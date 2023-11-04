// Package huggingfacehub can be used to download models and tokenizers from HuggingFace Hub,
// it is a port of huggingFace_hub python library to Go.
//
// The main functionality is provided by the `Download` method.
//
// Features supported:
//
// * Cache system that maches HuggingFace Hub (so same cache can be shared with Python).
// * Locked files (to guarantee only one download when multiple workers are trying to download simultaneously the same model).
// * Allow arbitrary progress function to be called (for progress bar).
// * Arbitrary revision.
//
// TODOs:
//
// * Add support for optional parameters.
// * Authentication tokens: should be relatively easy.
// * Resume downloads from interrupted connections.
// * Check disk-space before starting to download.

package huggingfacehub

import (
	"bytes"
	"context"
	"fmt"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"text/template"
	"time"
)

var SessionId string

// panicf generates an error message and panics with it, in one function.
func panicf(format string, args ...any) {
	err := errors.Errorf(format, args...)
	panic(err)
}

func init() {
	sessionUUID, err := uuid.NewRandom()
	if err != nil {
		panicf("failed generating UUID for SessionId: %v", err)
	}
	SessionId = strings.Replace(sessionUUID.String(), "-", "", -1)
}

var (
	// DefaultDirCreationPerm is used when creating new cache subdirectories.
	DefaultDirCreationPerm = os.FileMode(0755)

	// DefaultFileCreationPerm is used when creating files inside the cache subdirectories.
	DefaultFileCreationPerm = os.FileMode(0644)
)

const (
	// Versions, as of the time of this writing.
	transformersVersion = "4.34.1"
	hubVersion          = "0.17.3"
	tokenizersVersion   = "0.0.1"
)

const (
	HeaderXRepoCommit = "X-Repo-Commit"
	HeaderXLinkedETag = "X-Linked-Etag"
	HeaderXLinkedSize = "X-Linked-Size"
)

func getEnvOr(key, defaultValue string) string {
	v := os.Getenv(key)
	if v == "" {
		return defaultValue
	}
	return v
}

// DefaultCacheDir for HuggingFace Hub, same used by the python library.
//
// Its prefix is either `${XDG_CACHE_HOME}` if set, or `~/.cache` otherwise. Followed by `/huggingface/hub/`.
// So typically: `~/.cache/huggingface/hub/`.
func DefaultCacheDir() string {
	cacheDir := getEnvOr("XDG_CACHE_HOME", path.Join(os.Getenv("HOME"), ".cache"))
	cacheDir = path.Join(cacheDir, "huggingface", "hub")
	return cacheDir
}

// HttpUserAgent returns a user agent to use with HuggingFace Hub API.
// Loosely based on https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L198.
func HttpUserAgent() string {
	return fmt.Sprintf("transfomers/%v; golang/%s; session_id/%s; gomlx_tokenizers/%v; hf_hub/%s",
		transformersVersion, runtime.Version(), SessionId, tokenizersVersion, hubVersion)
}

// RepoIdSeparator is used to separate repository/model names parts when mapping to file names.
// Likely only for internal use.
const RepoIdSeparator = "--"

// RepoFolderName returns a serialized version of a hf.co repo name and type, safe for disk storage
// as a single non-nested folder.
//
// Based on github.com/huggingface/huggingface_hub repo_folder_name.
func RepoFolderName(repoId, repoType string) string {
	parts := []string{repoType + "s"}
	parts = append(parts, strings.Split(repoId, "/")...)
	return strings.Join(parts, RepoIdSeparator)
}

var (
	RepoTypesUrlPrefixes = map[string]string{
		"dataset": "datasets/",
		"space":   "spaces/",
	}

	DefaultRevision = "main"

	HuggingFaceUrlTemplate = template.Must(template.New("hf_url").Parse(
		"https://huggingface.co/{{.RepoId}}/resolve/{{.Revision}}/{{.Filename}}"))
)

// GetUrl is based on the `hf_hub_url` function defined in the [huggingface_hub](https://github.com/huggingface/huggingface_hub) library.
func GetUrl(repoId, fileName, repoType, revision string) string {
	if prefix, found := RepoTypesUrlPrefixes[repoType]; found {
		repoId = prefix + repoId
	}
	if revision == "" {
		revision = DefaultRevision
	}
	var buf bytes.Buffer
	err := HuggingFaceUrlTemplate.Execute(&buf,
		struct{ RepoId, Revision, Filename string }{repoId, revision, fileName})
	if err != nil {
		panicf("HuggingFaceUrlTemplate failed (!? pls report the bug, this shouldn't happen) with %+v", err)
	}
	url := buf.String()
	return url
}

// GetHeaders is based on the `build_hf_headers` function defined in the [huggingface_hub](https://github.com/huggingface/huggingface_hub) library.
// TODO: add support for authentication token.
func GetHeaders(userAgent, token string) map[string]string {
	return map[string]string{
		"user-agent": userAgent,
	}
}

// ProgressFn is a function called while downloading a file.
// It will be called with `progress=0` and `downloaded=0` at the first call, when download starts.
type ProgressFn func(progress, downloaded, total int, eof bool)

// progressReader implements a reader that calls progressFn after each read.
type progressReader struct {
	reader            io.Reader
	downloaded, total int
	progressFn        ProgressFn
}

// Read implements io.Reader, and report number of bytes read to progressFn.
func (r *progressReader) Read(dst []byte) (n int, err error) {
	n, err = r.reader.Read(dst)
	r.downloaded += n
	if err != nil && err != io.EOF {
		// No progress.
		return
	}
	r.progressFn(n, r.downloaded, r.total, err == io.EOF)
	return
}

// Download returns file either from cache or by downloading from HuggingFace Hub.
//
// TODO: a version with optional parameters.
//
// Args:
//
//   - `ctx` for the requests. There may be more than one request, the first being an `HEAD` HTTP.
//   - `client` used to make HTTP requests. I can be created with `&httpClient{}`.
//   - `repoId` and `fileName`: define the file and repository (model) name to download.
//   - `repoType`: usually "model".
//   - `revision`: default is "main", but a commitHash can be given.
//   - `cacheDir`: directory where to store the downloaded files, or reuse if previously downloaded.
//     Consider using the output from `DefaultCacheDir()` if in doubt.
//   - `token`: used for authentication. TODO: not implemented yet.
//   - `forceDownload`: if set to true, it will download the contents of the file even if there is a local copy.
//   - `localOnly`: does not use network, not even for reading the metadata.
//   - `progressFn`: is called during the download of a file. It is called synchronously and expected to be fast/
//     instantaneous. If the UI can be blocking, arrange it to be handled on a separate GoRoutine.
//
// On success it returns the `filePath` to the downloaded file, and its `commitHash`. Otherwise it returns an error.
func Download(ctx context.Context, client *http.Client,
	repoId, repoType, revision, fileName, cacheDir, token string,
	forceDownload, forceLocal bool, progressFn ProgressFn) (filePath, commitHash string, err error) {
	if cacheDir == "" {
		err = errors.New("Download() requires a cacheDir, even if temporary, to store the results of the download")
		return
	}
	cacheDir = path.Clean(cacheDir)
	userAgent := HttpUserAgent()
	if token != "" {
		// TODO, for now no token support.
		err = errors.Errorf("no support yet for authentication token while attemption to download %q from %q",
			fileName, repoId)
		return
	}
	folderName := RepoFolderName(repoId, repoType)

	// Find storage directory and if necessary create directories on disk.
	storageDir := path.Join(cacheDir, folderName)
	err = os.MkdirAll(storageDir, DefaultDirCreationPerm)
	if err != nil {
		err = errors.Wrapf(err, "failed to create cache directory %q:", storageDir)
		return
	}

	// Join the path parts of fileName using the current OS separator.
	relativeFilePath := path.Clean(path.Join(strings.Split(fileName, "/")...))

	// Local-only:
	if forceLocal {
		commitHash, err = readCommitHashForRevision(storageDir, revision)
		if err != nil {
			err = errors.WithMessagef(err, "while trying to load %q from repo %q from disk", fileName, repoId)
			return
		}
		filePath = getSnapshotPath(storageDir, commitHash, relativeFilePath)
		if !FileExists(filePath) {
			err = errors.Errorf("Download() with forceLocal, but file %q from repo %q not found in cache -- should be in %q", fileName, repoId, filePath)
			return
		}
		return
	}

	// URL and headers for request.
	url := GetUrl(repoId, fileName, repoType, revision)
	headers := GetHeaders(userAgent, token)

	// Get file Metadata.
	var metadata *HFFileMetadata
	metadata, err = getFileMetadata(ctx, client, url, token, headers)
	if err != nil {
		return
	}
	commitHash = metadata.CommitHash
	if commitHash == "" {
		err = errors.Errorf("resource %q for %q doesn't seem to be on huggingface.co (missing commit header)",
			fileName, repoId)
		return
	}
	etag := metadata.ETag
	if etag == "" {
		err = errors.Errorf("resource %q for %q doesn't have an ETag, not able to ensure reproduceability",
			fileName, repoId)
		return
	}

	var urlToDownload = url
	if metadata.Location != url {
		// In the case of a redirect, remove authorization header when downloading blob
		delete(headers, "authorization")
		urlToDownload = metadata.Location
	}

	// Make blob and snapshot paths (and create its directories).
	blobPath := path.Join(storageDir, "blobs", etag)
	snapshotPath := getSnapshotPath(storageDir, commitHash, relativeFilePath)
	for _, p := range []string{blobPath, snapshotPath} {
		dir := path.Dir(p)
		err = os.MkdirAll(dir, DefaultDirCreationPerm)
		if err != nil {
			err = errors.Wrapf(err, "cannot create cache directory %q for downloading %q from %q",
				dir, fileName, repoId)
			return
		}
	}

	// Maps the reference of revision to commitHash received. It's a no-op if they are the same.
	err = cacheCommitHashForSpecificRevision(storageDir, commitHash, revision)
	if err != nil {
		err = errors.WithMessagef(err, "while downloading %q from %q", fileName, repoId)
		return
	}

	// Use snapshot cached file, if available.
	if FileExists(snapshotPath) && !forceDownload {
		filePath = snapshotPath
		return
	}

	// If the generic blob is available (downloaded under a different name), link it and use it.
	if FileExists(blobPath) && !forceDownload {
		// ... create link
		err = createSymLink(snapshotPath, blobPath)
		if err != nil {
			err = errors.WithMessagef(err, "while downloading %q from %q", fileName, repoId)
			return
		}
		filePath = snapshotPath
		return
	}

	// TODO: pre-check disk space availability.

	// Lock file to avoid parallel downloads.
	lockPath := blobPath + ".lock"
	errLock := execOnFileLock(ctx, lockPath, func() {
		if FileExists(blobPath) && !forceDownload {
			// Some other process (or goroutine) already downloaded the file.
			return
		}

		// Create tmpFile where to download.
		var (
			tmpFile       *os.File
			tmpFileClosed bool
		)

		tmpFile, err = os.CreateTemp(cacheDir, "tmp_blob")
		if err != nil {
			err = errors.Wrapf(err, "creating temporary file for download in %q", cacheDir)
			return
		}
		var tmpFilePath = tmpFile.Name()
		defer func() {
			// If we exit with an error, make sure to close and remove unfinished temporary file.
			if !tmpFileClosed {
				_ = tmpFile.Close()
				_ = os.Remove(tmpFilePath)
			}
		}()

		// Connect and download with an HTTP GET.
		var resp *http.Response
		resp, err = client.Get(urlToDownload)
		if err != nil {
			err = errors.Wrapf(err, "failed request to download file to %q", urlToDownload)
			return
		}
		defer resp.Body.Close()

		// Replace reader with one that reports the progress, if requested.
		var r io.Reader = resp.Body
		if progressFn != nil {
			r = &progressReader{
				reader:     r,
				downloaded: 0,
				total:      metadata.Size,
				progressFn: progressFn,
			}
			progressFn(0, 0, metadata.Size, false) // Do initial call with 0 downloaded.
		}

		// Download.
		_, err := io.Copy(tmpFile, r)
		if err != nil {
			err = errors.Wrapf(err, "failed to download file from %q", urlToDownload)
			return
		}

		// Download succeeded, move to our target location.
		tmpFileClosed = true
		if err = tmpFile.Close(); err != nil {
			err = errors.Wrapf(err, "failed to close temporary download file %q", tmpFilePath)
			return
		}
		if err = os.Rename(tmpFilePath, blobPath); err != nil {
			err = errors.Wrapf(err, "failed to move downloaded file %q to %q", tmpFilePath, blobPath)
			return
		}
		if err = createSymLink(snapshotPath, blobPath); err != nil {
			return
		}
	})
	if err == nil && errLock != nil {
		err = errLock
	}
	if err != nil {
		err = errors.WithMessagef(err, "while downloading %q from %q", fileName, repoId)
		return
	}
	filePath = snapshotPath
	return
}

// HFFileMetadata used by HuggingFace Hub.
type HFFileMetadata struct {
	CommitHash, ETag, Location string
	Size                       int
}

func removeQuotes(str string) string {
	return strings.TrimRight(strings.TrimLeft(str, "\""), "\"")
}

// getFileMetadata: make a "HEAD" HTTP request and return the response with the header.
func getFileMetadata(ctx context.Context, client *http.Client, url, token string, headers map[string]string) (metadata *HFFileMetadata, err error) {
	// Create a request to download the tokenizer.
	var req *http.Request
	req, err = http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		err = errors.Wrap(err, "failed request for metadata: ")
		return
	}

	// Include requested headers, plus prevent any compression => we want to know the real size of the file.
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Accept-Encoding", "identity")

	// Make the request and download the tokenizer.
	resp, err := client.Do(req)
	if err != nil {
		err = errors.Wrap(err, "failed request for metadata: ")
		return
	}

	// TODO: handle redirects.
	defer func() { _ = resp.Body.Close() }()
	var contents []byte
	contents, err = io.ReadAll(resp.Body)
	if err != nil {
		err = errors.Wrapf(err, "failed reading response (%d) for metadata: ", resp.StatusCode)
		return
	}

	// Check status code.
	if resp.StatusCode != 200 {
		err = errors.Errorf("request for metadata from %q failed with the following message: %q",
			url, contents)
		return
	}

	metadata = &HFFileMetadata{
		CommitHash: resp.Header.Get(HeaderXRepoCommit),
	}
	metadata.ETag = resp.Header.Get(HeaderXLinkedETag)
	if metadata.ETag == "" {
		metadata.ETag = resp.Header.Get("ETag")
	}
	metadata.ETag = removeQuotes(metadata.ETag)
	metadata.Location = resp.Header.Get("Location")
	if metadata.Location == "" {
		metadata.Location = resp.Request.URL.String()
	}

	if sizeStr := resp.Header.Get(HeaderXLinkedSize); sizeStr != "" {
		metadata.Size, err = strconv.Atoi(sizeStr)
		if err != nil {
			err = nil // Discard
			metadata.Size = 0
		}
	}
	if metadata.Size == 0 {
		metadata.Size = int(resp.ContentLength)
	}
	return
}

// getSnapshotPath returns the "snapshot" path/link to the given commitHash and relativeFilePath.
func getSnapshotPath(storageDir, commitHash, relativeFilePath string) string {
	snapshotPath := path.Join(storageDir, "snapshots")
	return path.Join(snapshotPath, commitHash, relativeFilePath)
}

// cacheCommitHashForSpecificRevision creates reference between a revision (tag, branch or truncated commit hash)
// and the corresponding commit hash.
//
// It does nothing if `revision` is already a proper `commit_hash` or reference is already cached.
func cacheCommitHashForSpecificRevision(storageDir, commitHash, revision string) error {
	if revision == commitHash {
		// Nothing to do.
		return nil
	}

	refPath := path.Join(storageDir, "refs", revision)
	err := os.MkdirAll(path.Dir(refPath), DefaultDirCreationPerm)
	if err != nil {
		return errors.Wrap(err, "failed to create reference subdirectory in cache")
	}
	if FileExists(refPath) {
		contents, err := os.ReadFile(refPath)
		if err != nil {
			return errors.Wrapf(err, "failed reading %q", refPath)
		}
		checkCommitHash := strings.Trim(string(contents), "\n")
		if checkCommitHash == commitHash {
			// Same as previously stored, all good.
			return nil
		}
	}

	// Save new reference.
	err = os.WriteFile(refPath, []byte(commitHash), DefaultFileCreationPerm)
	if err != nil {
		return errors.Wrapf(err, "failed creating file %q", refPath)
	}
	return nil
}

// readCommitHashForRevision from disk.
// Notice revision can be a commitHash: if we don't find a revision file, we assume that is the case.
func readCommitHashForRevision(storageDir, revision string) (commitHash string, err error) {
	refPath := path.Join(storageDir, "refs", revision)
	if !FileExists(refPath) {
		commitHash = revision
		return
	}

	var contents []byte
	contents, err = os.ReadFile(refPath)
	if err != nil {
		err = errors.Wrapf(err, "failed reading %q", refPath)
		return
	}
	commitHash = strings.Trim(string(contents), "\n")
	return
}

// FileExists returns true if file or directory exists.
func FileExists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	if errors.Is(err, os.ErrNotExist) {
		return false
	}
	panic(err)
}

// createSymlink creates a symbolic link named dst pointing to src, using a relative path if possible.
//
// We use relative paths because:
// * It's what `huggingface_hub` library does, and we want to keep things compatible.
// * If the cache folder is moved or backed up, links won't break.
// * Relative paths seem better handled on Windows -- although Windows is not yet fully supported for this package.
//
// Example layout:
//
//	└── [ 128]  snapshots
//	  ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
//	  │   ├── [  52]  README.md -> ../../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
//	  │   └── [  76]  pytorch_model.bin -> ../../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
func createSymLink(dst, src string) error {
	relLink, err := filepath.Rel(path.Dir(dst), src)
	if err != nil {
		relLink = src // Take the absolute path instead.
	}
	if err = os.Symlink(relLink, dst); err != nil {
		err = errors.Wrapf(err, "while symlink'ing %q to %q using %q", src, dst, relLink)
	}
	return err
}

// onFileLock locks the given file, executes the function, unlocks again and returns.
func execOnFileLock(ctx context.Context, lockPath string, fn func()) (err error) {
	var f *os.File
	f, err = os.OpenFile(lockPath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, DefaultFileCreationPerm)
	if err != nil {
		err = errors.Wrapf(err, "while locking %q", lockPath)
		return
	}
	defer f.Close()

	// Acquire lock or return an error if context is canceled (due to time out).
	for {
		err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			break
		}
		if !errors.Is(err, syscall.EAGAIN) {
			err = errors.Wrapf(err, "while locking %q", lockPath)
			return err
		}

		// Wait from 1 to 2 seconds.
		timeDuration := time.Millisecond * time.Duration(1000+rand.Intn(1000))
		select {
		case <-ctx.Done():
			err = errors.Errorf("context cancelled (timedout?) while waiting for lock to download %q", lockPath)
			return
		case <-time.NewTimer(timeDuration).C:
			// Nothing, just continues to the next attempt.
		}
	}

	// Setup clean up in a deferred function, so it happens even if `fn()` panics.
	defer func() {
		err = os.Remove(lockPath)
		if err != nil {
			err = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		}
		if err != nil {
			err = errors.Wrapf(err, "unlocking and removing file %q", lockPath)
		}
	}()

	// We got the lock, run the function.
	fn()

	return
}

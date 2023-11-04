# huggingfacehub
Download models and tokenizers from HuggingFace Hub, a port of huggingFace_hub python library to Go 

## Introduction

A simple, straight-forward port of [github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) library for Go.

It is still not very ergonomic -- all parameters are obligatory. So a bit annoying to use, but hopefully end users will instead never have to use this library
directly, and instead use it though [github.com/gomlx/tokenizers] and [github.com/gomlx/transformers].

Features supported:

* Cache system that maches HuggingFace Hub (so same cache can be shared with Python).
* Locked files (to guarantee only one download when multiple workers are trying to download simultaneously the same model).
* Allow arbitrary progress function to be called (for progress bar).
* Arbitrary revision.

TODOs:

* Add support for optional parameters.
* Authentication tokens: should be relatively easy.
* Resume downloads from interrupted connections.
* Check disk-space before starting to download.

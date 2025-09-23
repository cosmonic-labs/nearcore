#!/usr/bin/env bash

set -xe

git fetch --all --tags
git checkout cosmonic/test/run-slow-test-on-wasmtime
perf record -k mono -ecpu-clock -F1000 -g --call-graph fp,65528 "$(cargo test --no-run -p near-vm-runner --features wasmtime_vm slow 2>&1 | tail -1 | cut -d' ' -f6 | tr -d '()')"
samply import -P 3333 perf.data

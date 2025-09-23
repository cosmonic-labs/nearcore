#!/usr/bin/env bash

set -xe

git fetch --all --tags
git checkout cosmonic/test/run-slow-test-on-wasmtime
perf record -k mono -ecpu-clock -F1000 -g --call-graph fp,65528 cargo test -p near-vm-runner --features wasmtime_vm slow_test_parallel_runtime_invocations
perf inject --jit --input perf.data --output perf.jit.data
samply import -P 3333 perf.jit.data

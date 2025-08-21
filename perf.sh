#!/usr/bin/env bash

set -xe

git fetch --all --tags
git checkout cosmonic/opt
cargo build --profile dev-release --config .cargo/config.profiling.toml -p neard
rm -rf ~/.near/data/contract.cache/
timeout 30 perf record -k mono -ecpu-clock -F1000 -g --call-graph fp,65528 ./target/dev-release/neard view-state apply-range --shard-id 6 --storage memtrie benchmark || :
perf inject --jit --input perf.data --output perf.jit.data
samply import -P 3333 perf.jit.data

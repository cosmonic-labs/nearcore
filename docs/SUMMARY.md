# Summary

[Introduction](README.md)

# Architecture

- [Overview](./architecture/README.md)
- [How neard works](./architecture/how/README.md)
  - [How Sync Works](./architecture/how/sync.md)
  - [Garbage Collection](./architecture/how/gc.md)
  - [How Epoch Works](./architecture/how/epoch.md)
  - [Transaction Routing](./architecture/how/tx_routing.md)
  - [Transactions And Receipts](./architecture/how/tx_receipts.md)
  - [Cross shard transactions - deep dive](./architecture/how/cross-shard.md)
  - [Gas](./architecture/how/gas.md)
  - [Receipt Congestion](./architecture/how/receipt-congestion.md)
  - [Meta transactions](./architecture/how/meta-tx.md)
  - [Serialization: Borsh, Json, ProtoBuf](./architecture/how/serialization.md)
  - [Proofs](./architecture/how/proofs.md)
  - [Resharding V2](./architecture/how/resharding_v2.md)
  - [Optimistic block](./architecture/how/optimistic_block.md)
- [How neard will work](./architecture/next/README.md)
  - [Catchup and state sync improvements](./architecture/next/catchup_and_state_sync.md)
  - [Malicious producers and phase 2](./architecture/next/malicious_chunk_producer_and_phase2.md)
- [Storage](./architecture/storage.md)
  - [Storage Request Flow](./architecture/storage/flow.md)
  - [Trie Storage](./architecture/storage/trie_storage.md)
  - [Database Format](./architecture/storage/database.md)
  - [Flat Storage](./architecture/storage/flat_storage.md)
- [Network](./architecture/network.md)
- [Gas Cost Parameters](./architecture/gas/README.md)
  - [Parameter Definitions](./architecture/gas/parameter_definition.md)
  - [Gas Profile](./architecture/gas/gas_profile.md)
  - [Runtime Parameter Estimator](./architecture/gas/estimator.md)

# Practices

- [Overview](./practices/README.md)
- [Rust 🦀](./practices/rust.md)
- [Workflows](./practices/workflows/README.md)
  - [Run a Node](./practices/workflows/run_a_node.md)
  - [Deploy a Contract](./practices/workflows/deploy_a_contract.md)
  - [Run Gas Estimations](./practices/workflows/gas_estimations.md)
  - [Localnet on many machines](./practices/workflows/localnet_on_many_machines.md)
  - [IO tracing](./practices/workflows/io_trace.md)
  - [Profiling](./practices/workflows/profiling.md)
  - [Benchmarks](./practices/workflows/benchmarks.md)
    - [Synthetic Workloads](./practices/workflows/benchmarking_synthetic_workloads.md)
    - [Native Transfer Chunk Application](./practices/workflows/benchmarking_chunk_application_on_native_transfers.md)
  - [Working with OpenTelemetry Traces](./practices/workflows/otel_traces.md)
  - [Futex contention](./practices/workflows/futex_contention.md)
- [Code Style](./practices/style.md)
- [Documentation](./practices/docs.md)
- [Tracking Issues](./practices/tracking_issues.md)
- [Security Vulnerabilities](./practices/security_vulnerabilities.md)
- [Fast Builds](./practices/fast_builds.md)
- [Testing](./practices/testing/README.md)
  - [Python Tests](./practices/testing/python_tests.md)
  - [Testing Utils](./practices/testing/test_utils.md)
  - [Test Coverage](./practices/testing/coverage.md)
- [Protocol Upgrade](./practices/protocol_upgrade.md)

# Advanced configuration

- [Networking](./advanced_configuration/networking.md)

# Custom test networks

- [Starting a network from mainnet state](./test_networks/mainnet_spoon.md)

# Misc

- [Overview](./misc/README.md)
- [State Sync Dump](./misc/state_sync_dump.md)
- [Archival node - recovery of missing data](./misc/archival_data_recovery.md)

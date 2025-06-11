use bencher::{Bencher, benchmark_group, benchmark_main};
use near_parameters::RuntimeConfig;
use near_parameters::vm::VMKind;
use runtime_benchmarks::bench_ft_transfer;

fn near_vm_2_ft_transfer(bench: &mut Bencher) {
    let config = RuntimeConfig::test();
    bench_ft_transfer(
        bench,
        near_parameters::vm::Config {
            vm_kind: VMKind::NearVm2,
            ..near_parameters::vm::Config::clone(&config.wasm_config)
        },
    );
}

fn wasmtime_ft_transfer(bench: &mut Bencher) {
    let config = RuntimeConfig::test();
    bench_ft_transfer(
        bench,
        near_parameters::vm::Config {
            vm_kind: VMKind::Wasmtime,
            ..near_parameters::vm::Config::clone(&config.wasm_config)
        },
    );
}

benchmark_group!(near_vm_2, near_vm_2_ft_transfer);
benchmark_group!(wasmtime, wasmtime_ft_transfer);

#[cfg(target_arch = "x86_64")]
benchmark_main!(near_vm_2, wasmtime);

#[cfg(not(target_arch = "x86_64"))]
benchmark_main!(wasmtime);

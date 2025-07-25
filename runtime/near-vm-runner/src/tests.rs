mod cache;
mod compile_errors;
#[cfg(feature = "prepare")]
mod fuzzers;
mod regression_tests;
mod rs_contract;
mod runtime_errors;
pub(crate) mod test_builder;
mod ts_contract;
mod wasm_validation;

use crate::logic::VMContext;
use near_parameters::RuntimeConfigStore;
use near_parameters::vm::VMKind;
use near_primitives_core::version::PROTOCOL_VERSION;

const CURRENT_ACCOUNT_ID: &str = "alice";
const SIGNER_ACCOUNT_ID: &str = "bob";
const SIGNER_ACCOUNT_PK: [u8; 3] = [0, 1, 2];
const PREDECESSOR_ACCOUNT_ID: &str = "carol";

pub(crate) fn test_vm_config(vm_kind: Option<VMKind>) -> near_parameters::vm::Config {
    let store = RuntimeConfigStore::test();
    let config = store.get_config(PROTOCOL_VERSION).wasm_config.clone();
    near_parameters::vm::Config {
        vm_kind: vm_kind.unwrap_or_else(|| config.vm_kind.replace_with_wasmtime_if_unsupported()),
        ..near_parameters::vm::Config::clone(&config)
    }
}

pub(crate) fn with_vm_variants(runner: impl Fn(VMKind) -> ()) {
    #[allow(unused)]
    let run = move |kind| {
        println!("running test with {kind:?}");
        runner(kind)
    };

    #[cfg(feature = "wasmtime_vm")]
    run(VMKind::Wasmtime);

    #[cfg(all(feature = "near_vm", target_arch = "x86_64"))]
    run(VMKind::NearVm);

    #[cfg(all(feature = "near_vm", target_arch = "x86_64"))]
    run(VMKind::NearVm2);
}

fn create_context(input: Vec<u8>) -> VMContext {
    VMContext {
        current_account_id: CURRENT_ACCOUNT_ID.parse().unwrap(),
        signer_account_id: SIGNER_ACCOUNT_ID.parse().unwrap(),
        signer_account_pk: Vec::from(&SIGNER_ACCOUNT_PK[..]),
        predecessor_account_id: PREDECESSOR_ACCOUNT_ID.parse().unwrap(),
        input,
        promise_results: Vec::new().into(),
        block_height: 10,
        block_timestamp: 42,
        epoch_height: 1,
        account_balance: 2u128,
        account_locked_balance: 0,
        storage_usage: 12,
        attached_deposit: 2u128,
        prepaid_gas: 10_u64.pow(14),
        random_seed: vec![0, 1, 2],
        view_config: None,
        output_data_receivers: vec![],
    }
}

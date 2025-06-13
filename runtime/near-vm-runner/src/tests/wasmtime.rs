#![allow(clippy::disallowed_types)]

use std::sync::Arc;

use anyhow::Context as _;
use near_parameters::RuntimeFeesConfig;
use near_parameters::vm::{Config, VMKind};
use near_primitives_core::code::ContractCode;

use crate::VM as _;
use crate::logic::ReturnData;
use crate::logic::mocks::mock_external::MockedExternal;
use crate::tests::{create_context, test_vm_config};
use crate::wasmtime_runner::WasmtimeVM;

fn encode(xs: &[u64]) -> Vec<u8> {
    xs.iter().flat_map(|it| it.to_le_bytes()).collect()
}

fn make_config() -> Arc<Config> {
    let mut config = test_vm_config();
    config.vm_kind = VMKind::Wasmtime;
    Arc::new(config)
}

#[test]
fn run_wasm() -> anyhow::Result<()> {
    // TODO: figure out what to do with Wasmer tests
    //let wasm = wat::parse_bytes(
    //    br#"(module
    //    (func $multiply (import "env" "multiply") (param i32 i32) (result i32))
    //    (func (export "add") (param i32 i32) (result i32)
    //       (i32.add (local.get 0)
    //                (local.get 1)))
    //    (func (export "double_then_add") (param i32 i32) (result i32)
    //       (i32.add (call $multiply (local.get 0) (i32.const 2))
    //                (call $multiply (local.get 1) (i32.const 2))))
    // )"#,
    //)
    //.context("failed to parse WAT")?;

    let fees = Arc::new(RuntimeFeesConfig::test());
    let mut external = MockedExternal::with_code(ContractCode::new(
        near_test_contracts::rs_contract().into(),
        None,
    ));

    let config = make_config();

    let vm = Box::new(WasmtimeVM::new(Arc::clone(&config)));

    let context = create_context(encode(&[10u64, 20u64]));
    let gas_counter = context.make_gas_counter(&config);

    let contract = vm.prepare(&external, None, gas_counter, "write_key_value");

    let result = contract.run(&mut external, &context, fees).context("failed to run contract")?;
    if let ReturnData::Value(value) = &result.return_data {
        let mut arr = [0u8; size_of::<u64>()];
        arr.copy_from_slice(&value);
        let res = u64::from_le_bytes(arr);
        assert_eq!(res, 0);
    } else {
        panic!("Value was not returned");
    }

    Ok(())
}

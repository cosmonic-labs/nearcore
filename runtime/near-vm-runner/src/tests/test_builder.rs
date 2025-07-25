use crate::logic::{
    ProtocolVersion, ReturnData, VMContext, VMOutcome, mocks::mock_external::MockedExternal,
};
use crate::runner::VMKindExt;
use near_parameters::vm::VMKind;
use near_parameters::{RuntimeConfig, RuntimeConfigStore, RuntimeFeesConfig};
use near_primitives_core::code::ContractCode;
use near_primitives_core::types::Gas;
use near_primitives_core::version::ProtocolFeature;
use std::{collections::HashSet, fmt::Write, sync::Arc};

pub(crate) fn test_builder() -> TestBuilder {
    let context = VMContext {
        current_account_id: "alice".parse().unwrap(),
        signer_account_id: "bob".parse().unwrap(),
        signer_account_pk: vec![0, 1, 2],
        predecessor_account_id: "carol".parse().unwrap(),
        input: Vec::new(),
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
    };
    let mut skip = HashSet::new();
    for kind in [VMKind::NearVm, VMKind::NearVm2, VMKind::Wasmtime] {
        if !kind.is_available() {
            skip.insert(kind);
        }
    }
    TestBuilder {
        code: ContractCode::new(Vec::new(), None),
        context,
        protocol_versions: vec![u32::MAX],
        skip,
        opaque_error: false,
        opaque_outcome: false,
        method: "main".into(),
    }
}

pub(crate) struct TestBuilder {
    code: ContractCode,
    context: VMContext,
    protocol_versions: Vec<ProtocolVersion>,
    skip: HashSet<VMKind>,
    opaque_error: bool,
    opaque_outcome: bool,
    method: String,
}

impl TestBuilder {
    pub(crate) fn wat(mut self, wat: &str) -> Self {
        let wasm = wat::parse_str(wat)
            .unwrap_or_else(|err| panic!("failed to parse input wasm: {err}\n{wat}"));
        self.code = ContractCode::new(wasm, None);
        self
    }

    pub(crate) fn wasm(mut self, wasm: &[u8]) -> Self {
        self.code = ContractCode::new(wasm.to_vec(), None);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn get_wasm(&self) -> &[u8] {
        self.code.code()
    }

    pub(crate) fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    pub(crate) fn gas(mut self, gas: Gas) -> Self {
        self.context.prepaid_gas = gas;
        self
    }

    pub(crate) fn opaque_error(mut self) -> Self {
        self.opaque_error = true;
        self
    }

    pub(crate) fn opaque_outcome(mut self) -> Self {
        self.opaque_outcome = true;
        self
    }

    pub(crate) fn skip_wasmtime(mut self) -> Self {
        self.skip.insert(VMKind::Wasmtime);
        self
    }

    pub(crate) fn skip_near_vm(mut self) -> Self {
        self.skip.insert(VMKind::NearVm);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn only_wasmtime(self) -> Self {
        self.skip_near_vm()
    }

    pub(crate) fn only_near_vm(self) -> Self {
        self.skip_wasmtime()
    }

    /// Add additional protocol features to this test.
    ///
    /// Tricky. Given `[feat1, feat2, feat3]`, this will run *four* tests for
    /// protocol versions `[feat1 - 1, feat2 - 1, feat3 - 1, PROTOCOL_VERSION]`.
    ///
    /// When using this method with `n` features, be sure to pass `n + 1`
    /// expectations to the `expects` method. For nightly features, you can
    /// `cfg` the relevant features and expect.
    #[allow(unused)]
    #[deprecated = "if test variation is necessary vary not by protocol features but by config changes"]
    #[allow(deprecated)]
    pub(crate) fn protocol_features(
        mut self,
        protocol_features: &'static [ProtocolFeature],
    ) -> Self {
        for feat in protocol_features {
            self = self.protocol_version(feat.protocol_version() - 1);
        }
        self
    }

    /// Add a protocol version to test.
    #[deprecated = "if test variation is necessary vary not by protocol features but by config changes"]
    pub(crate) fn protocol_version(mut self, protocol_version: ProtocolVersion) -> Self {
        self.protocol_versions.push(protocol_version - 1);
        self
    }

    #[deprecated = "if test variation is necessary vary not by protocol features but by config changes"]
    pub(crate) fn only_protocol_versions(
        mut self,
        protocol_versions: Vec<ProtocolVersion>,
    ) -> Self {
        self.protocol_versions = protocol_versions;
        self
    }

    #[track_caller]
    pub(crate) fn expect(self, want: &expect_test::Expect) {
        self.expects(std::iter::once(want))
    }

    pub(crate) fn configs(&self) -> impl Iterator<Item = Arc<RuntimeConfig>> {
        let runtime_config_store = RuntimeConfigStore::new(None);
        self.protocol_versions
            .clone()
            .into_iter()
            .map(move |pv| Arc::clone(runtime_config_store.get_config(pv)))
    }

    #[track_caller]
    pub(crate) fn expects<'a, I>(mut self, wants: I)
    where
        I: IntoIterator<Item = &'a expect_test::Expect>,
        I::IntoIter: ExactSizeIterator,
    {
        self.protocol_versions.sort();
        let mut runtime_config_store = RuntimeConfigStore::new(None);
        let wants = wants.into_iter();
        assert_eq!(
            wants.len(),
            self.protocol_versions.len(),
            "specified {} protocol versions but only {} expectation",
            self.protocol_versions.len(),
            wants.len(),
        );

        for (want, &protocol_version) in wants.zip(&self.protocol_versions) {
            let mut results = vec![];
            for vm_kind in [VMKind::NearVm, VMKind::Wasmtime] {
                if self.skip.contains(&vm_kind) {
                    continue;
                }

                let runtime_config = runtime_config_store.get_config_mut(protocol_version);
                Arc::get_mut(&mut Arc::get_mut(runtime_config).unwrap().wasm_config)
                    .unwrap()
                    .vm_kind = vm_kind;
                let mut fake_external = MockedExternal::with_code(self.code.clone_for_tests());
                let config = runtime_config.wasm_config.clone();
                let fees = Arc::new(RuntimeFeesConfig::test());
                let context = self.context.clone();
                let gas_counter = context.make_gas_counter(&config);
                let Some(runtime) = vm_kind.runtime(config) else {
                    panic!("runtime for {:?} has not been compiled", vm_kind);
                };
                println!("Running {:?} for protocol version {}", vm_kind, protocol_version);
                let outcome = runtime
                    .prepare(&fake_external, None, gas_counter, &self.method)
                    .run(&mut fake_external, &context, fees)
                    .expect("execution failed");

                let mut got = String::new();

                if !self.opaque_outcome {
                    fmt_outcome_without_abort(&outcome, &mut got).unwrap();
                    writeln!(&mut got).unwrap();
                }

                if let Some(err) = outcome.aborted {
                    let err_str = err.to_string();
                    assert!(
                        err_str.len() < 1000,
                        "errors should be bounded in size to prevent abuse \
                         via exhausting the storage space. Got: {err_str}"
                    );
                    if self.opaque_error {
                        writeln!(&mut got, "Err: ...").unwrap();
                    } else {
                        writeln!(&mut got, "Err: {err_str}").unwrap();
                    }
                };

                results.push((vm_kind, got));
            }

            if !results.is_empty() {
                want.assert_eq(&results[0].1);
                for i in 1..results.len() {
                    if results[i].1 != results[0].1 {
                        panic!(
                            "Inconsistent VM Output:\n{:?}:\n{}\n\n{:?}:\n{}",
                            results[0].0, results[0].1, results[i].0, results[i].1
                        )
                    }
                }
            }
        }
    }
}

fn fmt_outcome_without_abort(
    outcome: &VMOutcome,
    out: &mut dyn std::fmt::Write,
) -> std::fmt::Result {
    let return_data_str = match &outcome.return_data {
        ReturnData::None => "None".to_string(),
        ReturnData::ReceiptIndex(_) => "Receipt".to_string(),
        ReturnData::Value(v) => format!("Value [{} bytes]", v.len()),
    };
    write!(
        out,
        "VMOutcome: balance {} storage_usage {} return data {} burnt gas {} used gas {}",
        outcome.balance,
        outcome.storage_usage,
        return_data_str,
        outcome.burnt_gas,
        outcome.used_gas
    )?;
    Ok(())
}

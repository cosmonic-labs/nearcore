use near_chain_configs::{Genesis, GenesisConfig, GenesisRecords, get_initial_supply};
use near_crypto::{InMemorySigner, Signer};
use near_parameters::ActionCosts;
use near_primitives::account::{AccessKey, Account, AccountContract};
use near_primitives::apply::ApplyChunkReason;
use near_primitives::bandwidth_scheduler::BlockBandwidthRequests;
use near_primitives::congestion_info::{BlockCongestionInfo, ExtendedCongestionInfo};
use near_primitives::hash::{CryptoHash, hash};
use near_primitives::receipt::Receipt;
use near_primitives::shard_layout::ShardUId;
use near_primitives::state_record::{StateRecord, state_record_to_account_id};
use near_primitives::test_utils::MockEpochInfoProvider;
use near_primitives::transaction::{ExecutionOutcomeWithId, SignedTransaction};
use near_primitives::types::{AccountId, AccountInfo, Balance};
use near_primitives::version::PROTOCOL_VERSION;
use near_primitives_core::account::id::AccountIdRef;
use near_store::ShardTries;
use near_store::genesis::GenesisStateApplier;
use near_store::test_utils::TestTriesBuilder;
use node_runtime::{ApplyState, Runtime, SignedValidPeriodTransactions};
use parking_lot::{Condvar, Mutex};
use random_config::random_config;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;

pub mod random_config;

/// Initial balance used in tests.
pub const TESTING_INIT_BALANCE: Balance = 1_000_000_000 * NEAR_BASE;

/// Validator's stake used in tests.
pub const TESTING_INIT_STAKE: Balance = 50_000_000 * NEAR_BASE;

/// One NEAR, divisible by 10^24.
pub const NEAR_BASE: Balance = 1_000_000_000_000_000_000_000_000;

pub struct StandaloneRuntime {
    pub apply_state: ApplyState,
    pub runtime: Runtime,
    pub tries: ShardTries,
    pub signer: Signer,
    pub root: CryptoHash,
    pub epoch_info_provider: MockEpochInfoProvider,
}

impl StandaloneRuntime {
    pub fn account_id(&self) -> AccountId {
        self.signer.get_account_id()
    }

    pub fn new(
        signer: Signer,
        state_records: &[StateRecord],
        tries: ShardTries,
        validators: Vec<AccountInfo>,
    ) -> Self {
        let mut runtime_config = random_config();
        let wasm_config = Arc::make_mut(&mut runtime_config.wasm_config);
        // Bumping costs to avoid inflation overflows.
        wasm_config.limit_config.max_total_prepaid_gas = 10u64.pow(15);
        let fees = Arc::make_mut(&mut runtime_config.fees);
        fees.action_fees[ActionCosts::new_action_receipt].execution =
            runtime_config.wasm_config.limit_config.max_total_prepaid_gas / 64;
        fees.action_fees[ActionCosts::new_data_receipt_base].execution =
            runtime_config.wasm_config.limit_config.max_total_prepaid_gas / 64;

        let runtime = Runtime::new();
        let genesis = Genesis::new(
            GenesisConfig {
                validators,
                total_supply: get_initial_supply(state_records),
                epoch_length: 60,
                ..Default::default()
            },
            GenesisRecords(state_records.to_vec()),
        )
        .unwrap();

        let mut account_ids: HashSet<AccountId> = HashSet::new();
        genesis.for_each_record(|record: &StateRecord| {
            account_ids.insert(state_record_to_account_id(record).clone());
        });
        let writers = std::sync::atomic::AtomicUsize::new(0);
        let shard_uid = genesis.config.shard_layout.shard_uids().next().unwrap();
        let root = GenesisStateApplier::apply(
            &writers,
            tries.clone(),
            shard_uid,
            &[],
            &runtime_config.fees.storage_usage_config,
            &genesis,
            account_ids,
        );
        let congestion_info = genesis
            .config
            .shard_layout
            .shard_ids()
            .map(|shard_id| (shard_id, ExtendedCongestionInfo::default()))
            .collect();
        let congestion_info = BlockCongestionInfo::new(congestion_info);

        let apply_state = ApplyState {
            apply_reason: ApplyChunkReason::UpdateTrackedShard,
            block_height: 1,
            prev_block_hash: Default::default(),
            shard_id: shard_uid.shard_id(),
            epoch_id: Default::default(),
            epoch_height: 0,
            gas_price: 100,
            block_timestamp: 0,
            gas_limit: None,
            random_seed: Default::default(),
            current_protocol_version: PROTOCOL_VERSION,
            config: Arc::new(runtime_config),
            cache: None,
            is_new_chunk: true,
            congestion_info,
            bandwidth_requests: BlockBandwidthRequests::empty(),
            trie_access_tracker_state: Default::default(),
        };

        Self {
            apply_state,
            runtime,
            tries,
            signer,
            root,
            epoch_info_provider: MockEpochInfoProvider::default(),
        }
    }

    pub fn process_block(
        &mut self,
        receipts: &[Receipt],
        transactions: Vec<SignedTransaction>,
    ) -> ProcessBlockOutcome {
        // TODO - the shard id is correct but the shard version is hardcoded. It
        // would be better to store the shard layout in self and read the uid
        // from there.
        let shard_id = self.apply_state.shard_id;
        let shard_uid = ShardUId::new(0, shard_id);
        let trie = self.tries.get_trie_for_shard(shard_uid, self.root);
        let validity = vec![true; transactions.len()];
        let transactions = SignedValidPeriodTransactions::new(transactions, validity);
        let apply_result = self
            .runtime
            .apply(
                trie,
                &None,
                &self.apply_state,
                receipts,
                transactions,
                &self.epoch_info_provider,
                Default::default(),
            )
            .unwrap();

        let mut store_update = self.tries.store_update();
        self.root = self.tries.apply_all(&apply_result.trie_changes, shard_uid, &mut store_update);
        store_update.commit().unwrap();
        self.apply_state.block_height += 1;

        self.apply_state.bandwidth_requests = BlockBandwidthRequests {
            shards_bandwidth_requests: [(shard_id, apply_result.bandwidth_requests)]
                .into_iter()
                .collect(),
        };

        let mut has_queued_receipts = false;
        if let Some(congestion_info) = apply_result.congestion_info {
            has_queued_receipts = congestion_info.receipt_bytes() > 0;

            self.apply_state.congestion_info.insert(
                shard_id,
                ExtendedCongestionInfo { missed_chunks_count: 0, congestion_info },
            );
        }

        ProcessBlockOutcome {
            outgoing_receipts: apply_result.outgoing_receipts,
            execution_outcomes: apply_result.outcomes,
            has_queued_receipts,
        }
    }
}

#[derive(Default)]
pub struct RuntimeMailbox {
    pub incoming_transactions: Vec<SignedTransaction>,
    pub incoming_receipts: Vec<Receipt>,
}

impl RuntimeMailbox {
    pub fn is_empty(&self) -> bool {
        self.incoming_receipts.is_empty() && self.incoming_transactions.is_empty()
    }
}

pub struct ProcessBlockOutcome {
    outgoing_receipts: Vec<Receipt>,
    execution_outcomes: Vec<ExecutionOutcomeWithId>,
    has_queued_receipts: bool,
}

#[derive(Default)]
pub struct RuntimeGroup {
    pub mailboxes: (Mutex<HashMap<AccountId, RuntimeMailbox>>, Condvar),
    pub state_records: Arc<Vec<StateRecord>>,
    pub signers: Vec<Signer>,
    pub validators: Vec<AccountInfo>,

    /// Account id of the runtime on which the transaction was executed mapped to the transactions.
    pub executed_transactions: Mutex<HashMap<AccountId, Vec<SignedTransaction>>>,
    /// Account id of the runtime on which the receipt was executed mapped to the list of the receipts.
    pub executed_receipts: Mutex<HashMap<AccountId, Vec<Receipt>>>,
    /// List of the transaction logs.
    pub transaction_logs: Mutex<Vec<ExecutionOutcomeWithId>>,
}

impl RuntimeGroup {
    pub fn new_with_account_ids(
        account_ids: Vec<AccountId>,
        num_existing_accounts: u64,
        contract_code: &[u8],
    ) -> Arc<Self> {
        let mut res = Self::default();
        assert!(num_existing_accounts <= account_ids.len() as u64);
        let (state_records, signers, validators) =
            Self::state_records_signers(account_ids, num_existing_accounts, contract_code);
        Arc::make_mut(&mut res.state_records).extend(state_records);

        for signer in signers {
            res.signers.push(signer.clone());
            res.mailboxes.0.lock().insert(signer.get_account_id(), Default::default());
        }
        res.validators = validators;
        Arc::new(res)
    }

    pub fn new(num_runtimes: u64, num_existing_accounts: u64, contract_code: &[u8]) -> Arc<Self> {
        let account_ids = (0..num_runtimes)
            .map(|i| AccountId::try_from(format!("near_{}", i)).unwrap())
            .collect();
        Self::new_with_account_ids(account_ids, num_existing_accounts, contract_code)
    }

    /// Get state records and signers for standalone runtimes.
    fn state_records_signers(
        account_ids: Vec<AccountId>,
        num_existing_accounts: u64,
        contract_code: &[u8],
    ) -> (Vec<StateRecord>, Vec<Signer>, Vec<AccountInfo>) {
        let code_hash = hash(contract_code);
        let mut state_records = vec![];
        let mut signers = vec![];
        let mut validators = vec![];
        for (i, account_id) in account_ids.into_iter().enumerate() {
            let signer = InMemorySigner::test_signer(&account_id);
            if (i as u64) < num_existing_accounts {
                state_records.push(StateRecord::Account {
                    account_id: account_id.clone(),
                    account: Account::new(
                        TESTING_INIT_BALANCE,
                        TESTING_INIT_STAKE,
                        AccountContract::from_local_code_hash(code_hash),
                        0,
                    ),
                });
                state_records.push(StateRecord::AccessKey {
                    account_id: account_id.clone(),
                    public_key: signer.public_key(),
                    access_key: AccessKey::full_access(),
                });
                state_records
                    .push(StateRecord::Contract { account_id, code: contract_code.to_vec() });
                validators.push(AccountInfo {
                    account_id: signer.get_account_id(),
                    public_key: signer.public_key(),
                    amount: TESTING_INIT_STAKE,
                });
            }
            signers.push(signer);
        }
        (state_records, signers, validators)
    }

    pub fn start_runtimes(
        group: Arc<Self>,
        transactions: Vec<SignedTransaction>,
    ) -> Vec<JoinHandle<()>> {
        for transaction in transactions {
            group
                .mailboxes
                .0
                .lock()
                .get_mut(transaction.transaction.signer_id())
                .unwrap()
                .incoming_transactions
                .push(transaction);
        }

        let mut handles = vec![];
        for signer in &group.signers {
            let signer = signer.clone();
            let state_records = Arc::clone(&group.state_records);
            let validators = group.validators.clone();
            let runtime_factory = move || {
                StandaloneRuntime::new(
                    signer,
                    &state_records,
                    TestTriesBuilder::new().build(),
                    validators,
                )
            };
            handles.push(Self::start_runtime_in_thread(group.clone(), runtime_factory));
        }
        handles
    }

    fn start_runtime_in_thread<F>(group: Arc<Self>, runtime_factory: F) -> JoinHandle<()>
    where
        F: FnOnce() -> StandaloneRuntime + Send + 'static,
    {
        thread::spawn(move || {
            let mut runtime = runtime_factory();
            loop {
                let account_id = runtime.account_id();

                let mut mailboxes = group.mailboxes.0.lock();
                loop {
                    if !mailboxes.get(&account_id).unwrap().is_empty() {
                        break;
                    }
                    if mailboxes.values().all(|m| m.is_empty()) {
                        return;
                    }
                    group.mailboxes.1.wait(&mut mailboxes);
                }

                let mailbox = mailboxes.get_mut(&account_id).unwrap();
                group
                    .executed_receipts
                    .lock()
                    .entry(account_id.clone())
                    .or_insert_with(Vec::new)
                    .extend(mailbox.incoming_receipts.clone());
                group
                    .executed_transactions
                    .lock()
                    .entry(account_id.clone())
                    .or_insert_with(Vec::new)
                    .extend(mailbox.incoming_transactions.clone());

                let ProcessBlockOutcome {
                    mut outgoing_receipts,
                    mut execution_outcomes,
                    mut has_queued_receipts,
                } = runtime.process_block(
                    &mailbox.incoming_receipts,
                    mailbox.incoming_transactions.clone(),
                );
                while has_queued_receipts {
                    let process_outcome = runtime.process_block(&[], vec![]);
                    outgoing_receipts.extend(process_outcome.outgoing_receipts);
                    execution_outcomes.extend(process_outcome.execution_outcomes);
                    has_queued_receipts = process_outcome.has_queued_receipts;
                }

                mailbox.incoming_receipts.clear();
                mailbox.incoming_transactions.clear();
                group.transaction_logs.lock().extend(execution_outcomes);
                for outgoing_receipts in outgoing_receipts {
                    let locked_other_mailbox =
                        mailboxes.get_mut(outgoing_receipts.receiver_id()).unwrap();
                    locked_other_mailbox.incoming_receipts.push(outgoing_receipts);
                }
                group.mailboxes.1.notify_all();
            }
        })
    }

    /// Get receipt that was executed by the given runtime based on hash.
    pub fn get_receipt(&self, executing_runtime: &str, hash: &CryptoHash) -> Receipt {
        self.executed_receipts
            .lock()
            .get(AccountIdRef::new_or_panic(executing_runtime))
            .expect("Runtime not found")
            .iter()
            .find_map(|r| if &r.get_hash() == hash { Some(r.clone()) } else { None })
            .expect("Runtime does not contain the receipt with the given hash.")
    }

    /// Get transaction log produced by the execution of given transaction/receipt
    /// identified by `producer_hash`.
    pub fn get_transaction_log(&self, producer_hash: &CryptoHash) -> ExecutionOutcomeWithId {
        self.transaction_logs
            .lock()
            .iter()
            .find_map(|tl| if &tl.id == producer_hash { Some(tl.clone()) } else { None })
            .expect("The execution log of the given receipt is missing")
    }

    pub fn get_receipt_debug(&self, hash: &CryptoHash) -> (AccountId, Receipt) {
        for (executed_runtime, tls) in self.executed_receipts.lock().iter() {
            if let Some(res) =
                tls.iter().find_map(|r| if &r.get_hash() == hash { Some(r.clone()) } else { None })
            {
                return (executed_runtime.clone(), res);
            }
        }
        unimplemented!()
    }
}

#[macro_export]
macro_rules! assert_receipts {
    ($group:ident, $transaction:ident ) => {{
        let transaction_log = $group.get_transaction_log(&$transaction.get_hash());
        transaction_log.outcome.receipt_ids
    }};
    ($group:ident, $from:expr => $receipt:ident @ $to:expr,
    $receipt_pat:pat,
    $receipt_assert:block,
    $actions_name:ident,
    $($action_name:ident, $action_pat:pat, $action_assert:block ),+ ) => {{
        let r = $group.get_receipt($to, $receipt);
        assert_eq!(r.predecessor_id().clone(), $from);
        assert_eq!(r.receiver_id().clone(), $to);
        match r.receipt() {
            $receipt_pat => {
                $receipt_assert
                let [$($action_name),* ] = &$actions_name[..] else { panic!("Incorrect number of actions") };
                $(
                    match $action_name {
                        $action_pat => {
                            $action_assert
                        }
                        _ => panic!("Action {:#?} does not satisfy the pattern {}", $action_name, stringify!($action_pat)),
                    }
                )*
            }
            _ => panic!("Receipt {:#?} does not satisfy the pattern {}", r, stringify!($receipt_pat)),
        }
        let receipt_log = $group.get_transaction_log(&r.get_hash());
        receipt_log.outcome.receipt_ids
    }};
}

/// A short form for refunds.
/// ```
/// assert_refund!(group, ref1 @ "near_0");
/// ```
/// expands into:
/// ```
/// assert_receipts!(group, "system" => ref1 @ "near_0",
///                  ReceiptEnum::Action(ActionReceipt{actions, ..}), {},
///                  actions,
///                  a0, Action::Transfer(TransferAction{..}), {}
///                  );
/// ```
#[macro_export]
macro_rules! assert_refund {
 ($group:ident, $receipt:ident @ $to:expr) => {
        assert_receipts!($group, "system" => $receipt @ $to,
                         ReceiptEnum::Action(ActionReceipt{actions, ..}), {},
                         actions,
                         a0, Action::Transfer(TransferAction{..}), {}
                        );
 }
}

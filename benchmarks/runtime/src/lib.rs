use std::sync::Arc;

use bencher::Bencher;
use integration_tests::node::{Node, RuntimeNode};
use near_chain_configs::{Genesis, NEAR_BASE};
use near_crypto::{KeyType, PublicKey};
use near_parameters::{ActionCosts, RuntimeConfig, RuntimeConfigStore};
use near_primitives::account::id::AccountType;
use near_primitives::transaction::{Action, FunctionCallAction};
use near_primitives::types::{AccountId, Balance};
use near_primitives::views::FinalExecutionOutcomeView;
use node_runtime::config::total_prepaid_gas;
use testlib::{
    fees_utils::FeeHelper,
    runtime_utils::{add_contract, alice_account, bob_account, carol_account},
};

pub fn fee_helper(node: &impl Node) -> FeeHelper {
    let store = RuntimeConfigStore::new(None);
    let config = RuntimeConfig::clone(store.get_config(node.genesis().config.protocol_version));
    FeeHelper::new(config, node.genesis().config.min_gas_price)
}

/// Construct an function call action with a FT transfer.
///
/// Returns the action and the number of bytes for gas charges.
pub fn ft_transfer_action(receiver: &str, amount: u128) -> (Action, u64) {
    let args: Vec<u8> = format!(
        r#"{{
        "receiver_id": "{receiver}",
        "amount": "{amount}"
    }}"#
    )
    .bytes()
    .collect();
    let method_name = "ft_transfer".to_owned();
    let num_bytes = method_name.len() + args.len();
    let action = Action::FunctionCall(Box::new(FunctionCallAction {
        method_name,
        args,
        gas: 20_000_000_000_000,
        deposit: 1,
    }));

    (action, num_bytes as u64)
}

/// Add NEAR token balance to maintain the storage of an account, which
/// registers the user in the fungible contract account.
pub fn ft_register_action(receiver: &str) -> Action {
    let args: Vec<u8> = format!(
        r#"{{
        "account_id": "{receiver}"
    }}"#
    )
    .bytes()
    .collect();
    Action::FunctionCall(Box::new(FunctionCallAction {
        method_name: "storage_deposit".to_owned(),
        args,
        gas: 20_000_000_000_000,
        deposit: NEAR_BASE,
    }))
}

/// Take a list of actions and execute them as a meta transaction, check
/// everything executes successfully, return balance differences for the sender,
/// relayer, and receiver.
///
/// This is a common checker function used by the tests below.
fn check_meta_tx_execution(
    node: &impl Node,
    actions: Vec<Action>,
    sender: AccountId,
    relayer: AccountId,
    receiver: AccountId,
) -> (FinalExecutionOutcomeView, i128, i128, i128) {
    let node_user = node.user();

    assert_eq!(
        relayer,
        node.account_id().unwrap(),
        "the relayer must be the signer in meta transactions"
    );

    let sender_before = node_user.view_balance(&sender).unwrap();
    let relayer_before = node_user.view_balance(&relayer).unwrap();
    let receiver_before = node_user.view_balance(&receiver).unwrap_or(0);
    let relayer_nonce_before = node_user
        .get_access_key(&relayer, &PublicKey::from_seed(KeyType::ED25519, relayer.as_ref()))
        .unwrap()
        .nonce;
    let user_pub_key = match sender.get_account_type() {
        AccountType::NearImplicitAccount => PublicKey::from_near_implicit_account(&sender).unwrap(),
        AccountType::EthImplicitAccount => {
            panic!("ETH-implicit accounts must not have access key");
        }
        AccountType::NamedAccount => PublicKey::from_seed(KeyType::ED25519, sender.as_ref()),
    };
    let user_nonce_before = node_user.get_access_key(&sender, &user_pub_key).unwrap().nonce;

    let tx_result =
        node_user.meta_tx(sender.clone(), receiver.clone(), relayer.clone(), actions).unwrap();

    // Execution of the transaction and all receipts should succeed
    tx_result.assert_success();

    // both nonces should be increased by 1
    let relayer_nonce = node_user
        .get_access_key(&relayer, &PublicKey::from_seed(KeyType::ED25519, relayer.as_ref()))
        .unwrap()
        .nonce;
    assert_eq!(relayer_nonce, relayer_nonce_before + 1);
    // user key must be checked for existence (to test DeleteKey action)
    if let Ok(user_nonce) = node_user
        .get_access_key(&sender, &PublicKey::from_seed(KeyType::ED25519, sender.as_ref()))
        .map(|key| key.nonce)
    {
        assert_eq!(user_nonce, user_nonce_before + 1);
    }

    let sender_after = node_user.view_balance(&sender).unwrap_or(0);
    let relayer_after = node_user.view_balance(&relayer).unwrap_or(0);
    let receiver_after = node_user.view_balance(&receiver).unwrap_or(0);

    let sender_diff = sender_after as i128 - sender_before as i128;
    let relayer_diff = relayer_after as i128 - relayer_before as i128;
    let receiver_diff = receiver_after as i128 - receiver_before as i128;
    (tx_result, sender_diff, relayer_diff, receiver_diff)
}

/// Call `check_meta_tx_execution` and perform gas checks specific to function calls.
///
/// This is a common checker function used by the tests below.
/// It works for action lists that consists multiple function calls but adding
/// other action will mess up the gas checks.
pub fn check_meta_tx_fn_call(
    node: &impl Node,
    actions: Vec<Action>,
    msg_len: u64,
    tokens_transferred: Balance,
    sender: AccountId,
    relayer: AccountId,
    receiver: AccountId,
) -> FinalExecutionOutcomeView {
    let fee_helper = fee_helper(node);
    let num_fn_calls = actions.len();
    let meta_tx_overhead_cost = fee_helper.meta_tx_overhead_cost(&actions, &receiver);
    let prepaid_gas = total_prepaid_gas(&actions).unwrap();

    let (tx_result, sender_diff, relayer_diff, receiver_diff) =
        check_meta_tx_execution(node, actions, sender, relayer, receiver);

    assert_eq!(sender_diff, 0, "sender should not pay for anything");

    // Assertions on receiver and relayer are tricky because of dynamic gas
    // costs and contract reward. We need to check in the function call receipt
    // how much gas was spent and subtract the base cost that is not part of the
    // dynamic cost. The contract reward can be inferred from that.

    // static send gas is paid and burnt upfront
    let static_send_gas = fee_helper.cfg().fee(ActionCosts::new_action_receipt).send_fee(false)
        + num_fn_calls as u64
            * fee_helper.cfg().fee(ActionCosts::function_call_base).send_fee(false)
        + msg_len * fee_helper.cfg().fee(ActionCosts::function_call_byte).send_fee(false);
    // static execution gas burnt in the same receipt as the function calls but
    // it doesn't contribute to the contract reward
    let static_exec_gas = fee_helper.cfg().fee(ActionCosts::new_action_receipt).exec_fee()
        + num_fn_calls as u64 * fee_helper.cfg().fee(ActionCosts::function_call_base).exec_fee()
        + msg_len * fee_helper.cfg().fee(ActionCosts::function_call_byte).exec_fee();

    // calculate contract rewards as reward("gas burnt in fn call receipt" - "static exec costs")
    let gas_burnt_for_function_call =
        tx_result.receipts_outcome[1].outcome.gas_burnt - static_exec_gas;
    let dyn_cost = fee_helper.gas_to_balance(gas_burnt_for_function_call);
    let contract_reward = fee_helper.gas_burnt_to_reward(gas_burnt_for_function_call);

    // Calculate cost of gas refund
    let gross_gas_refund = prepaid_gas - gas_burnt_for_function_call;
    let refund_penalty = fee_helper.gas_refund_cost(gross_gas_refund);

    // the relayer pays all gas and tokens
    let gas_cost = meta_tx_overhead_cost
        + refund_penalty
        + fee_helper.gas_to_balance(static_exec_gas + static_send_gas);
    let expected_relayer_cost = (gas_cost + tokens_transferred + dyn_cost) as i128;
    assert_eq!(relayer_diff, -expected_relayer_cost, "unexpected relayer balance");

    // the receiver gains transferred tokens and the contract reward
    let expected_receiver_gain = (tokens_transferred + contract_reward) as i128;
    assert_eq!(receiver_diff, expected_receiver_gain, "unexpected receiver balance");

    tx_result
}

pub fn bench_ft_transfer(
    bench: &mut Bencher,
    wasm_config: impl Into<Arc<near_parameters::vm::Config>>,
) {
    let relayer = alice_account();
    let sender = bob_account();
    let ft_contract = carol_account();
    let receiver = "david.near";

    let mut genesis = Genesis::test(vec![alice_account(), bob_account(), carol_account()], 3);
    add_contract(&mut genesis, &ft_contract, near_test_contracts::ft_contract().to_vec());

    let mut config = {
        let store = RuntimeConfigStore::new(None);
        RuntimeConfig::clone(store.get_config(genesis.config.protocol_version))
    };
    config.wasm_config = wasm_config.into();
    let node = RuntimeNode::new_from_genesis_and_config(&relayer, genesis, config);

    // A BUNCH OF TEST SETUP
    // initialize the contract
    node.user()
        .function_call(
            relayer.clone(),
            ft_contract.clone(),
            "new_default_meta",
            // make the relayer (alice) owner, makes initialization easier
            br#"{"owner_id": "alice.near", "total_supply": "1000000"}"#.to_vec(),
            30_000_000_000_000,
            0,
        )
        .expect("FT contract initialization failed")
        .assert_success();

    // register sender & receiver FT accounts
    let actions = vec![ft_register_action(sender.as_ref()), ft_register_action(&receiver)];
    node.user()
        .sign_and_commit_actions(relayer.clone(), ft_contract.clone(), actions)
        .expect("registering FT accounts")
        .assert_success();
    // initialize sender balance
    let actions = vec![ft_transfer_action(sender.as_ref(), 10_000).0];
    node.user()
        .sign_and_commit_actions(relayer.clone(), ft_contract.clone(), actions)
        .expect("initializing sender balance failed")
        .assert_success();

    bench.iter(|| {
        // START OF META TRANSACTION
        // 1% fee to the relayer
        let (action0, bytes0) = ft_transfer_action(relayer.as_ref(), 10);
        // the actual transfer
        let (action1, bytes1) = ft_transfer_action(receiver, 1000);
        let actions = vec![action0, action1];

        check_meta_tx_fn_call(
            &node,
            actions,
            bytes0 + bytes1,
            2,
            sender.clone(),
            relayer.clone(),
            ft_contract.clone(),
        )
        .assert_success();
    });
}

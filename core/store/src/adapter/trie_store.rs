use super::{StoreAdapter, StoreUpdateAdapter, StoreUpdateHolder};
use crate::db::DBSlice;
use crate::{DBCol, KeyForStateChanges, STATE_SNAPSHOT_KEY, Store, StoreUpdate, TrieChanges};
use borsh::BorshDeserialize;
use near_primitives::errors::{MissingTrieValue, MissingTrieValueContext, StorageError};
use near_primitives::hash::CryptoHash;
use near_primitives::shard_layout::{ShardUId, get_block_shard_uid};
use near_primitives::types::RawStateChangesWithTrieKey;
use std::io;
use std::num::NonZero;
use std::sync::Arc;

#[derive(Clone)]
pub struct TrieStoreAdapter {
    store: Store,
}

impl StoreAdapter for TrieStoreAdapter {
    fn store_ref(&self) -> &Store {
        &self.store
    }
}

impl TrieStoreAdapter {
    pub fn new(store: Store) -> Self {
        Self { store }
    }

    pub fn store_update(&self) -> TrieStoreUpdateAdapter<'static> {
        TrieStoreUpdateAdapter { store_update: StoreUpdateHolder::Owned(self.store.store_update()) }
    }

    /// Here we are first trying to get the value with shard_uid as prefix.
    /// If that fails, we try to get the value with the mapped shard_uid as prefix.
    ///
    /// Note we should not first get the mapping and then try to get the value with the mapped
    /// shard_uid as prefix. This doesn't work with SplitDB, which has hot and cold stores.
    ///
    /// Since the mapping is deleted from the hot store after resharding is completed but never
    /// from the cold store, SplitDB (archive nodes) would return the mapping from the cold store,
    /// which can lead to a MissingTrieValue, for example when the value is in the hot store but not
    /// yet copied to the cold store.
    fn get_ref(&self, shard_uid: ShardUId, hash: &CryptoHash) -> Result<DBSlice<'_>, StorageError> {
        match self.get_ref_inner(shard_uid, hash) {
            Ok(value) => Ok(value),
            Err(err) => match maybe_get_shard_uid_mapping(&self.store, shard_uid) {
                Some(mapped_shard_uid) => self.get_ref_inner(mapped_shard_uid, hash),
                None => Err(err),
            },
        }
    }

    fn get_ref_inner(
        &self,
        shard_uid: ShardUId,
        hash: &CryptoHash,
    ) -> Result<DBSlice<'_>, StorageError> {
        let key = get_key_from_shard_uid_and_hash(shard_uid, hash);
        self.store
            .get(DBCol::State, key.as_ref())
            .map_err(|_| StorageError::StorageInternalError)?
            .ok_or(StorageError::MissingTrieValue(MissingTrieValue {
                context: MissingTrieValueContext::TrieStorage,
                hash: *hash,
            }))
    }

    /// Replaces shard_uid prefix with a mapped value according to mapping strategy in Resharding V3.
    /// For this, it does extra read from `DBCol::StateShardUIdMapping`.
    ///
    /// For more details, see `get_shard_uid_mapping()`.
    pub fn get(&self, shard_uid: ShardUId, hash: &CryptoHash) -> Result<Arc<[u8]>, StorageError> {
        let val = self.get_ref(shard_uid, hash)?;
        Ok(val.into())
    }

    pub fn get_ser<T: BorshDeserialize>(
        &self,
        shard_uid: ShardUId,
        hash: &CryptoHash,
    ) -> Result<T, StorageError> {
        let val = self.get_ref(shard_uid, hash)?;
        T::try_from_slice(&val).map_err(|e| StorageError::StorageInconsistentState(e.to_string()))
    }

    pub fn get_state_snapshot_hash(&self) -> Result<CryptoHash, StorageError> {
        let val = self
            .store
            .get_ser(DBCol::BlockMisc, STATE_SNAPSHOT_KEY)
            .map_err(|_| StorageError::StorageInternalError)?
            .ok_or(StorageError::StorageInternalError)?;
        Ok(val)
    }

    #[cfg(test)]
    pub fn iter_raw_bytes(&self) -> crate::db::DBIterator {
        self.store.iter_raw_bytes(DBCol::State)
    }
}

pub struct TrieStoreUpdateAdapter<'a> {
    store_update: StoreUpdateHolder<'a>,
}

impl Into<StoreUpdate> for TrieStoreUpdateAdapter<'static> {
    fn into(self) -> StoreUpdate {
        self.store_update.into()
    }
}

impl TrieStoreUpdateAdapter<'static> {
    pub fn commit(self) -> io::Result<()> {
        let store_update: StoreUpdate = self.into();
        store_update.commit()
    }
}

impl<'a> StoreUpdateAdapter for TrieStoreUpdateAdapter<'a> {
    fn store_update(&mut self) -> &mut StoreUpdate {
        &mut self.store_update
    }
}

impl<'a> TrieStoreUpdateAdapter<'a> {
    pub fn new(store_update: &'a mut StoreUpdate) -> Self {
        Self { store_update: StoreUpdateHolder::Reference(store_update) }
    }

    pub fn decrement_refcount_by(
        &mut self,
        shard_uid: ShardUId,
        hash: &CryptoHash,
        decrement: NonZero<u32>,
    ) {
        // For Resharding V3, along with decrementing the refcount of the child shard_uid, we also need to
        // decrement the refcount of the parent shard_uid, if it exists.
        let mapped_shard_uid = maybe_get_shard_uid_mapping(&self.store_update.store, shard_uid);
        for shard_uid in [shard_uid].into_iter().chain(mapped_shard_uid.into_iter()) {
            let key = get_key_from_shard_uid_and_hash(shard_uid, hash);
            self.store_update.decrement_refcount_by(DBCol::State, key.as_ref(), decrement);
        }
    }

    pub fn decrement_refcount(&mut self, shard_uid: ShardUId, hash: &CryptoHash) {
        self.decrement_refcount_by(shard_uid, hash, NonZero::new(1).unwrap());
    }

    pub fn increment_refcount_by(
        &mut self,
        shard_uid: ShardUId,
        hash: &CryptoHash,
        data: &[u8],
        increment: NonZero<u32>,
    ) {
        // For Resharding V3, along with incrementing the refcount of the child shard_uid, we also need to
        // increment the refcount of the parent shard_uid, if it exists.
        let mapped_shard_uid = maybe_get_shard_uid_mapping(&self.store_update.store, shard_uid);
        for shard_uid in [shard_uid].into_iter().chain(mapped_shard_uid.into_iter()) {
            let key = get_key_from_shard_uid_and_hash(shard_uid, hash);
            self.store_update.increment_refcount_by(DBCol::State, key.as_ref(), data, increment);
        }
    }

    pub fn set_state_snapshot_hash(&mut self, hash: Option<CryptoHash>) {
        let key = STATE_SNAPSHOT_KEY;
        match hash {
            Some(hash) => self.store_update.set_ser(DBCol::BlockMisc, key, &hash).unwrap(),
            None => self.store_update.delete(DBCol::BlockMisc, key),
        }
    }

    pub fn set_trie_changes(
        &mut self,
        shard_uid: ShardUId,
        block_hash: &CryptoHash,
        trie_changes: &TrieChanges,
    ) {
        let key = get_block_shard_uid(block_hash, &shard_uid);
        self.store_update.set_ser(DBCol::TrieChanges, &key, trie_changes).unwrap();
    }

    pub fn set_state_changes(
        &mut self,
        key: KeyForStateChanges,
        value: &RawStateChangesWithTrieKey,
    ) {
        self.store_update.set(
            DBCol::StateChanges,
            key.as_ref(),
            &borsh::to_vec(&value).expect("Borsh serialize cannot fail"),
        )
    }

    /// Set the mapping from `child_shard_uid` to `parent_shard_uid`.
    /// Used by Resharding V3 for State mapping.
    pub fn set_shard_uid_mapping(&mut self, child_shard_uid: ShardUId, parent_shard_uid: ShardUId) {
        self.store_update.set(
            DBCol::StateShardUIdMapping,
            child_shard_uid.to_bytes().as_ref(),
            &borsh::to_vec(&parent_shard_uid).expect("Borsh serialize cannot fail"),
        )
    }

    pub fn delete_shard_uid_mapping(&mut self, child_shard_uid: ShardUId) {
        self.store_update.delete(DBCol::StateShardUIdMapping, child_shard_uid.to_bytes().as_ref());
    }

    /// Remove State of any shard that uses `shard_uid_db_key_prefix` as database key prefix.
    /// That is potentially State of any descendant of the shard with the given `ShardUId`.
    /// Use with caution, as it might potentially remove the State of a descendant shard that is still in use!
    pub fn delete_shard_uid_prefixed_state(&mut self, shard_uid_db_key_prefix: ShardUId) {
        let key_from = shard_uid_db_key_prefix.to_bytes();
        let key_to = ShardUId::get_upper_bound_db_key(&key_from);
        self.store_update.delete_range(DBCol::State, &key_from, &key_to);
    }

    pub fn delete_all_state(&mut self) {
        self.store_update.delete_all(DBCol::State)
    }
}

/// Get the `ShardUId` mapping for child_shard_uid. If the mapping does not exist, map the shard to itself.
/// Used by Resharding V3 for State mapping.
///
/// It is kept out of `TrieStoreAdapter`, so that `TrieStoreUpdateAdapter` can use it without
/// cloning `store` each time, see https://github.com/near/nearcore/pull/12232#discussion_r1804810508.
pub fn get_shard_uid_mapping(store: &Store, child_shard_uid: ShardUId) -> ShardUId {
    maybe_get_shard_uid_mapping(store, child_shard_uid).unwrap_or(child_shard_uid)
}

/// Get the `ShardUId` mapping for child_shard_uid. If the mapping does not exist, return None.
fn maybe_get_shard_uid_mapping(store: &Store, child_shard_uid: ShardUId) -> Option<ShardUId> {
    store
        .caching_get_ser::<ShardUId>(DBCol::StateShardUIdMapping, &child_shard_uid.to_bytes())
        .unwrap_or_else(|_| {
            panic!("get_shard_uid_mapping() failed for child_shard_uid = {}", child_shard_uid)
        })
        .map(|v| *v)
}

/// Get the key for the given `shard_uid` and `hash`.
/// The key is a 40-byte array, where the first 8 bytes are the `shard_uid` and the last 32 bytes are the `hash`.
fn get_key_from_shard_uid_and_hash(shard_uid: ShardUId, hash: &CryptoHash) -> [u8; 40] {
    let mut key = [0; 40];
    key[0..8].copy_from_slice(&shard_uid.to_bytes());
    key[8..].copy_from_slice(hash.as_ref());
    key
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use near_primitives::errors::{MissingTrieValue, StorageError};
    use near_primitives::hash::CryptoHash;
    use near_primitives::shard_layout::ShardUId;

    use crate::adapter::StoreAdapter;
    use crate::db::metadata::{DB_VERSION, DbKind};
    use crate::test_utils::{create_test_node_storage_with_cold, create_test_store};

    const ONE: std::num::NonZeroU32 = match std::num::NonZeroU32::new(1) {
        Some(num) => num,
        None => panic!(),
    };

    #[test]
    fn test_trie_store_adapter() {
        let store = create_test_store().trie_store();
        let shard_uids: Vec<ShardUId> =
            (0..3).map(|i| ShardUId { version: 0, shard_id: i }).collect();
        let dummy_hash = CryptoHash::default();

        assert_matches!(
            store.get(shard_uids[0], &dummy_hash),
            Err(StorageError::MissingTrieValue(MissingTrieValue { context: _, hash: _ }))
        );
        {
            let mut store_update = store.store_update();
            store_update.increment_refcount_by(shard_uids[0], &dummy_hash, &[0], ONE);
            store_update.increment_refcount_by(shard_uids[1], &dummy_hash, &[1], ONE);
            store_update.increment_refcount_by(shard_uids[2], &dummy_hash, &[2], ONE);
            store_update.commit().unwrap();
        }
        assert_eq!(*store.get(shard_uids[0], &dummy_hash).unwrap(), [0]);
        {
            let mut store_update = store.store_update();
            store_update.delete_all_state();
            store_update.commit().unwrap();
        }
        assert_matches!(
            store.get(shard_uids[0], &dummy_hash),
            Err(StorageError::MissingTrieValue(_))
        );
    }

    #[test]
    fn test_shard_uid_mapping() {
        let store = create_test_store().trie_store();
        let parent_shard = ShardUId { version: 0, shard_id: 0 };
        let child_shard = ShardUId { version: 0, shard_id: 1 };
        let dummy_hash = CryptoHash::default();
        // Write some data to `parent_shard`.
        {
            let mut store_update = store.store_update();
            store_update.increment_refcount_by(parent_shard, &dummy_hash, &[0], ONE);
            store_update.commit().unwrap();
        }
        // The data is not yet visible to child shard, because the mapping has not been set yet.
        assert_matches!(
            store.get(child_shard, &dummy_hash),
            Err(StorageError::MissingTrieValue(_))
        );
        // Set the shard_uid mapping from `child_shard` to `parent_shard`.
        {
            let mut store_update = store.store_update();
            store_update.set_shard_uid_mapping(child_shard, parent_shard);
            store_update.commit().unwrap();
        }
        // The data is now visible to both `parent_shard` and `child_shard`.
        assert_eq!(*store.get(child_shard, &dummy_hash).unwrap(), [0]);
        assert_eq!(*store.get(parent_shard, &dummy_hash).unwrap(), [0]);
        // Remove the data using `parent_shard` UId.
        {
            let mut store_update = store.store_update();
            store_update.decrement_refcount(parent_shard, &dummy_hash);
            store_update.commit().unwrap();
        }
        // The data is now not visible to any shard.
        assert_matches!(
            store.get(child_shard, &dummy_hash),
            Err(StorageError::MissingTrieValue(_))
        );
        assert_matches!(
            store.get(parent_shard, &dummy_hash),
            Err(StorageError::MissingTrieValue(_))
        );
        // Restore the data now using the `child_shard` UId.
        {
            let mut store_update = store.store_update();
            store_update.increment_refcount_by(child_shard, &dummy_hash, &[0], ONE);
            store_update.commit().unwrap();
        }
        // The data is now visible to both shards again.
        assert_eq!(*store.get(child_shard, &dummy_hash).unwrap(), [0]);
        assert_eq!(*store.get(parent_shard, &dummy_hash).unwrap(), [0]);
        // Remove the data using `child_shard` UId.
        {
            let mut store_update = store.store_update();
            store_update.decrement_refcount_by(child_shard, &dummy_hash, ONE);
            store_update.commit().unwrap();
        }
        // The data is not visible to any shard again.
        assert_matches!(
            store.get(child_shard, &dummy_hash),
            Err(StorageError::MissingTrieValue(_))
        );
    }

    // Simulate a scenario where we have an archival node with split store configured.
    // A resharding has recently completed so the hot_store doesn't have the shard_uid mapping
    // but the cold_store does.
    // The data we are trying to read is currently in the hot store but cold store loop has not yet
    // copied it to the cold store.
    #[test]
    fn test_split_store_shard_uid_mapping() {
        let (storage, ..) = create_test_node_storage_with_cold(DB_VERSION, DbKind::Hot);
        let hot_store = storage.get_hot_store().trie_store();
        let cold_store = storage.get_cold_store().unwrap().trie_store();
        let store = storage.get_split_store().unwrap().trie_store();

        let parent_shard = ShardUId { version: 0, shard_id: 0 };
        let child_shard = ShardUId { version: 0, shard_id: 1 };
        let dummy_hash = CryptoHash::default();

        // Set the shard_uid mapping from `child_shard` to `parent_shard` in cold store ONLY.
        {
            let mut store_update = cold_store.store_update();
            store_update.set_shard_uid_mapping(child_shard, parent_shard);
            store_update.commit().unwrap();
        }
        // Write some data to `child_shard` in hot store ONLY.
        {
            let mut store_update = hot_store.store_update();
            store_update.increment_refcount_by(child_shard, &dummy_hash, &[0], ONE);
            store_update.commit().unwrap();
        }

        // Now try to read the data from split store. It should be present
        assert_eq!(*store.get(child_shard, &dummy_hash).unwrap(), [0]);
    }
}

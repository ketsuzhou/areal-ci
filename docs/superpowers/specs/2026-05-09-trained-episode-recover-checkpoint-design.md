# Trained Episode ID Tracking in Recover Checkpoint

**Date**: 2026-05-09

## Problem

When `CacheAwarePPOTrainer` resumes from a checkpoint, `_init_tree_components()` calls
`reset_trained_flags()`, which sets **all** nodes' trained flags to `False`. This causes
nodes that were already trained to be retrained — wasting compute and potentially
degrading model quality.

The tree checkpoint (`mcts_trees/`) does save `_trained` flags and is loaded in
`CROSS_TRAINING` mode, but `reset_trained_flags()` wipes them immediately after.

## Solution

Track trained episode IDs in a `trained_episodes.json` sidecar file within the recover
checkpoint directory. On resume, restore trained flags from this file instead of
resetting all flags.

The recover checkpoint's trained episode list is the single source of truth — the tree
checkpoint's `_trained` flags are ignored on resume.

## Design

### File format

`trained_episodes.json` in the recover checkpoint directory:

```json
{
  "trained_episode_ids": ["query_42", "query_43_0_abcd1234"]
}
```

### File location

```
{recover_checkpoint_dir}/trained_episodes.json
```

where
`recover_checkpoint_dir = Saver.get_recover_checkpoint_path(     experiment_name, trial_name, fileroot)`.

### Code changes

All changes are in `customized_areal/tree_search/`. No changes to core `areal/`.

#### 1. `trainer.py` — `_save_recover_checkpoint()`

After saving the tree checkpoint (existing logic), collect trained episode IDs from
`self.tree_store` and write them via `TreeCheckpointManager.save_trained_episodes()`.

```python
def _save_recover_checkpoint(self, epoch, epoch_step, global_step):
    super()._save_recover_checkpoint(epoch, epoch_step, global_step)
    if self.cache_config.enabled and self.tree_backup_config.mode == CacheMode.CROSS_TRAINING:
        self.tree_checkpoint_manager.save(self.tree_store)
        # NEW: save trained episode IDs to recover checkpoint dir
        recover_dir = Saver.get_recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.cluster.fileroot,
        )
        TreeCheckpointManager.save_trained_episodes(recover_dir, self.tree_store)
```

#### 2. `trainer.py` — `_init_tree_components()`

Replace unconditional `reset_trained_flags()` with:

1. Compute the recover checkpoint directory path.
1. Try to load `trained_episodes.json` via
   `TreeCheckpointManager.load_trained_episodes()`.
1. If loaded successfully, call `self.tree_store.mark_episodes_trained(episode_ids)` —
   this marks only those episodes' nodes as trained and sets all others to `False`.
1. If the file doesn't exist (fresh start), fall back to `reset_trained_flags()`.

```python
# In _init_tree_components(), replace:
#   self.tree_store.reset_trained_flags()
# with:
recover_dir = Saver.get_recover_checkpoint_path(
    self.config.experiment_name,
    self.config.trial_name,
    self.config.cluster.fileroot,
)
trained_episodes = TreeCheckpointManager.load_trained_episodes(recover_dir)
if trained_episodes is not None:
    self.tree_store.mark_episodes_trained(trained_episodes)
    logger.info(
        f"Restored trained flags for {len(trained_episodes)} episodes "
        f"from recover checkpoint"
    )
else:
    self.tree_store.reset_trained_flags()
```

#### 3. `checkpoint.py` — `TreeCheckpointManager`

Add two static methods:

- `save_trained_episodes(recover_checkpoint_dir, tree_store)`: Collect episode_ids from
  nodes where `tree_store._trained[node_id] is True`, write to `trained_episodes.json`
  using atomic write (`.tmp` + `os.replace`).

- `load_trained_episodes(recover_checkpoint_dir) -> set[str] | None`: Read
  `trained_episodes.json`, return the set of episode_ids. Return `None` if the file
  doesn't exist or is corrupt.

#### 4. `mcts_tree_store.py` — `MCTSTreeStore`

Add one method:

- `mark_episodes_trained(episode_ids: set[str])`: For each node in `_node_id_to_key`,
  check if its `episode_id` (retrievable via the node's record in `trajectories`) is in
  the given set. Set `_trained[node_id] = True` for matches, `False` for non-matches.

  Implementation approach: iterate over `trajectories` (keyed by query_id), check each
  record's `episode_id`, and set the trained flag on matching `node_id`s. Reset all
  other flags to `False`.

### Error handling

- **Missing or corrupt `trained_episodes.json`**: Fall back to `reset_trained_flags()`.
  Safe default — worst case is retraining some nodes.
- **Episode ID not in tree store**: Skip silently. The cache may have been cleared
  between runs.
- **Atomic writes**: Use `.tmp` file + `os.replace()`, consistent with existing
  `TreeCheckpointManager.save()`.

### What stays the same

- The tree checkpoint (`mcts_trees/`) continues to save `_trained` flags in its
  `metadata.json` as before — no change to existing tree checkpoint logic.
- `RecoverInfo`, `RecoverHandler`, and base `PPOTrainer` are not modified.
- The tree checkpoint save/load in `_cache_aware_prepare_batch()` is unchanged.

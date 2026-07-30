# Trainer Readability & Logging Improvements Design

**Date**: 2026-04-28 **Scope**: `customized_areal/tree_search/trainer.py`, `config.py`,
`__init__.py`, tests, README

## Goal

Improve readability and debug efficiency of the `CacheAwarePPOTrainer` by:

1. Removing replay mode code entirely
1. Decomposing long methods, cleaning up naming, adding inline comments and improved
   docstrings
1. Adding tiered logging (INFO for milestones, DEBUG for per-trajectory details)

## Section 1: Replay Mode Removal

Remove all replay-specific code from:

- **`config.py`**: Remove `replay: bool = False` field from `RolloutCacheConfig`
- **`trainer.py`**:
  - Remove `getattr(tree_store, "_replay_mode", False)` guard in
    `patch_ppo_actor_for_tree_backup` (step 5 — `record_training_step` always runs)
  - Remove `self.cache_config.replay` branch in `__init__` (already removed in
    uncommitted changes)
  - Remove `_load_untrained_from_tree_store`, `_generate_from_dataloader`,
    `_replay_prepare_batch` methods (already removed in uncommitted changes)
  - Remove `replay` references from docstrings
- **`tests/test_tree_search/test_cache_trainer.py`**: Remove test classes:
  - `TestLoadUntrainedFromTreeStore`
  - `TestGenerateFromDataloader`
  - `TestReplayPrepareBatchFallback`
  - `TestReplayFallbackProgression`
  - `TestReplayTrainCleanup`
- **`README.md`**: Remove replay mode section, update data flow diagrams and component
  reference table

## Section 2: Readability Improvements

### 2a. Method Decomposition

Break `CacheAwarePPOTrainer.__init__` into focused helpers:

```python
def __init__(self, config, cache_config, tree_backup_config, ...):
    self.cache_config = cache_config or RolloutCacheConfig()
    self.tree_backup_config = tree_backup_config or TreeBackupConfig()
    super().__init__(config, train_dataset, valid_dataset)
    if self.cache_config.enabled and self.tree_backup_config.mode != TreeBackupMode.OFF:
        self._init_tree_components()
        self._init_patches()
        logger.info(...)

def _init_tree_components(self):
    """Create tree store, advantage computer, and checkpoint manager."""
    turn_splitter = make_turn_splitter(...)
    self.tree_store = MCTSTreeStore(turn_splitter)
    self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
    self.tree_checkpoint_manager = TreeCheckpointManager(...)
    if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
        if self.tree_checkpoint_manager.exists():
            self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
            logger.info("Loaded MCTS tree checkpoint with cached rollouts")
    self.tree_store.reset_trained_flags()
    self._batch_builder = _CacheAwareBatchBuilder(...)

def _init_patches(self):
    """Apply monkey-patches for tree backup and query_id injection."""
    patch_ppo_actor_for_tree_backup(...)
    _patch_wrap_openai_agent_for_query_id(self.actor)
```

### 2b. Naming Cleanup

- `_mark_batch_trained` (free function) and `_mark_trajectories_trained` (instance
  method) do the same thing. Keep the free function `_mark_batch_trained` (used by the
  patched closure). Remove the instance method `_mark_trajectories_trained` — verify it
  is not called externally. If it has no external callers, delete it.

### 2c. Inline Comments

Add short comments explaining non-obvious invariants:

- **Monkey-patching over subclassing**: The patch modifies `PPOActor.compute_advantages`
  at the class level so all PPOActor instances (including those created by the base
  PPOTrainer) use the tree backup version. A subclass override would only apply if we
  also subclassed the actor.
- **Why concat_padded_tensors is avoided**: `concat_padded_tensors` keeps only the first
  dict's value for non-tensor, non-list keys, which would lose per-trajectory
  `_mcts_query_id` and `_mcts_seq_id`.
- **advantage_mode == TREE vs GAE**: In TREE mode, tree Q-values replace GAE advantages.
  In GAE mode, trajectories are still inserted into the tree (for caching and MCTS
  statistics) but the original GAE advantages are preserved.
- **query_id derivation fallback chain**: `split_prompts` prefers dataset `query_id`
  string (from `prompt["query_id"]`), falls back to `prompt["_mcts_query_id"]`, then to
  MD5 hash from tokenizer via `get_query_id_from_messages`.
- **\_split_grouped_trajectories batch_size == 1 fast path**: When batch_size == 1 the
  trajectory is already individual — appending as-is avoids unnecessary tensor slicing.

### 2d. Docstring Improvements

Improve docstrings to document side effects and preconditions:

- `patch_ppo_actor_for_tree_backup`: Document that it modifies
  `PPOActor.compute_advantages` at the class level (affects all instances), is
  idempotent (won't stack patches), and must be cleaned up via `unpatch_ppo_actor()`.
- `_cache_aware_prepare_batch`: Document the "all-or-nothing" cache strategy — if any
  prompt lacks sufficient cache, all prompts are regenerated. Document return type (list
  of per-sample dicts with shape \[1, seq_len\]).
- `_CacheAwareBatchBuilder.split_prompts`: Document the query_id derivation fallback
  chain and the structure of returned dicts.
- `CacheAwarePPOTrainer.train`: Document that `prepare_batch` is monkey-patched during
  training and restored in the `finally` block, so the patch never leaks on error.

## Section 3: Tiered Logging

### INFO Level (Milestones)

| Location                     | Message                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| `__init__`                   | `Cache-aware training enabled (mode=X, advantage=Y, n_samples=Z)` |
| `_init_tree_components`      | `Loaded MCTS tree checkpoint with cached rollouts`                |
| `_init_patches`              | `Patched compute_advantages for tree backup (advantage_mode=X)`   |
| `_init_patches`              | `Patched _wrap_openai_agent to use QueryIDProxyWorkflow`          |
| `_cache_aware_prepare_batch` | `Cache-aware rollout: N cached (all from cache)`                  |
| `_cache_aware_prepare_batch` | `Cache-aware rollout: 0 cached, N newly generated`                |
| `_cache_aware_prepare_batch` | `Cache-aware rollout: N cached + M newly generated`               |
| `_save_recover_checkpoint`   | `Saved MCTS tree checkpoint with rollout cache`                   |
| `train()` finally            | `Restored original prepare_batch`                                 |

### DEBUG Level (Per-trajectory Details)

| Location                          | Message                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| `_tree_backup_compute_advantages` | `Step A: GAE completed for N trajectories`                              |
| `_tree_backup_compute_advantages` | `Step B: Inserted N trajectories into tree (K new queries, M existing)` |
| `_tree_backup_compute_advantages` | `Step C: Computed tree advantages for N trajectories (mode=X)`          |
| `_tree_backup_compute_advantages` | `Step D: Marked N trajectories as trained`                              |
| `_mark_batch_trained`             | `Marked trained: query_id=X, seq_id=Y` (one line per pair, in batch)    |
| `split_prompts`                   | `Prompt query_id=X: Y untrained (need Z, have W)`                       |
| `_split_grouped_trajectories`     | `Split grouped trajectory (batch_size=N) into N individual items`       |

### Notes

- Don't log per-token in hot paths
- Batch DEBUG logs where possible (e.g., mark trained: log count or batch summary, not
  one line per trajectory in a tight loop)
- Keep existing WARNING logs (engine missing `_wrap_openai_agent`)

## Files Changed

| File                                           | Change                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------- |
| `customized_areal/tree_search/config.py`       | Remove `replay` field                                                         |
| `customized_areal/tree_search/trainer.py`      | Remove replay, decompose __init__, clean naming, add comments/docstrings/logs |
| `customized_areal/tree_search/__init__.py`     | No changes needed                                                             |
| `tests/test_tree_search/test_cache_trainer.py` | Remove replay test classes                                                    |
| `customized_areal/tree_search/README.md`       | Remove replay sections, update docs                                           |

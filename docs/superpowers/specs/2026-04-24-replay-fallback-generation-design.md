# Replay Fallback Generation Design

**Date**: 2026-04-24

## Problem

`_replay_prepare_batch` only loads trajectories from recorded `_training_history`. As
training progresses, paths get marked as trained and consumed. Eventually the tree has
no untrained paths left, and replay history may be exhausted. At that point, the method
returns `[]`, stalling training. The method should also generate new paths when
cached/replay paths are insufficient.

## Design

### 3-Level Fallback in `_replay_prepare_batch`

Each training step tries three sources in order:

1. **Replay from training history**: Load trajectories from
   `_training_history[global_step]` using `load_trajectory_by_seq_id`. If any
   trajectories loaded, return them (pure replay).

1. **Cached untrained from tree store**: Iterate all `query_id`s in `tree_store.trees`
   where `get_untrained_count > 0`, load up to `n_samples` per query via
   `load_trajectories`. If any cached trajectories loaded, return them.

1. **Fresh generation from dataloader**: Pull a batch from the dataloader (via
   `cycle_dataloader`), generate new rollouts via `self.actor.rollout_batch`. New
   trajectories are automatically inserted into the tree store by the patched
   `compute_advantages`.

Each fallback level logs which source was used with trajectory counts.

### New Helper Methods

**`_load_untrained_from_tree_store()`**: Iterates `tree_store.trees`, collects untrained
trajectories from all query_ids with `get_untrained_count > 0`, returns them as a flat
list. Limits to `cache_config.n_samples` per query_id.

**`_generate_from_dataloader(dataloader, workflow, workflow_kwargs, group_size)`**:
Lazily initializes a `_replay_dataloader_iter` (using `cycle_dataloader`), pulls a
batch, calls `self.actor.rollout_batch`, returns the generated trajectories.

### Modified `_replay_prepare_batch`

```python
def _replay_prepare_batch(self, dataloader, workflow, ...):
    global_step = self._replay_global_step

    # Level 1: Replay from training history
    if global_step in self.tree_store._training_history:
        pairs = self.tree_store._training_history[global_step]
        trajs = [self.tree_store.load_trajectory_by_seq_id(qid, sid)
                 for qid, sid in pairs]
        trajs = [t for t in trajs if t is not None]
        if trajs:
            self._replay_global_step += 1
            logger.info(f"Replay step {global_step}: {len(trajs)} from history")
            return trajs
        logger.warning(f"Replay step {global_step}: all missing, falling back")

    # Level 2: Cached untrained from tree store
    cached_trajs = self._load_untrained_from_tree_store()
    if cached_trajs:
        self._replay_global_step += 1
        logger.info(f"Replay step {global_step}: {len(cached_trajs)} cached untrained")
        return cached_trajs

    # Level 3: Fresh generation from dataloader
    self._replay_global_step += 1
    return self._generate_from_dataloader(dataloader, workflow, workflow_kwargs, group_size)
```

### Cleanup

In `CacheAwarePPOTrainer.train()` finally block, also clean up `_replay_dataloader_iter`
alongside `_cache_dataloader_iter`.

## Edge Cases

- **No training history at all**: Existing `ValueError` in `__init__` is preserved. The
  fallback only activates after initial replay history is exhausted or missing for a
  specific step.
- **`_replay_global_step` exceeds max key in `_training_history`**: The
  `global_step not in _training_history` check handles this, falls to Level 2/3.
- **Level 2 returns some but not enough**: Return whatever is available. The training
  loop handles varying batch sizes.
- **Dataloader exhausted**: `cycle_dataloader` wraps forever, so this cannot happen.
- **Mixed fallback within a step**: Each step uses exactly one source (the first that
  yields trajectories). No mixing within a single step.

## Files Changed

| File                                             | Change                                                                                                                           |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/trainer.py`        | Modify `_replay_prepare_batch`, add `_load_untrained_from_tree_store`, add `_generate_from_dataloader`, update `train()` cleanup |
| `tests/test_tree_search/test_mcts_tree_store.py` | Add unit tests for fallback behavior                                                                                             |

## Testing Plan

1. **Replay returns trajectories when available**: Mock `_training_history` with valid
   entries, verify `_replay_prepare_batch` returns them.
1. **Falls to Level 2 when replay step missing**: Populate tree store with untrained
   paths, no history for current step, verify cached trajectories returned.
1. **Falls to Level 3 when both exhausted**: Empty tree store and no history, mock
   `rollout_batch`, verify fresh generation called.
1. **`_load_untrained_from_tree_store` multi-query**: Insert trajectories under multiple
   query_ids, verify all are collected.
1. **`_generate_from_dataloader` lazy init**: Verify dataloader iterator created on
   first call, reused on subsequent calls.

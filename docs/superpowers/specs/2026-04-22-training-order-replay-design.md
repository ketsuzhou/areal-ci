# Training Order Recording & Replay Design

## Goal

Record the exact order of trajectories used at each training step in the
`CacheAwarePPOTrainer`, stored in trie leaf nodes, and support deterministic replay of
previous training runs using cached trajectories only.

## Background

`CacheAwarePPOTrainer` uses an MCTS tree (`MCTSTreeStore` + `TrieNode`) to cache rollout
trajectories. Currently there is no record of which trajectories were used at which
training step, making it impossible to reproduce a previous training run exactly. The
rollout order is non-deterministic due to `random.shuffle()` in
`BatchTaskDispatcher.wait_results()` and dataloader shuffling.

## Design

### 1. TrieNode: Add `training_steps` Field

Add `training_steps: list[int]` to `TrieNode` (default empty list).

When a trajectory (`seq_id`) is trained at `global_step`, append `global_step` to the
leaf node of that trajectory's path (the last node from
`root.get_path_nodes(seq_id)[-1]`).

The leaf node uniquely identifies a trajectory (paired with `query_id`). The
`training_steps` list records at which global steps the trajectory participated in
training.

### 2. Recording Training Order

**When**: Inside the patched `_tree_backup_compute_advantages`, after tree backup and
marking trajectories as trained.

**How**:

- Add `MCTSTreeStore.record_training_step(global_step, trajectories)` method
- For each trajectory with `_mcts_seq_id`: find the leaf node via
  `root.get_path_nodes(seq_id)[-1]`, append `global_step` to `leaf.training_steps`
- For grouped trajectories with `_mcts_seq_ids`: same for each `seq_id`
- Also maintain `_training_history: dict[int, list[tuple[str, int]]]` on
  `MCTSTreeStore`, mapping `global_step -> [(query_id, seq_id), ...]` in the exact order
  trajectories appear in the input list. This captures cross-query_id ordering within a
  step (a single step may use trajectories from multiple different prompts/query_ids).

**Why both leaf-level and store-level**:

- Leaf `training_steps` answers "when was this trajectory used?" (useful for analytics,
  pruning, debugging)
- Store `_training_history` answers "what was the exact training order at step N?"
  (required for replay). Cross-query_id ordering cannot be reconstructed from leaf data
  alone.
- `_training_history` is the source of truth for replay; leaf `training_steps` is
  secondary metadata.

**Threading `global_step`**:

- Inject `traj["_global_step"] = global_step` on each trajectory dict in the base
  `RLTrainer.train()` loop, before calling `actor.compute_advantages(rollout_batch)`
- The patched method reads `_global_step` from the trajectory dicts

### 3. Replay Mode

**Config**: Add `replay: bool = False` to `RolloutCacheConfig`.

When `replay=True`:

1. **At trainer init**: After loading the tree checkpoint, `_training_history` is
   already populated from the checkpoint (see serialization below). No scan needed — the
   dict maps `global_step -> list[(query_id, seq_id)]` in exact original order.

1. **Override `_cache_aware_prepare_batch`**:

   - Look up `_training_history[global_step]`
   - Load each trajectory via `tree_store.load_trajectory_by_seq_id(query_id, seq_id)`
   - Skip the dataloader entirely
   - Skip all rollout generation
   - Return the trajectories in the recorded order

1. **New `MCTSTreeStore` methods**:

   - `load_trajectory_by_seq_id(query_id, seq_id)` — load a single trajectory by its
     exact seq_id (similar to existing `load_trajectories` but for a known seq_id)
   - `build_training_history()` — fallback: if `_training_history` is empty (e.g., old
     checkpoints without it), scan all leaf nodes to reconstruct training history from
     `training_steps` lists. Order within each step is best-effort (uses
     `root.sequence_ids` ordering within each query_id, but cross-query_id order may
     differ from original)

1. **Validation**: If replay is enabled but no leaves have training_steps, raise
   `ValueError("Cannot replay: no training history found in tree checkpoint. Run a training session first.")`

1. **Step alignment**: Replay uses the same `global_step` counter. Recovery and
   checkpointing work identically.

### 4. Checkpoint Serialization

**`TreeCheckpointManager` changes**:

- `_serialize_node`: Include `training_steps` in serialized dict if non-empty
- `_deserialize_node`: Restore `training_steps` from serialized dict (default empty
  list)
- `save`: Add `_training_history` to `metadata.json` (serialized as
  `{str(global_step): [[query_id, seq_id], ...], ...}`)
- `load`: Restore `_training_history` from `metadata.json`. If absent (old checkpoint),
  leave empty (will be populated by `build_training_history()` fallback on replay)

Follows the existing pattern for optional per-node fields (`logprobs`, `versions`).

### 5. Error Handling

- **Empty training_steps on replay**: Raise `ValueError` with clear message
- **Missing trajectories**: If `(query_id, seq_id)` in history is unavailable, log
  warning and skip that step
- **Partial history**: If only some steps have recorded training_steps (interrupted
  training), replay available steps and log warnings for gaps
- **Missing `_global_step`**: Fall back gracefully if the field is not set on
  trajectories (skip recording rather than crash)

## Files to Modify

| File                                              | Changes                                                                                                                                                                     |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/trie_node.py`       | Add `training_steps: list[int]` field                                                                                                                                       |
| `customized_areal/tree_search/mcts_tree_store.py` | Add `record_training_step()`, `load_trajectory_by_seq_id()`, `build_training_history()` methods; add `_training_history` dict and `_training_history` serialization support |
| `customized_areal/tree_search/trainer.py`         | Call `record_training_step` in patched method; add replay mode to `_cache_aware_prepare_batch`                                                                              |
| `customized_areal/tree_search/checkpoint.py`      | Serialize/deserialize `training_steps` on nodes                                                                                                                             |
| `customized_areal/tree_search/config.py`          | Add `replay: bool` to `RolloutCacheConfig`                                                                                                                                  |
| `areal/trainer/rl_trainer.py`                     | Inject `_global_step` on trajectories before `compute_advantages`                                                                                                           |

## Out of Scope

- Full state replay (model weights, optimizer state, RNG seeds) — use existing
  checkpoint/recovery for that
- Order-constrained generation (generate new trajectories but enforce same order)
- Pruning or compression of training_steps lists for long runs

# Design: Rollout Batch Consistency Test

## Problem

When `CacheAwarePPOTrainer` calls `self.actor.rollout_batch()` or
`self.actor.prepare_batch()`, individual trajectory dicts from each
`workflow.arun_episode()` call pass through several transformations:

1. **GroupedRolloutWorkflow**: runs `arun_episode` `group_size` times and concatenates
   results via `concat_padded_tensors` — padding shorter sequences and keeping only the
   first dict's value for non-tensor keys.
1. **\_merge_cached_and_new**: splits grouped trajectories back into individual
   `[1, seq_len]` dicts to preserve per-sample metadata like `_mcts_seq_id`.
1. **WorkflowExecutor/Dispatcher**: collects results, applying `should_accept_fn`
   filtering.

There is no test verifying that per-agent data survives these transformations intact.

## Scope

Two tiers of tests:

**Tier 1 — CPU unit tests** (no GPU required): verify data transformation logic in
`concat_padded_tensors`, `_merge_cached_and_new`, and `GroupedRolloutWorkflow`:

1. **`GroupedRolloutWorkflow`**: tensor values and shapes are preserved after
   `concat_padded_tensors` concatenation.
1. **`_merge_cached_and_new`**: grouped trajectories are correctly split back into
   individual items with all metadata preserved.
1. **`concat_padded_tensors`**: non-tensor scalar values are taken from the first dict;
   list values are flat-concatenated.
1. **End-to-end per-agent → batch roundtrip**: simulating the full path from individual
   `arun_episode` results through grouping, merging, and verifying every tensor field
   matches the originals.

**Tier 2 — GPU integration tests** (require CUDA): use a real model on GPU to run actual
inference via `RemoteSGLangEngine` / `RemotevLLMEngine` and verify that `prepare_batch`
/ `rollout_batch` results are consistent with individual `arun_episode` outputs:

5. **Per-agent vs prepare_batch consistency**: run individual `arun_episode` calls and
   collect results, then run `prepare_batch` with same inputs, compare tensor values at
   non-padded positions.
1. **group_size>1 consistency**: verify grouped rollout preserves all per-agent data
   after concat and split.

## Test Cases

### 1. `test_grouped_rollout_preserves_tensor_values`

- Create 3 individual trajectory dicts with different seq lengths (shape `[1, seq_len]`
  each)
- Run `concat_padded_tensors` on them (same logic as `GroupedRolloutWorkflow`)
- Verify:
  - Result shape is `[3, max_seq_len]` for each tensor key
  - Values at non-padded positions match the originals exactly
  - `attention_mask` is 1 for original positions, 0 for padding

### 2. `test_grouped_rollout_padding_correctness`

- Create 2 trajectory dicts with different seq lengths
- Verify that shorter trajectory is right-padded and attention_mask is correct

### 3. `test_merge_cached_and_new_preserves_count`

- 2 cached trajs (shape `[1, s1]`) + 2 new grouped trajs (shape `[2, s2]` each)
- Verify output has 6 items total (2 + 2\*2)

### 4. `test_merge_cached_and_new_preserves_tensor_values`

- After merge, verify each individual trajectory's tensor values match the originals
  (accounting for padding removal for grouped trajs)

### 5. `test_merge_preserves_metadata`

- Grouped traj has `_mcts_seq_ids=[10, 20]` and `_mcts_query_id="q1"`
- After merge, each split traj has `_mcts_seq_id` = 10 or 20 and `_mcts_query_id="q1"`

### 6. `test_merge_handles_single_batch_trajs`

- Grouped traj with batch_size=1 should pass through unchanged

### 7. `test_concat_padded_tensors_non_tensor_keys`

- Verify scalar keys keep first dict's value
- Verify list keys are flat-concatenated

### 8. `test_end_to_end_roundtrip`

- Simulate full path: create N individual trajectories → group via
  `concat_padded_tensors` → merge via `_merge_cached_and_new` → verify every tensor
  value matches original at non-padded positions

### 9. `test_gpu_prepare_batch_vs_individual_episodes` (GPU required)

- Start a real inference engine (SGLang/vLLM) on GPU
- Run `prepare_batch` with a small dataset and `group_size=1`
- Also run individual `arun_episode` calls with the same data
- Compare: for each trajectory in the batch result, verify `input_ids`,
  `attention_mask`, `logprobs`, `rewards` match the corresponding individual result at
  non-padded positions
- Use `torch.testing.assert_close` with `rtol=1e-4, atol=1e-4` for float tensors
  (logprobs, rewards) and exact match for int tensors

### 10. `test_gpu_grouped_rollout_consistency` (GPU required)

- Run with `group_size=2`
- Verify that after `GroupedRolloutWorkflow` concatenation, all per-agent data is
  preserved in the grouped result
- Verify that splitting via `_merge_cached_and_new` recovers the original per-agent
  trajectories

## File Location

`tests/test_tree_search/test_batch_consistency.py`

## Implementation Notes

- Tier 1 tests (1-8): pure CPU, no GPU required
- Tier 2 tests (9-10): `@pytest.mark.skipif(not CUDA_AVAILABLE)`, require a model at
  `MODEL_PATH`, start SGLang/vLLM server
- Use `torch.testing.assert_close` with `rtol=0, atol=0` for exact matching (integer
  tensors) in CPU tests
- Use `torch.testing.assert_close` with `rtol=1e-4, atol=1e-4` for float tensors in GPU
  tests
- Helper function `_make_traj(batch_size, seq_len, **metadata)` creates realistic
  trajectory dicts with input_ids, attention_mask, loss_mask, logprobs, rewards, and
  optional metadata keys
- GPU tests follow existing patterns from `tests/test_tree_training.py` for engine setup
  and teardown
- No mocking of distributed primitives needed

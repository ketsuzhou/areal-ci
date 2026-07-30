# TPFC Tree Search Pipeline Hardening

**Date**: 2026-05-08 **Scope**: Bugfix + hardening pass on the TPFC tree search training
pipeline **Files affected**: `customized_areal/tree_search/`, `customized_areal/tpfc/`,
`tests/test_tree_search/`

## 1. Fix test API mismatches

`tests/test_tree_search/test_mcts_tree_store.py` calls `is_trained(query_id, node_id)`,
`set_trained(query_id, node_id, ...)`, `get_reward(query_id, node_id)` with a query_id
first argument, but the actual methods only take `node_id`. Also, `test_clear`
references `_next_seq_id` which was renamed to `_next_node_id`.

**Fix**: Update all test calls to match the current 1-arg API. Fix the attribute name in
`test_clear`.

## 2. Sanitize query_id for checkpoint filenames

`checkpoint.py:31` uses `query_{query_id}.json` with no sanitization. If `query_id`
contains `/`, `\`, `:`, or other special chars, this creates invalid paths or overwrites
files.

**Fix**: Replace non-alphanumeric/underscore/dash characters with `_` before using as a
filename.

## 3. Slice response-only fields in `_node_to_tensor_dict`

`mcts_tree_store.py:117-120` adds `topk_ids`, `topk_logp`, `distill_reward`,
`teacher_logp` at full sequence length, but they are documented as "response-only
(aligned to loss_mask==1 positions)".

**Fix**: Slice them to the response portion using `resp_start:resp_end` before adding to
the trajectory dict.

## 4. Add public accessors to MCTSTreeStore

`advantage.py` directly accesses `tree_store._rewards`, `_normalized_advantages`,
`_q_values`. This breaks encapsulation and makes future refactoring risky.

**Fix**: Add `get_reward()` (already exists but takes node_id), `get_q_value()`,
`set_normalized_advantage()`, `get_normalized_advantage()` public methods to
MCTSTreeStore. Update `advantage.py` to use them.

## 5. Mixed cache+generation strategy

`trainer.py:284-309`: When any prompt lacks cache, ALL prompts are regenerated, wasting
cached data and growing the tree with duplicates.

**Fix**: Load cached trajectories for prompts that have them. Generate only for missing
prompts. Concatenate both lists. This avoids duplicate insertions and respects existing
cache.

## 6. Move hardcoded credentials to env vars

`backend_run.py` has `DEFAULT_REFRESH_TOKEN`, `DEFAULT_AGENT_ID`, `DEFAULT_USER_ID`.
`tpfc_agent.py` hardcodes judge model name and OpenRouter defaults.

**Fix**: Read from env vars with the current values as fallbacks. No behavior change —
just removes hardcoded secrets from source code.

## 7. Atomic checkpoint save

`checkpoint.py:25-53` saves metadata and query files independently — a crash between
them leaves partial state.

**Fix**: Write each file to a temp path (`.tmp` suffix) then `os.replace()` to the final
path. This ensures readers never see partial writes.

## 8. Log warning on single-sample GRPO normalization

`advantage.py:56-59`: When only 1 sample per query, advantages are silently set to 0.0.
This makes the model effectively ignore those trajectories with no indication.

**Fix**: Log a warning once when single-sample normalization occurs.

## 9. Handle empty trajectory list

`trainer.py:311-315`: Returning `[]` can cause downstream shape errors in the training
loop.

**Fix**: Raise a `RuntimeError` with a clear message so the training loop can skip/retry
the step, rather than propagating an empty batch.

## 10. Log warning on Q-value fallback

`advantage.py:78-81`: Silent fallback from normalized advantage to raw Q-value masks
potential bugs.

**Fix**: Log a warning when the fallback path is hit.

## 11. Rename `trajs` after conversion

`trainer.py:342-347`: `trajs` holds `Node` objects, then gets reassigned to
`list[dict]`. This makes the code hard to follow.

**Fix**: Use `nodes` for the Node list and `converted_trajs` for the tensor dict list.

## 12. Normalize log levels

- Demote per-trajectory debug messages from INFO to DEBUG (e.g. "Inserted N trajectories
  into tree")
- Promote cache-hit/miss counts from DEBUG to INFO (these are important for
  understanding training behavior)
- Keep epoch-level summaries at INFO

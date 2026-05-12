# Eliminate Tree Search Patches on grouped_workflow.py and workflow_executor.py

**Date**: 2026-05-11
**Status**: Draft

## Problem

Tree search training uses 3 monkey-patches managed by `TreeSearchPatches`:

| Patch | Target | Purpose |
|-------|--------|---------|
| Patch 1 | `PPOActor.compute_advantages` | Save/restore tree advantages around GAE |
| Patch 2 | `engine._wrap_openai_agent` | Wrap with `TreeSearchGroupedRolloutWorkflow(QueryIDProxyWorkflow(...))` |
| Patch 2b | `engine._resolve_workflow` | Prevent double-wrapping with `GroupedRolloutWorkflow` |
| Patch 3 | `engine.workflow_executor` | Replace with `TreeSearchWorkflowExecutor` to handle `list[Node]` returns |

These patches make the code fragile, hard to debug, and tightly coupled. The goal is to eliminate all patches by moving tree operations (insertion, advantage computation) into `QueryIDProxyWorkflow` and having it return a batched tensor dict that the base `WorkflowExecutor` handles natively.

## Design

### Data flow

**Current** (3 patches):
```
_wrap_openai_agent → TreeSearchGroupedRolloutWorkflow(QueryIDProxyWorkflow)
  → arun_episode returns list[Node]
  → TreeSearchWorkflowExecutor bypasses Interaction→tensor conversion
  → _TreeSearchRolloutResult wraps list[Node]
  → trainer._cache_aware_prepare_batch receives list[Node], does tree insert + advantage
```

**New** (no patches on grouped_workflow/workflow_executor):
```
_wrap_openai_agent → QueryIDProxyWorkflow (with tree_store + advantage_computer)
  → arun_episode: runs episodes, builds list[Node]
  → inserts nodes into tree_store
  → computes tree advantages (if mode=TREE)
  → converts list[Node] → batched tensor dict (via _node_to_tensor_dict + concat_padded_tensors)
  → returns dict[str, Any]
  → base WorkflowExecutor handles tensor dict natively
  → trainer receives tensor dicts, extracts node_id/query_id metadata for marking trained
```

### 1. QueryIDProxyWorkflow changes

**New constructor args**:
- `tree_store: MCTSTreeStore | None` — tree store for node insertion
- `advantage_computer: TreeAdvantageComputer | None` — computes tree advantages
- `advantage_mode: AdvantageMode` — controls whether tree advantages are computed

**Existing constructor arg kept**:
- `group_size: int` — already exists on QueryIDProxyWorkflow; handles grouping internally so no outer `TreeSearchGroupedRolloutWorkflow` wrapper is needed

**New `arun_episode` behavior** (after building `list[Node]`):
1. Insert nodes: `tree_store.insert_batch(nodes)`
2. Compute advantages (if `advantage_mode == TREE`): `advantage_computer.compute(nodes)`
3. Mark nodes as trained: `tree_store.set_trained(node_id, True)` for each node
4. Convert to batched tensor dict: call `_nodes_to_batched_tensor_dict(nodes)` which internally uses `_node_to_tensor_dict` + `concat_padded_tensors`
5. Return `dict[str, Any]` (the batched tensor dict) instead of `list[Node]`

Note: Checkpoint saving stays in the trainer (after rollout_batch returns), since tree_store
is shared state. This avoids coupling the workflow to checkpoint logic.

### 2. _nodes_to_batched_tensor_dict helper

New function in `proxy_workflow.py`:

```python
def _nodes_to_batched_tensor_dict(nodes: list[Node]) -> dict[str, Any]:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    Metadata (query_id, node_id, episode_id, turn_idx) is stored as
    single-element lists in each per-Node dict so that concat_padded_tensors
    flat-concatenates them across nodes.
    """
```

Key: `query_id`, `node_id`, `episode_id`, `turn_idx` are stored as single-element lists
(e.g., `["query_123"]`) matching the pattern `InteractionWithTokenLogpReward.to_tensor_dict`
already uses for `node_id`. This ensures `concat_padded_tensors` concatenates them
correctly across turns.

### 3. _node_to_tensor_dict changes

In `mcts_tree_store.py`, `_node_to_tensor_dict` currently stores `query_id` and `node_id`
as plain strings. Change them to single-element lists:

```python
# Before:
traj["query_id"] = query_id
traj["node_id"] = node_id

# After:
traj["query_id"] = [query_id]
traj["node_id"] = [node_id]
traj["episode_id"] = [node.episode_id or ""]
traj["turn_idx"] = [node.turn_idx or 0]
```

### 4. TreeSearchPatches simplification

| Patch | Action | Reason |
|-------|--------|--------|
| Patch 1 (GAE backup/restore) | **Removed** | Advantages pre-computed in proxy_workflow |
| Patch 2 (_wrap_openai_agent) | **Simplified** — only creates `QueryIDProxyWorkflow` with tree_store/advantage_computer args. No more `TreeSearchGroupedRolloutWorkflow` wrapper. | |
| Patch 2b (_resolve_workflow) | **Removed** | No double-wrapping possible |
| Patch 3 (workflow_executor) | **Removed** | Base executor handles tensor dict returns |
| Patch 4 (distill loss) | **Kept** | Still needed for non-GRPO loss modes |

### 5. Files deleted

- `customized_areal/tree_search/grouped_workflow.py` — grouping now handled entirely by QueryIDProxyWorkflow
- `customized_areal/tree_search/workflow_executor.py` — base WorkflowExecutor handles tensor dict returns

### 6. CacheAwarePPOTrainer._cache_aware_prepare_batch simplification

**Before** (tree ops in trainer):
1. Rollout → list[Node]
2. Insert into tree
3. Compute tree advantages
4. Mark trained
5. Save checkpoint
6. Convert Nodes → tensor dicts

**After** (tree ops in proxy_workflow):
1. Rollout → tensor dicts (with query_id, node_id metadata)
2. Save checkpoint (if `cache_mode == CROSS_TRAINING`)
3. Return tensor dicts for PPO

The trainer no longer does tree insertion, advantage computation, or node→tensor conversion.
Those happen in proxy_workflow. It still:
- Handles cache loading (cached trajectories from tree_store)
- Saves checkpoints (tree_store is shared state, accessible from trainer)

### 7. Mark-trained flow

Nodes are marked as trained inside `QueryIDProxyWorkflow.arun_episode` immediately after
tree insertion and advantage computation, using `tree_store.set_trained(node_id, True)`.
The trainer no longer needs to handle mark-trained — it's done at the source.

### 8. InteractionWithTokenLogpReward.to_tensor_dict compatibility

The base `to_tensor_dict` already stores `node_id` as `[self.interaction_id]` (line 209).
The new `_node_to_tensor_dict` will follow the same pattern, adding `query_id`, `episode_id`,
`turn_idx` as single-element lists. This is consistent with how `concat_padded_tensors`
handles list keys.

### 9. Group size handling

`QueryIDProxyWorkflow` already handles `group_size > 1` internally (lines 279-322 in current
code). With the elimination of `TreeSearchGroupedRolloutWorkflow`, the `_resolve_workflow`
double-wrapping prevention patch is no longer needed. The `_wrap_openai_agent` patch just
creates `QueryIDProxyWorkflow` directly with `group_size` set.

## Impact on existing tests

- `test_mcts_tree_store.py`: Update tests that call `_node_to_tensor_dict` to expect list
  values for `query_id`/`node_id`/`episode_id`/`turn_idx`
- Any tests of `TreeSearchGroupedRolloutWorkflow` or `TreeSearchWorkflowExecutor`: remove
  or adapt

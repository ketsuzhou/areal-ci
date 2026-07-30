# Design: Use `interaction_id` as `node_id`

**Status**: approved | **Date**: 2026-05-11

## Motivation

`node_id` is currently a monotonically increasing `int` assigned by
`MCTSTreeStore._insert_single`. This causes two known bugs (BUGS.md #5: default `0`
collision with dedup check, #9: `_next_node_id` checkpoint restore defaults to wrong
value). The inference engine already assigns a globally unique `interaction_id` (UUID
string) to each completion — reuse it as `node_id` and eliminate the counter entirely.

## Changes

### Node dataclass (`mcts_tree_store.py`)

```
node_id: int = 0           →  node_id: str = ""
parent_node_id: int | None →  parent_node_id: str | None
```

### `interactions_dict_to_nodes` (`proxy_workflow.py`)

- Pass `node_id=interaction_id` (the dict key) to `Node()` constructor.
- When `interaction.parent is not None`, set
  `parent_node_id=interaction.parent.interaction_id`.

### `MCTSTreeStore` (`mcts_tree_store.py`)

- All 10 internal dicts change key type `int` → `str`: `_visit_counts`, `_total_values`,
  `_q_values`, `_trained`, `_rewards`, `_normalized_advantages`, `_normalized_returns`,
  `_node_id_to_key`, `_query_node_ids` (list value type), `_turn_nodes` (value type).
- Remove `_next_node_id`.
- `_turn_nodes`: `dict[str, int]` → `dict[str, str]` (turn_id → node_id, both strings).
- `_insert_single`: read `node.node_id` instead of assigning from counter. Assert it is
  non-empty — an empty `node_id` at insert time is a bug.
- `insert_batch` dedup sentinel: `existing_id != 0` → `existing_id != ""`.
- All method signatures taking `node_id: int` → `node_id: str`. Return type `list[int]`
  → `list[str]`.

### Consumers (no logic changes, type-compatible)

| File                   | What changes                                                                    |
| ---------------------- | ------------------------------------------------------------------------------- |
| `trainer.py:419`       | `node_id = node.node_id` — binding, type flows from Node                        |
| `advantage.py:49-52`   | Dict keys / set members — `int` → `str`                                         |
| `_node_to_tensor_dict` | `traj["node_id"]` stores string instead of int                                  |
| `checkpoint.py`        | Serialize/deserialize `node_id` as string; remove `_next_node_id` from metadata |

### Checkpoint format

Old checkpoints with int `node_id` and `_next_node_id` are incompatible. No migration
path — development-phase code, start fresh.

## What stays the same

- `episode_id` — still assigned by `QueryIDProxyWorkflow` via `uuid.uuid4().hex`
- `query_id` — still from `data["query_id"]`
- `turn_idx` — still loop index in `interactions_dict_to_nodes`
- All advantage computation, tree backup, loss logic — only dict key type changes

## Files touched

1. `customized_areal/tree_search/mcts_tree_store.py` — Node dataclass, MCTSTreeStore
   class, `_node_to_tensor_dict`
1. `customized_areal/tree_search/proxy_workflow.py` — `interactions_dict_to_nodes`,
   `arun_episode`
1. `customized_areal/tree_search/checkpoint.py` — serialize/deserialize
1. `customized_areal/tree_search/trainer.py` — remove `node_id` local binding (no
   semantic change)
1. `customized_areal/tree_search/advantage.py` — type annotations

# Design: Add turn_idx to Node

## Problem

Node has no field to indicate a node's turn position within its episode. The tensor dict
converter `_node_to_tensor_dict` hardcodes `_turn_idx_in_episode = 0` and
`_num_turns_in_episode = 1`, so downstream training code and MCTS tree logic cannot
determine turn ordering.

## Solution

Add a `turn_idx: int` field to the Node dataclass (1-based per episode, default 0
meaning unset). Set it at Node creation time in proxy_workflow and grouped_workflow.
Propagate through checkpoint serialization and tensor dict conversion.

## Changes

### 1. Node dataclass (mcts_tree_store.py)

Add field:

```python
turn_idx: int = 0  # 1-based turn position within episode
```

### 2. proxy_workflow.py `_interactions_to_nodes`

Set `turn_idx` via enumeration:

```python
for turn_idx, (interaction_id, interaction) in enumerate(interactions.items(), start=1):
    ...
    node = Node(
        ...
        turn_idx=turn_idx,
    )
```

### 3. grouped_workflow.py

Set `turn_idx` on each node after setting `episode_id`:

```python
for turn_idx, node in enumerate(result, start=1):
    node.episode_id = episode_id
    node.query_id = query_id
    node.turn_idx = turn_idx
```

### 4. \_node_to_tensor_dict (mcts_tree_store.py)

Replace hardcoded values with Node data:

```python
def _node_to_tensor_dict(
    node: Node, query_id: str, node_id: int, num_turns_in_episode: int = 1
) -> dict[str, Any]:
    ...
    traj["_turn_idx_in_episode"] = node.turn_idx
    traj["_num_turns_in_episode"] = num_turns_in_episode
```

Callers compute `num_turns_in_episode` by counting nodes sharing the same `episode_id`.

### 5. trainer.py callers

Two call sites need updating to pass `num_turns_in_episode`:

**CacheAwarePPOTrainer.\_load_cached_trajs** (line ~120): Build an `episode_id → count`
map from the loaded nodes, then pass it:

```python
episode_sizes: dict[str, int] = {}
for node in nodes:
    episode_sizes[node.episode_id] = episode_sizes.get(node.episode_id, 0) + 1
for node in nodes:
    traj_dict = _node_to_tensor_dict(
        node, query_id, node.node_id,
        num_turns_in_episode=episode_sizes.get(node.episode_id, 1),
    )
```

**TreeBackupPPOTrainer.compute_loss** (line ~368): Same pattern — build `episode_sizes`
from `nodes`, pass to `_node_to_tensor_dict`.

### 6. Checkpoint (checkpoint.py)

Add `turn_idx` to `_serialize_record` and `_deserialize_record`:

- Serialize: `"turn_idx": node.turn_idx`
- Deserialize: `turn_idx=data.get("turn_idx", 0)`

## What doesn't change

- `node_id` remains the global auto-incrementing unique identifier assigned by
  MCTSTreeStore
- `MCTSTreeStore` internal indexing (`_node_id_to_key`, `_query_node_ids`) is unchanged
- `num_turns_in_episode` is not stored on Node; it is computed at export time from
  episode grouping

## Scope

- `customized_areal/tree_search/mcts_tree_store.py` — Node dataclass,
  `_node_to_tensor_dict`
- `customized_areal/tree_search/proxy_workflow.py` — `_interactions_to_nodes`
- `customized_areal/tree_search/grouped_workflow.py` — `arun_episode`
- `customized_areal/tree_search/trainer.py` — two `_node_to_tensor_dict` call sites
- `customized_areal/tree_search/checkpoint.py` — serialize/deserialize

# Design: `train_id`-based Trained Determination

**Date**: 2026-05-12
**Status**: Approved

## Problem

The current "trained" tracking uses a boolean `_trained: dict[str, bool]` keyed by `node_id`. When a new training run starts, `reset_trained_flags()` sets ALL nodes to untrained, which means `get_untrained_count()` returns the full count for every query. This forces the workflow to reload cached nodes even though they came from the same training run. With `train_id`, the logic becomes: a node is "trained" only if its `train_id` matches the current run's `train_id`. This correctly distinguishes nodes trained in the current run from those trained in a previous run.

## Design

### 1. `train_id` Generation

In `train_tpfc_tree_search.py`, generate a UUID at startup and set it as an environment variable (e.g., `TRAIN_ID`). Downstream components read it via `os.environ`.

### 2. Node Changes (`mcts_tree_store.py`)

Add `train_id: str = ""` field to the `Node` dataclass.

- Default `""` means "not yet trained" (or loaded from a pre-`train_id` checkpoint).
- When trained, stamped with the current run's `train_id`.

### 3. MCTSTreeStore Changes (`mcts_tree_store.py`)

- Add `current_train_id: str` field, read from `os.environ["TRAIN_ID"]` in `__init__`.
- `set_trained(node_id)` → sets `node.train_id = self.current_train_id`.
- `is_trained(node_id)` → returns `node.train_id == self.current_train_id`.
- `get_untrained_count(query_id)` → counts nodes where `train_id != self.current_train_id`.
- `get_untrained_node_ids(query_id, n_samples)` → same comparison.
- `load_trajectories(query_id, n_samples)` → loads "untrained" nodes (those with mismatched or empty `train_id`).
- Remove `_trained: dict[str, bool]` (no longer needed).
- Remove `reset_trained_flags()` (or make it a no-op; comparison logic handles it).
- `mark_episodes_trained()` → updated to stamp `train_id` instead of setting boolean flags.
- `clear()` → updated to not reset the removed `_trained` dict.

### 4. Checkpoint Changes (`checkpoint.py`)

- `_serialize_record`: add `"train_id": node.train_id` to the serialized dict.
- `_deserialize_record`: read `train_id` from the dict (default `""` for old checkpoints).
- `metadata.json`: add `"current_train_id"` field.
- `save_trained_episodes`: filter by `train_id == current_train_id` instead of `is_trained()`.
- `load_trained_episodes`: unchanged (returns episode ID set, caller interprets).

### 5. Workflow Changes (`tree_search_grouped_workflow.py`)

In `TreeSearchGroupedRolloutWorkflow.__init__`:
- Remove `self.tree_store.reset_trained_flags()` call.
- `current_train_id` is already set on the store (from env), so old nodes are automatically "untrained."

In `arun_episode`:
- `set_trained(node.node_id, True)` → unchanged call site; the method now stamps `train_id` instead of setting a boolean.

### 6. Training Script Changes (`train_tpfc_tree_search.py`)

Generate `train_id` at startup and export to env:

```python
import uuid
os.environ["TRAIN_ID"] = uuid.uuid4().hex
```

### 7. Config Changes (`config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml`)

Remove `agent.train_id: ""` (or leave empty — env var takes precedence for workflow, but agent may still use it).

## Data Flow

```
train_tpfc_tree_search.py
  └─ os.environ["TRAIN_ID"] = uuid
       │
       ▼
TreeSearchGroupedRolloutWorkflow.__init__()
  └─ MCTSTreeStore.__init__()
       └─ self.current_train_id = os.environ["TRAIN_ID"]
       └─ loads checkpoint → Node.train_id values preserved
            │
            ▼
arun_episode():
  ├─ get_untrained_count(query_id) → nodes where train_id != current_train_id
  ├─ load cached nodes (those with old/missing train_id)
  ├─ generate fresh episodes (only for missing slots)
  ├─ insert fresh nodes (train_id defaults to "")
  ├─ set_trained(node_id) → node.train_id = current_train_id
  └─ save checkpoint → train_id persisted per-node
```

## Backward Compatibility

Old checkpoints have no `train_id` field in serialized nodes. `_deserialize_record` defaults missing `train_id` to `""`, and `"" != current_train_id`, so old nodes are correctly treated as "untrained" and available for reuse.

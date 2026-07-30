# Tree Search Bug Fixes Design

Fix all 14 remaining bugs from the code review (`CODE_REVIEW.md`), covering
`mcts_tree_store.py`, `advantage.py`, `checkpoint.py`, `trainer.py`,
`grouped_workflow.py`, config YAMLs, and `_node_to_tensor_dict`.

Bugs #7, #9, #10, #18 are already addressed by the existing `patches.py` refactor and
are excluded.

______________________________________________________________________

## Bug #1 (Critical): `insert_batch` does not skip already-inserted nodes

**File:** `mcts_tree_store.py:204-212`

`insert_batch` calls `_insert_single` for every node, even those loaded from cache that
already have a `node_id`. This creates duplicates in all internal dicts, inflating visit
counts, biasing Q-values, and causing unbounded memory growth.

**Fix:** Skip nodes whose `node_id` is already present in `_node_id_to_key`.

```python
def insert_batch(self, trajectories: list[Node]) -> None:
    for node in trajectories:
        existing_id = getattr(node, "node_id", 0)
        if existing_id != 0 and existing_id in self._node_id_to_key:
            continue
        query_id = getattr(node, "query_id", None) or ""
        self._insert_single(query_id, node)
```

______________________________________________________________________

## Bug #2 (High): Population variance in GRPO normalization

**File:** `advantage.py:80`

Uses population variance (`/ N`) instead of Bessel-corrected sample variance
(`/ (N-1)`). With `n_samples=4`, this underestimates variance by ~25%, compressing the
advantage signal.

**Fix:**

```python
var_q = sum((q - mean_q) ** 2 for q in q_values) / max(len(q_values) - 1, 1)
```

Note: `training/loss.py:352-354` already uses `(num_valid - 1).clamp(min=1)`, so no fix
needed there.

______________________________________________________________________

## Bug #3 (High): `query_id` lost on checkpoint deserialization

**File:** `checkpoint.py:95-131`

`_serialize_record` does not save `query_id` and `_deserialize_record` does not restore
it. Nodes loaded from checkpoints have `query_id=None`, causing
`TreeAdvantageComputer._get_query_id()` to silently skip them.

**Fix:** Store `query_id` in `_serialize_record` and restore it in `_deserialize_record`
via `object.__setattr__` (since `query_id` is not a dataclass field).

```python
@staticmethod
def _serialize_record(node: Node) -> dict:
    data = {
        "input_ids": node.input_ids,
        # ... existing fields ...
        "query_id": getattr(node, "query_id", ""),
    }
    # ... optional fields ...
    return data

@staticmethod
def _deserialize_record(data: dict) -> Node:
    node = Node(
        # ... existing fields ...
    )
    query_id = data.get("query_id", "")
    if query_id:
        object.__setattr__(node, "query_id", query_id)
    return node
```

______________________________________________________________________

## Bug #4 (Medium): `split_prompts` fallback checks the same key twice

**File:** `trainer.py:317`

```python
query_id = prompt.get("query_id") or prompt.get("query_id") or ""
```

Both branches check `"query_id"`. Copy-paste error — simplify to:

```python
query_id = prompt.get("query_id") or ""
```

Also fix the docstring (lines 303-306) that lists two identical fallback descriptions.

______________________________________________________________________

## Bug #6 (Medium): No safeguard if tree advantages lost in tensor pipeline

**Status:** Already fixed. The warning is implemented in
`patches.py:_build_tree_backup_compute_advantages` (lines 121-126), which logs when tree
advantages are missing in TREE mode.

No additional work needed.

______________________________________________________________________

## Bug #8 (Medium): `episode_id` duplicates across empty query_id and epochs

**File:** `grouped_workflow.py:49`

When `query_id` is empty, `episode_id` degenerates to `"0"`, `"1"`, etc., causing
collisions across queries. In CROSS_TRAINING mode, same-named episode_ids from different
epochs also collide.

**Fix:** Add a UUID suffix for guaranteed uniqueness, and warn on empty `query_id`.

```python
import uuid

query_id = data.get("query_id") or ""
if not query_id:
    logger.warning(
        "query_id is empty; episode_id will not be unique across queries"
    )

for group_idx, result in enumerate(valid_results):
    episode_id = f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
    # ...
```

______________________________________________________________________

## Bug #11 (Low): Stale dataloader iterator persists across train() calls

**File:** `trainer.py:538-539`

If `train()`'s finally block doesn't execute cleanly, a stale iterator persists. The
finally block already deletes it, but as extra safety, reset at the start of
`_cache_aware_prepare_batch`.

**Fix:** Keep the existing lazy init in `_cache_aware_prepare_batch`, but add a safety
delete at the very start of `train()` before the try block. This ensures a stale
iterator from a previous crashed `train()` call is cleaned up:

```python
# In train(), before the try block:
if hasattr(self, "_cache_dataloader_iter"):
    del self._cache_dataloader_iter
```

______________________________________________________________________

## Bug #12 (Low): Plaintext API key in config YAML

**Files:** Multiple config YAMLs under `customized_areal/tpfc/configs/`

API keys committed in config files risk credential exposure.

**Fix:** The codebase uses OmegaConf-style `${...}` interpolation in YAML configs.
Replace the hardcoded key with an env var reference:

```yaml
swanlab:
    api_key: ${oc.env:SWANLAB_API_KEY}
```

If `oc.env` is not available, use the `os.environ.get` pattern in the Python code that
reads this config, and replace the YAML value with a placeholder:

```yaml
swanlab:
    api_key: ""  # Set SWANLAB_API_KEY environment variable
```

______________________________________________________________________

## Bug #13 (Readability): Unused `query_id` parameter in `set_trained`

**File:** `mcts_tree_store.py:232`

`set_trained(self, query_id, seq_id, trained)` never uses `query_id`.

**Fix:** Remove `query_id` from the signature. Update the caller in
`trainer.py:_mark_batch_trained` to not pass it.

```python
def set_trained(self, node_id: int, trained: bool = True) -> None:
    self._trained[node_id] = trained
```

______________________________________________________________________

## Bug #14 (Readability): `dict[int, None]` used as a set

**File:** `advantage.py:59`

```python
query_seq_sets: dict[str, dict[int, None]] = {}
```

**Fix:** Use `dict[str, set[int]]`:

```python
query_node_sets: dict[str, set[int]] = {}
# ...
qset = query_node_sets.setdefault(query_id, set())
# ...
qset.add(seq_id)
```

______________________________________________________________________

## Bug #15 (Readability): Misleading comment "flat list of per-episode dicts"

**File:** `trainer.py:569`

The comment says "per-episode dicts" but these are `Node` objects.

**Fix:** Change comment to "flat list of Node objects".

______________________________________________________________________

## Bug #16 (Readability): Inconsistent naming `seq_id` vs `node_id`

The same concept is called `seq_id` in `MCTSTreeStore` internals but `node_id` on `Node`
objects. Unify to `node_id`.

**Renames in `mcts_tree_store.py`:**

| Old name                              | New name                 |
| ------------------------------------- | ------------------------ |
| `_seq_id_to_key`                      | `_node_id_to_key`        |
| `_query_seq_ids`                      | `_query_node_ids`        |
| `_next_seq_id`                        | `_next_node_id`          |
| `_insert_single` param `seq_id`       | `node_id`                |
| `get_advantages` param `seq_id`       | `node_id`                |
| `get_prompt_mask` param `seq_id`      | `node_id`                |
| `set_trained` param `seq_id`          | `node_id`                |
| `is_trained` param `seq_id`           | `node_id`                |
| `get_reward` param `seq_id`           | `node_id`                |
| `get_untrained_seq_ids`               | `get_untrained_node_ids` |
| `load_trajectories` local `seq_id`    | `node_id`                |
| `_backup` param `seq_id`              | `node_id`                |
| All internal dict keys using `seq_id` | `node_id`                |
| `_node_to_tensor_dict` param `seq_id` | `node_id`                |

**Renames in `advantage.py`:**

| Old name                         | New name          |
| -------------------------------- | ----------------- |
| `query_seq_sets`                 | `query_node_sets` |
| local `seq_id`                   | `node_id`         |
| `_compute_single` param `seq_id` | `node_id`         |

**Renames in `trainer.py`:**

| Old name                                    | New name                 |
| ------------------------------------------- | ------------------------ |
| `_mark_batch_trained` local `seq_id`        | `node_id`                |
| `_cache_aware_prepare_batch` local `seq_id` | `node_id`                |
| `get_untrained_seq_ids` call                | `get_untrained_node_ids` |

**Renames in `checkpoint.py`:**

| Old name                    | New name         |
| --------------------------- | ---------------- |
| `_next_seq_id` in metadata  | `_next_node_id`  |
| `seq_id_to_key` in metadata | `node_id_to_key` |
| `query_seq_ids` in metadata | `query_node_ids` |

Note: Checkpoint metadata key renames are **backward-incompatible**. Existing
checkpoints won't load. Since the code review notes "Old TrieNode-based checkpoints are
incompatible and must be discarded", and this is a relatively new module, the rename is
acceptable. Add a backward-compat shim that tries the new key name first, then falls
back to the old key name.

______________________________________________________________________

## Bug #17 (Readability): `_node_to_tensor_dict` is repetitive

**File:** `mcts_tree_store.py:87-150`

Six near-identical `if X is not None` blocks for optional tensor fields.

**Fix:** Extract a helper:

```python
def _optional_tensor_field(
    traj: dict, key: str, values: list, dtype: torch.dtype
) -> None:
    if values is not None:
        traj[key] = torch.tensor(values, dtype=dtype).unsqueeze(0)
```

Replace the six blocks with calls:

```python
_optional_tensor_field(traj, "topk_ids", node.topk_ids, torch.int32)
_optional_tensor_field(traj, "topk_logp", node.topk_logp, torch.float32)
_optional_tensor_field(traj, "distill_reward", node.distill_reward, torch.float32)
_optional_tensor_field(traj, "teacher_logp", node.teacher_logp, torch.float32)
```

The `logp` and advantage/returns fields have different logic (slicing, dim checks) so
they stay as-is.

______________________________________________________________________

## Summary

| #   | Severity    | File                | Status                                           |
| --- | ----------- | ------------------- | ------------------------------------------------ |
| 1   | Critical    | mcts_tree_store.py  | Fix: skip already-inserted nodes                 |
| 2   | High        | advantage.py        | Fix: Bessel-corrected variance                   |
| 3   | High        | checkpoint.py       | Fix: serialize/deserialize query_id              |
| 4   | Medium      | trainer.py          | Fix: remove duplicate key check                  |
| 6   | Medium      | —                   | Already fixed by patches.py                      |
| 8   | Medium      | grouped_workflow.py | Fix: UUID suffix + warning                       |
| 11  | Low         | trainer.py          | Fix: reset iterator at train() start             |
| 12  | Low         | config YAMLs        | Fix: replace hardcoded API key                   |
| 13  | Readability | mcts_tree_store.py  | Fix: remove unused query_id param                |
| 14  | Readability | advantage.py        | Fix: use set\[int\] instead of dict\[int, None\] |
| 15  | Readability | trainer.py          | Fix: correct comment                             |
| 16  | Readability | multiple            | Fix: rename seq_id → node_id                     |
| 17  | Readability | mcts_tree_store.py  | Fix: extract optional tensor helper              |

# TPFC Tree Search Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix bugs and harden the TPFC tree search training pipeline for stability,
correctness, and readability.

**Architecture:** Surgical fixes across 6 files — each task is isolated to one or two
files. No rearchitecting. Tests are updated alongside code changes.

**Tech Stack:** Python 3.12+, PyTorch, pytest

______________________________________________________________________

### Task 1: Fix test API mismatches in test_mcts_tree_store.py

**Files:**

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`
- Test: same file

The tests call `is_trained(query_id, node_id)`, `set_trained(query_id, node_id, ...)`,
`get_reward(query_id, node_id)` with a query_id first argument, but the actual methods
only take `node_id`. Also `test_clear` references `_next_seq_id` which was renamed to
`_next_node_id`.

- [ ] **Step 1: Fix `is_trained` / `set_trained` / `get_reward` call signatures**

In `test_mcts_tree_store.py`, these classes/methods need fixing:

`TestMCTSTreeStoreTrainedFlag` (lines 299-337):

- `test_trained_flag_default_false` (line 303):
  `store.is_trained("q1", traj["node_id"])` → `store.is_trained(traj["node_id"])`

- `test_set_trained` (line 310): `store.set_trained("q1", traj["node_id"], True)` →
  `store.set_trained(traj["node_id"], True)`

- `test_get_untrained_count` (line 321): `store.set_trained("q1", t1["node_id"], True)`
  → `store.set_trained(t1["node_id"], True)`

- `test_reset_trained_flags` (line 327):
  `store.set_trained("q1", traj["node_id"], True)` →
  `store.set_trained(traj["node_id"], True)` ; line 328:
  `store.is_trained("q1", traj["node_id"])` → `store.is_trained(traj["node_id"])`

- `test_get_reward` (lines 335-336): `store.get_reward("q1", t1["node_id"])` →
  `store.get_reward(t1["node_id"])` ; same for t2

- [ ] **Step 2: Fix `_next_seq_id` → `_next_node_id` in `test_clear`**

In `TestMCTSTreeStoreClear.test_clear` (line 424):

- `assert store._next_seq_id == 0` → `assert store._next_node_id == 1` (Note:
  `_next_node_id` starts at 1, not 0. After inserting one trajectory it will be 2. But
  after `clear()`, it resets to 1. So assert `store._next_node_id == 1`.)

- [ ] **Step 3: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_tree_search/test_mcts_tree_store.py
git commit -m "fix: correct test API mismatches in test_mcts_tree_store"
```

______________________________________________________________________

### Task 2: Sanitize query_id for checkpoint filenames

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:29-33`
- Test: `tests/test_tree_search/test_checkpoint.py` (create)

If `query_id` contains `/`, `\`, `:`, etc., `query_{query_id}.json` creates invalid
paths.

- [ ] **Step 1: Write the test**

Create `tests/test_tree_search/test_checkpoint.py`:

```python
import json
import os
import tempfile

from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


def _make_node(query_id: str = "q1", reward: float = 1.0) -> Node:
    return Node(
        input_ids=[1, 2, 3, 4, 5],
        loss_mask=[0, 0, 1, 1, 1],
        logprobs=[0.0] * 5,
        versions=[-1] * 5,
        outcome_reward=reward,
        query_id=query_id,
    )


class TestTreeCheckpointManager:
    def test_sanitize_query_id_special_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = TreeCheckpointManager(tmpdir)
            store = MCTSTreeStore()
            node = _make_node(query_id="path/with:special\\chars")
            store.insert_batch([node])
            mgr.save(store)
            loaded = mgr.load()
            assert "path/with:special\\chars" in loaded.trajectories

    def test_round_trip_simple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = TreeCheckpointManager(tmpdir)
            store = MCTSTreeStore()
            node = _make_node(query_id="q1", reward=2.0)
            store.insert_batch([node])
            mgr.save(store)
            loaded = mgr.load()
            assert "q1" in loaded.trajectories
            assert loaded.trajectories["q1"][0].outcome_reward == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_checkpoint.py -v`
Expected: FAIL — `test_sanitize_query_id_special_chars` will write to an invalid path or
fail on load

- [ ] **Step 3: Implement sanitization**

In `checkpoint.py`, add a helper and use it in `save()` and `load()`:

```python
import re

def _sanitize_filename(query_id: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    return re.sub(r"[^\w\-.]", "_", query_id)
```

In `save()` (line 31), change:

```python
filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
```

to:

```python
filepath = os.path.join(self.save_dir, f"query_{_sanitize_filename(query_id)}.json")
```

In `load()` (line 91), change:

```python
query_id = filename[len("query_") : -len(".json")]
```

to:

```python
sanitized = filename[len("query_") : -len(".json")]
# Map sanitized filename back to original query_id using metadata
query_id = None
for qid in tree_store.trajectories if hasattr(tree_store, 'trajectories') else []:
    if _sanitize_filename(qid) == sanitized:
        query_id = qid
        break
if query_id is None:
    query_id = sanitized  # fallback
```

Actually, the simpler approach: store a `query_id → sanitized_filename` mapping in
metadata. Update `save()` to also write:

```python
# In save(), add to metadata dict:
"query_id_to_file": {
    qid: _sanitize_filename(qid) for qid in tree_store.trajectories
},
```

And in `load()`, use the mapping:

```python
query_id_to_file = metadata.get("query_id_to_file", {})
file_to_query = {v: k for k, v in query_id_to_file.items()}
# ...
for filename in os.listdir(self.save_dir):
    if not filename.startswith("query_") or not filename.endswith(".json"):
        continue
    sanitized = filename[len("query_") : -len(".json")]
    query_id = file_to_query.get(sanitized, sanitized)
    # ... rest of load logic
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_checkpoint.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_tree_search/test_checkpoint.py
git commit -m "fix: sanitize query_id for checkpoint filenames"
```

______________________________________________________________________

### Task 3: Slice response-only fields in `_node_to_tensor_dict`

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:92-98, 117-120`
- Test: `tests/test_tree_search/test_mcts_tree_store.py`

`topk_ids`, `topk_logp`, `distill_reward`, `teacher_logp` are documented as
"response-only" but are added at full sequence length.

- [ ] **Step 1: Write the test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestNodeToTensorDict:
    def test_response_only_fields_sliced(self):
        from customized_areal.tree_search.mcts_tree_store import (
            Node,
            _node_to_tensor_dict,
        )

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[0.0, 0.0, -0.3, -0.4, -0.5],
            versions=[-1, -1, 1, 1, 1],
            outcome_reward=1.0,
            query_id="q1",
            node_id=1,
            topk_ids=[[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            topk_logp=[[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0], [-9.0, -10.0]],
            distill_reward=[[0.1], [0.2], [0.3], [0.4], [0.5]],
            teacher_logp=[[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]],
        )
        result = _node_to_tensor_dict(node, "q1", 1)
        # Response portion is positions 2:5 (loss_mask==1)
        assert result["topk_ids"].shape == (1, 3, 2)  # [1, resp_len, topk]
        assert result["topk_logp"].shape == (1, 3, 2)
        assert result["distill_reward"].shape == (1, 3, 1)
        assert result["teacher_logp"].shape == (1, 3, 1)

    def test_logp_already_sliced(self):
        """logp is already correctly sliced — verify it stays that way."""
        from customized_areal.tree_search.mcts_tree_store import (
            Node,
            _node_to_tensor_dict,
        )

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[0.0, 0.0, -0.3, -0.4, -0.5],
            versions=[-1, -1, 1, 1, 1],
            outcome_reward=1.0,
            query_id="q1",
            node_id=1,
        )
        result = _node_to_tensor_dict(node, "q1", 1)
        assert result["logp"].shape == (1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestNodeToTensorDict -v`
Expected: FAIL — `topk_ids` shape will be `(1, 5, 2)` instead of `(1, 3, 2)`

- [ ] **Step 3: Implement the fix**

In `mcts_tree_store.py`, replace `_optional_tensor_field` and the response-only field
calls:

Replace the `_optional_tensor_field` function (lines 92-97):

```python
def _optional_tensor_field(
    traj: dict[str, Any], key: str, values: list | None, dtype: torch.dtype
) -> None:
    """Add an unsqueezed tensor to traj if values is not None."""
    if values is not None:
        traj[key] = torch.tensor(values, dtype=dtype).unsqueeze(0)
```

with:

```python
def _optional_tensor_field(
    traj: dict[str, Any], key: str, values: list | None, dtype: torch.dtype,
    start: int = 0, end: int | None = None,
) -> None:
    """Add an unsqueezed tensor to traj if values is not None.

    If start/end are given, slice values[start:end] before conversion.
    """
    if values is not None:
        sliced = values[start:end] if end is not None else values[start:]
        traj[key] = torch.tensor(sliced, dtype=dtype).unsqueeze(0)
```

Update the calls in `_node_to_tensor_dict` (lines 117-120):

```python
    _optional_tensor_field(traj, "topk_ids", node.topk_ids, torch.int32, resp_start, resp_end)
    _optional_tensor_field(traj, "topk_logp", node.topk_logp, torch.float32, resp_start, resp_end)
    _optional_tensor_field(traj, "distill_reward", node.distill_reward, torch.float32, resp_start, resp_end)
    _optional_tensor_field(traj, "teacher_logp", node.teacher_logp, torch.float32, resp_start, resp_end)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "fix: slice response-only fields to response portion in _node_to_tensor_dict"
```

______________________________________________________________________

### Task 4: Add public accessors to MCTSTreeStore and update advantage.py

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py` (add methods)

- Modify: `customized_areal/tree_search/advantage.py` (use public methods)

- [ ] **Step 1: Add public accessors to MCTSTreeStore**

In `mcts_tree_store.py`, add these methods after the existing `get_reward` method (after
line 241):

```python
    def get_q_value(self, node_id: int) -> float:
        return self._q_values.get(node_id, 0.0)

    def set_normalized_advantage(self, node_id: int, value: float) -> None:
        self._normalized_advantages[node_id] = value

    def get_normalized_advantage(self, node_id: int, default: float = 0.0) -> float:
        return self._normalized_advantages.get(node_id, default)
```

- [ ] **Step 2: Update advantage.py to use public accessors**

In `advantage.py`, replace direct dict accesses:

Line 55:

```python
            q_values = [self.tree_store._rewards.get(nid, 0.0) for nid in node_ids]
```

→

```python
            q_values = [self.tree_store.get_reward(nid) for nid in node_ids]
```

Lines 57-58:

```python
                    self.tree_store._normalized_advantages[nid] = 0.0
```

→

```python
                    self.tree_store.set_normalized_advantage(nid, 0.0)
```

Lines 63-66:

```python
            for nid, q in zip(node_ids, q_values):
                self.tree_store._normalized_advantages[nid] = (q - mean_q) / (
                    std_q + self.grpo_eps
                )
```

→

```python
            for nid, q in zip(node_ids, q_values):
                self.tree_store.set_normalized_advantage(
                    nid, (q - mean_q) / (std_q + self.grpo_eps)
                )
```

Lines 78-81:

```python
            advantages = mask.float() * self.tree_store._normalized_advantages.get(
                node_id,
                self.tree_store._q_values.get(node_id, 0.0),
            )
```

→

```python
            advantages = mask.float() * self.tree_store.get_normalized_advantage(
                node_id,
                default=self.tree_store.get_q_value(node_id),
            )
```

- [ ] **Step 3: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py customized_areal/tree_search/advantage.py
git commit -m "refactor: add public accessors to MCTSTreeStore, remove private dict access from advantage.py"
```

______________________________________________________________________

### Task 5: Mixed cache+generation strategy

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:282-315`

Currently: if any prompt lacks cache, ALL prompts are regenerated. Fix: load cached for
prompts that have cache, generate only for missing prompts, concatenate.

- [ ] **Step 1: Implement the fix**

Replace lines 282-315 in `trainer.py`:

```python
        # Split into cached / needs-generation
        cached_items, need_gen_items = self._batch_builder.split_prompts(raw_batch)

        # All prompts have enough cache -> use cache only
        if not need_gen_items:
            trajs = list(self._batch_builder.load_cached_trajectories(cached_items))
            logger.info(f"Cache-aware rollout: {len(trajs)} cached (all from cache)")
        else:
            # Any prompt lacks cache -> regenerate all prompts via rollout_batch
            n_samples = self.cache_config.n_samples
            all_prompts = [item["prompt"] for item in cached_items] + [
                item["prompt"] for item in need_gen_items
            ]

            logger.info(
                f"Generating trajectories for {len(all_prompts)} query "
                f"(group_size={n_samples})"
            )
            new_trajs = self.actor.rollout_batch(
                all_prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )

            # TreeSearchWorkflowExecutor already returns flat list of Node objects
            trajs = new_trajs if new_trajs else []

            logger.info(f"Cache-aware rollout: 0 cached, {len(trajs)} newly generated")

        if not trajs:
            logger.warning(
                "No trajectories available for this step; returning empty batch"
            )
            return []
```

with:

```python
        # Split into cached / needs-generation
        cached_items, need_gen_items = self._batch_builder.split_prompts(raw_batch)

        # Load cached trajectories for prompts that have them
        cached_trajs: list = []
        if cached_items:
            cached_trajs = list(
                self._batch_builder.load_cached_trajectories(cached_items)
            )

        # Generate trajectories for prompts that need them
        generated_trajs: list = []
        if need_gen_items:
            n_samples = self.cache_config.n_samples
            gen_prompts = [item["prompt"] for item in need_gen_items]

            logger.info(
                f"Generating trajectories for {len(gen_prompts)} queries "
                f"(group_size={n_samples})"
            )
            new_trajs = self.actor.rollout_batch(
                gen_prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            if new_trajs:
                generated_trajs = new_trajs

        trajs = cached_trajs + generated_trajs
        logger.info(
            f"Cache-aware rollout: {len(cached_trajs)} cached + "
            f"{len(generated_trajs)} generated = {len(trajs)} total"
        )

        if not trajs:
            raise RuntimeError(
                "No trajectories available for this training step "
                "(both cache and generation returned empty). "
                "Check rollout engine and dataset."
            )
```

Note: This also implements items 9 (raise RuntimeError instead of returning \[\]) and 11
(use `trajs` consistently — the rename to `nodes`/`converted_trajs` is in Task 9).

- [ ] **Step 2: Run existing tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "fix: mixed cache+generation strategy instead of all-or-nothing regeneration"
```

______________________________________________________________________

### Task 6: Move hardcoded credentials to env vars

**Files:**

- Modify: `customized_areal/tpfc/backend_run.py:47-53`

- Modify: `customized_areal/tpfc/tpfc_agent.py:68-69`

- [ ] **Step 1: Fix backend_run.py**

Replace lines 47-53:

```python
DEFAULT_REFRESH_TOKEN = "4uhiohwgwp7e"
DEFAULT_AGENT_ID = "ef383a00-7c6b-4117-a2af-6d3ab9dbb8bb"
# DEFAULT_AGENT_ID = None

DEFAULT_USER_ID = "13183c90-ac94-403e-893e-c53552ad429d"
LE_AGENT_API_URL = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")
```

with:

```python
DEFAULT_REFRESH_TOKEN = os.environ.get("TPFC_REFRESH_TOKEN", "")
DEFAULT_AGENT_ID = os.environ.get("TPFC_AGENT_ID", "")
DEFAULT_USER_ID = os.environ.get("TPFC_USER_ID", "")
LE_AGENT_API_URL = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")
```

Also in `run_backend()` (line 603), add a validation check after the defaults are
resolved:

```python
    user_id = user_id or DEFAULT_USER_ID
    if not user_id:
        raise ValueError(
            "user_id is required. Set TPFC_USER_ID env var or pass user_id argument."
        )
```

And in `_resolve_agent_id()`, update the default check (currently line 296-297):

```python
    if agent_id is not None:
        return agent_id
```

This stays the same, but now `DEFAULT_AGENT_ID` is empty string when not set — so the
`agent_id is not None` check will be True with empty string. Fix:

```python
    if agent_id:
        return agent_id
```

- [ ] **Step 2: Fix tpfc_agent.py**

Replace lines 68-69:

```python
    model_name = "z-ai/glm-5.1"
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
```

with:

```python
    model_name = judge_model_name or os.environ.get("TPFC_JUDGE_MODEL", "z-ai/glm-5.1")
    base_url = judge_base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = judge_api_key or os.environ.get("OPENROUTER_API_KEY", "")
```

This ensures the function arguments (`judge_model_name`, `judge_base_url`,
`judge_api_key`) are actually used when provided, falling back to env vars.

- [ ] **Step 3: Run tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests PASS (these files aren't tested by tree_search tests, but verify
nothing breaks)

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tpfc/backend_run.py customized_areal/tpfc/tpfc_agent.py
git commit -m "fix: move hardcoded credentials to env vars with proper fallbacks"
```

______________________________________________________________________

### Task 7: Atomic checkpoint save

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:25-53`

- [ ] **Step 1: Implement atomic saves**

Replace the `save()` method (lines 25-53):

```python
    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save per-query trajectory records (atomic per file)
        query_id_to_file: dict[str, str] = {}
        for query_id, records in tree_store.trajectories.items():
            data = {"records": [self._serialize_record(r) for r in records]}
            sanitized = _sanitize_filename(query_id)
            query_id_to_file[query_id] = sanitized
            filepath = os.path.join(self.save_dir, f"query_{sanitized}.json")
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)

        # Save metadata (atomic)
        metadata = {
            "next_node_id": tree_store._next_node_id,
            "node_id_to_key": {
                str(k): [v[0], v[1]] for k, v in tree_store._node_id_to_key.items()
            },
            "query_node_ids": {k: v for k, v in tree_store._query_node_ids.items()},
            "visit_counts": {str(k): v for k, v in tree_store._visit_counts.items()},
            "total_values": {str(k): v for k, v in tree_store._total_values.items()},
            "q_values": {str(k): v for k, v in tree_store._q_values.items()},
            "trained": {str(k): v for k, v in tree_store._trained.items()},
            "rewards": {str(k): v for k, v in tree_store._rewards.items()},
            "normalized_advantages": {
                str(k): v for k, v in tree_store._normalized_advantages.items()
            },
            "turn_nodes": tree_store._turn_nodes,
            "query_id_to_file": query_id_to_file,
        }
        meta_path = os.path.join(self.save_dir, "metadata.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(metadata, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_meta, meta_path)
```

- [ ] **Step 2: Update `load()` to use query_id_to_file mapping**

Replace the load loop (lines 88-97):

```python
        # Load per-query trajectory records
        query_id_to_file = metadata.get("query_id_to_file", {})
        file_to_query = {v: k for k, v in query_id_to_file.items()}
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            sanitized = filename[len("query_") : -len(".json")]
            query_id = file_to_query.get(sanitized, sanitized)
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            store.trajectories[query_id] = [
                self._deserialize_record(r) for r in data["records"]
            ]
```

- [ ] **Step 3: Run tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_checkpoint.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py
git commit -m "fix: atomic checkpoint save with write-then-rename"
```

______________________________________________________________________

### Task 8: Log warnings on silent edge cases in advantage.py

**Files:**

- Modify: `customized_areal/tree_search/advantage.py:56-59, 78-81`

- [ ] **Step 1: Add warning for single-sample GRPO normalization**

Replace lines 56-59:

```python
            if len(q_values) < 2:
                for nid in node_ids:
                    self.tree_store._normalized_advantages[nid] = 0.0
                continue
```

with:

```python
            if len(q_values) < 2:
                logger.warning(
                    "Only %d sample(s) for query_id=%s — GRPO normalization "
                    "produces zero advantages (model will ignore this trajectory). "
                    "Consider increasing n_samples.",
                    len(q_values),
                    query_id,
                )
                for nid in node_ids:
                    self.tree_store.set_normalized_advantage(nid, 0.0)
                continue
```

- [ ] **Step 2: Add warning for Q-value fallback**

Replace lines 78-81:

```python
            advantages = mask.float() * self.tree_store._normalized_advantages.get(
                node_id,
                self.tree_store._q_values.get(node_id, 0.0),
            )
```

with (already using public accessors from Task 4, just add the warning):

```python
            norm_adv = self.tree_store.get_normalized_advantage(node_id)
            if norm_adv == 0.0 and self.tree_store.get_q_value(node_id) != 0.0:
                logger.warning(
                    "Normalized advantage missing for node_id=%d (query_id=%s), "
                    "falling back to raw Q-value=%.4f",
                    node_id,
                    query_id,
                    self.tree_store.get_q_value(node_id),
                )
            advantages = mask.float() * (
                norm_adv if norm_adv != 0.0
                else self.tree_store.get_q_value(node_id)
            )
```

- [ ] **Step 3: Run tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_advantage.py -v`
Expected: All tests PASS (warnings go to log, test assertions unchanged)

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/advantage.py
git commit -m "fix: add warnings for single-sample normalization and Q-value fallback"
```

______________________________________________________________________

### Task 9: Rename `trajs` → `nodes` / `converted_trajs` in trainer.py

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Rename variables in `_cache_aware_prepare_batch`**

In `trainer.py`, in the `_cache_aware_prepare_batch` method, rename:

- The main `trajs` variable that holds `Node` objects → `nodes`
- The converted list → `converted_trajs`

Update the variable throughout the method. The tree operations section (insert_batch,
compute, mark_batch_trained) and the conversion section should use `nodes`. The distill
loss weight injection uses `converted_trajs`.

Here is the full replacement for the method body from the split_prompts call onward
(after the Task 5 changes):

```python
        # Split into cached / needs-generation
        cached_items, need_gen_items = self._batch_builder.split_prompts(raw_batch)

        # Load cached trajectories for prompts that have them
        cached_nodes: list = []
        if cached_items:
            cached_nodes = list(
                self._batch_builder.load_cached_trajectories(cached_items)
            )

        # Generate trajectories for prompts that need them
        generated_nodes: list = []
        if need_gen_items:
            n_samples = self.cache_config.n_samples
            gen_prompts = [item["prompt"] for item in need_gen_items]

            logger.info(
                f"Generating trajectories for {len(gen_prompts)} queries "
                f"(group_size={n_samples})"
            )
            new_trajs = self.actor.rollout_batch(
                gen_prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            if new_trajs:
                generated_nodes = new_trajs

        nodes = cached_nodes + generated_nodes
        logger.info(
            f"Cache-aware rollout: {len(cached_nodes)} cached + "
            f"{len(generated_nodes)} generated = {len(nodes)} total"
        )

        if not nodes:
            raise RuntimeError(
                "No trajectories available for this training step "
                "(both cache and generation returned empty). "
                "Check rollout engine and dataset."
            )

        # --- Tree operations (while query_id / node_id are available) ---

        # Insert trajectories into the MCTS tree
        self.tree_store.insert_batch(nodes)
        logger.debug(f"Inserted {len(nodes)} trajectories into tree")

        # Compute tree advantages (stashed on Node fields, flow through to tensors)
        if self.tree_backup_config.advantage_mode == AdvantageMode.TREE:
            self.tree_advantage_computer.compute(nodes)
            logger.debug(
                f"Computed tree advantages for {len(nodes)} trajectories (mode=TREE)"
            )

        # Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(self.tree_store, nodes)
        logger.debug(f"Marked {len(nodes)} trajectories as trained")

        # Save tree checkpoint (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.debug("Saved MCTS tree checkpoint after tree operations")

        # --- End tree operations ---

        # Convert Nodes to tensor dicts for the downstream PPO pipeline.
        converted_trajs: list[dict[str, Any]] = []
        for node in nodes:
            query_id = node.query_id
            node_id = node.node_id
            converted_trajs.append(_node_to_tensor_dict(node, query_id, node_id))

        # Inject distillation loss weights into trajectory dicts
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            for traj in converted_trajs:
                if self.tree_backup_config.loss_mode == LossMode.DISTILL:
                    traj["rl_loss_weight"] = 0.0
                else:
                    traj["rl_loss_weight"] = self.tree_backup_config.rl_loss_weight
                traj["distill_loss_weight"] = (
                    self.tree_backup_config.distill_loss_weight
                )

        return converted_trajs
```

Also update the docstring to reflect the mixed strategy instead of "all-or-nothing".

- [ ] **Step 2: Run tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor: rename trajs → nodes/converted_trajs for clarity in _cache_aware_prepare_batch"
```

______________________________________________________________________

### Task 10: Normalize log levels

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`
- Modify: `customized_areal/tree_search/patches.py`

Rules:

- Per-step details (inserted N trajectories, computed advantages, marked trained, saved
  checkpoint) → DEBUG (already DEBUG in trainer.py, verify)

- Cache hit/miss counts → INFO (already INFO in trainer.py, verify)

- Patch apply/restore → INFO (already INFO in patches.py, verify)

- Check for any DEBUG messages that should be INFO or vice versa

- [ ] **Step 1: Audit and fix log levels**

In `trainer.py`, current state after Task 9:

- `logger.info(f"Cache-aware rollout: ...")` → INFO ✓ (important operational signal)
- `logger.debug(f"Inserted {len(nodes)} trajectories into tree")` → DEBUG ✓ (per-step
  detail)
- `logger.debug(f"Computed tree advantages...")` → DEBUG ✓
- `logger.debug(f"Marked {len(nodes)} trajectories as trained")` → DEBUG ✓
- `logger.debug("Saved MCTS tree checkpoint after tree operations")` → DEBUG ✓

In `trainer.__init__` (line 177):

- `logger.info(f"Cache-aware training enabled ...")` → INFO ✓ (startup summary)

In `patches.py`:

- `logger.warning("Engine has no _wrap_openai_agent method; ...")` → WARNING ✓
- `logger.debug("Skipping outer GroupedRolloutWorkflow wrapper ...")` → DEBUG ✓
- `logger.info(f"Applied tree search patches ...")` → INFO ✓
- `logger.info("Restored all tree search patches")` → INFO ✓
- `logger.warning("TreeSearchPatches.apply() called twice; skipping")` → WARNING ✓

All levels are already correct. No changes needed.

- [ ] **Step 2: Mark as complete (no changes required)**

No commit needed — log levels are already correct after the other tasks' changes.

______________________________________________________________________

### Task 11: Final integration test

**Files:**

- Run all tree_search tests

- Run pre-commit

- [ ] **Step 1: Run all tree search tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run pre-commit**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && pre-commit run --all-files`
Expected: All checks PASS (may auto-fix formatting — if so, commit the fixes)

- [ ] **Step 3: Final commit if pre-commit made changes**

```bash
git add -u
git commit -m "style: pre-commit auto-fixes"
```

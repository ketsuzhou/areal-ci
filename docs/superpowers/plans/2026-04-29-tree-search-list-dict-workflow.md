# Tree Search List\[Dict\] Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stacked tensor dicts with per-episode TrajectoryRecord-like dicts
(Python lists) in the tree search pipeline, and add a custom
`TreeSearchWorkflowExecutor` that handles `list[dict]` returns from `arun_episode`.

**Architecture:** `proxy_workflow.arun_episode` returns `list[dict]` (per-turn),
`grouped_workflow.arun_episode` merges turns into per-episode dicts and returns
`list[dict]`, `TreeSearchWorkflowExecutor` flattens results from `rollout_batch`, and
`MCTSTreeStore.insert_batch` accepts per-episode dicts with Python lists directly. New
fields (`logp`, `topk_ids`, `topk_logp`, `distill_reward`, `teacher_logp`) are added to
`TrajectoryRecord` and persisted through the tree store.

**Tech Stack:** Python 3.12+, PyTorch, asyncio

______________________________________________________________________

## File Structure

| Action | File                                                | Responsibility                                                                                                                |
| ------ | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Modify | `customized_areal/tree_search/mcts_tree_store.py`   | TrajectoryRecord new fields, insert_batch accepts list-of-list dicts, load_trajectories returns new fields, checkpoint compat |
| Modify | `customized_areal/tree_search/checkpoint.py`        | Serialize/deserialize new TrajectoryRecord fields                                                                             |
| Create | `customized_areal/tree_search/workflow_executor.py` | TreeSearchWorkflowExecutor: handles list\[dict\] from arun_episode, flattens rollout_batch results                            |
| Modify | `customized_areal/tree_search/proxy_workflow.py`    | arun_episode returns list\[dict\] with Python lists, extracts new fields from InteractionWithTokenLogpReward                  |
| Modify | `customized_areal/tree_search/grouped_workflow.py`  | arun_episode merges per-turn dicts into per-episode dicts, returns list\[dict\]                                               |
| Modify | `customized_areal/tree_search/trainer.py`           | Patch workflow_executor, remove \_split_to_turn_dicts, update advantage computation for list-based dicts                      |
| Modify | `customized_areal/tree_search/advantage.py`         | Handle list-based input_ids (list\[int\] instead of tensor) for advantage computation                                         |
| Modify | `tests/test_tree_search/test_mcts_tree_store.py`    | Tests for new TrajectoryRecord fields, insert/load with list-of-list dicts                                                    |

______________________________________________________________________

### Task 1: Add new fields to TrajectoryRecord and update insert_batch for list-based dicts

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:24-38` (TrajectoryRecord)

- Modify: `customized_areal/tree_search/mcts_tree_store.py:162-229` (insert_batch)

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write failing tests for TrajectoryRecord new fields and list-based
  insert_batch**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestTrajectoryRecordNewFields:
    def test_new_fields_default_none(self):
        record = TrajectoryRecord(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[-0.1, -0.2, -0.3], versions=[0, 0, 1],
            reward=1.0, turn_response_starts=[2], turn_response_ends=[3],
        )
        assert record.logp is None
        assert record.topk_ids is None
        assert record.topk_logp is None
        assert record.distill_reward is None
        assert record.teacher_logp is None

    def test_new_fields_set(self):
        record = TrajectoryRecord(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[-0.1, -0.2, -0.3], versions=[0, 0, 1],
            reward=1.0, turn_response_starts=[2], turn_response_ends=[3],
            logp=[-0.3], topk_ids=[[3, 5, 7]], topk_logp=[[-0.3, -1.2, -3.0]],
            distill_reward=[[0.5]], teacher_logp=[[-0.2, -1.0, -2.5]],
        )
        assert record.logp == [-0.3]
        assert record.topk_ids == [[3, 5, 7]]
        assert record.topk_logp == [[-0.3, -1.2, -3.0]]
        assert record.distill_reward == [[0.5]]
        assert record.teacher_logp == [[-0.2, -1.0, -2.5]]


def _make_list_traj(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    logprobs: list[float] | None = None,
    versions: list[int] | None = None,
    query_id: str | None = None,
    logp: list[float] | None = None,
    topk_ids: list[list[int]] | None = None,
    topk_logp: list[list[float]] | None = None,
    distill_reward: list[list[float]] | None = None,
    teacher_logp: list[list[float]] | None = None,
    turn_ids: list[str] | None = None,
    parent_turn_ids: list[str | None] | None = None,
    turn_rewards: list[float] | None = None,
    outcome_reward: float | None = None,
) -> dict[str, Any]:
    """Build a per-episode dict with Python lists (TrajectoryRecord-like format)."""
    traj: dict[str, Any] = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "reward": reward,
    }
    if logprobs is not None:
        traj["logprobs"] = logprobs
    if versions is not None:
        traj["versions"] = versions
    if query_id is not None:
        traj["_mcts_query_id"] = query_id
    if logp is not None:
        traj["logp"] = logp
    if topk_ids is not None:
        traj["topk_ids"] = topk_ids
    if topk_logp is not None:
        traj["topk_logp"] = topk_logp
    if distill_reward is not None:
        traj["distill_reward"] = distill_reward
    if teacher_logp is not None:
        traj["teacher_logp"] = teacher_logp
    if turn_ids is not None:
        traj["turn_ids"] = turn_ids
    if parent_turn_ids is not None:
        traj["parent_turn_ids"] = parent_turn_ids
    if turn_rewards is not None:
        traj["turn_rewards"] = turn_rewards
    if outcome_reward is not None:
        traj["outcome_reward"] = outcome_reward
    return traj


class TestMCTSTreeStoreInsertListDict:
    def test_insert_list_dict_basic(self):
        store = MCTSTreeStore()
        traj = _make_list_traj(
            [1, 2, 3, 4, 5], [0, 0, 1, 1, 1],
            reward=2.0, query_id="q1",
        )
        store.insert_batch([traj])
        assert "_mcts_seq_id" in traj
        assert len(store.trajectories["q1"]) == 1
        record = store.trajectories["q1"][0]
        assert record.input_ids == [1, 2, 3, 4, 5]
        assert record.loss_mask == [0, 0, 1, 1, 1]
        assert record.reward == 2.0

    def test_insert_list_dict_with_new_fields(self):
        store = MCTSTreeStore()
        traj = _make_list_traj(
            [1, 2, 3, 4, 5], [0, 0, 1, 1, 1],
            reward=2.0, query_id="q1",
            logp=[-0.3, -0.4, -0.5],
            topk_ids=[[3, 5], [4, 6], [5, 7]],
            topk_logp=[[-0.3, -1.2], [-0.4, -1.5], [-0.5, -2.0]],
            distill_reward=[[0.5], [0.3], [0.1]],
            teacher_logp=[[-0.2, -1.0], [-0.3, -1.1], [-0.4, -1.3]],
        )
        store.insert_batch([traj])
        record = store.trajectories["q1"][0]
        assert record.logp == [-0.3, -0.4, -0.5]
        assert record.topk_ids == [[3, 5], [4, 6], [5, 7]]
        assert record.topk_logp == [[-0.3, -1.2], [-0.4, -1.5], [-0.5, -2.0]]
        assert record.distill_reward == [[0.5], [0.3], [0.1]]
        assert record.teacher_logp == [[-0.2, -1.0], [-0.3, -1.1], [-0.4, -1.3]]

    def test_insert_list_dict_with_turn_metadata(self):
        store = MCTSTreeStore()
        traj = _make_list_traj(
            [1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 1, 1, 0, 0, 1, 1],
            reward=1.0, query_id="q1",
            turn_ids=["t1", "t2"],
            parent_turn_ids=[None, "t1"],
            turn_rewards=[0.5, 0.5],
            outcome_reward=1.0,
        )
        store.insert_batch([traj])
        record = store.trajectories["q1"][0]
        assert record.turn_ids == ["t1", "t2"]
        assert record.parent_turn_ids == [None, "t1"]
        assert record.turn_rewards == [0.5, 0.5]
        assert record.outcome_reward == 1.0
        assert record.turn_response_starts == [2, 6]
        assert record.turn_response_ends == [4, 8]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTrajectoryRecordNewFields tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreInsertListDict -v`
Expected: FAIL — `TrajectoryRecord` doesn't have the new fields yet, and `insert_batch`
doesn't handle list-based dicts.

- [ ] **Step 3: Add new fields to TrajectoryRecord**

In `customized_areal/tree_search/mcts_tree_store.py`, update the `TrajectoryRecord`
dataclass:

```python
@dataclass
class TrajectoryRecord:
    """Stores a complete multi-turn trajectory for cache storage."""

    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    reward: float
    turn_response_starts: list[int]
    turn_response_ends: list[int]
    # Episode metadata for tree search:
    turn_ids: list[str] | None = None
    parent_turn_ids: list[str | None] | None = None
    turn_rewards: list[float] | None = None
    outcome_reward: float = 0.0
    # Top-k and distillation fields:
    logp: list[float] | None = None
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
```

- [ ] **Step 4: Add `_is_list_dict` helper and `_insert_list_dict` method to
  MCTSTreeStore**

Add a helper to detect list-based dicts (input_ids is `list`, not `torch.Tensor`):

```python
def _is_list_dict(traj: dict[str, Any]) -> bool:
    """Check if a trajectory dict uses Python lists instead of tensors."""
    input_ids = traj.get("input_ids")
    return isinstance(input_ids, list)
```

Add `_insert_list_dict` method to `MCTSTreeStore`:

```python
def _insert_list_dict(self, traj: dict[str, Any]) -> None:
    """Insert a per-episode dict with Python lists (TrajectoryRecord-like)."""
    if "_mcts_seq_id" in traj or "_mcts_seq_ids" in traj:
        return

    query_id = traj.get("_mcts_query_id", "")
    if not query_id:
        query_id = _get_query_id_list(traj)

    input_ids = traj["input_ids"]
    loss_mask = traj["loss_mask"]
    logprobs = traj.get("logprobs", [0.0] * len(input_ids))
    versions = traj.get("versions", [0] * len(input_ids))
    reward = traj.get("reward", 0.0)

    # Use provided turn boundaries or compute from loss_mask
    if "turn_response_starts" in traj and "turn_response_ends" in traj:
        starts = traj["turn_response_starts"]
        ends = traj["turn_response_ends"]
    else:
        starts, ends = _find_turn_boundaries(loss_mask)

    record = TrajectoryRecord(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=logprobs,
        versions=versions,
        reward=reward,
        turn_response_starts=starts,
        turn_response_ends=ends,
        turn_ids=traj.get("turn_ids"),
        parent_turn_ids=traj.get("parent_turn_ids"),
        turn_rewards=traj.get("turn_rewards"),
        outcome_reward=traj.get("outcome_reward", reward),
        logp=traj.get("logp"),
        topk_ids=traj.get("topk_ids"),
        topk_logp=traj.get("topk_logp"),
        distill_reward=traj.get("distill_reward"),
        teacher_logp=traj.get("teacher_logp"),
    )
    seq_id = self._insert_single(query_id, record)
    traj["_mcts_seq_id"] = seq_id
    traj["_mcts_query_id"] = query_id
```

Add `_get_query_id_list` helper (same as `_get_query_id` but for list-based input):

```python
def _get_query_id_list(traj: dict[str, Any]) -> str:
    """Derive a query ID from the prompt tokens in a list-based trajectory."""
    loss_mask = traj["loss_mask"]
    input_ids = traj["input_ids"]
    prompt_tokens = [ids for ids, lm in zip(input_ids, loss_mask) if lm == 0]
    prompt_str = ",".join(str(t) for t in prompt_tokens)
    return hashlib.md5(prompt_str.encode()).hexdigest()
```

- [ ] **Step 5: Update `insert_batch` to route list-based dicts**

In the `insert_batch` method, add a check at the top of the per-trajectory loop:

```python
def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
    for traj in trajectories:
        if "_mcts_seq_id" in traj or "_mcts_seq_ids" in traj:
            continue
        if _is_list_dict(traj):
            self._insert_list_dict(traj)
            continue
        if "_episode_idx" in traj:
            per_turn_dicts.append(traj)
        else:
            legacy_dicts.append(traj)
    # ... rest unchanged
```

Note: The full method body needs restructuring to handle the early-return for list
dicts. Move the per_turn_dicts/legacy_dicts accumulation into the same loop.

- [ ] **Step 6: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTrajectoryRecordNewFields tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreInsertListDict -v`
Expected: PASS

- [ ] **Step 7: Run existing tests to verify no regressions**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: All existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add new fields to TrajectoryRecord and insert_batch support for list-based dicts"
```

______________________________________________________________________

### Task 2: Update checkpoint serialization for new TrajectoryRecord fields

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:83-104`

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write failing test for checkpoint round-trip with new fields**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestMCTSTreeStoreCheckpointNewFields:
    def test_checkpoint_round_trip_with_new_fields(self, tmp_path):
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore()
        traj = _make_list_traj(
            [1, 2, 3, 4, 5], [0, 0, 1, 1, 1],
            reward=2.0, query_id="q1",
            logp=[-0.3, -0.4, -0.5],
            topk_ids=[[3, 5], [4, 6], [5, 7]],
            topk_logp=[[-0.3, -1.2], [-0.4, -1.5], [-0.5, -2.0]],
            distill_reward=[[0.5], [0.3], [0.1]],
            teacher_logp=[[-0.2, -1.0], [-0.3, -1.1], [-0.4, -1.3]],
        )
        store.insert_batch([traj])

        mgr = TreeCheckpointManager(str(tmp_path))
        mgr.save(store)

        loaded = mgr.load()
        record = loaded.trajectories["q1"][0]
        assert record.logp == [-0.3, -0.4, -0.5]
        assert record.topk_ids == [[3, 5], [4, 6], [5, 7]]
        assert record.topk_logp == [[-0.3, -1.2], [-0.4, -1.5], [-0.5, -2.0]]
        assert record.distill_reward == [[0.5], [0.3], [0.1]]
        assert record.teacher_logp == [[-0.2, -1.0], [-0.3, -1.1], [-0.4, -1.3]]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreCheckpointNewFields -v`
Expected: FAIL — checkpoint doesn't serialize/deserialize new fields.

- [ ] **Step 3: Update `_serialize_record` and `_deserialize_record`**

In `customized_areal/tree_search/checkpoint.py`:

```python
@staticmethod
def _serialize_record(record: TrajectoryRecord) -> dict:
    data = {
        "input_ids": record.input_ids,
        "loss_mask": record.loss_mask,
        "logprobs": record.logprobs,
        "versions": record.versions,
        "reward": record.reward,
        "turn_response_starts": record.turn_response_starts,
        "turn_response_ends": record.turn_response_ends,
        "turn_ids": record.turn_ids,
        "parent_turn_ids": record.parent_turn_ids,
        "turn_rewards": record.turn_rewards,
        "outcome_reward": record.outcome_reward,
        "logp": record.logp,
        "topk_ids": record.topk_ids,
        "topk_logp": record.topk_logp,
        "distill_reward": record.distill_reward,
        "teacher_logp": record.teacher_logp,
    }
    return data

@staticmethod
def _deserialize_record(data: dict) -> TrajectoryRecord:
    return TrajectoryRecord(
        input_ids=data["input_ids"],
        loss_mask=data["loss_mask"],
        logprobs=data["logprobs"],
        versions=data["versions"],
        reward=data["reward"],
        turn_response_starts=data["turn_response_starts"],
        turn_response_ends=data["turn_response_ends"],
        turn_ids=data.get("turn_ids"),
        parent_turn_ids=data.get("parent_turn_ids"),
        turn_rewards=data.get("turn_rewards"),
        outcome_reward=data.get("outcome_reward", data["reward"]),
        logp=data.get("logp"),
        topk_ids=data.get("topk_ids"),
        topk_logp=data.get("topk_logp"),
        distill_reward=data.get("distill_reward"),
        teacher_logp=data.get("teacher_logp"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreCheckpointNewFields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): checkpoint serialization for new TrajectoryRecord fields"
```

______________________________________________________________________

### Task 3: Update load_trajectories to return new fields and support list-based output

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:354-446` (load_trajectories)

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write failing test for load_trajectories with new fields**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestMCTSTreeStoreLoadListDict:
    def test_load_list_dict_with_new_fields(self):
        store = MCTSTreeStore()
        traj = _make_list_traj(
            [1, 2, 3, 4, 5], [0, 0, 1, 1, 1],
            reward=2.0, query_id="q1",
            logp=[-0.3, -0.4, -0.5],
            topk_ids=[[3, 5], [4, 6], [5, 7]],
            topk_logp=[[-0.3, -1.2], [-0.4, -1.5], [-0.5, -2.0]],
            distill_reward=[[0.5], [0.3], [0.1]],
            teacher_logp=[[-0.2, -1.0], [-0.3, -1.1], [-0.4, -1.3]],
            turn_ids=["t1"], parent_turn_ids=[None],
            turn_rewards=[2.0], outcome_reward=2.0,
        )
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert len(loaded) == 1
        t = loaded[0]
        # list-based output
        assert isinstance(t["input_ids"], list)
        assert t["logp"] == [-0.3, -0.4, -0.5]
        assert t["topk_ids"] == [[3, 5], [4, 6], [5, 7]]
        assert t["topk_logp"] == [[-0.3, -1.2], [-0.4, -1.5], [-0.5, -2.0]]
        assert t["distill_reward"] == [[0.5], [0.3], [0.1]]
        assert t["teacher_logp"] == [[-0.2, -1.0], [-0.3, -1.1], [-0.4, -1.3]]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreLoadListDict -v`
Expected: FAIL — `load_trajectories` currently returns tensor-based dicts, not
list-based dicts with new fields.

- [ ] **Step 3: Update `load_trajectories` to return list-based dicts with new fields**

When the record has `turn_ids` (i.e., it was inserted as a list-based dict), return
list-based dicts instead of tensor dicts. When the record has no `turn_ids` (legacy),
keep returning tensor dicts for backward compatibility.

Replace `load_trajectories` in `mcts_tree_store.py`:

```python
def load_trajectories(self, query_id: str, n_samples: int) -> list[dict[str, Any]]:
    """Load untrained trajectories.

    For records with turn metadata (inserted via list-based dicts), returns
    per-episode list-based dicts. For legacy records (tensor-based), returns
    per-turn tensor dicts (backward compatible).
    """
    if query_id not in self.trajectories:
        return []

    untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
    result: list[dict[str, Any]] = []
    for seq_id in untrained_ids:
        qid, idx = self._seq_id_to_key[seq_id]
        record = self.trajectories[qid][idx]

        # List-based output (when record was inserted with list dict)
        if record.turn_ids is not None:
            traj: dict[str, Any] = {
                "input_ids": record.input_ids,
                "loss_mask": record.loss_mask,
                "logprobs": record.logprobs,
                "versions": record.versions,
                "reward": record.reward,
                "turn_response_starts": record.turn_response_starts,
                "turn_response_ends": record.turn_response_ends,
                "turn_ids": record.turn_ids,
                "parent_turn_ids": record.parent_turn_ids,
                "turn_rewards": record.turn_rewards,
                "outcome_reward": record.outcome_reward,
                "_mcts_query_id": query_id,
                "_mcts_seq_id": seq_id,
            }
            if record.logp is not None:
                traj["logp"] = record.logp
            if record.topk_ids is not None:
                traj["topk_ids"] = record.topk_ids
            if record.topk_logp is not None:
                traj["topk_logp"] = record.topk_logp
            if record.distill_reward is not None:
                traj["distill_reward"] = record.distill_reward
            if record.teacher_logp is not None:
                traj["teacher_logp"] = record.teacher_logp
            result.append(traj)
            continue

        # Legacy tensor-based output (backward compatible)
        seq_len = len(record.input_ids)
        full_input_ids = torch.tensor(record.input_ids, dtype=torch.int32)
        full_loss_mask = torch.tensor(record.loss_mask, dtype=torch.int32)
        full_logprobs = torch.tensor(record.logprobs, dtype=torch.float32)
        full_versions = torch.tensor(record.versions, dtype=torch.int32)
        full_attention = torch.ones(seq_len, dtype=torch.bool)

        if not record.turn_response_starts:
            result.append({
                "input_ids": full_input_ids.unsqueeze(0),
                "loss_mask": full_loss_mask.unsqueeze(0),
                "logprobs": full_logprobs.unsqueeze(0),
                "versions": full_versions.unsqueeze(0),
                "attention_mask": full_attention.unsqueeze(0),
                "rewards": torch.tensor([record.reward], dtype=torch.float32).unsqueeze(0),
                "_mcts_query_id": query_id,
                "_mcts_seq_id": seq_id,
            })
            continue

        n_turns = len(record.turn_response_starts)
        for t in range(n_turns):
            start = record.turn_response_starts[t]
            end = record.turn_response_ends[t]
            turn_seq_len = end
            turn_reward = record.turn_rewards[t] if record.turn_rewards and t < len(record.turn_rewards) else 0.0
            turn_id = record.turn_ids[t] if t < len(record.turn_ids) else ""
            parent_turn_id = record.parent_turn_ids[t] if record.parent_turn_ids and t < len(record.parent_turn_ids) else None

            result.append({
                "input_ids": full_input_ids[:turn_seq_len].unsqueeze(0),
                "loss_mask": full_loss_mask[:turn_seq_len].unsqueeze(0),
                "logprobs": full_logprobs[:turn_seq_len].unsqueeze(0),
                "versions": full_versions[:turn_seq_len].unsqueeze(0),
                "attention_mask": full_attention[:turn_seq_len].unsqueeze(0),
                "rewards": torch.tensor([turn_reward], dtype=torch.float32),
                "_mcts_query_id": query_id,
                "_mcts_seq_id": seq_id,
                "_episode_idx": idx,
                "_turn_idx_in_episode": t,
                "_turn_id": turn_id,
                "_parent_turn_id": parent_turn_id,
                "_turn_reward": turn_reward,
                "_outcome_reward": record.outcome_reward,
                "_num_turns_in_episode": n_turns,
            })

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: All tests pass, including new and existing ones.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): load_trajectories returns list-based dicts with new fields"
```

______________________________________________________________________

### Task 4: Create TreeSearchWorkflowExecutor

**Files:**

- Create: `customized_areal/tree_search/workflow_executor.py`

- [ ] **Step 1: Write the TreeSearchWorkflowExecutor**

Create `customized_areal/tree_search/workflow_executor.py`:

```python
# customized_areal/tree_search/workflow_executor.py
"""WorkflowExecutor subclass for tree search that handles list[dict] returns.

When arun_episode returns list[dict] (per-episode TrajectoryRecord-like dicts
with Python lists), the base WorkflowExecutor's assert fails because it expects
a single dict. This subclass overrides the task creation and result collection
to handle list[dict] returns and flatten them in rollout_batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from areal.infra.workflow_executor import WorkflowExecutor, _RolloutTaskInput
from areal.utils import logging

logger = logging.getLogger("TreeSearchWorkflowExecutor")


@dataclass
class _TreeSearchRolloutResult:
    task_id: int
    trajectories: list[dict[str, Any]]


class TreeSearchWorkflowExecutor(WorkflowExecutor):
    """WorkflowExecutor that handles list[dict] returns from arun_episode.

    Overrides:
    - _create_workflow_task: accepts list[dict] from arun_episode,
      skips InteractionWithTokenLogpReward conversion, stores as
      _TreeSearchRolloutResult
    - wait: extracts trajectories from _TreeSearchRolloutResult
    - rollout_batch: flattens list[list[dict]] to list[dict]
    """

    def _create_workflow_task(self, pending_task: _RolloutTaskInput):
        """Override to accept list[dict] from arun_episode."""
        from areal.infra.workflow_executor import (
            _RolloutResult,
            stats_tracker,
            trace_session_event,
        )
        from areal.infra.perf_tracer import perf_tracer
        from areal.infra.context import workflow_context, WorkflowContext
        from areal.experimental.openai.types import InteractionWithTokenLogpReward
        from areal.utils.data import concat_padded_tensors

        async def _execute_workflow() -> _TreeSearchRolloutResult | None:
            task_id = pending_task.task_id
            perf_tracer.set_task_id(task_id)
            workflow_context.set(
                WorkflowContext(is_eval=pending_task.is_eval, task_id=task_id)
            )

            manager = self.staleness_manager
            should_accept_fn = pending_task.should_accept_fn
            should_accept: bool | None = None
            reason: str | None = None

            try:
                result = await pending_task.workflow.arun_episode(
                    self.inference_engine, pending_task.data
                )

                # Handle None (rejected)
                if result is None:
                    should_accept_traj = False
                    reason = "returned_none"
                elif isinstance(result, list):
                    # list[dict] — tree search per-episode dicts
                    if should_accept_fn is None:
                        should_accept = True
                    else:
                        # Accept if any dict passes the filter
                        should_accept = any(
                            should_accept_fn(d) for d in result
                        ) if result else False
                    should_accept_traj = bool(should_accept)
                    if not should_accept_traj and should_accept_fn is not None:
                        reason = "rejected"
                elif isinstance(result, dict):
                    # Legacy dict — check for InteractionWithTokenLogpReward
                    if all(
                        isinstance(v, InteractionWithTokenLogpReward)
                        for v in result.values()
                    ):
                        result = concat_padded_tensors(
                            [v.to_tensor_dict() for v in result.values()]
                        )

                    if should_accept_fn is None:
                        should_accept = True
                    else:
                        should_accept = bool(should_accept_fn(result))
                    should_accept_traj = bool(should_accept)
                    if not should_accept_traj and should_accept_fn is not None:
                        reason = "rejected"

                    # Wrap single dict in list for uniform handling
                    result = [result]
                else:
                    should_accept_traj = False
                    reason = f"unexpected_return_type_{type(result).__name__}"

                if should_accept_traj:
                    manager.on_rollout_accepted()
                    stats_tracker.get("rollout").scalar(accepted=1)
                    trace_session_event(
                        "mark_finalized", task_id=task_id, status="accepted",
                    )
                    assert result is not None and isinstance(result, list)
                    return _TreeSearchRolloutResult(
                        task_id=task_id, trajectories=result
                    )

                manager.on_rollout_rejected()
                stats_tracker.get("rollout").scalar(rejected=1)
                trace_session_event(
                    "mark_finalized", task_id=task_id, status="rejected",
                    reason=reason,
                )
                return None

            except Exception as exc:
                manager.on_rollout_rejected()
                stats_tracker.get("rollout").scalar(rejected=1)
                trace_session_event(
                    "mark_finalized", task_id=task_id, status="failed",
                    reason="workflow_exception",
                )
                if self.logger is not None:
                    self.logger.error(
                        "Workflow execution failed: %s", exc, exc_info=True
                    )
                return None

        return _execute_workflow

    def wait(self, count, timeout=None, raise_timeout=True):
        """Wait for results, extracting list[dict] from _TreeSearchRolloutResult."""
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")
        output = []
        for r in results:
            if r is None:
                output.append(None)
            elif isinstance(r, _TreeSearchRolloutResult):
                output.append(r.trajectories)
            else:
                # Legacy _RolloutResult — wrap in list
                output.append([r.trajectory])
        return output

    def rollout_batch(self, data, workflow):
        """Submit a batch and return flattened list[dict]."""
        from areal.infra.perf_tracer import perf_tracer

        perf_tracer.instant(
            "tree_search_workflow_executor.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(data=item, workflow=workflow)
        results = self.wait(count=len(data))
        # Flatten: filter None, then flatten list[list[dict]] -> list[dict]
        flat = []
        for r in results:
            if r is not None:
                flat.extend(r)
        return flat
```

- [ ] **Step 2: Verify the file parses correctly**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.workflow_executor import TreeSearchWorkflowExecutor; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/workflow_executor.py
git commit -m "feat(tree-search): add TreeSearchWorkflowExecutor for list[dict] returns"
```

______________________________________________________________________

### Task 5: Update proxy_workflow.py to return list\[dict\]

**Files:**

- Modify: `customized_areal/tree_search/proxy_workflow.py`

- [ ] **Step 1: Rewrite `arun_episode` to return list\[dict\]**

Replace the `arun_episode` method in `QueryIDProxyWorkflow`:

```python
async def arun_episode(
    self, engine, data: dict[str, Any]
) -> list[dict[str, Any]] | None:
    # Extract query_id from the input data before it gets lost
    query_id = data.get("query_id", "")

    # Run the base episode logic
    result = await super().arun_episode(engine, data)

    if result is None:
        return None

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    if isinstance(result, dict) and all(
        isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
    ):
        # Convert InteractionWithTokenLogpReward chain to per-turn list[dict]
        sorted_interactions = list(result.values())
        turn_dicts = _interactions_to_turn_dicts(sorted_interactions)
        for td in turn_dicts:
            if query_id:
                td["_mcts_query_id"] = query_id
        return turn_dicts if turn_dicts else None

    # If result is already a list (e.g. from a wrapped workflow), inject query_id
    if isinstance(result, list):
        for td in result:
            if query_id and isinstance(td, dict):
                td["_mcts_query_id"] = query_id
        return result

    # If result is already a tensor dict (shouldn't normally happen),
    # wrap in list and inject query_id
    if isinstance(result, dict) and query_id:
        result["_mcts_query_id"] = query_id
    return [result] if result is not None else None
```

- [ ] **Step 2: Add `_interactions_to_turn_dicts` helper function**

Add to `proxy_workflow.py`:

```python
def _interactions_to_turn_dicts(
    interactions: list[InteractionWithTokenLogpReward],
) -> list[dict[str, Any]]:
    """Convert a list of InteractionWithTokenLogpReward to per-turn list-based dicts.

    Each output dict has Python lists (not tensors), matching the
    TrajectoryRecord-like format used by the tree search pipeline.
    """
    from customized_areal.tree_search.mcts_tree_store import _find_turn_boundaries

    turn_dicts: list[dict[str, Any]] = []

    for interaction in interactions:
        resp = interaction.model_response
        if resp is None:
            continue

        seq = resp.input_tokens + resp.output_tokens
        seq_len = len(seq)

        # Build logprobs, loss_mask, versions
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        # Pad to seq_len if needed
        while len(logprobs) < seq_len:
            logprobs.append(0.0)
        while len(loss_mask) < seq_len:
            loss_mask.append(0)
        while len(versions) < seq_len:
            versions.append(-1)

        # Response-only fields
        response_ids = list(resp.output_tokens)
        logp = list(resp.output_logprobs)

        # Top-k fields from output_top_logprobs
        topk_ids: list[list[int]] = []
        topk_logp: list[list[float]] = []
        if resp.output_top_logprobs is not None:
            for pos_topk in resp.output_top_logprobs:
                ids = [tid for tid, _ in pos_topk]
                lps = [lp for _, lp in pos_topk]
                topk_ids.append(ids)
                topk_logp.append(lps)

        # Placeholder for distill_reward and teacher_logp
        # (populated later by distillation pipeline)
        n_response = len(resp.output_tokens)
        distill_reward: list[list[float]] = []
        teacher_logp: list[list[float]] = []

        # Reward
        reward = interaction.reward if interaction.reward is not None else 0.0

        # Turn boundaries
        starts, ends = _find_turn_boundaries(loss_mask)

        # Episode metadata
        turn_id = interaction.interaction_id or ""
        parent_turn_id = (
            interaction.parent.interaction_id
            if interaction.parent is not None
            else None
        )

        turn_dict: dict[str, Any] = {
            "input_ids": seq,
            "loss_mask": loss_mask,
            "logprobs": logprobs,
            "versions": versions,
            "reward": reward,
            "turn_response_starts": starts,
            "turn_response_ends": ends,
            "turn_ids": [turn_id],
            "parent_turn_ids": [parent_turn_id],
            "turn_rewards": [reward],
            "outcome_reward": reward,
            "response_ids": response_ids,
            "logp": logp,
            "topk_ids": topk_ids,
            "topk_logp": topk_logp,
            "distill_reward": distill_reward,
            "teacher_logp": teacher_logp,
        }
        turn_dicts.append(turn_dict)

    return turn_dicts
```

- [ ] **Step 3: Verify the file parses correctly**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/proxy_workflow.py
git commit -m "feat(tree-search): proxy_workflow returns list[dict] with TrajectoryRecord-like format"
```

______________________________________________________________________

### Task 6: Update grouped_workflow.py to return list\[dict\] of per-episode dicts

**Files:**

- Modify: `customized_areal/tree_search/grouped_workflow.py`

- [ ] **Step 1: Rewrite `arun_episode` to merge per-turn dicts into per-episode dicts**

The key change: instead of calling `concat_padded_tensors` to stack tensors, merge
per-turn list-based dicts into per-episode list-based dicts.

Replace the `arun_episode` method in `TreeSearchGroupedRolloutWorkflow`:

```python
async def arun_episode(
    self, engine: InferenceEngine, data: dict[str, Any]
) -> list[dict[str, Any]] | None:
    results = await asyncio.gather(
        *[
            self.workflow.arun_episode(engine, data)
            for _ in range(self.group_size)
        ]
    )

    valid_results = [r for r in results if r is not None]

    if not valid_results:
        return None

    if len(valid_results) < len(results):
        self.logger.warning(
            f"TreeSearchGroupedWorkflow: "
            f"{len(results) - len(valid_results)}/{len(results)} "
            "trajectories returned None, using remaining results"
        )

    # Check if results are list[dict] (new format from proxy_workflow)
    first = valid_results[0]
    if isinstance(first, list) and (not first or isinstance(first[0], dict)):
        # list[list[dict]] — per-turn dicts, merge into per-episode dicts
        query_id = data.get("query_id", "")
        episode_dicts: list[dict[str, Any]] = []
        for turn_dict_list in valid_results:
            if not turn_dict_list:
                continue
            merged = _merge_turn_dicts_to_episode(turn_dict_list)
            if query_id:
                merged["_mcts_query_id"] = query_id
            episode_dicts.append(merged)
        return episode_dicts if episode_dicts else None

    # Fallback: tensor dict results (legacy behavior)
    first_result = valid_results[0]
    if isinstance(first_result, dict):
        concatenated = concat_padded_tensors(valid_results)
        return [concatenated] if concatenated is not None else None

    return None
```

- [ ] **Step 2: Add `_merge_turn_dicts_to_episode` function**

Add to `grouped_workflow.py`:

```python
def _merge_turn_dicts_to_episode(
    turn_dicts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-turn list-based dicts into a single per-episode dict.

    Concatenates input_ids, loss_mask, logprobs, versions.
    Merges turn_ids, parent_turn_ids, turn_rewards.
    Recomputes turn_response_starts/ends for the full episode.
    Merges response-only fields (logp, topk_ids, topk_logp,
    distill_reward, teacher_logp, response_ids) across turns.
    """
    all_input_ids: list[int] = []
    all_loss_mask: list[int] = []
    all_logprobs: list[float] = []
    all_versions: list[int] = []
    all_turn_ids: list[str] = []
    all_parent_turn_ids: list[str | None] = []
    all_turn_rewards: list[float] = []
    all_logp: list[float] = []
    all_topk_ids: list[list[int]] = []
    all_topk_logp: list[list[float]] = []
    all_distill_reward: list[list[float]] = []
    all_teacher_logp: list[list[float]] = []
    all_response_ids: list[int] = []
    outcome_reward: float = 0.0

    for turn in turn_dicts:
        offset = len(all_input_ids)
        all_input_ids.extend(turn["input_ids"])
        all_loss_mask.extend(turn["loss_mask"])
        all_logprobs.extend(turn.get("logprobs", [0.0] * len(turn["input_ids"])))
        all_versions.extend(turn.get("versions", [0] * len(turn["input_ids"])))

        # Turn metadata
        for tid in turn.get("turn_ids", []):
            all_turn_ids.append(tid)
        for ptid in turn.get("parent_turn_ids", []):
            all_parent_turn_ids.append(ptid)
        for tr in turn.get("turn_rewards", []):
            all_turn_rewards.append(tr)
        outcome_reward = turn.get("outcome_reward", turn.get("reward", 0.0))

        # Response-only fields: concatenate across turns
        all_logp.extend(turn.get("logp", []))
        all_topk_ids.extend(turn.get("topk_ids", []))
        all_topk_logp.extend(turn.get("topk_logp", []))
        all_distill_reward.extend(turn.get("distill_reward", []))
        all_teacher_logp.extend(turn.get("teacher_logp", []))
        all_response_ids.extend(turn.get("response_ids", []))

    starts, ends = _find_turn_boundaries(all_loss_mask)

    result: dict[str, Any] = {
        "input_ids": all_input_ids,
        "loss_mask": all_loss_mask,
        "logprobs": all_logprobs,
        "versions": all_versions,
        "reward": outcome_reward,
        "turn_response_starts": starts,
        "turn_response_ends": ends,
        "turn_ids": all_turn_ids,
        "parent_turn_ids": all_parent_turn_ids,
        "turn_rewards": all_turn_rewards,
        "outcome_reward": outcome_reward,
        "response_ids": all_response_ids,
        "logp": all_logp if all_logp else None,
        "topk_ids": all_topk_ids if all_topk_ids else None,
        "topk_logp": all_topk_logp if all_topk_logp else None,
        "distill_reward": all_distill_reward if all_distill_reward else None,
        "teacher_logp": all_teacher_logp if all_teacher_logp else None,
    }

    return result
```

Add the import at the top of `grouped_workflow.py`:

```python
from customized_areal.tree_search.mcts_tree_store import _find_turn_boundaries
```

- [ ] **Step 3: Remove `_split_to_turn_dicts` function and
  `_sort_interactions_by_creation`, `_collect_episode_metadata`,
  `EPISODE_LEVEL_METADATA_KEYS`**

These are no longer needed since the new pipeline doesn't use stacked tensor dicts.

- [ ] **Step 4: Verify the file parses correctly**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.grouped_workflow import TreeSearchGroupedRolloutWorkflow; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/grouped_workflow.py
git commit -m "feat(tree-search): grouped_workflow returns list[dict] of per-episode dicts"
```

______________________________________________________________________

### Task 7: Update advantage.py to handle list-based dicts

**Files:**

- Modify: `customized_areal/tree_search/advantage.py`

- [ ] **Step 1: Update `compute` to handle `list[int]` input_ids**

The `compute` method currently accesses `traj["input_ids"]` as a tensor
(`input_ids.shape[-1]`, `input_ids.shape[1]`). For list-based dicts, `input_ids` is
`list[int]` and `len()` is used instead.

Replace the `compute` method:

```python
def compute(self, trajectories: list[dict[str, Any]]) -> None:
    """Replace GAE advantages with tree Q-values. Mutates trajectories in-place.

    Handles both tensor-based and list-based trajectory dicts.
    For list-based dicts, input_ids is list[int] and seq_len = len(input_ids).
    """
    # Collect all (query_id, seq_id) pairs for GRPO normalization
    query_groups: dict[str, list[int]] = {}

    for traj in trajectories:
        query_id = traj.get("_mcts_query_id")
        if query_id is None:
            continue
        if "_mcts_seq_ids" in traj:
            for seq_id in traj["_mcts_seq_ids"]:
                query_groups.setdefault(query_id, []).append(seq_id)
        elif "_mcts_seq_id" in traj:
            query_groups.setdefault(query_id, []).append(traj["_mcts_seq_id"])

    # Per-query GRPO normalization
    for query_id, seq_ids in query_groups.items():
        q_values = [self.tree_store._rewards.get(sid, 0.0) for sid in seq_ids]
        if len(q_values) < 2:
            self.tree_store._normalized_advantages[seq_ids[0]] = q_values[0] if q_values else 0.0
            continue
        mean_q = sum(q_values) / len(q_values)
        var_q = sum((q - mean_q) ** 2 for q in q_values) / len(q_values)
        std_q = var_q**0.5
        for sid, q in zip(seq_ids, q_values):
            self.tree_store._normalized_advantages[sid] = (q - mean_q) / (
                std_q + self.grpo_eps
            )

    # Compute per-trajectory advantages using normalized Q-values
    for traj in trajectories:
        query_id = traj.get("_mcts_query_id")
        if query_id is None:
            continue
        input_ids = traj["input_ids"]

        if "_mcts_seq_ids" in traj:
            seq_ids = traj["_mcts_seq_ids"]
            all_advantages = []
            for seq_id in seq_ids:
                if isinstance(input_ids, torch.Tensor):
                    seq_len = input_ids.shape[1]
                else:
                    seq_len = len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)
                adv = self._compute_single(traj, query_id, seq_id, seq_len)
                all_advantages.append(adv)
            advantages = torch.stack(all_advantages, dim=0)
        elif "_mcts_seq_id" in traj:
            seq_id = traj["_mcts_seq_id"]
            if isinstance(input_ids, torch.Tensor):
                seq_len = input_ids.shape[-1]
            else:
                seq_len = len(input_ids)
            advantages = self._compute_single(traj, query_id, seq_id, seq_len)
            if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1:
                advantages = advantages.unsqueeze(0)
        else:
            continue

        traj["advantages"] = advantages
        traj["returns"] = advantages.clone()
```

- [ ] **Step 2: Verify the file parses correctly**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.advantage import TreeAdvantageComputer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/advantage.py
git commit -m "feat(tree-search): advantage.py handles list-based input_ids"
```

______________________________________________________________________

### Task 8: Update trainer.py — patch workflow executor, remove \_split_to_turn_dicts

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Update imports and remove \_split_to_turn_dicts import**

In `customized_areal/tree_search/trainer.py`, change:

```python
from customized_areal.tree_search.grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
    _split_to_turn_dicts,
)
```

to:

```python
from customized_areal.tree_search.grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.workflow_executor import TreeSearchWorkflowExecutor
```

- [ ] **Step 2: Update `_patch_wrap_openai_agent_for_tree_search` to also patch the
  workflow executor**

Add workflow executor patching after the existing patch. In the `_tree_search_wrap`
inner function, after creating the `TreeSearchGroupedRolloutWorkflow`, also swap the
engine's `workflow_executor`:

```python
def _tree_search_wrap(agent: Any, proxy_addr: str):
    from areal.api.cli_args import OpenAIProxyConfig

    openai_cfg = engine.config.openai or OpenAIProxyConfig()
    inner = original_wrap(agent, proxy_addr)
    workflow = TreeSearchGroupedRolloutWorkflow(
        workflow=inner,
        group_size=group_size,
        logger=logger,
    )

    # Replace the workflow executor with a TreeSearchWorkflowExecutor
    # that handles list[dict] returns from arun_episode
    if hasattr(engine, "workflow_executor"):
        old_executor = engine.workflow_executor
        new_executor = TreeSearchWorkflowExecutor(
            config=engine.config,
            inference_engine=engine,
            staleness_manager=old_executor._staleness_manager,
        )
        new_executor.initialize(
            logger=old_executor.logger,
            train_data_parallel_size=old_executor._staleness_manager._max_concurrent_rollouts if hasattr(old_executor._staleness_manager, '_max_concurrent_rollouts') else None,
        )
        engine.workflow_executor = new_executor
        logger.info("Replaced workflow_executor with TreeSearchWorkflowExecutor")

    return workflow
```

- [ ] **Step 3: Update `_unpatch_wrap_openai_agent` to restore the original workflow
  executor**

```python
def _unpatch_wrap_openai_agent(rollout_engine: Any) -> None:
    """Restore the original _wrap_openai_agent method and workflow executor."""
    engine = rollout_engine
    if hasattr(engine, "_original_wrap_openai_agent"):
        engine._wrap_openai_agent = engine._original_wrap_openai_agent
        del engine._original_wrap_openai_agent
        logger.info("Restored original _wrap_openai_agent")
    if hasattr(engine, "_original_workflow_executor"):
        engine.workflow_executor = engine._original_workflow_executor
        del engine._original_workflow_executor
        logger.info("Restored original workflow_executor")
```

Also store the original executor in the patch function (add before replacing):

```python
engine._original_workflow_executor = old_executor
```

- [ ] **Step 4: Update `_cache_aware_prepare_batch` to remove `_split_to_turn_dicts`
  call**

Replace:

```python
trajs = _split_to_turn_dicts(new_trajs) if new_trajs else []
```

with:

```python
trajs = new_trajs if new_trajs else []
```

(The `new_trajs` from `rollout_batch` via `TreeSearchWorkflowExecutor` is already a flat
`list[dict]` of per-episode dicts.)

- [ ] **Step 5: Update the n_new count calculation for list-based dicts**

Replace:

```python
n_new = sum(t["input_ids"].shape[0] for t in trajs) if trajs else 0
```

with:

```python
n_new = sum(len(t["input_ids"]) for t in trajs) if trajs else 0
```

- [ ] **Step 6: Verify the file parses correctly**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.trainer import CacheAwarePPOTrainer; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Run all tree search tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): patch workflow_executor, remove _split_to_turn_dicts from trainer"
```

______________________________________________________________________

### Task 9: Run pre-commit and full test suite

**Files:**

- All modified files

- [ ] **Step 1: Run pre-commit on all changed files**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && pre-commit run --all-files`
Expected: All checks pass. Fix any formatting/linting issues.

- [ ] **Step 2: Run the full tree search test suite**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -v`
Expected: All tests pass.

- [ ] **Step 3: Final commit if any formatting fixes**

```bash
git add -A
git commit -m "style: pre-commit fixes for tree search list-dict pipeline"
```

______________________________________________________________________

## Self-Review

**1. Spec coverage:**

- Per-dict structure with all fields → Task 1 (TrajectoryRecord), Task 5 (proxy_workflow
  extraction), Task 6 (grouped_workflow merge)
- Data flow (per-turn → per-episode → per-query) → Tasks 5, 6, 8
- TreeSearchWorkflowExecutor → Task 4
- proxy_workflow returns list\[dict\] → Task 5
- grouped_workflow returns list\[dict\] → Task 6
- trainer.py patches → Task 8
- mcts_tree_store insert/load with new fields → Tasks 1, 2, 3
- advantage.py list-based support → Task 7
- checkpoint serialization → Task 2

**2. Placeholder scan:** No TBDs, TODOs, or "implement later" patterns.

**3. Type consistency:**

- `TrajectoryRecord.logp` is `list[float] | None` — consistent in insert (Task 1), load
  (Task 3), proxy extraction (Task 5), merge (Task 6)
- `topk_ids` is `list[list[int]] | None` — consistent throughout
- `response_ids` is `list[int]` — consistent in proxy (Task 5) and merge (Task 6)
- `input_ids` in list dicts is `list[int]` — handled in advantage.py (Task 7) and
  trainer.py (Task 8)

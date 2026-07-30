# train_id-based Trained Determination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace boolean `_trained` flag with `train_id`-based comparison so a Node is
"trained" only when its `train_id` matches the current run's.

**Architecture:** Add `train_id` field to Node dataclass. MCTSTreeStore reads
`current_train_id` from `TRAIN_ID` env var and compares
`node.train_id == self.current_train_id` in all trained-check methods. Checkpoint
serializes `train_id` per-node. Training script generates UUID and exports to env.

**Tech Stack:** Python 3.12+, dataclasses, json

______________________________________________________________________

### Task 1: Add `train_id` field to Node dataclass

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:36-60`

- [ ] **Step 1: Add `train_id` field to Node**

```python
@dataclass
class Node:
    """A single turn in a multi-turn conversation tree.
    ...
    """

    # Core sequence (full turn: prompt + response)
    input_ids: list[int]
    loss_mask: list[int]  # 0=prompt, 1=response
    logprobs: list[float]  # full sequence (0.0 on prompt positions)
    versions: list[int]  # policy version (-1 on prompt)

    # Tree structure
    node_id: str = ""  # globally unique interaction ID (UUID from inference engine)
    parent_node_id: str | None = None  # parent interaction ID (None for root)
    episode_id: str = ""  # groups turns into a trajectory path
    turn_idx: int = 0  # 1-based turn position within episode
    query_id: str = ""  # dataset query identifier
    train_id: str = ""  # training run that trained this node; "" means untrained

    # Reward
    outcome_reward: float = 0.0

    # Tree-computed advantages/returns (set by TreeAdvantageComputer)
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
```

- [ ] **Step 2: Run existing tests to verify nothing breaks with the new default field**

```bash
uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v
```

Expected: All tests pass (new field defaults to `""`, no behavior change).

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "feat: add train_id field to Node dataclass"
```

______________________________________________________________________

### Task 2: Replace `_trained` dict with `train_id`-based logic in MCTSTreeStore

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:170-350`

- [ ] **Step 1: Add `current_train_id` to `__init__` and remove `_trained`**

In `MCTSTreeStore.__init__`, replace:

```python
self._trained: dict[str, bool] = {}
```

with:

```python
self.current_train_id: str = os.environ.get("TRAIN_ID", "")
```

And add `import os` at the top of the file.

- [ ] **Step 2: Rewrite `_insert_single` to remove `_trained[` references**

In `_insert_single`, remove the line:

```python
self._trained[node_id] = False
```

Nodes default `train_id=""` from the dataclass, so they are automatically "untrained."

- [ ] **Step 3: Rewrite `set_trained`**

Replace:

```python
def set_trained(self, node_id: str, trained: bool = True) -> None:
    self._trained[node_id] = trained
```

with:

```python
def set_trained(self, node_id: str, trained: bool = True) -> None:
    """Stamp the node with current_train_id to mark it as trained."""
    if not trained:
        return
    key = self._node_id_to_key.get(node_id)
    if key is None:
        return
    query_id, idx = key
    node = self.trajectories[query_id][idx]
    if isinstance(node, dict):
        node["train_id"] = self.current_train_id
    else:
        node.train_id = self.current_train_id
```

- [ ] **Step 4: Rewrite `is_trained`**

Replace:

```python
def is_trained(self, node_id: str) -> bool:
    return self._trained.get(node_id, False)
```

with:

```python
def is_trained(self, node_id: str) -> bool:
    """A node is trained if its train_id matches the current run's train_id."""
    key = self._node_id_to_key.get(node_id)
    if key is None:
        return False
    query_id, idx = key
    node = self.trajectories[query_id][idx]
    if isinstance(node, dict):
        return node.get("train_id", "") == self.current_train_id
    return node.train_id == self.current_train_id
```

- [ ] **Step 5: Rewrite `get_untrained_count`**

Replace `self._trained.get(node_id, False)` with `not self.is_trained(node_id)`:

```python
def get_untrained_count(self, query_id: str) -> int:
    if query_id not in self._query_node_ids:
        return 0
    return sum(
        1
        for node_id in self._query_node_ids[query_id]
        if not self.is_trained(node_id)
    )
```

- [ ] **Step 6: Rewrite `get_untrained_node_ids`**

Replace `self._trained.get(node_id, False)` with `not self.is_trained(node_id)`:

```python
def get_untrained_node_ids(self, query_id: str, n_samples: int) -> list[str]:
    if query_id not in self._query_node_ids:
        return []
    result: list[str] = []
    for node_id in self._query_node_ids[query_id]:
        if not self.is_trained(node_id):
            result.append(node_id)
            if len(result) >= n_samples:
                break
    return result
```

- [ ] **Step 7: Remove `reset_trained_flags` method**

Delete the entire method:

```python
def reset_trained_flags(self) -> None:
    for key in self._trained:
        self._trained[key] = False
```

- [ ] **Step 8: Rewrite `mark_episodes_trained`**

Replace the boolean-based logic with `train_id` stamping:

```python
def mark_episodes_trained(self, episode_ids: set[str]) -> None:
    """Set train_id based on episode IDs.

    Nodes whose episode_id is in the given set are stamped with
    current_train_id. All other nodes have train_id cleared.
    Episode IDs not present in the store are silently ignored.
    """
    for query_id, records in self.trajectories.items():
        for node in records:
            if isinstance(node, dict):
                nid_val = node.get("episode_id", "")
            else:
                nid_val = node.episode_id
            if nid_val in episode_ids:
                if isinstance(node, dict):
                    node["train_id"] = self.current_train_id
                else:
                    node.train_id = self.current_train_id
            else:
                if isinstance(node, dict):
                    node["train_id"] = ""
                else:
                    node.train_id = ""
```

- [ ] **Step 9: Update `clear` to remove `_trained` references**

In `clear()`, remove the line:

```python
self._trained.clear()
```

- [ ] **Step 10: Run tests to verify**

```bash
uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v
```

Expected: Some tests will fail because they test the old `_trained`-based behavior.
That's expected — we'll update tests next.

- [ ] **Step 11: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "feat: replace _trained dict with train_id-based comparison"
```

______________________________________________________________________

### Task 3: Update checkpoint serialization for `train_id`

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:110-151` (serialize/deserialize),
  `:39-48` (metadata save), `:66-86` (metadata load), `:154-166` (save_trained_episodes)

- Modify: `customized_areal/tree_search/migrate_checkpoint.py:150-162` (metadata
  rebuild)

- [ ] **Step 1: Add `train_id` to `_serialize_record`**

In `TreeCheckpointManager._serialize_record`, add after `"query_id": node.query_id`:

```python
"train_id": node.train_id,
```

- [ ] **Step 2: Add `train_id` to `_deserialize_record`**

In `TreeCheckpointManager._deserialize_record`, add to the `Node(...)` constructor call
after `query_id=data.get("query_id", "")`:

```python
train_id=data.get("train_id", ""),
```

- [ ] **Step 3: Update metadata save to include `current_train_id`**

In `save()`, in the metadata dict, replace:

```python
"trained": {k: v for k, v in tree_store._trained.items()},
```

with:

```python
"current_train_id": tree_store.current_train_id,
```

- [ ] **Step 4: Update metadata load to read `current_train_id`**

In `load()`, replace:

```python
store._trained = {k: v for k, v in metadata.get("trained", {}).items()}
```

with:

```python
store.current_train_id = metadata.get("current_train_id", "")
```

- [ ] **Step 5: Update `save_trained_episodes` to use `train_id` comparison**

Replace the `is_trained` check with direct `train_id` comparison:

```python
@staticmethod
def save_trained_episodes(
    recover_checkpoint_dir: str, tree_store: MCTSTreeStore
) -> None:
    """Save trained episode IDs to the recover checkpoint directory."""
    trained_ids: set[str] = set()
    for query_id, records in tree_store.trajectories.items():
        for node in records:
            if isinstance(node, dict):
                if node.get("train_id", "") == tree_store.current_train_id:
                    trained_ids.add(node.get("episode_id", ""))
            else:
                if node.train_id == tree_store.current_train_id:
                    trained_ids.add(node.episode_id)
    data = {"trained_episode_ids": sorted(trained_ids)}
    os.makedirs(recover_checkpoint_dir, exist_ok=True)
    filepath = os.path.join(recover_checkpoint_dir, "trained_episodes.json")
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)
```

- [ ] **Step 6: Update `migrate_checkpoint.py` metadata rebuild**

In `migrate_checkpoint.py`, replace:

```python
"trained": meta.get("trained", {}),
```

with:

```python
"current_train_id": meta.get("current_train_id", ""),
```

- [ ] **Step 7: Run checkpoint tests**

```bash
uv run pytest tests/test_tree_search/test_checkpoint.py -v
```

Expected: Failure on `test_load_preserves_trained_flags` (uses old `_trained` API).
Other tests may also fail. We'll fix tests in the next task.

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py customized_areal/tree_search/migrate_checkpoint.py
git commit -m "feat: serialize train_id in checkpoint and update metadata format"
```

______________________________________________________________________

### Task 4: Remove `reset_trained_flags()` call from workflow

**Files:**

- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py:233`

- [ ] **Step 1: Remove the `reset_trained_flags()` call**

In `TreeSearchGroupedRolloutWorkflow.__init__`, delete the line:

```python
self.tree_store.reset_trained_flags()
```

The `train_id` comparison handles this automatically — nodes from old runs have
mismatched `train_id` and are treated as untrained.

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py
git commit -m "feat: remove reset_trained_flags call from workflow init"
```

______________________________________________________________________

### Task 5: Generate and export `train_id` in training script

**Files:**

- Modify: `customized_areal/tpfc/scripts/train_tpfc_tree_search.py:1-12`

- [ ] **Step 1: Add UUID generation and env export**

Add `import os` and `import uuid` to the imports. Add before
`config, _ = load_expr_config(...)`:

```python
import os
import uuid

# Generate a unique train_id for this training run
if "TRAIN_ID" not in os.environ:
    os.environ["TRAIN_ID"] = uuid.uuid4().hex
logger.info("Train ID: %s", os.environ["TRAIN_ID"])
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tpfc/scripts/train_tpfc_tree_search.py
git commit -m "feat: generate train_id UUID and export to TRAIN_ID env var"
```

______________________________________________________________________

### Task 6: Update tests

**Files:**

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`

- Modify: `tests/test_tree_search/test_checkpoint.py`

- Modify: `tests/test_treesearch_bugfixes.py`

- [ ] **Step 1: Update MCTSTreeStore tests — `TestMCTSTreeStoreTrainedFlag`**

Replace the class with `train_id`-based tests:

```python
class TestMCTSTreeStoreTrainId:
    def test_train_id_default_empty(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        assert traj.train_id == ""

    def test_set_trained_stamps_current_train_id(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        store.set_trained(traj.node_id, True)
        assert store.is_trained(traj.node_id) is True
        assert traj.train_id == "run_001"

    def test_is_trained_false_when_different_train_id(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_002"
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        traj.train_id = "run_001"  # old run
        assert store.is_trained(traj.node_id) is False

    def test_is_trained_false_when_empty_train_id(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        assert traj.train_id == ""
        assert store.is_trained(traj.node_id) is False

    def test_get_untrained_count_with_different_train_ids(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_002"
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="t1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1", node_id="t2")
        t3 = _make_traj([7, 8, 9], [0, 0, 1], reward=0.3, query_id="q1", node_id="t3")
        store.insert_batch([t1, t2, t3])
        t1.train_id = "run_002"  # trained in current run
        t2.train_id = "run_001"  # trained in old run
        # t3.train_id = "" (untrained)
        assert store.get_untrained_count("q1") == 2  # t2 and t3
        store.set_trained(t2.node_id, True)
        assert store.get_untrained_count("q1") == 1  # only t3

    def test_mark_episodes_trained(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        n3 = Node(
            input_ids=[7, 8, 9], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3], versions=[0, 0, 0],
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=0.3, query_id="q1",
        )
        store.insert_batch([n1, n2, n3])
        store.mark_episodes_trained({"ep_a"})
        assert store.is_trained(n1.node_id) is True
        assert n1.train_id == "run_001"
        assert store.is_trained(n2.node_id) is False
        assert n2.train_id == ""
        assert store.is_trained(n3.node_id) is True
        assert n3.train_id == "run_001"

    def test_mark_episodes_trained_resets_others(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)
        store.set_trained(n2.node_id, True)
        store.mark_episodes_trained({"ep_b"})
        assert store.is_trained(n1.node_id) is False
        assert n1.train_id == ""
        assert store.is_trained(n2.node_id) is True
        assert n2.train_id == "run_001"

    def test_mark_episodes_trained_empty_set(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        store.set_trained(n1.node_id, True)
        store.mark_episodes_trained(set())
        assert store.is_trained(n1.node_id) is False
        assert n1.train_id == ""

    def test_mark_episodes_trained_unknown_episode(self):
        """Episode IDs not in the store are silently ignored."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        store.mark_episodes_trained({"nonexistent_episode"})
        assert store.is_trained(n1.node_id) is False
        assert n1.train_id == ""
```

Remove the old `TestMCTSTreeStoreTrainedFlag` class and `test_reset_trained_flags`
method entirely.

- [ ] **Step 2: Update `TestSetTrainedSignature` in test_treesearch_bugfixes.py**

The test `test_set_trained_accepts_node_id_only` needs `current_train_id` to be set:

```python
class TestSetTrainedSignature:
    """Bug #13: set_trained should not take unused query_id parameter."""

    def test_set_trained_accepts_node_id_only(self):
        store = MCTSTreeStore()
        store.current_train_id = "test_run"
        node = _make_node()
        store.insert_batch([node])
        node_id = node.node_id

        store.set_trained(node_id, True)
        assert store.is_trained(node_id) is True
        assert store.get_reward(node_id) == 1.0
```

- [ ] **Step 3: Update checkpoint test — `test_load_preserves_trained_flags`**

Replace with a `train_id`-based test:

```python
def test_load_preserves_train_id(self, tmp_path):
    manager = TreeCheckpointManager(str(tmp_path))
    store = _make_store_with_data()
    store.current_train_id = "run_001"
    node_ids = store._query_node_ids["q1"]
    store.set_trained(node_ids[0], True)
    manager.save(store)

    loaded = manager.load()
    assert loaded.current_train_id == "run_001"
    assert loaded.is_trained(node_ids[0]) is True
```

- [ ] **Step 4: Update integration test — `TestTrainedEpisodesRestoreIntegration`**

The `test_save_restore_cycle` test needs `current_train_id` set on both stores:

```python
def test_save_restore_cycle(self, tmp_path):
    """Full save -> load -> mark_episodes_trained cycle."""
    store = MCTSTreeStore()
    store.current_train_id = "run_001"
    n1 = Node(
        input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
        episode_id="ep_1", outcome_reward=1.0, query_id="q1",
    )
    n2 = Node(
        input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
        episode_id="ep_2", outcome_reward=0.5, query_id="q2",
    )
    store.insert_batch([n1, n2])
    store.set_trained(n1.node_id, True)

    recover_dir = str(tmp_path / "recover_checkpoint")
    TreeCheckpointManager.save_trained_episodes(recover_dir, store)

    fresh_store = MCTSTreeStore()
    fresh_store.current_train_id = "run_001"
    fresh_n1 = Node(
        input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
        episode_id="ep_1", outcome_reward=1.0, query_id="q1",
    )
    fresh_n2 = Node(
        input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
        episode_id="ep_2", outcome_reward=0.5, query_id="q2",
    )
    fresh_store.insert_batch([fresh_n1, fresh_n2])

    trained_episodes = TreeCheckpointManager.load_trained_episodes(recover_dir)
    assert trained_episodes is not None
    fresh_store.mark_episodes_trained(trained_episodes)

    assert fresh_store.is_trained(fresh_n1.node_id) is True
    assert fresh_store.is_trained(fresh_n2.node_id) is False
```

- [ ] **Step 5: Update `test_untrained_nodes_not_in_saved_episodes`**

Add `store.current_train_id = "run_001"` before `store.set_trained(n1.node_id, True)`.

- [ ] **Step 6: Run all tree search tests**

```bash
uv run pytest tests/test_tree_search/ tests/test_treesearch_bugfixes.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add tests/test_tree_search/test_mcts_tree_store.py tests/test_tree_search/test_checkpoint.py tests/test_treesearch_bugfixes.py
git commit -m "test: update tests for train_id-based trained determination"
```

______________________________________________________________________

### Task 7: Clean up config — remove `agent.train_id`

**Files:**

- Modify:
  `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml:19`

- [ ] **Step 1: Remove `train_id` from agent config**

In the YAML config, change:

```yaml
agent:
  trial_name: ${trial_name}
  train_id: ""
  user_id: ""
```

to:

```yaml
agent:
  trial_name: ${trial_name}
  user_id: ""
```

The `TPFCAgent` still accepts `train_id` in its constructor (it's optional), so this is
backward-compatible. If an agent needs a `train_id`, it'll get `""` by default.

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml
git commit -m "config: remove agent.train_id from tree search config"
```

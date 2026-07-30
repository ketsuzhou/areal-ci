# Training Order Recording & Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the exact order of trajectories used at each training step in
`CacheAwarePPOTrainer` (stored in trie leaf nodes and store-level history), and support
deterministic replay of previous training runs using cached trajectories only.

**Architecture:** Add `training_steps` list to `TrieNode` leaf nodes and
`_training_history` dict to `MCTSTreeStore`. Inject `_global_step` into trajectory dicts
before `compute_advantages`. In the patched method, call `record_training_step` to
persist order. Add a `replay` mode to `RolloutCacheConfig` that loads from
`_training_history` instead of the dataloader.

**Tech Stack:** Python 3.12+ | PyTorch | dataclasses | JSON serialization

______________________________________________________________________

## File Structure

| File                                              | Responsibility                                                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `customized_areal/tree_search/trie_node.py`       | Add `training_steps` field to `TrieNode`                                                                                       |
| `customized_areal/tree_search/mcts_tree_store.py` | Add `_training_history`, `record_training_step()`, `load_trajectory_by_seq_id()`, `build_training_history()`, update `clear()` |
| `customized_areal/tree_search/checkpoint.py`      | Serialize/deserialize `training_steps` on nodes and `_training_history` in metadata                                            |
| `customized_areal/tree_search/config.py`          | Add `replay: bool` to `RolloutCacheConfig`                                                                                     |
| `customized_areal/tree_search/trainer.py`         | Call `record_training_step` in patched method; add replay mode to `_cache_aware_prepare_batch`                                 |
| `areal/trainer/rl_trainer.py`                     | Inject `_global_step` on trajectories before `compute_advantages`                                                              |
| `tests/test_tree_search/test_mcts_tree_store.py`  | Unit tests for all new functionality                                                                                           |

______________________________________________________________________

### Task 1: Add `training_steps` field to TrieNode

**Files:**

- Modify: `customized_areal/tree_search/trie_node.py:10-28`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestTrieNodeTrainingSteps:
    def test_training_steps_default_empty(self):
        from customized_areal.tree_search.trie_node import TrieNode

        node = TrieNode(tree_id=0)
        assert node.training_steps == []

    def test_training_steps_append(self):
        from customized_areal.tree_search.trie_node import TrieNode

        node = TrieNode(tree_id=0)
        node.training_steps.append(5)
        node.training_steps.append(10)
        assert node.training_steps == [5, 10]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTrieNodeTrainingSteps -v`
Expected: FAIL — `TrieNode.__init__()` got an unexpected keyword argument
`training_steps` (or similar TypeError from dataclass)

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/trie_node.py`, add `training_steps` field to the
`TrieNode` dataclass:

```python
@dataclass
class TrieNode:
    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    training_steps: list[int] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTrieNodeTrainingSteps -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All
existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/trie_node.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add training_steps field to TrieNode"
```

______________________________________________________________________

### Task 2: Add `record_training_step` to MCTSTreeStore

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:42-368`

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestMCTSTreeStoreRecordTrainingStep:
    def test_record_training_step_single_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajectories = [{"_mcts_query_id": "q1", "_mcts_seq_id": seq_id}]
        store.record_training_step(0, trajectories)
        root = store.trees["q1"]
        leaf = root.get_path_nodes(seq_id)[-1]
        assert leaf.training_steps == [0]
        assert 0 in store._training_history
        assert store._training_history[0] == [("q1", seq_id)]

    def test_record_training_step_multiple_trajectories(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        trajectories = [
            {"_mcts_query_id": "q1", "_mcts_seq_id": s0},
            {"_mcts_query_id": "q2", "_mcts_seq_id": s1},
        ]
        store.record_training_step(3, trajectories)
        # Leaf of q1/s0
        leaf0 = store.trees["q1"].get_path_nodes(s0)[-1]
        assert leaf0.training_steps == [3]
        # Leaf of q2/s1
        leaf1 = store.trees["q2"].get_path_nodes(s1)[-1]
        assert leaf1.training_steps == [3]
        # Store-level history preserves order
        assert store._training_history[3] == [("q1", s0), ("q2", s1)]

    def test_record_training_step_grouped_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        trajectories = [
            {"_mcts_query_id": "q1", "_mcts_seq_ids": [s0, s1]},
        ]
        store.record_training_step(1, trajectories)
        leaf0 = store.trees["q1"].get_path_nodes(s0)[-1]
        leaf1 = store.trees["q1"].get_path_nodes(s1)[-1]
        assert leaf0.training_steps == [1]
        assert leaf1.training_steps == [1]
        assert store._training_history[1] == [("q1", s0), ("q1", s1)]

    def test_record_training_step_skips_missing_global_step(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajectories = [{"_mcts_query_id": "q1", "_mcts_seq_id": seq_id}]
        # No _global_step key — should skip gracefully
        store.record_training_step(None, trajectories)
        leaf = store.trees["q1"].get_path_nodes(seq_id)[-1]
        assert leaf.training_steps == []
        assert len(store._training_history) == 0

    def test_record_training_step_same_trajectory_multiple_steps(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajectories = [{"_mcts_query_id": "q1", "_mcts_seq_id": seq_id}]
        store.record_training_step(0, trajectories)
        store.record_training_step(5, trajectories)
        leaf = store.trees["q1"].get_path_nodes(seq_id)[-1]
        assert leaf.training_steps == [0, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreRecordTrainingStep -v`
Expected: FAIL —
`AttributeError: 'MCTSTreeStore' object has no attribute 'record_training_step'`

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/mcts_tree_store.py`, add `_training_history` to
`__init__` and implement `record_training_step`:

In `__init__`, add after `self._rewards`:

```python
self._training_history: dict[int, list[tuple[str, int]]] = {}
```

Add the new method after `insert_batch`:

```python
def record_training_step(
    self, global_step: int | None, trajectories: list[dict[str, Any]]
) -> None:
    """Record that the given trajectories were used for training at global_step.

    Appends global_step to each trajectory's leaf node training_steps list
    and stores the ordered (query_id, seq_id) list in _training_history.
    Skips gracefully if global_step is None.
    """
    if global_step is None:
        return

    ordered_pairs: list[tuple[str, int]] = []

    for traj in trajectories:
        query_id = traj.get("_mcts_query_id")
        if query_id is None:
            continue

        # Single trajectory
        seq_id = traj.get("_mcts_seq_id")
        if seq_id is not None and query_id in self.trees:
            root = self.trees[query_id]
            path_nodes = root.get_path_nodes(seq_id)
            if path_nodes:
                leaf = path_nodes[-1]
                leaf.training_steps.append(global_step)
            ordered_pairs.append((query_id, seq_id))
            continue

        # Grouped trajectory
        seq_ids = traj.get("_mcts_seq_ids")
        if seq_ids is not None and query_id in self.trees:
            root = self.trees[query_id]
            for sid in seq_ids:
                path_nodes = root.get_path_nodes(sid)
                if path_nodes:
                    leaf = path_nodes[-1]
                    leaf.training_steps.append(global_step)
                ordered_pairs.append((query_id, sid))

    if ordered_pairs:
        self._training_history[global_step] = ordered_pairs
```

Also update `clear()` to include `_training_history`:

```python
def clear(self) -> None:
    """Reset all trees, stats, and cursors."""
    self.trees.clear()
    self._next_seq_id = 0
    self._cursors.clear()
    self._visit_counts.clear()
    self._total_values.clear()
    self._q_values.clear()
    self._trained.clear()
    self._rewards.clear()
    self._training_history.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreRecordTrainingStep -v`
Expected: PASS

- [ ] **Step 5: Run all tree search tests**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All
tests PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add record_training_step and _training_history to MCTSTreeStore"
```

______________________________________________________________________

### Task 3: Add `load_trajectory_by_seq_id` to MCTSTreeStore

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py`

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestMCTSTreeStoreLoadBySeqId:
    def test_load_trajectory_by_seq_id(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        traj = store.load_trajectory_by_seq_id("q1", seq_id)
        assert traj is not None
        assert traj["input_ids"].shape[0] == 1
        assert traj["rewards"].item() == 1.0
        assert traj["_mcts_query_id"] == "q1"
        assert traj["_mcts_seq_id"] == seq_id

    def test_load_trajectory_by_seq_id_unknown(self):
        store = MCTSTreeStore(_two_turn_splitter)
        result = store.load_trajectory_by_seq_id("nonexistent", 0)
        assert result is None

    def test_load_trajectory_by_seq_id_matches_load_trajectories(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        # Mark s0 as trained so load_trajectories returns s1
        store.set_trained("q1", s0, True)
        trajs = store.load_trajectories("q1", n_samples=1)
        assert len(trajs) == 1
        assert trajs[0]["_mcts_seq_id"] == s1
        # load_by_seq_id can still load s0 regardless of trained flag
        traj = store.load_trajectory_by_seq_id("q1", s0)
        assert traj is not None
        assert traj["_mcts_seq_id"] == s0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreLoadBySeqId -v`
Expected: FAIL —
`AttributeError: 'MCTSTreeStore' object has no attribute 'load_trajectory_by_seq_id'`

- [ ] **Step 3: Write minimal implementation**

Add to `customized_areal/tree_search/mcts_tree_store.py`, after `load_trajectories`:

```python
def load_trajectory_by_seq_id(
    self, query_id: str, seq_id: int
) -> dict[str, Any] | None:
    """Load a single trajectory by its exact seq_id.

    Unlike load_trajectories, this ignores the trained flag and returns
    the trajectory regardless. Returns None if query_id or seq_id not found.
    """
    if query_id not in self.trees:
        return None

    root = self.trees[query_id]
    if seq_id not in root.sequence_ids:
        return None

    path_nodes = root.get_path_nodes(seq_id)

    all_tokens = []
    all_logprobs = []
    all_versions = []
    prompt_len_total = 0

    for node in path_nodes:
        all_tokens.extend(node.tokens)
        if node.logprobs:
            all_logprobs.extend(node.logprobs)
        else:
            all_logprobs.extend([0.0] * len(node.tokens))
        if node.versions:
            all_versions.extend(node.versions)
        else:
            all_versions.extend([0] * len(node.tokens))
        prompt_len_total += node.prompt_len

    seq_len = len(all_tokens)
    if seq_len == 0:
        return None

    input_ids = torch.tensor(all_tokens, dtype=torch.int32).unsqueeze(0)
    logprobs_t = torch.tensor(all_logprobs, dtype=torch.float32).unsqueeze(0)
    versions_t = torch.tensor(all_versions, dtype=torch.int32).unsqueeze(0)
    attention_mask = torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)

    loss_mask = torch.zeros(seq_len, dtype=torch.int32)
    loss_mask[prompt_len_total:] = 1
    loss_mask = loss_mask.unsqueeze(0)

    reward_val = self.get_reward(query_id, seq_id)
    rewards = torch.tensor([reward_val], dtype=torch.float32).unsqueeze(0)

    return {
        "input_ids": input_ids,
        "logprobs": logprobs_t,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "rewards": rewards,
        "versions": versions_t,
        "_mcts_query_id": query_id,
        "_mcts_seq_id": seq_id,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreLoadBySeqId -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add load_trajectory_by_seq_id to MCTSTreeStore"
```

______________________________________________________________________

### Task 4: Add `build_training_history` to MCTSTreeStore

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py`

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestMCTSTreeStoreBuildTrainingHistory:
    def _insert_and_record(self, store, query_id, tokens, reward, step):
        seq_id = store.insert_trajectory(query_id, tokens, reward=reward)
        trajectories = [{"_mcts_query_id": query_id, "_mcts_seq_id": seq_id}]
        store.record_training_step(step, trajectories)
        return seq_id

    def test_build_training_history_from_leaves(self):
        store = MCTSTreeStore(_two_turn_splitter)
        self._insert_and_record(store, "q1", [1, 2, 10, 3, 4], 1.0, 0)
        self._insert_and_record(store, "q2", [5, 6, 10, 7, 8], 0.5, 0)
        self._insert_and_record(store, "q1", [1, 2, 10, 3, 5], 0.3, 1)

        # Clear history and rebuild from leaves
        store._training_history.clear()
        store.build_training_history()

        assert 0 in store._training_history
        assert 1 in store._training_history
        # Step 0 should have q1 and q2 trajectories
        step0_pairs = store._training_history[0]
        assert len(step0_pairs) == 2
        # Step 1 should have one trajectory
        assert len(store._training_history[1]) == 1

    def test_build_training_history_empty(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.build_training_history()
        assert store._training_history == {}

    def test_build_training_history_preserves_existing(self):
        store = MCTSTreeStore(_two_turn_splitter)
        self._insert_and_record(store, "q1", [1, 2, 10, 3, 4], 1.0, 0)
        # _training_history already has step 0 — build_training_history should not overwrite
        original = store._training_history[0]
        store.build_training_history()
        assert store._training_history[0] == original
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store_store.py::TestMCTSTreeStoreBuildTrainingHistory -v`
Expected: FAIL —
`AttributeError: 'MCTSTreeStore' object has no attribute 'build_training_history'`

- [ ] **Step 3: Write minimal implementation**

Add to `customized_areal/tree_search/mcts_tree_store.py`, after
`load_trajectory_by_seq_id`:

```python
def build_training_history(self) -> None:
    """Reconstruct _training_history from leaf node training_steps.

    Fallback for old checkpoints that lack _training_history in metadata.
    Within each global_step, order is best-effort: trajectories are ordered
    by their seq_id position in root.sequence_ids per query_id.
    Cross-query_id ordering is not guaranteed.
    Does not overwrite existing _training_history entries.
    """
    if self._training_history:
        return

    # Collect (global_step, query_id, seq_id) from all leaves
    step_entries: dict[int, list[tuple[str, int, int]]] = {}
    for query_id, root in self.trees.items():
        for seq_id in set(root.sequence_ids):
            path_nodes = root.get_path_nodes(seq_id)
            if not path_nodes:
                continue
            leaf = path_nodes[-1]
            for step in leaf.training_steps:
                if step not in step_entries:
                    step_entries[step] = []
                # Use seq_id position in root.sequence_ids for ordering
                try:
                    order = root.sequence_ids.index(seq_id)
                except ValueError:
                    order = seq_id
                step_entries[step].append((query_id, seq_id, order))

    # Build _training_history, sorted by seq_id order within each step
    for step in sorted(step_entries.keys()):
        entries = step_entries[step]
        entries.sort(key=lambda x: x[2])
        self._training_history[step] = [(qid, sid) for qid, sid, _ in entries]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreBuildTrainingHistory -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add build_training_history to MCTSTreeStore"
```

______________________________________________________________________

### Task 5: Update TreeCheckpointManager for training_steps and \_training_history

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:83-122`

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
import os
import tempfile

from customized_areal.tree_search.checkpoint import TreeCheckpointManager


class TestTreeCheckpointTrainingHistory:
    def test_save_load_training_steps_and_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MCTSTreeStore(_two_turn_splitter)
            s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
            s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
            trajectories = [
                {"_mcts_query_id": "q1", "_mcts_seq_id": s0},
                {"_mcts_query_id": "q2", "_mcts_seq_id": s1},
            ]
            store.record_training_step(0, trajectories)

            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            loaded = mgr.load(_two_turn_splitter)

            # Check _training_history preserved
            assert 0 in loaded._training_history
            assert loaded._training_history[0] == [("q1", s0), ("q2", s1)]

            # Check leaf training_steps preserved
            leaf0 = loaded.trees["q1"].get_path_nodes(s0)[-1]
            assert leaf0.training_steps == [0]
            leaf1 = loaded.trees["q2"].get_path_nodes(s1)[-1]
            assert leaf1.training_steps == [0]

    def test_load_old_checkpoint_without_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MCTSTreeStore(_two_turn_splitter)
            s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)

            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            # Manually remove training_history from metadata to simulate old checkpoint
            metadata_path = os.path.join(tmpdir, "mcts_trees", "metadata.json")
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            del metadata["training_history"]
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            loaded = mgr.load(_two_turn_splitter)
            assert loaded._training_history == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTreeCheckpointTrainingHistory -v`
Expected: FAIL — `training_steps` not serialized/deserialized, `training_history` not in
metadata

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/checkpoint.py`, update `_serialize_node` to include
`training_steps`:

```python
def _serialize_node(self, node: TrieNode) -> dict:
    result = {
        "tree_id": node.tree_id,
        "start_idx": node.start_idx,
        "end_idx": node.end_idx,
        "tokens": node.tokens,
        "sequence_ids": list(node.sequence_ids),
        "children": {
            str(key): self._serialize_node(child)
            for key, child in node.children.items()
        },
    }
    if node.prompt_len > 0:
        result["prompt_len"] = node.prompt_len
    if node.logprobs:
        result["logprobs"] = node.logprobs
    if node.versions:
        result["versions"] = node.versions
    if node.training_steps:
        result["training_steps"] = node.training_steps
    return result
```

Update `_deserialize_node` to restore `training_steps`:

```python
def _deserialize_node(
    self, data: dict, parent: TrieNode | None, tree_id: int
) -> TrieNode:
    node = TrieNode(
        tree_id=tree_id,
        start_idx=data["start_idx"],
        end_idx=data["end_idx"],
        tokens=data["tokens"],
        sequence_ids=data["sequence_ids"],
        prompt_len=data.get("prompt_len", 0),
        logprobs=data.get("logprobs", []),
        versions=data.get("versions", []),
        training_steps=data.get("training_steps", []),
    )
    if parent is not None:
        node.ancestors = parent.ancestors + [parent]
    for key_str, child_data in data["children"].items():
        key = int(key_str)
        child = self._deserialize_node(child_data, parent=node, tree_id=tree_id)
        node.children[key] = child
    return node
```

Update `save` to include `_training_history` in metadata:

```python
def save(self, tree_store: MCTSTreeStore) -> None:
    os.makedirs(self.save_dir, exist_ok=True)
    for query_id, root in tree_store.trees.items():
        tree_data = {"root": self._serialize_node(root)}
        filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
        with open(filepath, "w") as f:
            json.dump(tree_data, f)

    trained_data = {
        f"{qid}:{sid}": trained
        for (qid, sid), trained in tree_store._trained.items()
    }
    rewards_data = {
        f"{qid}:{sid}": reward for (qid, sid), reward in tree_store._rewards.items()
    }
    training_history_data = {
        str(step): [[qid, sid] for qid, sid in pairs]
        for step, pairs in tree_store._training_history.items()
    }

    metadata = {
        "next_seq_id": tree_store._next_seq_id,
        "trained": trained_data,
        "rewards": rewards_data,
        "training_history": training_history_data,
    }
    with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
```

Update `load` to restore `_training_history`:

```python
def load(self, turn_splitter: Callable[[list[int]], list[Turn]]) -> MCTSTreeStore:
    store = MCTSTreeStore(turn_splitter)
    with open(os.path.join(self.save_dir, "metadata.json")) as f:
        metadata = json.load(f)
    store._next_seq_id = metadata["next_seq_id"]

    trained_data = metadata.get("trained", {})
    rewards_data = metadata.get("rewards", {})
    for key_str, trained in trained_data.items():
        qid, sid = key_str.rsplit(":", 1)
        store._trained[(qid, int(sid))] = trained
    for key_str, reward in rewards_data.items():
        qid, sid = key_str.rsplit(":", 1)
        store._rewards[(qid, int(sid))] = reward

    # Restore training_history (absent in old checkpoints)
    training_history_data = metadata.get("training_history", {})
    for step_str, pairs in training_history_data.items():
        store._training_history[int(step_str)] = [
            (qid, sid) for qid, sid in pairs
        ]

    for filename in os.listdir(self.save_dir):
        if not filename.startswith("query_") or not filename.endswith(".json"):
            continue
        query_id = filename[len("query_") : -len(".json")]
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath) as f:
            tree_data = json.load(f)
        root = self._deserialize_node(
            tree_data["root"], parent=None, tree_id=len(store.trees)
        )
        root.sequence_ids = list(root.sequence_ids)
        store.trees[query_id] = root

    store.rebuild_mcts_stats()

    return store
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTreeCheckpointTrainingHistory -v`
Expected: PASS

- [ ] **Step 5: Run all tree search tests**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All
tests PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): serialize training_steps and _training_history in checkpoints"
```

______________________________________________________________________

### Task 6: Add `replay` field to RolloutCacheConfig

**Files:**

- Modify: `customized_areal/tree_search/config.py:18-22`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestRolloutCacheConfig:
    def test_default_replay_is_false(self):
        from customized_areal.tree_search.config import RolloutCacheConfig

        config = RolloutCacheConfig()
        assert config.replay is False

    def test_replay_can_be_set(self):
        from customized_areal.tree_search.config import RolloutCacheConfig

        config = RolloutCacheConfig(replay=True)
        assert config.replay is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestRolloutCacheConfig -v`
Expected: FAIL — `__init__()` got an unexpected keyword argument `replay`

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/config.py`, add `replay` field:

```python
@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1
    replay: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestRolloutCacheConfig -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/config.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add replay field to RolloutCacheConfig"
```

______________________________________________________________________

### Task 7: Inject `_global_step` in RLTrainer and call `record_training_step` in patched method

**Files:**

- Modify: `areal/trainer/rl_trainer.py:650`

- Modify: `customized_areal/tree_search/trainer.py:75-91`

- [ ] **Step 1: Inject `_global_step` in RLTrainer.train() loop**

In `areal/trainer/rl_trainer.py`, right before the `compute_advantages` call at line
650, add the injection:

```python
            # Inject global_step into trajectories for tree backup recording
            for traj in rollout_batch:
                traj["_global_step"] = global_step

            with (
                stats_tracker.record_timing("compute_advantage"),
                perf_tracer.trace_scope(
                    "train.compute_advantage",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                adv_batch = self.actor.compute_advantages(rollout_batch)
```

- [ ] **Step 2: Call `record_training_step` in the patched
  `_tree_backup_compute_advantages`**

In `customized_areal/tree_search/trainer.py`, update the patched method to call
`record_training_step` after marking trained:

```python
    def _tree_backup_compute_advantages(
        self, data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # 1. Run original GAE pipeline (KL rewards, scaling, normalization, etc.)
        result = original_compute_advantages(self, data)

        # 2. Insert trajectories into tree with raw rewards, compute tree Q-values
        tree_store.insert_batch(result)
        tree_advantage_computer.compute(result)

        # 3. Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(tree_store, result)

        # 4. Record training step order for replay
        global_step = result[0].get("_global_step") if result else None
        tree_store.record_training_step(global_step, result)

        # advantages/returns already overwritten by compute()
        # kl_rewards, tot_rewards, loss_mask, logprobs preserved from GAE
        return result
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All
tests PASS

- [ ] **Step 4: Commit**

```bash
git add areal/trainer/rl_trainer.py customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): inject _global_step and call record_training_step in patched method"
```

______________________________________________________________________

### Task 8: Add replay mode to CacheAwarePPOTrainer

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:219-433`

- [ ] **Step 1: Write the replay `_cache_aware_prepare_batch` override**

In `customized_areal/tree_search/trainer.py`, add a `_replay_prepare_batch` method to
`CacheAwarePPOTrainer`:

```python
def _replay_prepare_batch(
    self,
    dataloader,
    workflow,
    workflow_kwargs=None,
    should_accept_fn=None,
    group_size=1,
    dynamic_bs=False,
):
    """Replay mode: load trajectories from recorded training history.

    Instead of pulling from the dataloader, loads the exact trajectories
    recorded at the current global_step from _training_history.
    """
    global_step = self._replay_global_step

    if global_step not in self.tree_store._training_history:
        if not self._training_history:
            raise ValueError(
                "Cannot replay: no training history found in tree checkpoint. "
                "Run a training session first."
            )
        logger.warning(
            f"Replay: no recorded trajectories for global_step {global_step}, skipping"
        )
        # Return empty to signal end of replay data
        return []

    pairs = self.tree_store._training_history[global_step]
    trajs = []
    for query_id, seq_id in pairs:
        traj = self.tree_store.load_trajectory_by_seq_id(query_id, seq_id)
        if traj is not None:
            trajs.append(traj)
        else:
            logger.warning(
                f"Replay: trajectory (query_id={query_id}, seq_id={seq_id}) "
                f"not found, skipping"
            )

    self._replay_global_step += 1
    logger.info(
        f"Replay step {global_step}: loaded {len(trajs)} trajectories from history"
    )
    return trajs
```

- [ ] **Step 2: Update `__init__` to handle replay mode**

In `CacheAwarePPOTrainer.__init__`, add replay initialization after the existing cache
setup block (after the `logger.info` call around line 280):

```python
            if self.cache_config.replay:
                if not self._training_history:
                    self.tree_store.build_training_history()
                if not self.tree_store._training_history:
                    raise ValueError(
                        "Cannot replay: no training history found in tree "
                        "checkpoint. Run a training session first."
                    )
                self._replay_global_step = 0
                logger.info(
                    f"Replay mode enabled: {len(self.tree_store._training_history)} "
                    f"training steps available"
                )
```

- [ ] **Step 3: Update `train()` to use replay prepare_batch**

In `CacheAwarePPOTrainer.train()`, add replay branch. The existing monkey-patch logic
(lines 387-433) currently always uses `_cache_aware_prepare_batch`. For replay, use
`_replay_prepare_batch` instead:

Replace the monkey-patch section:

```python
        # Monkey-patch prepare_batch with cache-aware or replay version
        original_prepare_batch = self.actor.prepare_batch

        if self.cache_config.replay:
            _prepare_batch_fn = self._replay_prepare_batch
        else:
            def _prepare_batch_fn(
                dataloader,
                workflow,
                workflow_kwargs=None,
                should_accept_fn=None,
                group_size=1,
                dynamic_bs=False,
            ):
                return self._cache_aware_prepare_batch(
                    dataloader=dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=should_accept_fn,
                    group_size=group_size,
                    dynamic_bs=dynamic_bs,
                )

        self.actor.prepare_batch = _prepare_batch_fn
```

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All
tests PASS

- [ ] **Step 5: Run pre-commit**

Run: `pre-commit run --all-files` Expected: All checks PASS (fix any issues)

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): add replay mode to CacheAwarePPOTrainer"
```

______________________________________________________________________

### Task 9: Final integration test

**Files:**

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write integration test for full record-replay cycle**

Add to `tests/test_tree_search/test_mcts_tree_store.py`:

```python
class TestTrainingOrderReplayIntegration:
    def test_record_and_replay_cycle(self):
        """Simulate recording training steps, saving checkpoint, loading, and replaying."""
        import tempfile

        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # === RECORD PHASE ===
            store = MCTSTreeStore(_two_turn_splitter)
            s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
            s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
            s2 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.3)

            # Simulate training step 0: use q1/s0 and q2/s1
            store.record_training_step(
                0,
                [
                    {"_mcts_query_id": "q1", "_mcts_seq_id": s0},
                    {"_mcts_query_id": "q2", "_mcts_seq_id": s1},
                ],
            )
            # Simulate training step 1: use q1/s2
            store.record_training_step(
                1,
                [{"_mcts_query_id": "q1", "_mcts_seq_id": s2}],
            )

            # Save checkpoint
            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            # === REPLAY PHASE ===
            loaded = mgr.load(_two_turn_splitter)

            # Step 0 replay
            assert 0 in loaded._training_history
            step0_pairs = loaded._training_history[0]
            assert len(step0_pairs) == 2
            assert step0_pairs[0] == ("q1", s0)
            assert step0_pairs[1] == ("q2", s1)

            # Load trajectories in replay order
            replay_trajs = []
            for query_id, seq_id in step0_pairs:
                traj = loaded.load_trajectory_by_seq_id(query_id, seq_id)
                assert traj is not None
                replay_trajs.append(traj)

            assert len(replay_trajs) == 2
            assert replay_trajs[0]["_mcts_query_id"] == "q1"
            assert replay_trajs[1]["_mcts_query_id"] == "q2"

            # Step 1 replay
            step1_pairs = loaded._training_history[1]
            assert len(step1_pairs) == 1
            assert step1_pairs[0] == ("q1", s2)

    def test_build_history_fallback_for_old_checkpoint(self):
        """Test that build_training_history can reconstruct from leaves alone."""
        import tempfile

        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MCTSTreeStore(_two_turn_splitter)
            s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
            store.record_training_step(
                0, [{"_mcts_query_id": "q1", "_mcts_seq_id": s0}]
            )

            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            # Simulate old checkpoint: remove training_history from metadata
            import json
            import os

            metadata_path = os.path.join(tmpdir, "mcts_trees", "metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)
            del metadata["training_history"]
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Load — _training_history should be empty
            loaded = mgr.load(_two_turn_splitter)
            assert loaded._training_history == {}

            # Fallback: build from leaves
            loaded.build_training_history()
            assert 0 in loaded._training_history
            assert len(loaded._training_history[0]) == 1
            assert loaded._training_history[0][0][0] == "q1"
```

- [ ] **Step 2: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestTrainingOrderReplayIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All
tests PASS

- [ ] **Step 4: Run pre-commit**

Run: `pre-commit run --all-files` Expected: All checks PASS (fix any issues)

- [ ] **Step 5: Commit**

```bash
git add tests/test_tree_search/test_mcts_tree_store.py
git commit -m "test(tree-search): add integration tests for training order record-replay"
```

______________________________________________________________________

## Self-Review Checklist

**1. Spec coverage:**

- TrieNode `training_steps` field → Task 1
- `record_training_step()` with leaf annotation + `_training_history` → Task 2
- `load_trajectory_by_seq_id()` → Task 3
- `build_training_history()` fallback → Task 4
- Checkpoint serialization (training_steps + \_training_history) → Task 5
- `RolloutCacheConfig.replay` field → Task 6
- `_global_step` injection + call in patched method → Task 7
- Replay mode in CacheAwarePPOTrainer → Task 8
- Integration test → Task 9
- Error handling (empty history, missing trajectories, missing `_global_step`) → covered
  in Task 2 (skip on None global_step), Task 8 (ValueError on empty, warnings on
  missing)
- `clear()` update → Task 2

**2. Placeholder scan:** No TBD/TODO/fill-in-later found. All steps contain complete
code.

**3. Type consistency:**

- `training_steps: list[int]` — consistent across TrieNode, serialization, tests
- `_training_history: dict[int, list[tuple[str, int]]]` — consistent across
  MCTSTreeStore, serialization (converted to `list[list]` for JSON), tests
- `record_training_step(global_step: int | None, trajectories: list[dict])` — consistent
  between MCTSTreeStore method and trainer call
- `load_trajectory_by_seq_id(query_id: str, seq_id: int) -> dict | None` — consistent
  usage in replay
- `RolloutCacheConfig.replay: bool = False` — consistent usage in trainer

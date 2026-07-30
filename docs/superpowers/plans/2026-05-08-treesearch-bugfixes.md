# Tree Search Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 13 remaining bugs from the code review across `mcts_tree_store.py`,
`advantage.py`, `checkpoint.py`, `trainer.py`, `grouped_workflow.py`, config YAMLs, and
`_node_to_tensor_dict`.

**Architecture:** Surgical per-file fixes in dependency order. Bug #16
(`seq_id → node_id` rename) and Bug #13 (`set_trained` signature) are done first since
they change API surfaces that other tasks depend on. Then correctness bugs (#1, #2, #3,
#8), then trainer.py fixes (#4, #11, #15), then config (#12), then helper extraction
(#17).

**Tech Stack:** Python 3.12+, PyTorch, pytest

______________________________________________________________________

## File Structure

| File                                                                              | Action | Responsibility                                                                                                   |
| --------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/mcts_tree_store.py`                                 | Modify | Bug #1 (insert_batch skip), #13 (remove unused param), #16 (seq_id→node_id rename), #17 (optional tensor helper) |
| `customized_areal/tree_search/advantage.py`                                       | Modify | Bug #2 (Bessel variance), #14 (dict→set), #16 (seq_id→node_id rename)                                            |
| `customized_areal/tree_search/checkpoint.py`                                      | Modify | Bug #3 (query_id serialization), #16 (metadata key rename)                                                       |
| `customized_areal/tree_search/grouped_workflow.py`                                | Modify | Bug #8 (episode_id UUID)                                                                                         |
| `customized_areal/tree_search/trainer.py`                                         | Modify | Bug #4 (duplicate key), #11 (stale iterator), #13 (caller update), #15 (comment), #16 (local var rename)         |
| `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml` | Modify | Bug #12 (API key)                                                                                                |
| `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct.yaml`             | Modify | Bug #12 (API key)                                                                                                |
| `customized_areal/tpfc/configs/config_tpfc.yaml`                                  | Modify | Bug #12 (API key)                                                                                                |
| `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-4B-Instruct.yaml`             | Modify | Bug #12 (API key)                                                                                                |
| `tests/test_treesearch_bugfixes.py`                                               | Create | Unit tests for all bug fixes                                                                                     |

______________________________________________________________________

### Task 1: Bug #16, #13, #14 — `seq_id → node_id` rename, `set_trained` signature, `dict→set`

This is the largest task and must come first since it changes API surfaces that later
tasks depend on.

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py`

- Modify: `customized_areal/tree_search/advantage.py`

- Modify: `customized_areal/tree_search/checkpoint.py`

- Modify: `customized_areal/tree_search/trainer.py`

- Test: `tests/test_treesearch_bugfixes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_treesearch_bugfixes.py`:

```python
"""Tests for tree search bug fixes."""

import pytest
import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


def _make_node(reward: float = 1.0) -> Node:
    """Create a minimal Node for testing."""
    return Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 1, 1],
        logprobs=[0.0, -0.5, -0.3],
        versions=[-1, 0, 0],
        outcome_reward=reward,
    )


class TestSetTrainedSignature:
    """Bug #13: set_trained should not take unused query_id parameter."""

    def test_set_trained_accepts_node_id_only(self):
        store = MCTSTreeStore()
        node = _make_node()
        store.insert_batch([node])
        node_id = node.node_id

        # Should work with just node_id (no query_id param)
        store.set_trained(node_id, True)
        assert store.is_trained(node_id) is True
        assert store.get_reward(node_id) == 1.0


class TestDictSetInsteadOfDictNone:
    """Bug #14: advantage computer should use set[int] not dict[int, None]."""

    def test_compute_uses_set_for_node_ids(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        for r in [1.0, 2.0, 3.0]:
            node = _make_node(reward=r)
            object.__setattr__(node, "query_id", "q1")
            store.insert_batch([node])

        computer = TreeAdvantageComputer(store)
        nodes = store.load_trajectories("q1", 3)
        computer.compute(nodes)

        for node in nodes:
            assert hasattr(node, "advantages")
            assert node.advantages is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py -v --no-header -x 2>&1 | tail -20`
Expected: FAIL — `set_trained` still expects `query_id` as first argument, `is_trained`
and `get_reward` too.

- [ ] **Step 3: Rename in `mcts_tree_store.py`**

Apply these renames throughout the file using Edit tool with `replace_all`:

| Old                     | New                      |
| ----------------------- | ------------------------ |
| `_seq_id_to_key`        | `_node_id_to_key`        |
| `_query_seq_ids`        | `_query_node_ids`        |
| `_next_seq_id`          | `_next_node_id`          |
| `get_untrained_seq_ids` | `get_untrained_node_ids` |

Then update method signatures to remove `query_id` from `set_trained`, `is_trained`,
`get_reward`, and rename `seq_id` parameter to `node_id` in all methods.

**`__init__`**:

```python
    def __init__(self) -> None:
        self.trajectories: dict[str, list[Node]] = {}
        self._node_id_to_key: dict[int, tuple[str, int]] = {}
        self._query_node_ids: dict[str, list[int]] = {}
        self._next_node_id: int = 0

        self._visit_counts: dict[int, int] = {}
        self._total_values: dict[int, float] = {}
        self._q_values: dict[int, float] = {}

        self._trained: dict[int, bool] = {}
        self._rewards: dict[int, float] = {}

        # Tree-search episode metadata
        self._turn_nodes: dict[str, int] = {}  # turn_id → node_id
        self._normalized_advantages: dict[int, float] = {}
```

**`_backup`**:

```python
    def _backup(self, node_id: int, reward: float) -> None:
        """Update MCTS stats for a single trajectory."""
        self._visit_counts[node_id] = self._visit_counts.get(node_id, 0) + 1
        self._total_values[node_id] = self._total_values.get(node_id, 0.0) + reward
        self._q_values[node_id] = self._total_values[node_id] / self._visit_counts[node_id]
```

**`_insert_single`**:

```python
    def _insert_single(self, query_id: str, node: Node) -> int:
        """Insert a single Node and assign a node_id."""
        node_id = self._next_node_id
        self._next_node_id += 1

        idx = len(self.trajectories.setdefault(query_id, []))
        self.trajectories[query_id].append(node)
        self._node_id_to_key[node_id] = (query_id, idx)
        self._query_node_ids.setdefault(query_id, []).append(node_id)

        # Assign node_id as the node's identifier
        node.node_id = node_id
        object.__setattr__(node, "query_id", query_id)

        self._backup(node_id, node.outcome_reward)
        self._trained[node_id] = False
        self._rewards[node_id] = node.outcome_reward

        return node_id
```

**`get_advantages`**:

```python
    def get_advantages(self, query_id: str, node_id: int) -> torch.Tensor:
        """Return per-token advantages: Q-value on response tokens, 0 on prompt."""
        qid, idx = self._node_id_to_key[node_id]
        node = self.trajectories[qid][idx]
        q_val = self._q_values.get(node_id, 0.0)
        seq_len = len(node.input_ids)
        advantages = torch.zeros(seq_len, dtype=torch.float32)
        starts, ends = _find_turn_boundaries(node.loss_mask)
        for start, end in zip(starts, ends):
            advantages[start:end] = q_val
        return advantages
```

**`get_prompt_mask`**:

```python
    def get_prompt_mask(self, query_id: str, node_id: int) -> torch.Tensor:
        """Return boolean mask: True for response tokens, False for prompt."""
        qid, idx = self._node_id_to_key[node_id]
        node = self.trajectories[qid][idx]
        return torch.tensor(node.loss_mask, dtype=torch.bool)
```

**`set_trained`** (Bug #13 — remove `query_id`):

```python
    def set_trained(self, node_id: int, trained: bool = True) -> None:
        self._trained[node_id] = trained
```

**`is_trained`** (remove `query_id`):

```python
    def is_trained(self, node_id: int) -> bool:
        return self._trained.get(node_id, False)
```

**`get_reward`** (remove `query_id`):

```python
    def get_reward(self, node_id: int) -> float:
        return self._rewards.get(node_id, 0.0)
```

**`get_untrained_count`**:

```python
    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self._query_node_ids:
            return 0
        return sum(
            1
            for node_id in self._query_node_ids[query_id]
            if not self._trained.get(node_id, False)
        )
```

**`get_untrained_node_ids`** (renamed from `get_untrained_seq_ids`):

```python
    def get_untrained_node_ids(self, query_id: str, n_samples: int) -> list[int]:
        if query_id not in self._query_node_ids:
            return []
        result: list[int] = []
        for node_id in self._query_node_ids[query_id]:
            if not self._trained.get(node_id, False):
                result.append(node_id)
                if len(result) >= n_samples:
                    break
        return result
```

**`load_trajectories`**:

```python
    def load_trajectories(self, query_id: str, n_samples: int) -> list[Node]:
        """Load untrained trajectories as Node objects."""
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_node_ids(query_id, n_samples)
        result: list[Node] = []
        for node_id in untrained_ids:
            qid, idx = self._node_id_to_key[node_id]
            node = self.trajectories[qid][idx]
            result.append(node)
        return result
```

**`clear`**:

```python
    def clear(self) -> None:
        """Reset all trajectories, stats, and indices."""
        self.trajectories.clear()
        self._node_id_to_key.clear()
        self._query_node_ids.clear()
        self._next_node_id = 0
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._trained.clear()
        self._rewards.clear()
        self._turn_nodes.clear()
        self._normalized_advantages.clear()
```

**`_node_to_tensor_dict`** — rename `seq_id` → `node_id` in signature only:

```python
def _node_to_tensor_dict(node: Node, query_id: str, node_id: int) -> dict[str, Any]:
```

(Inside the function body, `seq_id` does not appear; `node_id` was already used as the
`"node_id"` key.)

- [ ] **Step 4: Rename in `advantage.py` (Bug #14 fix + rename)**

Rewrite the `compute` method body. Replace `query_seq_sets: dict[str, dict[int, None]]`
with `query_node_sets: dict[str, set[int]]`, and rename all `seq_id` locals to
`node_id`.

Full rewritten file content for `advantage.py`:

```python
# customized_areal/tree_search/advantage.py
from __future__ import annotations

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

GRPO_NORM_EPS = 1e-8


class TreeAdvantageComputer:
    """Replace GAE advantages with tree Q-values from MCTS backup.

    Reads query_id and node_id from Node objects. Sets advantages
    and returns on the Node in-place via object.__setattr__.

    Supports per-query GRPO normalization: Q-values are normalized within
    each query group (all episodes for the same query), producing
    zero-mean unit-variance advantages.
    """

    def __init__(self, tree_store: MCTSTreeStore, grpo_eps: float = GRPO_NORM_EPS):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps

    def _compute_single(self, node_id: int, seq_len: int, query_id: str) -> torch.Tensor:
        """Compute tree Q-value advantages for a single sample."""
        normalized_q = self.tree_store._normalized_advantages.get(node_id)
        if normalized_q is None:
            normalized_q = self.tree_store._q_values.get(node_id, 0.0)

        prompt_mask = self.tree_store.get_prompt_mask(query_id, node_id)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        common_len = min(seq_len, prompt_mask.shape[0])
        mask[:common_len] = prompt_mask[:common_len]

        return mask.float() * normalized_q

    @staticmethod
    def _get_query_id(traj: Node) -> str | None:
        """Extract query_id from Node."""
        return getattr(traj, "query_id", None)

    @staticmethod
    def _get_seq_len(traj: Node) -> int:
        """Get sequence length from Node."""
        return len(traj.input_ids)

    def compute(self, trajectories: list[Node]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates Nodes in-place.

        Sets node.advantages/node.returns via object.__setattr__.

        After inserting all trajectories, performs per-query GRPO normalization
        of Q-values so that episodes within the same query group have
        zero-mean unit-variance advantages.
        """
        # Collect unique (query_id, node_id) pairs for GRPO normalization.
        query_node_sets: dict[str, set[int]] = {}

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            nset = query_node_sets.setdefault(query_id, set())

            node_id = getattr(traj, "node_id", None)
            if node_id is not None:
                nset.add(node_id)

        # Per-query GRPO normalization (deduplicated node_ids)
        for query_id, node_id_set in query_node_sets.items():
            node_ids = list(node_id_set)
            q_values = [self.tree_store._rewards.get(nid, 0.0) for nid in node_ids]
            if len(q_values) < 2:
                if node_ids:
                    self.tree_store._normalized_advantages[node_ids[0]] = q_values[0]
                continue
            mean_q = sum(q_values) / len(q_values)
            var_q = sum((q - mean_q) ** 2 for q in q_values) / max(len(q_values) - 1, 1)
            std_q = var_q**0.5
            for nid, q in zip(node_ids, q_values):
                self.tree_store._normalized_advantages[nid] = (q - mean_q) / (
                    std_q + self.grpo_eps
                )

        # Compute per-trajectory advantages using normalized Q-values
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue

            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            seq_len = len(traj.input_ids)
            advantages = self._compute_single(node_id, seq_len, query_id)
            object.__setattr__(traj, "advantages", advantages)
            object.__setattr__(traj, "returns", advantages.clone())
```

- [ ] **Step 5: Rename in `checkpoint.py` (metadata key rename + backward-compat)**

Update `save` method to use new key names:

```python
    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save per-query trajectory records
        for query_id, records in tree_store.trajectories.items():
            data = {"records": [self._serialize_record(r) for r in records]}
            filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
            with open(filepath, "w") as f:
                json.dump(data, f)

        # Save metadata (indices, stats, tracking)
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
        }
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
```

Update `load` method with backward-compat fallback for old key names:

```python
    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # Backward-compat: try new key names first, fall back to old names
        store._next_node_id = metadata.get("next_node_id", metadata.get("next_seq_id", 0))
        node_id_to_key_raw = metadata.get(
            "node_id_to_key", metadata.get("seq_id_to_key", {})
        )
        store._node_id_to_key = {
            int(k): (v[0], v[1]) for k, v in node_id_to_key_raw.items()
        }
        store._query_node_ids = metadata.get(
            "query_node_ids", metadata.get("query_seq_ids", {})
        )
        store._visit_counts = {
            int(k): v for k, v in metadata.get("visit_counts", {}).items()
        }
        store._total_values = {
            int(k): v for k, v in metadata.get("total_values", {}).items()
        }
        store._q_values = {int(k): v for k, v in metadata.get("q_values", {}).items()}
        store._trained = {int(k): v for k, v in metadata.get("trained", {}).items()}
        store._rewards = {int(k): v for k, v in metadata.get("rewards", {}).items()}
        store._normalized_advantages = {
            int(k): v for k, v in metadata.get("normalized_advantages", {}).items()
        }
        store._turn_nodes = metadata.get("turn_nodes", {})

        # Load per-query trajectory records
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            query_id = filename[len("query_") : -len(".json")]
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            store.trajectories[query_id] = [
                self._deserialize_record(r) for r in data["records"]
            ]

        return store
```

- [ ] **Step 6: Rename in `trainer.py`**

Update `_mark_batch_trained` to use new `set_trained(node_id, True)` signature and
rename local `seq_id` → `node_id`:

```python
def _mark_batch_trained(tree_store: MCTSTreeStore, trajectories: list[Node]) -> None:
    """Mark all trajectories in a batch as trained after tree backup."""
    count = 0
    for traj in trajectories:
        node_id = getattr(traj, "node_id", None)
        if node_id is not None:
            tree_store.set_trained(node_id, True)
            count += 1
    if count:
        logger.debug(f"Marked {count} trajectories as trained")
```

Update `_cache_aware_prepare_batch` local `seq_id` → `node_id` (line 622 area):

```python
        converted: list[dict[str, Any]] = []
        for t in trajs:
            query_id = getattr(t, "query_id", "")
            node_id = getattr(t, "node_id", 0)
            converted.append(_node_to_tensor_dict(t, query_id, node_id))
```

- [ ] **Step 7: Run tests to verify they pass**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py -v --no-header -x 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 8: Verify syntax of all modified files**

Run:
`python -c "import ast; [ast.parse(open(f).read()) for f in ['customized_areal/tree_search/mcts_tree_store.py', 'customized_areal/tree_search/advantage.py', 'customized_areal/tree_search/checkpoint.py', 'customized_areal/tree_search/trainer.py']]; print('OK')"`
Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py customized_areal/tree_search/advantage.py customized_areal/tree_search/checkpoint.py customized_areal/tree_search/trainer.py tests/test_treesearch_bugfixes.py
git commit -m "refactor: rename seq_id→node_id, fix set_trained signature, use set instead of dict (Bugs #13, #14, #16)"
```

______________________________________________________________________

### Task 2: Bug #1 — `insert_batch` skip already-inserted nodes

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:204-212`

- Test: `tests/test_treesearch_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestInsertBatchSkipDuplicates:
    """Bug #1: insert_batch should skip already-inserted nodes."""

    def test_insert_batch_skips_nodes_with_existing_node_id(self):
        store = MCTSTreeStore()
        node = _make_node()
        store.insert_batch([node])
        assert node.node_id != 0  # assigned by first insert

        first_id = node.node_id
        first_count = len(store.trajectories.get("", []))

        # Re-insert the same node (simulating cache reuse)
        store.insert_batch([node])

        # Should NOT create a duplicate
        assert node.node_id == first_id
        assert len(store.trajectories.get("", [])) == first_count

    def test_insert_batch_allows_new_nodes(self):
        store = MCTSTreeStore()
        node_a = _make_node(reward=1.0)
        node_b = _make_node(reward=2.0)
        store.insert_batch([node_a])
        store.insert_batch([node_b])
        assert node_a.node_id != node_b.node_id
        assert len(store.trajectories.get("", [])) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestInsertBatchSkipDuplicates -v --no-header -x 2>&1 | tail -20`
Expected: FAIL — `insert_batch` does not skip duplicates, so the node count will be 2
instead of 1.

- [ ] **Step 3: Implement the fix**

In `customized_areal/tree_search/mcts_tree_store.py`, replace the `insert_batch` method:

```python
    def insert_batch(self, trajectories: list[Node]) -> None:
        """Insert Node trajectories into the store.

        Each Node is inserted directly. Nodes that already have a
        node_id assigned (loaded from cache) are skipped.
        """
        for node in trajectories:
            existing_id = getattr(node, "node_id", 0)
            if existing_id != 0 and existing_id in self._node_id_to_key:
                continue  # already inserted (loaded from cache)
            query_id = getattr(node, "query_id", None) or ""
            self._insert_single(query_id, node)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestInsertBatchSkipDuplicates -v --no-header -x 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_treesearch_bugfixes.py
git commit -m "fix: insert_batch skips already-inserted nodes (Bug #1)"
```

______________________________________________________________________

### Task 3: Bug #2 — Bessel-corrected variance in GRPO normalization

**Files:**

- Modify: `customized_areal/tree_search/advantage.py`
- Test: `tests/test_treesearch_bugfixes.py`

Note: The variance fix was already applied in Task 1 Step 4 (the rewritten
`advantage.py` uses `max(len(q_values) - 1, 1)`). This task adds the test to verify it.

- [ ] **Step 1: Write the test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestBesselVariance:
    """Bug #2: GRPO normalization should use Bessel-corrected variance."""

    def test_uses_sample_variance_not_population(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        # Insert 4 nodes with known rewards
        for r in [1.0, 2.0, 3.0, 4.0]:
            node = _make_node(reward=r)
            object.__setattr__(node, "query_id", "q1")
            store.insert_batch([node])

        computer = TreeAdvantageComputer(store)
        nodes = store.load_trajectories("q1", 4)
        computer.compute(nodes)

        # Population variance of [1,2,3,4] = 1.25, std = 1.118
        # Sample variance of [1,2,3,4] = 5/3 = 1.667, std = 1.291
        # With sample variance: (1 - 2.5) / (1.291 + 1e-8) ≈ -1.161
        # With population variance: (1 - 2.5) / (1.118 + 1e-8) ≈ -2.236
        first_adv = nodes[0].advantages
        response_adv = first_adv[first_adv != 0]
        assert response_adv.numel() > 0
        # Should be close to -1.161 (sample), not -2.236 (population)
        assert abs(response_adv[0].item() - (-1.161)) < 0.05, (
            f"Expected sample variance normalization (~-1.161), got {response_adv[0].item()}"
        )
```

- [ ] **Step 2: Run test to verify it passes (fix already applied in Task 1)**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestBesselVariance -v --no-header -x 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_treesearch_bugfixes.py
git commit -m "test: add Bessel-corrected variance test (Bug #2)"
```

______________________________________________________________________

### Task 4: Bug #3 — `query_id` lost on checkpoint deserialization

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:94-131`

- Test: `tests/test_treesearch_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestQueryIdCheckpoint:
    """Bug #3: query_id lost on checkpoint deserialization."""

    def test_query_id_survives_save_load(self, tmp_path):
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore()
        node = _make_node()
        object.__setattr__(node, "query_id", "test_query_123")
        store.insert_batch([node])
        assert getattr(node, "query_id", None) == "test_query_123"

        manager = TreeCheckpointManager(str(tmp_path))
        manager.save(store)

        loaded = manager.load()
        loaded_nodes = loaded.trajectories.get("test_query_123", [])
        assert len(loaded_nodes) == 1
        assert getattr(loaded_nodes[0], "query_id", None) == "test_query_123"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestQueryIdCheckpoint -v --no-header -x 2>&1 | tail -20`
Expected: FAIL — `query_id` returns `None` on deserialized nodes.

- [ ] **Step 3: Implement the fix**

In `customized_areal/tree_search/checkpoint.py`, update `_serialize_record` to include
`query_id`:

```python
    @staticmethod
    def _serialize_record(node: Node) -> dict:
        data = {
            "input_ids": node.input_ids,
            "loss_mask": node.loss_mask,
            "logprobs": node.logprobs,
            "versions": node.versions,
            "outcome_reward": node.outcome_reward,
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "episode_id": node.episode_id,
            "query_id": getattr(node, "query_id", ""),
        }
        if node.topk_ids is not None:
            data["topk_ids"] = node.topk_ids
        if node.topk_logp is not None:
            data["topk_logp"] = node.topk_logp
        if node.distill_reward is not None:
            data["distill_reward"] = node.distill_reward
        if node.teacher_logp is not None:
            data["teacher_logp"] = node.teacher_logp
        return data
```

Update `_deserialize_record` to restore `query_id`:

```python
    @staticmethod
    def _deserialize_record(data: dict) -> Node:
        node = Node(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            logprobs=data["logprobs"],
            versions=data["versions"],
            outcome_reward=data.get("outcome_reward", data.get("reward", 0.0)),
            node_id=data.get("node_id", 0),
            parent_node_id=data.get("parent_node_id"),
            episode_id=data.get("episode_id", ""),
            topk_ids=data.get("topk_ids"),
            topk_logp=data.get("topk_logp"),
            distill_reward=data.get("distill_reward"),
            teacher_logp=data.get("teacher_logp"),
        )
        query_id = data.get("query_id", "")
        if query_id:
            object.__setattr__(node, "query_id", query_id)
        return node
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestQueryIdCheckpoint -v --no-header -x 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_treesearch_bugfixes.py
git commit -m "fix: serialize/deserialize query_id in checkpoint (Bug #3)"
```

______________________________________________________________________

### Task 5: Bug #8 — `episode_id` duplicates across empty query_id and epochs

**Files:**

- Modify: `customized_areal/tree_search/grouped_workflow.py:44-53`

- Test: `tests/test_treesearch_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestEpisodeIdUniqueness:
    """Bug #8: episode_id should be unique across queries and epochs."""

    def test_episode_ids_are_unique_even_with_empty_query_id(self):
        from customized_areal.tree_search.grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from unittest.mock import MagicMock, AsyncMock
        import asyncio

        # Create two groups of nodes with empty query_id
        node_a = _make_node()
        node_b = _make_node()
        node_c = _make_node()
        node_d = _make_node()

        inner = MagicMock()
        inner.arun_episode = AsyncMock(side_effect=[
            [node_a, node_b],
            [node_c, node_d],
        ])

        wf = TreeSearchGroupedRolloutWorkflow(
            workflow=inner, group_size=2, logger=MagicMock()
        )
        result = asyncio.get_event_loop().run_until_complete(
            wf.arun_episode(MagicMock(), {"query_id": ""})
        )

        episode_ids = [n.episode_id for n in result]
        assert len(episode_ids) == len(set(episode_ids)), (
            f"episode_ids not unique: {episode_ids}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestEpisodeIdUniqueness -v --no-header -x 2>&1 | tail -20`
Expected: FAIL — with empty `query_id`, both groups get duplicate episode_ids like `"0"`
and `"1"`.

- [ ] **Step 3: Implement the fix**

In `customized_areal/tree_search/grouped_workflow.py`, add `import uuid` at the top and
replace lines 44-53:

```python
        first = valid_results[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], Node):
            query_id = data.get("query_id") or ""
            if not query_id:
                logger.warning(
                    "query_id is empty; episode_id will not be unique across queries"
                )
            all_nodes: list[Node] = []
            for group_idx, result in enumerate(valid_results):
                episode_id = f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
                for node in result:
                    node.episode_id = episode_id
                    object.__setattr__(node, "query_id", query_id)
                all_nodes.extend(result)
            return all_nodes if all_nodes else None
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py::TestEpisodeIdUniqueness -v --no-header -x 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/grouped_workflow.py tests/test_treesearch_bugfixes.py
git commit -m "fix: add UUID suffix to episode_id for uniqueness (Bug #8)"
```

______________________________________________________________________

### Task 6: Bug #4, #11, #15 — `trainer.py` fixes

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Fix Bug #4 — duplicate key check in `split_prompts`**

In `trainer.py`, line 317, replace:

```python
            query_id = prompt.get("query_id") or prompt.get("query_id") or ""
```

with:

```python
            query_id = prompt.get("query_id") or ""
```

Also fix the docstring (lines 303-306) — replace:

```python
        """Split prompts into cached and needs-generation groups.

        Query ID derivation fallback chain:
        1. ``prompt["query_id"]`` — dataset-provided string (preferred)
        2. ``prompt["query_id"]`` — from prior injection
        3. Empty string (no tree lookup possible)
```

with:

```python
        """Split prompts into cached and needs-generation groups.

        Query ID derivation fallback chain:
        1. ``prompt["query_id"]`` — dataset-provided string (preferred)
        2. Empty string (no tree lookup possible)
```

- [ ] **Step 2: Fix Bug #11 — stale dataloader iterator**

In `trainer.py`, in the `train()` method, add a safety delete before the `try` block.
After the line `self.actor.prepare_batch = _prepare_batch_fn`, add:

```python
        # Safety: reset stale iterator from a previous crashed train() call
        if hasattr(self, "_cache_dataloader_iter"):
            del self._cache_dataloader_iter
```

- [ ] **Step 3: Fix Bug #15 — misleading comment**

In `trainer.py`, line 569, replace:

```python
            # TreeSearchWorkflowExecutor already returns flat list of per-episode dicts
```

with:

```python
            # TreeSearchWorkflowExecutor already returns flat list of Node objects
```

- [ ] **Step 4: Verify no syntax errors**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/trainer.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "fix: trainer.py bugs #4 (duplicate key), #11 (stale iterator), #15 (comment)"
```

______________________________________________________________________

### Task 7: Bug #12 — Replace plaintext API keys in config YAMLs

**Files:**

- Modify:
  `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml`

- Modify: `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct.yaml`

- Modify: `customized_areal/tpfc/configs/config_tpfc.yaml`

- Modify: `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-4B-Instruct.yaml`

- [ ] **Step 1: Replace the API key in each YAML file**

In each of the 4 config files, find the `swanlab:` section and replace:

```yaml
    api_key: WVB9KhdtDjWVAozYQgHu5
```

with:

```yaml
    api_key: ${oc.env:SWANLAB_API_KEY,}
```

This uses OmegaConf's `oc.env` resolver to read from the `SWANLAB_API_KEY` environment
variable. The trailing comma provides an empty-string default if the env var is not set.

- [ ] **Step 2: Verify YAML syntax**

Run:
`python -c "from omegaconf import OmegaConf; [OmegaConf.load(f) for f in ['customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml', 'customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct.yaml', 'customized_areal/tpfc/configs/config_tpfc.yaml', 'customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-4B-Instruct.yaml']]; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tpfc/configs/
git commit -m "fix: replace hardcoded API key with env var reference (Bug #12)"
```

______________________________________________________________________

### Task 8: Bug #17 — Extract optional tensor field helper in `_node_to_tensor_dict`

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py`

- [ ] **Step 1: Add the helper function**

Add before `_node_to_tensor_dict` in `mcts_tree_store.py`:

```python
def _optional_tensor_field(
    traj: dict[str, Any], key: str, values: list | None, dtype: torch.dtype
) -> None:
    """Add an unsqueezed tensor to traj if values is not None."""
    if values is not None:
        traj[key] = torch.tensor(values, dtype=dtype).unsqueeze(0)
```

- [ ] **Step 2: Replace the 4 repetitive blocks in `_node_to_tensor_dict`**

Replace the 4 `if X is not None` blocks for `topk_ids`, `topk_logp`, `distill_reward`,
`teacher_logp` with:

```python
    # Response-only fields: extract response portion from full sequence
    resp_start, resp_end = _response_span(node.loss_mask)
    _optional_tensor_field(traj, "topk_ids", node.topk_ids, torch.int32)
    _optional_tensor_field(traj, "topk_logp", node.topk_logp, torch.float32)
    _optional_tensor_field(traj, "distill_reward", node.distill_reward, torch.float32)
    _optional_tensor_field(traj, "teacher_logp", node.teacher_logp, torch.float32)
```

- [ ] **Step 3: Verify syntax**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/mcts_tree_store.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor: extract _optional_tensor_field helper (Bug #17)"
```

______________________________________________________________________

### Task 9: Final verification

- [ ] **Step 1: Run all bugfix tests**

Run: `python -m pytest tests/test_treesearch_bugfixes.py -v --no-header 2>&1 | tail -30`
Expected: All PASS

- [ ] **Step 2: Run existing patches tests (regression check)**

Run: `python -m pytest tests/test_treesearch_patches.py -v --no-header 2>&1 | tail -30`
Expected: All PASS

- [ ] **Step 3: Run pre-commit on all changed files**

Run:
`pre-commit run --files customized_areal/tree_search/mcts_tree_store.py customized_areal/tree_search/advantage.py customized_areal/tree_search/checkpoint.py customized_areal/tree_search/trainer.py customized_areal/tree_search/grouped_workflow.py tests/test_treesearch_bugfixes.py customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct.yaml customized_areal/tpfc/configs/config_tpfc.yaml customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-4B-Instruct.yaml 2>&1 | tail -20`
Expected: All checks pass

- [ ] **Step 4: Fix any formatting issues and re-commit if needed**

```bash
git add -u
git commit -m "style: apply pre-commit fixes for tree search bug fixes"
```

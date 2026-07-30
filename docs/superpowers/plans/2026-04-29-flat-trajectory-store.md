# Flat Trajectory Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `TrieNode`-based MCTS tree with a flat `TrajectoryRecord` store
that correctly preserves full multi-turn context.

**Architecture:** `TrajectoryRecord` dataclass stores complete, unpadded trajectories
with turn boundaries derived from `loss_mask`. `MCTSTreeStore` manages records keyed by
`query_id` with `seq_id`-based MCTS stats (no per-node stats). No trie deduplication —
each trajectory is stored independently.

**Tech Stack:** Python 3.12+, PyTorch, JSON checkpoints

______________________________________________________________________

## File Structure

| File                                               | Responsibility                                      |
| -------------------------------------------------- | --------------------------------------------------- |
| `customized_areal/tree_search/mcts_tree_store.py`  | `TrajectoryRecord` dataclass + flat `MCTSTreeStore` |
| `customized_areal/tree_search/checkpoint.py`       | JSON checkpoint save/load for new format            |
| `customized_areal/tree_search/advantage.py`        | Tree advantage computation (minor update)           |
| `customized_areal/tree_search/config.py`           | Remove `assistant_marker` field                     |
| `customized_areal/tree_search/trainer.py`          | Remove `make_turn_splitter` usage                   |
| `customized_areal/tree_search/__init__.py`         | Remove deleted exports                              |
| `customized_areal/tree_search/trie_node.py`        | **DELETE**                                          |
| `customized_areal/tree_search/turn_splitter.py`    | **DELETE**                                          |
| `tests/test_tree_search/test_mcts_tree_store.py`   | Rewrite for new API                                 |
| `tests/test_tree_search/test_checkpoint.py`        | Rewrite for new format                              |
| `tests/test_tree_search/test_trie_node.py`         | **DELETE**                                          |
| `tests/test_tree_search/test_turn_splitter.py`     | **DELETE**                                          |
| `tests/test_tree_search/test_advantage.py`         | Update to new store API                             |
| `tests/test_tree_search/test_trainer.py`           | Update to new patch signature                       |
| `tests/test_tree_search/test_cache_trainer.py`     | Update to new store API                             |
| `tests/test_tree_search/test_batch_consistency.py` | Remove `_split_grouped_trajectories` references     |

______________________________________________________________________

### Task 1: Write `TrajectoryRecord` + `_find_turn_boundaries` + tests

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py`

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test for `_find_turn_boundaries`**

Replace the entire `tests/test_tree_search/test_mcts_tree_store.py` with:

```python
import torch

from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    TrajectoryRecord,
    _find_turn_boundaries,
)


class TestFindTurnBoundaries:
    def test_single_turn(self):
        """Single 0→1 transition: one response region."""
        starts, ends = _find_turn_boundaries([0, 0, 0, 1, 1])
        assert starts == [3]
        assert ends == [5]

    def test_multi_turn(self):
        """Multiple 0→1 and 1→0 transitions: multiple response regions."""
        starts, ends = _find_turn_boundaries([0, 0, 1, 1, 0, 0, 1, 1])
        assert starts == [2, 6]
        assert ends == [4, 8]

    def test_all_zeros(self):
        """All prompt tokens: no response regions."""
        starts, ends = _find_turn_boundaries([0, 0, 0, 0])
        assert starts == []
        assert ends == []

    def test_all_ones(self):
        """All response tokens: single region starting at 0."""
        starts, ends = _find_turn_boundaries([1, 1, 1, 1])
        assert starts == [0]
        assert ends == [4]

    def test_empty(self):
        """Empty loss_mask."""
        starts, ends = _find_turn_boundaries([])
        assert starts == []
        assert ends == []

    def test_response_at_end(self):
        """Response tokens extend to end of sequence."""
        starts, ends = _find_turn_boundaries([0, 0, 1, 1, 1])
        assert starts == [2]
        assert ends == [5]

    def test_three_turns(self):
        """Three response regions with prompt separators."""
        loss_mask = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        starts, ends = _find_turn_boundaries(loss_mask)
        assert starts == [2, 6, 10]
        assert ends == [4, 8, 12]


class TestTrajectoryRecord:
    def test_creation(self):
        record = TrajectoryRecord(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            reward=1.0,
            turn_response_starts=[2],
            turn_response_ends=[5],
        )
        assert len(record.input_ids) == 5
        assert record.reward == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestFindTurnBoundaries -v`
Expected: FAIL with `ImportError: cannot import name '_find_turn_boundaries'`

- [ ] **Step 3: Implement `TrajectoryRecord` and `_find_turn_boundaries`**

Replace the entire `customized_areal/tree_search/mcts_tree_store.py` with:

```python
# customized_areal/tree_search/mcts_tree_store.py
"""Flat trajectory store with MCTS statistics.

Replaces the TrieNode-based trie with a per-query list of TrajectoryRecord
objects. Each record stores the complete, unpadded sequence from the rollout,
with turn boundaries derived from loss_mask transitions.

This correctly preserves full multi-turn context (including system prompts,
user questions, and growing conversation history) that the trie structure
discarded when it only stored assistant marker tokens as prompt_tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch


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


def _find_turn_boundaries(
    loss_mask: list[int],
) -> tuple[list[int], list[int]]:
    """Scan loss_mask for 0→1 and 1→0 transitions.

    Returns (turn_response_starts, turn_response_ends) where each pair
    defines a half-open range [start, end) of response tokens.
    """
    starts: list[int] = []
    ends: list[int] = []
    in_response = False
    for i, v in enumerate(loss_mask):
        if v == 1 and not in_response:
            starts.append(i)
            in_response = True
        elif v == 0 and in_response:
            ends.append(i)
            in_response = False
    if in_response:
        ends.append(len(loss_mask))
    return starts, ends


def _get_query_id(traj: dict[str, Any]) -> str:
    """Derive a query ID from the prompt tokens in a trajectory."""
    loss_mask = traj["loss_mask"]
    input_ids = traj["input_ids"]
    # Use the 1D shape if present, otherwise use the first batch item
    if input_ids.dim() == 2:
        lm = loss_mask[0]
        ids = input_ids[0]
    else:
        lm = loss_mask
        ids = input_ids
    prompt_tokens = ids[lm == 0].tolist()
    prompt_str = ",".join(str(t) for t in prompt_tokens)
    return hashlib.md5(prompt_str.encode()).hexdigest()


class MCTSTreeStore:
    """Flat trajectory store with MCTS statistics.

    Manages multiple trajectories per query, tracks MCTS statistics
    (visit counts, Q-values) per trajectory, and provides cache-aware
    loading of untrained trajectories.
    """

    def __init__(self) -> None:
        self.trajectories: dict[str, list[TrajectoryRecord]] = {}
        self._seq_id_to_key: dict[int, tuple[str, int]] = {}
        self._query_seq_ids: dict[str, list[int]] = {}
        self._next_seq_id: int = 0

        self._visit_counts: dict[int, int] = {}
        self._total_values: dict[int, float] = {}
        self._q_values: dict[int, float] = {}

        self._trained: dict[int, bool] = {}
        self._rewards: dict[int, float] = {}

    def _backup(self, seq_id: int, reward: float) -> None:
        """Update MCTS stats for a single trajectory."""
        self._visit_counts[seq_id] = self._visit_counts.get(seq_id, 0) + 1
        self._total_values[seq_id] = self._total_values.get(seq_id, 0.0) + reward
        self._q_values[seq_id] = self._total_values[seq_id] / self._visit_counts[seq_id]

    def _make_record(
        self, traj: dict[str, Any], idx: int, seq_len: int
    ) -> TrajectoryRecord:
        """Extract an unpadded sample from traj[idx] and derive turn boundaries."""
        input_ids = traj["input_ids"][idx, :seq_len].tolist()
        loss_mask = traj["loss_mask"][idx, :seq_len].tolist()
        logprobs = (
            traj["logprobs"][idx, :seq_len].tolist()
            if "logprobs" in traj
            else [0.0] * seq_len
        )
        versions = (
            traj["versions"][idx, :seq_len].tolist()
            if "versions" in traj
            else [0] * seq_len
        )
        rewards = traj["rewards"]
        reward = rewards[idx].item() if rewards.dim() >= 1 else rewards.item()

        starts, ends = _find_turn_boundaries(loss_mask)
        return TrajectoryRecord(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            reward=reward,
            turn_response_starts=starts,
            turn_response_ends=ends,
        )

    def _insert_single(
        self, query_id: str, record: TrajectoryRecord
    ) -> int:
        """Insert a single TrajectoryRecord and assign a seq_id."""
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        idx = len(self.trajectories.setdefault(query_id, []))
        self.trajectories[query_id].append(record)
        self._seq_id_to_key[seq_id] = (query_id, idx)
        self._query_seq_ids.setdefault(query_id, []).append(seq_id)

        self._backup(seq_id, record.reward)
        self._trained[seq_id] = False
        self._rewards[seq_id] = record.reward

        return seq_id

    def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
        """Insert trajectories into the store.

        Handles both individual (batch_size=1) and grouped (batch_size>1)
        trajectory dicts. Padding is stripped per sample using attention_mask.
        Turn boundaries are derived from loss_mask.

        Trajectories that already carry _mcts_seq_id or _mcts_seq_ids
        are skipped (loaded from cache).
        """
        for traj in trajectories:
            if "_mcts_seq_id" in traj or "_mcts_seq_ids" in traj:
                continue

            input_ids = traj["input_ids"]
            batch_size = input_ids.shape[0]

            if batch_size == 1:
                query_id = traj.get("_mcts_query_id") or _get_query_id(traj)
                seq_len = int(traj["attention_mask"].sum())
                record = self._make_record(traj, 0, seq_len)
                seq_id = self._insert_single(query_id, record)
                traj["_mcts_seq_id"] = seq_id
                traj["_mcts_query_id"] = query_id
            else:
                seq_ids: list[int] = []
                query_id = traj.get("_mcts_query_id")
                for i in range(batch_size):
                    single = {
                        "input_ids": input_ids[i : i + 1],
                        "loss_mask": traj["loss_mask"][i : i + 1],
                        "rewards": traj["rewards"][i : i + 1],
                    }
                    qid = query_id or _get_query_id(single)
                    if query_id is None:
                        query_id = qid
                    seq_len = int(traj["attention_mask"][i].sum())
                    record = self._make_record(traj, i, seq_len)
                    seq_id = self._insert_single(qid, record)
                    seq_ids.append(seq_id)

                traj["_mcts_seq_ids"] = seq_ids
                traj["_mcts_query_id"] = query_id

    def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return per-token advantages: Q-value on response tokens, 0 on prompt tokens."""
        qid, idx = self._seq_id_to_key[seq_id]
        record = self.trajectories[qid][idx]
        q_val = self._q_values.get(seq_id, 0.0)
        seq_len = len(record.input_ids)
        advantages = torch.zeros(seq_len, dtype=torch.float32)
        for start, end in zip(record.turn_response_starts, record.turn_response_ends):
            advantages[start:end] = q_val
        return advantages

    def get_prompt_mask(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return boolean mask: True for response tokens, False for prompt."""
        qid, idx = self._seq_id_to_key[seq_id]
        record = self.trajectories[qid][idx]
        return torch.tensor(record.loss_mask, dtype=torch.bool)

    def set_trained(self, query_id: str, seq_id: int, trained: bool = True) -> None:
        self._trained[seq_id] = trained

    def is_trained(self, query_id: str, seq_id: int) -> bool:
        return self._trained.get(seq_id, False)

    def get_reward(self, query_id: str, seq_id: int) -> float:
        return self._rewards.get(seq_id, 0.0)

    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self._query_seq_ids:
            return 0
        return sum(
            1
            for seq_id in self._query_seq_ids[query_id]
            if not self._trained.get(seq_id, False)
        )

    def get_untrained_seq_ids(
        self, query_id: str, n_samples: int
    ) -> list[int]:
        if query_id not in self._query_seq_ids:
            return []
        result: list[int] = []
        for seq_id in self._query_seq_ids[query_id]:
            if not self._trained.get(seq_id, False):
                result.append(seq_id)
                if len(result) >= n_samples:
                    break
        return result

    def load_trajectories(
        self, query_id: str, n_samples: int
    ) -> list[dict[str, Any]]:
        """Load untrained trajectories as [1, seq_len] dicts.

        Returns stored input_ids/loss_mask directly — no reconstruction.
        """
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
        result: list[dict[str, Any]] = []
        for seq_id in untrained_ids:
            qid, idx = self._seq_id_to_key[seq_id]
            record = self.trajectories[qid][idx]
            seq_len = len(record.input_ids)
            result.append(
                {
                    "input_ids": torch.tensor(
                        record.input_ids, dtype=torch.int32
                    ).unsqueeze(0),
                    "loss_mask": torch.tensor(
                        record.loss_mask, dtype=torch.int32
                    ).unsqueeze(0),
                    "logprobs": torch.tensor(
                        record.logprobs, dtype=torch.float32
                    ).unsqueeze(0),
                    "versions": torch.tensor(
                        record.versions, dtype=torch.int32
                    ).unsqueeze(0),
                    "attention_mask": torch.ones(
                        seq_len, dtype=torch.bool
                    ).unsqueeze(0),
                    "rewards": torch.tensor(
                        [record.reward], dtype=torch.float32
                    ).unsqueeze(0),
                    "_mcts_query_id": query_id,
                    "_mcts_seq_id": seq_id,
                }
            )
        return result

    def reset_trained_flags(self) -> None:
        for key in self._trained:
            self._trained[key] = False

    def clear(self) -> None:
        """Reset all trajectories, stats, and indices."""
        self.trajectories.clear()
        self._seq_id_to_key.clear()
        self._query_seq_ids.clear()
        self._next_seq_id = 0
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._trained.clear()
        self._rewards.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestFindTurnBoundaries tests/test_tree_search/test_mcts_tree_store.py::TestTrajectoryRecord -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "refactor(tree-search): add TrajectoryRecord and _find_turn_boundaries"
```

______________________________________________________________________

### Task 2: Write tests for the new `MCTSTreeStore` API

**Files:**

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Add store tests to the test file**

Append the following test classes to `tests/test_tree_search/test_mcts_tree_store.py`
(after `TestTrajectoryRecord`):

```python
def _make_traj(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    logprobs: list[float] | None = None,
    versions: list[int] | None = None,
    query_id: str | None = None,
) -> dict[str, Any]:
    """Create a single trajectory dict (batch_size=1) for testing."""
    seq_len = len(input_ids)
    traj = {
        "input_ids": torch.tensor([input_ids], dtype=torch.int32),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.int32),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
    }
    if logprobs is not None:
        traj["logprobs"] = torch.tensor([logprobs], dtype=torch.float32)
    if versions is not None:
        traj["versions"] = torch.tensor([versions], dtype=torch.int32)
    if query_id is not None:
        traj["_mcts_query_id"] = query_id
    return traj


class TestMCTSTreeStoreInsertBatch:
    def test_insert_single_trajectory(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        assert "_mcts_seq_id" in traj
        assert traj["_mcts_query_id"] == "q1"
        assert len(store.trajectories["q1"]) == 1

    def test_insert_two_trajectories_same_query(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        assert len(store.trajectories["q1"]) == 2
        assert t1["_mcts_seq_id"] != t2["_mcts_seq_id"]

    def test_insert_grouped_trajectory(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.int32),
            "rewards": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool
            ),
            "_mcts_query_id": "q1",
        }
        store.insert_batch([traj])
        assert "_mcts_seq_ids" in traj
        assert len(traj["_mcts_seq_ids"]) == 2
        # Second sample should be trimmed to 3 tokens (attention_mask sum)
        record1 = store.trajectories["q1"][1]
        assert len(record1.input_ids) == 3

    def test_insert_skips_already_inserted(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        seq_id_1 = traj["_mcts_seq_id"]
        # Insert again — should skip
        store.insert_batch([traj])
        assert traj["_mcts_seq_id"] == seq_id_1
        assert len(store.trajectories["q1"]) == 1

    def test_insert_stores_logprobs_and_versions(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 1, 1, 1],
            query_id="q1",
        )
        store.insert_batch([traj])
        record = store.trajectories["q1"][0]
        assert record.logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]
        assert record.versions == [0, 0, 1, 1, 1]


class TestMCTSTreeStoreAdvantages:
    def test_get_advantages_single_turn(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        seq_id = traj["_mcts_seq_id"]
        adv = store.get_advantages("q1", seq_id)
        assert adv.shape == torch.Size([5])
        # Prompt tokens (0,1) should be 0
        assert torch.allclose(adv[:2], torch.zeros(2))
        # Response tokens (2,3,4) should be Q-value = 2.0
        assert torch.allclose(adv[2:], torch.full((3,), 2.0))

    def test_get_advantages_multi_turn(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
            query_id="q1",
        )
        store.insert_batch([traj])
        seq_id = traj["_mcts_seq_id"]
        adv = store.get_advantages("q1", seq_id)
        # Prompt tokens: 0
        assert torch.allclose(adv[:2], torch.zeros(2))
        # Turn 1 response (2,3): 0.75
        assert torch.allclose(adv[2:4], torch.full((2,), 0.75))
        # Turn 2 prompt (4,5): 0
        assert torch.allclose(adv[4:6], torch.zeros(2))
        # Turn 2 response (6,7): 0.75
        assert torch.allclose(adv[6:8], torch.full((2,), 0.75))

    def test_get_prompt_mask(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], query_id="q1")
        store.insert_batch([traj])
        mask = store.get_prompt_mask("q1", traj["_mcts_seq_id"])
        assert mask.tolist() == [False, False, True, True, True]


class TestMCTSTreeStoreTrainedFlag:
    def test_trained_flag_default_false(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        assert store.is_trained("q1", traj["_mcts_seq_id"]) is False

    def test_set_trained(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.set_trained("q1", traj["_mcts_seq_id"], True)
        assert store.is_trained("q1", traj["_mcts_seq_id"]) is True

    def test_get_untrained_count(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        t3 = _make_traj([7, 8, 9], [0, 0, 1], reward=0.3, query_id="q1")
        store.insert_batch([t1, t2, t3])
        assert store.get_untrained_count("q1") == 3
        store.set_trained("q1", t1["_mcts_seq_id"], True)
        assert store.get_untrained_count("q1") == 2

    def test_reset_trained_flags(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.set_trained("q1", traj["_mcts_seq_id"], True)
        store.reset_trained_flags()
        assert store.is_trained("q1", traj["_mcts_seq_id"]) is False

    def test_get_reward(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        assert store.get_reward("q1", t1["_mcts_seq_id"]) == 1.0
        assert store.get_reward("q1", t2["_mcts_seq_id"]) == 0.5


class TestMCTSTreeStoreLoadTrajectories:
    def test_load_trajectories_basic(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=1.0, query_id="q1")
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert len(loaded) == 1
        t = loaded[0]
        assert t["input_ids"].shape[0] == 1
        assert t["input_ids"].shape[1] == 5
        assert t["rewards"].item() == 1.0
        assert t["_mcts_query_id"] == "q1"
        assert t["_mcts_seq_id"] == traj["_mcts_seq_id"]

    def test_load_trajectories_only_untrained(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        store.set_trained("q1", t1["_mcts_seq_id"], True)
        loaded = store.load_trajectories("q1", n_samples=2)
        assert len(loaded) == 1
        assert loaded[0]["rewards"].item() == 0.5

    def test_load_trajectories_preserves_loss_mask(self):
        """Verify that loss_mask is preserved exactly as stored (no reconstruction bugs)."""
        store = MCTSTreeStore()
        loss_mask = [0, 0, 1, 1, 0, 0, 1, 1]
        traj = _make_traj(
            [1, 2, 3, 4, 5, 6, 7, 8], loss_mask, reward=1.0, query_id="q1"
        )
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        torch.testing.assert_close(
            loaded[0]["loss_mask"].squeeze(0),
            torch.tensor(loss_mask, dtype=torch.int32),
        )

    def test_load_trajectories_unknown_query(self):
        store = MCTSTreeStore()
        assert store.load_trajectories("nonexistent", n_samples=1) == []

    def test_load_trajectories_attention_mask_all_ones(self):
        """Loaded trajectories have no padding, so attention_mask should be all ones."""
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], query_id="q1")
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert loaded[0]["attention_mask"].all()


class TestMCTSTreeStoreClear:
    def test_clear(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.clear()
        assert len(store.trajectories) == 0
        assert store._next_seq_id == 0
        assert len(store._visit_counts) == 0
        assert len(store._q_values) == 0


class TestMCTSTreeStoreMCTSStats:
    def test_backup_updates_stats(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        seq_id = traj["_mcts_seq_id"]
        assert store._visit_counts[seq_id] == 1
        assert store._q_values[seq_id] == 2.0

    def test_two_trajectories_average_q_value(self):
        """Two trajectories with same query_id: Q-values are per-seq_id, not averaged."""
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1")
        store.insert_batch([t1, t2])
        # Each seq_id gets its own Q-value (reward), not averaged
        assert store._q_values[t1["_mcts_seq_id"]] == 1.0
        assert store._q_values[t2["_mcts_seq_id"]] == 0.0
```

- [ ] **Step 2: Run store tests**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v` Expected: All new
tests PASS. (Old test classes from the previous version will FAIL because they reference
the old API — that's expected, we're replacing them.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_tree_search/test_mcts_tree_store.py
git commit -m "test(tree-search): add new MCTSTreeStore tests for flat store"
```

______________________________________________________________________

### Task 3: Rewrite `checkpoint.py` for the new format + tests

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py`

- Modify: `tests/test_tree_search/test_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint tests**

Replace `tests/test_tree_search/test_checkpoint.py` with:

```python
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _make_store_with_data() -> MCTSTreeStore:
    """Create a store with sample data for checkpoint tests."""
    import torch

    store = MCTSTreeStore()
    t1 = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "rewards": torch.tensor([2.0], dtype=torch.float32),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "_mcts_query_id": "q1",
    }
    t2 = {
        "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.int32),
        "loss_mask": torch.tensor([[0, 0, 1]], dtype=torch.int32),
        "rewards": torch.tensor([0.5], dtype=torch.float32),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "_mcts_query_id": "q2",
    }
    store.insert_batch([t1, t2])
    return store


class TestTreeCheckpointManager:
    def test_save_and_load(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)
        assert manager.exists()

        loaded = manager.load()
        assert len(loaded.trajectories) == 2
        assert "q1" in loaded.trajectories
        assert "q2" in loaded.trajectories

    def test_exists_false_when_no_dir(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path / "nonexistent"))
        assert not manager.exists()

    def test_save_creates_directory(self, tmp_path):
        save_dir = str(tmp_path / "new_dir")
        manager = TreeCheckpointManager(save_dir)
        store = _make_store_with_data()
        manager.save(store)
        import os

        assert os.path.isdir(os.path.join(save_dir, "mcts_trees"))

    def test_load_preserves_seq_id_counter(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        import torch

        t3 = {
            "input_ids": torch.tensor([[9, 10]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 1]], dtype=torch.int32),
            "rewards": torch.tensor([1.0], dtype=torch.float32),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "_mcts_query_id": "q3",
        }
        loaded.insert_batch([t3])
        assert t3["_mcts_seq_id"] == 2

    def test_load_preserves_trajectory_data(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        # Verify trajectory content
        record_q1 = loaded.trajectories["q1"][0]
        assert record_q1.input_ids == [1, 2, 3, 4, 5]
        assert record_q1.loss_mask == [0, 0, 1, 1, 1]
        assert record_q1.reward == 2.0
        record_q2 = loaded.trajectories["q2"][0]
        assert record_q2.input_ids == [6, 7, 8]
        assert record_q2.reward == 0.5

    def test_load_preserves_mcts_stats(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        # Q-values should be preserved directly (no rebuild needed)
        seq_ids = loaded._query_seq_ids["q1"]
        assert loaded._q_values[seq_ids[0]] == 2.0
        seq_ids = loaded._query_seq_ids["q2"]
        assert loaded._q_values[seq_ids[0]] == 0.5

    def test_load_preserves_trained_flags(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        # Mark one as trained
        seq_ids = store._query_seq_ids["q1"]
        store.set_trained("q1", seq_ids[0], True)
        manager.save(store)

        loaded = manager.load()
        assert loaded.is_trained("q1", seq_ids[0]) is True

    def test_load_preserves_turn_boundaries(self, tmp_path):
        import torch

        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore()
        # Multi-turn trajectory: two response regions
        traj = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.int32),
            "rewards": torch.tensor([0.75], dtype=torch.float32),
            "attention_mask": torch.ones(1, 8, dtype=torch.bool),
            "_mcts_query_id": "q1",
        }
        store.insert_batch([traj])
        manager.save(store)

        loaded = manager.load()
        record = loaded.trajectories["q1"][0]
        assert record.turn_response_starts == [2, 6]
        assert record.turn_response_ends == [4, 8]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tree_search/test_checkpoint.py -v` Expected: FAIL (old
checkpoint.py still uses TrieNode)

- [ ] **Step 3: Rewrite checkpoint.py**

Replace `customized_areal/tree_search/checkpoint.py` with:

```python
# customized_areal/tree_search/checkpoint.py
"""Checkpoint save/load for the flat TrajectoryRecord store.

Unlike the old TrieNode-based format, MCTS stats are keyed by seq_id (int)
and serialize directly — no rebuild_mcts_stats() needed after loading.
Old TrieNode-based checkpoints are incompatible and must be discarded.
"""

from __future__ import annotations

import json
import os

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, TrajectoryRecord


class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")

    def exists(self) -> bool:
        return os.path.isdir(self.save_dir) and os.path.isfile(
            os.path.join(self.save_dir, "metadata.json")
        )

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
            "next_seq_id": tree_store._next_seq_id,
            "seq_id_to_key": {
                str(k): [v[0], v[1]]
                for k, v in tree_store._seq_id_to_key.items()
            },
            "query_seq_ids": {
                k: v for k, v in tree_store._query_seq_ids.items()
            },
            "visit_counts": {str(k): v for k, v in tree_store._visit_counts.items()},
            "total_values": {
                str(k): v for k, v in tree_store._total_values.items()
            },
            "q_values": {str(k): v for k, v in tree_store._q_values.items()},
            "trained": {str(k): v for k, v in tree_store._trained.items()},
            "rewards": {str(k): v for k, v in tree_store._rewards.items()},
        }
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)

        store._next_seq_id = metadata["next_seq_id"]
        store._seq_id_to_key = {
            int(k): (v[0], v[1]) for k, v in metadata["seq_id_to_key"].items()
        }
        store._query_seq_ids = metadata["query_seq_ids"]
        store._visit_counts = {
            int(k): v for k, v in metadata["visit_counts"].items()
        }
        store._total_values = {
            int(k): v for k, v in metadata["total_values"].items()
        }
        store._q_values = {int(k): v for k, v in metadata["q_values"].items()}
        store._trained = {int(k): v for k, v in metadata["trained"].items()}
        store._rewards = {int(k): v for k, v in metadata["rewards"].items()}

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

    @staticmethod
    def _serialize_record(record: TrajectoryRecord) -> dict:
        return {
            "input_ids": record.input_ids,
            "loss_mask": record.loss_mask,
            "logprobs": record.logprobs,
            "versions": record.versions,
            "reward": record.reward,
            "turn_response_starts": record.turn_response_starts,
            "turn_response_ends": record.turn_response_ends,
        }

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
        )
```

- [ ] **Step 4: Run checkpoint tests**

Run: `uv run pytest tests/test_tree_search/test_checkpoint.py -v` Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_tree_search/test_checkpoint.py
git commit -m "refactor(tree-search): rewrite checkpoint for flat TrajectoryRecord store"
```

______________________________________________________________________

### Task 4: Update `config.py`, `advantage.py`, `trainer.py`, `__init__.py`

This is an atomic change — all these files import each other and must be updated
together.

**Files:**

- Modify: `customized_areal/tree_search/config.py`

- Modify: `customized_areal/tree_search/advantage.py`

- Modify: `customized_areal/tree_search/trainer.py`

- Modify: `customized_areal/tree_search/__init__.py`

- [ ] **Step 1: Update `config.py` — remove `assistant_marker`**

In `customized_areal/tree_search/config.py`, remove `assistant_marker` from
`TreeBackupConfig`:

```python
from dataclasses import dataclass
from enum import Enum


class TreeBackupMode(str, Enum):
    OFF = "off"
    IN_TRAINING = "in_training"
    CROSS_TRAINING = "cross_training"


class AdvantageMode(str, Enum):
    GAE = "gae"
    TREE = "tree"


@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.GAE


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1
```

- [ ] **Step 2: Update `advantage.py` — remove `Turn` import, use new store API**

The `TreeAdvantageComputer` doesn't need changes since `get_advantages` and
`get_prompt_mask` signatures are unchanged. Just remove the `Turn` import and verify the
code works with the new store.

In `customized_areal/tree_search/advantage.py`, the file is already clean — no `Turn` or
`TrieNode` imports. Verify it only imports from `mcts_tree_store`. No changes needed if
it already uses `self.tree_store.get_advantages` and `self.tree_store.get_prompt_mask`.

- [ ] **Step 3: Update `trainer.py` — remove `make_turn_splitter` usage**

In `customized_areal/tree_search/trainer.py`:

- Remove `from customized_areal.tree_search.turn_splitter import make_turn_splitter`
  (line 35)
- In `_init_tree_components` (line 299-321): remove
  `turn_splitter = make_turn_splitter(...)` and pass nothing to `MCTSTreeStore()` and
  `TreeCheckpointManager.load()`
- The `MCTSTreeStore()` constructor now takes no arguments

Replace `_init_tree_components` with:

```python
def _init_tree_components(self) -> None:
    """Create tree store, advantage computer, and checkpoint manager."""
    self.tree_store = MCTSTreeStore()
    self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
    self.tree_checkpoint_manager = TreeCheckpointManager(
        self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
    )

    # Load existing tree checkpoint if available (CROSS_TRAINING mode)
    if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
        if self.tree_checkpoint_manager.exists():
            self.tree_store = self.tree_checkpoint_manager.load()
            logger.info("Loaded MCTS tree checkpoint with cached rollouts")

    # Reset trained flags for a fresh training run
    self.tree_store.reset_trained_flags()

    self._batch_builder = _CacheAwareBatchBuilder(
        self.tree_store, self.cache_config.n_samples, self.tokenizer
    )
```

- [ ] **Step 4: Update `__init__.py` — remove deleted exports**

Replace `customized_areal/tree_search/__init__.py` with:

```python
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import (
    AdvantageMode,
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, TrajectoryRecord
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

__all__ = [
    "AdvantageMode",
    "CacheAwarePPOTrainer",
    "MCTSTreeStore",
    "QueryIDProxyWorkflow",
    "RolloutCacheConfig",
    "TreeAdvantageComputer",
    "TreeBackupConfig",
    "TreeBackupMode",
    "TreeCheckpointManager",
    "TrajectoryRecord",
]
```

- [ ] **Step 5: Run tests to verify nothing is broken at import level**

Run:
`uv run python -c "from customized_areal.tree_search import MCTSTreeStore, TrajectoryRecord, TreeCheckpointManager, CacheAwarePPOTrainer; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/config.py customized_areal/tree_search/advantage.py customized_areal/tree_search/trainer.py customized_areal/tree_search/__init__.py
git commit -m "refactor(tree-search): update consumers for flat TrajectoryRecord store"
```

______________________________________________________________________

### Task 5: Delete old files and their tests

**Files:**

- Delete: `customized_areal/tree_search/trie_node.py`

- Delete: `customized_areal/tree_search/turn_splitter.py`

- Delete: `tests/test_tree_search/test_trie_node.py`

- Delete: `tests/test_tree_search/test_turn_splitter.py`

- [ ] **Step 1: Delete the files**

```bash
rm customized_areal/tree_search/trie_node.py
rm customized_areal/tree_search/turn_splitter.py
rm tests/test_tree_search/test_trie_node.py
rm tests/test_tree_search/test_turn_splitter.py
```

- [ ] **Step 2: Verify no remaining imports reference deleted files**

Run:
`grep -r "from customized_areal.tree_search.trie_node\|from customized_areal.tree_search.turn_splitter" customized_areal/ tests/ --include="*.py"`
Expected: No matches (all references were updated in Task 4)

- [ ] **Step 3: Commit**

```bash
git add -u customized_areal/tree_search/trie_node.py customized_areal/tree_search/turn_splitter.py tests/test_tree_search/test_trie_node.py tests/test_tree_search/test_turn_splitter.py
git commit -m "refactor(tree-search): delete TrieNode and turn_splitter (replaced by flat store)"
```

______________________________________________________________________

### Task 6: Update remaining test files

**Files:**

- Modify: `tests/test_tree_search/test_advantage.py`

- Modify: `tests/test_tree_search/test_trainer.py`

- Modify: `tests/test_tree_search/test_cache_trainer.py`

- Modify: `tests/test_tree_search/test_batch_consistency.py`

- [ ] **Step 1: Rewrite `test_advantage.py`**

Replace `tests/test_tree_search/test_advantage.py` with:

```python
import torch

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _make_traj(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    query_id: str = "q1",
) -> dict:
    """Create a single trajectory dict for testing."""
    seq_len = len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.int32),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.int32),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "_mcts_query_id": query_id,
    }


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0)
        store.insert_batch([traj])
        computer.compute([traj])
        assert "advantages" in traj
        assert "returns" in traj
        # Advantages should be zeroed for prompt tokens (first 2)
        assert torch.allclose(traj["advantages"][:2], torch.zeros(2))
        # Response tokens should have q_value = 2.0
        assert torch.allclose(traj["advantages"][2:], torch.full((3,), 2.0))

    def test_compute_returns_equal_advantages(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        traj = _make_traj([1, 10, 3], [0, 0, 1], reward=1.0)
        store.insert_batch([traj])
        computer.compute([traj])
        assert torch.allclose(traj["returns"], traj["advantages"])

    def test_compute_multi_turn_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        traj = _make_traj(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
        )
        store.insert_batch([traj])
        computer.compute([traj])
        # Prompt tokens: 0
        assert torch.allclose(traj["advantages"][:2], torch.zeros(2))
        # Turn 1 response: 0.75
        assert torch.allclose(traj["advantages"][2:4], torch.full((2,), 0.75))
        # Turn 2 prompt: 0
        assert torch.allclose(traj["advantages"][4:6], torch.zeros(2))
        # Turn 2 response: 0.75
        assert torch.allclose(traj["advantages"][6:8], torch.full((2,), 0.75))

    def test_compute_two_trajectories(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        t1 = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        t2 = _make_traj([5, 6, 7, 8], [0, 0, 1, 1], reward=0.5, query_id="q2")
        store.insert_batch([t1, t2])
        computer.compute([t1, t2])
        assert "advantages" in t1
        assert "advantages" in t2
```

- [ ] **Step 2: Rewrite `test_trainer.py`**

Replace `tests/test_tree_search/test_trainer.py` with:

```python
# tests/test_tree_search/test_trainer.py
from unittest.mock import MagicMock

from customized_areal.tree_search.config import AdvantageMode, TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.trainer import (
    patch_ppo_actor_for_tree_backup,
    unpatch_ppo_actor,
)

from areal.trainer.ppo.actor import PPOActor


class TestPatchOuterMethod:
    def setup_method(self):
        unpatch_ppo_actor()

    def teardown_method(self):
        unpatch_ppo_actor()

    def test_patch_replaces_compute_advantages(self):
        original = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.TREE)
        assert PPOActor.compute_advantages is not original
        assert hasattr(PPOActor, "_original_compute_advantages")

    def test_unpatch_restores_original(self):
        original = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.TREE)
        unpatch_ppo_actor()
        assert PPOActor.compute_advantages is original
        assert not hasattr(PPOActor, "_original_compute_advantages")

    def test_double_patch_replaces_previous(self):
        original = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.TREE)
        first_patched = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.GAE)
        # _original should always point to the TRUE original
        assert PPOActor._original_compute_advantages is original
        unpatch_ppo_actor()
        assert PPOActor.compute_advantages is original


class TestUnpatchSafety:
    def setup_method(self):
        unpatch_ppo_actor()

    def teardown_method(self):
        unpatch_ppo_actor()

    def test_unpatch_without_patch_is_safe(self):
        unpatch_ppo_actor()


class TestTreeBackupConfigDefaults:
    def test_default_mode_is_off(self):
        config = TreeBackupConfig()
        assert config.mode == TreeBackupMode.OFF

    def test_no_assistant_marker_field(self):
        config = TreeBackupConfig()
        assert not hasattr(config, "assistant_marker")
```

- [ ] **Step 3: Rewrite `test_cache_trainer.py`**

Replace `tests/test_tree_search/test_cache_trainer.py` with:

```python
import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _make_traj_for_store(
    input_ids: list[int], loss_mask: list[int], *, reward: float = 1.0, query_id: str = "q1"
) -> dict:
    seq_len = len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.int32),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.int32),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "_mcts_query_id": query_id,
    }


class TestCacheAwareBatchBuilder:
    def test_build_batch_fully_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(4):
            traj = _make_traj_for_store(
                [1, 2, 3, 4, 5 + i], [0, 0, 1, 1, 1], reward=1.0 / (i + 1), query_id="q1"
            )
            store.insert_batch([traj])

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 4
        assert cached[0]["need_gen_count"] == 0

    def test_build_batch_partially_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(2):
            traj = _make_traj_for_store(
                [1, 2, 3, 4, 5 + i], [0, 0, 1, 1, 1], reward=1.0 / (i + 1), query_id="q1"
            )
            store.insert_batch([traj])

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 2
        assert cached[0]["need_gen_count"] == 2

    def test_build_batch_not_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 0
        assert len(need_gen) == 1

    def test_load_cached_trajectories(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        t1 = _make_traj_for_store([1, 2, 3, 4], [0, 0, 1, 1], reward=1.0, query_id="q1")
        t2 = _make_traj_for_store([5, 6, 7, 8], [0, 0, 1, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        cached, _ = builder.split_prompts([{"_mcts_query_id": "q1"}])
        loaded = builder.load_cached_trajectories(cached)
        assert len(loaded) == 2
```

- [ ] **Step 4: Update `test_batch_consistency.py` — remove
  `_split_grouped_trajectories`**

In `tests/test_tree_search/test_batch_consistency.py`:

- Remove `from customized_areal.tree_search.trainer import _split_grouped_trajectories`
  (line 21)
- Remove or comment out `TestSplitGroupedTrajectories` class and `TestEndToEndRoundtrip`
  class (they test the deleted function)
- Keep `TestConcatPaddedTensorsPreservesValues` and `TestGroupedRolloutWorkflow` and
  `TestGPUBatchConsistency`

Remove these classes:

- `TestSplitGroupedTrajectories` (lines 148-228)
- `TestEndToEndRoundtrip` (lines 310-376)

And update the docstring at the top to remove references to
`_split_grouped_trajectories`.

- [ ] **Step 5: Run all tree_search tests**

Run:
`uv run pytest tests/test_tree_search/ -v --ignore=tests/test_tree_search/test_batch_consistency.py`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_tree_search/test_advantage.py tests/test_tree_search/test_trainer.py tests/test_tree_search/test_cache_trainer.py tests/test_tree_search/test_batch_consistency.py
git commit -m "test(tree-search): update tests for flat TrajectoryRecord store"
```

______________________________________________________________________

### Task 7: Run full test suite and fix any remaining issues

**Files:**

- Potentially any file with remaining references

- [ ] **Step 1: Check for any remaining references to deleted modules**

Run:
`grep -r "trie_node\|turn_splitter\|Turn\|TrieNode\|make_turn_splitter\|_split_grouped_trajectories" customized_areal/ tests/ --include="*.py" | grep -v __pycache__ | grep -v "\.pyc" | grep -v "docs/superpowers" | grep -v "TRAINING_PIPELINE"`
Expected: No matches in source/test code (docs and pipeline docs are OK)

- [ ] **Step 2: Run full tree_search test suite**

Run: `uv run pytest tests/test_tree_search/ -v` Expected: All PASS

- [ ] **Step 3: Run pre-commit**

Run: `pre-commit run --files customized_areal/tree_search/ tests/test_tree_search/`
Expected: PASS (or fix any formatting/linting issues)

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "style(tree-search): pre-commit fixes for flat store refactor"
```

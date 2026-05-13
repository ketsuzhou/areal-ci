# Episode-Aware cached_count and Advantage Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the semantic mismatch where `cached_count` counts untrained nodes instead of untrained episodes, and change `TreeAdvantageComputer.compute()` to normalize per-episode instead of per-node.

**Architecture:** Add two episode-aware methods to `MCTSTreeStore` that derive episode grouping from existing `episode_id` attributes on Nodes. Rewrite `TreeAdvantageComputer.compute()` to group by `(query_id, episode_id)`. Update the workflow to call the new episode-aware methods.

**Tech Stack:** Python 3.12+, PyTorch, pytest

---

### Task 1: Add `get_untrained_episode_count` to MCTSTreeStore

**Files:**
- Modify: `customized_areal/tree_search/mcts_tree_store.py:307-314`
- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py` after `TestMCTSTreeStoreTrainId`:

```python
class TestGetUntrainedEpisodeCount:
    def test_no_episodes(self):
        store = MCTSTreeStore()
        assert store.get_untrained_episode_count("q1") == 0

    def test_unknown_query(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        assert store.get_untrained_episode_count("q_other") == 0

    def test_all_untrained(self):
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
        assert store.get_untrained_episode_count("q1") == 2

    def test_multi_node_episodes(self):
        """An episode with multiple nodes counts as one episode."""
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
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n3 = Node(
            input_ids=[7, 8, 9], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2, n3])
        assert store.get_untrained_episode_count("q1") == 2

    def test_trained_episode_excluded(self):
        """An episode where all nodes are trained is excluded from count."""
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
        assert store.get_untrained_episode_count("q1") == 1

    def test_partially_trained_episode_still_counted(self):
        """An episode is untrained if any of its nodes is untrained."""
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
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1, n2])
        # Train only one node of the episode
        store.set_trained(n1.node_id, True)
        # Episode is still untrained because n2 is untrained
        assert store.get_untrained_episode_count("q1") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestGetUntrainedEpisodeCount -v`
Expected: FAIL with `AttributeError: 'MCTSTreeStore' object has no attribute 'get_untrained_episode_count'`

- [ ] **Step 3: Write minimal implementation**

Add to `customized_areal/tree_search/mcts_tree_store.py` after `get_untrained_count` (line 314):

```python
    def get_untrained_episode_count(self, query_id: str) -> int:
        """Count untrained episodes for a query.

        An episode is untrained if any of its nodes is untrained
        (train_id != current_train_id).
        """
        if query_id not in self._query_node_ids:
            return 0
        episode_has_untrained: dict[str, bool] = {}
        for node_id in self._query_node_ids[query_id]:
            key = self._node_id_to_key.get(node_id)
            if key is None:
                continue
            qid, idx = key
            node = self.trajectories[qid][idx]
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_has_untrained:
                episode_has_untrained[ep_id] = False
            if not self.is_trained(node_id):
                episode_has_untrained[ep_id] = True
        return sum(1 for v in episode_has_untrained.values() if v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestGetUntrainedEpisodeCount -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat: add get_untrained_episode_count to MCTSTreeStore"
```

---

### Task 2: Add `load_untrained_episodes` to MCTSTreeStore

**Files:**
- Modify: `customized_areal/tree_search/mcts_tree_store.py`
- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py` after `TestGetUntrainedEpisodeCount`:

```python
class TestLoadUntrainedEpisodes:
    def test_no_episodes(self):
        store = MCTSTreeStore()
        assert store.load_untrained_episodes("q1", n_episodes=1) == []

    def test_unknown_query(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        assert store.load_untrained_episodes("q_other", n_episodes=1) == []

    def test_returns_all_nodes_from_episode(self):
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
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1, n2])
        loaded = store.load_untrained_episodes("q1", n_episodes=1)
        assert len(loaded) == 2
        assert loaded[0].node_id == "ep_a_1"
        assert loaded[1].node_id == "ep_a_2"

    def test_respects_n_episodes_limit(self):
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
            node_id="ep_c_1", episode_id="ep_c", outcome_reward=0.3, query_id="q1",
        )
        store.insert_batch([n1, n2, n3])
        loaded = store.load_untrained_episodes("q1", n_episodes=2)
        # 2 episodes, each with 1 node = 2 nodes total
        assert len(loaded) == 2
        episode_ids = {n.episode_id for n in loaded}
        assert len(episode_ids) == 2

    def test_skips_trained_episodes(self):
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
        loaded = store.load_untrained_episodes("q1", n_episodes=2)
        assert len(loaded) == 1
        assert loaded[0].node_id == "ep_b_1"

    def test_preserves_episode_order(self):
        """Episodes are returned in insertion order."""
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
        loaded = store.load_untrained_episodes("q1", n_episodes=2)
        assert loaded[0].episode_id == "ep_a"
        assert loaded[1].episode_id == "ep_b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestLoadUntrainedEpisodes -v`
Expected: FAIL with `AttributeError: 'MCTSTreeStore' object has no attribute 'load_untrained_episodes'`

- [ ] **Step 3: Write minimal implementation**

Add to `customized_areal/tree_search/mcts_tree_store.py` after `get_untrained_episode_count`:

```python
    def load_untrained_episodes(self, query_id: str, n_episodes: int) -> list[Node]:
        """Load nodes from up to n_episodes untrained episodes.

        Returns all nodes belonging to the first n_episodes untrained
        episodes (in insertion order). An episode is untrained if any
        of its nodes is untrained.
        """
        if query_id not in self._query_node_ids:
            return []
        # Build episode_id → list of (node_id, query_id, idx) in insertion order
        episode_nodes: dict[str, list[tuple[str, str, int]]] = {}
        episode_order: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            key = self._node_id_to_key.get(node_id)
            if key is None:
                continue
            qid, idx = key
            node = self.trajectories[qid][idx]
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_nodes:
                episode_nodes[ep_id] = []
                episode_order.append(ep_id)
            episode_nodes[ep_id].append((node_id, qid, idx))
        # Select up to n_episodes untrained episodes
        selected: list[Node] = []
        count = 0
        for ep_id in episode_order:
            if count >= n_episodes:
                break
            # Check if any node in this episode is untrained
            is_untrained = False
            for node_id, qid, idx in episode_nodes[ep_id]:
                if not self.is_trained(node_id):
                    is_untrained = True
                    break
            if not is_untrained:
                continue
            count += 1
            for node_id, qid, idx in episode_nodes[ep_id]:
                selected.append(self.trajectories[qid][idx])
        return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestLoadUntrainedEpisodes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat: add load_untrained_episodes to MCTSTreeStore"
```

---

### Task 3: Rewrite `TreeAdvantageComputer.compute()` for episode-level normalization

**Files:**
- Modify: `customized_areal/tree_search/advantage.py:35-86`
- Test: `tests/test_tree_search/test_advantage.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_advantage.py` after `TestTreeAdvantageComputer`:

```python
class TestTreeAdvantageComputerEpisodeLevel:
    def test_episode_level_normalization(self):
        """GRPO normalization operates across episodes, not individual nodes."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        # Episode A: 2 turns, reward 1.0
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[-1, -1, 0],
            outcome_reward=1.0, query_id="q1", node_id="ep_a_1", episode_id="ep_a",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[-1, -1, 0],
            outcome_reward=1.0, query_id="q1", node_id="ep_a_2", episode_id="ep_a",
        )
        # Episode B: 1 turn, reward 0.0
        n3 = Node(
            input_ids=[7, 8, 9], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3], versions=[-1, -1, 0],
            outcome_reward=0.0, query_id="q1", node_id="ep_b_1", episode_id="ep_b",
        )
        store.insert_batch([n1, n2, n3])
        computer.compute([n1, n2, n3])
        # Both nodes in episode A get the same normalized value
        assert n1.advantages is not None
        assert n2.advantages is not None
        assert n3.advantages is not None
        # Response positions (loss_mask=1) in same episode get same advantage
        assert abs(n1.advantages[2].item() - n2.advantages[2].item()) < 1e-6
        # Episode A and B have different rewards, so different advantages
        assert abs(n1.advantages[2].item() - n3.advantages[2].item()) > 0.1
        # Prompt positions are zero
        assert n1.advantages[0].item() == 0.0
        assert n1.advantages[1].item() == 0.0

    def test_episode_level_zero_mean(self):
        """Per-episode GRPO normalization preserves zero-mean property."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        # 2 episodes with rewards 1.0 and -1.0 → mean=0, std=1.0
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[-1, -1, 0],
            outcome_reward=1.0, query_id="q1", node_id="ep_a_1", episode_id="ep_a",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[-1, -1, 0],
            outcome_reward=-1.0, query_id="q1", node_id="ep_b_1", episode_id="ep_b",
        )
        store.insert_batch([n1, n2])
        computer.compute([n1, n2])
        # Response positions: ep_a gets +1.0, ep_b gets -1.0
        assert abs(n1.advantages[2].item() - 1.0) < 1e-5
        assert abs(n2.advantages[2].item() + 1.0) < 1e-5

    def test_episode_level_backward_compat_single_node(self):
        """Single-node trajectories (no episode_id) still work: each node is its own episode."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        t1 = _make_node([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_node([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1")
        store.insert_batch([t1, t2])
        computer.compute([t1, t2])
        # Two nodes with no episode_id → each is its own "episode"
        # Same behavior as before: non-zero normalized values
        assert not torch.allclose(t1.advantages, torch.zeros(3))
        assert not torch.allclose(t2.advantages, torch.zeros(3))

    def test_episode_level_single_episode_zero_advantage(self):
        """A single episode in the query group gets zero advantage (std=0)."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[-1, -1, 0],
            outcome_reward=1.0, query_id="q1", node_id="ep_a_1", episode_id="ep_a",
        )
        store.insert_batch([n1])
        computer.compute([n1])
        assert torch.allclose(n1.advantages, torch.zeros(3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tree_search/test_advantage.py::TestTreeAdvantageComputerEpisodeLevel -v`
Expected: FAIL — current `compute()` normalizes per-node, so `n1.advantages[2]` will not equal `n2.advantages[2]` for nodes in the same episode

- [ ] **Step 3: Rewrite `compute()` implementation**

Replace the body of `TreeAdvantageComputer.compute()` in `customized_areal/tree_search/advantage.py` (lines 35-86) with:

```python
    def compute(self, trajectories: list[Node]) -> None:
        """Replace GAE advantages with per-episode GRPO-normalized outcome_rewards.

        Groups nodes by (query_id, episode_id). Each episode contributes one
        reward (all nodes in an episode share the same outcome_reward).
        GRPO normalization operates across episodes within each query group.
        The normalized return is broadcast to all response positions in
        every node of the episode.
        """
        # Build query_id → {episode_id → [node_ids]} and per-episode reward
        query_episodes: dict[str, dict[str, list[str]]] = {}
        episode_rewards: dict[str, float] = {}  # episode_id → reward

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            ep_id = getattr(traj, "episode_id", "") or node_id
            ep_map = query_episodes.setdefault(query_id, {})
            ep_map.setdefault(ep_id, []).append(node_id)
            # All nodes in an episode share the same outcome_reward
            if ep_id not in episode_rewards:
                episode_rewards[ep_id] = traj.outcome_reward

        # Per-query GRPO normalization of per-episode rewards
        for query_id, ep_map in query_episodes.items():
            ep_ids = list(ep_map.keys())
            rewards = [episode_rewards[eid] for eid in ep_ids]
            if len(rewards) < 2:
                for eid in ep_ids:
                    for nid in ep_map[eid]:
                        self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = var_r**0.5
            for eid, r in zip(ep_ids, rewards):
                norm_val = (r - mean_r) / (std_r + self.grpo_eps)
                for nid in ep_map[eid]:
                    self.tree_store.set_normalized_return(nid, norm_val)

        # Compute per-trajectory advantages and returns
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            mask = traj.loss_mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            norm_return = self.tree_store.get_normalized_return(node_id)
            traj.advantages = mask.float() * norm_return
            traj.returns = mask.float() * norm_return
```

- [ ] **Step 4: Run the new episode-level tests**

Run: `uv run pytest tests/test_tree_search/test_advantage.py::TestTreeAdvantageComputerEpisodeLevel -v`
Expected: PASS

- [ ] **Step 5: Run existing advantage tests to verify backward compatibility**

Run: `uv run pytest tests/test_tree_search/test_advantage.py -v`
Expected: All existing `TestTreeAdvantageComputer` tests still PASS. Key: the single-node tests use `_make_node` which does not set `episode_id`, so `ep_id` falls back to `node_id` — each node becomes its own "episode", preserving the old behavior.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/advantage.py tests/test_tree_search/test_advantage.py
git commit -m "feat: episode-level GRPO normalization in TreeAdvantageComputer"
```

---

### Task 4: Update workflow to use episode-aware methods

**Files:**
- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py:299-341`

- [ ] **Step 1: Update `arun_episode` to use episode-aware methods**

In `customized_areal/tree_search/tree_search_grouped_workflow.py`, change lines 300-302:

From:
```python
        cached_count = (
            self.tree_store.get_untrained_count(query_id) if query_id else 0
        )
```

To:
```python
        cached_count = (
            self.tree_store.get_untrained_episode_count(query_id) if query_id else 0
        )
```

Change lines 339-341:

From:
```python
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_trajectories(
                query_id, cached_count
            )
```

To:
```python
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_untrained_episodes(
                query_id, cached_count
            )
```

- [ ] **Step 2: Run the full test suite for tree_search**

Run: `uv run pytest tests/test_tree_search/ -v`
Expected: All tests PASS

- [ ] **Step 3: Run pre-commit**

Run: `pre-commit run --all-files`
Expected: All checks PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py
git commit -m "feat: use episode-aware caching in TreeSearchGroupedWorkflow"
```

---

### Task 5: Update docstrings and class-level docs

**Files:**
- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py:183-191`

- [ ] **Step 1: Update the class docstring**

In `customized_areal/tree_search/tree_search_grouped_workflow.py`, change the class docstring (lines 183-191):

From:
```python
    """GroupedRolloutWorkflow with tree-search cache reuse, tree ops, and checkpoint.

    Wraps the base OpenAIProxyWorkflow and overrides arun_episode to:
    1. Check cache: how many untrained episodes exist for this query?
    2. Generate only the needed fresh episodes (group_size - cached_count)
    3. Convert fresh results to Nodes, load cached Nodes
    4. Combine cached + fresh Nodes (total = group_size)
    5. Insert fresh Nodes into tree_store
    6. Compute tree advantages (if advantage_mode == TREE)
    7. Mark all nodes as trained
    8. Save tree checkpoint (if cache_mode == CROSS_TRAINING)
    9. Return batched tensor dict
    """
```

To:
```python
    """GroupedRolloutWorkflow with tree-search cache reuse, tree ops, and checkpoint.

    Wraps the base OpenAIProxyWorkflow and overrides arun_episode to:
    1. Check cache: how many untrained episodes exist for this query?
    2. Generate only the needed fresh episodes (group_size - cached_count)
    3. Convert fresh results to Nodes, load cached episode Nodes
    4. Combine cached + fresh Nodes (total = group_size episodes)
    5. Insert fresh Nodes into tree_store
    6. Compute tree advantages per-episode (if advantage_mode == TREE)
    7. Mark all nodes as trained
    8. Save tree checkpoint (if cache_mode == CROSS_TRAINING)
    9. Return batched tensor dict
    """
```

Also update the module docstring (line 8):

From:
```python
- Does per-query cache lookup to determine how many fresh episodes are needed
```

To:
```python
- Does per-query cache lookup to determine how many fresh episodes are needed (episode-level counting)
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py
git commit -m "docs: update docstrings for episode-aware caching"
```

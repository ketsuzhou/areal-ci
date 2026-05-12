# Add turn_idx to Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `turn_idx` field (1-based per episode) to the Node dataclass so
downstream training and MCTS code can determine turn ordering within episodes.

**Architecture:** Add `turn_idx: int = 0` to Node, set it at creation time in
proxy_workflow and grouped_workflow, propagate through `_node_to_tensor_dict`,
checkpoint serialization, and both trainer.py call sites.

**Tech Stack:** Python 3.12+, pytest

______________________________________________________________________

### Task 1: Add turn_idx field to Node dataclass

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:23-58`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestTurnIdx:
    """Feature: turn_idx field on Node for per-episode turn ordering."""

    def test_node_has_turn_idx_default_zero(self):
        node = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 1, 1],
            logprobs=[0.0, -0.5, -0.3],
            versions=[-1, 0, 0],
            outcome_reward=1.0,
        )
        assert node.turn_idx == 0

    def test_node_turn_idx_can_be_set(self):
        node = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 1, 1],
            logprobs=[0.0, -0.5, -0.3],
            versions=[-1, 0, 0],
            outcome_reward=1.0,
            turn_idx=3,
        )
        assert node.turn_idx == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdx -v` Expected: FAIL —
`Node.__init__()` got an unexpected keyword argument `turn_idx`

- [ ] **Step 3: Add turn_idx field to Node**

In `customized_areal/tree_search/mcts_tree_store.py`, add the field after `episode_id`
(line 44):

```python
    episode_id: str = ""  # groups turns into a trajectory path
    turn_idx: int = 0  # 1-based turn position within episode
    query_id: str = ""  # dataset query identifier
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdx -v` Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_treesearch_bugfixes.py
git commit -m "feat: add turn_idx field to Node dataclass (1-based per episode)"
```

______________________________________________________________________

### Task 2: Set turn_idx in proxy_workflow.\_interactions_to_nodes

**Files:**

- Modify: `customized_areal/tree_search/proxy_workflow.py:65-149`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestTurnIdxInInteractionsToNodes:
    """proxy_workflow._interactions_to_nodes sets turn_idx 1-based."""

    def test_interactions_to_nodes_sets_turn_idx(self):
        from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow
        from areal.experimental.openai.types import InteractionWithTokenLogpReward
        from unittest.mock import MagicMock

        wf = QueryIDProxyWorkflow.__new__(QueryIDProxyWorkflow)

        # Build two mock interactions
        def make_interaction():
            inter = MagicMock(spec=InteractionWithTokenLogpReward)
            inter.chat_template_type = "individual"
            inter.parent = None
            inter.reward = 1.0
            resp = MagicMock()
            resp.input_tokens = [1, 2]
            resp.output_tokens = [3, 4]
            resp.input_ids = [1, 2]
            resp.output_ids = [3, 4]
            resp.input_len = 2
            resp.output_len = 2
            resp.output_logprobs = [-0.5, -0.3]
            resp.output_versions = [0, 0]
            resp.output_top_logprobs = None
            inter.model_response = resp
            return inter

        interactions = {"turn_a": make_interaction(), "turn_b": make_interaction()}
        nodes = wf._interactions_to_nodes(interactions)

        assert len(nodes) == 2
        assert nodes[0].turn_idx == 1
        assert nodes[1].turn_idx == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxInInteractionsToNodes -v`
Expected: FAIL — `nodes[0].turn_idx == 0` (not 1)

- [ ] **Step 3: Implement turn_idx in \_interactions_to_nodes**

In `customized_areal/tree_search/proxy_workflow.py`, change line 74 from:

```python
        for interaction_id, interaction in interactions.items():
```

to:

```python
        for turn_idx, (interaction_id, interaction) in enumerate(interactions.items(), start=1):
```

And add `turn_idx=turn_idx,` to the Node constructor (after `node_id=0,` at line 139):

```python
            node = Node(
                input_ids=seq_tokens,
                loss_mask=loss_mask,
                logprobs=logprobs,
                versions=versions,
                outcome_reward=outcome_reward,
                node_id=0,
                turn_idx=turn_idx,
                episode_id="",
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxInInteractionsToNodes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/proxy_workflow.py tests/test_treesearch_bugfixes.py
git commit -m "feat: set turn_idx in proxy_workflow._interactions_to_nodes"
```

______________________________________________________________________

### Task 3: Set turn_idx in grouped_workflow

**Files:**

- Modify: `customized_areal/tree_search/grouped_workflow.py:50-57`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestTurnIdxInGroupedWorkflow:
    """grouped_workflow sets turn_idx 1-based per episode."""

    def test_grouped_workflow_sets_turn_idx(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        # Create nodes without turn_idx set
        node_a = _make_node()
        node_b = _make_node()
        node_c = _make_node()
        node_d = _make_node()

        inner = MagicMock()
        inner.arun_episode = AsyncMock(
            side_effect=[
                [node_a, node_b],
                [node_c, node_d],
            ]
        )

        wf = TreeSearchGroupedRolloutWorkflow(
            workflow=inner, group_size=2, logger=MagicMock()
        )
        result = asyncio.run(wf.arun_episode(MagicMock(), {"query_id": "q1"}))

        # Group 0: nodes[0] and nodes[1] share episode, turn_idx 1 and 2
        assert result[0].turn_idx == 1
        assert result[1].turn_idx == 2
        # Group 1: nodes[2] and nodes[3] share different episode, turn_idx 1 and 2
        assert result[2].turn_idx == 1
        assert result[3].turn_idx == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxInGroupedWorkflow -v`
Expected: FAIL — `result[0].turn_idx == 0` (not 1)

- [ ] **Step 3: Implement turn_idx in grouped_workflow**

In `customized_areal/tree_search/grouped_workflow.py`, change lines 54-57 from:

```python
                for node in result:
                    node.episode_id = episode_id
                    node.query_id = query_id
```

to:

```python
                for turn_idx, node in enumerate(result, start=1):
                    node.episode_id = episode_id
                    node.query_id = query_id
                    node.turn_idx = turn_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxInGroupedWorkflow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/grouped_workflow.py tests/test_treesearch_bugfixes.py
git commit -m "feat: set turn_idx in grouped_workflow"
```

______________________________________________________________________

### Task 4: Fix \_node_to_tensor_dict to use turn_idx

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:109-163`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestTurnIdxInTensorDict:
    """_node_to_tensor_dict uses node.turn_idx and num_turns_in_episode."""

    def test_tensor_dict_uses_turn_idx(self):
        from customized_areal.tree_search.mcts_tree_store import _node_to_tensor_dict

        node = _make_node()
        node.turn_idx = 2
        traj = _node_to_tensor_dict(node, "q1", 1, num_turns_in_episode=3)
        assert traj["_turn_idx_in_episode"] == 2
        assert traj["_num_turns_in_episode"] == 3

    def test_tensor_dict_defaults(self):
        from customized_areal.tree_search.mcts_tree_store import _node_to_tensor_dict

        node = _make_node()
        # turn_idx=0 (default), num_turns_in_episode defaults to 1
        traj = _node_to_tensor_dict(node, "q1", 1)
        assert traj["_turn_idx_in_episode"] == 0
        assert traj["_num_turns_in_episode"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxInTensorDict -v`
Expected: FAIL —
`_node_to_tensor_dict() got an unexpected keyword argument 'num_turns_in_episode'`

- [ ] **Step 3: Update \_node_to_tensor_dict signature and body**

In `customized_areal/tree_search/mcts_tree_store.py`, change the function signature
(line 109) from:

```python
def _node_to_tensor_dict(node: Node, query_id: str, node_id: int) -> dict[str, Any]:
```

to:

```python
def _node_to_tensor_dict(node: Node, query_id: str, node_id: int, num_turns_in_episode: int = 1) -> dict[str, Any]:
```

And change lines 161-162 from:

```python
    traj["_turn_idx_in_episode"] = 0
    traj["_num_turns_in_episode"] = 1
```

to:

```python
    traj["_turn_idx_in_episode"] = node.turn_idx
    traj["_num_turns_in_episode"] = num_turns_in_episode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxInTensorDict -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_treesearch_bugfixes.py
git commit -m "feat: _node_to_tensor_dict uses node.turn_idx and num_turns_in_episode param"
```

______________________________________________________________________

### Task 5: Update trainer.py call sites

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:113-124`

- Modify: `customized_areal/tree_search/trainer.py:364-368`

- [ ] **Step 1: Update CacheAwarePPOTrainer.load_cached_trajectories**

In `customized_areal/tree_search/trainer.py`, replace lines 113-124:

```python
        all_trajs = []
        for item in cached_prompts:
            query_id = item["query_id"]
            if not query_id:
                continue
            nodes = self.tree_store.load_trajectories(query_id, self.n_samples)
            for node in nodes:
                traj_dict = _node_to_tensor_dict(
                    node, query_id, getattr(node, "node_id", 0)
                )
                all_trajs.append(traj_dict)
        return all_trajs
```

with:

```python
        all_trajs = []
        for item in cached_prompts:
            query_id = item["query_id"]
            if not query_id:
                continue
            nodes = self.tree_store.load_trajectories(query_id, self.n_samples)
            episode_sizes: dict[str, int] = {}
            for node in nodes:
                episode_sizes[node.episode_id] = episode_sizes.get(node.episode_id, 0) + 1
            for node in nodes:
                traj_dict = _node_to_tensor_dict(
                    node,
                    query_id,
                    node.node_id,
                    num_turns_in_episode=episode_sizes.get(node.episode_id, 1),
                )
                all_trajs.append(traj_dict)
        return all_trajs
```

- [ ] **Step 2: Update TreeBackupPPOTrainer.compute_loss**

In `customized_areal/tree_search/trainer.py`, replace lines 364-368:

```python
        # Convert Nodes to tensor dicts for the downstream PPO pipeline.
        converted_trajs: list[dict[str, Any]] = []
        for node in nodes:
            query_id = node.query_id
            node_id = node.node_id
            converted_trajs.append(_node_to_tensor_dict(node, query_id, node_id))
```

with:

```python
        # Convert Nodes to tensor dicts for the downstream PPO pipeline.
        converted_trajs: list[dict[str, Any]] = []
        episode_sizes: dict[str, int] = {}
        for node in nodes:
            episode_sizes[node.episode_id] = episode_sizes.get(node.episode_id, 0) + 1
        for node in nodes:
            query_id = node.query_id
            node_id = node.node_id
            converted_trajs.append(
                _node_to_tensor_dict(
                    node, query_id, node_id,
                    num_turns_in_episode=episode_sizes.get(node.episode_id, 1),
                )
            )
```

- [ ] **Step 3: Run existing tree search tests to verify no regressions**

Run:
`uv run pytest tests/test_treesearch_bugfixes.py tests/test_treesearch_patches.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat: pass num_turns_in_episode to _node_to_tensor_dict in trainer call sites"
```

______________________________________________________________________

### Task 6: Add turn_idx to checkpoint serialization

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:126-164`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_treesearch_bugfixes.py`:

```python
class TestTurnIdxCheckpoint:
    """turn_idx survives checkpoint save/load."""

    def test_turn_idx_survives_save_load(self, tmp_path):
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore()
        node = _make_node()
        node.query_id = "q1"
        node.turn_idx = 3
        store.insert_batch([node])

        manager = TreeCheckpointManager(str(tmp_path))
        manager.save(store)

        loaded = manager.load()
        loaded_nodes = loaded.trajectories.get("q1", [])
        assert len(loaded_nodes) == 1
        assert loaded_nodes[0].turn_idx == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxCheckpoint -v`
Expected: FAIL — `loaded_nodes[0].turn_idx == 0` (not 3, because deserialization doesn't
read it)

- [ ] **Step 3: Add turn_idx to \_serialize_record**

In `customized_areal/tree_search/checkpoint.py`, in `_serialize_record` (after line 136
`"episode_id": node.episode_id,`), add:

```python
            "turn_idx": node.turn_idx,
```

- [ ] **Step 4: Add turn_idx to \_deserialize_record**

In `customized_areal/tree_search/checkpoint.py`, in `_deserialize_record` (after line
158 `episode_id=data.get("episode_id", ""),`), add:

```python
            turn_idx=data.get("turn_idx", 0),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_treesearch_bugfixes.py::TestTurnIdxCheckpoint -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_treesearch_bugfixes.py
git commit -m "feat: serialize/deserialize turn_idx in TreeCheckpointManager"
```

______________________________________________________________________

### Task 7: Run full test suite and verify

- [ ] **Step 1: Run all tree search tests**

Run:
`uv run pytest tests/test_treesearch_bugfixes.py tests/test_treesearch_patches.py tests/test_tree_training.py -v`
Expected: All PASS

- [ ] **Step 2: Run pre-commit**

Run: `pre-commit run --all-files` Expected: PASS (or only pre-existing warnings
unrelated to changed files)

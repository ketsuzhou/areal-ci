import itertools

import torch

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

_node_id_counter = itertools.count(1)


def _make_node(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    query_id: str = "q1",
    node_id: str | None = None,
) -> Node:
    if node_id is None:
        node_id = f"n{next(_node_id_counter)}"
    return Node(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=[0.0] * len(input_ids),
        versions=[-1] * len(input_ids),
        outcome_reward=reward,
        query_id=query_id,
        node_id=node_id,
    )


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        node = _make_node([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0)
        store.insert_batch([node])
        computer.compute([node])
        assert node.advantages is not None
        assert node.returns is not None
        # Prompt positions (loss_mask=0) get zero advantage/return
        assert torch.allclose(node.advantages[:2], torch.zeros(2))
        assert torch.allclose(node.returns[:2], torch.zeros(2))
        # Single-sample: zero-mean normalization produces 0.0 for both
        assert torch.allclose(node.advantages[2:], torch.zeros(3))
        assert torch.allclose(node.returns[2:], torch.zeros(3))

    def test_compute_returns_from_outcome_reward(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        node = _make_node([1, 10, 3], [0, 0, 1], reward=1.0)
        store.insert_batch([node])
        computer.compute([node])
        # Single-sample: advantage = 0 (zero-mean), returns = 0 (zero-mean)
        assert torch.allclose(node.advantages, torch.zeros(3))
        assert torch.allclose(node.returns, torch.zeros(3))

    def test_compute_multi_turn_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        node = _make_node(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
        )
        store.insert_batch([node])
        computer.compute([node])
        # Single-sample: zero-mean normalization → advantage = 0, returns = 0
        assert torch.allclose(node.advantages, torch.zeros(8))
        assert torch.allclose(node.returns, torch.zeros(8))

    def test_compute_two_trajectories_different_queries(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        t1 = _make_node([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        t2 = _make_node([5, 6, 7, 8], [0, 0, 1, 1], reward=0.5, query_id="q2")
        store.insert_batch([t1, t2])
        computer.compute([t1, t2])
        assert t1.advantages is not None
        assert t2.advantages is not None
        # Different queries → single-sample per query → zero advantages/returns
        assert torch.allclose(t1.advantages, torch.zeros(5))
        assert torch.allclose(t1.returns, torch.zeros(5))
        assert torch.allclose(t2.advantages, torch.zeros(4))
        assert torch.allclose(t2.returns, torch.zeros(4))

    def test_compute_returns_normalized_per_query(self):
        """Returns use per-query GRPO normalization on outcome_reward."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        t1 = _make_node([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_node([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1")
        store.insert_batch([t1, t2])
        computer.compute([t1, t2])
        # Two samples with different rewards → non-zero normalized returns
        assert not torch.allclose(t1.returns, torch.zeros(3))
        assert not torch.allclose(t2.returns, torch.zeros(3))
        # Prompt positions are zero
        assert t1.returns[0].item() == 0.0
        assert t1.returns[1].item() == 0.0
        # Response positions: normalized values (zero-mean property)
        assert abs(t1.returns[2].item() + t2.returns[2].item()) < 1e-6
        # Q-value advantages are independent from returns
        # (Q-values == outcome_reward for single-insert nodes, so they
        # happen to match here, but they come from different normalization paths)


class TestTreeAdvantageComputerEpisodeLevel:
    def test_episode_level_normalization(self):
        """GRPO normalization operates across episodes, not individual nodes."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        # Episode A: 2 turns, reward 1.0
        n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[-1, -1, 0],
            outcome_reward=1.0,
            query_id="q1",
            node_id="ep_a_1",
            episode_id="ep_a",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[-1, -1, 0],
            outcome_reward=1.0,
            query_id="q1",
            node_id="ep_a_2",
            episode_id="ep_a",
        )
        # Episode B: 1 turn, reward 0.0
        n3 = Node(
            input_ids=[7, 8, 9],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3],
            versions=[-1, -1, 0],
            outcome_reward=0.0,
            query_id="q1",
            node_id="ep_b_1",
            episode_id="ep_b",
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
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[-1, -1, 0],
            outcome_reward=1.0,
            query_id="q1",
            node_id="ep_a_1",
            episode_id="ep_a",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[-1, -1, 0],
            outcome_reward=-1.0,
            query_id="q1",
            node_id="ep_b_1",
            episode_id="ep_b",
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
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[-1, -1, 0],
            outcome_reward=1.0,
            query_id="q1",
            node_id="ep_a_1",
            episode_id="ep_a",
        )
        store.insert_batch([n1])
        computer.compute([n1])
        assert torch.allclose(n1.advantages, torch.zeros(3))

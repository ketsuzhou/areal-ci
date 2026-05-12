"""Tests for tree search bug fixes."""

import itertools

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

_counter = itertools.count(1)


def _make_node(reward: float = 1.0) -> Node:
    """Create a minimal Node for testing."""
    return Node(
        node_id=str(next(_counter)),
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
        store.current_train_id = "test_run"
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
            node.query_id = "q1"
            store.insert_batch([node])

        computer = TreeAdvantageComputer(store)
        nodes = store.load_trajectories("q1", 3)
        computer.compute(nodes)

        for node in nodes:
            assert node.advantages is not None


class TestInsertBatchSkipDuplicates:
    """Bug #1: insert_batch should skip already-inserted nodes."""

    def test_insert_batch_skips_nodes_with_existing_node_id(self):
        store = MCTSTreeStore()
        node = _make_node()
        store.insert_batch([node])
        # First node gets ID 1 (_next_node_id starts at 1)
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


class TestQueryIdCheckpoint:
    """Bug #3: query_id lost on checkpoint deserialization."""

    def test_query_id_survives_save_load(self, tmp_path):
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore()
        node = _make_node()
        node.query_id = "test_query_123"
        store.insert_batch([node])
        assert node.query_id == "test_query_123"

        manager = TreeCheckpointManager(str(tmp_path))
        manager.save(store)

        loaded = manager.load()
        loaded_nodes = loaded.trajectories.get("test_query_123", [])
        assert len(loaded_nodes) == 1
        assert getattr(loaded_nodes[0], "query_id", None) == "test_query_123"


class TestBesselVariance:
    """Bug #2: GRPO normalization should use Bessel-corrected variance."""

    def test_uses_sample_variance_not_population(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        # Insert 4 nodes with known rewards
        for r in [1.0, 2.0, 3.0, 4.0]:
            node = _make_node(reward=r)
            node.query_id = "q1"
            store.insert_batch([node])

        computer = TreeAdvantageComputer(store)
        nodes = store.load_trajectories("q1", 4)
        computer.compute(nodes)

        # Population variance of [1,2,3,4] = 1.25, std = 1.118
        # (1.0 - 2.5) / (1.118 + eps) ≈ -1.342
        first_adv = nodes[0].advantages
        response_adv = first_adv[first_adv != 0]
        assert response_adv.numel() > 0
        assert abs(response_adv[0].item() - (-1.342)) < 0.05, (
            f"Expected population variance normalization (~-1.342), got {response_adv[0].item()}"
        )




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


class TestTurnIdxInInteractionsToNodes:
    """tree_search_grouped_workflow.interactions_dict_to_nodes sets turn_idx 1-based."""

    def test_interactions_to_nodes_sets_turn_idx(self):
        from unittest.mock import MagicMock

        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

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
        nodes = interactions_dict_to_nodes(interactions)

        assert len(nodes) == 2
        assert nodes[0].turn_idx == 1
        assert nodes[1].turn_idx == 2


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

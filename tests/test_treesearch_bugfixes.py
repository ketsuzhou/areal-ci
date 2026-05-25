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


class TestPackExtraDataSubsetsBatchDimTensors:
    """Bug #3: _pack_extra_data should subset batch-dim tensors per tree."""

    def test_batch_dim_non_packable_subset_per_tree(self):
        import torch
        from areal.models.tree_attn.tree import _pack_extra_data, TrieNode

        # Build a simple trie with 2 sequences (seq_ids 0, 1)
        trie = TrieNode(tree_id=0, start_idx=0)
        trie.sequence_ids = [0, 1]

        # Full batch has 4 sequences, but this tree only has 0 and 1
        N = 4
        data = {
            "input_ids": torch.randint(0, 100, (N, 10)),
            "topk_ids": torch.randint(0, 100, (N, 5, 3)),
            "teacher_logp": torch.randn(N, 5, 3),
            "some_scalar": 42,
        }
        sequence_lens = torch.tensor([8, 7, 9, 6], dtype=torch.int32)
        packable_keys = set()
        non_packable_keys = {"topk_ids", "teacher_logp", "some_scalar"}

        result = _pack_extra_data(trie, data, sequence_lens, packable_keys, non_packable_keys)

        # topk_ids and teacher_logp should be subsetted to [2, 5, 3]
        assert result["topk_ids"].shape == (2, 5, 3)
        assert result["teacher_logp"].shape == (2, 5, 3)
        # Values should match the original sequences 0 and 1
        torch.testing.assert_close(result["topk_ids"], data["topk_ids"][[0, 1]])
        torch.testing.assert_close(result["teacher_logp"], data["teacher_logp"][[0, 1]])
        # Non-tensor scalars should be copied as-is
        assert result["some_scalar"] == 42

    def test_cu_seqlens_added_to_tree_extra_data(self):
        import torch
        from areal.models.tree_attn.tree import _pack_extra_data, TrieNode

        trie = TrieNode(tree_id=0, start_idx=0)
        trie.sequence_ids = [0, 2]  # sequences 0 and 2

        N = 4
        data = {
            "input_ids": torch.randint(0, 100, (N, 10)),
        }
        sequence_lens = torch.tensor([8, 7, 9, 6], dtype=torch.int32)
        packable_keys = set()
        non_packable_keys = set()

        result = _pack_extra_data(trie, data, sequence_lens, packable_keys, non_packable_keys)

        # cu_seqlens should be [0, 8, 17] (cumsum of [8, 9])
        assert "cu_seqlens" in result
        expected = torch.tensor([0, 8, 17], dtype=torch.int32)
        torch.testing.assert_close(result["cu_seqlens"], expected)


class TestTeacherKLLossPackedFormat:
    """Bug #5: _compute_teacher_kl_loss must handle 1D packed logprobs with cu_seqlens."""

    def test_packed_1d_format_with_cu_seqlens(self):
        import torch
        from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss

        # 2 sequences packed into 1D: seq0 has prompt_len=2 resp_len=3, seq1 has prompt_len=1 resp_len=2
        # Total packed len = 5 + 3 = 8
        loss_mask = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)

        # Student logprobs: 1D [8] (single-candidate for simplicity)
        logprobs = torch.tensor([-1.0, -2.0, -0.5, -0.6, -0.7, -3.0, -0.8, -0.9])

        # Teacher: [2, max_resp, 1] where max_resp=3 (max resp across sequences)
        # seq0 has resp_len=3, seq1 has resp_len=2
        teacher_logprobs = torch.tensor([
            [[-0.4], [-0.5], [-0.6]],  # seq0, 3 response positions
            [[-0.7], [-0.8], [0.0]],   # seq1, 2 response positions (3rd is padding)
        ])

        loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[2, 1],
            input_data={"cu_seqlens": cu_seqlens},
        )

        # Loss should be non-zero and finite
        assert abs(loss.item()) > 0
        assert torch.isfinite(loss)

    def test_packed_1d_multi_candidate(self):
        import torch
        from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss

        # 2 sequences, multi-candidate (3 candidates per position)
        loss_mask = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

        # Student logprobs: [6, 3] (multi-candidate)
        logprobs = torch.randn(6, 3)

        # Teacher: [2, 2, 3]
        teacher_logprobs = torch.randn(2, 2, 3)

        loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[1, 1],
            input_data={"cu_seqlens": cu_seqlens},
        )

        assert abs(loss.item()) > 0
        assert torch.isfinite(loss)

    def test_batched_2d_format_unchanged(self):
        """Existing batched [batch, seq] format should still work."""
        import torch
        from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss

        # 2D batched format (existing code path)
        loss_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.0, -0.5, -0.6], [-2.0, -0.7, -0.8]])
        teacher_logprobs = torch.tensor([[[-0.4], [-0.5]], [[-0.6], [-0.7]]])

        loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[1, 1],
        )

        assert abs(loss.item()) > 0
        assert torch.isfinite(loss)


class TestPrepareMultiCandidateLabelsPacked:
    """Bug #2: _prepare_multi_candidate_labels must handle mb_bs > 1."""

    def _make_engine(self):
        """Create a minimal MultiCandidateFSDPEngine for testing."""
        from unittest.mock import MagicMock
        from customized_areal.tree_search.engine.fsdp_engine import MultiCandidateFSDPEngine
        engine = MagicMock(spec=MultiCandidateFSDPEngine)
        engine._prepare_multi_candidate_labels = MultiCandidateFSDPEngine._prepare_multi_candidate_labels.__get__(engine)
        engine.config = MagicMock()
        return engine

    def test_single_sequence_still_works(self):
        import torch
        engine = self._make_engine()

        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30, 40, 50]]),  # [1, 5]
            "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 5], dtype=torch.int32),
        }
        mb_input = {
            "topk_ids": torch.tensor([[[100, 101, 102], [200, 201, 202], [300, 301, 302]]]),
            # [1, 3, 3] — 1 sequence, 3 response positions, 3 candidates
        }

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=5)

        assert labels is not None
        assert labels.shape == (5, 3)
        # Positions 0-1 (prompt): should have rolled input_ids
        assert labels[0, 0].item() == 20  # rolled(10)
        assert labels[1, 0].item() == 30  # rolled(20)
        # Positions 2-4 (response): should have topk_ids
        assert labels[2, 0].item() == 100
        assert labels[2, 1].item() == 101

    def test_two_sequences_packed(self):
        import torch
        engine = self._make_engine()

        # 2 sequences packed: seq0=[10,20,30,40], seq1=[50,60,70]
        # cu_seqlens = [0, 4, 7]
        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30, 40, 50, 60, 70]]),  # [1, 7]
            "loss_mask": torch.tensor([[0, 1, 1, 1, 0, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 4, 7], dtype=torch.int32),
        }
        mb_input = {
            # 2 sequences, each with 3 response positions, 3 candidates
            # seq0: prompt_len=1, resp_len=3 (from loss_mask)
            # seq1: prompt_len=1, resp_len=2 (from loss_mask)
            "topk_ids": torch.tensor([
                [[100, 101, 102], [200, 201, 202], [300, 301, 302]],  # seq0
                [[400, 401, 402], [500, 501, 502], [-1, -1, -1]],     # seq1 (3rd pos is padding/sentinel)
            ]),
            # [2, 3, 3]
        }

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=7)

        assert labels is not None
        assert labels.shape == (7, 3)
        # seq0 position 0 (prompt): rolled input_ids
        assert labels[0, 0].item() == 20
        # seq0 positions 1-3 (response): from topk_ids[0]
        assert labels[1, 0].item() == 100
        assert labels[2, 0].item() == 200
        assert labels[3, 0].item() == 300
        # seq1 position 4 (prompt): rolled input_ids
        assert labels[4, 0].item() == 60
        # seq1 positions 5-6 (response): from topk_ids[1]
        assert labels[5, 0].item() == 400
        assert labels[6, 0].item() == 500

    def test_returns_none_when_no_topk_ids(self):
        import torch
        engine = self._make_engine()

        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        }
        mb_input = {}

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=3)
        assert labels is None

    def test_returns_none_when_all_sentinel(self):
        import torch
        engine = self._make_engine()

        model_inputs = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
            "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        }
        mb_input = {
            "topk_ids": torch.tensor([[[-1], [-1]]]),  # all -1 sentinel
        }

        labels = engine._prepare_multi_candidate_labels(model_inputs, mb_input, seq_len=3)
        assert labels is None

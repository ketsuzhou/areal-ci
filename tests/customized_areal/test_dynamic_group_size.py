"""Tests for dynamic group_size feature."""

import pytest

from customized_areal.tree_search.config import TreeBackupConfig


class TestTreeBackupConfigDynamicFields:
    def test_defaults(self):
        cfg = TreeBackupConfig()
        assert cfg.dynamic_group_size is False
        assert cfg.initial_group_size == 4
        assert cfg.max_group_size == 64
        assert cfg.uncertainty_threshold == pytest.approx(0.05)
        assert cfg.reward_type == "binary"

    def test_custom_values(self):
        cfg = TreeBackupConfig(
            dynamic_group_size=True,
            initial_group_size=8,
            max_group_size=128,
            uncertainty_threshold=0.1,
            reward_type="continuous",
        )
        assert cfg.dynamic_group_size is True
        assert cfg.initial_group_size == 8
        assert cfg.max_group_size == 128
        assert cfg.uncertainty_threshold == pytest.approx(0.1)
        assert cfg.reward_type == "continuous"

    def test_initial_group_size_must_be_positive(self):
        with pytest.raises(ValueError, match="initial_group_size"):
            TreeBackupConfig(initial_group_size=0)

    def test_max_group_size_must_be_at_least_initial(self):
        with pytest.raises(ValueError, match="max_group_size"):
            TreeBackupConfig(initial_group_size=10, max_group_size=5)

    def test_uncertainty_threshold_must_be_non_negative(self):
        with pytest.raises(ValueError, match="uncertainty_threshold"):
            TreeBackupConfig(uncertainty_threshold=-0.01)

    def test_reward_type_must_be_valid(self):
        with pytest.raises(ValueError, match="reward_type"):
            TreeBackupConfig(reward_type="unknown")


class TestComputeQueryUncertainty:
    def test_binary_all_success(self):
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        # 3 episodes all reward=1, 2 steps each
        # Beta(4,1): var = 4*1 / (25*6) = 4/150
        u = compute_query_uncertainty([1.0, 1.0, 1.0], [2, 2, 2], "binary")
        expected_var = (4 * 1) / (25 * 6)
        assert u == pytest.approx(expected_var * 2.0)

    def test_binary_mixed(self):
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        # 4 episodes, 2 successes, 2 steps each
        # Beta(3,3): var = 9 / (36*7) = 9/252
        u = compute_query_uncertainty([1.0, 0.0, 1.0, 0.0], [2, 2, 2, 2], "binary")
        expected_var = (3 * 3) / (36 * 7)
        assert u == pytest.approx(expected_var * 2.0)

    def test_continuous_n_ge_2(self):
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        # 4 episodes rewards [0.5, 1.5, 0.5, 1.5], 3 steps each
        u = compute_query_uncertainty([0.5, 1.5, 0.5, 1.5], [3, 3, 3, 3], "continuous")
        # Just verify it's finite and positive
        assert u > 0
        assert u != float("inf")

    def test_continuous_n_1(self):
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        # n=1: NIG posterior should be well-defined under weak prior
        u = compute_query_uncertainty([0.5], [3], "continuous")
        assert u > 0
        assert u != float("inf")

    def test_zero_episodes(self):
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        assert compute_query_uncertainty([], [], "binary") == float("inf")

    def test_more_steps_higher_uncertainty(self):
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        u_short = compute_query_uncertainty([1.0, 0.0], [1, 1], "binary")
        u_long = compute_query_uncertainty([1.0, 0.0], [10, 10], "binary")
        assert u_long > u_short


class TestShouldDiscardQuery:
    def test_all_zero(self):
        from customized_areal.tree_search.core.uncertainty import should_discard_query

        assert should_discard_query([0.0, 0.0, 0.0]) is True

    def test_all_one(self):
        from customized_areal.tree_search.core.uncertainty import should_discard_query

        assert should_discard_query([1.0, 1.0, 1.0]) is True

    def test_mixed(self):
        from customized_areal.tree_search.core.uncertainty import should_discard_query

        assert should_discard_query([0.0, 1.0, 0.0]) is False

    def test_single_episode_kept(self):
        from customized_areal.tree_search.core.uncertainty import should_discard_query

        assert should_discard_query([0.5]) is False

    def test_empty_discarded(self):
        from customized_areal.tree_search.core.uncertainty import should_discard_query

        assert should_discard_query([]) is True


class TestWorkflowConstructorDynamicFields:
    def test_explicit_dynamic_config(self):
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import MagicMock

        base = MagicMock()
        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=16,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=8,
            max_group_size=32,
            uncertainty_threshold=0.1,
            reward_type="continuous",
        )
        assert wf.dynamic_group_size is True
        assert wf.initial_group_size == 8
        assert wf.max_group_size == 32
        assert wf.uncertainty_threshold == pytest.approx(0.1)
        assert wf.reward_type == "continuous"
        assert wf.group_size == 8

    def test_fallback_initial_from_group_size(self):
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import MagicMock

        base = MagicMock()
        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=16,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
        )
        assert wf.initial_group_size == 16
        assert wf.group_size == 16

    def test_no_dynamic_backward_compat(self):
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import MagicMock

        base = MagicMock()
        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=16,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
        )
        assert wf.dynamic_group_size is False
        assert wf.group_size == 16


class TestZeroVarianceDiscard:
    @pytest.mark.asyncio
    async def test_discard_identical_rewards(self):
        """2+ episodes with identical rewards -> return None."""
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import AsyncMock, MagicMock

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=2,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
        )

        def make_nodes(result, query_id, group_idx):
            nodes = []
            for i in range(2):
                node = Node(
                    input_ids=[1, 2, 3],
                    loss_mask=[0, 1, 1],
                    logprobs=[0.0, -0.5, -0.3],
                    versions=[-1, 0, 0],
                    outcome_reward=1.0,
                )
                node.query_id = query_id
                node.episode_id = f"ep_{group_idx}"
                node.node_id = f"node_{group_idx}_{i}"
                node.turn_idx = i + 1
                nodes.append(node)
            return nodes

        wf._result_to_nodes = make_nodes
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_discard"})
        assert result is None

    @pytest.mark.asyncio
    async def test_keep_mixed_rewards(self):
        """Mixed rewards -> result is NOT None."""
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import AsyncMock, MagicMock

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=2,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
        )

        call_count = 0

        def make_nodes(result, query_id, group_idx):
            nonlocal call_count
            call_count += 1
            reward = 1.0 if call_count % 2 == 0 else 0.0
            nodes = []
            for i in range(2):
                node = Node(
                    input_ids=[1, 2, 3],
                    loss_mask=[0, 1, 1],
                    logprobs=[0.0, -0.5, -0.3],
                    versions=[-1, 0, 0],
                    outcome_reward=reward,
                )
                node.query_id = query_id
                node.episode_id = f"ep_{group_idx}"
                node.node_id = f"node_{group_idx}_{i}"
                node.turn_idx = i + 1
                nodes.append(node)
            return nodes

        wf._result_to_nodes = make_nodes
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_mixed"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_keep_single_episode(self):
        """Single episode is not discarded even if only one reward value."""
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import AsyncMock, MagicMock

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=1,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
        )

        def make_nodes(result, query_id, group_idx):
            node = Node(
                input_ids=[1, 2, 3],
                loss_mask=[0, 1, 1],
                logprobs=[0.0, -0.5, -0.3],
                versions=[-1, 0, 0],
                outcome_reward=1.0,
            )
            node.query_id = query_id
            node.episode_id = "ep_0"
            node.node_id = "node_0_0"
            node.turn_idx = 1
            return [node]

        wf._result_to_nodes = make_nodes
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_single"})
        assert result is not None


class TestDynamicSamplingLoop:
    @pytest.mark.asyncio
    async def test_stops_at_threshold(self):
        """Dynamic mode samples until uncertainty drops below threshold."""
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import AsyncMock, MagicMock

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=4,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=2,
            max_group_size=10,
            uncertainty_threshold=0.5,
            reward_type="binary",
        )

        episode_count = 0

        def make_nodes(result, query_id, group_idx):
            nonlocal episode_count
            episode_count += 1
            # Mixed rewards: as episodes grow, Beta posterior narrows
            reward = 1.0 if episode_count % 2 == 0 else 0.0
            nodes = []
            for i in range(2):
                node = Node(
                    input_ids=[1, 2, 3],
                    loss_mask=[0, 1, 1],
                    logprobs=[0.0, -0.5, -0.3],
                    versions=[-1, 0, 0],
                    outcome_reward=reward,
                )
                node.query_id = query_id
                node.episode_id = f"ep_{episode_count}"
                node.node_id = f"node_{episode_count}_{i}"
                node.turn_idx = i + 1
                nodes.append(node)
            return nodes

        wf._result_to_nodes = make_nodes
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_dyn1"})
        # Should have sampled at least initial_group_size
        assert episode_count >= 2
        # Should have a result (mixed rewards = not discarded)
        assert result is not None

    @pytest.mark.asyncio
    async def test_respects_max_group_size(self):
        """Dynamic mode caps at max_group_size even if uncertainty is high."""
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import AsyncMock, MagicMock

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=4,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=2,
            max_group_size=4,
            uncertainty_threshold=1e-12,
            reward_type="binary",
        )

        episode_count = 0

        def make_nodes(result, query_id, group_idx):
            nonlocal episode_count
            episode_count += 1
            # Always alternating → uncertainty stays high
            reward = 1.0 if episode_count % 2 == 0 else 0.0
            nodes = []
            for i in range(3):
                node = Node(
                    input_ids=[1, 2, 3],
                    loss_mask=[0, 1, 1],
                    logprobs=[0.0, -0.5, -0.3],
                    versions=[-1, 0, 0],
                    outcome_reward=reward,
                )
                node.query_id = query_id
                node.episode_id = f"ep_{episode_count}"
                node.node_id = f"node_{episode_count}_{i}"
                node.turn_idx = i + 1
                nodes.append(node)
            return nodes

        wf._result_to_nodes = make_nodes
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_dyn2"})
        # Should cap at max_group_size
        assert episode_count <= 4

    @pytest.mark.asyncio
    async def test_tolerates_failed_extra_samples(self):
        """Dynamic loop continues if an extra episode fails."""
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node
        from customized_areal.tree_search.config import AdvantageMode, LossMode, CacheMode
        from unittest.mock import AsyncMock, MagicMock

        base = MagicMock()
        # First call succeeds, second call fails (returns None)
        base.arun_episode = AsyncMock(side_effect=[{}, None, {}, None, {}])

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=4,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=2,
            max_group_size=6,
            uncertainty_threshold=1e-12,
            reward_type="binary",
        )

        episode_count = 0

        def make_nodes(result, query_id, group_idx):
            nonlocal episode_count
            episode_count += 1
            reward = 1.0 if episode_count % 2 == 0 else 0.0
            nodes = []
            for i in range(2):
                node = Node(
                    input_ids=[1, 2, 3],
                    loss_mask=[0, 1, 1],
                    logprobs=[0.0, -0.5, -0.3],
                    versions=[-1, 0, 0],
                    outcome_reward=reward,
                )
                node.query_id = query_id
                node.episode_id = f"ep_{episode_count}"
                node.node_id = f"node_{episode_count}_{i}"
                node.turn_idx = i + 1
                nodes.append(node)
            return nodes

        wf._result_to_nodes = make_nodes
        # Should not crash even with failed episodes
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_dyn3"})
        # Result may be None if all discarded, or a valid dict
        # The key assertion: no crash
        assert result is None or isinstance(result, dict)

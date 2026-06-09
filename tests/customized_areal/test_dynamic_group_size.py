"""Tests for dynamic group_size feature."""

from types import SimpleNamespace
from unittest.mock import MagicMock

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
        from unittest.mock import MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

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
        from unittest.mock import MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

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
        from unittest.mock import MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

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

    def test_dynamic_constructor_rejects_invalid_bounds(self):
        from unittest.mock import MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        with pytest.raises(ValueError, match="max_group_size"):
            TreeSearchGroupedRolloutWorkflow(
                MagicMock(),
                group_size=4,
                checkpoint_dir="/tmp/test_ckpt",
                advantage_mode=AdvantageMode.TREE,
                loss_mode=LossMode.GRPO,
                cache_mode=CacheMode.OFF,
                dynamic_group_size=True,
                initial_group_size=8,
                max_group_size=4,
            )

    def test_dynamic_constructor_rejects_invalid_reward_type(self):
        from unittest.mock import MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        with pytest.raises(ValueError, match="reward_type"):
            TreeSearchGroupedRolloutWorkflow(
                MagicMock(),
                group_size=4,
                checkpoint_dir="/tmp/test_ckpt",
                advantage_mode=AdvantageMode.TREE,
                loss_mode=LossMode.GRPO,
                cache_mode=CacheMode.OFF,
                dynamic_group_size=True,
                reward_type="unknown",
            )


class TestWorkflowFailureHandling:
    @pytest.mark.asyncio
    async def test_retry_episode_retries_raised_exception(self, monkeypatch):
        """Raised workflow exceptions are retried like None results."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        base = MagicMock()
        base.arun_episode = AsyncMock(side_effect=[RuntimeError("boom"), {"ok": True}])
        monkeypatch.setattr(
            "customized_areal.tree_search.core.customized_grouped_workflow.asyncio.sleep",
            AsyncMock(),
        )

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=1,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
        )

        result = await wf._retry_episode(MagicMock(), {}, group_idx=0, max_retries=2)
        assert result == {"ok": True}
        assert base.arun_episode.await_count == 2

    @pytest.mark.asyncio
    async def test_branch_cleanup_runs_when_branch_rollout_raises(self):
        """A failed branch rollout still marks the branch candidate consumed."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
            SampleSource,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

        wf = TreeSearchGroupedRolloutWorkflow(
            MagicMock(),
            group_size=1,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            sample_source=SampleSource.BRANCH,
        )
        candidate = Node(
            input_ids=[1, 2],
            loss_mask=[0, 1],
            logprobs=[0.0, -0.1],
            versions=[-1, 0],
            node_id="candidate",
            query_id="q_branch",
            task_id="task",
            need_branch=True,
            branch_sandbox_id="sandbox",
        )
        wf.tree_store.trajectories["q_branch"] = [candidate]
        wf._prepare_branch_task = AsyncMock(return_value="branch_task")
        wf._retry_episode = AsyncMock(side_effect=RuntimeError("branch failed"))
        wf._cleanup_branch = AsyncMock()

        with pytest.raises(RuntimeError, match="branch failed"):
            await wf._run_fresh_episode(MagicMock(), {"query_id": "q_branch"}, 0, "q_branch")

        wf._cleanup_branch.assert_awaited_once_with(candidate)

    @pytest.mark.asyncio
    async def test_dynamic_distill_mode_does_not_generate_without_cache(self):
        """Dynamic DISTILL mode matches fixed mode and consumes cache only."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})
        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=2,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.DISTILL,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=2,
            max_group_size=4,
        )

        result = await wf.arun_episode(MagicMock(), {"query_id": "q_distill_empty"})
        assert result is None
        base.arun_episode.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_conversion_failure_does_not_mark_nodes_trained(self, monkeypatch):
        """Cache entries stay untrained when tensor batch conversion fails."""
        from unittest.mock import AsyncMock, MagicMock

        import customized_areal.tree_search.core.customized_grouped_workflow as workflow_mod
        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
            node = Node(
                input_ids=[1, 2, 3],
                loss_mask=[0, 1, 1],
                logprobs=[0.0, -0.5, -0.3],
                versions=[-1, 0, 0],
                outcome_reward=float(group_idx),
                node_id=f"node_{group_idx}",
                episode_id=f"ep_{group_idx}",
                query_id=query_id,
                turn_idx=1,
            )
            return [node]

        wf._result_to_nodes = make_nodes

        def fail_convert(*args, **kwargs):
            raise RuntimeError("convert failed")

        monkeypatch.setattr(workflow_mod, "_nodes_to_batched_tensor_dict", fail_convert)

        result = await wf.arun_episode(MagicMock(), {"query_id": "q_convert_fail"})
        assert result is None
        assert wf.tree_store.get_untrained_episode_count("q_convert_fail") == 2
        for node in wf.tree_store.trajectories["q_convert_fail"]:
            assert node.train_id == ""


class TestZeroVarianceDiscard:
    @pytest.mark.asyncio
    async def test_discard_identical_rewards(self):
        """2+ episodes with identical rewards -> return None."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
    async def test_discarded_cached_episodes_are_not_reused(self):
        """Discarded episodes leave train_id untouched but are excluded from cache reuse."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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

        cached_nodes = []
        for ep_idx in range(2):
            for turn_idx in range(2):
                node = Node(
                    input_ids=[1, 2, 3],
                    loss_mask=[0, 1, 1],
                    logprobs=[0.0, -0.5, -0.3],
                    versions=[-1, 0, 0],
                    outcome_reward=1.0,
                    node_id=f"cached_{ep_idx}_{turn_idx}",
                    episode_id=f"ep_{ep_idx}",
                    query_id="q_cached_discard",
                    turn_idx=turn_idx + 1,
                )
                cached_nodes.append(node)
        wf.tree_store.insert_batch(cached_nodes)

        result = await wf.arun_episode(MagicMock(), {"query_id": "q_cached_discard"})
        assert result is None
        assert wf.tree_store.get_untrained_episode_count("q_cached_discard") == 0
        for node in cached_nodes:
            assert node.train_id == ""
            assert wf.tree_store.is_discarded(node.node_id) is True

    @pytest.mark.asyncio
    async def test_keep_mixed_rewards(self):
        """Mixed rewards -> result is NOT None."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
        await wf.arun_episode(MagicMock(), {"query_id": "q_dyn2"})
        # Should cap at max_group_size
        assert episode_count <= 4

    @pytest.mark.asyncio
    async def test_tolerates_failed_extra_samples(self):
        """Dynamic loop continues if an extra episode fails."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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

    @pytest.mark.asyncio
    async def test_failed_additions_are_bounded(self):
        """Dynamic loop stops after a bounded number of failed additions."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        wf = TreeSearchGroupedRolloutWorkflow(
            MagicMock(),
            group_size=2,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            cache_mode=CacheMode.OFF,
            dynamic_group_size=True,
            initial_group_size=1,
            max_group_size=4,
            uncertainty_threshold=0.0,
            reward_type="binary",
        )
        wf._run_fresh_episode = AsyncMock(return_value=None)

        result = await wf.arun_episode(MagicMock(), {"query_id": "q_fail_cap"})
        assert result is None
        assert wf._run_fresh_episode.await_count == 1 + max(3, wf.max_group_size)


class TestPrecomputedAdvantages:
    def _make_minimal_trainer(self, rollout_batch):
        from areal.trainer.rl_trainer import PPOTrainer, _EmptyDataLoader

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.config = SimpleNamespace(
            total_train_epochs=1,
            total_train_steps=1,
            dynamic_bs=False,
            actor=SimpleNamespace(
                should_compute_prox_logp=lambda: False,
            ),
            memory_profiler=None,
            teacher=SimpleNamespace(rl_loss_weight=1.0, distill_loss_weight=1.0),
            rollout=SimpleNamespace(agent=None),
            gconfig=SimpleNamespace(n_samples=1),
        )
        trainer.recover_info = None
        trainer.train_dataloader = _EmptyDataLoader(batch_size=1, steps_per_epoch=1)
        trainer.actor = MagicMock()
        trainer.actor.prepare_batch.return_value = rollout_batch
        trainer.actor.get_device_stats.return_value = MagicMock(log=MagicMock())
        trainer.actor.step_lr_scheduler = MagicMock()
        trainer.actor.compute_advantages = MagicMock(return_value=rollout_batch)
        trainer.actor.ppo_update = MagicMock(side_effect=RuntimeError("stop_after_update"))
        trainer.rollout = MagicMock()
        trainer.critic = None
        trainer.ref = None
        trainer.teacher = None
        trainer._should_offload_rollout = False
        trainer._should_offload_actor = False
        trainer._should_offload_critic = False
        trainer._should_offload_ref = False
        trainer._should_offload_teacher = False
        trainer._requires_proxy_workflow = MagicMock(return_value=False)
        trainer.saver = MagicMock(maybe_wait_for_staging=MagicMock())
        return trainer

    def test_trainer_skips_compute_advantages_for_precomputed_rollout(self):
        from areal.trainer.rl_trainer import PPOTrainer

        rollout_batch = [
            {"input_ids": [1, 2, 3], "advantages": [0.5], "returns": [0.5]},
            {"input_ids": [4, 5, 6], "advantages": [0.3], "returns": [0.3]},
        ]
        trainer = self._make_minimal_trainer(rollout_batch)

        with pytest.raises(RuntimeError, match="stop_after_update"):
            PPOTrainer.train(trainer, workflow=object())

        trainer.actor.compute_advantages.assert_not_called()

    def test_trainer_computes_advantages_when_missing(self):
        from areal.trainer.rl_trainer import PPOTrainer

        rollout_batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5, 6]},
        ]
        trainer = self._make_minimal_trainer(rollout_batch)

        with pytest.raises(RuntimeError, match="stop_after_update"):
            PPOTrainer.train(trainer, workflow=object())

        trainer.actor.compute_advantages.assert_called_once_with(rollout_batch)


class TestEndToEndWorkflow:
    @pytest.mark.asyncio
    async def test_fixed_mode_unchanged(self):
        """Fixed mode produces a valid tensor dict output."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

        base = MagicMock()
        base.arun_episode = AsyncMock(return_value={})

        wf = TreeSearchGroupedRolloutWorkflow(
            base,
            group_size=2,
            checkpoint_dir="/tmp/test_ckpt",
            advantage_mode=AdvantageMode.GAE,
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
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_e2e_fixed"})
        assert result is not None
        assert "input_ids" in result

    @pytest.mark.asyncio
    async def test_dynamic_binary_rewards(self):
        """Dynamic mode with binary rewards produces valid output."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
            max_group_size=8,
            uncertainty_threshold=0.5,
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
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_e2e_binary"})
        assert result is not None
        assert "input_ids" in result

    @pytest.mark.asyncio
    async def test_dynamic_continuous_rewards(self):
        """Dynamic mode with continuous rewards produces valid output."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
            max_group_size=8,
            uncertainty_threshold=0.5,
            reward_type="continuous",
        )

        episode_count = 0

        def make_nodes(result, query_id, group_idx):
            nonlocal episode_count
            episode_count += 1
            reward = 0.3 + 0.4 * (episode_count % 3) / 2
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
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_e2e_cont"})
        assert result is not None
        assert "input_ids" in result

    @pytest.mark.asyncio
    async def test_discard_after_identical_rewards(self):
        """E2E: dynamic mode discards when all rewards are identical."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
            uncertainty_threshold=0.001,
            reward_type="binary",
        )

        def make_nodes(result, query_id, group_idx):
            # All rewards = 1.0
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
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_e2e_discard"})
        assert result is None

    @pytest.mark.asyncio
    async def test_tree_mode_advantages_in_output(self):
        """TREE-mode workflow output contains precomputed advantages."""
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.config import (
            AdvantageMode,
            CacheMode,
            LossMode,
        )
        from customized_areal.tree_search.core.customized_grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )
        from customized_areal.tree_search.core.tree_store import Node

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
        result = await wf.arun_episode(MagicMock(), {"query_id": "q_e2e_tree"})
        assert result is not None
        assert "advantages" in result

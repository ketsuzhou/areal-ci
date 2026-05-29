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

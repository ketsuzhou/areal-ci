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

# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM-judge step-level process-reward config fields."""

import pytest

from customized_areal.tree_search.config import Config


class TestJudgeConfigDefaults:
    def test_disabled_by_default(self):
        cfg = Config()
        assert cfg.enable_judge_process_reward is False

    def test_default_values(self):
        cfg = Config()
        assert cfg.judge_process_reward_beta == 0.2
        assert cfg.judge_model_name == ""
        assert cfg.judge_max_concurrency == 4

    def test_target_scale_stays_one(self):
        # The convex blend keeps the return in [0, 1], matching the critic's
        # [0, 1] output range, so no target rescaling is needed.
        cfg = Config(enable_judge_process_reward=True)
        assert cfg.critic_target_scale == 1.0


class TestJudgeConfigToggling:
    def test_enable_with_valid_beta(self):
        cfg = Config(enable_judge_process_reward=True, judge_process_reward_beta=0.5)
        assert cfg.enable_judge_process_reward is True
        assert cfg.judge_process_reward_beta == 0.5

    @pytest.mark.parametrize("beta", [0.0, 0.2, 1.0])
    def test_beta_boundaries_ok(self, beta):
        cfg = Config(judge_process_reward_beta=beta)
        assert cfg.judge_process_reward_beta == beta


class TestJudgeConfigValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_beta(self, bad):
        with pytest.raises(ValueError, match="judge_process_reward_beta"):
            Config(judge_process_reward_beta=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_bad_max_concurrency(self, bad):
        with pytest.raises(ValueError, match="judge_max_concurrency"):
            Config(judge_max_concurrency=bad)

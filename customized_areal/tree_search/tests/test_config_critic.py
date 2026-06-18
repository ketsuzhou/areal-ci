# SPDX-License-Identifier: Apache-2.0
"""Tests for generative-critic config fields and GAE gating."""

import warnings

import pytest

from customized_areal.tree_search.config import AdvantageMode, Config


class TestCriticConfigDefaults:
    def test_disabled_by_default(self):
        cfg = Config()
        assert cfg.enable_generative_critic is False

    def test_default_values(self):
        cfg = Config()
        assert cfg.critic_avg_success_rate == 0.29
        assert cfg.critic_gamma == 1.0
        assert cfg.critic_lambda == 0.95
        assert cfg.critic_score_max == 10
        assert cfg.critic_target_scale == 1.0
        assert cfg.critic_max_new_tokens == 1024
        assert cfg.critic_temperature == 0.0
        assert cfg.critic_loss_weight == 1.0

    def test_disabled_keeps_advantage_mode(self):
        # When the critic is off, the default TREE advantage mode is preserved.
        cfg = Config()
        assert cfg.advantage_mode == AdvantageMode.TREE


class TestCriticGaeGating:
    def test_enable_forces_gae(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = Config(enable_generative_critic=True)
        assert cfg.advantage_mode == AdvantageMode.GAE

    def test_enable_with_tree_warns_and_overrides(self):
        with pytest.warns(UserWarning, match="requires advantage_mode=GAE"):
            cfg = Config(
                enable_generative_critic=True,
                advantage_mode=AdvantageMode.TREE,
            )
        assert cfg.advantage_mode == AdvantageMode.GAE

    def test_enable_with_gae_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = Config(
                enable_generative_critic=True,
                advantage_mode=AdvantageMode.GAE,
            )
        assert cfg.advantage_mode == AdvantageMode.GAE


class TestCriticConfigValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_success_rate(self, bad):
        with pytest.raises(ValueError, match="critic_avg_success_rate"):
            Config(critic_avg_success_rate=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_gamma(self, bad):
        with pytest.raises(ValueError, match="critic_gamma"):
            Config(critic_gamma=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_lambda(self, bad):
        with pytest.raises(ValueError, match="critic_lambda"):
            Config(critic_lambda=bad)

    def test_bad_score_max(self):
        with pytest.raises(ValueError, match="critic_score_max"):
            Config(critic_score_max=0)

    def test_bad_target_scale(self):
        with pytest.raises(ValueError, match="critic_target_scale"):
            Config(critic_target_scale=0.0)

    def test_bad_loss_weight(self):
        with pytest.raises(ValueError, match="critic_loss_weight"):
            Config(critic_loss_weight=-1.0)

    def test_bad_max_new_tokens(self):
        with pytest.raises(ValueError, match="critic_max_new_tokens"):
            Config(critic_max_new_tokens=0)

    def test_bad_temperature(self):
        with pytest.raises(ValueError, match="critic_temperature"):
            Config(critic_temperature=-0.5)

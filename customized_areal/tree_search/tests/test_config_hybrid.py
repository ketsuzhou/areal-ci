# SPDX-License-Identifier: Apache-2.0
"""Tests for HYBRID_GAE advantage mode and hybrid config knobs (Task 3)."""

import warnings

from customized_areal.tree_search.config import AdvantageMode, Config


class TestHybridEnum:
    def test_string_round_trip(self):
        assert AdvantageMode("hybrid_gae") is AdvantageMode.HYBRID_GAE
        assert AdvantageMode.HYBRID_GAE.value == "hybrid_gae"


class TestHybridDefaults:
    def test_defaults_present(self):
        cfg = Config()
        assert cfg.hybrid_mc_min_visits == 5
        assert cfg.hybrid_critic_var_floor == 1e-3
        assert cfg.branch_td_threshold == 0.0


class TestHybridCriticGating:
    def test_enable_with_hybrid_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = Config(
                enable_generative_critic=True,
                advantage_mode=AdvantageMode.HYBRID_GAE,
            )
        assert cfg.advantage_mode == AdvantageMode.HYBRID_GAE

    def test_enable_with_hybrid_string_coerced(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = Config(
                enable_generative_critic=True,
                advantage_mode="hybrid_gae",
            )
        assert cfg.advantage_mode == AdvantageMode.HYBRID_GAE

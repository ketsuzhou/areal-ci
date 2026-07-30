import pytest

from customized_areal.tree_search.config import (
    CacheMode,
    Config,
    DistillKLMode,
    LossMode,
)


class TestCacheMode:
    def test_off_is_default(self):
        config = Config()
        assert config.mode == CacheMode.OFF

    def test_enum_values(self):
        assert CacheMode.OFF == "off"
        assert CacheMode.IN_TRAINING == "in_training"
        assert CacheMode.CROSS_TRAINING == "cross_training"

    def test_default_checkpoint_dir_empty(self):
        config = Config()
        assert config.checkpoint_dir == ""

    def test_custom_values(self):
        config = Config(
            mode=CacheMode.CROSS_TRAINING,
            checkpoint_dir="/tmp/mcts",
        )
        assert config.mode == CacheMode.CROSS_TRAINING
        assert config.checkpoint_dir == "/tmp/mcts"


class TestDistillConfig:
    def test_distill_kl_mode_defaults_to_reverse(self):
        config = Config()
        assert config.distill_kl_mode == DistillKLMode.REVERSE

    def test_distill_env_defaults_are_representable(self):
        config = Config(
            topk_distill=True,
            teacher_provider="external",
            teacher_base_url="http://teacher:8001",
            teacher_model_name="qwen-397b",
            teacher_top_k=5,
            diagnose_model_name="qwen-397b",
            distill_kl_mode=DistillKLMode.FORWARD,
        )

        assert config.topk_distill is True
        assert config.teacher_provider == "external"
        assert config.teacher_top_k == 5
        assert config.distill_kl_mode == DistillKLMode.FORWARD

    def test_loss_mode_enum(self):
        assert LossMode.GRPO == "grpo"
        assert LossMode.DISTILL == "distill"
        assert LossMode.BOTH == "both"


class TestClipCovConfig:
    def test_clip_cov_defaults_disabled(self):
        config = Config()

        assert config.use_clip_cov is False
        assert config.clip_cov_clip_ratio == 0.0002
        assert config.clip_cov_lb == 1.0
        assert config.clip_cov_ub == 5.0

    def test_clip_cov_custom_values(self):
        config = Config(
            use_clip_cov=True,
            clip_cov_clip_ratio=0.01,
            clip_cov_lb=0.5,
            clip_cov_ub=2.0,
        )

        assert config.use_clip_cov is True
        assert config.clip_cov_clip_ratio == 0.01
        assert config.clip_cov_lb == 0.5
        assert config.clip_cov_ub == 2.0

    def test_clip_cov_rejects_invalid_ratio(self):
        with pytest.raises(ValueError, match="clip_cov_clip_ratio"):
            Config(clip_cov_clip_ratio=1.1)

    def test_clip_cov_rejects_invalid_bounds(self):
        with pytest.raises(ValueError, match="clip_cov_lb"):
            Config(clip_cov_lb=2.0, clip_cov_ub=2.0)


class TestMuonConfig:
    def test_muon_defaults_disabled(self):
        config = Config()

        assert config.use_muon_optimizer is False
        assert config.muon_momentum == 0.95
        assert config.muon_adam_lr == 3e-4
        assert config.muon_ns_steps == 5
        assert config.muon_nesterov is True

    def test_muon_custom_values(self):
        config = Config(
            use_muon_optimizer=True,
            muon_momentum=0.9,
            muon_adam_lr=1e-4,
            muon_ns_steps=3,
            muon_nesterov=False,
        )

        assert config.use_muon_optimizer is True
        assert config.muon_momentum == 0.9
        assert config.muon_adam_lr == 1e-4
        assert config.muon_ns_steps == 3
        assert config.muon_nesterov is False

    def test_muon_rejects_invalid_momentum(self):
        with pytest.raises(ValueError, match="muon_momentum"):
            Config(muon_momentum=1.0)


class TestFreshQueryConfig:
    def test_fresh_query_defaults_disabled(self):
        config = Config()

        assert config.use_fresh_query is False
        assert config.fresh_query_table == ""

    def test_fresh_query_accepts_configured_table(self):
        config = Config(use_fresh_query=True, fresh_query_table="query_bank")

        assert config.use_fresh_query is True
        assert config.fresh_query_table == "query_bank"

    def test_fresh_query_uses_env_table(self, monkeypatch):
        monkeypatch.setenv("FRESH_QUERY_TABLE", "env_query_bank")

        config = Config(use_fresh_query=True)

        assert config.fresh_query_table == "env_query_bank"

    def test_fresh_query_requires_table(self, monkeypatch):
        monkeypatch.delenv("FRESH_QUERY_TABLE", raising=False)

        with pytest.raises(ValueError, match="fresh_query_table"):
            Config(use_fresh_query=True)

    def test_muon_rejects_invalid_aux_adam_lr(self):
        with pytest.raises(ValueError, match="muon_adam_lr"):
            Config(muon_adam_lr=0)

    def test_muon_rejects_invalid_ns_steps(self):
        with pytest.raises(ValueError, match="muon_ns_steps"):
            Config(muon_ns_steps=0)

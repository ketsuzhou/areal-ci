from customized_areal.tree_search.config import CacheMode, LossMode, TreeBackupConfig


class TestCacheMode:
    def test_off_is_default(self):
        config = TreeBackupConfig()
        assert config.mode == CacheMode.OFF

    def test_enum_values(self):
        assert CacheMode.OFF == "off"
        assert CacheMode.IN_TRAINING == "in_training"
        assert CacheMode.CROSS_TRAINING == "cross_training"

    def test_default_checkpoint_dir_empty(self):
        config = TreeBackupConfig()
        assert config.checkpoint_dir == ""

    def test_custom_values(self):
        config = TreeBackupConfig(
            mode=CacheMode.CROSS_TRAINING,
            checkpoint_dir="/tmp/mcts",
        )
        assert config.mode == CacheMode.CROSS_TRAINING
        assert config.checkpoint_dir == "/tmp/mcts"


class TestDistillConfig:
    def test_distill_env_defaults_are_representable(self):
        config = TreeBackupConfig(
            topk_distill=True,
            teacher_provider="external",
            teacher_base_url="http://teacher:8001",
            teacher_model_name="qwen-397b",
            teacher_top_k=5,
            diagnose_model_name="qwen-397b",
        )

        assert config.topk_distill is True
        assert config.teacher_provider == "external"
        assert config.teacher_top_k == 5

    def test_loss_mode_enum(self):
        assert LossMode.GRPO == "grpo"
        assert LossMode.DISTILL == "distill"
        assert LossMode.BOTH == "both"

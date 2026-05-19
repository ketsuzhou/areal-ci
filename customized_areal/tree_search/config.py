from dataclasses import dataclass
from enum import Enum


class CacheMode(str, Enum):
    OFF = "off"
    IN_TRAINING = "in_training"
    CROSS_TRAINING = "cross_training"


class AdvantageMode(str, Enum):
    GAE = "gae"
    TREE = "tree"


class LossMode(str, Enum):
    GRPO = "grpo"
    DISTILL = "distill"
    BOTH = "both"


@dataclass
class TreeBackupConfig:
    mode: CacheMode = CacheMode.OFF
    enabled: bool = True
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.TREE
    loss_mode: LossMode = LossMode.GRPO
    max_reasoning_tokens: int = 1000
    rl_loss_weight: float = 1.0
    distill_loss_weight: float = 0.005
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    overlong_reward_penalty: bool = False
    overlong_tokens: int | None = None
    overlong_penalty_factor: float | None = None
    topk_distill: bool = False
    teacher_provider: str = "external"
    teacher_base_url: str = "http://localhost:8001"
    teacher_model_name: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_timeout: float = 60.0
    teacher_missing_logprob: float = -23.0
    diagnose_model_name: str = ""
    diagnose_max_tokens: int = 1024
    diagnose_temperature: float = 0.0
    diagnose_base_url: str = ""
    diagnose_api_key: str = ""
    strict_distill_json: bool = True


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1

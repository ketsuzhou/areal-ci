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
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.TREE
    loss_mode: LossMode = LossMode.GRPO
    rl_loss_weight: float = 1.0
    distill_loss_weight: float = 0.005
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    overlong_reward_penalty: bool = False
    overlong_tokens: int | None = None
    overlong_penalty_factor: float | None = None


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1

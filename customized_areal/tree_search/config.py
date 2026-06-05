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


class DistillKLMode(str, Enum):
    FORWARD = "forward_kl"
    REVERSE = "reverse_kl"


class SampleSource(str, Enum):
    SCRATCH = "scratch"
    BRANCH = "branch"
    MIXED = "mixed"


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
    teacher_api_key: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_max_concurrency: int = 4
    teacher_timeout: float = 300.0
    teacher_missing_logprob: float = -23.0
    teacher_backend: str = "openai"
    diagnose_model_name: str = ""
    diagnose_max_tokens: int = 1024
    diagnose_temperature: float = 0.0
    diagnose_base_url: str = ""
    diagnose_api_key: str = ""
    strict_distill_json: bool = True
    sample_source: SampleSource = SampleSource.SCRATCH
    branch_probability: float = 0.5
    dynamic_group_size: bool = False
    initial_group_size: int = 4
    max_group_size: int = 64
    uncertainty_threshold: float = 0.05
    reward_type: str = "binary"
    distill_kl_mode: DistillKLMode = DistillKLMode.REVERSE
    max_distill_tokens: int = 0

    def __post_init__(self) -> None:
        self.distill_kl_mode = DistillKLMode(self.distill_kl_mode)
        if self.initial_group_size < 1:
            raise ValueError(
                f"initial_group_size must be >= 1, got {self.initial_group_size}"
            )
        if self.max_group_size < self.initial_group_size:
            raise ValueError(
                f"max_group_size ({self.max_group_size}) must be >= "
                f"initial_group_size ({self.initial_group_size})"
            )
        if self.uncertainty_threshold < 0:
            raise ValueError(
                f"uncertainty_threshold must be >= 0, got {self.uncertainty_threshold}"
            )
        if self.reward_type not in {"binary", "continuous"}:
            raise ValueError(
                f"reward_type must be 'binary' or 'continuous', "
                f"got {self.reward_type!r}"
            )


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1

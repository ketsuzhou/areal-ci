import os
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
class Config:
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
    use_clip_cov: bool = False
    clip_cov_clip_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    use_muon_optimizer: bool = False
    muon_momentum: float = 0.95
    muon_adam_lr: float = 3e-4
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    use_fresh_query: bool = False
    fresh_query_table: str = ""

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
        if not 0.0 <= self.clip_cov_clip_ratio <= 1.0:
            raise ValueError(
                f"clip_cov_clip_ratio must be in [0, 1], got {self.clip_cov_clip_ratio}"
            )
        if self.clip_cov_lb >= self.clip_cov_ub:
            raise ValueError(
                f"clip_cov_lb ({self.clip_cov_lb}) must be < "
                f"clip_cov_ub ({self.clip_cov_ub})"
            )
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError(
                f"muon_momentum must be in [0, 1), got {self.muon_momentum}"
            )
        if self.muon_adam_lr <= 0:
            raise ValueError(f"muon_adam_lr must be > 0, got {self.muon_adam_lr}")
        if self.muon_ns_steps < 1:
            raise ValueError(f"muon_ns_steps must be >= 1, got {self.muon_ns_steps}")
        if self.use_fresh_query:
            self.fresh_query_table = self.fresh_query_table or os.environ.get(
                "FRESH_QUERY_TABLE", ""
            )
            if not self.fresh_query_table:
                raise ValueError(
                    "fresh_query_table must be set when use_fresh_query=True "
                    "(or set FRESH_QUERY_TABLE)"
                )


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1

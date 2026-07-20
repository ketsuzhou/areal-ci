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
    HYBRID_GAE = "hybrid_gae"
    VERSIONED_BACKUP = "versioned_backup"


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
    # Weight of the co-trained generative-critic value loss in the combined
    # actor+critic objective (``actor_PG + critic_loss_weight * critic_value_loss``)
    # when ``advantage_mode == GAE`` for DAG runs (Phase 3). Start small to limit
    # gradient interference on the shared trunk.
    critic_loss_weight: float = 0.5
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

    # Generative critic (shared-model) settings.
    #
    # When ``enable_generative_critic`` is True, the actor's own model (same
    # weights, same SGLang server) doubles as a generative critic: it scores
    # partial-solution states by emitting an integer in ``[0, critic_score_max]``
    # via a success-probability prompt. The soft expected value over those digit
    # tokens is used as the state value ``v_phi(s_t)``, which drives GAE
    # advantages for the actor. The same shared model is trained with a combined
    # ``actor_loss + critic_loss_weight * critic_loss`` objective, where the
    # critic loss is an expected-value soft regression toward the tree-stored
    # MCTS ``q_value`` target.
    enable_generative_critic: bool = False
    critic_avg_success_rate: float = 0.29
    critic_gamma: float = 1.0
    critic_lambda: float = 0.95
    critic_score_max: int = 10
    critic_target_scale: float = 1.0
    critic_max_new_tokens: int = 1024
    critic_temperature: float = 0.0
    critic_loss_weight: float = 1.0
    # Unified TD/MC critic target. The regression target is a convex blend
    #   y = (1 - w) * y_td + w * y_mc
    # where y_mc is the MCTS Monte-Carlo q_value and y_td is an n-step
    # bootstrapped return. ``critic_mc_weight`` is the fixed weight ``w``
    # (1.0 == previous pure-MCTS behavior, 0.0 == pure n-step TD).
    # ``critic_td_n_steps`` is the TD horizon. When ``critic_mc_adaptive`` is
    # True the weight is determined per node from MCTS visit counts (and, when
    # fed back, critic error) with scale ``critic_mc_c``.
    critic_mc_weight: float = 1.0
    critic_td_n_steps: int = 1
    critic_mc_adaptive: bool = False
    critic_mc_c: float = 4.0

    # Variance-aware hybrid GAE (advantage_mode=HYBRID_GAE). On branched nodes
    # with enough MCTS samples, the noisy critic value v_theta(s_t) is blended
    # with a leave-one-out Monte-Carlo value via inverse-variance weighting.
    # ``hybrid_mc_min_visits`` is the minimum node visit count required before
    # MC substitution applies (the LOO set then has >= value-1 samples).
    # ``hybrid_critic_var_floor`` floors the critic's error variance so a
    # perfectly-confident critic does not appear infinitely reliable.
    # ``hybrid_critic_error_var`` is the static prior on the critic's *error*
    # variance E[(v_theta - V)^2] (the critic regression MSE), used as
    # ``var_theta`` in the inverse-variance blend so it is unit-consistent with
    # the MC mean's sampling variance ``var_mc``. With returns in [0, 1] a
    # moderately-trained critic (RMSE ~0.22) sits near 0.05, comparable to
    # ``var_mc`` at the minimum visit count, so MC and critic blend ~50/50 there
    # and MC gains weight as visits accumulate. When a live critic-MSE EMA is
    # wired in (AdaptiveMCWeight) it overrides this prior per rollout.
    hybrid_mc_min_visits: int = 5
    hybrid_critic_var_floor: float = 1e-3
    hybrid_critic_error_var: float = 0.05
    # Fixed absolute TD-error threshold for the branch-selection gate: only
    # need_branch candidates whose |delta_t| (from critic values) meets this
    # threshold are eligible; survivors are then ranked by entropy. 0.0 keeps
    # the previous entropy-only behavior.
    branch_td_threshold: float = 0.0

    # ARE-4 ΔV advantage (advantage_mode=VERSIONED_BACKUP). A(s_t) =
    # G_latest^(n)(s_t) - V_own(s_t) with a variance-threshold adaptive
    # horizon: bootstrap at the first descendant s_{t+d} where the critic is
    # trustworthy (critic_error_var(s_{t+d}) <= versioned_rho * var_mc(s_{t+d})).
    # ``versioned_rho`` in (0, 1] (default 1: critic trusted once its error
    # variance is no larger than the MC mean's sampling variance).
    # ``versioned_n_min`` / ``versioned_n_max`` floor/ceiling the horizon; if the
    # trust criterion is never met within n_max (or the episode ends first) the
    # return is pure MC with a 0 terminal bootstrap.
    versioned_rho: float = 1.0
    versioned_n_min: int = 1
    versioned_n_max: int = 10

    def __post_init__(self) -> None:
        self.distill_kl_mode = DistillKLMode(self.distill_kl_mode)
        if self.critic_loss_weight < 0:
            raise ValueError(
                f"critic_loss_weight must be >= 0, got {self.critic_loss_weight}"
            )
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

        # Generative-critic validation and GAE gating.
        self.advantage_mode = AdvantageMode(self.advantage_mode)
        if not 0.0 <= self.critic_avg_success_rate <= 1.0:
            raise ValueError(
                "critic_avg_success_rate must be in [0, 1], got "
                f"{self.critic_avg_success_rate}"
            )
        if not 0.0 <= self.critic_gamma <= 1.0:
            raise ValueError(f"critic_gamma must be in [0, 1], got {self.critic_gamma}")
        if not 0.0 <= self.critic_lambda <= 1.0:
            raise ValueError(
                f"critic_lambda must be in [0, 1], got {self.critic_lambda}"
            )
        if self.critic_score_max < 1:
            raise ValueError(
                f"critic_score_max must be >= 1, got {self.critic_score_max}"
            )
        if self.critic_target_scale <= 0:
            raise ValueError(
                f"critic_target_scale must be > 0, got {self.critic_target_scale}"
            )
        if self.critic_loss_weight < 0:
            raise ValueError(
                f"critic_loss_weight must be >= 0, got {self.critic_loss_weight}"
            )
        if self.critic_max_new_tokens < 1:
            raise ValueError(
                f"critic_max_new_tokens must be >= 1, got {self.critic_max_new_tokens}"
            )
        if self.critic_temperature < 0:
            raise ValueError(
                f"critic_temperature must be >= 0, got {self.critic_temperature}"
            )
        if not 0.0 <= self.critic_mc_weight <= 1.0:
            raise ValueError(
                f"critic_mc_weight must be in [0, 1], got {self.critic_mc_weight}"
            )
        if self.critic_td_n_steps < 1:
            raise ValueError(
                f"critic_td_n_steps must be >= 1, got {self.critic_td_n_steps}"
            )
        if self.critic_mc_c <= 0:
            raise ValueError(f"critic_mc_c must be > 0, got {self.critic_mc_c}")
        # ARE-4 versioned-backup hyperparameter validation.
        if not 0.0 < self.versioned_rho <= 1.0:
            raise ValueError(
                f"versioned_rho must be in (0, 1], got {self.versioned_rho}"
            )
        if self.versioned_n_min < 1:
            raise ValueError(
                f"versioned_n_min must be >= 1, got {self.versioned_n_min}"
            )
        if self.versioned_n_max < self.versioned_n_min:
            raise ValueError(
                f"versioned_n_max ({self.versioned_n_max}) must be >= "
                f"versioned_n_min ({self.versioned_n_min})"
            )
        if self.enable_generative_critic:
            # The generative critic supplies bootstrapped state values, so the
            # actor advantage must be GAE or the variance-aware HYBRID_GAE
            # (which also consumes the critic). Auto-switch from the default
            # TREE mode and warn if a conflicting mode was set explicitly.
            if self.advantage_mode not in (
                AdvantageMode.GAE,
                AdvantageMode.HYBRID_GAE,
                AdvantageMode.VERSIONED_BACKUP,
            ):
                import warnings

                warnings.warn(
                    "enable_generative_critic=True requires "
                    "advantage_mode=GAE, HYBRID_GAE, or VERSIONED_BACKUP; "
                    f"overriding advantage_mode={self.advantage_mode.value!r} -> 'gae'.",
                    stacklevel=2,
                )
                self.advantage_mode = AdvantageMode.GAE


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1

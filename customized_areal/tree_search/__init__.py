# NOTE: only torch-free, dependency-light symbols (config enums/dataclasses) are
# imported eagerly. Torch-heavy symbols (advantage / checkpoint / tree_store) and
# distillation types (which pull optional deps like ``openai``) are loaded lazily
# via ``__getattr__`` below, so torch-free subpackages such as
# ``customized_areal.tree_search.agents`` can be imported and unit-tested without the
# training stack installed.
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    Config,
    DistillKLMode,
    LossMode,
    RolloutCacheConfig,
)

__all__ = [
    "AdvantageMode",
    "CustomizedPPOTrainer",
    "DiagnosisTurn",
    "DistillKLMode",
    "EpisodeDiagnosis",
    "InteractionWithTokenLevelReward",
    "LossMode",
    "MCTSTreeStore",
    "Node",
    "PositionRewardInfo",
    "TreeSearchGroupedRolloutWorkflow",
    "RolloutCacheConfig",
    "TreeAdvantageComputer",
    "GAEAdvantageComputer",
    "HybridGAEAdvantageComputer",
    "VersionedBackupAdvantageComputer",
    "Config",
    "CacheMode",
    "TreeCheckpointManager",
]


def __getattr__(name):
    # Lazy imports for torch-heavy core components (keeps the package importable
    # without torch for torch-free subpackages like ``dag``).
    if name == "TreeAdvantageComputer":
        from .core.advantage import TreeAdvantageComputer

        return TreeAdvantageComputer
    if name == "GAEAdvantageComputer":
        from .core.advantage import GAEAdvantageComputer

        return GAEAdvantageComputer
    if name == "HybridGAEAdvantageComputer":
        from .core.advantage import HybridGAEAdvantageComputer

        return HybridGAEAdvantageComputer
    if name == "VersionedBackupAdvantageComputer":
        from .core.advantage import VersionedBackupAdvantageComputer

        return VersionedBackupAdvantageComputer
    if name == "TreeCheckpointManager":
        from .core.checkpoint import TreeCheckpointManager

        return TreeCheckpointManager
    if name in ("MCTSTreeStore", "Node"):
        from .core import tree_store

        return getattr(tree_store, name)
    if name in (
        "DiagnosisTurn",
        "EpisodeDiagnosis",
        "InteractionWithTokenLevelReward",
        "PositionRewardInfo",
    ):
        from .distilling import distill_types

        return getattr(distill_types, name)
    # Lazy imports for distillation components
    if name == "OnPolicyDistillConfig":
        from .distilling.config import OnPolicyDistillConfig

        return OnPolicyDistillConfig
    if name == "OnPolicyDistillAgent":
        from .distilling.agent import OnPolicyDistillAgent

        return OnPolicyDistillAgent
    if name == "TeacherConfig":
        from .distilling.teacher_client import TeacherConfig

        return TeacherConfig
    if name == "TeacherClient":
        from .distilling.teacher_client import TeacherClient

        return TeacherClient
    if name == "MultiCandidateFSDPEngine":
        from .engine import MultiCandidateFSDPEngine

        return MultiCandidateFSDPEngine
    if name == "MultiCandidateFSDPPPOActor":
        from .engine import MultiCandidateFSDPPPOActor

        return MultiCandidateFSDPPPOActor
    if name == "CustomizedPPOTrainer":
        from .training.trainer import CustomizedPPOTrainer

        return CustomizedPPOTrainer
    if name == "TreeSearchGroupedRolloutWorkflow":
        from .core.customized_grouped_workflow import TreeSearchGroupedRolloutWorkflow

        return TreeSearchGroupedRolloutWorkflow
    if name == "grpo_distill_loss_fn":
        from .training.loss import grpo_distill_loss_fn

        return grpo_distill_loss_fn
    if name == "gather_logprobs_entropy_multi_candidates":
        from .training.logprobs import gather_logprobs_entropy_multi_candidates

        return gather_logprobs_entropy_multi_candidates
    if name == "_compute_token_rewards":
        from .distilling.reward_compute import _compute_token_rewards

        return _compute_token_rewards
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

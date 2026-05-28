from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
    RolloutCacheConfig,
    TreeBackupConfig,
)
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    EpisodeDiagnosis,
    InteractionWithTokenLevelReward,
    PositionRewardInfo,
)
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.training.trainer import CustomizedPPOTrainer
from customized_areal.tree_search.tree_search_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)

__all__ = [
    "AdvantageMode",
    "CustomizedPPOTrainer",
    "DiagnosisTurn",
    "EpisodeDiagnosis",
    "InteractionWithTokenLevelReward",
    "LossMode",
    "MCTSTreeStore",
    "Node",
    "PositionRewardInfo",
    "TreeSearchGroupedRolloutWorkflow",
    "RolloutCacheConfig",
    "TreeAdvantageComputer",
    "TreeBackupConfig",
    "CacheMode",
    "TreeCheckpointManager",
]


def __getattr__(name):
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

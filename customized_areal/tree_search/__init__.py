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
    InteractionWithTokenLevelReward,
    PositionRewardInfo,
)
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.tree_search_grouped_workflow import TreeSearchGroupedRolloutWorkflow
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

__all__ = [
    "AdvantageMode",
    "CacheAwarePPOTrainer",
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
        from .core.config import OnPolicyDistillConfig

        return OnPolicyDistillConfig
    if name == "OnPolicyDistillAgent":
        from .core.agent import OnPolicyDistillAgent

        return OnPolicyDistillAgent
    if name == "TeacherConfig":
        from .core.teacher_client import TeacherConfig

        return TeacherConfig
    if name == "TeacherClient":
        from .core.teacher_client import TeacherClient

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
        from .core.reward_compute import _compute_token_rewards

        return _compute_token_rewards
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

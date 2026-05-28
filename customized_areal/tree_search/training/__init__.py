"""Training components for on-policy distillation.

This module contains:
- actor.py: MultiCandidateFSDPPPOActor and distill-loss patching functions
- loss.py: grpo_distill_loss_fn for combined GRPO + position-level loss
- logprobs.py: Log probability and entropy computation utilities
- trainer.py: CustomizedPPOTrainer for tree search PPO training
"""

from .actor import (
    MultiCandidateFSDPPPOActor,
    patch_ppo_actor_class_to_use_distill_loss,
    unpatch_ppo_actor_distill_loss,
)
from .logprobs import gather_logprobs_entropy_multi_candidates
from .loss import grpo_distill_loss_fn
from .trainer import CustomizedPPOTrainer

__all__ = [
    "MultiCandidateFSDPPPOActor",
    "patch_ppo_actor_class_to_use_distill_loss",
    "unpatch_ppo_actor_distill_loss",
    "grpo_distill_loss_fn",
    "gather_logprobs_entropy_multi_candidates",
    "CustomizedPPOTrainer",
]

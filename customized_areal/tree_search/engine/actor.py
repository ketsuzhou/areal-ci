"""Custom PPO Actor using MultiCandidateFSDPEngine.

This module provides MultiCandidateFSDPPPOActor which extends
MultiCandidateFSDPEngine and wraps PPOActor for on-policy distillation.
"""

from __future__ import annotations

from typing import Any

import torch

from areal.api import Scheduler
from areal.api.cli_args import PPOActorConfig
from areal.utils import logging

from .fsdp_engine import MultiCandidateFSDPEngine

logger = logging.getLogger("MultiCandidateFSDPPPOActor")


class MultiCandidateFSDPPPOActor(MultiCandidateFSDPEngine):
    """PPO Actor implementation using MultiCandidateFSDPEngine backend.

    This actor extends MultiCandidateFSDPEngine instead of standard FSDPEngine
    to enable multi-candidate logprob gathering for on-policy distillation.

    The key difference from FSDPPPOActor:
    - Uses MultiCandidateFSDPEngine as base class
    - Supports position-level rewards via _prepare_multi_candidate_labels
    - Computes logprobs for all candidate tokens, not just chosen tokens

    Example usage:
        actor = MultiCandidateFSDPPPOActor(config)
        actor.initialize(role="actor", ...)
        actor.ppo_update(batch)  # Uses grpo_distill_loss_fn
    """

    def __init__(self, config: PPOActorConfig):
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

        # Patch PPOActor._ppo_update on the worker process to use
        # grpo_distill_loss_fn.  The controller-side patch in
        # CacheAwarePPOTrainer.train() only affects the controller process;
        # workers receive ppo_update via RPC and run the original
        # _ppo_update → grpo_loss_fn, which crashes on teacher_logp shape
        # mismatch (response-aligned vs full-sequence-length).
        from customized_areal.tree_search.training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        patch_ppo_actor_class_to_use_distill_loss()

        logger.info("MultiCandidateFSDPPPOActor initialized")

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> list[torch.Tensor] | None:
        """Compute log probabilities for given trajectories."""
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Compute advantages for given trajectories."""
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        """Perform PPO update on the given batch."""
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        """Create a controller for this actor class.

        This method is used by the trainer to create a controller
        that manages distributed workers.
        """
        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(
            train_engine=cls,
            config=config,
            scheduler=scheduler,
        )

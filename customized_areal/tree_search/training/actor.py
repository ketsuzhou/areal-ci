"""Custom PPO Actor with distill-loss patching and multi-candidate support.

This module provides:
- MultiCandidateFSDPPPOActor: PPO actor using MultiCandidateFSDPEngine for
  on-policy distillation with multi-candidate logprob gathering.
- patch/unpatch functions to swap PPOActor._ppo_update with grpo_distill_loss_fn.
"""

from __future__ import annotations

import copy
import functools
from typing import Any

import torch

from areal.api import Scheduler
from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging, stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list

from ..engine.fsdp_engine import MultiCandidateFSDPEngine

logger = logging.getLogger("OnPolicyDistill")

_patch_applied = False
_original_ppo_update = None


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
        super().__init__(config)
        self.actor = PPOActor(config, self)

        # Patch PPOActor._ppo_update on the worker process to use
        # grpo_distill_loss_fn.  The controller-side patch in
        # CustomizedPPOTrainer.train() only affects the controller process;
        # workers receive ppo_update via RPC and run the original
        # _ppo_update → grpo_loss_fn, which crashes on teacher_logp shape
        # mismatch (response-aligned vs full-sequence-length).
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


def patch_ppo_actor_class_to_use_distill_loss() -> None:
    """Patch PPOActor class to use grpo_distill_loss_fn globally.

    This replaces PPOActor._ppo_update with a version that uses
    grpo_distill_loss_fn instead of standard grpo_loss_fn.

    Only patches once, even if called multiple times.

    Note:
    -----
    For multi-candidate gathering during training, the AReaL engine must be
    modified to pass logits to the loss function. See ENGINE_MODIFICATION.md
    in the training directory.
    """
    global _patch_applied, _original_ppo_update
    if _patch_applied:
        return

    _original_ppo_update = PPOActor._ppo_update

    def _ppo_update_with_distill_loss(self, data: dict[str, Any]) -> None:
        """PPO update using grpo_distill_loss_fn."""
        from .loss import grpo_distill_loss_fn

        # Log reward stats before removing them (Bug 2 fix)
        reward_score = data.get("rewards")
        if reward_score is not None and isinstance(reward_score, torch.Tensor):
            correct_n = (reward_score > 0).bool()
            incorrect_n = (reward_score <= 0).bool()
            stats_tracker.denominator(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                correct_n_seqs=correct_n,
                incorrect_n_seqs=incorrect_n,
            )
            attn_mask = data.get("attention_mask")
            if attn_mask is not None:
                stats_tracker.stat(
                    task_reward=reward_score.float(),
                    denominator="n_seqs",
                )

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        # Extract position_rewards before splitting so we can distribute
        # the correct subset to each minibatch.  position_rewards is a
        # Python list and cannot be split by the generic tensor-based
        # minibatch splitter.
        position_rewards = data.pop("position_rewards", None)

        self.engine.train()

        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        # Distribute position_rewards to minibatches based on sample_index.
        # Each PositionRewardInfo.sample_index indicates which batch item
        # it belongs to.  We use the forward_indices from the minibatch
        # split to determine which samples are in which minibatch.
        if position_rewards is not None:
            _distribute_position_rewards(mb_inputs, position_rewards)

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_distill_loss_fn,
                        config=self.config,
                        current_version=current_version,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

    PPOActor._ppo_update = _ppo_update_with_distill_loss
    _patch_applied = True
    logger.info("PPOActor class patched to use grpo_distill_loss_fn")


def unpatch_ppo_actor_distill_loss() -> None:
    """Restore the original PPOActor._ppo_update method.

    Must be called after patch_ppo_actor_class_to_use_distill_loss().
    """
    global _patch_applied, _original_ppo_update
    if _patch_applied and _original_ppo_update is not None:
        PPOActor._ppo_update = _original_ppo_update
        _original_ppo_update = None
        _patch_applied = False
        logger.info("Restored original PPOActor._ppo_update")


def _distribute_position_rewards(mb_inputs, position_rewards: list) -> None:
    """Distribute position_rewards to minibatches based on sample_index.

    Each PositionRewardInfo has a sample_index indicating which batch item
    it belongs to.  The MicroBatchList.forward_indices maps batch items to
    their position in the reordered minibatch sequence.  We use this to
    determine which position_rewards belong to which minibatch.
    """
    if not position_rewards:
        return

    forward_indices = mb_inputs.forward_indices
    # Build mapping: original batch index -> minibatch index
    batch_size = len(forward_indices)
    mb_assignment: list[int | None] = [None] * batch_size
    offset = 0
    for i, mb in enumerate(mb_inputs.mbs):
        mb_bs = mb["attention_mask"].shape[0]
        for j in range(mb_bs):
            orig_idx = int(forward_indices[offset + j])
            mb_assignment[orig_idx] = i
        offset += mb_bs

    # Group position_rewards by minibatch and rebase sample_index to the
    # sample's local index inside that minibatch.
    per_mb_prs: dict[int, list] = {}
    for pr in position_rewards:
        if pr.sample_index >= len(mb_assignment):
            logger.warning(
                "position_reward sample_index=%d exceeds batch_size=%d, "
                "dropping position=%d",
                pr.sample_index,
                len(mb_assignment),
                pr.position,
            )
            continue
        mb_i = mb_assignment[pr.sample_index]
        if mb_i is None:
            logger.warning(
                "position_reward sample_index=%d not mapped to any minibatch, "
                "dropping position=%d",
                pr.sample_index,
                pr.position,
            )
            continue
        local_sample_index = None
        offset = 0
        for i, mb in enumerate(mb_inputs.mbs):
            mb_bs = mb["attention_mask"].shape[0]
            if i == mb_i:
                for local_idx in range(mb_bs):
                    orig_idx = int(forward_indices[offset + local_idx])
                    if orig_idx == pr.sample_index:
                        local_sample_index = local_idx
                        break
                break
            offset += mb_bs
        if local_sample_index is None:
            logger.warning(
                "position_reward sample_index=%d could not be rebased for minibatch %d, "
                "dropping position=%d",
                pr.sample_index,
                mb_i,
                pr.position,
            )
            continue
        mb_pr = copy.copy(pr)
        mb_pr.sample_index = local_sample_index
        per_mb_prs.setdefault(mb_i, []).append(mb_pr)

    # Attach to minibatches
    for i, mb in enumerate(mb_inputs.mbs):
        if i in per_mb_prs:
            mb["position_rewards"] = per_mb_prs[i]
        else:
            mb["position_rewards"] = []

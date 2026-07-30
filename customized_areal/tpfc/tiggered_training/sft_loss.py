"""Patch PPO training to use a supervised token loss.

This keeps the rollout/training loop in ``areal.trainer.rl_trainer.PPOTrainer``
but replaces ``PPOActor._ppo_update`` with an SFT-style masked NLL objective.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from areal.api import Scheduler
from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list

logger = logging.getLogger("TriggeredSFTLoss")

_patch_applied = False
_original_ppo_update = None


class TriggeredSFTFSDPPPOActor(FSDPPPOActor):
    """FSDP PPO actor whose worker process uses triggered SFT loss."""

    def __init__(self, config):
        patch_ppo_actor_to_use_triggered_sft_loss()
        super().__init__(config)

    @classmethod
    def as_controller(cls, config, scheduler: Scheduler):
        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(
            train_engine=cls,
            config=config,
            scheduler=scheduler,
        )


def triggered_sft_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, Any],
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked SFT loss over rollout completion tokens."""
    del entropy

    loss_mask = input_data["loss_mask"].bool()
    valid_tokens = loss_mask.count_nonzero().clamp(min=1)
    masked_logprobs = torch.where(loss_mask, logprobs, 0.0)
    loss = -masked_logprobs.sum() / valid_tokens

    with torch.no_grad():
        stats_tracker.denominator(
            n_tokens=infer_token_denominator(input_data, loss_mask),
            n_valid_tokens=loss_mask,
        )
        stats_tracker.stat(
            sft_logp=masked_logprobs.detach(),
            sft_nll=-masked_logprobs.detach(),
            denominator="n_valid_tokens",
        )
        stats_tracker.scalar(sft_loss=loss.detach())
        if vocab_min_logits is not None and vocab_max_logits is not None:
            stats_tracker.stat(
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
                denominator="n_tokens",
            )

    return loss


def patch_ppo_actor_to_use_triggered_sft_loss() -> None:
    """Patch ``PPOActor._ppo_update`` to train with ``triggered_sft_loss_fn``."""
    global _patch_applied, _original_ppo_update
    if _patch_applied:
        return

    _original_ppo_update = PPOActor._ppo_update

    def _ppo_update_with_triggered_sft_loss(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"].bool()
        reward_score = data.get("rewards")
        seqlens = attn_mask.sum(-1)

        denominators: dict[str, torch.Tensor] = {
            "n_tokens": infer_token_denominator(data, loss_mask),
            "n_valid_tokens": loss_mask,
        }
        if isinstance(reward_score, torch.Tensor):
            denominators.update(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                correct_n_seqs=(reward_score > 0).bool(),
                incorrect_n_seqs=(reward_score <= 0).bool(),
            )
        stats_tracker.denominator(**denominators)

        if isinstance(reward_score, torch.Tensor):
            prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
            stats_tracker.stat(
                task_reward=reward_score.float(),
                prompt_len=prompt_lens.float(),
                seq_len=seqlens.float(),
                denominator="n_seqs",
            )
        stats_tracker.scalar(use_triggered_sft_loss=1)

        for key in [
            "advantages",
            "returns",
            "rewards",
            "tot_rewards",
            "kl_rewards",
            "ref_logp",
            "teacher_logp",
            "prox_logp",
        ]:
            data.pop(key, None)

        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(triggered_sft_loss_fn),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

    PPOActor._ppo_update = _ppo_update_with_triggered_sft_loss
    _patch_applied = True
    logger.info("PPOActor class patched to use triggered SFT loss")


def unpatch_ppo_actor_triggered_sft_loss() -> None:
    """Restore the original PPO update method."""
    global _patch_applied, _original_ppo_update
    if _patch_applied and _original_ppo_update is not None:
        PPOActor._ppo_update = _original_ppo_update
        _original_ppo_update = None
        _patch_applied = False
        logger.info("Restored original PPOActor._ppo_update")

"""Combined GRPO and teacher distillation loss."""

from __future__ import annotations

import torch

from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker

from .distill import (
    _compute_distill_reweighted_advantages,
    _compute_teacher_kl_loss,
    _select_chosen_logprobs,
)
from .grpo import _compute_grpo_loss, _resolve_proximal_logp


def grpo_distill_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    config,
    current_version: int | None = None,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
):
    """Combined GRPO and distillation loss function.

    This function computes:
    1. GRPO loss using standard PPO objective with advantages
    2. Position-level GRPO loss using position_rewards:
       - Uses pre-computed logprobs (with gradients) for all candidates
       - Uses old logprobs from rollout for off-policy importance weighting
       - Computes: -E[importance_weight * reward * logp]

    Parameters
    ----------
    logprobs : torch.Tensor
        Log probabilities for all candidates [seq_len, num_candidates] or [seq_len].
        These are computed by the engine and have gradient information.
    entropy : torch.Tensor
        Entropy values for the current policy [seq_len].
    input_data : dict
        Dictionary containing:
        - logprobs: Old log probabilities from rollout (for importance sampling)
        - advantages: Advantage estimates
        - loss_mask: Mask indicating which positions to compute loss on
        - position_rewards: Position-wise rewards with candidate info
          Each PositionRewardInfo should have:
          - candidate_token_ids: list[int] - token IDs for all candidates
          - rewards: list[float] - rewards for each candidate
          - logprobs: list[float] - OLD logprobs from rollout (for importance weighting)
        - rl_loss_weight: Weight for GRPO loss (default: 1.0)
        - distill_loss_weight: Weight for distillation loss (default: 0.005)
    config : PPOActorConfig
        PPO actor configuration.
    current_version : int | None, optional
        Current weight version for version alignment.
    vocab_min_logits, vocab_max_logits : torch.Tensor | None
        Min/max logits for numerical stability (passed by engine).

    Returns
    -------
    torch.Tensor
        Combined loss (GRPO + position-level GRPO).
    """

    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()

    teacher_logprobs = input_data.get("teacher_logp")
    rl_loss_weight = input_data.get("rl_loss_weight", 1.0)
    distill_loss_weight = input_data.get("distill_loss_weight", 0.005)
    distill_kl_mode = getattr(config, "distill_kl_mode", "reverse_kl")
    distill_loss_mode = input_data.get(
        "distill_loss_mode", getattr(config, "distill_loss_mode", "evidence")
    )

    # Determine prompt length per sample from loss_mask (0 = prompt, 1 = output)
    if loss_mask.dim() > 1:
        # [batch, seq_len] -> per-sample prompt_len
        first_true = loss_mask.bool().cumsum(dim=1) == 1
        prompt_lens = first_true.int().argmax(dim=1).tolist()
    else:
        # [seq_len] -> single sample
        prompt_len = (loss_mask.bool().cumsum(dim=0) == 1).int().argmax(dim=0).item()
        prompt_lens = [prompt_len]

    prox_logp_gt = input_data.get("prox_logp")
    entropy = entropy.detach()
    chosen_logprobs = _select_chosen_logprobs(logprobs, loss_mask)

    distill_stat = None
    distill_evidence_stat = None

    if (
        teacher_logprobs is not None
        and distill_loss_mode in {"evidence", "rlsd"}
        and rl_loss_weight != 0
    ):
        distill_eps = input_data.get(
            "distill_eps_clip", getattr(config, "distill_eps_clip", None)
        )
        distill_lambda = input_data.get(
            "distill_mixing_coeff", getattr(config, "distill_mixing_coeff", 1.0)
        )
        reweighted_advantages, evidence_stat = _compute_distill_reweighted_advantages(
            advantages=advantages,
            student_context_logprobs=old_logp,
            teacher_logprobs=teacher_logprobs,
            loss_mask=loss_mask,
            prompt_lens=prompt_lens,
            input_data=input_data,
            eps_clip=distill_eps,
            mixing_coeff=distill_lambda,
        )

        coeffs = _resolve_proximal_logp(
            prox_logp_gt=prox_logp_gt,
            prox_logp_method=getattr(config, "prox_logp_method", "recompute"),
            old_logp=old_logp,
            logprobs=chosen_logprobs.detach(),
            versions=input_data.get("versions"),
            current_version=current_version,
        )

        loss, stat = _compute_grpo_loss(
            logprobs=chosen_logprobs,
            old_logp=old_logp,
            advantages=reweighted_advantages,
            eps_clip=config.eps_clip,
            eps_clip_higher=config.eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=config.c_clip,
            proximal_logprobs=coeffs,
            rejection_sampling=getattr(config, "rejection_sampling", None),
            importance_sampling_level=config.importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
        loss = rl_loss_weight * loss
        distill_evidence_stat = evidence_stat | {
            "advantage": reweighted_advantages.detach()
        }
    elif rl_loss_weight == 0 and teacher_logprobs is not None:
        # DISTILL mode: only teacher KL loss, no GRPO loss.
        teacher_kl_loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=prompt_lens,
            input_data=input_data,
            distill_kl_mode=distill_kl_mode,
        )
        loss = distill_loss_weight * teacher_kl_loss
        distill_stat = teacher_kl_loss.detach()
        stat = {
            "loss": torch.zeros_like(loss),
            "clip_mask": torch.zeros_like(loss_mask),
            "dual_clip_mask": torch.zeros_like(loss_mask),
            "importance_weight": torch.zeros_like(chosen_logprobs.float()),
            "approx_kl": torch.zeros_like(chosen_logprobs.float()),
        }
    else:
        coeffs = _resolve_proximal_logp(
            prox_logp_gt=prox_logp_gt,
            prox_logp_method=getattr(config, "prox_logp_method", "recompute"),
            old_logp=old_logp,
            logprobs=chosen_logprobs.detach(),
            versions=input_data.get("versions"),
            current_version=current_version,
        )

        loss, stat = _compute_grpo_loss(
            logprobs=chosen_logprobs,
            old_logp=old_logp,
            advantages=advantages,
            eps_clip=config.eps_clip,
            eps_clip_higher=config.eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=config.c_clip,
            proximal_logprobs=coeffs,
            rejection_sampling=getattr(config, "rejection_sampling", None),
            importance_sampling_level=config.importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )

        if teacher_logprobs is not None:
            teacher_kl_loss = _compute_teacher_kl_loss(
                teacher_logprobs=teacher_logprobs,
                logprobs=logprobs,
                loss_mask=loss_mask,
                prompt_lens=prompt_lens,
                input_data=input_data,
                distill_kl_mode=distill_kl_mode,
            )
            loss = rl_loss_weight * loss + distill_loss_weight * teacher_kl_loss
            distill_stat = teacher_kl_loss.detach()

    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    if distill_stat is not None:
        # Expand distill_stat to match the shape of loss_mask for stats_tracker.
        # Use tensor directly to avoid GPU-CPU sync from .item().
        distill_loss_expanded = torch.full(
            loss_mask.shape,
            distill_stat,
            dtype=torch.float32,
            device=loss_mask.device,
        )
        stats_tracker.stat(
            distill_loss=distill_loss_expanded,
            denominator="n_valid_tokens",
        )

    if distill_evidence_stat is not None:
        stats_tracker.stat(
            distill_delta=distill_evidence_stat["delta"],
            distill_evidence_weight=distill_evidence_stat["evidence_weight"],
            distill_credit_weight=distill_evidence_stat["credit_weight"],
            distill_advantage=distill_evidence_stat["advantage"],
            denominator="n_valid_tokens",
        )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=chosen_logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )

    return loss

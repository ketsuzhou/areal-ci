"""Clip-cov PPO loss functions.

Implements covariance-aware PPO clipping from
https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

Adapted to AReaL conventions (loss_mask, proximal_logprobs, stat dict format).
"""

from __future__ import annotations

import torch


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean over valid (masked) elements."""
    return (x * mask.float()).sum() / (mask.float().sum() + 1e-8)


def clip_cov_ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    clip_ratio: float = 0.0002,
    clip_cov_lb: float = 1.0,
    clip_cov_ub: float = 5.0,
) -> tuple[torch.Tensor, dict]:
    """Covariance-aware PPO actor loss.

    Extends standard PPO clipping with gradient masking for tokens whose
    advantage-logprob covariance falls within [clip_cov_lb, clip_cov_ub].
    Tokens already clipped by standard PPO are excluded from cov selection.

    Args:
        logprobs: Current policy log-probabilities.
        proximal_logprobs: Proximal policy log-probabilities (for PPO ratio).
        old_logprobs: Behavior policy log-probabilities (unused here, kept for API compat).
        advantages: Advantage estimates.
        eps_clip: PPO clipping parameter.
        loss_mask: Boolean mask for valid tokens.
        eps_clip_higher: Asymmetric higher clipping bound.
        c_clip: Dual clipping parameter (must be > 1.0).
        clip_ratio: Fraction of valid tokens to zero via cov clipping.
        clip_cov_lb: Lower bound of covariance range for candidate selection.
        clip_cov_ub: Upper bound of covariance range for candidate selection.

    Returns:
        Tuple of (loss_scalar, stat_dict) compatible with AReaL's ppo_actor_loss_fn.
    """
    loss_mask_count = loss_mask.count_nonzero()
    if loss_mask_count == 0:
        loss_mask_count = 1

    # Compute ratio (AReaL convention: masked ratio is 0)
    mask_float = loss_mask.float()
    ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0)

    # Standard PPO clipped loss
    eps_higher = eps_clip if eps_clip_higher is None else eps_clip_higher
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_higher)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio

    # Identify tokens clipped by standard PPO (where clipping increases the loss)
    clip_by_origin = (pg_losses2 > pg_losses1) & loss_mask
    pg_loss = torch.max(pg_losses1, pg_losses2)

    # Dual clipping
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_by_origin)

    # Compute per-token covariance between advantages and logprobs
    adv_mean = _masked_mean(advantages, mask_float)
    logp_mean = _masked_mean(logprobs.detach(), mask_float)
    cov_all = (advantages - adv_mean) * (logprobs.detach() - logp_mean)

    # Mask out tokens ineligible for cov selection
    cov_all = cov_all.clone()
    cov_all[~loss_mask] = float("-inf")
    cov_all[clip_by_origin] = float("-inf")

    # Select tokens within the covariance range
    candidates = (cov_all > clip_cov_lb) & (cov_all < clip_cov_ub) & loss_mask
    candidate_indices = torch.nonzero(candidates)

    # Randomly select up to clip_num tokens
    clip_num = max(int(clip_ratio * loss_mask_count), 1)

    if len(candidate_indices) > 0:
        perm = torch.randperm(len(candidate_indices))
        selected = candidate_indices[perm[: min(clip_num, len(candidate_indices))]]
    else:
        selected = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    # Build corr mask: 1 everywhere, 0 for selected tokens
    corr = torch.ones_like(advantages)
    if len(selected) > 0:
        corr[selected[:, 0], selected[:, 1]] = 0.0

    # Apply corr mask to loss
    pg_loss = pg_loss * corr

    # Aggregate loss (AReaL convention)
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0.0).sum() / loss_mask_count

    # Build stat dict (AReaL-compatible)
    clip_mask = clip_by_origin & loss_mask
    dual_clip_mask = (
        dual_clip_mask & loss_mask if c_clip is not None else dual_clip_mask
    )
    clip_cov_mask = (corr == 0) & loss_mask

    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
        clip_cov_mask=clip_cov_mask,
    )
    return pg_loss, stat


def clip_cov_grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    clip_ratio: float = 0.0002,
    clip_cov_lb: float = 1.0,
    clip_cov_ub: float = 5.0,
    prox_logp_method: str = "recompute",
    current_version: int | None = None,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """GRPO-compatible wrapper for clip-cov loss.

    Resolves proximal log-probs and delegates to clip_cov_ppo_actor_loss_fn.
    Does not support SAPO, decoupled loss, or teacher KL distillation.
    """
    from areal.trainer.ppo.actor import _resolve_proximal_logp

    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")

    entropy = entropy.detach()

    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    loss, stat = clip_cov_ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=prox_logp,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        clip_ratio=clip_ratio,
        clip_cov_lb=clip_cov_lb,
        clip_cov_ub=clip_cov_ub,
    )
    return loss, stat

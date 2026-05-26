"""Combined GRPO and Position-Level GRPO Loss Function.

This module provides a loss function that supports both standard GRPO training and
position-level GRPO using position_rewards from token_reward/cache.py.

The position-level GRPO loss treats candidates at each position as different samples
and computes reward-weighted log probability: -EGRPO[log p(·) * A(·)]

For multi-candidate training:
- Engine computes multi-candidate logprobs: [seq_len, num_candidates]
- Loss function receives logprobs directly (already has gradients)
- Uses old logprobs from rollout for off-policy importance weighting
"""

from __future__ import annotations

import torch

from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker
from areal.utils.logging import getLogger

logger = getLogger("DistillLoss")


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

    if rl_loss_weight == 0 and teacher_logprobs is not None:
        # DISTILL mode: only teacher KL loss, no GRPO loss.
        teacher_kl_loss = _compute_teacher_kl_loss(
            teacher_logprobs=teacher_logprobs,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=prompt_lens,
            input_data=input_data,
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
            prox_logp_method=getattr(config, "prox_clip", "recompute"),
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


def _select_chosen_logprobs(
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Select chosen-token logprobs from optional candidate dimensions.

    Assumes ``loss_mask`` is never expanded to match multi-candidate
    ``logprobs`` shape.  Standard shapes are ``[seq_len]`` or
    ``[batch, seq_len]``.
    """
    if logprobs.dim() == 1:
        return logprobs

    # Multi-candidate: logprobs has more dims than loss_mask.
    if logprobs.dim() > loss_mask.dim():
        if logprobs.dim() == 2:
            return logprobs[:, 0]
        if logprobs.dim() == 3:
            return logprobs[..., 0]

    # Same dims: single-candidate (return as-is).
    if logprobs.dim() == loss_mask.dim() and logprobs.dim() == 2:
        return logprobs

    raise ValueError(f"Unsupported logprobs shape for distill loss: {logprobs.shape}")


def _compute_teacher_kl_loss(
    teacher_logprobs: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int],
    input_data: dict | None = None,
) -> torch.Tensor:
    """Compute teacher KL distillation loss from batched teacher_logprobs tensor.

    teacher_logprobs is response-aligned with shape [batch, resp_len, max_candidates].
    Position i in the response maps to absolute sequence position prompt_len + i.

    Supports both batched (loss_mask 2D) and 1D packed (loss_mask 1D + cu_seqlens)
    formats for logprobs and loss_mask.
    """
    if teacher_logprobs.numel() == 0:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    terms: list[torch.Tensor] = []
    mask = loss_mask.bool()
    batch_size = teacher_logprobs.shape[0]
    max_resp = teacher_logprobs.shape[1]

    is_multi_candidate = logprobs.dim() > loss_mask.dim() or (
        logprobs.dim() == loss_mask.dim() and logprobs.shape != loss_mask.shape
    )

    # 1D packed format: use cu_seqlens for per-sequence boundaries
    cu_seqlens = (input_data or {}).get("cu_seqlens")
    if mask.dim() == 1 and cu_seqlens is not None:
        for b in range(len(cu_seqlens) - 1):
            if b >= batch_size:
                break
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            # Recompute prompt_len from loss_mask for this segment
            seg_mask = mask[start:end]
            pl = prompt_lens[b] if b < len(prompt_lens) else 0
            if seg_mask.any():
                pl = int(seg_mask.int().argmax().item())
            resp_len = (end - start) - pl
            n_pos = min(resp_len, max_resp)
            if n_pos == 0:
                continue

            if is_multi_candidate:
                num_cand = min(
                    teacher_logprobs.shape[2],
                    logprobs.shape[1] if logprobs.dim() == 2 else logprobs.shape[2],
                )
                student = logprobs[start + pl : start + pl + n_pos, :num_cand]
                teacher = teacher_logprobs[b, :n_pos, :num_cand]
                valid = teacher.abs().sum(dim=-1) > 1e-8
                if valid.any():
                    terms.append((student[valid] - teacher[valid]).reshape(-1))
            else:
                student = logprobs[start + pl : start + pl + n_pos]
                teacher = teacher_logprobs[b, :n_pos, 0]
                valid = teacher.abs() > 1e-8
                if valid.any():
                    terms.append((student[valid] - teacher[valid]).reshape(-1))

        if not terms:
            return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)
        return torch.cat(terms).mean()

    # Original batched format: loss_mask is 2D [batch, seq_len]
    is_batched = mask.dim() == 2 or logprobs.dim() >= 3

    for b in range(batch_size):
        pl = prompt_lens[b] if b < len(prompt_lens) else 0
        resp_len = mask[b, pl:].sum().item() if is_batched else mask[pl:].sum().item()
        n_pos = min(resp_len, max_resp)
        if n_pos == 0:
            continue

        if is_multi_candidate:
            num_cand = min(
                teacher_logprobs.shape[2],
                logprobs.shape[2] if logprobs.dim() == 3 else logprobs.shape[1],
            )
            if is_batched:
                student = logprobs[b, pl : pl + n_pos, :num_cand]
            else:
                student = logprobs[pl : pl + n_pos, :num_cand]
            teacher = teacher_logprobs[b, :n_pos, :num_cand]
            valid = teacher.abs().sum(dim=-1) > 1e-8
            if valid.any():
                terms.append((student[valid] - teacher[valid]).reshape(-1))
        else:
            if is_batched:
                student = logprobs[b, pl : pl + n_pos]
            else:
                student = logprobs[pl : pl + n_pos]
            teacher = teacher_logprobs[b, :n_pos, 0]
            valid = teacher.abs() > 1e-8
            if valid.any():
                terms.append((student[valid] - teacher[valid]).reshape(-1))

    if not terms:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    return torch.cat(terms).mean()


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor | None:
    """Resolve proximal log probabilities based on method."""
    if prox_logp_gt is not None:
        return prox_logp_gt

    if prox_logp_method == "recompute":
        return old_logp

    if versions is not None and current_version is not None:
        return logprobs[versions == current_version]

    return old_logp


def _compute_grpo_loss(
    logprobs: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_higher: float | None,
    loss_mask: torch.Tensor,
    c_clip: float | None,
    proximal_logprobs: torch.Tensor | None,
    rejection_sampling: object | None,
    importance_sampling_level: str,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO/PPO loss."""
    from areal.utils.functional import ppo_actor_loss_fn

    proximal_logp = old_logp if proximal_logprobs is None else proximal_logprobs

    return ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logp,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        rejection_sampling=rejection_sampling,
        importance_sampling_level=importance_sampling_level,
        cu_seqlens=cu_seqlens,
    )


def _compute_position_level_grpo_loss(
    position_rewards: list,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int] | int = 0,
) -> torch.Tensor:
    """Compute position-level GRPO loss using pre-computed multi-candidate logprobs.

    This function uses the logprobs computed by the engine (which have gradients)
    and combines them with rewards and old logprobs from rollout.

    Parameters
    ----------
    position_rewards : list
        List of PositionRewardInfo objects with:
        - candidate_token_ids: list[int] - token IDs for all candidates
        - rewards: list[float] - rewards for each candidate
        - logprobs: list[float] - OLD logprobs from rollout (for importance weighting)
    logprobs : torch.Tensor
        Current policy logprobs for all candidates [seq_len, num_candidates].
        These are computed by the engine and have gradient information.
    loss_mask : torch.Tensor
        Mask indicating which tokens to compute loss on.
    prompt_len : int
        Number of prompt tokens. PositionRewardInfo.position is 0-indexed
        from the first output token, so we add prompt_len to get the
        absolute position in the logprobs tensor.

    Returns
    -------
    torch.Tensor
        GRPO loss tensor (scalar).
    """
    if not position_rewards:
        return torch.tensor(0.0, dtype=torch.float32, device=loss_mask.device)

    # Gather valid positions and data into padded tensors
    positions = []
    reward_rows = []
    old_logprob_rows = []
    has_old_mask_rows = []
    max_candidates = logprobs.shape[1]

    for pr in position_rewards:
        if not pr.rewards:
            continue
        # Bug 3 fix: use per-sample prompt_len
        if isinstance(prompt_lens, list):
            pl = (
                prompt_lens[pr.sample_index]
                if pr.sample_index < len(prompt_lens)
                else 0
            )
        else:
            pl = prompt_lens
        position = pr.position + pl
        if position >= logprobs.shape[0]:
            logger.warning(
                "Skipping position %d + prompt_len=%d = %d: exceeds logprobs length %d",
                pr.position,
                pl,
                position,
                logprobs.shape[0],
            )
            continue
        if position < 0:
            continue
        num_candidates = min(len(pr.rewards), max_candidates)
        positions.append(position)
        reward_rows.append(pr.rewards[:num_candidates])
        if pr.logprobs and len(pr.logprobs) >= num_candidates:
            old_logprob_rows.append(pr.logprobs[:num_candidates])
            has_old_mask_rows.append([True] * num_candidates)
        else:
            old_logprob_rows.append([0.0] * num_candidates)
            has_old_mask_rows.append([False] * num_candidates)

    if not positions:
        return torch.tensor(0.0, dtype=torch.float32, device=loss_mask.device)

    n_pos = len(positions)
    device = logprobs.device
    positions_t = torch.tensor(positions, device=device)

    # Build padded tensors [n_pos, max_candidates]
    rewards_t = torch.zeros(n_pos, max_candidates, dtype=torch.float32, device=device)
    old_logprobs_t = torch.zeros(
        n_pos, max_candidates, dtype=torch.float32, device=device
    )
    candidate_mask = torch.zeros(n_pos, max_candidates, dtype=torch.bool, device=device)
    has_old_mask = torch.zeros(n_pos, max_candidates, dtype=torch.bool, device=device)

    for i in range(n_pos):
        num = len(reward_rows[i])
        rewards_t[i, :num] = torch.tensor(
            reward_rows[i], dtype=torch.float32, device=device
        )
        old_logprobs_t[i, :num] = torch.tensor(
            old_logprob_rows[i], dtype=torch.float32, device=device
        )
        candidate_mask[i, :num] = True
        has_old_mask[i, :num] = torch.tensor(
            has_old_mask_rows[i], dtype=torch.bool, device=device
        )

    # Get new logprobs for all positions at once [n_pos, max_candidates]
    new_logprobs = logprobs[positions_t, :]

    # Compute importance weights with clipping
    importance_weights = torch.ones_like(new_logprobs)
    iw = torch.exp(new_logprobs.detach() - old_logprobs_t).clamp(max=10.0)
    importance_weights = torch.where(has_old_mask, iw, importance_weights)

    # Compute GRPO advantages: normalize rewards within each position group
    num_valid = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1)
    reward_mean = (rewards_t * candidate_mask).sum(dim=1, keepdim=True) / num_valid

    # Unbiased std (match original torch.std behavior)
    var = ((rewards_t - reward_mean) ** 2 * candidate_mask).sum(dim=1, keepdim=True) / (
        num_valid - 1
    ).clamp(min=1)
    reward_std = torch.sqrt(var)
    reward_std = torch.where(num_valid > 1, reward_std, torch.zeros_like(reward_std))

    advantages = (rewards_t - reward_mean) / (reward_std + 1e-8)
    advantages = advantages * candidate_mask  # mask padding

    # Compute weighted loss with importance sampling
    weighted_advantages = importance_weights * advantages
    loss_per_position = -(weighted_advantages * new_logprobs).sum(
        dim=1
    ) / num_valid.squeeze(1)

    total_weight = importance_weights.sum(dim=1)
    loss_per_position = torch.where(
        total_weight > 0, loss_per_position / total_weight, loss_per_position
    )

    # Pad or truncate to match loss_mask output length
    output_len = loss_mask.sum().int()
    n_loss = loss_per_position.shape[0]
    if n_loss < output_len:
        padding = torch.zeros((output_len - n_loss), dtype=torch.float32, device=device)
        loss_per_position = torch.cat([loss_per_position, padding])
    elif n_loss > output_len:
        loss_per_position = loss_per_position[:output_len]

    grpo_loss = loss_per_position.sum() / loss_mask.sum().clamp(min=1).float()
    return grpo_loss

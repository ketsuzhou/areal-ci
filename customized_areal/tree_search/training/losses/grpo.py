"""GRPO, proximal-logprob, and position-level loss helpers."""

from __future__ import annotations

import torch

from areal.utils.logging import getLogger

logger = getLogger("DistillLoss")


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

# SPDX-License-Identifier: Apache-2.0
"""Combined VIMPO actor loss math.

VIMPO (Variational Implicit Multi-turn Policy Optimization) combines two
loss terms with different normalization denominators in a single backward:

1. **PPO token-mean numerator** (``ppo_sum``): standard PPO clipped
   surrogate, summed over valid predict positions.  The caller divides by
   ``valid_token_count`` (all-reduced across DP) to get a per-token mean.

2. **Terminal episode-mean numerator** (``value_sum``): half-squared
   residual between the per-episode summed terminal-value prediction
   ``beta * sum(logp_theta - logp_ref - KL)`` and the episode's centered
   reward target.  The caller divides by ``episode_count`` (all-reduced
   across DP) to get a per-episode mean.

``vimpo_loss_terms`` returns the raw numerators plus the two counts so the
engine can all-reduce the counts independently before scaling.  This is
necessary because the two means use *different* denominators, which the
generic ``train_batch`` single-``loss_weight_fn`` API cannot express.

Reference tensors (``vimpo_ref_sample_logp``, ``vimpo_candidate_kl``,
``advantages``, ``prox_logp``) are detached inside ``vimpo_loss_terms`` so
no gradient flows through them - the only differentiable path is the
current policy ``logprobs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "VIMPOLossTerms",
    "vimpo_loss_terms",
    "vimpo_loss_fn",
    "validate_vimpo_episode_consistency",
]


@dataclass(frozen=True)
class VIMPOLossTerms:
    """Numerators and counts for the combined VIMPO loss.

    The caller (engine) divides ``ppo_sum`` / ``valid_token_count`` and
    ``value_sum`` / ``episode_count`` using two separately all-reduced
    denominators, then combines them via ``vimpo_loss_fn``.

    Fields
    ------
    ppo_sum
        PPO clipped-surrogate numerator summed over valid predict
        positions (token-mean denominator: ``valid_token_count``).
    value_sum
        ``0.5 * residual.square().sum()`` over complete episodes
        (episode-mean denominator: ``episode_count``).
    valid_token_count
        Number of True entries in ``vimpo_predict_mask``.
    episode_count
        Number of complete episodes with at least one valid position.
    terminal_prediction
        Per-episode summed terminal-value prediction
        ``beta * sum(logp - logp_ref - KL)`` (shape ``[episode_count]``).
    terminal_target
        Per-episode centered reward target (shape ``[episode_count]``).
    terminal_residual
        ``terminal_prediction - terminal_target`` (shape ``[episode_count]``).
    """

    ppo_sum: torch.Tensor
    value_sum: torch.Tensor
    valid_token_count: torch.Tensor
    episode_count: torch.Tensor
    terminal_prediction: torch.Tensor
    terminal_target: torch.Tensor
    terminal_residual: torch.Tensor


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise ``ValueError`` if a floating-point tensor has non-finite values."""
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(
            f"VIMPO batch field {name!r} contains non-finite values (inf/nan)"
        )


def _validate_constant_per_row(
    name: str, tensor: torch.Tensor, mask: torch.Tensor
) -> None:
    """For 2D+ tensors, assert every row is constant over its VALID positions.

    VIMPO per-row metadata (``vimpo_episode_index``, ``vimpo_turn_index``,
    ``vimpo_centered_reward``, ``vimpo_expected_turn_count``) is broadcast to
    all valid positions in a row, but the batch is right-padded with zeros by
    ``concat_padded_tensors``. Constancy is therefore checked over the masked
    (valid predict) positions only: the first valid element is the row's
    value. A row whose valid values differ indicates a batching bug; raise
    early so the loss math doesn't silently average inconsistent values.
    """
    if tensor.ndim < 2:
        return
    for row in range(tensor.shape[0]):
        valid = tensor[row][mask[row]]
        if valid.numel() > 0 and not torch.all(valid == valid[0]):
            raise ValueError(
                f"VIMPO batch field {name!r} must be constant within each row"
            )


def validate_vimpo_episode_consistency(
    data: dict[str, torch.Tensor], mask: torch.Tensor
) -> None:
    """Validate per-episode consistency of a VIMPO batch.

    Shape-agnostic: works on the 2D raw batch ``[bs, seq_len]`` (called
    from ``MultiCandidateFSDPEngine._validate_vimpo_batch`` before
    ``optimizer_zero_grad``) and on the 1D packed tensors ``[total_tokens]``
    (called from ``vimpo_loss_terms`` inside the forward-backward callback).
    The boolean-mask operations (``mask & (episode_indices == episode)``,
    ``masked_select``, ``torch.unique``) behave identically under both
    shapes.

    Performs ONLY the two per-episode consistency checks; it does NOT
    compute predictions/targets (those stay in ``vimpo_loss_terms``):

    1. **Turn-count completeness**: when both ``vimpo_turn_index`` and
       ``vimpo_expected_turn_count`` are present, every episode's distinct
       turn-index count must equal its expected turn count.
    2. **Centered-reward agreement**: within each episode, every valid
       position's ``vimpo_centered_reward`` must agree.

    Parameters
    ----------
    data
        VIMPO batch dict (must contain ``vimpo_episode_index`` and
        ``vimpo_centered_reward``; optionally ``vimpo_turn_index`` and
        ``vimpo_expected_turn_count``).
    mask
        Boolean predict mask matching the shape of
        ``data["vimpo_episode_index"]``.

    Raises
    ------
    ValueError
        If an episode is incomplete or has inconsistent centered rewards.
    """
    episode_indices = data["vimpo_episode_index"]
    has_turn_index = "vimpo_turn_index" in data
    has_expected_count = "vimpo_expected_turn_count" in data
    turn_index = data.get("vimpo_turn_index")
    expected_count = data.get("vimpo_expected_turn_count")

    for episode in torch.unique(episode_indices[mask], sorted=True):
        episode_mask = mask & (episode_indices == episode)

        # Validate complete turn count (when both fields are present).
        if has_turn_index and has_expected_count and turn_index is not None:
            turn_values = torch.unique(turn_index[episode_mask])
            expected = int(expected_count[episode_mask].reshape(-1)[0].item())
            if len(turn_values) != expected:
                raise ValueError(
                    f"episode {int(episode)} is incomplete: "
                    f"expected {expected} turns, got {len(turn_values)}"
                )

        # Centered reward must be constant within the episode.
        repeated_target = (
            data["vimpo_centered_reward"].masked_select(episode_mask).float()
        )
        if not torch.allclose(
            repeated_target, repeated_target[0].expand_as(repeated_target)
        ):
            raise ValueError(
                f"episode {int(episode)} has inconsistent centered rewards"
            )


def vimpo_loss_terms(
    logprobs: torch.Tensor,
    data: dict[str, torch.Tensor],
    *,
    beta: float,
    eps_clip: float,
    eps_clip_higher: float | None,
) -> VIMPOLossTerms:
    """Compute VIMPO loss numerators (PPO token-mean + terminal episode-mean).

    Parameters
    ----------
    logprobs
        Current-policy sampled-token log-probabilities (differentiable).
        Shape ``[B, S]`` (raw batch) or ``[total_tokens]`` (packed).
    data
        VIMPO batch dict.  Required keys: ``vimpo_predict_mask``,
        ``vimpo_episode_index``, ``vimpo_centered_reward``,
        ``vimpo_ref_sample_logp``, ``vimpo_candidate_kl``, ``advantages``,
        and at least one of ``prox_logp`` / ``logprobs`` (old/proximal).
        Optional: ``vimpo_turn_index`` + ``vimpo_expected_turn_count``
        (trigger per-episode completeness validation).
    beta
        Terminal-value temperature scaling the policy-reference KL.
    eps_clip
        PPO clip ratio lower/upper bound (``1 - eps_clip`` / ``1 + eps_clip``).
    eps_clip_higher
        Decoupled upper clip bound.  ``None`` falls back to ``eps_clip``.

    Returns
    -------
    VIMPOLossTerms
        Numerators and counts; the caller scales and combines them.

    Raises
    ------
    ValueError
        If inputs are non-finite, per-row metadata is inconsistent, an
        episode is incomplete (fewer turns than ``vimpo_expected_turn_count``),
        centered rewards vary within an episode, or there are zero valid
        tokens / episodes.
    """
    mask = data["vimpo_predict_mask"].bool()

    # --- Validate finite inputs ---
    _validate_finite("logprobs", logprobs)
    for key in (
        "vimpo_ref_sample_logp",
        "vimpo_candidate_kl",
        "advantages",
        "vimpo_centered_reward",
    ):
        if key in data:
            _validate_finite(key, data[key])
    for key in ("prox_logp", "logprobs"):
        if key in data:
            _validate_finite(key, data[key])

    # --- Validate constant-per-row metadata over valid positions (2D raw
    # batch only; padded columns are zero-filled and excluded) ---
    for key in (
        "vimpo_episode_index",
        "vimpo_turn_index",
        "vimpo_centered_reward",
        "vimpo_expected_turn_count",
    ):
        if key in data:
            _validate_constant_per_row(key, data[key], mask)

    # --- Validate nonzero token count ---
    valid_token_count = mask.sum()
    if valid_token_count.item() == 0:
        raise ValueError(
            "VIMPO batch has no valid predict positions "
            "(vimpo_predict_mask is all False)"
        )

    # --- PPO clipped-surrogate numerator (token-mean) ---
    # Match ppo_actor_loss_fn clipping: ratio = exp(logp - proximal),
    # clipped_ratio = clamp(ratio, 1-eps, 1+eps_higher), take the min of
    # (ratio*adv, clipped_ratio*adv) and negate.  The numerator is the sum
    # over valid positions; the caller divides by valid_token_count.
    old = data.get("prox_logp", data["logprobs"]).detach().float()
    advantage = data["advantages"].detach().float()
    ratio = torch.exp(logprobs.float() - old)
    lower = 1.0 - eps_clip
    upper = 1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher)
    ppo_token = -torch.minimum(ratio * advantage, ratio.clamp(lower, upper) * advantage)
    ppo_sum = ppo_token.masked_select(mask).sum()

    # --- Terminal episode-mean numerator ---
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    episode_indices = data["vimpo_episode_index"]

    # Validate per-episode consistency (shape-agnostic; also called from
    # ``_validate_vimpo_batch`` before ``optimizer_zero_grad`` so a partial
    # episode or inconsistent centered reward fails before any optimizer
    # mutation).
    validate_vimpo_episode_consistency(data, mask)

    for episode in torch.unique(episode_indices[mask], sorted=True):
        episode_mask = mask & (episode_indices == episode)

        # Per-episode terminal-value prediction:
        #   beta * sum(logp_theta - logp_ref - KL) over valid positions.
        prediction = beta * (
            logprobs.float()
            - data["vimpo_ref_sample_logp"].detach().float()
            - data["vimpo_candidate_kl"].detach().float()
        )
        predictions.append(prediction.masked_select(episode_mask).sum())

        # Per-episode target: the centered reward is broadcast to all
        # positions in the episode; consistency validated above.
        repeated_target = (
            data["vimpo_centered_reward"].masked_select(episode_mask).float()
        )
        targets.append(repeated_target[0].detach())

    # --- Validate nonzero episode count ---
    episode_count = len(predictions)
    if episode_count == 0:
        raise ValueError(
            "VIMPO batch has no complete episodes (no valid positions in any episode)"
        )

    prediction_tensor = torch.stack(predictions)
    target_tensor = torch.stack(targets)
    residual = prediction_tensor - target_tensor
    value_sum = 0.5 * residual.square().sum()

    return VIMPOLossTerms(
        ppo_sum=ppo_sum,
        value_sum=value_sum,
        valid_token_count=valid_token_count,
        episode_count=torch.tensor(episode_count, device=logprobs.device),
        terminal_prediction=prediction_tensor,
        terminal_target=target_tensor,
        terminal_residual=residual,
    )


def vimpo_loss_fn(
    terms: VIMPOLossTerms,
    *,
    actor_coeff: float,
    value_loss_weight: float,
    global_valid_tokens: torch.Tensor,
    global_episodes: torch.Tensor,
    dp_size: int = 1,
) -> torch.Tensor:
    """Scale VIMPO loss terms into a single backward-able scalar.

    Combines the PPO token-mean numerator and the terminal episode-mean
    numerator using two *separate* distributed denominators::

        loss = dp_size * (
            actor_coeff * terms.ppo_sum / global_valid_tokens
            + value_loss_weight * terms.value_sum / global_episodes
        )

    The ``dp_size`` factor compensates for FSDP's gradient all-reduce
    averaging: each DP rank computes a local loss, and the all-reduce
    divides gradients by ``dp_size``.  Multiplying by ``dp_size`` here
    ensures the effective gradient matches the mathematical loss.

    Parameters
    ----------
    terms
        Output of :func:`vimpo_loss_terms`.
    actor_coeff
        PPO actor loss coefficient (multiplies the token-mean term).
    value_loss_weight
        Terminal value loss coefficient (multiplies the episode-mean term).
    global_valid_tokens
        All-reduced valid token count (float64 scalar).
    global_episodes
        All-reduced complete episode count (float64 scalar).
    dp_size
        Data-parallel world size (gradient-compensation factor).
    """
    # Cast denominators to the terms' dtype so the loss scalar matches the
    # model parameter dtype (the all-reduced denominators are float64 for
    # numerical stability across DP ranks).
    dtype = terms.ppo_sum.dtype
    return dp_size * (
        actor_coeff * terms.ppo_sum / global_valid_tokens.to(dtype)
        + value_loss_weight * terms.value_sum / global_episodes.to(dtype)
    )

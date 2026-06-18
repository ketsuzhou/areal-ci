# SPDX-License-Identifier: Apache-2.0
"""Generative-critic soft-regression loss and training-batch builder.

The shared model is trained as a critic by an *expected-value soft regression*:
at the answer position the model produces a next-token distribution; we restrict
it to the integer-label tokens ``0..score_max``, renormalize, and compute the
expected normalized value ``v = Sum_i p_i * (i / score_max)``. The loss is
``MSE(v, target)`` where ``target = clamp(q_value / target_scale, 0, 1)`` and the
``q_value`` is the tree-stored MCTS value for the node.

The combined objective applied to the shared model is::

    loss = actor_loss + critic_loss_weight * critic_loss

Approximation note
------------------
A single answer position has one next-token distribution, so multi-token labels
(e.g. ``"10"``) are represented by their *leading* token. When two labels share a
leading token (``"1"`` and ``"10"`` for ``score_max=10``) the approximation
double-counts that token; use ``score_max=9`` for a collision-free scale if this
matters.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def expected_value_from_label_logits(
    answer_logits: torch.Tensor,
    leading_token_ids: list[int],
    score_max: int,
) -> torch.Tensor:
    """Differentiable expected value from answer-position logits.

    Parameters
    ----------
    answer_logits : torch.Tensor
        Logits at the answer position, shape ``[B, V]``.
    leading_token_ids : list[int]
        Leading token id for each label ``0..score_max`` (length ``score_max+1``),
        ordered by label.
    score_max : int
        Maximum integer label.

    Returns
    -------
    torch.Tensor
        Expected normalized value per sample, shape ``[B]`` in ``[0, 1]``.
    """
    if score_max < 1:
        raise ValueError(f"score_max must be >= 1, got {score_max}")
    if answer_logits.dim() != 2:
        raise ValueError(
            f"answer_logits must be 2D [B, V], got shape {tuple(answer_logits.shape)}"
        )
    if len(leading_token_ids) != score_max + 1:
        raise ValueError(
            f"leading_token_ids must have length score_max+1={score_max + 1}, "
            f"got {len(leading_token_ids)}"
        )

    logprobs = F.log_softmax(answer_logits.float(), dim=-1)  # [B, V]
    idx = torch.tensor(leading_token_ids, device=answer_logits.device, dtype=torch.long)
    label_logprobs = logprobs[:, idx]  # [B, L]
    label_probs = torch.softmax(label_logprobs, dim=-1)  # renormalize over labels
    scores = (
        torch.arange(score_max + 1, device=answer_logits.device, dtype=torch.float32)
        / score_max
    )  # value of each label
    return (label_probs * scores).sum(dim=-1)  # [B]


def critic_softreg_loss_fn(
    answer_logits: torch.Tensor,
    targets: torch.Tensor,
    leading_token_ids: list[int],
    score_max: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MSE between the expected critic value and the (clamped) q_value target.

    ``targets`` must already be normalized to ``[0, 1]`` (see
    :func:`build_critic_training_batch`).
    """
    value = expected_value_from_label_logits(
        answer_logits, leading_token_ids, score_max
    )
    targets = targets.to(value.dtype)
    loss = F.mse_loss(value, targets)
    stats = {
        "critic_value": value.detach(),
        "critic_target": targets.detach(),
        "critic_loss": loss.detach(),
    }
    return loss, stats


def expected_value_from_label_logprobs(
    candidate_logprobs: torch.Tensor,
    score_max: int,
) -> torch.Tensor:
    """Differentiable expected value from per-candidate logprobs.

    This is the variant consumed by the multi-candidate engine path: the engine
    gathers the logprobs of the digit-label tokens ``0..score_max`` at the answer
    position, producing ``candidate_logprobs`` of shape ``[B, score_max+1]``
    (ordered by label). The logprobs are renormalized into a categorical over the
    labels and the expected normalized value is returned.
    """
    if candidate_logprobs.dim() != 2:
        raise ValueError(
            "candidate_logprobs must be 2D [B, score_max+1], got shape "
            f"{tuple(candidate_logprobs.shape)}"
        )
    if candidate_logprobs.shape[1] != score_max + 1:
        raise ValueError(
            f"candidate_logprobs second dim must be score_max+1={score_max + 1}, "
            f"got {candidate_logprobs.shape[1]}"
        )
    label_probs = torch.softmax(candidate_logprobs.float(), dim=-1)
    scores = (
        torch.arange(
            score_max + 1, device=candidate_logprobs.device, dtype=torch.float32
        )
        / score_max
    )
    return (label_probs * scores).sum(dim=-1)


def critic_softreg_loss_from_logprobs(
    candidate_logprobs: torch.Tensor,
    targets: torch.Tensor,
    score_max: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft-regression MSE loss from per-candidate logprobs (engine path)."""
    value = expected_value_from_label_logprobs(candidate_logprobs, score_max)
    targets = targets.to(value.dtype)
    loss = F.mse_loss(value, targets)
    stats = {
        "critic_value": value.detach(),
        "critic_target": targets.detach(),
        "critic_loss": loss.detach(),
    }
    return loss, stats


def combined_actor_critic_loss(
    actor_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    critic_loss_weight: float,
) -> torch.Tensor:
    """Combine the actor and critic losses for the shared-model update."""
    return actor_loss + critic_loss_weight * critic_loss


def build_critic_training_batch(
    client: Any,
    nodes: list[Any],
    *,
    targets: list[float] | None = None,
    tree_store: Any | None = None,
    target_scale: float = 1.0,
    score_max: int | None = None,
) -> dict[str, Any]:
    """Build a critic regression batch from episode nodes.

    Parameters
    ----------
    client : CriticValueClient
        Provides ``build_prompt_ids`` and ``_digit_ids`` (the same prompt used at
        rollout, for train/rollout consistency).
    nodes : list[Node]
        Nodes to score.
    targets : list[float] | None
        Raw q_value targets per node. If None, read from ``tree_store`` via
        ``get_q_value(node_id)`` falling back to ``node.outcome_reward``.
    tree_store : MCTSTreeStore | None
        Source of q_values when ``targets`` is None.
    target_scale : float
        Divisor applied to q_values before clamping to ``[0, 1]``.
    score_max : int | None
        Overrides the client's ``score_max`` (defaults to the client's value).

    Returns
    -------
    dict with keys:
        - ``prompt_ids``: list[list[int]] critic prompts.
        - ``answer_pos``: LongTensor[B], index whose next-token distribution is
          the score (the last prompt position).
        - ``targets``: FloatTensor[B] clamped to [0, 1].
        - ``leading_token_ids``: list[int] length score_max+1.
    """
    if target_scale <= 0:
        raise ValueError(f"target_scale must be > 0, got {target_scale}")
    sm = score_max if score_max is not None else client.score_max

    if targets is None:
        targets = []
        for node in nodes:
            node_id = getattr(node, "node_id", "")
            q = None
            if tree_store is not None and node_id:
                q = tree_store.get_q_value(node_id)
            if q is None:
                q = getattr(node, "outcome_reward", 0.0)
            targets.append(float(q))

    if len(targets) != len(nodes):
        raise ValueError(f"targets length {len(targets)} != nodes length {len(nodes)}")

    prompt_ids: list[list[int]] = []
    answer_pos: list[int] = []
    clamped_targets: list[float] = []
    for node, q in zip(nodes, targets):
        ids = list(client.build_prompt_ids(node))
        if not ids:
            raise ValueError("critic prompt produced no tokens")
        prompt_ids.append(ids)
        answer_pos.append(len(ids) - 1)
        clamped_targets.append(min(1.0, max(0.0, q / target_scale)))

    leading_token_ids = [client._digit_ids[i][0] for i in range(sm + 1)]

    return {
        "prompt_ids": prompt_ids,
        "answer_pos": torch.tensor(answer_pos, dtype=torch.long),
        "targets": torch.tensor(clamped_targets, dtype=torch.float32),
        "leading_token_ids": leading_token_ids,
    }

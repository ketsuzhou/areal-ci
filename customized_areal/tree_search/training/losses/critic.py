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


# ---------------------------------------------------------------------------
# Unified TD / MC critic target
# ---------------------------------------------------------------------------
#
# The critic regression target is a convex blend of two estimators of the same
# state value ``v_phi(s_t)``, both expressed in normalized ``[0, 1]`` space::
#
#     y_mc_t = q_mcts(s_t) / target_scale                          # Monte-Carlo (MCTS)
#     y_td_t = (Sum_{k<n} gamma^k r_{t+k}) / target_scale
#              + gamma^n * v(s_{t+n})                               # n-step bootstrap
#     y_t    = (1 - w) * y_td_t + w * y_mc_t   (clamped to [0, 1])
#
# ``w`` (``mc_weight``) recovers the previous behavior exactly at ``w = 1`` (pure
# MCTS target) and yields pure n-step TD at ``w = 0``. ``w`` may be a fixed float
# or determined automatically per node via :class:`AdaptiveMCWeight`.
#
# Rewards are sparse: only the terminal turn of an episode carries
# ``outcome_reward``; the terminal bootstrap is ``v(s_{T+1}) = 0``. The bootstrap
# value ``v(s_{t+n})`` is the critic's stored ``Node.value`` (a constant target,
# i.e. semi-gradient TD).


def _episode_groups(nodes: list[Any]) -> dict[tuple[str, str], list[Any]]:
    """Group nodes by ``(query_id, episode_id)``.

    Nodes without an ``episode_id`` are treated as standalone single-turn
    episodes keyed by their ``node_id`` (mirrors ``GAEAdvantageComputer``).
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    for n in nodes:
        node_id = getattr(n, "node_id", None)
        if node_id is None:
            continue
        query_id = getattr(n, "query_id", "") or ""
        ep_id = getattr(n, "episode_id", "") or node_id
        groups.setdefault((query_id, ep_id), []).append(n)
    return groups


class AdaptiveMCWeight:
    """Automatically determine the MC/TD blend weight per node.

    The weight follows an inverse-variance (reliability) rule::

        mc_weight_t = N_t * eps2 / (N_t * eps2 + c)

    where ``N_t`` is the node's MCTS visit count (reliability of the MC target,
    which grows with visits) and ``eps2`` is an EMA of the critic regression MSE
    (the TD bootstrap is unreliable while the critic is inaccurate). Limits:

    * ``eps2`` large (critic still bad) -> weight -> 1 (rely on MCTS / MC).
    * ``eps2`` small (critic mature)    -> weight -> 0 (rely on TD bootstrap).
    * ``N_t`` large (well-visited node) -> weight -> 1 (MCTS Q reliable).
    * ``N_t`` small                     -> weight -> 0 (bootstrap instead).

    During ``warmup_steps`` (before the critic-error EMA is meaningful) the
    weight is forced to ``1.0`` so a randomly-initialized critic never poisons
    the target. The weight is clamped to ``[w_min, w_max]`` so neither estimator
    is ever fully discarded.

    Note
    ----
    ``update_critic_error`` must be fed the training-side critic MSE to close the
    loop. When that feedback is unavailable (e.g. the target is built in a
    different process than the critic loss), ``eps2`` stays at its initial value
    and the controller degenerates to a pure visit-count rule
    ``N_t / (N_t + c / eps2_init)``.
    """

    def __init__(
        self,
        c: float = 4.0,
        ema_beta: float = 0.95,
        warmup_steps: int = 50,
        w_min: float = 0.05,
        w_max: float = 0.95,
        eps2_init: float = 1.0,
    ) -> None:
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not 0.0 <= w_min <= w_max <= 1.0:
            raise ValueError(
                f"require 0 <= w_min <= w_max <= 1, got w_min={w_min}, w_max={w_max}"
            )
        if eps2_init <= 0:
            raise ValueError(f"eps2_init must be > 0, got {eps2_init}")
        self.c = c
        self.ema_beta = ema_beta
        self.warmup_steps = warmup_steps
        self.w_min = w_min
        self.w_max = w_max
        self._eps2 = eps2_init
        self._steps = 0

    def update_critic_error(self, critic_mse: float) -> None:
        """Feed the observed critic regression MSE (one critic step)."""
        mse = float(critic_mse)
        if mse < 0:
            raise ValueError(f"critic_mse must be >= 0, got {mse}")
        self._eps2 = self.ema_beta * self._eps2 + (1.0 - self.ema_beta) * mse
        self._steps += 1

    def live_critic_error_var(self) -> float | None:
        """Current EMA of the critic regression MSE, or ``None`` if not yet fed.

        This is the critic's *error* variance ``E[(v_theta - V)^2]`` -- an
        estimator-error quantity in the same units as the MC mean's sampling
        variance ``var_mc``. ``HybridGAEAdvantageComputer`` consumes it as
        ``var_theta`` for a unit-consistent inverse-variance blend. Returns
        ``None`` until ``update_critic_error`` has been called at least once, so
        callers can fall back to a static prior rather than the uninformative
        ``eps2_init``.
        """
        if self._steps <= 0:
            return None
        return self._eps2

    def weight(self, visit_count: int) -> float:
        """Per-node MC weight in ``[w_min, w_max]`` (1.0 during warmup)."""
        if self._steps < self.warmup_steps:
            return 1.0
        n = max(int(visit_count), 1)
        w = (n * self._eps2) / (n * self._eps2 + self.c)
        return min(self.w_max, max(self.w_min, w))


def compute_critic_targets(
    nodes: list[Any],
    *,
    tree_store: Any | None = None,
    mc_weight: float | AdaptiveMCWeight | None = 1.0,
    n_steps: int = 1,
    gamma: float = 1.0,
    target_scale: float = 1.0,
    judge_beta: float = 0.0,
    judge_score_max: int = 10,
) -> list[float]:
    """Unified critic regression targets in ``[0, 1]``, aligned to ``nodes`` order.

    Blends the MCTS Monte-Carlo target with an n-step bootstrapped TD target::

        y_t = (1 - w) * y_td_t + w * y_mc_t    (clamped to [0, 1])

    Parameters
    ----------
    nodes : list[Node]
        Nodes to score. Grouped into episodes internally for the TD horizon.
    tree_store : MCTSTreeStore | None
        Source of MCTS ``q_value`` (MC target) and ``_visit_counts`` (adaptive
        weighting). When None, the MC target falls back to ``node.outcome_reward``
        and adaptive weighting uses a visit count of 1.
    mc_weight : float | AdaptiveMCWeight | None
        Weight on the MC target. ``1.0`` (or ``None``) reproduces the pure-MCTS
        target. ``0.0`` is pure n-step TD. An :class:`AdaptiveMCWeight` computes a
        per-node weight from MCTS visit counts and critic error.
    n_steps : int
        Horizon of the TD component (``1`` == one-step TD).
    gamma : float
        Discount factor.
    target_scale : float
        Divisor applied to ``q_value`` and ``outcome_reward`` before blending.

    Returns
    -------
    list[float]
        Targets clamped to ``[0, 1]``, in the same order as ``nodes``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if target_scale <= 0:
        raise ValueError(f"target_scale must be > 0, got {target_scale}")

    adaptive = isinstance(mc_weight, AdaptiveMCWeight)
    if not adaptive:
        fixed_w = 1.0 if mc_weight is None else float(mc_weight)
        if not 0.0 <= fixed_w <= 1.0:
            raise ValueError(f"mc_weight must be in [0, 1], got {fixed_w}")

    target_by_id: dict[str, float] = {}
    for group in _episode_groups(nodes).values():
        ordered = sorted(group, key=lambda x: getattr(x, "turn_idx", 0))
        n_turns = len(ordered)
        values = [float(getattr(x, "value", 0.0) or 0.0) for x in ordered]
        # Sparse reward (terminal-only) unless LLM-judge shaping is enabled, in
        # which case the per-turn reward becomes dense. The helper falls back to
        # the sparse array when no judge signal exists, so judge_beta == 0 keeps
        # the target byte-for-byte unchanged.
        if judge_beta > 0.0 and tree_store is not None:
            from customized_areal.tree_search.core.process_reward import (
                build_episode_process_rewards,
            )

            dense = build_episode_process_rewards(
                ordered,
                tree_store,
                beta=judge_beta,
                score_max=judge_score_max,
            )
            rewards = [r / target_scale for r in dense]
        else:
            rewards = [0.0] * n_turns
            if n_turns:
                rewards[-1] = float(getattr(ordered[-1], "outcome_reward", 0.0)) / (
                    target_scale
                )

        for t, node in enumerate(ordered):
            node_id = getattr(node, "node_id", "")

            # MC target: MCTS q_value, fall back to outcome_reward.
            q = None
            if tree_store is not None and node_id:
                q = tree_store.get_q_value(node_id)
            if q is None:
                q = getattr(node, "outcome_reward", 0.0)
            y_mc = float(q) / target_scale

            # n-step TD target with terminal bootstrap v(s_{T+1}) = 0.
            horizon = min(n_steps, n_turns - t)
            y_td = 0.0
            disc = 1.0
            for k in range(horizon):
                y_td += disc * rewards[t + k]
                disc *= gamma
            boot_idx = t + n_steps
            if boot_idx < n_turns:  # s_{t+n} exists and is non-terminal
                y_td += disc * values[boot_idx]  # disc == gamma ** n_steps here

            # Blend weight (per node when adaptive).
            if adaptive:
                visit_count = 1
                if tree_store is not None and node_id:
                    visit_count = tree_store._visit_counts.get(node_id, 1)
                w = mc_weight.weight(visit_count)
            else:
                w = fixed_w

            y = (1.0 - w) * y_td + w * y_mc
            target_by_id[node_id] = min(1.0, max(0.0, y))

    return [target_by_id.get(getattr(n, "node_id", ""), 0.0) for n in nodes]

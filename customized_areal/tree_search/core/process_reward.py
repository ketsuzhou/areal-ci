# SPDX-License-Identifier: Apache-2.0
# customized_areal/tree_search/core/process_reward.py
"""Dense per-turn process rewards from accumulated LLM-judge credit.

A larger judge model assigns each assistant turn an integer credit; those raw
scores accumulate per node in the tree store (one score per episode that
traverses a -- possibly shared -- node). This module converts the accumulated
scores for one ordered episode into a dense per-turn reward ``r_t`` that shapes
training on top of the verified terminal ``outcome_reward``.

Reward construction (per episode, turns ordered ascending by ``turn_idx``)::

    mean_raw_t = mean(judge_scores[node_id])      # mean across traversing episodes
    jbar_t     = mean_raw_t / sum_t mean_raw_t     # per-episode credit distribution
    r_t        = beta * jbar_t                      # intermediate turns (t < T)
    r_T        = (1 - beta) * outcome_reward + beta * jbar_T   # terminal turn

The episode return is::

    G = sum_t r_t = beta * sum_t jbar_t + (1 - beta) * outcome_reward
                  = beta + (1 - beta) * outcome_reward      (since sum_t jbar_t = 1)

which lies in ``[0, 1]`` for ``outcome_reward, beta in [0, 1]``. Re-normalizing
``mean_raw`` within the episode (rather than dividing by ``score_max``) keeps
``sum_t jbar_t == 1`` exactly even when shared nodes were averaged across
branched episodes, so the bound holds regardless of branching.

Fallback: when no turn in the episode has a usable judge signal
(``sum_t mean_raw_t == 0`` or every node is unjudged) the sparse construction is
returned -- intermediate ``0`` and terminal ``outcome_reward`` -- so behaviour is
byte-for-byte identical to the no-judge path and the verified outcome is never
down-weighted by ``(1 - beta)`` without a judge signal.
"""

from __future__ import annotations

from typing import Any


def build_episode_process_rewards(
    ordered_nodes: list[Any],
    tree_store: Any | None,
    *,
    beta: float,
    score_max: int,
) -> list[float]:
    """Dense per-turn rewards for one episode (nodes already ordered by turn).

    Parameters
    ----------
    ordered_nodes : list[Node]
        Episode turns ordered ascending by ``turn_idx`` (caller's responsibility).
    tree_store : MCTSTreeStore | None
        Source of accumulated judge scores via ``get_mean_judge_score(node_id)``.
        When None, the sparse fallback is always returned.
    beta : float
        Convex shaping weight in ``[0, 1]``. ``0`` reproduces the sparse reward.
    score_max : int
        Judge scale (used only for validation; normalization is per-episode).

    Returns
    -------
    list[float]
        Per-turn reward aligned to ``ordered_nodes``.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if score_max < 1:
        raise ValueError(f"score_max must be >= 1, got {score_max}")

    n = len(ordered_nodes)
    if n == 0:
        return []

    terminal_outcome = float(getattr(ordered_nodes[-1], "outcome_reward", 0.0))

    def _sparse() -> list[float]:
        rewards = [0.0] * n
        rewards[-1] = terminal_outcome
        return rewards

    # No shaping requested or no store -> sparse, identical to the legacy path.
    if beta <= 0.0 or tree_store is None:
        return _sparse()

    # Gather mean raw judge score per turn (None == unjudged node).
    mean_raw: list[float | None] = []
    for node in ordered_nodes:
        node_id = getattr(node, "node_id", "")
        m = tree_store.get_mean_judge_score(node_id) if node_id else None
        mean_raw.append(m)

    total = sum(m for m in mean_raw if m is not None)
    # No usable judge signal -> sparse fallback (do not down-weight the outcome).
    if total <= 0.0:
        return _sparse()

    rewards: list[float] = []
    for t, m in enumerate(mean_raw):
        jbar = (m / total) if m is not None else 0.0
        if t == n - 1:
            rewards.append((1.0 - beta) * terminal_outcome + beta * jbar)
        else:
            rewards.append(beta * jbar)
    return rewards

"""DAG advantage assembler -- the GAE-replaces-GRPO core (Phase 3, Task 9).

Torch-free orchestration that the trainer calls for DAG runs: take the DAG
nodes in global completion order (each carrying the critic value ``V_{t+1}`` in
``node.value`` and rewards in ``process_reward`` / ``outcome_reward``), run the
global joint-state GAE (``gae.compute_global_gae``), and return per-node
advantages + returns.

The trainer then (Task 8 helpers): broadcasts each node advantage onto that
turn's actor tokens (``critic_advantage.broadcast_node_advantages``) and
regresses the generative critic's expected-score value against the node returns
(``critic_advantage.critic_huber_loss``), combining
``actor_PG + critic_loss_weight * critic_value_loss`` on the shared trunk.

``explained_variance`` is the standard critic-quality metric to log (GRPO's
group baseline is gone, so value quality drives gradient variance).
"""

from __future__ import annotations

from dataclasses import dataclass

from customized_areal.tree_search.dag.gae import (
    compute_global_gae,
    events_from_nodes,
)


@dataclass(frozen=True)
class AssembledAdvantages:
    """Per-node GAE outputs keyed by ``node_id``."""

    advantages: dict[str, float]
    returns: dict[str, float]
    baseline_values: dict[str, float]


def assemble_node_advantages(
    ordered_nodes: list,
    *,
    initial_value: float,
    gamma: float,
    lam: float,
    terminal_bootstrap: float = 0.0,
) -> AssembledAdvantages:
    """Run global GAE over ``ordered_nodes`` (global completion order).

    ``ordered_nodes`` are duck-typed ``AgentRunNode``-like objects with
    ``node_id``, ``value`` (``V_{t+1}``), ``process_reward`` and
    ``outcome_reward``. Returns per-node advantages/returns/baselines.
    """
    events = events_from_nodes(ordered_nodes)
    results = compute_global_gae(
        events,
        initial_value=initial_value,
        gamma=gamma,
        lam=lam,
        terminal_bootstrap=terminal_bootstrap,
    )
    return AssembledAdvantages(
        advantages={r.node_id: r.advantage for r in results},
        returns={r.node_id: r.return_ for r in results},
        baseline_values={r.node_id: r.baseline_value for r in results},
    )


def explained_variance(predictions: list[float], targets: list[float]) -> float:
    """``1 - Var(target - pred) / Var(target)`` -- critic-quality metric.

    Returns 1.0 for perfect prediction, ~0.0 for predicting the target mean,
    and can be negative for worse-than-mean predictions. Returns 0.0 when the
    targets have zero variance (degenerate).
    """
    n = len(targets)
    if n == 0 or len(predictions) != n:
        raise ValueError("predictions and targets must be equal, non-empty lengths")
    mean_t = sum(targets) / n
    var_t = sum((t - mean_t) ** 2 for t in targets) / n
    if var_t == 0.0:
        return 0.0
    var_resid = sum((t - p) ** 2 for t, p in zip(targets, predictions)) / n
    return 1.0 - var_resid / var_t


__all__ = [
    "AssembledAdvantages",
    "assemble_node_advantages",
    "explained_variance",
]

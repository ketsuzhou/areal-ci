"""DAG-aware reward backup (Phase 3, load-bearing).

Distributes a terminal verifier reward along DAG edges so sub-agent runs
that contributed to a successful root get credit. Fan-in joins get explicit
per-agent credit from the caller — no fixed sum/mean/max aggregation
(spec §2 decision 8, §5.4).

Uses stdlib :mod:`logging` so the package stays importable without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from customized_areal.tree_search.agents.execution_dag import ExecutionDAG

if TYPE_CHECKING:
    # Node is referenced only in the ``branch_backup`` annotation; import under
    # TYPE_CHECKING to keep this module importable without the tree_store/torch
    # stack and to avoid a circular import.
    from customized_areal.tree_search.core.tree_store import Node

logger = logging.getLogger("DagBackup")


@dataclass(frozen=True)
class CreditAssignment:
    """Explicit per-agent credit at a fan-in join (spec §2 decision 8).

    The caller (verifier or credit assigner) decides how much each incoming
    agent contributed; this module does not apply a sum/mean/max rule.
    """

    per_node: dict[str, float]


def distribute_reward_over_dag(
    dag: ExecutionDAG,
    *,
    terminal_reward: float,
    terminal_node_id: str,
    fan_in_credit: CreditAssignment | None = None,
) -> dict[str, float]:
    """Distribute ``terminal_reward`` backward along DAG edges.

    Returns a ``{node_id: credit}`` map. The terminal node gets the full
    reward; each ancestor along an incoming edge gets its child's full credit
    (no attenuation). Fan-in joins (multiple parents) consume
    ``fan_in_credit`` if provided — each parent's credit is set to its
    explicit share multiplied by the node's credit; otherwise the node's
    reward is split equally among parents as a default.
    """
    if terminal_node_id not in dag:
        raise KeyError(f"terminal node {terminal_node_id!r} not in DAG")

    credit: dict[str, float] = {nid: 0.0 for nid in dag.event_ids()}
    credit[terminal_node_id] = terminal_reward

    # Walk backward from the terminal node. For each node, find its parents
    # (incoming edges) and propagate credit.
    visited: set[str] = set()
    queue = [terminal_node_id]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        parents = [e.src for e in dag.edges if e.dst == nid]
        if not parents:
            continue
        if len(parents) > 1 and fan_in_credit is not None:
            # Explicit per-agent credit at the fan-in join.
            for p in parents:
                share = fan_in_credit.per_node.get(p, 0.0)
                credit[p] = max(credit[p], share * credit[nid])
        else:
            # Single parent or no explicit fan-in credit: split equally.
            share = credit[nid] / len(parents)
            for p in parents:
                credit[p] += share
        queue.extend(parents)

    return credit


def branch_backup(
    parent: Node,
    *,
    branch_return: float,
    discount: float,
    visit_count: int = 1,
) -> None:
    """MCTS-style value backup from a branch to its fork point (checkpoint node).

    Propagates a discounted branch return to ``parent`` (the checkpoint node the
    branch forked from): ``parent.value`` becomes the running mean of discounted
    branch returns over visits, and ``parent.visit_count`` accumulates. Used by
    tree-search branching so future branch selection can rank checkpoints by
    value. Distinct from :func:`distribute_reward_over_dag`, which assigns a
    terminal reward across the multi-agent execution DAG.
    """
    discounted = discount * branch_return
    current_value = parent.value if parent.value is not None else 0.0
    new_count = parent.visit_count + visit_count
    parent.value = (
        current_value * parent.visit_count + discounted * visit_count
    ) / new_count
    parent.visit_count = new_count

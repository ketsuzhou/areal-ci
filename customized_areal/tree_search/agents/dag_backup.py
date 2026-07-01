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

from customized_areal.tree_search.agents.execution_dag import ExecutionDAG

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
    backup_decay: float = 1.0,
) -> dict[str, float]:
    """Distribute ``terminal_reward`` backward along DAG edges.

    Returns a ``{node_id: credit}`` map. The terminal node gets the full
    reward; each ancestor along an incoming edge gets ``backup_decay`` of
    its child's credit (so credit attenuates with distance from the outcome
    when ``backup_decay < 1.0``). Fan-in joins (multiple parents) consume
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
            # Single parent or no explicit fan-in credit: split with decay.
            share = credit[nid] * backup_decay / len(parents)
            for p in parents:
                credit[p] += share
        queue.extend(parents)

    return credit

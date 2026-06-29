"""Bidirectional codec between an ExecutionDAG and its linear Event log.

Forward (``dag_to_events``): linearize the DAG into completion-ordered Events
for reward backup. Reverse (``events_to_dag``): losslessly rebuild the DAG from
a persisted Event log, then (``replay_prefix_for``) derive a branch replay
prefix shaped to ``BranchMaterializer.materialize``'s inputs.

Pure: no mutation of inputs, no I/O, torch-free. All failures raise ``DAGError``.
"""

from __future__ import annotations

from collections.abc import Sequence

from customized_areal.tree_search.dag.event_model import Event
from customized_areal.tree_search.dag.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
)


def _adjacency(
    dag: ExecutionDAG,
) -> tuple[
    dict[str, list[tuple[str, EdgeType]]], dict[str, list[tuple[str, EdgeType]]]
]:
    """Build per-node incoming/outgoing typed-edge lists from ``dag.edges``."""
    incoming: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.node_ids()}
    outgoing: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.node_ids()}
    for e in dag.edges:
        outgoing[e.src].append((e.dst, e.type))
        incoming[e.dst].append((e.src, e.type))
    return incoming, outgoing


def _validate_topological(dag: ExecutionDAG, order: Sequence[str]) -> None:
    index_of = {nid: i for i, nid in enumerate(order)}
    for e in dag.edges:
        if index_of[e.src] >= index_of[e.dst]:
            raise DAGError(
                f"ordering is not topological: edge {e.src!r}->{e.dst!r} has "
                f"index {index_of[e.src]} >= {index_of[e.dst]}"
            )


def dag_to_events(
    dag: ExecutionDAG,
    *,
    ordering: Sequence[str] | None = None,
    messages_by_node: dict | None = None,
) -> list[Event]:
    """Linearize ``dag`` into completion-ordered Events.

    ``ordering`` is an explicit completion order (list of node_id). If omitted,
    ``dag.topological_order()`` is used (deterministic; valid because completion
    order is always a topological order). ``messages_by_node`` overrides the
    per-node transcript payload (else ``node.metadata['messages']``).
    """
    node_ids = dag.node_ids()
    if ordering is None:
        order = [n.node_id for n in dag.topological_order()]
    else:
        order = list(ordering)
        if sorted(order) != sorted(node_ids):
            raise DAGError("ordering is not a permutation of the DAG node ids")
        _validate_topological(dag, order)

    incoming, outgoing = _adjacency(dag)
    msgs_map = messages_by_node or {}
    events: list[Event] = []
    for idx, nid in enumerate(order):
        node = dag.get(nid)
        payload = msgs_map.get(nid)
        if payload is None:
            payload = node.metadata.get("messages", ())
        events.append(
            Event(
                node_id=node.node_id,
                agent_id=node.agent_id,
                issue_id=node.issue_id,
                task_id=node.task_id,
                completion_index=idx,
                incoming_edges=tuple(incoming[nid]),
                outgoing_edges=tuple(outgoing[nid]),
                session_id=node.session_id,
                completion_time=node.metadata.get("completion_time"),
                branch_seq=node.branch_seq,
                branch_issue_id=node.branch_issue_id,
                branch_env_snapshot_id=node.branch_env_snapshot_id,
                value=node.value,
                process_reward=node.process_reward,
                outcome_reward=node.outcome_reward,
                messages=tuple(dict(m) for m in payload),
                metadata=dict(node.metadata),
            )
        )
    return events


__all__ = ["dag_to_events"]

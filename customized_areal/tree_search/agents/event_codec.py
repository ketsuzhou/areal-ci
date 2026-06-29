"""Bidirectional codec between an ExecutionDAG and its linear Event log.

Forward (``dag_to_events``): linearize the DAG into completion-ordered Events
for reward backup. Reverse (``events_to_dag``): losslessly rebuild the DAG from
a persisted Event log, then (``replay_prefix_for``) derive a branch replay
prefix shaped to ``BranchMaterializer.materialize``'s inputs.

Pure: no mutation of inputs, no I/O, torch-free. All failures raise ``DAGError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from customized_areal.tree_search.agents.event_model import Event, message_timeline
from customized_areal.tree_search.agents.execution_dag import (
    AgentRunNode,
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


def events_to_dag(events: Sequence[Event]) -> ExecutionDAG:
    """Losslessly rebuild an ExecutionDAG from a linear Event log.

    Steps (each failure raises ``DAGError``):
      1. completion_index must be dense 0..n-1, unique, non-negative.
      2. node_ids must be unique (no duplicate node_id across events).
      3. edge lists must be symmetric (every A.outgoing (A->B) has a matching
         B.incoming (A->B) with the same EdgeType).
      4. add nodes (faithful AgentRunNode; messages/index stay in the log only).
      5. add edges (idempotent).
      6. enforce the topological-order invariant: for every edge src->dst,
         index(src) < index(dst).
    """
    events = list(events)
    if not events:
        return ExecutionDAG()

    indices = sorted(e.completion_index for e in events)
    if indices != list(range(len(events))):
        raise DAGError(
            f"completion_index must be dense 0..{len(events) - 1}, got {indices}"
        )
    ordered = sorted(events, key=lambda e: e.completion_index)
    index_of = {e.node_id: e.completion_index for e in ordered}
    if len(index_of) != len(ordered):
        raise DAGError("duplicate node_id across events")

    incoming_set = {
        (src, e.node_id, t) for e in ordered for (src, t) in e.incoming_edges
    }
    outgoing_set = {
        (e.node_id, dst, t) for e in ordered for (dst, t) in e.outgoing_edges
    }
    if incoming_set != outgoing_set:
        diff = incoming_set ^ outgoing_set
        raise DAGError(
            f"edge symmetry mismatch (incoming XOR outgoing): {sorted(diff)}"
        )

    dag = ExecutionDAG()
    for e in ordered:
        dag.add_node(
            AgentRunNode(
                node_id=e.node_id,
                agent_id=e.agent_id,
                issue_id=e.issue_id,
                task_id=e.task_id,
                session_id=e.session_id,
                branch_seq=e.branch_seq,
                branch_issue_id=e.branch_issue_id,
                branch_env_snapshot_id=e.branch_env_snapshot_id,
                process_reward=e.process_reward,
                outcome_reward=e.outcome_reward,
                value=e.value,
                metadata=dict(e.metadata),
            )
        )

    for src, dst, t in sorted(incoming_set):
        dag.add_edge(src, dst, t)

    for src, dst, _ in incoming_set:
        if index_of[src] >= index_of[dst]:
            raise DAGError(
                f"event order is not topological: edge {src!r}->{dst!r} has "
                f"index {index_of[src]} >= {index_of[dst]}"
            )
    return dag


__all__ = ["ReplayPrefix", "dag_to_events", "events_to_dag", "replay_prefix_for"]


@dataclass(frozen=True)
class ReplayPrefix:
    """Branch replay data, shaped to ``BranchMaterializer.materialize`` inputs."""

    replay_messages: list[dict]
    task_id: str
    seq: int
    source_issue_id: str
    branch_env_snapshot_id: str | None


def replay_prefix_for(
    events: Sequence[Event],
    *,
    branch_point: tuple[str, int],
) -> ReplayPrefix:
    """Derive the replay prefix for a branch point from a linear Event log.

    ``branch_point = (task_id, seq)`` where ``seq`` is the ``task_message.seq``
    the run is allowed to branch at. Locates the unique Event with matching
    ``task_id`` and ``branch_seq == seq`` (zero or multiple matches -> DAGError),
    collects that node's ancestors (plus the node itself) in completion order,
    and flattens their message payloads via ``message_timeline``.
    """
    task_id, seq = branch_point
    dag = events_to_dag(events)
    matches = [e for e in events if e.task_id == task_id and e.branch_seq == seq]
    if len(matches) != 1:
        raise DAGError(
            f"branch point (task_id={task_id!r}, seq={seq}) matched {len(matches)} "
            f"nodes; expected exactly 1"
        )
    branch_ev = matches[0]
    ancestor_ids = dag.ancestors(branch_ev.node_id) | {branch_ev.node_id}
    prefix_events = sorted(
        (e for e in events if e.node_id in ancestor_ids),
        key=lambda e: e.completion_index,
    )
    return ReplayPrefix(
        replay_messages=message_timeline(prefix_events),
        task_id=task_id,
        seq=seq,
        source_issue_id=branch_ev.issue_id,
        branch_env_snapshot_id=branch_ev.branch_env_snapshot_id,
    )

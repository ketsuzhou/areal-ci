"""Bidirectional codec between an ExecutionDAG and its linear SuperNode log.

Forward (``dag_to_supernodes``): linearize the DAG into completion-ordered
SuperNodes for reward backup. Reverse (``supernodes_to_dag``): losslessly
rebuild the DAG from a persisted SuperNode log, then
(``replay_prefix_for``) derive a branch replay prefix shaped to
``BranchMaterializer.materialize``'s inputs.

Pure: no mutation of inputs, no I/O, torch-free. All failures raise ``DAGError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from customized_areal.tree_search.agents.event_model import message_timeline
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def _adjacency(
    dag: ExecutionDAG,
) -> tuple[
    dict[str, list[tuple[str, EdgeType]]], dict[str, list[tuple[str, EdgeType]]]
]:
    """Build per-node incoming/outgoing typed-edge lists from ``dag.edges``."""
    incoming: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.event_ids()}
    outgoing: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.event_ids()}
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


def dag_to_supernodes(
    dag: ExecutionDAG,
    *,
    ordering: Sequence[str] | None = None,
    nodes_by_segment: dict | None = None,
) -> list[SuperNode]:
    """Linearize ``dag`` into completion-ordered SuperNodes.

    ``ordering`` is an explicit completion order (list of node_id). If omitted,
    ``dag.topological_order()`` is used (deterministic; valid because completion
    order is always a topological order). ``nodes_by_segment`` overrides the
    per-segment ``nodes`` payload (else ``segment.metadata['nodes']``).
    """
    event_ids = dag.event_ids()
    if ordering is None:
        order = [n.node_id for n in dag.topological_order()]
    else:
        order = list(ordering)
        if sorted(order) != sorted(event_ids):
            raise DAGError("ordering is not a permutation of the DAG event ids")
        _validate_topological(dag, order)

    incoming, outgoing = _adjacency(dag)
    nodes_map = nodes_by_segment or {}
    supers: list[SuperNode] = []
    for idx, nid in enumerate(order):
        node = dag.get(nid)
        payload = nodes_map.get(nid)
        if payload is None:
            payload = node.metadata.get("nodes", [])
        super_node = SuperNode(
            node_id=node.node_id,
            agent_id=node.agent_id,
            issue_id=node.issue_id,
            task_id=node.task_id,
            closing_event=node.closing_event,
            closing_event_target=node.closing_event_target,
            session_id=node.session_id,
            completion_index=idx,
            completion_time=node.metadata.get("completion_time"),
            incoming_edges=tuple(incoming[nid]),
            outgoing_edges=tuple(outgoing[nid]),
            branch_seq=node.branch_seq,
            branch_issue_id=node.branch_issue_id,
            branch_env_snapshot_id=node.branch_env_snapshot_id,
            value=node.value,
            process_reward=node.process_reward,
            outcome_reward=node.outcome_reward,
            sandbox_ids=list(node.sandbox_ids),
            issue_snapshot_id=node.issue_snapshot_id,
            env_state=dict(node.env_state),
            nodes=list(payload),
            metadata=dict(node.metadata),
        )
        supers.append(super_node)
    return supers


def supernodes_to_dag(supers: Sequence[SuperNode]) -> ExecutionDAG:
    """Losslessly rebuild an ExecutionDAG from a linear SuperNode log.

    Steps (each failure raises ``DAGError``):
      1. completion_index must be dense 0..n-1, unique, non-negative.
      2. node_ids must be unique (no duplicate node_id across SuperNodes).
      3. edge lists must be symmetric (every A.outgoing (A->B) has a matching
         B.incoming (A->B) with the same EdgeType).
      4. add SuperNodes (faithful copy of all fields; ``completion_index`` and
         ``completion_time`` stay in the log only -- the latter is preserved
         via ``metadata['completion_time']`` by ``dag_to_supernodes``).
      5. add edges (idempotent).
      6. enforce the topological-order invariant: for every edge src->dst,
         index(src) < index(dst).
    """
    supers = list(supers)
    if not supers:
        return ExecutionDAG()

    indices = sorted(s.completion_index for s in supers)
    if indices != list(range(len(supers))):
        raise DAGError(
            f"completion_index must be dense 0..{len(supers) - 1}, got {indices}"
        )
    ordered = sorted(supers, key=lambda s: s.completion_index)
    index_of = {s.node_id: s.completion_index for s in ordered}
    if len(index_of) != len(ordered):
        raise DAGError("duplicate node_id across SuperNodes")

    incoming_set = {
        (src, s.node_id, t) for s in ordered for (src, t) in s.incoming_edges
    }
    outgoing_set = {
        (s.node_id, dst, t) for s in ordered for (dst, t) in s.outgoing_edges
    }
    if incoming_set != outgoing_set:
        diff = incoming_set ^ outgoing_set
        raise DAGError(
            f"edge symmetry mismatch (incoming XOR outgoing): {sorted(diff)}"
        )

    dag = ExecutionDAG()
    for s in ordered:
        dag.add_event(
            SuperNode(
                node_id=s.node_id,
                agent_id=s.agent_id,
                issue_id=s.issue_id,
                task_id=s.task_id,
                closing_event=s.closing_event,
                closing_event_target=s.closing_event_target,
                session_id=s.session_id,
                branch_seq=s.branch_seq,
                branch_issue_id=s.branch_issue_id,
                branch_env_snapshot_id=s.branch_env_snapshot_id,
                process_reward=s.process_reward,
                outcome_reward=s.outcome_reward,
                value=s.value,
                sandbox_ids=list(s.sandbox_ids),
                issue_snapshot_id=s.issue_snapshot_id,
                env_state=dict(s.env_state),
                nodes=list(s.nodes),
                metadata=dict(s.metadata),
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


@dataclass(frozen=True)
class ReplayPrefix:
    """Branch replay data, shaped to ``BranchMaterializer.materialize`` inputs."""

    replay_messages: list[dict]
    task_id: str
    seq: int
    source_issue_id: str
    branch_env_snapshot_id: str | None


def replay_prefix_for(
    supers: Sequence[SuperNode],
    *,
    branch_point: tuple[str, int],
) -> ReplayPrefix:
    """Derive the replay prefix for a branch point from a linear SuperNode log.

    ``branch_point = (task_id, seq)`` where ``seq`` is the ``task_message.seq``
    the run is allowed to branch at. Locates the unique SuperNode with matching
    ``task_id`` and ``branch_seq == seq`` (zero or multiple matches -> DAGError),
    collects that node's ancestors (plus the node itself) in completion order,
    and flattens their message payloads via ``message_timeline``.
    """
    task_id, seq = branch_point
    dag = supernodes_to_dag(supers)
    matches = [s for s in supers if s.task_id == task_id and s.branch_seq == seq]
    if len(matches) != 1:
        raise DAGError(
            f"branch point (task_id={task_id!r}, seq={seq}) matched {len(matches)} "
            f"nodes; expected exactly 1"
        )
    branch_super = matches[0]
    ancestor_ids = dag.ancestors(branch_super.node_id) | {branch_super.node_id}
    prefix_supers = sorted(
        (s for s in supers if s.node_id in ancestor_ids),
        key=lambda s: s.completion_index,
    )
    return ReplayPrefix(
        replay_messages=message_timeline(prefix_supers),
        task_id=task_id,
        seq=seq,
        source_issue_id=branch_super.issue_id,
        branch_env_snapshot_id=branch_super.branch_env_snapshot_id,
    )


__all__ = [
    "ReplayPrefix",
    "dag_to_supernodes",
    "replay_prefix_for",
    "supernodes_to_dag",
]

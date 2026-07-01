"""Assemble SuperNodes from Multica's segment specs + proxy interactions.

Consumes Multica's pre-defined segments (does NOT infer boundaries). Maps each
agent's list[Node] into the segments Multica defined, by the start_turn_idx /
end_turn_idx range on each SegmentSpec.

Maintains the unified parent_node_id chain (causal flattening):
  - Within one run: n_{k+1}.parent_node_id = n_k.node_id
  - Cross-agent delegation (A's n3 delegates -> B's n1'):
    n1'.parent_node_id = n3.node_id
  - Cross-agent completion (B completes -> A continues@n4):
    n4.parent_node_id = B's terminal node_id
  - MENTION edges record topology only; they do NOT set parent_node_id.

Stamps TeamEnvSnapshot onto each SuperNode. Binds session_id to each SuperNode
(from session_to_agent_run).

Pure: no I/O, no mutation of inputs, torch-free. All failures raise DAGError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


@dataclass(frozen=True)
class SegmentSpec:
    """One communication-bounded segment, as defined by Multica.

    Multica decides segment boundaries at communication events during
    execution; AReal does not infer them.
    """

    segment_id: str
    agent_run_id: str
    issue_id: str
    task_id: str
    closing_event: EdgeType | None
    closing_event_target_segment: str | None
    # 1-based inclusive turn range within this agent run.
    start_turn_idx: int
    end_turn_idx: int


@dataclass(frozen=True)
class EdgeSpec:
    """One typed DAG edge between segments, as defined by Multica."""

    src_segment_id: str
    dst_segment_id: str
    type: EdgeType


@dataclass(frozen=True)
class TeamEnvSnapshot:
    """Team-wide environment state at a branching point, from Multica.

    One per segment, captured at the moment that segment's closing
    communication event fired. Multica owns the env state; AReal stamps
    it onto the corresponding SuperNode without interpretation.
    """

    sandbox_ids: list[str]
    issue_snapshot_id: str | None
    env_state: dict


@dataclass(frozen=True)
class DagResult:
    """Multica's completion payload: the DAG of segments + env snapshots."""

    session_ids: list[str]
    session_to_agent_run: dict[str, str]
    segments: list[SegmentSpec]
    edges: list[EdgeSpec]
    env_snapshots: dict[str, TeamEnvSnapshot]


class SuperNodeAssembler:
    """Assemble SuperNodes from Multica's segment specs + proxy interactions."""

    def assemble(
        self,
        *,
        sessions_nodes: dict[str, list[Any]],
        dag_result: DagResult,
    ) -> tuple[list[SuperNode], ExecutionDAG, str]:
        """Returns (super_nodes, dag, root_terminal_node_id).

        root_terminal_node_id is the terminal node of the DAG's unique sink
        -- the starting point for backup_episode_terminal.
        """
        # ── Step 1: resolve session -> agent_run -> list[Node] ─────────
        agent_run_to_session: dict[str, str] = {}
        for session_id, agent_run_id in dag_result.session_to_agent_run.items():
            agent_run_to_session[agent_run_id] = session_id

        # ── Step 2: slice each agent run's list[Node] into segments ────
        segment_id_to_nodes: dict[str, list[Any]] = {}
        # Track per-run turn coverage to detect gaps/overlaps.
        run_coverage: dict[str, list[tuple[int, int, str]]] = {}
        for spec in dag_result.segments:
            session_id = agent_run_to_session.get(spec.agent_run_id)
            if session_id is None:
                raise DAGError(
                    f"segment {spec.segment_id!r} references dangling "
                    f"agent_run_id {spec.agent_run_id!r}"
                )
            agent_nodes = sessions_nodes.get(session_id)
            if agent_nodes is None:
                raise DAGError(
                    f"segment {spec.segment_id!r}: session {session_id!r} "
                    f"not in sessions_nodes"
                )
            if spec.start_turn_idx < 1 or spec.end_turn_idx < spec.start_turn_idx:
                raise DAGError(
                    f"segment {spec.segment_id!r}: invalid turn range "
                    f"[{spec.start_turn_idx}, {spec.end_turn_idx}]"
                )
            if spec.end_turn_idx > len(agent_nodes):
                raise DAGError(
                    f"segment {spec.segment_id!r}: turn range out of range "
                    f"(end_turn_idx={spec.end_turn_idx} > "
                    f"len(nodes)={len(agent_nodes)})"
                )
            segment_nodes = agent_nodes[spec.start_turn_idx - 1 : spec.end_turn_idx]
            if not segment_nodes:
                raise DAGError(f"segment {spec.segment_id!r}: slice is empty")
            segment_id_to_nodes[spec.segment_id] = segment_nodes
            run_coverage.setdefault(spec.agent_run_id, []).append(
                (spec.start_turn_idx, spec.end_turn_idx, spec.segment_id)
            )
        # Validate dense, non-overlapping coverage per run.
        for agent_run_id, ranges in run_coverage.items():
            ranges_sorted = sorted(ranges, key=lambda r: r[0])
            for i in range(1, len(ranges_sorted)):
                prev_end = ranges_sorted[i - 1][1]
                cur_start = ranges_sorted[i][0]
                if cur_start <= prev_end:
                    raise DAGError(
                        f"agent_run {agent_run_id!r}: segments "
                        f"{ranges_sorted[i - 1][2]!r} and "
                        f"{ranges_sorted[i][2]!r} overlap or are not dense"
                    )

        # ── Step 3: construct each SuperNode ──────────────────────────
        segment_id_to_super: dict[str, SuperNode] = {}
        supers: list[SuperNode] = []
        for spec in dag_result.segments:
            env_snapshot = dag_result.env_snapshots.get(spec.segment_id)
            if env_snapshot is None:
                raise DAGError(
                    f"segment {spec.segment_id!r}: missing env_snapshots entry"
                )
            session_id = agent_run_to_session[spec.agent_run_id]
            super_node = SuperNode(
                node_id=self._fresh_uuid(),
                agent_id=spec.agent_run_id,  # use run id as agent_id stand-in
                issue_id=spec.issue_id,
                task_id=spec.task_id,
                closing_event=spec.closing_event,
                closing_event_target=None,  # resolved after all SuperNodes built
                session_id=session_id,
                sandbox_ids=list(env_snapshot.sandbox_ids),
                issue_snapshot_id=env_snapshot.issue_snapshot_id,
                env_state=dict(env_snapshot.env_state),
                nodes=list(segment_id_to_nodes[spec.segment_id]),
                metadata={"_segment_id": spec.segment_id},
            )
            segment_id_to_super[spec.segment_id] = super_node
            supers.append(super_node)
        # Backfill closing_event_target by resolving segment_id -> SuperNode UUID.
        for spec in dag_result.segments:
            if spec.closing_event_target_segment is None:
                continue
            target_super = segment_id_to_super.get(spec.closing_event_target_segment)
            if target_super is None:
                raise DAGError(
                    f"segment {spec.segment_id!r}: closing_event_target_segment "
                    f"{spec.closing_event_target_segment!r} is unknown"
                )
            segment_id_to_super[
                spec.segment_id
            ].closing_event_target = target_super.node_id

        # ── Step 4: build ExecutionDAG from EdgeSpecs ──────────────────
        dag = ExecutionDAG()
        for super_node in supers:
            dag.add_event(super_node)
        for edge_spec in dag_result.edges:
            src_super = segment_id_to_super.get(edge_spec.src_segment_id)
            dst_super = segment_id_to_super.get(edge_spec.dst_segment_id)
            if src_super is None or dst_super is None:
                raise DAGError(
                    f"edge references unknown segment: "
                    f"{edge_spec.src_segment_id!r} -> {edge_spec.dst_segment_id!r}"
                )
            dag.add_edge(src_super.node_id, dst_super.node_id, edge_spec.type)
        # Assign completion_index from topological order.
        topo = dag.topological_order()
        for idx, super_node in enumerate(topo):
            super_node.completion_index = idx
        # Populate incoming_edges / outgoing_edges on each SuperNode.
        for super_node in supers:
            incoming = [
                (e.src, e.type) for e in dag.edges if e.dst == super_node.node_id
            ]
            outgoing = [
                (e.dst, e.type) for e in dag.edges if e.src == super_node.node_id
            ]
            super_node.incoming_edges = tuple(incoming)
            super_node.outgoing_edges = tuple(outgoing)

        # ── Step 5: set the unified parent_node_id chain ──────────────
        # (a) Within-segment sequential.
        for super_node in supers:
            for i in range(1, len(super_node.nodes)):
                super_node.nodes[i].parent_node_id = super_node.nodes[i - 1].node_id
        # Precompute, per segment, the set of incoming DELEGATION/COMPLETION
        # edges (for the (b) gate + (c)/(d) overrides).
        blocking_incoming: dict[str, list[EdgeSpec]] = {}
        for edge_spec in dag_result.edges:
            if edge_spec.type in (EdgeType.DELEGATION, EdgeType.COMPLETION):
                blocking_incoming.setdefault(edge_spec.dst_segment_id, []).append(
                    edge_spec
                )
        # (b) Within-run cross-segment (default): for each run, sort segments
        # by start_turn_idx; for segment[k] (k>0) with no blocking incoming
        # edge, set its first node's parent to segment[k-1]'s terminal.
        segments_by_run: dict[str, list[SegmentSpec]] = {}
        for spec in dag_result.segments:
            segments_by_run.setdefault(spec.agent_run_id, []).append(spec)
        for agent_run_id, run_segments in segments_by_run.items():
            run_segments_sorted = sorted(run_segments, key=lambda s: s.start_turn_idx)
            for k in range(1, len(run_segments_sorted)):
                cur_spec = run_segments_sorted[k]
                if cur_spec.segment_id in blocking_incoming:
                    continue  # (c) or (d) will set the parent
                prev_spec = run_segments_sorted[k - 1]
                prev_super = segment_id_to_super[prev_spec.segment_id]
                cur_super = segment_id_to_super[cur_spec.segment_id]
                if cur_super.nodes:
                    prev_terminal = prev_super.terminal_node
                    if prev_terminal is not None:
                        cur_super.nodes[0].parent_node_id = prev_terminal.node_id
        # (c) + (d): for each blocking incoming edge, set dst's first node
        # parent to src's terminal node_id.
        for edge_spec in dag_result.edges:
            if edge_spec.type not in (EdgeType.DELEGATION, EdgeType.COMPLETION):
                continue
            src_super = segment_id_to_super[edge_spec.src_segment_id]
            dst_super = segment_id_to_super[edge_spec.dst_segment_id]
            src_terminal = src_super.terminal_node
            if src_terminal is None or not dst_super.nodes:
                continue
            dst_super.nodes[0].parent_node_id = src_terminal.node_id
        # MENTION edges: no parent_node_id set (topology-only), by omission.

        # ── Step 6: identify unique sink ──────────────────────────────
        sinks = [s for s in supers if not s.outgoing_edges]
        if len(sinks) != 1:
            raise DAGError(
                f"execution DAG has multiple sinks (found {len(sinks)}); "
                f"sinks={[s.node_id for s in sinks]}"
            )
        root_terminal = sinks[0].terminal_node
        if root_terminal is None:
            raise DAGError(f"sink SuperNode {sinks[0].node_id!r} has no terminal node")
        return supers, dag, root_terminal.node_id

    @staticmethod
    def _fresh_uuid() -> str:
        import uuid

        return str(uuid.uuid4())


__all__ = [
    "DagResult",
    "EdgeSpec",
    "SegmentSpec",
    "SuperNodeAssembler",
    "TeamEnvSnapshot",
]

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
  - Fan-in join (multiple DELEGATION/COMPLETION edges into one segment):
    the first source terminal becomes parent_node_id, the rest become
    extra_parent_node_ids, so reward backup credits every incoming branch.
  - MENTION edges record topology only; they do NOT set parent_node_id.

Stamps TeamEnvSnapshot onto each SuperNode. Binds session_id to each SuperNode
(from session_to_agent_run).

Torch-free, no I/O. Container structures (lists, dicts, DAG) are not mutated;
Node.parent_node_id IS set in place (Step 5's contract). All failures raise DAGError.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)
from customized_areal.tree_search.agents.multica_dag_client import AssembledDag

logger = logging.getLogger("SuperNodeAssembler")


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


class TensorResolver(Protocol):
    """Resolves a segment's ``tensor_ref`` to its token/logprob tensors.

    Implementations call the v2 data_proxy ``/data/<shard_id>`` / ``/data/batch``
    endpoints. Returns a torch-free dict (``input_ids``, ``loss_mask``,
    ``logprobs``, ``versions``, ``attention_mask``, ``rewards``) so the assembler
    stays torch-free and unit-testable.
    """

    def resolve(self, tensor_ref: dict[str, Any]) -> dict[str, Any]: ...


def _aggregate_process_reward(scores: list[int], score_max: int) -> float:
    """Normalized mean of a segment's diagnosis step scores -> ``[0, 1]``.

    The diagnosis agent scores each LLM output (turn) in ``[0, score_max]``;
    the v2 GAE consumes a per-segment ``SuperNode.process_reward`` (one GAE
    step per segment), so the per-turn scores are aggregated to a segment
    reward. Returns 0.0 when the segment was not scored (sparse - diagnosis
    did not run / did not cover it) or when ``score_max`` is 0 (diagnosis
    scoring not configured). Absence stays 0.0 - never a fabricated reward.
    """
    if not scores or score_max <= 0:
        return 0.0
    return (sum(scores) / len(scores)) / score_max


class SuperNodeAssembler:
    """Assemble SuperNodes from Multica's segment specs + proxy interactions."""

    def assemble_from_refs(
        self,
        dag: AssembledDag,
        resolver: TensorResolver,
    ) -> ExecutionDAG | None:
        """Build an ExecutionDAG from an AssembledDag by resolving tensor refs.

        This is the v2 segment-DAG consumer path. Unlike :meth:`assemble`
        (which slices an agent's ``list[Node]`` by ``start_turn_idx`` /
        ``end_turn_idx``), each segment's payload comes from resolving its
        ``tensor_ref`` via ``resolver`` - no turn indices, no message text, no
        judge scores cross this boundary (per the locked architecture).

        Design choices:
          - ``segment_id`` is used as the SuperNode ``node_id`` so the
            AssembledDag edges (which reference segment ids) resolve directly.
          - Resolved tensors are stored on ``metadata["tensors"]``; segment
            identity (``segment_id``, ``trajectory_id``) is stored in metadata
            too. SuperNode has no ``payload`` field, so metadata is the
            non-invasive attachment point (the old ``assemble`` path is
            unaffected).
          - ``task_id`` is "" because the v2 SegmentSpec dropped it (the locked
            segment shape carries ``trajectory_id`` + ``tensor_ref`` instead).
          - Acyclicity is validated via :meth:`ExecutionDAG.topological_order`,
            which raises ``DAGError`` on a cycle. ``completion_index`` is
            stamped from that order, mirroring :meth:`assemble`.

        Args:
            dag: The AssembledDag fetched from Multica (structure only).
            resolver: Resolves each segment's ``tensor_ref`` to tensors.

        Returns:
            The assembled ExecutionDAG (one SuperNode per segment), or None when
            the dag has no segments (empty trajectory).

        Raises:
            DAGError: On a cycle, a dangling edge, or an unknown EdgeType.
        """
        # Empty trajectory (no segments recorded) -> None so callers skip the
        # episode rather than train on an empty graph.
        if not dag.segments:
            return None
        agent_run_to_session: dict[str, str] = {
            agent_run_id: session_id
            for session_id, agent_run_id in dag.session_to_agent_run.items()
        }

        # Index diagnosis step rewards by segment_id so each SuperNode's
        # process_reward can be the normalized mean of its per-turn scores.
        # score_max (served by /dag) is the diagnosis agent's scoring scale; 0
        # means diagnosis scoring was not configured -> sparse (0.0). Rewards
        # whose segment_id has no matching segment are dropped + logged (never
        # applied to a wrong SuperNode, never fatal).
        segment_ids = {seg.segment_id for seg in dag.segments}
        scores_by_segment: dict[str, list[int]] = {}
        for sr in dag.step_rewards:
            if sr.segment_id not in segment_ids:
                logger.warning(
                    "dropping step reward for unknown segment %r (seq=%d); "
                    "no matching SuperNode",
                    sr.segment_id,
                    sr.seq,
                )
                continue
            scores_by_segment.setdefault(sr.segment_id, []).append(sr.score)

        edag = ExecutionDAG()
        for seg in dag.segments:
            tensors = resolver.resolve(seg.tensor_ref)
            env = seg.env_snapshot or {}
            closing_event = (
                EdgeType(seg.closing_event) if seg.closing_event else None
            )
            super_node = SuperNode(
                node_id=seg.segment_id,
                agent_id=seg.agent_run_id,
                issue_id=seg.issue_id,
                task_id="",
                closing_event=closing_event,
                session_id=agent_run_to_session.get(seg.agent_run_id),
                sandbox_ids=list(env.get("sandbox_ids", [])),
                issue_snapshot_id=env.get("issue_snapshot_id"),
                env_state=dict(env.get("env_state", {})),
                nodes=[],
                metadata={
                    "segment_id": seg.segment_id,
                    "trajectory_id": seg.trajectory_id,
                    "tensors": tensors,
                },
            )
            edag.add_event(super_node)
            # Aggregate the segment's per-turn diagnosis scores into the
            # per-segment GAE reward (events_from_nodes consumes
            # super_node.process_reward). 0.0 when unscored or score_max is 0.
            super_node.process_reward = _aggregate_process_reward(
                scores_by_segment.get(seg.segment_id, []), dag.score_max
            )

        for edge in dag.edges:
            edag.add_edge(
                edge.src_segment_id,
                edge.dst_segment_id,
                EdgeType(edge.type),
                branch_from_segment_id=edge.branch_from_segment_id,
                branch_from_checkpoint_id=edge.branch_from_checkpoint_id,
            )

        # Validate acyclicity and stamp completion_index from topological order.
        # topological_order raises DAGError if the graph contains a cycle.
        for idx, super_node in enumerate(edag.topological_order()):
            super_node.completion_index = idx
        return edag

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
            session_id = agent_run_to_session[agent_run_id]
            node_count = len(sessions_nodes[session_id])
            # First segment must start at turn 1.
            if ranges_sorted[0][0] != 1:
                raise DAGError(
                    f"agent_run {agent_run_id!r}: segments do not start at "
                    f"turn 1 (first start_turn_idx={ranges_sorted[0][0]}); "
                    f"dense coverage required"
                )
            # No gaps and no overlap between consecutive segments.
            for i in range(1, len(ranges_sorted)):
                prev_end = ranges_sorted[i - 1][1]
                cur_start = ranges_sorted[i][0]
                if cur_start != prev_end + 1:
                    raise DAGError(
                        f"agent_run {agent_run_id!r}: segments "
                        f"{ranges_sorted[i - 1][2]!r} and "
                        f"{ranges_sorted[i][2]!r} do not form dense coverage "
                        f"(gap or overlap: prev_end={prev_end}, "
                        f"cur_start={cur_start})"
                    )
            # Last segment must end at the last turn.
            if ranges_sorted[-1][1] != node_count:
                raise DAGError(
                    f"agent_run {agent_run_id!r}: segments do not cover "
                    f"through turn {node_count} (last end_turn_idx="
                    f"{ranges_sorted[-1][1]}); dense coverage required"
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
                agent_id=spec.agent_run_id,  # Multica identifies runs; agent_id resolution is deferred to a later phase.
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
        # (c) + (d): a segment's first node depends on the terminal(s) of the
        # source segment(s) of its incoming DELEGATION/COMPLETION edge(s). A
        # join (fan-in) has multiple such edges: the first source terminal
        # (ordered deterministically by src_segment_id) becomes parent_node_id;
        # the rest become extra_parent_node_ids. Reward backup follows every
        # parent, so all incoming branches of a join receive credit.
        for dst_segment_id, blocking_edges in blocking_incoming.items():
            dst_super = segment_id_to_super[dst_segment_id]
            if not dst_super.nodes:
                continue
            parent_terminals: list[str] = []
            for edge_spec in sorted(blocking_edges, key=lambda e: e.src_segment_id):
                src_terminal = segment_id_to_super[
                    edge_spec.src_segment_id
                ].terminal_node
                if src_terminal is not None:
                    parent_terminals.append(src_terminal.node_id)
            if not parent_terminals:
                continue
            dst_super.nodes[0].parent_node_id = parent_terminals[0]
            if len(parent_terminals) > 1:
                dst_super.nodes[0].extra_parent_node_ids = parent_terminals[1:]
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
        return str(uuid.uuid4())


__all__ = [
    "DagResult",
    "EdgeSpec",
    "SegmentSpec",
    "SuperNodeAssembler",
    "TeamEnvSnapshot",
]

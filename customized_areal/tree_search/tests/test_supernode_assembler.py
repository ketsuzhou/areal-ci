"""Unit tests for SuperNodeAssembler (spec §5.4 assemble algorithm).

Pure, torch-free. Uses SimpleNamespace stand-ins for Node (the assembler only
reads .node_id and .parent_node_id; Node is torch-lazy and heavy to build).

Fixture: a 3-segment DAG --
  planner (1 segment, 3 turns) --delegation--> worker (1 segment, 2 turns)
  worker --completion--> synthesizer (1 segment, 1 turn)
The DAG's unique sink is the synthesizer segment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
)
from customized_areal.tree_search.agents.supernode_assembler import (
    DagResult,
    EdgeSpec,
    SegmentSpec,
    SuperNodeAssembler,
    TeamEnvSnapshot,
)


def _node(node_id: str) -> SimpleNamespace:
    """Build a Node-like stand-in (the assembler reads .node_id / .parent_node_id)."""
    return SimpleNamespace(node_id=node_id, parent_node_id=None)


def _nodes(*ids: str) -> list:
    return [_node(i) for i in ids]


def _planner_worker_synthesizer() -> tuple[dict, DagResult]:
    """3-segment DAG: planner delegates to worker; worker completes to synth."""
    # Agent run "planner" has 3 turns; worker has 2; synthesizer has 1.
    sessions_nodes = {
        "sess-planner": _nodes("p1", "p2", "p3"),
        "sess-worker": _nodes("w1", "w2"),
        "sess-synth": _nodes("s1"),
    }
    dag_result = DagResult(
        session_ids=["sess-planner", "sess-worker", "sess-synth"],
        session_to_agent_run={
            "sess-planner": "run-planner",
            "sess-worker": "run-worker",
            "sess-synth": "run-synth",
        },
        segments=[
            SegmentSpec(
                segment_id="seg-planner",
                agent_run_id="run-planner",
                issue_id="iss-planner",
                task_id="task-root",
                closing_event=EdgeType.DELEGATION,
                closing_event_target_segment="seg-worker",
                start_turn_idx=1,
                end_turn_idx=3,
            ),
            SegmentSpec(
                segment_id="seg-worker",
                agent_run_id="run-worker",
                issue_id="iss-worker",
                task_id="task-root",
                closing_event=EdgeType.COMPLETION,
                closing_event_target_segment="seg-synth",
                start_turn_idx=1,
                end_turn_idx=2,
            ),
            SegmentSpec(
                segment_id="seg-synth",
                agent_run_id="run-synth",
                issue_id="iss-synth",
                task_id="task-root",
                closing_event=None,
                closing_event_target_segment=None,
                start_turn_idx=1,
                end_turn_idx=1,
            ),
        ],
        edges=[
            EdgeSpec("seg-planner", "seg-worker", EdgeType.DELEGATION),
            EdgeSpec("seg-worker", "seg-synth", EdgeType.COMPLETION),
        ],
        env_snapshots={
            "seg-planner": TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id="iss-snap-planner",
                env_state={"phase": "plan"},
            ),
            "seg-worker": TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id="iss-snap-worker",
                env_state={"phase": "work"},
            ),
            "seg-synth": TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id=None,
                env_state={"phase": "done"},
            ),
        },
    )
    return sessions_nodes, dag_result


# -- Step 1+2: slicing ----------------------------------------------------


def test_assemble_slices_each_agent_run_by_turn_range():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert [n.node_id for n in by_seg["seg-planner"].nodes] == ["p1", "p2", "p3"]
    assert [n.node_id for n in by_seg["seg-worker"].nodes] == ["w1", "w2"]
    assert [n.node_id for n in by_seg["seg-synth"].nodes] == ["s1"]


def test_assemble_rejects_out_of_range_turn_indices():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Corrupt: planner segment claims turns 1..5 but only 3 exist.
    dag_result.segments[0] = SegmentSpec(
        segment_id="seg-planner",
        agent_run_id="run-planner",
        issue_id="iss-planner",
        task_id="task-root",
        closing_event=EdgeType.DELEGATION,
        closing_event_target_segment="seg-worker",
        start_turn_idx=1,
        end_turn_idx=5,
    )
    with pytest.raises(DAGError, match="out of range"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


def test_assemble_rejects_overlapping_segments_within_run():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Make two segments of the same run overlap on turn 2.
    dag_result.segments.append(
        SegmentSpec(
            segment_id="seg-planner-2",
            agent_run_id="run-planner",
            issue_id="iss-planner",
            task_id="task-root",
            closing_event=None,
            closing_event_target_segment=None,
            start_turn_idx=2,
            end_turn_idx=3,
        )
    )
    with pytest.raises(DAGError, match="overlap"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


def test_assemble_rejects_non_dense_segments_within_run():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Replace planner's single [1,3] segment with [1,1] and [3,3]; turn 2
    # is uncovered, so coverage is not dense.
    dag_result.segments[0] = SegmentSpec(
        segment_id="seg-planner",
        agent_run_id="run-planner",
        issue_id="iss-planner",
        task_id="task-root",
        closing_event=EdgeType.DELEGATION,
        closing_event_target_segment="seg-worker",
        start_turn_idx=1,
        end_turn_idx=1,
    )
    dag_result.segments.append(
        SegmentSpec(
            segment_id="seg-planner-2",
            agent_run_id="run-planner",
            issue_id="iss-planner",
            task_id="task-root",
            closing_event=None,
            closing_event_target_segment=None,
            start_turn_idx=3,
            end_turn_idx=3,
        )
    )
    with pytest.raises(DAGError, match="dense coverage"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


# -- Step 3: SuperNode construction --------------------------------------


def test_assemble_stamps_env_snapshot_on_each_supernode():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert by_seg["seg-planner"].sandbox_ids == ["sb-p", "sb-w", "sb-s"]
    assert by_seg["seg-planner"].issue_snapshot_id == "iss-snap-planner"
    assert by_seg["seg-planner"].env_state == {"phase": "plan"}
    assert by_seg["seg-synth"].env_state == {"phase": "done"}


def test_assemble_binds_session_id_to_each_supernode():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert by_seg["seg-planner"].session_id == "sess-planner"
    assert by_seg["seg-worker"].session_id == "sess-worker"
    assert by_seg["seg-synth"].session_id == "sess-synth"


def test_assemble_sets_closing_event_and_target():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert by_seg["seg-planner"].closing_event is EdgeType.DELEGATION
    # closing_event_target is resolved to the other SuperNode's UUID.
    assert by_seg["seg-planner"].closing_event_target == by_seg["seg-worker"].node_id
    assert by_seg["seg-synth"].closing_event is None


def test_assemble_rejects_missing_env_snapshot():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    del dag_result.env_snapshots["seg-worker"]
    with pytest.raises(DAGError, match="env_snapshots"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


# -- Step 4: DAG construction --------------------------------------------


def test_assemble_builds_execution_dag_from_edges():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    _, dag, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in dag.events}
    # planner -> worker (delegation), worker -> synth (completion)
    assert {(e.src, e.dst, e.type) for e in dag.edges} == {
        (
            by_seg["seg-planner"].node_id,
            by_seg["seg-worker"].node_id,
            EdgeType.DELEGATION,
        ),
        (
            by_seg["seg-worker"].node_id,
            by_seg["seg-synth"].node_id,
            EdgeType.COMPLETION,
        ),
    }


def test_assemble_assigns_dense_completion_index():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    indices = sorted(s.completion_index for s in supers)
    assert indices == list(range(len(supers)))


def test_assemble_rejects_unknown_segment_in_edge():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    dag_result.edges.append(EdgeSpec("seg-planner", "seg-ghost", EdgeType.MENTION))
    with pytest.raises(DAGError, match="unknown segment"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


# -- Step 5: parent_node_id chain (causal flattening) --------------------


def test_assemble_sets_within_segment_parent_chain():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # planner segment: p1 (root, parent None) -> p2 -> p3
    p_nodes = by_seg["seg-planner"].nodes
    assert p_nodes[0].parent_node_id is None
    assert p_nodes[1].parent_node_id == p_nodes[0].node_id
    assert p_nodes[2].parent_node_id == p_nodes[1].node_id


def test_assemble_sets_cross_agent_delegation_parent():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # planner's terminal (p3) delegates to worker; worker's first turn (w1)
    # has parent_node_id = p3.node_id.
    planner_terminal = by_seg["seg-planner"].terminal_node
    worker_first = by_seg["seg-worker"].nodes[0]
    assert worker_first.parent_node_id == planner_terminal.node_id


def test_assemble_sets_cross_agent_completion_parent():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # worker's terminal (w2) completes to synthesizer; synthesizer's first
    # turn (s1) has parent_node_id = w2.node_id.
    worker_terminal = by_seg["seg-worker"].terminal_node
    synth_first = by_seg["seg-synth"].nodes[0]
    assert synth_first.parent_node_id == worker_terminal.node_id


def test_assemble_mention_edge_does_not_set_parent():
    """A MENTION edge records topology only; it must NOT set parent_node_id."""
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Add a peer MENTION edge: planner mentions synth (non-blocking).
    dag_result.edges.append(EdgeSpec("seg-planner", "seg-synth", EdgeType.MENTION))
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # synth's first turn's parent must still be worker's terminal (COMPLETION),
    # NOT planner's terminal (MENTION). MENTION is topology-only.
    worker_terminal = by_seg["seg-worker"].terminal_node
    synth_first = by_seg["seg-synth"].nodes[0]
    assert synth_first.parent_node_id == worker_terminal.node_id


# -- Step 6: root_terminal_node_id ---------------------------------------


def test_assemble_returns_unique_sink_terminal_node_id():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, root_terminal = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert root_terminal == by_seg["seg-synth"].terminal_node.node_id


def test_assemble_rejects_multiple_sinks():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Add a second sink segment: its own run/session, no edges in or out.
    dag_result.segments.append(
        SegmentSpec(
            segment_id="seg-orphan",
            agent_run_id="run-orphan",
            issue_id="iss-orphan",
            task_id="task-root",
            closing_event=None,
            closing_event_target_segment=None,
            start_turn_idx=1,
            end_turn_idx=1,
        )
    )
    sessions_nodes["sess-orphan"] = _nodes("o1")
    dag_result.session_ids.append("sess-orphan")
    dag_result.session_to_agent_run["sess-orphan"] = "run-orphan"
    dag_result.env_snapshots["seg-orphan"] = TeamEnvSnapshot(
        sandbox_ids=[], issue_snapshot_id=None, env_state={}
    )
    with pytest.raises(DAGError, match="multiple sinks"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


def test_assemble_rejects_dangling_agent_run_id():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Reference an agent_run_id that has no session.
    dag_result.segments[0] = SegmentSpec(
        segment_id="seg-planner",
        agent_run_id="run-ghost",
        issue_id="iss-planner",
        task_id="task-root",
        closing_event=EdgeType.DELEGATION,
        closing_event_target_segment="seg-worker",
        start_turn_idx=1,
        end_turn_idx=3,
    )
    with pytest.raises(DAGError, match="dangling"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )

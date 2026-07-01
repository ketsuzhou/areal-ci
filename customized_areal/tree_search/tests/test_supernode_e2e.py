"""End-to-end smoke test for Phase 1a: assemble -> insert -> backup.

Builds a 3-segment planner->worker->synthesizer DAG via SuperNodeAssembler,
inserts the SuperNodes into MCTSTreeStore, runs backup_episode_terminal from
the root terminal, and verifies credit flows along the unified parent_node_id
chain across all three segments (cross-SuperNode + cross-agent).
"""

from __future__ import annotations

from types import SimpleNamespace

from customized_areal.tree_search.agents.execution_dag import EdgeType
from customized_areal.tree_search.agents.supernode_assembler import (
    DagResult,
    EdgeSpec,
    SegmentSpec,
    SuperNodeAssembler,
    TeamEnvSnapshot,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore


def _node(node_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        parent_node_id=None,
        episode_id="ep",
        turn_idx=1,
        outcome_reward=0.0,
        train_id="",
        discarded=False,
    )


def test_e2e_assemble_insert_backup_flows_across_segments():
    # 1. Build sessions_nodes + DagResult for a 3-segment DAG.
    sessions_nodes = {
        "sess-planner": [_node("p1"), _node("p2"), _node("p3")],
        "sess-worker": [_node("w1"), _node("w2")],
        "sess-synth": [_node("s1")],
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
            seg_id: TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id=None,
                env_state={},
            )
            for seg_id in ("seg-planner", "seg-worker", "seg-synth")
        },
    )

    # 2. Assemble -> (super_nodes, dag, root_terminal_node_id).
    supers, dag, root_terminal = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    # root_terminal is the synthesizer's single turn (s1).
    assert root_terminal == "s1"

    # 3. Insert into MCTSTreeStore (no insert-time backup; we'll back up manually).
    store = MCTSTreeStore()
    store.insert_super_batch(supers, query_id="q", backup=False)

    # 4. Back up a reward of 1.0 from the root terminal.
    store.backup_episode_terminal(root_terminal, 1.0)

    # 5. Verify credit flows along the unified parent_node_id chain:
    #    s1 <- w2 <- w1 <- p3 <- p2 <- p1
    #    (synth's s1 has parent w2; w2's parent is w1; w1's parent is p3
    #    via the delegation edge; p3 <- p2 <- p1 within the planner segment.)
    for node_id in ("s1", "w2", "w1", "p3", "p2", "p1"):
        assert store.get_visit_count(node_id) == 1, (
            f"node {node_id!r} not visited by backup"
        )
        assert store.get_q_value(node_id) == 1.0, (
            f"node {node_id!r} did not receive reward 1.0"
        )

    # 6. The unique sink is the synthesizer SuperNode; verify it has no
    #    outgoing edges and its terminal node is root_terminal.
    sinks = [s for s in supers if not s.outgoing_edges]
    assert len(sinks) == 1
    assert sinks[0].terminal_node.node_id == root_terminal

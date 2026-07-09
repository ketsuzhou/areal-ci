"""Tests for EdgeType.BRANCH + fork provenance fields (Task 3.1).

A ``BRANCH`` edge records which segment + checkpoint it forked from. The
provenance fields round-trip through ``ExecutionDAG.to_records`` /
``from_records``; legacy edges (no provenance) serialize unchanged.
"""

from __future__ import annotations

from customized_areal.tree_search.agents.execution_dag import (
    Edge,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def test_edge_type_branch_value():
    assert EdgeType.BRANCH == "branch"


def test_branch_edge_carries_provenance():
    e = Edge(
        src="seg_parent",
        dst="seg_child",
        type=EdgeType.BRANCH,
        branch_from_segment_id="seg_parent",
        branch_from_checkpoint_id="ckpt_42",
    )
    assert e.branch_from_segment_id == "seg_parent"
    assert e.branch_from_checkpoint_id == "ckpt_42"
    # Non-BRANCH edges default provenance to None.
    legacy = Edge(src="a", dst="b", type=EdgeType.DELEGATION)
    assert legacy.branch_from_segment_id is None
    assert legacy.branch_from_checkpoint_id is None


def test_branch_edge_round_trips_through_records():
    dag = ExecutionDAG()
    dag.add_event(
        SuperNode(node_id="seg_parent", agent_id="r1", issue_id="i1", task_id="")
    )
    dag.add_event(
        SuperNode(node_id="seg_child", agent_id="r2", issue_id="i2", task_id="")
    )
    dag.add_edge(
        "seg_parent",
        "seg_child",
        EdgeType.BRANCH,
        branch_from_segment_id="seg_parent",
        branch_from_checkpoint_id="ckpt_42",
    )

    runs, edges = dag.to_records()
    assert edges == [
        {
            "src": "seg_parent",
            "dst": "seg_child",
            "type": "branch",
            "branch_from_segment_id": "seg_parent",
            "branch_from_checkpoint_id": "ckpt_42",
        }
    ]

    restored = ExecutionDAG.from_records(runs, edges)
    r_edge = restored.edges[0]
    assert r_edge.type is EdgeType.BRANCH
    assert r_edge.branch_from_segment_id == "seg_parent"
    assert r_edge.branch_from_checkpoint_id == "ckpt_42"


def test_non_branch_edges_records_omit_provenance():
    # Legacy edges (no provenance) serialize without the provenance keys, so
    # existing records / round-trips are unchanged.
    dag = ExecutionDAG()
    dag.add_event(SuperNode(node_id="a", agent_id="r1", issue_id="i1", task_id=""))
    dag.add_event(SuperNode(node_id="b", agent_id="r2", issue_id="i2", task_id=""))
    dag.add_edge("a", "b", EdgeType.DELEGATION)
    _, edges = dag.to_records()
    assert edges == [{"src": "a", "dst": "b", "type": "delegation"}]

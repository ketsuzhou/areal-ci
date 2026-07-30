"""Unit tests for the torch-free agent-execution DAG model.

These tests must run without the training stack (no torch). They cover the
contract from the design doc §6: topological order, cycle detection, fork/join
node identification, edge idempotency, ancestor/descendant traversal, and the
``from_records`` builder (explicit edges + delegation inference).
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    Edge,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def _node(
    node_id: str, *, issue_id: str = "", parent_issue_id: str | None = None
) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        issue_id=issue_id or f"issue-{node_id}",
        task_id=f"task-{node_id}",
    )


def _two_run_delegation_dag() -> ExecutionDAG:
    """planner (run A) delegates to worker (run B)."""
    dag = ExecutionDAG()
    dag.add_event(_node("A", issue_id="i-parent"))
    dag.add_event(_node("B", issue_id="i-child"))
    dag.add_edge("A", "B", EdgeType.DELEGATION)
    return dag


# -- node/edge construction ------------------------------------------------


def test_add_event_rejects_duplicate_event_id() -> None:
    dag = ExecutionDAG()
    dag.add_event(_node("A"))
    with pytest.raises(DAGError, match="duplicate event_id"):
        dag.add_event(_node("A"))


def test_add_edge_rejects_unknown_endpoints() -> None:
    dag = ExecutionDAG()
    dag.add_event(_node("A"))
    with pytest.raises(DAGError, match="unknown dst"):
        dag.add_edge("A", "missing", EdgeType.DELEGATION)
    with pytest.raises(DAGError, match="unknown src"):
        dag.add_edge("missing", "A", EdgeType.DELEGATION)


def test_add_edge_rejects_self_loop() -> None:
    dag = ExecutionDAG()
    dag.add_event(_node("A"))
    with pytest.raises(DAGError, match="self-loop"):
        dag.add_edge("A", "A", EdgeType.MENTION)


def test_add_edge_is_idempotent() -> None:
    dag = _two_run_delegation_dag()
    before = len(dag.edges)
    # Adding the identical edge again must not duplicate it.
    dag.add_edge("A", "B", EdgeType.DELEGATION)
    assert len(dag.edges) == before
    # A different edge type between the same pair is a distinct edge.
    dag.add_edge("A", "B", EdgeType.MENTION)
    assert len(dag.edges) == before + 1


def test_get_unknown_event_raises() -> None:
    dag = ExecutionDAG()
    with pytest.raises(DAGError, match="unknown event"):
        dag.get("nope")


# -- structural queries ----------------------------------------------------


def test_parents_children_and_degrees() -> None:
    dag = _two_run_delegation_dag()
    assert [n.node_id for n in dag.children("A")] == ["B"]
    assert [n.node_id for n in dag.parents("B")] == ["A"]
    assert dag.out_degree("A") == 1
    assert dag.in_degree("B") == 1
    assert dag.in_degree("A") == 0
    assert dag.out_degree("B") == 0


def test_roots_and_leaves() -> None:
    dag = _two_run_delegation_dag()
    assert [n.node_id for n in dag.roots()] == ["A"]
    assert [n.node_id for n in dag.leaves()] == ["B"]


def test_fork_and_join_events() -> None:
    # A delegates to B and C (fork); B and C both complete to D (join).
    dag = ExecutionDAG()
    for nid in ("A", "B", "C", "D"):
        dag.add_event(_node(nid))
    dag.add_edge("A", "B", EdgeType.DELEGATION)
    dag.add_edge("A", "C", EdgeType.DELEGATION)
    dag.add_edge("B", "D", EdgeType.COMPLETION)
    dag.add_edge("C", "D", EdgeType.COMPLETION)

    assert [n.node_id for n in dag.fork_events()] == ["A"]
    assert [n.node_id for n in dag.join_events()] == ["D"]


# -- traversal -------------------------------------------------------------


def test_descendants_cross_run_boundary() -> None:
    # A -> B -> C chain: descendants(A) must reach C across the run boundary.
    dag = ExecutionDAG()
    for nid in ("A", "B", "C"):
        dag.add_event(_node(nid))
    dag.add_edge("A", "B", EdgeType.DELEGATION)
    dag.add_edge("B", "C", EdgeType.DELEGATION)
    assert dag.descendants("A") == {"B", "C"}
    assert dag.ancestors("C") == {"A", "B"}


def test_ancestors_descendants_unknown_event_raises() -> None:
    dag = _two_run_delegation_dag()
    with pytest.raises(DAGError):
        dag.ancestors("missing")
    with pytest.raises(DAGError):
        dag.descendants("missing")


def test_topological_order_respects_edges() -> None:
    dag = ExecutionDAG()
    for nid in ("A", "B", "C", "D"):
        dag.add_event(_node(nid))
    dag.add_edge("A", "B", EdgeType.DELEGATION)
    dag.add_edge("A", "C", EdgeType.DELEGATION)
    dag.add_edge("B", "D", EdgeType.COMPLETION)
    dag.add_edge("C", "D", EdgeType.COMPLETION)

    order = [n.node_id for n in dag.topological_order()]
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")
    assert dag.is_acyclic() is True


def test_cycle_detection() -> None:
    dag = ExecutionDAG()
    for nid in ("A", "B", "C"):
        dag.add_event(_node(nid))
    dag.add_edge("A", "B", EdgeType.MENTION)
    dag.add_edge("B", "C", EdgeType.MENTION)
    dag.add_edge("C", "A", EdgeType.MENTION)

    assert dag.is_acyclic() is False
    with pytest.raises(DAGError, match="cycle"):
        dag.topological_order()


def test_iter_topo_matches_topological_order() -> None:
    dag = _two_run_delegation_dag()
    assert [n.node_id for n in dag.iter_topo()] == [
        n.node_id for n in dag.topological_order()
    ]


# -- from_records builder --------------------------------------------------


def test_from_records_with_explicit_edges() -> None:
    runs = [
        {"node_id": "A", "agent_id": "ag", "issue_id": "i1", "task_id": "t1"},
        {"node_id": "B", "agent_id": "ag", "issue_id": "i2", "task_id": "t2"},
    ]
    edges = [{"src": "A", "dst": "B", "type": "delegation"}]
    dag = ExecutionDAG.from_records(runs, edges)
    assert len(dag) == 2
    assert dag.edges == [Edge(src="A", dst="B", type=EdgeType.DELEGATION)]


def test_from_records_infers_delegation_from_parent_issue_id() -> None:
    runs = [
        {"node_id": "A", "agent_id": "ag", "issue_id": "i-parent", "task_id": "t1"},
        {
            "node_id": "B",
            "agent_id": "ag",
            "issue_id": "i-child",
            "task_id": "t2",
            "parent_issue_id": "i-parent",
        },
    ]
    dag = ExecutionDAG.from_records(runs)
    assert [n.node_id for n in dag.children("A")] == ["B"]
    assert dag.edges[0].type is EdgeType.DELEGATION


def test_from_records_carries_optional_fields() -> None:
    runs = [
        {
            "node_id": "A",
            "agent_id": "ag",
            "issue_id": "i1",
            "task_id": "t1",
            "env_id": "env-7",
            "outcome_reward": 1.0,
        },
    ]
    dag = ExecutionDAG.from_records(runs)
    node = dag.get("A")
    assert node.env_id == "env-7"
    assert node.outcome_reward == 1.0


# -- SuperNode: nodes + env snapshot --------------------------------------


def _super(node_id: str, *, nodes=None, sandbox_ids=None) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        issue_id=f"issue-{node_id}",
        task_id=f"task-{node_id}",
        nodes=list(nodes or []),
        sandbox_ids=list(sandbox_ids or []),
    )


def test_supernode_defaults_empty_nodes_and_env():
    s = SuperNode(node_id="s1", agent_id="a", issue_id="i", task_id="t")
    assert s.nodes == []
    assert s.sandbox_ids == []
    assert s.issue_snapshot_id is None
    assert s.env_state == {}
    assert s.closing_event is None
    assert s.terminal_node is None
    assert s.branch_node_id is None


def test_supernode_terminal_node_is_last():
    # Lightweight stand-ins; Node is torch-lazy and heavy to build, so use
    # SimpleNamespace for the structural test (terminal_node only reads .node_id).
    from types import SimpleNamespace

    n0 = SimpleNamespace(node_id="n0")
    n1 = SimpleNamespace(node_id="n1")
    s = _super("s1", nodes=[n0, n1])
    assert s.terminal_node is n1
    assert s.branch_node_id == "n1"


def test_supernode_branch_node_id_none_when_empty():
    s = _super("s1")
    assert s.terminal_node is None
    assert s.branch_node_id is None

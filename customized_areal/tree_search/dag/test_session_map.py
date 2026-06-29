"""Tests for the {agent_run -> session_id} map on the execution DAG.

The verifier agent needs to address rewards per RL ``session_id`` (design
decision 3). Each :class:`AgentRunNode` carries the ``session_id`` assigned at
``rl_start_session`` time; :meth:`ExecutionDAG.session_map` exposes the mapping
for the whole task, and the mapping survives a checkpoint round-trip.

Torch-free -- runs without the training stack.
"""

from __future__ import annotations

from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    EdgeType,
    ExecutionDAG,
)


def _run(node_id: str, *, session_id: str | None = None) -> AgentRunNode:
    return AgentRunNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        issue_id=f"issue-{node_id}",
        task_id=f"task-{node_id}",
        session_id=session_id,
    )


def test_agent_run_node_session_id_defaults_to_none() -> None:
    node = AgentRunNode(node_id="A", agent_id="ag", issue_id="i", task_id="t")
    assert node.session_id is None


def test_session_map_returns_node_to_session() -> None:
    dag = ExecutionDAG()
    dag.add_node(_run("A", session_id="sess-A"))
    dag.add_node(_run("B", session_id="sess-B"))
    dag.add_edge("A", "B", EdgeType.DELEGATION)

    assert dag.session_map() == {"A": "sess-A", "B": "sess-B"}


def test_session_map_includes_none_for_session_less_node() -> None:
    dag = ExecutionDAG()
    dag.add_node(_run("A", session_id="sess-A"))
    dag.add_node(_run("B"))  # no session started yet

    assert dag.session_map() == {"A": "sess-A", "B": None}


def test_set_session_id_populates_node() -> None:
    dag = ExecutionDAG()
    dag.add_node(_run("A"))
    dag.set_session_id("A", "sess-A")
    assert dag.get("A").session_id == "sess-A"
    assert dag.session_map() == {"A": "sess-A"}


def test_to_records_round_trip_preserves_session_id() -> None:
    dag = ExecutionDAG()
    dag.add_node(_run("A", session_id="sess-A"))
    dag.add_node(_run("B", session_id="sess-B"))
    dag.add_edge("A", "B", EdgeType.DELEGATION)

    runs, edges = dag.to_records()
    rebuilt = ExecutionDAG.from_records(runs, edges)

    assert rebuilt.session_map() == {"A": "sess-A", "B": "sess-B"}
    assert [e.type for e in rebuilt.edges] == [EdgeType.DELEGATION]

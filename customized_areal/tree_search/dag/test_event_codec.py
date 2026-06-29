"""Tests for the bidirectional DAG <-> linear Event codec.

The multi-lane fixture mirrors the worked example in CRITIC_GAE_INTEGRATION.md:
    O0 --delegation--> R0 ; O0 --delegation--> C0 ; R0 --mention--> C0 ;
    C0 --delegation--> T0 ; T0 --completion--> C1 ; R0 --completion--> O1 ;
    C1 --completion--> O1
Global completion order: O0, R0, C0, T0, C1, O1.

Torch-free.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.event_codec import dag_to_events
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    DAGError,
    EdgeType,
    ExecutionDAG,
)

ORDER = ["O0", "R0", "C0", "T0", "C1", "O1"]
EDGES = [
    ("O0", "R0", EdgeType.DELEGATION),
    ("O0", "C0", EdgeType.DELEGATION),
    ("R0", "C0", EdgeType.MENTION),
    ("C0", "T0", EdgeType.DELEGATION),
    ("T0", "C1", EdgeType.COMPLETION),
    ("R0", "O1", EdgeType.COMPLETION),
    ("C1", "O1", EdgeType.COMPLETION),
]


def _build_dag() -> ExecutionDAG:
    dag = ExecutionDAG()
    for i, nid in enumerate(ORDER):
        node = AgentRunNode(
            node_id=nid, agent_id=nid[0], issue_id=f"iss-{nid}", task_id=f"task-{nid}"
        )
        node.value = 0.1 * i
        node.process_reward = 0.0
        node.metadata = {
            "messages": [{"role": "assistant", "content": f"{nid}-out"}],
            "completion_time": float(i),
        }
        dag.add_node(node)
    for src, dst, t in EDGES:
        dag.add_edge(src, dst, t)
    dag.get("O1").outcome_reward = 1.0
    return dag


def test_dag_to_events_uses_explicit_ordering_and_dense_index() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    assert [e.node_id for e in events] == ORDER
    assert [e.completion_index for e in events] == [0, 1, 2, 3, 4, 5]


def test_dag_to_events_fills_both_edge_directions() -> None:
    dag = _build_dag()
    events = {e.node_id: e for e in dag_to_events(dag, ordering=ORDER)}
    assert set(events["C0"].incoming_edges) == {
        ("O0", EdgeType.DELEGATION),
        ("R0", EdgeType.MENTION),
    }
    assert events["C0"].outgoing_edges == (("T0", EdgeType.DELEGATION),)
    assert events["O0"].incoming_edges == ()
    assert set(events["O0"].outgoing_edges) == {
        ("R0", EdgeType.DELEGATION),
        ("C0", EdgeType.DELEGATION),
    }


def test_dag_to_events_copies_node_fields_and_messages() -> None:
    dag = _build_dag()
    events = {e.node_id: e for e in dag_to_events(dag, ordering=ORDER)}
    o1 = events["O1"]
    assert o1.outcome_reward == 1.0
    assert o1.value == pytest.approx(0.5)
    assert o1.task_id == "task-O1"
    assert o1.messages == ({"role": "assistant", "content": "O1-out"},)
    assert o1.completion_time == 5.0


def test_dag_to_events_falls_back_to_topological_order() -> None:
    dag = _build_dag()
    events = dag_to_events(dag)
    order = [e.node_id for e in events]
    pos = {nid: i for i, nid in enumerate(order)}
    for src, dst, _ in EDGES:
        assert pos[src] < pos[dst]


def test_dag_to_events_rejects_non_permutation_ordering() -> None:
    dag = _build_dag()
    with pytest.raises(DAGError):
        dag_to_events(dag, ordering=["O0", "R0"])


def test_dag_to_events_rejects_non_topological_ordering() -> None:
    dag = _build_dag()
    bad = ["R0", "O0", "C0", "T0", "C1", "O1"]
    with pytest.raises(DAGError):
        dag_to_events(dag, ordering=bad)


def test_dag_to_events_messages_by_node_overrides_metadata() -> None:
    dag = _build_dag()
    override = {"O0": [{"role": "assistant", "content": "override"}]}
    events = {
        e.node_id: e
        for e in dag_to_events(dag, ordering=ORDER, messages_by_node=override)
    }
    assert events["O0"].messages == ({"role": "assistant", "content": "override"},)
    assert events["R0"].messages == ({"role": "assistant", "content": "R0-out"},)

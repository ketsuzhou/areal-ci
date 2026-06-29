"""Tests for the bidirectional DAG <-> linear Event codec.

The multi-lane fixture mirrors the worked example in CRITIC_GAE_INTEGRATION.md:
    O0 --delegation--> R0 ; O0 --delegation--> C0 ; R0 --mention--> C0 ;
    C0 --delegation--> T0 ; T0 --completion--> C1 ; R0 --completion--> O1 ;
    C1 --completion--> O1
Global completion order: O0, R0, C0, T0, C1, O1.

Torch-free.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from customized_areal.tree_search.dag.critic_observation import (
    build_critic_observations,
)
from customized_areal.tree_search.dag.event_codec import (
    ReplayPrefix,
    dag_to_events,
    events_to_dag,
    replay_prefix_for,
)
from customized_areal.tree_search.dag.event_model import Event, message_timeline
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    DAGError,
    EdgeType,
    ExecutionDAG,
)
from customized_areal.tree_search.dag.gae import GlobalEvent, events_from_nodes

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


def test_events_to_dag_round_trip_rebuilds_nodes_and_edges() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    rebuilt = events_to_dag(events)
    assert sorted(rebuilt.node_ids()) == sorted(ORDER)
    orig = {(e.src, e.dst, e.type) for e in dag.edges}
    back = {(e.src, e.dst, e.type) for e in rebuilt.edges}
    assert back == orig
    assert rebuilt.get("O1").outcome_reward == 1.0
    assert rebuilt.get("C0").value == pytest.approx(0.2)


def test_events_to_dag_rejects_non_dense_index() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    events[2] = replace(events[2], completion_index=0)
    with pytest.raises(DAGError, match="dense"):
        events_to_dag(events)


def test_events_to_dag_rejects_duplicate_node_id() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    # Reuse O0's node_id for R0 while keeping completion_index dense and unique
    # so the duplicate-node_id branch fires (not the density check).
    events[1] = replace(events[1], node_id=events[0].node_id)
    with pytest.raises(DAGError, match="duplicate node_id"):
        events_to_dag(events)


def test_events_to_dag_rejects_asymmetric_edges() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    events = [
        replace(e, incoming_edges=(("R0", EdgeType.MENTION),))
        if e.node_id == "C0"
        else e
        for e in events
    ]
    with pytest.raises(DAGError, match="symmetry mismatch"):
        events_to_dag(events)


def test_events_to_dag_rejects_non_topological_index() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    swapped = []
    for e in events:
        if e.node_id == "O0":
            swapped.append(replace(e, completion_index=1))
        elif e.node_id == "R0":
            swapped.append(replace(e, completion_index=0))
        else:
            swapped.append(e)
    with pytest.raises(DAGError, match="not topological"):
        events_to_dag(swapped)


def test_events_to_dag_empty_returns_empty_dag() -> None:
    rebuilt = events_to_dag([])
    assert rebuilt.node_ids() == []


def test_full_dict_round_trip_identity() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    redecoded = [Event.from_dict(e.to_dict()) for e in events]
    assert redecoded == events
    dag_a = events_to_dag(events)
    dag_b = events_to_dag(redecoded)
    assert {(e.src, e.dst, e.type) for e in dag_a.edges} == {
        (e.src, e.dst, e.type) for e in dag_b.edges
    }
    assert sorted(dag_a.node_ids()) == sorted(dag_b.node_ids())


def _build_dag_with_branch() -> ExecutionDAG:
    dag = _build_dag()
    c1 = dag.get("C1")
    c1.branch_seq = 7
    c1.branch_env_snapshot_id = "snap-C1"
    return dag


def test_replay_prefix_for_returns_ancestor_slice() -> None:
    dag = _build_dag_with_branch()
    events = dag_to_events(dag, ordering=ORDER)
    prefix = replay_prefix_for(events, branch_point=("task-C1", 7))
    assert isinstance(prefix, ReplayPrefix)
    assert prefix.task_id == "task-C1"
    assert prefix.seq == 7
    assert prefix.source_issue_id == "iss-C1"
    assert prefix.branch_env_snapshot_id == "snap-C1"
    contents = [m["content"] for m in prefix.replay_messages]
    assert contents == ["O0-out", "R0-out", "C0-out", "T0-out", "C1-out"]


def test_replay_prefix_for_unknown_branch_point_raises() -> None:
    dag = _build_dag_with_branch()
    events = dag_to_events(dag, ordering=ORDER)
    with pytest.raises(DAGError):
        replay_prefix_for(events, branch_point=("task-C1", 999))


def test_replay_prefix_for_ambiguous_branch_point_raises() -> None:
    dag = _build_dag_with_branch()
    dag.get("C0").task_id = "task-dup"
    dag.get("C0").branch_seq = 42
    dag.get("T0").task_id = "task-dup"
    dag.get("T0").branch_seq = 42
    events = dag_to_events(dag, ordering=ORDER)
    with pytest.raises(DAGError):
        replay_prefix_for(events, branch_point=("task-dup", 42))


def _old_events_from_nodes(ordered_nodes):
    """Snapshot of the pre-refactor logic, for parity comparison."""
    out = []
    for node in ordered_nodes:
        value = getattr(node, "value", None)
        out.append(
            GlobalEvent(
                node_id=node.node_id,
                value=float(value) if value is not None else 0.0,
                reward=float(node.process_reward) + float(node.outcome_reward),
            )
        )
    return out


def test_events_from_nodes_parity_with_old_logic() -> None:
    dag = _build_dag()
    nodes = [dag.get(nid) for nid in ORDER]
    assert events_from_nodes(nodes) == _old_events_from_nodes(nodes)


def test_events_from_nodes_unscored_value_is_zero() -> None:
    node = AgentRunNode(node_id="n", agent_id="a", issue_id="i", task_id="t")
    node.process_reward = 0.25
    (ev,) = events_from_nodes([node])
    assert ev == GlobalEvent(node_id="n", value=0.0, reward=0.25)


def test_critic_observations_match_message_timeline_of_events() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    timeline_from_events = message_timeline(events)
    obs_from_events = build_critic_observations(timeline_from_events)
    hand_built = [
        {"role": "assistant", "content": f"{nid}-out", "node_id": nid} for nid in ORDER
    ]
    obs_hand = build_critic_observations(hand_built)
    assert [o.node_id for o in obs_from_events] == [o.node_id for o in obs_hand]
    assert [o.value_index for o in obs_from_events] == [o.value_index for o in obs_hand]
    assert obs_from_events[0].node_id is None
    assert len(obs_from_events) == len(ORDER) + 1

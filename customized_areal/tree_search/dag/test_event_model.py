"""Tests for the canonical linear Event model (DAG trajectory log).

Torch-free.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.event_model import Event, message_timeline
from customized_areal.tree_search.dag.execution_dag import DAGError, EdgeType


def _sample_event() -> Event:
    return Event(
        node_id="C0",
        agent_id="coder",
        issue_id="iss-2",
        task_id="task-2",
        completion_index=2,
        incoming_edges=(("O0", EdgeType.DELEGATION), ("R0", EdgeType.MENTION)),
        outgoing_edges=(("T0", EdgeType.DELEGATION),),
        session_id="sess-2",
        completion_time=3.0,
        branch_seq=5,
        branch_issue_id="iss-2-fork",
        branch_env_snapshot_id="snap-2",
        value=0.6,
        process_reward=0.1,
        outcome_reward=0.0,
        messages=({"role": "assistant", "content": "draft"},),
        metadata={"k": "v"},
    )


def test_to_dict_is_json_safe() -> None:
    d = _sample_event().to_dict()
    assert d["incoming_edges"] == [["O0", "delegation"], ["R0", "mention"]]
    assert d["outgoing_edges"] == [["T0", "delegation"]]
    assert isinstance(d["messages"], list)
    assert d["completion_index"] == 2


def test_from_dict_to_dict_round_trip_equal() -> None:
    e = _sample_event()
    assert Event.from_dict(e.to_dict()) == e


def test_from_dict_coerces_edge_type_strings() -> None:
    e = _sample_event()
    rebuilt = Event.from_dict(e.to_dict())
    assert rebuilt.incoming_edges[0] == ("O0", EdgeType.DELEGATION)
    assert isinstance(rebuilt.incoming_edges[0][1], EdgeType)


def test_from_dict_unknown_edge_type_raises() -> None:
    raw = _sample_event().to_dict()
    raw["incoming_edges"] = [["O0", "not-a-real-type"]]
    with pytest.raises(DAGError):
        Event.from_dict(raw)


def test_from_dict_missing_required_field_raises() -> None:
    raw = _sample_event().to_dict()
    del raw["node_id"]
    with pytest.raises(DAGError):
        Event.from_dict(raw)


def test_defaults_are_empty() -> None:
    e = Event(node_id="n", agent_id="a", issue_id="i", task_id="t", completion_index=0)
    assert e.incoming_edges == ()
    assert e.outgoing_edges == ()
    assert e.messages == ()
    assert e.value is None
    assert e.process_reward == 0.0
    assert e.metadata == {}


def test_message_timeline_orders_by_completion_index_and_tags_output() -> None:
    e1 = Event(
        node_id="O0",
        agent_id="orch",
        issue_id="i",
        task_id="t",
        completion_index=1,
        messages=(
            {"role": "user", "content": "ctx"},
            {"role": "assistant", "content": "plan"},
        ),
    )
    e0 = Event(
        node_id="seed",
        agent_id="orch",
        issue_id="i",
        task_id="t",
        completion_index=0,
        messages=({"role": "assistant", "content": "boot"},),
    )
    timeline = message_timeline([e1, e0])
    assert [m["content"] for m in timeline] == ["boot", "ctx", "plan"]
    assert timeline[0]["node_id"] == "seed"
    assert "node_id" not in timeline[1]
    assert timeline[2]["node_id"] == "O0"


def test_message_timeline_does_not_mutate_source() -> None:
    src = {"role": "assistant", "content": "x"}
    e = Event(
        node_id="n",
        agent_id="a",
        issue_id="i",
        task_id="t",
        completion_index=0,
        messages=(src,),
    )
    message_timeline([e])
    assert "node_id" not in src

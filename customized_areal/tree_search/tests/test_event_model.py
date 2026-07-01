"""Tests for the message_timeline helper (Event class removed).

After the SuperNode unification, event_model.py no longer defines Event;
the linear-log role is absorbed by SuperNode. message_timeline survives
because the critic observation builder consumes it. It now takes SuperNodes
(or any object with .completion_index, .node_id, and .nodes).
"""

from __future__ import annotations

from types import SimpleNamespace

from customized_areal.tree_search.agents.event_model import (
    EdgeRef,
    message_timeline,
)


def test_edgeref_is_alias_for_tuple():
    # EdgeRef is a type alias: tuple[str, EdgeType].
    from customized_areal.tree_search.agents.execution_dag import EdgeType

    ref: EdgeRef = ("n0", EdgeType.DELEGATION)
    assert ref[0] == "n0"
    assert ref[1] is EdgeType.DELEGATION


def _super(node_id, completion_index, messages):
    """Build a SuperNode-like object with the fields message_timeline reads.

    ``message_timeline`` reads ``nodes[i].messages`` (the SuperNode shape);
    the ``.messages`` attribute is NOT set, so the fallback path is the only
    path exercised.
    """
    return SimpleNamespace(
        node_id=node_id,
        completion_index=completion_index,
        nodes=[SimpleNamespace(messages=[m]) for m in messages],
    )


def test_message_timeline_orders_by_completion_index_and_tags_output():
    s1 = _super(
        "O0",
        completion_index=1,
        messages=[
            {"role": "user", "content": "ctx"},
            {"role": "assistant", "content": "plan"},
        ],
    )
    s0 = _super(
        "seed",
        completion_index=0,
        messages=[
            {"role": "assistant", "content": "boot"},
        ],
    )
    timeline = message_timeline([s1, s0])
    assert [m["content"] for m in timeline] == ["boot", "ctx", "plan"]
    assert timeline[0]["node_id"] == "seed"
    assert "node_id" not in timeline[1]
    assert timeline[2]["node_id"] == "O0"


def test_message_timeline_does_not_mutate_source():
    src = {"role": "assistant", "content": "x"}
    s = _super("n", completion_index=0, messages=[src])
    message_timeline([s])
    assert "node_id" not in src


def test_event_class_removed():
    # Importing Event must now fail; it was absorbed by SuperNode.
    from customized_areal.tree_search.agents import event_model

    assert not hasattr(event_model, "Event")

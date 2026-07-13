import pytest

from customized_areal.tree_search.agents.dag_backup import distribute_reward_over_dag
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    SuperNode,
)
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    EdgeSpec,
    SegmentSpec,
)
from customized_areal.tree_search.agents.supernode_assembler import SuperNodeAssembler


class FakeResolver:
    """Records calls and returns a fixed torch-free tensor dict."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, tensor_ref: dict) -> dict:
        self.calls.append(tensor_ref["shard_id"])
        return {
            "input_ids": [1, 2],
            "loss_mask": [1, 1],
            "logprobs": [0.0, 0.0],
            "versions": [1, 1],
            "attention_mask": [1, 1],
            "rewards": [0.0, 0.0],
        }


def _dag() -> AssembledDag:
    return AssembledDag(
        segments=[
            SegmentSpec(
                segment_id="seg-1",
                agent_run_id="ar-1",
                issue_id="i-1",
                trajectory_id=0,
                tensor_ref={"shard_id": "sh-1", "node_addr": "x"},
                closing_event="completion",
                env_snapshot={},
            ),
            SegmentSpec(
                segment_id="seg-2",
                agent_run_id="ar-1",
                issue_id="i-1",
                trajectory_id=1,
                tensor_ref={"shard_id": "sh-2", "node_addr": "x"},
                closing_event=None,
                env_snapshot={},
            ),
        ],
        edges=[EdgeSpec(src_segment_id="seg-1", dst_segment_id="seg-2", type="completion")],
        session_to_agent_run={"s-1": "ar-1"},
    )


def test_assemble_from_refs_builds_one_supernode_per_segment():
    dag = _dag()
    resolver = FakeResolver()
    edag = SuperNodeAssembler().assemble_from_refs(dag, resolver)

    # one SuperNode per segment, one resolve per segment
    assert len(edag.events) == 2
    assert len(resolver.calls) == 2

    # segment_id is the node_id; tensors resolved onto metadata
    seg1 = edag.get("seg-1")
    assert seg1.metadata["segment_id"] == "seg-1"
    assert seg1.metadata["trajectory_id"] == 0
    assert seg1.metadata["tensors"]["input_ids"] == [1, 2]
    # session_id reverse-mapped from session_to_agent_run
    assert seg1.session_id == "s-1"
    # closing_event coerced to EdgeType
    assert seg1.closing_event is not None
    assert seg1.closing_event.value == "completion"

    # edge preserved (src/dst are segment ids == node ids)
    assert len(edag.edges) == 1
    assert edag.edges[0].src == "seg-1"
    assert edag.edges[0].dst == "seg-2"
    assert edag.edges[0].type.value == "completion"

    # completion_index stamped from topological order (seg-1 before seg-2)
    assert edag.get("seg-1").completion_index == 0
    assert edag.get("seg-2").completion_index == 1


def test_assemble_from_refs_returns_none_for_empty_trajectory():
    # A dag with no segments (no trajectory recorded) -> None so callers skip
    # the episode rather than train on an empty graph.
    dag = AssembledDag(segments=[], edges=[], session_to_agent_run={})
    resolver = FakeResolver()
    assert SuperNodeAssembler().assemble_from_refs(dag, resolver) is None
    assert resolver.calls == []  # no segments resolved


def test_assemble_from_refs_rejects_cycle():
    dag = AssembledDag(
        segments=[
            SegmentSpec("a", "ar", "i", 0, {"shard_id": "a"}, None, {}),
            SegmentSpec("b", "ar", "i", 1, {"shard_id": "b"}, None, {}),
        ],
        edges=[
            EdgeSpec("a", "b", "mention"),
            EdgeSpec("b", "a", "mention"),
        ],
        session_to_agent_run={"s": "ar"},
    )
    with pytest.raises(DAGError):
        SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())


def test_assemble_from_refs_env_snapshot_stamped():
    dag = AssembledDag(
        segments=[
            SegmentSpec(
                segment_id="seg-x",
                agent_run_id="ar-x",
                issue_id="i-x",
                trajectory_id=0,
                tensor_ref={"shard_id": "sh-x"},
                closing_event=None,
                env_snapshot={
                    "sandbox_ids": ["sb-1", "sb-2"],
                    "issue_snapshot_id": "isnap-1",
                    "env_state": {"foo": 1},
                },
            ),
        ],
        edges=[],
        session_to_agent_run={"s-x": "ar-x"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    node = edag.get("seg-x")
    assert node.sandbox_ids == ["sb-1", "sb-2"]
    assert node.issue_snapshot_id == "isnap-1"
    assert node.env_state == {"foo": 1}


def test_assemble_from_refs_populates_edge_tuples():
    """v2-assembled SuperNodes carry topology in incoming_edges /
    outgoing_edges (not just edag.edges) so a to_dict() round-trip preserves it."""
    dag = _dag()  # seg-1 --completion--> seg-2
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())

    seg1 = edag.get("seg-1")
    seg2 = edag.get("seg-2")
    assert seg1.outgoing_edges == (("seg-2", EdgeType.COMPLETION),)
    assert seg1.incoming_edges == ()
    assert seg2.incoming_edges == (("seg-1", EdgeType.COMPLETION),)
    assert seg2.outgoing_edges == ()


def test_assemble_from_refs_leaf_segment_has_empty_edge_tuples():
    """A segment with no edges has empty edge tuples (regression guard)."""
    dag = AssembledDag(
        segments=[
            SegmentSpec("seg-solo", "ar", "i", 0, {"shard_id": "s"}, None, {}),
        ],
        edges=[],
        session_to_agent_run={"s": "ar"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    solo = edag.get("seg-solo")
    assert solo.incoming_edges == ()
    assert solo.outgoing_edges == ()


def test_assemble_from_refs_round_trip_preserves_topology_and_visit_count():
    """assemble_from_refs -> to_dict -> from_dict preserves edges, edge tuples,
    visit_count, closing_event, and tensors (the v2-path lossless invariant)."""
    dag = AssembledDag(
        segments=[
            SegmentSpec("seg-a", "ar", "i", 0, {"shard_id": "a"}, "completion", {}),
            SegmentSpec("seg-b", "ar", "i", 1, {"shard_id": "b"}, None, {}),
        ],
        edges=[EdgeSpec("seg-a", "seg-b", "completion")],
        session_to_agent_run={"s": "ar"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    # Seed visit_count on seg-a (as branch_backup would) to assert non-zero
    # round-trip - the field that was previously dropped by to_dict/from_dict.
    edag.get("seg-a").visit_count = 4

    a = SuperNode.from_dict(edag.get("seg-a").to_dict())
    b = SuperNode.from_dict(edag.get("seg-b").to_dict())
    # Topology survives (tuples populated by assemble_from_refs).
    assert a.outgoing_edges == (("seg-b", EdgeType.COMPLETION),)
    assert a.incoming_edges == ()
    assert b.incoming_edges == (("seg-a", EdgeType.COMPLETION),)
    assert b.outgoing_edges == ()
    # visit_count survives (serialized by to_dict/from_dict).
    assert a.visit_count == 4
    assert b.visit_count == 0
    # closing_event + tensors survive.
    assert a.closing_event == EdgeType.COMPLETION
    assert b.closing_event is None
    assert a.metadata["tensors"]["input_ids"] == [1, 2]
    assert b.metadata["tensors"]["input_ids"] == [1, 2]


def test_assemble_from_refs_fan_in_credits_all_parents():
    """A segment with two incoming COMPLETION edges (fan-in join) ->
    distribute_reward_over_dag credits every parent (Bug #2 class, v2 path)."""
    dag = AssembledDag(
        segments=[
            SegmentSpec("seg-child-a", "ar-a", "i", 0, {"shard_id": "ca"}, "completion", {}),
            SegmentSpec("seg-child-b", "ar-b", "i", 0, {"shard_id": "cb"}, "completion", {}),
            SegmentSpec("seg-parent", "ar-c", "i", 1, {"shard_id": "p"}, None, {}),
        ],
        edges=[
            EdgeSpec("seg-child-a", "seg-parent", "completion"),
            EdgeSpec("seg-child-b", "seg-parent", "completion"),
        ],
        session_to_agent_run={"s-a": "ar-a", "s-b": "ar-b", "s-c": "ar-c"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    credit = distribute_reward_over_dag(
        edag, terminal_reward=1.0, terminal_node_id="seg-parent"
    )
    # Default (no fan_in_credit): split equally across the two parents -> 0.5 each.
    assert credit["seg-child-a"] > 0.0
    assert credit["seg-child-b"] > 0.0
    assert credit["seg-child-a"] == credit["seg-child-b"] == 0.5

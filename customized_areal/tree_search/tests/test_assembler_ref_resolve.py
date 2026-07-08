import pytest

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
    with pytest.raises(Exception):
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

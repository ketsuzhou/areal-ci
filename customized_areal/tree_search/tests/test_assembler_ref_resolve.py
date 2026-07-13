import logging

import pytest

from customized_areal.tree_search.agents.execution_dag import DAGError
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    EdgeSpec,
    SegmentSpec,
    StepReward,
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


def _dag_with_rewards() -> AssembledDag:
    """_dag() with seg-1 carrying two diagnosis step rewards (scores 8, 6)."""
    dag = _dag()
    dag.step_rewards = [
        StepReward(segment_id="seg-1", seq=1, score=8, rationale="good"),
        StepReward(segment_id="seg-1", seq=2, score=6, rationale="ok"),
    ]
    dag.score_max = 10
    return dag


def test_assemble_from_refs_aggregates_step_rewards_to_process_reward():
    dag = _dag_with_rewards()
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    # mean([8, 6]) / score_max(10) = 0.7 -> SuperNode.process_reward (per-segment
    # GAE reward; dag_advantage.events_from_nodes consumes super_node.process_reward).
    assert edag.get("seg-1").process_reward == pytest.approx(0.7)
    # seg-2 has no step_rewards -> sparse 0.0 (no fabricated default).
    assert edag.get("seg-2").process_reward == 0.0


def test_assemble_from_refs_score_max_zero_is_sparse():
    # score_max 0 means diagnosis scoring was not configured: no normalization,
    # process_reward stays 0.0 (absence distinguishable, not a fabricated reward).
    dag = _dag_with_rewards()
    dag.score_max = 0
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    assert edag.get("seg-1").process_reward == 0.0


class _ListHandler(logging.Handler):
    """Captures LogRecords directly on a named logger (robust to root-handler
    config - caplog's root handler does not always catch a fresh stdlib logger
    with no root handlers installed)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_assemble_from_refs_drops_unmatched_step_rewards():
    # A step reward whose segment_id has no matching segment is dropped + logged,
    # not applied and not fatal. The matching rewards still land and the ghost
    # creates no spurious SuperNode.
    dag = _dag_with_rewards()
    dag.step_rewards.append(
        StepReward(segment_id="seg-ghost", seq=1, score=9, rationale="x")
    )
    asm_logger = logging.getLogger("SuperNodeAssembler")
    handler = _ListHandler()
    prev_level = asm_logger.level
    asm_logger.addHandler(handler)
    asm_logger.setLevel(logging.WARNING)
    try:
        edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    finally:
        asm_logger.removeHandler(handler)
        asm_logger.setLevel(prev_level)
    assert edag.get("seg-1").process_reward == pytest.approx(0.7)
    assert len(edag.events) == 2  # ghost dropped, no spurious SuperNode
    assert any("seg-ghost" in r.getMessage() for r in handler.records)

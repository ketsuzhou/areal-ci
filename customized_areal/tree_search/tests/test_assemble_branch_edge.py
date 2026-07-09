"""Tests for BRANCH edge parsing in assemble_from_refs (Task 3.2).

Multica emits ``branch`` edges carrying fork provenance; AReaL parses them into
``EdgeType.BRANCH`` edges with ``branch_from_segment_id`` /
``branch_from_checkpoint_id`` on the assembled ``ExecutionDAG``.
"""

from __future__ import annotations

from customized_areal.tree_search.agents.execution_dag import EdgeType
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    EdgeSpec,
    SegmentSpec,
)
from customized_areal.tree_search.agents.supernode_assembler import SuperNodeAssembler


class _FakeResolver:
    def resolve(self, tensor_ref):  # noqa: D401 - minimal fake
        return {}


def _seg(segment_id: str, agent_run_id: str) -> SegmentSpec:
    return SegmentSpec(
        segment_id=segment_id,
        agent_run_id=agent_run_id,
        issue_id="i",
        trajectory_id=0,
        tensor_ref={"shard_id": f"shard_{segment_id}"},
        closing_event=None,
        env_snapshot={},
    )


def test_assemble_from_refs_parses_branch_edge_with_provenance():
    dag = AssembledDag(
        segments=[_seg("seg_parent", "r1"), _seg("seg_child", "r2")],
        edges=[
            EdgeSpec(
                src_segment_id="seg_parent",
                dst_segment_id="seg_child",
                type="branch",
                branch_from_segment_id="seg_parent",
                branch_from_checkpoint_id="ckpt_42",
            )
        ],
        session_to_agent_run={"sess1": "r1", "sess2": "r2"},
    )

    edag = SuperNodeAssembler().assemble_from_refs(dag, _FakeResolver())

    branch_edges = [e for e in edag.edges if e.type is EdgeType.BRANCH]
    assert len(branch_edges) == 1
    be = branch_edges[0]
    assert be.src == "seg_parent"
    assert be.dst == "seg_child"
    assert be.branch_from_segment_id == "seg_parent"
    assert be.branch_from_checkpoint_id == "ckpt_42"


def test_assemble_from_refs_non_branch_edges_have_no_provenance():
    # A delegation edge (no provenance) assembles without fork provenance.
    dag = AssembledDag(
        segments=[_seg("seg_a", "r1"), _seg("seg_b", "r2")],
        edges=[
            EdgeSpec(
                src_segment_id="seg_a", dst_segment_id="seg_b", type="delegation"
            )
        ],
        session_to_agent_run={"sess1": "r1", "sess2": "r2"},
    )

    edag = SuperNodeAssembler().assemble_from_refs(dag, _FakeResolver())

    assert len(edag.edges) == 1
    e = edag.edges[0]
    assert e.type is EdgeType.DELEGATION
    assert e.branch_from_segment_id is None
    assert e.branch_from_checkpoint_id is None

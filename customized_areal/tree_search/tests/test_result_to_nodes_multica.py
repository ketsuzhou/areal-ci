"""Tests for _result_to_nodes multica branch (SuperNode-preserving).

The multica path returns the assembled ExecutionDAG's SuperNodes without
flattening them to Nodes, so the multi-segment edge structure survives to
_finalize_episode. ``_result_to_nodes`` is a pure transformation (it does not
read ``self``), so the heavy workflow constructor is bypassed via ``__new__``.
"""

from __future__ import annotations

from customized_areal.tree_search.agents.execution_dag import (
    EdgeType,
    ExecutionDAG,
    SuperNode,
)
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)


def _workflow() -> TreeSearchGroupedRolloutWorkflow:
    # _result_to_nodes does not read self; bypass __init__.
    return TreeSearchGroupedRolloutWorkflow.__new__(TreeSearchGroupedRolloutWorkflow)


def _two_segment_dag() -> ExecutionDAG:
    sn1 = SuperNode(node_id="seg1", agent_id="r1", issue_id="i1", task_id="")
    sn2 = SuperNode(node_id="seg2", agent_id="r2", issue_id="i2", task_id="")
    edag = ExecutionDAG()
    edag.add_event(sn1)
    edag.add_event(sn2)
    edag.add_edge("seg1", "seg2", EdgeType.DELEGATION)
    return edag


def test_result_to_nodes_multica_preserves_supernodes_and_edges():
    edag = _two_segment_dag()
    result = {"assembled_dag": object(), "execution_dag": edag}

    nodes = _workflow()._result_to_nodes(result, query_id="q1", group_idx=0)

    # Returns SuperNodes (not flattened to Nodes), in topological order.
    assert nodes is not None
    assert [n.node_id for n in nodes] == ["seg1", "seg2"]
    assert all(isinstance(n, SuperNode) for n in nodes)

    # query_id/group_idx stamped onto each SuperNode via metadata.
    for sn in nodes:
        assert sn.metadata["query_id"] == "q1"
        assert sn.metadata["group_idx"] == 0

    # Edge structure preserved: each SuperNode is self-contained.
    assert nodes[0].outgoing_edges == (("seg2", EdgeType.DELEGATION),)
    assert nodes[0].incoming_edges == ()
    assert nodes[1].incoming_edges == (("seg1", EdgeType.DELEGATION),)
    assert nodes[1].outgoing_edges == ()


def test_result_to_nodes_multica_does_not_corrupt_dag_topology():
    edag = _two_segment_dag()
    _workflow()._result_to_nodes(
        {"assembled_dag": object(), "execution_dag": edag},
        query_id="q1",
        group_idx=0,
    )
    # Stamping metadata/edges onto the SuperNodes must not alter the DAG itself.
    assert [e.type for e in edag.edges] == [EdgeType.DELEGATION]
    assert len(edag.events) == 2


def test_result_to_nodes_non_multica_dict_not_captured_by_multica_branch():
    # A dict without "execution_dag" must NOT be routed to the multica branch;
    # it falls through to the single-agent path (here: unparseable -> None).
    nodes = _workflow()._result_to_nodes({"foo": "bar"}, query_id="q1", group_idx=0)
    assert nodes is None

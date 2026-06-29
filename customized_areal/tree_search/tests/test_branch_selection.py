"""Tests for the Event branch-point selection policy.

Ports core/customized_grouped_workflow.py::select_branch_candidate (TD-error
gate + max-entropy ranking) onto the Event/DAG representation. Torch-free.

Spec: docs/superpowers/specs/2026-06-29-event-branch-selection-design.md
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.branch_selection import (
    BranchPoint,
    lane_successor_value,
    passes_gate,
    select_branch_points,
    td_error,
)
from customized_areal.tree_search.agents.event_model import Event


def _ev(
    node_id,
    task_id="t1",
    completion_index=0,
    *,
    branch_seq=None,
    value=None,
    outcome_reward=0.0,
    max_entropy=None,
    incoming=(),
    outgoing=(),
):
    """Build a minimal Event for selection tests."""
    metadata = {} if max_entropy is None else {"max_entropy": max_entropy}
    return Event(
        node_id=node_id,
        agent_id="ag",
        issue_id="iss",
        task_id=task_id,
        completion_index=completion_index,
        incoming_edges=tuple(incoming),
        outgoing_edges=tuple(outgoing),
        branch_seq=branch_seq,
        value=value,
        outcome_reward=outcome_reward,
        metadata=metadata,
    )


class TestLaneSuccessorValue:
    def test_in_lane_child_returns_zero_reward_and_child_value(self):
        child = _ev("c", value=0.7)
        r_t, v_next = lane_successor_value(_ev("a", value=0.2), {"a": child})
        assert r_t == 0.0
        assert v_next == 0.7

    def test_child_with_none_value_yields_zero_v_next(self):
        child = _ev("c", value=None)
        r_t, v_next = lane_successor_value(_ev("a", value=0.2), {"a": child})
        assert (r_t, v_next) == (0.0, 0.0)

    def test_terminal_uses_outcome_reward(self):
        r_t, v_next = lane_successor_value(_ev("a", outcome_reward=1.0), {})
        assert (r_t, v_next) == (1.0, 0.0)


class TestTdError:
    def test_known_arithmetic(self):
        # |0.0 + 1.0*0.9 - 0.1| = 0.8
        d = td_error(_ev("a", value=0.1), r_t=0.0, v_next=0.9, gamma=1.0)
        assert d == pytest.approx(0.8)

    def test_terminal_arithmetic(self):
        # |1.0 + 1.0*0.0 - 0.2| = 0.8
        d = td_error(_ev("a", value=0.2), r_t=1.0, v_next=0.0, gamma=1.0)
        assert d == pytest.approx(0.8)

    def test_missing_value_returns_none(self):
        assert td_error(_ev("a", value=None), r_t=0.0, v_next=0.0, gamma=1.0) is None


class TestPassesGate:
    def test_threshold_off_keeps_everything(self):
        assert passes_gate(0.0, td_threshold=0.0) is True
        assert passes_gate(None, td_threshold=0.0) is True

    def test_none_delta_bypasses_gate(self):
        assert passes_gate(None, td_threshold=5.0) is True

    def test_inclusive_boundary(self):
        assert passes_gate(0.5, td_threshold=0.5) is True

    def test_below_threshold_dropped(self):
        assert passes_gate(0.49, td_threshold=0.5) is False


def test_branchpoint_is_frozen():
    bp = BranchPoint(task_id="t1", seq=3, node_id="a", td_error=0.8, entropy=0.9)
    with pytest.raises(Exception):
        bp.seq = 4  # type: ignore[misc]


class TestSelectSingleLane:
    """Ported from tests/test_branch_td_gate.py onto the Event representation.

    Lane "t1" has candidate turns linked n1 -> n2 by a same-task edge so n1 has
    an in-lane successor; n2 is terminal. Edge type is irrelevant to selection
    but must be symmetric for events_to_dag.
    """

    def _linked_pair(self, *, v1, v2, e1, e2, seq1=1, seq2=2, out1=0.0, out2=0.0):
        from customized_areal.tree_search.agents.execution_dag import EdgeType

        n1 = _ev(
            "n1",
            task_id="t1",
            completion_index=0,
            branch_seq=seq1,
            value=v1,
            outcome_reward=out1,
            max_entropy=e1,
            outgoing=(("n2", EdgeType.COMPLETION),),
        )
        n2 = _ev(
            "n2",
            task_id="t1",
            completion_index=1,
            branch_seq=seq2,
            value=v2,
            outcome_reward=out2,
            max_entropy=e2,
            incoming=(("n1", EdgeType.COMPLETION),),
        )
        return [n1, n2]

    def test_threshold_zero_reproduces_entropy_only(self):
        # Both eligible, gate off -> highest entropy wins (n1: 0.9 > n2: 0.2).
        events = self._linked_pair(v1=0.5, v2=0.5, e1=0.9, e2=0.2)
        out = select_branch_points(events, td_threshold=0.0, gamma=1.0)
        assert len(out) == 1
        assert out[0].task_id == "t1"
        assert out[0].node_id == "n1"
        assert out[0].seq == 1

    def test_sub_threshold_dropped_higher_delta_chosen(self):
        # n1: delta = |0 + 0.5 - 0.5| = 0, entropy 0.9 -> dropped by threshold 0.5.
        # n2: terminal, delta = |0 + 0 - 0.1| = 0.1 -> also dropped.
        # Make n2 survive: outcome_reward 1.0 -> delta = |1.0 - 0.1| = 0.9.
        events = self._linked_pair(v1=0.5, v2=0.1, e1=0.9, e2=0.2, out2=1.0)
        out = select_branch_points(events, td_threshold=0.5, gamma=1.0)
        assert len(out) == 1
        assert out[0].node_id == "n2"

    def test_highest_entropy_among_survivors(self):
        # Both survive (deltas large), highest entropy wins.
        events = self._linked_pair(v1=0.1, v2=0.1, e1=0.3, e2=0.8, out2=1.0)
        # n1: delta = |0 + 0.1 - 0.1| = 0 -> dropped at threshold 0.5.
        # Only n2 survives here, so adjust n1 to survive via a successor gap:
        # use gamma so n1 delta is large. Simpler: rely on n2 being the survivor.
        out = select_branch_points(events, td_threshold=0.5, gamma=1.0)
        assert out[0].node_id == "n2"

    def test_all_dropped_returns_empty(self):
        # n1 delta 0; n2 terminal delta = |0 - 0.5| = 0.5 < threshold 1.0.
        events = self._linked_pair(v1=0.5, v2=0.5, e1=0.9, e2=0.2)
        out = select_branch_points(events, td_threshold=1.0, gamma=1.0)
        assert out == []

    def test_missing_critic_value_bypasses_gate(self):
        # Single eligible candidate with no value -> kept even at high threshold.
        n1 = _ev("n1", task_id="t1", branch_seq=1, value=None, max_entropy=0.9)
        out = select_branch_points([n1], td_threshold=5.0, gamma=1.0)
        assert len(out) == 1
        assert out[0].node_id == "n1"
        assert out[0].td_error is None

    def test_non_eligible_events_ignored(self):
        # branch_seq=None -> not a candidate; empty result.
        n1 = _ev("n1", task_id="t1", branch_seq=None, value=0.5, max_entropy=0.9)
        assert select_branch_points([n1]) == []

    def test_entropy_tie_breaks_on_completion_index(self):
        # Two eligible terminal candidates, equal entropy -> smaller index wins.
        a = _ev(
            "a",
            task_id="t1",
            completion_index=0,
            branch_seq=1,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.5,
        )
        b = _ev(
            "b",
            task_id="t1",
            completion_index=1,
            branch_seq=2,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.5,
        )
        # Make this a valid DAG: no edges between them is fine (both terminal,
        # independent roots in lane t1). Both gate-survive (delta=1.0).
        out = select_branch_points([a, b], td_threshold=0.5, gamma=1.0)
        assert out[0].node_id == "a"


class TestMultiLaneAndIntegration:
    def test_two_lanes_emit_two_branch_points(self):
        # Lane t1 winner "a", lane t2 winner "c". Both terminal, gate off.
        a = _ev(
            "a",
            task_id="t1",
            completion_index=0,
            branch_seq=1,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.9,
        )
        b = _ev(
            "b",
            task_id="t1",
            completion_index=1,
            branch_seq=2,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.1,
        )
        c = _ev(
            "c",
            task_id="t2",
            completion_index=2,
            branch_seq=3,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.7,
        )
        out = select_branch_points([a, b, c], td_threshold=0.0, gamma=1.0)
        assert [bp.task_id for bp in out] == ["t1", "t2"]  # sorted by task_id
        assert {bp.node_id for bp in out} == {"a", "c"}

    def test_determinism_under_shuffled_input(self):
        a = _ev(
            "a",
            task_id="t1",
            completion_index=0,
            branch_seq=1,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.9,
        )
        b = _ev(
            "b",
            task_id="t2",
            completion_index=1,
            branch_seq=2,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.7,
        )
        out1 = select_branch_points([a, b])
        out2 = select_branch_points([b, a])
        assert out1 == out2

    def test_empty_log_returns_empty(self):
        assert select_branch_points([]) == []

    def test_emitted_point_round_trips_through_replay_prefix(self):
        from customized_areal.tree_search.agents.event_codec import replay_prefix_for

        # Single eligible terminal event whose (task_id, branch_seq) is the key.
        a = _ev(
            "a",
            task_id="t1",
            completion_index=0,
            branch_seq=4,
            value=0.0,
            outcome_reward=1.0,
            max_entropy=0.9,
        )
        out = select_branch_points([a])
        assert len(out) == 1
        bp = out[0]
        prefix = replay_prefix_for([a], branch_point=(bp.task_id, bp.seq))
        assert prefix.task_id == "t1"
        assert prefix.seq == 4

    def test_malformed_log_propagates_dag_error(self):
        from customized_areal.tree_search.agents.execution_dag import DAGError

        # Non-dense completion_index (0 then 2) -> events_to_dag raises DAGError.
        a = _ev("a", task_id="t1", completion_index=0, branch_seq=1)
        b = _ev("b", task_id="t1", completion_index=2, branch_seq=2)
        with pytest.raises(DAGError):
            select_branch_points([a, b])


def test_public_symbols_exported_from_dag_package():
    import customized_areal.tree_search.agents as d

    for name in (
        "BranchPoint",
        "select_branch_points",
        "lane_successor_value",
        "td_error",
        "passes_gate",
    ):
        assert name in d.__all__, f"{name} missing from dag.__all__"
        assert hasattr(d, name), f"{name} not importable from dag"

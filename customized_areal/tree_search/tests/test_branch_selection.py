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

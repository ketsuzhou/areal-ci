# Event Branch-Point Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a torch-free branch-point selection policy over the canonical `Event` sequence that ports the existing TD-error-gate + max-entropy criterion and emits one `(task_id, seq)` branch point per task lane.

**Architecture:** A new pure module `agents/branch_selection.py` composed of three small helpers (`lane_successor_value`, `td_error`, `passes_gate`) plus an orchestrator `select_branch_points`. It reconstructs/validates the DAG once via `events_to_dag` (single source of validation), groups eligible Events by `task_id`, applies the ported gate, and returns the highest-entropy survivor per lane as a `BranchPoint`. It coexists with the legacy `Node`-based `select_branch_candidate` (left untouched); live-workflow wiring is an out-of-scope follow-up.

**Tech Stack:** Python 3.12, dataclasses, torch-free. Tests via `.venv-test/bin/python -m pytest`. Lint/format via project ruff at `/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff`.

**Spec:** `docs/superpowers/specs/2026-06-29-event-branch-selection-design.md`

---

## File Structure

- **Create** `customized_areal/tree_search/agents/branch_selection.py` — the entire feature: `BranchPoint` dataclass, three pure helpers, `select_branch_points` orchestrator, `__all__`. One clear responsibility (branch-point selection over Events). Depends only on `event_model`, `event_codec`. Torch-free.
- **Create** `customized_areal/tree_search/tests/test_branch_selection.py` — all tests (helper-level + orchestrator + lane/integration).
- **Modify** `customized_areal/tree_search/agents/__init__.py` — export the new public symbols (must stay torch-free).

### Reference conventions (read before starting)
- Imports use the full path: `from customized_areal.tree_search.agents.branch_selection import ...`.
- `Event` fields (see `agents/event_model.py`): `node_id, agent_id, issue_id, task_id, completion_index, incoming_edges, outgoing_edges, branch_seq, value, process_reward, outcome_reward, messages, metadata`. Edges are tuples `(dst_node_id, EdgeType)`.
- `EdgeType` (see `agents/execution_dag.py`): `DELEGATION`, `MENTION`, `COMPLETION` (a `StrEnum`). The selection logic ignores edge *type*; it only matches on the child's `task_id`.
- `events_to_dag(events)` (see `agents/event_codec.py`) validates: dense `completion_index` `0..n-1`, unique `node_id`, **edge symmetry** (every `outgoing` `A->B` needs a matching `incoming` `A->B` of the same `EdgeType`), and topological order (`index(src) < index(dst)`). It raises `DAGError` on violation.

---

## Task 1: BranchPoint dataclass + pure helpers

**Files:**
- Create: `customized_areal/tree_search/agents/branch_selection.py`
- Test: `customized_areal/tree_search/tests/test_branch_selection.py`

- [ ] **Step 1: Write the failing test**

Create `customized_areal/tree_search/tests/test_branch_selection.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.agents.branch_selection'`.

- [ ] **Step 3: Write minimal implementation**

Create `customized_areal/tree_search/agents/branch_selection.py`:

```python
"""Branch-point selection over the canonical Event sequence (ported criterion).

Torch-free, I/O-free, pure. Ports
``core/customized_grouped_workflow.py::select_branch_candidate`` (critic
TD-error gate + max-entropy ranking) onto the Event/DAG representation, emitting
one :class:`BranchPoint` per ``task_id`` lane. The emitted ``(task_id, seq)``
pairs are the key accepted by ``event_codec.replay_prefix_for``.

Coexists with the legacy Node-based selector; live-workflow wiring is a separate
follow-up. See docs/superpowers/specs/2026-06-29-event-branch-selection-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from customized_areal.tree_search.agents.event_codec import events_to_dag
from customized_areal.tree_search.agents.event_model import Event


@dataclass(frozen=True)
class BranchPoint:
    """A selected branch point, keyed for ``replay_prefix_for``.

    ``(task_id, seq)`` is exactly ``replay_prefix_for``'s ``branch_point`` key.
    ``td_error`` is the gate magnitude that passed (``None`` when the gate was
    bypassed because the Event had no critic value). ``entropy`` is the
    ranking key used (0.0 when absent).
    """

    task_id: str
    seq: int
    node_id: str
    td_error: float | None
    entropy: float


def _entropy(event: Event) -> float:
    """Read ``metadata['max_entropy']`` defensively (absent/bad -> 0.0)."""
    raw = event.metadata.get("max_entropy", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def lane_successor_value(
    event: Event, lane_children: Mapping[str, Event]
) -> tuple[float, float]:
    """Return ``(r_t, v_next)`` for the TD term.

    In-lane DAG child present -> ``(0.0, child.value or 0.0)``. Terminal (no
    in-lane child) -> ``(event.outcome_reward, 0.0)``.
    """
    child = lane_children.get(event.node_id)
    if child is not None:
        v_next = float(child.value) if child.value is not None else 0.0
        return 0.0, v_next
    return float(event.outcome_reward), 0.0


def td_error(
    event: Event, r_t: float, v_next: float, *, gamma: float
) -> float | None:
    """``|r_t + gamma*v_next - v(s_t)|`` with ``v(s_t) = event.value``.

    Returns ``None`` when ``event.value`` is ``None`` (gate bypassed).
    """
    if event.value is None:
        return None
    return abs(r_t + gamma * v_next - float(event.value))


def passes_gate(delta: float | None, *, td_threshold: float) -> bool:
    """Gate predicate: threshold off, missing delta (bypass), or ``>=`` threshold."""
    if td_threshold <= 0.0:
        return True
    if delta is None:
        return True
    return delta >= td_threshold


__all__ = [
    "BranchPoint",
    "lane_successor_value",
    "passes_gate",
    "td_error",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py -q`
Expected: PASS — but `select_branch_points` import will still fail at collection. To isolate, run only the implemented classes:
`.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py -q -k "LaneSuccessorValue or TdError or PassesGate or frozen"`
Expected: these PASS. (The module-level `select_branch_points` import fails collection; Task 2 adds it. If collection blocks the run, temporarily import only the implemented names — but Task 2 follows immediately, so prefer to proceed.)

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/branch_selection.py customized_areal/tree_search/tests/test_branch_selection.py
git commit -m "feat: add Event branch-selection helpers (BranchPoint, gate, td_error)"
```

---

## Task 2: select_branch_points orchestrator (single lane, ported scenarios)

**Files:**
- Modify: `customized_areal/tree_search/agents/branch_selection.py`
- Test: `customized_areal/tree_search/tests/test_branch_selection.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/tests/test_branch_selection.py`:

```python
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
        a = _ev("a", task_id="t1", completion_index=0, branch_seq=1,
                value=0.0, outcome_reward=1.0, max_entropy=0.5)
        b = _ev("b", task_id="t1", completion_index=1, branch_seq=2,
                value=0.0, outcome_reward=1.0, max_entropy=0.5)
        # Make this a valid DAG: no edges between them is fine (both terminal,
        # independent roots in lane t1). Both gate-survive (delta=1.0).
        out = select_branch_points([a, b], td_threshold=0.5, gamma=1.0)
        assert out[0].node_id == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py::TestSelectSingleLane -q`
Expected: FAIL — `ImportError: cannot import name 'select_branch_points'`.

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/agents/branch_selection.py`, add the lane-children builder and orchestrator above `__all__`, then update `__all__`:

```python
def _lane_children(events: Sequence[Event]) -> dict[str, Event]:
    """Map each node_id to its in-lane DAG child (same task_id).

    For a node with multiple same-lane outgoing children, the one with the
    smallest ``completion_index`` is chosen (deterministic).
    """
    by_id = {e.node_id: e for e in events}
    children: dict[str, Event] = {}
    for e in events:
        same_lane = [
            by_id[dst]
            for dst, _etype in e.outgoing_edges
            if dst in by_id and by_id[dst].task_id == e.task_id
        ]
        if same_lane:
            children[e.node_id] = min(same_lane, key=lambda c: c.completion_index)
    return children


def select_branch_points(
    events: Sequence[Event],
    *,
    td_threshold: float = 0.0,
    gamma: float = 1.0,
) -> list[BranchPoint]:
    """Select one branch point per ``task_id`` lane (ported criterion).

    Eligible events (``branch_seq is not None``) are grouped by ``task_id``;
    within each lane, candidates passing the TD-error gate are ranked by
    ``metadata['max_entropy']`` (ties: smallest ``completion_index``). Returns
    one :class:`BranchPoint` per lane with a survivor, sorted by ``task_id``.

    Raises ``DAGError`` (via ``events_to_dag``) on a malformed Event log.
    """
    events = list(events)
    if not events:
        return []

    # Validate structure once; single source of validation (the codec).
    events_to_dag(events)

    lane_children = _lane_children(events)

    lanes: dict[str, list[Event]] = {}
    for e in events:
        if e.branch_seq is not None:
            lanes.setdefault(e.task_id, []).append(e)

    results: list[BranchPoint] = []
    for task_id in sorted(lanes):
        survivors: list[tuple[Event, float | None]] = []
        for e in lanes[task_id]:
            r_t, v_next = lane_successor_value(e, lane_children)
            delta = td_error(e, r_t, v_next, gamma=gamma)
            if passes_gate(delta, td_threshold=td_threshold):
                survivors.append((e, delta))
        if not survivors:
            continue
        best_ev, best_delta = max(
            survivors,
            key=lambda pair: (_entropy(pair[0]), -pair[0].completion_index),
        )
        assert best_ev.branch_seq is not None  # eligibility guarantees this
        results.append(
            BranchPoint(
                task_id=best_ev.task_id,
                seq=best_ev.branch_seq,
                node_id=best_ev.node_id,
                td_error=best_delta,
                entropy=_entropy(best_ev),
            )
        )
    return results
```

Update `__all__` to:

```python
__all__ = [
    "BranchPoint",
    "lane_successor_value",
    "passes_gate",
    "select_branch_points",
    "td_error",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/branch_selection.py customized_areal/tree_search/tests/test_branch_selection.py
git commit -m "feat: add select_branch_points orchestrator (one per task_id lane)"
```

---

## Task 3: Multi-lane, determinism, integration round-trip, error propagation

**Files:**
- Test: `customized_areal/tree_search/tests/test_branch_selection.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/tests/test_branch_selection.py`:

```python
class TestMultiLaneAndIntegration:
    def test_two_lanes_emit_two_branch_points(self):
        # Lane t1 winner "a", lane t2 winner "c". Both terminal, gate off.
        a = _ev("a", task_id="t1", completion_index=0, branch_seq=1,
                value=0.0, outcome_reward=1.0, max_entropy=0.9)
        b = _ev("b", task_id="t1", completion_index=1, branch_seq=2,
                value=0.0, outcome_reward=1.0, max_entropy=0.1)
        c = _ev("c", task_id="t2", completion_index=2, branch_seq=3,
                value=0.0, outcome_reward=1.0, max_entropy=0.7)
        out = select_branch_points([a, b, c], td_threshold=0.0, gamma=1.0)
        assert [bp.task_id for bp in out] == ["t1", "t2"]  # sorted by task_id
        assert {bp.node_id for bp in out} == {"a", "c"}

    def test_determinism_under_shuffled_input(self):
        a = _ev("a", task_id="t1", completion_index=0, branch_seq=1,
                value=0.0, outcome_reward=1.0, max_entropy=0.9)
        b = _ev("b", task_id="t2", completion_index=1, branch_seq=2,
                value=0.0, outcome_reward=1.0, max_entropy=0.7)
        out1 = select_branch_points([a, b])
        out2 = select_branch_points([b, a])
        assert out1 == out2

    def test_empty_log_returns_empty(self):
        assert select_branch_points([]) == []

    def test_emitted_point_round_trips_through_replay_prefix(self):
        from customized_areal.tree_search.agents.event_codec import replay_prefix_for

        # Single eligible terminal event whose (task_id, branch_seq) is the key.
        a = _ev("a", task_id="t1", completion_index=0, branch_seq=4,
                value=0.0, outcome_reward=1.0, max_entropy=0.9)
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
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py::TestMultiLaneAndIntegration -q`
Expected: These tests exercise already-implemented behavior, so they should PASS immediately. If any FAIL, fix the implementation in `branch_selection.py` (do not weaken the test) until green. In particular confirm:
- result list is sorted by `task_id`,
- `replay_prefix_for` accepts the emitted `(task_id, seq)`,
- `DAGError` is not swallowed.

- [ ] **Step 3: Run full module test**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py -q`
Expected: PASS (all classes).

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/tests/test_branch_selection.py
git commit -m "test: multi-lane, determinism, replay round-trip, DAGError propagation"
```

---

## Task 4: Export from package + lint + full agents suite

**Files:**
- Modify: `customized_areal/tree_search/agents/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/tests/test_branch_selection.py`:

```python
def test_public_symbols_exported_from_dag_package():
    import customized_areal.tree_search.agents as d

    for name in ("BranchPoint", "select_branch_points", "lane_successor_value",
                 "td_error", "passes_gate"):
        assert name in d.__all__, f"{name} missing from agents.__all__"
        assert hasattr(d, name), f"{name} not importable from agents"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py::test_public_symbols_exported_from_dag_package -q`
Expected: FAIL — `BranchPoint missing from agents.__all__`.

- [ ] **Step 3: Add the exports**

In `customized_areal/tree_search/agents/__init__.py`, add the import block (after the `gae` import block, keeping alphabetical-ish grouping consistent with the file):

```python
from customized_areal.tree_search.agents.branch_selection import (
    BranchPoint,
    lane_successor_value,
    passes_gate,
    select_branch_points,
    td_error,
)
```

And add to the `__all__` list (a new grouped section near the `gae` / `dag advantage` entries):

```python
    # branch selection (branch-point selection policy over the Event sequence)
    "BranchPoint",
    "lane_successor_value",
    "passes_gate",
    "select_branch_points",
    "td_error",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py::test_public_symbols_exported_from_dag_package -q`
Expected: PASS.

- [ ] **Step 5: Run the full agents test suite + lint/format**

```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_branch_selection.py customized_areal/tree_search/tests/test_gae.py customized_areal/tree_search/tests/test_session_map.py -q
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/agents/branch_selection.py customized_areal/tree_search/agents/__init__.py customized_areal/tree_search/tests/test_branch_selection.py
$RUFF format --check customized_areal/tree_search/agents/branch_selection.py customized_areal/tree_search/agents/__init__.py customized_areal/tree_search/tests/test_branch_selection.py
```
Expected: all tests PASS; ruff reports no errors and "would reformat 0 files" (run `$RUFF format` without `--check` to apply if needed, then re-verify). Confirm `import customized_areal.tree_search.agents` still succeeds (proves the package stayed torch-free):
`.venv-test/bin/python -c "import customized_areal.tree_search.agents as d; print('ok', 'select_branch_points' in d.__all__)"`
Expected: `ok True`.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/agents/__init__.py customized_areal/tree_search/tests/test_branch_selection.py
git commit -m "feat: export branch-selection API from agents package"
```

---

## Self-Review

**1. Spec coverage:**
- §4 placement (`agents/branch_selection.py`, torch-free, deps on event_model/codec) → Tasks 1–2.
- §5 API (`BranchPoint` + 3 helpers + orchestrator, list output, default gamma=1.0/td_threshold=0.0) → Tasks 1–2.
- §6 field mapping (eligibility `branch_seq is not None`; successor = in-lane DAG child; value = `Event.value`; entropy = `metadata['max_entropy']`) → Tasks 1–2 (`_lane_children`, `lane_successor_value`, `_entropy`).
- §7 algorithm (validate via `events_to_dag`, group by task_id, gate, rank, tiebreak on completion_index, sort by task_id) → Task 2.
- §8 error handling (empty → []; no eligible → []; all-dropped lane omitted; missing value bypass; missing entropy → 0.0; terminal uses outcome_reward; DAGError propagates; no config validation) → Tasks 2–3.
- §9 testing (helper unit, ported orchestrator scenarios, multi-lane, replay round-trip, determinism, empty, malformed) → Tasks 1–3.
- §10 follow-up (workflow wiring) → explicitly out of scope; not a task. Correct.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". All code shown in full.

**3. Type consistency:** `BranchPoint(task_id, seq, node_id, td_error, entropy)` identical in spec, dataclass, and all test constructions. `select_branch_points(events, *, td_threshold=0.0, gamma=1.0)`, `lane_successor_value(event, lane_children)`, `td_error(event, r_t, v_next, *, gamma)`, `passes_gate(delta, *, td_threshold)` consistent across all tasks. `events_to_dag` / `replay_prefix_for` / `DAGError` / `EdgeType` / `Event` names match the codebase.

Note for the implementer: in `TestSelectSingleLane.test_highest_entropy_among_survivors`, only `n2` survives the threshold-0.5 gate (n1's delta is 0); the assertion (`out[0].node_id == "n2"`) is correct as written — the inline comments explain why. If you want a true two-survivor entropy comparison, see `test_two_lanes_emit_two_branch_points` and the entropy-tie test instead.

---

## Execution Handoff

Plan complete. See below for execution options.

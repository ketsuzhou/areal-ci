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

from customized_areal.tree_search.agents.event_codec import (
    events_to_dag,
)
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


__all__ = [
    "BranchPoint",
    "lane_successor_value",
    "passes_gate",
    "select_branch_points",
    "td_error",
]

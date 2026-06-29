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

from collections.abc import Mapping, Sequence  # noqa: F401  # Task 2 uses Sequence
from dataclasses import dataclass

from customized_areal.tree_search.agents.event_codec import (
    events_to_dag,  # noqa: F401  # Task 2 uses events_to_dag
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


__all__ = [
    "BranchPoint",
    "lane_successor_value",
    "passes_gate",
    "td_error",
]

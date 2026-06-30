"""Canonical linear-event model for the agent-execution DAG (trajectory log).

This is the single source of truth for the *linear* (completion-ordered) view
of an :class:`ExecutionDAG`. One ``Event`` == one DAG node (a completed agent
turn). Forward/reverse conversion lives in ``event_codec``; this module holds
only the data definition + serialization + the derived message-timeline view.

Framework B: the linear order is the global completion order, which is always a
topological order of the causal DAG ("prefix = cut" -- see
``CRITIC_GAE_INTEGRATION.md``). The codec persists/reconstructs that order via
the explicit ``completion_index``.

Torch-free and I/O-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from customized_areal.tree_search.dag.execution_dag import DAGError, EdgeType

EdgeRef = tuple[str, EdgeType]


@dataclass(frozen=True)
class Event:
    """One completed turn in the global completion-ordered trajectory.

    ``completion_index`` is the authoritative linear position (0-based, dense).
    Edges are stored in BOTH directions for fidelity; ``event_codec`` validates
    their symmetry on decode. ``messages`` is the turn's transcript slice, with
    the turn's own output expected as the last element (see ``message_timeline``).
    """

    node_id: str
    agent_id: str
    issue_id: str
    task_id: str
    completion_index: int
    incoming_edges: tuple[EdgeRef, ...] = ()
    outgoing_edges: tuple[EdgeRef, ...] = ()
    session_id: str | None = None
    completion_time: float | None = None
    branch_seq: int | None = None
    branch_issue_id: str | None = None
    branch_env_snapshot_id: str | None = None
    value: float | None = None
    process_reward: float = 0.0
    outcome_reward: float = 0.0
    messages: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Emit a plain JSON-safe dict (EdgeType -> str, tuples -> lists)."""
        return {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "issue_id": self.issue_id,
            "task_id": self.task_id,
            "completion_index": self.completion_index,
            "incoming_edges": [[s, t.value] for s, t in self.incoming_edges],
            "outgoing_edges": [[d, t.value] for d, t in self.outgoing_edges],
            "session_id": self.session_id,
            "completion_time": self.completion_time,
            "branch_seq": self.branch_seq,
            "branch_issue_id": self.branch_issue_id,
            "branch_env_snapshot_id": self.branch_env_snapshot_id,
            "value": self.value,
            "process_reward": self.process_reward,
            "outcome_reward": self.outcome_reward,
            "messages": [dict(m) for m in self.messages],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        """Exact inverse of :meth:`to_dict`. Raises ``DAGError`` on bad input."""
        try:
            return cls(
                node_id=d["node_id"],
                agent_id=d["agent_id"],
                issue_id=d["issue_id"],
                task_id=d["task_id"],
                completion_index=d["completion_index"],
                incoming_edges=cls._coerce_edges(d.get("incoming_edges", ())),
                outgoing_edges=cls._coerce_edges(d.get("outgoing_edges", ())),
                session_id=d.get("session_id"),
                completion_time=d.get("completion_time"),
                branch_seq=d.get("branch_seq"),
                branch_issue_id=d.get("branch_issue_id"),
                branch_env_snapshot_id=d.get("branch_env_snapshot_id"),
                value=d.get("value"),
                process_reward=d.get("process_reward", 0.0),
                outcome_reward=d.get("outcome_reward", 0.0),
                messages=tuple(dict(m) for m in d.get("messages", ())),
                metadata=dict(d.get("metadata", {})),
            )
        except KeyError as exc:
            raise DAGError(f"Event.from_dict missing required field: {exc}") from exc

    @staticmethod
    def _coerce_edges(raw: Iterable) -> tuple[EdgeRef, ...]:
        out: list[EdgeRef] = []
        for item in raw:
            nid, etype = item[0], item[1]
            if not isinstance(etype, EdgeType):
                try:
                    etype = EdgeType(etype)
                except ValueError as exc:
                    raise DAGError(f"unknown EdgeType: {etype!r}") from exc
            out.append((nid, etype))
        return tuple(out)


def message_timeline(events: Sequence[Event]) -> list[dict]:
    """Derived message-level view across events in completion order.

    Concatenates each event's ``messages`` payload in ``completion_index``
    order. The LAST message of each event's payload is the turn's own output and
    is tagged with the event's ``node_id`` (so the critic frontier builder can
    detect turn outputs); earlier payload messages pass through untagged. Tagged
    copies are emitted; the source payloads are not mutated.
    """
    ordered = sorted(events, key=lambda e: e.completion_index)
    timeline: list[dict] = []
    for ev in ordered:
        msgs = list(ev.messages)
        for i, m in enumerate(msgs):
            tagged = dict(m)
            if i == len(msgs) - 1:
                tagged["node_id"] = ev.node_id
            timeline.append(tagged)
    return timeline


__all__ = ["EdgeRef", "Event", "message_timeline"]

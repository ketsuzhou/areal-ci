"""Edge ref type + message-timeline helper for the linear Event-log view.

After the SuperNode unification, this module no longer defines ``Event``;
the linear-log role is absorbed by :class:`SuperNode` (which carries
``completion_index`` and the ``nodes`` payload directly). What survives here:

- :data:`EdgeRef` -- the ``(node_id, EdgeType)`` alias used across the codec.
- :func:`message_timeline` -- the message-level view consumed by the critic
  observation builder.

Torch-free and I/O-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from customized_areal.tree_search.agents.execution_dag import EdgeType

EdgeRef = tuple[str, EdgeType]


class _HasCompletionIndex(Protocol):
    """Structural type for objects message_timeline can consume.

    SuperNode satisfies this: ``completion_index`` is its linear position and
    ``node_id`` identifies the segment. Messages come from the ``nodes`` field
    (each node's serialized messages) or a legacy ``messages`` tuple on the
    object itself.
    """

    node_id: str
    completion_index: int


def _extract_messages(obj: object) -> list[dict]:
    """Return the message payload for one SuperNode-like object.

    Prefers ``obj.messages`` (the legacy Event shape: a tuple of message dicts);
    falls back to flattening ``obj.nodes[i].messages`` (the SuperNode shape).
    Each node's own output is the LAST message of its payload.
    """
    messages = getattr(obj, "messages", None)
    if messages is not None:
        return [dict(m) for m in messages]
    # SuperNode path: each inner node carries its own message dict(s).
    out: list[dict] = []
    nodes = getattr(obj, "nodes", None) or ()
    for n in nodes:
        node_msgs = getattr(n, "messages", None) or ()
        for m in node_msgs:
            out.append(dict(m))
    return out


def message_timeline(events: Sequence) -> list[dict]:
    """Derived message-level view across SuperNodes in completion order.

    Concatenates each SuperNode's message payload in ``completion_index``
    order. The LAST message of each SuperNode's payload is the segment's
    own terminal output and is tagged with the SuperNode's ``node_id`` (so the
    critic frontier builder can detect turn outputs); earlier payload messages
    pass through untagged. Tagged copies are emitted; the source payloads are
    not mutated.
    """
    ordered = sorted(events, key=lambda e: e.completion_index)
    timeline: list[dict] = []
    for ev in ordered:
        msgs = _extract_messages(ev)
        for i, m in enumerate(msgs):
            tagged = dict(m)
            if i == len(msgs) - 1:
                tagged["node_id"] = ev.node_id
            timeline.append(tagged)
    return timeline


__all__ = ["EdgeRef", "message_timeline"]

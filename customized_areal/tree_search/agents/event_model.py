"""Edge ref type + message-timeline helper for the linear SuperNode log.

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

from customized_areal.tree_search.agents.execution_dag import EdgeType

EdgeRef = tuple[str, EdgeType]


def _extract_messages(obj: object) -> list[dict]:
    """Return the message payload for one SuperNode-like object.

    Flattens ``obj.nodes[i].messages`` -- the SuperNode shape where each inner
    node carries its own message dict(s). Each returned dict is a shallow copy
    so callers can tag entries without mutating the source payload.
    """
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

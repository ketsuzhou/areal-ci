"""Critic-agent observation builder -- global joint-state frontier (Phase 3, Task 5).

Framework B: the critic value ``V_t`` is the value of the **global joint state**
of all agents + the environment at a moment. That moment maps to a *frontier*
(cut) over the execution DAG: the set of all turns, across every agent lane,
that have completed up to a point in the global completion order.

The critic agent is triggered **when any sub-task agent finishes a turn**. The
global timeline is the completion order of turns; this builder takes that
ordered message list and produces the value-observation sequence.

Next-state indexing convention (correctness-critical -- keeps the value baseline
action-independent):

  - ``V_0``      = the global state BEFORE any turn output (entering state of
    turn 0); ``node_id = None``.
  - ``V_{j+1}``  = the frontier up to and INCLUDING the j-th turn output (the
    state right after turn j completed). This is what the critic computes when
    turn j finishes; it has seen turn j's action and is used only as the
    *next-state bootstrap*.
  - Turn j's own baseline ``V_j`` is the observation produced at turn (j-1)'s
    completion, which does NOT include turn j's output -> changing turn j's
    output never changes ``V_j``.

Each turn-output message carries a ``node_id`` linking the produced value (and,
downstream, the GAE advantage) back to the DAG node / actor token span.

Torch-free: no training-stack imports.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fields exposed to the critic. The critic sees only a filtered view of the
# state -- raw training internals (logprobs, token ids, tensors) never leak in.
DEFAULT_CRITIC_FIELDS: tuple[str, ...] = ("role", "content", "agent_id", "node_id")


@dataclass(frozen=True)
class CriticObservation:
    """One value-observation in the global trajectory.

    ``value_index`` is the ``t`` in ``V_t``. ``node_id`` is the DAG node whose
    turn completion produced this (next-)state, or ``None`` for the initial
    state ``V_0``. ``messages`` is the filtered global frontier; ``text`` is the
    rendered critic prompt body.
    """

    value_index: int
    node_id: str | None
    messages: tuple[dict, ...]
    text: str


def _filter_fields(msg: dict, fields: tuple[str, ...]) -> dict:
    return {k: msg[k] for k in fields if k in msg}


def _render(messages: tuple[dict, ...]) -> str:
    lines = []
    for m in messages:
        agent = m.get("agent_id", "-")
        node = m.get("node_id", "-")
        role = m.get("role", "")
        content = m.get("content", "")
        lines.append(f"[{agent}|{node}] {role}: {content}")
    return "\n".join(lines)


def _frontier(
    messages: list[dict], upto_exclusive: int, fields: tuple[str, ...]
) -> tuple[dict, ...]:
    return tuple(_filter_fields(m, fields) for m in messages[:upto_exclusive])


def build_critic_observations(
    messages: list[dict],
    *,
    fields: tuple[str, ...] = DEFAULT_CRITIC_FIELDS,
) -> list[CriticObservation]:
    """Build the global value-observation sequence ``[V_0, V_1, ..., V_T]``.

    ``messages`` is the global completion-ordered timeline. A message that
    carries a ``node_id`` is a *turn output* (a critic trigger / DAG node); other
    messages (seed prompt, user/tool/context) are part of the state but do not
    themselves produce a value.

    Returns ``len(turn_outputs) + 1`` observations. ``V_0`` is the frontier
    before the first turn output; ``V_{j+1}`` is the frontier including the
    j-th turn output (next-state convention).
    """
    output_positions = [i for i, m in enumerate(messages) if m.get("node_id")]

    observations: list[CriticObservation] = []

    # V_0: the global state entering turn 0 (everything before the first output).
    first = output_positions[0] if output_positions else len(messages)
    init_frontier = _frontier(messages, first, fields)
    observations.append(
        CriticObservation(
            value_index=0,
            node_id=None,
            messages=init_frontier,
            text=_render(init_frontier),
        )
    )

    # V_{j+1}: frontier including the j-th turn output.
    for j, p in enumerate(output_positions):
        frontier = _frontier(messages, p + 1, fields)
        observations.append(
            CriticObservation(
                value_index=j + 1,
                node_id=messages[p]["node_id"],
                messages=frontier,
                text=_render(frontier),
            )
        )
    return observations


def build_observation_after_turn(
    messages: list[dict],
    *,
    turn_output_index: int,
    value_index: int,
    fields: tuple[str, ...] = DEFAULT_CRITIC_FIELDS,
) -> CriticObservation:
    """Build a single observation online when one turn completes.

    ``turn_output_index`` is the position (in the global timeline) of the turn
    output that just completed; the frontier includes it. ``value_index`` is the
    next-state index ``j+1`` to assign (the caller tracks the global event
    count). Used by the online rollout path where the full sequence is not yet
    known.
    """
    frontier = _frontier(messages, turn_output_index + 1, fields)
    node_id = messages[turn_output_index].get("node_id")
    return CriticObservation(
        value_index=value_index,
        node_id=node_id,
        messages=frontier,
        text=_render(frontier),
    )


__all__ = [
    "DEFAULT_CRITIC_FIELDS",
    "CriticObservation",
    "build_critic_observations",
    "build_observation_after_turn",
]

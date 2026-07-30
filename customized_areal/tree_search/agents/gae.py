"""Global joint-state GAE over the DAG event sequence (Phase 3, Task 7).

Framework B: the multi-agent system is one shared policy acting through
whichever agent moves next. GAE therefore runs over the **global,
completion-ordered turn sequence** (a linearization of the execution DAG that
respects causality), not over per-task_id lanes. Each turn's advantage is
attached to the DAG node that produced it, for broadcast onto that turn's actor
tokens (Task 8).

Recurrence (next-state indexing -- see ``critic_observation``):

    delta_t = r_t + gamma * V_{t+1} - V_t
    A_t     = delta_t + gamma * lam * A_{t+1}     (backward, A_T = 0)
    R_t     = A_t + V_t

where ``V_t`` (the baseline for turn t) is the value the critic produced at the
PREVIOUS event (or ``initial_value`` for t=0) -- action-independent -- and
``V_{t+1}`` (= ``GlobalEvent.value``) is the next-state value produced when turn
t completed. The terminal bootstrap is 0 and the verifier terminal reward enters
as the last event's ``reward``.

Torch-free: plain floats so the ``dag`` package stays importable without the
training stack. The trainer maps these per-node results onto tensors (Task 8).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalEvent:
    """One turn completion in the global trajectory.

    ``value`` is ``V_{t+1}`` -- the next-state value the critic produced when
    this turn finished. ``reward`` is the step reward ``r_t`` (a per-node
    process signal; the terminal event also carries the verifier reward).
    """

    node_id: str
    value: float
    reward: float


@dataclass(frozen=True)
class NodeGAEResult:
    """GAE output for one turn/node."""

    node_id: str
    advantage: float
    return_: float
    baseline_value: float  # V_t actually used as the baseline


def compute_global_gae(
    events: list[GlobalEvent],
    *,
    initial_value: float,
    gamma: float,
    lam: float,
    terminal_bootstrap: float = 0.0,
) -> list[NodeGAEResult]:
    """Compute GAE advantages/returns over the global event sequence.

    Parameters
    ----------
    events:
        Turns in global completion order. ``events[t].value`` is ``V_{t+1}``.
    initial_value:
        ``V_0`` -- the value of the state entering turn 0.
    gamma, lam:
        Discount and GAE lambda.
    terminal_bootstrap:
        Value of the state after the final turn; 0.0 for a terminated episode.
    """
    n = len(events)
    if n == 0:
        return []

    # baseline[t] = V_t : V_0 = initial_value ; V_t = events[t-1].value (t>=1).
    baseline = [initial_value] + [events[t - 1].value for t in range(1, n)]

    advantages = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        v_t = baseline[t]
        # V_{t+1}: the next event's baseline (= events[t].value) except at the
        # terminal step, where the bootstrap is terminal_bootstrap (0).
        v_next = terminal_bootstrap if t == n - 1 else events[t].value
        delta = events[t].reward + gamma * v_next - v_t
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    return [
        NodeGAEResult(
            node_id=events[t].node_id,
            advantage=advantages[t],
            return_=advantages[t] + baseline[t],
            baseline_value=baseline[t],
        )
        for t in range(n)
    ]


def events_from_nodes(ordered_nodes: list) -> list[GlobalEvent]:
    """Build the global event sequence from DAG nodes in completion order.

    Duck-typed against training ``Node`` objects (with a compatibility fallback
    for older SuperNode-only callers). Diagnosis process rewards are attached to
    Nodes; SuperNodes are topology-only.
    """
    events: list[GlobalEvent] = []
    for node in ordered_nodes:
        events.append(
            GlobalEvent(
                node_id=node.node_id,
                value=float(getattr(node, "value", 0.0) or 0.0),
                reward=float(getattr(node, "process_reward", 0.0))
                + float(getattr(node, "outcome_reward", 0.0)),
            )
        )
    return events


__all__ = [
    "GlobalEvent",
    "NodeGAEResult",
    "compute_global_gae",
    "events_from_nodes",
]

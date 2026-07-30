"""Node-advantage broadcast + critic value loss target (Phase 3, Task 8).

Bridges the torch-free global GAE (``gae.py``) to the tensors the actor/critic
update consumes:

  - **Actor:** each DAG node = one actor turn, so the node-level advantage
    ``A_k`` is broadcast to every actor token of turn k (via the token->node
    map / loss mask). :func:`assign_token_advantages` is the pure-Python core;
    :func:`broadcast_node_advantages` returns the torch tensor.
  - **Critic:** the node return ``R_k`` is the value-regression target
    (:func:`value_targets_from_gae`), and :func:`critic_huber_loss` is the Huber
    between the differentiable expected-score value (``critic_score``) and the
    target returns.

torch is imported lazily; the pure-Python helpers are usable (and tested)
without the training stack. Not re-exported from ``dag/__init__``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from customized_areal.tree_search.agents.gae import NodeGAEResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def assign_token_advantages(
    token_node_ids: Sequence[str | None],
    node_advantages: dict[str, float],
    *,
    default: float = 0.0,
) -> list[float]:
    """Broadcast each node's advantage to all of its turn's tokens.

    ``token_node_ids[i]`` is the DAG node that produced token ``i`` (or ``None``
    for non-action/context tokens). Tokens whose node has no advantage (unknown
    node or ``None``) get ``default``.
    """
    return [
        node_advantages.get(nid, default) if nid is not None else default
        for nid in token_node_ids
    ]


def value_targets_from_gae(results: list[NodeGAEResult]) -> dict[str, float]:
    """``{node_id: return_}`` -- the per-node critic value-regression targets."""
    return {r.node_id: r.return_ for r in results}


def huber_loss_py(pred: float, target: float, *, delta: float = 1.0) -> float:
    """Pure-Python Huber loss (reference / torch-free use)."""
    err = abs(pred - target)
    if err <= delta:
        return 0.5 * err * err
    return delta * (err - 0.5 * delta)


def broadcast_node_advantages(
    token_node_ids: Sequence[str | None],
    node_advantages: dict[str, float],
    *,
    default: float = 0.0,
    device: object = None,
    dtype: object = None,
) -> torch.Tensor:
    """Torch version of :func:`assign_token_advantages` -> a 1-D tensor."""
    import torch

    values = assign_token_advantages(token_node_ids, node_advantages, default=default)
    return torch.tensor(
        values,
        dtype=dtype or torch.float32,
        device=device,
    )


def critic_huber_loss(
    expected_values: torch.Tensor,
    target_returns: torch.Tensor,
    *,
    delta: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked-mean Huber loss between expected-score values and GAE returns.

    ``expected_values`` are the differentiable per-node values from
    :func:`critic_score.expected_score_value`; ``target_returns`` are the GAE
    node returns. Gradients flow into ``expected_values`` (and thus the shared
    trunk).
    """
    import torch

    loss = torch.nn.functional.huber_loss(
        expected_values.float(),
        target_returns.float(),
        delta=delta,
        reduction="none",
    )
    if mask is not None:
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        return (loss * m).sum() / denom
    return loss.mean()


def combined_actor_critic_loss(
    actor_loss: torch.Tensor,
    critic_value_loss: torch.Tensor,
    *,
    critic_loss_weight: float,
) -> torch.Tensor:
    """Combined co-trained objective: ``actor_PG + w * critic_value_loss``.

    Both terms backprop into the shared trunk. ``critic_loss_weight`` should be
    small initially to limit gradient interference (see ``Config``).
    """
    return actor_loss + critic_loss_weight * critic_value_loss


__all__ = [
    "assign_token_advantages",
    "broadcast_node_advantages",
    "combined_actor_critic_loss",
    "critic_huber_loss",
    "huber_loss_py",
    "value_targets_from_gae",
]

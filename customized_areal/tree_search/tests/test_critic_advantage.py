"""Tests for node-advantage broadcast + critic value targets (Phase 3, Task 8).

Each DAG node = one actor turn. The node-level GAE advantage A_k is broadcast to
every actor token of turn k (via the loss mask / token->node map); the node
return R_k is the critic value-regression target, and the critic value loss is a
Huber between the differentiable expected-score value and R_k.

Pure-Python core is always tested; the torch path is behind ``importorskip``.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.critic_advantage import (
    assign_token_advantages,
    huber_loss_py,
    value_targets_from_gae,
)
from customized_areal.tree_search.agents.gae import NodeGAEResult


def test_assign_token_advantages_broadcasts_per_turn() -> None:
    token_node_ids = ["n0", "n0", "n0", "n1", "n1", None]
    node_adv = {"n0": 1.0, "n1": -0.5}
    out = assign_token_advantages(token_node_ids, node_adv)
    # All of turn n0's tokens carry 1.0, n1's carry -0.5, context token -> 0.0.
    assert out == [1.0, 1.0, 1.0, -0.5, -0.5, 0.0]


def test_assign_token_advantages_unknown_node_is_zero() -> None:
    out = assign_token_advantages(["n0", "ghost"], {"n0": 0.7})
    assert out == [0.7, 0.0]


def test_value_targets_from_gae_returns_node_returns() -> None:
    results = [
        NodeGAEResult(node_id="n0", advantage=1.0, return_=1.0, baseline_value=0.0),
        NodeGAEResult(node_id="n1", advantage=0.3, return_=1.0, baseline_value=0.7),
    ]
    assert value_targets_from_gae(results) == {"n0": 1.0, "n1": 1.0}


def test_huber_loss_py_quadratic_and_linear_regions() -> None:
    # |err| <= delta -> 0.5*err^2 ; else delta*(|err|-0.5*delta)
    assert huber_loss_py(0.0, 0.5, delta=1.0) == pytest.approx(0.125)  # 0.5*0.25
    assert huber_loss_py(0.0, 2.0, delta=1.0) == pytest.approx(1.5)  # 1*(2-0.5)


def test_broadcast_node_advantages_torch_matches_python() -> None:
    torch = pytest.importorskip("torch")
    from customized_areal.tree_search.agents.critic_advantage import (
        broadcast_node_advantages,
    )

    token_node_ids = ["n0", "n0", "n1", None]
    node_adv = {"n0": 1.0, "n1": -0.5}
    out = broadcast_node_advantages(token_node_ids, node_adv)
    expected = torch.tensor([1.0, 1.0, -0.5, 0.0])
    assert torch.allclose(out, expected)


def test_critic_huber_loss_torch_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from customized_areal.tree_search.agents.critic_advantage import critic_huber_loss

    values = torch.tensor([0.2, 0.9], requires_grad=True)
    targets = torch.tensor([1.0, 1.0])
    loss = critic_huber_loss(values, targets, delta=1.0)
    loss.backward()
    assert loss.item() > 0
    assert values.grad is not None and torch.any(values.grad != 0)

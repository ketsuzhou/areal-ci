"""Trainer integration smoke test for co-trained critic + GAE (Phase 3, Task 9).

The full combined-update step (PPO actor PG + critic value loss on the shared
trunk, with TreeAdvantageComputer bypassed) requires the training stack
(torch + FSDP/Megatron + GPU), which is not available here. That live step is
skipped with an explanation; the combinable loss helper it relies on is tested
behind ``importorskip``.

See ``CRITIC_GAE_INTEGRATION.md`` for the wiring points.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_combined_actor_critic_loss_weights_and_backprops() -> None:
    torch = pytest.importorskip("torch")
    from customized_areal.tree_search.dag.critic_advantage import (
        combined_actor_critic_loss,
    )

    actor = torch.tensor(2.0, requires_grad=True)
    critic = torch.tensor(4.0, requires_grad=True)
    total = combined_actor_critic_loss(actor, critic, critic_loss_weight=0.5)
    assert float(total.item()) == pytest.approx(2.0 + 0.5 * 4.0)
    total.backward()
    # d/d actor = 1 ; d/d critic = critic_loss_weight = 0.5
    assert float(actor.grad.item()) == pytest.approx(1.0)
    assert float(critic.grad.item()) == pytest.approx(0.5)


@pytest.mark.skipif(
    not _HAS_TORCH,
    reason="DAG critic+GAE combined train step requires torch + FSDP + GPU "
    "(training stack not available in this environment); see "
    "CRITIC_GAE_INTEGRATION.md for the wiring and run on a GPU node (Task 10).",
)
def test_dag_critic_gae_combined_train_step_smoke() -> None:  # pragma: no cover
    # Live smoke: rollout over a 2-run DAG -> assemble_node_advantages ->
    # broadcast to actor tokens -> critic_huber_loss -> combined update, with
    # GRPO/TreeAdvantageComputer bypassed. Requires a GPU training stack.
    pytest.skip("requires GPU training stack; validated in Task 10 e2e")

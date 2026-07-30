# SPDX-License-Identifier: Apache-2.0
"""Tests for the generative-critic soft-regression loss and batch builder."""

import torch

from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.training.losses.critic import (
    build_critic_training_batch,
    combined_actor_critic_loss,
    critic_softreg_loss_fn,
    expected_value_from_label_logits,
)

# Distinct leading token ids per label to avoid the documented 1/10 collision.
LEADING = list(range(11))  # labels 0..10 -> token ids 0..10
SCORE_MAX = 10


def _logits_one_hot(label, vocab=16, batch=1, big=20.0):
    logits = torch.zeros(batch, vocab)
    logits[:, label] = big
    return logits


class TestExpectedValueFromLogits:
    def test_all_mass_on_max(self):
        v = expected_value_from_label_logits(_logits_one_hot(10), LEADING, SCORE_MAX)
        assert torch.allclose(v, torch.tensor([1.0]), atol=1e-4)

    def test_all_mass_on_zero(self):
        v = expected_value_from_label_logits(_logits_one_hot(0), LEADING, SCORE_MAX)
        assert torch.allclose(v, torch.tensor([0.0]), atol=1e-4)

    def test_bad_shape(self):
        try:
            expected_value_from_label_logits(torch.zeros(3), LEADING, SCORE_MAX)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_bad_label_count(self):
        try:
            expected_value_from_label_logits(_logits_one_hot(5), [0, 1, 2], SCORE_MAX)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestCriticLoss:
    def test_zero_when_mass_on_target(self):
        # target = 1.0 -> all mass on label 10
        logits = _logits_one_hot(10)
        targets = torch.tensor([1.0])
        loss, stats = critic_softreg_loss_fn(logits, targets, LEADING, SCORE_MAX)
        assert loss.item() < 1e-6
        assert torch.allclose(stats["critic_value"], torch.tensor([1.0]), atol=1e-4)

    def test_gradient_reduces_loss(self):
        # Start from uniform logits (value ~0.5), target 1.0. A few SGD steps
        # should reduce the loss.
        logits = torch.zeros(1, 16, requires_grad=True)
        targets = torch.tensor([1.0])
        loss0, _ = critic_softreg_loss_fn(logits, targets, LEADING, SCORE_MAX)
        params = logits
        for _ in range(20):
            loss, _ = critic_softreg_loss_fn(params, targets, LEADING, SCORE_MAX)
            (grad,) = torch.autograd.grad(loss, params)
            params = (params - 5.0 * grad).detach().requires_grad_(True)
        new_loss, _ = critic_softreg_loss_fn(params, targets, LEADING, SCORE_MAX)
        assert new_loss.item() < loss0.item()

    def test_batch_mse(self):
        logits = torch.cat([_logits_one_hot(10), _logits_one_hot(0)], dim=0)
        targets = torch.tensor([1.0, 1.0])
        loss, stats = critic_softreg_loss_fn(logits, targets, LEADING, SCORE_MAX)
        # values ~ [1.0, 0.0]; mse = (0 + 1)/2 = 0.5
        assert abs(loss.item() - 0.5) < 1e-3


class TestCombinedLoss:
    def test_weighted_sum(self):
        a = torch.tensor(2.0)
        c = torch.tensor(4.0)
        out = combined_actor_critic_loss(a, c, 0.5)
        assert out.item() == 4.0


class _FakeClient:
    score_max = 10

    def __init__(self):
        self._digit_ids = {i: [i] for i in range(11)}

    def build_prompt_ids(self, node):
        # length proportional to turn so answer_pos differs
        return [1] * (node.turn_idx + 2)


def _node(node_id, turn_idx, reward):
    return Node(
        input_ids=[0, 0],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[0, 0],
        node_id=node_id,
        turn_idx=turn_idx,
        outcome_reward=reward,
    )


class TestBatchBuilder:
    def test_targets_from_store_q_value_clamped(self):
        store = MCTSTreeStore()
        nodes = [_node("a", 1, 0.0), _node("b", 2, 0.0)]
        # inject q_values via the store's backup mechanism
        store._q_values["a"] = 0.4
        store._q_values["b"] = 3.0  # will be clamped after /scale
        client = _FakeClient()
        batch = build_critic_training_batch(
            client, nodes, tree_store=store, target_scale=2.0, score_max=10
        )
        # a: 0.4/2 = 0.2 ; b: 3.0/2 = 1.5 -> clamp 1.0
        assert torch.allclose(batch["targets"], torch.tensor([0.2, 1.0]))
        # answer_pos = len(prompt)-1 = (turn+2)-1
        assert batch["answer_pos"].tolist() == [2, 3]
        assert batch["leading_token_ids"] == list(range(11))

    def test_explicit_targets(self):
        nodes = [_node("a", 0, 0.0)]
        client = _FakeClient()
        batch = build_critic_training_batch(
            client, nodes, targets=[0.5], target_scale=1.0
        )
        assert torch.allclose(batch["targets"], torch.tensor([0.5]))
        assert batch["prompt_ids"] == [[1, 1]]

    def test_store_default_qvalue_zero(self):
        nodes = [_node("a", 0, 0.7)]
        client = _FakeClient()
        store = MCTSTreeStore()  # no q_value for "a" -> get_q_value returns 0.0
        batch = build_critic_training_batch(
            client, nodes, tree_store=store, target_scale=1.0
        )
        assert torch.allclose(batch["targets"], torch.tensor([0.0]))

    def test_fallback_to_outcome_reward_without_store(self):
        # No tree_store and no explicit targets -> fall back to outcome_reward.
        nodes = [_node("a", 0, 0.7)]
        client = _FakeClient()
        batch = build_critic_training_batch(client, nodes, target_scale=1.0)
        assert torch.allclose(batch["targets"], torch.tensor([0.7]))

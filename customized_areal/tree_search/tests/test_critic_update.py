# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared-model critic regression step and combined-loss patch."""

import torch

from customized_areal.tree_search.training.critic_update import (
    build_critic_minibatch,
    critic_multicandidate_loss_fn,
    run_critic_regression_step,
)


def _critic_train_data():
    return {
        "prompt_ids": [[1, 2, 3], [4, 5]],
        "answer_pos": torch.tensor([2, 1], dtype=torch.long),
        "targets": torch.tensor([0.8, 0.2], dtype=torch.float32),
        "leading_token_ids": list(range(11)),  # labels 0..10
    }


class TestCriticMultiCandidateLoss:
    def test_zero_when_value_matches_target(self):
        # candidate logprobs all mass on label 8 -> value 0.8; target 0.8 -> ~0
        logprobs = torch.full((1, 11), -1e9)
        logprobs[0, 8] = 0.0
        input_data = {
            "critic_targets": torch.tensor([0.8]),
            "critic_score_max": 10,
        }
        loss = critic_multicandidate_loss_fn(logprobs, None, input_data)
        assert loss.item() < 1e-5

    def test_positive_when_mismatch(self):
        logprobs = torch.full((2, 11), -1e9)
        logprobs[0, 0] = 0.0  # value 0
        logprobs[1, 10] = 0.0  # value 1
        input_data = {
            "critic_targets": torch.tensor([1.0, 0.0]),
            "critic_score_max": 10,
        }
        loss = critic_multicandidate_loss_fn(logprobs, None, input_data)
        assert loss.item() > 0.9  # both maximally wrong -> mse ~1.0


class TestBuildCriticMinibatch:
    def test_shapes_and_masks(self):
        mb = build_critic_minibatch(_critic_train_data(), pad_token_id=0)
        assert mb["input_ids"].shape == (2, 3)
        assert mb["attention_mask"].tolist() == [
            [True, True, True],
            [True, True, False],
        ]
        # loss_mask 1 only at answer positions (2 and 1)
        assert mb["loss_mask"].tolist() == [[0, 0, 1], [0, 1, 0]]
        # topk_ids: [B, 1, n_labels]
        assert mb["topk_ids"].shape == (2, 1, 11)
        assert mb["topk_ids"][0, 0].tolist() == list(range(11))
        assert int(mb["critic_score_max"]) == 10
        assert torch.allclose(mb["critic_targets"], torch.tensor([0.8, 0.2]))


class _FakeEngine:
    def __init__(self):
        self.train_batch_calls = 0
        self.last_loss = None

    def train_batch(self, mb, loss_fn, loss_weight_fn, **kwargs):
        self.train_batch_calls += 1
        # Simulate the multi-candidate engine: gather candidate logprobs at the
        # single answer position per sample and invoke the loss fn.
        b = mb["input_ids"].shape[0]
        logprobs = torch.full((b, 11), -1e9)
        for i in range(b):
            logprobs[i, 5] = 0.0
        self.last_loss = loss_fn(logprobs, None, mb)
        return self.last_loss


class _FakeActor:
    def __init__(self):
        self.engine = _FakeEngine()


class TestRunCriticRegressionStep:
    def test_runs_when_data_present(self):
        actor = _FakeActor()
        ran = run_critic_regression_step(
            actor, _critic_train_data(), critic_loss_weight=0.5
        )
        assert ran is True
        assert actor.engine.train_batch_calls == 1
        # value 0.5 for both; targets 0.8, 0.2 -> mse = (0.09+0.09)/2 = 0.09;
        # weighted by 0.5 -> 0.045
        assert abs(actor.engine.last_loss.item() - 0.045) < 1e-4

    def test_skips_when_no_data(self):
        actor = _FakeActor()
        ran = run_critic_regression_step(actor, None, critic_loss_weight=0.5)
        assert ran is False
        assert actor.engine.train_batch_calls == 0

    def test_engine_error_is_guarded(self):
        class _BadEngine:
            def train_batch(self, *a, **k):
                raise RuntimeError("boom")

        class _BadActor:
            engine = _BadEngine()

        ran = run_critic_regression_step(
            _BadActor(), _critic_train_data(), critic_loss_weight=1.0
        )
        assert ran is False


class TestCombinedLossPatch:
    def test_patch_and_unpatch(self):
        from customized_areal.tree_search.training.actor import (
            patch_ppo_actor_class_to_use_combined_critic_loss,
            unpatch_combined_critic_loss,
        )

        from areal.trainer.ppo.actor import PPOActor

        original = PPOActor._ppo_update
        patch_ppo_actor_class_to_use_combined_critic_loss(0.5)
        assert PPOActor._ppo_update is not original
        # idempotent
        patch_ppo_actor_class_to_use_combined_critic_loss(0.5)
        unpatch_combined_critic_loss()
        assert PPOActor._ppo_update is original

    def test_patched_update_runs_critic_then_inner(self):
        from customized_areal.tree_search.training.actor import (
            patch_ppo_actor_class_to_use_combined_critic_loss,
            unpatch_combined_critic_loss,
        )

        from areal.trainer.ppo.actor import PPOActor

        calls = {"inner": 0, "critic": 0}
        original = PPOActor._ppo_update

        def fake_inner(self, data):
            calls["inner"] += 1
            # critic_train_data must have been popped before inner runs
            assert "critic_train_data" not in data

        PPOActor._ppo_update = fake_inner
        try:
            patch_ppo_actor_class_to_use_combined_critic_loss(0.5)

            actor = _FakeActor()
            data = {"x": 1, "critic_train_data": _critic_train_data()}
            PPOActor._ppo_update(actor, data)
            assert calls["inner"] == 1
            assert actor.engine.train_batch_calls == 1
        finally:
            unpatch_combined_critic_loss()
            PPOActor._ppo_update = original

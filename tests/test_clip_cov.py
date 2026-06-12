"""Tests for clip_cov PPO loss module."""

import pytest
import torch

from customized_areal.clip_cov.config import ClipCovConfig
from customized_areal.clip_cov.loss import (
    clip_cov_grpo_loss_fn,
    clip_cov_ppo_actor_loss_fn,
)


class TestClipCovPpoActorLossFn:
    """Tests for clip_cov_ppo_actor_loss_fn."""

    @pytest.fixture
    def basic_data(self):
        batch_size, seq_len = 4, 8
        torch.manual_seed(42)
        return {
            "logprobs": torch.randn(batch_size, seq_len),
            "proximal_logprobs": torch.randn(batch_size, seq_len),
            "old_logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "eps_clip": 0.2,
        }

    def test_returns_scalar_loss_and_stat_dict(self, basic_data):
        loss, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=basic_data["logprobs"],
            proximal_logprobs=basic_data["proximal_logprobs"],
            old_logprobs=basic_data["old_logprobs"],
            advantages=basic_data["advantages"],
            eps_clip=basic_data["eps_clip"],
            loss_mask=basic_data["loss_mask"],
        )
        assert loss.ndim == 0
        assert isinstance(stat, dict)
        assert "loss" in stat
        assert "importance_weight" in stat
        assert "approx_kl" in stat
        assert "clip_mask" in stat
        assert "clip_cov_mask" in stat

    def test_stat_shapes_match_input(self, basic_data):
        _, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=basic_data["logprobs"],
            proximal_logprobs=basic_data["proximal_logprobs"],
            old_logprobs=basic_data["old_logprobs"],
            advantages=basic_data["advantages"],
            eps_clip=basic_data["eps_clip"],
            loss_mask=basic_data["loss_mask"],
        )
        shape = basic_data["logprobs"].shape
        assert stat["loss"].shape == shape
        assert stat["importance_weight"].shape == shape
        assert stat["approx_kl"].shape == shape
        assert stat["clip_mask"].shape == shape
        assert stat["clip_cov_mask"].shape == shape

    def test_loss_is_finite(self, basic_data):
        loss, _ = clip_cov_ppo_actor_loss_fn(
            logprobs=basic_data["logprobs"],
            proximal_logprobs=basic_data["proximal_logprobs"],
            old_logprobs=basic_data["old_logprobs"],
            advantages=basic_data["advantages"],
            eps_clip=basic_data["eps_clip"],
            loss_mask=basic_data["loss_mask"],
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_all_masked_gives_zero_loss(self):
        logprobs = torch.randn(2, 4)
        proximal_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 4)
        advantages = torch.randn(2, 4)
        loss_mask = torch.zeros(2, 4, dtype=torch.bool)

        loss, _ = clip_cov_ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
        )
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_no_clip_cov_equals_standard_ppo_loss(self):
        """When no tokens fall within the cov range, clip_cov loss equals standard PPO loss."""
        torch.manual_seed(0)
        logprobs = torch.randn(2, 4)
        proximal_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 4)
        advantages = torch.randn(2, 4)
        loss_mask = torch.ones(2, 4, dtype=torch.bool)
        eps_clip = 0.2

        from areal.utils.functional import ppo_actor_loss_fn

        ppo_loss, _ = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=eps_clip,
            loss_mask=loss_mask,
        )

        # Use bounds that exclude all covariance values (very tight range)
        cov_loss, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=eps_clip,
            loss_mask=loss_mask,
            clip_cov_lb=1e10,
            clip_cov_ub=1e11,
        )
        # No tokens selected for cov clipping when bounds exclude all
        assert stat["clip_cov_mask"].sum() == 0
        assert torch.allclose(cov_loss, ppo_loss, atol=1e-5)

    def test_clip_cov_mask_is_boolean(self, basic_data):
        _, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=basic_data["logprobs"],
            proximal_logprobs=basic_data["proximal_logprobs"],
            old_logprobs=basic_data["old_logprobs"],
            advantages=basic_data["advantages"],
            eps_clip=basic_data["eps_clip"],
            loss_mask=basic_data["loss_mask"],
        )
        assert stat["clip_cov_mask"].dtype == torch.bool

    def test_asymmetric_clipping(self):
        torch.manual_seed(1)
        logprobs = torch.randn(2, 4)
        proximal_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 4)
        advantages = torch.randn(2, 4)
        loss_mask = torch.ones(2, 4, dtype=torch.bool)

        loss, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            eps_clip_higher=0.4,
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_dual_clip(self):
        torch.manual_seed(2)
        logprobs = torch.randn(2, 4)
        proximal_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 4)
        advantages = torch.randn(2, 4)
        loss_mask = torch.ones(2, 4, dtype=torch.bool)

        loss, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            c_clip=3.0,
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert "dual_clip_mask" in stat


class TestClipCovGrpoLossFn:
    """Tests for clip_cov_grpo_loss_fn wrapper."""

    @pytest.fixture
    def input_data(self):
        batch_size, seq_len = 2, 4
        torch.manual_seed(42)
        return {
            "logprobs": torch.randn(batch_size, seq_len),
            "prox_logp": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }

    def test_returns_scalar_loss_and_dict(self, input_data):
        logprobs = torch.randn(2, 4)
        entropy = torch.randn(2, 4)
        loss, stat = clip_cov_grpo_loss_fn(
            logprobs=logprobs,
            entropy=entropy,
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )
        assert loss.ndim == 0
        assert isinstance(stat, dict)

    def test_loss_is_finite(self, input_data):
        logprobs = torch.randn(2, 4)
        entropy = torch.randn(2, 4)
        loss, _ = clip_cov_grpo_loss_fn(
            logprobs=logprobs,
            entropy=entropy,
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestClipCovPatch:
    """Tests for PPOActor monkey-patching."""

    def test_patch_is_idempotent(self):
        from customized_areal.clip_cov.patch import (
            patch_ppo_actor_to_use_clip_cov_loss,
            unpatch_ppo_actor_clip_cov_loss,
        )

        config = ClipCovConfig()
        try:
            patch_ppo_actor_to_use_clip_cov_loss(config)
            # Calling again should not raise
            patch_ppo_actor_to_use_clip_cov_loss(config)
        finally:
            unpatch_ppo_actor_clip_cov_loss()

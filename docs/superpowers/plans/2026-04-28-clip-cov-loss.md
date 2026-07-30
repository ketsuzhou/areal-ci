# Clip-Cov PPO Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement covariance-aware PPO clipping loss (`clip_cov`) as a standalone
monkey-patch module in `customized_areal/clip_cov/`.

**Architecture:** A new `customized_areal/clip_cov/` module containing a config
dataclass, the core loss function adapted from the PRIME-RL reference, a grpo-compatible
wrapper, and a patch function that replaces `PPOActor._ppo_update`. No changes to
`areal/`.

**Tech Stack:** Python 3.12+ | PyTorch | AReaL (monkey-patching `PPOActor`)

______________________________________________________________________

## File Structure

| File                                    | Responsibility                                                              |
| --------------------------------------- | --------------------------------------------------------------------------- |
| `customized_areal/clip_cov/__init__.py` | Public API: exports `ClipCovConfig`, `patch_ppo_actor_to_use_clip_cov_loss` |
| `customized_areal/clip_cov/config.py`   | `ClipCovConfig` dataclass with clip_cov hyperparameters                     |
| `customized_areal/clip_cov/loss.py`     | `clip_cov_ppo_actor_loss_fn` (core) + `clip_cov_grpo_loss_fn` (wrapper)     |
| `customized_areal/clip_cov/patch.py`    | `patch_ppo_actor_to_use_clip_cov_loss()` — class-level monkey-patch         |
| `tests/test_clip_cov.py`                | Unit tests for loss function and patch                                      |

______________________________________________________________________

### Task 1: Create ClipCovConfig

**Files:**

- Create: `customized_areal/clip_cov/config.py`

- [ ] **Step 1: Write config.py**

```python
"""Clip-cov PPO loss configuration."""

from dataclasses import dataclass


@dataclass
class ClipCovConfig:
    """Configuration for covariance-aware PPO clipping.

    Attributes:
        clip_ratio: Fraction of valid tokens to zero via covariance clipping.
        clip_cov_lb: Lower bound of covariance range for candidate selection.
        clip_cov_ub: Upper bound of covariance range for candidate selection.
    """

    clip_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/clip_cov/config.py
git commit -m "feat(clip-cov): add ClipCovConfig dataclass"
```

______________________________________________________________________

### Task 2: Write failing tests for clip_cov_ppo_actor_loss_fn

**Files:**

- Create: `tests/test_clip_cov.py`

- [ ] **Step 1: Write test file with core loss function tests**

```python
"""Tests for clip_cov PPO loss module."""

import pytest
import torch

from customized_areal.clip_cov.config import ClipCovConfig
from customized_areal.clip_cov.loss import clip_cov_grpo_loss_fn, clip_cov_ppo_actor_loss_fn


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
        """When clip_cov_lb=-inf and clip_cov_ub=inf, no tokens are selected
        for cov clipping, so clip_cov loss equals standard PPO loss."""
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
            behave_imp_weight_mode="disabled",
        )

        cov_loss, stat = clip_cov_ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=eps_clip,
            loss_mask=loss_mask,
            clip_cov_lb=float("-inf"),
            clip_cov_ub=float("inf"),
        )
        # No tokens selected for cov clipping when bounds are infinite
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
        from customized_areal.clip_cov.patch import patch_ppo_actor_to_use_clip_cov_loss

        config = ClipCovConfig()
        patch_ppo_actor_to_use_clip_cov_loss(config)
        # Calling again should not raise
        patch_ppo_actor_to_use_clip_cov_loss(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_clip_cov.py -v --tb=short 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.clip_cov'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_clip_cov.py
git commit -m "test(clip-cov): add failing tests for clip_cov loss"
```

______________________________________________________________________

### Task 3: Implement clip_cov_ppo_actor_loss_fn

**Files:**

- Create: `customized_areal/clip_cov/loss.py`

- [ ] **Step 1: Write the core loss function**

```python
"""Clip-cov PPO loss functions.

Implements covariance-aware PPO clipping from
https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

Adapted to AReaL conventions (loss_mask, proximal_logprobs, stat dict format).
"""

from __future__ import annotations

import torch


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean over valid (masked) elements."""
    return (x * mask.float()).sum() / (mask.float().sum() + 1e-8)


def clip_cov_ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    clip_ratio: float = 0.0002,
    clip_cov_lb: float = 1.0,
    clip_cov_ub: float = 5.0,
) -> tuple[torch.Tensor, dict]:
    """Covariance-aware PPO actor loss.

    Extends standard PPO clipping with gradient masking for tokens whose
    advantage-logprob covariance falls within [clip_cov_lb, clip_cov_ub].
    Tokens already clipped by standard PPO are excluded from cov selection.

    Args:
        logprobs: Current policy log-probabilities.
        proximal_logprobs: Proximal policy log-probabilities (for PPO ratio).
        old_logprobs: Behavior policy log-probabilities (unused here, kept for API compat).
        advantages: Advantage estimates.
        eps_clip: PPO clipping parameter.
        loss_mask: Boolean mask for valid tokens.
        eps_clip_higher: Asymmetric higher clipping bound.
        c_clip: Dual clipping parameter (must be > 1.0).
        clip_ratio: Fraction of valid tokens to zero via cov clipping.
        clip_cov_lb: Lower bound of covariance range for candidate selection.
        clip_cov_ub: Upper bound of covariance range for candidate selection.

    Returns:
        Tuple of (loss_scalar, stat_dict) compatible with AReaL's ppo_actor_loss_fn.
    """
    loss_mask_count = loss_mask.count_nonzero() or 1

    # Compute ratio (AReaL convention: masked ratio is 0)
    mask_float = loss_mask.float()
    ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0)

    # Standard PPO clipped loss
    eps_higher = eps_clip if eps_clip_higher is None else eps_clip_higher
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_higher)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio

    # Identify tokens clipped by standard PPO (where clipping increases the loss)
    clip_by_origin = (pg_losses2 > pg_losses1) & loss_mask
    pg_loss = torch.max(pg_losses1, pg_losses2)

    # Dual clipping
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_by_origin)

    # Compute per-token covariance between advantages and logprobs
    adv_mean = _masked_mean(advantages, mask_float)
    logp_mean = _masked_mean(logprobs.detach(), mask_float)
    cov_all = (advantages - adv_mean) * (logprobs.detach() - logp_mean)

    # Mask out tokens ineligible for cov selection
    cov_all = cov_all.clone()
    cov_all[~loss_mask] = float("-inf")
    cov_all[clip_by_origin] = float("-inf")

    # Select tokens within the covariance range
    candidates = (cov_all > clip_cov_lb) & (cov_all < clip_cov_ub) & loss_mask
    candidate_indices = torch.nonzero(candidates)

    # Randomly select up to clip_num tokens
    clip_num = max(int(clip_ratio * loss_mask_count.item()), 1)

    if len(candidate_indices) > 0:
        perm = torch.randperm(len(candidate_indices))
        selected = candidate_indices[perm[: min(clip_num, len(candidate_indices))]]
    else:
        selected = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    # Build corr mask: 1 everywhere, 0 for selected tokens
    corr = torch.ones_like(advantages)
    if len(selected) > 0:
        corr[selected[:, 0], selected[:, 1]] = 0.0

    # Apply corr mask to loss
    pg_loss = pg_loss * corr

    # Aggregate loss (AReaL convention)
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0.0).sum() / loss_mask_count

    # Build stat dict (AReaL-compatible)
    clip_mask = clip_by_origin & loss_mask
    dual_clip_mask = dual_clip_mask & loss_mask if c_clip is not None else dual_clip_mask
    clip_cov_mask = (corr == 0) & loss_mask

    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
        clip_cov_mask=clip_cov_mask,
    )
    return pg_loss, stat


def clip_cov_grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    clip_ratio: float = 0.0002,
    clip_cov_lb: float = 1.0,
    clip_cov_ub: float = 5.0,
    prox_logp_method: str = "recompute",
    current_version: int | None = None,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """GRPO-compatible wrapper for clip-cov loss.

    Resolves proximal log-probs and delegates to clip_cov_ppo_actor_loss_fn.
    Does not support SAPO, decoupled loss, or teacher KL distillation.
    """
    from areal.trainer.ppo.actor import _resolve_proximal_logp

    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")

    entropy = entropy.detach()

    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    loss, stat = clip_cov_ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=prox_logp,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        clip_ratio=clip_ratio,
        clip_cov_lb=clip_cov_lb,
        clip_cov_ub=clip_cov_ub,
    )
    return loss, stat
```

- [ ] **Step 2: Run core loss tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_clip_cov.py::TestClipCovPpoActorLossFn -v --tb=short 2>&1 | tail -20`
Expected: All `TestClipCovPpoActorLossFn` tests PASS

- [ ] **Step 3: Run wrapper tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_clip_cov.py::TestClipCovGrpoLossFn -v --tb=short 2>&1 | tail -20`
Expected: All `TestClipCovGrpoLossFn` tests PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/clip_cov/loss.py
git commit -m "feat(clip-cov): implement clip_cov_ppo_actor_loss_fn and wrapper"
```

______________________________________________________________________

### Task 4: Create patch module and __init__.py

**Files:**

- Create: `customized_areal/clip_cov/patch.py`

- Create: `customized_areal/clip_cov/__init__.py`

- [ ] **Step 1: Write patch.py**

```python
"""Monkey-patch PPOActor to use clip-cov loss."""

from __future__ import annotations

import functools
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list

from .config import ClipCovConfig
from .loss import clip_cov_grpo_loss_fn

logger = logging.getLogger("ClipCov")

_patch_applied = False


def patch_ppo_actor_to_use_clip_cov_loss(config: ClipCovConfig) -> None:
    """Patch PPOActor._ppo_update to use clip-cov loss.

    Idempotent: calling multiple times has no additional effect.

    Args:
        config: ClipCovConfig with clip_ratio, clip_cov_lb, clip_cov_ub.
    """
    global _patch_applied
    if _patch_applied:
        return

    def _ppo_update_with_clip_cov_loss(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code (mirrors PPOActor._ppo_update) ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=infer_token_denominator(data, loss_mask),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
            clip_cov_clip_ratio=config.clip_ratio,
            clip_cov_lb=config.clip_cov_lb,
            clip_cov_ub=config.clip_cov_ub,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        clip_cov_grpo_loss_fn,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        clip_ratio=config.clip_ratio,
                        clip_cov_lb=config.clip_cov_lb,
                        clip_cov_ub=config.clip_cov_ub,
                        prox_logp_method=self.config.prox_logp_method,
                        current_version=current_version,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

    PPOActor._ppo_update = _ppo_update_with_clip_cov_loss
    _patch_applied = True
    logger.info(
        "PPOActor class patched to use clip_cov loss "
        "(clip_ratio=%.4f, lb=%.1f, ub=%.1f)",
        config.clip_ratio,
        config.clip_cov_lb,
        config.clip_cov_ub,
    )
```

- [ ] **Step 2: Write __init__.py**

```python
"""Clip-cov PPO loss module.

Provides covariance-aware PPO clipping as a monkey-patch for AReaL's PPOActor.

Usage:
    from customized_areal.clip_cov import ClipCovConfig, patch_ppo_actor_to_use_clip_cov_loss

    config = ClipCovConfig(clip_ratio=0.0002, clip_cov_lb=1.0, clip_cov_ub=5.0)
    patch_ppo_actor_to_use_clip_cov_loss(config)
"""

from .config import ClipCovConfig
from .patch import patch_ppo_actor_to_use_clip_cov_loss

__all__ = ["ClipCovConfig", "patch_ppo_actor_to_use_clip_cov_loss"]
```

- [ ] **Step 3: Run patch tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_clip_cov.py::TestClipCovPatch -v --tb=short 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 4: Run all clip_cov tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_clip_cov.py -v --tb=short 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/clip_cov/patch.py customized_areal/clip_cov/__init__.py
git commit -m "feat(clip-cov): add PPOActor patch and module init"
```

______________________________________________________________________

### Task 5: Run pre-commit and final verification

**Files:**

- All files in `customized_areal/clip_cov/`

- `tests/test_clip_cov.py`

- [ ] **Step 1: Run pre-commit on all new files**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pre-commit run --files customized_areal/clip_cov/config.py customized_areal/clip_cov/loss.py customized_areal/clip_cov/patch.py customized_areal/clip_cov/__init__.py tests/test_clip_cov.py 2>&1`
Expected: All checks PASS (or auto-fix applied)

- [ ] **Step 2: If pre-commit made changes, re-commit the fixes**

```bash
git add customized_areal/clip_cov/ tests/test_clip_cov.py
git commit -m "style(clip-cov): pre-commit formatting fixes"
```

- [ ] **Step 3: Run full test suite one more time**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_clip_cov.py -v 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 4: Verify import works**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.clip_cov import ClipCovConfig, patch_ppo_actor_to_use_clip_cov_loss; print('OK')" 2>&1`
Expected: `OK`

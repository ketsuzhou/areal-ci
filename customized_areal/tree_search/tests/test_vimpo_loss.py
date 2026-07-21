# SPDX-License-Identifier: Apache-2.0
"""Tests for the combined VIMPO actor loss (T6).

Two layers:

1. Pure loss-math tests (``vimpo_loss_terms``) - verify the terminal
   episode-mean numerator, PPO token-mean numerator, reference-tensor
   detachment, and partial-episode rejection.
2. Engine tests (``MultiCandidateFSDPEngine.train_vimpo_batch``) - verify
   one zero-grad / backward / optimizer-step with two separate distributed
   denominators, the scaling formula, all-reduce over ``dp_group``, and
   validation before ``optimizer_zero_grad``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.distributed as dist

from customized_areal.tree_search.engine.fsdp_engine import (
    MultiCandidateFSDPEngine,
)
from customized_areal.tree_search.training.losses.vimpo import (
    VIMPOLossTerms,
    vimpo_loss_fn,
    vimpo_loss_terms,
)

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import (
    MicroBatchList,
    pack_tensor_dict,
)

# =============================================================================
# Helpers
# =============================================================================


def _complete_two_turn_batch() -> dict[str, torch.Tensor]:
    """Build a complete 2-turn VIMPO episode (2 rows, episode 0, turns 1 & 2).

    Every field is shaped ``[2, 3]`` (2 rows, 3 positions).  Row 0 is turn 1,
    row 1 is turn 2.  ``vimpo_episode_index`` is 0 for both rows.  Position 2
    in each row is a terminal/pad position (``vimpo_predict_mask=False``).
    """
    return {
        "vimpo_predict_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "vimpo_episode_index": torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long),
        "vimpo_turn_index": torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long),
        "vimpo_centered_reward": torch.tensor(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32
        ),
        "vimpo_ref_sample_logp": torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.4, -0.4, 0.0]], dtype=torch.float32
        ),
        "vimpo_candidate_kl": torch.tensor(
            [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0]], dtype=torch.float32
        ),
        "logprobs": torch.tensor(
            [[-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0]], dtype=torch.float32
        ),
        "prox_logp": torch.tensor(
            [[-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0]], dtype=torch.float32
        ),
        "advantages": torch.tensor(
            [[1.0, 1.0, 0.0], [1.5, 1.5, 0.0]], dtype=torch.float32
        ),
    }


# =============================================================================
# Step 1: Pure loss-math tests
# =============================================================================


def test_terminal_loss_averages_complete_episodes_not_tokens() -> None:
    """``vimpo_loss_terms`` returns numerators that the caller divides by
    episode_count / valid_token_count.  Reference tensors (ref_sample_logp,
    candidate_kl, advantages, prox_logp) are detached inside the function so
    no gradient flows through them."""
    current = torch.tensor([[-0.2, -0.4, 0.0], [-0.3, 0.0, 0.0]], requires_grad=True)
    data = {
        "vimpo_predict_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "vimpo_episode_index": torch.tensor([[0, 0, 0], [1, 1, 1]]),
        "vimpo_centered_reward": torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]),
        "vimpo_ref_sample_logp": torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.0, 0.0]], requires_grad=True
        ),
        "vimpo_candidate_kl": torch.tensor(
            [[0.1, 0.2, 0.0], [0.3, 0.0, 0.0]], requires_grad=True
        ),
        "logprobs": torch.tensor(
            [[-0.25, -0.45, 0.0], [-0.35, 0.0, 0.0]], requires_grad=True
        ),
        "prox_logp": torch.tensor(
            [[-0.25, -0.45, 0.0], [-0.35, 0.0, 0.0]], requires_grad=True
        ),
        "advantages": torch.tensor(
            [[1.0, 2.0, 0.0], [-1.0, 0.0, 0.0]], requires_grad=True
        ),
    }
    terms = vimpo_loss_terms(
        current, data, beta=0.5, eps_clip=0.2, eps_clip_higher=None
    )
    assert terms.episode_count == 2
    assert terms.valid_token_count == 3
    total = (
        terms.value_sum / terms.episode_count
        + 0.005 * terms.ppo_sum / terms.valid_token_count
    )
    total.backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert data["vimpo_ref_sample_logp"].grad is None
    assert data["vimpo_candidate_kl"].grad is None
    assert data["advantages"].grad is None
    assert data["prox_logp"].grad is None


def test_loss_rejects_partial_episode_before_backward() -> None:
    """When ``vimpo_expected_turn_count`` is present and an episode has fewer
    turns than expected, ``vimpo_loss_terms`` raises ``ValueError`` BEFORE
    returning (so no backward can happen on a partial episode)."""
    data = _complete_two_turn_batch()
    data["vimpo_expected_turn_count"] = torch.full_like(data["vimpo_turn_index"], 2)
    partial = {key: value[:1] for key, value in data.items()}
    with pytest.raises(ValueError, match="episode 0 is incomplete"):
        vimpo_loss_terms(
            torch.zeros_like(partial["vimpo_ref_sample_logp"], requires_grad=True),
            partial,
            beta=5e-4,
            eps_clip=0.2,
            eps_clip_higher=None,
        )


def test_loss_rejects_inconsistent_centered_reward() -> None:
    """``vimpo_centered_reward`` must be constant within each episode; a
    mismatch across rows of the same episode raises ``ValueError``."""
    current = torch.tensor([[-0.2, -0.4], [-0.3, -0.1]], requires_grad=True)
    data = {
        "vimpo_predict_mask": torch.tensor([[True, True], [True, True]]),
        "vimpo_episode_index": torch.tensor([[0, 0], [0, 0]]),
        "vimpo_centered_reward": torch.tensor([[0.5, 0.5], [-0.5, -0.5]]),
        "vimpo_ref_sample_logp": torch.tensor([[-0.5, -0.5], [-0.4, -0.4]]),
        "vimpo_candidate_kl": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "logprobs": torch.tensor([[-0.25, -0.45], [-0.35, -0.15]]),
        "prox_logp": torch.tensor([[-0.25, -0.45], [-0.35, -0.15]]),
        "advantages": torch.tensor([[1.0, 2.0], [1.5, 0.5]]),
    }
    with pytest.raises(ValueError, match="inconsistent centered rewards"):
        vimpo_loss_terms(current, data, beta=0.5, eps_clip=0.2, eps_clip_higher=None)


def test_loss_rejects_non_finite_inputs() -> None:
    """Non-finite reference/logprob values raise ``ValueError``."""
    current = torch.tensor([[-0.2, -0.4]], requires_grad=True)
    data = {
        "vimpo_predict_mask": torch.tensor([[True, True]]),
        "vimpo_episode_index": torch.tensor([[0, 0]]),
        "vimpo_centered_reward": torch.tensor([[0.5, 0.5]]),
        "vimpo_ref_sample_logp": torch.tensor([[-0.5, float("nan")]]),
        "vimpo_candidate_kl": torch.tensor([[0.1, 0.2]]),
        "logprobs": torch.tensor([[-0.25, -0.45]]),
        "prox_logp": torch.tensor([[-0.25, -0.45]]),
        "advantages": torch.tensor([[1.0, 2.0]]),
    }
    with pytest.raises(ValueError, match="non-finite"):
        vimpo_loss_terms(current, data, beta=0.5, eps_clip=0.2, eps_clip_higher=None)


def test_loss_rejects_zero_valid_tokens() -> None:
    """An empty predict mask raises ``ValueError``."""
    current = torch.tensor([[-0.2, -0.4]], requires_grad=True)
    data = {
        "vimpo_predict_mask": torch.tensor([[False, False]]),
        "vimpo_episode_index": torch.tensor([[0, 0]]),
        "vimpo_centered_reward": torch.tensor([[0.5, 0.5]]),
        "vimpo_ref_sample_logp": torch.tensor([[-0.5, -0.5]]),
        "vimpo_candidate_kl": torch.tensor([[0.1, 0.2]]),
        "logprobs": torch.tensor([[-0.25, -0.45]]),
        "prox_logp": torch.tensor([[-0.25, -0.45]]),
        "advantages": torch.tensor([[1.0, 2.0]]),
    }
    with pytest.raises(ValueError, match="no valid"):
        vimpo_loss_terms(current, data, beta=0.5, eps_clip=0.2, eps_clip_higher=None)


def test_vimpo_loss_fn_scales_per_formula() -> None:
    """``vimpo_loss_fn`` combines PPO token-mean and terminal episode-mean
    using two separate denominators and the dp_size gradient-compensation
    factor."""
    ppo_sum = torch.tensor(3.0)
    value_sum = torch.tensor(1.5)
    terms = VIMPOLossTerms(
        ppo_sum=ppo_sum,
        value_sum=value_sum,
        valid_token_count=torch.tensor(6),
        episode_count=torch.tensor(2),
        terminal_prediction=torch.tensor([0.0]),
        terminal_target=torch.tensor([0.0]),
        terminal_residual=torch.tensor([0.0]),
    )
    loss = vimpo_loss_fn(
        terms,
        actor_coeff=0.005,
        value_loss_weight=1.0,
        global_valid_tokens=torch.tensor(12.0),
        global_episodes=torch.tensor(4.0),
        dp_size=2,
    )
    expected = 2 * (0.005 * 3.0 / 12.0 + 1.0 * 1.5 / 4.0)
    torch.testing.assert_close(loss, torch.tensor(expected), rtol=1e-6, atol=1e-7)


# =============================================================================
# Step 4: Engine tests (one zero-grad / backward / step, two denominators)
# =============================================================================


def _engine_batch() -> dict[str, torch.Tensor]:
    """Minimal VIMPO batch for engine tests (1 row, 4 positions, 2 predict)."""
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[True, True, True, True]]),
        "vimpo_predict_mask": torch.tensor([[True, True, False, False]]),
        "vimpo_episode_index": torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        "vimpo_centered_reward": torch.tensor(
            [[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32
        ),
        "vimpo_ref_sample_logp": torch.tensor(
            [[-0.5, -0.5, 0.0, 0.0]], dtype=torch.float32
        ),
        "vimpo_candidate_kl": torch.tensor([[0.1, 0.2, 0.0, 0.0]], dtype=torch.float32),
        "logprobs": torch.tensor([[-0.25, -0.45, 0.0, 0.0]], dtype=torch.float32),
        "prox_logp": torch.tensor([[-0.25, -0.45, 0.0, 0.0]], dtype=torch.float32),
        "advantages": torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.float32),
    }


class _RecordingEngine(MultiCandidateFSDPEngine):
    """Recording fake engine built via ``__new__`` (no model, no FSDP, no GPU).

    Overrides ``forward_backward_batch`` / ``optimizer_zero_grad`` /
    ``optimizer_step`` to record call counts, and
    ``_compute_logprobs_entropy`` to return a preset logprobs tensor so the
    process callback's scaling can be exercised end-to-end.
    """

    def __init__(
        self,
        *,
        dp_size: int = 1,
        device: torch.device | None = None,
        fake_logprobs: torch.Tensor | None = None,
    ) -> None:
        # NOTE: bypass FSDPEngine.__init__ entirely; set only what we need.
        self._dp_size = dp_size
        self._device = device or torch.device("cpu")
        self._fake_logprobs = fake_logprobs
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.forward_backward_calls = 0
        self.captured_process_fn: Any = None
        self.captured_losses: list[torch.Tensor] = []
        # ``_prepare_vimpo_mb_list`` logs via ``self.logger``.
        import logging as _stdlib_logging

        self.logger = _stdlib_logging.getLogger("RecordingEngine")

    # --- minimal attribute stubs ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parallel_helper(self) -> Any:
        class _PH:
            dp_size = self._dp_size
            tp_size = 1
            sp_size = 1
            tp_group = None

        return _PH()

    @property
    def dp_group(self) -> Any:
        return None

    @property
    def config(self) -> Any:
        class _Cfg:
            mb_spec = MicroBatchSpec(n_mbs=1)
            pad_to_maximum = False
            temperature = 1.0

        return _Cfg()

    @property
    def enable_tree_training(self) -> bool:
        return False

    @property
    def model_config(self) -> Any:
        class _MC:
            model_type = "qwen2"

        return _MC()

    # --- overrides ---

    def _ensure_ready(self) -> None:
        pass

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
        labels_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Return a tensor matching the padded input length so the process
        # callback's ``logprobs[:-pad_length]`` trim produces the right
        # content-length slice.  The first ``len(fake_logprobs)`` positions
        # carry the preset values; the rest are zero (will be trimmed).
        padded_len = inputs["input_ids"].shape[-1]
        fake = torch.zeros(padded_len, dtype=self._fake_logprobs.dtype)
        n = min(len(self._fake_logprobs), padded_len)
        fake[:n] = self._fake_logprobs[:n]
        entropy = torch.zeros(padded_len, dtype=self._fake_logprobs.dtype)
        return fake, entropy

    def optimizer_zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def optimizer_step(self) -> dict[str, float]:
        self.step_calls += 1
        return {"update_successful": 1.0, "grad_norm": 0.0, "lr": 1e-4}

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Any,
        forward_only: bool = False,
    ) -> None:
        self.forward_backward_calls += 1
        self.captured_process_fn = process_output_fn
        for mb_item in mb_list:
            ctx_dict = {
                "model_inputs": mb_item.padded_mb,
                "mb_input": mb_item.orig_mb,
                "pad_length": mb_item.padding_length,
                "ulysses_pad_size": 0,
                "trie_node": None,
            }
            fake_logits = torch.zeros(1, 1, 4)
            loss = process_output_fn(fake_logits, ctx_dict)
            if loss is not None:
                self.captured_losses.append(loss.detach())


def _make_recording_engine(
    *,
    dp_size: int = 1,
    fake_logprobs: torch.Tensor | None = None,
) -> _RecordingEngine:
    """Build a ``_RecordingEngine`` bypassing ``__init__``."""
    engine = _RecordingEngine.__new__(_RecordingEngine)
    engine.__init__(dp_size=dp_size, fake_logprobs=fake_logprobs)
    return engine


def test_train_vimpo_batch_calls_each_method_once(monkeypatch) -> None:
    """``train_vimpo_batch`` calls zero_grad, forward_backward, step exactly
    once each (one backward, one optimizer step)."""
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: None)
    fake_lp = torch.tensor([-0.2, -0.4, 0.0, 0.0])
    engine = _make_recording_engine(dp_size=1, fake_logprobs=fake_lp)
    engine.train_vimpo_batch(
        _engine_batch(),
        actor_coeff=0.005,
        value_loss_weight=1.0,
        beta=0.5,
        eps_clip=0.2,
    )
    assert engine.zero_grad_calls == 1
    assert engine.forward_backward_calls == 1
    assert engine.step_calls == 1


def test_train_vimpo_batch_validates_before_zero_grad(monkeypatch) -> None:
    """Invalid input (missing required field) raises before
    ``optimizer_zero_grad`` is called."""
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: None)
    fake_lp = torch.tensor([-0.2, -0.4, 0.0, 0.0])
    engine = _make_recording_engine(dp_size=1, fake_logprobs=fake_lp)
    bad_batch = _engine_batch()
    del bad_batch["vimpo_ref_sample_logp"]
    with pytest.raises(ValueError, match="vimpo_ref_sample_logp"):
        engine.train_vimpo_batch(
            bad_batch,
            actor_coeff=0.005,
            value_loss_weight=1.0,
            beta=0.5,
            eps_clip=0.2,
        )
    assert engine.zero_grad_calls == 0
    assert engine.forward_backward_calls == 0
    assert engine.step_calls == 0


def test_train_vimpo_batch_all_reduces_denominators(monkeypatch) -> None:
    """``train_vimpo_batch`` all-reduces ``[valid_token_count,
    episode_count]`` (float64) over ``dp_group`` once."""
    all_reduced: list[torch.Tensor] = []

    def _spy_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        all_reduced.append(tensor.clone())
        return None

    monkeypatch.setattr(dist, "all_reduce", _spy_all_reduce)
    fake_lp = torch.tensor([-0.2, -0.4, 0.0, 0.0])
    engine = _make_recording_engine(dp_size=2, fake_logprobs=fake_lp)
    engine.train_vimpo_batch(
        _engine_batch(),
        actor_coeff=0.005,
        value_loss_weight=1.0,
        beta=0.5,
        eps_clip=0.2,
    )
    # Exactly two all_reduce calls: valid_token_count + episode_count.
    assert len(all_reduced) == 2
    # Both must be float64 scalars.
    for t in all_reduced:
        assert t.dtype == torch.float64
        assert t.numel() == 1
    # Local values: valid_token_count=2 (2 predict positions), episode_count=1.
    torch.testing.assert_close(
        all_reduced[0].float(), torch.tensor(2.0), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        all_reduced[1].float(), torch.tensor(1.0), rtol=1e-6, atol=1e-7
    )


def test_train_vimpo_batch_scales_per_formula(monkeypatch) -> None:
    """The per-microbatch loss matches
    ``dp_size * (actor_coeff * ppo_sum / global_valid_tokens
                 + value_loss_weight * value_sum / global_episodes)``."""
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: None)

    fake_lp = torch.tensor([-0.2, -0.4, 0.0, 0.0])
    dp_size = 2
    actor_coeff = 0.005
    value_loss_weight = 1.0
    beta = 0.5
    eps_clip = 0.2

    engine = _make_recording_engine(dp_size=dp_size, fake_logprobs=fake_lp)
    engine.train_vimpo_batch(
        _engine_batch(),
        actor_coeff=actor_coeff,
        value_loss_weight=value_loss_weight,
        beta=beta,
        eps_clip=eps_clip,
    )

    # The fake forward_backward_batch called the process fn once per mb.
    assert len(engine.captured_losses) == 1
    actual_loss = engine.captured_losses[0]

    # Independently compute expected loss using vimpo_loss_terms + vimpo_loss_fn.
    # After pack+unsqueeze+trim, the mb_input tensors are [1, 4] -> squeeze -> [4].
    # pad_length is 0 (single mb, no padding needed for 4 tokens).
    batch = _engine_batch()
    packed = pack_tensor_dict(
        {
            **batch,
            "position_ids": torch.arange(4).unsqueeze(0).expand(1, -1),
        }
    )
    # Squeeze to 1D for vimpo_loss_terms (matching what the process callback does).
    data_1d: dict[str, torch.Tensor] = {}
    for key in (
        "vimpo_predict_mask",
        "vimpo_episode_index",
        "vimpo_centered_reward",
        "vimpo_ref_sample_logp",
        "vimpo_candidate_kl",
        "logprobs",
        "prox_logp",
        "advantages",
    ):
        t = packed[key]
        if t.ndim >= 1 and t.shape[0] == 1:
            t = t.squeeze(0)
        data_1d[key] = t

    terms = vimpo_loss_terms(
        fake_lp, data_1d, beta=beta, eps_clip=eps_clip, eps_clip_higher=None
    )
    global_valid_tokens = terms.valid_token_count.to(torch.float64)
    global_episodes = terms.episode_count.to(torch.float64)
    expected_loss = vimpo_loss_fn(
        terms,
        actor_coeff=actor_coeff,
        value_loss_weight=value_loss_weight,
        global_valid_tokens=global_valid_tokens,
        global_episodes=global_episodes,
        dp_size=dp_size,
    )
    torch.testing.assert_close(
        actual_loss, expected_loss.detach(), rtol=1e-5, atol=1e-6
    )

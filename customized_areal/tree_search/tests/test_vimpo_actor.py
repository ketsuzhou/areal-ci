# SPDX-License-Identifier: Apache-2.0
"""Tests for VIMPOFSDPPPOActor orchestration (Task 7).

Verifies the ``compute_advantages`` ordering (validate identity -> actor
snapshot -> reference score -> advantage), that a reference failure propagates
before any optimizer mutation (OpenSpec 5.4), and that ``ppo_update`` calls
``train_vimpo_batch`` exactly once per complete PPO minibatch (never the
generic ``train_batch``).

The actor is built via ``__new__`` (no FSDP/GPU/model) with fake scorer,
advantage-computer, and engine dependencies - the same pattern as
``test_vimpo_loss.py``'s ``_RecordingEngine``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from customized_areal.tree_search.engine.fsdp_engine import VIMPOCandidateStats
from customized_areal.tree_search.training.actor import VIMPOFSDPPPOActor
from customized_areal.tree_search.training.vimpo_reference import (
    ReferenceScore,
)

# =============================================================================
# Fake dependencies
# =============================================================================


class _RecordingOptimizer:
    """Records zero_grad / step calls (guards the reference-failure test)."""

    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.zero_grad_calls += 1

    def step(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step_calls += 1
        return {"update_successful": 1.0, "grad_norm": 0.0, "lr": 1e-4}


class _RecordingScorer:
    """Fake frozen-reference scorer that records validate/score events."""

    def __init__(
        self,
        events: list[str],
        *,
        scores: list[ReferenceScore] | None = None,
        fail_on_score: bool = False,
    ) -> None:
        self._events = events
        self._scores = scores
        self._fail = fail_on_score

    def validate_identity(self, expected: Any) -> None:
        self._events.append("validate_identity")

    def score(self, requests: list[Any]) -> list[ReferenceScore]:
        if self._fail:
            raise RuntimeError("reference unavailable")
        self._events.append("reference_score")
        if self._scores is not None:
            return self._scores
        return [
            ReferenceScore(
                key=r.key,
                sampled_logp=-0.1,
                candidate_logp=[-0.1] * len(r.candidate_token_ids),
            )
            for r in requests
        ]

    def close(self) -> None:
        pass


class _RecordingAdvantage:
    """Fake VIMPOAdvantageComputer that records the advantage event."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def compute(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._events.append("advantage")
        mask = batch["vimpo_predict_mask"]
        shape = mask.shape
        batch["vimpo_candidate_kl"] = torch.zeros(shape, dtype=torch.float32)
        batch["vimpo_retained_mass"] = torch.zeros(shape, dtype=torch.float32)
        batch["advantages"] = torch.zeros(shape, dtype=torch.float32)
        return batch


# =============================================================================
# Batch fixtures
# =============================================================================


def _vimpo_rollout_batch() -> dict[str, torch.Tensor]:
    """A single-trajectory VIMPO rollout batch (1 row, 4 positions)."""
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
        "vimpo_episode_index": torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        "vimpo_turn_index": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
    }


def _fake_candidate_stats() -> VIMPOCandidateStats:
    """VIMPOCandidateStats matching ``_vimpo_rollout_batch`` (2 predict positions)."""
    return VIMPOCandidateStats(
        sampled_logp=torch.tensor([[-0.1, -0.2, 0.0, 0.0]], dtype=torch.float32),
        candidate_ids=torch.tensor(
            [[[10, 11], [12, 13], [-1, -1], [-1, -1]]], dtype=torch.long
        ),
        candidate_logp=torch.tensor(
            [[[-0.1, -0.2], [-0.3, -0.4], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        ),
        retained_mass=torch.tensor([[0.9, 0.8, 0.0, 0.0]], dtype=torch.float32),
        predict_mask=torch.tensor([[True, True, False, False]]),
    )


def _complete_enriched_batch() -> dict[str, torch.Tensor]:
    """A complete enriched VIMPO batch ready for ``ppo_update`` (1 row, 4 positions)."""
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "vimpo_predict_mask": torch.tensor([[True, True, False, False]]),
        "vimpo_episode_index": torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        "vimpo_turn_index": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
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


# =============================================================================
# Fake actor builders
# =============================================================================


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        vimpo_top_k=2,
        vimpo_beta=5e-4,
        vimpo_actor_coeff=5e-3,
        vimpo_value_loss_weight=1.0,
        vimpo_gamma=1.0,
        vimpo_lambda=1.0,
        vimpo_whiten_advantages=False,
        vimpo_detach_kl=True,
        vimpo_ref_base_url="http://fake",
        vimpo_ref_timeout=1.0,
        vimpo_ref_max_concurrency=1,
        vimpo_ref_max_retries=0,
        eps_clip=0.2,
        eps_clip_higher=None,
        ppo_n_minibatches=1,
        path="/fake/model",
    )


def _fake_vimpo_actor(events: list[str]) -> VIMPOFSDPPPOActor:
    """Actor whose scorer/snapshot/advantage all record into ``events``."""
    actor = VIMPOFSDPPPOActor.__new__(VIMPOFSDPPPOActor)
    actor.config = _fake_config()
    actor.reference_scorer = _RecordingScorer(events)
    actor.vimpo_advantage = _RecordingAdvantage(events)
    actor.optimizer = _RecordingOptimizer()
    actor.mp_group = None
    actor.dp_group = None
    stats = _fake_candidate_stats()

    def _snapshot(data: Any, *, top_k: int) -> VIMPOCandidateStats:
        events.append("actor_snapshot")
        return stats

    actor.compute_vimpo_candidate_stats = _snapshot  # type: ignore[method-assign]
    return actor


def _fake_vimpo_actor_with_failing_reference() -> tuple[
    VIMPOFSDPPPOActor, _RecordingOptimizer
]:
    """Actor whose reference scorer raises on score (optimizer recorded)."""
    events: list[str] = []
    optimizer = _RecordingOptimizer()
    actor = VIMPOFSDPPPOActor.__new__(VIMPOFSDPPPOActor)
    actor.config = _fake_config()
    actor.reference_scorer = _RecordingScorer(events, fail_on_score=True)
    actor.vimpo_advantage = _RecordingAdvantage(events)
    actor.optimizer = optimizer
    actor.mp_group = None
    actor.dp_group = None
    stats = _fake_candidate_stats()

    def _snapshot(data: Any, *, top_k: int) -> VIMPOCandidateStats:
        events.append("actor_snapshot")
        return stats

    actor.compute_vimpo_candidate_stats = _snapshot  # type: ignore[method-assign]
    return actor, optimizer


def _fake_vimpo_actor_with_engine() -> VIMPOFSDPPPOActor:
    """Actor whose train_vimpo_batch / train_batch are recording stubs."""
    events: list[str] = []
    actor = _fake_vimpo_actor(events)
    actor.train_vimpo_batch_calls = 0
    actor.train_batch_calls = 0

    def _record_vimpo(input_: Any, **kwargs: Any) -> dict[str, float]:
        actor.train_vimpo_batch_calls += 1
        return {
            "update_successful": 1.0,
            "vimpo/ppo_actor_loss": 0.0,
            "vimpo/value_loss": 0.0,
            "vimpo/combined_loss": 0.0,
            "vimpo/terminal_rmse": 0.0,
        }

    def _record_batch(*args: Any, **kwargs: Any) -> dict[str, float]:
        actor.train_batch_calls += 1
        return {}

    actor.train_vimpo_batch = _record_vimpo  # type: ignore[method-assign]
    actor.train_batch = _record_batch  # type: ignore[method-assign]
    return actor


# =============================================================================
# Step 1: compute_advantages orchestration tests
# =============================================================================


def test_compute_advantages_orders_snapshot_reference_then_advantage() -> None:
    """compute_advantages runs validate_identity -> actor_snapshot ->
    reference_score -> advantage, in that exact order."""
    events: list[str] = []
    actor = _fake_vimpo_actor(events)
    enriched = actor.compute_advantages([_vimpo_rollout_batch()])
    assert events == [
        "validate_identity",
        "actor_snapshot",
        "reference_score",
        "advantage",
    ]
    assert "vimpo_candidate_kl" in enriched[0]
    assert enriched[0]["advantages"].requires_grad is False


def test_reference_failure_happens_before_optimizer_mutation() -> None:
    """When the reference scorer fails, compute_advantages raises before any
    optimizer zero_grad / step (OpenSpec 5.4)."""
    actor, optimizer = _fake_vimpo_actor_with_failing_reference()
    with pytest.raises(RuntimeError, match="reference unavailable"):
        actor.compute_advantages([_vimpo_rollout_batch()])
    assert optimizer.zero_grad_calls == 0
    assert optimizer.step_calls == 0


def test_ppo_update_uses_one_combined_train_call() -> None:
    """ppo_update calls train_vimpo_batch exactly once for one complete PPO
    minibatch and never calls the generic train_batch."""
    actor = _fake_vimpo_actor_with_engine()
    actor.ppo_update([_complete_enriched_batch()])
    assert actor.train_vimpo_batch_calls == 1
    assert actor.train_batch_calls == 0


def test_destroy_closes_reference_scorer(monkeypatch) -> None:
    """destroy() closes the reference scorer before delegating to the engine."""
    closed: list[bool] = []

    class _CloseableScorer:
        def close(self) -> None:
            closed.append(True)

    actor = VIMPOFSDPPPOActor.__new__(VIMPOFSDPPPOActor)
    actor.reference_scorer = _CloseableScorer()
    # Stub the heavy super().destroy() (FSDPEngine.destroy needs model/optimizer).
    from customized_areal.tree_search.engine.fsdp_engine import (
        MultiCandidateFSDPEngine,
    )

    monkeypatch.setattr(MultiCandidateFSDPEngine, "destroy", lambda self: None)
    actor.destroy()
    assert closed == [True]


def test_retained_mass_quantiles_subsample_over_quantile_limit(monkeypatch) -> None:
    """Valid-token counts above the torch.quantile ceiling (2**24) are
    subsampled deterministically (uniform stride) instead of erroring."""
    import customized_areal.tree_search.training.actor as actor_module

    # Shrink the ceiling so the 2-valid-token fake batch trips the guard.
    monkeypatch.setattr(actor_module, "_QUANTILE_INPUT_LIMIT", 1)
    seen_numel: list[int] = []
    real_quantile = torch.quantile

    def _spy_quantile(values: torch.Tensor, q: torch.Tensor, **kwargs: Any):
        seen_numel.append(values.numel())
        return real_quantile(values, q, **kwargs)

    monkeypatch.setattr(torch, "quantile", _spy_quantile)
    actor = _fake_vimpo_actor([])
    actor.compute_advantages([_vimpo_rollout_batch()])
    # 2 valid tokens with ceiling 1 -> stride-2 subsample of 1 element.
    assert seen_numel == [1]

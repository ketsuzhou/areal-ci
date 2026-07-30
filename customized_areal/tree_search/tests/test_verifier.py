"""Tests for the Phase 2 verifier slice used by Phase 4."""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.verifier import (
    ObjectiveVerifier,
    Verifier,
    VerifierResult,
)


def test_verifier_result_defaults() -> None:
    r = VerifierResult(success=True, reward=0.5, source="objective")
    assert r.rationale == ""
    assert r.per_step_signals == {}


@pytest.mark.asyncio
async def test_objective_verifier_success_returns_reward_one() -> None:
    verifier = ObjectiveVerifier(check=lambda run: True)
    result = await verifier.verify({"task_id": "t1"})
    assert result.success is True
    assert result.reward == 1.0
    assert result.source == "objective"


@pytest.mark.asyncio
async def test_objective_verifier_failure_returns_reward_zero() -> None:
    verifier = ObjectiveVerifier(check=lambda run: False)
    result = await verifier.verify({"task_id": "t1"})
    assert result.success is False
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_objective_verifier_raising_check_is_failure_not_crash() -> None:
    def boom(run: dict) -> bool:
        raise RuntimeError("check exploded")

    verifier = ObjectiveVerifier(check=boom)
    result = await verifier.verify({"task_id": "t1"})
    assert result.success is False
    assert result.reward == 0.0
    assert "exploded" in result.rationale


def test_objective_verifier_satisfies_protocol() -> None:
    verifier = ObjectiveVerifier(check=lambda run: True)
    assert isinstance(verifier, Verifier)

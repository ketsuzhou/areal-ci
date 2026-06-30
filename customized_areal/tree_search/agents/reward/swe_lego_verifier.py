"""Hybrid SWE-Lego verifier: objective tests + generative critic + semi-resolved.

Composes three layers (spec §5.3). The objective layer short-circuits when
decisive; the blend runs only in the mixed middle. Uses stdlib
:mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("SweLegoVerifier")


@dataclass(frozen=True)
class ObjectiveOutcome:
    fully_passes: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    failing_count_reduced: bool


class _Objective(Protocol):
    async def run_tests(
        self, sandbox_id: str, fail_to_pass: list[str], pass_to_pass: list[str]
    ) -> ObjectiveOutcome: ...


class _Critic(Protocol):
    async def score(
        self, transcript: str, acceptance_criteria: str, objective: ObjectiveOutcome
    ) -> float: ...


class _RlSession(Protocol):
    async def set_reward(self, *, session_id: str, reward: float) -> None: ...


@dataclass(frozen=True)
class BlendWeights:
    objective: float = 0.7
    generative: float = 0.2
    semi_resolved: float = 0.1


class SweLegoVerifier:
    """Hybrid verifier producing a terminal reward per agent run."""

    def __init__(
        self,
        *,
        objective: _Objective,
        critic: _Critic,
        rl_session: _RlSession,
        weights: BlendWeights | None = None,
    ) -> None:
        self._objective = objective
        self._critic = critic
        self._rl = rl_session
        self._weights = weights or BlendWeights()

    async def verify_and_reward(
        self,
        *,
        agent_run_id: str,
        sandbox_id: str,
        session_id: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        transcript: str,
        acceptance_criteria: str,
    ) -> VerifierResult:
        obj = await self._objective.run_tests(sandbox_id, fail_to_pass, pass_to_pass)

        # Short-circuit: objective fully decisive.
        if obj.fully_passes:
            reward = 1.0
            await self._rl.set_reward(session_id=session_id, reward=reward)
            return VerifierResult(
                success=True,
                reward=reward,
                source="objective",
                rationale="FAIL_TO_PASS fully passes",
            )
        if obj.f2p_passed == 0 and not obj.failing_count_reduced:
            reward = 0.0
            await self._rl.set_reward(session_id=session_id, reward=reward)
            return VerifierResult(
                success=False,
                reward=reward,
                source="objective",
                rationale="FAIL_TO_PASS fully fails, no failing-count reduction",
            )

        # Mixed middle: blend.
        gen_score, source = await self._safe_critic_score(
            transcript, acceptance_criteria, obj
        )
        obj_score = obj.f2p_passed / obj.f2p_total if obj.f2p_total else 0.0
        semi = 1.0 if obj.failing_count_reduced else 0.0
        w = self._weights
        reward = (
            w.objective * obj_score + w.generative * gen_score + w.semi_resolved * semi
        )
        await self._rl.set_reward(session_id=session_id, reward=reward)
        return VerifierResult(
            success=obj.fully_passes,
            reward=reward,
            source=source,
            rationale=f"blend: obj={obj_score:.2f} gen={gen_score:.2f} semi={semi:.2f}",
            per_step_signals={"semi_resolved": semi},
        )

    async def _safe_critic_score(
        self, transcript: str, criteria: str, obj: ObjectiveOutcome
    ) -> tuple[float, str]:
        try:
            return await self._critic.score(transcript, criteria, obj), "hybrid"
        except Exception as exc:
            logger.warning("critic failed; using 0.0 for generative term: %s", exc)
            return 0.0, "default"

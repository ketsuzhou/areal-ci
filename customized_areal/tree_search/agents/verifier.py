"""Verifier types for DAG RL training.

This is the minimal, forward-compatible slice of the Phase 2 "verifier agent"
that Phase 4 depends on: the :class:`VerifierResult` dataclass, the
:class:`Verifier` Protocol, and the deterministic :class:`ObjectiveVerifier`.
The LLM-judge fallback verifier is intentionally left for the full Phase 2
implementation -- everything here matches the canonical Phase 2 spec so it can
be extended without breaking changes.

The module uses the stdlib :mod:`logging` (not ``areal.utils.logging``) so the
``dag`` package stays importable without the heavy training stack -- consistent
with the rest of the package (``environment`` / ``execution_dag``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("Verifier")


@dataclass(frozen=True)
class VerifierResult:
    """Outcome of verifying one agent run.

    ``reward`` is the terminal outcome reward in ``[0.0, 1.0]``; ``success`` is
    the boolean interpretation; ``source`` records which path produced the
    result (``"objective"``, ``"llm_judge"``, or ``"default"``). ``rationale``
    is a human-readable explanation and ``per_step_signals`` carries optional
    per-step process signals for DAG reward backup (Phase 3).
    """

    success: bool
    reward: float
    source: str
    rationale: str = ""
    per_step_signals: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    """Async verifier seam: maps a run dict to a :class:`VerifierResult`."""

    async def verify(self, run: dict[str, Any]) -> VerifierResult: ...


ObjectiveCheck = Callable[[dict[str, Any]], bool]


class ObjectiveVerifier:
    """Runs a deterministic objective check when one is available.

    The check callable receives the run dict and returns ``True``/``False``.
    Used when task success is decidable by running tests, checking build
    status, etc. -- no LLM call needed. A check that raises is treated as
    failure (reward 0.0) rather than propagating, so a flaky check never
    crashes finalization.
    """

    def __init__(self, *, check: ObjectiveCheck) -> None:
        self._check = check

    async def verify(self, run: dict[str, Any]) -> VerifierResult:
        try:
            ok = bool(self._check(run))
        except Exception as exc:
            logger.warning(
                "objective check raised; treating as failure (task_id=%s): %s",
                run.get("task_id"),
                exc,
            )
            return VerifierResult(
                success=False,
                reward=0.0,
                source="objective",
                rationale=f"check raised: {exc}",
            )
        return VerifierResult(
            success=ok,
            reward=1.0 if ok else 0.0,
            source="objective",
            rationale="objective check" if ok else "objective check failed",
        )

"""Verifier-driven finalizer + trajectory harvest (Phase 2, Task 4).

Replaces the constant ``set_reward(1.0)``. At task finalize:

  1. run the :class:`AgenticVerifier` -- the pi agent reviews the collaboration
     and assigns a reward per RL ``session_id``;
  2. write each session's reward authoritatively via the bridge -- this
     enforces the **reward-before-export** ordering and lands the neutral
     fallback when the agent failed;
  3. harvest each session's reward-stamped trajectory via
     ``/export_trajectories`` (terminal -- it revokes the session).

The pi agent also calls ``/rl/set_reward`` itself (via the ``verifier-rl``
extension); the authoritative write here is idempotent w.r.t. that and
guarantees the ordering on the Python side regardless of the agent path.

This module supersedes ``rl_session.RLSessionRewardWriter`` /
``integration.finalize_with_verifier`` (the older single-verifier path), which
remain for backward compatibility.

Torch-free: stdlib :mod:`logging` only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from customized_areal.tree_search.dag.agentic_verifier import (
    AgenticVerifier,
    VerifierRun,
)

logger = logging.getLogger("VerifierFinalizer")


class RewardWriter(Protocol):
    """Writes a reward to one RL session (db_bridge ``set_reward`` seam)."""

    async def set_reward(self, *, session_id: str, reward: float) -> None: ...


class TrajectoryHarvester(Protocol):
    """Exports (and revokes) one session's reward-stamped trajectory."""

    async def export(self, *, session_id: str) -> None: ...


@dataclass
class FinalizeResult:
    """Outcome of one task finalize: the verifier run + harvested sessions."""

    run: VerifierRun
    exported: list[str] = field(default_factory=list)


class VerifierFinalizer:
    """Runs the verifier, writes rewards, then harvests trajectories.

    ``reward_writer`` and ``harvester`` are separate seams but a single
    db_bridge client typically implements both.
    """

    def __init__(
        self,
        *,
        verifier: AgenticVerifier,
        reward_writer: RewardWriter,
        harvester: TrajectoryHarvester,
    ) -> None:
        self._verifier = verifier
        self._reward_writer = reward_writer
        self._harvester = harvester

    async def finalize(
        self,
        *,
        task_id: str,
        transcripts: dict[str, str],
        session_map: dict[str, str | None],
        acceptance_criteria: list[str],
    ) -> FinalizeResult:
        run = await self._verifier.verify_task(
            task_id=task_id,
            transcripts=transcripts,
            session_map=session_map,
            acceptance_criteria=acceptance_criteria,
        )
        reward_map = run.reward_map()

        # Step 1: write every reward BEFORE any export (export is terminal).
        for session_id, reward in reward_map.items():
            await self._reward_writer.set_reward(session_id=session_id, reward=reward)

        # Step 2: harvest each session exactly once (idempotent).
        exported: list[str] = []
        for session_id in reward_map:
            if session_id in exported:
                continue
            try:
                await self._harvester.export(session_id=session_id)
                exported.append(session_id)
            except Exception:
                logger.warning(
                    "export_trajectories failed (task_id=%s session_id=%s)",
                    task_id,
                    session_id,
                    exc_info=True,
                )
        logger.info(
            "finalize complete (task_id=%s ok=%s rewards=%d exported=%d)",
            task_id,
            run.ok,
            len(reward_map),
            len(exported),
        )
        return FinalizeResult(run=run, exported=exported)


__all__ = [
    "FinalizeResult",
    "RewardWriter",
    "TrajectoryHarvester",
    "VerifierFinalizer",
]

"""RL session reward writer -- replaces the constant ``set_reward(1.0)``.

At run finalization the verifier result is written to the RL session via the
bridge's ``set_reward``, then ``end_session`` is called. If ``set_reward``
fails, the session is left open (``end_session`` is NOT called) so the
trajectory is not lost -- the caller decides whether to retry. See design §5
Phase 4 Task 11/12.

Uses stdlib :mod:`logging` to keep the ``dag`` package importable without the
heavy training stack.
"""

from __future__ import annotations

import logging
from typing import Protocol

from customized_areal.tree_search.dag.verifier import VerifierResult

logger = logging.getLogger("RLSessionRewardWriter")


class RLBridgeClient(Protocol):
    """Subset of the db_bridge client used by :class:`RLSessionRewardWriter`."""

    async def set_reward(self, *, session_id: str, reward: float) -> None: ...
    async def end_session(self, *, session_id: str) -> None: ...


class RLSessionRewardWriter:
    """Writes the verifier-driven reward to the RL session at finalization.

    Replaces the constant ``set_reward(1.0)`` documented in
    ``db_bridge/README.md``. The actual call site lives in le-agent (out of
    this repo); this writer is what the le-agent finalizer calls into.
    """

    def __init__(self, *, bridge_client: RLBridgeClient) -> None:
        self._bridge = bridge_client

    async def finalize(
        self,
        *,
        session_id: str,
        verifier_result: VerifierResult,
    ) -> None:
        """Set reward from the verifier result, then end the session.

        Raises if ``set_reward`` fails -- ``end_session`` is NOT called in that
        case (the session stays open so the caller can retry).
        """
        try:
            await self._bridge.set_reward(
                session_id=session_id,
                reward=float(verifier_result.reward),
            )
        except Exception as exc:
            logger.error(
                "set_reward failed; leaving session open (session_id=%s): %s",
                session_id,
                exc,
            )
            raise
        await self._bridge.end_session(session_id=session_id)
        logger.info(
            "RL session finalized (session_id=%s reward=%s source=%s)",
            session_id,
            verifier_result.reward,
            verifier_result.source,
        )

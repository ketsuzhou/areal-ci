"""Tests for RLSessionRewardWriter -- verifier-driven reward to the RL session."""

from __future__ import annotations

from typing import Any

import pytest

from customized_areal.tree_search.agents.rl_session import RLSessionRewardWriter
from customized_areal.tree_search.agents.verifier import VerifierResult


class _FakeBridgeClient:
    """Fake client recording set_reward / end_session calls in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def set_reward(self, *, session_id: str, reward: float) -> None:
        self.events.append(("set_reward", {"session_id": session_id, "reward": reward}))

    async def end_session(self, *, session_id: str) -> None:
        self.events.append(("end_session", {"session_id": session_id}))


@pytest.mark.asyncio
async def test_writer_sets_verifier_reward_before_end() -> None:
    bridge = _FakeBridgeClient()
    writer = RLSessionRewardWriter(bridge_client=bridge)

    result = VerifierResult(success=True, reward=0.75, source="llm_judge")
    await writer.finalize(session_id="sess-1", verifier_result=result)

    assert bridge.events == [
        ("set_reward", {"session_id": "sess-1", "reward": 0.75}),
        ("end_session", {"session_id": "sess-1"}),
    ]


@pytest.mark.asyncio
async def test_writer_skips_end_on_set_reward_failure() -> None:
    class _FailingBridge(_FakeBridgeClient):
        async def set_reward(self, *, session_id: str, reward: float) -> None:
            raise RuntimeError("gateway down")

    bridge = _FailingBridge()
    writer = RLSessionRewardWriter(bridge_client=bridge)
    result = VerifierResult(success=True, reward=1.0, source="objective")

    with pytest.raises(RuntimeError, match="gateway down"):
        await writer.finalize(session_id="sess-1", verifier_result=result)

    # end_session was NOT called -- the session stays open for retry.
    assert all(name != "end_session" for name, _ in bridge.events)

"""Tests for the retained workflow-integration helper ``finalize_with_verifier``.

The cloud-branch helpers (``materialize_cloud_branch`` / ``cleanup_cloud_branch``)
were removed with the branch transport; branching is now driven through the
env-dispatch primitive. Only verifier finalization remains here.
"""

from __future__ import annotations

from typing import Any

import pytest

from customized_areal.tree_search.agents.integration import (
    finalize_with_verifier,
)
from customized_areal.tree_search.agents.rl_session import RLSessionRewardWriter
from customized_areal.tree_search.agents.verifier import ObjectiveVerifier


class _RecordingBridge:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def set_reward(self, *, session_id: str, reward: float) -> None:
        self.events.append(("set_reward", {"session_id": session_id, "reward": reward}))

    async def end_session(self, *, session_id: str) -> None:
        self.events.append(("end_session", {"session_id": session_id}))


@pytest.mark.asyncio
async def test_finalize_with_verifier_writes_verifier_reward() -> None:
    bridge = _RecordingBridge()
    writer = RLSessionRewardWriter(bridge_client=bridge)
    verifier = ObjectiveVerifier(check=lambda run: True)

    result = await finalize_with_verifier(
        verifier=verifier,
        writer=writer,
        session_id="sess-1",
        run={"task_id": "t1", "check_output": "ok"},
    )

    assert result.reward == 1.0  # ObjectiveVerifier returns 1.0 on success
    assert bridge.events == [
        ("set_reward", {"session_id": "sess-1", "reward": 1.0}),
        ("end_session", {"session_id": "sess-1"}),
    ]

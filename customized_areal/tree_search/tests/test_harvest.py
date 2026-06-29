"""Tests for the verifier-driven finalizer + trajectory harvest (Phase 2, Task 4).

At task finalize the new path:
  1. runs the :class:`AgenticVerifier` (the pi agent assigns per-session reward),
  2. writes the per-session reward authoritatively via the bridge (enforces the
     reward-before-export ordering, and lands the neutral fallback on error),
  3. harvests each session's reward-stamped trajectory via /export_trajectories.

This replaces the constant ``set_reward(1.0)``. Export is terminal, so it must
never run before the reward is set.

Torch-free.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from customized_areal.tree_search.agents.agentic_verifier import AgenticVerifier
from customized_areal.tree_search.agents.harvest import (
    FinalizeResult,
    VerifierFinalizer,
)


class _FakeLauncher:
    def __init__(self, *, output: str | None = None, raise_exc: Exception | None = None):
        self._output = output
        self._raise = raise_exc

    async def run(self, *, prompt: str, judge_model: str) -> str:
        if self._raise is not None:
            raise self._raise
        assert self._output is not None
        return self._output


class _RecordingBridge:
    """Records set_reward / export calls in global order (reward-before-export)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def set_reward(self, *, session_id: str, reward: float) -> None:
        self.events.append(("set_reward", {"session_id": session_id, "reward": reward}))

    async def export(self, *, session_id: str) -> None:
        self.events.append(("export", {"session_id": session_id}))


def _canned(rewards: list[dict]) -> str:
    return "ok\n```json\n" + json.dumps({"rewards": rewards}) + "\n```"


_SESSION_MAP = {"A": "sess-A", "B": "sess-B"}
_TRANSCRIPTS = {"A": "a", "B": "b"}
_CRITERIA = ["done"]


@pytest.mark.asyncio
async def test_finalize_sets_verifier_rewards_not_constant_one() -> None:
    launcher = _FakeLauncher(
        output=_canned(
            [
                {"session_id": "sess-A", "reward": 0.8, "rationale": "x"},
                {"session_id": "sess-B", "reward": 0.2, "rationale": "y"},
            ]
        )
    )
    bridge = _RecordingBridge()
    finalizer = VerifierFinalizer(
        verifier=AgenticVerifier(launcher=launcher, judge_model="j"),
        reward_writer=bridge,
        harvester=bridge,
    )
    result = await finalizer.finalize(
        task_id="t1",
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    assert isinstance(result, FinalizeResult)
    rewards = {
        e[1]["session_id"]: e[1]["reward"]
        for e in bridge.events
        if e[0] == "set_reward"
    }
    assert rewards == {"sess-A": 0.8, "sess-B": 0.2}
    assert 1.0 not in rewards.values()
    assert set(result.exported) == {"sess-A", "sess-B"}


@pytest.mark.asyncio
async def test_finalize_sets_reward_before_export_for_each_session() -> None:
    launcher = _FakeLauncher(
        output=_canned([{"session_id": "sess-A", "reward": 0.5, "rationale": "x"}])
    )
    bridge = _RecordingBridge()
    finalizer = VerifierFinalizer(
        verifier=AgenticVerifier(launcher=launcher, judge_model="j"),
        reward_writer=bridge,
        harvester=bridge,
    )
    await finalizer.finalize(
        task_id="t1",
        transcripts={"A": "a"},
        session_map={"A": "sess-A"},
        acceptance_criteria=_CRITERIA,
    )
    # All set_reward events precede any export event (reward-before-export).
    first_export = next(i for i, e in enumerate(bridge.events) if e[0] == "export")
    assert all(e[0] == "set_reward" for e in bridge.events[:first_export])


@pytest.mark.asyncio
async def test_finalize_error_path_writes_neutral_then_exports() -> None:
    launcher = _FakeLauncher(raise_exc=RuntimeError("boom"))
    bridge = _RecordingBridge()
    finalizer = VerifierFinalizer(
        verifier=AgenticVerifier(launcher=launcher, judge_model="j", neutral_reward=0.0),
        reward_writer=bridge,
        harvester=bridge,
    )
    result = await finalizer.finalize(
        task_id="t1",
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    assert result.run.ok is False
    rewards = {
        e[1]["session_id"]: e[1]["reward"]
        for e in bridge.events
        if e[0] == "set_reward"
    }
    assert rewards == {"sess-A": 0.0, "sess-B": 0.0}
    assert set(result.exported) == {"sess-A", "sess-B"}


@pytest.mark.asyncio
async def test_finalize_harvest_is_idempotent_per_session() -> None:
    launcher = _FakeLauncher(
        output=_canned(
            [
                {"session_id": "sess-A", "reward": 1.0, "rationale": "x"},
                {"session_id": "sess-A", "reward": 1.0, "rationale": "dup"},
            ]
        )
    )
    bridge = _RecordingBridge()
    finalizer = VerifierFinalizer(
        verifier=AgenticVerifier(launcher=launcher, judge_model="j"),
        reward_writer=bridge,
        harvester=bridge,
    )
    await finalizer.finalize(
        task_id="t1",
        transcripts={"A": "a"},
        session_map={"A": "sess-A"},
        acceptance_criteria=_CRITERIA,
    )
    exports = [e for e in bridge.events if e[0] == "export"]
    assert len(exports) == 1  # exported once despite duplicate verdict

"""Tests for the AgenticVerifier driver (Phase 2, Task 3).

The driver launches a pi verifier agent on a fixed judge model, feeds it the
per-agent transcripts + session map + acceptance criteria, and collects the
per-session rewards the agent assigned. On any launch/parse error it falls back
to a safe neutral reward for every session instead of crashing finalization.

Torch-free -- runs without the training stack.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.agentic_verifier import (
    AgenticVerifier,
    VerifierRun,
    parse_verifier_output,
)


class _FakeLauncher:
    """Records the prompt it was given and returns a canned agent output."""

    def __init__(
        self, *, output: str | None = None, raise_exc: Exception | None = None
    ):
        self._output = output
        self._raise = raise_exc
        self.calls: list[dict] = []

    async def run(self, *, prompt: str, judge_model: str) -> str:
        self.calls.append({"prompt": prompt, "judge_model": judge_model})
        if self._raise is not None:
            raise self._raise
        assert self._output is not None
        return self._output


_TASK = "task-1"
_SESSION_MAP = {"A": "sess-A", "B": "sess-B"}
_TRANSCRIPTS = {"A": "planner did X", "B": "worker did Y"}
_CRITERIA = ["the bug is fixed", "tests pass"]


def _canned_output(rewards: list[dict]) -> str:
    import json

    return (
        "I reviewed the collaboration.\n\n```json\n"
        + json.dumps({"rewards": rewards})
        + "\n```\n"
    )


def test_parse_verifier_output_extracts_json_block() -> None:
    out = _canned_output(
        [{"session_id": "sess-A", "reward": 1.0, "rationale": "great"}]
    )
    rewards = parse_verifier_output(out)
    assert rewards == [("sess-A", 1.0, "great")]


def test_parse_verifier_output_raises_on_missing_block() -> None:
    with pytest.raises(ValueError):
        parse_verifier_output("no json here")


@pytest.mark.asyncio
async def test_verify_task_collects_per_session_rewards() -> None:
    launcher = _FakeLauncher(
        output=_canned_output(
            [
                {"session_id": "sess-A", "reward": 1.0, "rationale": "did the work"},
                {"session_id": "sess-B", "reward": 0.0, "rationale": "no progress"},
            ]
        )
    )
    verifier = AgenticVerifier(launcher=launcher, judge_model="judge/fixed-v1")
    run = await verifier.verify_task(
        task_id=_TASK,
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    assert isinstance(run, VerifierRun)
    assert run.ok is True
    assert run.reward_map() == {"sess-A": 1.0, "sess-B": 0.0}
    # The judge model was passed through to the launcher.
    assert launcher.calls[0]["judge_model"] == "judge/fixed-v1"
    # Acceptance criteria + transcripts are present in the prompt.
    assert "tests pass" in launcher.calls[0]["prompt"]
    assert "worker did Y" in launcher.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_verify_task_clamps_rewards_to_unit_interval() -> None:
    launcher = _FakeLauncher(
        output=_canned_output(
            [
                {"session_id": "sess-A", "reward": 5.0, "rationale": "over"},
                {"session_id": "sess-B", "reward": -3.0, "rationale": "under"},
            ]
        )
    )
    verifier = AgenticVerifier(launcher=launcher, judge_model="j")
    run = await verifier.verify_task(
        task_id=_TASK,
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    assert run.reward_map() == {"sess-A": 1.0, "sess-B": 0.0}


@pytest.mark.asyncio
async def test_verify_task_ignores_unknown_session_ids() -> None:
    launcher = _FakeLauncher(
        output=_canned_output(
            [{"session_id": "sess-GHOST", "reward": 1.0, "rationale": "?"}]
        )
    )
    verifier = AgenticVerifier(launcher=launcher, judge_model="j")
    run = await verifier.verify_task(
        task_id=_TASK,
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    # Ghost session is not in the session map -> dropped.
    assert run.reward_map() == {}


@pytest.mark.asyncio
async def test_verify_task_neutral_fallback_on_launcher_error() -> None:
    launcher = _FakeLauncher(raise_exc=RuntimeError("agent crashed"))
    verifier = AgenticVerifier(launcher=launcher, judge_model="j", neutral_reward=0.0)
    run = await verifier.verify_task(
        task_id=_TASK,
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    assert run.ok is False
    assert run.error is not None and "agent crashed" in run.error
    # Neutral reward applied to every (non-None) session, no crash.
    assert run.reward_map() == {"sess-A": 0.0, "sess-B": 0.0}


@pytest.mark.asyncio
async def test_verify_task_neutral_fallback_on_parse_error() -> None:
    launcher = _FakeLauncher(output="garbled, no json block")
    verifier = AgenticVerifier(launcher=launcher, judge_model="j", neutral_reward=0.0)
    run = await verifier.verify_task(
        task_id=_TASK,
        transcripts=_TRANSCRIPTS,
        session_map=_SESSION_MAP,
        acceptance_criteria=_CRITERIA,
    )
    assert run.ok is False
    assert run.reward_map() == {"sess-A": 0.0, "sess-B": 0.0}

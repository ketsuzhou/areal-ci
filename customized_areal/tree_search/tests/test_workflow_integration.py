"""Tests for the adapted workflow-integration helpers (Tasks 12/15).

These cover the module-level helpers in ``dag.integration`` that replace the
plan's fictional ``TreeSearchGroupedWorkflow`` methods. The candidate is a tiny
``SimpleNamespace`` standing in for a tree-store ``Node`` (so these tests need
neither torch nor the heavy workflow module).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from customized_areal.tree_search.agents.integration import (
    cleanup_cloud_branch,
    finalize_with_verifier,
    materialize_cloud_branch,
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


@pytest.mark.asyncio
async def test_materialize_cloud_branch_passes_turn_idx_as_seq() -> None:
    calls: list[dict[str, Any]] = []

    class _FakeMaterializer:
        async def materialize(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(branch_run_id="branch-1")

    candidate = SimpleNamespace(
        task_id="t1",
        turn_idx=2,
        branch_sandbox_id=None,
        branch_issue_id=None,
        branch_env_snapshot_id="snap-9",
    )

    branch_run_id = await materialize_cloud_branch(
        materializer=_FakeMaterializer(),  # type: ignore[arg-type]
        candidate=candidate,
        source_sandbox_id="sbx-1",
        source_issue_id="issue-1",
        replay_messages=[{"role": "user", "content": "hi"}],
    )

    assert branch_run_id == "branch-1"
    assert calls[0]["source_sandbox_id"] == "sbx-1"
    assert calls[0]["source_issue_id"] == "issue-1"
    assert calls[0]["task_id"] == "t1"
    assert calls[0]["seq"] == 2  # turn_idx -> seq


@pytest.mark.asyncio
async def test_cleanup_cloud_branch_deletes_sandbox_and_issue() -> None:
    cleaned: list[str] = []
    deleted: list[str] = []

    class _FakeEnv:
        async def cleanup(self, sandbox_id: str) -> None:
            cleaned.append(sandbox_id)

    class _FakeForker:
        async def delete_fork(self, *, issue_id: str) -> None:
            deleted.append(issue_id)

    candidate = SimpleNamespace(
        task_id="t1",
        turn_idx=2,
        branch_sandbox_id="forked-sbx-1",
        branch_issue_id="forked-issue-1",
        branch_env_snapshot_id="snap-9",
    )

    await cleanup_cloud_branch(
        env=_FakeEnv(),  # type: ignore[arg-type]
        forker=_FakeForker(),  # type: ignore[arg-type]
        candidate=candidate,
    )

    assert cleaned == ["forked-sbx-1"]
    assert deleted == ["forked-issue-1"]


@pytest.mark.asyncio
async def test_cleanup_cloud_branch_is_best_effort_on_errors() -> None:
    class _FailingEnv:
        async def cleanup(self, sandbox_id: str) -> None:
            raise RuntimeError("sandbox gone")

    class _FailingForker:
        async def delete_fork(self, *, issue_id: str) -> None:
            raise RuntimeError("issue gone")

    candidate = SimpleNamespace(
        task_id="t1",
        turn_idx=2,
        branch_sandbox_id="forked-sbx-1",
        branch_issue_id="forked-issue-1",
        branch_env_snapshot_id="snap-9",
    )

    # Must not raise -- cleanup is best-effort.
    await cleanup_cloud_branch(
        env=_FailingEnv(),  # type: ignore[arg-type]
        forker=_FailingForker(),  # type: ignore[arg-type]
        candidate=candidate,
    )

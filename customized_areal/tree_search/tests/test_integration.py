"""Tests for MulticaIssueForker + BranchMaterializer.

``httpx.MockTransport`` takes a single dispatching handler (it has no
``add_route``), so each test routes on ``request.method`` / ``request.url.path``
inside one handler -- mirroring ``test_environment.py``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from customized_areal.tree_search.dag.environment import ForkResult, SnapshotResult
from customized_areal.tree_search.dag.integration import (
    BranchMaterializer,
    MulticaIssueForker,
)

ISSUE = "00000000-0000-0000-0000-000000000001"
TASK = "00000000-0000-0000-0000-000000000002"


# --------------------------------------------------------------------------- #
# MulticaIssueForker
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multica_issue_forker_posts_to_fork_endpoint() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(201, json={"forked_issue_id": "issue-forked"})

    forker = MulticaIssueForker(
        base_url="http://multica.test",
        transport=httpx.MockTransport(handler),
        api_key="key-1",
    )
    forked_id = await forker.fork(issue_id=ISSUE, task_id=TASK, seq=5)

    assert forked_id == "issue-forked"
    assert seen["method"] == "POST"
    assert seen["path"] == f"/api/issues/{ISSUE}/fork"
    assert seen["params"] == {"task_id": TASK, "seq": "5"}


@pytest.mark.asyncio
async def test_multica_issue_forker_raises_on_non_201() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    forker = MulticaIssueForker(
        base_url="http://multica.test", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RuntimeError, match="status=500"):
        await forker.fork(issue_id=ISSUE, task_id=TASK, seq=1)


@pytest.mark.asyncio
async def test_multica_issue_forker_delete_calls_endpoint() -> None:
    called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(f"{request.method} {request.url.path}")
        return httpx.Response(204)

    forker = MulticaIssueForker(
        base_url="http://multica.test", transport=httpx.MockTransport(handler)
    )
    await forker.delete_fork(issue_id=ISSUE)
    assert called == [f"DELETE /api/issues/{ISSUE}/fork"]


@pytest.mark.asyncio
async def test_multica_issue_forker_delete_treats_404_as_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    forker = MulticaIssueForker(
        base_url="http://multica.test", transport=httpx.MockTransport(handler)
    )
    # Should not raise.
    await forker.delete_fork(issue_id=ISSUE)


def test_multica_issue_forker_requires_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MULTICA_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="MULTICA_BASE_URL"):
        MulticaIssueForker()


# --------------------------------------------------------------------------- #
# BranchMaterializer
# --------------------------------------------------------------------------- #
class _FakeEnv:
    """Fake ForkableEnvironment -- records calls, returns canned ids."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cleanup_calls: list[str] = []

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        self.calls.append(f"snapshot:{sandbox_id}")
        return SnapshotResult(snapshot_id="snap-1", source_sandbox_id=sandbox_id)

    async def fork(
        self, *, source_sandbox_id: str | None = None, snapshot_id: str | None = None
    ) -> ForkResult:
        self.calls.append(f"fork:src={source_sandbox_id},snap={snapshot_id}")
        return ForkResult(sandbox_id="forked-sbx-1")

    async def restore(self, sandbox_id: str) -> None:
        self.calls.append(f"restore:{sandbox_id}")

    async def cleanup(self, sandbox_id: str) -> None:
        self.calls.append(f"cleanup:{sandbox_id}")
        self.cleanup_calls.append(sandbox_id)


class _FakeStarter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_branch(
        self,
        *,
        forked_sandbox_id: str,
        forked_issue_id: str,
        replay_messages: list[dict[str, Any]],
        drop_prior_session_id: bool,
    ) -> str:
        self.calls.append(
            {
                "forked_sandbox_id": forked_sandbox_id,
                "forked_issue_id": forked_issue_id,
                "replay_messages": replay_messages,
                "drop_prior_session_id": drop_prior_session_id,
            }
        )
        return "branch-run-1"


def _forker_with_fake_fork() -> MulticaIssueForker:
    """A forker whose HTTP fork is stubbed (no transport needed)."""
    forker = MulticaIssueForker(
        base_url="http://m.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(204)),
    )

    async def fake_fork(*, issue_id: str, task_id: str, seq: int) -> str:
        return "issue-forked"

    forker.fork = fake_fork  # type: ignore[method-assign]
    return forker


@pytest.mark.asyncio
async def test_branch_materializer_orchestrates_steps_in_order() -> None:
    env = _FakeEnv()
    forker = _forker_with_fake_fork()
    starter = _FakeStarter()

    materializer = BranchMaterializer(env=env, forker=forker, starter=starter)
    result = await materializer.materialize(
        source_sandbox_id="sbx-1",
        source_issue_id="issue-1",
        task_id="task-1",
        seq=5,
        replay_messages=[{"role": "user", "content": "hi"}],
    )

    assert env.calls == ["snapshot:sbx-1", "fork:src=None,snap=snap-1"]
    assert starter.calls == [
        {
            "forked_sandbox_id": "forked-sbx-1",
            "forked_issue_id": "issue-forked",
            "replay_messages": [{"role": "user", "content": "hi"}],
            "drop_prior_session_id": True,
        }
    ]
    assert result.branch_run_id == "branch-run-1"
    assert result.forked_sandbox_id == "forked-sbx-1"
    assert result.forked_issue_id == "issue-forked"
    assert result.snapshot_id == "snap-1"


@pytest.mark.asyncio
async def test_branch_materializer_rolls_back_on_start_branch_failure() -> None:
    env = _FakeEnv()
    forker = _forker_with_fake_fork()
    delete_calls: list[str] = []

    async def spy_delete(*, issue_id: str) -> None:
        delete_calls.append(issue_id)

    forker.delete_fork = spy_delete  # type: ignore[method-assign]

    class _FailingStarter(_FakeStarter):
        async def start_branch(self, **kwargs: Any) -> str:
            raise RuntimeError("branch start failed")

    materializer = BranchMaterializer(env=env, forker=forker, starter=_FailingStarter())
    with pytest.raises(RuntimeError, match="branch start failed"):
        await materializer.materialize(
            source_sandbox_id="sbx-1",
            source_issue_id="issue-1",
            task_id="task-1",
            seq=5,
            replay_messages=[],
        )

    # Rollback: forked sandbox cleaned up AND forked issue deleted.
    assert "forked-sbx-1" in env.cleanup_calls
    assert delete_calls == ["issue-forked"]


@pytest.mark.asyncio
async def test_branch_materializer_rolls_back_sandbox_on_issue_fork_failure() -> None:
    env = _FakeEnv()
    forker = _forker_with_fake_fork()

    async def failing_fork(*, issue_id: str, task_id: str, seq: int) -> str:
        raise RuntimeError("issue fork failed")

    forker.fork = failing_fork  # type: ignore[method-assign]

    materializer = BranchMaterializer(env=env, forker=forker, starter=_FakeStarter())
    with pytest.raises(RuntimeError, match="issue fork failed"):
        await materializer.materialize(
            source_sandbox_id="sbx-1",
            source_issue_id="issue-1",
            task_id="task-1",
            seq=5,
            replay_messages=[],
        )
    # The forked sandbox is cleaned up; no start_branch happened.
    assert "forked-sbx-1" in env.cleanup_calls

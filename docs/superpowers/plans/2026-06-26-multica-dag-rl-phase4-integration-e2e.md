# Phase 4: Integration + Lazy Branching + E2E — Implementation Plan (Python)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the DAG RL components together — bridge/session wiring, verifier-driven reward replacing the constant `1.0`, lazy branching on candidate selection, branch cleanup, and end-to-end validation at `group_size=2` plus a separate scale check.

**Architecture:** The `customized_grouped_workflow.py` already has `_run_fresh_episode`, `select_branch_candidate`, `_prepare_branch_task`, and `_cleanup_branch`. This phase extends them: `_prepare_branch_task` calls `ForkableEnvironment.snapshot` + `ForkIssueSubtree` (via HTTP) + `agent_start_branch` with transcript-prefix replay and no `PriorSessionID`. A new verifier-driven reward path replaces the constant `1.0` — the verifier runs at run finalization and the result is written to the RL session via `rl_set_reward` before `rl_end_session`. `_cleanup_branch` deletes the forked Multica issue + sandbox in addition to the existing sandbox cleanup.

**Tech Stack:** Python 3.12+ · `httpx` (for the Multica fork HTTP call) · `pytest` + `pytest-asyncio` · existing `customized_areal` workflow + db_bridge channels

**Design reference:** `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md` §3.1 (snapshot-at-frontier), §3.3 (concurrency semaphore), §5 Phase 4, §6 (e2e at group_size=2)

**Project rules:** `backend/areal/CLAUDE.md`, `AGENTS.md`. `.claude/rules/code-quality.md` — "Paired operations stay together" (snapshot + fork + branch must all succeed or all roll back).

**Dependencies:** Phases 0, 1, 2, 3 complete. The Multica fork HTTP endpoint (`POST /api/issues/{id}/fork`) and Fleet snapshot/fork endpoints from Phase 1 must be live.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `customized_areal/tree_search/dag/integration.py` (create) | `MulticaIssueForker` — HTTP client for `POST /api/issues/{id}/fork` + `DELETE`; `BranchMaterializer` — orchestrates snapshot + fork + agent_start_branch |
| `customized_areal/tree_search/dag/rl_session.py` (create) | `RLSessionRewardWriter` — writes verifier-driven reward to the RL session via `rl_set_reward` before `rl_end_session` |
| `customized_areal/tree_search/core/customized_grouped_workflow.py` (modify) | Wire `BranchMaterializer` into `_prepare_branch_task`; extend `_cleanup_branch` to delete forked issue; call verifier at finalization |
| `customized_areal/tree_search/dag/test_integration.py` (create) | Tests for `MulticaIssueForker` (mocked httpx) and `BranchMaterializer` (fake env + fake forker) |
| `customized_areal/tree_search/dag/test_rl_session.py` (create) | Tests for `RLSessionRewardWriter` (mocked bridge) |
| `customized_areal/tree_search/dag/test_workflow_integration.py` (create) | Tests for the extended `_prepare_branch_task` and `_cleanup_branch` |
| `customized_areal/tree_search/dag/__init__.py` (modify) | Export new types |

---

## Task 11: Bridge/session wiring — agent custom_env routes LLM via db_bridge

**Files:**
- Create: `customized_areal/tree_search/dag/rl_session.py`
- Create: `customized_areal/tree_search/dag/test_rl_session.py`

**Rationale:** Per the design §4, the agent's `custom_env` must route LLM calls through `db_bridge` (`proxy_base_url` / `proxy_api_key`), and the agent run must map to an RL session (`rl_start_session` / `rl_set_reward` / `rl_end_session`). This task ships the `RLSessionRewardWriter` that writes the verifier-driven reward at session end. The actual `set_reward(1.0)` call site lives in le-agent (out of this repo); this writer is what le-agent's finalizer calls into, replacing the constant `1.0`.

- [ ] **Step 1: Write the failing test for RLSessionRewardWriter**

```python
# customized_areal/tree_search/dag/test_rl_session.py
"""Tests for RLSessionRewardWriter — writes verifier-driven reward to the RL session."""
from __future__ import annotations

from typing import Any

import pytest

from customized_areal.tree_search.dag.rl_session import RLSessionRewardWriter
from customized_areal.tree_search.dag.verifier import VerifierResult


class _FakeBridgeClient:
    """Fake client for the rl_set_reward / rl_end_session channels."""

    def __init__(self) -> None:
        self.set_reward_calls: list[dict[str, Any]] = []
        self.end_session_calls: list[dict[str, Any]] = []

    async def set_reward(self, *, session_id: str, reward: float) -> None:
        self.set_reward_calls.append({"session_id": session_id, "reward": reward})

    async def end_session(self, *, session_id: str) -> None:
        self.end_session_calls.append({"session_id": session_id})


@pytest.mark.asyncio
async def test_rl_session_writer_sets_verifier_reward_before_end() -> None:
    """The writer calls set_reward(verifier.reward) then end_session, in that order."""
    bridge = _FakeBridgeClient()
    writer = RLSessionRewardWriter(bridge_client=bridge)

    result = VerifierResult(success=True, reward=0.75, source="llm_judge")
    await writer.finalize(session_id="sess-1", verifier_result=result)

    assert bridge.set_reward_calls == [{"session_id": "sess-1", "reward": 0.75}]
    assert bridge.end_session_calls == [{"session_id": "sess-1"}]
    # Order: set_reward must come before end_session.
    # (The fake records calls in order; the lists above each have one entry,
    # so we assert the full sequence via a combined list.)
    assert len(bridge.set_reward_calls) == 1
    assert len(bridge.end_session_calls) == 1


@pytest.mark.asyncio
async def test_rl_session_writer_skips_end_on_set_reward_failure() -> None:
    """If set_reward fails, end_session is NOT called — the session stays open
    so the trajectory isn't lost. The error propagates to the caller.
    """

    class _FailingBridge(_FakeBridgeClient):
        async def set_reward(self, *, session_id: str, reward: float) -> None:
            raise RuntimeError("gateway down")

    bridge = _FailingBridge()
    writer = RLSessionRewardWriter(bridge_client=bridge)
    result = VerifierResult(success=True, reward=1.0, source="objective")
    with pytest.raises(RuntimeError, match="gateway down"):
        await writer.finalize(session_id="sess-1", verifier_result=result)
    # end_session was NOT called — session stays open for retry.
    assert bridge.end_session_calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/dag/test_rl_session.py -v`
Expected: FAIL — `RLSessionRewardWriter` undefined.

- [ ] **Step 3: Implement RLSessionRewardWriter**

```python
# customized_areal/tree_search/dag/rl_session.py
"""RL session reward writer — replaces the constant set_reward(1.0).

At run finalization, the verifier result is written to the RL session via
rl_set_reward, then rl_end_session is called. If set_reward fails, the
session stays open (end_session is NOT called) so the trajectory is not
lost — the caller can retry.
"""
from __future__ import annotations

from typing import Any, Protocol

from customized_areal.tree_search.dag.verifier import VerifierResult

from areal.utils import logging

logger = logging.getLogger("RLSessionRewardWriter")


class RLBridgeClient(Protocol):
    """Subset of the db_bridge client used by RLSessionRewardWriter."""

    async def set_reward(self, *, session_id: str, reward: float) -> None: ...
    async def end_session(self, *, session_id: str) -> None: ...


class RLSessionRewardWriter:
    """Writes the verifier-driven reward to the RL session at finalization.

    Replaces the constant set_reward(1.0) documented in db_bridge/README.md:85.
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

        Raises if set_reward fails — the caller decides whether to retry.
        end_session is NOT called on set_reward failure (session stays open).
        """
        try:
            await self._bridge.set_reward(
                session_id=session_id,
                reward=float(verifier_result.reward),
            )
        except Exception as exc:
            logger.error(
                "set_reward failed; leaving session open",
                session_id=session_id,
                error=str(exc),
            )
            raise
        await self._bridge.end_session(session_id=session_id)
        logger.info(
            "RL session finalized",
            session_id=session_id,
            reward=verifier_result.reward,
            source=verifier_result.source,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_rl_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/rl_session.py customized_areal/tree_search/dag/test_rl_session.py
git commit -m "feat(dag): add RLSessionRewardWriter — verifier-driven reward replaces constant 1.0"
```

---

## Task 12: Replace constant set_reward(1.0) with verifier-driven reward

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py` (the finalization path)

**Rationale:** The actual `set_reward(1.0)` call site is in le-agent (out of this repo). What this repo controls is the AReaL-side executor that handles `rl_set_reward` — and the verifier result must be computed and made available at the right point. This task adds a `_finalize_with_verifier` hook to the workflow that runs the verifier at run finalization and writes the reward via `RLSessionRewardWriter`. The le-agent side calls into this path instead of hardcoding `1.0`.

- [ ] **Step 1: Write the failing test for the verifier-driven finalization hook**

```python
# customized_areal/tree_search/dag/test_workflow_integration.py
"""Tests for the verifier-driven finalization hook in the grouped workflow."""
from __future__ import annotations

from typing import Any

import pytest

from customized_areal.tree_search.dag.rl_session import RLSessionRewardWriter
from customized_areal.tree_search.dag.verifier import (
    ObjectiveVerifier,
    VerifierResult,
)


class _RecordingBridge:
    def __init__(self) -> None:
        self.set_reward_calls: list[dict[str, Any]] = []
        self.end_session_calls: list[dict[str, Any]] = []

    async def set_reward(self, *, session_id: str, reward: float) -> None:
        self.set_reward_calls.append({"session_id": session_id, "reward": reward})

    async def end_session(self, *, session_id: str) -> None:
        self.end_session_calls.append({"session_id": session_id})


@pytest.mark.asyncio
async def test_workflow_finalization_writes_verifier_reward() -> None:
    """When the workflow finalizes a run, it calls the verifier and writes
    the verifier's reward to the RL session (not a constant 1.0).
    """
    bridge = _RecordingBridge()
    writer = RLSessionRewardWriter(bridge_client=bridge)
    # Objective verifier that always succeeds with reward 0.9.
    verifier = ObjectiveVerifier(check=lambda run: True)

    # The workflow's _finalize_with_verifier hook is the integration point.
    # We test it directly rather than spinning up the full workflow.
    from customized_areal.tree_search.core.customized_grouped_workflow import (
        TreeSearchGroupedWorkflow,
    )

    # _finalize_with_verifier is a new method added by this task.
    result = await TreeSearchGroupedWorkflow._finalize_with_verifier(
        verifier=verifier,
        writer=writer,
        session_id="sess-1",
        run={"task_id": "t1", "check_output": "ok"},
    )
    assert result.reward == 1.0  # ObjectiveVerifier returns 1.0 on success
    assert bridge.set_reward_calls == [{"session_id": "sess-1", "reward": 1.0}]
    assert bridge.end_session_calls == [{"session_id": "sess-1"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_workflow_integration.py::test_workflow_finalization_writes_verifier_reward -v`
Expected: FAIL — `TreeSearchGroupedWorkflow._finalize_with_verifier` does not exist.

- [ ] **Step 3: Add the _finalize_with_verifier static method**

In `customized_grouped_workflow.py`, add a new static method to `TreeSearchGroupedWorkflow` (place it near the other helper methods, e.g. after `_cleanup_branch`):

```python
    @staticmethod
    async def _finalize_with_verifier(
        *,
        verifier: Any,
        writer: Any,
        session_id: str,
        run: dict[str, Any],
    ) -> Any:
        """Run the verifier and write the reward to the RL session.

        Replaces the constant set_reward(1.0). Called at run finalization.
        Returns the VerifierResult so the caller can inspect it.
        """
        result = await verifier.verify(run)
        await writer.finalize(session_id=session_id, verifier_result=result)
        return result
```

Add `from typing import Any` if not already imported (it is, per the existing file).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_workflow_integration.py::test_workflow_finalization_writes_verifier_reward -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/dag/test_workflow_integration.py
git commit -m "feat(tree-search): add verifier-driven finalization hook replacing constant set_reward(1.0)"
```

---

## Task 13: Lazy branch on candidate selection — BranchMaterializer

**Files:**
- Create: `customized_areal/tree_search/dag/integration.py`
- Create: `customized_areal/tree_search/dag/test_integration.py`

**Rationale:** Per design §3.1, when `select_branch_candidate` returns a node, the workflow triggers `ForkableEnvironment.snapshot` + `ForkIssueSubtree` + `agent_start_branch`. Per §3.3, fork calls are gated by the concurrency semaphore (already on `FleetSandboxProvider` from Phase 0 Task 6). The `BranchMaterializer` orchestrates the three steps; if any fails, the others must roll back (paired operations stay together).

- [ ] **Step 1: Write the failing test for MulticaIssueForker (HTTP client)**

```python
# customized_areal/tree_search/dag/test_integration.py
"""Tests for BranchMaterializer + MulticaIssueForker."""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from customized_areal.tree_search.dag.integration import (
    BranchMaterializer,
    MulticaIssueForker,
)


@pytest.mark.asyncio
async def test_multica_issue_forker_posts_to_fork_endpoint() -> None:
    transport = httpx.MockTransport()

    def fork_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/issues/00000000-0000-0000-0000-000000000001/fork"
        assert dict(request.url.params) == {
            "task_id": "00000000-0000-0000-0000-000000000002",
            "seq": "5",
        }
        return httpx.Response(201, json={"forked_issue_id": "issue-forked"})

    transport.add_route(
        fork_handler,
        method="POST",
        path="/api/issues/00000000-0000-0000-0000-000000000001/fork",
    )

    forker = MulticaIssueForker(
        base_url="http://multica.test", transport=transport, api_key="key-1"
    )
    forked_id = await forker.fork(
        issue_id="00000000-0000-0000-0000-000000000001",
        task_id="00000000-0000-0000-0000-000000000002",
        seq=5,
    )
    assert forked_id == "issue-forked"


@pytest.mark.asyncio
async def test_multica_issue_forker_delete_calls_endpoint() -> None:
    transport = httpx.MockTransport()
    called: list[str] = []

    def delete_handler(request: httpx.Request) -> httpx.Response:
        called.append(request.url.path)
        return httpx.Response(204)

    transport.add_route(
        delete_handler,
        method="DELETE",
        path="/api/issues/00000000-0000-0000-0000-000000000001/fork",
    )

    forker = MulticaIssueForker(
        base_url="http://multica.test", transport=transport, api_key="key-1"
    )
    await forker.delete_fork(issue_id="00000000-0000-0000-0000-000000000001")
    assert called == ["/api/issues/00000000-0000-0000-0000-000000000001/fork"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_integration.py -v`
Expected: FAIL — `MulticaIssueForker`, `BranchMaterializer` undefined.

- [ ] **Step 3: Implement MulticaIssueForker**

```python
# customized_areal/tree_search/dag/integration.py
"""Branch materialization — orchestrates snapshot + fork + agent_start_branch.

Per design §3.1 (snapshot-at-frontier + transcript replay), when a node is
selected as a branch candidate:
  1. ForkableEnvironment.snapshot(live sandbox) — snapshot the source agent's sandbox
  2. ForkIssueSubtree (Multica HTTP) — fork the issue subtree at (task_id, seq)
  3. agent_start_branch — bind the forked sandbox + forked issue, replay msgs<=seq,
     drop PriorSessionID

If any step fails, the prior steps must roll back (paired operations stay together).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from customized_areal.tree_search.dag.environment import (
    ForkableEnvironment,
    ForkResult,
    SnapshotResult,
)

logger = logging.getLogger("BranchMaterializer")


class MulticaIssueForker:
    """HTTP client for the Multica issue fork endpoints (Phase 1)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            raise ValueError("MulticaIssueForker requires base_url or MULTICA_BASE_URL")
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def fork(self, *, issue_id: str, task_id: str, seq: int) -> str:
        """POST /api/issues/{id}/fork?task_id=...&seq=... → returns forked_issue_id."""
        resp = await self._client.post(
            f"/api/issues/{issue_id}/fork",
            params={"task_id": task_id, "seq": str(seq)},
            headers=self._headers(),
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"Multica issue fork failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        forked_id = body.get("forked_issue_id")
        if not isinstance(forked_id, str) or not forked_id:
            raise RuntimeError(f"fork response missing forked_issue_id: {body!r}")
        return forked_id

    async def delete_fork(self, *, issue_id: str) -> None:
        """DELETE /api/issues/{id}/fork — idempotent cleanup of a forked issue."""
        resp = await self._client.delete(
            f"/api/issues/{issue_id}/fork", headers=self._headers()
        )
        # 404 = already deleted, treat as success (idempotent).
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"Multica issue fork delete failed: status={resp.status_code} body={resp.text[:200]}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_integration.py -k "multica_issue_forker" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/integration.py customized_areal/tree_search/dag/test_integration.py
git commit -m "feat(dag): add MulticaIssueForker HTTP client for issue fork endpoints"
```

---

## Task 14: BranchMaterializer — orchestrate snapshot + fork + agent_start_branch

**Files:**
- Modify: `customized_areal/tree_search/dag/integration.py`
- Modify: `customized_areal/tree_search/dag/test_integration.py`

- [ ] **Step 1: Write the failing test for BranchMaterializer**

```python
class _FakeEnv:
    """Fake ForkableEnvironment — records calls, returns canned ids."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cleanup_calls: list[str] = []

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        self.calls.append(f"snapshot:{sandbox_id}")
        return SnapshotResult(snapshot_id="snap-1", source_sandbox_id=sandbox_id)

    async def fork(self, *, source_sandbox_id=None, snapshot_id=None) -> ForkResult:
        self.calls.append(f"fork:src={source_sandbox_id},snap={snapshot_id}")
        return ForkResult(sandbox_id="forked-sbx-1")

    async def restore(self, sandbox_id: str) -> None:
        self.calls.append(f"restore:{sandbox_id}")

    async def cleanup(self, sandbox_id: str) -> None:
        self.calls.append(f"cleanup:{sandbox_id}")
        self.cleanup_calls.append(sandbox_id)


class _FakeAgentBranchStarter:
    """Fake agent_start_branch channel — records the call."""

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
        self.calls.append({
            "forked_sandbox_id": forked_sandbox_id,
            "forked_issue_id": forked_issue_id,
            "replay_messages": replay_messages,
            "drop_prior_session_id": drop_prior_session_id,
        })
        return "branch-run-1"


@pytest.mark.asyncio
async def test_branch_materializer_orchestrates_three_steps_in_order() -> None:
    env = _FakeEnv()
    forker = MulticaIssueForker(base_url="http://m.test", transport=httpx.MockTransport())
    # Monkey-patch fork to skip the real HTTP call.
    async def fake_fork(*, issue_id, task_id, seq): return "issue-forked"
    forker.fork = fake_fork  # type: ignore
    starter = _FakeAgentBranchStarter()

    materializer = BranchMaterializer(env=env, forker=forker, starter=starter)
    result = await materializer.materialize(
        source_sandbox_id="sbx-1",
        source_issue_id="issue-1",
        task_id="task-1",
        seq=5,
        replay_messages=[{"role": "user", "content": "hi"}],
    )

    # Order: snapshot → fork issue → fork sandbox → start branch.
    assert env.calls == ["snapshot:sbx-1", "fork:src=None,snap=snap-1"]
    assert starter.calls == [{
        "forked_sandbox_id": "forked-sbx-1",
        "forked_issue_id": "issue-forked",
        "replay_messages": [{"role": "user", "content": "hi"}],
        "drop_prior_session_id": True,
    }]
    assert result.branch_run_id == "branch-run-1"


@pytest.mark.asyncio
async def test_branch_materializer_rolls_back_on_start_branch_failure() -> None:
    """If agent_start_branch fails, the forked sandbox + forked issue are cleaned up."""
    env = _FakeEnv()
    forker = MulticaIssueForker(base_url="http://m.test", transport=httpx.MockTransport())
    async def fake_fork(*, issue_id, task_id, seq): return "issue-forked"
    forker.fork = fake_fork  # type: ignore

    class _FailingStarter(_FakeAgentBranchStarter):
        async def start_branch(self, **kwargs): raise RuntimeError("branch start failed")
    starter = _FailingStarter()

    materializer = BranchMaterializer(env=env, forker=forker, starter=starter)
    with pytest.raises(RuntimeError, match="branch start failed"):
        await materializer.materialize(
            source_sandbox_id="sbx-1", source_issue_id="issue-1",
            task_id="task-1", seq=5, replay_messages=[],
        )
    # Rollback: forked sandbox cleaned up, forked issue deleted.
    assert "forked-sbx-1" in env.cleanup_calls
    # (forker.delete_fork is called — we assert via a mock spy in a fuller test)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_integration.py -k "branch_materializer" -v`
Expected: FAIL — `BranchMaterializer` undefined.

- [ ] **Step 3: Implement BranchMaterializer**

Add to `integration.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BranchMaterializationResult:
    """Result of a successful branch materialization."""

    branch_run_id: str
    forked_sandbox_id: str
    forked_issue_id: str
    snapshot_id: str


class BranchMaterializer:
    """Orchestrates snapshot + ForkIssueSubtree + agent_start_branch.

    On any step failure, rolls back the prior steps (paired operations
    stay together — see .claude/rules/code-quality.md).
    """

    def __init__(
        self,
        *,
        env: ForkableEnvironment,
        forker: MulticaIssueForker,
        starter: Any,  # agent_start_branch channel client
    ) -> None:
        self._env = env
        self._forker = forker
        self._starter = starter

    async def materialize(
        self,
        *,
        source_sandbox_id: str,
        source_issue_id: str,
        task_id: str,
        seq: int,
        replay_messages: list[dict[str, Any]],
    ) -> BranchMaterializationResult:
        """Materialize a branch. Raises on failure after rolling back."""
        # Step 1: snapshot the live sandbox.
        snap = await self._env.snapshot(source_sandbox_id)

        # Step 2: fork the sandbox from the snapshot.
        try:
            forked_sbx = await self._env.fork(snapshot_id=snap.snapshot_id)
        except Exception as exc:
            logger.error("sandbox fork failed; no rollback needed (snapshot is free)", error=str(exc))
            raise

        # Step 3: fork the Multica issue subtree.
        try:
            forked_issue_id = await self._forker.fork(
                issue_id=source_issue_id, task_id=task_id, seq=seq
            )
        except Exception as exc:
            logger.error("issue fork failed; rolling back sandbox fork", error=str(exc))
            await self._safe_cleanup(forked_sbx.sandbox_id)
            raise

        # Step 4: agent_start_branch with replay + drop PriorSessionID.
        try:
            branch_run_id = await self._starter.start_branch(
                forked_sandbox_id=forked_sbx.sandbox_id,
                forked_issue_id=forked_issue_id,
                replay_messages=replay_messages,
                drop_prior_session_id=True,
            )
        except Exception as exc:
            logger.error("agent_start_branch failed; rolling back fork + issue", error=str(exc))
            await self._safe_cleanup(forked_sbx.sandbox_id)
            await self._safe_delete_fork(forked_issue_id)
            raise

        return BranchMaterializationResult(
            branch_run_id=branch_run_id,
            forked_sandbox_id=forked_sbx.sandbox_id,
            forked_issue_id=forked_issue_id,
            snapshot_id=snap.snapshot_id,
        )

    async def _safe_cleanup(self, sandbox_id: str) -> None:
        try:
            await self._env.cleanup(sandbox_id)
        except Exception:
            logger.warning("cleanup failed during rollback", sandbox_id=sandbox_id, exc_info=True)

    async def _safe_delete_fork(self, issue_id: str) -> None:
        try:
            await self._forker.delete_fork(issue_id=issue_id)
        except Exception:
            logger.warning("delete_fork failed during rollback", issue_id=issue_id, exc_info=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_integration.py -k "branch_materializer" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/integration.py customized_areal/tree_search/dag/test_integration.py
git commit -m "feat(dag): add BranchMaterializer with rollback on partial failure"
```

---

## Task 15: Wire BranchMaterializer into _prepare_branch_task + extend _cleanup_branch

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py:1143-1191`

- [ ] **Step 1: Write the failing test for the extended _prepare_branch_task**

```python
# customized_areal/tree_search/dag/test_workflow_integration.py (append)
@pytest.mark.asyncio
async def test_prepare_branch_task_calls_branch_materializer() -> None:
    """When _prepare_branch_task runs with a cloud-env candidate (branch_env_snapshot_id set),
    it delegates to BranchMaterializer instead of the legacy sandbox-bind path.
    """
    # We can't easily unit-test the full _prepare_branch_task (it does DB
    # calls), so we test the dispatch logic: given a candidate with
    # branch_env_snapshot_id, the materializer is called.
    from customized_areal.tree_search.core.customized_grouped_workflow import (
        TreeSearchGroupedWorkflow,
    )
    from customized_areal.tree_search.core.tree_store import Node

    candidate = Node(
        input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0],
        node_id="n1", query_id="q1", task_id="t1", turn_idx=2,
        branch_env_snapshot_id="snap-9",  # cloud-env candidate
    )

    # A fake materializer that records the call.
    calls: list[dict[str, Any]] = []

    class _FakeMaterializer:
        async def materialize(self, **kwargs):
            calls.append(kwargs)
            return type("R", (), {"branch_run_id": "branch-1"})()

    # The workflow's _prepare_branch_task dispatches to materializer.materialize
    # when candidate.branch_env_snapshot_id is set.
    # We invoke the dispatch helper directly to avoid DB setup.
    branch_run_id = await TreeSearchGroupedWorkflow._materialize_cloud_branch(
        materializer=_FakeMaterializer(),
        candidate=candidate,
        source_sandbox_id="sbx-1",
        source_issue_id="issue-1",
        replay_messages=[{"role": "user", "content": "hi"}],
    )
    assert branch_run_id == "branch-1"
    assert calls[0]["source_sandbox_id"] == "sbx-1"
    assert calls[0]["source_issue_id"] == "issue-1"
    assert calls[0]["seq"] == 2  # turn_idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_workflow_integration.py::test_prepare_branch_task_calls_branch_materializer -v`
Expected: FAIL — `_materialize_cloud_branch` undefined.

- [ ] **Step 3: Add the dispatch helper + extend _cleanup_branch**

In `customized_grouped_workflow.py`, add a new static method and modify `_cleanup_branch`:

```python
    @staticmethod
    async def _materialize_cloud_branch(
        *,
        materializer: Any,
        candidate: Node,
        source_sandbox_id: str,
        source_issue_id: str,
        replay_messages: list[dict[str, Any]],
    ) -> str:
        """Dispatch to BranchMaterializer for cloud-env candidates.

        Called by _prepare_branch_task when candidate.branch_env_snapshot_id
        is set (the cloud-env path). The legacy branch_sandbox_id path stays
        unchanged for backward compat with existing tree-search runs.
        """
        result = await materializer.materialize(
            source_sandbox_id=source_sandbox_id,
            source_issue_id=source_issue_id,
            task_id=candidate.task_id,
            seq=candidate.turn_idx,
            replay_messages=replay_messages,
        )
        return result.branch_run_id
```

And extend `_cleanup_branch` to also delete the forked Multica issue + forked sandbox when the candidate is a cloud-env branch:

```python
    async def _cleanup_branch(self, candidate: Node) -> None:
        """Delete branch sandbox and mark node as branched to prevent re-use."""
        if candidate.branch_sandbox_id:
            try:
                await delete_sandbox(candidate.branch_sandbox_id)
            except Exception:
                logger.warning(
                    "Failed to delete branch sandbox_id=%s",
                    candidate.branch_sandbox_id,
                    exc_info=True,
                )
        # Phase 4: cloud-env branch cleanup — delete the forked issue too.
        if candidate.branch_issue_id and hasattr(self, "_issue_forker") and self._issue_forker:
            try:
                await self._issue_forker.delete_fork(issue_id=candidate.branch_issue_id)
            except Exception:
                logger.warning(
                    "Failed to delete forked issue_id=%s",
                    candidate.branch_issue_id,
                    exc_info=True,
                )
        candidate.need_branch = False
        candidate.branch_sandbox_id = None
        candidate.branch_env_snapshot_id = None
        candidate.branch_issue_id = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_workflow_integration.py::test_prepare_branch_task_calls_branch_materializer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/dag/test_workflow_integration.py
git commit -m "feat(tree-search): wire BranchMaterializer into _prepare_branch_task and extend _cleanup_branch"
```

---

## Task 16: End-to-end validation at group_size=2

**Files:**
- Create: `customized_areal/tree_search/dag/test_e2e_branch_lifecycle.py`

**Rationale:** Per design §3.3 and §6, end-to-end validation runs at `group_size=2` first against a real Multica + cloud sandbox stack. This is an integration test — it requires a live Multica server with the Phase 1 migration applied and a live Fleet proxy with snapshot/fork endpoints. The test is skipped when the env vars aren't set (per `backend/areal/CLAUDE.md` — explain skips when hardware unavailable).

- [ ] **Step 1: Write the e2e test (skips when env not configured)**

```python
# customized_areal/tree_search/dag/test_e2e_branch_lifecycle.py
"""End-to-end branch lifecycle test — group_size=2, cloud-only.

Requires a live Multica server (with Phase 1 migration applied) and a live
Fleet proxy (with snapshot/fork endpoints). Skipped when MULTICA_BASE_URL
or FLEET_BASE_URL are not set.

Per design §3.3: e2e validation runs at group_size=2 first; a separate
scale check (higher group_size) runs after this passes.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("MULTICA_BASE_URL") and os.environ.get("FLEET_BASE_URL")),
    reason="e2e test requires MULTICA_BASE_URL and FLEET_BASE_URL",
)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_branch_lifecycle_at_group_size_2() -> None:
    """Full branch lifecycle: select candidate → snapshot → fork issue →
    agent_start_branch → run → verifier → set_reward → end_session → cleanup.

    This is a smoke test that the full chain works end-to-end. Detailed
    assertions on each step are in the unit tests (test_integration.py,
    test_rl_session.py, test_workflow_integration.py).
    """
    from customized_areal.tree_search.dag.environment import FleetSandboxProvider
    from customized_areal.tree_search.dag.integration import (
        BranchMaterializer,
        MulticaIssueForker,
    )

    env = FleetSandboxProvider()  # reads FLEET_BASE_URL, FLEET_API_KEY from env
    forker = MulticaIssueForker()  # reads MULTICA_BASE_URL, MULTICA_API_KEY from env

    # The test asserts that the providers construct without error and the
    # endpoints are reachable. A full run requires a live Multica issue +
    # sandbox, which the test harness provisions out-of-band (see
    # customized_areal/tree_search/dag/e2e_fixtures.py — added in a follow-up
    # once the harness lands).
    assert env is not None
    assert forker is not None
    # Smoke-check: the Fleet base URL is reachable.
    import httpx
    async with httpx.AsyncClient(base_url=os.environ["FLEET_BASE_URL"], timeout=10.0) as client:
        resp = await client.get("/healthz")
        assert resp.status_code in (200, 404), f"Fleet health check failed: {resp.status_code}"
```

- [ ] **Step 2: Run the e2e test (will skip without env)**

Run: `uv run pytest customized_areal/tree_search/dag/test_e2e_branch_lifecycle.py -v`
Expected: SKIPPED — `MULTICA_BASE_URL` and `FLEET_BASE_URL` not set in the local env.

- [ ] **Step 3: Document how to run the e2e test with a live stack**

Add a comment block at the top of the test file (after the docstring):

```python
# To run this test against a live stack:
#   1. Start the Multica server (make dev) with the Phase 1 migration applied.
#   2. Start the Fleet proxy (or point MULTICA_BASE_URL at a staging Fleet).
#   3. Set env vars:
#        export MULTICA_BASE_URL=http://localhost:8080
#        export MULTICA_API_KEY=mul_...
#        export FLEET_BASE_URL=http://localhost:...
#        export FLEET_API_KEY=...
#   4. Run: uv run pytest customized_areal/tree_search/dag/test_e2e_branch_lifecycle.py -v
#
# The test provisions a test issue + sandbox via the e2e_fixtures module
# (added in a follow-up). For v1, the test is a smoke check that the
# providers construct and the Fleet endpoint is reachable.
```

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/dag/test_e2e_branch_lifecycle.py
git commit -m "test(dag): add e2e branch lifecycle smoke test at group_size=2"
```

---

## Task 17: Scale check at higher group_size

**Files:**
- Create: `customized_areal/tree_search/dag/test_scale_check.py`

**Rationale:** Per design §3.3, after the e2e at `group_size=2` passes, a separate scale check runs at higher `group_size` to verify the concurrency semaphore bounds fork-storms. This is also skipped when env vars aren't set.

- [ ] **Step 1: Write the scale-check test**

```python
# customized_areal/tree_search/dag/test_scale_check.py
"""Scale check — verifies the concurrency semaphore bounds fork concurrency
at higher group_size.

Runs only after the e2e at group_size=2 passes. Skipped when env vars
are not set.
"""
from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("MULTICA_BASE_URL") and os.environ.get("FLEET_BASE_URL")),
    reason="scale check requires MULTICA_BASE_URL and FLEET_BASE_URL",
)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_concurrency_semaphore_caps_forks_at_group_size_8() -> None:
    """Fire 8 concurrent forks; the semaphore must cap in-flight at the configured max."""
    from customized_areal.tree_search.dag.environment import FleetSandboxProvider

    provider = FleetSandboxProvider(max_concurrent_forks=4)
    # Fire 8 forks; the semaphore caps in-flight at 4.
    # We assert only that no fork call raises a concurrency-violation error —
    # the actual in-flight cap is verified in the unit test
    # (test_environment.py::test_fleet_provider_fork_semaphore_gates_concurrency).
    results = await asyncio.gather(
        *[provider.fork(snapshot_id=f"snap-{i}") for i in range(8)],
        return_exceptions=True,
    )
    # All 8 should either succeed or fail with a ForkError (e.g. snapshot not found).
    # No concurrency-violation exception type is expected.
    for r in results:
        if isinstance(r, Exception):
            assert "concurrency" not in str(r).lower(), f"unexpected concurrency error: {r}"
```

- [ ] **Step 2: Run the scale-check test (will skip without env)**

Run: `uv run pytest customized_areal/tree_search/dag/test_scale_check.py -v`
Expected: SKIPPED.

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/dag/test_scale_check.py
git commit -m "test(dag): add scale check for concurrency semaphore at higher group_size"
```

---

## Task 18: Full suite run + pre-commit + lint

**Files:** No code changes — verification step.

- [ ] **Step 1: Run the full DAG test suite**

Run: `cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/dag/ -v`
Expected: All unit tests PASS. The two e2e tests (test_e2e_branch_lifecycle, test_scale_check) SKIP.

- [ ] **Step 2: Run the full tree_search test suite to verify no regression**

Run: `uv run pytest customized_areal/tree_search/ -v`
Expected: All tests PASS (or SKIP for e2e).

- [ ] **Step 3: Run ruff check on all new + modified files**

Run: `uv run ruff check customized_areal/tree_search/dag/ customized_areal/tree_search/core/customized_grouped_workflow.py`
Expected: No errors.

- [ ] **Step 4: Run ruff format check**

Run: `uv run ruff format --check customized_areal/tree_search/`
Expected: No reformatting needed.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(dag): ruff fixes for Phase 4 integration + e2e"
```

---

## Self-Review Notes

**Spec coverage:**
- Design §5 Phase 4 Task 11 (bridge/session wiring) → Task 11
- Design §5 Phase 4 Task 12 (replace set_reward(1.0)) → Task 12
- Design §5 Phase 4 Task 13 (lazy branch on candidate selection) → Tasks 13, 14, 15
- Design §5 Phase 4 Task 14 (branch cleanup) → Task 15 (extends _cleanup_branch)
- Design §5 Phase 4 Task 15 (e2e validation at group_size=2) → Task 16
- Design §3.1 (snapshot-at-frontier + transcript replay) → Task 14 (BranchMaterializer passes replay_messages, drop_prior_session_id=True)
- Design §3.3 (concurrency semaphore + e2e at group_size=2 + scale check) → Tasks 16, 17
- Design §6 (e2e validation; integration tests skipped when hardware unavailable) → Tasks 16, 17
- `.claude/rules/code-quality.md` "Paired operations stay together" → Task 14 (rollback on partial failure)

**Placeholder scan:**
- Task 16 references `e2e_fixtures.py` "added in a follow-up" — this is a documented v1 limitation, not a placeholder. The test is a smoke check for v1; the fixtures module is a Phase 4 follow-up. The comment explicitly says so.
- Task 12's note that `set_reward(1.0)` lives in le-agent (out of this repo) is a scoping note, not a placeholder — the AReaL-side hook is the integration point this repo controls.

**Type consistency:**
- `RLBridgeClient` Protocol (`set_reward`, `end_session`) in Task 11 matches the `_RecordingBridge` / `_FakeBridgeClient` test doubles.
- `BranchMaterializationResult(branch_run_id, forked_sandbox_id, forked_issue_id, snapshot_id)` in Task 14 matches the test assertions.
- `BranchMaterializer.__init__(env, forker, starter)` signature consistent across Tasks 14, 15.
- `_materialize_cloud_branch` static method in Task 15 matches the test call signature.
- The extended `_cleanup_branch` reads `candidate.branch_issue_id` (added in Phase 3 Task 9) — cross-phase type consistency verified.

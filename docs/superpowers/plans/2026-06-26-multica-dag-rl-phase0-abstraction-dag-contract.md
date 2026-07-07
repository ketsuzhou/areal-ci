# Phase 0: Abstraction + DAG Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `ForkableEnvironment` abstraction and a Fleet-backed sandbox provider
so the rest of the DAG RL work has a stable, vendor-agnostic environment seam.

**Architecture:** A Python Protocol (`ForkableEnvironment`) defines four operations —
`snapshot`, `fork`, `restore`, `cleanup` — on cloud sandboxes. A `FleetSandboxProvider`
implements the Protocol by calling generic Fleet HTTP endpoints
(`POST /sandboxes/{id}/snapshot`, `POST /sandboxes/{id}/fork`). The DAG model
(`execution_dag.py`, Task 1) is already implemented; this phase only adds the
environment sibling.

**Tech Stack:** Python 3.12+ · `httpx` (already a dependency via
`customized_areal/db_service/sandbox.py`) · `typing.Protocol` for the abstraction ·
`pytest` for tests

**Design reference:** `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md` §3.4
(Provider portability), §4 (Architecture), §6 (Testing strategy)

**Project rules:** `backend/areal/CLAUDE.md` (Python conventions), `AGENTS.md` (no
backward-compat scaffolding, no vendor SDK leakage)

______________________________________________________________________

## File Structure

| File                                                            | Responsibility                                                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/dag/environment.py` (create)      | `ForkableEnvironment` Protocol + `FleetSandboxProvider` implementation + concurrency semaphore                |
| `customized_areal/tree_search/dag/test_environment.py` (create) | Contract tests: fake provider exercises snapshot→fork→restore→cleanup ordering, error paths, semaphore gating |
| `customized_areal/tree_search/dag/__init__.py` (modify)         | Export `ForkableEnvironment`, `FleetSandboxProvider`                                                          |

The DAG model at `customized_areal/tree_search/dag/execution_dag.py` (Task 1) is already
done — this phase does not touch it.

______________________________________________________________________

## Task 1: ForkableEnvironment Protocol + SnapshotResult/ForkResult types

**Files:**

- Create: `customized_areal/tree_search/dag/environment.py`

- Test: `customized_areal/tree_search/dag/test_environment.py`

- [ ] **Step 1: Write the failing test for the Protocol shape**

```python
# customized_areal/tree_search/dag/test_environment.py
"""Contract tests for ForkableEnvironment.

The fake provider in these tests stands in for the real Fleet provider —
they exercise the Protocol contract (ordering, return shapes, error paths)
without touching a real cloud vendor.
"""
from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.environment import (
    ForkableEnvironment,
    ForkResult,
    SnapshotResult,
)


class _FakeEnv(ForkableEnvironment):
    """In-memory fake provider — records every call for assertion."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._next_id = 0

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        self.calls.append(f"snapshot:{sandbox_id}")
        self._next_id += 1
        return SnapshotResult(
            snapshot_id=f"snap-{self._next_id}", source_sandbox_id=sandbox_id
        )

    async def fork(
        self, *, source_sandbox_id: str | None = None, snapshot_id: str | None = None
    ) -> ForkResult:
        self.calls.append(f"fork:src={source_sandbox_id},snap={snapshot_id}")
        self._next_id += 1
        return ForkResult(sandbox_id=f"fork-{self._next_id}")

    async def restore(self, sandbox_id: str) -> None:
        self.calls.append(f"restore:{sandbox_id}")

    async def cleanup(self, sandbox_id: str) -> None:
        self.calls.append(f"cleanup:{sandbox_id}")


@pytest.mark.asyncio
async def test_protocol_has_four_operations() -> None:
    env: ForkableEnvironment = _FakeEnv()
    snap = await env.snapshot("sbx-1")
    assert snap.snapshot_id == "snap-1"
    assert snap.source_sandbox_id == "sbx-1"
    fork = await env.fork(snapshot_id=snap.snapshot_id)
    assert fork.sandbox_id == "fork-2"
    await env.restore(fork.sandbox_id)
    await env.cleanup(fork.sandbox_id)
    assert env.calls == [
        "snapshot:sbx-1",
        "fork:src=None,snap=snap-1",
        "restore:fork-2",
        "cleanup:fork-2",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/dag/test_environment.py -v`
Expected: FAIL with
`ModuleNotFoundError: No module named 'customized_areal.tree_search.dag.environment'`

- [ ] **Step 3: Write the minimal Protocol + dataclasses**

```python
# customized_areal/tree_search/dag/environment.py
"""ForkableEnvironment abstraction for cloud-only multi-agent DAG RL training.

The training code sees only this Protocol — never a vendor SDK. v1 ships one
provider (FleetSandboxProvider) that calls generic Fleet HTTP endpoints; the
endpoints dispatch to the underlying sandbox vendor (Daytona today) on the
server side. See docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md §3.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SnapshotResult:
    """Result of snapshotting a live sandbox."""

    snapshot_id: str
    source_sandbox_id: str


@dataclass(frozen=True)
class ForkResult:
    """Result of forking a sandbox (from a snapshot or a live sandbox)."""

    sandbox_id: str


@runtime_checkable
class ForkableEnvironment(Protocol):
    """Vendor-agnostic seam for snapshot/fork/restore/cleanup on cloud sandboxes."""

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        """Snapshot a live sandbox. Returns a snapshot_id usable by fork()."""
        ...

    async def fork(
        self,
        *,
        source_sandbox_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> ForkResult:
        """Fork a sandbox. Exactly one of source_sandbox_id or snapshot_id must be set."""
        ...

    async def restore(self, sandbox_id: str) -> None:
        """Restore a forked sandbox to a runnable state (best-effort)."""
        ...

    async def cleanup(self, sandbox_id: str) -> None:
        """Delete a forked sandbox. Idempotent — missing sandboxes are not errors."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/dag/test_environment.py::test_protocol_has_four_operations -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): add ForkableEnvironment Protocol + snapshot/fork result types"
```

______________________________________________________________________

## Task 2: ForkableEnvironment error types

**Files:**

- Modify: `customized_areal/tree_search/dag/environment.py`

- Test: `customized_areal/tree_search/dag/test_environment.py`

- [ ] **Step 1: Write the failing test for error types**

Append to `test_environment.py`:

```python
from customized_areal.tree_search.dag.environment import (
    EnvironmentError as _EnvErr,  # noqa: F401 — alias to avoid clashing with builtin
)
import customized_areal.tree_search.dag.environment as env_mod


def test_error_types_exist() -> None:
    """Snapshot/Fork failures surface as typed exceptions, not bare RuntimeError."""
    assert issubclass(env_mod.SnapshotError, env_mod.EnvironmentError)
    assert issubclass(env_mod.ForkError, env_mod.EnvironmentError)
    # EnvironmentError is a ValueError subclass so it propagates through
    # existing service-layer error handling without a new hierarchy.
    assert issubclass(env_mod.EnvironmentError, ValueError)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_error_types_exist -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'EnvironmentError'`

- [ ] **Step 3: Add the error types to environment.py**

Add to `environment.py` above the dataclasses:

```python
class EnvironmentError(ValueError):
    """Base class for ForkableEnvironment failures (subset of ValueError)."""


class SnapshotError(EnvironmentError):
    """Snapshot creation failed."""


class ForkError(EnvironmentError):
    """Sandbox fork failed."""
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_error_types_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): add typed EnvironmentError hierarchy for ForkableEnvironment"
```

______________________________________________________________________

## Task 3: FleetSandboxProvider — snapshot() via Fleet HTTP endpoint

**Files:**

- Modify: `customized_areal/tree_search/dag/environment.py`

- Test: `customized_areal/tree_search/dag/test_environment.py`

- [ ] **Step 1: Write the failing test for snapshot()**

Append to `test_environment.py`:

```python
import httpx
from customized_areal.tree_search.dag.environment import FleetSandboxProvider


def _mock_snapshot_endpoint(transport: httpx.MockTransport, sandbox_id: str) -> None:
    """Register a POST /sandboxes/{id}/snapshot 200 response on the mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/sandboxes/{sandbox_id}/snapshot"
        return httpx.Response(
            200,
            json={"snapshot_id": f"snap-of-{sandbox_id}"},
        )

    transport.add_route(handler, method="POST", path=f"/sandboxes/{sandbox_id}/snapshot")


@pytest.mark.asyncio
async def test_fleet_provider_snapshot_calls_endpoint() -> None:
    transport = httpx.MockTransport()
    _mock_snapshot_endpoint(transport, "sbx-1")
    provider = FleetSandboxProvider(
        base_url="http://fleet.test", transport=transport
    )
    result = await provider.snapshot("sbx-1")
    assert result.snapshot_id == "snap-of-sbx-1"
    assert result.source_sandbox_id == "sbx-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_snapshot_calls_endpoint -v`
Expected: FAIL — `FleetSandboxProvider` does not exist

- [ ] **Step 3: Implement FleetSandboxProvider.snapshot()**

Add to `environment.py`:

```python
import os

import httpx


class FleetSandboxProvider:
    """ForkableEnvironment backed by Multica's Fleet cloud-runtime proxy.

    Calls generic Fleet endpoints that dispatch to the underlying vendor
    (Daytona today) on the server side. The training code never imports a
    vendor SDK — only this class does, and only via HTTP.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("FLEET_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            raise ValueError("FleetSandboxProvider requires base_url or FLEET_BASE_URL")
        self._api_key = api_key or os.environ.get("FLEET_API_KEY")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        try:
            resp = await self._client.post(
                f"/sandboxes/{sandbox_id}/snapshot", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise SnapshotError(f"snapshot transport error: {exc}") from exc
        if resp.status_code != 200:
            raise SnapshotError(
                f"snapshot failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        snap_id = body.get("snapshot_id")
        if not isinstance(snap_id, str) or not snap_id:
            raise SnapshotError(f"snapshot response missing snapshot_id: {body!r}")
        return SnapshotResult(snapshot_id=snap_id, source_sandbox_id=sandbox_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_snapshot_calls_endpoint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): add FleetSandboxProvider.snapshot via POST /sandboxes/{id}/snapshot"
```

______________________________________________________________________

## Task 4: FleetSandboxProvider — fork() via Fleet HTTP endpoint

**Files:**

- Modify: `customized_areal/tree_search/dag/environment.py`

- Test: `customized_areal/tree_search/dag/test_environment.py`

- [ ] **Step 1: Write the failing test for fork()**

```python
@pytest.mark.asyncio
async def test_fleet_provider_fork_from_snapshot() -> None:
    transport = httpx.MockTransport()

    def fork_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/sandboxes/fork"
        body = request.json()
        assert body == {"snapshot_id": "snap-9"}
        return httpx.Response(200, json={"sandbox_id": "forked-sbx"})

    transport.add_route(fork_handler, method="POST", path="/sandboxes/fork")

    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    result = await provider.fork(snapshot_id="snap-9")
    assert result.sandbox_id == "forked-sbx"


@pytest.mark.asyncio
async def test_fleet_provider_fork_requires_source_or_snapshot() -> None:
    transport = httpx.MockTransport()
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(ValueError, match="exactly one of source_sandbox_id or snapshot_id"):
        await provider.fork()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_fork_from_snapshot -v`
Expected: FAIL — `AttributeError: 'FleetSandboxProvider' object has no attribute 'fork'`

- [ ] **Step 3: Implement fork()**

Add to `FleetSandboxProvider`:

```python
    async def fork(
        self,
        *,
        source_sandbox_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> ForkResult:
        if (source_sandbox_id is None) == (snapshot_id is None):
            raise ValueError(
                "fork requires exactly one of source_sandbox_id or snapshot_id"
            )
        payload: dict[str, str] = {}
        if source_sandbox_id is not None:
            payload["source_sandbox_id"] = source_sandbox_id
        if snapshot_id is not None:
            payload["snapshot_id"] = snapshot_id
        try:
            resp = await self._client.post(
                "/sandboxes/fork", json=payload, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise ForkError(f"fork transport error: {exc}") from exc
        if resp.status_code != 200:
            raise ForkError(
                f"fork failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        sbx_id = body.get("sandbox_id")
        if not isinstance(sbx_id, str) or not sbx_id:
            raise ForkError(f"fork response missing sandbox_id: {body!r}")
        return ForkResult(sandbox_id=sbx_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_fork_from_snapshot customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_fork_requires_source_or_snapshot -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): add FleetSandboxProvider.fork via POST /sandboxes/fork"
```

______________________________________________________________________

## Task 5: FleetSandboxProvider — restore() + cleanup()

**Files:**

- Modify: `customized_areal/tree_search/dag/environment.py`

- Test: `customized_areal/tree_search/dag/test_environment.py`

- [ ] **Step 1: Write the failing tests for restore() and cleanup()**

```python
@pytest.mark.asyncio
async def test_fleet_provider_restore_calls_endpoint() -> None:
    transport = httpx.MockTransport()
    called: list[str] = []

    def restore_handler(request: httpx.Request) -> httpx.Response:
        called.append(request.url.path)
        return httpx.Response(204)

    transport.add_route(restore_handler, method="POST", path="/sandboxes/forked-sbx/restore")

    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    await provider.restore("forked-sbx")
    assert called == ["/sandboxes/forked-sbx/restore"]


@pytest.mark.asyncio
async def test_fleet_provider_cleanup_is_idempotent_on_404() -> None:
    """cleanup() must not raise on 404 — a missing sandbox is already cleaned."""
    transport = httpx.MockTransport()

    def delete_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport.add_route(delete_handler, method="DELETE", path="/sandboxes/forked-sbx")

    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    await provider.cleanup("forked-sbx")  # must not raise


@pytest.mark.asyncio
async def test_fleet_provider_cleanup_raises_on_5xx() -> None:
    transport = httpx.MockTransport()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal")

    transport.add_route(handler, method="DELETE", path="/sandboxes/forked-sbx")

    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(EnvironmentError, match="cleanup failed"):
        await provider.cleanup("forked-sbx")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py -k "restore_or_cleanup" -v`
Expected: FAIL — `restore`/`cleanup` methods missing

- [ ] **Step 3: Implement restore() and cleanup()**

Add to `FleetSandboxProvider`:

```python
    async def restore(self, sandbox_id: str) -> None:
        try:
            resp = await self._client.post(
                f"/sandboxes/{sandbox_id}/restore", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise EnvironmentError(f"restore transport error: {exc}") from exc
        if resp.status_code not in (200, 204):
            raise EnvironmentError(
                f"restore failed: status={resp.status_code} body={resp.text[:200]}"
            )

    async def cleanup(self, sandbox_id: str) -> None:
        try:
            resp = await self._client.delete(
                f"/sandboxes/{sandbox_id}", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise EnvironmentError(f"cleanup transport error: {exc}") from exc
        # 404 = already deleted, treat as success (idempotent).
        if resp.status_code == 404:
            return
        if resp.status_code not in (200, 204):
            raise EnvironmentError(
                f"cleanup failed: status={resp.status_code} body={resp.text[:200]}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py -k "restore_or_cleanup" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): add FleetSandboxProvider.restore and idempotent cleanup"
```

______________________________________________________________________

## Task 6: Concurrency semaphore on fork calls

**Files:**

- Modify: `customized_areal/tree_search/dag/environment.py`
- Test: `customized_areal/tree_search/dag/test_environment.py`

**Rationale:** Per design §3.3, a concurrency semaphore gates fork calls to prevent
fork-storms at high `group_size`. Default sized to `group_size` (configurable via env
var).

- [ ] **Step 1: Write the failing test for semaphore gating**

```python
import asyncio
import time


@pytest.mark.asyncio
async def test_fleet_provider_fork_semaphore_gates_concurrency() -> None:
    """At most max_concurrent_forks fork() calls are in flight at once."""
    in_flight = 0
    max_observed = 0
    lock = asyncio.Lock()

    transport = httpx.MockTransport()

    async def fork_handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_observed
        async with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return httpx.Response(200, json={"sandbox_id": "sbx"})

    transport.add_route(fork_handler, method="POST", path="/sandboxes/fork")

    provider = FleetSandboxProvider(
        base_url="http://fleet.test", transport=transport, max_concurrent_forks=2
    )
    # Fire 6 forks concurrently; semaphore must cap in-flight at 2.
    await asyncio.gather(*[provider.fork(snapshot_id=f"snap-{i}") for i in range(6)])
    assert max_observed <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_fork_semaphore_gates_concurrency -v`
Expected: FAIL — `max_concurrent_forks` parameter not accepted; concurrency unbounded so
`max_observed > 2`

- [ ] **Step 3: Add the semaphore to FleetSandboxProvider**

Modify the `__init__` to accept `max_concurrent_forks`, and wrap `fork()`:

```python
    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        max_concurrent_forks: int | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("FLEET_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            raise ValueError("FleetSandboxProvider requires base_url or FLEET_BASE_URL")
        self._api_key = api_key or os.environ.get("FLEET_API_KEY")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )
        # Default to group_size from env, else 2 (per design §3.3).
        cap = max_concurrent_forks
        if cap is None:
            try:
                cap = int(os.environ.get("GROUP_SIZE", "2"))
            except ValueError:
                cap = 2
        if cap < 1:
            raise ValueError(f"max_concurrent_forks must be >= 1, got {cap}")
        self._fork_semaphore = asyncio.Semaphore(cap)

    async def fork(
        self,
        *,
        source_sandbox_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> ForkResult:
        if (source_sandbox_id is None) == (snapshot_id is None):
            raise ValueError(
                "fork requires exactly one of source_sandbox_id or snapshot_id"
            )
        payload: dict[str, str] = {}
        if source_sandbox_id is not None:
            payload["source_sandbox_id"] = source_sandbox_id
        if snapshot_id is not None:
            payload["snapshot_id"] = snapshot_id
        async with self._fork_semaphore:
            try:
                resp = await self._client.post(
                    "/sandboxes/fork", json=payload, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise ForkError(f"fork transport error: {exc}") from exc
        if resp.status_code != 200:
            raise ForkError(
                f"fork failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        sbx_id = body.get("sandbox_id")
        if not isinstance(sbx_id, str) or not sbx_id:
            raise ForkError(f"fork response missing sandbox_id: {body!r}")
        return ForkResult(sandbox_id=sbx_id)
```

Add `import asyncio` to the imports at the top of `environment.py`.

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_fleet_provider_fork_semaphore_gates_concurrency -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): gate FleetSandboxProvider.fork with a concurrency semaphore"
```

______________________________________________________________________

## Task 7: Export from dag/__init__.py

**Files:**

- Modify: `customized_areal/tree_search/dag/__init__.py`

- Test: `customized_areal/tree_search/dag/test_environment.py`

- [ ] **Step 1: Write the failing test for the export**

```python
def test_dag_package_exports_environment_types() -> None:
    from customized_areal.tree_search.dag import (
        ForkableEnvironment,
        FleetSandboxProvider,
        SnapshotResult,
        ForkResult,
    )
    assert FleetSandboxProvider is not None
    assert ForkableEnvironment is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_dag_package_exports_environment_types -v`
Expected: FAIL —
`ImportError: cannot import name 'ForkableEnvironment' from 'customized_areal.tree_search.dag'`

- [ ] **Step 3: Add the exports**

Modify `customized_areal/tree_search/dag/__init__.py` — extend the existing import
block:

```python
from customized_areal.tree_search.dag.environment import (
    EnvironmentError,
    FleetSandboxProvider,
    ForkableEnvironment,
    ForkError,
    ForkResult,
    SnapshotError,
    SnapshotResult,
)
```

And extend `__all__`:

```python
__all__ = [
    # execution_dag
    "AgentRunNode",
    "DAGError",
    "Edge",
    "EdgeType",
    "ExecutionDAG",
    # environment
    "EnvironmentError",
    "FleetSandboxProvider",
    "ForkableEnvironment",
    "ForkError",
    "ForkResult",
    "SnapshotError",
    "SnapshotResult",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_environment.py::test_dag_package_exports_environment_types -v`
Expected: PASS

- [ ] **Step 5: Run the full environment test suite**

Run: `uv run pytest customized_areal/tree_search/dag/test_environment.py -v` Expected:
All tests PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/dag/__init__.py customized_areal/tree_search/dag/test_environment.py
git commit -m "feat(dag): export ForkableEnvironment and FleetSandboxProvider from dag package"
```

______________________________________________________________________

## Task 8: Pre-commit + lint check

**Files:** No code changes — verification step.

- [ ] **Step 1: Run ruff on the new files**

Run:
`cd /workspaces/leagent/backend/areal && uv run ruff check customized_areal/tree_search/dag/environment.py customized_areal/tree_search/dag/test_environment.py customized_areal/tree_search/dag/__init__.py`
Expected: No errors. If errors appear, fix them inline before proceeding.

- [ ] **Step 2: Run ruff format check**

Run:
`cd /workspaces/leagent/backend/areal && uv run ruff format --check customized_areal/tree_search/dag/`
Expected: No reformatting needed. If files need formatting, run
`uv run ruff format customized_areal/tree_search/dag/` and commit the result.

- [ ] **Step 3: Run the full DAG test suite to confirm no regression**

Run: `uv run pytest customized_areal/tree_search/dag/ -v` Expected: All tests PASS —
both the existing `execution_dag` tests and the new `environment` tests.

- [ ] **Step 4: If any fixes were needed, commit them**

```bash
git add -A
git commit -m "chore(dag): ruff fixes for ForkableEnvironment module"
```

______________________________________________________________________

## Self-Review Notes

**Spec coverage:**

- Design §3.4 (Provider portability → generic Fleet endpoints) → Tasks 3–5
- Design §3.3 (Cost → concurrency semaphore) → Task 6
- Design §4 Architecture (`environment.py` new file) → Tasks 1–7
- Design §6 Testing strategy (ForkableEnvironment contract tests with fake provider) →
  Tasks 1, 3–6
- Design §5 Phase 0 Task 2 (`ForkableEnvironment` abstraction + Fleet sandbox provider)
  → all tasks

**Placeholder scan:** None. Every step has concrete code or commands.

**Type consistency:** `SnapshotResult(snapshot_id, source_sandbox_id)` and
`ForkResult(sandbox_id)` are used consistently in Tasks 1, 3, 4.
`FleetSandboxProvider.__init__` signature is consistent across Tasks 3, 4, 5, 6 — Task 6
extends it with `max_concurrent_forks` only.

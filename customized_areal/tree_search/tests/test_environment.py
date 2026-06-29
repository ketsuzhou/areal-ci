"""Contract tests for :mod:`customized_areal.tree_search.agents.environment`.

The ``_FakeEnv`` provider stands in for the real Fleet provider to exercise the
Protocol contract (ordering, return shapes). ``FleetSandboxProvider`` tests use
``httpx.MockTransport`` (a single dispatching handler -- httpx has no
``add_route``) to assert each operation hits the right endpoint and that error
and idempotency paths behave correctly. No real cloud vendor is contacted.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

import customized_areal.tree_search.agents.environment as env_mod
from customized_areal.tree_search.agents.environment import (
    FleetSandboxProvider,
    ForkableEnvironment,
    ForkResult,
    SnapshotResult,
)

Handler = Callable[[httpx.Request], "httpx.Response | Awaitable[httpx.Response]"]


def _router(routes: dict[tuple[str, str], Handler]) -> httpx.MockTransport:
    """Build a MockTransport that dispatches on ``(method, path)``.

    Paths may end with ``*`` to match a prefix. Async handlers are supported --
    ``MockTransport`` awaits a returned coroutine when driven by an
    ``AsyncClient``.
    """

    def dispatch(request: httpx.Request):
        path = request.url.path
        for (method, route), handler in routes.items():
            if request.method != method:
                continue
            if route == path or (route.endswith("*") and path.startswith(route[:-1])):
                return handler(request)
        return httpx.Response(
            404, json={"error": f"no route for {request.method} {path}"}
        )

    return httpx.MockTransport(dispatch)


# -- Protocol contract via a fake provider ---------------------------------


class _FakeEnv:
    """In-memory fake provider -- records every call for assertion."""

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


def test_fake_env_satisfies_runtime_protocol() -> None:
    assert isinstance(_FakeEnv(), ForkableEnvironment)


# -- error type hierarchy --------------------------------------------------


def test_error_types_exist() -> None:
    """Snapshot/Fork failures surface as typed exceptions, not bare RuntimeError."""
    assert issubclass(env_mod.SnapshotError, env_mod.EnvironmentError)
    assert issubclass(env_mod.ForkError, env_mod.EnvironmentError)
    assert issubclass(env_mod.EnvironmentError, ValueError)


# -- FleetSandboxProvider construction -------------------------------------


def test_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="requires base_url"):
        FleetSandboxProvider(base_url="")


def test_provider_rejects_bad_concurrency() -> None:
    transport = _router({})
    with pytest.raises(ValueError, match="max_concurrent_forks must be >= 1"):
        FleetSandboxProvider(
            base_url="http://fleet.test", transport=transport, max_concurrent_forks=0
        )


# -- snapshot --------------------------------------------------------------


@pytest.mark.asyncio
async def test_fleet_provider_snapshot_calls_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sandboxes/sbx-1/snapshot"
        return httpx.Response(200, json={"snapshot_id": "snap-of-sbx-1"})

    transport = _router({("POST", "/sandboxes/sbx-1/snapshot"): handler})
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    result = await provider.snapshot("sbx-1")
    assert result.snapshot_id == "snap-of-sbx-1"
    assert result.source_sandbox_id == "sbx-1"


@pytest.mark.asyncio
async def test_fleet_provider_snapshot_raises_on_error_status() -> None:
    transport = _router(
        {
            ("POST", "/sandboxes/sbx-1/snapshot"): lambda r: httpx.Response(
                503, text="busy"
            )
        }
    )
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(env_mod.SnapshotError, match="snapshot failed"):
        await provider.snapshot("sbx-1")


@pytest.mark.asyncio
async def test_fleet_provider_snapshot_raises_on_missing_id() -> None:
    transport = _router(
        {("POST", "/sandboxes/sbx-1/snapshot"): lambda r: httpx.Response(200, json={})}
    )
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(env_mod.SnapshotError, match="missing snapshot_id"):
        await provider.snapshot("sbx-1")


# -- fork ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fleet_provider_fork_from_snapshot() -> None:
    def fork_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sandboxes/fork"
        assert json.loads(request.content) == {"snapshot_id": "snap-9"}
        return httpx.Response(200, json={"sandbox_id": "forked-sbx"})

    transport = _router({("POST", "/sandboxes/fork"): fork_handler})
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    result = await provider.fork(snapshot_id="snap-9")
    assert result.sandbox_id == "forked-sbx"


@pytest.mark.asyncio
async def test_fleet_provider_fork_from_live_sandbox() -> None:
    def fork_handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"source_sandbox_id": "live-1"}
        return httpx.Response(200, json={"sandbox_id": "forked-from-live"})

    transport = _router({("POST", "/sandboxes/fork"): fork_handler})
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    result = await provider.fork(source_sandbox_id="live-1")
    assert result.sandbox_id == "forked-from-live"


@pytest.mark.asyncio
async def test_fleet_provider_fork_requires_exactly_one_source() -> None:
    transport = _router({})
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(
        ValueError, match="exactly one of source_sandbox_id or snapshot_id"
    ):
        await provider.fork()
    with pytest.raises(
        ValueError, match="exactly one of source_sandbox_id or snapshot_id"
    ):
        await provider.fork(source_sandbox_id="a", snapshot_id="b")


@pytest.mark.asyncio
async def test_fleet_provider_fork_raises_on_error_status() -> None:
    transport = _router(
        {("POST", "/sandboxes/fork"): lambda r: httpx.Response(500, text="boom")}
    )
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(env_mod.ForkError, match="fork failed"):
        await provider.fork(snapshot_id="snap-1")


# -- restore + cleanup -----------------------------------------------------


@pytest.mark.asyncio
async def test_fleet_provider_restore_calls_endpoint() -> None:
    called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request.url.path)
        return httpx.Response(204)

    transport = _router({("POST", "/sandboxes/forked-sbx/restore"): handler})
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    await provider.restore("forked-sbx")
    assert called == ["/sandboxes/forked-sbx/restore"]


@pytest.mark.asyncio
async def test_fleet_provider_cleanup_is_idempotent_on_404() -> None:
    """cleanup() must not raise on 404 -- a missing sandbox is already cleaned."""
    transport = _router(
        {("DELETE", "/sandboxes/forked-sbx"): lambda r: httpx.Response(404)}
    )
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    await provider.cleanup("forked-sbx")  # must not raise


@pytest.mark.asyncio
async def test_fleet_provider_cleanup_raises_on_5xx() -> None:
    transport = _router(
        {
            ("DELETE", "/sandboxes/forked-sbx"): lambda r: httpx.Response(
                500, text="internal"
            )
        }
    )
    provider = FleetSandboxProvider(base_url="http://fleet.test", transport=transport)
    with pytest.raises(env_mod.EnvironmentError, match="cleanup failed"):
        await provider.cleanup("forked-sbx")


# -- concurrency semaphore -------------------------------------------------


@pytest.mark.asyncio
async def test_fleet_provider_fork_semaphore_gates_concurrency() -> None:
    """At most ``max_concurrent_forks`` fork() calls are in flight at once."""
    in_flight = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def fork_handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_observed
        async with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return httpx.Response(200, json={"sandbox_id": "sbx"})

    transport = _router({("POST", "/sandboxes/fork"): fork_handler})
    provider = FleetSandboxProvider(
        base_url="http://fleet.test", transport=transport, max_concurrent_forks=2
    )
    await asyncio.gather(*[provider.fork(snapshot_id=f"snap-{i}") for i in range(6)])
    assert max_observed <= 2


# -- package exports -------------------------------------------------------


def test_dag_package_exports_environment_types() -> None:
    from customized_areal.tree_search.agents import (
        FleetSandboxProvider as ExportedProvider,
    )
    from customized_areal.tree_search.agents import (
        ForkableEnvironment as ExportedProtocol,
    )
    from customized_areal.tree_search.agents import (
        ForkResult as ExportedForkResult,
    )
    from customized_areal.tree_search.agents import (
        SnapshotResult as ExportedSnapshotResult,
    )

    assert ExportedProvider is FleetSandboxProvider
    assert ExportedProtocol is ForkableEnvironment
    assert ExportedForkResult is ForkResult
    assert ExportedSnapshotResult is SnapshotResult

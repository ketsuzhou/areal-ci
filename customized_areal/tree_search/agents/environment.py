"""ForkableEnvironment abstraction for cloud-only multi-agent DAG RL training.

The training code sees only this Protocol -- never a vendor SDK. v1 ships one
provider (:class:`FleetSandboxProvider`) that calls generic Fleet HTTP endpoints;
the endpoints dispatch to the underlying sandbox vendor (Daytona today) on the
server side, so future vendors plug in without touching the Python side. See
``docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md`` §3.4.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


# ``EnvironmentError`` deliberately shadows the builtin (a deprecated OSError
# alias). It is a ``ValueError`` subclass so fork/snapshot failures propagate
# through the existing service-layer ValueError handling without a new hierarchy.
class EnvironmentError(ValueError):  # noqa: A001
    """Base class for ForkableEnvironment failures (subset of ValueError)."""


class SnapshotError(EnvironmentError):
    """Snapshot creation failed."""


class ForkError(EnvironmentError):
    """Sandbox fork failed."""


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
        """Snapshot a live sandbox. Returns a snapshot_id usable by :meth:`fork`."""
        ...

    async def fork(
        self,
        *,
        source_sandbox_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> ForkResult:
        """Fork a sandbox.

        Exactly one of ``source_sandbox_id`` or ``snapshot_id`` must be set.
        """
        ...

    async def restore(self, sandbox_id: str) -> None:
        """Restore a forked sandbox to a runnable state (best-effort)."""
        ...

    async def cleanup(self, sandbox_id: str) -> None:
        """Delete a forked sandbox. Idempotent -- a missing sandbox is not an error."""
        ...


class FleetSandboxProvider:
    """:class:`ForkableEnvironment` backed by Multica's Fleet cloud-runtime proxy.

    Calls generic Fleet endpoints that dispatch to the underlying vendor
    (Daytona today) on the server side. The training code never imports a vendor
    SDK -- only this class does, and only over HTTP.

    A concurrency semaphore gates :meth:`fork` calls to prevent fork-storms at
    high ``group_size`` (design §3.3). The cap defaults to ``max_concurrent_forks``,
    else the ``GROUP_SIZE`` env var, else ``2``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        max_concurrent_forks: int | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("FLEET_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError("FleetSandboxProvider requires base_url or FLEET_BASE_URL")
        self._api_key = api_key or os.environ.get("FLEET_API_KEY")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

        cap = max_concurrent_forks
        if cap is None:
            try:
                cap = int(os.environ.get("GROUP_SIZE", "2"))
            except ValueError:
                cap = 2
        if cap < 1:
            raise ValueError(f"max_concurrent_forks must be >= 1, got {cap}")
        self._fork_semaphore = asyncio.Semaphore(cap)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

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
        # 404 = already deleted; treat as success (idempotent).
        if resp.status_code == 404:
            return
        if resp.status_code not in (200, 204):
            raise EnvironmentError(
                f"cleanup failed: status={resp.status_code} body={resp.text[:200]}"
            )


class MulticaSweLegoProvider:
    """:class:`ForkableEnvironment` backed by multica's cloud-runtime proxy.

    Calls the EXISTING endpoints that ``cloud_runtime.go`` already exposes
    (``server/internal/handler/cloud_runtime.go:108-127``):
      POST /api/v1/sandboxes/{id}/snapshot  → SnapshotResult
      POST /api/v1/sandboxes/fork           → ForkResult
      POST /api/v1/sandboxes/{id}/restore   → None
      DELETE /api/v1/sandboxes/{id}         → None   (idempotent on 404)

    Identical surface to :class:`FleetSandboxProvider`, so the two providers are
    interchangeable — only the injected provider class differs. ``base_url`` /
    ``api_key`` default to ``MULTICA_BASE_URL`` / ``MULTICA_API_KEY``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        max_concurrent_forks: int | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError(
                "MulticaSweLegoProvider requires base_url or MULTICA_BASE_URL"
            )
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

        cap = max_concurrent_forks
        if cap is None:
            try:
                cap = int(os.environ.get("GROUP_SIZE", "2"))
            except ValueError:
                cap = 2
        if cap < 1:
            raise ValueError(f"max_concurrent_forks must be >= 1, got {cap}")
        self._fork_semaphore = asyncio.Semaphore(cap)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        try:
            resp = await self._client.post(
                f"/api/v1/sandboxes/{sandbox_id}/snapshot",
                headers=self._headers(),
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
                    "/api/v1/sandboxes/fork", json=payload, headers=self._headers()
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

    async def restore(self, sandbox_id: str) -> None:
        try:
            resp = await self._client.post(
                f"/api/v1/sandboxes/{sandbox_id}/restore", headers=self._headers()
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
                f"/api/v1/sandboxes/{sandbox_id}", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise EnvironmentError(f"cleanup transport error: {exc}") from exc
        if resp.status_code == 404:
            return
        if resp.status_code not in (200, 204):
            raise EnvironmentError(
                f"cleanup failed: status={resp.status_code} body={resp.text[:200]}"
            )

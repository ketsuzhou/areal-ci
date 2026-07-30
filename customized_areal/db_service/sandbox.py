"""Sandbox lifecycle and cleanup operations.

Ported from le-agent-dev/backend/core/infra/sandbox and
le-agent-dev/backend/core/tasks/service.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import httpx

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_DAYTONA_DELETE_TIMEOUT = max(_env_float("DAYTONA_DELETE_TIMEOUT", 120), 1.0)
_DAYTONA_DELETE_RETRIES = max(_env_int("DAYTONA_DELETE_RETRIES", 2), 0)
_DAYTONA_DELETE_RETRY_DELAY = max(_env_float("DAYTONA_DELETE_RETRY_DELAY", 2), 0.0)
_DAYTONA_DELETE_CONCURRENCY = max(_env_int("DAYTONA_DELETE_CONCURRENCY", 4), 1)
_SANDBOX_LOOKUP_RETRIES = max(_env_int("SANDBOX_LOOKUP_RETRIES", 2), 0)
_SANDBOX_LOOKUP_RETRY_DELAY = max(_env_float("SANDBOX_LOOKUP_RETRY_DELAY", 2), 0.0)
_SANDBOX_LOOKUP_CONCURRENCY = max(_env_int("SANDBOX_LOOKUP_CONCURRENCY", 8), 1)
_delete_semaphore = asyncio.Semaphore(_DAYTONA_DELETE_CONCURRENCY)
_lookup_semaphore = asyncio.Semaphore(_SANDBOX_LOOKUP_CONCURRENCY)
_TRANSIENT_LOOKUP_ERRORS = (
    TimeoutError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
)


def _get_daytona():
    """Return a lazy-initialized async Daytona client from env vars."""
    try:
        from daytona import AsyncDaytona, DaytonaConfig
    except ImportError as exc:
        raise RuntimeError(
            "daytona SDK is required for sandbox operations. "
            "Install it or ensure it is available in the environment."
        ) from exc

    if not hasattr(_get_daytona, "_client"):
        config = DaytonaConfig(
            api_key=os.environ.get("DAYTONA_API_KEY"),
            api_url=os.environ.get("DAYTONA_SERVER_URL"),
            target=os.environ.get("DAYTONA_TARGET"),
        )
        _get_daytona._client = AsyncDaytona(config)
    return _get_daytona._client


async def delete_sandbox(
    sandbox_id: str,
    *,
    timeout: float = _DAYTONA_DELETE_TIMEOUT,
    max_retries: int = _DAYTONA_DELETE_RETRIES,
) -> None:
    """Delete a Daytona sandbox by its ID."""
    daytona = _get_daytona()
    async with _delete_semaphore:
        for attempt in range(max_retries + 1):
            try:
                sandbox = await daytona.get(sandbox_id)
                await daytona.delete(sandbox, timeout=timeout)
                logger.info("Sandbox deleted: %s", sandbox_id)
                return
            except Exception as exc:
                if attempt == max_retries:
                    raise
                sleep_for = min(_DAYTONA_DELETE_RETRY_DELAY * (2**attempt), 30.0)
                logger.warning(
                    "Failed to delete sandbox_id=%s (attempt %d/%d, %s: %s); "
                    "retrying in %.1fs",
                    sandbox_id,
                    attempt + 1,
                    max_retries + 1,
                    type(exc).__name__,
                    exc,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)


def _sandbox_id_from_created(created) -> str | None:
    sandbox_id = getattr(created, "id", None)
    if sandbox_id:
        return str(sandbox_id)
    if isinstance(created, dict):
        sandbox_id = created.get("id") or created.get("sandbox_id")
        if sandbox_id:
            return str(sandbox_id)
    return None


async def clone_sandbox(source_sandbox_id: str) -> str | None:
    """Clone a Daytona sandbox if the installed SDK exposes a clone path."""
    daytona = _get_daytona()
    source = await daytona.get(source_sandbox_id)

    if hasattr(daytona, "clone"):
        cloned = await daytona.clone(source)
        return _sandbox_id_from_created(cloned)

    if hasattr(source, "_experimental_fork"):
        cloned = await source._experimental_fork()
        return _sandbox_id_from_created(cloned)

    if hasattr(source, "create_snapshot") and hasattr(daytona, "create"):
        from daytona import CreateSandboxFromSnapshotParams

        snapshot = await source.create_snapshot()
        snapshot_id = getattr(snapshot, "id", None) or str(snapshot)
        created = await daytona.create(
            CreateSandboxFromSnapshotParams(snapshot=snapshot_id)
        )
        return _sandbox_id_from_created(created)

    if hasattr(source, "_experimental_create_snapshot") and hasattr(daytona, "create"):
        from daytona import CreateSandboxFromSnapshotParams

        snapshot_id = f"branch-{source_sandbox_id}-{uuid.uuid4().hex}"
        await source._experimental_create_snapshot(snapshot_id)
        created = await daytona.create(
            CreateSandboxFromSnapshotParams(snapshot=snapshot_id)
        )
        return _sandbox_id_from_created(created)

    logger.warning(
        "Daytona sandbox clone/snapshot API unavailable for sandbox_id=%s",
        source_sandbox_id,
    )
    return None


async def bind_sandbox_to_task(
    client,
    *,
    sandbox_id: str,
    task_id: str,
    account_id: str,
) -> None:
    """Bind an existing Daytona sandbox id to a newly created task row."""
    await (
        client.table("sandboxes")
        .insert(
            {"sandbox_id": sandbox_id, "task_id": task_id, "account_id": account_id}
        )
        .execute()
    )


async def _lookup_sandbox_id_for_task(
    client,
    task_id: str,
    *,
    max_retries: int = _SANDBOX_LOOKUP_RETRIES,
) -> str | None:
    retry_delay = _SANDBOX_LOOKUP_RETRY_DELAY
    async with _lookup_semaphore:
        for attempt in range(max_retries + 1):
            try:
                sandbox_result = (
                    await client.table("sandboxes")
                    .select("sandbox_id")
                    .eq("task_id", task_id)
                    .maybe_single()
                    .execute()
                )
                if sandbox_result and sandbox_result.data:
                    return sandbox_result.data.get("sandbox_id")
                return None
            except _TRANSIENT_LOOKUP_ERRORS as exc:
                if attempt == max_retries:
                    raise
                sleep_for = min(retry_delay * (2**attempt), 30.0)
                logger.warning(
                    "Failed to look up sandbox for deletion: task_id=%s "
                    "(attempt %d/%d, %s: %s); retrying in %.1fs",
                    task_id,
                    attempt + 1,
                    max_retries + 1,
                    type(exc).__name__,
                    exc,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)


async def cleanup_sandbox_for_task(client, task_id: str) -> None:
    """Delete sandbox via Daytona API for a task.

    Only deletes the Daytona sandbox; DB records are left intact.
    """
    try:
        sandbox_id = await _lookup_sandbox_id_for_task(client, task_id)
        if sandbox_id:
            try:
                await delete_sandbox(sandbox_id)
            except Exception as exc:
                logger.warning(
                    "Failed to delete sandbox for task_id=%s sandbox_id=%s "
                    "after %d attempts (%s: %s)",
                    task_id,
                    sandbox_id,
                    _DAYTONA_DELETE_RETRIES + 1,
                    type(exc).__name__,
                    exc,
                )
    except Exception as exc:
        logger.warning(
            "Failed to look up sandbox for deletion: task_id=%s "
            "after %d attempts (%s: %s)",
            task_id,
            _SANDBOX_LOOKUP_RETRIES + 1,
            type(exc).__name__,
            exc,
        )

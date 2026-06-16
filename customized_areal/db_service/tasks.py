"""Task-related database operations."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_TASK_WRITE_RETRIES = max(_env_int("SUPABASE_TASK_WRITE_RETRIES", 2), 0)
_TASK_WRITE_RETRY_DELAY = max(_env_float("SUPABASE_TASK_WRITE_RETRY_DELAY", 1.0), 0.0)
_TRANSIENT_TASK_WRITE_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
)


class TaskStatus(StrEnum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


async def _execute_task_write_with_retry(
    query,
    *,
    operation: str,
    task_id: str | None = None,
    table: str = "tasks",
    max_retries: int = _TASK_WRITE_RETRIES,
) -> Any:
    """Execute a task-table write with bounded retry on transient transport errors."""
    retry_delay = _TASK_WRITE_RETRY_DELAY
    started_at = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "Starting DB write: table=%s operation=%s task_id=%s attempt=%d/%d",
                table,
                operation,
                task_id,
                attempt + 1,
                max_retries + 1,
            )
            result = await query.execute()
            logger.info(
                "Completed DB write: table=%s operation=%s task_id=%s attempt=%d/%d "
                "elapsed=%.3fs",
                table,
                operation,
                task_id,
                attempt + 1,
                max_retries + 1,
                time.monotonic() - started_at,
            )
            return result
        except _TRANSIENT_TASK_WRITE_ERRORS as exc:
            if attempt == max_retries:
                logger.error(
                    "DB write failed after retries: table=%s operation=%s task_id=%s "
                    "attempt=%d/%d elapsed=%.3fs error=%s: %s",
                    table,
                    operation,
                    task_id,
                    attempt + 1,
                    max_retries + 1,
                    time.monotonic() - started_at,
                    type(exc).__name__,
                    exc,
                )
                raise
            sleep_for = min(retry_delay * (2**attempt), 30.0)
            logger.warning(
                "Transient task write failure: table=%s operation=%s task_id=%s "
                "(attempt %d/%d, %s: %s); retrying in %.1fs",
                table,
                operation,
                task_id,
                attempt + 1,
                max_retries + 1,
                type(exc).__name__,
                exc,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)


async def create_task(
    *,
    client,
    account_id: str,
    parent_task_id: str | None = None,
    agent_id: str,
    agent_run_id: str | None = None,
    name: str | None = None,
    project_id: str | None = None,
) -> str:
    """Create a new task record.

    Args:
        client: Supabase client instance (from DBConnection.get_client())
        account_id: The account ID to associate with the task
        parent_task_id: Optional parent task ID for subtasks
        agent_id: The agent ID to associate with the task
        agent_run_id: Optional agent run ID
        name: Optional task name
        project_id: Optional project ID

    Returns:
        The task_id of the created task.

    Example:
        >>> db = DBConnection()
        >>> client = await db.get_client()
        >>> task_id = await create_task(
        ...     client=client,
        ...     account_id="user-123",
        ...     agent_id="agent-456",
        ...     name="My Task"
        ... )
    """
    task_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    task_data: dict[str, Any] = {
        "task_id": task_id,
        "account_id": account_id,
        "user_id": account_id,
        "status": TaskStatus.PENDING,
        "created_at": now,
    }

    if parent_task_id is not None:
        task_data["parent_task_id"] = parent_task_id
    task_data["agent_id"] = agent_id
    if agent_run_id is not None:
        task_data["agent_run_id"] = agent_run_id
    if name is not None:
        task_data["name"] = name
    if project_id is not None:
        task_data["project_id"] = project_id

    await _execute_task_write_with_retry(
        client.table("tasks").insert(task_data),
        operation="create_task",
        task_id=task_id,
        table="tasks",
    )

    return task_id


async def update_task_status(
    client,
    task_id: str,
    status: TaskStatus,
    *,
    error: str | None = None,
) -> None:
    """Update a task's status with appropriate timestamps.

    Args:
        client: Supabase client instance
        task_id: The task ID to update
        status: New task status
        error: Optional error message (for FAILED status)
    """
    now = datetime.now(UTC).isoformat()

    update_data: dict[str, Any] = {"status": status}

    if status == TaskStatus.RUNNING:
        update_data["started_at"] = now
    elif status in (TaskStatus.COMPLETED, TaskStatus.CANCELED, TaskStatus.FAILED):
        update_data["completed_at"] = now

    if error is not None:
        update_data["error"] = error

    await _execute_task_write_with_retry(
        client.table("tasks").update(update_data).eq("task_id", task_id),
        operation="update_task_status",
        task_id=task_id,
        table="tasks",
    )

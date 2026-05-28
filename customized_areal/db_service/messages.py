"""Message-related database operations."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from .connection import DBConnection

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0

# ── Table constants ──

TABLE_MESSAGES = "messages"
COLUMNS_PUBLIC = "message_id, role, content, created_at, updated_at"
COPY_EXCLUDED_COLUMNS = {"message_id", "task_id", "created_at", "updated_at"}

_db = DBConnection()


async def _get_client():
    """Get database client."""
    return await _db.get_client()


async def get_llm_messages(
    task_id: str,
    return_raw: bool = False,
) -> list[dict[str, Any]]:
    """Get all LLM messages for a task from the database.

    Args:
        task_id: The task ID to fetch messages for.
        return_raw: If True, return raw message content as-is. If False,
            parse messages for LLM consumption.

    Returns:
        A list of message dictionaries with 'role', 'content', and 'message_id' keys.

    Example:
        >>> messages = await get_llm_messages("task-123")
        >>> print(messages)
        [
            {"role": "user", "content": "Hello", "message_id": "msg-1"},
            {"role": "assistant", "content": "Hi!", "message_id": "msg-2"},
        ]
    """
    try:
        raw_messages = await _query_messages_by_task(task_id)
        if return_raw:
            return raw_messages
        return _parse_messages(raw_messages)
    except httpx.ReadTimeout as e:
        logger.error(
            "ReadTimeout fetching messages for task %s after retries: %s",
            task_id,
            e,
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(f"Failed to get messages for task {task_id}: {e}", exc_info=True)
        return []


async def add_message(
    task_id: str,
    role: str,
    content: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict | None:
    """Add a message to the messages table.

    Args:
        task_id: The task ID to associate the message with.
        role: The message role (e.g., 'user', 'assistant', 'system').
        content: The message content as a dictionary.
        metadata: Optional metadata dictionary.

    Example:
        >>> await add_message(
        ...     task_id="task-123",
        ...     role="user",
        ...     content={"role": "user", "content": "Hello!"},
        ... )
    """
    client = await _get_client()

    data: dict[str, Any] = {
        "task_id": task_id,
        "role": role,
        "content": content,
    }
    if metadata:
        data["metadata"] = metadata

    try:
        result = await client.table(TABLE_MESSAGES).insert(data).execute()
        if result.data and result.data[0].get("message_id"):
            return result.data[0]
        return None
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to add message: {e}", exc_info=True)
        raise


def truncate_messages_before_turn(
    raw_messages: list[dict[str, Any]],
    assistant_turn_idx: int,
) -> list[dict[str, Any]]:
    """Return sanitized raw messages before the 1-indexed assistant turn."""
    if assistant_turn_idx < 1:
        raise ValueError("assistant_turn_idx must be >= 1")

    assistant_count = 0
    prefix: list[dict[str, Any]] = []
    for row in raw_messages:
        if row.get("role") == "assistant":
            assistant_count += 1
            if assistant_count == assistant_turn_idx:
                return prefix
        copied = {
            key: value for key, value in row.items() if key not in COPY_EXCLUDED_COLUMNS
        }
        prefix.append(copied)
    raise ValueError(
        f"assistant turn {assistant_turn_idx} not found; found {assistant_count} assistant turns"
    )


async def copy_messages_to_task(
    client,
    *,
    task_id: str,
    messages: list[dict[str, Any]],
) -> None:
    """Copy raw messages to a task after stripping source DB identity columns."""
    if not messages:
        return

    rows = []
    for message in messages:
        row = {
            key: value
            for key, value in message.items()
            if key not in COPY_EXCLUDED_COLUMNS
        }
        row["task_id"] = task_id
        rows.append(row)
    await client.table(TABLE_MESSAGES).insert(rows).execute()


# ── Private: query helpers ──


async def _query_messages_by_task(
    task_id: str,
    order: str = "asc",
    include_metadata: bool = False,
) -> list[dict]:
    """Query messages by task_id with pagination."""
    client = await _get_client()
    columns = "*" if include_metadata else COLUMNS_PUBLIC
    all_messages: list[dict] = []
    batch_size = 1000
    offset = 0

    while True:
        query = client.table(TABLE_MESSAGES).select(columns).eq("task_id", task_id)
        result = await _execute_with_retry(
            query.order("created_at", desc=(order == "desc")).range(
                offset, offset + batch_size - 1
            ),
            task_id=task_id,
            offset=offset,
        )
        if not result.data:
            break
        all_messages.extend(result.data)
        if len(result.data) < batch_size:
            break
        offset += batch_size

    return all_messages


async def _execute_with_retry(
    query, *, task_id: str, offset: int, max_retries: int = _MAX_RETRIES
):
    """Execute a query with exponential backoff retry on transient timeouts."""
    for attempt in range(max_retries + 1):
        try:
            return await query.execute()
        except httpx.ReadTimeout:
            if attempt == max_retries:
                raise
            delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "ReadTimeout querying messages (task=%s, offset=%d), "
                "retry %d/%d in %.1fs",
                task_id,
                offset,
                attempt + 1,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)


def _parse_messages(raw_messages: list[dict]) -> list[dict[str, Any]]:
    """Parse raw DB messages into LLM-ready format.

    Handles:
    - JSON string content (legacy format)
    - Dict content (current JSONB format)
    - Empty user message filtering
    """
    messages = []
    for item in raw_messages:
        content = item.get("content")
        metadata = item.get("metadata", {})
        is_compressed = False

        # Handle compressed content in metadata
        if isinstance(metadata, dict) and metadata.get("compressed"):
            compressed_content = metadata.get("compressed_content")
            if compressed_content:
                if isinstance(compressed_content, dict):
                    compressed_content["message_id"] = item["message_id"]
                    messages.append(compressed_content)
                    continue
                else:
                    content = compressed_content
                    is_compressed = True

        if isinstance(content, str):
            try:
                parsed_item = json.loads(content)
                parsed_item["message_id"] = item["message_id"]
                # Filter empty user messages
                if parsed_item.get("role") == "user":
                    msg_content = parsed_item.get("content", "")
                    if isinstance(msg_content, str) and not msg_content.strip():
                        continue
                messages.append(parsed_item)
            except json.JSONDecodeError:
                if is_compressed:
                    messages.append(
                        {
                            "role": "user",
                            "content": content,
                            "message_id": item["message_id"],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "message_id": item["message_id"],
                        }
                    )
        elif isinstance(content, dict):
            content["message_id"] = item["message_id"]
            # Filter empty user messages
            if content.get("role") == "user":
                msg_content = content.get("content", "")
                if isinstance(msg_content, str) and not msg_content.strip():
                    continue
            messages.append(content)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": str(content),
                    "message_id": item["message_id"],
                }
            )

    return messages

"""TPFC backend task runner."""

# ruff: noqa: E402

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from httpx import Timeout

# Add parent of 'customized_areal' to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from supabase import create_async_client
from supabase.lib.client_options import AsyncClientOptions

from customized_areal.db_service import (
    AgentCreateRequest,
    AgentService,
    cleanup_sandbox_for_task,
    create_task,
    get_agent_loader,
)

# Auth/token layer (extracted into auth.py). Names without an alias are used
# directly by the orchestration logic below; the ``X as X`` re-exports keep auth
# helpers importable as ``backend_run.<name>`` for external callers and tests.
from customized_areal.db_service.auth import (
    DEFAULT_REFRESH_TOKEN,
    AuthTokenExpiredError,
    SharedTokenManager,
    _mint_legacy_hs256_auth_token,
    _safe_response_json,
    _token_diagnostics,
)
from customized_areal.db_service.auth import (
    _base64url_json as _base64url_json,
)
from customized_areal.db_service.auth import (
    _decode_jwt_header as _decode_jwt_header,
)
from customized_areal.db_service.auth import (
    _decode_jwt_payload as _decode_jwt_payload,
)
from customized_areal.db_service.auth import (
    _is_auth_token_usable as _is_auth_token_usable,
)
from customized_areal.db_service.auth import (
    _is_refresh_token_usable as _is_refresh_token_usable,
)
from customized_areal.db_service.auth import (
    _is_token_valid as _is_token_valid,
)
from customized_areal.db_service.auth import (
    _login_with_credentials as _login_with_credentials,
)
from customized_areal.db_service.auth import (
    _refresh_access_token as _refresh_access_token,
)
from customized_areal.db_service.connection import (
    _SUPABASE_CONNECT_TIMEOUT,
    _SUPABASE_KEEPALIVE_EXPIRY,
    _SUPABASE_POOL_TIMEOUT,
    _SUPABASE_READ_TIMEOUT,
    _SUPABASE_WRITE_TIMEOUT,
    _build_async_supabase_httpx_client,
    _describe_supabase_url,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("BackendRun")

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_AGENT_ID = os.environ.get("TPFC_AGENT_ID", "")
DEFAULT_USER_ID = os.environ.get("TPFC_USER_ID", "")
DEFAULT_BRIDGE_USER_ID = os.environ.get("TPFC_BRIDGE_USER_ID") or os.environ.get(
    "BRIDGE_USER_ID", ""
)
LE_AGENT_API_URL = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")

_AUTH_ERROR_STATUS_CODES = {401, 403}
_RUN_TIMEOUT = 1500
_TERMINAL_STATUSES = {"completed", "failed", "stopped", "canceled"}
TERMINAL_SSE_EVENTS = {"task_end", "error"}
_BRIDGE_USER_HEADER = "X-Bridge-User-Id"

_TRANSIENT_HTTP_ERRORS = (
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.WriteTimeout,
)


def _agent_start_http_timeout() -> Timeout:
    return Timeout(connect=30.0, read=_RUN_TIMEOUT, write=300.0, pool=60.0)


def _auth_headers(auth_token: str, user_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {auth_token}"}
    bridge_user_id = DEFAULT_BRIDGE_USER_ID or user_id or DEFAULT_USER_ID
    if bridge_user_id:
        headers[_BRIDGE_USER_HEADER] = bridge_user_id
    return headers


@dataclass(frozen=True)
class BackendRunResult:
    messages: list[dict[str, Any]]
    final_answer: str | None
    log_path: str
    task_id: str
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    trace: Any | None = None

    @property
    def _legacy_tuple(
        self,
    ) -> tuple[list[dict[str, Any]], str | None, str, Any | None]:
        return (self.messages, self.final_answer, self.log_path, self.trace)

    def __iter__(self):
        return iter(self._legacy_tuple)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return self._legacy_tuple[index]


def _prepare_form_data(
    task_id: str,
    task_description: str | None,
    agent_id: str,
    model_name: str | None,
    base_url: str | None,
    api_key: str | None,
    tags: list[str] | None,
) -> dict:
    """Build the multipart form data for the agent start API."""
    form_data = {
        "task_id": task_id,
        "prompt": task_description,
        "agent_id": agent_id,
        "skip_check_pending": True,
        "clean_filename": False,
    }
    if model_name is not None:
        form_data["model_name"] = model_name

    if base_url is not None:
        form_data["proxy_base_url"] = base_url
    if api_key is not None:
        form_data["proxy_api_key"] = api_key
    if tags is not None:
        form_data["tags"] = ",".join(tags)

    logger.info(
        "_prepare_form_data: proxy_base_url=%s, proxy_api_key=%s",
        form_data.get("proxy_base_url"),
        form_data.get("proxy_api_key")[:8] + "..."
        if form_data.get("proxy_api_key")
        else None,
    )
    return form_data


def _save_tpfc_agent_id(agent_id: str, env_path: Path = ENV_PATH) -> None:
    """Persist the TPFC agent ID for later backend runs."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"TPFC_AGENT_ID={agent_id}\n"

    if not env_path.exists():
        env_path.write_text(line, encoding="utf-8")
    else:
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        for idx, existing_line in enumerate(lines):
            if existing_line.lstrip().startswith("TPFC_AGENT_ID="):
                lines[idx] = line
                break
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] = f"{lines[-1]}\n"
            lines.append(line)
        env_path.write_text("".join(lines), encoding="utf-8")

    os.environ["TPFC_AGENT_ID"] = agent_id
    global DEFAULT_AGENT_ID
    DEFAULT_AGENT_ID = agent_id


async def _agent_exists(client, agent_id: str) -> bool:
    """Return whether the agent ID exists in the agents table."""
    result = (
        await client.table("agents")
        .select("agent_id")
        .eq("agent_id", agent_id)
        .limit(1)
        .execute()
    )
    return bool(result.data)


async def _resolve_agent_id(client, user_id: str, agent_id: str | None) -> str:
    """Return an existing agent_id or create and persist a default one."""
    agent_id = agent_id or os.environ.get("TPFC_AGENT_ID", "")
    if agent_id:
        if await _agent_exists(client, agent_id):
            return agent_id
        logger.warning(
            "Configured TPFC_AGENT_ID=%s was not found in database; creating a new agent",
            agent_id,
        )

    agent_service = AgentService(client)
    from customized_areal.tpfc.config.builtin import TPFC_CONFIG

    created_agent = await agent_service.create_agent(
        user_id,
        AgentCreateRequest(
            name=TPFC_CONFIG["name"],
            config=TPFC_CONFIG["config"],
            is_default=TPFC_CONFIG.get("is_default", False),
        ),
    )
    agent_id = created_agent.agent_id
    loader = await get_agent_loader()
    await loader.load_agent(agent_id, user_id, load_config=True)
    _save_tpfc_agent_id(agent_id)
    logger.info("Created agent: %s", agent_id)
    return agent_id


async def _start_agent_run(
    api_base_url: str,
    auth_token: str,
    form_data: dict,
    task_file_path: list[str] | None,
    user_id: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Start the agent run via HTTP and return the JSON response."""
    retry_delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=_agent_start_http_timeout()
            ) as http_client:
                files = []
                file_handles = []
                try:
                    if task_file_path:
                        for file_path in task_file_path:
                            if os.path.exists(file_path):
                                fh = open(file_path, "rb")
                                file_handles.append(fh)
                                files.append(
                                    ("files", (os.path.basename(file_path), fh, None))
                                )
                            else:
                                logger.warning("File not found: %s", file_path)

                    response = await http_client.post(
                        f"{api_base_url}/api/agent/start",
                        headers=_auth_headers(auth_token, user_id),
                        data=form_data,
                        files=files if files else None,
                    )
                finally:
                    for fh in file_handles:
                        fh.close()

                if response.status_code in _AUTH_ERROR_STATUS_CODES:
                    raise AuthTokenExpiredError(
                        f"Backend rejected auth token while starting agent run: "
                        f"{response.status_code} - {response.text}"
                    )

                if response.status_code != 200:
                    logger.error(
                        "Failed to start agent run via API: status_code=%s, response=%s",
                        response.status_code,
                        response.text,
                    )
                    raise RuntimeError(
                        f"Failed to start agent run: {response.status_code} - {response.text}"
                    )

                return _safe_response_json(response)
        except _TRANSIENT_HTTP_ERRORS as exc:
            if attempt < max_retries:
                logger.warning(
                    "Transient HTTP error starting agent run (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries + 1,
                    retry_delay,
                    exc,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0)
            else:
                raise


async def _start_agent_run_with_refresh(
    api_base_url: str,
    token_manager: SharedTokenManager,
    auth_token: str,
    form_data: dict,
    task_file_path: list[str] | None,
    user_id: str | None = None,
) -> tuple[dict, str]:
    """Start the agent run with refresh-token and credential-login fallbacks."""
    try:
        result = await _start_agent_run(
            api_base_url=api_base_url,
            auth_token=auth_token,
            form_data=form_data,
            task_file_path=task_file_path,
            user_id=user_id,
        )
        return result, auth_token
    except AuthTokenExpiredError:
        logger.warning(
            "Backend auth token expired while starting run; refreshing token"
        )
        fresh_token = await token_manager.get_valid_token(force_refresh=True)
        try:
            result = await _start_agent_run(
                api_base_url=api_base_url,
                auth_token=fresh_token,
                form_data=form_data,
                task_file_path=task_file_path,
                user_id=user_id,
            )
            return result, fresh_token
        except AuthTokenExpiredError:
            logger.warning(
                "Backend rejected refreshed auth token; attempting credential login"
            )
            login_token = await token_manager.login_and_store_token()
            try:
                result = await _start_agent_run(
                    api_base_url=api_base_url,
                    auth_token=login_token,
                    form_data=form_data,
                    task_file_path=task_file_path,
                    user_id=user_id,
                )
                return result, login_token
            except AuthTokenExpiredError:
                legacy_token = _mint_legacy_hs256_auth_token(login_token)
                if legacy_token:
                    logger.warning(
                        "Backend rejected login-issued token; trying HS256 "
                        "compatibility token"
                    )
                    try:
                        result = await _start_agent_run(
                            api_base_url=api_base_url,
                            auth_token=legacy_token,
                            form_data=form_data,
                            task_file_path=task_file_path,
                            user_id=user_id,
                        )
                        return result, legacy_token
                    except AuthTokenExpiredError:
                        pass
                raise AuthTokenExpiredError(
                    "Backend rejected both refreshed and login-issued Supabase "
                    f"tokens. Check that LE_AGENT_API_URL={api_base_url!r} uses "
                    "the same SUPABASE_URL/SUPABASE_ANON_KEY project as this "
                    f"client. Token diagnostics: {_token_diagnostics(login_token)}"
                )


async def _start_branch_agent_run_for_task(
    *,
    client,
    api_base_url: str,
    auth_token: str,
    task_id: str,
    account_id: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    """Start a run for an existing task whose messages are already in the DB."""
    del client
    endpoint = os.environ.get("LE_AGENT_BRANCH_RUN_ENDPOINT", "/api/agent/start-branch")
    url = (
        endpoint
        if endpoint.startswith(("http://", "https://"))
        else f"{api_base_url}{endpoint}"
    )
    async with httpx.AsyncClient(timeout=_agent_start_http_timeout()) as http_client:
        request_body = {
            "task_id": task_id,
            "model_name": model_name,
            "proxy_base_url": base_url,
            "proxy_api_key": api_key,
            "stream": False,
        }
        logger.info(
            "_start_branch_agent_run_for_task: proxy_base_url=%s, proxy_api_key=%s",
            base_url,
            api_key[:8] + "..." if api_key else None,
        )
        response = await http_client.post(
            url,
            headers=_auth_headers(auth_token, account_id),
            json=request_body,
        )
    if response.status_code in _AUTH_ERROR_STATUS_CODES:
        raise AuthTokenExpiredError(
            f"Backend rejected auth token while starting branch task run: "
            f"{response.status_code} - {response.text}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to start branch task run: {response.status_code} - {response.text}"
        )
    return _safe_response_json(response)


async def _start_branch_agent_run_for_task_with_refresh(
    *,
    client,
    api_base_url: str,
    token_manager: SharedTokenManager,
    auth_token: str,
    task_id: str,
    account_id: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[dict[str, Any], str]:
    """Start a branch task run with the same auth fallbacks as the normal path."""

    async def _start_with_token(token: str) -> dict[str, Any]:
        return await _start_branch_agent_run_for_task(
            client=client,
            api_base_url=api_base_url,
            auth_token=token,
            task_id=task_id,
            account_id=account_id,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )

    try:
        return await _start_with_token(auth_token), auth_token
    except AuthTokenExpiredError:
        logger.warning(
            "Backend auth token expired while starting branch run; refreshing token"
        )
        fresh_token = await token_manager.get_valid_token(force_refresh=True)
        try:
            return await _start_with_token(fresh_token), fresh_token
        except AuthTokenExpiredError:
            logger.warning(
                "Backend rejected refreshed auth token for branch run; "
                "attempting credential login"
            )
            login_token = await token_manager.login_and_store_token()
            try:
                return await _start_with_token(login_token), login_token
            except AuthTokenExpiredError:
                legacy_token = _mint_legacy_hs256_auth_token(login_token)
                if legacy_token:
                    logger.warning(
                        "Backend rejected login-issued token for branch run; "
                        "trying HS256 compatibility token"
                    )
                    try:
                        return await _start_with_token(legacy_token), legacy_token
                    except AuthTokenExpiredError:
                        pass
                raise AuthTokenExpiredError(
                    "Backend rejected refreshed and login-issued tokens while "
                    f"starting branch run. Token diagnostics: "
                    f"{_token_diagnostics(login_token)}"
                )


def _task_stream_url(api_base_url: str, task_id: str, auth_token: str) -> str:
    return (
        f"{api_base_url}/api/tasks/{task_id}/stream?token={quote(auth_token, safe='')}"
    )


async def _wait_for_agent_run(
    client,
    task_id: str,
    agent_run_id: str | None,
    api_base_url: str | None = None,
    auth_token: str | None = None,
    token_manager: SharedTokenManager | None = None,
    timeout: int = 9000,
) -> str:
    """Wait until the agent run reaches a terminal state.

    If *api_base_url* and *auth_token* are provided, attempts to consume the
    task-level SSE stream from the backend first with auto-reconnect. Falls
    back to database polling if the stream endpoint is unavailable or the
    stream ends without a terminal status.
    """
    start_time = time.time()
    status = "pending"
    streamed = False
    last_event_id: str | None = None

    def _time_left() -> float:
        return timeout - (time.time() - start_time)

    if api_base_url and auth_token:
        sse_retry_delay = 1.0

        while _time_left() > 0:
            stream_url = _task_stream_url(api_base_url, task_id, auth_token)
            headers: dict[str, str] = {}
            if last_event_id is not None:
                headers["last-event-id"] = last_event_id

            try:
                async with httpx.AsyncClient(
                    timeout=_time_left() + 10.0
                ) as http_client:
                    async with http_client.stream(
                        "GET",
                        stream_url,
                        headers=headers,
                        timeout=_time_left() + 10.0,
                    ) as response:
                        if response.status_code == 200:
                            streamed = True
                            sse_retry_delay = 1.0
                            current_event = "message"
                            current_data_parts: list[str] = []

                            async for raw_line in response.aiter_lines():
                                if _time_left() <= 0:
                                    break

                                line = raw_line.strip()
                                # SSE comment (e.g. keepalive) — ignore
                                if line.startswith(":"):
                                    continue
                                if line.startswith("id:"):
                                    last_event_id = line[3:].strip() or last_event_id
                                    continue
                                if line.startswith("event:"):
                                    current_event = line[6:].strip()
                                    continue
                                if line.startswith("data:"):
                                    current_data_parts.append(line[5:].strip())
                                    continue
                                # Empty line = end of SSE message
                                if line == "":
                                    if current_data_parts:
                                        data_str = "\n".join(current_data_parts)
                                        current_data_parts = []
                                        try:
                                            event = json.loads(data_str)
                                        except json.JSONDecodeError:
                                            event = {}

                                        # Derive status from event data when available
                                        event_status = event.get("status")
                                        if event_status:
                                            status = event_status

                                        # Terminal event types close the stream
                                        if current_event in TERMINAL_SSE_EVENTS:
                                            # task_end carries status; error is a failure
                                            if current_event == "error":
                                                status = "failed"
                                            break

                                        logger.debug(
                                            "SSE event: type=%s status=%s task_id=%s",
                                            current_event,
                                            status,
                                            task_id,
                                        )
                                    current_event = "message"

                            if status in _TERMINAL_STATUSES:
                                break
                            # Stream ended without terminal event — may need reconnect
                            if current_event not in TERMINAL_SSE_EVENTS:
                                logger.warning(
                                    "SSE stream ended for task_id=%s, reconnecting in %.1fs",
                                    task_id,
                                    sse_retry_delay,
                                )
                                await asyncio.sleep(min(sse_retry_delay, _time_left()))
                                sse_retry_delay = min(sse_retry_delay * 2, 30.0)
                                continue
                        if (
                            response.status_code in _AUTH_ERROR_STATUS_CODES
                            and token_manager is not None
                        ):
                            logger.warning(
                                "SSE stream auth failed for task_id=%s; refreshing token",
                                task_id,
                            )
                            auth_token = await token_manager.get_valid_token(
                                force_refresh=True
                            )
                            streamed = False
                            sse_retry_delay = 1.0
                            continue
                        else:
                            logger.warning(
                                "SSE stream endpoint returned %s for task_id=%s, falling back to polling",
                                response.status_code,
                                task_id,
                            )
                            break
            except httpx.ConnectError as exc:
                logger.warning(
                    "SSE connect error for task_id=%s, retrying in %.1fs: %s",
                    task_id,
                    sse_retry_delay,
                    exc,
                )
                await asyncio.sleep(min(sse_retry_delay, _time_left()))
                sse_retry_delay = min(sse_retry_delay * 2, 30.0)
                continue
            except Exception as exc:
                logger.warning(
                    "SSE error for task_id=%s, retrying in %.1fs: %s",
                    task_id,
                    sse_retry_delay,
                    exc,
                )
                await asyncio.sleep(min(sse_retry_delay, _time_left()))
                sse_retry_delay = min(sse_retry_delay * 2, 30.0)
                continue

    if not streamed or status not in _TERMINAL_STATUSES:
        retry_delay = 1.0
        while _time_left() > 0:
            try:
                if agent_run_id:
                    agent_run = (
                        await client.table("agent_runs")
                        .select("status, error, completed_at")
                        .eq("id", agent_run_id)
                        .single()
                        .execute()
                    )
                    run_data = getattr(agent_run, "data", None)
                    if isinstance(run_data, dict) and "status" in run_data:
                        status = run_data["status"]
                else:
                    # No agent_run_id (queued status) — poll task for active run
                    task_row = (
                        await client.table("tasks")
                        .select("status")
                        .eq("task_id", task_id)
                        .single()
                        .execute()
                    )
                    task_data = getattr(task_row, "data", None)
                    if isinstance(task_data, dict) and "status" in task_data:
                        task_status = task_data["status"]
                        if task_status in _TERMINAL_STATUSES:
                            status = task_status
            except Exception as exc:
                if _time_left() <= 0:
                    break
                sleep_for = min(retry_delay, _time_left())
                logger.warning(
                    "DB poll failed for task_id=%s, retrying in %.1fs: %s",
                    task_id,
                    sleep_for,
                    exc,
                )
                await asyncio.sleep(sleep_for)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            if status in _TERMINAL_STATUSES:
                break

            if _time_left() <= 0:
                break
            await asyncio.sleep(min(30.0, _time_left()))
            retry_delay = 1.0
    if status not in _TERMINAL_STATUSES:
        logger.error(
            "Timeout waiting for agent run to complete: task_id=%s",
            task_id,
        )
        raise TimeoutError(
            f"Agent run {agent_run_id} did not complete within {timeout} seconds"
        )

    return status


async def _get_llm_messages_with_client(client, task_id: str) -> list[dict[str, Any]]:
    """Fetch raw LLM messages using the caller-owned Supabase client."""
    all_messages: list[dict[str, Any]] = []
    batch_size = 1000
    offset = 0

    while True:
        query = (
            client.table("messages")
            .select("message_id, role, content, created_at, updated_at")
            .eq("task_id", task_id)
            .order("created_at", desc=False)
            .range(offset, offset + batch_size - 1)
        )
        result = await _execute_message_query_with_retry(
            query, task_id=task_id, offset=offset
        )
        data = getattr(result, "data", None) or []
        all_messages.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    return all_messages


async def _get_raw_messages_with_client(client, task_id: str) -> list[dict[str, Any]]:
    all_messages: list[dict[str, Any]] = []
    batch_size = 1000
    offset = 0

    while True:
        query = (
            client.table("messages")
            .select("message_id, role, content, created_at, updated_at, metadata")
            .eq("task_id", task_id)
            .order("created_at", desc=False)
            .range(offset, offset + batch_size - 1)
        )
        result = await _execute_message_query_with_retry(
            query, task_id=task_id, offset=offset
        )
        data = getattr(result, "data", None) or []
        if not data:
            break
        all_messages.extend(data)
        offset += batch_size

    return all_messages


async def _execute_message_query_with_retry(
    query, *, task_id: str, offset: int, max_retries: int = 3
):
    """Execute a message query with retry on transient read failures."""
    retry_delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return await query.execute()
        except _TRANSIENT_HTTP_ERRORS:
            if attempt == max_retries:
                raise
            logger.warning(
                "Transient error querying messages (task=%s, offset=%d), "
                "retry %d/%d in %.1fs",
                task_id,
                offset,
                attempt + 1,
                max_retries,
                retry_delay,
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30.0)


async def _get_terminal_error(client, task_id: str, agent_run_id: str | None) -> str:
    """Return a concise backend error for a terminal non-success status."""
    try:
        if agent_run_id:
            result = (
                await client.table("agent_runs")
                .select("error, status")
                .eq("id", agent_run_id)
                .maybe_single()
                .execute()
            )
            data = getattr(result, "data", None)
            if isinstance(data, dict) and data.get("error"):
                return str(data["error"])

        result = (
            await client.table("tasks")
            .select("error, status")
            .eq("task_id", task_id)
            .maybe_single()
            .execute()
        )
        data = getattr(result, "data", None)
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except Exception as exc:
        logger.warning(
            "Failed to fetch terminal error for task_id=%s: %s",
            task_id,
            exc,
        )

    return "no backend error details available"


def _extract_final_answer(messages: list[dict]) -> str | None:
    """Extract text inside the first <answer> tag from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("content", "")
        elif isinstance(content, list):
            content = "".join(
                p.get("text", p.get("content", "")) if isinstance(p, dict) else str(p)
                for p in content
            )

        if isinstance(content, str):
            matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if matches:
                return matches[-1].strip()

    return None


async def _create_shortlived_db_client():
    """Create a fresh Supabase async client with a per-run connection pool.

    The returned client (and its underlying httpx pool) should be closed by the
    caller via ``await _close_db_client(client)`` once the run finishes.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
        "SUPABASE_ANON_KEY"
    )
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and a key (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY) "
            "environment variables must be set."
        )
    max_connections = 10
    max_keepalive_connections = 5
    httpx_client = _build_async_supabase_httpx_client(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )
    logger.info(
        "Creating short-lived Supabase client for backend run: url=%s trust_env=%s "
        "max_connections=%s max_keepalive=%s keepalive_expiry=%s timeout=(connect=%s read=%s write=%s pool=%s)",
        _describe_supabase_url(supabase_url),
        False,
        max_connections,
        max_keepalive_connections,
        _SUPABASE_KEEPALIVE_EXPIRY,
        _SUPABASE_CONNECT_TIMEOUT,
        _SUPABASE_READ_TIMEOUT,
        _SUPABASE_WRITE_TIMEOUT,
        _SUPABASE_POOL_TIMEOUT,
    )
    options = AsyncClientOptions(httpx_client=httpx_client)
    return await create_async_client(supabase_url, supabase_key, options)


async def _close_db_client(client) -> None:
    """Close a Supabase async client and its underlying httpx pool."""
    try:
        httpx_client = getattr(getattr(client, "options", None), "httpx_client", None)
        if httpx_client is not None:
            await httpx_client.aclose()
    except Exception as exc:
        logger.warning("Error closing DB client: %s", exc)


async def run_backend(
    task_description: str | None,
    task_file_path: list[str] | None,
    log_path: str = "./log.json",
    task_id: str = "",
    gt: str = "",
    tags: list[str] | None = None,
    user_id: str | None = None,
    model_name: str | None = None,
    server_manager=None,
    tokenizer=None,
    agent_id: str | None = DEFAULT_AGENT_ID,
    base_url: str | None = None,
    api_key: str | None = None,
    refresh_token: str | None = DEFAULT_REFRESH_TOKEN,
    seed_messages_already_inserted: bool = False,
):
    # Per-run client: avoids exhausting the singleton pool under concurrent rollouts
    client = await _create_shortlived_db_client()
    terminal_status: str | None = None
    agent_started = False

    user_id = user_id or DEFAULT_USER_ID
    if not user_id:
        await _close_db_client(client)
        raise ValueError(
            "user_id is required. Set TPFC_USER_ID env var or pass user_id argument."
        )
    if seed_messages_already_inserted:
        if not task_id:
            await _close_db_client(client)
            raise ValueError(
                "task_id is required when seed_messages_already_inserted=True"
            )
        if not model_name:
            await _close_db_client(client)
            raise ValueError(
                "model_name is required when seed_messages_already_inserted=True"
            )

    try:
        token_manager = SharedTokenManager(
            refresh_token=refresh_token or DEFAULT_REFRESH_TOKEN
        )
        auth_token = await token_manager.get_valid_token()

        resolved_agent_id = None
        if not seed_messages_already_inserted:
            resolved_agent_id = await _resolve_agent_id(client, user_id, agent_id)

            task_id = await create_task(
                client=client,
                account_id=user_id,
                agent_id=resolved_agent_id,
                name=task_description[:100] if task_description else None,
            )
            logger.info("Task created: %s", task_id)
            print("Task created: %s", task_id)

        async def _do_run():
            nonlocal agent_started, terminal_status, auth_token

            if seed_messages_already_inserted:
                (
                    result,
                    auth_token,
                ) = await _start_branch_agent_run_for_task_with_refresh(
                    client=client,
                    api_base_url=LE_AGENT_API_URL,
                    token_manager=token_manager,
                    auth_token=auth_token,
                    task_id=task_id,
                    account_id=user_id,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
                logger.info("Branch agent run started for task: %s", result)
            else:
                form_data = _prepare_form_data(
                    task_id=task_id,
                    task_description=task_description,
                    agent_id=resolved_agent_id,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    tags=tags,
                )

                result, auth_token = await _start_agent_run_with_refresh(
                    api_base_url=LE_AGENT_API_URL,
                    token_manager=token_manager,
                    auth_token=auth_token,
                    form_data=form_data,
                    task_file_path=task_file_path,
                    user_id=user_id,
                )
                logger.info("Agent run started via API: %s", result)
            agent_started = True

            agent_run_id = result.get("agent_run_id")
            start_status = result.get("status")

            if start_status == "queued" and not agent_run_id:
                logger.warning(
                    "Agent run queued (slot occupied) for task_id=%s, waiting via task stream",
                    task_id,
                )

            terminal_status = await _wait_for_agent_run(
                client,
                task_id=task_id,
                agent_run_id=agent_run_id,
                api_base_url=LE_AGENT_API_URL,
                auth_token=auth_token,
                token_manager=token_manager,
                timeout=_RUN_TIMEOUT,
            )

            if terminal_status != "completed":
                error = await _get_terminal_error(client, task_id, agent_run_id)
                raise RuntimeError(
                    f"Agent run ended with status={terminal_status!r} "
                    f"for task_id={task_id}, backend_error={error!r}"
                )

            messages = await _get_llm_messages_with_client(client, task_id)
            raw_messages = await _get_raw_messages_with_client(client, task_id)
            final_boxed_answer = _extract_final_answer(messages)

            return BackendRunResult(
                messages=messages,
                raw_messages=raw_messages,
                final_answer=final_boxed_answer,
                log_path=log_path,
                task_id=task_id,
            )

        try:
            return await asyncio.wait_for(_do_run(), timeout=_RUN_TIMEOUT)
        except TimeoutError:
            logger.warning(
                "TPFCAgent run timed out after %.0f minutes for task_id=%s",
                _RUN_TIMEOUT / 60,
                task_id,
            )
            raise
        finally:
            # Guaranteed to run on normal exit, exception, or CancelledError
            # (the latter fires when asyncio.run() is cancelled by SIGINT/Ctrl+C).
            if terminal_status in _TERMINAL_STATUSES or not agent_started:
                await cleanup_sandbox_for_task(client, task_id)
            else:
                logger.warning(
                    "Skipping sandbox cleanup for task_id=%s because the backend "
                    "run may still be active (last_status=%s)",
                    task_id,
                    terminal_status,
                )
    finally:
        await _close_db_client(client)


if __name__ == "__main__":
    task_description = (
        "The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. "
        "What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet."
    )
    task_file_path = [
        "/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/dataset/gaia-benchmark/gaia/2023/validation/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx"
    ]
    gt = "Time-Parking 2: Parallel Universe"

    messages, final_answer, log_path, _trace = asyncio.run(
        run_backend(
            task_description=task_description,
            task_file_path=task_file_path,
            gt=gt,
            tags=["debug", "0421"],
            user_id=DEFAULT_USER_ID,
            model_name="areal/qwen/qwen3_5-9b",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL"),
            refresh_token=DEFAULT_REFRESH_TOKEN,
        )
    )

    print("Messages:", messages)
    print("Final boxed answer:", final_answer)
    print("Log path:", log_path)

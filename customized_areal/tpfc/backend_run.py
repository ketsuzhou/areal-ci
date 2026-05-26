import asyncio
import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

# Add parent of 'customized_areal' to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

from customized_areal.db_service import (
    AgentCreateRequest,
    AgentService,
    DBConnection,
    cleanup_sandbox_for_task,
    create_task,
    get_agent_loader,
    get_llm_messages,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Initialize logger
try:
    from areal.utils.logging import getLogger

    logger = getLogger("BackendRun")
except ImportError:
    import logging

    logger = logging.getLogger("BackendRun")

DEFAULT_REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN", "")
DEFAULT_AGENT_ID = os.environ.get("TPFC_AGENT_ID", "")

DEFAULT_USER_ID = os.environ.get("TPFC_USER_ID", "")
SUPABASE_AUTH_EMAIL = os.environ.get("SUPABASE_AUTH_EMAIL", "")
SUPABASE_AUTH_PASSWORD = os.environ.get("SUPABASE_AUTH_PASSWORD", "")
LE_AGENT_API_URL = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")
TOKEN_REFRESH_MARGIN = 60  # seconds; refresh shortly before expiry.
_REFRESH_WAIT_TIMEOUT = 30  # seconds to wait for another process to finish refreshing
_REFRESH_WAIT_INTERVAL = 1  # seconds between token re-reads while waiting
_AUTH_ERROR_STATUS_CODES = {401, 403}
_TOKEN_HTTP_TIMEOUT = 30.0

_SHARED_TOKEN_FILE = Path(__file__).parent / ".shared_auth_token.json"


class _FileLock:
    """Context manager for fcntl file locking."""

    def __init__(self, fd: Any, lock_type: int):
        self._fd = fd
        self._lock_type = lock_type

    def __enter__(self) -> None:
        fcntl.flock(self._fd, self._lock_type)

    def __exit__(self, *args: Any) -> bool:
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        return False


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verifying signature."""
    if not token:
        return {}

    try:
        token_parts = token.split(".")
        if len(token_parts) < 2:
            return {}

        payload_b64 = token_parts[1]
        padding_needed = 4 - len(payload_b64) % 4
        if padding_needed != 4:
            payload_b64 += "=" * padding_needed
        payload_json = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_json)
    except Exception as e:
        logger.warning("Failed to decode JWT payload: %s", e)
        return {}


def _decode_jwt_header(token: str) -> dict:
    """Decode JWT header without verifying signature."""
    if not token:
        return {}

    try:
        token_parts = token.split(".")
        if len(token_parts) < 1:
            return {}

        header_b64 = token_parts[0]
        padding_needed = 4 - len(header_b64) % 4
        if padding_needed != 4:
            header_b64 += "=" * padding_needed
        header_json = base64.urlsafe_b64decode(header_b64)
        return json.loads(header_json)
    except Exception as e:
        logger.warning("Failed to decode JWT header: %s", e)
        return {}


def _is_token_valid(token: str, margin: int = TOKEN_REFRESH_MARGIN) -> bool:
    """Return True if *token* is still valid for at least *margin* seconds."""
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    return isinstance(exp, int | float) and exp >= time.time() + margin


def _is_auth_token_usable(token: str, margin: int = TOKEN_REFRESH_MARGIN) -> bool:
    """Return True for LeAgent-compatible Supabase access tokens."""
    header = _decode_jwt_header(token)
    return (
        header.get("alg") == "ES256"
        and isinstance(header.get("kid"), str)
        and _is_token_valid(token, margin=margin)
    )


def _is_refresh_token_usable(refresh_token: str | None) -> bool:
    """Return False for missing or locally expired JWT refresh tokens.

    Supabase refresh tokens are usually opaque strings, so a non-JWT token is
    considered usable and validated by the refresh endpoint. If a deployment
    uses JWT-shaped refresh tokens, check the local expiry before calling the
    endpoint.
    """
    if not refresh_token:
        return False
    if refresh_token.count(".") != 2:
        return True
    return _is_token_valid(refresh_token, margin=0)


def _base64url_json(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode()


def _base64url_decode(data: str) -> bytes:
    padding_needed = 4 - len(data) % 4
    if padding_needed != 4:
        data += "=" * padding_needed
    return base64.urlsafe_b64decode(data)


def _verify_hs256_signature(token: str, jwt_secret: str) -> bool:
    """Return True when token is an HS256 JWT signed by jwt_secret."""
    try:
        header = _decode_jwt_header(token)
        if header.get("alg") != "HS256":
            return False

        signing_input, signature_b64 = token.rsplit(".", 1)
        expected_signature = hmac.new(
            jwt_secret.encode(),
            signing_input.encode(),
            hashlib.sha256,
        ).digest()
        actual_signature = _base64url_decode(signature_b64)
        return hmac.compare_digest(expected_signature, actual_signature)
    except (ValueError, OSError):
        return False


def _configured_jwt_secret_valid(jwt_secret: str) -> bool:
    """Check whether SUPABASE_JWT_SECRET matches configured Supabase keys."""
    configured_tokens = [
        os.environ.get("SUPABASE_ANON_KEY", ""),
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    ]
    return any(
        token and _verify_hs256_signature(token, jwt_secret)
        for token in configured_tokens
    )


def _mint_legacy_hs256_auth_token(access_token: str) -> str | None:
    """Mint an HS256 compatibility token from a validated Supabase access token.

    Some older LeAgent deployments validate Supabase JWTs locally with
    SUPABASE_JWT_SECRET and only accept HS256 tokens. Newer Supabase projects can
    issue ES256 access tokens. This fallback keeps the same claims, signs them
    with the configured legacy secret, and is only used after a real Supabase
    refresh/login has succeeded but LeAgent still rejects the token.
    """
    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        return None
    if not _configured_jwt_secret_valid(jwt_secret):
        logger.warning(
            "SUPABASE_JWT_SECRET does not verify configured Supabase keys; "
            "skipping HS256 compatibility token"
        )
        return None

    claims = _decode_jwt_payload(access_token)
    if not claims or not _is_token_valid(access_token, margin=0):
        return None

    now = int(time.time())
    claims["iat"] = now
    claims["exp"] = min(int(claims.get("exp", now + 3600)), now + 3600)
    claims.setdefault("aud", "authenticated")
    claims.setdefault("role", "authenticated")

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_base64url_json(header)}.{_base64url_json(claims)}"
    signature = hmac.new(
        jwt_secret.encode(),
        signing_input.encode(),
        hashlib.sha256,
    ).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{signing_input}.{signature_b64}"


class AuthTokenExpiredError(RuntimeError):
    """Raised when the backend rejects an auth token."""


class SharedTokenManager:
    """File-based auth token manager for multi-process sharing.

    Stores the access_token in a JSON file so multiple processors can
    read the same token. When a processor detects the token is expired,
    it refreshes the token via _refresh_access_token and writes the new
    token back to the file.

    Three-layer locking:
    - asyncio.Lock: serializes within a single process (intra-process coroutines)
    - Refresh-in-progress lock file (.refresh.lock): serializes the full refresh
      lifecycle across processes, preventing refresh storms and single-use
      refresh-token failures
    - Shared/exclusive fcntl on the token file: guards reads/writes of the
      token file itself (atomic via os.replace, so readers never see partial writes)
    """

    _async_lock: asyncio.Lock | None = None

    def __init__(
        self,
        token_file: Path = _SHARED_TOKEN_FILE,
        refresh_token: str = DEFAULT_REFRESH_TOKEN,
    ):
        self.token_file = token_file
        self.refresh_token = refresh_token
        # One async lock per class (shared across instances in the same process)
        if SharedTokenManager._async_lock is None:
            SharedTokenManager._async_lock = asyncio.Lock()

    def read_token_payload(self) -> dict[str, Any] | None:
        """Read the shared token payload from file."""
        try:
            with open(self.token_file) as f, _FileLock(f, fcntl.LOCK_SH):
                data = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read shared auth token file %s: %s", self.token_file, exc
            )
            return None

        if not isinstance(data, dict):
            logger.warning(
                "Ignoring invalid shared auth token payload in %s", self.token_file
            )
            return None

        refresh_token = data.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            self.refresh_token = refresh_token
        return data

    def read_token(self) -> str | None:
        """Read the shared access token from file."""
        data = self.read_token_payload()
        if data is None:
            return None

        access_token = data.get("access_token")
        return access_token if isinstance(access_token, str) and access_token else None

    def write_token(self, access_token: str, refresh_token: str | None = None) -> None:
        """Write the access token (and optionally a new refresh token) to the shared file atomically."""
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"access_token": access_token, "updated_at": time.time()}
        if refresh_token:
            payload["refresh_token"] = refresh_token
            self.refresh_token = refresh_token
        tmp = self.token_file.with_name(f".{self.token_file.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.token_file)
        self._fsync_parent_dir()
        logger.info("Shared auth token updated in file: %s", self.token_file)

    async def get_valid_token(self, force_refresh: bool = False) -> str:
        """Get a valid access token, refreshing if necessary.

        Uses a refresh-in-progress lock file to serialize the full refresh
        lifecycle across processes. If another process holds the lock, this
        process waits and re-reads the token file (the other process likely
        just refreshed it).
        """
        auth_token = self.read_token()
        if not force_refresh and auth_token and _is_auth_token_usable(auth_token):
            return auth_token

        async with SharedTokenManager._async_lock:
            # Double-check after acquiring async lock.
            auth_token = self.read_token()
            if not force_refresh and auth_token and _is_auth_token_usable(auth_token):
                return auth_token

            if force_refresh:
                logger.info("Shared auth token rejected by backend, refreshing...")
            else:
                logger.info(
                    "Shared auth token expired or about to expire, refreshing..."
                )

            # Acquire refresh-in-progress lock; wait if another process is refreshing.
            lock_fd = self._try_acquire_refresh_lock()
            if lock_fd is None:
                # Another process is refreshing — wait for it to finish, then re-read.
                auth_token = await self._wait_for_valid_token(
                    force_refresh=force_refresh
                )
                if auth_token:
                    return auth_token
                # Still invalid after waiting — try acquiring lock again.
                lock_fd = self._try_acquire_refresh_lock()
                if lock_fd is None:
                    auth_token = await self._wait_for_valid_token(
                        force_refresh=force_refresh
                    )
                    if auth_token:
                        return auth_token
                    raise RuntimeError(
                        "Failed to acquire refresh lock and no valid token available"
                    )

            try:
                # Re-read under refresh lock — another process may have refreshed
                # between our first read and lock acquisition.
                auth_token = self.read_token()
                if (
                    not force_refresh
                    and auth_token
                    and _is_auth_token_usable(auth_token)
                ):
                    return auth_token

                new_token, new_refresh = await _refresh_access_token(self.refresh_token)
                self.write_token(new_token, new_refresh)
                return new_token
            finally:
                self._release_refresh_lock(lock_fd)

    async def login_and_store_token(self) -> str:
        """Get fresh tokens via credentials, bypassing the refresh token."""
        async with SharedTokenManager._async_lock:
            lock_fd = self._try_acquire_refresh_lock()
            if lock_fd is None:
                auth_token = await self._wait_for_valid_token(force_refresh=True)
                if auth_token:
                    return auth_token
                lock_fd = self._try_acquire_refresh_lock()
                if lock_fd is None:
                    raise RuntimeError(
                        "Failed to acquire refresh lock for credential login"
                    )

            try:
                access_token, refresh_token = await _login_with_env_credentials()
                self.write_token(access_token, refresh_token)
                return access_token
            finally:
                self._release_refresh_lock(lock_fd)

    @property
    def _refresh_lock_file(self) -> Path:
        return self.token_file.with_suffix(".refresh.lock")

    def _try_acquire_refresh_lock(self) -> int | None:
        """Non-blocking acquire of the refresh-in-progress lock file.

        Returns the fd on success, or None if another process holds the lock.
        """
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._refresh_lock_file, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (OSError, BlockingIOError):
            try:
                os.close(fd)
            except (NameError, OSError):
                pass
            return None

    def _release_refresh_lock(self, fd: int) -> None:
        """Release the refresh-in-progress lock."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except OSError:
            pass

    async def _wait_for_valid_token(self, force_refresh: bool = False) -> str | None:
        """Poll the token file until a valid token appears or timeout."""
        deadline = time.monotonic() + _REFRESH_WAIT_TIMEOUT
        current_token = self.read_token() if force_refresh else None
        while time.monotonic() < deadline:
            await asyncio.sleep(_REFRESH_WAIT_INTERVAL)
            auth_token = self.read_token()
            if (
                auth_token
                and _is_auth_token_usable(auth_token)
                and (not force_refresh or auth_token != current_token)
            ):
                return auth_token
        return None

    def _fsync_parent_dir(self) -> None:
        """Best-effort fsync for durability of the atomic token-file replace."""
        try:
            dir_fd = os.open(self.token_file.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass


async def _refresh_access_token(refresh_token: str) -> tuple[str, str]:
    """Refresh Supabase access token using refresh_token.

    Returns (access_token, refresh_token). Falls back to email/password login
    if the refresh token is expired or invalid.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_anon_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set to refresh token"
        )

    async with httpx.AsyncClient(timeout=_TOKEN_HTTP_TIMEOUT) as client:
        if not _is_refresh_token_usable(refresh_token):
            logger.warning(
                "REFRESH_TOKEN is missing or locally expired; attempting email/password login"
            )
            return await _login_with_credentials(
                supabase_url, supabase_anon_key, client
            )

        try:
            resp = await client.post(
                f"{supabase_url}/auth/v1/token?grant_type=refresh_token",
                headers={
                    "apikey": supabase_anon_key,
                    "Content-Type": "application/json",
                },
                json={"refresh_token": refresh_token},
            )
        except httpx.HTTPError as exc:
            raise RuntimeError("Failed to refresh Supabase access token") from exc

        if resp.status_code == 200:
            data = _safe_response_json(resp)
            new_access_token = data.get("access_token")
            new_refresh_token = data.get("refresh_token") or refresh_token
            if not isinstance(new_access_token, str) or not _is_token_valid(
                new_access_token,
                margin=0,
            ):
                raise RuntimeError(
                    "Refresh response did not include a valid access_token"
                )
            logger.info("Access token refreshed successfully")
            return new_access_token, new_refresh_token

        # Refresh token expired or rejected; get a fresh refresh token via login.
        logger.warning(
            "Refresh token failed (status=%s), attempting email/password login",
            resp.status_code,
        )
        return await _login_with_credentials(supabase_url, supabase_anon_key, client)


async def _login_with_credentials(
    supabase_url: str, supabase_anon_key: str, client: httpx.AsyncClient | None = None
) -> tuple[str, str]:
    """Sign in with email/password to obtain fresh tokens.

    Requires SUPABASE_AUTH_EMAIL and SUPABASE_AUTH_PASSWORD environment variables.
    Returns (access_token, refresh_token).
    """
    email = os.environ.get("SUPABASE_AUTH_EMAIL")
    password = os.environ.get("SUPABASE_AUTH_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "Refresh token expired and SUPABASE_AUTH_EMAIL / SUPABASE_AUTH_PASSWORD "
            "are not set. Set them in .env to enable automatic re-login."
        )

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    try:
        try:
            resp = await client.post(
                f"{supabase_url}/auth/v1/token?grant_type=password",
                headers={
                    "apikey": supabase_anon_key,
                    "Content-Type": "application/json",
                },
                json={"email": email, "password": password},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("Failed to re-authenticate with Supabase") from exc

        data = _safe_response_json(resp)
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token", "")
        if not isinstance(access_token, str) or not _is_token_valid(
            access_token, margin=0
        ):
            raise RuntimeError("Login response did not include a valid access_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise RuntimeError("Login response did not include a refresh_token")
        logger.info("Re-authenticated via email/password successfully")
        return access_token, refresh_token
    finally:
        if own_client:
            await client.aclose()


async def _login_with_env_credentials() -> tuple[str, str]:
    """Sign in with configured Supabase credentials using a dedicated client."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_anon_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set to login")
    return await _login_with_credentials(supabase_url, supabase_anon_key)


def _safe_response_json(response: httpx.Response) -> dict[str, Any]:
    """Return response JSON as a dict with a clearer error for malformed bodies."""
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {response.url}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected JSON object from {response.url}, got {type(data)}"
        )
    return data


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

    return form_data


async def _resolve_agent_id(client, user_id: str, agent_id: str | None) -> str:
    """Return the provided agent_id or create a default one if missing."""
    if agent_id:
        return agent_id

    agent_service = AgentService(client)
    # from customized_areal.tpfc.config.builtin import TPFC_CONFIG
    from customized_areal.tpfc.config.builtin_new import TPFC_CONFIG

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
    logger.info("Created agent: %s", agent_id)
    return agent_id


_TRANSIENT_HTTP_ERRORS = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
)


async def _start_agent_run(
    api_base_url: str,
    auth_token: str,
    form_data: dict,
    task_file_path: list[str] | None,
    max_retries: int = 3,
) -> dict:
    """Start the agent run via HTTP and return the JSON response."""
    retry_delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=300.0) as http_client:
                files = []
                file_handles = []
                try:
                    if task_file_path:
                        for file_path in task_file_path:
                            if os.path.exists(file_path):
                                fh = open(file_path, "rb")
                                file_handles.append(fh)
                                files.append(("files", (os.path.basename(file_path), fh, None)))
                            else:
                                logger.warning("File not found: %s", file_path)

                    response = await http_client.post(
                        f"{api_base_url}/api/agent/start",
                        headers={"Authorization": f"Bearer {auth_token}"},
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

                return response.json()
        except _TRANSIENT_HTTP_ERRORS as exc:
            if attempt < max_retries:
                logger.warning(
                    "Transient HTTP error starting agent run (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, retry_delay, exc,
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
) -> tuple[dict, str]:
    """Start the agent run with refresh-token and credential-login fallbacks."""
    try:
        result = await _start_agent_run(
            api_base_url=api_base_url,
            auth_token=auth_token,
            form_data=form_data,
            task_file_path=task_file_path,
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
                )
                return result, login_token
            except AuthTokenExpiredError:
                raise AuthTokenExpiredError(
                    "Backend rejected both refreshed and login-issued Supabase "
                    f"tokens. Check that LE_AGENT_API_URL={api_base_url!r} uses "
                    "the same SUPABASE_URL/SUPABASE_ANON_KEY project as this "
                    f"client. Token diagnostics: {_token_diagnostics(login_token)}"
                )


TERMINAL_SSE_EVENTS = {"task_end", "error"}


def _token_diagnostics(token: str) -> dict[str, Any]:
    """Return non-secret token metadata useful for auth mismatch debugging."""
    header = _decode_jwt_header(token)
    payload = _decode_jwt_payload(token)
    return {
        "alg": header.get("alg"),
        "kid": header.get("kid"),
        "iss": payload.get("iss"),
        "sub": payload.get("sub"),
        "aud": payload.get("aud"),
        "role": payload.get("role"),
        "exp": payload.get("exp"),
    }


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

                            if status in {"completed", "failed", "stopped"}:
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

    if not streamed or status not in {"completed", "failed", "stopped"}:
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
                        if task_status in {"completed", "failed", "stopped"}:
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

            if status in {"completed", "failed", "stopped"}:
                break

            if _time_left() <= 0:
                break
            await asyncio.sleep(min(30.0, _time_left()))
            retry_delay = 1.0
        else:
            logger.error(
                "Timeout waiting for agent run to complete: task_id=%s",
                task_id,
            )
            raise TimeoutError(
                f"Agent run {agent_run_id} did not complete within {timeout} seconds"
            )

    return status


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
):

    db = DBConnection()
    client = await db.client

    user_id = user_id or DEFAULT_USER_ID
    if not user_id:
        raise ValueError(
            "user_id is required. Set TPFC_USER_ID env var or pass user_id argument."
        )
    resolved_agent_id = await _resolve_agent_id(client, user_id, agent_id)

    task_id = await create_task(
        client=client,
        account_id=user_id,
        agent_id=resolved_agent_id,
        name=task_description[:100] if task_description else None,
    )
    logger.info("Task created: %s", task_id)


    async def _do_run():
        token_manager = SharedTokenManager(
            refresh_token=refresh_token or DEFAULT_REFRESH_TOKEN
        )
        auth_token = await token_manager.get_valid_token()

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
        )
        logger.info("Agent run started via API: %s", result)

        agent_run_id = result.get("agent_run_id")
        start_status = result.get("status")

        if start_status == "queued" and not agent_run_id:
            logger.warning(
                "Agent run queued (slot occupied) for task_id=%s, waiting via task stream",
                task_id,
            )

        await _wait_for_agent_run(
            client,
            task_id=task_id,
            agent_run_id=agent_run_id,
            api_base_url=LE_AGENT_API_URL,
            auth_token=auth_token,
            token_manager=token_manager,
        )

        messages = await get_llm_messages(task_id, return_raw=True)
        final_boxed_answer = _extract_final_answer(messages)

        return messages, final_boxed_answer, log_path, None


    try:
        return await asyncio.wait_for(_do_run(), timeout=900)
    except TimeoutError:
        logger.warning("TPFCAgent run timed out after 15 minutes for task_id=%s", task_id)
        raise
    finally:
        # Guaranteed to run on normal exit, exception, or CancelledError
        # (the latter fires when asyncio.run() is cancelled by SIGINT/Ctrl+C).
        await cleanup_sandbox_for_task(client, task_id)


if __name__ == "__main__":
    task_description = (
        "The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. "
        "What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet."
    )
    task_file_path = [
        "/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/dataset/gaia-benchmark/gaia/2023/validation/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx"
    ]
    gt = "Time-Parking 2: Parallel Universe"

    # task_description = "今天北京天气怎么样"
    # task_file_path = []
    gt = ""

    messages, final_answer, log_path, _trace = asyncio.run(
        run_backend(
            task_description=task_description,
            task_file_path=task_file_path,
            gt=gt,
            tags=["debug", "0421"],
            user_id=DEFAULT_USER_ID,
            model_name="openrouter/qwen/qwen3-vl-8b-thinking",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            # api_key='aaa',
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            refresh_token=DEFAULT_REFRESH_TOKEN,
        )
    )

    print("Messages:", messages)
    print("Final boxed answer:", final_answer)
    print("Log path:", log_path)

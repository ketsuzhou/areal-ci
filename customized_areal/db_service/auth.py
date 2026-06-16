"""Supabase authentication and shared auth-token management for TPFC.

This module is intentionally free of any ``db_service`` / ``supabase`` client
dependency so it can be imported cheaply and tested in isolation. It owns:

- JWT decoding / validation helpers
- HS256 legacy compatibility-token minting
- the file-based, multi-process ``SharedTokenManager``
- Supabase refresh / email-password login flows
"""

import asyncio
import base64
import fcntl
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    _DOTENV_FILE = Path(__file__).parent.parent / ".env"
    load_dotenv(_DOTENV_FILE)
except ImportError:
    _DOTENV_FILE = Path(__file__).parent.parent / ".env"
    pass  # python-dotenv not installed, rely on environment variables

logger = logging.getLogger("BackendRun")

DEFAULT_REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN", "")
TOKEN_REFRESH_MARGIN = 60  # seconds; refresh shortly before expiry.
_REFRESH_WAIT_TIMEOUT = 30  # seconds to wait for another process to finish refreshing
_REFRESH_WAIT_INTERVAL = 1  # seconds between token re-reads while waiting
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
    alg = header.get("alg")
    if alg == "ES256":
        return isinstance(header.get("kid"), str) and _is_token_valid(
            token, margin=margin
        )
    return alg == "HS256" and _is_token_valid(token, margin=margin)


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


def _mint_hs256_auth_token_for_user(user_id: str, email: str | None = None) -> str:
    """Mint an HS256 Supabase-compatible JWT for a known self-hosted auth user."""
    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET") or os.environ.get(
        "AUTH_JWT_SECRET"
    )
    if not jwt_secret:
        raise RuntimeError(
            "SUPABASE_JWT_SECRET or AUTH_JWT_SECRET must be set to mint a local "
            "self-hosted Supabase auth token"
        )

    now = int(time.time())
    claims: dict[str, Any] = {
        "aud": "authenticated",
        "exp": now + 3600,
        "iat": now,
        "iss": f"{os.environ.get('SUPABASE_URL', '').rstrip('/')}/auth/v1",
        "role": "authenticated",
        "sub": user_id,
    }
    if email:
        claims["email"] = email

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_base64url_json(header)}.{_base64url_json(claims)}"
    signature = hmac.new(
        jwt_secret.encode(),
        signing_input.encode(),
        hashlib.sha256,
    ).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{signing_input}.{signature_b64}"


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
        """Write shared auth state and persist rotated refresh tokens."""
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
        if refresh_token:
            self._write_refresh_token_to_env(refresh_token)
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
                try:
                    access_token, refresh_token = await _login_with_env_credentials()
                except RuntimeError:
                    user_id = os.environ.get("TPFC_USER_ID")
                    if not user_id:
                        raise
                    logger.warning(
                        "Supabase credential login failed; minting local HS256 token "
                        "for self-hosted TPFC user"
                    )
                    access_token, refresh_token = (
                        _mint_hs256_auth_token_for_user(
                            user_id=user_id,
                            email=os.environ.get("SUPABASE_AUTH_EMAIL"),
                        ),
                        "",
                    )
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

    def _write_refresh_token_to_env(self, refresh_token: str) -> None:
        """Persist the latest refresh token back to the dotenv file."""
        os.environ["REFRESH_TOKEN"] = refresh_token
        lines: list[str] = []
        try:
            if _DOTENV_FILE.exists():
                lines = _DOTENV_FILE.read_text(encoding="utf-8").splitlines(
                    keepends=True
                )
        except OSError as exc:
            logger.warning("Failed to read dotenv file %s: %s", _DOTENV_FILE, exc)
            return

        updated = False
        new_lines: list[str] = []
        for line in lines:
            if line.startswith("REFRESH_TOKEN="):
                new_lines.append(f"REFRESH_TOKEN={refresh_token}\n")
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] = f"{new_lines[-1]}\n"
            new_lines.append(f"REFRESH_TOKEN={refresh_token}\n")

        tmp = _DOTENV_FILE.with_name(f".{_DOTENV_FILE.name}.{os.getpid()}.tmp")
        try:
            _DOTENV_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, _DOTENV_FILE)
            try:
                dir_fd = os.open(_DOTENV_FILE.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        except OSError as exc:
            logger.warning(
                "Failed to update REFRESH_TOKEN in %s: %s", _DOTENV_FILE, exc
            )
            try:
                if tmp.exists():
                    tmp.unlink()
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

    async with httpx.AsyncClient(
        timeout=_TOKEN_HTTP_TIMEOUT, trust_env=False
    ) as client:
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
        try:
            return await _login_with_credentials(supabase_url, supabase_anon_key, client)
        except RuntimeError:
            user_id = os.environ.get("TPFC_USER_ID")
            if not user_id:
                raise
            logger.warning(
                "Supabase credential login failed; minting local HS256 token for "
                "self-hosted TPFC user"
            )
            return _mint_hs256_auth_token_for_user(
                user_id=user_id,
                email=os.environ.get("SUPABASE_AUTH_EMAIL"),
            ), ""


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
        client = httpx.AsyncClient(trust_env=False)

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
            if resp.status_code != 200:
                details = _safe_response_json(resp)
                raise RuntimeError(
                    "Failed to re-authenticate with Supabase: "
                    f"status={resp.status_code}, "
                    f"error_code={details.get('error_code')!r}, "
                    f"msg={details.get('msg')!r}"
                )
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

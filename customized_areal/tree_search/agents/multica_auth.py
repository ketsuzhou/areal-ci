"""Authentication and local credential storage for direct MultiCA API calls."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import stat
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

DEFAULT_CREDENTIALS_PATH = Path(__file__).with_name("credentials.json")
_CREDENTIAL_VERSION = 1


class MulticaAuthError(RuntimeError):
    """Raised when MultiCA authentication cannot be resolved safely."""


def normalize_base_url(base_url: str) -> str:
    """Return a normalized absolute HTTP(S) MultiCA base URL."""
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise MulticaAuthError(
            "MultiCA base URL must be an absolute HTTP(S) URL without a query or fragment"
        )
    return value


def _credentials_path(credentials_path: Path | None) -> Path:
    return credentials_path or DEFAULT_CREDENTIALS_PATH


def login_guidance(base_url: str) -> str:
    return (
        "Run `python -m customized_areal.tree_search.agents.multica_auth "
        f"login --base-url {base_url}`."
    )


def save_credentials(
    base_url: str,
    api_key: str,
    *,
    credentials_path: Path | None = None,
) -> None:
    """Atomically save a base-URL-bound PAT with owner-only permissions."""
    if not api_key:
        raise MulticaAuthError("MultiCA API key cannot be empty")
    path = _credentials_path(credentials_path)
    payload = {
        "version": _CREDENTIAL_VERSION,
        "base_url": normalize_base_url(base_url),
        "api_key": api_key,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(raw_tmp_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_saved_api_key(
    base_url: str,
    *,
    credentials_path: Path | None = None,
) -> str:
    """Load a saved PAT only when it belongs to ``base_url``."""
    normalized_url = normalize_base_url(base_url)
    path = _credentials_path(credentials_path)
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise MulticaAuthError(
            f"No saved MultiCA credential exists. {login_guidance(normalized_url)}"
        ) from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise MulticaAuthError("MultiCA credential path must be a regular file")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise MulticaAuthError(
            f"MultiCA credential file permissions are too broad; run `chmod 600 {path}`"
        )

    try:
        with path.open() as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise MulticaAuthError("Failed to read the saved MultiCA credential") from exc

    if not isinstance(payload, dict) or payload.get("version") != _CREDENTIAL_VERSION:
        raise MulticaAuthError("Unsupported MultiCA credential file format")
    saved_url = payload.get("base_url")
    api_key = payload.get("api_key")
    if not isinstance(saved_url, str) or not isinstance(api_key, str) or not api_key:
        raise MulticaAuthError("Invalid MultiCA credential file format")
    try:
        normalized_saved_url = normalize_base_url(saved_url)
    except MulticaAuthError as exc:
        raise MulticaAuthError("Invalid MultiCA credential file format") from exc
    if normalized_saved_url != normalized_url:
        raise MulticaAuthError(
            "Saved credentials belong to a different MultiCA server. "
            + login_guidance(normalized_url)
        )
    return api_key


def resolve_api_key(
    base_url: str,
    explicit_api_key: str | None = None,
    *,
    credentials_path: Path | None = None,
) -> str:
    """Resolve explicit, environment, then saved MultiCA credentials."""
    if explicit_api_key:
        return explicit_api_key
    environment_api_key = os.environ.get("MULTICA_API_KEY")
    if environment_api_key:
        return environment_api_key
    return load_saved_api_key(base_url, credentials_path=credentials_path)


def _response_json(
    response: httpx.Response,
    *,
    endpoint: str,
    expected_status: int,
) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise MulticaAuthError(
            f"MultiCA authentication request to {endpoint} failed with status "
            f"{response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise MulticaAuthError(
            f"MultiCA authentication request to {endpoint} returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise MulticaAuthError(
            f"MultiCA authentication request to {endpoint} returned an invalid response"
        )
    return payload


def _auth_request(
    client: httpx.Client,
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> httpx.Response:
    try:
        return client.request(method, endpoint, **kwargs)
    except httpx.RequestError:
        pass
    raise MulticaAuthError(
        f"MultiCA authentication network request failed for {endpoint}"
    )


def login(
    base_url: str,
    email: str,
    code: str | Callable[[], str],
    *,
    credentials_path: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Exchange an email verification code for a validated, persisted PAT."""
    normalized_url = normalize_base_url(base_url)
    normalized_email = email.strip().lower()
    if not normalized_email:
        raise MulticaAuthError("Email cannot be empty")

    client_kwargs: dict[str, Any] = {"base_url": normalized_url, "timeout": 30.0}
    if transport is not None:
        client_kwargs["transport"] = transport
    with httpx.Client(**client_kwargs) as client:
        send_response = _auth_request(
            client, "POST", "/auth/send-code", json={"email": normalized_email}
        )
        _response_json(send_response, endpoint="/auth/send-code", expected_status=200)

        raw_code = code() if callable(code) else code
        normalized_code = raw_code.strip()
        if not normalized_code:
            raise MulticaAuthError("Verification code cannot be empty")
        verify_response = _auth_request(
            client,
            "POST",
            "/auth/verify-code",
            json={"email": normalized_email, "code": normalized_code},
        )
        verify_payload = _response_json(
            verify_response, endpoint="/auth/verify-code", expected_status=200
        )
        jwt_token = verify_payload.get("token")
        if not isinstance(jwt_token, str) or not jwt_token:
            raise MulticaAuthError(
                "MultiCA authentication response did not include a login token"
            )

        resolved_hostname = hostname or socket.gethostname() or "unknown"
        token_response = _auth_request(
            client,
            "POST",
            "/api/tokens",
            json={
                "name": f"AReaL ({resolved_hostname})",
                "expires_in_days": 90,
            },
            headers={"Authorization": f"Bearer {jwt_token}"},
        )
        token_payload = _response_json(
            token_response, endpoint="/api/tokens", expected_status=201
        )
        api_key = token_payload.get("token")
        if not isinstance(api_key, str) or not api_key:
            raise MulticaAuthError(
                "MultiCA token response did not include a personal access token"
            )

        me_response = _auth_request(
            client,
            "GET",
            "/api/me",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        me_payload = _response_json(
            me_response, endpoint="/api/me", expected_status=200
        )

    save_credentials(normalized_url, api_key, credentials_path=credentials_path)
    return me_payload


def main(argv: list[str] | None = None) -> int:
    """Run the explicit terminal MultiCA authentication command."""
    parser = argparse.ArgumentParser(description="Authenticate AReaL with MultiCA")
    subparsers = parser.add_subparsers(dest="command", required=True)
    login_parser = subparsers.add_parser(
        "login", description="Create and save a MultiCA personal access token"
    )
    login_parser.add_argument(
        "--base-url",
        default=os.environ.get("MULTICA_BASE_URL"),
        help="MultiCA API base URL (defaults to MULTICA_BASE_URL)",
    )
    args = parser.parse_args(argv)
    if not args.base_url:
        login_parser.error("--base-url or MULTICA_BASE_URL is required")

    email = input("Email: ")
    try:
        account = login(
            args.base_url,
            email,
            lambda: getpass.getpass("Verification code: "),
        )
    except MulticaAuthError as exc:
        print(f"MultiCA login failed: {exc}", file=sys.stderr)
        return 1

    account_email = account.get("email")
    if isinstance(account_email, str) and account_email:
        print(f"Authenticated as {account_email}")
    else:
        print("Authenticated with MultiCA")
    print(f"Credentials saved to {DEFAULT_CREDENTIALS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

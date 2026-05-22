import base64
import hashlib
import hmac
import json
import time

import pytest

from customized_areal.tpfc import backend_run


def _jwt_with_exp(exp: int, header: dict | None = None) -> str:
    header = header or {"alg": "none"}
    header_b64 = (
        base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"{header_b64}.{payload}.sig"


def _hs256_jwt(payload: dict, secret: str) -> str:
    header = backend_run._base64url_json({"alg": "HS256", "typ": "JWT"})
    payload_b64 = backend_run._base64url_json(payload)
    signing_input = f"{header}.{payload_b64}"
    signature = hmac.new(
        secret.encode(),
        signing_input.encode(),
        hashlib.sha256,
    ).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{signing_input}.{signature_b64}"


def test_refresh_token_usable_handles_opaque_and_expired_jwt():
    """Opaque refresh tokens are endpoint-validated; expired JWT tokens fail locally."""
    assert backend_run._is_refresh_token_usable("opaque-refresh-token")
    assert backend_run._is_refresh_token_usable(_jwt_with_exp(int(time.time()) + 3600))
    assert not backend_run._is_refresh_token_usable(_jwt_with_exp(int(time.time()) - 1))
    assert not backend_run._is_refresh_token_usable("")


def test_auth_token_usable_requires_es256_kid():
    """Shared access tokens must be compatible with LeAgent JWKS verification."""
    exp = int(time.time()) + 3600
    assert backend_run._is_auth_token_usable(
        _jwt_with_exp(exp, {"alg": "ES256", "kid": "key-id", "typ": "JWT"})
    )
    assert not backend_run._is_auth_token_usable(
        _jwt_with_exp(exp, {"alg": "HS256", "typ": "JWT"})
    )
    assert not backend_run._is_auth_token_usable(
        _jwt_with_exp(exp, {"alg": "ES256", "typ": "JWT"})
    )


def test_mint_legacy_hs256_auth_token_preserves_claims(monkeypatch):
    """HS256 fallback tokens preserve user claims and use the configured secret."""
    source_token = _jwt_with_exp(int(time.time()) + 3600)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")
    monkeypatch.setenv(
        "SUPABASE_ANON_KEY",
        _hs256_jwt({"iss": "supabase", "role": "anon"}, "test-secret"),
    )

    token = backend_run._mint_legacy_hs256_auth_token(source_token)

    assert token is not None
    assert backend_run._decode_jwt_header(token)["alg"] == "HS256"
    assert backend_run._decode_jwt_payload(token)["exp"] <= int(time.time()) + 3600


def test_mint_legacy_hs256_auth_token_rejects_mismatched_secret(monkeypatch):
    """HS256 fallback is skipped when SUPABASE_JWT_SECRET is inconsistent."""
    source_token = _jwt_with_exp(int(time.time()) + 3600)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "wrong-secret")
    monkeypatch.setenv(
        "SUPABASE_ANON_KEY",
        _hs256_jwt({"iss": "supabase", "role": "anon"}, "right-secret"),
    )
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    assert backend_run._mint_legacy_hs256_auth_token(source_token) is None


def test_shared_token_manager_round_trips_token_payload(tmp_path):
    """Shared token files carry access and refresh tokens across manager instances."""
    token_file = tmp_path / "shared_auth.json"
    access_token = _jwt_with_exp(int(time.time()) + 3600)
    refresh_token = "refresh-v1"

    writer = backend_run.SharedTokenManager(token_file=token_file)
    writer.write_token(access_token, refresh_token)

    reader = backend_run.SharedTokenManager(token_file=token_file)
    assert reader.read_token() == access_token
    assert reader.refresh_token == refresh_token


@pytest.mark.asyncio
async def test_refresh_access_token_uses_login_when_refresh_token_expired(
    monkeypatch,
):
    """Expired REFRESH_TOKEN values are replaced by login-issued refresh tokens."""
    access_token = _jwt_with_exp(int(time.time()) + 3600)

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)
            self.url = "https://example.supabase.co/auth/v1/token"

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("unexpected status")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.posts = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers, json):
            self.posts.append((url, json))
            assert "grant_type=password" in url
            return FakeResponse(
                200,
                {"access_token": access_token, "refresh_token": "refresh-v2"},
            )

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.setenv("SUPABASE_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("SUPABASE_AUTH_PASSWORD", "password")
    monkeypatch.setattr(backend_run.httpx, "AsyncClient", FakeAsyncClient)

    token, refresh_token = await backend_run._refresh_access_token(
        _jwt_with_exp(int(time.time()) - 1)
    )

    assert token == access_token
    assert refresh_token == "refresh-v2"


@pytest.mark.asyncio
async def test_start_agent_run_tries_login_after_refreshed_token_rejected(
    monkeypatch,
):
    """Backend 401 after refresh triggers one credential-login retry."""
    calls = []

    async def fake_start_agent_run(api_base_url, auth_token, form_data, task_file_path):
        calls.append(auth_token)
        if auth_token in {"old-token", "refreshed-token"}:
            raise backend_run.AuthTokenExpiredError("invalid")
        return {"agent_run_id": "run-id"}

    async def fake_get_valid_token(self, force_refresh=False):
        assert force_refresh
        return "refreshed-token"

    async def fake_login_and_store_token(self):
        return "login-token"

    monkeypatch.setattr(backend_run, "_start_agent_run", fake_start_agent_run)
    monkeypatch.setattr(
        backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token
    )
    monkeypatch.setattr(
        backend_run.SharedTokenManager,
        "login_and_store_token",
        fake_login_and_store_token,
    )

    result, token = await backend_run._start_agent_run_with_refresh(
        api_base_url="http://backend",
        token_manager=backend_run.SharedTokenManager(),
        auth_token="old-token",
        form_data={},
        task_file_path=[],
    )

    assert result == {"agent_run_id": "run-id"}
    assert token == "login-token"
    assert calls == ["old-token", "refreshed-token", "login-token"]

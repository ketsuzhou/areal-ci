from types import SimpleNamespace

import pytest

from customized_areal.db_service import connection as db_connection
from customized_areal.tpfc import backend_run


def test_sync_supabase_httpx_client_disables_env_proxy():
    client = db_connection._build_sync_supabase_httpx_client()
    try:
        assert client._trust_env is False
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_supabase_httpx_client_disables_env_proxy():
    client = db_connection._build_async_supabase_httpx_client()
    try:
        assert client._trust_env is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_shortlived_db_client_uses_proxy_free_httpx(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_create_async_client(url, key, options):
        captured["url"] = url
        captured["key"] = key
        captured["options"] = options
        return SimpleNamespace(options=options)

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setattr(backend_run, "create_async_client", fake_create_async_client)

    client = await backend_run._create_shortlived_db_client()
    httpx_client = captured["options"].httpx_client
    try:
        assert httpx_client._trust_env is False
        assert client.options.httpx_client is httpx_client
    finally:
        await httpx_client.aclose()


@pytest.mark.asyncio
async def test_login_with_credentials_owns_proxy_free_httpx_client(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "token", "refresh_token": "refresh"}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        async def post(self, *args, **kwargs):
            return _FakeResponse()

        async def aclose(self):
            return None

    monkeypatch.setenv("SUPABASE_AUTH_EMAIL", "user@example.com")
    monkeypatch.setenv("SUPABASE_AUTH_PASSWORD", "secret")
    monkeypatch.setattr(backend_run.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(backend_run, "_is_token_valid", lambda *args, **kwargs: True)

    access_token, refresh_token = await backend_run._login_with_credentials(
        "https://example.supabase.co",
        "anon-key",
        client=None,
    )

    assert captured["kwargs"]["trust_env"] is False
    assert access_token == "token"
    assert refresh_token == "refresh"

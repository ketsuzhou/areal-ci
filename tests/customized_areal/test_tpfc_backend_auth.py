import asyncio
import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest

from customized_areal.db_service import auth
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


def test_shared_token_manager_write_token_updates_refresh_token_in_dotenv(
    monkeypatch, tmp_path
):
    """Rotated refresh tokens are persisted for future processes."""
    token_file = tmp_path / "shared_auth.json"
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("REFRESH_TOKEN=refresh-v1\nOTHER=value\n", encoding="utf-8")
    access_token = _jwt_with_exp(
        int(time.time()) + 3600,
        {"alg": "ES256", "kid": "key-id", "typ": "JWT"},
    )

    monkeypatch.setattr(auth, "_DOTENV_FILE", dotenv_file)

    manager = backend_run.SharedTokenManager(
        token_file=token_file,
        refresh_token="refresh-v1",
    )
    manager.write_token(access_token, "refresh-v2")

    payload = json.loads(token_file.read_text(encoding="utf-8"))
    assert payload["refresh_token"] == "refresh-v2"
    assert "REFRESH_TOKEN=refresh-v2\n" in dotenv_file.read_text(encoding="utf-8")
    assert "OTHER=value\n" in dotenv_file.read_text(encoding="utf-8")


def test_save_tpfc_agent_id_updates_dotenv_and_process_env(monkeypatch, tmp_path):
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "OTHER=value\nTPFC_AGENT_ID=old-agent\nREFRESH_TOKEN=refresh\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(backend_run, "DEFAULT_AGENT_ID", "old-agent")

    backend_run._save_tpfc_agent_id("new-agent", dotenv_file)

    assert dotenv_file.read_text(encoding="utf-8") == (
        "OTHER=value\nTPFC_AGENT_ID=new-agent\nREFRESH_TOKEN=refresh\n"
    )
    assert backend_run.DEFAULT_AGENT_ID == "new-agent"
    assert backend_run.os.environ["TPFC_AGENT_ID"] == "new-agent"


@pytest.mark.asyncio
async def test_resolve_agent_id_creates_and_persists_when_configured_id_missing(
    monkeypatch,
):
    class FakeQuery:
        def __init__(self, data):
            self.data = data

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        async def execute(self):
            return SimpleNamespace(data=self.data)

    class FakeClient:
        def table(self, name):
            assert name == "agents"
            return FakeQuery([])

    class FakeAgentService:
        def __init__(self, client):
            assert isinstance(client, FakeClient)

        async def create_agent(self, user_id, data):
            assert user_id == "user"
            assert data.name == "TPFC"
            return SimpleNamespace(agent_id="created-agent")

    class FakeLoader:
        async def load_agent(self, agent_id, user_id, load_config=True):
            assert agent_id == "created-agent"
            assert user_id == "user"
            assert load_config is True

    async def fake_get_agent_loader():
        return FakeLoader()

    saved_agent_ids = []
    monkeypatch.setenv("TPFC_AGENT_ID", "missing-agent")
    monkeypatch.setattr(backend_run, "AgentService", FakeAgentService)
    monkeypatch.setattr(backend_run, "get_agent_loader", fake_get_agent_loader)
    monkeypatch.setattr(
        backend_run,
        "_save_tpfc_agent_id",
        lambda agent_id: saved_agent_ids.append(agent_id),
    )

    agent_id = await backend_run._resolve_agent_id(FakeClient(), "user", None)

    assert agent_id == "created-agent"
    assert saved_agent_ids == ["created-agent"]


@pytest.mark.asyncio
async def test_shared_token_manager_refreshes_and_persists_rotated_refresh_token(
    monkeypatch, tmp_path
):
    """Refreshing under the shared lock updates both token stores."""
    token_file = tmp_path / "shared_auth.json"
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("REFRESH_TOKEN=refresh-v1\n", encoding="utf-8")
    refreshed_access_token = _jwt_with_exp(
        int(time.time()) + 3600,
        {"alg": "ES256", "kid": "key-id", "typ": "JWT"},
    )

    async def fake_refresh(refresh_token):
        assert refresh_token == "refresh-v1"
        return refreshed_access_token, "refresh-v2"

    monkeypatch.setattr(auth, "_DOTENV_FILE", dotenv_file)
    monkeypatch.setattr(auth, "_refresh_access_token", fake_refresh)

    manager = backend_run.SharedTokenManager(
        token_file=token_file,
        refresh_token="refresh-v1",
    )
    token = await manager.get_valid_token()

    assert token == refreshed_access_token
    assert manager.refresh_token == "refresh-v2"
    payload = json.loads(token_file.read_text(encoding="utf-8"))
    assert payload["refresh_token"] == "refresh-v2"
    assert dotenv_file.read_text(encoding="utf-8") == "REFRESH_TOKEN=refresh-v2\n"


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


@pytest.mark.asyncio
async def test_start_agent_run_tries_legacy_token_after_login_rejected(monkeypatch):
    """Backend 401 after login tries the HS256 compatibility token once."""
    calls = []

    async def fake_start_agent_run(api_base_url, auth_token, form_data, task_file_path):
        calls.append(auth_token)
        if auth_token != "legacy-token":
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
    monkeypatch.setattr(
        backend_run, "_mint_legacy_hs256_auth_token", lambda token: "legacy-token"
    )

    result, token = await backend_run._start_agent_run_with_refresh(
        api_base_url="http://backend",
        token_manager=backend_run.SharedTokenManager(),
        auth_token="old-token",
        form_data={},
        task_file_path=[],
    )

    assert result == {"agent_run_id": "run-id"}
    assert token == "legacy-token"
    assert calls == ["old-token", "refreshed-token", "login-token", "legacy-token"]


@pytest.mark.asyncio
async def test_start_agent_run_uses_run_timeout_budget(monkeypatch):
    """Agent start requests should not time out before the run budget expires."""
    captured_timeouts = []

    class FakeResponse:
        status_code = 200
        text = '{"agent_run_id": "run-id"}'

        def json(self):
            return {"agent_run_id": "run-id"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured_timeouts.append(kwargs.get("timeout"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(backend_run.httpx, "AsyncClient", FakeAsyncClient)

    result = await backend_run._start_agent_run(
        api_base_url="http://backend",
        auth_token="token",
        form_data={},
        task_file_path=[],
        max_retries=0,
    )

    assert result == {"agent_run_id": "run-id"}
    assert captured_timeouts[0].read == backend_run._RUN_TIMEOUT


@pytest.mark.asyncio
async def test_run_backend_returns_named_result_and_tuple_compat(monkeypatch):
    class FakeClient:
        pass

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_resolve_agent_id(client, user_id, agent_id):
        return "agent"

    async def fake_create_task(**kwargs):
        return "task-id"

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        return {"agent_run_id": "run-id", "status": "running"}, "token"

    async def fake_wait_for_agent_run(*args, **kwargs):
        return "completed"

    parsed_messages = [{"role": "assistant", "content": "<answer>42</answer>"}]
    raw_messages = [
        {
            "message_id": "m1",
            "role": "assistant",
            "content": {"role": "assistant", "content": "<answer>42</answer>"},
            "metadata": {"entropy_stats": {"max_entropy": 3.2}, "need_branch": True},
        }
    ]

    async def fake_get_messages(client, task_id):
        assert task_id == "task-id"
        return parsed_messages

    async def fake_get_raw_messages(client, task_id):
        assert task_id == "task-id"
        return raw_messages

    async def fake_cleanup(client, task_id):
        return None

    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "_resolve_agent_id", fake_resolve_agent_id)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(
        backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token
    )
    monkeypatch.setattr(
        backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh
    )
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", fake_wait_for_agent_run)
    monkeypatch.setattr(backend_run, "_get_llm_messages_with_client", fake_get_messages)
    monkeypatch.setattr(
        backend_run, "_get_raw_messages_with_client", fake_get_raw_messages
    )
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    result = await backend_run.run_backend("task", [], user_id="user")

    assert result.task_id == "task-id"
    assert result.messages == parsed_messages
    assert result.raw_messages == raw_messages
    assert result.final_answer == "42"
    assert len(result) == 4
    assert result[0] == parsed_messages
    assert result[1:3] == ("42", "./log.json")
    messages, answer, log_path, trace = result
    assert messages == parsed_messages
    assert answer == "42"
    assert log_path == "./log.json"
    assert trace is None


@pytest.mark.asyncio
async def test_run_backend_branch_mode_starts_existing_task_without_prompt_append(
    monkeypatch,
):
    class FakeClient:
        pass

    create_task_calls = []
    normal_start_calls = []
    branch_start_calls = []
    cleanup_calls = []

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_create_task(**kwargs):
        create_task_calls.append(kwargs)
        raise AssertionError("branch mode must not create a task")

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        normal_start_calls.append(kwargs)
        raise AssertionError(
            "branch mode must not append a prompt through /agent/start"
        )

    async def fake_start_branch_agent_run_for_task_with_refresh(**kwargs):
        branch_start_calls.append(kwargs)
        return {"agent_run_id": "run-id", "status": "pending"}, "token"

    async def fake_wait_for_agent_run(*args, **kwargs):
        assert kwargs["task_id"] == "branch-task-id"
        assert kwargs["agent_run_id"] == "run-id"
        return "completed"

    parsed_messages = [{"role": "assistant", "content": "<answer>branch</answer>"}]
    raw_messages = [
        {
            "message_id": "m1",
            "role": "assistant",
            "content": {"role": "assistant", "content": "<answer>branch</answer>"},
            "metadata": {"need_branch": True},
        }
    ]

    async def fake_get_messages(client, task_id):
        assert task_id == "branch-task-id"
        return parsed_messages

    async def fake_get_raw_messages(client, task_id):
        assert task_id == "branch-task-id"
        return raw_messages

    async def fake_cleanup(client, task_id):
        cleanup_calls.append(task_id)

    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(
        backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token
    )
    monkeypatch.setattr(
        backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh
    )
    monkeypatch.setattr(
        backend_run,
        "_start_branch_agent_run_for_task_with_refresh",
        fake_start_branch_agent_run_for_task_with_refresh,
    )
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", fake_wait_for_agent_run)
    monkeypatch.setattr(backend_run, "_get_llm_messages_with_client", fake_get_messages)
    monkeypatch.setattr(
        backend_run, "_get_raw_messages_with_client", fake_get_raw_messages
    )
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    result = await backend_run.run_backend(
        task_description=None,
        task_file_path=["should-not-upload.txt"],
        task_id="branch-task-id",
        user_id="user",
        model_name="openai/test-model",
        seed_messages_already_inserted=True,
    )

    assert create_task_calls == []
    assert normal_start_calls == []
    assert len(branch_start_calls) == 1
    assert branch_start_calls[0]["task_id"] == "branch-task-id"
    assert branch_start_calls[0]["model_name"] == "openai/test-model"
    assert branch_start_calls[0]["base_url"] is None
    assert branch_start_calls[0]["api_key"] is None
    assert result.task_id == "branch-task-id"
    assert result.messages == parsed_messages
    assert result.raw_messages == raw_messages
    assert result.final_answer == "branch"
    assert cleanup_calls == ["branch-task-id"]


@pytest.mark.asyncio
async def test_run_backend_branch_mode_rejects_missing_task_id(monkeypatch):
    close_calls = []

    class FakeClient:
        pass

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        close_calls.append(client)

    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)

    with pytest.raises(ValueError, match="task_id is required"):
        await backend_run.run_backend(
            task_description=None,
            task_file_path=[],
            user_id="user",
            seed_messages_already_inserted=True,
        )

    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_start_branch_agent_run_for_task_calls_authenticated_endpoint(
    monkeypatch,
):
    posts = []

    class FakeResponse:
        status_code = 200
        text = json.dumps({"task_id": "task-id", "agent_run_id": "run-id"})
        url = "http://backend/api/agent/start-branch"

        def json(self):
            return {"task_id": "task-id", "agent_run_id": "run-id"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            posts.append((url, headers, json))
            return FakeResponse()

    monkeypatch.delenv("LE_AGENT_BRANCH_RUN_ENDPOINT", raising=False)
    monkeypatch.setattr(backend_run.httpx, "AsyncClient", FakeAsyncClient)

    result = await backend_run._start_branch_agent_run_for_task(
        client=object(),
        api_base_url="http://backend",
        auth_token="auth-token",
        task_id="task-id",
        account_id="user-id",
        model_name="openai/test",
        base_url="http://proxy",
        api_key="proxy-key",
    )

    assert result == {"task_id": "task-id", "agent_run_id": "run-id"}
    assert posts == [
        (
            "http://backend/api/agent/start-branch",
            {"Authorization": "Bearer auth-token"},
            {
                "task_id": "task-id",
                "model_name": "openai/test",
                "proxy_base_url": "http://proxy",
                "proxy_api_key": "proxy-key",
                "stream": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_get_raw_messages_with_client_query_shape():
    row = {
        "message_id": "m1",
        "role": "assistant",
        "content": {"role": "assistant", "content": "hello"},
        "created_at": "2026-05-27T00:00:00Z",
        "updated_at": "2026-05-27T00:00:01Z",
        "metadata": {"need_branch": True},
    }

    class FakeResult:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self):
            self.selects = []
            self.eqs = []
            self.orders = []
            self.ranges = []
            self.execute_count = 0

        def select(self, columns):
            self.selects.append(columns)
            return self

        def eq(self, column, value):
            self.eqs.append((column, value))
            return self

        def order(self, column, desc):
            self.orders.append((column, desc))
            return self

        def range(self, start, end):
            self.ranges.append((start, end))
            return self

        async def execute(self):
            self.execute_count += 1
            if self.execute_count == 1:
                return FakeResult([row])
            return FakeResult([])

    class FakeClient:
        def __init__(self):
            self.tables = []
            self.query = FakeQuery()

        def table(self, name):
            self.tables.append(name)
            return self.query

    client = FakeClient()

    result = await backend_run._get_raw_messages_with_client(client, "task-id")

    assert result == [row]
    assert client.tables == ["messages", "messages"]
    assert client.query.selects == [
        "message_id, role, content, created_at, updated_at, metadata",
        "message_id, role, content, created_at, updated_at, metadata",
    ]
    assert client.query.eqs == [("task_id", "task-id"), ("task_id", "task-id")]
    assert client.query.orders == [("created_at", False), ("created_at", False)]
    assert client.query.ranges == [(0, 999), (1000, 1999)]


@pytest.mark.asyncio
async def test_run_backend_failed_terminal_status_raises_and_cleans_up(monkeypatch):
    """Failed backend terminal status is surfaced instead of returning messages."""
    cleanup_calls = []

    class FakeClient:
        pass

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_resolve_agent_id(client, user_id, agent_id):
        return "agent"

    async def fake_create_task(**kwargs):
        return "task-id"

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        return {"agent_run_id": "run-id", "status": "running"}, "token"

    async def fake_wait_for_agent_run(*args, **kwargs):
        return "failed"

    async def fail_get_messages(*args, **kwargs):
        raise AssertionError("messages should not be fetched for failed runs")

    async def fake_cleanup(client, task_id):
        cleanup_calls.append(task_id)

    monkeypatch.setenv("TPFC_USER_ID", "user")
    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "_resolve_agent_id", fake_resolve_agent_id)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(
        backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token
    )
    monkeypatch.setattr(
        backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh
    )
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", fake_wait_for_agent_run)

    async def fake_get_terminal_error(*args):
        return "boom"

    monkeypatch.setattr(backend_run, "_get_terminal_error", fake_get_terminal_error)
    monkeypatch.setattr(backend_run, "_get_llm_messages_with_client", fail_get_messages)
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    with pytest.raises(RuntimeError, match="status='failed'.*boom"):
        await backend_run.run_backend("task", [], user_id="user")

    assert cleanup_calls == ["task-id"]


@pytest.mark.asyncio
async def test_run_backend_timeout_skips_cleanup_for_active_run(monkeypatch):
    """Timeout does not delete the sandbox while the backend may still be active."""
    cleanup_calls = []

    class FakeClient:
        pass

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_resolve_agent_id(client, user_id, agent_id):
        return "agent"

    async def fake_create_task(**kwargs):
        return "task-id"

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        return {"agent_run_id": "run-id", "status": "running"}, "token"

    async def slow_wait_for_agent_run(*args, **kwargs):
        await asyncio.sleep(1)
        return "completed"

    async def fake_cleanup(client, task_id):
        cleanup_calls.append(task_id)

    monkeypatch.setattr(backend_run, "_RUN_TIMEOUT", 0.01)
    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "_resolve_agent_id", fake_resolve_agent_id)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(
        backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token
    )
    monkeypatch.setattr(
        backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh
    )
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", slow_wait_for_agent_run)
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    with pytest.raises(TimeoutError):
        await backend_run.run_backend("task", [], user_id="user")

    assert cleanup_calls == []


def test_prepare_form_data_includes_proxy_override():
    """Explicit api_key/base_url are forwarded as proxy_api_key/proxy_base_url."""
    form_data = backend_run._prepare_form_data(
        task_id="t1",
        task_description="test",
        agent_id="a1",
        model_name="openrouter/qwen/test",
        base_url="http://proxy",
        api_key="aaa",
        tags=None,
    )
    assert form_data["proxy_api_key"] == "aaa"
    assert form_data["proxy_base_url"] == "http://proxy"


def test_prepare_form_data_omits_proxy_when_none():
    """When api_key/base_url are None, proxy fields are absent (no overwrite)."""
    form_data = backend_run._prepare_form_data(
        task_id="t1",
        task_description="test",
        agent_id="a1",
        model_name="openrouter/qwen/test",
        base_url=None,
        api_key=None,
        tags=None,
    )
    assert "proxy_api_key" not in form_data
    assert "proxy_base_url" not in form_data


@pytest.mark.asyncio
async def test_start_branch_agent_run_for_task_sends_proxy_override(monkeypatch):
    """Branch start sends proxy_api_key/proxy_base_url in JSON body."""
    posts = []

    class FakeResponse:
        status_code = 200
        text = json.dumps({"task_id": "t1", "agent_run_id": "r1"})
        url = "http://backend/api/agent/start-branch"

        def json(self):
            return {"task_id": "t1", "agent_run_id": "r1"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            posts.append((url, headers, json))
            return FakeResponse()

    monkeypatch.delenv("LE_AGENT_BRANCH_RUN_ENDPOINT", raising=False)
    monkeypatch.setattr(backend_run.httpx, "AsyncClient", FakeAsyncClient)

    await backend_run._start_branch_agent_run_for_task(
        client=object(),
        api_base_url="http://backend",
        auth_token="auth-token",
        task_id="t1",
        account_id="user",
        model_name="openrouter/qwen/test",
        base_url="http://proxy",
        api_key="aaa",
    )

    assert len(posts) == 1
    body = posts[0][2]
    assert body["proxy_api_key"] == "aaa"
    assert body["proxy_base_url"] == "http://proxy"


@pytest.mark.asyncio
async def test_start_branch_agent_run_for_task_omits_proxy_when_none(monkeypatch):
    """Branch start omits proxy fields when api_key/base_url are None."""
    posts = []

    class FakeResponse:
        status_code = 200
        text = json.dumps({"task_id": "t1", "agent_run_id": "r1"})
        url = "http://backend/api/agent/start-branch"

        def json(self):
            return {"task_id": "t1", "agent_run_id": "r1"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            posts.append((url, headers, json))
            return FakeResponse()

    monkeypatch.delenv("LE_AGENT_BRANCH_RUN_ENDPOINT", raising=False)
    monkeypatch.setattr(backend_run.httpx, "AsyncClient", FakeAsyncClient)

    await backend_run._start_branch_agent_run_for_task(
        client=object(),
        api_base_url="http://backend",
        auth_token="auth-token",
        task_id="t1",
        account_id="user",
        model_name="openrouter/qwen/test",
        base_url=None,
        api_key=None,
    )

    assert len(posts) == 1
    body = posts[0][2]
    assert body["proxy_api_key"] is None
    assert body["proxy_base_url"] is None


@pytest.mark.asyncio
async def test_run_backend_forwards_proxy_override_to_normal_start(monkeypatch):
    """run_backend passes api_key/base_url as proxy fields in normal (non-branch) mode."""

    class FakeClient:
        pass

    captured_form_data = {}

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_resolve_agent_id(client, user_id, agent_id):
        return "agent"

    async def fake_create_task(**kwargs):
        return "task-id"

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        nonlocal captured_form_data
        captured_form_data = kwargs.get("form_data", {})
        return {"agent_run_id": "run-id", "status": "running"}, "token"

    async def fake_wait_for_agent_run(*args, **kwargs):
        return "completed"

    async def fake_get_messages(client, task_id):
        return [{"role": "assistant", "content": "<answer>42</answer>"}]

    async def fake_get_raw_messages(client, task_id):
        return []

    async def fake_cleanup(client, task_id):
        return None

    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "_resolve_agent_id", fake_resolve_agent_id)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(
        backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token
    )
    monkeypatch.setattr(
        backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh
    )
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", fake_wait_for_agent_run)
    monkeypatch.setattr(backend_run, "_get_llm_messages_with_client", fake_get_messages)
    monkeypatch.setattr(
        backend_run, "_get_raw_messages_with_client", fake_get_raw_messages
    )
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    await backend_run.run_backend(
        "task",
        [],
        user_id="user",
        model_name="openrouter/qwen/test",
        api_key="aaa",
        base_url="http://proxy",
    )

    assert captured_form_data.get("proxy_api_key") == "aaa"
    assert captured_form_data.get("proxy_base_url") == "http://proxy"

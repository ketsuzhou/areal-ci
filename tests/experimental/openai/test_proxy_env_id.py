"""Test suite for env_id in StartSessionRequest and SessionData."""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import (
    SessionData,
    StartSessionRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADMIN_KEY = "test-admin-key"


@pytest.fixture(autouse=True)
def _reset_server_globals(monkeypatch):
    """Reset all module-level globals before each test."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)


httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


# ---------------------------------------------------------------------------
# Tests: StartSessionRequest model
# ---------------------------------------------------------------------------


class TestStartSessionRequestModel:
    def test_start_session_accepts_session_ref_without_task_id(self):
        req = StartSessionRequest(session_ref="binding-123", env_id="env_abc")

        assert req.canonical_session_ref == "binding-123"
        assert req.task_id is None

    def test_start_session_keeps_legacy_task_id_compatible(self):
        req = StartSessionRequest(task_id="task-legacy")

        assert req.canonical_session_ref == "task-legacy"
        assert req.session_ref is None

    @pytest.mark.parametrize(
        "payload",
        [{}, {"session_ref": "binding-123", "task_id": "task-legacy"}],
    )
    def test_start_session_rejects_missing_or_conflicting_reference(self, payload):
        with pytest.raises(ValidationError, match="exactly one"):
            StartSessionRequest(**payload)

    def test_start_session_accepts_env_id(self):
        """StartSessionRequest accepts and stores env_id parameter."""
        req = StartSessionRequest(task_id="t1", env_id="env_abc")
        assert req.env_id == "env_abc"
        assert req.task_id == "t1"

    def test_start_session_env_id_optional(self):
        """StartSessionRequest has env_id optional with None default."""
        req = StartSessionRequest(task_id="t1")
        assert req.env_id is None
        assert req.task_id == "t1"


# ---------------------------------------------------------------------------
# Tests: SessionData stores env_id
# ---------------------------------------------------------------------------


class TestSessionDataEnvId:
    def test_session_data_persists_env_id(self):
        """SessionData stores env_id when provided in constructor."""
        session = SessionData(session_id="s1", env_id="env_abc")
        assert session.session_id == "s1"
        assert session.env_id == "env_abc"

    def test_session_data_env_id_optional(self):
        """SessionData env_id defaults to None when not provided."""
        session = SessionData(session_id="s1")
        assert session.session_id == "s1"
        assert session.env_id is None


# ---------------------------------------------------------------------------
# Tests: start_session endpoint integration
# ---------------------------------------------------------------------------


class TestStartSessionEndpointEnvId:
    @pytest.mark.asyncio
    async def test_start_session_uses_session_ref_as_session_namespace(
        self, monkeypatch
    ):
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"session_ref": "binding-123", "env_id": "env_abc"},
            )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "binding-123-0"

    @pytest.mark.asyncio
    async def test_start_session_passes_env_id_to_session_data(self, monkeypatch):
        """start_session endpoint passes env_id from request to SessionData."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t1", "env_id": "env_abc"},
            )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Check that SessionData has env_id set
        session = srv._session_cache[session_id]
        assert session.env_id == "env_abc"

    @pytest.mark.asyncio
    async def test_start_session_without_env_id_sets_none(self, monkeypatch):
        """start_session without env_id leaves SessionData.env_id as None."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t1"},
            )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Check that SessionData has env_id as None
        session = srv._session_cache[session_id]
        assert session.env_id is None

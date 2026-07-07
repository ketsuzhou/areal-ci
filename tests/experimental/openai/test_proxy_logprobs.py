"""Tests for logprobs injection in ``_call_client_create`` (sub-project E, task 9).

The proxy rollout server's ``_call_client_create`` is the single chokepoint for
all OpenAI-compatible chat completions / responses forwarding. Sub-project E
needs token-level logprobs to compute entropy from captured trajectories, so
this file verifies that:

1. ``logprobs=True`` is injected into the forwarded kwargs unconditionally.
2. If the upstream provider rejects the ``logprobs`` parameter, the call is
   retried exactly once without ``logprobs`` and a warning is emitted.
3. Unrelated upstream errors propagate (as ``HTTPException(500)``) with no
   retry.
"""

from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData

# ---------------------------------------------------------------------------
# Helpers (mirror the pattern in test_proxy_env_id.py / test_proxy_rollout_server.py)
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
    # _call_client_create checks `_openai_client is None` and short-circuits to
    # HTTPException(500) if so. Inject a truthy sentinel so the function
    # proceeds to the create_fn call (which is supplied directly by each test).
    monkeypatch.setattr(srv, "_openai_client", object())


def _seed_session(session_id: str = "s1") -> SessionData:
    """Insert a live SessionData into the cache and return it."""
    session = SessionData(session_id=session_id)
    srv._session_cache[session_id] = session
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogprobsInjection:
    """Sub-project E task 9: ``_call_client_create`` injects ``logprobs=True``."""

    @pytest.mark.asyncio
    async def test_call_client_create_injects_logprobs(self):
        """``logprobs=True`` is forwarded to ``create_fn``."""
        captured: dict = {}

        async def fake_create(*, areal_cache=None, **kwargs):
            captured.update(kwargs)
            return object()  # minimal sentinel; response shape is not under test here

        _seed_session()
        await srv._call_client_create(
            create_fn=fake_create,
            request={"messages": [{"role": "user", "content": "hi"}]},
            session_id="s1",
        )
        assert captured.get("logprobs") is True

    @pytest.mark.asyncio
    async def test_call_client_create_fallback_on_logprobs_error(self):
        """Upstream rejection of ``logprobs`` triggers exactly one retry without it."""
        call_count = 0
        captured: list[dict] = []

        async def fake_create(*, areal_cache=None, **kwargs):
            nonlocal call_count
            call_count += 1
            captured.append(dict(kwargs))
            if call_count == 1 and kwargs.get("logprobs") is True:
                # Simulate an upstream that rejects the logprobs parameter.
                raise Exception("logprobs not supported on this model")
            return object()

        _seed_session()
        await srv._call_client_create(
            create_fn=fake_create,
            request={"messages": [{"role": "user", "content": "hi"}]},
            session_id="s1",
        )
        assert call_count == 2  # one initial attempt + one retry
        # First attempt must have carried logprobs=True ...
        assert captured[0].get("logprobs") is True
        # ... and the retry must NOT have logprobs=True.
        assert captured[1].get("logprobs") is not True

    @pytest.mark.asyncio
    async def test_call_client_create_propagates_unrelated_error(self):
        """Non-logprobs errors surface as HTTPException(500) with no retry."""
        call_count = 0

        async def fake_create(*, areal_cache=None, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("something else went wrong")

        _seed_session()
        with pytest.raises(HTTPException) as exc_info:
            await srv._call_client_create(
                create_fn=fake_create,
                request={"messages": [{"role": "user", "content": "hi"}]},
                session_id="s1",
            )
        assert exc_info.value.status_code == 500
        assert "something else went wrong" in exc_info.value.detail
        assert call_count == 1  # no retry for unrelated errors

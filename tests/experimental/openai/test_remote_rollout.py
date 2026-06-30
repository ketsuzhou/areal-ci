# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenRouter remote rollout proxy.

Spec: docs/superpowers/specs/2026-06-29-openrouter-remote-rollout-proxy-design.md
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from areal.api import cli_args as cli_args_module
from areal.api.cli_args import PPOActorConfig
from areal.experimental.openai.cache import InteractionCache

# ---------------------------------------------------------------------------
# Tests: PPOActorConfig enable_remote_rollout warning (spec tests 18-20)
# ---------------------------------------------------------------------------


# Areal's getLogger() replaces Logger.root/manager on import, which orphans
# caplog's root-attached handler and breaks `logging.getLogger("CLIArgs")`
# lookups. Attach a handler directly to the exact logger object the
# cli_args module holds, so capture is robust to that root replacement.
class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def cliargs_capture():
    handler = _ListHandler()
    logger = cli_args_module.logger
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


class TestEnableRemoteRolloutWarning:
    def test_warns_when_recompute_disabled(self, cliargs_capture):
        """enable_remote_rollout=True with no recompute path → warning."""
        PPOActorConfig(
            enable_remote_rollout=True,
            recompute_logprob=False,
            use_decoupled_loss=False,
        )
        assert any(
            "enable_remote_rollout" in rec.getMessage()
            and "recompute" in rec.getMessage()
            for rec in cliargs_capture.records
        ), (
            f"expected recompute warning, got: {[r.getMessage() for r in cliargs_capture.records]}"
        )

    def test_no_warning_when_recompute_logprob_true(self, cliargs_capture):
        """enable_remote_rollout=True + recompute_logprob=True → no warning."""
        PPOActorConfig(
            enable_remote_rollout=True,
            recompute_logprob=True,
            use_decoupled_loss=False,
        )
        assert not any(
            "enable_remote_rollout" in rec.getMessage()
            for rec in cliargs_capture.records
        ), f"unexpected warning: {[r.getMessage() for r in cliargs_capture.records]}"

    def test_no_warning_when_decoupled_loss_true(self, cliargs_capture):
        """enable_remote_rollout=True + use_decoupled_loss=True → no warning."""
        PPOActorConfig(
            enable_remote_rollout=True,
            recompute_logprob=False,
            use_decoupled_loss=True,
        )
        assert not any(
            "enable_remote_rollout" in rec.getMessage()
            for rec in cliargs_capture.records
        ), f"unexpected warning: {[r.getMessage() for r in cliargs_capture.records]}"


# ---------------------------------------------------------------------------
# Tests: finish-reason mapping (spec test 14)
# ---------------------------------------------------------------------------


class TestMapFinishReason:
    def test_length(self):
        from areal.experimental.openai.proxy.remote_rollout import _map_finish_reason

        assert _map_finish_reason("length") == "length"

    def test_tool_calls(self):
        from areal.experimental.openai.proxy.remote_rollout import _map_finish_reason

        assert _map_finish_reason("tool_calls") == "tool_calls"

    def test_stop(self):
        from areal.experimental.openai.proxy.remote_rollout import _map_finish_reason

        assert _map_finish_reason("stop") == "stop"

    def test_content_filter(self):
        from areal.experimental.openai.proxy.remote_rollout import _map_finish_reason

        assert _map_finish_reason("content_filter") == "stop"

    def test_none(self):
        from areal.experimental.openai.proxy.remote_rollout import _map_finish_reason

        assert _map_finish_reason(None) == "stop"

    def test_unknown(self):
        from areal.experimental.openai.proxy.remote_rollout import _map_finish_reason

        assert _map_finish_reason("some-new-future-reason") == "stop"


# ---------------------------------------------------------------------------
# Tests: recompute gate (spec tests 21-22)
# ---------------------------------------------------------------------------


def _make_remote_client(recompute_enabled: bool):
    """Construct a RemoteRolloutClient with stubbed tokenizer."""
    from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    return RemoteRolloutClient(
        tokenizer=tokenizer,
        chat_template_type="hf",
        recompute_enabled=recompute_enabled,
    )


async def _async_return(value):
    """Helper: make a sync lambda usable as an awaitable."""
    return value


class TestRecomputeGate:
    @pytest.mark.asyncio
    async def test_first_request_fails_when_recompute_disabled(self):
        """recompute_enabled=False → first remote request raises 500; gate stays closed."""
        from fastapi import HTTPException

        client = _make_remote_client(recompute_enabled=False)
        cache = InteractionCache()
        request = {"model": "remote:openai/gpt-4o-mini", "messages": []}

        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)

        assert exc_info.value.status_code == 500
        assert "recompute_logprob" in exc_info.value.detail
        # Gate stays closed: a second call also fails.
        with pytest.raises(HTTPException) as exc_info_2:
            await client.create_completion(request, cache)
        assert exc_info_2.value.status_code == 500

    @pytest.mark.asyncio
    async def test_gate_opens_after_first_success(self, monkeypatch):
        """recompute_enabled=True → first request succeeds; gate flips; later requests skip check."""
        client = _make_remote_client(recompute_enabled=True)
        assert client._recompute_verified is False

        # Stub the OpenRouter call to return a minimal completion.
        fake_completion = ChatCompletion(
            id="chatcmpl-remote-1",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                )
            ],
            created=0,
            model="openai/gpt-4o-mini",
            object="chat.completion",
        )
        monkeypatch.setattr(
            client, "_call_openrouter", lambda *a, **kw: _async_return(fake_completion)
        )
        # Stub prompt tokenization to avoid needing a real tokenizer.
        monkeypatch.setattr(
            "areal.experimental.openai.proxy.remote_rollout.apply_chat_template",
            lambda *a, **kw: [1, 2, 3],
        )
        monkeypatch.setattr(client.tokenizer, "encode", lambda *a, **kw: [10, 20])

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = await client.create_completion(request, cache)

        assert result.id == "chatcmpl-remote-1"
        assert client._recompute_verified is True

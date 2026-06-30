# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenRouter remote rollout proxy.

Spec: docs/superpowers/specs/2026-06-29-openrouter-remote-rollout-proxy-design.md
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from areal.api import cli_args as cli_args_module
from areal.api.cli_args import PPOActorConfig
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward

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


# ---------------------------------------------------------------------------
# Real tokenizer fixture (spec tests 8-13).
#
# Deviation from plan: load via AutoTokenizer.from_pretrained directly instead
# of tests.utils.get_model_path + areal.utils.hf_utils.load_hf_tokenizer.
# The plan's path imports areal.utils.testing_utils, which builds a
# module-level DENSE_MODEL_PATHS dict that calls get_model_path for ~8 models
# at import time — most are not cached in this environment, so the import
# hangs on network downloads. load_hf_tokenizer also passes force_download=True
# which forces a network call even when the tokenizer is cached. Direct
# AutoTokenizer.from_pretrained uses the HF cache without forcing a download.
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


def _make_chat_completion(
    completion_id: str = "chatcmpl-remote-1",
    content: str = "Hello there.",
    finish_reason: str = "stop",
    tool_calls=None,
) -> ChatCompletion:
    """Build a minimal ChatCompletion as OpenRouter would return."""
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return ChatCompletion(
        id=completion_id,
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=message,
            )
        ],
        created=0,
        model="openai/gpt-4o-mini",
        object="chat.completion",
    )


# ---------------------------------------------------------------------------
# Tests: remote client happy path and failures (spec tests 8-13)
# ---------------------------------------------------------------------------


class TestCreateCompletionHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_stores_interaction(self, real_tokenizer, monkeypatch):
        """OpenRouter returns a completion → InteractionWithTokenLogpReward stored correctly."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer,
            chat_template_type="hf",
            recompute_enabled=True,
        )
        fake_completion = _make_chat_completion(
            completion_id="chatcmpl-remote-xyz",
            content="The answer is 42.",
            finish_reason="stop",
        )
        monkeypatch.setattr(
            client,
            "_call_openrouter",
            lambda *a, **kw: _async_return(fake_completion),
        )

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the answer?"}],
        }
        result = await client.create_completion(request, cache)

        # Returned completion is the remote one, unmodified.
        assert result is fake_completion
        # Cache contains the interaction under the remote ID.
        assert "chatcmpl-remote-xyz" in cache
        interaction = cache["chatcmpl-remote-xyz"]
        assert isinstance(interaction, InteractionWithTokenLogpReward)
        assert interaction.completion is fake_completion
        # ModelResponse built with local tokenization.
        assert interaction.model_response is not None
        mr = interaction.model_response
        assert len(mr.input_tokens) > 0
        # Output tokens = local encoding of "The answer is 42." + EOS.
        expected_output = real_tokenizer.encode(
            "The answer is 42.", add_special_tokens=False
        ) + [real_tokenizer.eos_token_id]
        assert mr.output_tokens == expected_output
        # Placeholder logprobs and versions.
        assert mr.output_logprobs == [0.0] * len(mr.output_tokens)
        assert mr.output_versions == [-1] * len(mr.output_tokens)
        assert mr.stop_reason == "stop"
        # Output message list preserved from remote.
        assert interaction.output_message_list == [
            fake_completion.choices[0].message.model_dump(exclude_none=True)
        ]
        # Gate flipped.
        assert client._recompute_verified is True


class TestCreateCompletionFailures:
    @pytest.mark.asyncio
    async def test_empty_remote_output_returns_400(self, real_tokenizer, monkeypatch):
        """Empty content and no tool_calls → 400, no cache entry."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )
        fake_completion = _make_chat_completion(content="", finish_reason="stop")
        monkeypatch.setattr(
            client, "_call_openrouter", lambda *a, **kw: _async_return(fake_completion)
        )
        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 400
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_openrouter_4xx_returns_502_no_cache(
        self, real_tokenizer, monkeypatch
    ):
        """OpenRouter APIStatusError → 502, no cache entry.

        Deviation from plan: stub the underlying OpenAI client (via
        _get_openrouter_client) rather than _call_openrouter, so the real
        _call_openrouter wrapping logic (APIStatusError → 502) actually runs.
        The plan's stub of _call_openrouter bypassed that wrapping.
        """
        from openai import APIStatusError

        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )

        async def _raise(*a, **kw):
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers.get.return_value = None
            raise APIStatusError(
                message="upstream 429",
                response=mock_response,
                body=None,
            )

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = _raise
        monkeypatch.setattr(
            client, "_get_openrouter_client", lambda: mock_openai_client
        )

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 502
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_openrouter_network_error_returns_504_no_cache(
        self, real_tokenizer, monkeypatch
    ):
        """OpenRouter APIError (network) → 504, no cache entry.

        Deviation from plan: stub the underlying OpenAI client (via
        _get_openrouter_client) rather than _call_openrouter, so the real
        _call_openrouter wrapping logic (APIError → 504) actually runs.
        """
        from openai import APIConnectionError

        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )

        async def _raise(*a, **kw):
            raise APIConnectionError(request=None)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = _raise
        monkeypatch.setattr(
            client, "_get_openrouter_client", lambda: mock_openai_client
        )

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 504
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_output_tokenization_failure_removes_cache_entry(
        self, real_tokenizer, monkeypatch
    ):
        """tokenizer.encode raising after cache insert → 500, cache entry removed.

        Deviation from plan: the flaky encode (raise on 2nd call) never
        triggered because apply_chat_template uses tokenizer.apply_chat_template,
        not tokenizer.encode — so encode is called exactly once (at step 6).
        Make encode raise on the first call to exercise the post-cache-insert
        cleanup path.
        """
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )
        fake_completion = _make_chat_completion(
            content="some output", finish_reason="stop"
        )
        monkeypatch.setattr(
            client, "_call_openrouter", lambda *a, **kw: _async_return(fake_completion)
        )

        def _failing_encode(text, add_special_tokens=False):
            raise RuntimeError("simulated encode failure")

        monkeypatch.setattr(real_tokenizer, "encode", _failing_encode)

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 500
        # Cache entry was inserted at step 5 then removed on failure.
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_500_before_network(
        self, real_tokenizer, monkeypatch
    ):
        """No OPENROUTER_API_KEY → 500 before any network call."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Reset the lazy client so _get_openrouter_client re-reads env.
        client._openrouter_client = None

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 500
        assert "OPENROUTER_API_KEY" in exc_info.value.detail
        assert len(cache) == 0

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.experimental.openai.cache import InteractionCache
from areal.utils import logging

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger("RemoteRollout")

_REMOTE_PREFIX = "remote:"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _map_finish_reason(finish_reason: str | None) -> str:
    """Map an OpenRouter/OpenAI finish_reason to a ModelResponse stop_reason.

    See spec "Resolved Implementation Questions > 3. Finish-reason mapping".
    Error completions are never cached (raised as HTTP 502/504 before this
    function is called), so "abort" is never produced here.
    """
    if finish_reason == "length":
        return "length"
    if finish_reason == "tool_calls":
        return "tool_calls"
    # "stop", "content_filter", None, and any future reason collapse to "stop".
    return "stop"


class RemoteRolloutClient:
    """OpenRouter-backed remote rollout client.

    Calls OpenRouter with the stripped model name, retokenizes the remote
    assistant output with the local tokenizer, builds a local ModelResponse
    with placeholder logprobs, and stores an InteractionWithTokenLogpReward
    in the session cache.

    Privacy: prompts, tool schemas, tool outputs, and conversation context
    are sent to OpenRouter and the selected upstream provider. Do not use
    remote rollout with data that must stay local.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        chat_template_type: str,
        engine_max_tokens: int | None = None,
        recompute_enabled: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template_type = chat_template_type
        self.engine_max_tokens = engine_max_tokens
        self.recompute_enabled = recompute_enabled
        self._recompute_verified: bool = False
        self._openrouter_client: AsyncOpenAI | None = None

    def _get_openrouter_client(self) -> AsyncOpenAI:
        """Lazily construct the OpenRouter AsyncOpenAI client from env vars.

        Raises HTTPException(500) if OPENROUTER_API_KEY is missing.
        """
        if self._openrouter_client is not None:
            return self._openrouter_client
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY is not set; cannot call OpenRouter for remote rollout.",
            )
        base_url = os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL)
        self._openrouter_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(
            "Constructed OpenRouter client (base_url=%s). "
            "Remote rollout sends prompts and conversation context to OpenRouter.",
            base_url,
        )
        return self._openrouter_client

    async def _call_openrouter(
        self, provider_model: str, forwarded: dict[str, Any]
    ) -> ChatCompletion:
        """Call OpenRouter and return the non-streaming ChatCompletion.

        Wrapped separately so tests can monkeypatch this method.
        """
        client = self._get_openrouter_client()
        try:
            return await client.chat.completions.create(
                model=provider_model,
                stream=False,
                n=1,
                **forwarded,
            )
        except HTTPException:
            raise
        except Exception as e:
            # Distinguish HTTP errors from network errors for the status code.
            from openai import APIError, APIStatusError

            if isinstance(e, APIStatusError):
                raise HTTPException(
                    status_code=502,
                    detail=f"OpenRouter upstream error: {e.status_code}: {e}",
                )
            if isinstance(e, APIError):
                raise HTTPException(
                    status_code=504,
                    detail=f"OpenRouter network error: {type(e).__name__}: {e}",
                )
            raise HTTPException(
                status_code=504,
                detail=f"OpenRouter request failed: {type(e).__name__}: {e}",
            )

    async def create_completion(
        self,
        request: dict[str, Any],
        session_cache: InteractionCache,
    ) -> ChatCompletion:
        """Process a remote rollout request through OpenRouter.

        Implements the 10-step per-request lifecycle from the design spec.
        """
        raise NotImplementedError("Implemented in Task 4")

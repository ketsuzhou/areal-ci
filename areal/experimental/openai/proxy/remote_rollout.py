# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import ModelResponse
from areal.experimental.openai._prompt_utils import (
    _ensure_message_dict_list,
    _extract_images_from_messages,
    apply_chat_template,
)
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
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
        The cache entry is created at step 5 (after the OpenRouter call
        succeeds and the completion ID is known). Every failure path
        from step 5 onward deletes the cache entry before raising.
        """
        model = request.get("model", "default")

        # --- Step 1: Parse & validate ---
        provider_model = model.removeprefix(_REMOTE_PREFIX)
        if not provider_model:
            raise HTTPException(status_code=400, detail="remote: model name is empty")
        if self.chat_template_type != "hf":
            raise HTTPException(
                status_code=400,
                detail="remote rollout MVP supports chat_template_type='hf' only",
            )
        if request.get("stream") is True:
            raise HTTPException(
                status_code=400, detail="remote streaming not supported in MVP"
            )
        if request.get("n", 1) != 1:
            raise HTTPException(status_code=400, detail="remote n != 1 not supported")
        if not self.recompute_enabled and not self._recompute_verified:
            raise HTTPException(
                status_code=500,
                detail="remote rollout requires actor.recompute_logprob=true "
                "or actor.use_decoupled_loss=true (placeholder remote logprobs "
                "would corrupt PPO ratios without recompute)",
            )

        messages_list = _ensure_message_dict_list(
            "messages", list(request.get("messages", []))
        )
        if not messages_list:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        # --- Step 2: Tokenize prompt locally (hf mode) ---
        tools_list = None
        if not _is_omitted(request.get("tools")):
            tools_list = list(request["tools"])
        extra_body = request.get("extra_body") or {}
        chat_template_kwargs = extra_body.get("chat_template_kwargs", {})

        image_data, messages_for_tokenizer, _ = _extract_images_from_messages(
            messages_list
        )
        tokenizer_messages = messages_for_tokenizer if image_data else messages_list
        try:
            prompt_token_ids = apply_chat_template(
                self.tokenizer,
                tokenizer_messages,
                tools=tools_list,
                add_generation_prompt=True,
                tokenize=True,
                **chat_template_kwargs,
            )
        except Exception as e:
            # Failure before cache insert — no cleanup needed.
            raise HTTPException(
                status_code=500,
                detail=f"prompt tokenization failed: {type(e).__name__}: {e}",
            )

        # --- Step 3: Call OpenRouter ---
        forwarded = _build_forwarded_params(request)
        remote_completion = await self._call_openrouter(provider_model, forwarded)

        # --- Step 4: Extract assistant output & determine cache key ---
        choice = remote_completion.choices[0]
        output_text = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        completion_id = remote_completion.id
        if not completion_id or completion_id in session_cache:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        if not output_text and not tool_calls:
            raise HTTPException(
                status_code=400,
                detail="remote returned empty completion (no content and no tool_calls)",
            )

        # --- Step 5: Insert interaction into cache ---
        interaction = InteractionWithTokenLogpReward(
            messages=deepcopy(messages_list),
            chat_template_type="hf",
        )
        session_cache[completion_id] = interaction

        try:
            # --- Step 6: Tokenize output locally ---
            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            output_tokens = output_tokens + [self.tokenizer.eos_token_id]
            if not output_tokens:
                raise ValueError("output tokenization produced no tokens")

            # --- Step 7: Map finish reason ---
            stop_reason = _map_finish_reason(choice.finish_reason)

            # --- Step 8: Build local ModelResponse ---
            model_response = ModelResponse(
                input_tokens=prompt_token_ids,
                output_tokens=output_tokens,
                output_logprobs=[0.0] * len(output_tokens),
                output_versions=[-1] * len(output_tokens),
                stop_reason=stop_reason,
                tokenizer=self.tokenizer,
            )

            # --- Step 9: Store interaction fields ---
            interaction.completion = remote_completion
            interaction.model_response = model_response
            interaction.output_message_list = [
                choice.message.model_dump(exclude_none=True)
            ]
        except HTTPException:
            del session_cache[completion_id]
            raise
        except Exception as e:
            del session_cache[completion_id]
            raise HTTPException(
                status_code=500,
                detail=f"remote rollout post-processing failed: {type(e).__name__}: {e}",
            )

        # --- Step 10: Flip the recompute gate and return ---
        self._recompute_verified = True
        return remote_completion


def _is_omitted(value: Any) -> bool:
    """True if value is None, NOT_GIVEN, or an Omit sentinel."""
    if value is None:
        return True
    try:
        from openai import NOT_GIVEN, Omit

        if value is NOT_GIVEN or isinstance(value, Omit):
            return True
    except ImportError:
        pass
    if hasattr(value, "__class__"):
        return value.__class__.__name__ in ("NotGiven", "Omit")
    return False


def _build_forwarded_params(request: dict[str, Any]) -> dict[str, Any]:
    """Build the kwargs dict to forward to OpenRouter.

    Forwards sampling/tool params. Strips model, stream, n, areal_cache, store
    (model is passed explicitly; the others are not OpenRouter params).
    """
    _STRIPPED = {
        "model",
        "stream",
        "n",
        "areal_cache",
        "store",
        "messages",
        "tools",
        "extra_body",
    }
    _FORWARDABLE = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "tool_choice",
        "frequency_penalty",
        "metadata",
        "tools",
    }
    forwarded: dict[str, Any] = {}
    for key, value in request.items():
        if key in _STRIPPED:
            continue
        if key not in _FORWARDABLE:
            continue
        if _is_omitted(value):
            continue
        forwarded[key] = value
    # tools is forwarded separately (re-added after the STRIPPED check above
    # because it appears in both sets — we want it forwarded, not stripped).
    if not _is_omitted(request.get("tools")):
        forwarded["tools"] = list(request["tools"])
    return forwarded

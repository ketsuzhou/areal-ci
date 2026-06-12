"""Async client for calling a remote teacher model inference API.

This module provides TeacherConfig and TeacherClient for querying a remote
teacher model (vLLM, SGLang, or Fireworks) to get logprobs for student
candidate tokens during on-policy distillation.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from areal.utils import logging

try:
    from dotenv import load_dotenv

    _DOTENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_DOTENV_FILE)
except ImportError:
    pass

logger = logging.getLogger("TeacherClient")

_DEFAULT_MISSING_LOGPROB = math.log(1e-10)


class TeacherServiceError(RuntimeError):
    """Raised when the remote teacher service cannot satisfy a request."""


@dataclass
class TeacherConfig:
    """Configuration for the remote teacher model.

    Attributes
    ----------
    teacher_base_url : str
        Base URL of the teacher model inference server.
    teacher_model_name : str
        Model name to pass in API requests (empty for single-model servers).
    teacher_api_key : str
        API key for authentication.
    teacher_top_k : int
        Number of top logprobs to request from the teacher.
    teacher_max_retries : int
        Maximum number of retries on transient HTTP failures.
    teacher_timeout : float
        Request timeout in seconds.
    teacher_missing_logprob : float
        Logprob value assigned to candidate tokens not found in the
        teacher's top-k response.
    teacher_backend : str
        Backend type: ``"openai"`` (vLLM-compatible /v1/completions),
        ``"sglang"`` (SGLang native /generate), or ``"fireworks"``
        (Fireworks /v1/chat/completions with echo logprobs).
    """

    teacher_base_url: str = "http://localhost:8001"
    teacher_model_name: str = ""
    teacher_api_key: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_timeout: float = 300.0
    teacher_missing_logprob: float = _DEFAULT_MISSING_LOGPROB
    teacher_backend: str = "openai"
    teacher_max_concurrency: int = 4

    _FIREWORKS_DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/"

    def __post_init__(self) -> None:
        if self.teacher_backend == "fireworks":
            if self.teacher_base_url == "http://localhost:8001":
                env_url = os.getenv(
                    "FIREWORKS_BASE_URL", self._FIREWORKS_DEFAULT_BASE_URL
                )
                self.teacher_base_url = env_url
            if not self.teacher_api_key:
                env_key = os.getenv("FIREWORKS_API_KEY", "")
                if env_key:
                    self.teacher_api_key = env_key


class TeacherClient:
    """Async client for calling a remote teacher model inference API.

    Supports three backends:

    - ``"openai"`` (default): vLLM-compatible ``/v1/completions`` endpoint
      with ``echo=True`` and text/token prompts.
    - ``"sglang"``: SGLang native ``/generate`` endpoint with ``input_ids``
      and ``top_k_logprobs_num``.
    - ``"fireworks"``: Fireworks ``/v1/chat/completions`` endpoint with
      ``echo=True`` and text prompts.  Requires a tokenizer to map between
      token IDs and token text.

    The underlying httpx.AsyncClient is created in __init__ so the client
    is ready to use immediately — no async context manager needed.
    Call ``await client.close()`` when done to release the connection pool.

    Parameters
    ----------
    config : TeacherConfig
        Teacher model configuration.
    """

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        base_url = config.teacher_base_url
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"teacher_base_url must start with http:// or https://, "
                f"got: {config.teacher_base_url!r}"
            )
        if not base_url.endswith("/"):
            base_url += "/"
        headers: dict[str, str] = {}
        if config.teacher_api_key:
            headers["Authorization"] = f"Bearer {config.teacher_api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(config.teacher_timeout),
            headers=headers,
        )
        self._semaphore = asyncio.Semaphore(config.teacher_max_concurrency)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_logprobs_for_candidates(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],
        tokenizer: Any = None,
    ) -> list[dict[int, float]]:
        """Get teacher logprobs for student candidate tokens at each position.

        Parameters
        ----------
        input_ids : list[int]
            Token IDs of the prompt/prefix.
        output_ids : list[int]
            Token IDs of the generated output.
        candidate_token_ids : list[list[int]]
            For each output position, a list of candidate token IDs from the
            student. Shape: ``[num_positions][num_candidates]``.
        tokenizer : Any, optional
            Tokenizer (reserved for future use, e.g. token ID validation).

        Returns
        -------
        list[dict[int, float]]
            For each output position, a mapping from token ID to teacher
            logprob. Length equals ``len(output_ids)``.

        Raises
        ------
        RuntimeError
            If the API returns an error after all retries are exhausted.
        """
        if self.config.teacher_backend == "sglang":
            return await self._get_logprobs_sglang(
                input_ids, output_ids, candidate_token_ids
            )
        if self.config.teacher_backend == "fireworks":
            return await self._get_logprobs_fireworks(
                input_ids, output_ids, candidate_token_ids, tokenizer
            )
        return await self._get_logprobs_openai(
            input_ids, output_ids, candidate_token_ids
        )

    async def _get_logprobs_sglang(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[dict[int, float]]:
        """Get teacher logprobs for student output tokens via SGLang /generate.

        Sends the full student sequence (input_ids + output_ids) as the prompt
        with ``logprob_start_len=len(input_ids)`` so SGLang returns top-k
        logprobs for the output positions.  The teacher model evaluates the
        student's tokens (forward pass over the full sequence) and we extract
        the teacher's probability distribution at each student output position.
        """
        num_output_tokens = len(output_ids)
        prompt_len = len(input_ids)
        all_ids = input_ids + output_ids

        logger.info(
            "Teacher logprob request (sglang): prompt_tokens=%d, output_tokens=%d, total_tokens=%d",
            prompt_len,
            num_output_tokens,
            len(all_ids),
        )

        payload: dict[str, Any] = {
            "input_ids": all_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            "return_logprob": True,
            "logprob_start_len": prompt_len,
            "top_logprobs_num": self.config.teacher_top_k,
            "stream": False,
        }

        response_data = await self._post_with_retries(payload)

        meta_info = response_data.get("meta_info", {})
        # SGLang returns input_top_logprobs for prompt positions covered
        # by logprob_start_len, and output_top_logprobs for generated tokens.
        input_top_logprobs = meta_info.get("input_top_logprobs")
        output_top_logprobs = meta_info.get("output_top_logprobs")

        # Combine: student output positions are in input_top_logprobs
        # (they are part of the "prompt" sent to SGLang).  The single
        # generated token's logprobs go in output_top_logprobs.
        all_top_logprobs: list[Any] = []
        if input_top_logprobs is not None:
            all_top_logprobs.extend(input_top_logprobs)
        if output_top_logprobs is not None:
            all_top_logprobs.extend(output_top_logprobs)

        if len(all_top_logprobs) < num_output_tokens:
            logger.warning(
                "SGLang teacher returned fewer logprob positions (%d) than "
                "expected (%d). Padding with missing_logprob.",
                len(all_top_logprobs),
                num_output_tokens,
            )

        result: list[dict[int, float]] = []
        missing_logprob = self.config.teacher_missing_logprob

        for pos_idx in range(num_output_tokens):
            teacher_logprob_map: dict[int, float] = {}
            if (
                pos_idx < len(all_top_logprobs)
                and all_top_logprobs[pos_idx] is not None
            ):
                pos_data = all_top_logprobs[pos_idx]
                # SGLang returns list of (logprob, token_id, token_text) tuples
                if isinstance(pos_data, list):
                    for entry in pos_data:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            lp, tid = entry[0], entry[1]
                            if isinstance(tid, int):
                                teacher_logprob_map[tid] = float(lp)
                elif isinstance(pos_data, dict):
                    for key, value in pos_data.items():
                        if isinstance(key, int):
                            teacher_logprob_map[key] = float(value)

            candidate_map: dict[int, float] = {}
            for tid in candidate_token_ids[pos_idx]:
                candidate_map[tid] = teacher_logprob_map.get(tid, missing_logprob)
            result.append(candidate_map)

        return result

    async def _get_logprobs_openai(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[dict[int, float]]:
        """Get teacher logprobs for student output tokens via vLLM /v1/completions.

        Sends the full student sequence (input_ids + output_ids) as the prompt
        with ``echo=True`` and ``max_tokens=1`` so the teacher model runs a
        forward pass over the entire student sequence.  The top-k logprobs
        at the student's output positions give the teacher's evaluation of the
        student's token choices.
        """
        num_output_tokens = len(output_ids)
        prompt_len = len(input_ids)
        all_ids = input_ids + output_ids

        logger.info(
            "Teacher logprob request: prompt_tokens=%d, output_tokens=%d, total_tokens=%d",
            prompt_len,
            num_output_tokens,
            len(all_ids),
        )

        payload: dict[str, Any] = {
            "prompt": all_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": self.config.teacher_top_k,
            "echo": True,
        }
        if self.config.teacher_model_name:
            payload["model"] = self.config.teacher_model_name

        response_data = await self._post_with_retries(payload)

        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError("Teacher API returned no choices")

        logprobs_data = choices[0].get("logprobs", {})
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        prompt_len = len(input_ids)
        output_top_logprobs = top_logprobs_list[
            prompt_len : prompt_len + num_output_tokens
        ]

        if len(output_top_logprobs) < num_output_tokens:
            logger.warning(
                "Teacher returned fewer logprob positions (%d) than "
                "expected (%d). Padding with missing_logprob.",
                len(output_top_logprobs),
                num_output_tokens,
            )

        result: list[dict[int, float]] = []
        missing_logprob = self.config.teacher_missing_logprob

        for pos_idx in range(num_output_tokens):
            teacher_logprob_map: dict[int, float] = {}
            if (
                pos_idx < len(output_top_logprobs)
                and output_top_logprobs[pos_idx] is not None
            ):
                for token_entry in output_top_logprobs[pos_idx]:
                    if isinstance(token_entry, dict):
                        tid = token_entry.get("token_id")
                        if tid is None:
                            tid = token_entry.get("id")
                        lp = token_entry.get("logprob", missing_logprob)
                        if tid is not None:
                            teacher_logprob_map[tid] = float(lp)

            candidate_map: dict[int, float] = {}
            for tid in candidate_token_ids[pos_idx]:
                candidate_map[tid] = teacher_logprob_map.get(tid, missing_logprob)
            result.append(candidate_map)

        return result

    async def _get_logprobs_fireworks(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],
        tokenizer: Any = None,
    ) -> list[dict[int, float]]:
        """Get teacher logprobs via Fireworks chat completions API with echo.

        Sends the full student sequence (prompt + output) decoded to text as a
        user message.  With ``echo=True`` the Fireworks API returns logprobs
        for every content token.  Output positions are identified via
        ``text_offset >= len(prompt_text)``.  Token text from ``top_logprobs``
        is mapped back to token IDs via the tokenizer.
        """
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required for the fireworks backend "
                "to convert between token IDs and token text"
            )

        num_output_tokens = len(output_ids)
        prompt_text = tokenizer.decode(input_ids)
        output_text = tokenizer.decode(output_ids)
        full_text = prompt_text + output_text
        prompt_char_len = len(prompt_text)

        logger.info(
            "Teacher logprob request (fireworks): prompt_tokens=%d, "
            "output_tokens=%d, prompt_chars=%d, output_chars=%d",
            len(input_ids),
            num_output_tokens,
            prompt_char_len,
            len(output_text),
        )

        payload: dict[str, Any] = {
            "model": self.config.teacher_model_name,
            "messages": [{"role": "user", "content": full_text}],
            "max_tokens": 1,
            "temperature": 0.6,
            "logprobs": self.config.teacher_top_k,
            "echo": True,
        }

        response_data = await self._post_fireworks_with_retries(payload)

        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError("Fireworks teacher API returned no choices")

        logprobs_data = choices[0].get("logprobs", {})
        text_offsets = logprobs_data.get("text_offset", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        # Collect top_logprobs for output positions:
        # text_offset >= prompt_char_len marks output tokens,
        # text_offset >= len(full_text) marks the 1 generated token (skip it).
        output_top_logprobs: list[dict[str, float]] = []
        for i, offset in enumerate(text_offsets):
            if prompt_char_len <= offset < len(full_text):
                lp = top_logprobs_list[i] if i < len(top_logprobs_list) else None
                output_top_logprobs.append(lp if lp else {})

        if len(output_top_logprobs) < num_output_tokens:
            logger.warning(
                "Fireworks teacher returned fewer output logprob positions "
                "(%d) than expected (%d). Padding with missing_logprob.",
                len(output_top_logprobs),
                num_output_tokens,
            )

        result: list[dict[int, float]] = []
        missing_logprob = self.config.teacher_missing_logprob

        for pos_idx in range(num_output_tokens):
            top_lp = (
                output_top_logprobs[pos_idx]
                if pos_idx < len(output_top_logprobs)
                else {}
            )

            # Build token_id → logprob map by encoding top_logprobs keys
            teacher_id_map: dict[int, float] = {}
            for token_text, logprob in top_lp.items():
                try:
                    encoded = tokenizer.encode(token_text, add_special_tokens=False)
                    if len(encoded) == 1:
                        teacher_id_map[encoded[0]] = float(logprob)
                except Exception:
                    pass

            candidate_map: dict[int, float] = {}
            for tid in candidate_token_ids[pos_idx]:
                if tid in teacher_id_map:
                    candidate_map[tid] = teacher_id_map[tid]
                else:
                    # Fallback: decode candidate ID and look up by text
                    try:
                        decoded = tokenizer.decode([tid])
                        candidate_map[tid] = float(top_lp.get(decoded, missing_logprob))
                    except Exception:
                        candidate_map[tid] = missing_logprob
            result.append(candidate_map)

        return result

    async def complete_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return a text completion from the teacher API."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        selected_model = model or self.config.teacher_model_name
        if selected_model:
            payload["model"] = selected_model

        response_data = await self._post_with_retries(payload)
        if not isinstance(response_data, Mapping):
            raise RuntimeError("Teacher API completion response must be a mapping")

        choices = response_data.get("choices", [])
        if isinstance(choices, str | bytes) or not isinstance(choices, Sequence):
            raise RuntimeError("Teacher API completion choices must be a sequence")
        if not choices:
            raise RuntimeError("Teacher API returned no completion choices")

        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise RuntimeError("Teacher API completion choice must be a mapping")

        text = choice.get("text")
        if isinstance(text, str):
            return text

        if "message" in choice:
            message = choice["message"]
            if not isinstance(message, Mapping):
                raise RuntimeError("Teacher API completion message must be a mapping")
            text = message.get("content")
            if isinstance(text, str):
                return text

        if text is None:
            raise RuntimeError("Teacher API completion choice contained no text")
        raise RuntimeError("Teacher API completion choice contained no text string")

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Call the LLM via chat completions API using teacher_api_key and teacher_base_url."""
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        selected_model = model or self.config.teacher_model_name
        if selected_model:
            payload["model"] = selected_model

        response_data = await self._post_with_retries_chat(payload)
        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError("Teacher chat API returned no choices")

        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise RuntimeError("Teacher chat choice must be a mapping")

        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("Teacher chat message must be a mapping")

        content = message.get("content")
        if isinstance(content, str):
            return content

        raise RuntimeError("Teacher chat completion choice contained no text content")

    _BACKEND_ENDPOINTS: dict[str, str] = {
        "openai": "v1/completions",
        "sglang": "generate",
        "fireworks": "v1/chat/completions",
    }

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retry logic, dispatching to the correct endpoint."""
        async with self._semaphore:
            backend = self.config.teacher_backend
            endpoint = self._BACKEND_ENDPOINTS.get(backend, "v1/completions")
            return await self._post_with_retries_inner(endpoint, backend, payload)

    @staticmethod
    def _retry_backoff(attempt: int, status_code: int | None = None) -> float:
        """Compute retry delay with jitter. Longer for server errors (5xx)."""
        base = 8 if (status_code is not None and status_code >= 500) else 1
        backoff = base * (2 ** (attempt - 1))
        jitter = random.uniform(0, backoff * 0.5)
        return backoff + jitter

    async def _post_with_retries_inner(
        self,
        endpoint: str,
        backend: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST to the given endpoint with retry logic."""
        max_retries = self.config.teacher_max_retries
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                logger.info(
                    "Teacher %s %s success (attempt %d/%d)",
                    backend,
                    endpoint,
                    attempt,
                    max_retries,
                )
                return response.json()
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as exc:
                last_exc = exc
                status_code = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                logger.warning(
                    "Teacher %s API request failed (attempt %d/%d): %s",
                    backend,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(self._retry_backoff(attempt, status_code))

        raise TeacherServiceError(
            f"Teacher {backend} API request failed after "
            f"{max_retries} retries: {last_exc}"
        ) from last_exc

    async def _post_with_retries_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the chat completions endpoint with retry logic."""
        async with self._semaphore:
            return await self._post_with_retries_inner(
                "v1/chat/completions", "chat", payload
            )


__all__ = ["TeacherConfig", "TeacherClient", "TeacherServiceError"]

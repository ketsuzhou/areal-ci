"""Async client for calling a remote teacher model inference API.

This module provides TeacherConfig and TeacherClient for querying a remote
teacher model (vLLM or SGLang) to get logprobs for student candidate
tokens during on-policy distillation.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from areal.utils import logging

logger = logging.getLogger("TeacherClient")

_DEFAULT_MISSING_LOGPROB = math.log(1e-10)


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
        Backend type: ``"openai"`` (vLLM-compatible /v1/completions) or
        ``"sglang"`` (SGLang native /generate).
    """

    teacher_base_url: str = "http://localhost:8001"
    teacher_model_name: str = ""
    teacher_api_key: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_timeout: float = 300.0
    teacher_missing_logprob: float = _DEFAULT_MISSING_LOGPROB
    teacher_backend: str = "openai"


class TeacherClient:
    """Async client for calling a remote teacher model inference API.

    Supports two backends:

    - ``"openai"`` (default): vLLM-compatible ``/v1/completions`` endpoint
      with ``echo=True`` and text/token prompts.
    - ``"sglang"``: SGLang native ``/generate`` endpoint with ``input_ids``
      and ``top_k_logprobs_num``.

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
        if not config.teacher_base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"teacher_base_url must start with http:// or https://, "
                f"got: {config.teacher_base_url!r}"
            )
        headers: dict[str, str] = {}
        if config.teacher_api_key:
            headers["Authorization"] = f"Bearer {config.teacher_api_key}"
        self._client = httpx.AsyncClient(
            base_url=config.teacher_base_url,
            timeout=httpx.Timeout(config.teacher_timeout),
            headers=headers,
        )

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

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retry logic, dispatching to the correct endpoint."""
        if self.config.teacher_backend == "sglang":
            return await self._post_sglang_with_retries(payload)
        return await self._post_openai_with_retries(payload)

    async def _post_sglang_with_retries(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST to SGLang /generate endpoint with retry logic."""
        max_retries = self.config.teacher_max_retries
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._client.post("/generate", json=payload)
                response.raise_for_status()
                logger.info(
                    "SGLang teacher /generate success (attempt %d/%d, "
                    "prompt_len=%d, output_len=%d)",
                    attempt,
                    max_retries,
                    len(payload.get("input_ids", [])),
                    payload.get("sampling_params", {}).get("max_new_tokens", 0),
                )
                return response.json()
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as exc:
                last_exc = exc
                logger.warning(
                    "SGLang teacher API request failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    backoff = 2 ** (attempt - 1)
                    await asyncio.sleep(backoff)

        raise RuntimeError(
            f"SGLang teacher API request failed after {max_retries} retries: "
            f"{last_exc}"
        ) from last_exc

    async def _post_openai_with_retries(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST to the completions endpoint with retry logic."""
        max_retries = self.config.teacher_max_retries
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._client.post("/v1/completions", json=payload)
                response.raise_for_status()
                logger.info(
                    "Teacher /v1/completions success (attempt %d/%d)", attempt, max_retries
                )
                return response.json()
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as exc:
                last_exc = exc
                logger.warning(
                    "Teacher API request failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    backoff = 2 ** (attempt - 1)
                    await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Teacher API request failed after {max_retries} retries: {last_exc}"
        ) from last_exc

    async def _post_with_retries_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the chat completions endpoint with retry logic."""
        max_retries = self.config.teacher_max_retries
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as exc:
                last_exc = exc
                logger.warning(
                    "Teacher chat API request failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    backoff = 2 ** (attempt - 1)
                    await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Teacher chat API request failed after {max_retries} retries: {last_exc}"
        ) from last_exc


__all__ = ["TeacherConfig", "TeacherClient"]

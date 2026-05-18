"""Async client for calling a remote teacher model inference API.

This module provides TeacherConfig and TeacherClient for querying a remote
teacher model (vLLM/SGLang compatible) to get logprobs for student candidate
tokens during on-policy distillation.

Example
-------
>>> config = TeacherConfig(teacher_base_url="http://localhost:8001")
>>> client = TeacherClient(config)
>>> async with client:
...     result = await client.get_logprobs_for_candidates(
...         input_ids=[1, 2, 3],
...         output_ids=[4, 5],
...         candidate_token_ids=[[4, 10, 20], [5, 30, 40]],
...     )
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

# Default missing logprob: log(1e-10) ~ -23.025
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
    teacher_top_k : int
        Number of top logprobs to request from the teacher.
    teacher_max_retries : int
        Maximum number of retries on transient HTTP failures.
    teacher_timeout : float
        Request timeout in seconds.
    teacher_missing_logprob : float
        Logprob value assigned to candidate tokens not found in the
        teacher's top-k response.
    """

    teacher_base_url: str = "http://localhost:8001"
    teacher_model_name: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_timeout: float = 60.0
    teacher_missing_logprob: float = _DEFAULT_MISSING_LOGPROB


class TeacherClient:
    """Async client for calling a remote teacher model inference API.

    Uses the vLLM/SGLang compatible completions endpoint to retrieve
    top-k logprobs from the teacher model for given candidate token IDs.

    Parameters
    ----------
    config : TeacherConfig
        Teacher model configuration.

    Example
    -------
    >>> config = TeacherConfig(teacher_base_url="http://localhost:8001")
    >>> client = TeacherClient(config)
    >>> async with client:
    ...     result = await client.get_logprobs_for_candidates(
    ...         input_ids=[1, 2, 3],
    ...         output_ids=[4, 5],
    ...         candidate_token_ids=[[4, 10, 20], [5, 30, 40]],
    ...     )
    """

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TeacherClient:
        self._client = httpx.AsyncClient(
            base_url=self.config.teacher_base_url,
            timeout=httpx.Timeout(self.config.teacher_timeout),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Raise if the client is not open, otherwise return it."""
        if self._client is None:
            raise RuntimeError(
                "TeacherClient is not open. Use 'async with client:' context manager."
            )
        return self._client

    async def get_logprobs_for_candidates(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],
        tokenizer: Any = None,
    ) -> list[dict[int, float]]:
        """Get teacher logprobs for student candidate tokens at each position.

        Calls the teacher completions API with echo=True to get logprobs for
        all output positions, then maps the student's candidate token IDs to
        teacher logprobs. Tokens not found in the teacher's top-k get the
        configured ``missing_logprob`` value.

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
            If the client is not open or the API returns an error after
            all retries are exhausted.
        """
        client = self._ensure_client()

        num_output_tokens = len(output_ids)
        if len(candidate_token_ids) != num_output_tokens:
            raise ValueError(
                f"candidate_token_ids length ({len(candidate_token_ids)}) "
                f"must match output_ids length ({num_output_tokens})"
            )

        # Build the completions request.
        # Send input_ids as the prompt (token IDs) and request max_tokens
        # equal to the output length. echo=True ensures logprobs are returned
        # for all positions including prompt tokens.
        payload: dict[str, Any] = {
            "prompt": input_ids,
            "max_tokens": num_output_tokens,
            "temperature": 0.0,
            "logprobs": self.config.teacher_top_k,
            "echo": True,
        }
        if self.config.teacher_model_name:
            payload["model"] = self.config.teacher_model_name

        response_data = await self._post_with_retries(payload)

        # Parse logprobs from the response.
        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError("Teacher API returned no choices")

        logprobs_data = choices[0].get("logprobs", {})
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        # The API returns logprobs for all tokens in the sequence
        # (prompt + generated). We only need the positions corresponding to
        # the output tokens, which start at index len(input_ids).
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

        # Map candidate token IDs to teacher logprobs for each position.
        result: list[dict[int, float]] = []
        missing_logprob = self.config.teacher_missing_logprob

        for pos_idx in range(num_output_tokens):
            candidate_map: dict[int, float] = {}

            # Build a lookup from the teacher's top-k for this position.
            teacher_logprob_map: dict[int, float] = {}
            if (
                pos_idx < len(output_top_logprobs)
                and output_top_logprobs[pos_idx] is not None
            ):
                for token_entry in output_top_logprobs[pos_idx]:
                    # Each entry is like {"token_id": int, "logprob": float}
                    # or {"token": str, "logprob": float} depending on API.
                    # vLLM/SGLang with prompt_token_ids returns token_id fields.
                    if isinstance(token_entry, dict):
                        tid = token_entry.get("token_id")
                        if tid is None:
                            # Fallback: some APIs return token string + id
                            tid = token_entry.get("id")
                        lp = token_entry.get("logprob", missing_logprob)
                        if tid is not None:
                            teacher_logprob_map[tid] = float(lp)

            # Map each candidate token ID to the teacher logprob.
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

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the completions endpoint with retry logic.

        Parameters
        ----------
        payload : dict[str, Any]
            Request payload for the completions API.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        client = self._ensure_client()
        max_retries = self.config.teacher_max_retries
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post("/v1/completions", json=payload)
                response.raise_for_status()
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
                    # Exponential backoff: 1s, 2s, 4s, ...
                    backoff = 2 ** (attempt - 1)
                    await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Teacher API request failed after {max_retries} retries: {last_exc}"
        ) from last_exc


__all__ = ["TeacherConfig", "TeacherClient"]

# SPDX-License-Identifier: Apache-2.0
"""Frozen SGLang reference candidate scorer for VIMPO.

The reference policy is ``pi_ref = pi_0`` - the actor's INITIAL checkpoint -
and is never weight-updated. ``SGLangVIMPOReferenceScorer`` validates that the
SGLang service serves that exact checkpoint (model path, revision, vocabulary,
special-token IDs, temperature 1, no quantization) via ``/get_model_info``,
then scores the actor's per-position candidate token IDs against that frozen
reference using full-softmax (``temperature=1.0``) log-probabilities.

Scoring is synchronous (suitable for PPO worker RPC): a bounded
``ThreadPoolExecutor`` wraps one shared ``httpx.Client``. Requests are sorted
by ``(len(prefix_ids), key)`` for radix-cache friendly submission; results are
restored to the caller's input order by ``key``.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from areal.utils import logging

__all__ = [
    "ReferenceIdentity",
    "ReferenceScore",
    "ReferenceScoreRequest",
    "SGLangVIMPOReferenceScorer",
    "VIMPOReferenceScorer",
]

logger = logging.getLogger("VIMPOReference")


# ---------------------------------------------------------------------------
# Immutable identity and request/response types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceIdentity:
    """Canonical identity of the frozen reference checkpoint.

    Every field is compared against ``/get_model_info``; any mismatch fails
    ``validate_identity`` before scoring begins. An empty ``revision`` is a
    wildcard (stock SGLang does not report one) and is not compared.
    """

    model_path: str
    revision: str
    vocab_size: int
    tokenizer_vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None


@dataclass(frozen=True)
class ReferenceScoreRequest:
    """One valid response position to score against the frozen reference.

    ``key`` is the caller's position identifier (restored in the result).
    ``prefix_ids`` is the prompt + already-generated tokens up to this position.
    ``sampled_token_id`` is the token the actor actually sampled.
    ``candidate_token_ids`` is the actor's ordered top-k candidate set.
    """

    key: tuple[int, int]
    prefix_ids: list[int]
    sampled_token_id: int
    candidate_token_ids: list[int]


@dataclass(frozen=True)
class ReferenceScore:
    """Frozen reference log-probabilities for one position.

    ``candidate_logp`` preserves the original ``candidate_token_ids`` order
    (not the deduplicated submission order).
    """

    key: tuple[int, int]
    sampled_logp: float
    candidate_logp: list[float]


class VIMPOReferenceScorer(Protocol):
    """Protocol for a frozen reference candidate scorer."""

    def validate_identity(self, expected: ReferenceIdentity) -> None:
        raise NotImplementedError

    def score(self, requests: list[ReferenceScoreRequest]) -> list[ReferenceScore]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGLang adapter
# ---------------------------------------------------------------------------


_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SGLangVIMPOReferenceScorer:
    """Synchronous SGLang adapter for frozen reference candidate scoring.

    Uses one shared ``httpx.Client`` (thread-safe for concurrent requests)
    wrapped by a ``ThreadPoolExecutor(max_workers=max_concurrency)``. Retries
    only ``httpx.TransportError``, HTTP 429, and HTTP 5xx up to ``max_retries``.
    After a transport reconnect, identity is re-validated before scoring resumes.
    The cumulative retry count is exposed as the ``retry_count`` attribute (read
    by the actor's ``vimpo/reference_retries`` metric).
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        max_concurrency: int = 8,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(base_url=self._base_url, timeout=timeout)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency
        )
        self._expected_identity: ReferenceIdentity | None = None
        self._needs_revalidate = False
        self._revalidate_lock = threading.Lock()
        self.retry_count = 0
        self._retry_count_lock = threading.Lock()
        self._closed = False

    # -- Identity validation ---------------------------------------------

    def validate_identity(self, expected: ReferenceIdentity) -> None:
        """Fetch ``/get_model_info`` and compare every field + temperature + quantized.

        Stock SGLang routes ``/get_model_info`` as GET-only, so the request
        uses GET (a POST would 405 against an unpatched server).
        """
        if self._closed:
            raise RuntimeError("scorer is closed")
        response = self._client.get(f"{self._base_url}/get_model_info")
        if response.status_code != 200:
            response.raise_for_status()
        info = response.json()
        self._compare_identity(info, expected)
        self._expected_identity = expected

    @staticmethod
    def _compare_identity(info: dict[str, Any], expected: ReferenceIdentity) -> None:
        """Raise ``ValueError`` naming the mismatched field.

        An empty expected ``revision`` is a wildcard: stock SGLang does not
        report a ``revision`` field in ``/get_model_info``, so pinning one
        would fail every real run. A non-empty expected revision is still
        compared exactly.

        Fields the server OMITS (``None``) are unverifiable - stock SGLang
        does not report vocabulary sizes or special-token IDs - so they are
        skipped with a logged warning (same precedent as the revision
        wildcard). Fields the server REPORTS must still match exactly.
        """
        pairs: list[tuple[str, Any, Any]] = [
            ("model_path", info.get("model_path"), expected.model_path),
            ("vocab_size", info.get("vocab_size"), expected.vocab_size),
            (
                "tokenizer_vocab_size",
                info.get("tokenizer_vocab_size"),
                expected.tokenizer_vocab_size,
            ),
            ("bos_token_id", info.get("bos_token_id"), expected.bos_token_id),
            ("eos_token_id", info.get("eos_token_id"), expected.eos_token_id),
            ("pad_token_id", info.get("pad_token_id"), expected.pad_token_id),
        ]
        if expected.revision:
            pairs.append(("revision", info.get("revision"), expected.revision))
        for field, server_val, expected_val in pairs:
            if server_val is None:
                logger.warning(
                    "VIMPO reference /get_model_info omits %r; identity field "
                    "is unverifiable and skipped (expected=%r)",
                    field,
                    expected_val,
                )
                continue
            if server_val != expected_val:
                raise ValueError(
                    f"{field} mismatch: server={server_val!r} expected={expected_val!r}"
                )
        # Temperature must be exactly 1.0 for full-softmax scoring.
        server_temp = info.get("temperature", 1.0)
        if server_temp != 1.0:
            raise ValueError(
                f"temperature mismatch: server={server_temp!r} expected=1.0"
            )
        # Quantized scoring is forbidden in paper-faithful mode.
        if info.get("quantized", False) is not False:
            raise ValueError(
                f"quantized mismatch: server={info.get('quantized')!r} expected=False"
            )

    def _maybe_revalidate(self) -> None:
        """Re-run identity validation after a transport reconnect (thread-safe)."""
        with self._revalidate_lock:
            if not self._needs_revalidate or self._expected_identity is None:
                return
            self._needs_revalidate = False
        self.validate_identity(self._expected_identity)

    # -- Scoring ---------------------------------------------------------

    def score(self, requests: list[ReferenceScoreRequest]) -> list[ReferenceScore]:
        """Score a batch of requests, returning results in input order."""
        if self._closed:
            raise RuntimeError("scorer is closed")
        if not requests:
            return []

        # Sort by (prefix length, key) for radix-cache friendly submission.
        sorted_reqs = sorted(requests, key=lambda r: (len(r.prefix_ids), r.key))
        futures = {
            self._executor.submit(self._score_one, req): req.key for req in sorted_reqs
        }
        results_by_key: dict[tuple[int, int], ReferenceScore] = {}
        for future in concurrent.futures.as_completed(futures):
            score = future.result()
            results_by_key[score.key] = score
        # Restore caller's input order.
        return [results_by_key[req.key] for req in requests]

    def _score_one(self, request: ReferenceScoreRequest) -> ReferenceScore:
        """Score a single request with retries and optional chunking."""
        deduped = self._dedup_token_ids(request)
        token_map = self._post_with_retry(request, deduped)
        return ReferenceScore(
            key=request.key,
            sampled_logp=token_map[request.sampled_token_id],
            candidate_logp=[token_map[tid] for tid in request.candidate_token_ids],
        )

    @staticmethod
    def _dedup_token_ids(request: ReferenceScoreRequest) -> list[int]:
        """Candidates first (original order), then sampled if not already present."""
        deduped = list(request.candidate_token_ids)
        seen = set(deduped)
        if request.sampled_token_id not in seen:
            deduped.append(request.sampled_token_id)
        return deduped

    def _record_retry(self) -> None:
        """Increment the retry counter (thread-safe; read as ``retry_count``)."""
        with self._retry_count_lock:
            self.retry_count += 1

    def _post_with_retry(
        self, request: ReferenceScoreRequest, token_ids: list[int]
    ) -> dict[int, float]:
        """Send ``/generate`` with retries; return ``{token_id: logprob}``.

        Retries only ``httpx.TransportError``, HTTP 429, and HTTP 5xx. A 400
        response carrying ``max_token_ids_logprob`` triggers chunking (the
        token-ID list is split into chunks at or below the reported limit and
        each chunk is scored independently, then joined by token ID).
        """
        payload = self._build_payload(request, token_ids)
        for attempt in range(self._max_retries + 1):
            # After a transport reconnect, re-validate identity before scoring.
            self._maybe_revalidate()

            try:
                response = self._client.post(f"{self._base_url}/generate", json=payload)
            except httpx.TransportError:
                if attempt < self._max_retries:
                    self._record_retry()
                    with self._revalidate_lock:
                        self._needs_revalidate = True
                    continue
                raise

            # Chunking signal: 400 with max_token_ids_logprob.
            if response.status_code == 400:
                body = self._safe_json(response)
                limit = body.get("max_token_ids_logprob")
                if (
                    limit is not None
                    and isinstance(limit, int)
                    and len(token_ids) > limit
                    and limit > 0
                ):
                    return self._post_chunked(request, token_ids, limit)
                response.raise_for_status()

            if response.status_code in _RETRYABLE_STATUS:
                if attempt < self._max_retries:
                    self._record_retry()
                    continue
                response.raise_for_status()

            if response.status_code != 200:
                response.raise_for_status()

            return self._parse_token_map(response, token_ids)

        raise RuntimeError("retry loop exhausted without result")

    def _post_chunked(
        self,
        request: ReferenceScoreRequest,
        token_ids: list[int],
        limit: int,
    ) -> dict[int, float]:
        """Split ``token_ids`` into chunks and score each independently."""
        token_map: dict[int, float] = {}
        for start in range(0, len(token_ids), limit):
            chunk = token_ids[start : start + limit]
            chunk_map = self._post_with_retry(request, chunk)
            token_map.update(chunk_map)
        return token_map

    @staticmethod
    def _build_payload(
        request: ReferenceScoreRequest, token_ids: list[int]
    ) -> dict[str, Any]:
        """Build the SGLang /generate scoring payload."""
        return {
            "input_ids": list(request.prefix_ids),
            "sampling_params": {"max_new_tokens": 1, "temperature": 1.0},
            "return_logprob": True,
            "token_ids_logprob": list(token_ids),
            "stream": False,
        }

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any]:
        """Parse JSON body; return ``{}`` on failure."""
        try:
            body = response.json()
            if isinstance(body, dict):
                return body
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    @staticmethod
    def _parse_token_map(
        response: httpx.Response, requested_ids: list[int]
    ) -> dict[int, float]:
        """Parse ``meta_info.token_ids_logprob`` into ``{token_id: logprob}``.

        Each entry is ``[logprob, token_id, ...]`` (SGLang convention). Every
        requested ID must appear exactly once with a finite logprob.
        """
        body = response.json()
        meta_info = body.get("meta_info")
        if not isinstance(meta_info, dict):
            raise ValueError("response missing meta_info")
        raw = meta_info.get("token_ids_logprob")
        if not isinstance(raw, list):
            raise ValueError("response missing meta_info.token_ids_logprob")

        token_map: dict[int, float] = {}
        for entry in raw:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError(f"malformed token_ids_logprob entry: {entry!r}")
            logprob = float(entry[0])
            token_id = int(entry[1])
            if token_id in token_map:
                raise ValueError(f"duplicate token_id {token_id} in response")
            token_map[token_id] = logprob

        missing = [tid for tid in requested_ids if tid not in token_map]
        if missing:
            raise ValueError(f"missing token IDs in response: {sorted(set(missing))}")

        for tid, logprob in token_map.items():
            if not math.isfinite(logprob):
                raise ValueError(f"non-finite logprob {logprob} for token_id {tid}")

        return token_map

    # -- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Idempotent shutdown of the executor and owned client."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False)
        if self._owns_client:
            self._client.close()

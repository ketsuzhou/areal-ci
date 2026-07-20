# SPDX-License-Identifier: Apache-2.0
"""Mocked HTTP contract tests for the frozen SGLang VIMPO reference scorer.

All tests use ``httpx.MockTransport`` - no real network access. The fixtures
``_scripted_scorer`` and ``_identity_scorer`` are complete local helpers that
also back the failure-mode tests (retry, no-retry, timeout, missing token,
non-finite, chunking, identity-change-after-reconnect, bounded workers,
idempotent close).
"""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from customized_areal.tree_search.training.vimpo_reference import (
    ReferenceIdentity,
    ReferenceScoreRequest,
    SGLangVIMPOReferenceScorer,
)

# ---------------------------------------------------------------------------
# Shared identity payload served by every mock /get_model_info handler.
# ---------------------------------------------------------------------------

_IDENTITY_INFO: dict = {
    "model_path": "/models/init",
    "revision": "main",
    "vocab_size": 8,
    "tokenizer_vocab_size": 8,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 2,
    "temperature": 1.0,
    "quantized": False,
}

_REF = ReferenceIdentity("/models/init", "main", 8, 8, 1, 2, 2)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


def _identity_scorer(**overrides) -> SGLangVIMPOReferenceScorer:
    """Scorer whose ``/get_model_info`` returns identity fields with overrides.

    Default fields match ``_REF`` so ``validate_identity(_REF)`` passes; any
    overridden field causes a mismatch detected by ``validate_identity``.
    """
    info = {**_IDENTITY_INFO, **overrides}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=info)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ref")
    return SGLangVIMPOReferenceScorer(
        "http://ref", timeout=2, max_concurrency=2, max_retries=0, client=client
    )


def _scripted_scorer(
    scores: dict[tuple[int, int], tuple[float, list[float]]],
    *,
    max_retries: int = 0,
    max_concurrency: int = 2,
    handler: type[httpx.MockTransport] | None = None,
) -> SGLangVIMPOReferenceScorer:
    """Scorer returning ``-token_id`` logprobs via ``httpx.MockTransport``.

    The ``scores`` dict documents the expected ``(sampled_logp, candidate_logp)``
    per key; the mock derives logprobs as ``-float(token_id)`` so the scripted
    values are naturally produced. ``handler`` may override the default handler
    for failure-mode tests.
    """
    if handler is not None:
        transport = httpx.MockTransport(handler)
    else:

        def _default_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/get_model_info":
                return httpx.Response(200, json=_IDENTITY_INFO)
            payload = json.loads(request.content)
            token_ids = payload["token_ids_logprob"]
            return httpx.Response(
                200,
                json={
                    "meta_info": {
                        "token_ids_logprob": [[-float(t), t, str(t)] for t in token_ids]
                    }
                },
            )

        transport = httpx.MockTransport(_default_handler)

    client = httpx.Client(transport=transport, base_url="http://ref")
    return SGLangVIMPOReferenceScorer(
        "http://ref",
        timeout=2,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        client=client,
    )


# ---------------------------------------------------------------------------
# Brief tests (verbatim)
# ---------------------------------------------------------------------------


def test_actor_selected_candidates_and_sample_dedup_keep_order() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        if request.url.path == "/get_model_info":
            return httpx.Response(
                200,
                json={
                    "model_path": "/models/init",
                    "revision": "main",
                    "vocab_size": 8,
                    "tokenizer_vocab_size": 8,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "pad_token_id": 2,
                    "temperature": 1.0,
                    "quantized": False,
                },
            )
        seen.append(payload)
        token_ids = payload["token_ids_logprob"]
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "token_ids_logprob": [
                        [-float(token), token, str(token)] for token in token_ids
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ref")
    scorer = SGLangVIMPOReferenceScorer(
        "http://ref", timeout=2, max_concurrency=2, max_retries=0, client=client
    )
    scorer.validate_identity(ReferenceIdentity("/models/init", "main", 8, 8, 1, 2, 2))
    result = scorer.score([ReferenceScoreRequest((0, 1), [1, 3], 5, [7, 5, 4])])
    assert seen[0]["token_ids_logprob"] == [7, 5, 4]
    assert result[0].candidate_logp == [-7.0, -5.0, -4.0]
    assert result[0].sampled_logp == -5.0


def test_out_of_order_completion_is_restored_by_key() -> None:
    requests = [
        ReferenceScoreRequest((0, 3), [1, 2, 3], 4, [5]),
        ReferenceScoreRequest((0, 1), [1], 2, [3]),
    ]
    scorer = _scripted_scorer({(0, 1): (-2.0, [-3.0]), (0, 3): (-4.0, [-5.0])})
    assert [score.key for score in scorer.score(requests)] == [(0, 3), (0, 1)]


@pytest.mark.parametrize(
    "field",
    [
        "model_path",
        "revision",
        "vocab_size",
        "tokenizer_vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ],
)
def test_identity_mismatch_fails_before_scoring(field: str) -> None:
    scorer = _identity_scorer(
        **{field: "wrong" if field in {"model_path", "revision"} else 99}
    )
    with pytest.raises(ValueError, match=field):
        scorer.validate_identity(
            ReferenceIdentity("/models/init", "main", 8, 8, 1, 2, 2)
        )


# ---------------------------------------------------------------------------
# Failure-mode tests (also mandated by the brief)
# ---------------------------------------------------------------------------


def test_retry_on_429_then_succeeds() -> None:
    """HTTP 429 is retried; subsequent 200 returns the score."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        payload = json.loads(request.content)
        token_ids = payload["token_ids_logprob"]
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "token_ids_logprob": [[-float(t), t, str(t)] for t in token_ids]
                }
            },
        )

    scorer = _scripted_scorer({}, max_retries=2, handler=handler)
    scorer.validate_identity(_REF)
    result = scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])
    assert result[0].sampled_logp == -2.0
    assert result[0].candidate_logp == [-3.0]
    assert calls["n"] == 2


def test_retry_on_500_then_succeeds() -> None:
    """HTTP 500 is retried; subsequent 200 returns the score."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        payload = json.loads(request.content)
        token_ids = payload["token_ids_logprob"]
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "token_ids_logprob": [[-float(t), t, str(t)] for t in token_ids]
                }
            },
        )

    scorer = _scripted_scorer({}, max_retries=2, handler=handler)
    scorer.validate_identity(_REF)
    result = scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])
    assert result[0].sampled_logp == -2.0
    assert calls["n"] == 2


def test_no_retry_on_400_raises() -> None:
    """HTTP 400 (non-chunking) is not retried; raises immediately."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    scorer = _scripted_scorer({}, max_retries=3, handler=handler)
    scorer.validate_identity(_REF)
    with pytest.raises(httpx.HTTPStatusError):
        scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])
    assert calls["n"] == 1


def test_timeout_exhaustion_raises() -> None:
    """httpx.TimeoutException (a TransportError) is retried up to max_retries."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        calls["n"] += 1
        raise httpx.TimeoutException("timed out")

    scorer = _scripted_scorer({}, max_retries=2, handler=handler)
    scorer.validate_identity(_REF)
    with pytest.raises(httpx.TimeoutException):
        scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])
    # 1 initial + 2 retries = 3 attempts
    assert calls["n"] == 3


def test_missing_token_id_raises() -> None:
    """Response omitting a requested token ID raises ValueError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        # Return only one of the two requested token IDs.
        return httpx.Response(
            200,
            json={"meta_info": {"token_ids_logprob": [[-3.0, 3, "3"]]}},
        )

    scorer = _scripted_scorer({}, max_retries=0, handler=handler)
    scorer.validate_identity(_REF)
    with pytest.raises(ValueError, match="missing"):
        scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])


def test_non_finite_score_raises() -> None:
    """A non-finite logprob in the response raises ValueError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        # Use raw content because json= cannot serialize float('inf').
        # Python's json.loads parses Infinity, which httpx.Response.json uses.
        return httpx.Response(
            200,
            content=b'{"meta_info": {"token_ids_logprob": '
            b'[[Infinity, 3, "3"], [-2.0, 2, "2"]]'
            b"}}",
            headers={"content-type": "application/json"},
        )

    scorer = _scripted_scorer({}, max_retries=0, handler=handler)
    scorer.validate_identity(_REF)
    with pytest.raises(ValueError, match="finite"):
        scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])


def test_candidate_chunking_joins_by_id() -> None:
    """When the service reports a token-ID limit, the adapter chunks and joins."""
    seen_chunks: list[list[int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        payload = json.loads(request.content)
        token_ids = payload["token_ids_logprob"]
        if len(token_ids) > 2:
            # Service reports its max token_ids_logprob capacity.
            return httpx.Response(400, json={"max_token_ids_logprob": 2})
        seen_chunks.append(list(token_ids))
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "token_ids_logprob": [[-float(t), t, str(t)] for t in token_ids]
                }
            },
        )

    scorer = _scripted_scorer({}, max_retries=0, handler=handler)
    scorer.validate_identity(_REF)
    # 4 candidates + 1 sampled (deduped) = 5 token IDs, chunked into sizes <= 2.
    result = scorer.score([ReferenceScoreRequest((0, 0), [1], 5, [7, 6, 5, 4])])
    # All 5 IDs (4 candidates + sampled 5 which is in candidates => deduped to [7,6,5,4])
    # Wait: candidates=[7,6,5,4], sampled=5. 5 is in candidates, so deduped=[7,6,5,4] (4 IDs).
    # Chunked into [7,6], [5,4].
    assert result[0].sampled_logp == -5.0
    assert result[0].candidate_logp == [-7.0, -6.0, -5.0, -4.0]
    # At least 2 chunks were sent.
    assert len(seen_chunks) >= 2
    # No chunk exceeded the limit.
    assert all(len(c) <= 2 for c in seen_chunks)


def test_identity_change_after_reconnect_raises() -> None:
    """After a transport error, identity is re-validated; mismatch raises."""
    state = {"transport_failed": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            if state["transport_failed"]:
                # Endpoint now serves a different model.
                return httpx.Response(
                    200, json={**_IDENTITY_INFO, "model_path": "/models/other"}
                )
            return httpx.Response(200, json=_IDENTITY_INFO)
        # First scoring attempt: transport error.
        if not state["transport_failed"]:
            state["transport_failed"] = True
            raise httpx.ConnectError("connection reset")
        # Retry: return a valid score (but identity re-validation happens first).
        payload = json.loads(request.content)
        token_ids = payload["token_ids_logprob"]
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "token_ids_logprob": [[-float(t), t, str(t)] for t in token_ids]
                }
            },
        )

    scorer = _scripted_scorer({}, max_retries=2, handler=handler)
    scorer.validate_identity(_REF)
    with pytest.raises(ValueError, match="model_path"):
        scorer.score([ReferenceScoreRequest((0, 0), [1], 2, [3])])


def test_bounded_worker_count() -> None:
    """At most ``max_concurrency`` requests run simultaneously."""
    max_in_flight = {"value": 0}
    current_in_flight = {"value": 0}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json=_IDENTITY_INFO)
        with lock:
            current_in_flight["value"] += 1
            if current_in_flight["value"] > max_in_flight["value"]:
                max_in_flight["value"] = current_in_flight["value"]
        # Simulate work so concurrency is observable.
        import time

        time.sleep(0.05)
        with lock:
            current_in_flight["value"] -= 1
        payload = json.loads(request.content)
        token_ids = payload["token_ids_logprob"]
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "token_ids_logprob": [[-float(t), t, str(t)] for t in token_ids]
                }
            },
        )

    scorer = _scripted_scorer({}, max_retries=0, max_concurrency=2, handler=handler)
    scorer.validate_identity(_REF)
    requests = [
        ReferenceScoreRequest((0, i), [i + 1], i + 10, [i + 20]) for i in range(6)
    ]
    scorer.score(requests)
    assert max_in_flight["value"] <= 2


def test_close_is_idempotent() -> None:
    """Calling close() twice does not raise."""
    scorer = _scripted_scorer({})
    scorer.close()
    scorer.close()  # must not raise


def test_temperature_mismatch_fails() -> None:
    """temperature != 1.0 fails identity validation."""
    scorer = _identity_scorer(temperature=0.5)
    with pytest.raises(ValueError, match="temperature"):
        scorer.validate_identity(_REF)


def test_quantized_fails() -> None:
    """quantized == True fails identity validation."""
    scorer = _identity_scorer(quantized=True)
    with pytest.raises(ValueError, match="quantized"):
        scorer.validate_identity(_REF)

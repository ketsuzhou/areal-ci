"""Tests for SGLangBackend.parse_generation_response output_top_logprobs parsing."""

import pytest

from areal.api.io_struct import HttpGenerationResult
from areal.engine.sglang_remote import SGLangBackend


@pytest.fixture
def backend():
    return SGLangBackend()


def _make_response(
    output_token_logprobs: list[tuple[float, int]],
    finish_reason: dict | None = None,
    output_top_logprobs: list | None = None,
    prompt_tokens: int = 5,
    completion_tokens: int = 0,
) -> dict:
    """Build a minimal SGLang response dict for testing."""
    if finish_reason is None:
        finish_reason = {"type": "stop", "message": ""}
    meta_info = {
        "finish_reason": finish_reason,
        "output_token_logprobs": output_token_logprobs,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if output_top_logprobs is not None:
        meta_info["output_top_logprobs"] = output_top_logprobs
    return {"meta_info": meta_info}


def test_parse_top_logprobs_when_present(backend):
    """Top-k logprobs from meta_info should be parsed into output_top_logprobs."""
    response = _make_response(
        output_token_logprobs=[(-0.5, 100), (-0.3, 200), (-0.1, 300)],
        output_top_logprobs=[
            {100: -0.5, 200: -1.2, 300: -2.0},
            {200: -0.3, 150: -0.9, 350: -1.5},
            {300: -0.1, 250: -0.8, 400: -1.1},
        ],
    )

    result = backend.parse_generation_response(response)

    assert isinstance(result, HttpGenerationResult)
    assert result.output_tokens == [100, 200, 300]
    assert result.output_logprobs == [-0.5, -0.3, -0.1]
    assert result.output_top_logprobs is not None
    assert len(result.output_top_logprobs) == 3

    # Position 0
    assert len(result.output_top_logprobs[0]) == 3
    assert (100, -0.5) in result.output_top_logprobs[0]
    assert (200, -1.2) in result.output_top_logprobs[0]
    assert (300, -2.0) in result.output_top_logprobs[0]

    # Position 1
    assert len(result.output_top_logprobs[1]) == 3
    assert (200, -0.3) in result.output_top_logprobs[1]
    assert (150, -0.9) in result.output_top_logprobs[1]
    assert (350, -1.5) in result.output_top_logprobs[1]

    # Position 2
    assert len(result.output_top_logprobs[2]) == 3


def test_parse_top_logprobs_backward_compatible_when_absent(backend):
    """When output_top_logprobs is not in meta_info, result should have None."""
    response = _make_response(
        output_token_logprobs=[(-0.5, 100), (-0.3, 200)],
    )

    result = backend.parse_generation_response(response)

    assert result.output_tokens == [100, 200]
    assert result.output_logprobs == [-0.5, -0.3]
    assert result.output_top_logprobs is None


def test_parse_top_logprobs_with_none_positions(backend):
    """None positions in output_top_logprobs should produce empty lists."""
    response = _make_response(
        output_token_logprobs=[(-0.5, 100), (-0.3, 200), (-0.1, 300)],
        output_top_logprobs=[
            {100: -0.5, 200: -1.2},
            None,
            {300: -0.1, 400: -0.8},
        ],
    )

    result = backend.parse_generation_response(response)

    assert result.output_top_logprobs is not None
    assert len(result.output_top_logprobs) == 3
    assert len(result.output_top_logprobs[0]) == 2
    assert result.output_top_logprobs[1] == []
    assert len(result.output_top_logprobs[2]) == 2


def test_parse_top_logprobs_skips_string_keys(backend):
    """String token keys in top_logprobs should be skipped (no tokenizer to convert)."""
    response = _make_response(
        output_token_logprobs=[(-0.5, 100)],
        output_top_logprobs=[
            {100: -0.5, "hello": -1.2, 200: -2.0},
        ],
    )

    result = backend.parse_generation_response(response)

    assert result.output_top_logprobs is not None
    assert len(result.output_top_logprobs) == 1
    # Only int keys should be kept
    assert len(result.output_top_logprobs[0]) == 2
    assert (100, -0.5) in result.output_top_logprobs[0]
    assert (200, -2.0) in result.output_top_logprobs[0]


def test_parse_abort_response_no_top_logprobs(backend):
    """Abort responses should still have output_top_logprobs as None."""
    response = _make_response(
        output_token_logprobs=[],
        finish_reason={"type": "abort", "message": "Abort before prefill"},
    )

    result = backend.parse_generation_response(response)

    assert result.output_tokens == []
    assert result.output_logprobs == []
    assert result.output_top_logprobs is None


def test_parse_top_logprobs_all_none_positions(backend):
    """All None positions should produce a list of empty lists."""
    response = _make_response(
        output_token_logprobs=[(-0.5, 100), (-0.3, 200)],
        output_top_logprobs=[None, None],
    )

    result = backend.parse_generation_response(response)

    assert result.output_top_logprobs is not None
    assert result.output_top_logprobs == [[], []]


def test_parse_top_logprobs_empty_dict_positions(backend):
    """Empty dict positions should produce empty lists."""
    response = _make_response(
        output_token_logprobs=[(-0.5, 100)],
        output_top_logprobs=[{}],
    )

    result = backend.parse_generation_response(response)

    assert result.output_top_logprobs is not None
    assert result.output_top_logprobs == [[]]

"""Tests for ModelResponse.output_top_logprobs field."""

from areal.api.io_struct import ModelResponse


def test_model_response_default_top_logprobs_is_none():
    """ModelResponse should default output_top_logprobs to None."""
    resp = ModelResponse()
    assert resp.output_top_logprobs is None


def test_model_response_top_logprobs_assignment():
    """ModelResponse should accept output_top_logprobs as list of list of tuples."""
    top_logprobs = [
        [(100, -0.5), (200, -1.2), (300, -2.0)],
        [(150, -0.3), (250, -0.9), (350, -1.5)],
    ]
    resp = ModelResponse(output_top_logprobs=top_logprobs)
    assert resp.output_top_logprobs == top_logprobs
    assert len(resp.output_top_logprobs) == 2


def test_model_response_backward_compatible_without_top_logprobs():
    """Code that doesn't use output_top_logprobs should work unchanged."""
    resp = ModelResponse(
        input_tokens=[1, 2, 3],
        output_tokens=[4, 5, 6],
        output_logprobs=[-0.1, -0.2, -0.3],
    )
    assert resp.input_len == 3
    assert resp.output_len == 3
    assert resp.output_logprobs == [-0.1, -0.2, -0.3]
    assert resp.output_top_logprobs is None

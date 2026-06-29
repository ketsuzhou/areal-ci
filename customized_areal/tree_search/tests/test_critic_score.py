"""Tests for the generative critic score (Phase 3, Task 6).

The critic is the actor in "critic mode" (one shared network): given a filtered
critic observation it generates a critique then ``<score>N</score>`` with N an
integer in [0, 10]. At rollout the integer is parsed (with retry on garbled
output) and clamped; for training the value is the differentiable EXPECTED score
over the 11 digit-token logits: ``V = sum_i softmax(logits)[i] * i / 10``.

torch is optional in this env, so the differentiable path is tested behind
``importorskip`` while the math is also verified with a pure-Python reference.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.critic_score import (
    SCORE_BUCKETS,
    build_critic_score_prompt,
    expected_value_from_logits_py,
    parse_score,
    score_to_value,
)


def test_score_buckets_is_eleven() -> None:
    assert SCORE_BUCKETS == 11  # integers 0..10 inclusive


def test_build_prompt_demands_score_tag_and_range() -> None:
    prompt = build_critic_score_prompt("[A|n0] assistant: did the thing")
    assert "<score>" in prompt
    assert "did the thing" in prompt
    assert "0" in prompt and "10" in prompt


def test_parse_score_extracts_integer() -> None:
    assert parse_score("analysis...\n<score>7</score>") == 7
    assert parse_score("<score> 3 </score>") == 3


def test_parse_score_clamps_out_of_range() -> None:
    assert parse_score("<score>15</score>") == 10
    assert parse_score("<score>-2</score>") == 0


def test_parse_score_raises_on_missing_tag() -> None:
    with pytest.raises(ValueError):
        parse_score("I think it is pretty good, maybe an 8")


def test_score_to_value_maps_to_unit_interval() -> None:
    assert score_to_value(0) == 0.0
    assert score_to_value(5) == 0.5
    assert score_to_value(10) == 1.0


def test_expected_value_uniform_is_half() -> None:
    logits = [0.0] * SCORE_BUCKETS
    assert expected_value_from_logits_py(logits) == pytest.approx(0.5)


def test_expected_value_one_hot_extremes() -> None:
    lo = [0.0] * SCORE_BUCKETS
    lo[0] = 100.0  # all mass on bucket 0
    assert expected_value_from_logits_py(lo) == pytest.approx(0.0, abs=1e-6)
    hi = [0.0] * SCORE_BUCKETS
    hi[10] = 100.0  # all mass on bucket 10
    assert expected_value_from_logits_py(hi) == pytest.approx(1.0, abs=1e-6)


def test_expected_value_two_point_distribution() -> None:
    # Equal mass on buckets 2 and 8 -> expected bucket 5 -> value 0.5.
    logits = [-1e9] * SCORE_BUCKETS
    logits[2] = 0.0
    logits[8] = 0.0
    assert expected_value_from_logits_py(logits) == pytest.approx(0.5, abs=1e-6)


def test_expected_score_value_torch_matches_reference_and_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from customized_areal.tree_search.dag.critic_score import expected_score_value

    logits = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]],
        requires_grad=True,
    )
    value = expected_score_value(logits)
    ref = expected_value_from_logits_py(logits.detach()[0].tolist())
    assert value.shape == (1,)
    assert float(value.item()) == pytest.approx(ref, abs=1e-6)
    # Differentiable: gradient flows back to the logits.
    value.sum().backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)

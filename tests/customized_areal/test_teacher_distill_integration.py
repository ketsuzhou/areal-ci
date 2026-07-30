"""Integration tests for teacher distillation pipeline.

Tests the full flow: TeacherConfig -> TeacherClient -> _compute_token_rewards
-> PositionRewardInfo, with mocked teacher API responses.
"""

from unittest.mock import AsyncMock

import pytest

from customized_areal.tree_search.distilling.distill_types import PositionRewardInfo
from customized_areal.tree_search.distilling.reward_compute import (
    _compute_token_rewards,
)
from customized_areal.tree_search.distilling.teacher_client import (
    TeacherClient,
    TeacherConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_teacher_top_logprobs(
    token_logprob_pairs: list[tuple[int, float]],
) -> list[dict]:
    """Build a single-position top_logprobs entry as returned by vLLM/SGLang.

    The real API returns a list of dicts like:
        [{"token_id": 100, "logprob": -0.5}, ...]
    """
    return [{"token_id": tid, "logprob": lp} for tid, lp in token_logprob_pairs]


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_teacher_distillation_pipeline():
    """Full pipeline: TeacherConfig -> TeacherClient -> _compute_token_rewards -> PositionRewardInfo."""
    # 1. Create TeacherConfig
    config = TeacherConfig(
        teacher_base_url="http://localhost:8001",
        teacher_model_name="test-teacher",
        teacher_top_k=3,
        teacher_max_retries=1,
        teacher_timeout=5.0,
    )
    assert config.teacher_top_k == 3

    # 2. Create TeacherClient and mock get_logprobs_for_candidates
    #    (we mock at this level rather than the HTTP layer to isolate the
    #    reward-computation logic from API parsing details)
    client = TeacherClient(config)
    client.get_logprobs_for_candidates = AsyncMock(  # type: ignore[attr-defined]
        return_value=[
            {100: -0.3, 200: -1.2, 300: -2.5},  # position 0
            {150: -0.6, 250: -1.0, 350: -1.8},  # position 1
            {400: -0.4, 500: -0.9, 600: -1.7},  # position 2
        ]
    )

    # 3. Define student output
    student_output_ids = [100, 150, 400]
    student_input_ids = [1, 2, 3, 4, 5]
    student_top_k_logprobs = [
        [(100, -0.5), (200, -1.0), (300, -2.0)],  # position 0
        [(150, -0.3), (250, -0.8), (350, -1.5)],  # position 1
        [(400, -0.2), (500, -0.7), (600, -1.3)],  # position 2
    ]

    # 4. Compute token rewards
    position_rewards = await _compute_token_rewards(
        student_output_ids=student_output_ids,
        student_input_ids=student_input_ids,
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=client,
        top_k=3,
    )

    # 5. Verify results
    assert len(position_rewards) == 3

    # Position 0: chosen token = 100
    pr0 = position_rewards[0]
    assert pr0.position == 0
    assert pr0.candidate_token_ids == [100, 200, 300]
    assert pr0.chosen_index == 0
    assert pr0.logprobs == [-0.5, -1.0, -2.0]
    # reward = student_logp - teacher_logp
    assert pr0.rewards[0] == pytest.approx(-0.5 - (-0.3), abs=1e-6)  # -0.2
    assert pr0.rewards[1] == pytest.approx(-1.0 - (-1.2), abs=1e-6)  # 0.2
    assert pr0.rewards[2] == pytest.approx(-2.0 - (-2.5), abs=1e-6)  # 0.5

    # Position 1: chosen token = 150
    pr1 = position_rewards[1]
    assert pr1.position == 1
    assert pr1.chosen_index == 0
    assert pr1.rewards[0] == pytest.approx(-0.3 - (-0.6), abs=1e-6)  # 0.3

    # Position 2: chosen token = 400
    pr2 = position_rewards[2]
    assert pr2.position == 2
    assert pr2.chosen_index == 0
    assert pr2.rewards[0] == pytest.approx(-0.2 - (-0.4), abs=1e-6)  # 0.2


@pytest.mark.asyncio
async def test_pipeline_with_missing_teacher_candidates():
    """Pipeline handles student candidates absent from teacher's top-k."""
    config = TeacherConfig(teacher_top_k=3, teacher_missing_logprob=-23.0)
    client = TeacherClient(config)

    # Teacher only has 2 of 3 candidates; 999 gets missing_logprob
    client.get_logprobs_for_candidates = AsyncMock(  # type: ignore[attr-defined]
        return_value=[
            {100: -0.3, 200: -1.2, 999: -23.0},
        ]
    )

    student_top_k_logprobs = [
        [(100, -0.5), (200, -1.0), (999, -3.0)],
    ]

    position_rewards = await _compute_token_rewards(
        student_output_ids=[100],
        student_input_ids=[1, 2, 3],
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=client,
        top_k=3,
    )

    assert len(position_rewards) == 1
    pr = position_rewards[0]
    # Token 999: student_logp=-3.0, teacher_logp=-23.0, reward=20.0
    assert pr.rewards[2] == pytest.approx(-3.0 - (-23.0), abs=1e-6)  # 20.0


@pytest.mark.asyncio
async def test_pipeline_empty_output():
    """Empty student output returns empty rewards list."""
    config = TeacherConfig(teacher_top_k=3)
    client = TeacherClient(config)

    result = await _compute_token_rewards(
        student_output_ids=[],
        student_input_ids=[1, 2, 3],
        student_top_k_logprobs=[],
        teacher_client=client,
        top_k=3,
    )
    assert result == []


def test_position_reward_info_validation():
    """PositionRewardInfo validates field consistency."""
    # Valid PositionRewardInfo
    pr = PositionRewardInfo(
        position=0,
        candidates=["a", "b"],
        candidate_token_ids=[100, 200],
        logprobs=[-0.5, -1.0],
        rewards=[0.2, -0.3],
        chosen_index=0,
    )
    assert pr.position == 0
    assert pr.chosen_token == "a"
    assert pr.chosen_reward == 0.2
    assert pr.chosen_logprob == -0.5

    # Mismatched lengths should raise ValueError
    with pytest.raises(ValueError):
        PositionRewardInfo(
            position=0,
            candidates=["a", "b"],
            candidate_token_ids=[100],
            logprobs=[-0.5],
            rewards=[0.2],
            chosen_index=0,
        )

    # chosen_index out of range should raise ValueError
    with pytest.raises(ValueError):
        PositionRewardInfo(
            position=0,
            candidates=["a", "b"],
            candidate_token_ids=[100, 200],
            logprobs=[-0.5, -1.0],
            rewards=[0.2, -0.3],
            chosen_index=5,
        )


@pytest.mark.asyncio
async def test_pipeline_reward_sign_consistency():
    """Verify reward sign semantics: student_logp - teacher_logp.

    Positive reward => student over-estimates the token relative to teacher.
    Negative reward => student under-estimates.
    """
    config = TeacherConfig(teacher_top_k=2)
    client = TeacherClient(config)

    # Position 0: student logp > teacher logp => positive reward
    # Position 1: student logp < teacher logp => negative reward
    client.get_logprobs_for_candidates = AsyncMock(  # type: ignore[attr-defined]
        return_value=[
            {100: -2.0},  # teacher: very unlikely => student over-estimates
            {200: -0.1},  # teacher: very likely => student under-estimates
        ]
    )

    student_top_k_logprobs = [
        [(100, -0.5)],  # student thinks P=exp(-0.5)
        [(200, -1.5)],  # student thinks P=exp(-1.5)
    ]

    position_rewards = await _compute_token_rewards(
        student_output_ids=[100, 200],
        student_input_ids=[1, 2],
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=client,
        top_k=2,
    )

    assert len(position_rewards) == 2
    # Position 0: -0.5 - (-2.0) = 1.5 (student over-estimates)
    assert position_rewards[0].rewards[0] == pytest.approx(1.5, abs=1e-6)
    # Position 1: -1.5 - (-0.1) = -1.4 (student under-estimates)
    assert position_rewards[1].rewards[0] == pytest.approx(-1.4, abs=1e-6)

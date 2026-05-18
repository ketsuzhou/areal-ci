"""Tests for _compute_token_rewards in reward_compute.py."""

import math
from unittest.mock import AsyncMock

import pytest

from customized_areal.tree_search.core.reward_compute import (
    _compute_token_rewards,
)
from customized_areal.tree_search.core.teacher_client import (
    TeacherClient,
    TeacherConfig,
)


def _make_mock_teacher_client(
    teacher_logprobs: list[dict[int, float]],
    missing_logprob: float = math.log(1e-10),
) -> TeacherClient:
    """Create a TeacherClient with a mocked get_logprobs_for_candidates.

    Parameters
    ----------
    teacher_logprobs : list[dict[int, float]]
        For each position, a mapping from token_id to teacher logprob.
    missing_logprob : float
        Value to use for missing teacher logprobs.

    Returns
    -------
    TeacherClient
        A TeacherClient whose ``get_logprobs_for_candidates`` returns
        *teacher_logprobs*.
    """
    config = TeacherConfig(teacher_missing_logprob=missing_logprob)
    client = TeacherClient(config)
    client.get_logprobs_for_candidates = AsyncMock(return_value=teacher_logprobs)
    return client


# ---------------------------------------------------------------------------
# Test: Basic reward computation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_reward_computation():
    """Test reward = student_logp - teacher_logp with known values."""
    teacher_logprobs = [
        {100: -0.5, 200: -1.0, 300: -2.0},  # position 0
        {400: -0.2, 500: -0.6},  # position 1
    ]
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[100, 400],
        student_input_ids=[10, 20, 30],
        student_top_k_logprobs=[
            [(100, -0.5), (200, -1.0), (300, -2.0)],  # position 0
            [(400, -0.2), (500, -0.6)],  # position 1
        ],
        teacher_client=client,
        top_k=10,
    )

    assert len(result) == 2

    # Position 0
    pos0 = result[0]
    assert pos0.position == 0
    assert pos0.candidate_token_ids == [100, 200, 300]
    assert pos0.candidates == ["100", "200", "300"]
    assert pos0.logprobs == pytest.approx([-0.5, -1.0, -2.0])
    assert pos0.teacher_logprobs == pytest.approx([-0.5, -1.0, -2.0])
    # reward = student_logp - teacher_logp
    assert pos0.rewards == pytest.approx([0.0, 0.0, 0.0])
    assert pos0.chosen_index == 0  # token 100 at index 0

    # Position 1
    pos1 = result[1]
    assert pos1.position == 1
    assert pos1.candidate_token_ids == [400, 500]
    assert pos1.candidates == ["400", "500"]
    assert pos1.logprobs == pytest.approx([-0.2, -0.6])
    assert pos1.teacher_logprobs == pytest.approx([-0.2, -0.6])
    assert pos1.rewards == pytest.approx([0.0, 0.0])
    assert pos1.chosen_index == 0  # token 400 at index 0


# ---------------------------------------------------------------------------
# Test: Missing teacher logprobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_teacher_logprobs():
    """Test that missing teacher logprobs use the configured default."""
    missing_lp = -50.0
    # Teacher only knows token 100 at position 0; tokens 200 and 300 are
    # missing from the teacher response.
    teacher_logprobs = [
        {100: -0.5},  # only token 100 present
    ]
    client = _make_mock_teacher_client(teacher_logprobs, missing_logprob=missing_lp)

    result = await _compute_token_rewards(
        student_output_ids=[100],
        student_input_ids=[10],
        student_top_k_logprobs=[
            [(100, -0.3), (200, -1.5), (300, -2.8)],
        ],
        teacher_client=client,
        top_k=10,
    )

    assert len(result) == 1
    pos0 = result[0]
    # Token 100: student -0.3, teacher -0.5 => reward 0.2
    assert pos0.teacher_logprobs == pytest.approx([-0.5, missing_lp, missing_lp])
    assert pos0.rewards[0] == pytest.approx(-0.3 - (-0.5))
    # Token 200: student -1.5, teacher missing_lp (-50.0) => reward 48.5
    assert pos0.rewards[1] == pytest.approx(-1.5 - missing_lp)
    # Token 300: student -2.8, teacher missing_lp (-50.0) => reward 47.2
    assert pos0.rewards[2] == pytest.approx(-2.8 - missing_lp)


# ---------------------------------------------------------------------------
# Test: Empty output returns empty list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_output_ids():
    """Test that empty student_output_ids returns empty list."""
    teacher_logprobs: list[dict[int, float]] = []
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[],
        student_input_ids=[10, 20],
        student_top_k_logprobs=[],
        teacher_client=client,
    )

    assert result == []


@pytest.mark.asyncio
async def test_empty_top_k_logprobs():
    """Test that empty student_top_k_logprobs returns empty list."""
    teacher_logprobs: list[dict[int, float]] = []
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[100],
        student_input_ids=[10],
        student_top_k_logprobs=[],
        teacher_client=client,
    )

    assert result == []


# ---------------------------------------------------------------------------
# Test: Chosen index is correctly identified
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chosen_index_first_position():
    """Test chosen_index when the generated token is the first candidate."""
    teacher_logprobs = [{10: -0.1, 20: -0.5, 30: -1.0}]
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[10],  # token 10 is at index 0
        student_input_ids=[1, 2],
        student_top_k_logprobs=[
            [(10, -0.1), (20, -0.5), (30, -1.0)],
        ],
        teacher_client=client,
    )

    assert result[0].chosen_index == 0


@pytest.mark.asyncio
async def test_chosen_index_middle_position():
    """Test chosen_index when the generated token is a middle candidate."""
    teacher_logprobs = [{10: -0.1, 20: -0.5, 30: -1.0}]
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[20],  # token 20 is at index 1
        student_input_ids=[1, 2],
        student_top_k_logprobs=[
            [(10, -0.1), (20, -0.5), (30, -1.0)],
        ],
        teacher_client=client,
    )

    assert result[0].candidate_token_ids == [20, 10, 30]
    assert result[0].chosen_index == 0


@pytest.mark.asyncio
async def test_chosen_index_last_position():
    """Test chosen_index when the generated token is the last candidate."""
    teacher_logprobs = [{10: -0.1, 20: -0.5, 30: -1.0}]
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[30],  # token 30 is at index 2
        student_input_ids=[1, 2],
        student_top_k_logprobs=[
            [(10, -0.1), (20, -0.5), (30, -1.0)],
        ],
        teacher_client=client,
    )

    assert result[0].candidate_token_ids == [30, 10, 20]
    assert result[0].chosen_index == 0


# ---------------------------------------------------------------------------
# Test: reward = student_logp - teacher_logp is exact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reward_exact_computation():
    """Test that reward is exactly student_logp minus teacher_logp."""
    teacher_logprobs = [
        {10: -0.3, 20: -0.8, 30: -1.5},
        {40: -0.1, 50: -2.0},
    ]
    client = _make_mock_teacher_client(teacher_logprobs)

    student_top_k = [
        [(10, -0.5), (20, -1.0), (30, -2.5)],
        [(40, -0.4), (50, -1.8)],
    ]

    result = await _compute_token_rewards(
        student_output_ids=[20, 50],
        student_input_ids=[1],
        student_top_k_logprobs=student_top_k,
        teacher_client=client,
        top_k=10,
    )

    # Position 0
    assert result[0].rewards[0] == pytest.approx(-0.5 - (-0.3))  # -0.2
    assert result[0].rewards[1] == pytest.approx(-1.0 - (-0.8))  # -0.2
    assert result[0].rewards[2] == pytest.approx(-2.5 - (-1.5))  # -1.0

    # Position 1. The chosen token is moved to index 0.
    assert result[1].candidate_token_ids == [50, 40]
    assert result[1].rewards[0] == pytest.approx(-1.8 - (-2.0))  # 0.2
    assert result[1].rewards[1] == pytest.approx(-0.4 - (-0.1))  # -0.3


# ---------------------------------------------------------------------------
# Test: top_k truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_k_truncation():
    """Test that only the top-k student candidates are used."""
    # Student provides 5 candidates but top_k=2 truncates to first 2.
    teacher_logprobs = [{10: -0.1, 20: -0.2}]
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[10],
        student_input_ids=[1],
        student_top_k_logprobs=[
            [(10, -0.5), (20, -1.0), (30, -1.5), (40, -2.0), (50, -2.5)],
        ],
        teacher_client=client,
        top_k=2,
    )

    assert len(result) == 1
    pos0 = result[0]
    assert pos0.candidate_token_ids == [10, 20]
    assert len(pos0.rewards) == 2
    assert len(pos0.logprobs) == 2


# ---------------------------------------------------------------------------
# Test: Teacher client is called with correct arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teacher_client_called_correctly():
    """Test that get_logprobs_for_candidates receives the right arguments."""
    teacher_logprobs = [{100: -0.5}]
    client = _make_mock_teacher_client(teacher_logprobs)

    await _compute_token_rewards(
        student_output_ids=[100],
        student_input_ids=[10, 20, 30],
        student_top_k_logprobs=[[(100, -0.3), (200, -1.0)]],
        teacher_client=client,
        top_k=5,
    )

    client.get_logprobs_for_candidates.assert_called_once_with(
        input_ids=[10, 20, 30],
        output_ids=[100],
        candidate_token_ids=[[100, 200]],
    )


# ---------------------------------------------------------------------------
# Test: Multiple positions with varying candidate counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_positions():
    """Test reward computation across multiple positions."""
    teacher_logprobs = [
        {10: -0.1, 20: -0.3},
        {30: -0.5},
        {40: -0.2, 50: -0.4, 60: -0.6},
    ]
    client = _make_mock_teacher_client(teacher_logprobs)

    result = await _compute_token_rewards(
        student_output_ids=[10, 30, 50],
        student_input_ids=[1],
        student_top_k_logprobs=[
            [(10, -0.2), (20, -0.4)],
            [(30, -0.5)],
            [(40, -0.1), (50, -0.3), (60, -0.7)],
        ],
        teacher_client=client,
        top_k=10,
    )

    assert len(result) == 3

    # Position 0
    assert result[0].position == 0
    assert result[0].candidate_token_ids == [10, 20]
    assert result[0].chosen_index == 0
    assert result[0].rewards == pytest.approx([-0.2 - (-0.1), -0.4 - (-0.3)])

    # Position 1
    assert result[1].position == 1
    assert result[1].candidate_token_ids == [30]
    assert result[1].chosen_index == 0
    assert result[1].rewards == pytest.approx([-0.5 - (-0.5)])

    # Position 2
    assert result[2].position == 2
    assert result[2].candidate_token_ids == [50, 40, 60]
    assert result[2].chosen_index == 0
    assert result[2].rewards == pytest.approx(
        [-0.3 - (-0.4), -0.1 - (-0.2), -0.7 - (-0.6)]
    )

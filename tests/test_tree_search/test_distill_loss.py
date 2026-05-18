import torch

from customized_areal.tree_search.distill_types import PositionRewardInfo
from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss


def test_compute_teacher_kl_loss_1d_uses_prompt_len_absolute_positions():
    logprobs = torch.tensor([-9.0, -1.0, -2.0, -3.0], requires_grad=True)
    loss_mask = torch.tensor([0, 1, 1, 1], dtype=torch.bool)
    position_rewards = [
        PositionRewardInfo(
            position=0,
            teacher_logprobs=[-0.25, -9.0],
            sample_index=0,
        ),
        PositionRewardInfo(
            position=2,
            teacher_logprobs=[-1.5],
            sample_index=0,
        ),
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[1],
    )

    expected = torch.tensor(((-1.0 + 0.25) + (-3.0 + 1.5)) / 2)
    torch.testing.assert_close(loss, expected)


def test_compute_teacher_kl_loss_2d_uses_candidate_columns():
    logprobs = torch.tensor(
        [
            [-9.0, -9.0, -9.0],
            [-1.0, -1.5, -2.0],
            [-2.5, -3.0, -3.5],
        ],
        requires_grad=True,
    )
    loss_mask = torch.tensor([0, 1, 1], dtype=torch.bool)
    position_rewards = [
        PositionRewardInfo(
            position=1,
            teacher_logprobs=[-2.0, -2.25],
            sample_index=0,
        ),
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[1],
    )

    expected = torch.tensor(((-2.5 + 2.0) + (-3.0 + 2.25)) / 2)
    torch.testing.assert_close(loss, expected)


def test_compute_teacher_kl_loss_empty_position_rewards_returns_device_scalar_zero():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logprobs = torch.tensor([-1.0, -2.0], device=device, requires_grad=True)
    loss_mask = torch.tensor([1, 1], dtype=torch.bool, device=device)

    loss = _compute_teacher_kl_loss(
        position_rewards=[],
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[0],
    )

    assert loss.shape == torch.Size([])
    assert loss.device == logprobs.device == loss_mask.device
    torch.testing.assert_close(loss, torch.tensor(0.0, device=logprobs.device))


def test_compute_teacher_kl_loss_ignores_missing_and_invalid_entries():
    logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss_mask = torch.tensor([1, 1], dtype=torch.bool)
    position_rewards = [
        PositionRewardInfo(position=0, teacher_logprobs=None, sample_index=0),
        PositionRewardInfo(position=10, teacher_logprobs=[-0.5], sample_index=0),
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[0],
    )

    assert loss.device == logprobs.device
    torch.testing.assert_close(loss, torch.tensor(0.0, device=logprobs.device))

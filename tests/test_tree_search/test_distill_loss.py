import torch

from customized_areal.tree_search.distill_types import PositionRewardInfo
from customized_areal.tree_search.training.actor import _distribute_position_rewards
from customized_areal.tree_search.training.loss import (
    _compute_teacher_kl_loss,
    _select_chosen_logprobs,
)


def test_select_chosen_logprobs_supports_distill_shapes():
    seq_logprobs = torch.tensor([-1.0, -2.0])
    seq_mask = torch.tensor([1, 1], dtype=torch.bool)
    torch.testing.assert_close(
        _select_chosen_logprobs(seq_logprobs, seq_mask),
        seq_logprobs,
    )

    seq_candidate_logprobs = torch.tensor([[-1.0, -1.5], [-2.0, -2.5]])
    torch.testing.assert_close(
        _select_chosen_logprobs(seq_candidate_logprobs, seq_mask),
        torch.tensor([-1.0, -2.0]),
    )

    batch_logprobs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    batch_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    torch.testing.assert_close(
        _select_chosen_logprobs(batch_logprobs, batch_mask),
        batch_logprobs,
    )

    batch_candidate_logprobs = torch.tensor(
        [[[-1.0, -1.5], [-2.0, -2.5]], [[-3.0, -3.5], [-4.0, -4.5]]]
    )
    torch.testing.assert_close(
        _select_chosen_logprobs(batch_candidate_logprobs, batch_mask),
        torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
    )


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


def test_compute_teacher_kl_loss_1d_uses_chosen_index():
    logprobs = torch.tensor([-1.0], requires_grad=True)
    loss_mask = torch.tensor([1], dtype=torch.bool)
    position_rewards = [
        PositionRewardInfo(
            position=0,
            teacher_logprobs=[-9.0, -0.25],
            chosen_index=1,
            sample_index=0,
        )
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[0],
    )

    torch.testing.assert_close(loss, torch.tensor(-0.75))


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


def test_compute_teacher_kl_loss_batched_single_candidate_uses_sample_index():
    logprobs = torch.tensor(
        [
            [-9.0, -1.0, -2.0],
            [-8.0, -3.0, -4.0],
        ],
        requires_grad=True,
    )
    loss_mask = torch.tensor(
        [
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=torch.bool,
    )
    position_rewards = [
        PositionRewardInfo(
            position=1,
            teacher_logprobs=[-0.5],
            sample_index=1,
        )
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[1, 1],
    )

    torch.testing.assert_close(loss, torch.tensor(-3.5))


def test_compute_teacher_kl_loss_batched_multi_candidate_uses_sample_index():
    logprobs = torch.tensor(
        [
            [
                [-9.0, -9.0],
                [-1.0, -1.5],
            ],
            [
                [-8.0, -8.5],
                [-3.0, -3.5],
            ],
        ],
        requires_grad=True,
    )
    loss_mask = torch.tensor(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=torch.bool,
    )
    position_rewards = [
        PositionRewardInfo(
            position=0,
            teacher_logprobs=[-0.5, -1.0],
            sample_index=1,
        )
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[1, 1],
    )

    expected = torch.tensor(((-3.0 + 0.5) + (-3.5 + 1.0)) / 2)
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


class _FakeMbInputs:
    def __init__(self):
        self.forward_indices = torch.tensor([2, 0, 1])
        self.mbs = [
            {"attention_mask": torch.ones(2, 4, dtype=torch.bool)},
            {"attention_mask": torch.ones(1, 4, dtype=torch.bool)},
        ]


def test_distribute_position_rewards_rebases_sample_index_to_minibatch_local():
    mb_inputs = _FakeMbInputs()
    rewards = [
        PositionRewardInfo(position=0, teacher_logprobs=[-0.5], sample_index=1),
        PositionRewardInfo(position=1, teacher_logprobs=[-0.6], sample_index=2),
    ]

    _distribute_position_rewards(mb_inputs, rewards)

    assert "position_rewards" in mb_inputs.mbs[0]
    assert "position_rewards" in mb_inputs.mbs[1]
    assert mb_inputs.mbs[0]["position_rewards"][0].sample_index == 0
    assert mb_inputs.mbs[1]["position_rewards"][0].sample_index == 0
    assert rewards[0].sample_index == 1
    assert rewards[1].sample_index == 2

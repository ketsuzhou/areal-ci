import pytest
import torch

from customized_areal.tree_search.core.advantage import (
    VIMPOAdvantageComputer,
    candidate_forward_kl,
    masked_distributed_whiten,
    masked_episode_reverse_lambda,
)


def test_candidate_kl_is_unrenormalized_and_exact_at_full_vocab() -> None:
    policy = torch.log(torch.tensor([[[0.50, 0.30, 0.20]]]))
    reference = torch.log(torch.tensor([[[0.25, 0.25, 0.50]]]))
    mask = torch.tensor([[True]])
    kl, mass = candidate_forward_kl(policy, reference, mask)
    expected = (policy.exp() * (policy - reference)).sum(-1)
    torch.testing.assert_close(kl, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(mass, torch.ones_like(mass), rtol=0, atol=1e-6)


def test_candidate_kl_keeps_missing_probability_mass_outside_topk() -> None:
    policy = torch.log(torch.tensor([[[0.50, 0.30]]]))
    reference = torch.log(torch.tensor([[[0.25, 0.25]]]))
    kl, mass = candidate_forward_kl(policy, reference, torch.tensor([[True]]))
    torch.testing.assert_close(mass, torch.tensor([[0.80]]), rtol=0, atol=1e-6)
    torch.testing.assert_close(
        kl,
        torch.tensor(
            [
                [
                    0.50 * torch.log(torch.tensor(2.0))
                    + 0.30 * torch.log(torch.tensor(1.2))
                ]
            ]
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reverse_lambda_crosses_turns_and_resets_between_episodes() -> None:
    td = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    mask = torch.tensor(
        [[True, True, False], [True, False, False], [True, False, False]]
    )
    episode = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    turn = torch.tensor([[1, 1, 1], [2, 2, 2], [1, 1, 1]])
    actual = masked_episode_reverse_lambda(td, mask, episode, turn, gamma=1.0, lam=0.5)
    expected = torch.tensor([[2.75, 3.50, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_whitening_one_token_returns_zero_and_detaches() -> None:
    values = torch.tensor([[4.0, 0.0]], requires_grad=True)
    actual = masked_distributed_whiten(
        values, torch.tensor([[True, False]]), group=None
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    assert actual.requires_grad is False


def test_vimpo_computer_detaches_kl_reference_and_advantage() -> None:
    policy = torch.log(torch.tensor([[[0.6, 0.3]]], requires_grad=True))
    reference = torch.log(torch.tensor([[[0.4, 0.2]]], requires_grad=True))
    batch = {
        "vimpo_predict_mask": torch.tensor([[True]]),
        "vimpo_sample_logp": torch.tensor([[-0.2]], requires_grad=True),
        "vimpo_ref_sample_logp": torch.tensor([[-0.4]], requires_grad=True),
        "vimpo_candidate_logp": policy,
        "vimpo_ref_candidate_logp": reference,
        "vimpo_episode_index": torch.tensor([[0]]),
        "vimpo_turn_index": torch.tensor([[1]]),
    }
    out = VIMPOAdvantageComputer(beta=0.5, gamma=1.0, lam=1.0, whiten=False).compute(
        batch
    )
    assert out["vimpo_candidate_kl"].requires_grad is False
    assert out["advantages"].requires_grad is False


def test_vimpo_computer_rejects_duplicate_candidate_ids() -> None:
    policy = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]], requires_grad=True))
    reference = torch.log(torch.tensor([[[0.4, 0.2, 0.4]]], requires_grad=True))
    batch = {
        "vimpo_predict_mask": torch.tensor([[True]]),
        "vimpo_sample_logp": torch.tensor([[-0.2]], requires_grad=True),
        "vimpo_ref_sample_logp": torch.tensor([[-0.4]], requires_grad=True),
        "vimpo_candidate_logp": policy,
        "vimpo_ref_candidate_logp": reference,
        "vimpo_episode_index": torch.tensor([[0]]),
        "vimpo_turn_index": torch.tensor([[1]]),
        "vimpo_candidate_ids": torch.tensor([[[1, 2, 2]]]),
    }
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        VIMPOAdvantageComputer(beta=0.5, gamma=1.0, lam=1.0, whiten=False).compute(
            batch
        )

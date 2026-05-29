import torch

from customized_areal.tree_search.training.loss import (
    _compute_subset_kl,
    _compute_teacher_kl_loss,
)


def test_compute_subset_kl_reverse_matches_weighted_reverse_kl():
    student = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-1.5, -3.0]])

    loss = _compute_subset_kl(student, teacher, "reverse_kl")

    student_norm = torch.log_softmax(student, dim=-1)
    teacher_norm = torch.log_softmax(teacher, dim=-1)
    expected = (student_norm.exp() * (student_norm - teacher_norm)).sum(dim=-1)
    torch.testing.assert_close(loss, expected)


def test_compute_subset_kl_forward_matches_weighted_forward_kl():
    student = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-1.5, -3.0]])

    loss = _compute_subset_kl(student, teacher, "forward_kl")

    student_norm = torch.log_softmax(student, dim=-1)
    teacher_norm = torch.log_softmax(teacher, dim=-1)
    expected = (teacher_norm.exp() * (teacher_norm - student_norm)).sum(dim=-1)
    torch.testing.assert_close(loss, expected)


def test_compute_teacher_kl_loss_uses_reverse_kl_by_default():
    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teacher_logprobs = torch.tensor([[[-1.5, -3.0]]])
    loss_mask = torch.tensor([[1]], dtype=torch.bool)

    loss = _compute_teacher_kl_loss(
        teacher_logprobs=teacher_logprobs,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[0],
    )

    student_norm = torch.log_softmax(logprobs, dim=-1)
    teacher_norm = torch.log_softmax(teacher_logprobs.squeeze(0), dim=-1)
    expected = (student_norm.exp() * (student_norm - teacher_norm)).sum(dim=-1).mean()
    torch.testing.assert_close(loss, expected)


def test_compute_teacher_kl_loss_supports_forward_kl():
    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teacher_logprobs = torch.tensor([[[-1.5, -3.0]]])
    loss_mask = torch.tensor([[1]], dtype=torch.bool)

    loss = _compute_teacher_kl_loss(
        teacher_logprobs=teacher_logprobs,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[0],
        distill_kl_mode="forward_kl",
    )

    student_norm = torch.log_softmax(logprobs, dim=-1)
    teacher_norm = torch.log_softmax(teacher_logprobs.squeeze(0), dim=-1)
    expected = (teacher_norm.exp() * (teacher_norm - student_norm)).sum(dim=-1).mean()
    torch.testing.assert_close(loss, expected)

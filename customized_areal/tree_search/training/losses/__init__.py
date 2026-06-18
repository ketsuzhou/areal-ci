"""Loss helpers for tree-search training."""

from .combined import grpo_distill_loss_fn
from .critic import (
    build_critic_training_batch,
    combined_actor_critic_loss,
    critic_softreg_loss_fn,
    critic_softreg_loss_from_logprobs,
    expected_value_from_label_logits,
    expected_value_from_label_logprobs,
)
from .distill import (
    _align_teacher_chosen_logprobs,
    _compute_distill_reweighted_advantages,
    _compute_position_reward_teacher_kl_loss,
    _compute_subset_kl,
    _compute_teacher_kl_loss,
    _select_chosen_logprobs,
)
from .grpo import (
    _compute_grpo_loss,
    _compute_position_level_grpo_loss,
    _resolve_proximal_logp,
)

__all__ = [
    "grpo_distill_loss_fn",
    "build_critic_training_batch",
    "combined_actor_critic_loss",
    "critic_softreg_loss_fn",
    "critic_softreg_loss_from_logprobs",
    "expected_value_from_label_logits",
    "expected_value_from_label_logprobs",
    "_align_teacher_chosen_logprobs",
    "_compute_distill_reweighted_advantages",
    "_compute_grpo_loss",
    "_compute_position_level_grpo_loss",
    "_compute_position_reward_teacher_kl_loss",
    "_compute_subset_kl",
    "_compute_teacher_kl_loss",
    "_resolve_proximal_logp",
    "_select_chosen_logprobs",
]

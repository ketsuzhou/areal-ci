"""Compatibility exports for tree-search GRPO/distillation losses.

The implementation is split across ``customized_areal.tree_search.training.losses``.
This module keeps the historical import path stable for runtime code and tests.
"""

from __future__ import annotations

from areal.utils.logging import getLogger

from .losses import grpo as _grpo
from .losses.combined import grpo_distill_loss_fn
from .losses.distill import (
    _align_teacher_chosen_logprobs,
    _compute_distill_reweighted_advantages,
    _compute_position_reward_teacher_kl_loss,
    _compute_subset_kl,
    _compute_teacher_kl_loss,
    _select_chosen_logprobs,
)
from .losses.grpo import _compute_grpo_loss, _resolve_proximal_logp

logger = getLogger("DistillLoss")


def _compute_position_level_grpo_loss(*args, **kwargs):
    """Compatibility wrapper that preserves patching of ``loss.logger``."""
    _grpo.logger = logger
    return _grpo._compute_position_level_grpo_loss(*args, **kwargs)


__all__ = [
    "grpo_distill_loss_fn",
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

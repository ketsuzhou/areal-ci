"""Backward-compatible imports for teacher reward computation helpers."""

from customized_areal.tree_search.distilling.reward_compute import (
    _compute_token_rewards,
)

__all__ = ["_compute_token_rewards"]

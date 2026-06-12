"""Clip-cov PPO loss module.

Provides covariance-aware PPO clipping as a monkey-patch for AReaL's PPOActor.

Usage:
    from customized_areal.clip_cov import ClipCovConfig, patch_ppo_actor_to_use_clip_cov_loss

    config = ClipCovConfig(clip_ratio=0.0002, clip_cov_lb=1.0, clip_cov_ub=5.0)
    patch_ppo_actor_to_use_clip_cov_loss(config)
"""

from .config import ClipCovConfig
from .patch import patch_ppo_actor_to_use_clip_cov_loss, unpatch_ppo_actor_clip_cov_loss

__all__ = [
    "ClipCovConfig",
    "patch_ppo_actor_to_use_clip_cov_loss",
    "unpatch_ppo_actor_clip_cov_loss",
]

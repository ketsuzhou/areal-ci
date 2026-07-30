"""Clip-cov PPO loss configuration."""

from dataclasses import dataclass


@dataclass
class ClipCovConfig:
    """Configuration for covariance-aware PPO clipping.

    Attributes:
        clip_ratio: Fraction of valid tokens to zero via covariance clipping.
        clip_cov_lb: Lower bound of covariance range for candidate selection.
        clip_cov_ub: Upper bound of covariance range for candidate selection.
    """

    clip_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0

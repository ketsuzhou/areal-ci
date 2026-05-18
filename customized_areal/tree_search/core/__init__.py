"""Core components for on-policy distillation.

This module contains:
- config.py: OnPolicyDistillConfig
- agent.py: OnPolicyDistillAgent and reward functions
- teacher_client.py: TeacherConfig and TeacherClient
- reward_compute.py: _compute_token_rewards for teacher/student logprob comparison
"""

from .reward_compute import _compute_token_rewards
from .teacher_client import TeacherClient, TeacherConfig
from .teacher_provider import (
    EngineTeacherProvider,
    ExternalTeacherProvider,
    TeacherProvider,
)

__all__ = [
    "EngineTeacherProvider",
    "ExternalTeacherProvider",
    "TeacherClient",
    "TeacherConfig",
    "TeacherProvider",
    "_compute_token_rewards",
]

"""Distilling components for on-policy distillation.

This module contains:
- config.py: OnPolicyDistillConfig
- agent.py: OnPolicyDistillAgent and reward functions
- teacher_client.py: TeacherConfig and TeacherClient
- diagnose_provider.py: DiagnoseProvider protocol and implementations
- reward_compute.py: _compute_token_rewards for teacher/student logprob comparison
"""

from .reward_compute import _compute_token_rewards
from .teacher_client import TeacherClient, TeacherConfig
from .diagnose_provider import (
    EngineDiagnoseProvider,
    ExternalDiagnoseProvider,
    DiagnoseProvider,
)

__all__ = [
    "EngineDiagnoseProvider",
    "ExternalDiagnoseProvider",
    "TeacherClient",
    "TeacherConfig",
    "DiagnoseProvider",
    "_compute_token_rewards",
]

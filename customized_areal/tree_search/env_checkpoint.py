"""Entropy-gated checkpoint decision helper for AReaL env-dispatch rollouts.

AReaL computes entropy from model logprobs at decision points and calls
:func:`should_create_entropy_checkpoint` before hitting the Multica checkpoint
API. When logprobs are unavailable (entropy is ``None``) or no threshold is
configured, the helper returns ``False`` so the rollout continues without
failing or creating a spurious checkpoint.
"""

from __future__ import annotations


def should_create_entropy_checkpoint(
    entropy: float | None, threshold: float | None
) -> bool:
    """Return ``True`` iff an entropy-gated checkpoint should be created.

    Skips when logprobs are unavailable (``entropy is None``) or no threshold
    is configured (``threshold is None``). When both are present, the checkpoint
    fires at or above the threshold (``>=``).
    """
    return entropy is not None and threshold is not None and entropy >= threshold

"""Generative critic score -- the actor in "critic mode" (Phase 3, Task 6).

The critic shares the actor's trunk (co-trained, one network). Conditioned on a
filtered critic observation it generates a short critique and then a score tag
``<score>N</score>`` with N an integer in [0, 10] -- modeled on the structured
output pattern of ``distilling.diagnose_provider.diagnose_episode``.

Two value paths:

  - **Rollout (sampling):** :func:`parse_score` extracts the integer (with a
    retry/clamp policy in the caller) and :func:`score_to_value` maps it to
    ``[0, 1]`` for the online per-node value store.
  - **Training (differentiable):** :func:`expected_score_value` computes the
    EXPECTED score over the 11 digit-token logits at the score position,
    ``V = sum_i softmax(logits)[i] * i / 10`` -- differentiable in the logits so
    the GAE value loss flows into the shared trunk.

This module imports torch lazily inside the differentiable function, so the
``dag`` package stays importable (and the parsing/prompt helpers usable) without
the training stack. It is intentionally NOT re-exported from ``dag/__init__``.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# Integer scores 0..10 inclusive -> 11 buckets.
SCORE_BUCKETS = 11
_SCORE_MIN = 0
_SCORE_MAX = SCORE_BUCKETS - 1

_SCORE_TAG = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.IGNORECASE)

_CRITIC_INSTRUCTION = (
    "You are the critic. Analyze the global state of the multi-agent task above "
    "(all agents and the environment so far) and judge how good this state is "
    "for eventually satisfying the task -- higher is better.\n\n"
    "First give a one-paragraph analysis, then output your value as an integer "
    "from 0 to 10 inclusive wrapped EXACTLY as <score>N</score> (0 = worst, "
    "10 = best). Output the tag once, last."
)


def build_critic_score_prompt(observation_text: str) -> str:
    """Render the critic prompt: the filtered observation + scoring instruction."""
    return f"{observation_text}\n\n{_CRITIC_INSTRUCTION}"


def parse_score(text: str) -> int:
    """Extract the integer from ``<score>N</score>`` and clamp to [0, 10].

    Raises :class:`ValueError` when no score tag is present so the caller can
    retry the generation (mirrors the diagnose-provider parse-retry policy).
    """
    match = _SCORE_TAG.search(text)
    if not match:
        raise ValueError("no <score>N</score> tag in critic output")
    raw = int(match.group(1))
    return max(_SCORE_MIN, min(_SCORE_MAX, raw))


def score_to_value(score: int) -> float:
    """Map an integer score in [0, 10] to a value in [0, 1]."""
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, int(score)))
    return clamped / _SCORE_MAX


def expected_value_from_logits_py(logits: list[float]) -> float:
    """Pure-Python reference: expected score / 10 from 11 bucket logits.

    ``V = sum_i softmax(logits)[i] * i / 10``. Used for testing the math without
    torch and as a backend-agnostic reference.
    """
    if len(logits) != SCORE_BUCKETS:
        raise ValueError(f"expected {SCORE_BUCKETS} logits, got {len(logits)}")
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    probs = [e / z for e in exps]
    expected_bucket = sum(p * i for i, p in enumerate(probs))
    return expected_bucket / _SCORE_MAX


def expected_score_value(bucket_logits: torch.Tensor) -> torch.Tensor:
    """Differentiable expected score in [0, 1] from the 11 digit-token logits.

    ``bucket_logits`` has shape ``(..., 11)`` -- the logits at the score position
    restricted to the integer tokens 0..10. Returns shape ``(...)`` with the
    expected score divided by 10. Gradients flow to ``bucket_logits`` so the
    value-regression loss trains the shared trunk.
    """
    import torch

    if bucket_logits.shape[-1] != SCORE_BUCKETS:
        raise ValueError(
            f"expected last dim {SCORE_BUCKETS}, got {bucket_logits.shape[-1]}"
        )
    probs = torch.softmax(bucket_logits.float(), dim=-1)
    buckets = torch.arange(SCORE_BUCKETS, dtype=probs.dtype, device=probs.device)
    expected_bucket = (probs * buckets).sum(dim=-1)
    return expected_bucket / _SCORE_MAX


__all__ = [
    "SCORE_BUCKETS",
    "build_critic_score_prompt",
    "expected_score_value",
    "expected_value_from_logits_py",
    "parse_score",
    "score_to_value",
]

# SPDX-License-Identifier: Apache-2.0
# customized_areal/tree_search/core/critic_prompt.py
"""Prompt construction and value-extraction utilities for the generative critic.

The generative critic reuses the actor's model (shared weights, shared SGLang
server). Given a partial solution -- the conversation through turn ``t`` -- it is
asked to emit a single integer in ``[0, score_max]`` estimating the probability
that the model will eventually succeed. The soft expected value over those digit
tokens is used as the state value ``v_phi(s_t)``.

This module is intentionally dependency-light: it operates on plain ``Node``-like
objects (anything exposing ``input_ids`` and ``loss_mask``) and a tokenizer-like
object exposing ``decode`` and ``encode``. This keeps the utilities unit-testable
with a tiny fake tokenizer and free of heavy runtime imports.

Reconstruction note
-------------------
Each ``Node`` stores the *fully chat-templated* token ids for the conversation up
through its own turn's response. To recover clean message *content* (without
template markup such as ``<|im_start|>``), spans are decoded with
``skip_special_tokens=True``. Roles are assigned from ``loss_mask`` turn
boundaries: response spans (``loss_mask == 1``) become ``assistant`` messages and
the surrounding context spans become ``user`` messages. Separating an embedded
system prompt from the first user span is not generally recoverable from tokens,
so the leading context span is treated as a single ``user`` message by default;
callers may pass ``system_prompt`` to prepend an explicit system message.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

DEFAULT_AVG_SUCCESS_RATE = 0.29


class TokenizerLike(Protocol):
    def decode(self, token_ids: list[int], skip_special_tokens: bool = ...) -> str: ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def _find_turn_boundaries(loss_mask: list[int]) -> tuple[list[int], list[int]]:
    """Scan loss_mask for 0->1 and 1->0 transitions.

    Returns (starts, ends) where each pair defines a half-open range
    [start, end) of response tokens. Mirrors tree_store._find_turn_boundaries
    but is reimplemented locally to avoid importing the torch-backed module.
    """
    starts: list[int] = []
    ends: list[int] = []
    in_response = False
    for i, v in enumerate(loss_mask):
        if v == 1 and not in_response:
            starts.append(i)
            in_response = True
        elif v == 0 and in_response:
            ends.append(i)
            in_response = False
    if in_response:
        ends.append(len(loss_mask))
    return starts, ends


def build_critic_instruction(
    avg_success_rate: float = DEFAULT_AVG_SUCCESS_RATE,
    score_max: int = 10,
) -> str:
    """Build the critic instruction text appended after the partial solution.

    The instruction asks the model to score the probability of eventual success
    as an integer in ``[0, score_max]`` (``score_max`` = most likely to succeed).
    """
    return (
        "Your job is to evaluate its probability of success, given the partial "
        "solution it has already generated. For your reference, the average "
        f"success rate of the model on the dataset is {avg_success_rate:.2f}.\n"
        "Instructions:\n"
        "1. Evaluate the difficulty of the problem.\n"
        "2. Recover the capabilities of the model based on the partial solution.\n"
        "3. Skim through the partial solution and detect any progress, error, or "
        "confusion.\n"
        "4. Analyze the probability of success if the model finishes the "
        "solution.\n"
        f"5. Output the final answer as an integer between 0 and {score_max} "
        f"inclusive ({score_max} most likely)."
    )


def reconstruct_messages_through_turn(
    node: Any,
    tokenizer: TokenizerLike,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Reconstruct conversation messages for the partial solution in ``node``.

    ``node`` must expose ``input_ids`` and ``loss_mask`` for the full sequence up
    through its turn's response. Response spans become ``assistant`` messages and
    context spans become ``user`` messages, with content decoded without special
    tokens. An optional explicit ``system_prompt`` is prepended.
    """
    input_ids = list(node.input_ids)
    loss_mask = list(node.loss_mask)
    if len(loss_mask) != len(input_ids):
        raise ValueError(
            f"input_ids (len {len(input_ids)}) and loss_mask (len {len(loss_mask)}) "
            "must have equal length"
        )

    starts, ends = _find_turn_boundaries(loss_mask)

    # Build an ordered list of (role, [start, end)) spans covering the sequence.
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for s, e in zip(starts, ends):
        if s > cursor:
            spans.append(("user", cursor, s))
        spans.append(("assistant", s, e))
        cursor = e
    if cursor < len(input_ids):
        spans.append(("user", cursor, len(input_ids)))

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    for role, s, e in spans:
        text = tokenizer.decode(input_ids[s:e], skip_special_tokens=True).strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def build_critic_messages(
    messages_through_turn_t: list[dict[str, str]],
    critic_instruction: str,
    instruction_role: str = "user",
) -> list[dict[str, str]]:
    """Append the critic instruction as a final message.

    Returns ``[*messages_through_turn_t, {role: instruction_role, content:
    critic_instruction}]``. The caller is responsible for applying the chat
    template (with ``add_generation_prompt=True``) to obtain token ids.
    """
    return [
        *messages_through_turn_t,
        {"role": instruction_role, "content": critic_instruction},
    ]


def digit_token_ids(
    tokenizer: TokenizerLike,
    score_max: int = 10,
) -> dict[int, list[int]]:
    """Resolve the token ids for each integer label ``0..score_max``.

    Labels that tokenize to multiple tokens (e.g. ``"10"``) keep all their token
    ids; the value-extraction helper treats a multi-token label's logprob as the
    joint (summed) logprob of its tokens.
    """
    labels: dict[int, list[int]] = {}
    for score in range(score_max + 1):
        ids = list(tokenizer.encode(str(score), add_special_tokens=False))
        if not ids:
            raise ValueError(f"tokenizer produced no tokens for label {score!r}")
        labels[score] = ids
    return labels


def expected_value_from_logprobs(
    label_logprobs: dict[int, float],
    score_max: int = 10,
) -> float:
    """Compute the expected value ``Sum_i p_i * (i / score_max)``.

    ``label_logprobs`` maps an integer label to its (aggregated) logprob. The
    logprobs are renormalized into a categorical distribution over the labels
    that are present, then the expected normalized score is returned in [0, 1].

    Returns ``0.0`` if no labels are present.
    """
    if score_max < 1:
        raise ValueError(f"score_max must be >= 1, got {score_max}")
    if not label_logprobs:
        return 0.0

    max_lp = max(label_logprobs.values())
    weights = {i: math.exp(lp - max_lp) for i, lp in label_logprobs.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0

    value = 0.0
    for i, w in weights.items():
        prob = w / total
        value += prob * (i / score_max)
    # Numerical guard.
    return min(1.0, max(0.0, value))


def variance_from_logprobs(
    label_logprobs: dict[int, float],
    score_max: int = 10,
) -> float:
    """Categorical variance of the critic's score distribution on [0, 1].

    Uses the same renormalization as :func:`expected_value_from_logprobs` and
    returns ``Sum_i p_i * (i/score_max - v)^2`` where ``v`` is the expected
    normalized value. This is the critic's own uncertainty about ``v`` and is
    used as ``var_theta`` in the variance-aware hybrid blend. A degenerate
    one-hot distribution yields ``0.0`` (the caller applies a variance floor).

    Returns ``0.0`` if no labels are present.
    """
    if score_max < 1:
        raise ValueError(f"score_max must be >= 1, got {score_max}")
    if not label_logprobs:
        return 0.0

    max_lp = max(label_logprobs.values())
    weights = {i: math.exp(lp - max_lp) for i, lp in label_logprobs.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0

    mean = 0.0
    for i, w in weights.items():
        mean += (w / total) * (i / score_max)

    var = 0.0
    for i, w in weights.items():
        prob = w / total
        diff = (i / score_max) - mean
        var += prob * diff * diff
    return max(0.0, var)

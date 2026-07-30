# SPDX-License-Identifier: Apache-2.0
# customized_areal/tree_search/core/judge_prompt.py
"""Prompt construction and score-extraction utilities for the LLM judge.

A larger judge model receives a *complete* multi-turn episode (every assistant
turn) together with the gold answer and assigns each assistant turn an integer
credit in ``[0, score_max]`` reflecting how much that turn contributed toward
reaching the gold answer. Those per-turn scores are later normalized into a
per-episode credit distribution and turned into a dense process reward.

This module mirrors :mod:`customized_areal.tree_search.core.critic_prompt`: it is
dependency-light (pure string/regex helpers) so it can be unit-tested without the
heavy runtime or a live model. The structured-output discipline (XML wrapped in
``\\`\\`\\`xml`` fences) matches ``ExternalDiagnoseProvider.diagnose_episode`` so the
same client and parsing assumptions apply.
"""

from __future__ import annotations

import re


def build_judge_instruction(score_max: int, gold_answer: str) -> str:
    """Build the judge instruction appended after the full episode.

    The judge is asked to score each assistant turn's contribution toward the
    gold answer as an integer in ``[0, score_max]`` (``score_max`` = most
    helpful) and to emit only structured XML.
    """
    if score_max < 1:
        raise ValueError(f"score_max must be >= 1, got {score_max}")
    return (
        "You are judging a multi-turn assistant trajectory above. Knowing the "
        "gold answer, assign each assistant turn an integer credit between 0 and "
        f"{score_max} inclusive ({score_max} = contributed the most toward "
        "reaching the gold answer, 0 = contributed nothing or hurt). Judge each "
        "turn by how much genuine progress it made toward the gold answer.\n"
        "You MUST output ONLY XML (no extra text) wrapped in ```xml fences. The "
        "XML must have a top-level <judgment> element containing a <turns> "
        "element. Each <turn> must include <turn_idx> (int, starts from 1) and "
        "<score> (int between 0 and "
        f"{score_max}). Include every assistant turn.\n\n"
        f"Gold answer: {gold_answer}\n\n"
        "Example response format:\n"
        "```xml\n"
        "<judgment>\n"
        "  <turns>\n"
        "    <turn>\n"
        "      <turn_idx>1</turn_idx>\n"
        f"      <score>{score_max}</score>\n"
        "    </turn>\n"
        "  </turns>\n"
        "</judgment>\n"
        "```"
    )


# Match a <turn> ... </turn> block, then pull turn_idx / score from within it.
_TURN_BLOCK_RE = re.compile(r"<turn>(.*?)</turn>", re.DOTALL | re.IGNORECASE)
_TURN_IDX_RE = re.compile(r"<turn_idx>\s*(-?\d+)\s*</turn_idx>", re.IGNORECASE)
_SCORE_RE = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.IGNORECASE)


def parse_turn_scores(text: str, score_max: int) -> dict[int, int]:
    """Extract ``{turn_idx: score}`` from a judge XML response.

    Robust to surrounding prose / ```xml fences. Scores are clamped into
    ``[0, score_max]``. Turn blocks missing a ``turn_idx`` or ``score`` (or with
    a non-positive ``turn_idx``) are skipped. When the same ``turn_idx`` appears
    more than once the last occurrence wins.

    Returns an empty dict if nothing parseable is found (drives the sparse
    fallback downstream).
    """
    if score_max < 1:
        raise ValueError(f"score_max must be >= 1, got {score_max}")
    if not text:
        return {}

    scores: dict[int, int] = {}
    for block in _TURN_BLOCK_RE.findall(text):
        idx_match = _TURN_IDX_RE.search(block)
        score_match = _SCORE_RE.search(block)
        if idx_match is None or score_match is None:
            continue
        turn_idx = int(idx_match.group(1))
        if turn_idx < 1:
            continue
        score = int(score_match.group(1))
        score = max(0, min(score_max, score))
        scores[turn_idx] = score
    return scores

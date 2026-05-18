"""Selected-turn distillation builders for tree-search trajectories."""

from __future__ import annotations

import json
from typing import Any

from customized_areal.tree_search.core.teacher_provider import TeacherProvider
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    EpisodeDiagnosis,
    PositionRewardInfo,
)
from customized_areal.tree_search.mcts_tree_store import Node


GUIDANCE_PROMPT_TEMPLATE = (
    "\n\nImprove this selected assistant turn using this guidance:\n{guidance}\n\n"
)


def parse_episode_diagnosis(raw_text: str) -> EpisodeDiagnosis:
    """Parse strict teacher diagnosis JSON into an EpisodeDiagnosis."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("episode diagnosis must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("episode diagnosis must be a JSON object")

    turns_payload = payload.get("turns")
    if not isinstance(turns_payload, list):
        raise ValueError("episode diagnosis must contain a top-level turns list")

    turns: list[DiagnosisTurn] = []
    for index, turn_payload in enumerate(turns_payload):
        if not isinstance(turn_payload, dict):
            raise ValueError(f"turns[{index}] must be a JSON object")

        turn_idx = turn_payload.get("turn_idx")
        should_improve = turn_payload.get("should_improve")
        guidance = turn_payload.get("guidance", "")

        if not isinstance(turn_idx, int) or isinstance(turn_idx, bool):
            raise ValueError(f"turns[{index}].turn_idx must be an integer")
        if not isinstance(should_improve, bool):
            raise ValueError(f"turns[{index}].should_improve must be a boolean")
        if not isinstance(guidance, str):
            raise ValueError(f"turns[{index}].guidance must be a string")

        turns.append(
            DiagnosisTurn(
                turn_idx=turn_idx,
                should_improve=should_improve,
                guidance=guidance,
            )
        )

    return EpisodeDiagnosis(turns=tuple(turns))


def response_token_span(loss_mask: list[int]) -> tuple[int, int]:
    """Return the first contiguous response span in a loss mask."""
    start: int | None = None
    for index, value in enumerate(loss_mask):
        if value == 1 and start is None:
            start = index
        elif value != 1 and start is not None:
            return start, index
    if start is None:
        return 0, 0
    return start, len(loss_mask)


def build_teacher_prompt_ids(
    node: Node, guidance: str, tokenizer: Any
) -> tuple[list[int], list[int]]:
    """Build teacher prompt IDs and selected response IDs for a node."""
    input_ids = _as_list(node.input_ids)
    loss_mask = _as_list(node.loss_mask)
    start, end = response_token_span(loss_mask)
    prefix_ids = input_ids[:start]
    generation_ids = input_ids[start:end]

    guidance_text = GUIDANCE_PROMPT_TEMPLATE.format(guidance=guidance.strip())
    guidance_ids = _encode(tokenizer, guidance_text)
    return prefix_ids + guidance_ids, generation_ids


async def selected_turn_to_position_rewards(
    node: Node,
    guidance: str,
    tokenizer: Any,
    provider: TeacherProvider,
    sample_index: int,
    topk_distill: bool,
    engine: Any,
    teacher_top_k: int,
) -> list[PositionRewardInfo]:
    """Convert one selected turn into position-level teacher rewards.

    Task 3 fully supports the single-candidate path. The top-k path keeps a
    minimal local skeleton so callers can opt in when cached top-k fields are
    already present; Task 4 will harden recomputation via ``engine``.
    """
    del engine
    del teacher_top_k

    prompt_ids, generation_ids = build_teacher_prompt_ids(node, guidance, tokenizer)
    if not generation_ids:
        return []

    logprobs = _as_list(node.logprobs)
    start, end = response_token_span(_as_list(node.loss_mask))
    generation_logprobs = [float(value) for value in logprobs[start:end]]

    if topk_distill and node.topk_ids is not None and node.topk_logp is not None:
        candidate_token_ids = [list(candidates) for candidates in node.topk_ids]
        student_logprobs = [
            [float(logprob) for logprob in position_logprobs]
            for position_logprobs in node.topk_logp
        ]
    else:
        candidate_token_ids = [[token_id] for token_id in generation_ids]
        student_logprobs = [[logprob] for logprob in generation_logprobs]

    teacher_logprobs = await provider.get_logprobs_for_prompt(
        prompt_ids=prompt_ids,
        generation_ids=generation_ids,
        candidate_token_ids=candidate_token_ids,
    )

    position_rewards: list[PositionRewardInfo] = []
    for position, (candidate_ids, student_lps, teacher_lps) in enumerate(
        zip(candidate_token_ids, student_logprobs, teacher_logprobs, strict=True)
    ):
        if len(candidate_ids) != len(student_lps) or len(candidate_ids) != len(
            teacher_lps
        ):
            raise ValueError(
                "candidate, student logprob, and teacher logprob lengths must match"
            )

        rewards = [
            student_logprob - teacher_logprob
            for student_logprob, teacher_logprob in zip(
                student_lps, teacher_lps, strict=True
            )
        ]
        position_rewards.append(
            PositionRewardInfo(
                position=position,
                candidates=[str(token_id) for token_id in candidate_ids],
                candidate_token_ids=list(candidate_ids),
                logprobs=list(student_lps),
                teacher_logprobs=list(teacher_lps),
                rewards=rewards,
                chosen_index=0,
                sample_index=sample_index,
            )
        )

    return position_rewards


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    return list(encoded)


def _as_list(values: Any) -> list:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


__all__ = [
    "build_teacher_prompt_ids",
    "parse_episode_diagnosis",
    "response_token_span",
    "selected_turn_to_position_rewards",
]

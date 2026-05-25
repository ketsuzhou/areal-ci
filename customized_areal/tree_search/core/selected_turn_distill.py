"""Selected-turn distillation builders for tree-search trajectories."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from customized_areal.tree_search.core.teacher_provider import TeacherProvider
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    EpisodeDiagnosis,
    PositionRewardInfo,
)
from customized_areal.tree_search.mcts_tree_store import Node

from areal.utils import logging

logger = logging.getLogger("SelectedTurnDistill")

GUIDANCE_PROMPT_TEMPLATE = (
    "\n\nImprove this selected assistant turn using this guidance:\n{guidance}\n\n"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_TURNS_JSON_RE = re.compile(r'\{\s*"turns"\s*:', re.DOTALL)
# " "turns" (extra space before key)
_BROKEN_TURNS_RE = re.compile(r'\{\s*"\s+"turns"\s*:', re.DOTALL)
# Find the last code fence marker
_LAST_FENCE_RE = re.compile(r"```(?:json)?\s*", re.DOTALL)


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON content from markdown code fences or plain text.

    Reasoning models often output chain-of-thought text before the JSON.
    Handles: complete fences, incomplete fences (no closing ```),
    bare JSON objects, and models that wrap output in XML tags.
    """
    # 1. Try complete code fences: ```json ... ```
    match = _FENCE_RE.search(text)
    if match:
        content = match.group(1).strip()
        if content:
            # Strip stray leading "json" / non-JSON prefixes before {
            brace = content.find("{")
            bracket = content.find("[")
            first = brace if brace >= 0 and (bracket < 0 or brace < bracket) else bracket
            if first >= 0:
                content = content[first:]
            return content

    # 2. Try incomplete code fence — opening ``` exists but no closing ```
    #    Take everything after the *last* opening fence marker.
    fence_markers = list(_LAST_FENCE_RE.finditer(text))
    if fence_markers:
        after_fence = text[fence_markers[-1].end() :]
    else:
        after_fence = text

    # 3. Strip leading XML/thinking tags (</think>, </reasoning>, </analysis>)
    after_fence = re.sub(
        r"</(?:think|reasoning|analysis|thought)>", "", after_fence
    ).strip()

    # 4. Search for JSON object with "turns" key in remaining text
    for pattern in (_TURNS_JSON_RE, _BROKEN_TURNS_RE):
        obj_match = pattern.search(after_fence)
        if obj_match:
            start = obj_match.start()
            depth = 0
            end = start
            for i, ch in enumerate(after_fence[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                raw = after_fence[start:end].strip()
                # Fix " "turns" → "turns"
                raw = re.sub(r'"\s+"turns"', '"turns"', raw)
                return raw

    # 5. Fall back to original text for the bare-JSON search
    for pattern in (_TURNS_JSON_RE, _BROKEN_TURNS_RE):
        obj_match = pattern.search(text)
        if obj_match:
            start = obj_match.start()
            depth = 0
            end = start
            for i, ch in enumerate(text[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                raw = text[start:end].strip()
                raw = re.sub(r'"\s+"turns"', '"turns"', raw)
                return raw

    return text.strip()


def parse_episode_diagnosis(raw_text: str | None) -> EpisodeDiagnosis:
    """Parse teacher diagnosis into an EpisodeDiagnosis.

    Handles JSON wrapped in markdown code fences and empty responses.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("episode diagnosis is empty — the model returned no content")

    cleaned = _extract_json_from_markdown(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        tail = raw_text[-300:] if len(raw_text) > 600 else ""
        logger.error(
            "Failed to parse episode diagnosis as JSON. "
            "len=%d, first 300=%r, last 300=%r",
            len(raw_text),
            raw_text[:300],
            tail,
        )
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
    """Return the current selected response span in a loss mask.

    In concat-mode multi-turn nodes, parent assistant spans are retained in
    the node loss mask and the current turn is appended at the end. The
    selected turn's generation is therefore the latest contiguous response
    span, not the earliest one.
    """
    start: int | None = None
    latest_span: tuple[int, int] | None = None
    for index, value in enumerate(loss_mask):
        if value == 1 and start is None:
            start = index
        elif value != 1 and start is not None:
            latest_span = (start, index)
            start = None
    if start is not None:
        latest_span = (start, len(loss_mask))
    if latest_span is None:
        return 0, 0
    return latest_span


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
    """Convert one selected turn into position-level teacher rewards."""
    prompt_ids, generation_ids = build_teacher_prompt_ids(node, guidance, tokenizer)
    if not generation_ids:
        return []

    logprobs = _as_list(node.logprobs)
    loss_mask = _as_list(node.loss_mask)
    start, end = response_token_span(loss_mask)
    generation_logprobs = [float(value) for value in logprobs[start:end]]

    if topk_distill:
        if node.topk_ids is None or node.topk_logp is None:
            topk_ids, topk_logp = await _recompute_student_topk(
                engine=engine,
                node=node,
                teacher_top_k=teacher_top_k,
            )
        else:
            topk_ids, topk_logp = _select_current_topk_rows(
                topk_ids=node.topk_ids,
                topk_logp=node.topk_logp,
                input_ids=_as_list(node.input_ids),
                loss_mask=loss_mask,
            )
        candidate_token_ids = []
        student_logprobs = []
        for generated_id, generated_logprob, candidates, position_logprobs in zip(
            generation_ids,
            generation_logprobs,
            topk_ids,
            topk_logp,
            strict=True,
        ):
            reordered_ids = [generated_id]
            reordered_logprobs = [generated_logprob]
            for token_id, logprob in zip(candidates, position_logprobs, strict=True):
                if token_id == generated_id:
                    continue
                reordered_ids.append(int(token_id))
                reordered_logprobs.append(float(logprob))
            candidate_token_ids.append(reordered_ids)
            student_logprobs.append(reordered_logprobs)
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


async def _recompute_student_topk(
    engine: Any,
    node: Node,
    teacher_top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    get_topk_logprobs = getattr(engine, "get_topk_logprobs", None)
    if get_topk_logprobs is None or not callable(get_topk_logprobs):
        raise NotImplementedError(
            "topk_distill requires engine.get_topk_logprobs for missing student top-k"
        )

    maybe_topk = get_topk_logprobs(
        input_ids=node.input_ids,
        loss_mask=node.loss_mask,
        top_k=teacher_top_k,
    )
    if not inspect.isawaitable(maybe_topk):
        raise NotImplementedError(
            "topk_distill requires awaitable engine.get_topk_logprobs for missing "
            "student top-k"
        )
    topk_ids, topk_logp = await maybe_topk
    selected_topk_ids, selected_topk_logp = _select_current_topk_rows(
        topk_ids=topk_ids,
        topk_logp=topk_logp,
        input_ids=_as_list(node.input_ids),
        loss_mask=_as_list(node.loss_mask),
    )
    node.topk_ids = selected_topk_ids
    node.topk_logp = selected_topk_logp
    return selected_topk_ids, selected_topk_logp


def _select_current_topk_rows(
    topk_ids: Any,
    topk_logp: Any,
    input_ids: list[int],
    loss_mask: list[int],
) -> tuple[list[list[int]], list[list[float]]]:
    ids_rows = _nested_list(topk_ids)
    logp_rows = _nested_list(topk_logp)
    start, end = response_token_span(loss_mask)
    response_len = end - start

    if len(ids_rows) != len(logp_rows):
        raise ValueError("top-k id rows and logprob rows must use the same layout")

    if len(ids_rows) == len(input_ids):
        selected_ids = ids_rows[start:end]
        selected_logp = logp_rows[start:end]
    elif len(ids_rows) == response_len:
        selected_ids = ids_rows
        selected_logp = logp_rows
    else:
        total_response_len = sum(1 for value in loss_mask if value == 1)
        if len(ids_rows) != total_response_len:
            raise ValueError(
                "top-k rows must be full-sequence, selected-response, or all-response aligned"
            )
        response_offset = sum(1 for value in loss_mask[:start] if value == 1)
        selected_ids = ids_rows[response_offset : response_offset + response_len]
        selected_logp = logp_rows[response_offset : response_offset + response_len]

    if len(selected_ids) != response_len or len(selected_logp) != response_len:
        raise ValueError("top-k rows must align with selected generation positions")
    return selected_ids, selected_logp


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


def _nested_list(rows: Any) -> list[list[Any]]:
    return [_as_list(row) for row in _as_list(rows)]


__all__ = [
    "build_teacher_prompt_ids",
    "parse_episode_diagnosis",
    "response_token_span",
    "selected_turn_to_position_rewards",
]

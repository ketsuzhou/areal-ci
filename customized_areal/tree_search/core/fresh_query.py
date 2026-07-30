# customized_areal/tree_search/core/fresh_query.py
"""Fresh-query selection/claim helpers for TreeSearchGroupedRolloutWorkflow.

A fresh query is consumed in two phases (see the workflow class):
``_select_fresh_query_data`` picks an eligible row WITHOUT claiming it, and
``_claim_fresh_query`` appends ``TRAIN_ID`` to ``used4train`` only after the
rollout produced a usable training batch — otherwise failed or
zero-variance-discarded rollouts would permanently consume the query.

This module holds the module-level helpers those two phases rely on. Heavy or
optional dependencies (the Supabase ``DBConnection``) stay as lazy imports in
the workflow methods so this module remains importable with only stdlib +
``areal.utils.logging`` available.
"""

from __future__ import annotations

from typing import Any

from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")

_FRESH_QUERY_SELECT_LIMIT = 100


def _normalize_used4train(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _apply_fresh_query_row(data: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    query_id = row.get("query_id")
    query = row.get("query")
    gold_answer = row.get("gold_answer")
    evaluation_rubric = row.get("evaluation_rubric")
    used4train = row.get("used4train")

    file_paths = row.get("file_paths")

    merged["query_id"] = str(query_id or "")
    merged["query"] = str(query or "")
    merged["answer"] = str(gold_answer or "")
    merged["evaluation_rubric"] = (
        evaluation_rubric if isinstance(evaluation_rubric, list) else []
    )
    merged["used4train"] = _normalize_used4train(used4train)
    merged["file_paths"] = file_paths if isinstance(file_paths, list) else []
    return merged


async def _execute_fresh_query_claim_update(
    query: Any,
    *,
    train_id: str,
) -> Any:
    query = _apply_train_id_not_contains_filter(query, train_id=train_id)
    return await query.execute()


def _apply_train_id_not_contains_filter(query: Any, *, train_id: str) -> Any:
    try:
        return query.not_.contains("used4train", [train_id])
    except AttributeError:
        logger.warning(
            "Supabase client does not expose not_.contains; fresh query claim "
            "will rely on the query_id update guard only"
        )
        return query


def _affected_row_count(result: Any) -> int:
    data = getattr(result, "data", None)
    if isinstance(data, list):
        return len(data)
    count = getattr(result, "count", None)
    return count if isinstance(count, int) else 0

# customized_areal/tree_search/core/distill_prep.py
"""Selected-turn distillation preparation for TreeSearchGroupedRolloutWorkflow.

Split out of ``customized_grouped_workflow.py`` as explicit-parameter free
functions (no ``self.config`` dependency) so the failure-recovery stub tests
can exercise them against a bare workflow instance:

- ``prepare_distill_for_episode`` — diagnose one episode, identify turns
  needing improvement, gather teacher logprobs, and build per-position reward
  info.
- ``prepare_distill_for_node_groups`` — the parallel multi-episode wrapper.
- ``setup_distill_provider`` — teacher client + diagnose provider construction
  from a :class:`Config` (only called on fully-constructed workflows).
- ``_filter_distill_episode_failure`` / ``_input_ids_to_messages`` — helpers.

Heavy dependencies (``distilling.*``) stay as function-local lazy imports so
the module remains importable in minimal (stubbed) environments.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from customized_areal.tree_search.config import Config, LossMode
from customized_areal.tree_search.core.tree_store import Node

from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")


def _filter_distill_episode_failure(
    nodes: list[Node], loss_mode: LossMode
) -> list[Node]:
    if loss_mode == LossMode.DISTILL:
        return []
    return nodes


def _input_ids_to_messages(
    input_ids: list[int], tokenizer: Any
) -> list[dict[str, str]]:
    """Convert full-context token IDs to a list of role/content message dicts.

    Uses ``apply_chat_template`` on a dummy conversation to derive the
    format markers, then parses the decoded token sequence with those
    markers.  Falls back to a single-user-message format for unrecognized
    templates.
    """
    # Derive the chat-template markers from a dummy round-trip.
    _DUMMY = [{"role": "user", "content": "X"}]
    try:
        formatted = tokenizer.apply_chat_template(_DUMMY, tokenize=False)
    except Exception:
        formatted = "<|im_start|>user\nX<|im_end|>\n"
    m_start = re.search(r"(<\S+?>)(system|user|assistant)", formatted)
    start_token = m_start.group(1) if m_start else "<|im_start|>"
    m_end = re.search(r"(<\S+?>)", formatted[::-1])
    end_token = m_end.group(1)[::-1] if m_end else "<|im_end|>"

    try:
        raw = tokenizer.decode(input_ids, skip_special_tokens=False)
    except TypeError:
        raw = tokenizer.decode(input_ids)

    pattern = re.compile(
        re.escape(start_token)
        + r"(system|user|assistant|tool)\s*\n(.*?)"
        + re.escape(end_token),
        re.DOTALL,
    )

    messages: list[dict[str, str]] = []
    for match in pattern.finditer(raw):
        role = match.group(1)
        content = match.group(2).strip()
        if content:
            if role == "tool":
                role = "user"
            messages.append({"role": role, "content": content})

    if not messages:
        try:
            fallback = tokenizer.decode(input_ids, skip_special_tokens=True)
        except TypeError:
            fallback = tokenizer.decode(input_ids)
        messages = [{"role": "user", "content": fallback}]

    return messages


async def setup_distill_provider(config: Config, engine: Any, tokenizer: Any = None):
    """Construct the teacher client + diagnose provider from ``config``.

    ``teacher_provider == "engine"`` resolves the teacher endpoint from the
    inference engine (proxy gateway or direct server address, with an SGLang
    backend probe); otherwise the external teacher/diagnose URLs come from
    ``config`` with environment-variable fallbacks for the diagnose side.
    """
    from customized_areal.tree_search.distilling.diagnose_provider import (
        ExternalDiagnoseProvider,
    )
    from customized_areal.tree_search.distilling.teacher_client import (
        TeacherClient,
        TeacherConfig,
    )

    if config.teacher_provider == "engine":
        proxy_addr = getattr(engine, "_proxy_gateway_addr", "") or ""
        engine_addrs = getattr(engine, "addresses", None) or []
        admin_api_key = getattr(engine.config, "admin_api_key", "") or ""
        if proxy_addr:
            teacher_base_url = proxy_addr
        elif engine_addrs:
            teacher_base_url = f"http://{engine_addrs[0]}"
        else:
            teacher_base_url = config.teacher_base_url

        # Detect backend type from engine's backend attribute
        backend_obj = getattr(engine, "backend", None)
        backend_cls_name = type(backend_obj).__name__ if backend_obj else ""
        if backend_cls_name == "SGLangBackend":
            teacher_backend = "sglang"
            # SGLang /generate is not available on the proxy gateway;
            # use the direct SGLang server address instead.
            if engine_addrs:
                teacher_base_url = f"http://{engine_addrs[0]}"
        else:
            teacher_backend = "openai"

        logger.info(
            "Teacher provider=engine, resolved teacher_base_url=%s, teacher_backend=%s",
            teacher_base_url,
            teacher_backend,
        )

        if not teacher_base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"teacher_base_url must start with http:// or https://, "
                f"got: {teacher_base_url!r}"
            )

        teacher_config = TeacherConfig(
            teacher_base_url=teacher_base_url,
            teacher_model_name=config.teacher_model_name,
            teacher_api_key=admin_api_key,
            teacher_top_k=config.teacher_top_k,
            teacher_max_retries=config.teacher_max_retries,
            teacher_timeout=config.teacher_timeout,
            teacher_missing_logprob=config.teacher_missing_logprob,
            teacher_backend=teacher_backend,
            teacher_max_concurrency=config.teacher_max_concurrency,
        )
        client = TeacherClient(teacher_config)
    else:
        teacher_config = TeacherConfig(
            teacher_base_url=config.teacher_base_url,
            teacher_model_name=config.teacher_model_name,
            teacher_api_key=config.teacher_api_key,
            teacher_top_k=config.teacher_top_k,
            teacher_max_retries=config.teacher_max_retries,
            teacher_timeout=config.teacher_timeout,
            teacher_missing_logprob=config.teacher_missing_logprob,
            teacher_backend=config.teacher_backend,
            teacher_max_concurrency=config.teacher_max_concurrency,
        )
        client = TeacherClient(teacher_config)

    diagnose_model_name = config.diagnose_model_name or "qwen/qwen3.7-max"
    diagnose_api_key = (
        config.diagnose_api_key
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("WORKSPACE_OPENAI_API_KEY", "")
    )
    diagnose_base_url = (
        config.diagnose_base_url
        or os.environ.get("OPENROUTER_BASE_URL", "")
        or os.environ.get("WORKSPACE_OPENAI_API_BASE", "")
    )
    provider = ExternalDiagnoseProvider(
        client=client,
        diagnose_model_name=diagnose_model_name,
        diagnose_temperature=config.diagnose_temperature,
        diagnose_max_tokens=config.diagnose_max_tokens,
        diagnose_base_url=diagnose_base_url,
        diagnose_api_key=diagnose_api_key,
        tokenizer=tokenizer,
    )

    return provider, client


async def prepare_distill_for_episode(
    nodes: list[Node],
    data: dict[str, Any],
    engine: Any,
    provider: Any,
    tokenizer: Any,
    *,
    loss_mode: LossMode,
    topk_distill: bool,
    teacher_top_k: int,
    max_distill_tokens: int,
) -> tuple[list[Node], dict[str, list[Any]]]:
    """Run selected-turn distillation for one episode's nodes.

    Diagnoses the episode (reusing cached guidance when present), then gathers
    teacher logprobs for the selected turns. Teacher backend failures degrade
    gracefully: a failed diagnosis skips distillation for the episode; a failed
    logprob request skips only that node; when every node failed, DISTILL mode
    drops the episode entirely (:func:`_filter_distill_episode_failure`).
    """
    from customized_areal.tree_search.distilling.selected_turn_distill import (
        parse_episode_diagnosis,
        selected_turn_to_position_rewards,
    )
    from customized_areal.tree_search.distilling.teacher_client import (
        TeacherServiceError,
    )

    if not nodes:
        return nodes, {}
    # Reuse cached guidance from a previous diagnosis to avoid the
    # expensive diagnose_episode call across training iterations.
    if nodes[-1].guidance:
        selected = nodes[-1].guidance
    else:
        # Build structured messages from the last node's full context,
        # then append the diagnosis instruction as the final user message.
        conversation = _input_ids_to_messages(nodes[-1].input_ids, tokenizer)
        gold_answer = str(data.get("answer", ""))

        raw = None
        diagnosis = None
        max_retries = 3
        base_temp = 0.7
        for retry in range(max_retries):
            try:
                temp = base_temp + retry * 0.3
                raw = await provider.diagnose_episode(
                    conversation, gold_answer, temperature=temp
                )
                diagnosis = parse_episode_diagnosis(raw)
                break
            except TeacherServiceError as exc:
                logger.warning(
                    "Diagnose request failed for episode_id=%s; proceeding "
                    "without guidance: %s",
                    nodes[0].episode_id,
                    exc,
                )
                diagnosis = None
                break
            except ValueError:
                if retry < max_retries - 1:
                    logger.warning(
                        "Diagnose parse failed (attempt %d/%d), retrying "
                        "with temperature=%.1f",
                        retry + 1,
                        max_retries,
                        base_temp + (retry + 1) * 0.3,
                    )
                else:
                    logger.error(
                        "Diagnose parse failed after %d attempts for episode_id=%s",
                        max_retries,
                        nodes[0].episode_id,
                    )
                    raise
        selected = diagnosis.selected_turns if diagnosis is not None else {}
        if selected:
            nodes[-1].guidance = dict(selected)
    if not selected:
        return nodes, {}

    async def _run_one_node(node: Node) -> tuple[str, list[Any]] | None:
        guidance = selected.get(node.turn_idx, "")
        if not guidance:
            return None
        try:
            rewards = await selected_turn_to_position_rewards(
                node=node,
                guidance=guidance,
                tokenizer=tokenizer,
                provider=provider,
                sample_index=0,
                topk_distill=topk_distill,
                engine=engine,
                teacher_top_k=teacher_top_k,
                max_distill_tokens=max_distill_tokens,
            )
        except TeacherServiceError as exc:
            logger.warning(
                "Teacher logprob request failed for episode_id=%s "
                "node_id=%s turn_idx=%s; skipping distill targets: %s",
                node.episode_id,
                node.node_id,
                node.turn_idx,
                exc,
            )
            return None
        if not rewards:
            return None
        node.teacher_logp = [reward.teacher_logprobs or [] for reward in rewards]
        node.topk_ids = [reward.candidate_token_ids for reward in rewards]
        return node.node_id, rewards

    # Run the last node first to warm the KV cache, then parallelize
    # the remaining nodes so they benefit from the pre-warmed cache.
    last_result = await _run_one_node(nodes[-1])
    other_results = await asyncio.gather(*[_run_one_node(node) for node in nodes[:-1]])

    rewards_by_node_id: dict[str, list[Any]] = {}
    for result in [last_result, *other_results]:
        if result is not None:
            node_id, rewards = result
            rewards_by_node_id[node_id] = rewards
    if not rewards_by_node_id:
        return _filter_distill_episode_failure(nodes, loss_mode), {}
    return nodes, rewards_by_node_id


async def prepare_distill_for_node_groups(
    node_groups: list[list[Node]],
    data: dict[str, Any],
    engine: Any,
    provider: Any,
    tokenizer: Any,
    *,
    loss_mode: LossMode,
    topk_distill: bool,
    teacher_top_k: int,
    max_distill_tokens: int,
) -> tuple[list[Node], dict[str, list[Any]]]:
    """Run :func:`prepare_distill_for_episode` across episodes in parallel."""

    async def _run_one(
        nodes: list[Node],
    ) -> tuple[list[Node], dict[str, list[Any]]]:
        try:
            return await prepare_distill_for_episode(
                nodes,
                data,
                engine,
                provider,
                tokenizer,
                loss_mode=loss_mode,
                topk_distill=topk_distill,
                teacher_top_k=teacher_top_k,
                max_distill_tokens=max_distill_tokens,
            )
        except Exception:
            logger.exception(
                "Selected-turn distillation failed for episode_id=%s",
                nodes[0].episode_id if nodes else "",
            )
            return _filter_distill_episode_failure(nodes, loss_mode), {}

    results = await asyncio.gather(*[_run_one(nodes) for nodes in node_groups])

    prepared_nodes: list[Node] = []
    rewards_by_node_id: dict[str, list[Any]] = {}
    for episode_nodes, episode_rewards in results:
        prepared_nodes.extend(episode_nodes)
        rewards_by_node_id.update(episode_rewards)
    return prepared_nodes, rewards_by_node_id

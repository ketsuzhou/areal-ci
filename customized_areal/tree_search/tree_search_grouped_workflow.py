# customized_areal/tree_search/tree_search_grouped_workflow.py
"""Tree-search-aware grouped rollout workflow with cache reuse and teacher distillation.

Consolidates the functionality of QueryIDProxyWorkflow,
TreeSearchGroupedRolloutWorkflow, and TreeSearchWorkflowExecutor into
a single class that:
- Loads/saves tree_store from a checkpoint directory
- Does per-query cache lookup to determine how many fresh episodes are needed (episode-level counting)
- Generates only the needed fresh episodes (partial cache reuse)
- Converts fresh results to Nodes, loads cached Nodes, combines them
- Performs teacher model reward computation (diagnosis + teacher logprob gathering) for distillation
- Inserts fresh Nodes into tree_store, computes advantages, marks trained
- Saves tree checkpoint
- Returns batched tensor dicts that the base WorkflowExecutor handles natively
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from customized_areal.db_service import (
    bind_sandbox_to_task,
    copy_messages_to_task,
    create_task,
    delete_sandbox,
    truncate_messages_before_turn,
)
from customized_areal.tpfc.backend_run import _get_raw_messages_with_client
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
    SampleSource,
)
from customized_areal.tree_search.mcts_tree_store import Node

from areal.api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")


@dataclass(frozen=True)
class EpisodeRunResult:
    result: Any
    task_id: str
    raw_messages: list[dict[str, Any]]


def _with_episode_metadata(
    result: Any,
    data: dict[str, Any],
) -> Any:
    task_id = data.get("_backend_run_task_id")
    raw_messages = data.get("_backend_run_raw_messages")
    if isinstance(task_id, str) and isinstance(raw_messages, list):
        return EpisodeRunResult(
            result=result,
            task_id=task_id,
            raw_messages=raw_messages,
        )
    return result


def choose_sample_source(
    mode: SampleSource,
    *,
    branch_probability: float,
    has_candidate: bool,
    random_value: float,
) -> SampleSource:
    if mode == SampleSource.SCRATCH or not has_candidate:
        return SampleSource.SCRATCH
    if mode == SampleSource.BRANCH:
        return SampleSource.BRANCH
    if mode == SampleSource.MIXED and random_value < branch_probability:
        return SampleSource.BRANCH
    return SampleSource.SCRATCH


def _max_entropy(node: Node) -> float:
    stats = node.entropy_stats or {}
    value = stats.get("max_entropy") if isinstance(stats, dict) else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def select_branch_candidate(nodes: list[Node], query_id: str) -> Node | None:
    candidates = [
        node
        for node in nodes
        if node.query_id == query_id
        and node.need_branch
        and bool(node.task_id)
        and bool(node.branch_sandbox_id)
    ]
    if not candidates:
        return None
    return max(candidates, key=_max_entropy)


async def build_branch_task(
    *,
    client: Any,
    account_id: str,
    agent_id: str,
    candidate: Node,
    name: str | None,
) -> str | None:
    if not candidate.task_id or not candidate.branch_sandbox_id:
        return None

    raw_messages = await _get_raw_messages_with_client(client, candidate.task_id)
    prefix = truncate_messages_before_turn(raw_messages, candidate.turn_idx)
    branch_sandbox_id = candidate.branch_sandbox_id

    branch_task_id = await create_task(
        client=client,
        account_id=account_id,
        agent_id=agent_id,
        name=name,
    )
    await bind_sandbox_to_task(
        client,
        sandbox_id=branch_sandbox_id,
        task_id=branch_task_id,
        account_id=account_id,
    )
    await copy_messages_to_task(client, task_id=branch_task_id, messages=prefix)

    return branch_task_id


def _assistant_metadata(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_by_turn: list[dict[str, Any]] = []
    for row in raw_messages:
        if row.get("role") != "assistant":
            continue
        metadata = row.get("metadata") or {}
        metadata_by_turn.append(metadata if isinstance(metadata, dict) else {})
    return metadata_by_turn


def annotate_nodes_from_run(
    nodes: list[Node],
    *,
    task_id: str,
    raw_messages: list[dict[str, Any]],
) -> None:
    """Copy TPFC assistant-message metadata onto Nodes by 1-indexed turn_idx."""
    assistant_meta = _assistant_metadata(raw_messages)
    for node in nodes:
        node.task_id = task_id
        if node.turn_idx < 1:
            continue
        idx = node.turn_idx - 1
        if idx >= len(assistant_meta):
            continue

        metadata = assistant_meta[idx]
        entropy_stats = metadata.get("entropy_stats")
        node.entropy_stats = entropy_stats if isinstance(entropy_stats, dict) else None
        node.need_branch = bool(metadata.get("need_branch"))
        branch_sandbox_id = metadata.get("branch_sandbox_id")
        node.branch_sandbox_id = (
            branch_sandbox_id
            if isinstance(branch_sandbox_id, str) and branch_sandbox_id
            else None
        )


def interactions_dict_to_nodes(interactions: dict[str, Any]) -> list[Node]:
    """Convert dict[str, InteractionWithTokenLogpReward] to list[Node].

    Each interaction becomes one Node representing a single turn.
    """
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    nodes: list[Node] = []

    for turn_idx, (interaction_id, interaction) in enumerate(
        interactions.items(), start=1
    ):
        if not isinstance(interaction, InteractionWithTokenLogpReward):
            logger.warning(
                "Skipping interaction %s (type=%s, expected InteractionWithTokenLogpReward)",
                interaction_id,
                type(interaction).__name__,
            )
            continue
        # When interactions are deserialized from the proxy server (via HTTP),
        # model_response is None but _cache contains the pre-computed tensor
        # dict. Use to_tensor_dict() which checks _cache first.
        resp = interaction.model_response
        if resp is not None:
            seq_tokens = resp.input_tokens + resp.output_tokens

            if (
                interaction.chat_template_type == "concat"
                and interaction.parent is not None
            ):
                from areal.infra.rpc.rtensor import RTensor

                parent_res = RTensor.localize(interaction.parent.to_tensor_dict())
                parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
                parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
                parent_versions = parent_res["versions"].squeeze(0).tolist()
                parent_len = len(parent_logprobs)
                assert parent_len == len(parent_loss_mask) == len(parent_versions)

                if resp.input_len > parent_len:
                    logprobs = (
                        parent_logprobs
                        + [0.0] * (resp.input_len - parent_len)
                        + resp.output_logprobs
                    )
                    loss_mask = (
                        parent_loss_mask
                        + [0] * (resp.input_len - parent_len)
                        + [1] * resp.output_len
                    )
                    versions = (
                        parent_versions
                        + [-1] * (resp.input_len - parent_len)
                        + resp.output_versions
                    )
                else:
                    logger.error(
                        "concat mode: resp.input_len (%d) <= parent_len (%d) — "
                        "expected monotonic growth. Zero-filling prompt context.",
                        resp.input_len,
                        parent_len,
                    )
                    logprobs = [0.0] * resp.input_len + resp.output_logprobs
                    loss_mask = [0] * resp.input_len + [1] * resp.output_len
                    versions = [-1] * resp.input_len + resp.output_versions
            else:
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions

            outcome_reward = (
                interaction.reward if interaction.reward is not None else 0.0
            )

            topk_ids: list[list[int]] = []
            topk_logp: list[list[float]] = []
            if resp.output_top_logprobs is not None:
                for pos_logprobs in resp.output_top_logprobs:
                    ids = []
                    logps = []
                    for token_id, lp in pos_logprobs:
                        ids.append(token_id)
                        logps.append(lp)
                    topk_ids.append(ids)
                    topk_logp.append(logps)
        elif interaction.has_tensor_data:
            # Deserialized from proxy server: _cache is set but model_response
            # is None. Extract fields from the pre-computed tensor dict.
            # to_tensor_dict() returns RTensor-wrapped tensors after proxy
            # deserialization; localize them to real tensors first.
            from areal.infra.rpc.rtensor import RTensor

            td = RTensor.localize(interaction.to_tensor_dict())
            seq_tokens = td["input_ids"].squeeze(0).tolist()
            logprobs = td["logprobs"].squeeze(0).tolist()
            loss_mask = td["loss_mask"].squeeze(0).tolist()
            versions = td["versions"].squeeze(0).tolist()
            outcome_reward = (
                interaction.reward if interaction.reward is not None else 0.0
            )
            topk_ids = []
            topk_logp = []
        else:
            logger.warning(
                "Skipping interaction %s: no tensor data (model_response and _cache are both None)",
                interaction_id,
            )
            continue

        pn_id: str | None = None
        if interaction.parent is not None:
            pn_id = interaction.parent.interaction_id

        node = Node(
            input_ids=seq_tokens,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            turn_idx=turn_idx,
            node_id=interaction_id,
            parent_node_id=pn_id,
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
        )

        nodes.append(node)

    return nodes


def _nodes_to_batched_tensor_dict(
    nodes: list[Node], max_tokens: int = 0, loss_mode: str | None = None
) -> dict[str, Any] | None:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    If max_tokens > 0, each node's sequence is truncated to max_tokens
    from the beginning before conversion.

    Returns None if nodes is empty.
    """
    if not nodes:
        return None

    from customized_areal.tree_search.mcts_tree_store import _node_to_tensor_dict

    from areal.utils.data import concat_padded_tensors

    tensor_dicts = [
        _node_to_tensor_dict(
            node,
            query_id=node.query_id or "",
            node_id=node.node_id,
            max_tokens=max_tokens,
            loss_mode=loss_mode,
        )
        for node in nodes
    ]
    return concat_padded_tensors(tensor_dicts)


def _filter_distill_episode_failure(
    nodes: list[Node], loss_mode: LossMode
) -> list[Node]:
    if loss_mode == LossMode.DISTILL:
        return []
    return nodes


def _set_position_reward_sample_indices(
    nodes: list[Node],
    rewards_by_node_id: dict[str, list[Any]],
) -> list[Any]:
    all_rewards: list[Any] = []
    for sample_index, node in enumerate(nodes):
        for reward in rewards_by_node_id.get(node.node_id, []):
            reward.sample_index = sample_index
            all_rewards.append(reward)
    return all_rewards


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


class TreeSearchGroupedRolloutWorkflow(RolloutWorkflow):
    """GroupedRolloutWorkflow with tree-search cache reuse, tree ops, and checkpoint.

    Wraps the base OpenAIProxyWorkflow and overrides arun_episode to:
    1. Check cache: how many untrained episodes exist for this query?
    2. Generate only the needed fresh episodes (group_size - cached_count)
    3. Convert fresh results to Nodes, load cached episode Nodes
    4. Teacher model reward computation (if loss_mode != GRPO):
       - Diagnose episodes to find turns needing improvement
       - Get teacher logprobs for candidate tokens at each position
       - Build PositionRewardInfo with candidate_token_ids, teacher_logprobs, and rewards
    5. Combine cached + fresh Nodes (total = group_size episodes)
    6. Insert fresh Nodes into tree_store
    7. Compute tree advantages per-episode (if advantage_mode == TREE)
    8. Mark all nodes as trained
    9. Save tree checkpoint (if cache_mode == CROSS_TRAINING)
    10. Return batched tensor dict (with distill weights and position_rewards if distilling)
    """

    _tokenizer_cache: dict[str, Any] = {}
    _tokenizer_lock = asyncio.Lock()

    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        checkpoint_dir: str,
        advantage_mode: AdvantageMode,
        loss_mode: LossMode,
        cache_mode: CacheMode,
        tokenizer_path: str = "",
        max_reasoning_tokens: int = 1000,
        rl_loss_weight: float = 1.0,
        distill_loss_weight: float = 0.005,
        topk_distill: bool = False,
        teacher_provider: str = "external",
        teacher_base_url: str = "http://localhost:8001",
        teacher_backend: str = "openai",
        teacher_model_name: str = "",
        teacher_api_key: str = "",
        teacher_top_k: int = 10,
        teacher_max_retries: int = 3,
        teacher_timeout: float = 300.0,
        teacher_missing_logprob: float = -23.0,
        diagnose_model_name: str = "",
        diagnose_max_tokens: int = 1024,
        diagnose_temperature: float = 0.0,
        diagnose_base_url: str = "",
        diagnose_api_key: str = "",
        strict_distill_json: bool = True,
        max_tokens: int = 0,
        sample_source: SampleSource = SampleSource.SCRATCH,
        branch_probability: float = 0.5,
    ) -> None:
        from customized_areal.tree_search.advantage import TreeAdvantageComputer
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager
        from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore

        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.advantage_mode = advantage_mode
        self.loss_mode = loss_mode
        self.cache_mode = cache_mode
        self.tokenizer_path = tokenizer_path
        self.max_reasoning_tokens = max_reasoning_tokens
        self.rl_loss_weight = rl_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.topk_distill = topk_distill
        self.teacher_provider = teacher_provider
        self.teacher_base_url = teacher_base_url
        self.teacher_backend = teacher_backend
        self.teacher_model_name = teacher_model_name
        self.teacher_api_key = teacher_api_key
        self.teacher_top_k = teacher_top_k
        self.teacher_max_retries = teacher_max_retries
        self.teacher_timeout = teacher_timeout
        self.teacher_missing_logprob = teacher_missing_logprob
        self.diagnose_model_name = diagnose_model_name
        self.diagnose_max_tokens = diagnose_max_tokens
        self.diagnose_temperature = diagnose_temperature
        self.diagnose_base_url = diagnose_base_url
        self.diagnose_api_key = diagnose_api_key
        self.strict_distill_json = strict_distill_json
        self.max_tokens = max_tokens
        self.sample_source = SampleSource(sample_source)
        self.branch_probability = branch_probability

        self.tree_checkpoint_manager = TreeCheckpointManager(checkpoint_dir)

        # Load existing tree checkpoint if present (CROSS_TRAINING mode)
        if self.cache_mode == CacheMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load()
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")
            else:
                self.tree_store = MCTSTreeStore()
        else:
            self.tree_store = MCTSTreeStore()

        self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)

    def _result_to_nodes(
        self, result: Any, query_id: str, group_idx: int
    ) -> list[Node] | None:
        """Convert a single arun_episode result to list[Node]."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        task_id: str | None = None
        raw_messages: list[dict[str, Any]] | None = None
        if isinstance(result, EpisodeRunResult):
            task_id = result.task_id
            raw_messages = result.raw_messages
            result = result.result

        if isinstance(result, dict) and all(
            isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
        ):
            nodes = interactions_dict_to_nodes(result)
        elif (
            isinstance(result, list)
            and result
            and isinstance(result[0], InteractionWithTokenLogpReward)
        ):
            converted = {str(i): v for i, v in enumerate(result)}
            nodes = interactions_dict_to_nodes(converted)
        else:
            return None

        episode_id = (
            f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
            if query_id
            else f"{group_idx}_{uuid.uuid4().hex[:8]}"
        )
        for turn_idx, node in enumerate(nodes, start=1):
            node.episode_id = episode_id
            node.query_id = query_id
            if not node.turn_idx:
                node.turn_idx = turn_idx
        if isinstance(task_id, str) and isinstance(raw_messages, list):
            annotate_nodes_from_run(nodes, task_id=task_id, raw_messages=raw_messages)
        return nodes

    async def _get_tokenizer(self):
        if self.loss_mode == LossMode.GRPO:
            return None
        if not self.tokenizer_path:
            raise ValueError(
                "tokenizer_path is required when tree-search distillation is enabled"
            )
        async with self._tokenizer_lock:
            tokenizer = self._tokenizer_cache.get(self.tokenizer_path)
            if tokenizer is None:
                from areal.utils.hf_utils import load_hf_tokenizer

                tokenizer = load_hf_tokenizer(self.tokenizer_path)
                self._tokenizer_cache[self.tokenizer_path] = tokenizer
            return tokenizer

    async def _setup_distill_provider(self, engine):
        from customized_areal.tree_search.core.teacher_client import (
            TeacherClient,
            TeacherConfig,
        )
        from customized_areal.tree_search.core.teacher_provider import (
            ExternalTeacherProvider,
        )

        if self.teacher_provider == "engine":
            proxy_addr = getattr(engine, "_proxy_gateway_addr", "") or ""
            engine_addrs = getattr(engine, "addresses", None) or []
            admin_api_key = getattr(engine.config, "admin_api_key", "") or ""
            if proxy_addr:
                teacher_base_url = proxy_addr
            elif engine_addrs:
                teacher_base_url = f"http://{engine_addrs[0]}"
            else:
                teacher_base_url = self.teacher_base_url

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
                "Teacher provider=engine, resolved teacher_base_url=%s, "
                "teacher_backend=%s",
                teacher_base_url,
                teacher_backend,
            )

            if not teacher_base_url.startswith(("http://", "https://")):
                raise ValueError(
                    f"teacher_base_url must start with http:// or https://, "
                    f"got: {teacher_base_url!r}"
                )

            config = TeacherConfig(
                teacher_base_url=teacher_base_url,
                teacher_model_name=self.teacher_model_name,
                teacher_api_key=admin_api_key,
                teacher_top_k=self.teacher_top_k,
                teacher_max_retries=self.teacher_max_retries,
                teacher_timeout=self.teacher_timeout,
                teacher_missing_logprob=self.teacher_missing_logprob,
                teacher_backend=teacher_backend,
            )
            client = TeacherClient(config)
        else:
            config = TeacherConfig(
                teacher_base_url=self.teacher_base_url,
                teacher_model_name=self.teacher_model_name,
                teacher_api_key=self.teacher_api_key,
                teacher_top_k=self.teacher_top_k,
                teacher_max_retries=self.teacher_max_retries,
                teacher_timeout=self.teacher_timeout,
                teacher_missing_logprob=self.teacher_missing_logprob,
                teacher_backend=self.teacher_backend,
            )
            client = TeacherClient(config)

        diagnose_model_name = "qwen/qwen3.5-397b-a17b"
        diagnose_api_key = os.environ.get("WORKSPACE_OPENAI_API_KEY")
        diagnose_base_url = os.environ.get("WORKSPACE_OPENAI_API_BASE")
        provider = ExternalTeacherProvider(
            client=client,
            diagnose_model_name=diagnose_model_name,
            diagnose_temperature=0,
            diagnose_base_url=diagnose_base_url,
            diagnose_api_key=diagnose_api_key,
        )

        return provider, client

    async def _prepare_distill_for_episode(
        self,
        nodes: list[Node],
        data: dict[str, Any],
        engine: Any,
        provider: Any,
        tokenizer: Any,
    ) -> tuple[list[Node], dict[str, list[Any]]]:
        from customized_areal.tree_search.core.selected_turn_distill import (
            parse_episode_diagnosis,
            selected_turn_to_position_rewards,
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
            selected = diagnosis.selected_turns
            if selected:
                nodes[-1].guidance = dict(selected)
        if not selected:
            if self.loss_mode == LossMode.DISTILL:
                return [], {}
            return nodes, {}

        async def _run_one_node(node: Node) -> tuple[str, list[Any]] | None:
            guidance = selected.get(node.turn_idx)
            if not guidance:
                return None
            rewards = await selected_turn_to_position_rewards(
                node=node,
                guidance=guidance,
                tokenizer=tokenizer,
                provider=provider,
                sample_index=0,
                topk_distill=self.topk_distill,
                engine=engine,
                teacher_top_k=self.teacher_top_k,
            )
            if not rewards:
                return None
            node.teacher_logp = [reward.teacher_logprobs or [] for reward in rewards]
            node.topk_ids = [reward.candidate_token_ids for reward in rewards]
            return node.node_id, rewards

        # Run the last node first to warm the KV cache, then parallelize
        # the remaining nodes so they benefit from the pre-warmed cache.
        last_result = await _run_one_node(nodes[-1])
        other_results = await asyncio.gather(
            *[_run_one_node(node) for node in nodes[:-1]]
        )

        rewards_by_node_id: dict[str, list[Any]] = {}
        for result in [last_result, *other_results]:
            if result is not None:
                node_id, rewards = result
                rewards_by_node_id[node_id] = rewards
        if self.loss_mode == LossMode.DISTILL and not rewards_by_node_id:
            return [], {}
        return nodes, rewards_by_node_id

    async def _retry_episode(
        self,
        engine,
        data: dict[str, Any],
        group_idx: int,
        max_retries: int = 1,
    ) -> Any:
        """Retry a failed episode until success or max_retries exhausted."""
        for attempt in range(1, max_retries + 1):
            result = await self.workflow.arun_episode(engine, data)
            if not isinstance(result, Exception) and result is not None:
                return result
            logger.warning(
                "Episode %s retry %d/%d %s",
                group_idx,
                attempt,
                max_retries,
                f"failed: {result}"
                if isinstance(result, Exception)
                else "returned None",
            )
            if isinstance(result, Exception):
                logger.warning(
                    "Episode %s retry %d traceback:\n%s",
                    group_idx,
                    attempt,
                    "".join(
                        traceback.format_exception(
                            type(result), result, result.__traceback__
                        )
                    ),
                )
            wait = 2**attempt
            logger.info(
                "Episode %s retry %d — waiting %ds before next attempt",
                group_idx,
                attempt,
                wait,
            )
            await asyncio.sleep(wait)
        logger.error(
            "Episode %s exhausted all %d retries — skipping", group_idx, max_retries
        )
        return None

    async def _run_fresh_episode(
        self,
        engine,
        data: dict[str, Any],
        group_idx: int,
        query_id: str,
    ) -> Any:
        episode_data = dict(data)
        all_query_nodes = self.tree_store.trajectories.get(query_id, [])
        candidate = select_branch_candidate(all_query_nodes, query_id)
        source = choose_sample_source(
            self.sample_source,
            branch_probability=self.branch_probability,
            has_candidate=candidate is not None,
            random_value=random.random(),
        )
        if source == SampleSource.BRANCH and candidate is not None:
            branch_data = dict(episode_data)
            try:
                branch_task_id = await self._prepare_branch_task(branch_data, candidate)
            except Exception as exc:
                logger.warning(
                    "Branch task preparation errored for query_id=%s; "
                    "falling back to scratch: %s",
                    query_id,
                    exc,
                )
                branch_task_id = None
            if branch_task_id:
                branch_data["task_id"] = branch_task_id
                branch_data["seed_messages_already_inserted"] = True
                result = await self._retry_episode(engine, branch_data, group_idx)
                return _with_episode_metadata(result, branch_data)
            logger.warning(
                "Branch task preparation failed for query_id=%s; falling back to scratch",
                query_id,
            )
        result = await self._retry_episode(engine, episode_data, group_idx)
        return _with_episode_metadata(result, episode_data)

    async def _prepare_branch_task(
        self,
        data: dict[str, Any],
        candidate: Node,
    ) -> str | None:
        from customized_areal.tpfc.backend_run import (
            DEFAULT_AGENT_ID,
            DEFAULT_USER_ID,
            _close_db_client,
            _create_shortlived_db_client,
            _resolve_agent_id,
        )

        account_id = str(data.get("user_id") or DEFAULT_USER_ID or "")
        if not account_id:
            logger.warning("Cannot branch TPFC episode without user/account id")
            return None

        client = await _create_shortlived_db_client()
        try:
            agent_id = await _resolve_agent_id(
                client,
                account_id,
                data.get("agent_id") or DEFAULT_AGENT_ID,
            )
            query = data.get("query")
            return await build_branch_task(
                client=client,
                account_id=account_id,
                agent_id=agent_id,
                candidate=candidate,
                name=str(query)[:100] if query else None,
            )
        finally:
            await _close_db_client(client)

    async def _prepare_distill_for_node_groups(
        self,
        node_groups: list[list[Node]],
        data: dict[str, Any],
        engine: Any,
        provider: Any,
        tokenizer: Any,
    ) -> tuple[list[Node], dict[str, list[Any]]]:
        async def _run_one(
            nodes: list[Node],
        ) -> tuple[list[Node], dict[str, list[Any]]]:
            try:
                return await self._prepare_distill_for_episode(
                    nodes=nodes,
                    data=data,
                    engine=engine,
                    provider=provider,
                    tokenizer=tokenizer,
                )
            except Exception:
                logger.exception(
                    "Selected-turn distillation failed for episode_id=%s",
                    nodes[0].episode_id if nodes else "",
                )
                return _filter_distill_episode_failure(nodes, self.loss_mode), {}

        results = await asyncio.gather(*[_run_one(nodes) for nodes in node_groups])

        prepared_nodes: list[Node] = []
        rewards_by_node_id: dict[str, list[Any]] = {}
        for episode_nodes, episode_rewards in results:
            prepared_nodes.extend(episode_nodes)
            rewards_by_node_id.update(episode_rewards)
        return prepared_nodes, rewards_by_node_id

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return await self._arun_episode_impl(engine, data)
        except Exception:
            logger.exception(
                "TreeSearchGroupedWorkflow.arun_episode failed for query_id=%s",
                data.get("query_id", ""),
            )
            return None

    async def _arun_episode_impl(
        self, engine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        query_id = data.get("query_id") or ""

        # 1. Check cache
        cached_count = (
            self.tree_store.get_untrained_episode_count(query_id) if query_id else 0
        )
        need_gen = max(0, self.group_size - cached_count)

        logger.info(
            "TreeSearchGroupedWorkflow: query_id=%s, group_size=%d, "
            "cached=%d, need_gen=%d",
            query_id,
            self.group_size,
            cached_count,
            need_gen,
        )

        # 2. Generate fresh episodes if needed
        fresh_nodes: list[Node] = []
        if need_gen > 0:
            results = await asyncio.gather(
                *[
                    self._run_fresh_episode(engine, data, group_idx, query_id)
                    for group_idx in range(need_gen)
                ],
                return_exceptions=True,
            )

            for group_idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Episode %d unrecoverable: %s", group_idx, result)
                    continue
                if result is None:
                    continue
                nodes = self._result_to_nodes(result, query_id, group_idx)
                if nodes:
                    fresh_nodes.extend(nodes)

        # 3. Load cached nodes
        cached_nodes: list[Node] = []
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_untrained_episodes(
                query_id, cached_count
            )
            # Reset versions to 0 so decoupled PPO treats cached rollouts
            # as coming from the current behavior policy
            # for node in cached_nodes:
            #     node.versions = [0 if m == 1 else -1 for m in node.loss_mask]

        provider_client = None
        try:
            # 4. Insert fresh nodes into tree
            if fresh_nodes:
                self.tree_store.insert_batch(fresh_nodes)

            # 5. Combine
            all_nodes = fresh_nodes + cached_nodes

            if not all_nodes:
                return None

            if self.loss_mode != LossMode.GRPO:
                tokenizer = await self._get_tokenizer()
                provider, provider_client = await self._setup_distill_provider(engine)
                all_nodes, _ = await self._prepare_distill_for_node_groups(
                    _group_nodes_by_episode(all_nodes),
                    data,
                    engine,
                    provider,
                    tokenizer,
                )

            # 6. Compute tree advantages
            if self.advantage_mode == AdvantageMode.TREE:
                self.tree_advantage_computer.compute(all_nodes)

            # 7. Mark all nodes as trained
            for node in all_nodes:
                if node.node_id:
                    self.tree_store.set_trained(node.node_id, True)

            # 8. Save tree checkpoint
            self.tree_checkpoint_manager.save_query(self.tree_store, query_id)

            # 9. Convert to batched tensor dict
            result_dict = _nodes_to_batched_tensor_dict(
                all_nodes, max_tokens=self.max_tokens,
                loss_mode=self.loss_mode.value,
            )

            return result_dict if result_dict else None
        finally:
            if provider_client is not None:
                await provider_client.close()


def _group_nodes_by_episode(nodes: list[Node]) -> list[list[Node]]:
    grouped: dict[str, list[Node]] = {}
    fallback_index = 0
    for node in nodes:
        episode_id = node.episode_id
        if not episode_id:
            episode_id = f"__missing_episode_{fallback_index}"
            fallback_index += 1
        grouped.setdefault(episode_id, []).append(node)
    return list(grouped.values())

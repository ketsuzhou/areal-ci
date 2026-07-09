# customized_areal/tree_search/core/customized_grouped_workflow.py
"""Tree-search-aware grouped rollout workflow with cache reuse and teacher distillation.

Consolidates the functionality of QueryIDProxyWorkflow,
TreeSearchGroupedRolloutWorkflow, and TreeSearchWorkflowExecutor into a single
``RolloutWorkflow`` subclass.

Architecture
~~~~~~~~~~~~
The main entry point is ``arun_episode``, which dispatches to either
``_arun_episode_fixed`` or ``_arun_episode_dynamic`` depending on
``dynamic_group_size``.  Both paths converge on ``_finalize_episode`` for
shared post-processing.

Fixed group_size path (``_arun_episode_fixed``):
  1. Query the tree_store for cached (untrained) episodes.
  2. Generate ``group_size - cached_count`` fresh episodes in parallel,
     each generated from scratch.  (Branching now lives only in the
     env-dispatch runner model; the wired grouped workflow no longer
     branches from cached nodes.)
  3. Combine cached + fresh nodes.

Dynamic group_size path (``_arun_episode_dynamic``):
  1. Same cache lookup and initial generation as fixed, but starting from
     ``initial_group_size`` instead of ``group_size``.
  2. Iteratively sample one more episode, recompute uncertainty U(q), and
     stop when U(q) falls below ``uncertainty_threshold`` or
     ``max_group_size`` is reached.  A consecutive-failure circuit breaker
     prevents infinite loops.

Shared finalization (``_finalize_episode``):
  4. Zero-variance discard: if all episodes have identical reward, insert
     fresh nodes, mark them discarded, save checkpoint, and return None.
  5. Insert fresh nodes into the tree_store.
  6. If ``loss_mode != GRPO``: run selected-turn distillation — diagnose
     the episode, identify turns needing improvement, gather teacher
     logprobs, and build per-position reward info.
  7. If ``advantage_mode == TREE``: compute GRPO-normalized tree advantages
     across episodes.
  8. Mark all nodes as trained, save the tree checkpoint, and convert to a
     batched tensor dict.

Helper functions
~~~~~~~~~~~~~~~~
- ``choose_sample_source`` — probabilistic selection between SCRATCH / BRANCH / MIXED.
- ``select_branch_candidate`` — pick the highest-entropy node eligible for branching.
- ``interactions_dict_to_nodes`` — convert inference-engine interactions to ``Node``
  objects, handling both live ``model_response`` and proxy-deserialized tensor caches.
- ``annotate_nodes_from_run`` — stamp TPFC assistant metadata (entropy, branch info)
  onto nodes by turn index.
- ``_input_ids_to_messages`` — heuristic token-ID → message-list conversion used by
  the diagnosis step.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass
import re
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
    SampleSource,
)
from customized_areal.tree_search.core.tree_store import Node
from customized_areal.tree_search.core.uncertainty import should_discard_query

from areal.api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")

_FRESH_QUERY_SELECT_LIMIT = 100


def _wrap_leaf_super(nodes: list[Node], *, super_id: str = "") -> SuperNode:
    """Wrap a single-agent episode's Nodes in one leaf SuperNode.

    Used by the single-agent path to keep the unified data model exercised
    while the multi-agent coordinator (Phase 1b/2) is not yet wired. The leaf
    SuperNode has no DAG edges and no comm events; backup walks parent_node_id
    inside it exactly as before.
    """
    return SuperNode(
        node_id=super_id or str(uuid.uuid4()),
        agent_id="",
        issue_id="",
        task_id=nodes[0].task_id if nodes else "",
        nodes=list(nodes),
    )


@dataclass(frozen=True)
class EpisodeRunResult:
    result: Any
    task_id: str
    raw_messages: list[dict[str, Any]]
    # node_id of the branch-point node this episode was branched from, when the
    # episode is a branch (None for scratch episodes). Used to link the
    # episode's first turn into the shared tree for MCTS root-ward backup.
    branch_point_node_id: str | None = None


def _with_episode_metadata(
    result: Any,
    data: dict[str, Any],
) -> Any:
    task_id = data.get("_backend_run_task_id")
    raw_messages = data.get("_backend_run_raw_messages")
    branch_point_node_id = data.get("_branch_point_node_id")
    if isinstance(task_id, str) and isinstance(raw_messages, list):
        return EpisodeRunResult(
            result=result,
            task_id=task_id,
            raw_messages=raw_messages,
            branch_point_node_id=(
                branch_point_node_id
                if isinstance(branch_point_node_id, str) and branch_point_node_id
                else None
            ),
        )
    return result


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


def _critic_value(node: Node, tree_store: Any | None) -> tuple[float, bool]:
    """Return ``(v(s_t), has_value)`` for a node from the critic.

    Prefers the tree store's recorded value (authoritative ``has_value``);
    falls back to ``Node.value`` where a non-zero value is treated as present.
    A missing value (``has_value is False``) makes the caller bypass the TD gate.
    """
    node_id = getattr(node, "node_id", "") or ""
    if tree_store is not None and node_id and tree_store.has_value(node_id):
        return float(tree_store.get_value(node_id)), True
    v = float(getattr(node, "value", 0.0) or 0.0)
    return v, v != 0.0


def select_branch_candidate(
    nodes: list[Node],
    query_id: str,
    tree_store: Any | None = None,
    td_threshold: float = 0.0,
    gamma: float = 1.0,
) -> Node | None:
    """Pick the best branch candidate, optionally gated by critic TD-error.

    Candidates are ``need_branch`` nodes (with a task) for this query.
    When ``td_threshold > 0`` and a tree store is supplied, a candidate is kept
    only if its critic TD-error magnitude meets the threshold::

        |delta_t| = |r_t + gamma * v(s_{t+1}) - v(s_t)|

    computed from critic values only, where ``v(s_{t+1})`` is the candidate's
    successor turn in the same episode (terminal: ``r_t = outcome_reward`` and
    ``v(s_{t+1}) = 0``). Candidates whose own critic value is unavailable bypass
    the gate (entropy-only fallback), so disabling the critic degrades to the
    previous entropy-only behavior. Surviving candidates are ranked by entropy.
    """
    candidates = [
        node
        for node in nodes
        if node.query_id == query_id and node.need_branch and bool(node.task_id)
    ]
    if not candidates:
        return None
    if tree_store is None or td_threshold <= 0.0:
        return max(candidates, key=_max_entropy)

    # Successor lookup over all query nodes (the successor need not be a
    # branch candidate itself).
    by_turn: dict[tuple[str, int], Node] = {}
    for n in nodes:
        if n.query_id == query_id and n.episode_id:
            by_turn[(n.episode_id, getattr(n, "turn_idx", 0))] = n

    gated: list[Node] = []
    for node in candidates:
        v_t, has_v = _critic_value(node, tree_store)
        if not has_v:
            # No critic value -> cannot gate; keep as entropy-only fallback.
            gated.append(node)
            continue
        successor = by_turn.get((node.episode_id, getattr(node, "turn_idx", 0) + 1))
        if successor is not None:
            r_t = 0.0
            v_next, _ = _critic_value(successor, tree_store)
        else:
            r_t = float(getattr(node, "outcome_reward", 0.0) or 0.0)
            v_next = 0.0
        delta = abs(r_t + gamma * v_next - v_t)
        if delta >= td_threshold:
            gated.append(node)
    if not gated:
        return None
    return max(gated, key=_max_entropy)


def _assistant_metadata(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_by_turn: list[dict[str, Any]] = []
    for row in raw_messages:
        if row.get("role") != "assistant":
            continue
        metadata = row.get("metadata") or {}
        metadata_by_turn.append(metadata if isinstance(metadata, dict) else {})
    return metadata_by_turn


def _topk_ids_from_metadata(
    metadata: dict[str, Any],
) -> list[list[int]] | None:
    top_logprobs = metadata.get("top_logprobs")
    if not isinstance(top_logprobs, list):
        return None

    topk_ids: list[list[int]] = []
    for pos_entry in top_logprobs:
        if not isinstance(pos_entry, list):
            return None
        pos_ids: list[int] = []
        for alt in pos_entry:
            if not isinstance(alt, dict):
                return None
            token_id = alt.get("token_id")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                return None
            pos_ids.append(token_id)
        topk_ids.append(pos_ids)
    return topk_ids


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
        env_id = metadata.get("env_id")
        node.env_id = env_id if isinstance(env_id, str) and env_id else None
        if node.topk_ids is None:
            topk_ids = _topk_ids_from_metadata(metadata)
            if topk_ids is not None:
                node.topk_ids = topk_ids


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

    from customized_areal.tree_search.core.tree_store import _node_to_tensor_dict

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
    """Grouped rollout workflow with tree-search cache reuse and teacher distillation.

    Wraps a base ``RolloutWorkflow`` and overrides ``arun_episode`` to produce
    ``group_size`` episodes per query, reusing cached (untrained) episodes from
    the tree_store and generating only the deficit fresh episodes.

    Two sampling modes:
      - **Fixed** (default): generate exactly ``group_size - cached_count`` episodes.
      - **Dynamic** (``dynamic_group_size=True``): start from ``initial_group_size``,
        then iteratively add episodes until uncertainty drops below threshold
        or ``max_group_size`` is reached.

    Fresh episodes are always generated from scratch; the wired workflow no
    longer branches from cached nodes (branching lives in the env-dispatch
    runner model).

    When ``loss_mode`` is DISTILL or BOTH, a teacher model diagnoses each
    episode to identify weak turns, then provides logprobs for distillation
    targets at those positions.

    Checkpointing is managed per-query via ``TreeCheckpointManager`` and is
    always performed (not gated on ``cache_mode``).  ``cache_mode`` only
    controls whether the checkpoint is *loaded* at init time.
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
        teacher_max_concurrency: int = 4,
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
        dynamic_group_size: bool = False,
        initial_group_size: int = 0,
        max_group_size: int = 64,
        uncertainty_threshold: float = 0.05,
        reward_type: str = "binary",
        distill_kl_mode: str = "reverse_kl",
        max_distill_tokens: int = 0,
        use_fresh_query: bool = False,
        fresh_query_table: str = "",
        enable_generative_critic: bool = False,
        critic_gamma: float = 1.0,
        critic_lambda: float = 0.95,
        critic_avg_success_rate: float = 0.29,
        critic_score_max: int = 10,
        critic_max_new_tokens: int = 1024,
        critic_temperature: float = 0.0,
        critic_target_scale: float = 1.0,
        critic_mc_weight: float = 1.0,
        critic_td_n_steps: int = 1,
        critic_mc_adaptive: bool = False,
        critic_mc_c: float = 4.0,
        hybrid_mc_min_visits: int = 5,
        hybrid_critic_var_floor: float = 1e-3,
        hybrid_critic_error_var: float = 0.05,
        branch_td_threshold: float = 0.0,
        enable_judge_process_reward: bool = False,
        judge_process_reward_beta: float = 0.2,
        judge_model_name: str = "",
        judge_max_concurrency: int = 4,
        multica_dag_enabled: bool = False,
        multica_dag_client=None,
    ) -> None:
        from customized_areal.tree_search.core.advantage import TreeAdvantageComputer
        from customized_areal.tree_search.core.checkpoint import TreeCheckpointManager
        from customized_areal.tree_search.core.tree_store import MCTSTreeStore

        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.advantage_mode = advantage_mode
        self.enable_generative_critic = enable_generative_critic
        self.critic_gamma = critic_gamma
        self.critic_lambda = critic_lambda
        self.critic_avg_success_rate = critic_avg_success_rate
        self.critic_score_max = critic_score_max
        self.critic_max_new_tokens = critic_max_new_tokens
        self.critic_temperature = critic_temperature
        self.critic_target_scale = critic_target_scale
        self.critic_mc_weight = critic_mc_weight
        self.critic_td_n_steps = critic_td_n_steps
        self.critic_mc_adaptive = critic_mc_adaptive
        self.critic_mc_c = critic_mc_c
        # Variance-aware hybrid GAE + TD-gated branching knobs.
        self.hybrid_mc_min_visits = hybrid_mc_min_visits
        self.hybrid_critic_var_floor = hybrid_critic_var_floor
        self.hybrid_critic_error_var = hybrid_critic_error_var
        self.branch_td_threshold = branch_td_threshold
        # LLM-judge process-reward shaping.
        self.enable_judge_process_reward = enable_judge_process_reward
        self.judge_process_reward_beta = judge_process_reward_beta
        self.judge_model_name = judge_model_name
        self.judge_max_concurrency = judge_max_concurrency
        # Episodes already scored by the judge (cache key = episode_id) so a
        # shared prefix accumulates one score per distinct episode, not per
        # training iteration.
        self._judged_episodes: set[str] = set()
        # Resolved blend weight passed to ``compute_critic_targets``: either the
        # fixed float ``critic_mc_weight`` or an adaptive controller. The
        # controller is stateful and persists across rollouts so its critic-error
        # EMA can be updated if/when training-side feedback is wired in.
        if critic_mc_adaptive:
            from customized_areal.tree_search.training.losses.critic import (
                AdaptiveMCWeight,
            )

            self._mc_mixer: Any = AdaptiveMCWeight(c=critic_mc_c)
        else:
            self._mc_mixer = critic_mc_weight
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
        self.teacher_max_concurrency = teacher_max_concurrency
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
        self.dynamic_group_size = dynamic_group_size
        self.initial_group_size = (
            initial_group_size if initial_group_size > 0 else group_size
        )
        self.max_group_size = max_group_size
        self.uncertainty_threshold = uncertainty_threshold
        self.reward_type = reward_type
        self.distill_kl_mode = distill_kl_mode
        self.max_distill_tokens = max_distill_tokens or max_tokens
        self.use_fresh_query = use_fresh_query
        self.fresh_query_table = fresh_query_table or os.environ.get(
            "FRESH_QUERY_TABLE", ""
        )
        if self.use_fresh_query and not self.fresh_query_table:
            raise ValueError(
                "fresh_query_table must be set when use_fresh_query=True "
                "(or set FRESH_QUERY_TABLE)"
            )
        if dynamic_group_size:
            if self.initial_group_size < 1:
                raise ValueError(
                    f"initial_group_size must be >= 1, got {self.initial_group_size}"
                )
            if self.max_group_size < self.initial_group_size:
                raise ValueError(
                    "max_group_size must be >= initial_group_size, "
                    f"got max_group_size={self.max_group_size}, "
                    f"initial_group_size={self.initial_group_size}"
                )
            if self.uncertainty_threshold < 0:
                raise ValueError(
                    "uncertainty_threshold must be non-negative, "
                    f"got {self.uncertainty_threshold}"
                )
            if self.reward_type not in ("binary", "continuous"):
                raise ValueError(
                    "reward_type must be 'binary' or 'continuous', "
                    f"got {self.reward_type!r}"
                )
            self.group_size = self.initial_group_size

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
        from customized_areal.tree_search.core.advantage import GAEAdvantageComputer

        self.gae_advantage_computer = GAEAdvantageComputer(
            self.tree_store,
            gamma=self.critic_gamma,
            lam=self.critic_lambda,
            judge_beta=(
                self.judge_process_reward_beta
                if self.enable_judge_process_reward
                else 0.0
            ),
            judge_score_max=self.critic_score_max,
        )
        from customized_areal.tree_search.core.advantage import (
            HybridGAEAdvantageComputer,
        )

        # When the adaptive MC mixer is active, let HybridGAE's critic-side
        # variance track the live critic-MSE EMA (falls back to the static prior
        # until the EMA has been fed).
        critic_error_var_fn = getattr(self._mc_mixer, "live_critic_error_var", None)
        self.hybrid_gae_advantage_computer = HybridGAEAdvantageComputer(
            self.tree_store,
            gamma=self.critic_gamma,
            lam=self.critic_lambda,
            judge_beta=(
                self.judge_process_reward_beta
                if self.enable_judge_process_reward
                else 0.0
            ),
            judge_score_max=self.critic_score_max,
            mc_min_visits=self.hybrid_mc_min_visits,
            critic_var_floor=self.hybrid_critic_var_floor,
            critic_error_var=self.hybrid_critic_error_var,
            critic_error_var_fn=critic_error_var_fn,
        )
        # Lazily constructed on first use (needs the tokenizer).
        self._critic_value_client = None
        # Multica DAG rollout (Phase 1b/2). Declared now so callers can pass
        # them, but the single-agent path runs unchanged until the coordinator
        # dispatch is wired.
        self._multica_dag_enabled = multica_dag_enabled
        self._multica_dag_client = multica_dag_client
        self._coordinator = None  # Phase 1b/2 wires TeamRolloutCoordinator

    async def _load_fresh_query_data(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        from customized_areal.db_service import DBConnection

        train_id = os.environ.get("TRAIN_ID", "")
        if not train_id:
            raise ValueError("TRAIN_ID must be set when use_fresh_query=True")

        client = await DBConnection().get_client()
        select_query = (
            client.table(self.fresh_query_table)
            .select("query_id,query,gold_answer,evaluation_rubric,used4train")
            .limit(_FRESH_QUERY_SELECT_LIMIT)
        )
        select_query = _apply_train_id_not_contains_filter(
            select_query,
            train_id=train_id,
        )
        result = await select_query.execute()
        rows = getattr(result, "data", None)
        if not isinstance(rows, list):
            logger.warning(
                "Fresh query table %s returned non-list data; skipping",
                self.fresh_query_table,
            )
            return None

        for row in rows:
            if not isinstance(row, dict):
                continue
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id:
                continue
            used4train = _normalize_used4train(row.get("used4train"))
            if train_id in used4train:
                continue

            next_used4train = [*used4train, train_id]
            update_query = (
                client.table(self.fresh_query_table)
                .update({"used4train": next_used4train})
                .eq("query_id", query_id)
            )
            claim_result = await _execute_fresh_query_claim_update(
                update_query,
                train_id=train_id,
            )
            if _affected_row_count(claim_result) < 1:
                logger.info(
                    "Fresh query claim lost race for query_id=%s train_id=%s; retrying",
                    query_id,
                    train_id,
                )
                continue

            logger.info(
                "Claimed fresh query query_id=%s for train_id=%s",
                query_id,
                train_id,
            )
            claimed_row = dict(row)
            claimed_row["used4train"] = next_used4train
            return _apply_fresh_query_row(data, claimed_row)

        logger.warning(
            "No eligible fresh query rows found in table=%s for train_id=%s",
            self.fresh_query_table,
            train_id,
        )
        return None

    def _result_to_nodes(
        self, result: Any, query_id: str, group_idx: int
    ) -> list[Node] | list[SuperNode] | None:
        """Convert a single arun_episode result to nodes.

        Single-agent path: a dict/list of ``InteractionWithTokenLogpReward`` ->
        ``list[Node]``. Multica multi-agent path: a dict carrying
        ``"execution_dag"`` (an ``ExecutionDAG[SuperNode]`` assembled by
        ``MultiAgentEnvDispatchWorkflow``) -> ``list[SuperNode]``, preserved (not
        flattened to ``Node``) so the multi-segment edge structure survives to
        ``_finalize_episode``. Returns ``None`` for an unparseable result.
        """
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        task_id: str | None = None
        raw_messages: list[dict[str, Any]] | None = None
        branch_point_node_id: str | None = None
        if isinstance(result, EpisodeRunResult):
            task_id = result.task_id
            raw_messages = result.raw_messages
            branch_point_node_id = result.branch_point_node_id
            result = result.result

        # Multica multi-agent path: the base workflow returned an assembled
        # ExecutionDAG[SuperNode] (one SuperNode per segment). Preserve the
        # SuperNodes - do NOT flatten to Nodes - so the multi-segment edge
        # structure survives to _finalize_episode. Each SuperNode is stamped
        # with query_id/group_idx (via metadata; SuperNode has no such fields)
        # and made self-contained: incoming_edges/outgoing_edges are populated
        # from the DAG, mirroring the legacy assemble() path.
        if isinstance(result, dict) and "execution_dag" in result:
            edag = result["execution_dag"]
            super_nodes = edag.topological_order()
            for sn in super_nodes:
                sn.incoming_edges = tuple(
                    (e.src, e.type) for e in edag.edges if e.dst == sn.node_id
                )
                sn.outgoing_edges = tuple(
                    (e.dst, e.type) for e in edag.edges if e.src == sn.node_id
                )
                sn.metadata["query_id"] = query_id
                sn.metadata["group_idx"] = group_idx
            return super_nodes or None

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
        # Link a branch episode's first turn to its branch-point node so the
        # MCTS root-ward backup aggregates returns at the shared prefix.
        if branch_point_node_id and nodes:
            nodes[0].parent_node_id = branch_point_node_id
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

    async def _get_tokenizer_unconditional(self):
        """Load the tokenizer regardless of loss_mode (needed by the critic)."""
        if not self.tokenizer_path:
            raise ValueError(
                "tokenizer_path is required when enable_generative_critic=True"
            )
        async with self._tokenizer_lock:
            tokenizer = self._tokenizer_cache.get(self.tokenizer_path)
            if tokenizer is None:
                from areal.utils.hf_utils import load_hf_tokenizer

                tokenizer = load_hf_tokenizer(self.tokenizer_path)
                self._tokenizer_cache[self.tokenizer_path] = tokenizer
            return tokenizer

    async def _annotate_judge_process_rewards(
        self, provider, all_nodes, data, tokenizer
    ) -> None:
        """Score each episode with the LLM judge and store per-node credit.

        For every (uncached) episode the judge sees the full trajectory plus the
        gold answer and assigns each turn an integer credit in
        ``[0, critic_score_max]``. Scores accumulate in the tree store keyed by
        ``node_id`` -- a shared prefix node receives one score per distinct
        episode that traverses it. Best-effort: a judge failure logs and skips
        that episode, leaving the sparse reward intact (graceful fallback).
        """
        from customized_areal.tree_search.distilling.teacher_client import (
            TeacherServiceError,
        )

        gold_answer = str(data.get("answer", ""))
        episodes = _group_nodes_by_episode(all_nodes)
        sem = asyncio.Semaphore(max(1, self.judge_max_concurrency))

        async def _score_one(nodes: list[Node]) -> None:
            if not nodes:
                return
            episode_id = nodes[0].episode_id or ""
            # Cache per episode so a shared prefix is not re-judged across
            # training iterations (it still gets one score per distinct episode).
            if episode_id and episode_id in self._judged_episodes:
                return
            ordered = sorted(nodes, key=lambda n: getattr(n, "turn_idx", 0))
            conversation = _input_ids_to_messages(ordered[-1].input_ids, tokenizer)
            try:
                async with sem:
                    scores = await provider.score_episode(
                        conversation,
                        gold_answer,
                        score_max=self.critic_score_max,
                        model_name=self.judge_model_name or None,
                    )
            except (TeacherServiceError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    "Judge scoring failed for episode_id=%s; using sparse reward: %s",
                    episode_id,
                    exc,
                )
                return
            if not scores:
                return
            for node in ordered:
                score = scores.get(int(getattr(node, "turn_idx", 0)))
                if score is None or not node.node_id:
                    continue
                self.tree_store.add_judge_score(node.node_id, float(score))
            if episode_id:
                self._judged_episodes.add(episode_id)

        await asyncio.gather(*[_score_one(nodes) for nodes in episodes])

    async def _annotate_critic_values(self, engine, all_nodes) -> None:
        """Compute generative-critic state values v_phi(s_t) onto Node.value.

        Values are computed per episode (so the critic sees the correct partial
        conversation through each turn) using the actor's shared inference
        engine. Stored on both the Node and the tree store.
        """
        from customized_areal.tree_search.core.critic_value_client import (
            CriticValueClient,
        )

        if self._critic_value_client is None:
            tokenizer = await self._get_tokenizer_unconditional()
            self._critic_value_client = CriticValueClient(
                tokenizer,
                score_max=self.critic_score_max,
                avg_success_rate=self.critic_avg_success_rate,
                max_new_tokens=self.critic_max_new_tokens,
                temperature=self.critic_temperature,
            )
        for episode_nodes in _group_nodes_by_episode(all_nodes):
            await self._critic_value_client.annotate_episode(
                engine, episode_nodes, self.tree_store
            )

    def _attach_critic_train_data(self, result_dict, all_nodes) -> None:
        """Attach critic soft-regression training data to the batch.

        Built from the same critic prompts used at rollout. The regression
        target unifies the MCTS Monte-Carlo ``q_value`` with an n-step
        bootstrapped TD return via :func:`compute_critic_targets`; the blend is
        controlled by ``self._mc_mixer`` (a fixed weight or an adaptive,
        visit-count-driven controller). With the defaults
        (``critic_mc_weight=1.0``) this reproduces the previous pure-MCTS target.
        Stored as a Python object that the patched ``_ppo_update`` pops before
        tensor ops (mirrors position_rewards). Best-effort: never raise into the
        rollout path.

        TODO(agent): close the adaptive loop by feeding the training-side critic
        MSE back into ``self._mc_mixer.update_critic_error(...)``. The critic loss
        is computed in ``run_critic_regression_step`` (training engine), which is
        a different component than this rollout-side target builder, so the
        feedback needs cross-boundary plumbing (shared stat / RPC). Until then the
        adaptive weight degenerates to a pure visit-count rule.
        """
        try:
            if self._critic_value_client is None or not all_nodes:
                return
            from customized_areal.tree_search.training.losses.critic import (
                build_critic_training_batch,
                compute_critic_targets,
            )

            nodes = list(all_nodes)
            # Unified target in [0, 1]; pass target_scale=1.0 to build_* since
            # compute_critic_targets already applied scaling and clamping.
            targets = compute_critic_targets(
                nodes,
                tree_store=self.tree_store,
                mc_weight=self._mc_mixer,
                n_steps=self.critic_td_n_steps,
                gamma=self.critic_gamma,
                target_scale=self.critic_target_scale,
                judge_beta=(
                    self.judge_process_reward_beta
                    if self.enable_judge_process_reward
                    else 0.0
                ),
                judge_score_max=self.critic_score_max,
            )
            critic_batch = build_critic_training_batch(
                self._critic_value_client,
                nodes,
                targets=targets,
                tree_store=self.tree_store,
                target_scale=1.0,
                score_max=self.critic_score_max,
            )
            result_dict["critic_train_data"] = critic_batch
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to attach critic_train_data (skipping critic step): %s", exc
            )

    async def _setup_distill_provider(self, engine, tokenizer=None):
        from customized_areal.tree_search.distilling.diagnose_provider import (
            ExternalDiagnoseProvider,
        )
        from customized_areal.tree_search.distilling.teacher_client import (
            TeacherClient,
            TeacherConfig,
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
                teacher_max_concurrency=self.teacher_max_concurrency,
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
                teacher_max_concurrency=self.teacher_max_concurrency,
            )
            client = TeacherClient(config)

        diagnose_model_name = self.diagnose_model_name or "qwen/qwen3.7-max"
        diagnose_api_key = (
            self.diagnose_api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("WORKSPACE_OPENAI_API_KEY", "")
        )
        diagnose_base_url = (
            self.diagnose_base_url
            or os.environ.get("OPENROUTER_BASE_URL", "")
            or os.environ.get("WORKSPACE_OPENAI_API_BASE", "")
        )
        provider = ExternalDiagnoseProvider(
            client=client,
            diagnose_model_name=diagnose_model_name,
            diagnose_temperature=self.diagnose_temperature,
            diagnose_max_tokens=self.diagnose_max_tokens,
            diagnose_base_url=diagnose_base_url,
            diagnose_api_key=diagnose_api_key,
            tokenizer=tokenizer,
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
                    topk_distill=self.topk_distill,
                    engine=engine,
                    teacher_top_k=self.teacher_top_k,
                    max_distill_tokens=self.max_distill_tokens,
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
        other_results = await asyncio.gather(
            *[_run_one_node(node) for node in nodes[:-1]]
        )

        rewards_by_node_id: dict[str, list[Any]] = {}
        for result in [last_result, *other_results]:
            if result is not None:
                node_id, rewards = result
                rewards_by_node_id[node_id] = rewards
        if not rewards_by_node_id:
            return _filter_distill_episode_failure(nodes, self.loss_mode), {}
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
            try:
                result = await self.workflow.arun_episode(engine, data)
            except Exception as exc:
                result = exc
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
        # Branching now lives only in the env-dispatch runner model; the wired
        # grouped workflow always generates fresh episodes from scratch.
        episode_data = dict(data)
        result = await self._retry_episode(engine, episode_data, group_idx)
        return _with_episode_metadata(result, episode_data)

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
        if self.use_fresh_query:
            fresh_data = await self._load_fresh_query_data(data)
            if fresh_data is None:
                return None
            data = fresh_data

        query_id = data.get("query_id") or ""
        try:
            if self.dynamic_group_size:
                return await self._arun_episode_dynamic(engine, data, query_id)
            return await self._arun_episode_fixed(engine, data, query_id)
        except Exception:
            logger.exception(
                "TreeSearchGroupedWorkflow.arun_episode failed for query_id=%s",
                query_id,
            )
            return None

    async def _arun_episode_fixed(
        self, engine, data: dict[str, Any], query_id: str
    ) -> dict[str, Any] | None:
        """Original fixed group_size logic (with zero-variance discard)."""
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

        fresh_nodes: list[Node] = []
        if need_gen > 0 and self.loss_mode != LossMode.DISTILL:
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

        cached_nodes: list[Node] = []
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_untrained_episodes(
                query_id, cached_count
            )

        return await self._finalize_episode(
            fresh_nodes, cached_nodes, engine, data, query_id
        )

    async def _arun_episode_dynamic(
        self, engine, data: dict[str, Any], query_id: str
    ) -> dict[str, Any] | None:
        """Dynamic group_size: iterative sampling with uncertainty threshold."""

        # 1. Initial round
        cached_count = (
            self.tree_store.get_untrained_episode_count(query_id) if query_id else 0
        )
        need_gen = max(0, self.initial_group_size - cached_count)

        logger.info(
            "TreeSearchGroupedWorkflow [dynamic]: query_id=%s, "
            "initial_group_size=%d, cached=%d, need_gen=%d, max=%d",
            query_id,
            self.initial_group_size,
            cached_count,
            need_gen,
            self.max_group_size,
        )

        fresh_nodes: list[Node] = []
        if need_gen > 0 and self.loss_mode != LossMode.DISTILL:
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

        cached_nodes: list[Node] = []
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_untrained_episodes(
                query_id, cached_count
            )

        # 2. Compute initial uncertainty
        all_nodes = fresh_nodes + cached_nodes
        uncertainty = self._compute_uncertainty_for_nodes(all_nodes)

        logger.info(
            "TreeSearchGroupedWorkflow [dynamic]: query_id=%s, "
            "initial U=%.6f, episodes=%d",
            query_id,
            uncertainty,
            self._count_episodes(all_nodes),
        )

        # 3. Iterative sampling loop
        next_group_idx = need_gen
        consecutive_failed_additions = 0
        max_failed_additions = max(3, self.max_group_size)
        while (
            self.loss_mode != LossMode.DISTILL
            and self._count_episodes(all_nodes) < self.max_group_size
        ):
            if uncertainty <= self.uncertainty_threshold:
                logger.info(
                    "TreeSearchGroupedWorkflow [dynamic]: query_id=%s "
                    "converged (U=%.6f <= threshold=%.6f), episodes=%d",
                    query_id,
                    uncertainty,
                    self.uncertainty_threshold,
                    self._count_episodes(all_nodes),
                )
                break

            # Generate one more episode
            result = await self._run_fresh_episode(
                engine, data, next_group_idx, query_id
            )
            next_group_idx += 1

            if isinstance(result, Exception) or result is None:
                consecutive_failed_additions += 1
                logger.warning(
                    "TreeSearchGroupedWorkflow [dynamic]: query_id=%s "
                    "additional episode failed (%d/%d), skipping",
                    query_id,
                    consecutive_failed_additions,
                    max_failed_additions,
                )
                if consecutive_failed_additions >= max_failed_additions:
                    logger.warning(
                        "TreeSearchGroupedWorkflow [dynamic]: query_id=%s "
                        "stopping after %d consecutive failed additions",
                        query_id,
                        consecutive_failed_additions,
                    )
                    break
                continue

            nodes = self._result_to_nodes(result, query_id, next_group_idx - 1)
            if nodes:
                fresh_nodes.extend(nodes)
                all_nodes = fresh_nodes + cached_nodes
                uncertainty = self._compute_uncertainty_for_nodes(all_nodes)
                consecutive_failed_additions = 0

                logger.info(
                    "TreeSearchGroupedWorkflow [dynamic]: query_id=%s "
                    "added episode, U=%.6f, episodes=%d",
                    query_id,
                    uncertainty,
                    self._count_episodes(all_nodes),
                )
            else:
                consecutive_failed_additions += 1
                logger.warning(
                    "TreeSearchGroupedWorkflow [dynamic]: query_id=%s "
                    "additional episode produced no nodes (%d/%d)",
                    query_id,
                    consecutive_failed_additions,
                    max_failed_additions,
                )
                if consecutive_failed_additions >= max_failed_additions:
                    logger.warning(
                        "TreeSearchGroupedWorkflow [dynamic]: query_id=%s "
                        "stopping after %d consecutive empty additions",
                        query_id,
                        consecutive_failed_additions,
                    )
                    break

        return await self._finalize_episode(
            fresh_nodes, cached_nodes, engine, data, query_id
        )

    def _compute_uncertainty_for_nodes(self, nodes: list[Node]) -> float:
        """Compute uncertainty U(q) from a list of nodes."""
        from customized_areal.tree_search.core.uncertainty import (
            compute_query_uncertainty,
        )

        episode_data: dict[str, tuple[float, int]] = {}
        for node in nodes:
            if not node.episode_id:
                continue
            if node.episode_id not in episode_data:
                episode_data[node.episode_id] = (node.outcome_reward, node.turn_idx)
            else:
                prev_reward, prev_max_turn = episode_data[node.episode_id]
                episode_data[node.episode_id] = (
                    prev_reward,
                    max(prev_max_turn, node.turn_idx),
                )

        if not episode_data:
            return float("inf")

        rewards = [r for r, _ in episode_data.values()]
        steps = [s for _, s in episode_data.values()]
        return compute_query_uncertainty(rewards, steps, self.reward_type)

    @staticmethod
    def _count_episodes(nodes: list[Node]) -> int:
        """Count distinct episodes in a node list."""
        return len({n.episode_id for n in nodes if n.episode_id})

    def _backup_fresh_with_returns(
        self, fresh_nodes: list[Node], judge_beta: float
    ) -> None:
        """Accumulate MC stats for freshly inserted episodes (deferred backup).

        Called only when ``judge_beta > 0`` (the insert-time backup was skipped).
        Every fresh episode -- scratch *and* branch -- is backed up with its own
        per-node discounted return-to-go ``g_t`` of the dense shaped reward, so
        ``q_value(s_t)`` estimates ``E[return-to-go]`` rather than
        ``E[outcome]``. Crucially this is done along the episode's **full**
        root-ward path: a branch episode's shared prefix (owned by another
        episode) therefore accumulates a *return-to-go* sample from this episode
        too, keeping the prefix node's MC aggregate a homogeneous set of
        return-to-go samples. That removes the mixed-estimand bias that arises
        when branch episodes contribute bare terminal outcomes to a shared
        branch point while scratch episodes contribute return-to-go.

        The full path is reconstructed by walking ``parent_node_id`` from the
        episode's terminal (not by sorting on ``turn_idx``, because branch
        suffixes restart their turn numbering at 1), so ordering is correct
        across the branch boundary.
        """
        from customized_areal.tree_search.core.process_reward import (
            build_episode_process_rewards,
            episode_returns_to_go,
        )

        for ep_nodes in _group_nodes_by_episode(fresh_nodes):
            ordered_fresh = sorted(ep_nodes, key=lambda n: getattr(n, "turn_idx", 0))
            if not ordered_fresh:
                continue
            terminal = ordered_fresh[-1]
            if not terminal.node_id:
                continue

            # Reconstruct the FULL causal path (branch suffix + shared prefix)
            # by walking parents from the terminal; reverse to ascending order.
            path: list[Node] = []
            seen: set[str] = set()
            current: str | None = terminal.node_id
            while current and current not in seen:
                node = self.tree_store.get_node(current)
                if node is None:
                    break
                seen.add(current)
                path.append(node)
                current = node.parent_node_id
            ordered = list(reversed(path))
            if not ordered:
                continue

            rewards = build_episode_process_rewards(
                ordered,
                self.tree_store,
                beta=judge_beta,
                score_max=self.critic_score_max,
            )
            returns = episode_returns_to_go(rewards, gamma=self.critic_gamma)
            g_by_id = {n.node_id: g for n, g in zip(ordered, returns) if n.node_id}
            self.tree_store.backup_path_returns(terminal.node_id, g_by_id)

    async def _finalize_episode(
        self,
        fresh_nodes: list[Node],
        cached_nodes: list[Node],
        engine: Any,
        data: dict[str, Any],
        query_id: str,
    ) -> dict[str, Any] | None:
        """Shared finalization: insert, distill, advantage, save, convert."""
        all_nodes = fresh_nodes + cached_nodes

        if not all_nodes:
            return None

        # Zero-variance discard: if all episodes have identical reward,
        # there is no learning signal for GRPO.
        episode_rewards: list[float] = []
        seen_episodes: set[str] = set()
        for node in all_nodes:
            if node.episode_id and node.episode_id not in seen_episodes:
                episode_rewards.append(node.outcome_reward)
                seen_episodes.add(node.episode_id)
        if should_discard_query(episode_rewards):
            if fresh_nodes:
                self.tree_store.insert_super_batch(
                    [_wrap_leaf_super(fresh_nodes)], query_id=query_id
                )
            for node in all_nodes:
                if node.node_id:
                    self.tree_store.set_discarded(node.node_id, True)
            self.tree_checkpoint_manager.save_query(self.tree_store, query_id)
            logger.info(
                "TreeSearchGroupedWorkflow: discarding query_id=%s — "
                "all %d episodes have identical reward",
                query_id,
                len(episode_rewards),
            )
            return None

        provider_client = None
        provider = None
        tokenizer = None
        # When dense judge shaping is active, defer the MC backup until after
        # the judge scores are annotated so it can propagate each node's
        # return-to-go (consistent with the dense reward) instead of the bare
        # terminal outcome. judge_beta == 0 keeps the insert-time backup.
        defer_backup = (
            self.enable_judge_process_reward and self.judge_process_reward_beta > 0.0
        )
        try:
            # Insert fresh nodes into tree
            if fresh_nodes:
                self.tree_store.insert_super_batch(
                    [_wrap_leaf_super(fresh_nodes)],
                    query_id=query_id,
                    backup=not defer_backup,
                )

            if self.loss_mode != LossMode.GRPO:
                tokenizer = await self._get_tokenizer()
                provider, provider_client = await self._setup_distill_provider(
                    engine, tokenizer
                )
                all_nodes, _ = await self._prepare_distill_for_node_groups(
                    _group_nodes_by_episode(all_nodes),
                    data,
                    engine,
                    provider,
                    tokenizer,
                )

            # If distillation filtered out all nodes, don't mark originals as trained
            # — that would consume cached data without producing training signal.
            if not all_nodes:
                logger.warning(
                    "TreeSearchGroupedWorkflow: distillation produced no usable nodes "
                    "for query_id=%s; skipping (nodes not marked as trained)",
                    query_id,
                )
                return None

            # Annotate LLM-judge step-level process rewards before advantages so
            # both the actor (GAE) and critic targets consume the dense reward.
            if (
                self.enable_judge_process_reward
                and self.judge_process_reward_beta > 0.0
            ):
                if tokenizer is None:
                    tokenizer = await self._get_tokenizer()
                if provider is None:
                    provider, provider_client = await self._setup_distill_provider(
                        engine, tokenizer
                    )
                await self._annotate_judge_process_rewards(
                    provider, all_nodes, data, tokenizer
                )

            # Deferred MC backup: now that judge scores exist, accumulate the
            # per-node return-to-go for freshly inserted episodes (SCRATCH) and
            # the terminal return for branched episodes. Skipped entirely when
            # judge_beta == 0 (the insert-time backup already ran).
            if defer_backup and fresh_nodes:
                self._backup_fresh_with_returns(
                    fresh_nodes, self.judge_process_reward_beta
                )

            # Compute generative-critic state values v_phi(s_t) before advantages.
            if self.enable_generative_critic:
                await self._annotate_critic_values(engine, all_nodes)

            # Compute tree advantages
            if self.advantage_mode == AdvantageMode.TREE:
                self.tree_advantage_computer.compute(all_nodes)
            elif self.advantage_mode == AdvantageMode.GAE:
                self.gae_advantage_computer.compute(all_nodes)
            elif self.advantage_mode == AdvantageMode.HYBRID_GAE:
                self.hybrid_gae_advantage_computer.compute(all_nodes)

            # Convert to batched tensor dict
            result_dict = _nodes_to_batched_tensor_dict(
                all_nodes,
                max_tokens=self.max_tokens,
                loss_mode=self.loss_mode.value,
            )

            if not result_dict:
                return None

            # Attach critic regression data (Python object, popped in _ppo_update
            # before tensor ops -- mirrors position_rewards). Enables the shared
            # model's combined soft-regression critic step.
            if self.enable_generative_critic:
                self._attach_critic_train_data(result_dict, all_nodes)

            # Mark nodes as trained only after the batch is materialized.
            for node in all_nodes:
                if node.node_id:
                    self.tree_store.set_trained(node.node_id, True)

            # Save tree checkpoint
            self.tree_checkpoint_manager.save_query(self.tree_store, query_id)

            return result_dict
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

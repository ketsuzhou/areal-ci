# customized_areal/tree_search/core/customized_grouped_workflow.py
"""Tree-search-aware grouped rollout workflow with cache reuse and teacher distillation.

Consolidates the functionality of QueryIDProxyWorkflow,
TreeSearchGroupedRolloutWorkflow, and TreeSearchWorkflowExecutor into a single
``RolloutWorkflow`` subclass.

Architecture
~~~~~~~~~~~~
The main entry point is ``arun_episode``. When ``use_fresh_query`` is enabled
it first runs the two-phase fresh-query flow — ``_select_fresh_query_data``
picks an eligible row WITHOUT claiming it and ``_claim_fresh_query`` consumes
it only after the episode produced a usable training batch (helpers live in
``core/fresh_query.py``). It then dispatches to either ``_arun_episode_fixed``
or ``_arun_episode_dynamic`` depending on ``dynamic_group_size``. Both paths
share ``_generate_initial_round`` (cache lookup + parallel fresh generation)
and converge on ``_finalize_episode`` for shared post-processing.

Fixed group_size path (``_arun_episode_fixed``):
  1. Query the tree_store for cached (untrained) episodes.
  2. Generate ``group_size - cached_count`` fresh episodes in parallel,
     each generated from scratch.  (Branching now lives only in the
     env-dispatch runner model; the wired grouped workflow no longer
     branches from cached nodes.)
  3. Combine cached + fresh nodes.

Dynamic group_size path (``_arun_episode_dynamic``):
  1. Same initial round as fixed, but starting from ``initial_group_size``
     instead of ``group_size``.
  2. Iteratively sample one more episode, recompute uncertainty U(q), and
     stop when U(q) falls below ``uncertainty_threshold`` or
     ``max_group_size`` is reached.  A consecutive-failure circuit breaker
     prevents infinite loops.

Shared finalization (``_finalize_episode``):
  4. Zero-variance discard: if all episodes have identical reward, insert
     fresh nodes, mark them discarded, save checkpoint, and return None.
  5. Insert fresh nodes into the tree_store.
  6. If ``loss_mode != GRPO``: run selected-turn distillation (orchestration
     lives in ``core/distill_prep.py``) — diagnose the episode, identify turns
     needing improvement, gather teacher logprobs, and build per-position
     reward info.
  7. If ``advantage_mode == TREE``: compute GRPO-normalized tree advantages
     across episodes.
  8. Mark all nodes as trained, save the tree checkpoint, and convert to a
     batched tensor dict.

Helper modules and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``core/fresh_query.py`` — fresh-query row normalization and claim helpers
  (``_apply_fresh_query_row``, ``_execute_fresh_query_claim_update``, ...).
- ``core/batch_convert.py`` — ``interactions_dict_to_nodes`` plus the batched
  tensor-dict converters for Nodes/SuperNodes and the VIMPO metadata stamper
  (re-exported here for backward compatibility).
- ``core/distill_prep.py`` — selected-turn distillation orchestration
  (``prepare_distill_for_episode`` / ``prepare_distill_for_node_groups`` /
  ``setup_distill_provider``); the class keeps thin delegating methods under
  the historical ``_prepare_distill_*`` / ``_setup_distill_provider`` names.
- ``annotate_nodes_from_run`` — stamp TPFC assistant metadata (entropy, branch
  info) onto nodes by turn index.
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
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    Config,
    LossMode,
    SampleSource,
)

# Re-exported for backward compatibility: tests import/patch these names on
# this module while the implementations live in the split-out helper modules.
from customized_areal.tree_search.core.batch_convert import (
    _nodes_to_batched_tensor_dict,
    _supernodes_to_batched_tensor_dict,
    annotate_vimpo_episode_metadata,
    interactions_dict_to_nodes,
)
from customized_areal.tree_search.core.distill_prep import (
    prepare_distill_for_episode,
    prepare_distill_for_node_groups,
    setup_distill_provider,
)
from customized_areal.tree_search.core.fresh_query import (
    _FRESH_QUERY_SELECT_LIMIT,
    _affected_row_count,
    _apply_fresh_query_row,
    _apply_train_id_not_contains_filter,
    _execute_fresh_query_claim_update,
    _normalize_used4train,
)
from customized_areal.tree_search.core.tree_store import Node
from customized_areal.tree_search.core.uncertainty import should_discard_query

from areal.api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")

# Circuit breaker for the dynamic-group-size sampling loop: stop after this
# many CONSECUTIVE failed episode additions. Deliberately small and
# independent of max_group_size — failures come with exponential backoff in
# _retry_episode, so a large budget would stall the rollout for minutes.
_MAX_CONSECUTIVE_FAILED_ADDITIONS = 3


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

    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        config: Config,
        *,
        tokenizer_path: str = "",
        max_tokens: int = 0,
        multica_dag_enabled: bool = False,
        multica_dag_client=None,
        multica_assembler=None,
        multica_resolver=None,
        multica_session_remover=None,
    ) -> None:
        from customized_areal.tree_search.core.advantage import TreeAdvantageComputer
        from customized_areal.tree_search.core.checkpoint import TreeCheckpointManager
        from customized_areal.tree_search.core.tree_store import MCTSTreeStore

        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.config = config
        self.workflow = workflow
        self.group_size = group_size
        self.advantage_mode = config.advantage_mode
        self.loss_mode = config.loss_mode
        self.cache_mode = config.mode
        self.enable_generative_critic = config.enable_generative_critic
        # Generative-critic knobs.
        self.critic_gamma = config.critic_gamma
        self.critic_lambda = config.critic_lambda
        self.critic_avg_success_rate = config.critic_avg_success_rate
        self.critic_score_max = config.critic_score_max
        self.critic_max_new_tokens = config.critic_max_new_tokens
        self.critic_temperature = config.critic_temperature
        self.critic_target_scale = config.critic_target_scale
        self.critic_td_n_steps = config.critic_td_n_steps
        # Variance-aware hybrid GAE + TD-gated branching knobs.
        self.hybrid_mc_min_visits = config.hybrid_mc_min_visits
        self.hybrid_critic_var_floor = config.hybrid_critic_var_floor
        self.hybrid_critic_error_var = config.hybrid_critic_error_var
        self.branch_td_threshold = config.branch_td_threshold
        # ARE-4 ΔV advantage (advantage_mode=VERSIONED_BACKUP) hyperparameters.
        self.versioned_rho = config.versioned_rho
        self.versioned_n_min = config.versioned_n_min
        self.versioned_n_max = config.versioned_n_max
        # Resolved blend weight passed to ``compute_critic_targets``: either the
        # fixed float ``critic_mc_weight`` or an adaptive controller. The
        # controller is stateful and persists across rollouts so its critic-error
        # EMA can be updated if/when training-side feedback is wired in.
        if config.critic_mc_adaptive:
            from customized_areal.tree_search.training.losses.critic import (
                AdaptiveMCWeight,
            )

            self._mc_mixer: Any = AdaptiveMCWeight(c=config.critic_mc_c)
        else:
            self._mc_mixer = config.critic_mc_weight
        self.tokenizer_path = tokenizer_path
        self.max_reasoning_tokens = config.max_reasoning_tokens
        self.rl_loss_weight = config.rl_loss_weight
        self.distill_loss_weight = config.distill_loss_weight
        self.topk_distill = config.topk_distill
        self.teacher_provider = config.teacher_provider
        self.teacher_base_url = config.teacher_base_url
        self.teacher_backend = config.teacher_backend
        self.teacher_model_name = config.teacher_model_name
        self.teacher_api_key = config.teacher_api_key
        self.teacher_top_k = config.teacher_top_k
        self.teacher_max_retries = config.teacher_max_retries
        self.teacher_timeout = config.teacher_timeout
        self.teacher_max_concurrency = config.teacher_max_concurrency
        self.teacher_missing_logprob = config.teacher_missing_logprob
        self.diagnose_model_name = config.diagnose_model_name
        self.diagnose_max_tokens = config.diagnose_max_tokens
        self.diagnose_temperature = config.diagnose_temperature
        self.diagnose_base_url = config.diagnose_base_url
        self.diagnose_api_key = config.diagnose_api_key
        self.strict_distill_json = config.strict_distill_json
        self.max_tokens = max_tokens
        self.sample_source = SampleSource(config.sample_source)
        self.branch_probability = config.branch_probability
        self.dynamic_group_size = config.dynamic_group_size
        self.initial_group_size = config.initial_group_size
        self.max_group_size = config.max_group_size
        self.uncertainty_threshold = config.uncertainty_threshold
        self.reward_type = config.reward_type
        self.distill_kl_mode = config.distill_kl_mode
        self.max_distill_tokens = config.max_distill_tokens or max_tokens
        self.use_fresh_query = config.use_fresh_query
        self.fresh_query_table = config.fresh_query_table
        # query_ids selected by in-flight arun_episode calls (claim is deferred
        # until a usable batch exists; see _select/_claim_fresh_query).
        self._inflight_fresh_queries: set[str] = set()
        # Per-instance tokenizer cache (a class-level dict/lock would leak
        # across instances and bind one asyncio.Lock to foreign event loops).
        self._tokenizer_cache: dict[str, Any] = {}
        self._tokenizer_lock = asyncio.Lock()
        if config.dynamic_group_size:
            # Field-level validation (initial >= 1, max >= initial, threshold
            # >= 0, reward_type) already lives in Config.__post_init__.
            self.group_size = config.initial_group_size

        self.tree_checkpoint_manager = TreeCheckpointManager(config.checkpoint_dir)

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
            judge_score_max=self.critic_score_max,
            mc_min_visits=self.hybrid_mc_min_visits,
            critic_var_floor=self.hybrid_critic_var_floor,
            critic_error_var=self.hybrid_critic_error_var,
            critic_error_var_fn=critic_error_var_fn,
        )
        from customized_areal.tree_search.core.advantage import (
            VersionedBackupAdvantageComputer,
        )

        self.versioned_backup_advantage_computer = VersionedBackupAdvantageComputer(
            self.tree_store,
            gamma=self.critic_gamma,
            lam=self.critic_lambda,
            judge_score_max=self.critic_score_max,
            mc_min_visits=self.hybrid_mc_min_visits,
            critic_var_floor=self.hybrid_critic_var_floor,
            critic_error_var=self.hybrid_critic_error_var,
            critic_error_var_fn=critic_error_var_fn,
            rho=self.versioned_rho,
            n_min=self.versioned_n_min,
            n_max=self.versioned_n_max,
        )
        # Lazily constructed on first use (needs the tokenizer).
        self._critic_value_client = None
        # Multica DAG rollout (Phase 1b/2). Declared now so callers can pass
        # them, but the single-agent path runs unchanged until the coordinator
        # dispatch is wired.
        self._multica_dag_enabled = multica_dag_enabled
        self._multica_dag_client = multica_dag_client
        self._coordinator = None  # Phase 1b/2 wires TeamRolloutCoordinator
        if multica_dag_enabled:
            # Wire the multica base workflow: one arun_episode = one Multica task
            # = N agents -> AssembledDag. multica_dag_client is the get_dag
            # poller; the  client (a distinct concern) is
            # constructed directly here as a MulticaEnvDispatchClient, which
            # reads MULTICA_BASE_URL / MULTICA_API_KEY from the environment.
            # The workflow's group_size is the squad size N (defaults to 1);
            # this grouped workflow's own group_size is M parallel tasks - a
            # different axis. Multica owns /rl/start_session + the
            # session_to_agent_run binding + per-agent credentials; AReaL never
            # calls start_session on this path. The passed-in ``workflow`` is
            # overridden on the multica path.
            from customized_areal.tree_search.agents.multi_agent_workflow import (
                MultiAgentEnvDispatchWorkflow,
            )

            self.workflow = MultiAgentEnvDispatchWorkflow(
                dag_client=multica_dag_client,
                assembler=multica_assembler,
                resolver=multica_resolver,
                session_remover=multica_session_remover,
            )

    async def _select_fresh_query_data(
        self,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, str]] | None:
        """Select an eligible fresh-query row WITHOUT claiming it.

        The claim (appending TRAIN_ID to ``used4train``) is deferred to
        :meth:`_claim_fresh_query`, which runs only after the episode produced
        a usable training batch — otherwise failed or zero-variance-discarded
        rollouts would permanently consume the query.

        Returns ``(merged_data, (query_id, train_id))``, or ``None`` when no
        eligible row exists. Rows already in-flight in this process (selected
        by another concurrent ``arun_episode``) are skipped.
        """
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
            if query_id in self._inflight_fresh_queries:
                continue
            used4train = _normalize_used4train(row.get("used4train"))
            if train_id in used4train:
                continue

            self._inflight_fresh_queries.add(query_id)
            logger.info(
                "Selected fresh query query_id=%s for train_id=%s (claim deferred)",
                query_id,
                train_id,
            )
            return _apply_fresh_query_row(data, row), (query_id, train_id)

        logger.warning(
            "No eligible fresh query rows found in table=%s for train_id=%s",
            self.fresh_query_table,
            train_id,
        )
        return None

    async def _claim_fresh_query(self, query_id: str, train_id: str) -> None:
        """Claim a previously selected fresh query after a successful rollout.

        Re-reads ``used4train`` so concurrent claims by other train runs are
        preserved, then appends ``train_id`` with a ``not_.contains`` guard so
        a retried/duplicated claim is a no-op. Best-effort: a lost race is
        logged, not raised.
        """
        from customized_areal.db_service import DBConnection

        client = await DBConnection().get_client()
        row_result = (
            await client.table(self.fresh_query_table)
            .select("used4train")
            .eq("query_id", query_id)
            .execute()
        )
        rows = getattr(row_result, "data", None)
        current = (
            _normalize_used4train(rows[0].get("used4train"))
            if isinstance(rows, list) and rows and isinstance(rows[0], dict)
            else []
        )
        if train_id in current:
            return
        update_query = (
            client.table(self.fresh_query_table)
            .update({"used4train": [*current, train_id]})
            .eq("query_id", query_id)
        )
        claim_result = await _execute_fresh_query_claim_update(
            update_query,
            train_id=train_id,
        )
        if _affected_row_count(claim_result) < 1:
            logger.warning(
                "Fresh query claim lost race for query_id=%s train_id=%s; "
                "the rollout may be re-used by another run",
                query_id,
                train_id,
            )
            return
        logger.info(
            "Claimed fresh query query_id=%s for train_id=%s",
            query_id,
            train_id,
        )

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

    async def _get_tokenizer(self, *, purpose: str):
        """Load the HF tokenizer for ``purpose`` (cached per instance).

        Raises ValueError when ``tokenizer_path`` is unset — a configuration
        error, not a transient per-episode failure. The blocking HF load runs
        in a thread so the rollout event loop stays responsive.
        """
        if not self.tokenizer_path:
            raise ValueError(f"tokenizer_path is required for {purpose}")
        async with self._tokenizer_lock:
            tokenizer = self._tokenizer_cache.get(self.tokenizer_path)
            if tokenizer is None:
                from areal.utils.hf_utils import load_hf_tokenizer

                tokenizer = await asyncio.to_thread(
                    load_hf_tokenizer, self.tokenizer_path
                )
                self._tokenizer_cache[self.tokenizer_path] = tokenizer
            return tokenizer

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
            tokenizer = await self._get_tokenizer(purpose="the generative critic")
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
        return await setup_distill_provider(self.config, engine, tokenizer)

    async def _prepare_distill_for_episode(
        self,
        nodes: list[Node],
        data: dict[str, Any],
        engine: Any,
        provider: Any,
        tokenizer: Any,
    ) -> tuple[list[Node], dict[str, list[Any]]]:
        # Thin delegate: reads only the mirror attributes (never self.config)
        # so the failure-recovery stub tests can drive it on a bare instance.
        return await prepare_distill_for_episode(
            nodes,
            data,
            engine,
            provider,
            tokenizer,
            loss_mode=self.loss_mode,
            topk_distill=self.topk_distill,
            teacher_top_k=self.teacher_top_k,
            max_distill_tokens=self.max_distill_tokens,
        )

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
            if attempt == max_retries:
                break
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
        # Thin delegate: reads only the mirror attributes (never self.config)
        # so the failure-recovery stub tests can drive it on a bare instance.
        return await prepare_distill_for_node_groups(
            node_groups,
            data,
            engine,
            provider,
            tokenizer,
            loss_mode=self.loss_mode,
            topk_distill=self.topk_distill,
            teacher_top_k=self.teacher_top_k,
            max_distill_tokens=self.max_distill_tokens,
        )

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        fresh_claim: tuple[str, str] | None = None
        if self.use_fresh_query:
            selected = await self._select_fresh_query_data(data)
            if selected is None:
                return None
            data, fresh_claim = selected

        query_id = data.get("query_id") or ""
        try:
            if self.dynamic_group_size:
                result = await self._arun_episode_dynamic(engine, data, query_id)
            else:
                result = await self._arun_episode_fixed(engine, data, query_id)
        except ValueError:
            # Configuration/programming errors (missing tokenizer_path,
            # inconsistent VIMPO metadata, ...) are not transient per-episode
            # failures — surface them instead of silently skipping every
            # query and training on nothing.
            logger.exception(
                "TreeSearchGroupedWorkflow.arun_episode hit a fatal error "
                "for query_id=%s",
                query_id,
            )
            raise
        except Exception:
            logger.exception(
                "TreeSearchGroupedWorkflow.arun_episode failed for query_id=%s",
                query_id,
            )
            return None
        finally:
            if fresh_claim is not None:
                self._inflight_fresh_queries.discard(fresh_claim[0])

        if result is not None and fresh_claim is not None:
            # Only consume the query once a usable batch actually exists.
            await self._claim_fresh_query(*fresh_claim)
        return result

    async def _generate_initial_round(
        self,
        engine,
        data: dict[str, Any],
        query_id: str,
        n_episodes: int,
        *,
        mode_label: str = "",
    ) -> tuple[list[Node], list[Node]]:
        """Shared initial round of the fixed/dynamic paths.

        Looks up cached (untrained) episodes for ``query_id``, generates the
        ``n_episodes - cached_count`` deficit of fresh episodes in parallel,
        loads the cached nodes, and returns ``(fresh_nodes, cached_nodes)``.
        DISTILL mode consumes the cache only (no fresh generation).
        ``mode_label`` distinguishes the callers in the log line ("" for
        fixed, " [dynamic]" for dynamic).
        """
        cached_count = (
            self.tree_store.get_untrained_episode_count(query_id) if query_id else 0
        )
        need_gen = max(0, n_episodes - cached_count)

        logger.info(
            "TreeSearchGroupedWorkflow%s: query_id=%s, group_size=%d, "
            "cached=%d, need_gen=%d",
            mode_label,
            query_id,
            n_episodes,
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

        return fresh_nodes, cached_nodes

    async def _arun_episode_fixed(
        self, engine, data: dict[str, Any], query_id: str
    ) -> dict[str, Any] | None:
        """Original fixed group_size logic (with zero-variance discard)."""
        fresh_nodes, cached_nodes = await self._generate_initial_round(
            engine, data, query_id, self.group_size
        )

        return await self._finalize_episode(
            fresh_nodes, cached_nodes, engine, data, query_id
        )

    @property
    def _branch_budget(self) -> int:
        """Per-query branch budget: max additional (branch) episodes beyond the
        initial SCRATCH round. Branches are capped at
        ``max_group_size - initial_group_size`` so the total (initial + branches)
        never exceeds ``max_group_size`` - even when early episodes fail and
        ``_count_episodes`` stays below ``max_group_size``.
        """
        return max(0, self.max_group_size - self.initial_group_size)

    async def _arun_episode_dynamic(
        self, engine, data: dict[str, Any], query_id: str
    ) -> dict[str, Any] | None:
        """Dynamic group_size: iterative sampling with uncertainty threshold."""

        # 1. Initial round
        fresh_nodes, cached_nodes = await self._generate_initial_round(
            engine,
            data,
            query_id,
            self.initial_group_size,
            mode_label=" [dynamic]",
        )
        # The group-idx counter for additional episodes starts where the
        # initial round left off. This is the same need_gen the helper
        # computed — the untrained count is unchanged between the two reads.
        cached_count = (
            self.tree_store.get_untrained_episode_count(query_id) if query_id else 0
        )
        next_group_idx = max(0, self.initial_group_size - cached_count)

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
        consecutive_failed_additions = 0
        max_failed_additions = _MAX_CONSECUTIVE_FAILED_ADDITIONS
        branch_budget = self._branch_budget
        branch_samples = 0  # successful additional (branch) episodes
        while (
            self.loss_mode != LossMode.DISTILL
            and self._count_episodes(all_nodes) < self.max_group_size
            and branch_samples < branch_budget
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
                branch_samples += 1

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

    async def _finalize_episode(
        self,
        fresh_nodes: list[Node],
        cached_nodes: list[Node],
        engine: Any,
        data: dict[str, Any],
        query_id: str,
    ) -> dict[str, Any] | None:
        """Shared finalization: insert, distill, advantage, save, convert."""
        # Multica multi-agent path: fresh_nodes are SuperNodes (one assembled
        # DAG). Finalize them directly - insert + DAG advantages + batch from
        # the resolved segment tensors - skipping the single-agent Node path
        # (zero-variance discard / distillation / judge / critic V are Change 2
        # concerns, deferred on this path).
        if fresh_nodes and isinstance(fresh_nodes[0], SuperNode):
            return await self._finalize_multica_episode(fresh_nodes, query_id)
        all_nodes = fresh_nodes + cached_nodes
        # A partially-trained cached episode is loaded whole by
        # load_untrained_episodes; drop its already-trained nodes so they are
        # neither re-batched (double training) nor re-marked below.
        all_nodes = [
            node
            for node in all_nodes
            if not node.node_id or not self.tree_store.is_trained(node.node_id)
        ]

        if not all_nodes:
            return None

        # Zero-variance discard: if all episodes have identical reward, there
        # is no learning signal for GRPO. Distillation modes keep the query —
        # an all-wrong (or all-right) group is exactly where teacher guidance
        # still provides a training signal.
        episode_rewards: list[float] = []
        seen_episodes: set[str] = set()
        for node in all_nodes:
            if node.episode_id and node.episode_id not in seen_episodes:
                episode_rewards.append(node.outcome_reward)
                seen_episodes.add(node.episode_id)
        if self.loss_mode == LossMode.GRPO and should_discard_query(episode_rewards):
            if fresh_nodes:
                self.tree_store.insert_super_batch(
                    [_wrap_leaf_super(fresh_nodes)], query_id=query_id
                )
            for node in all_nodes:
                if node.node_id:
                    self.tree_store.set_discarded(node.node_id, True)
            await asyncio.to_thread(
                self.tree_checkpoint_manager.save_query, self.tree_store, query_id
            )
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
        try:
            # Insert fresh nodes into tree
            if fresh_nodes:
                # ARE-4: the fresh rollout IS the latest exploration. Record
                # its policy version so new/old nodes resolve correctly, and
                # back up its returns into the latest-only stats (record_latest)
                # so V_latest can be computed new-only.
                fresh_vids = [
                    n.version_id
                    for n in fresh_nodes
                    if getattr(n, "version_id", -1) >= 0
                ]
                if fresh_vids:
                    self.tree_store.set_latest_version(max(fresh_vids))
                self.tree_store.insert_super_batch(
                    [_wrap_leaf_super(fresh_nodes)],
                    query_id=query_id,
                    backup=True,
                    record_latest=True,
                )

            if self.loss_mode != LossMode.GRPO:
                tokenizer = await self._get_tokenizer(
                    purpose="tree-search distillation"
                )
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

            # VIMPO is critic-free: it skips the generative-critic value
            # annotation, skips every Node advantage computer (its
            # VIMPOAdvantageComputer runs later on the batched dict), and never
            # attaches critic_train_data. It only stamps episode identity +
            # centered terminal-reward targets here so tensorization can emit
            # the vimpo_* metadata keys.
            is_vimpo = self.advantage_mode == AdvantageMode.VIMPO

            # Compute generative-critic state values v_phi(s_t) before advantages.
            if not is_vimpo and self.enable_generative_critic:
                await self._annotate_critic_values(engine, all_nodes)

            # Compute tree advantages
            if self.advantage_mode == AdvantageMode.TREE:
                self.tree_advantage_computer.compute(all_nodes)
            elif self.advantage_mode == AdvantageMode.GAE:
                self.gae_advantage_computer.compute(all_nodes)
            elif self.advantage_mode == AdvantageMode.HYBRID_GAE:
                self.hybrid_gae_advantage_computer.compute(all_nodes)
            elif self.advantage_mode == AdvantageMode.VERSIONED_BACKUP:
                self.versioned_backup_advantage_computer.compute(all_nodes)
            elif is_vimpo:
                annotate_vimpo_episode_metadata(all_nodes)

            # Convert to batched tensor dict
            result_dict = _nodes_to_batched_tensor_dict(
                all_nodes,
                max_tokens=self.max_tokens,
                loss_mode=self.loss_mode.value,
                advantage_mode=self.advantage_mode.value if is_vimpo else None,
            )

            if not result_dict:
                return None

            # Attach critic regression data (Python object, popped in _ppo_update
            # before tensor ops -- mirrors position_rewards). Enables the shared
            # model's combined soft-regression critic step.
            if not is_vimpo and self.enable_generative_critic:
                self._attach_critic_train_data(result_dict, all_nodes)

            # Mark nodes as trained only after the batch is materialized.
            for node in all_nodes:
                if node.node_id:
                    self.tree_store.set_trained(node.node_id, True)

            # Save tree checkpoint
            await asyncio.to_thread(
                self.tree_checkpoint_manager.save_query, self.tree_store, query_id
            )

            return result_dict
        finally:
            if provider_client is not None:
                await provider_client.close()

    async def _finalize_multica_episode(
        self,
        super_nodes: list[SuperNode],
        query_id: str,
    ) -> dict[str, Any] | None:
        """Finalize a multica multi-agent episode (``list[SuperNode]``).

        Inserts the SuperNodes via ``insert_super_batch`` (NOT
        ``_wrap_leaf_super`` -- the multi-segment edge structure is preserved),
        computes DAG-level GAE advantages via ``assemble_node_advantages`` per
        assembled DAG (grouped by ``group_idx``), and builds the training batch
        from each segment's resolved tensors (``metadata["tensors"]``). Rewards
        and critic values are placeholder zero on this path (Change 2 wires the
        judge + critic V), so advantages are zero -- the point is that the
        multica data path runs end-to-end. ``super_nodes`` may span M parallel
        rollouts (one DAG each); per-episode grouping keeps independent DAGs
        from being chained by the global GAE (which propagates backward, so a
        later episode's reward would otherwise contaminate an earlier one).
        """
        from customized_areal.tree_search.agents.dag_advantage import (
            AssembledAdvantages,
            assemble_node_advantages,
        )

        if not super_nodes:
            return None

        self.tree_store.insert_super_batch(super_nodes, query_id=query_id, backup=True)
        # MCTS branch backup: for each BRANCH edge, propagate the branch's
        # terminal return (the child segment's outcome_reward) to the parent
        # (fork) segment's value via branch_backup. This is the cross-episode
        # value signal for future branch selection; distinct from the per-episode
        # GAE advantages computed below. (Branch session cleanup - removing the
        # branched sessions - happens in MultiAgentEnvDispatchWorkflow.arun_episode.)
        from customized_areal.tree_search.agents.dag_backup import branch_backup
        from customized_areal.tree_search.agents.execution_dag import EdgeType

        super_by_id = {sn.node_id: sn for sn in super_nodes}
        for sn in super_nodes:
            for dst_id, etype in sn.outgoing_edges:
                if etype is EdgeType.BRANCH:
                    child = super_by_id.get(dst_id)
                    if child is None:
                        continue
                    branch_backup(
                        sn,
                        branch_return=float(child.outcome_reward),
                        discount=self.critic_gamma,
                    )

        # Group by episode (group_idx, stamped in _result_to_nodes) and run GAE
        # per assembled DAG. A single global pass would chain independent
        # episodes -- GAE propagates backward, so a later episode's reward would
        # contaminate an earlier episode's advantages.
        per_episode: dict[int, list[SuperNode]] = {}
        order: list[int] = []
        for sn in super_nodes:
            gi = sn.metadata.get("group_idx", 0)
            if gi not in per_episode:
                per_episode[gi] = []
                order.append(gi)
            per_episode[gi].append(sn)
        adv: dict[str, float] = {}
        returns: dict[str, float] = {}
        baselines: dict[str, float] = {}
        for gi in order:
            ep_adv = assemble_node_advantages(
                per_episode[gi],
                initial_value=0.0,
                gamma=self.critic_gamma,
                lam=self.critic_lambda,
            )
            adv.update(ep_adv.advantages)
            returns.update(ep_adv.returns)
            baselines.update(ep_adv.baseline_values)
        merged_advantages = AssembledAdvantages(
            advantages=adv, returns=returns, baseline_values=baselines
        )
        result_dict = _supernodes_to_batched_tensor_dict(
            super_nodes,
            merged_advantages,
            max_tokens=self.max_tokens,
            loss_mode=self.loss_mode.value,
        )
        if not result_dict:
            return None
        await asyncio.to_thread(
            self.tree_checkpoint_manager.save_query, self.tree_store, query_id
        )
        return result_dict


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

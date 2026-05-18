# customized_areal/tree_search/tree_search_grouped_workflow.py
"""Tree-search-aware grouped rollout workflow with cache reuse.

Consolidates the functionality of QueryIDProxyWorkflow,
TreeSearchGroupedRolloutWorkflow, and TreeSearchWorkflowExecutor into
a single class that:
- Loads/saves tree_store from a checkpoint directory
- Does per-query cache lookup to determine how many fresh episodes are needed (episode-level counting)
- Generates only the needed fresh episodes (partial cache reuse)
- Converts fresh results to Nodes, loads cached Nodes, combines them
- Inserts fresh Nodes into tree_store, computes advantages, marks trained
- Saves tree checkpoint
- Returns batched tensor dicts that the base WorkflowExecutor handles natively
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any

from customized_areal.tree_search.config import AdvantageMode, CacheMode, LossMode
from customized_areal.tree_search.mcts_tree_store import Node

from areal.api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")


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
                parent_res = interaction.parent.to_tensor_dict()
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

            outcome_reward = interaction.reward if interaction.reward is not None else 0.0

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
            td = interaction.to_tensor_dict()
            seq_tokens = td["input_ids"].squeeze(0).tolist()
            logprobs = td["logprobs"].squeeze(0).tolist()
            loss_mask = td["loss_mask"].squeeze(0).tolist()
            versions = td["versions"].squeeze(0).tolist()
            outcome_reward = interaction.reward if interaction.reward is not None else 0.0
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


def _nodes_to_batched_tensor_dict(nodes: list[Node]) -> dict[str, Any] | None:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    Returns None if nodes is empty.
    """
    if not nodes:
        return None

    from areal.utils.data import concat_padded_tensors
    from customized_areal.tree_search.mcts_tree_store import _node_to_tensor_dict

    tensor_dicts = [
        _node_to_tensor_dict(
            node,
            query_id=node.query_id or "",
            node_id=node.node_id,
        )
        for node in nodes
    ]
    return concat_padded_tensors(tensor_dicts)


class TreeSearchGroupedRolloutWorkflow(RolloutWorkflow):
    """GroupedRolloutWorkflow with tree-search cache reuse, tree ops, and checkpoint.

    Wraps the base OpenAIProxyWorkflow and overrides arun_episode to:
    1. Check cache: how many untrained episodes exist for this query?
    2. Generate only the needed fresh episodes (group_size - cached_count)
    3. Convert fresh results to Nodes, load cached episode Nodes
    4. Combine cached + fresh Nodes (total = group_size episodes)
    5. Insert fresh Nodes into tree_store
    6. Compute tree advantages per-episode (if advantage_mode == TREE)
    7. Mark all nodes as trained
    8. Save tree checkpoint (if cache_mode == CROSS_TRAINING)
    9. Return batched tensor dict
    """

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
        self.max_reasoning_tokens = max_reasoning_tokens
        self.rl_loss_weight = rl_loss_weight
        self.distill_loss_weight = distill_loss_weight

        # Tokenizer path for episode validation (shared via class-level cache)
        self._tokenizer_path = tokenizer_path

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

    def _result_to_nodes(self, result: Any, query_id: str, group_idx: int) -> list[Node] | None:
        """Convert a single arun_episode result to list[Node]."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

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
        return nodes

    # Class-level tokenizer cache: same path shares one instance across all workflow objects
    _shared_tokenizers: dict[str, Any] = {}

    @property
    def tokenizer(self):
        """Lazy-load tokenizer, shared across instances with the same path."""
        if self._tokenizer_path not in self._shared_tokenizers:
            if not self._tokenizer_path:
                raise ValueError(
                    "tokenizer_path is required for episode validation "
                    "(reasoning filter). Set it in __init__ or via "
                    "TREE_SEARCH_TOKENIZER_PATH env var."
                )
            from transformers import AutoTokenizer
            self._shared_tokenizers[self._tokenizer_path] = AutoTokenizer.from_pretrained(
                self._tokenizer_path, trust_remote_code=True
            )
            logger.info("Loaded tokenizer from %s (shared cache)", self._tokenizer_path)
        return self._shared_tokenizers[self._tokenizer_path]

    def _is_episode_valid(self, nodes: list[Node]) -> bool:
        """Reject episode if any node lacks <reasoning>...</reasoning> in its
        loss-masked (response) portion, or if reasoning content exceeds
        max_reasoning_tokens in length.
        """
        import re

        for node in nodes:
            resp_ids = [
                tid for tid, m in zip(node.input_ids, node.loss_mask) if m == 1
            ]
            if not resp_ids:
                return False
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=False)
            m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            if m is None:
                return False
            if len(m.group(1)) > self.max_reasoning_tokens:
                return False
        return True

    async def _retry_episode(
        self, engine, data: dict[str, Any], query_id: str, group_idx: int,
        max_retries: int = 5,
    ) -> list[Node] | None:
        """Retry a failed episode until success or max_retries exhausted.

        On success, converts the result to Nodes and inserts them into
        tree_store immediately, so that other concurrent episodes can
        see the newly inserted nodes.

        Episodes that fail the _is_episode_valid filter (missing or
        overlong <reasoning>) are treated as failed attempts and retried.
        """
        for attempt in range(1, max_retries + 1):
            result = await self.workflow.arun_episode(engine, data)
            if not isinstance(result, Exception) and result is not None:
                nodes = self._result_to_nodes(result, query_id, group_idx)
                if nodes and self._is_episode_valid(nodes):
                    self.tree_store.insert_batch(nodes)
                    return nodes
                if nodes:
                    logger.warning(
                        "Episode %s retry %d/%d rejected: "
                        "invalid <reasoning> (missing or >%d chars)",
                        group_idx, attempt, max_retries,
                        self.max_reasoning_tokens,
                    )
                else:
                    logger.warning(
                        "Episode %s retry %d/%d returned None after _result_to_nodes",
                        group_idx, attempt, max_retries,
                    )
            else:
                logger.warning(
                    "Episode %s retry %d/%d %s",
                    group_idx, attempt, max_retries,
                    f"failed: {result}" if isinstance(result, Exception) else "returned None",
                )
                if isinstance(result, Exception):
                    logger.warning(
                        "Episode %s retry %d traceback:\n%s",
                        group_idx, attempt,
                        "".join(traceback.format_exception(type(result), result, result.__traceback__)),
                    )
            wait = 2 ** attempt
            logger.info("Episode %s retry %d — waiting %ds before next attempt", group_idx, attempt, wait)
            await asyncio.sleep(wait)
        logger.error(
            "Episode %s exhausted all %d retries — skipping", group_idx, max_retries
        )
        return None

    async def arun_episode(
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
                    self._retry_episode(engine, data, query_id, group_idx)
                    for group_idx in range(need_gen)
                ],
                return_exceptions=True,
            )

            for group_idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Episode %d unrecoverable: %s", group_idx, result
                    )
                    continue
                if result is None:
                    continue
                fresh_nodes.extend(result)

        # 3. Load cached nodes
        cached_nodes: list[Node] = []
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_untrained_episodes(
                query_id, cached_count
            )
            # Reset versions to 0 so decoupled PPO treats cached rollouts
            # as coming from the current behavior policy
            for node in cached_nodes:
                node.versions = [0 if m == 1 else -1 for m in node.loss_mask]

        # 4. Combine
        all_nodes = fresh_nodes + cached_nodes

        if not all_nodes:
            return None

        # 5. (fresh nodes already inserted into tree_store by _retry_episode)

        # 6. Compute tree advantages
        if self.advantage_mode == AdvantageMode.TREE:
            self.tree_advantage_computer.compute(all_nodes)

        # 7. Mark all nodes as trained
        for node in all_nodes:
            if node.node_id:
                self.tree_store.set_trained(node.node_id, True)

        # 8. Save tree checkpoint
        if self.cache_mode == CacheMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)

        # 9. Convert to batched tensor dict
        result_dict = _nodes_to_batched_tensor_dict(all_nodes)

        # 10. Inject distill loss weights
        if result_dict is not None and self.loss_mode != LossMode.GRPO:
            if self.loss_mode == LossMode.DISTILL:
                result_dict["rl_loss_weight"] = 0.0
            else:
                result_dict["rl_loss_weight"] = self.rl_loss_weight
            result_dict["distill_loss_weight"] = self.distill_loss_weight

        return result_dict

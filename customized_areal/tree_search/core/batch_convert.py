# customized_areal/tree_search/core/batch_convert.py
"""Conversion of rollout results (Nodes / SuperNodes) to batched tensor dicts.

Holds the batch-conversion helpers split out of
``customized_grouped_workflow.py``:

- ``interactions_dict_to_nodes`` — convert inference-engine interactions to
  ``Node`` objects, handling both live ``model_response`` and
  proxy-deserialized tensor caches.
- ``_nodes_to_batched_tensor_dict`` — batch a list of ``Node`` into one padded
  tensor dict.
- ``_supernodes_to_batched_tensor_dict`` — the multica parallel of the above
  for ``SuperNode`` segments.
- ``annotate_vimpo_episode_metadata`` — stamp VIMPO episode identity +
  centered terminal-reward onto nodes.

Top-level imports are limited to stdlib, the tree-store / execution-DAG data
types, and ``areal.utils.logging``; heavy dependencies (``RTensor``,
``concat_padded_tensors``, ``_node_to_tensor_dict``, torch, the interaction
types) stay as function-local lazy imports so the module remains importable in
minimal (stubbed) environments.
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.core.tree_store import Node, version_id_from_versions

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
                    # The parent/child token alignment is broken; zero-filling
                    # would silently corrupt the training sample. Skip it.
                    logger.warning(
                        "Skipping interaction %s: concat mode resp.input_len (%d) "
                        "<= parent_len (%d) — expected monotonic growth.",
                        interaction_id,
                        resp.input_len,
                        parent_len,
                    )
                    continue
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
            version_id=version_id_from_versions(versions),
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
        )

        nodes.append(node)

    return nodes


def _nodes_to_batched_tensor_dict(
    nodes: list[Node],
    max_tokens: int = 0,
    loss_mode: str | None = None,
    advantage_mode: str | None = None,
) -> dict[str, Any] | None:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    If max_tokens > 0, each node's sequence is truncated to max_tokens
    from the beginning before conversion.

    ``advantage_mode`` is forwarded to ``_node_to_tensor_dict`` so the VIMPO
    branch can emit its episode-identity / centered-target metadata tensors;
    pass ``"vimpo"`` only from the VIMPO branch of ``_finalize_episode``.

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
            advantage_mode=advantage_mode,
        )
        for node in nodes
    ]
    return concat_padded_tensors(tensor_dicts)


def annotate_vimpo_episode_metadata(nodes: list[Node]) -> None:
    """Stamp VIMPO episode identity + centered terminal-reward onto nodes.

    Groups nodes by ``query_id`` then ``episode_id`` (falling back to
    ``node_id`` when ``episode_id`` is empty). Within a query, the per-episode
    outcome reward is mean-subtracted across that query's distinct episodes to
    form ``vimpo_centered_reward`` (the policy-implied terminal-value target).
    ``vimpo_query_index`` orders the query within the batch (queries are
    sorted for determinism) and ``vimpo_episode_index`` is a globally-unique,
    monotonically increasing episode counter over distinct episodes -- not
    turns -- so every turn of one episode shares one index.

    Called only from the VIMPO branch of ``_finalize_episode`` (VIMPO is
    critic-free and skips the Node advantage computers). Raises ``ValueError``
    on a missing ``query_id``/``episode_id``, a duplicate ``turn_idx`` within
    an episode, or an inconsistent ``outcome_reward`` across one episode's
    turns.
    """
    grouped: dict[str, dict[str, list[Node]]] = {}
    for node in nodes:
        if not node.query_id:
            raise ValueError("VIMPO requires a non-empty query_id")
        episode_id = node.episode_id or node.node_id
        if not episode_id:
            raise ValueError("VIMPO requires a non-empty episode_id or node_id")
        grouped.setdefault(node.query_id, {}).setdefault(episode_id, []).append(node)
    episode_counter = 0
    for query_index, query_id in enumerate(sorted(grouped)):
        episodes = grouped[query_id]
        rewards: dict[str, float] = {}
        for episode_id, episode_nodes in episodes.items():
            turns = [node.turn_idx for node in episode_nodes]
            if len(turns) != len(set(turns)):
                raise ValueError(f"duplicate turn_idx in episode {episode_id!r}")
            values = {float(node.outcome_reward) for node in episode_nodes}
            if len(values) != 1:
                raise ValueError(
                    f"inconsistent outcome_reward in episode {episode_id!r}"
                )
            rewards[episode_id] = values.pop()
        mean_reward = sum(rewards.values()) / len(rewards)
        for episode_id in sorted(episodes):
            for node in episodes[episode_id]:
                node.vimpo_query_index = query_index
                node.vimpo_episode_index = episode_counter
                node.vimpo_centered_reward = rewards[episode_id] - mean_reward
            episode_counter += 1


def _supernodes_to_batched_tensor_dict(
    super_nodes: list[SuperNode],
    advantages: Any,
    *,
    max_tokens: int = 0,
    loss_mode: str | None = None,
) -> dict[str, Any] | None:
    """Convert multica SuperNodes to a batched tensor dict.

    The multica parallel of :func:`_nodes_to_batched_tensor_dict`: each SuperNode
    is one segment whose resolved tensors live in ``metadata["tensors"]``
    (torch-free lists: ``input_ids`` / ``loss_mask`` / ``logprobs`` /
    ``versions``). The per-segment scalar advantage from
    :func:`assemble_node_advantages` is broadcast to the response span (where
    ``loss_mask == 1``), mirroring the per-token advantage the Node path
    carries. ``topk_ids`` is a -1 sentinel (the trainer fills it) and
    ``teacher_logp`` is zeros when ``loss_mode != "grpo"`` -- distillation is a
    Change 2 concern and is not sourced from the segment tensors here.

    Returns ``None`` if ``super_nodes`` is empty.
    """
    if not super_nodes:
        return None
    from customized_areal.tree_search.core.tree_store import (
        _lazy_torch,
        _response_span,
    )

    from areal.utils.data import concat_padded_tensors

    torch = _lazy_torch()
    tensor_dicts: list[dict[str, Any]] = []
    for sn in super_nodes:
        t = sn.metadata.get("tensors") or {}
        input_ids = list(t.get("input_ids", []))
        loss_mask = list(t.get("loss_mask", []))
        logprobs = list(t.get("logprobs", []))
        versions = list(t.get("versions", []))
        if max_tokens > 0 and len(input_ids) > max_tokens:
            cut = len(input_ids) - max_tokens
            input_ids = input_ids[cut:]
            loss_mask = loss_mask[cut:]
            logprobs = logprobs[cut:]
            versions = versions[cut:]
        seq_len = len(input_ids)
        resp_start, resp_end = _response_span(loss_mask)
        resp_len = max(0, resp_end - resp_start)
        adv = (
            float(advantages.advantages.get(sn.node_id, 0.0))
            if advantages is not None
            else 0.0
        )
        traj: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "rewards": torch.tensor(
                float(sn.outcome_reward), dtype=torch.float32
            ).unsqueeze(0),
            "topk_ids": torch.full((1, resp_len, 1), -1, dtype=torch.int32),
            "advantages": torch.full((1, resp_len), adv, dtype=torch.float32),
        }
        if loss_mode != "grpo":
            traj["teacher_logp"] = torch.zeros(1, resp_len, 1, dtype=torch.float32)
        tensor_dicts.append(traj)
    return concat_padded_tensors(tensor_dicts)

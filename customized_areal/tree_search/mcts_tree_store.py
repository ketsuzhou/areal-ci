#!/usr/bin/env python3
# customized_areal/tree_search/mcts_tree_store.py
"""Flat trajectory store with MCTS statistics.

Replaces the TrieNode-based trie with a per-query list of Node
objects. Each record stores the complete, unpadded sequence from the rollout,
with turn boundaries derived from loss_mask transitions.

This correctly preserves full multi-turn context (including system prompts,
user questions, and growing conversation history) that the trie structure
discarded when it only stored assistant marker tokens as prompt_tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Node:
    """A single turn in a multi-turn conversation tree.

    Each Node represents one assistant response turn, including its prompt
    context (all tokens from the beginning of the conversation up through
    this turn's response). Nodes are linked via node_id/parent_node_id and
    grouped into episodes via episode_id.

    node_id is the globally unique interaction ID (UUID string from the
    inference engine). query_id is
    set as metadata. advantages/returns are set by the advantage computer.
    """

    # Core sequence (full turn: prompt + response)
    input_ids: list[int]
    loss_mask: list[int]  # 0=prompt, 1=response
    logprobs: list[float]  # full sequence (0.0 on prompt positions)
    versions: list[int]  # policy version (-1 on prompt)

    # Tree structure
    node_id: str = ""  # globally unique interaction ID (UUID from inference engine)
    parent_node_id: str | None = None  # parent interaction ID (None for root)
    episode_id: str = ""  # groups turns into a trajectory path
    turn_idx: int = 0  # 1-based turn position within episode
    query_id: str = ""  # dataset query identifier
    train_id: str = ""  # training run that trained this node; "" means untrained

    # Reward
    outcome_reward: float = 0.0

    # Tree-computed advantages/returns (set by TreeAdvantageComputer)
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None


def _find_turn_boundaries(
    loss_mask: list[int],
) -> tuple[list[int], list[int]]:
    """Scan loss_mask for 0→1 and 1→0 transitions.

    Returns (turn_response_starts, turn_response_ends) where each pair
    defines a half-open range [start, end) of response tokens.
    """
    starts: list[int] = []
    ends: list[int] = []
    in_response = False
    for i, v in enumerate(loss_mask):
        if v == 1 and not in_response:
            starts.append(i)
            in_response = True
        elif v == 0 and in_response:
            ends.append(i)
            in_response = False
    if in_response:
        ends.append(len(loss_mask))
    return starts, ends


def _response_span(loss_mask: list[int]) -> tuple[int, int]:
    """Return (start, end) of the current response region in loss_mask."""
    starts, ends = _find_turn_boundaries(loss_mask)
    if starts:
        return starts[-1], ends[-1]
    return 0, 0


def _optional_tensor_field(
    traj: dict[str, Any],
    key: str,
    values: list | None,
    dtype: torch.dtype,
    start: int = 0,
    end: int | None = None,
    loss_mask: list[int] | None = None,
) -> None:
    """Add an unsqueezed tensor to traj if values is not None.

    Supports full-sequence rows, all-response rows, and current-response rows.
    If start/end are given and the field has full-sequence rows, slice by
    absolute token offsets. Response-only and current-response fields are
    already relative and are exported as-is.
    """
    if values is not None:
        sliced = values
        if end is not None:
            value_len = len(values)
            if loss_mask is not None:
                seq_len = len(loss_mask)
                starts, ends = _find_turn_boundaries(loss_mask)
                total_response_len = sum(e - s for s, e in zip(starts, ends))
                current_response_len = (ends[-1] - starts[-1]) if starts else 0
                if value_len == seq_len:
                    sliced = values[start:end]
                elif value_len in (total_response_len, current_response_len):
                    sliced = values
                else:
                    sliced = values[start:end]
            else:
                sliced = values[start:end]
        traj[key] = torch.tensor(sliced, dtype=dtype).unsqueeze(0)


def _node_to_tensor_dict(
    node: Node, query_id: str, node_id: str, num_turns_in_episode: int = 1
) -> dict[str, Any]:
    """Convert a single Node to a tensor dict with shape [1, seq_len]."""
    seq_len = len(node.input_ids)
    traj: dict[str, Any] = {
        "input_ids": torch.tensor(node.input_ids, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.tensor(node.loss_mask, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(node.logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(node.versions, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "rewards": torch.tensor(node.outcome_reward, dtype=torch.float32).unsqueeze(0),
        "query_id": [query_id],
        "node_id": [node_id],
        "episode_id": [node.episode_id or ""],
        "turn_idx": [node.turn_idx or 0],
    }
    # Response-only fields: extract response portion from full sequence
    resp_start, resp_end = _response_span(node.loss_mask)
    _optional_tensor_field(
        traj,
        "topk_ids",
        node.topk_ids,
        torch.int32,
        resp_start,
        resp_end,
        node.loss_mask,
    )
    _optional_tensor_field(
        traj,
        "topk_logp",
        node.topk_logp,
        torch.float32,
        resp_start,
        resp_end,
        node.loss_mask,
    )
    _optional_tensor_field(
        traj,
        "distill_reward",
        node.distill_reward,
        torch.float32,
        resp_start,
        resp_end,
        node.loss_mask,
    )
    _optional_tensor_field(
        traj,
        "teacher_logp",
        node.teacher_logp,
        torch.float32,
        resp_start,
        resp_end,
        node.loss_mask,
    )
    # Derived from logprobs for response tokens
    if resp_end > resp_start:
        traj["logp"] = torch.tensor(
            node.logprobs[resp_start:resp_end], dtype=torch.float32
        ).unsqueeze(0)
    # Carry advantages if set by advantage computer
    if node.advantages is not None:
        traj["advantages"] = (
            node.advantages.unsqueeze(0)
            if node.advantages.dim() == 1
            else node.advantages
        )
    if node.returns is not None:
        traj["returns"] = (
            node.returns.unsqueeze(0) if node.returns.dim() == 1 else node.returns
        )
    # Turn metadata
    traj["_turn_id"] = node.node_id
    if node.parent_node_id is not None:
        traj["_parent_turn_id"] = node.parent_node_id
    traj["_turn_reward"] = node.outcome_reward
    traj["_outcome_reward"] = node.outcome_reward
    traj["_episode_idx"] = 0
    traj["_turn_idx_in_episode"] = node.turn_idx
    traj["_num_turns_in_episode"] = num_turns_in_episode
    return traj


class MCTSTreeStore:
    """Flat trajectory store with MCTS statistics.

    Manages multiple trajectories per query, tracks MCTS statistics
    (visit counts, Q-values) per trajectory (keyed by node_id, a string
    interaction ID), and provides cache-aware loading of untrained
    trajectories.
    """

    def __init__(self) -> None:
        self.trajectories: dict[str, list[Node]] = {}
        self._node_id_to_key: dict[str, tuple[str, int]] = {}
        self._query_node_ids: dict[str, list[str]] = {}

        self._visit_counts: dict[str, int] = {}
        self._total_values: dict[str, float] = {}
        self._q_values: dict[str, float] = {}

        self.current_train_id: str = os.environ.get("TRAIN_ID", "")
        self._rewards: dict[str, float] = {}

        # Tree-search episode metadata
        self._turn_nodes: dict[str, str] = {}  # turn_id → node_id
        self._normalized_advantages: dict[str, float] = {}
        self._normalized_returns: dict[str, float] = {}

    def _backup(self, node_id: str, reward: float) -> None:
        """Update MCTS stats for a single trajectory."""
        self._visit_counts[node_id] = self._visit_counts.get(node_id, 0) + 1
        self._total_values[node_id] = self._total_values.get(node_id, 0.0) + reward
        self._q_values[node_id] = (
            self._total_values[node_id] / self._visit_counts[node_id]
        )

    def _insert_single(self, query_id: str, node: Node) -> str:
        """Insert a single Node, reading node_id from the node itself.

        Supports both Node dataclass instances and plain dicts (the
        latter arriving when tree search patches aren't active on the
        remote engine and _convert_trajs_to_nodes hasn't converted yet).
        """
        node_id = node.node_id if isinstance(node, Node) else node.get("node_id", "")
        if not node_id:
            raise ValueError(
                "Node must have a non-empty node_id (interaction_id) before insert"
            )

        idx = len(self.trajectories.setdefault(query_id, []))
        self.trajectories[query_id].append(node)
        self._node_id_to_key[node_id] = (query_id, idx)
        self._query_node_ids.setdefault(query_id, []).append(node_id)

        if isinstance(node, dict):
            node["node_id"] = node_id
            node["query_id"] = query_id
            outcome_reward = node.get("outcome_reward", node.get("reward", 0.0))
        else:
            node.node_id = node_id
            node.query_id = query_id
            outcome_reward = node.outcome_reward

        self._backup(node_id, outcome_reward)
        self._rewards[node_id] = outcome_reward

        return node_id

    def insert_batch(self, trajectories: list[Node]) -> None:
        """Insert Node trajectories into the store.

        Each Node is inserted directly. Nodes that already have a
        node_id assigned (loaded from cache) are skipped.
        """
        for node in trajectories:
            existing_id = getattr(node, "node_id", "")
            if existing_id != "" and existing_id in self._node_id_to_key:
                continue
            query_id = (
                node.get("query_id", "")
                if isinstance(node, dict)
                else (node.query_id or "")
            )
            self._insert_single(query_id, node)

    def set_trained(self, node_id: str, trained: bool = True) -> None:
        """Stamp the node with current_train_id to mark it as trained."""
        if not trained:
            return
        key = self._node_id_to_key.get(node_id)
        if key is None:
            return
        query_id, idx = key
        node = self.trajectories[query_id][idx]
        if isinstance(node, dict):
            node["train_id"] = self.current_train_id
        else:
            node.train_id = self.current_train_id

    def is_trained(self, node_id: str) -> bool:
        """A node is trained if its train_id matches the current run's train_id.

        An empty train_id means the node has never been trained, so it
        is always considered untrained regardless of current_train_id.
        """
        key = self._node_id_to_key.get(node_id)
        if key is None:
            return False
        query_id, idx = key
        node = self.trajectories[query_id][idx]
        if isinstance(node, dict):
            train_id = node.get("train_id", "")
        else:
            train_id = node.train_id
        return bool(train_id) and train_id == self.current_train_id

    def get_reward(self, node_id: str) -> float:
        return self._rewards.get(node_id, 0.0)

    def get_q_value(self, node_id: str) -> float:
        return self._q_values.get(node_id, 0.0)

    def set_normalized_advantage(self, node_id: str, value: float) -> None:
        self._normalized_advantages[node_id] = value

    def get_normalized_advantage(self, node_id: str, default: float = 0.0) -> float:
        return self._normalized_advantages.get(node_id, default)

    def has_normalized_advantage(self, node_id: str) -> bool:
        return node_id in self._normalized_advantages

    def set_normalized_return(self, node_id: str, value: float) -> None:
        self._normalized_returns[node_id] = value

    def get_normalized_return(self, node_id: str, default: float = 0.0) -> float:
        return self._normalized_returns.get(node_id, default)

    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self._query_node_ids:
            return 0
        return sum(
            1
            for node_id in self._query_node_ids[query_id]
            if not self.is_trained(node_id)
        )

    def get_untrained_episode_count(self, query_id: str) -> int:
        """Count untrained episodes for a query.

        An episode is untrained if any of its nodes is untrained
        (train_id != current_train_id).
        """
        if query_id not in self._query_node_ids:
            return 0
        episode_has_untrained: dict[str, bool] = {}
        for node_id in self._query_node_ids[query_id]:
            key = self._node_id_to_key.get(node_id)
            if key is None:
                continue
            qid, idx = key
            node = self.trajectories[qid][idx]
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_has_untrained:
                episode_has_untrained[ep_id] = False
            if not self.is_trained(node_id):
                episode_has_untrained[ep_id] = True
        return sum(1 for v in episode_has_untrained.values() if v)

    def load_untrained_episodes(self, query_id: str, n_episodes: int) -> list[Node]:
        """Load nodes from up to n_episodes untrained episodes.

        Returns all nodes belonging to the first n_episodes untrained
        episodes (in insertion order). An episode is untrained if any
        of its nodes is untrained.
        """
        if query_id not in self._query_node_ids:
            return []
        # Build episode_id → list of (node_id, query_id, idx) in insertion order
        episode_nodes: dict[str, list[tuple[str, str, int]]] = {}
        episode_order: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            key = self._node_id_to_key.get(node_id)
            if key is None:
                continue
            qid, idx = key
            node = self.trajectories[qid][idx]
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_nodes:
                episode_nodes[ep_id] = []
                episode_order.append(ep_id)
            episode_nodes[ep_id].append((node_id, qid, idx))
        # Select up to n_episodes untrained episodes
        selected: list[Node] = []
        count = 0
        for ep_id in episode_order:
            if count >= n_episodes:
                break
            # Check if any node in this episode is untrained
            is_untrained = False
            for node_id, qid, idx in episode_nodes[ep_id]:
                if not self.is_trained(node_id):
                    is_untrained = True
                    break
            if not is_untrained:
                continue
            count += 1
            for node_id, qid, idx in episode_nodes[ep_id]:
                selected.append(self.trajectories[qid][idx])
        return selected

    def get_untrained_node_ids(self, query_id: str, n_samples: int) -> list[str]:
        if query_id not in self._query_node_ids:
            return []
        result: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            if not self.is_trained(node_id):
                result.append(node_id)
                if len(result) >= n_samples:
                    break
        return result

    def load_trajectories(self, query_id: str, n_samples: int) -> list[Node]:
        """Load untrained trajectories as Node objects.

        Returns per-turn Node objects. Callers can:
        - Read Node attributes directly for advantage computation
        - Convert to tensor dicts via _node_to_tensor_dict() for training

        Each Node carries query_id and node_id set during
            insertion (accessible as regular attributes).
        """
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_node_ids(query_id, n_samples)
        result: list[Node] = []
        for node_id in untrained_ids:
            qid, idx = self._node_id_to_key[node_id]
            node = self.trajectories[qid][idx]
            result.append(node)
        return result

    def mark_episodes_trained(self, episode_ids: set[str]) -> None:
        """Set train_id based on episode IDs.

        Nodes whose episode_id is in the given set are stamped with
        current_train_id. All other nodes have train_id cleared.
        Episode IDs not present in the store are silently ignored.
        """
        for query_id, records in self.trajectories.items():
            for node in records:
                if isinstance(node, dict):
                    nid_val = node.get("episode_id", "")
                else:
                    nid_val = node.episode_id
                if nid_val in episode_ids:
                    if isinstance(node, dict):
                        node["train_id"] = self.current_train_id
                    else:
                        node.train_id = self.current_train_id
                else:
                    if isinstance(node, dict):
                        node["train_id"] = ""
                    else:
                        node.train_id = ""

    def clear(self) -> None:
        """Reset all trajectories, stats, and indices."""
        self.trajectories.clear()
        self._node_id_to_key.clear()
        self._query_node_ids.clear()
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._rewards.clear()
        self._turn_nodes.clear()
        self._normalized_advantages.clear()
        self._normalized_returns.clear()

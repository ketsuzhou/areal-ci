#!/usr/bin/env python3
# customized_areal/tree_search/core/tree_store.py
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

from customized_areal.tree_search.agents.execution_dag import SuperNode


def _lazy_torch():
    """Import and return torch on first use.

    tree_store stays importable without torch. Tensor-consuming code paths
    (_node_to_tensor_dict, _optional_tensor_field) call this at first use;
    a missing torch surfaces as ImportError at the call site, not at module
    load — which is the desired behavior since torch is required for any
    tensor path but not for importing the data model.
    """
    import torch

    return torch


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
    parent_node_id: str | None = None  # primary causal parent (None for root)
    # Additional causal parents at DAG join points (fan-in). None/empty for the
    # common single-parent case; set by SuperNodeAssembler when a segment's
    # first turn has multiple incoming DELEGATION/COMPLETION edges. Reward
    # backup follows the primary parent AND these, so every incoming branch of
    # a join receives credit (no fan-in credit loss).
    extra_parent_node_ids: list[str] | None = None
    episode_id: str = ""  # groups turns into a trajectory path
    turn_idx: int = 0  # 1-based turn position within episode
    query_id: str = ""  # dataset query identifier
    train_id: str = ""  # training run that trained this node; "" means untrained
    discarded: bool = False  # excluded from future cache reuse without marking trained
    task_id: str = ""  # TPFC backend task that produced this node
    entropy_stats: dict[str, Any] | None = None
    need_branch: bool = False
    # Terminal env-dispatch environment ID for this turn (the branch frontier).
    # Populated from backend per-turn metadata ``env_id``; a branch is created
    # via ``env_dispatch(mode="branch", env_id=node.env_id)``.
    env_id: str | None = None

    # Reward
    outcome_reward: float = 0.0

    # Per-node credit from DAG reward backup (Phase 3). When set (not None),
    # TreeAdvantageComputer uses this for per-node GRPO normalization instead
    # of the flat episode-level outcome_reward broadcast.
    credit: float | None = None

    # Generative-critic state value v_phi(s_t) for the partial solution through
    # this turn. Written by the critic value client; consumed by
    # GAEAdvantageComputer. 0.0 when the critic is disabled.
    value: float = 0.0

    # Critic's own uncertainty about ``value``: the categorical variance of the
    # critic's score distribution. Used as var_theta in the variance-aware
    # hybrid blend. 0.0 when the critic is one-hot or disabled.
    value_variance: float = 0.0

    # Tree-computed advantages/returns (set by TreeAdvantageComputer or
    # GAEAdvantageComputer). Typed Any (not torch.Tensor) so this module imports
    # cleanly without torch; tensor construction is deferred to lazy import in
    # _node_to_tensor_dict.
    advantages: Any = None
    returns: Any = None

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
    guidance: dict[int, str] | None = None  # turn_idx → guidance text for leaf nodes


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
    dtype: Any,
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
        torch = _lazy_torch()
        traj[key] = torch.tensor(sliced, dtype=dtype).unsqueeze(0)


def _node_to_tensor_dict(
    node: Node,
    query_id: str,
    node_id: str,
    max_tokens: int = 0,
    loss_mode: str | None = None,
) -> dict[str, Any]:
    """Convert a single Node to a tensor dict with shape [1, seq_len].

    If max_tokens > 0, the sequence is truncated to the last max_tokens
    tokens before conversion (full-sequence fields sliced, response-aligned
    fields trimmed to remaining output positions).
    """
    torch = _lazy_torch()

    input_ids = node.input_ids
    logprobs = node.logprobs
    loss_mask = node.loss_mask
    versions = node.versions
    topk_ids = node.topk_ids
    topk_logp = node.topk_logp
    teacher_logp = node.teacher_logp

    if max_tokens > 0 and len(input_ids) > max_tokens:
        truncate_len = len(input_ids) - max_tokens
        input_ids = input_ids[truncate_len:]
        logprobs = logprobs[truncate_len:]
        loss_mask = loss_mask[truncate_len:]
        versions = versions[truncate_len:]

        output_kept = sum(1 for m in loss_mask if m == 1)

        def _trim(field: list | None) -> list | None:
            if field is not None and output_kept < len(field):
                return field[-output_kept:] if output_kept > 0 else []
            return field

        topk_ids = _trim(topk_ids)
        topk_logp = _trim(topk_logp)
        teacher_logp = _trim(teacher_logp)

    seq_len = len(input_ids)
    traj: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "rewards": torch.tensor(node.outcome_reward, dtype=torch.float32).unsqueeze(0),
    }
    # Response-only fields: extract response portion from full sequence
    resp_start, resp_end = _response_span(loss_mask)
    resp_len = resp_end - resp_start

    # topk_ids is always emitted for key consistency across the batch.
    # teacher_logp is only emitted when loss_mode != "grpo".
    if topk_ids is not None:
        _optional_tensor_field(
            traj,
            "topk_ids",
            topk_ids,
            torch.int32,
            resp_start,
            resp_end,
            loss_mask,
        )
    else:
        # -1 sentinel: _prepare_multi_candidate_labels detects this and
        # fills those positions with the actual next token instead.
        traj["topk_ids"] = torch.full((1, resp_len, 1), -1, dtype=torch.int32)

    if loss_mode != "grpo":
        if teacher_logp is not None:
            _optional_tensor_field(
                traj,
                "teacher_logp",
                teacher_logp,
                torch.float32,
                resp_start,
                resp_end,
                loss_mask,
            )
        else:
            traj["teacher_logp"] = torch.zeros(1, resp_len, 1, dtype=torch.float32)

    # Carry advantages if set by advantage computer
    if node.advantages is not None:
        traj["advantages"] = (
            node.advantages.unsqueeze(0)
            if node.advantages.dim() == 1
            else node.advantages
        )
    return traj


class MCTSTreeStore:
    """Unified store: SuperNodes (segments) hold DAG topology, comm-event
    provenance, and team env snapshots; Nodes hold all MCTS stats and the
    unified causal parent_node_id chain.

    Reward backup: Node-level only (parent_node_id encodes the full DAG
    causal order across agents and segments).

    Node payloads may be either ``Node`` objects (local pipeline) or dicts
    (remote-engine wire format). Methods that read node fields branch on
    ``isinstance(node, dict)`` for this reason.
    """

    def __init__(self) -> None:
        # Primary storage: query_id -> SuperNodes in insertion order.
        self.trajectories: dict[str, list[SuperNode]] = {}
        # SuperNode-level index: SuperNode UUID -> (query_id, idx_in_trajectories)
        self._super_id_to_key: dict[str, tuple[str, int]] = {}
        # Node-level index: node_id -> (super_node_id, idx_in_super_node.nodes)
        self._node_id_to_super: dict[str, tuple[str, int]] = {}
        self._query_node_ids: dict[str, list[str]] = {}

        # Per-Node MCTS stats (unchanged).
        self._visit_counts: dict[str, int] = {}
        self._total_values: dict[str, float] = {}
        self._q_values: dict[str, float] = {}
        self._sum_sq_values: dict[str, float] = {}

        self.current_train_id: str = os.environ.get("TRAIN_ID", "")
        self._rewards: dict[str, float] = {}

        self._turn_nodes: dict[str, str] = {}
        self._normalized_advantages: dict[str, float] = {}
        self._normalized_returns: dict[str, float] = {}
        self._values: dict[str, float] = {}
        self._value_variances: dict[str, float] = {}
        self._judge_scores: dict[str, list[float]] = {}

    # -- SuperNode / Node lookup -----------------------------------------

    def get_super_node(self, super_node_id: str):
        """Return the indexed SuperNode (or None if absent)."""
        key = self._super_id_to_key.get(super_node_id)
        if key is None:
            return None
        query_id, idx = key
        return self.trajectories[query_id][idx]

    def get_node(self, node_id: str):
        """Return the indexed Node (or None) by walking into its SuperNode."""
        key = self._node_id_to_super.get(node_id)
        if key is None:
            return None
        super_node_id, idx_in_nodes = key
        super_node = self.get_super_node(super_node_id)
        if super_node is None:
            return None
        return super_node.nodes[idx_in_nodes]

    def _node_parent_ids(self, node_id: str) -> list[str]:
        """Return all causal parents of an indexed node (primary + extras).

        The primary ``parent_node_id`` plus any ``extra_parent_node_ids`` set by
        the assembler at DAG join points. Empty list for the root / unknown
        node. Backup follows every parent, so fan-in joins credit all incoming
        branches (the ``visited`` guard prevents double-counting a shared
        ancestor within a single backup walk).
        """
        node = self.get_node(node_id)
        if node is None:
            return []
        if isinstance(node, dict):
            primary = node.get("parent_node_id")
            extras = node.get("extra_parent_node_ids") or ()
        else:
            primary = getattr(node, "parent_node_id", None)
            extras = getattr(node, "extra_parent_node_ids", None) or ()
        parents: list[str] = []
        if primary:
            parents.append(primary)
        parents.extend(p for p in extras if p)
        return parents

    # -- MCTS backup (unchanged algorithm; walks parent_node_id) ---------

    def _backup_node(self, node_id: str, reward: float) -> None:
        """Add one Monte-Carlo sample (``reward``) to a single node's stats."""
        self._visit_counts[node_id] = self._visit_counts.get(node_id, 0) + 1
        self._total_values[node_id] = self._total_values.get(node_id, 0.0) + reward
        self._sum_sq_values[node_id] = (
            self._sum_sq_values.get(node_id, 0.0) + reward * reward
        )
        self._q_values[node_id] = (
            self._total_values[node_id] / self._visit_counts[node_id]
        )

    def _backup_path(self, terminal_node_id: str, reward: float) -> None:
        """Propagate one episode's return root-ward along the parent DAG.

        Walks ``parent_node_id`` plus ``extra_parent_node_ids`` across SuperNode
        and agent boundaries (the unified causal chain, generalized to a DAG at
        fan-in joins). A ``visited`` guard makes the walk robust to malformed
        cycles and ensures each ancestor is credited exactly once per episode
        even when several branches share it: on a cycle the walk silently stops
        (no raise) and already-visited nodes keep their accumulated reward.
        """
        visited: set[str] = set()
        stack: list[str] = [terminal_node_id]
        while stack:
            current = stack.pop()
            if current in visited or current not in self._node_id_to_super:
                continue
            visited.add(current)
            self._backup_node(current, reward)
            for parent_id in self._node_parent_ids(current):
                if parent_id not in visited:
                    stack.append(parent_id)

    def backup_episode_terminal(self, terminal_node_id: str, reward: float) -> None:
        """Public entry: root-ward backup of one episode's terminal return.

        Walks parent_node_id from terminal_node_id across SuperNode/agent
        boundaries (causal flattening).
        """
        self._backup_path(terminal_node_id, float(reward))

    def backup_path_returns(
        self, terminal_node_id: str, returns_by_node_id: dict[str, float]
    ) -> None:
        """Root-ward backup assigning each node on the path its own return-to-go.

        Walks the parent DAG (``parent_node_id`` + ``extra_parent_node_ids``);
        the ``visited`` guard credits each node once per call.
        """
        visited: set[str] = set()
        stack: list[str] = [terminal_node_id]
        while stack:
            current = stack.pop()
            if current in visited or current not in self._node_id_to_super:
                continue
            visited.add(current)
            g = returns_by_node_id.get(current)
            if g is not None:
                self._backup_node(current, float(g))
            for parent_id in self._node_parent_ids(current):
                if parent_id not in visited:
                    stack.append(parent_id)

    # -- Insertion -------------------------------------------------------

    def insert_super_batch(
        self, supers: list[SuperNode], backup: bool = True, query_id: str = ""
    ) -> None:
        """Insert a batch of SuperNodes under ``query_id``.

        Indexes each SuperNode and each Node inside it. When ``backup`` is
        True, runs a root-ward backup per episode among freshly inserted nodes
        (same algorithm as the old insert_batch, but walking into SuperNodes).
        """
        inserted_node_ids: list[str] = []
        for super_node in supers:
            # Index the SuperNode.
            super_id = super_node.node_id
            if not super_id:
                raise ValueError(
                    "SuperNode must have a non-empty node_id before insert"
                )
            if super_id in self._super_id_to_key:
                continue  # idempotent
            qid = query_id or getattr(super_node, "query_id", "") or ""
            idx = len(self.trajectories.setdefault(qid, []))
            self.trajectories[qid].append(super_node)
            self._super_id_to_key[super_id] = (qid, idx)
            # Index each Node inside the SuperNode.
            for node_idx, node in enumerate(super_node.nodes):
                node_id = (
                    node.get("node_id", "") if isinstance(node, dict) else node.node_id
                )
                if not node_id:
                    continue
                if node_id in self._node_id_to_super:
                    continue  # idempotent across supers
                self._node_id_to_super[node_id] = (super_id, node_idx)
                self._query_node_ids.setdefault(qid, []).append(node_id)
                inserted_node_ids.append(node_id)
                # Record the node's own reward.
                if isinstance(node, dict):
                    outcome_reward = node.get("outcome_reward", node.get("reward", 0.0))
                else:
                    outcome_reward = node.outcome_reward
                self._rewards[node_id] = outcome_reward
        if backup:
            self._backup_inserted_episodes(inserted_node_ids)

    def _backup_inserted_episodes(self, node_ids: list[str]) -> None:
        """Run a root-ward backup once per episode among freshly inserted nodes.

        Nodes that carry an ``episode_id`` are grouped by episode and backed up
        from the highest-turn node (the episode terminal). Nodes WITHOUT an
        ``episode_id`` (e.g. multi-agent SuperNodes assembled from Multica
        segments, where credit flows along the cross-agent parent DAG rather
        than a single episode chain) are backed up once per *leaf* -- a node
        that is not a parent of any other freshly-inserted node. This avoids
        the previous per-node backup, which walked the full parent chain from
        every node and inflated shared ancestors' visit counts.
        """
        episode_terminal: dict[str, tuple[int, str, float]] = {}
        no_episode_ids: list[str] = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node is None:
                continue
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "") or ""
                turn_idx = int(node.get("turn_idx", 0) or 0)
                reward = float(
                    node.get("outcome_reward", node.get("reward", 0.0)) or 0.0
                )
            else:
                ep_id = node.episode_id or ""
                turn_idx = int(node.turn_idx or 0)
                reward = float(node.outcome_reward or 0.0)
            if not ep_id:
                no_episode_ids.append(node_id)
                continue
            prev = episode_terminal.get(ep_id)
            if prev is None or turn_idx >= prev[0]:
                episode_terminal[ep_id] = (turn_idx, node_id, reward)
        for _turn_idx, terminal_id, reward in episode_terminal.values():
            self._backup_path(terminal_id, reward)
        if no_episode_ids:
            # A leaf is a node that no other empty-episode node points to as a
            # parent; back up once from each leaf so shared ancestors get one
            # sample per distinct terminal (not one per node on the path).
            no_ep_set = set(no_episode_ids)
            has_child = set()
            for node_id in no_episode_ids:
                for parent_id in self._node_parent_ids(node_id):
                    if parent_id in no_ep_set:
                        has_child.add(parent_id)
            for node_id in no_episode_ids:
                if node_id not in has_child:
                    self._backup_path(node_id, self._rewards.get(node_id, 0.0))

    # -- Per-Node accessors (unchanged signatures) -----------------------

    def set_trained(self, node_id: str, trained: bool = True) -> None:
        if not trained:
            return
        node = self.get_node(node_id)
        if node is None:
            return
        if isinstance(node, dict):
            node["train_id"] = self.current_train_id
        else:
            node.train_id = self.current_train_id

    def set_discarded(self, node_id: str, discarded: bool = True) -> None:
        node = self.get_node(node_id)
        if node is None:
            return
        if isinstance(node, dict):
            node["discarded"] = discarded
        else:
            node.discarded = discarded

    def is_discarded(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        if isinstance(node, dict):
            return bool(node.get("discarded", False))
        return node.discarded

    def is_trained(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        if isinstance(node, dict):
            train_id = node.get("train_id", "")
        else:
            train_id = node.train_id
        return bool(train_id) and train_id == self.current_train_id

    def get_reward(self, node_id: str) -> float:
        return self._rewards.get(node_id, 0.0)

    def get_q_value(self, node_id: str) -> float:
        return self._q_values.get(node_id, 0.0)

    def get_visit_count(self, node_id: str) -> int:
        return self._visit_counts.get(node_id, 0)

    def get_total_value(self, node_id: str) -> float:
        return self._total_values.get(node_id, 0.0)

    def get_sum_sq_value(self, node_id: str) -> float:
        return self._sum_sq_values.get(node_id, 0.0)

    def get_loo_value_and_variance(
        self, node_id: str, excluded_reward: float
    ) -> tuple[float, float, int]:
        """Leave-one-out MC value, variance-of-the-mean, and LOO sample size.

        Returns (loo_mean, var_of_mean, n_loo). var_of_mean is the larger of:
          - the sample-variance-of-the-mean after removing ``excluded_reward``
            (clamped to >= 0; divided by n_loo for the mean's variance), and
          - a Beta(a, b) prior floor where a = s' + 1, b = (n_loo - s') + 1,
            s' = sum of rewards excluding the LOO sample (clamped to [0, n_loo]).
            This floor prevents zero-variance estimates when all LOO samples agree.
        n_loo < 2 returns (0.0, -1.0, max(n_loo, 0)) -- not enough samples.
        """
        n = self._visit_counts.get(node_id, 0)
        n_loo = n - 1
        if n_loo < 2:
            return 0.0, -1.0, max(n_loo, 0)
        total = self._total_values.get(node_id, 0.0)
        sum_sq = self._sum_sq_values.get(node_id, 0.0)
        s_prime = total - excluded_reward
        q_prime = sum_sq - excluded_reward * excluded_reward
        loo_mean = s_prime / n_loo
        loo_var = (q_prime - s_prime * s_prime / n_loo) / (n_loo - 1)
        if loo_var < 0.0:
            loo_var = 0.0
        var_mc = loo_var / n_loo
        s_clamped = min(max(s_prime, 0.0), float(n_loo))
        a = s_clamped + 1.0
        b = (n_loo - s_clamped) + 1.0
        nn = a + b
        var_floor = (a * b) / (nn * nn * (nn + 1.0))
        if var_floor > var_mc:
            var_mc = var_floor
        return loo_mean, var_mc, n_loo

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

    def set_value(self, node_id: str, value: float) -> None:
        self._values[node_id] = value

    def get_value(self, node_id: str, default: float = 0.0) -> float:
        return self._values.get(node_id, default)

    def has_value(self, node_id: str) -> bool:
        return node_id in self._values

    def set_value_variance(self, node_id: str, variance: float) -> None:
        self._value_variances[node_id] = variance

    def get_value_variance(self, node_id: str, default: float = 0.0) -> float:
        return self._value_variances.get(node_id, default)

    def add_judge_score(self, node_id: str, score: float) -> None:
        self._judge_scores.setdefault(node_id, []).append(float(score))

    def get_judge_scores(self, node_id: str) -> list[float]:
        return list(self._judge_scores.get(node_id, []))

    def get_mean_judge_score(self, node_id: str) -> float | None:
        scores = self._judge_scores.get(node_id)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self._query_node_ids:
            return 0
        return sum(
            1
            for node_id in self._query_node_ids[query_id]
            if not self.is_trained(node_id) and not self.is_discarded(node_id)
        )

    def get_untrained_episode_count(self, query_id: str) -> int:
        if query_id not in self._query_node_ids:
            return 0
        episode_has_untrained: dict[str, bool] = {}
        for node_id in self._query_node_ids[query_id]:
            node = self.get_node(node_id)
            if node is None:
                continue
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_has_untrained:
                episode_has_untrained[ep_id] = False
            if not self.is_trained(node_id) and not self.is_discarded(node_id):
                episode_has_untrained[ep_id] = True
        return sum(1 for v in episode_has_untrained.values() if v)

    def load_untrained_episodes(self, query_id: str, n_episodes: int) -> list:
        if query_id not in self._query_node_ids:
            return []
        episode_nodes: dict[str, list[str]] = {}
        episode_order: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            node = self.get_node(node_id)
            if node is None:
                continue
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_nodes:
                episode_nodes[ep_id] = []
                episode_order.append(ep_id)
            episode_nodes[ep_id].append(node_id)
        selected: list = []
        count = 0
        for ep_id in episode_order:
            if count >= n_episodes:
                break
            is_untrained = False
            for node_id in episode_nodes[ep_id]:
                if not self.is_trained(node_id) and not self.is_discarded(node_id):
                    is_untrained = True
                    break
            if not is_untrained:
                continue
            count += 1
            for node_id in episode_nodes[ep_id]:
                node = self.get_node(node_id)
                if node is not None:
                    selected.append(node)
        return selected

    def get_untrained_node_ids(self, query_id: str, n_samples: int) -> list[str]:
        if query_id not in self._query_node_ids:
            return []
        result: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            if not self.is_trained(node_id) and not self.is_discarded(node_id):
                result.append(node_id)
                if len(result) >= n_samples:
                    break
        return result

    def load_trajectories(self, query_id: str, n_samples: int) -> list:
        if query_id not in self.trajectories:
            return []
        untrained_ids = self.get_untrained_node_ids(query_id, n_samples)
        result: list = []
        for node_id in untrained_ids:
            node = self.get_node(node_id)
            if node is not None:
                result.append(node)
        return result

    def mark_episodes_trained(self, episode_ids: set[str]) -> None:
        for query_id, supers in self.trajectories.items():
            for super_node in supers:
                for node in super_node.nodes:
                    if isinstance(node, dict):
                        ep_id = node.get("episode_id", "")
                    else:
                        ep_id = node.episode_id
                    if ep_id in episode_ids:
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
        self._super_id_to_key.clear()
        self._node_id_to_super.clear()
        self._query_node_ids.clear()
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._sum_sq_values.clear()
        self._rewards.clear()
        self._turn_nodes.clear()
        self._normalized_advantages.clear()
        self._normalized_returns.clear()
        self._values.clear()
        self._value_variances.clear()

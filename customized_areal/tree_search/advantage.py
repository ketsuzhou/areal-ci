# customized_areal/tree_search/advantage.py
from __future__ import annotations

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

from areal.utils import logging

GRPO_NORM_EPS = 1e-8

logger = logging.getLogger("TreeAdvantageComputer")


class TreeAdvantageComputer:
    """Replace GAE advantages with per-query GRPO-normalized outcome_rewards.

    Reads query_id and node_id from Node objects. Sets advantages
    and returns on the Node in-place.

    outcome_rewards are normalized within each query group (all episodes
    for the same query), producing zero-mean unit-variance values for both
    advantages and returns.
    """

    def __init__(self, tree_store: MCTSTreeStore, grpo_eps: float = GRPO_NORM_EPS):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps

    @staticmethod
    def _get_query_id(traj: Node) -> str | None:
        """Extract query_id from Node."""
        return traj.query_id or None

    def compute(self, trajectories: list[Node]) -> None:
        """Replace GAE advantages with per-query GRPO-normalized outcome_rewards.

        Both advantages and returns are set to the same per-query GRPO-normalized
        outcome_reward, broadcast across response positions via prompt mask.
        """
        # Collect unique (query_id → set of node_ids) and reward_per_node
        query_node_sets: dict[str, set[str]] = {}
        node_rewards: dict[str, float] = {}

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            nset = query_node_sets.setdefault(query_id, set())

            node_id = getattr(traj, "node_id", None)
            if node_id is not None:
                nset.add(node_id)
                node_rewards[node_id] = traj.outcome_reward

        # Per-query GRPO normalization of outcome_rewards for returns
        for query_id, node_id_set in query_node_sets.items():
            node_ids = list(node_id_set)
            rewards = [node_rewards[nid] for nid in node_ids]
            if len(rewards) < 2:
                for nid in node_ids:
                    self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = var_r**0.5
            for nid, r in zip(node_ids, rewards):
                self.tree_store.set_normalized_return(
                    nid, (r - mean_r) / (std_r + self.grpo_eps)
                )

        # Compute per-trajectory advantages and returns
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue

            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            mask = traj.loss_mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            norm_return = self.tree_store.get_normalized_return(node_id)
            traj.advantages = mask.float() * norm_return
            traj.returns = mask.float() * norm_return

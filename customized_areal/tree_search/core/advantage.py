# customized_areal/tree_search/advantage.py
from __future__ import annotations

import torch

from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node

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
        """Replace GAE advantages with per-episode GRPO-normalized outcome_rewards.

        Groups nodes by (query_id, episode_id). Each episode contributes one
        reward (all nodes in an episode share the same outcome_reward).
        GRPO normalization operates across episodes within each query group.
        The normalized return is broadcast to all response positions in
        every node of the episode.
        """
        # Build query_id → {episode_id → [node_ids]} and per-episode reward
        query_episodes: dict[str, dict[str, list[str]]] = {}
        episode_rewards: dict[str, float] = {}  # episode_id → reward

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            ep_id = getattr(traj, "episode_id", "") or node_id
            ep_map = query_episodes.setdefault(query_id, {})
            ep_map.setdefault(ep_id, []).append(node_id)
            # All nodes in an episode share the same outcome_reward
            if ep_id not in episode_rewards:
                episode_rewards[ep_id] = traj.outcome_reward

        # Per-query GRPO normalization of per-episode rewards
        for query_id, ep_map in query_episodes.items():
            ep_ids = list(ep_map.keys())
            rewards = [episode_rewards[eid] for eid in ep_ids]
            if len(rewards) < 2:
                for eid in ep_ids:
                    for nid in ep_map[eid]:
                        self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = var_r**0.5
            for eid, r in zip(ep_ids, rewards):
                norm_val = (r - mean_r) / (std_r + self.grpo_eps)
                for nid in ep_map[eid]:
                    self.tree_store.set_normalized_return(nid, norm_val)

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


gae_logger = logging.getLogger("GAEAdvantageComputer")


class GAEAdvantageComputer:
    """Generalized Advantage Estimation over episode turns.

    Each ``Node`` is one turn (one state ``s_t``). The generative critic
    supplies a bootstrapped state value ``v_phi(s_t)`` (read from ``Node.value``).
    Rewards are sparse: ``r_t = 0`` for intermediate turns and
    ``r_T = outcome_reward`` at the terminal (last) turn of the episode, with a
    zero terminal bootstrap ``v(s_{T+1}) = 0``.

    For each episode (grouped by ``(query_id, episode_id)`` and ordered by
    ``turn_idx``)::

        delta_t = r_t + gamma * v(s_{t+1}) - v(s_t)
        A_t     = delta_t + gamma * lam * A_{t+1}       (A_{T+1} = 0)
        ret_t   = A_t + v(s_t)

    ``A_t`` and ``ret_t`` are broadcast over each node's response positions
    (``loss_mask == 1``), mirroring ``TreeAdvantageComputer``.
    """

    def __init__(
        self,
        tree_store: MCTSTreeStore,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> None:
        self.tree_store = tree_store
        self.gamma = gamma
        self.lam = lam

    @staticmethod
    def _node_value(node: Node) -> float:
        return float(getattr(node, "value", 0.0) or 0.0)

    def _assign(self, node: Node, advantage: float, ret: float) -> None:
        mask = node.loss_mask
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        mask_f = mask.float()
        node.advantages = mask_f * advantage
        node.returns = mask_f * ret

    def compute(self, trajectories: list[Node]) -> None:
        """Compute GAE advantages/returns in-place on the given nodes."""
        # Group nodes into episodes. Nodes without an episode_id are treated as
        # standalone single-turn episodes keyed by node_id.
        episodes: dict[tuple[str, str], list[Node]] = {}
        for traj in trajectories:
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            query_id = traj.query_id or ""
            ep_id = traj.episode_id or node_id
            episodes.setdefault((query_id, ep_id), []).append(traj)

        for nodes in episodes.values():
            # Order turns ascending; ties broken by insertion order (stable).
            ordered = sorted(nodes, key=lambda n: getattr(n, "turn_idx", 0))
            n_turns = len(ordered)

            values = [self._node_value(n) for n in ordered]
            # Sparse reward: only the terminal turn carries outcome_reward.
            rewards = [0.0] * n_turns
            if n_turns > 0:
                rewards[-1] = float(ordered[-1].outcome_reward)

            advantages = [0.0] * n_turns
            next_adv = 0.0
            next_value = 0.0  # terminal bootstrap v(s_{T+1}) = 0
            for t in range(n_turns - 1, -1, -1):
                delta = rewards[t] + self.gamma * next_value - values[t]
                next_adv = delta + self.gamma * self.lam * next_adv
                advantages[t] = next_adv
                next_value = values[t]

            for n, adv, val in zip(ordered, advantages, values):
                ret = adv + val
                self._assign(n, adv, ret)
                if getattr(n, "node_id", None):
                    self.tree_store.set_normalized_advantage(n.node_id, adv)
                    self.tree_store.set_normalized_return(n.node_id, ret)

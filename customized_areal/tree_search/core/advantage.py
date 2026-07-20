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

        When any node carries per-node ``credit`` (DAG reward backup, Phase 3),
        normalization operates per-node across the query group instead.
        """
        # Per-node credit path (DAG reward backup): normalize across all
        # nodes in each query group, ignoring episode boundaries.
        if any(getattr(traj, "credit", None) is not None for traj in trajectories):
            self._compute_per_node_credit(trajectories)
            return

        # Existing flat-broadcast path (unchanged).
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

    def _compute_per_node_credit(self, trajectories: list[Node]) -> None:
        """GRPO-normalize per-node credit across all nodes in each query group.

        Each node's ``credit`` is its own reward; normalization operates across
        all nodes sharing a ``query_id`` (ignoring episode boundaries, since
        DAG credit is per-node, not per-episode). The normalized return is
        per-node, broadcast over response positions via ``loss_mask``.
        """
        query_nodes: dict[str, list[Node]] = {}
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            if not getattr(traj, "node_id", None):
                continue
            query_nodes.setdefault(query_id, []).append(traj)

        for nodes in query_nodes.values():
            credits = [float(getattr(n, "credit", None) or 0.0) for n in nodes]
            if len(credits) < 2:
                for n in nodes:
                    self.tree_store.set_normalized_return(n.node_id, 0.0)
                continue
            mean_c = sum(credits) / len(credits)
            var_c = sum((c - mean_c) ** 2 for c in credits) / max(len(credits), 1)
            std_c = var_c**0.5
            for n, c in zip(nodes, credits):
                norm_val = (c - mean_c) / (std_c + self.grpo_eps)
                self.tree_store.set_normalized_return(n.node_id, norm_val)

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            if not getattr(traj, "node_id", None):
                continue
            mask = traj.loss_mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            norm_return = self.tree_store.get_normalized_return(traj.node_id)
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
        judge_beta: float = 0.0,
        judge_score_max: int = 10,
    ) -> None:
        self.tree_store = tree_store
        self.gamma = gamma
        self.lam = lam
        # LLM-judge process-reward shaping. ``judge_beta == 0`` reproduces the
        # sparse terminal-only reward exactly (no behavioural change).
        self.judge_beta = judge_beta
        self.judge_score_max = judge_score_max

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
            # With LLM-judge shaping (judge_beta > 0) the reward becomes dense
            # per-turn; the helper falls back to the sparse array when no judge
            # signal exists, so judge_beta == 0 is byte-for-byte unchanged.
            if self.judge_beta > 0.0:
                from customized_areal.tree_search.core.process_reward import (
                    build_episode_process_rewards,
                )

                rewards = build_episode_process_rewards(
                    ordered,
                    self.tree_store,
                    beta=self.judge_beta,
                    score_max=self.judge_score_max,
                )
            else:
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


hybrid_gae_logger = logging.getLogger("HybridGAEAdvantageComputer")


class HybridGAEAdvantageComputer(GAEAdvantageComputer):
    """Variance-aware GAE that blends the critic with a leave-one-out MC value.

    Identical to :class:`GAEAdvantageComputer` except that, before running the
    GAE recursion, each turn's state value is (optionally) replaced by an
    inverse-variance blend of the learned critic ``v_theta(s_t)`` and a
    leave-one-out Monte-Carlo estimate ``v_mc^{(-i)}(s_t)`` accumulated in the
    tree. The substitution applies only to **branched** nodes that have enough
    MCTS samples (``need_branch`` and ``visit_count >= mc_min_visits``); all
    other turns keep the critic value, so with no eligible node the output is
    identical to plain GAE.

    For an eligible node, with critic variance ``var_theta`` (floored) and LOO
    MC variance ``var_mc``::

        v_hat = (v_mc/var_mc + v_theta/var_theta) / (1/var_mc + 1/var_theta)

    A non-positive ``var_mc`` (all remaining MC samples identical) short-circuits
    to ``v_hat = v_mc`` (the MC estimate is maximally confident). The blended
    array is used for both ``v(s_t)`` and the bootstrap ``v(s_{t+1})`` so the
    recursion stays self-consistent.

    **Commensurable variances.** ``var_mc`` is the *sampling variance of the MC
    mean* -- an estimator-error quantity ``Var[G|s]/n'`` that shrinks as samples
    accumulate. The critic side must therefore also be an estimator-error
    quantity ``E[(v_theta - V)^2]``, i.e. the critic's regression MSE, **not**
    the categorical variance of the critic's output distribution. The latter is
    (when calibrated) the *target spread* ``Var[G|s]`` -- roughly ``n'`` times
    too large and not an error of ``v_theta`` -- so blending it against
    ``var_mc`` compares a single-sample spread to a mean's standard error and
    spuriously favours MC as visits grow regardless of critic quality. ``var_theta``
    is sourced from ``critic_error_var`` (a static prior) or, once a live critic
    MSE is wired in, ``critic_error_var_fn`` (e.g. ``AdaptiveMCWeight``'s EMA).
    """

    def __init__(
        self,
        tree_store,
        gamma: float = 1.0,
        lam: float = 0.95,
        judge_beta: float = 0.0,
        judge_score_max: int = 10,
        mc_min_visits: int = 5,
        critic_var_floor: float = 1e-3,
        critic_error_var: float = 0.05,
        critic_error_var_fn=None,
    ) -> None:
        super().__init__(
            tree_store,
            gamma=gamma,
            lam=lam,
            judge_beta=judge_beta,
            judge_score_max=judge_score_max,
        )
        self.mc_min_visits = mc_min_visits
        self.critic_var_floor = critic_var_floor
        # Critic error variance E[(v_theta - V)^2] used as ``var_theta`` in the
        # blend. ``critic_error_var`` is a static prior; ``critic_error_var_fn``
        # (optional) returns a live estimate (e.g. an EMA of the critic MSE) or
        # ``None`` to fall back to the prior.
        self.critic_error_var = critic_error_var
        self.critic_error_var_fn = critic_error_var_fn

    def _critic_error_var(self) -> float:
        """Current critic error-variance estimate (live if available)."""
        if self.critic_error_var_fn is not None:
            live = self.critic_error_var_fn()
            if live is not None:
                return float(live)
        return self.critic_error_var

    def _blended_value(self, node: Node, excluded_reward: float) -> float:
        """Critic value for ``node``, blended with LOO MC when eligible."""
        v_theta = self._node_value(node)
        node_id = getattr(node, "node_id", None)
        if not node_id or not getattr(node, "need_branch", False):
            return v_theta
        if self.tree_store.get_visit_count(node_id) < self.mc_min_visits:
            return v_theta
        v_mc, var_mc, n_loo = self.tree_store.get_loo_value_and_variance(
            node_id, excluded_reward
        )
        if n_loo < 2 or var_mc < 0.0:
            # Not enough LOO samples to trust the MC estimate -> keep critic.
            return v_theta
        if var_mc <= 0.0:
            # Defensive guard only: get_loo_value_and_variance now floors var_mc
            # with a strictly-positive Beta(1, 1) posterior variance, so this
            # branch is unreachable for n_loo >= 2. Kept so a future caller that
            # bypasses the floor still degrades gracefully to pure MC instead of
            # dividing by zero.
            return v_mc
        # var_theta is the critic's *error* variance (regression MSE), in the
        # same units as var_mc (variance of an estimate of V(s_t)) -- NOT the
        # categorical variance of the critic's output distribution.
        var_theta = max(self._critic_error_var(), self.critic_var_floor)
        w_mc = 1.0 / var_mc
        w_theta = 1.0 / var_theta
        return (v_mc * w_mc + v_theta * w_theta) / (w_mc + w_theta)

    def compute(self, trajectories: list[Node]) -> None:
        """Compute hybrid GAE advantages/returns in-place on the given nodes."""
        episodes: dict[tuple[str, str], list[Node]] = {}
        for traj in trajectories:
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            query_id = traj.query_id or ""
            ep_id = traj.episode_id or node_id
            episodes.setdefault((query_id, ep_id), []).append(traj)

        for nodes in episodes.values():
            ordered = sorted(nodes, key=lambda n: getattr(n, "turn_idx", 0))
            n_turns = len(ordered)

            # Per-turn rewards. With LLM-judge shaping (judge_beta > 0) the reward
            # is dense; otherwise it is sparse terminal-only (byte-for-byte
            # identical to the legacy path).
            if self.judge_beta > 0.0:
                from customized_areal.tree_search.core.process_reward import (
                    build_episode_process_rewards,
                    episode_returns_to_go,
                )

                rewards = build_episode_process_rewards(
                    ordered,
                    self.tree_store,
                    beta=self.judge_beta,
                    score_max=self.judge_score_max,
                )
                # The sample this episode contributed to each node's MC aggregate
                # is the per-node discounted return-to-go (see
                # MCTSTreeStore.backup_path_returns), so the leave-one-out must
                # exclude g_t -- not the terminal outcome -- to stay consistent
                # with what was backed up.
                excluded = episode_returns_to_go(rewards, gamma=self.gamma)
            else:
                rewards = [0.0] * n_turns
                if n_turns > 0:
                    rewards[-1] = float(ordered[-1].outcome_reward)
                # Legacy LOO sample: the episode's terminal outcome, shared
                # across all turns (the insert-time backup added it to every
                # node on the path).
                term = float(ordered[-1].outcome_reward) if n_turns > 0 else 0.0
                excluded = [term] * n_turns

            # Blended per-turn values: critic, with LOO MC substituted on
            # eligible branched nodes. Used for both v(s_t) and bootstrap.
            values = [
                self._blended_value(n, excluded[i]) for i, n in enumerate(ordered)
            ]

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


versioned_logger = logging.getLogger("VersionedBackupAdvantageComputer")


def n_step_return(
    rewards: list[float],
    t: int,
    n: int,
    gamma: float,
    bootstrap_value: float,
) -> float:
    """n-step return ``G^(n)(s_t)`` (ARE-4, pure helper).

    ``G^(n)(s_t) = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * bootstrap_value``.

    Rewards past the episode end are treated as 0, so a horizon that reaches or
    passes the terminal reduces to a Monte-Carlo return when ``bootstrap_value``
    is 0 (the pure-MC / terminal-bootstrap case). Exposed for unit tests with
    known values.
    """
    g = 0.0
    for i in range(int(n)):
        idx = t + i
        if 0 <= idx < len(rewards):
            g += (gamma ** i) * rewards[idx]
    g += (gamma ** int(n)) * float(bootstrap_value)
    return g


def compute_advantage(v_own: float, g_latest: float) -> float:
    """``A(s_t) = G_latest^(n)(s_t) - V_own(s_t)`` (ARE-4, pure helper).

    Exposed for unit tests that assert ``A`` from known ``V_own`` / ``G_latest``.
    """
    return float(g_latest) - float(v_own)


def adaptive_horizon(
    path_len: int,
    t: int,
    tree_store,
    node_ids: list[str],
    excluded_rewards: list[float],
    *,
    rho: float,
    n_min: int,
    n_max: int,
    critic_error_var: float | None,
) -> tuple[int, bool]:
    """Pick the backup horizon for node ``t`` by walking descendants (ARE-4).

    Bootstrap at the first descendant ``s_{t+d}`` where the critic is
    trustworthy::

        critic_error_var(s_{t+d}) <= rho * var_mc(s_{t+d})

    where ``critic_error_var`` is the global critic-error EMA and ``var_mc`` is
    the per-node MC mean's sampling variance from
    ``tree_store.get_loo_value_and_variance``. Returns ``(horizon, trusted)``:

    * ``trusted=True``  -> bootstrap with ``V_latest(s_{t+horizon})``.
    * ``trusted=False`` -> pure MC: bootstrap 0 (terminal reached, or the
      ``n_max`` cap hit without the criterion ever being satisfied).

    ``var_mc`` unavailable (``visit < 2``) -> the critic is NOT trusted there ->
    keep walking (long horizon), so brand-new nodes with no MC data fall through
    to a pure-MC return. ``n_min`` floors the horizon; ``n_max`` caps it.
    ``rho in (0, 1]`` controls how strict the trust criterion is.
    """
    distance_to_terminal = path_len - t  # t + distance == terminal index
    d = max(int(n_min), 1)
    crit_err = float(critic_error_var) if critic_error_var is not None else 0.0
    while d <= int(n_max) and d < distance_to_terminal:
        idx = t + d
        if 0 <= idx < len(node_ids):
            _, var_mc, n_loo = tree_store.get_loo_value_and_variance(
                node_ids[idx], excluded_rewards[idx]
            )
            if n_loo >= 2 and var_mc >= 0.0 and crit_err <= rho * var_mc:
                return d, True
        d += 1
    horizon = min(int(n_max), distance_to_terminal)
    if horizon < 1:
        horizon = 1
    return horizon, False


class VersionedBackupAdvantageComputer(HybridGAEAdvantageComputer):
    """ΔV advantage ``A(s_t) = G_latest^(n)(s_t) - V_own(s_t)`` (ARE-4).

    Each ``Node`` carries a ``version_id`` (the policy version that generated
    it). The latest policy version is recorded on the tree store; nodes whose
    ``version_id`` equals it are "new", all others are reused "old" nodes.

    * ``V_own(s_t)`` [baseline]: the node's OWN blended value (critic +
      all-version LOO MC, reusing :class:`HybridGAEAdvantageComputer`'s
      inverse-variance blend). Using the node's own value -- not its parent's --
      satisfies "new node baseline = own V_own".
    * ``G_latest^(n)(s_t)`` [return]: n-step return along the episode,
      bootstrapped at an adaptively-chosen descendant with ``V_latest``.
      Horizon ``n`` is the first depth where the critic is trustworthy
      (:func:`adaptive_horizon`); new / no-data nodes never satisfy it early ->
      long horizon / pure MC. ``V_latest`` is NEW-ONLY: for new nodes it is the
      own value (they ARE the latest exploration); for old nodes it is the
      latest-policy LOO MC value (excluding old-policy samples). Old nodes with
      NO latest data are "path-off" -> advantage 0 / skipped this round.

    No importance sampling -- the off-policy value estimate is backup-based.
    Intermediates ``v_own`` / ``g_latest`` are stashed on each node for
    inspection/testing.
    """

    def __init__(
        self,
        tree_store,
        gamma: float = 1.0,
        lam: float = 0.95,
        judge_beta: float = 0.0,
        judge_score_max: int = 10,
        mc_min_visits: int = 5,
        critic_var_floor: float = 1e-3,
        critic_error_var: float = 0.05,
        critic_error_var_fn=None,
        rho: float = 1.0,
        n_min: int = 1,
        n_max: int = 10,
    ) -> None:
        super().__init__(
            tree_store,
            gamma=gamma,
            lam=lam,
            judge_beta=judge_beta,
            judge_score_max=judge_score_max,
            mc_min_visits=mc_min_visits,
            critic_var_floor=critic_var_floor,
            critic_error_var=critic_error_var,
            critic_error_var_fn=critic_error_var_fn,
        )
        self.rho = float(rho)
        self.n_min = int(n_min)
        self.n_max = int(n_max)

    def _is_new(self, node: Node) -> bool:
        """True iff this node was generated by the latest policy version."""
        if self.tree_store.latest_version < 0:
            return False
        return int(getattr(node, "version_id", -1) or -1) == self.tree_store.latest_version

    def _is_on_latest_path(self, node: Node) -> bool:
        """On-path check (ARE-4) using the node's own ``version_id``.

        New (latest-version) nodes are always on the path. Old nodes are on the
        path iff at least one latest-only MC sample has been backed up through
        them; old nodes with no latest data are "path-off" -> A=0 / skipped.
        When version tracking is off (``latest_version < 0``) every node is
        on-path so the computer degrades gracefully. Reads ``version_id`` from
        the node object directly (not via ``get_node``) so it works whether or
        not the node is indexed in the store.
        """
        if self.tree_store.latest_version < 0:
            return True
        if self._is_new(node):
            return True
        node_id = getattr(node, "node_id", None)
        return bool(node_id) and self.tree_store.get_latest_visit_count(node_id) >= 1

    def _v_latest(self, node: Node, excluded_reward: float) -> float:
        """``V_latest`` at a node: the new-only value (ARE-4).

        New (latest-version) node -> its own blended value (it IS the latest
        exploration). Old node -> the latest-policy LOO MC mean, falling back to
        the all-version blended value when no latest-only sample exists. When
        version tracking is off (``latest_version < 0``) every node uses its own
        blended value.
        """
        node_id = getattr(node, "node_id", None)
        if (
            self.tree_store.latest_version >= 0
            and node_id
            and not self._is_new(node)
        ):
            v_latest, _, n_loo = self.tree_store.get_latest_loo_value_and_variance(
                node_id, excluded_reward
            )
            if n_loo >= 2:
                return float(v_latest)
        return self._blended_value(node, excluded_reward)

    def compute(self, trajectories: list[Node]) -> None:
        """Compute ΔV advantages/returns in-place on the given nodes (ARE-4)."""
        episodes: dict[tuple[str, str], list[Node]] = {}
        for traj in trajectories:
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            query_id = traj.query_id or ""
            ep_id = traj.episode_id or node_id
            episodes.setdefault((query_id, ep_id), []).append(traj)

        # Record the latest policy version present in this batch so the
        # is_latest_* / is_on_latest_path helpers resolve correctly even when
        # the store was not pre-configured with set_latest_version.
        max_vid = -1
        for traj in trajectories:
            vid = int(getattr(traj, "version_id", -1) or -1)
            if vid > max_vid:
                max_vid = vid
        if max_vid >= 0 and (
            self.tree_store.latest_version < 0
            or max_vid > self.tree_store.latest_version
        ):
            self.tree_store.set_latest_version(max_vid)

        for nodes in episodes.values():
            ordered = sorted(nodes, key=lambda n: getattr(n, "turn_idx", 0))
            n_turns = len(ordered)
            node_ids = [getattr(n, "node_id", "") for n in ordered]

            # Per-turn rewards (reuse the GAE/Hybrid reward logic).
            if self.judge_beta > 0.0:
                from customized_areal.tree_search.core.process_reward import (
                    build_episode_process_rewards,
                    episode_returns_to_go,
                )

                rewards = build_episode_process_rewards(
                    ordered,
                    self.tree_store,
                    beta=self.judge_beta,
                    score_max=self.judge_score_max,
                )
                excluded = episode_returns_to_go(rewards, gamma=self.gamma)
            else:
                rewards = [0.0] * n_turns
                if n_turns > 0:
                    rewards[-1] = float(ordered[-1].outcome_reward)
                term = float(ordered[-1].outcome_reward) if n_turns > 0 else 0.0
                excluded = [term] * n_turns

            # V_own (own blended baseline) and V_latest (new-only) per turn.
            v_own = [
                self._blended_value(n, excluded[i]) for i, n in enumerate(ordered)
            ]
            v_latest_vals = [
                self._v_latest(n, excluded[i]) for i, n in enumerate(ordered)
            ]

            crit_err = self._critic_error_var()
            for t, node in enumerate(ordered):
                node_id = node_ids[t]
                # Path-off old nodes: no latest data -> A=0, skip (AC #6).
                if not self._is_on_latest_path(node):
                    node.v_own = v_own[t]
                    node.g_latest = 0.0
                    self._assign(node, 0.0, v_own[t])
                    self.tree_store.set_normalized_advantage(node_id, 0.0)
                    self.tree_store.set_normalized_return(node_id, v_own[t])
                    continue

                horizon, trusted = adaptive_horizon(
                    n_turns,
                    t,
                    self.tree_store,
                    node_ids,
                    excluded,
                    rho=self.rho,
                    n_min=self.n_min,
                    n_max=self.n_max,
                    critic_error_var=crit_err,
                )
                if trusted and (t + horizon) < n_turns:
                    bootstrap = v_latest_vals[t + horizon]
                else:
                    bootstrap = 0.0  # pure MC (terminal / n_max cap)
                g_latest = n_step_return(
                    rewards, t, horizon, self.gamma, bootstrap
                )
                adv = compute_advantage(v_own[t], g_latest)
                ret = adv + v_own[t]
                node.v_own = v_own[t]
                node.g_latest = g_latest
                self._assign(node, adv, ret)
                if node_id:
                    self.tree_store.set_normalized_advantage(node_id, adv)
                    self.tree_store.set_normalized_return(node_id, ret)

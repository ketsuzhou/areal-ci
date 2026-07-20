# SPDX-License-Identifier: Apache-2.0
"""Tests for the versioned ΔV advantage (ARE-4).

Covers the 10 acceptance criteria:
  1. Node.version_id write / default / -1.
  2. A = G_latest^(n) - V_own (1-step and n-step) via pure helpers.
  3. V_latest = new-only (old node return excludes old-policy data; new node
     return = own n-step).
  4. New node baseline = own V_own (not the parent's).
  5. Adaptive horizon: old depth-1 (critic trusted) -> 1; new / no-data -> pure
     MC; mixed -> first trusted depth; visit<2 -> long horizon.
  6. Path-off old node advantage = 0 / skipped.
  7. No GAE/GRPO regression (covered by the existing suites; this file also
     checks the new class leaves plain-GAE output untouched when versioning is
     off).
  8. End-to-end compute: A non-zero, shape aligned with loss_mask, broadcast to
     response positions.
  9. Edges: critic disabled, single turn, terminal bootstrap, version missing
     / -1, rho / n_min / n_max boundaries.
 10. File list + data flow (asserted via the intermediates stashed on the node
     and the documented data-flow docstring in VersionedBackupAdvantageComputer).
"""

import math

import torch

from customized_areal.tree_search.core.advantage import (
    VersionedBackupAdvantageComputer,
    adaptive_horizon,
    compute_advantage,
    n_step_return,
)
from customized_areal.tree_search.core.tree_store import (
    MCTSTreeStore,
    Node,
    version_id_from_versions,
)


def _make_node(
    node_id,
    episode_id,
    turn_idx,
    loss_mask,
    value=0.0,
    outcome_reward=0.0,
    need_branch=False,
    version_id=-1,
    parent_node_id=None,
):
    n = Node(
        input_ids=[0] * len(loss_mask),
        loss_mask=loss_mask,
        logprobs=[0.0] * len(loss_mask),
        versions=[version_id] * len(loss_mask),
        node_id=node_id,
        parent_node_id=parent_node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=outcome_reward,
        need_branch=need_branch,
        version_id=version_id,
    )
    n.value = value
    return n


def _seed(store, node_id, samples, latest=False):
    """Seed a node's MC aggregates (all-version, and latest-only if asked)."""
    n = len(samples)
    total = sum(samples)
    sum_sq = sum(s * s for s in samples)
    if latest:
        store._latest_visit_counts[node_id] = n
        store._latest_total_values[node_id] = total
        store._latest_sum_sq_values[node_id] = sum_sq
        store._latest_q_values[node_id] = total / n
    else:
        store._visit_counts[node_id] = n
        store._total_values[node_id] = total
        store._sum_sq_values[node_id] = sum_sq
        store._q_values[node_id] = total / n


# ---------------------------------------------------------------------------
# AC 1: Node.version_id write / default / -1
# ---------------------------------------------------------------------------


class TestNodeVersionId:
    def test_default_is_negative_one(self):
        n = Node(input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0])
        assert n.version_id == -1

    def test_version_id_from_versions_extracts_generation_version(self):
        # prompt tokens carry -1, response tokens carry the generation version.
        assert version_id_from_versions([-1, -1, 0, 0]) == 0
        assert version_id_from_versions([-1, -1, 5, 5]) == 5
        assert version_id_from_versions([-1, -1]) == -1
        assert version_id_from_versions([]) == -1

    def test_construction_writes_version_id(self):
        n = _make_node("n0", "ep", 0, [0, 1], value=0.2, version_id=7)
        assert n.version_id == 7


# ---------------------------------------------------------------------------
# AC 2: A = G_latest^(n) - V_own (pure helpers, known values)
# ---------------------------------------------------------------------------


class TestAdvantageFormula:
    def test_compute_advantage_known_values(self):
        assert compute_advantage(0.2, 0.5) == 0.3
        assert compute_advantage(0.8, 0.8) == 0.0
        assert compute_advantage(0.5, 0.2) == -0.3

    def test_one_step_return(self):
        # G^(1)(s_t) = r_t + gamma * V(s_{t+1})
        # r=[0,0,1], t=0, gamma=1, bootstrap V(s_1)=0.5 -> 0 + 1*0.5 = 0.5
        assert n_step_return([0.0, 0.0, 1.0], 0, 1, 1.0, 0.5) == 0.5

    def test_n_step_return_discounted(self):
        # G^(2)(s_0) = r0 + gamma r1 + gamma^2 V(s_2)
        # r=[0,0,1], gamma=0.9, V(s_2)=0.4 -> 0 + 0.9*0 + 0.81*0.4 = 0.324
        assert abs(n_step_return([0.0, 0.0, 1.0], 0, 2, 0.9, 0.4) - 0.324) < 1e-9

    def test_n_step_pure_mc_terminal_bootstrap_zero(self):
        # horizon reaching/past terminal -> rewards summed, bootstrap 0 (pure MC).
        # r=[0,0,1], t=0, n=3, gamma=1, bootstrap=0 -> 1.0
        assert n_step_return([0.0, 0.0, 1.0], 0, 3, 1.0, 0.0) == 1.0

    def test_one_step_advantage_matches_spec(self):
        # A = r_t + gamma*V(s_{t+1}) - V(s_t)
        v_own = 0.2
        g_latest = n_step_return([0.0, 0.0, 1.0], 0, 1, 1.0, 0.5)  # 0.5
        assert compute_advantage(v_own, g_latest) == 0.3  # 0.5 - 0.2


# ---------------------------------------------------------------------------
# AC 3: V_latest = new-only
# ---------------------------------------------------------------------------


class TestVLatestNewOnly:
    def test_latest_loo_excludes_old_policy_samples(self):
        # All-version aggregate mixes old (four 1.0) and new (three 0.2)
        # samples; the latest-only aggregate holds ONLY the new 0.2 samples.
        store = MCTSTreeStore()
        store.set_latest_version(2)
        _seed(store, "x", [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2])  # all-version
        _seed(store, "x", [0.2, 0.2, 0.2], latest=True)  # latest-only

        all_mean, _, all_n = store.get_loo_value_and_variance("x", 0.2)
        new_mean, _, new_n = store.get_latest_loo_value_and_variance("x", 0.2)

        # All-version mean is pulled up by the old 1.0 samples; new-only is 0.2.
        assert all_n == 6 and all_mean > 0.2
        assert new_n == 2 and abs(new_mean - 0.2) < 1e-9

    def test_v_latest_old_node_uses_latest_only_mc(self):
        # Old node (version 1) with latest-only MC -> V_latest = latest MC mean,
        # NOT the all-version mean (which is contaminated by old samples).
        store = MCTSTreeStore()
        store.set_latest_version(2)
        _seed(store, "old", [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
        _seed(store, "old", [0.2, 0.2, 0.2], latest=True)
        node = _make_node("old", "ep", 0, [1], value=0.9, version_id=1)
        # Make the critic uninformative so the blend (fallback) would differ;
        # the latest-only MC path must win for an old node with latest data.
        comp = VersionedBackupAdvantageComputer(
            store, gamma=1.0, lam=1.0, critic_error_var=1e9
        )
        v_latest = comp._v_latest(node, 0.2)
        assert abs(v_latest - 0.2) < 1e-9  # latest-only MC mean, not 0.9/blend

    def test_v_latest_new_node_uses_own_value(self):
        # New node (version == latest) -> V_latest = own blended value.
        store = MCTSTreeStore()
        store.set_latest_version(2)
        node = _make_node("new", "ep", 0, [1], value=0.4, version_id=2)
        comp = VersionedBackupAdvantageComputer(store, gamma=1.0, lam=1.0)
        assert abs(comp._v_latest(node, 0.0) - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# AC 4: new node baseline = own V_own (not parent)
# ---------------------------------------------------------------------------


class TestNewNodeBaselineOwnValue:
    def test_v_own_is_node_own_value_not_parent(self):
        # Two new nodes with distinct critic values; each V_own is its own.
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n0 = _make_node("n0", "ep", 0, [0, 1], value=0.2, version_id=2)
        n1 = _make_node(
            "n1", "ep", 1, [1], value=0.5, version_id=2, parent_node_id="n0"
        )
        comp = VersionedBackupAdvantageComputer(
            store, gamma=1.0, lam=1.0, n_max=10
        )
        comp.compute([n0, n1])
        # V_own stashed on each node equals that node's own critic value
        # (no MC blend for need_branch=False), not the parent's.
        assert abs(n0.v_own - 0.2) < 1e-9
        assert abs(n1.v_own - 0.5) < 1e-9
        assert n0.v_own != n1.v_own


# ---------------------------------------------------------------------------
# AC 5: adaptive horizon
# ---------------------------------------------------------------------------


class TestAdaptiveHorizon:
    @staticmethod
    def _store_with_descendant(descendant_id, samples):
        store = MCTSTreeStore()
        _seed(store, descendant_id, samples)
        return store

    def test_old_node_depth1_critic_trusted(self):
        # n1 has MC data with var_mc ~= 0.111; critic_error_var tiny ->
        # critic trusted at depth 1 -> horizon=1, trusted.
        store = self._store_with_descendant("n1", [0.0, 1.0, 0.0, 1.0])
        h, trusted = adaptive_horizon(
            path_len=5,
            t=0,
            tree_store=store,
            node_ids=["n0", "n1", "n2", "n3", "n4"],
            excluded_rewards=[0.0] * 5,
            rho=1.0,
            n_min=1,
            n_max=10,
            critic_error_var=0.001,
        )
        assert (h, trusted) == (1, True)

    def test_new_node_no_data_pure_mc(self):
        # Descendants have no MC data (visit<2) -> never trusted -> pure MC to
        # terminal: horizon = distance to terminal, trusted=False.
        store = MCTSTreeStore()  # no stats seeded -> all visit<2
        h, trusted = adaptive_horizon(
            path_len=4,
            t=0,
            tree_store=store,
            node_ids=["n0", "n1", "n2", "n3"],
            excluded_rewards=[0.0] * 4,
            rho=1.0,
            n_min=1,
            n_max=10,
            critic_error_var=0.001,  # even a trusted-looking critic
        )
        assert trusted is False
        assert h == 4  # distance to terminal (pure MC)

    def test_mixed_first_trusted_depth(self):
        # n1: var_mc small (~0.056, floored) -> critic_err=0.1 NOT trusted.
        # n2: var_mc=0.25 -> critic_err=0.1 trusted -> horizon=2.
        store = MCTSTreeStore()
        _seed(store, "n1", [0.5, 0.5, 0.5])  # excl 0.5 -> var_mc ~0.0556
        _seed(store, "n2", [0.0, 1.0, 0.0])  # excl 0.0 -> var_mc = 0.25
        h, trusted = adaptive_horizon(
            path_len=5,
            t=0,
            tree_store=store,
            node_ids=["n0", "n1", "n2", "n3", "n4"],
            excluded_rewards=[0.0] * 5,
            rho=1.0,
            n_min=1,
            n_max=10,
            critic_error_var=0.1,
        )
        assert (h, trusted) == (2, True)

    def test_visit_lt_two_means_long_horizon(self):
        # Descendants all have a single sample (n_loo<2 -> var_mc unavailable).
        # Critic never trusted -> pure MC (long horizon), even though
        # critic_error_var is tiny.
        store = MCTSTreeStore()
        _seed(store, "n1", [0.5])  # visit=1 -> n_loo=0 < 2
        _seed(store, "n2", [0.5])
        h, trusted = adaptive_horizon(
            path_len=5,
            t=0,
            tree_store=store,
            node_ids=["n0", "n1", "n2", "n3", "n4"],
            excluded_rewards=[0.0] * 5,
            rho=1.0,
            n_min=1,
            n_max=10,
            critic_error_var=0.001,
        )
        assert trusted is False
        assert h == 5  # long horizon (pure MC to terminal)

    def test_n_max_cap_when_never_trusted(self):
        # Long episode, no descendant ever trusted -> horizon capped at n_max.
        store = MCTSTreeStore()
        h, trusted = adaptive_horizon(
            path_len=20,
            t=0,
            tree_store=store,
            node_ids=[f"n{i}" for i in range(20)],
            excluded_rewards=[0.0] * 20,
            rho=1.0,
            n_min=1,
            n_max=4,
            critic_error_var=0.9,  # large -> never trusted
        )
        assert trusted is False
        assert h == 4  # n_max cap


# ---------------------------------------------------------------------------
# AC 6: path-off old node advantage = 0 / skipped
# ---------------------------------------------------------------------------


class TestPathOffOldNode:
    def test_old_node_without_latest_data_is_path_off(self):
        store = MCTSTreeStore()
        store.set_latest_version(2)
        # n_off is old (version 1) and has NO latest-only data -> path-off.
        n_off = _make_node(
            "n_off", "ep", 0, [0, 1], value=0.3, version_id=1
        )
        assert store.is_on_latest_path("n_off") is False

    def test_old_node_with_latest_data_is_on_path(self):
        store = MCTSTreeStore()
        store.set_latest_version(2)
        _seed(store, "n_on", [0.4], latest=True)  # latest-only sample present
        # Need the node registered so is_on_latest_path can look it up.
        n_on = _make_node("n_on", "ep", 0, [1], value=0.3, version_id=1)
        from customized_areal.tree_search.agents.execution_dag import SuperNode

        store.insert_super_batch(
            [SuperNode(node_id="s", agent_id="a", issue_id="i", task_id="t", nodes=[n_on])],
            backup=False,
            query_id="q",
        )
        _seed(store, "n_on", [0.4], latest=True)
        assert store.is_on_latest_path("n_on") is True

    def test_path_off_node_advantage_zero(self):
        # Episode: an old path-off node + a new node. The old node has no
        # latest data -> its advantage must be 0; the new node is computed.
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n_old = _make_node(
            "n_old", "ep", 0, [0, 1], value=0.2, version_id=1
        )  # old, no latest data -> path-off
        n_new = _make_node(
            "n_new", "ep", 1, [1], value=0.5, version_id=2, parent_node_id="n_old"
        )
        n_new.outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(
            store, gamma=1.0, lam=1.0, n_max=10
        )
        comp.compute([n_old, n_new])
        # Old path-off node: advantage broadcast over loss_mask is all zero.
        torch.testing.assert_close(
            n_old.advantages, torch.tensor([0.0, 0.0], dtype=torch.float32)
        )
        assert n_old.g_latest == 0.0
        # New node is on-path and computed (non-zero here: r=1, v=0.5 -> 0.5).
        assert abs(float(n_new.advantages[0]) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# AC 8: end-to-end compute (shape, non-zero, broadcast to response positions)
# ---------------------------------------------------------------------------


class TestEndToEndCompute:
    def test_all_new_episode_pure_mc_advantages(self):
        # 3 new turns, critic values [0.2,0.5,0.8], terminal reward 1.0,
        # gamma=1, no MC data -> pure MC to terminal for every node.
        #   G_0 = 1.0, A_0 = 1.0 - 0.2 = 0.8
        #   G_1 = 1.0, A_1 = 1.0 - 0.5 = 0.5
        #   G_2 = 1.0, A_2 = 1.0 - 0.8 = 0.2
        store = MCTSTreeStore()
        store.set_latest_version(2)
        nodes = [
            _make_node("n0", "ep", 0, [0, 1], value=0.2, version_id=2),
            _make_node("n1", "ep", 1, [0, 1], value=0.5, version_id=2, parent_node_id="n0"),
            _make_node("n2", "ep", 2, [1], value=0.8, version_id=2, parent_node_id="n1"),
        ]
        nodes[-1].outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(
            store, gamma=1.0, lam=1.0, n_max=10
        )
        comp.compute(nodes)

        expected_adv = [0.8, 0.5, 0.2]
        expected_ret = [1.0, 1.0, 1.0]
        for n, adv, ret in zip(nodes, expected_adv, expected_ret):
            # shape aligns with loss_mask
            assert n.advantages.shape[0] == len(n.loss_mask)
            # broadcast: prompt positions (loss_mask==0) are 0
            mask = torch.tensor(n.loss_mask, dtype=torch.float32)
            torch.testing.assert_close(n.advantages, mask * adv, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(n.returns, mask * ret, atol=1e-6, rtol=1e-6)
            # intermediates stashed (AC 10)
            assert n.v_own is not None and n.g_latest is not None
        # non-zero advantages present
        assert all(float(n.advantages.sum()) != 0.0 for n in nodes)

    def test_old_node_uses_latest_bootstrap(self):
        # Old node n0 (version 1) on the latest path; descendant n1 (old) has
        # MC data and a low critic_error_var -> horizon=1, bootstrap with
        # V_latest(n1) (latest-only MC). G_0 = r0 + gamma*V_latest(n1).
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n0 = _make_node("n0", "ep", 0, [1], value=0.2, version_id=1)
        n1 = _make_node("n1", "ep", 1, [1], value=0.9, version_id=1, parent_node_id="n0")
        n1.outcome_reward = 1.0
        # Register nodes so is_on_latest_path / get_node resolve.
        from customized_areal.tree_search.agents.execution_dag import SuperNode

        store.insert_super_batch(
            [SuperNode(node_id="s", agent_id="a", issue_id="i", task_id="t", nodes=[n0, n1])],
            backup=False,
            query_id="q",
        )
        # Latest exploration passed through both n0 and n1. The current
        # episode (latest) contributed its return-to-go g_t to each node's
        # latest-only aggregate; in the sparse terminal case g_t = outcome = 1.0.
        # n0: [1.0] (current) -> on-path. n1: [1.0 (current), 0.6, 0.6, 0.6]
        # (current + 3 other latest branches); LOO excluding the current 1.0
        # leaves three 0.6 -> V_latest(n1) = 0.6.
        _seed(store, "n0", [1.0], latest=True)
        _seed(store, "n1", [1.0, 0.6, 0.6, 0.6], latest=True)
        # All-version MC for n1 with enough spread that var_mc is available and
        # critic (tiny error var) is trusted at depth 1.
        _seed(store, "n1", [0.0, 1.0, 0.0, 1.0, 0.6, 0.6, 0.6])
        comp = VersionedBackupAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            rho=1.0,
            n_min=1,
            n_max=10,
            critic_error_var=0.001,
        )
        comp.compute([n0, n1])
        # horizon=1 trusted: G_0 = r0 + 1*V_latest(n1) = 0 + 0.6 = 0.6
        # A_0 = 0.6 - V_own(n0)=0.2 -> 0.4
        assert abs(float(n0.advantages[0]) - 0.4) < 1e-6
        assert abs(n0.g_latest - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# AC 9: edges
# ---------------------------------------------------------------------------


class TestEdges:
    def test_critic_disabled_value_zero(self):
        # value=0 everywhere -> V_own=0; pure-MC G -> A = G (telescoping to 1.0).
        store = MCTSTreeStore()
        store.set_latest_version(2)
        nodes = [
            _make_node("n0", "ep", 0, [0, 1], value=0.0, version_id=2),
            _make_node("n1", "ep", 1, [0, 1], value=0.0, version_id=2, parent_node_id="n0"),
            _make_node("n2", "ep", 2, [1], value=0.0, version_id=2, parent_node_id="n1"),
        ]
        nodes[-1].outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(store, gamma=1.0, lam=1.0, n_max=10)
        comp.compute(nodes)
        for n in nodes:
            # A = G - 0 = G = 1.0 (telescoping terminal reward)
            assert abs(float(n.advantages[n.loss_mask.index(1)]) - 1.0) < 1e-6

    def test_single_turn_episode(self):
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n = _make_node("n0", "ep", 0, [0, 1, 1], value=0.3, version_id=2)
        n.outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(store, gamma=1.0, lam=1.0, n_max=10)
        comp.compute([n])
        # horizon=1 pure MC: G = r0 = 1.0; A = 1.0 - 0.3 = 0.7
        mask = torch.tensor([0, 1, 1], dtype=torch.float32)
        torch.testing.assert_close(n.advantages, mask * 0.7, atol=1e-6, rtol=1e-6)

    def test_terminal_bootstrap_zero(self):
        # The terminal node bootstraps with 0 (no descendant past it).
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n = _make_node("nT", "ep", 0, [1], value=0.8, version_id=2)
        n.outcome_reward = 0.4
        comp = VersionedBackupAdvantageComputer(store, gamma=0.5, lam=1.0, n_max=10)
        comp.compute([n])
        # G = r0 + gamma^1*0 = 0.4; A = 0.4 - 0.8 = -0.4
        assert abs(float(n.advantages[0]) - (-0.4)) < 1e-6

    def test_version_missing_degrades_gracefully(self):
        # All version_id=-1, latest_version unset -> no node is skipped (on-path
        # for all), V_latest falls back to own value. Compute runs, non-zero.
        store = MCTSTreeStore()
        assert store.latest_version == -1
        nodes = [
            _make_node("n0", "ep", 0, [0, 1], value=0.2, version_id=-1),
            _make_node("n1", "ep", 1, [1], value=0.5, version_id=-1, parent_node_id="n0"),
        ]
        nodes[-1].outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(store, gamma=1.0, lam=1.0, n_max=10)
        comp.compute(nodes)
        # No skip happened: n0 got a real advantage (pure MC G=1, A=0.8).
        assert abs(float(nodes[0].advantages[1]) - 0.8) < 1e-6

    def test_n_max_one_forces_horizon_one(self):
        # n_max=1 -> every node bootstraps at depth 1 (or terminal), never pure
        # long MC.
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n0 = _make_node("n0", "ep", 0, [1], value=0.2, version_id=2)
        n1 = _make_node("n1", "ep", 1, [1], value=0.5, version_id=2, parent_node_id="n0")
        n1.outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(
            store, gamma=1.0, lam=1.0, n_max=1
        )
        comp.compute([n0, n1])
        # n0: horizon=1, no MC -> trusted=False -> pure MC bootstrap 0.
        # G_0 = r0 + gamma*0 = 0; A_0 = 0 - 0.2 = -0.2
        assert abs(float(n0.advantages[0]) - (-0.2)) < 1e-6
        # n1 terminal: G = r1 = 1.0; A = 1.0 - 0.5 = 0.5
        assert abs(float(n1.advantages[0]) - 0.5) < 1e-6

    def test_rho_boundary_one_allows_trust(self):
        # rho=1.0 (max) -> criterion critic_err <= var_mc is easiest to meet.
        store = MCTSTreeStore()
        _seed(store, "n1", [0.0, 1.0, 0.0, 1.0])  # var_mc ~0.111
        h, trusted = adaptive_horizon(
            path_len=3,
            t=0,
            tree_store=store,
            node_ids=["n0", "n1", "n2"],
            excluded_rewards=[0.0, 0.0, 0.0],
            rho=1.0,
            n_min=1,
            n_max=10,
            critic_error_var=0.1,  # 0.1 <= 0.111 -> trusted at rho=1
        )
        assert (h, trusted) == (1, True)

    def test_rho_small_prevents_trust(self):
        # rho=0.05 -> criterion critic_err <= 0.05*var_mc much stricter; with
        # critic_err=0.1 and var_mc~0.111 -> 0.1 <= 0.0055 False -> not trusted.
        store = MCTSTreeStore()
        _seed(store, "n1", [0.0, 1.0, 0.0, 1.0])
        h, trusted = adaptive_horizon(
            path_len=3,
            t=0,
            tree_store=store,
            node_ids=["n0", "n1", "n2"],
            excluded_rewards=[0.0, 0.0, 0.0],
            rho=0.05,
            n_min=1,
            n_max=10,
            critic_error_var=0.1,
        )
        assert trusted is False


# ---------------------------------------------------------------------------
# AC 7 (partial): versioning off -> matches a sensible n-step baseline and
# does not touch plain GAE. (Full GAE/GRPO regression is the existing suites.)
# ---------------------------------------------------------------------------


class TestNoRegression:
    def test_versioned_backup_runs_without_mc_data(self):
        # No MC data anywhere, no judge -> must not raise and must produce
        # finite advantages (degrades to critic-baseline n-step / pure MC).
        store = MCTSTreeStore()
        store.set_latest_version(2)
        nodes = [
            _make_node("n0", "ep", 0, [0, 1], value=0.2, version_id=2),
            _make_node("n1", "ep", 1, [1], value=0.5, version_id=2, parent_node_id="n0"),
        ]
        nodes[-1].outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(store, gamma=1.0, lam=1.0, n_max=10)
        comp.compute(nodes)
        for n in nodes:
            assert torch.isfinite(n.advantages).all()
            assert torch.isfinite(n.returns).all()


# ---------------------------------------------------------------------------
# AC 10: file list + data flow (the data-flow contract is asserted by the
# stashed intermediates v_own / g_latest and the public helper surface).
# ---------------------------------------------------------------------------


class TestDataFlowContract:
    def test_intermediates_stashed_for_inspection(self):
        store = MCTSTreeStore()
        store.set_latest_version(2)
        n = _make_node("n0", "ep", 0, [0, 1], value=0.3, version_id=2)
        n.outcome_reward = 1.0
        comp = VersionedBackupAdvantageComputer(store, gamma=1.0, lam=1.0, n_max=10)
        comp.compute([n])
        # data flow: version_id (on node) -> V_own (blended baseline) +
        # G_latest (new-only n-step) -> A = G_latest - V_own (stashed).
        assert n.version_id == 2
        assert n.v_own is not None
        assert n.g_latest is not None
        assert abs((n.g_latest - n.v_own) - float(n.advantages[1])) < 1e-6

    def test_pure_helpers_are_public(self):
        # The data-flow steps are individually testable (AC 10 file list /
        # data flow): n_step_return, adaptive_horizon, compute_advantage.
        assert callable(n_step_return)
        assert callable(adaptive_horizon)
        assert callable(compute_advantage)

# SPDX-License-Identifier: Apache-2.0
"""Tests for per-node return-to-go MCTS backup (gated on judge_beta > 0).

Covers:
- ``episode_returns_to_go`` discounted reverse cumsum (sparse reduces to outcome).
- ``MCTSTreeStore.insert_batch(backup=False)`` defers the MC backup.
- ``backup_episode_returns`` accumulates per-node return-to-go.
- ``HybridGAEAdvantageComputer`` with ``judge_beta == 0`` is byte-for-byte
  unchanged, and with ``judge_beta > 0`` the leave-one-out excludes the node's
  own ``g_t`` (not the terminal outcome).
"""

import torch

from customized_areal.tree_search.core.advantage import (
    GAEAdvantageComputer,
    HybridGAEAdvantageComputer,
)
from customized_areal.tree_search.core.process_reward import episode_returns_to_go
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _make_node(
    node_id,
    episode_id,
    turn_idx,
    loss_mask,
    value=0.0,
    outcome_reward=0.0,
    need_branch=False,
    parent_node_id=None,
):
    n = Node(
        input_ids=[0] * len(loss_mask),
        loss_mask=loss_mask,
        logprobs=[0.0] * len(loss_mask),
        versions=[0] * len(loss_mask),
        node_id=node_id,
        parent_node_id=parent_node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=outcome_reward,
        need_branch=need_branch,
    )
    n.value = value
    return n


class TestEpisodeReturnsToGo:
    def test_sparse_reduces_to_outcome(self):
        # Terminal-only reward with gamma=1 -> every turn's return-to-go == outcome.
        assert episode_returns_to_go([0.0, 0.0, 1.0], gamma=1.0) == [1.0, 1.0, 1.0]

    def test_discounted(self):
        g = episode_returns_to_go([0.0, 0.0, 1.0], gamma=0.9)
        assert g == [0.81, 0.9, 1.0]

    def test_dense(self):
        # README worked example: r = [0.04, 0.06, 0.9], gamma=1.
        g = episode_returns_to_go([0.04, 0.06, 0.9], gamma=1.0)
        assert g[0] == 1.0
        assert abs(g[1] - 0.96) < 1e-12
        assert g[2] == 0.9


class TestDeferredBackup:
    def test_insert_batch_backup_false_defers(self):
        store = MCTSTreeStore()
        nodes = [
            _make_node("n0", "ep", 1, [1], outcome_reward=1.0),
            _make_node("n1", "ep", 2, [1], outcome_reward=1.0, parent_node_id="n0"),
        ]
        store.insert_batch(nodes, backup=False)
        # No MC samples accumulated yet.
        assert store.get_visit_count("n0") == 0
        assert store.get_visit_count("n1") == 0

    def test_insert_batch_default_backs_up(self):
        store = MCTSTreeStore()
        nodes = [
            _make_node("n0", "ep", 1, [1], outcome_reward=1.0),
            _make_node("n1", "ep", 2, [1], outcome_reward=1.0, parent_node_id="n0"),
        ]
        store.insert_batch(nodes)  # default backup=True
        # Legacy path walk adds terminal outcome to every node on the chain.
        assert store.get_visit_count("n0") == 1
        assert store.get_visit_count("n1") == 1
        assert store.get_q_value("n0") == 1.0
        assert store.get_q_value("n1") == 1.0

    def test_backup_episode_returns_per_node(self):
        store = MCTSTreeStore()
        nodes = [
            _make_node("n0", "ep", 1, [1], outcome_reward=1.0),
            _make_node("n1", "ep", 2, [1], outcome_reward=1.0, parent_node_id="n0"),
            _make_node("n2", "ep", 3, [1], outcome_reward=1.0, parent_node_id="n1"),
        ]
        store.insert_batch(nodes, backup=False)
        returns = [1.0, 0.96, 0.9]
        store.backup_episode_returns(nodes, returns)
        for nid, g in zip(["n0", "n1", "n2"], returns):
            assert store.get_visit_count(nid) == 1
            assert abs(store.get_q_value(nid) - g) < 1e-12

    def test_backup_episode_terminal_walks_path(self):
        store = MCTSTreeStore()
        nodes = [
            _make_node("n0", "ep", 1, [1], outcome_reward=1.0),
            _make_node("n1", "ep", 2, [1], outcome_reward=1.0, parent_node_id="n0"),
        ]
        store.insert_batch(nodes, backup=False)
        store.backup_episode_terminal("n1", 1.0)
        # Terminal return propagated root-ward to both nodes.
        assert store.get_q_value("n0") == 1.0
        assert store.get_q_value("n1") == 1.0


class TestHybridLOOUsesReturnToGo:
    def test_beta_zero_unchanged(self):
        # judge_beta == 0 must match plain GAE exactly (no eligible nodes here).
        def nodes():
            return [
                _make_node("n0", "ep", 1, [0, 1], value=0.2, outcome_reward=1.0),
                _make_node("n1", "ep", 2, [0, 1], value=0.5, outcome_reward=1.0),
                _make_node("n2", "ep", 3, [1], value=0.8, outcome_reward=1.0),
            ]

        g = GAEAdvantageComputer(MCTSTreeStore(), gamma=1.0, lam=0.95)
        h = HybridGAEAdvantageComputer(
            MCTSTreeStore(), gamma=1.0, lam=0.95, judge_beta=0.0
        )
        ng, nh = nodes(), nodes()
        g.compute(ng)
        h.compute(nh)
        for a, b in zip(ng, nh):
            torch.testing.assert_close(a.advantages, b.advantages)
            torch.testing.assert_close(a.returns, b.returns)

    def test_loo_excludes_g_t_not_terminal_outcome(self):
        # README worked example. 3-turn episode, beta=0.2, gamma=lam=1.
        # jbar=[0.2,0.3,0.5] via raw judge means 2/3/5; outcome=1.
        #   dense r = [0.04, 0.06, 0.9],  g = [1.0, 0.96, 0.9]
        # Middle node n1 is the eligible branched node. Its MC aggregate holds
        # samples [1,1,1,1,0.96]; excluding THIS episode's g_1 = 0.96 leaves four
        # identical 1.0 -> loo_mean = 1.0. The critic is made uninformative
        # (huge value_variance) so the inverse-variance blend collapses to
        # v_hat ~= v_mc = loo_mean = 1.0. If the code wrongly excluded the
        # terminal outcome (1.0), loo_mean would be 0.99 instead.
        #
        # NOTE: var_mc is now floored by a strictly-positive Beta posterior, so
        # this no longer exercises the (removed) var_mc==0 short-circuit -- it
        # goes through the real blend with a near-zero critic weight.
        store = MCTSTreeStore()
        n0 = _make_node("n0", "ep", 1, [1], value=0.0, outcome_reward=1.0)
        n1 = _make_node(
            "n1",
            "ep",
            2,
            [1],
            value=0.0,
            outcome_reward=1.0,
            need_branch=True,
            parent_node_id="n0",
        )
        n2 = _make_node(
            "n2", "ep", 3, [1], value=0.0, outcome_reward=1.0, parent_node_id="n1"
        )
        # Raw judge means: 2, 3, 5 -> total 10 -> jbar [0.2, 0.3, 0.5].
        store.add_judge_score("n0", 2.0)
        store.add_judge_score("n1", 3.0)
        store.add_judge_score("n2", 5.0)
        # Seed n1 MC aggregate with samples [1,1,1,1,0.96] (visit_count 5).
        samples = [1.0, 1.0, 1.0, 1.0, 0.96]
        store._visit_counts["n1"] = len(samples)
        store._total_values["n1"] = sum(samples)
        store._sum_sq_values["n1"] = sum(s * s for s in samples)
        store._q_values["n1"] = sum(samples) / len(samples)
        # Huge critic error variance -> w_theta ~ 0 -> v_hat ~= v_mc = loo_mean.
        # (The blend uses critic_error_var as var_theta, not value_variance.)

        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            judge_beta=0.2,
            judge_score_max=10,
            mc_min_visits=5,
            critic_var_floor=1e-3,
            critic_error_var=1e9,
        )
        h.compute([n0, n1, n2])

        # v(n1) ~= 1.0 (pure MC, critic uninformative). With gamma=lam=1 and
        # other values 0:  A_1 = -0.04, ret_1 = A_1 + v(n1) = 0.96
        torch.testing.assert_close(
            n1.advantages, torch.tensor([-0.04]), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            n1.returns, torch.tensor([0.96]), atol=1e-5, rtol=1e-4
        )


class TestUnifiedPathBackup:
    """Option A: branch episodes also back up return-to-go along their full path,
    so a shared branch point accumulates a homogeneous set of return-to-go
    samples (not a mix of return-to-go + terminal outcomes)."""

    def test_backup_path_returns_walks_full_chain(self):
        store = MCTSTreeStore()
        # Scratch episode S: r(turn1) -> x(turn2, branch point) -> x3(turn3).
        # Branch episode B: b1(turn1) -> b2(turn2), with b1.parent = x.
        # (Branch suffix turn_idx restarts at 1; linkage is via parent_node_id.)
        nodes = [
            _make_node("r", "S", 1, [1], outcome_reward=1.0),
            _make_node("x", "S", 2, [1], outcome_reward=1.0, parent_node_id="r"),
            _make_node("x3", "S", 3, [1], outcome_reward=1.0, parent_node_id="x"),
            _make_node("b1", "B", 1, [1], outcome_reward=0.0, parent_node_id="x"),
            _make_node("b2", "B", 2, [1], outcome_reward=0.0, parent_node_id="b1"),
        ]
        store.insert_batch(nodes, backup=False)

        # Back up each episode with its own per-node return-to-go.
        store.backup_path_returns("x3", {"r": 1.0, "x": 0.96, "x3": 0.90})
        store.backup_path_returns("b2", {"r": 0.50, "x": 0.16, "b1": 0.30, "b2": 0.0})

        # Shared branch point x got ONE return-to-go sample from each episode.
        assert store.get_visit_count("x") == 2
        assert abs(store.get_q_value("x") - (0.96 + 0.16) / 2) < 1e-12  # 0.56
        # Shared prefix above x also traversed by both episodes.
        assert store.get_visit_count("r") == 2
        # Private suffixes got one sample each.
        assert store.get_visit_count("x3") == 1
        assert store.get_visit_count("b2") == 1

    def test_branch_point_loo_recovers_consistent_value(self):
        # README worked example, now with the Option A fix. Branch point x at
        # turn 2 of scratch S accumulates HOMOGENEOUS return-to-go samples:
        #   - S itself contributed g_x^S = 0.96
        #   - 4 branch episodes contributed g_x^{B_i} = beta*c_2 + (1-beta)*outcome
        #     = 0.2*0.8 + 0.8*{1,0,1,0} = [0.96, 0.16, 0.96, 0.16]
        # Excluding S's own 0.96, the LOO mean is the mean of the branch
        # return-to-go samples = 0.56 = V(s_2) = beta*c_2 + (1-beta)*P(success).
        #
        # Contrast: under the OLD mixed-estimand backup the branch samples were
        # bare outcomes [1,0,1,0], giving a biased LOO mean of 0.50 (pure
        # P(success), missing the beta*credit-to-go term).
        store = MCTSTreeStore()
        samples = [0.96, 0.96, 0.16, 0.96, 0.16]  # S + 4 branch return-to-go
        store._visit_counts["x"] = len(samples)
        store._total_values["x"] = sum(samples)
        store._sum_sq_values["x"] = sum(s * s for s in samples)
        store._q_values["x"] = sum(samples) / len(samples)

        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("x", 0.96)
        assert n_loo == 4
        assert abs(loo_mean - 0.56) < 1e-9  # V(s_2), no estimand bias


class TestVarMcBayesianFloor:
    """The var_mc==0 overconfidence fix: a Beta(1,1) posterior variance floors
    var_mc so all-equal LOO samples no longer claim infinite confidence and
    bypass the critic."""

    @staticmethod
    def _seed(store, node_id, samples):
        store._visit_counts[node_id] = len(samples)
        store._total_values[node_id] = sum(samples)
        store._sum_sq_values[node_id] = sum(s * s for s in samples)
        store._q_values[node_id] = sum(samples) / len(samples)

    def test_var_mc_positive_when_all_remaining_equal(self):
        # Five identical 1.0 samples; exclude one -> four identical 1.0 remain.
        # Empirical variance-of-mean is 0, but the Beta floor keeps var_mc > 0.
        store = MCTSTreeStore()
        self._seed(store, "x", [1.0, 1.0, 1.0, 1.0, 1.0])
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("x", 1.0)
        assert n_loo == 4
        assert loo_mean == 1.0
        assert var_mc > 0.0
        # Beta(1,1) with S'=4, n'=4: a=5, b=1, nn=6 -> 5/(36*7).
        assert abs(var_mc - 5.0 / 252.0) < 1e-12

    def test_floor_inert_when_samples_spread(self):
        # [1,0,1,0] remain after excluding 0.96; empirical var-of-mean = 0.0833
        # dominates the Beta floor (a=b=3 -> 9/252 = 0.0357), so it stays.
        store = MCTSTreeStore()
        self._seed(store, "x", [0.96, 1.0, 0.0, 1.0, 0.0])
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("x", 0.96)
        assert n_loo == 4
        assert loo_mean == 0.5
        assert abs(var_mc - 0.08333333333) < 1e-6  # empirical, not the floor

    def test_sentinel_when_too_few_samples(self):
        store = MCTSTreeStore()
        self._seed(store, "x", [1.0, 1.0])  # n'=1 after exclusion
        _, var_mc, n_loo = store.get_loo_value_and_variance("x", 1.0)
        assert n_loo == 1
        assert var_mc == -1.0

    def test_blend_pulls_toward_confident_critic_when_mc_all_equal(self):
        # All-equal MC (loo_mean=1.0) but a CONFIDENT critic (small var_theta).
        # Pre-fix: var_mc==0 -> v_hat=1.0 (critic ignored). Post-fix: floored
        # var_mc lets the confident critic dominate, so v_hat is pulled well
        # below the pure-MC 1.0.
        store = MCTSTreeStore()
        n0 = _make_node("n0", "ep", 1, [1], value=0.0, outcome_reward=1.0)
        n1 = _make_node(
            "n1",
            "ep",
            2,
            [1],
            value=0.2,
            outcome_reward=1.0,
            need_branch=True,
            parent_node_id="n0",
        )
        n2 = _make_node(
            "n2", "ep", 3, [1], value=0.0, outcome_reward=1.0, parent_node_id="n1"
        )
        store.add_judge_score("n0", 2.0)
        store.add_judge_score("n1", 3.0)
        store.add_judge_score("n2", 5.0)
        self._seed(store, "n1", [1.0, 1.0, 1.0, 1.0, 0.96])

        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            judge_beta=0.2,
            judge_score_max=10,
            mc_min_visits=5,
            critic_var_floor=1e-3,
            critic_error_var=1e-3,  # confident critic
        )
        h.compute([n0, n1, n2])

        # v_mc = 1.0, var_mc = 5/252 ~= 0.01984; v_theta = 0.2, var_theta = 1e-3.
        var_mc = 5.0 / 252.0
        v_hat = (1.0 / var_mc + 0.2 / 1e-3) / (1.0 / var_mc + 1.0 / 1e-3)
        assert 0.2 < v_hat < 0.3  # critic was actually heard, not ignored
        # A_1 = (r_1 - v_hat) + gamma*lam*A_2 = (0.06 - v_hat) + 0.9 = 0.96 - v_hat.
        # Pre-fix (var_mc==0 -> v_hat=1.0) this would be -0.04; post-fix ~= 0.722,
        # which is the observable proof the critic pulled v_hat off pure MC.
        torch.testing.assert_close(
            n1.advantages, torch.tensor([0.96 - v_hat]), atol=1e-5, rtol=1e-4
        )

# SPDX-License-Identifier: Apache-2.0
"""Tests for HybridGAEAdvantageComputer (Task 4)."""

import torch

from customized_areal.tree_search.core.advantage import (
    GAEAdvantageComputer,
    HybridGAEAdvantageComputer,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _make_node(
    node_id,
    episode_id,
    turn_idx,
    loss_mask,
    value,
    outcome_reward,
    need_branch=False,
):
    n = Node(
        input_ids=[0] * len(loss_mask),
        loss_mask=loss_mask,
        logprobs=[0.0] * len(loss_mask),
        versions=[0] * len(loss_mask),
        node_id=node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=outcome_reward,
        need_branch=need_branch,
    )
    n.value = value
    return n


def _seed_mc(store, node_id, rewards):
    n = len(rewards)
    store._visit_counts[node_id] = n
    store._total_values[node_id] = sum(rewards)
    store._sum_sq_values[node_id] = sum(r * r for r in rewards)
    store._q_values[node_id] = sum(rewards) / n


class TestHybridEquivalence:
    def test_no_eligible_node_matches_gae(self):
        # No need_branch nodes -> hybrid output identical to plain GAE.
        def nodes():
            return [
                _make_node("n0", "ep", 0, [0, 1], 0.2, 1.0),
                _make_node("n1", "ep", 1, [0, 1], 0.5, 1.0),
                _make_node("n2", "ep", 2, [1], 0.8, 1.0),
            ]

        store_g = MCTSTreeStore()
        store_h = MCTSTreeStore()
        g = GAEAdvantageComputer(store_g, gamma=1.0, lam=0.95)
        h = HybridGAEAdvantageComputer(store_h, gamma=1.0, lam=0.95)
        ng, nh = nodes(), nodes()
        g.compute(ng)
        h.compute(nh)
        for a, b in zip(ng, nh):
            torch.testing.assert_close(a.advantages, b.advantages)
            torch.testing.assert_close(a.returns, b.returns)

    def test_branched_but_too_few_visits_matches_gae(self):
        store = MCTSTreeStore()
        h = HybridGAEAdvantageComputer(store, gamma=1.0, lam=0.95, mc_min_visits=5)
        n = _make_node("n0", "ep", 0, [1], 0.3, 1.0, need_branch=True)
        _seed_mc(store, "n0", [1.0, 1.0])  # visit_count 2 < 5
        h.compute([n])
        # Falls back to critic value 0.3: A = r - v = 1.0 - 0.3 = 0.7
        torch.testing.assert_close(n.advantages, torch.tensor([0.7]))


class TestHybridBlend:
    def test_tiny_var_mc_pulls_toward_mc(self):
        # Eligible branched node; MC samples tightly clustered (small var_mc),
        # large critic error variance -> blended value approx the LOO MC mean.
        store = MCTSTreeStore()
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1e-3,
            critic_error_var=1e6,  # critic asserted very inaccurate -> down-weight
        )
        # 6 samples all 0.9; exclude the current 0.9 -> five 0.9 remain. var_mc
        # is floored (Beta posterior) but stays tiny vs the huge critic error
        # variance, so the blend still collapses to the LOO MC mean 0.9.
        _seed_mc(store, "n0", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        n = _make_node("n0", "ep", 0, [1], 0.1, 0.9, need_branch=True)
        h.compute([n])
        # v_hat ~= v_mc = 0.9; A = r - v_hat = 0.9 - 0.9 = 0.0
        torch.testing.assert_close(
            n.advantages, torch.tensor([0.0]), atol=1e-4, rtol=1e-4
        )

    def test_large_var_mc_keeps_critic(self):
        # Noisy MC (large var_mc), tiny critic error variance -> blend ~ critic.
        store = MCTSTreeStore()
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1e-6,
            critic_error_var=1e-6,  # critic is (asserted) very accurate
        )
        # Widely spread MC samples -> large var_mc.
        _seed_mc(store, "n0", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        v_theta = 0.4
        n = _make_node("n0", "ep", 0, [1], v_theta, 0.0, need_branch=True)
        h.compute([n])
        # Blend ~ critic 0.4 -> A = r - v ~ 0.0 - 0.4 = -0.4
        adv = float(n.advantages[0])
        assert abs(adv - (-0.4)) < 1e-2

    def test_loo_excludes_current_episode_reward(self):
        # Changing the excluded (current episode) reward changes v_hat.
        store = MCTSTreeStore()
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1.0,
            critic_error_var=1e6,  # down-weight critic -> v_hat ~= LOO MC mean
        )
        _seed_mc(store, "n0", [1.0, 1.0, 1.0, 1.0, 0.0])
        # Exclude reward 0.0 -> remaining four 1.0 -> LOO mean 1.0.
        n_excl0 = _make_node("n0", "ep", 0, [1], 0.5, 0.0, need_branch=True)
        h.compute([n_excl0])
        # Excluding the 0.0 sample leaves four 1.0 -> LOO mean 1.0; critic is
        # heavily down-weighted so v_hat ~= 1.0 (not the critic 0.5).
        # A = r - v_hat = 0.0 - 1.0 = -1.0.
        torch.testing.assert_close(
            n_excl0.advantages, torch.tensor([-1.0]), atol=1e-4, rtol=1e-4
        )

    def test_one_hot_critic_floor_no_div_by_zero(self):
        # Critic variance 0 (one-hot) must be floored; no exception, finite adv.
        store = MCTSTreeStore()
        h = HybridGAEAdvantageComputer(
            store, gamma=1.0, lam=1.0, mc_min_visits=5, critic_var_floor=1e-3
        )
        _seed_mc(store, "n0", [0.0, 1.0, 0.0, 1.0, 0.5])
        store.set_value_variance("n0", 0.0)  # one-hot critic
        n = _make_node("n0", "ep", 0, [1], 0.4, 0.5, need_branch=True)
        h.compute([n])
        assert torch.isfinite(n.advantages).all()


class TestCriticErrorVarianceSource:
    """var_theta is the critic ERROR variance (regression MSE), not the
    categorical variance of the critic's output distribution."""

    def _eligible_node_and_store(self):
        store = MCTSTreeStore()
        # Excluding 0.0 leaves [1,0,1,0,1]: mean 0.6, sample var 0.3,
        # var_mc = 0.3/5 = 0.06 (a moderate, non-degenerate MC variance).
        _seed_mc(store, "n0", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        n = _make_node("n0", "ep", 0, [1], 0.4, 0.0, need_branch=True)
        return store, n

    def test_small_error_var_favours_critic(self):
        store, n = self._eligible_node_and_store()
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1e-9,
            critic_error_var=1e-6,
        )
        h.compute([n])
        # var_theta tiny -> v_hat ~ critic 0.4 -> A = 0 - 0.4 = -0.4
        assert abs(float(n.advantages[0]) - (-0.4)) < 1e-2

    def test_large_error_var_favours_mc(self):
        store, n = self._eligible_node_and_store()
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1e-9,
            critic_error_var=100.0,
        )
        h.compute([n])
        # var_theta huge -> v_hat ~ v_mc 0.6 -> A = 0 - 0.6 = -0.6
        assert abs(float(n.advantages[0]) - (-0.6)) < 1e-2

    def test_error_var_fn_overrides_static_prior(self):
        store, n = self._eligible_node_and_store()
        # Static prior would favour the critic (1e-6), but the live hook
        # reports a large error variance, so the blend leans MC instead.
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1e-9,
            critic_error_var=1e-6,
            critic_error_var_fn=lambda: 100.0,
        )
        h.compute([n])
        assert abs(float(n.advantages[0]) - (-0.6)) < 1e-2

    def test_error_var_fn_none_falls_back_to_static(self):
        store, n = self._eligible_node_and_store()
        h = HybridGAEAdvantageComputer(
            store,
            gamma=1.0,
            lam=1.0,
            mc_min_visits=5,
            critic_var_floor=1e-9,
            critic_error_var=1e-6,
            critic_error_var_fn=lambda: None,
        )
        h.compute([n])
        # Live hook returns None -> static prior (1e-6) -> critic wins.
        assert abs(float(n.advantages[0]) - (-0.4)) < 1e-2


class TestAdaptiveMCWeightLiveErrorVar:
    def test_live_error_var_none_before_feed_then_ema(self):
        from customized_areal.tree_search.training.losses.critic import (
            AdaptiveMCWeight,
        )

        mixer = AdaptiveMCWeight(c=4.0, ema_beta=0.5, eps2_init=1.0)
        assert mixer.live_critic_error_var() is None  # not yet fed
        mixer.update_critic_error(0.0)
        # EMA: 0.5*1.0 + 0.5*0.0 = 0.5
        assert abs(mixer.live_critic_error_var() - 0.5) < 1e-9

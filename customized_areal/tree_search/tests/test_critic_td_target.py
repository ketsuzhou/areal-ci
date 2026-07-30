# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified TD/MC critic target and adaptive MC weighting."""

import pytest

from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.training.losses.critic import (
    AdaptiveMCWeight,
    compute_critic_targets,
)


def _node(node_id, turn_idx, *, value=0.0, reward=0.0, episode_id="ep", query_id="q"):
    return Node(
        input_ids=[0, 0],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[0, 0],
        node_id=node_id,
        turn_idx=turn_idx,
        value=value,
        outcome_reward=reward,
        episode_id=episode_id,
        query_id=query_id,
    )


class TestComputeCriticTargets:
    def test_mc_weight_one_reproduces_mcts_qvalue(self):
        # Backward-compat: mc_weight=1.0 == q_value / target_scale (clamped).
        store = MCTSTreeStore()
        store._q_values["a"] = 0.4
        store._q_values["b"] = 3.0  # clamps to 1.0 after /scale=2.0
        nodes = [_node("a", 1), _node("b", 2)]
        targets = compute_critic_targets(
            nodes, tree_store=store, mc_weight=1.0, target_scale=2.0
        )
        assert targets == pytest.approx([0.2, 1.0])

    def test_none_weight_defaults_to_mc(self):
        store = MCTSTreeStore()
        store._q_values["a"] = 0.5
        targets = compute_critic_targets(
            [_node("a", 1)], tree_store=store, mc_weight=None
        )
        assert targets == pytest.approx([0.5])

    def test_one_step_td_pure(self):
        # 2-turn episode, mc_weight=0, n_steps=1, gamma=1.
        # turn0 target = gamma * v(turn1); terminal target = reward (no bootstrap).
        nodes = [
            _node("a", 1, value=0.3),
            _node("b", 2, value=0.8, reward=1.0),
        ]
        targets = compute_critic_targets(nodes, mc_weight=0.0, n_steps=1, gamma=1.0)
        assert targets == pytest.approx([0.8, 1.0])

    def test_one_step_td_discounted(self):
        nodes = [
            _node("a", 1, value=0.5),
            _node("b", 2, value=0.9, reward=1.0),
        ]
        targets = compute_critic_targets(nodes, mc_weight=0.0, n_steps=1, gamma=0.5)
        # t0: 0 + 0.5 * 0.9 = 0.45 ; terminal: 1.0
        assert targets == pytest.approx([0.45, 1.0])

    def test_large_nsteps_is_monte_carlo_return(self):
        # n_steps beyond horizon -> full discounted return, ignoring critic value.
        nodes = [
            _node("a", 1, value=0.3, reward=0.0),
            _node("b", 2, value=0.8, reward=1.0),
        ]
        targets = compute_critic_targets(nodes, mc_weight=0.0, n_steps=10, gamma=0.9)
        # t0: r0 + 0.9 * r1 = 0 + 0.9 * 1.0 = 0.9 ; terminal: 1.0
        assert targets == pytest.approx([0.9, 1.0])

    def test_blend_half(self):
        store = MCTSTreeStore()
        store._q_values["a"] = 0.2
        store._q_values["b"] = 1.0
        nodes = [
            _node("a", 1, value=0.3),
            _node("b", 2, value=0.8, reward=1.0),
        ]
        # td: [0.8, 1.0] ; mc: [0.2, 1.0] ; blend w=0.5 -> [0.5, 1.0]
        targets = compute_critic_targets(
            nodes, tree_store=store, mc_weight=0.5, n_steps=1, gamma=1.0
        )
        assert targets == pytest.approx([0.5, 1.0])

    def test_clamped_to_unit_interval(self):
        nodes = [_node("a", 1, value=0.0, reward=5.0, episode_id="")]
        targets = compute_critic_targets(nodes, mc_weight=0.0, n_steps=1)
        assert targets == pytest.approx([1.0])

    def test_alignment_across_episodes(self):
        # Two episodes; output order must match input order.
        nodes = [
            _node("a", 1, value=0.3, episode_id="e1"),
            _node("x", 1, value=0.1, reward=0.0, episode_id="e2"),
            _node("b", 2, value=0.8, reward=1.0, episode_id="e1"),
            _node("y", 2, value=0.5, reward=0.0, episode_id="e2"),
        ]
        targets = compute_critic_targets(nodes, mc_weight=0.0, n_steps=1, gamma=1.0)
        # e1: a->0.8, b->1.0 ; e2: x->0.5 (boot on y), y->0.0 (terminal reward 0)
        assert targets == pytest.approx([0.8, 0.5, 1.0, 0.0])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_steps": 0},
            {"gamma": 1.5},
            {"gamma": -0.1},
            {"target_scale": 0.0},
            {"mc_weight": 1.5},
            {"mc_weight": -0.1},
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(ValueError):
            compute_critic_targets([_node("a", 1)], **kwargs)


class TestAdaptiveMCWeight:
    def test_warmup_forces_mc(self):
        mixer = AdaptiveMCWeight(warmup_steps=3, c=4.0)
        assert mixer.weight(1) == 1.0
        mixer.update_critic_error(1.0)
        assert mixer.weight(100) == 1.0  # still in warmup

    def test_inverse_variance_formula(self):
        mixer = AdaptiveMCWeight(c=4.0, warmup_steps=0, eps2_init=1.0)
        # N * eps2 / (N * eps2 + c) with eps2=1, c=4
        assert mixer.weight(4) == pytest.approx(0.5)
        assert mixer.weight(1) == pytest.approx(0.2)

    def test_high_visit_trusts_mc(self):
        mixer = AdaptiveMCWeight(c=4.0, warmup_steps=0, w_max=0.95)
        assert mixer.weight(10_000) == pytest.approx(0.95)  # clamped

    def test_low_error_trusts_td(self):
        mixer = AdaptiveMCWeight(c=4.0, warmup_steps=0, ema_beta=0.0, w_min=0.05)
        # ema_beta=0 -> eps2 jumps to the observed mse immediately.
        mixer.update_critic_error(0.0)
        assert mixer.weight(4) == pytest.approx(0.05)  # eps2->0 -> weight->w_min

    def test_per_node_weight_via_visit_counts(self):
        store = MCTSTreeStore()
        store._q_values["a"] = 0.2
        store._q_values["b"] = 0.2
        store._visit_counts["a"] = 10_000  # reliable MCTS -> trust MC
        store._visit_counts["b"] = 1  # noisy -> trust TD
        nodes = [
            _node("a", 1, value=0.9),
            _node("b", 2, value=0.9, reward=0.2),
        ]
        mixer = AdaptiveMCWeight(c=4.0, warmup_steps=0, eps2_init=1.0)
        targets = compute_critic_targets(
            nodes, tree_store=store, mc_weight=mixer, n_steps=1, gamma=1.0
        )
        # a: td=0.9, mc=0.2, w~0.95 -> ~0.235 ; b: terminal td=0.2, mc=0.2 -> 0.2
        assert targets[0] == pytest.approx(0.235, abs=1e-3)
        assert targets[1] == pytest.approx(0.2, abs=1e-3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"c": 0.0},
            {"ema_beta": 1.0},
            {"warmup_steps": -1},
            {"w_min": 0.6, "w_max": 0.5},
            {"eps2_init": 0.0},
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(ValueError):
            AdaptiveMCWeight(**kwargs)

    def test_update_critic_error_negative_raises(self):
        with pytest.raises(ValueError):
            AdaptiveMCWeight(warmup_steps=0).update_critic_error(-1.0)


class TestComputeCriticTargetsJudgeShaping:
    def test_dense_rewards_in_td_target(self):
        # judge scores equal -> jbar=[0.5,0.5]; beta=0.5; outcome=1.0.
        # dense rewards = [0.25, 0.75]; pure TD (mc_weight=0), n_steps large.
        store = MCTSTreeStore()
        store.add_judge_score("a", 4)
        store.add_judge_score("b", 4)
        nodes = [
            _node("a", 1, value=0.3),
            _node("b", 2, value=0.8, reward=1.0),
        ]
        targets = compute_critic_targets(
            nodes,
            tree_store=store,
            mc_weight=0.0,
            n_steps=10,
            gamma=1.0,
            judge_beta=0.5,
            judge_score_max=10,
        )
        # Monte-Carlo return from dense rewards: t0 = r0 + r1 = 0.25 + 0.75 = 1.0;
        # terminal = r1 = 0.75.
        assert targets == pytest.approx([1.0, 0.75])

    def test_judge_beta_zero_unchanged(self):
        store = MCTSTreeStore()
        store.add_judge_score("a", 4)
        store.add_judge_score("b", 4)
        nodes = [
            _node("a", 1, value=0.3),
            _node("b", 2, value=0.8, reward=1.0),
        ]
        baseline = compute_critic_targets(
            nodes, tree_store=store, mc_weight=0.0, n_steps=1, gamma=1.0
        )
        with_flag_off = compute_critic_targets(
            nodes,
            tree_store=store,
            mc_weight=0.0,
            n_steps=1,
            gamma=1.0,
            judge_beta=0.0,
        )
        assert with_flag_off == pytest.approx(baseline)

    def test_no_judge_signal_falls_back(self):
        # beta>0 but no scores -> sparse terminal-only behavior.
        store = MCTSTreeStore()
        nodes = [
            _node("a", 1, value=0.3),
            _node("b", 2, value=0.8, reward=1.0),
        ]
        targets = compute_critic_targets(
            nodes,
            tree_store=store,
            mc_weight=0.0,
            n_steps=1,
            gamma=1.0,
            judge_beta=0.5,
        )
        # Identical to the sparse one-step TD target.
        assert targets == pytest.approx([0.8, 1.0])

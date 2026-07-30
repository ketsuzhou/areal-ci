# SPDX-License-Identifier: Apache-2.0
"""Tests for build_episode_process_rewards (dense judge process rewards)."""

import math

import pytest

from customized_areal.tree_search.core.process_reward import (
    build_episode_process_rewards,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _node(node_id: str, turn_idx: int, outcome: float = 0.0) -> Node:
    return Node(
        input_ids=[1, 2],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[-1, -1],
        node_id=node_id,
        turn_idx=turn_idx,
        outcome_reward=outcome,
    )


class TestSparseFallback:
    def test_no_store(self):
        nodes = [_node("a", 1), _node("b", 2, outcome=1.0)]
        rewards = build_episode_process_rewards(nodes, None, beta=0.5, score_max=10)
        assert rewards == [0.0, 1.0]

    def test_beta_zero(self):
        store = MCTSTreeStore()
        store.add_judge_score("a", 5)
        store.add_judge_score("b", 5)
        nodes = [_node("a", 1), _node("b", 2, outcome=1.0)]
        rewards = build_episode_process_rewards(nodes, store, beta=0.0, score_max=10)
        assert rewards == [0.0, 1.0]

    def test_no_judge_signal(self):
        # No node judged -> sparse, outcome NOT down-weighted by (1 - beta).
        store = MCTSTreeStore()
        nodes = [_node("a", 1), _node("b", 2, outcome=1.0)]
        rewards = build_episode_process_rewards(nodes, store, beta=0.5, score_max=10)
        assert rewards == [0.0, 1.0]

    def test_all_zero_scores(self):
        store = MCTSTreeStore()
        store.add_judge_score("a", 0)
        store.add_judge_score("b", 0)
        nodes = [_node("a", 1), _node("b", 2, outcome=1.0)]
        rewards = build_episode_process_rewards(nodes, store, beta=0.5, score_max=10)
        assert rewards == [0.0, 1.0]

    def test_empty(self):
        assert (
            build_episode_process_rewards([], MCTSTreeStore(), beta=0.5, score_max=10)
            == []
        )


class TestDensePath:
    def test_basic_distribution(self):
        store = MCTSTreeStore()
        store.add_judge_score("a", 3)  # jbar = 3/9
        store.add_judge_score("b", 6)  # jbar = 6/9 (terminal)
        nodes = [_node("a", 1), _node("b", 2, outcome=1.0)]
        beta = 0.4
        rewards = build_episode_process_rewards(nodes, store, beta=beta, score_max=10)
        jbar_a, jbar_b = 3 / 9, 6 / 9
        assert math.isclose(rewards[0], beta * jbar_a)
        assert math.isclose(rewards[1], (1 - beta) * 1.0 + beta * jbar_b)

    def test_return_bound_and_value(self):
        store = MCTSTreeStore()
        store.add_judge_score("a", 2)
        store.add_judge_score("b", 5)
        store.add_judge_score("c", 3)
        nodes = [
            _node("a", 1),
            _node("b", 2),
            _node("c", 3, outcome=0.0),
        ]
        beta = 0.3
        rewards = build_episode_process_rewards(nodes, store, beta=beta, score_max=10)
        G = sum(rewards)
        # G = beta + (1 - beta) * outcome ; outcome = 0
        assert math.isclose(G, beta)
        assert 0.0 <= G <= 1.0

    def test_return_bound_success(self):
        store = MCTSTreeStore()
        store.add_judge_score("a", 4)
        store.add_judge_score("b", 4)
        nodes = [_node("a", 1), _node("b", 2, outcome=1.0)]
        for beta in (0.0, 0.2, 0.5, 1.0):
            rewards = build_episode_process_rewards(
                nodes, store, beta=beta, score_max=10
            )
            G = sum(rewards)
            assert math.isclose(G, beta + (1 - beta) * 1.0)
            assert 0.0 <= G <= 1.0

    def test_partial_scoring_unjudged_turn_zero(self):
        # Middle turn unjudged contributes 0 credit but episode still dense.
        store = MCTSTreeStore()
        store.add_judge_score("a", 5)
        store.add_judge_score("c", 5)
        nodes = [_node("a", 1), _node("b", 2), _node("c", 3, outcome=1.0)]
        beta = 0.5
        rewards = build_episode_process_rewards(nodes, store, beta=beta, score_max=10)
        assert math.isclose(rewards[1], 0.0)  # unjudged middle
        G = sum(rewards)
        assert 0.0 <= G <= 1.0

    def test_branching_average_then_renormalize_keeps_bound(self):
        # Shared prefix node judged by two episodes -> mean; bound still holds.
        store = MCTSTreeStore()
        store.add_judge_score("shared", 8)
        store.add_judge_score("shared", 2)  # mean 5
        store.add_judge_score("leaf", 5)
        nodes = [_node("shared", 1), _node("leaf", 2, outcome=1.0)]
        beta = 0.6
        rewards = build_episode_process_rewards(nodes, store, beta=beta, score_max=10)
        # mean_raw = [5, 5] -> jbar = [0.5, 0.5]
        assert math.isclose(rewards[0], beta * 0.5)
        assert math.isclose(rewards[1], (1 - beta) * 1.0 + beta * 0.5)
        assert math.isclose(sum(rewards), beta + (1 - beta) * 1.0)


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_beta(self, bad):
        with pytest.raises(ValueError, match="beta"):
            build_episode_process_rewards([], MCTSTreeStore(), beta=bad, score_max=10)

    def test_bad_score_max(self):
        with pytest.raises(ValueError, match="score_max"):
            build_episode_process_rewards([], MCTSTreeStore(), beta=0.5, score_max=0)

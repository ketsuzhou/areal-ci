# SPDX-License-Identifier: Apache-2.0
"""Tests for the leave-one-out MC value/variance helper (Task 1)."""

from customized_areal.tree_search.core.tree_store import MCTSTreeStore


def _populate(store, node_id, rewards):
    """Directly seed a node's MC aggregates with the given sample returns."""
    n = len(rewards)
    store._visit_counts[node_id] = n
    store._total_values[node_id] = sum(rewards)
    store._sum_sq_values[node_id] = sum(r * r for r in rewards)
    store._q_values[node_id] = sum(rewards) / n


class TestLeaveOneOut:
    def test_identical_returns_zero_variance(self):
        store = MCTSTreeStore()
        _populate(store, "a", [1.0, 1.0, 1.0, 1.0, 1.0])
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("a", 1.0)
        assert n_loo == 4
        assert loo_mean == 1.0
        assert var_mc == 0.0

    def test_known_mixed_returns(self):
        # Samples [0, 1, 0, 1, 0], exclude one 0.0 -> remaining [0,1,1,0] effectively
        # via aggregate removal: S=2, Q=2 -> S'=2, Q'=2, n'=4.
        store = MCTSTreeStore()
        _populate(store, "a", [0.0, 1.0, 0.0, 1.0, 0.0])
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("a", 0.0)
        assert n_loo == 4
        # loo_mean = S'/n' = 2/4 = 0.5
        assert abs(loo_mean - 0.5) < 1e-9
        # loo_var = (Q' - S'^2/n')/(n'-1) = (2 - 4/4)/3 = (2-1)/3 = 1/3
        # var_mc = loo_var / n' = (1/3)/4 = 1/12
        assert abs(var_mc - (1.0 / 12.0)) < 1e-9

    def test_excluding_reward_removes_that_sample(self):
        store = MCTSTreeStore()
        _populate(store, "a", [0.0, 0.0, 0.0, 0.0, 4.0])
        # Exclude the 4.0 outlier -> remaining all-zero -> mean 0, var 0.
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("a", 4.0)
        assert n_loo == 4
        assert abs(loo_mean) < 1e-9
        assert abs(var_mc) < 1e-9

    def test_insufficient_loo_samples_returns_sentinel(self):
        store = MCTSTreeStore()
        _populate(store, "a", [1.0, 0.0])  # n=2 -> n_loo=1 < 2
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("a", 1.0)
        assert var_mc == -1.0
        assert n_loo == 1

    def test_missing_node_returns_sentinel(self):
        store = MCTSTreeStore()
        loo_mean, var_mc, n_loo = store.get_loo_value_and_variance("nope", 0.0)
        assert var_mc == -1.0
        assert n_loo == 0

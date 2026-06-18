# SPDX-License-Identifier: Apache-2.0
"""Tests for GAEAdvantageComputer, Node.value, and store value accessors."""

import torch

from customized_areal.tree_search.core.advantage import GAEAdvantageComputer
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _make_node(node_id, episode_id, turn_idx, loss_mask, value, outcome_reward):
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
    )
    n.value = value
    return n


class TestStoreValueAccessors:
    def test_set_get_has_value(self):
        store = MCTSTreeStore()
        assert store.has_value("a") is False
        assert store.get_value("a", default=-1.0) == -1.0
        store.set_value("a", 0.7)
        assert store.has_value("a") is True
        assert store.get_value("a") == 0.7

    def test_clear_resets_values(self):
        store = MCTSTreeStore()
        store.set_value("a", 0.5)
        store.clear()
        assert store.has_value("a") is False


class TestNodeValueField:
    def test_default_zero(self):
        n = Node(input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0])
        assert n.value == 0.0


class TestGAEComputer:
    def test_closed_form_three_turns_gamma1_lambda095(self):
        # Episode with 3 turns; values v=[0.2,0.5,0.8]; terminal reward=1.0.
        # gamma=1, lambda=0.95. Sparse reward only at terminal.
        store = MCTSTreeStore()
        gae = GAEAdvantageComputer(store, gamma=1.0, lam=0.95)
        nodes = [
            _make_node("n0", "ep", 0, [0, 1], 0.2, 1.0),
            _make_node("n1", "ep", 1, [0, 1], 0.5, 1.0),
            _make_node("n2", "ep", 2, [1], 0.8, 1.0),
        ]
        gae.compute(nodes)

        # Manual GAE:
        # deltas (next_value: v(s_{t+1}), terminal bootstrap 0):
        # delta2 = r2 + g*0 - v2 = 1.0 - 0.8 = 0.2
        # delta1 = 0 + 1*0.8 - 0.5 = 0.3
        # delta0 = 0 + 1*0.5 - 0.2 = 0.3
        # A2 = 0.2
        # A1 = delta1 + g*l*A2 = 0.3 + 0.95*0.2 = 0.49
        # A0 = delta0 + g*l*A1 = 0.3 + 0.95*0.49 = 0.7655
        a = {0: 0.7655, 1: 0.49, 2: 0.2}
        v = {0: 0.2, 1: 0.5, 2: 0.8}
        for i, n in enumerate(nodes):
            adv_vals = n.advantages[n.advantages != 0]
            # advantage broadcast over response positions
            expected_adv = a[i]
            torch.testing.assert_close(
                n.advantages,
                torch.tensor(n.loss_mask, dtype=torch.float32) * expected_adv,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                n.returns,
                torch.tensor(n.loss_mask, dtype=torch.float32) * (a[i] + v[i]),
                rtol=1e-5,
                atol=1e-6,
            )
            del adv_vals

    def test_single_turn_episode(self):
        # T=0 terminal: delta0 = r - v; A0 = r - v; ret0 = r.
        store = MCTSTreeStore()
        gae = GAEAdvantageComputer(store, gamma=1.0, lam=0.95)
        n = _make_node("n0", "ep", 0, [0, 1, 1], 0.3, 1.0)
        gae.compute([n])
        mask = torch.tensor([0, 1, 1], dtype=torch.float32)
        torch.testing.assert_close(n.advantages, mask * (1.0 - 0.3))
        torch.testing.assert_close(n.returns, mask * 1.0)

    def test_missing_values_default_zero(self):
        # Nodes with no value set -> treated as 0.0, runs without error.
        store = MCTSTreeStore()
        gae = GAEAdvantageComputer(store)
        n = Node(
            input_ids=[0, 0],
            loss_mask=[0, 1],
            logprobs=[0.0, 0.0],
            versions=[0, 0],
            node_id="x",
            episode_id="ep",
            turn_idx=0,
            outcome_reward=1.0,
        )
        gae.compute([n])
        assert n.advantages is not None
        # A0 = r - v = 1.0 - 0.0 = 1.0
        torch.testing.assert_close(
            n.advantages, torch.tensor([0.0, 1.0], dtype=torch.float32)
        )

    def test_store_records_advantage_and_return(self):
        store = MCTSTreeStore()
        gae = GAEAdvantageComputer(store, gamma=1.0, lam=0.95)
        n = _make_node("n0", "ep", 0, [1], 0.4, 1.0)
        gae.compute([n])
        assert store.get_normalized_advantage("n0") == 1.0 - 0.4
        assert store.get_normalized_return("n0") == 1.0

    def test_two_episodes_independent(self):
        store = MCTSTreeStore()
        gae = GAEAdvantageComputer(store, gamma=1.0, lam=1.0)
        nodes = [
            _make_node("a0", "epA", 0, [1], 0.0, 1.0),
            _make_node("b0", "epB", 0, [1], 0.0, 0.0),
        ]
        gae.compute(nodes)
        # epA terminal reward 1.0 -> A0 = 1.0; epB reward 0 -> A0 = 0.0
        torch.testing.assert_close(nodes[0].advantages, torch.tensor([1.0]))
        torch.testing.assert_close(nodes[1].advantages, torch.tensor([0.0]))

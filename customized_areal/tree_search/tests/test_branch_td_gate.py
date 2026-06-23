# SPDX-License-Identifier: Apache-2.0
"""Tests for the TD-error gate in select_branch_candidate (Task 6)."""

from customized_areal.tree_search.core.customized_grouped_workflow import (
    select_branch_candidate,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _cand(
    node_id,
    episode_id,
    turn_idx,
    entropy,
    value=0.0,
    outcome_reward=0.0,
    need_branch=True,
):
    return Node(
        input_ids=[0, 0],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[0, 0],
        node_id=node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=outcome_reward,
        need_branch=need_branch,
        task_id="task" if need_branch else "",
        branch_sandbox_id="sb" if need_branch else None,
        entropy_stats={"max_entropy": entropy},
    )


def _store(values):
    s = MCTSTreeStore()
    for nid, v in values.items():
        s.set_value(nid, v)
    return s


class TestThresholdZeroReproducesEntropy:
    def test_picks_highest_entropy(self):
        nodes = [
            _cand("a", "epA", 1, entropy=0.9, value=0.5),
            _cand("b", "epB", 1, entropy=0.2, value=0.5),
        ]
        store = _store({"a": 0.5, "b": 0.5})
        chosen = select_branch_candidate(
            nodes, "q", tree_store=store, td_threshold=0.0, gamma=1.0
        )
        assert chosen.node_id == "a"

    def test_no_tree_store_entropy_only(self):
        nodes = [
            _cand("a", "epA", 1, entropy=0.9),
            _cand("b", "epB", 1, entropy=0.2),
        ]
        chosen = select_branch_candidate(nodes, "q")
        assert chosen.node_id == "a"


class TestGate:
    def test_sub_threshold_dropped(self):
        # A: high entropy but delta 0 (value == successor). B: lower entropy,
        # delta 0.8. Threshold 0.5 -> A dropped, B chosen despite lower entropy.
        nodes = [
            _cand("a", "epA", 1, entropy=0.9, value=0.5),
            Node(  # successor of a, value 0.5 -> delta_a = 0
                input_ids=[0, 0],
                loss_mask=[0, 1],
                logprobs=[0.0, 0.0],
                versions=[0, 0],
                node_id="a2",
                episode_id="epA",
                turn_idx=2,
                query_id="q",
                value=0.5,
            ),
            _cand("b", "epB", 1, entropy=0.2, value=0.1),
            Node(  # successor of b, value 0.9 -> delta_b = 0.8
                input_ids=[0, 0],
                loss_mask=[0, 1],
                logprobs=[0.0, 0.0],
                versions=[0, 0],
                node_id="b2",
                episode_id="epB",
                turn_idx=2,
                query_id="q",
                value=0.9,
            ),
        ]
        store = _store({"a": 0.5, "a2": 0.5, "b": 0.1, "b2": 0.9})
        chosen = select_branch_candidate(
            nodes, "q", tree_store=store, td_threshold=0.5, gamma=1.0
        )
        assert chosen.node_id == "b"

    def test_highest_entropy_among_survivors(self):
        # Two survivors both above threshold -> highest entropy wins.
        nodes = [
            _cand("a", "epA", 1, entropy=0.3, value=0.1, outcome_reward=1.0),
            _cand("b", "epB", 1, entropy=0.8, value=0.1, outcome_reward=1.0),
        ]
        store = _store({"a": 0.1, "b": 0.1})
        # both terminal: delta = |1.0 - 0.1| = 0.9 >= 0.5
        chosen = select_branch_candidate(
            nodes, "q", tree_store=store, td_threshold=0.5, gamma=1.0
        )
        assert chosen.node_id == "b"

    def test_all_dropped_returns_none(self):
        nodes = [_cand("a", "epA", 1, entropy=0.9, value=0.5)]
        store = _store({"a": 0.5})  # terminal delta = |0 - 0.5| = 0.5
        chosen = select_branch_candidate(
            nodes, "q", tree_store=store, td_threshold=1.0, gamma=1.0
        )
        assert chosen is None

    def test_terminal_uses_outcome_reward(self):
        nodes = [_cand("a", "epA", 1, entropy=0.9, value=0.2, outcome_reward=1.0)]
        store = _store({"a": 0.2})
        # terminal delta = |1.0 + 0 - 0.2| = 0.8 >= 0.8 -> kept
        chosen = select_branch_candidate(
            nodes, "q", tree_store=store, td_threshold=0.8, gamma=1.0
        )
        assert chosen.node_id == "a"

    def test_missing_critic_value_bypasses_gate(self):
        # Candidate with no critic value bypasses the gate (entropy fallback),
        # so it is kept even with a high threshold.
        nodes = [_cand("a", "epA", 1, entropy=0.9, value=0.0)]
        store = MCTSTreeStore()  # no value stored for "a"
        chosen = select_branch_candidate(
            nodes, "q", tree_store=store, td_threshold=5.0, gamma=1.0
        )
        assert chosen.node_id == "a"

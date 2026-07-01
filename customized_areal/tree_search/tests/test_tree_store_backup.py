# SPDX-License-Identifier: Apache-2.0
"""Tests for MCTS root-ward backup and branch-point aggregation (Task 0)."""

import uuid

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _wrap_leaf(nodes, *, super_id=None):
    """Wrap a list[Node] in a single leaf SuperNode for the single-agent path."""
    return SuperNode(
        node_id=super_id or str(uuid.uuid4()),
        agent_id="a",
        issue_id="i",
        task_id="t",
        nodes=list(nodes),
    )


def _make_node(node_id, episode_id, turn_idx, outcome_reward, parent_node_id=None):
    return Node(
        input_ids=[0, 0],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[0, 0],
        node_id=node_id,
        parent_node_id=parent_node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=outcome_reward,
    )


class TestRootWardBackup:
    def test_linear_episode_each_node_visited_once(self):
        # A single 3-turn linear episode: every node has exactly one sample.
        store = MCTSTreeStore()
        nodes = [
            _make_node("n0", "ep", 1, 1.0, parent_node_id=None),
            _make_node("n1", "ep", 2, 1.0, parent_node_id="n0"),
            _make_node("n2", "ep", 3, 1.0, parent_node_id="n1"),
        ]
        store.insert_super_batch([_wrap_leaf(nodes)], query_id="q")
        for nid in ("n0", "n1", "n2"):
            assert store.get_visit_count(nid) == 1
            assert store.get_q_value(nid) == 1.0
            assert store.get_total_value(nid) == 1.0
            assert store.get_sum_sq_value(nid) == 1.0

    def test_branch_point_aggregates_across_episodes(self):
        # Parent episode p0->p1 (reward 1.0). Branch episode b0->b1 (reward 0.0)
        # whose first turn links to branch point p0. p0 should accumulate both
        # episodes' returns; p1 and the branch leaves stay at one sample.
        store = MCTSTreeStore()
        parent = [
            _make_node("p0", "epP", 1, 1.0, parent_node_id=None),
            _make_node("p1", "epP", 2, 1.0, parent_node_id="p0"),
        ]
        store.insert_super_batch([_wrap_leaf(parent)], query_id="q")
        branch = [
            _make_node("b0", "epB", 2, 0.0, parent_node_id="p0"),
            _make_node("b1", "epB", 3, 0.0, parent_node_id="b0"),
        ]
        store.insert_super_batch([_wrap_leaf(branch)], query_id="q")

        # Branch point p0 saw both episodes: returns {1.0, 0.0}.
        assert store.get_visit_count("p0") == 2
        assert store.get_total_value("p0") == 1.0
        assert store.get_q_value("p0") == 0.5
        # sum of squares = 1.0^2 + 0.0^2 = 1.0
        assert store.get_sum_sq_value("p0") == 1.0

        # Non-shared nodes stay at one sample.
        assert store.get_visit_count("p1") == 1
        assert store.get_visit_count("b0") == 1
        assert store.get_visit_count("b1") == 1
        assert store.get_q_value("b0") == 0.0

    def test_cached_episode_not_double_counted(self):
        # Re-inserting the same nodes (cache hit) must not re-run the backup.
        store = MCTSTreeStore()
        nodes = [
            _make_node("n0", "ep", 1, 1.0, parent_node_id=None),
            _make_node("n1", "ep", 2, 1.0, parent_node_id="n0"),
        ]
        store.insert_super_batch([_wrap_leaf(nodes)], query_id="q")
        store.insert_super_batch(
            [_wrap_leaf(nodes)], query_id="q"
        )  # node_ids already present -> skipped
        assert store.get_visit_count("n0") == 1
        assert store.get_visit_count("n1") == 1

    def test_clear_resets_sum_sq(self):
        store = MCTSTreeStore()
        store.insert_super_batch(
            [_wrap_leaf([_make_node("n0", "ep", 1, 1.0)])], query_id="q"
        )
        assert store.get_sum_sq_value("n0") == 1.0
        store.clear()
        assert store.get_sum_sq_value("n0") == 0.0
        assert store.get_visit_count("n0") == 0

    def test_standalone_node_without_episode_id(self):
        store = MCTSTreeStore()
        store.insert_super_batch(
            [_wrap_leaf([_make_node("solo", "", 0, 0.7)])], query_id="q"
        )
        assert store.get_visit_count("solo") == 1
        assert store.get_q_value("solo") == 0.7

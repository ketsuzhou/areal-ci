"""Tests for the unified MCTSTreeStore (trajectories holds SuperNodes).

SuperNodes hold DAG topology + team env snapshot; Nodes (inside SuperNode.nodes)
hold all MCTS stats + the unified parent_node_id chain. backup_episode_terminal
walks parent_node_id across SuperNode and agent boundaries (causal flattening).

Uses SimpleNamespace stand-ins for Node (the store reads .node_id /
.parent_node_id / .outcome_reward / .episode_id / .turn_idx / .train_id /
.discarded).
"""

from __future__ import annotations

from types import SimpleNamespace

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.core.tree_store import MCTSTreeStore


def _node(
    node_id, *, parent_node_id=None, episode_id="ep", turn_idx=1, outcome_reward=0.0
):
    return SimpleNamespace(
        node_id=node_id,
        parent_node_id=parent_node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        outcome_reward=outcome_reward,
        train_id="",
        discarded=False,
    )


def _super(node_id, *, nodes, session_id=None):
    return SuperNode(
        node_id=node_id,
        agent_id="a",
        issue_id="i",
        task_id="t",
        session_id=session_id,
        nodes=nodes,
    )


def test_insert_super_batch_indexes_super_and_node_levels():
    store = MCTSTreeStore()
    s1 = _super("s1", nodes=[_node("n1"), _node("n2", parent_node_id="n1")])
    s2 = _super("s2", nodes=[_node("n3", parent_node_id="n2")])  # cross-super link
    store.insert_super_batch([s1, s2], query_id="q")
    assert store.get_super_node("s1") is s1
    assert store.get_super_node("s2") is s2
    assert store.get_node("n1") is s1.nodes[0]
    assert store.get_node("n2") is s1.nodes[1]
    assert store.get_node("n3") is s2.nodes[0]


def test_backup_episode_terminal_walks_across_super_boundary():
    # n1 -> n2 (in s1) -> n3 (in s2, parent=n2). Reward 1.0 backed up from n3.
    store = MCTSTreeStore()
    n1 = _node("n1", outcome_reward=0.0)
    n2 = _node("n2", parent_node_id="n1", outcome_reward=0.0)
    n3 = _node("n3", parent_node_id="n2", outcome_reward=0.0)
    s1 = _super("s1", nodes=[n1, n2])
    s2 = _super("s2", nodes=[n3])
    store.insert_super_batch([s1, s2], query_id="q", backup=False)
    store.backup_episode_terminal("n3", 1.0)
    # All three nodes get +1.0; n3 is terminal, n2 and n1 are ancestors via
    # the cross-super parent_node_id chain.
    assert store.get_visit_count("n3") == 1
    assert store.get_q_value("n3") == 1.0
    assert store.get_visit_count("n2") == 1
    assert store.get_q_value("n2") == 1.0
    assert store.get_visit_count("n1") == 1
    assert store.get_q_value("n1") == 1.0


def test_single_agent_leaf_super_preserves_backup_behavior():
    """Single-agent path wraps Nodes in a leaf SuperNode; backup still walks
    the parent_node_id chain inside that one SuperNode."""
    store = MCTSTreeStore()
    # Simulate what the workflow does: build a leaf SuperNode wrapping the
    # episode's Nodes, then insert_super_batch with backup=True.
    n1 = _node("n1", episode_id="ep", turn_idx=1, outcome_reward=1.0)
    n2 = _node(
        "n2", episode_id="ep", turn_idx=2, outcome_reward=1.0, parent_node_id="n1"
    )
    leaf = _super("s-ep", nodes=[n1, n2])
    store.insert_super_batch([leaf], query_id="q", backup=True)
    # n2 is the episode terminal (highest turn_idx); backup walks n2 -> n1.
    assert store.get_visit_count("n2") == 1
    assert store.get_q_value("n2") == 1.0
    assert store.get_visit_count("n1") == 1
    assert store.get_q_value("n1") == 1.0

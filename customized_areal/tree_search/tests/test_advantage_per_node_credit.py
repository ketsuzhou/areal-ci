"""Tests for per-node credit consumption in TreeAdvantageComputer (Phase 3).

When nodes carry `credit` (from DAG reward backup), GRPO normalization
operates per-node across the query group — not flat-broadcast per episode.
"""

from __future__ import annotations

from customized_areal.tree_search.core.advantage import TreeAdvantageComputer
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _make_node(
    node_id: str,
    episode_id: str,
    query_id: str = "q1",
    credit: float | None = None,
    outcome_reward: float = 0.0,
) -> Node:
    return Node(
        input_ids=[1, 2, 3, 4],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.5, -0.5],
        versions=[-1, -1, 0, 0],
        node_id=node_id,
        episode_id=episode_id,
        query_id=query_id,
        outcome_reward=outcome_reward,
        credit=credit,
    )


def test_advantage_consumes_per_node_credit():
    """When nodes carry `credit`, GRPO normalization operates per-node."""
    store = MCTSTreeStore()
    computer = TreeAdvantageComputer(store)
    nodes = [
        _make_node("n1", "e1", credit=1.0),
        _make_node("n2", "e2", credit=0.5),
        _make_node("n3", "e3", credit=0.0),
    ]
    computer.compute(nodes)
    # Sum the response-position advantages (mask=[0,0,1,1], so sum = 2 * norm_return).
    adv = {n.node_id: float(n.advantages.sum()) for n in nodes}
    assert adv["n1"] > adv["n2"] > adv["n3"]


def test_advantage_flat_broadcast_when_no_credit():
    """Backward-compat: without credit, episodes with the same reward get the same advantage."""
    store = MCTSTreeStore()
    computer = TreeAdvantageComputer(store)
    nodes = [
        _make_node("n1", "e1", outcome_reward=1.0),
        _make_node("n2", "e2", outcome_reward=1.0),
    ]
    computer.compute(nodes)
    adv = {n.node_id: float(n.advantages.sum()) for n in nodes}
    assert adv["n1"] == adv["n2"]

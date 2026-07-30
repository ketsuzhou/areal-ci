"""Node-level normalization for MultiCA diagnosis scores."""

import pytest

from customized_areal.tree_search.agents.supernode_assembler import (
    normalize_diagnosis_node_rewards,
)
from customized_areal.tree_search.core.tree_store import Node


def _node(node_id: str) -> Node:
    return Node(
        input_ids=[1],
        loss_mask=[1],
        logprobs=[0.0],
        versions=[1],
        node_id=node_id,
    )


def test_normalize_diagnosis_node_rewards_averages_episode_scores_globally():
    shared = _node("shared")
    leaf = _node("leaf")
    shared.episode_scores = {"episode-a": 8.0, "episode-b": 4.0}
    leaf.episode_scores = {"episode-a": 2.0}

    normalize_diagnosis_node_rewards([shared, leaf])

    # Node means are 6 and 2; one global normalization yields 0.75 and 0.25.
    assert shared.process_reward == pytest.approx(0.75)
    assert leaf.process_reward == pytest.approx(0.25)
    assert shared.process_reward + leaf.process_reward == pytest.approx(1.0)


def test_normalize_diagnosis_node_rewards_uses_uniform_zero_total_fallback():
    first = _node("first")
    second = _node("second")
    first.episode_scores = {"episode-a": 0.0}
    second.episode_scores = {"episode-a": 0.0, "episode-b": 0.0}

    normalize_diagnosis_node_rewards([first, second])

    assert first.process_reward == pytest.approx(0.5)
    assert second.process_reward == pytest.approx(0.5)

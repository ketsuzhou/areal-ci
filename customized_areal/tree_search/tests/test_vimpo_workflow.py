import pytest
import torch

from customized_areal.tree_search.core.customized_grouped_workflow import (
    annotate_vimpo_episode_metadata,
)
from customized_areal.tree_search.core.tree_store import Node, _node_to_tensor_dict


def _node(query: str, episode: str, turn: int, reward: float) -> Node:
    return Node(
        input_ids=[10, 11, 12],
        logprobs=[0.0, -0.2, -0.3],
        loss_mask=[0, 1, 1],
        versions=[0, 0, 0],
        outcome_reward=reward,
        node_id=f"{episode}-{turn}",
        query_id=query,
        episode_id=episode,
        turn_idx=turn,
    )


def test_centered_rewards_count_distinct_episodes_not_turns() -> None:
    nodes = [_node("q", "a", 1, 1.0), _node("q", "a", 2, 1.0), _node("q", "b", 1, 0.0)]
    annotate_vimpo_episode_metadata(nodes)
    assert [n.vimpo_centered_reward for n in nodes] == [0.5, 0.5, -0.5]
    assert nodes[0].vimpo_episode_index == nodes[1].vimpo_episode_index
    assert nodes[2].vimpo_episode_index != nodes[0].vimpo_episode_index


def test_single_episode_query_has_zero_target() -> None:
    nodes = [_node("q", "a", 1, 0.7)]
    annotate_vimpo_episode_metadata(nodes)
    assert nodes[0].vimpo_centered_reward == pytest.approx(0.0)


def test_tensorizer_uses_next_token_coordinates() -> None:
    node = _node("q", "a", 1, 1.0)
    annotate_vimpo_episode_metadata([node])
    data = _node_to_tensor_dict(node, "q", node.node_id, advantage_mode="vimpo")
    assert data["vimpo_predict_mask"].dtype is torch.bool
    assert data["vimpo_predict_mask"].tolist() == [[True, True, False]]
    assert data["vimpo_episode_index"].shape == data["input_ids"].shape


def test_metadata_rejects_duplicate_turn_and_inconsistent_reward() -> None:
    with pytest.raises(ValueError, match="duplicate turn_idx"):
        annotate_vimpo_episode_metadata([_node("q", "a", 1, 1), _node("q", "a", 1, 1)])
    with pytest.raises(ValueError, match="inconsistent outcome_reward"):
        annotate_vimpo_episode_metadata([_node("q", "a", 1, 1), _node("q", "a", 2, 0)])

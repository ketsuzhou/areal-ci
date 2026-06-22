# SPDX-License-Identifier: Apache-2.0
"""Tests for the workflow judge-annotation step _annotate_judge_process_rewards.

Uses ``object.__new__`` to exercise the method without the heavy workflow
constructor, and patches the module-level ``_input_ids_to_messages`` so no real
tokenizer is needed.
"""

import pytest

import customized_areal.tree_search.core.customized_grouped_workflow as wf_mod
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _node(node_id, episode_id, turn_idx):
    return Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, 0.0],
        versions=[-1, -1, -1],
        node_id=node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
    )


class _FakeProvider:
    def __init__(self, scores_by_call, raise_exc=False):
        self.scores_by_call = scores_by_call
        self.raise_exc = raise_exc
        self.calls = 0

    async def score_episode(
        self, conversation, gold_answer, *, score_max, model_name=None
    ):
        self.calls += 1
        if self.raise_exc:
            raise RuntimeError("judge down")
        return self.scores_by_call


def _make_workflow(store, *, beta=0.5, score_max=10, concurrency=4):
    w = object.__new__(TreeSearchGroupedRolloutWorkflow)
    w.tree_store = store
    w.judge_process_reward_beta = beta
    w.critic_score_max = score_max
    w.judge_model_name = ""
    w.judge_max_concurrency = concurrency
    w._judged_episodes = set()
    return w


@pytest.fixture(autouse=True)
def _patch_messages(monkeypatch):
    monkeypatch.setattr(
        wf_mod,
        "_input_ids_to_messages",
        lambda input_ids, tokenizer: [{"role": "user", "content": "x"}],
    )


@pytest.mark.asyncio
async def test_annotation_populates_judge_scores():
    store = MCTSTreeStore()
    w = _make_workflow(store)
    provider = _FakeProvider({1: 7, 2: 3})
    nodes = [_node("n1", "ep", 1), _node("n2", "ep", 2)]
    await w._annotate_judge_process_rewards(
        provider, nodes, {"answer": "gold"}, tokenizer=None
    )
    assert store.get_judge_scores("n1") == [7.0]
    assert store.get_judge_scores("n2") == [3.0]
    assert "ep" in w._judged_episodes


@pytest.mark.asyncio
async def test_provider_error_leaves_store_empty():
    store = MCTSTreeStore()
    w = _make_workflow(store)
    provider = _FakeProvider({}, raise_exc=True)
    nodes = [_node("n1", "ep", 1), _node("n2", "ep", 2)]
    await w._annotate_judge_process_rewards(
        provider, nodes, {"answer": "gold"}, tokenizer=None
    )
    assert store.get_mean_judge_score("n1") is None
    assert store.get_mean_judge_score("n2") is None
    assert "ep" not in w._judged_episodes


@pytest.mark.asyncio
async def test_cache_avoids_rejudging_same_episode():
    store = MCTSTreeStore()
    w = _make_workflow(store)
    provider = _FakeProvider({1: 5, 2: 5})
    nodes = [_node("n1", "ep", 1), _node("n2", "ep", 2)]
    await w._annotate_judge_process_rewards(provider, nodes, {"answer": "g"}, None)
    await w._annotate_judge_process_rewards(provider, nodes, {"answer": "g"}, None)
    assert provider.calls == 1  # second call skipped via cache
    assert store.get_judge_scores("n1") == [5.0]


@pytest.mark.asyncio
async def test_shared_node_accumulates_across_episodes():
    # Two episodes share prefix node "shared"; each contributes one score.
    store = MCTSTreeStore()
    w = _make_workflow(store)
    ep_a = [_node("shared", "epA", 1), _node("a_leaf", "epA", 2)]
    ep_b = [_node("shared", "epB", 1), _node("b_leaf", "epB", 2)]
    await w._annotate_judge_process_rewards(
        _FakeProvider({1: 8, 2: 2}), ep_a, {"answer": "g"}, None
    )
    await w._annotate_judge_process_rewards(
        _FakeProvider({1: 4, 2: 6}), ep_b, {"answer": "g"}, None
    )
    assert store.get_judge_scores("shared") == [8.0, 4.0]
    assert store.get_mean_judge_score("shared") == 6.0

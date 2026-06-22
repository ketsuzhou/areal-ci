# SPDX-License-Identifier: Apache-2.0
"""Integration test: LLM-judge process rewards reach both actor and critic.

Exercises the wired path end-to-end at the reward-consumer level:
annotation -> tree_store judge scores -> GAE advantages (actor) AND
compute_critic_targets (critic). Also asserts the flag-off path is byte-for-byte
the sparse baseline.
"""

import pytest

import customized_areal.tree_search.core.customized_grouped_workflow as wf_mod
from customized_areal.tree_search.core.advantage import GAEAdvantageComputer
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.training.losses.critic import compute_critic_targets


def _node(node_id, turn_idx, value, outcome):
    n = Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, 0.0],
        versions=[-1, -1, -1],
        node_id=node_id,
        episode_id="ep",
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=outcome,
    )
    n.value = value
    return n


class _FakeProvider:
    async def score_episode(
        self, conversation, gold_answer, *, score_max, model_name=None
    ):
        return {1: 4, 2: 4}  # equal -> jbar = [0.5, 0.5]


def _make_workflow(store, *, enabled, beta=0.5, score_max=10):
    w = object.__new__(TreeSearchGroupedRolloutWorkflow)
    w.tree_store = store
    w.enable_judge_process_reward = enabled
    w.judge_process_reward_beta = beta
    w.critic_score_max = score_max
    w.judge_model_name = ""
    w.judge_max_concurrency = 4
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
async def test_judge_rewards_reach_actor_and_critic():
    store = MCTSTreeStore()
    w = _make_workflow(store, enabled=True, beta=0.5)
    nodes = [
        _node("n0", 1, value=0.0, outcome=1.0),
        _node("n1", 2, value=0.0, outcome=1.0),
    ]

    await w._annotate_judge_process_rewards(
        _FakeProvider(), nodes, {"answer": "gold"}, tokenizer=None
    )
    # Both nodes scored.
    assert store.get_mean_judge_score("n0") == 4.0
    assert store.get_mean_judge_score("n1") == 4.0

    # Actor: GAE wired with judge_beta (as the workflow constructor does).
    gae = GAEAdvantageComputer(
        store,
        gamma=1.0,
        lam=1.0,
        judge_beta=w.judge_process_reward_beta
        if w.enable_judge_process_reward
        else 0.0,
        judge_score_max=w.critic_score_max,
    )
    gae.compute(nodes)
    # dense rewards = [0.25, 0.75] -> A1=0.75, A0=1.0
    assert abs(float(nodes[0].advantages[-1]) - 1.0) < 1e-6
    assert abs(float(nodes[1].advantages[-1]) - 0.75) < 1e-6

    # Critic: dense MC return from process rewards.
    targets = compute_critic_targets(
        nodes,
        tree_store=store,
        mc_weight=0.0,
        n_steps=10,
        gamma=1.0,
        target_scale=1.0,
        judge_beta=w.judge_process_reward_beta
        if w.enable_judge_process_reward
        else 0.0,
        judge_score_max=w.critic_score_max,
    )
    assert targets == pytest.approx([1.0, 0.75])


@pytest.mark.asyncio
async def test_flag_off_reproduces_sparse_baseline():
    store = MCTSTreeStore()
    w = _make_workflow(store, enabled=False, beta=0.5)
    nodes = [
        _node("n0", 1, value=0.2, outcome=1.0),
        _node("n1", 2, value=0.8, outcome=1.0),
    ]

    # No annotation runs when the flag is off (caller gates on it); store empty.
    assert store.get_mean_judge_score("n0") is None

    judge_beta = w.judge_process_reward_beta if w.enable_judge_process_reward else 0.0
    gae = GAEAdvantageComputer(store, gamma=1.0, lam=0.95, judge_beta=judge_beta)
    gae.compute(nodes)

    # Baseline sparse GAE (no judge): identical computer with judge_beta=0.
    store_b = MCTSTreeStore()
    nodes_b = [
        _node("n0", 1, value=0.2, outcome=1.0),
        _node("n1", 2, value=0.8, outcome=1.0),
    ]
    gae_b = GAEAdvantageComputer(store_b, gamma=1.0, lam=0.95)
    gae_b.compute(nodes_b)

    assert float(nodes[0].advantages[-1]) == pytest.approx(
        float(nodes_b[0].advantages[-1])
    )
    assert float(nodes[1].advantages[-1]) == pytest.approx(
        float(nodes_b[1].advantages[-1])
    )

    targets = compute_critic_targets(
        nodes,
        tree_store=store,
        mc_weight=0.0,
        n_steps=1,
        gamma=1.0,
        judge_beta=judge_beta,
    )
    targets_b = compute_critic_targets(
        nodes_b, tree_store=store_b, mc_weight=0.0, n_steps=1, gamma=1.0
    )
    assert targets == pytest.approx(targets_b)

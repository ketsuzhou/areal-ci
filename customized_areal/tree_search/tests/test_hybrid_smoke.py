# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test of the HYBRID_GAE pipeline (Task 5, no GPU/engine).

Exercises: branch-point MCTS aggregation -> critic value+variance annotation ->
HybridGAEAdvantageComputer over an eligible branched node, asserting per-turn
advantages/returns are produced and the blend shifts the branched turn away
from the pure-critic GAE result.
"""

import asyncio
import uuid

import torch

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.core.advantage import (
    GAEAdvantageComputer,
    HybridGAEAdvantageComputer,
)
from customized_areal.tree_search.core.critic_value_client import CriticValueClient
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _wrap_leaf(nodes):
    return SuperNode(
        node_id=str(uuid.uuid4()),
        agent_id="",
        issue_id="",
        task_id=nodes[0].query_id if nodes else "",
        nodes=list(nodes),
    )


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(t) for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [10 + i for i in range(len(messages))]
        if add_generation_prompt:
            ids.append(999)
        return ids


def _node(node_id, episode_id, turn_idx, reward, parent=None, need_branch=False):
    return Node(
        input_ids=[20, 21],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[0, 0],
        node_id=node_id,
        parent_node_id=parent,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q1",
        outcome_reward=reward,
        need_branch=need_branch,
    )


def test_hybrid_end_to_end():
    store = MCTSTreeStore()

    # Parent episode: p0 (branch point, need_branch) -> p1 terminal, reward 1.0.
    parent_nodes = [
        _node("p0", "epP", 1, 1.0, parent=None, need_branch=True),
        _node("p1", "epP", 2, 1.0, parent="p0"),
    ]
    store.insert_super_batch([_wrap_leaf(parent_nodes)], query_id="q1")
    # Five branch episodes through p0 with reward 0.0 -> p0 visit_count = 6 >= 5.
    for k in range(5):
        branch_nodes = [
            _node(f"b{k}_0", f"epB{k}", 2, 0.0, parent="p0"),
            _node(f"b{k}_1", f"epB{k}", 3, 0.0, parent=f"b{k}_0"),
        ]
        store.insert_super_batch([_wrap_leaf(branch_nodes)], query_id="q1")
    assert store.get_visit_count("p0") == 6

    # Critic annotation (soft): equal mass on labels 4 and 6 -> value 0.5,
    # non-zero categorical variance for every node.
    async def soft_fn(engine, prompt_ids):
        return {4: 0.0, 6: 0.0}

    client = CriticValueClient(_FakeTokenizer(), score_max=10, logprob_query_fn=soft_fn)
    _run(client.annotate_episode(None, parent_nodes, store))
    assert store.get_value("p0") == 0.5
    assert store.get_value_variance("p0") > 0.0

    # Hybrid vs plain GAE on the parent episode.
    hybrid = HybridGAEAdvantageComputer(
        store, gamma=1.0, lam=1.0, mc_min_visits=5, critic_var_floor=1e-3
    )
    hybrid.compute(parent_nodes)
    for n in parent_nodes:
        assert n.advantages is not None and n.returns is not None
        assert torch.isfinite(n.advantages).all()

    # p0 is eligible (need_branch + visits>=5); its LOO MC value (mean over the
    # five 0.0 branch returns excluding p0's own 1.0) pulls its blended value
    # below the pure-critic 0.5, so the hybrid advantage at p0 differs from GAE.
    gae_store = MCTSTreeStore()
    gae = GAEAdvantageComputer(gae_store, gamma=1.0, lam=1.0)
    gae_nodes = [
        _node("p0", "epP", 1, 1.0, need_branch=True),
        _node("p1", "epP", 2, 1.0, parent="p0"),
    ]
    for n in gae_nodes:
        n.value = 0.5
    gae.compute(gae_nodes)
    assert not torch.allclose(parent_nodes[0].advantages, gae_nodes[0].advantages)

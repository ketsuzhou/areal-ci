# SPDX-License-Identifier: Apache-2.0
"""Tests for CriticValueClient and its integration with GAEAdvantageComputer."""

import asyncio
from dataclasses import dataclass, field

from customized_areal.tree_search.core.advantage import GAEAdvantageComputer
from customized_areal.tree_search.core.critic_value_client import CriticValueClient
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        toks = [t for t in token_ids if not (skip_special_tokens and t < 0)]
        return " ".join(str(t) for t in toks)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        # Deterministic: one token per message plus a generation marker.
        ids = [hash(m["role"]) % 100 for m in messages]
        if add_generation_prompt:
            ids.append(999)
        return ids


@dataclass
class _FakeResponse:
    output_tokens: list = field(default_factory=list)


class _FakeEngine:
    """Async engine returning a fixed integer answer in output_tokens."""

    def __init__(self, answer_token):
        self.answer_token = answer_token
        self.calls = 0

    async def agenerate(self, req):
        self.calls += 1
        return _FakeResponse(output_tokens=[self.answer_token])


def _node(node_id, episode_id, turn_idx, loss_mask, reward):
    return Node(
        input_ids=[20, 21, 30, 31][: len(loss_mask)]
        if len(loss_mask) <= 4
        else [0] * len(loss_mask),
        loss_mask=loss_mask,
        logprobs=[0.0] * len(loss_mask),
        versions=[0] * len(loss_mask),
        node_id=node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q",
        outcome_reward=reward,
    )


class TestDefaultQueryParse:
    def test_parses_integer_one_hot_value(self):
        client = CriticValueClient(_FakeTokenizer(), score_max=10)
        engine = _FakeEngine(answer_token=7)
        node = _node("n0", "ep", 1, [0, 0, 1, 1], 1.0)
        v = _run(client.compute_value(engine, node))
        # one-hot at label 7 -> 7/10
        assert v == 0.7
        assert engine.calls == 1

    def test_out_of_range_returns_zero(self):
        client = CriticValueClient(_FakeTokenizer(), score_max=5)
        engine = _FakeEngine(answer_token=99)  # decodes to "99", out of [0,5]
        node = _node("n0", "ep", 1, [0, 1], 1.0)
        v = _run(client.compute_value(engine, node))
        assert v == 0.0


class TestInjectedSoftQuery:
    def test_soft_value_from_logprobs(self):
        async def soft_fn(engine, prompt_ids):
            # equal mass on 0 and 10 -> expected 0.5
            return {0: 0.0, 10: 0.0}

        client = CriticValueClient(
            _FakeTokenizer(), score_max=10, logprob_query_fn=soft_fn
        )
        node = _node("n0", "ep", 1, [0, 1], 1.0)
        v = _run(client.compute_value(None, node))
        assert v == 0.5


class TestAnnotateEpisode:
    def test_writes_node_value_and_store(self):
        client = CriticValueClient(_FakeTokenizer(), score_max=10)
        engine = _FakeEngine(answer_token=3)
        store = MCTSTreeStore()
        nodes = [
            _node("n0", "ep", 1, [0, 1], 0.0),
            _node("n1", "ep", 2, [0, 1], 1.0),
        ]
        _run(client.annotate_episode(engine, nodes, store))
        for n in nodes:
            assert n.value == 0.3
            assert store.get_value(n.node_id) == 0.3


class TestIntegrationWithGAE:
    def test_values_drive_gae(self):
        # Inject a soft query that returns a value depending on turn so we can
        # check GAE consumes Node.value end-to-end.
        async def soft_fn(engine, prompt_ids):
            return {5: 0.0}  # value 0.5 for every node

        client = CriticValueClient(
            _FakeTokenizer(), score_max=10, logprob_query_fn=soft_fn
        )
        store = MCTSTreeStore()
        nodes = [
            _node("n0", "ep", 1, [0, 1], 1.0),
            _node("n1", "ep", 2, [1], 1.0),
        ]
        _run(client.annotate_episode(None, nodes, store))
        assert all(n.value == 0.5 for n in nodes)

        gae = GAEAdvantageComputer(store, gamma=1.0, lam=0.95)
        gae.compute(nodes)
        # terminal (turn 2): delta = 1.0 - 0.5 = 0.5 -> A=0.5
        # turn1: delta = 0 + 1*0.5 - 0.5 = 0.0; A = 0.0 + 0.95*0.5 = 0.475
        import torch

        torch.testing.assert_close(
            nodes[1].advantages, torch.tensor([0.5]), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            nodes[0].advantages,
            torch.tensor([0.0, 0.475]),
            rtol=1e-5,
            atol=1e-6,
        )

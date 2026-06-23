# SPDX-License-Identifier: Apache-2.0
"""Tests for critic categorical variance (Task 2)."""

import asyncio

from customized_areal.tree_search.core.critic_prompt import (
    expected_value_from_logprobs,
    variance_from_logprobs,
)
from customized_areal.tree_search.core.critic_value_client import CriticValueClient
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(t) for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [hash(m["role"]) % 100 for m in messages]
        if add_generation_prompt:
            ids.append(999)
        return ids


def _node(node_id, reward):
    return Node(
        input_ids=[20, 21],
        loss_mask=[0, 1],
        logprobs=[0.0, 0.0],
        versions=[0, 0],
        node_id=node_id,
        episode_id="ep",
        turn_idx=1,
        query_id="q",
        outcome_reward=reward,
    )


class TestVarianceFromLogprobs:
    def test_one_hot_zero_variance(self):
        assert variance_from_logprobs({7: 0.0}, score_max=10) == 0.0

    def test_symmetric_two_point(self):
        # Equal mass on 0 and 10 (normalized 0 and 1): mean 0.5,
        # variance = 0.5*(0.5)^2 + 0.5*(0.5)^2 = 0.25.
        v = expected_value_from_logprobs({0: 0.0, 10: 0.0}, score_max=10)
        var = variance_from_logprobs({0: 0.0, 10: 0.0}, score_max=10)
        assert abs(v - 0.5) < 1e-9
        assert abs(var - 0.25) < 1e-9

    def test_empty_returns_zero(self):
        assert variance_from_logprobs({}, score_max=10) == 0.0


class TestClientVariance:
    def test_compute_value_and_variance_soft(self):
        async def soft_fn(engine, prompt_ids):
            return {0: 0.0, 10: 0.0}

        client = CriticValueClient(
            _FakeTokenizer(), score_max=10, logprob_query_fn=soft_fn
        )
        value, variance = _run(
            client.compute_value_and_variance(None, _node("n0", 1.0))
        )
        assert abs(value - 0.5) < 1e-9
        assert abs(variance - 0.25) < 1e-9

    def test_annotate_episode_stores_variance(self):
        async def soft_fn(engine, prompt_ids):
            return {0: 0.0, 10: 0.0}

        client = CriticValueClient(
            _FakeTokenizer(), score_max=10, logprob_query_fn=soft_fn
        )
        store = MCTSTreeStore()
        nodes = [_node("n0", 0.0), _node("n1", 1.0)]
        _run(client.annotate_episode(None, nodes, store))
        for n in nodes:
            assert abs(n.value - 0.5) < 1e-9
            assert abs(n.value_variance - 0.25) < 1e-9
            assert abs(store.get_value_variance(n.node_id) - 0.25) < 1e-9

    def test_one_hot_fallback_zero_variance(self):
        # No soft fn -> default parse path -> one-hot -> variance 0.0.
        client = CriticValueClient(_FakeTokenizer(), score_max=10)

        class _Eng:
            async def agenerate(self, req):
                class R:
                    output_tokens = [4]

                return R()

        value, variance = _run(
            client.compute_value_and_variance(_Eng(), _node("n0", 1.0))
        )
        assert value == 0.4
        assert variance == 0.0


class TestStoreVarianceAccessors:
    def test_set_get_default(self):
        store = MCTSTreeStore()
        assert store.get_value_variance("a", default=-1.0) == -1.0
        store.set_value_variance("a", 0.2)
        assert store.get_value_variance("a") == 0.2

    def test_clear_resets(self):
        store = MCTSTreeStore()
        store.set_value_variance("a", 0.2)
        store.clear()
        assert store.get_value_variance("a") == 0.0

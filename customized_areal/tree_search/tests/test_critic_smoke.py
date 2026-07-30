# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test of the generative-critic pipeline (no GPU/engine).

Exercises the full logical flow: critic value computation -> Node.value -> GAE
advantages -> critic training batch -> critic minibatch -> soft-regression loss,
asserting advantages/returns are set and the critic loss + stats are produced.
"""

import asyncio
import os

import torch
import yaml

from customized_areal.tree_search.config import AdvantageMode, Config
from customized_areal.tree_search.core.advantage import GAEAdvantageComputer
from customized_areal.tree_search.core.critic_value_client import CriticValueClient
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.training.critic_update import (
    build_critic_minibatch,
    critic_multicandidate_loss_fn,
)
from customized_areal.tree_search.training.losses.critic import (
    build_critic_training_batch,
)

CONFIG_PATH = (
    "customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B_generative_critic.yaml"
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        toks = [t for t in token_ids if not (skip_special_tokens and t < 0)]
        return " ".join(str(t) for t in toks)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [10 + i for i in range(len(messages))]
        if add_generation_prompt:
            ids.append(999)
        return ids


def _node(node_id, episode_id, turn_idx, loss_mask, reward):
    return Node(
        input_ids=[20, 21, 30, 31][: len(loss_mask)],
        loss_mask=loss_mask,
        logprobs=[0.0] * len(loss_mask),
        versions=[0] * len(loss_mask),
        node_id=node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        query_id="q1",
        outcome_reward=reward,
    )


def test_example_config_parses_with_critic_enabled():
    if not os.path.isfile(CONFIG_PATH):
        import pytest

        pytest.skip(f"config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    ts = raw["tree_search"]
    assert ts["enable_generative_critic"] is True
    assert ts["advantage_mode"] == "gae"
    # The critic block must round-trip through Config validation.
    cfg = Config(
        enable_generative_critic=ts["enable_generative_critic"],
        advantage_mode=ts["advantage_mode"],
        critic_avg_success_rate=ts["critic_avg_success_rate"],
        critic_gamma=ts["critic_gamma"],
        critic_lambda=ts["critic_lambda"],
        critic_score_max=ts["critic_score_max"],
        critic_target_scale=ts["critic_target_scale"],
        critic_loss_weight=ts["critic_loss_weight"],
    )
    assert cfg.advantage_mode == AdvantageMode.GAE


def test_end_to_end_pipeline():
    # Soft critic: equal mass on labels 4 and 6 -> value 0.5 for every node.
    async def soft_fn(engine, prompt_ids):
        return {4: 0.0, 6: 0.0}

    tokenizer = _FakeTokenizer()
    client = CriticValueClient(
        tokenizer, score_max=10, avg_success_rate=0.29, logprob_query_fn=soft_fn
    )
    store = MCTSTreeStore()
    nodes = [
        _node("n0", "ep", 1, [0, 1], 1.0),
        _node("n1", "ep", 2, [0, 1], 1.0),
        _node("n2", "ep", 3, [1], 1.0),
    ]
    # inject q_values as regression targets
    for nid, q in [("n0", 0.3), ("n1", 0.6), ("n2", 0.9)]:
        store._q_values[nid] = q

    # 1. critic values onto nodes
    _run(client.annotate_episode(None, nodes, store))
    assert all(n.value == 0.5 for n in nodes)

    # 2. GAE advantages from critic values
    gae = GAEAdvantageComputer(store, gamma=1.0, lam=0.95)
    gae.compute(nodes)
    assert all(n.advantages is not None and n.returns is not None for n in nodes)

    # 3. critic training batch + minibatch + loss
    batch = build_critic_training_batch(
        client, nodes, tree_store=store, target_scale=1.0
    )
    assert torch.allclose(batch["targets"], torch.tensor([0.3, 0.6, 0.9]))

    mb = build_critic_minibatch(batch, pad_token_id=0)
    assert mb["loss_mask"].sum().item() == 3  # one answer position per node

    # Simulate the engine gathering candidate logprobs at the answer positions:
    # put all mass on label 3 -> value 0.3 for each sample.
    logprobs = torch.full((3, 11), -1e9)
    logprobs[:, 3] = 0.0
    loss = critic_multicandidate_loss_fn(logprobs, None, mb, critic_loss_weight=1.0)
    # values 0.3 vs targets [0.3,0.6,0.9] -> mse = (0+0.09+0.36)/3 = 0.15
    assert abs(loss.item() - 0.15) < 1e-4

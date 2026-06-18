# SPDX-License-Identifier: Apache-2.0
"""Tests for critic prompt construction and value-extraction utilities."""

import math
from dataclasses import dataclass

import pytest

from customized_areal.tree_search.core.critic_prompt import (
    build_critic_instruction,
    build_critic_messages,
    digit_token_ids,
    expected_value_from_logprobs,
    reconstruct_messages_through_turn,
)


@dataclass
class _FakeNode:
    input_ids: list
    loss_mask: list


class _FakeTokenizer:
    """A trivial tokenizer: each int id maps to a word token; special tokens
    (negative ids) are dropped when skip_special_tokens=True.

    Vocabulary for decode: positive ids -> f"w{id}". For encode of digit labels
    we map each character to its ord, so "10" -> two tokens.
    """

    SPECIAL = {-1, -2}

    def decode(self, token_ids, skip_special_tokens=False):
        words = []
        for t in token_ids:
            if skip_special_tokens and t in self.SPECIAL:
                continue
            words.append(self._word(t))
        return " ".join(words)

    def _word(self, t):
        mapping = {
            10: "system",
            11: "You",
            12: "are",
            20: "user",
            21: "question",
            30: "assistant",
            31: "answer1",
            40: "followup",
            50: "answer2",
        }
        return mapping.get(t, f"w{t}")

    def encode(self, text, add_special_tokens=False):
        # Each character becomes a token id = ord(char). "10" -> [49, 48].
        return [ord(c) for c in text]


class TestCriticInstruction:
    def test_contains_success_rate_and_scale(self):
        text = build_critic_instruction(avg_success_rate=0.29, score_max=10)
        assert "0.29" in text
        assert "between 0 and 10" in text

    def test_custom_values(self):
        text = build_critic_instruction(avg_success_rate=0.5, score_max=9)
        assert "0.50" in text
        assert "between 0 and 9" in text


class TestReconstructMessages:
    def test_single_turn(self):
        # context tokens [-1, 20, 21] (special dropped), response [30, 31]
        node = _FakeNode(input_ids=[-1, 20, 21, 30, 31], loss_mask=[0, 0, 0, 1, 1])
        msgs = reconstruct_messages_through_turn(node, _FakeTokenizer())
        assert msgs == [
            {"role": "user", "content": "user question"},
            {"role": "assistant", "content": "assistant answer1"},
        ]

    def test_multi_turn_through_turn_t(self):
        # user / assistant / user / assistant
        node = _FakeNode(
            input_ids=[20, 21, 30, 31, 40, 50],
            loss_mask=[0, 0, 1, 1, 0, 1],
        )
        msgs = reconstruct_messages_through_turn(node, _FakeTokenizer())
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert msgs[2]["content"] == "followup"

    def test_system_prompt_prepended(self):
        node = _FakeNode(input_ids=[20, 21, 30, 31], loss_mask=[0, 0, 1, 1])
        msgs = reconstruct_messages_through_turn(
            node, _FakeTokenizer(), system_prompt="SYS"
        )
        assert msgs[0] == {"role": "system", "content": "SYS"}

    def test_length_mismatch_raises(self):
        node = _FakeNode(input_ids=[1, 2, 3], loss_mask=[0, 1])
        with pytest.raises(ValueError, match="equal length"):
            reconstruct_messages_through_turn(node, _FakeTokenizer())


class TestBuildCriticMessages:
    def test_appends_instruction(self):
        base = [{"role": "user", "content": "q"}]
        out = build_critic_messages(base, "INSTR")
        assert out[-1] == {"role": "user", "content": "INSTR"}
        assert out[:-1] == base
        # original not mutated
        assert len(base) == 1


class TestDigitTokenIds:
    def test_resolves_all_labels(self):
        labels = digit_token_ids(_FakeTokenizer(), score_max=10)
        assert set(labels.keys()) == set(range(11))
        # "10" tokenizes to two tokens with the fake tokenizer.
        assert labels[10] == [ord("1"), ord("0")]
        assert labels[5] == [ord("5")]


class TestExpectedValue:
    def test_all_mass_on_max(self):
        lp = {i: (-1e9 if i != 10 else 0.0) for i in range(11)}
        v = expected_value_from_logprobs(lp, score_max=10)
        assert v == pytest.approx(1.0)

    def test_all_mass_on_zero(self):
        lp = {i: (-1e9 if i != 0 else 0.0) for i in range(11)}
        v = expected_value_from_logprobs(lp, score_max=10)
        assert v == pytest.approx(0.0)

    def test_uniform_two_labels(self):
        # equal logprob on 0 and 10 -> expected normalized value 0.5
        lp = {0: 0.0, 10: 0.0}
        v = expected_value_from_logprobs(lp, score_max=10)
        assert v == pytest.approx(0.5)

    def test_matches_manual_softmax(self):
        lp = {0: math.log(0.2), 5: math.log(0.3), 10: math.log(0.5)}
        v = expected_value_from_logprobs(lp, score_max=10)
        # already-normalized probs -> 0.2*0 + 0.3*0.5 + 0.5*1.0 = 0.65
        assert v == pytest.approx(0.65)

    def test_empty_returns_zero(self):
        assert expected_value_from_logprobs({}, score_max=10) == 0.0

    def test_bad_score_max(self):
        with pytest.raises(ValueError, match="score_max"):
            expected_value_from_logprobs({0: 0.0}, score_max=0)

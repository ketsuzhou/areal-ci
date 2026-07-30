# SPDX-License-Identifier: Apache-2.0
"""Tests for the LLM-judge prompt + score parser."""

import pytest

from customized_areal.tree_search.core.judge_prompt import (
    build_judge_instruction,
    parse_turn_scores,
)


class TestBuildJudgeInstruction:
    def test_includes_gold_and_scale(self):
        text = build_judge_instruction(score_max=10, gold_answer="42")
        assert "42" in text
        assert "0 and 10" in text
        assert "<judgment>" in text
        assert "<score>" in text

    def test_bad_score_max(self):
        with pytest.raises(ValueError, match="score_max"):
            build_judge_instruction(score_max=0, gold_answer="x")


class TestParseTurnScores:
    def test_well_formed(self):
        text = (
            "```xml\n"
            "<judgment><turns>"
            "<turn><turn_idx>1</turn_idx><score>7</score></turn>"
            "<turn><turn_idx>2</turn_idx><score>3</score></turn>"
            "</turns></judgment>\n"
            "```"
        )
        assert parse_turn_scores(text, score_max=10) == {1: 7, 2: 3}

    def test_clamps_out_of_range(self):
        text = (
            "<turn><turn_idx>1</turn_idx><score>15</score></turn>"
            "<turn><turn_idx>2</turn_idx><score>-4</score></turn>"
        )
        assert parse_turn_scores(text, score_max=10) == {1: 10, 2: 0}

    def test_skips_malformed_turns(self):
        text = (
            "<turn><turn_idx>1</turn_idx><score>5</score></turn>"
            "<turn><turn_idx>2</turn_idx></turn>"  # missing score
            "<turn><score>9</score></turn>"  # missing turn_idx
            "<turn><turn_idx>0</turn_idx><score>9</score></turn>"  # idx < 1
        )
        assert parse_turn_scores(text, score_max=10) == {1: 5}

    def test_surrounding_prose_ignored(self):
        text = (
            "Here is my judgement:\n"
            "<judgment><turns>"
            "<turn><turn_idx>3</turn_idx><score>8</score></turn>"
            "</turns></judgment>\n"
            "Thanks!"
        )
        assert parse_turn_scores(text, score_max=10) == {3: 8}

    def test_duplicate_turn_last_wins(self):
        text = (
            "<turn><turn_idx>1</turn_idx><score>2</score></turn>"
            "<turn><turn_idx>1</turn_idx><score>6</score></turn>"
        )
        assert parse_turn_scores(text, score_max=10) == {1: 6}

    def test_empty_and_garbage(self):
        assert parse_turn_scores("", score_max=10) == {}
        assert parse_turn_scores("no xml here", score_max=10) == {}

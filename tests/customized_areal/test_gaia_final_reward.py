"""Tests for gaia_final_reward.compute_reward and helpers."""

from unittest.mock import patch

from customized_areal.tpfc.gaia_final_reward import (
    compute_reward,
    evaluate_final_answer,
    extract_answer,
    parse_judge_score,
)

# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_basic_answer_tag(self):
        text = "Some reasoning\n<answer>42</answer>"
        assert extract_answer(text) == "42"

    def test_multiple_answer_tags_returns_last(self):
        text = "<answer>first</answer> more text <answer>second</answer>"
        assert extract_answer(text) == "second"

    def test_no_answer_tag(self):
        assert extract_answer("no tags here") is None

    def test_nested_content_stripped(self):
        text = "<answer>  hello world  </answer>"
        assert extract_answer(text) == "hello world"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "<answer>list answer</answer>"},
        ]
        assert extract_answer(content) == "list answer"

    def test_dict_content(self):
        content = {"text": "<answer>dict answer</answer>"}
        assert extract_answer(content) == "dict answer"


# ---------------------------------------------------------------------------
# parse_judge_score
# ---------------------------------------------------------------------------


class TestParseJudgeScore:
    def test_score_1(self):
        assert parse_judge_score("<score>1.0</score>") == 1.0

    def test_score_0(self):
        assert parse_judge_score("<score>0.0</score>") == 0.0

    def test_score_09_rounds_to_1(self):
        assert parse_judge_score("<score>0.9</score>") == 1.0

    def test_score_below_09(self):
        assert parse_judge_score("<score>0.8</score>") == 0.0

    def test_missing_score_tags(self):
        assert parse_judge_score("no tags") == 0.0

    def test_score_with_surrounding_text(self):
        text = "Thinking... <score>1.0</score> done"
        assert parse_judge_score(text) == 1.0

    def test_multiple_score_tags_uses_last(self):
        text = "<score>0.0</score> revised <score>1.0</score>"
        assert parse_judge_score(text) == 1.0


# ---------------------------------------------------------------------------
# evaluate_final_answer (mock the LLM call)
# ---------------------------------------------------------------------------


class TestEvaluateFinalAnswer:
    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_correct_answer(self, mock_llm):
        mock_llm.return_value = "<score>1.0</score>"
        result = evaluate_final_answer(
            model_answer="42",
            ground_truth="42",
            user_query="What is 6*7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["answer_reward"] == 1.0
        assert result["judge_prompt"] is not None
        assert result["judge_response_text"] == "<score>1.0</score>"

    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_wrong_answer(self, mock_llm):
        mock_llm.return_value = "<score>0.0</score>"
        result = evaluate_final_answer(
            model_answer="41",
            ground_truth="42",
            user_query="What is 6*7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["answer_reward"] == 0.0


# ---------------------------------------------------------------------------
# compute_reward (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------


class TestComputeReward:
    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_correct_answer_with_answer_tag(self, mock_llm):
        mock_llm.return_value = "<score>1.0</score>"
        result = compute_reward(
            response_text="Reasoning... <answer>42</answer>",
            ground_truth="42",
            user_query="What is 6*7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["raw_answer"] == "42"
        assert result["answer_reward"] == 1.0
        assert result["judge_prompt"] is not None

    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_wrong_answer_with_answer_tag(self, mock_llm):
        mock_llm.return_value = "<score>0.0</score>"
        result = compute_reward(
            response_text="<answer>41</answer>",
            ground_truth="42",
            user_query="What is 6*7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["raw_answer"] == "41"
        assert result["answer_reward"] == 0.0

    def test_no_answer_tag_returns_zero(self):
        result = compute_reward(
            response_text="No answer tag here",
            ground_truth="42",
            user_query="What is 6*7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["raw_answer"] is None
        assert result["answer_reward"] == 0
        assert result["judge_prompt"] is None

    def test_no_ground_truth_returns_zero(self):
        result = compute_reward(
            response_text="<answer>42</answer>",
            ground_truth=None,
            user_query="What is 6*7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["answer_reward"] == 0

    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_explicit_base_url_not_overridden_by_env(self, mock_llm):
        mock_llm.return_value = "<score>1.0</score>"
        with patch.dict(
            "os.environ",
            {
                "WORKSPACE_OPENAI_API_BASE": "http://env-url",
                "WORKSPACE_OPENAI_API_KEY": "env-key",
            },
        ):
            compute_reward(
                response_text="<answer>42</answer>",
                ground_truth="42",
                user_query="q",
                model_name="test-model",
                base_url="http://explicit-url",
                api_key="explicit-key",
            )
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert (
            call_kwargs.kwargs.get("base_url") == "http://explicit-url"
            or call_kwargs[1].get("base_url") == "http://explicit-url"
        )

    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_env_fallback_when_no_explicit_params(self, mock_llm):
        mock_llm.return_value = "<score>1.0</score>"
        with patch.dict(
            "os.environ",
            {
                "WORKSPACE_OPENAI_API_BASE": "http://env-url",
                "WORKSPACE_OPENAI_API_KEY": "env-key",
            },
        ):
            compute_reward(
                response_text="<answer>42</answer>",
                ground_truth="42",
                user_query="q",
                model_name="test-model",
                base_url=None,
                api_key=None,
            )
        call_kwargs = mock_llm.call_args
        assert (
            call_kwargs.kwargs.get("base_url") == "http://env-url"
            or call_kwargs[1].get("base_url") == "http://env-url"
        )

    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_realistic_llm_response_with_thinking(self, mock_llm):
        mock_llm.return_value = (
            "The model answer is 42 and the ground truth is also 42. "
            "The answer is factually correct and complete.\n"
            "<score>1.0</score>"
        )
        result = compute_reward(
            response_text="Let me think... The answer is 42.\n<answer>42</answer>",
            ground_truth="42",
            user_query="What is 6 times 7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["raw_answer"] == "42"
        assert result["answer_reward"] == 1.0

    @patch("customized_areal.tpfc.gaia_final_reward.call_openai_compatible_model")
    def test_realistic_llm_response_incorrect(self, mock_llm):
        mock_llm.return_value = (
            "The model answered 41 but the ground truth is 42. "
            "This is incorrect.\n"
            "<score>0.0</score>"
        )
        result = compute_reward(
            response_text="I think it's 41.\n<answer>41</answer>",
            ground_truth="42",
            user_query="What is 6 times 7?",
            model_name="test-model",
            base_url="http://fake",
            api_key="fake-key",
        )
        assert result["raw_answer"] == "41"
        assert result["answer_reward"] == 0.0

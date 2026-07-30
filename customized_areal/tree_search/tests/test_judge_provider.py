# SPDX-License-Identifier: Apache-2.0
"""Tests for ExternalDiagnoseProvider.score_episode (LLM-judge scoring)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from customized_areal.tree_search.distilling.diagnose_provider import (
    ExternalDiagnoseProvider,
)
from customized_areal.tree_search.distilling.teacher_client import TeacherServiceError

_JUDGE_XML = (
    "```xml\n"
    "<judgment><turns>"
    "<turn><turn_idx>1</turn_idx><score>9</score></turn>"
    "<turn><turn_idx>2</turn_idx><score>4</score></turn>"
    "</turns></judgment>\n"
    "```"
)


def _make_provider_with_openai(content_or_exc):
    """Build a provider whose OpenAI client returns canned content or raises."""
    provider = ExternalDiagnoseProvider(
        client=MagicMock(),
        diagnose_model_name="judge-model",
        diagnose_base_url="http://judge.local/v1",
        diagnose_api_key="unused",
    )

    fake_client = MagicMock()
    if isinstance(content_or_exc, Exception):
        fake_client.chat.completions.create.side_effect = content_or_exc
    else:
        message = SimpleNamespace(content=content_or_exc)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice]
        )
    provider._openai_client = fake_client
    return provider


@pytest.mark.asyncio
async def test_score_episode_parses_scores():
    provider = _make_provider_with_openai(_JUDGE_XML)
    conversation = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "step 1"},
        {"role": "assistant", "content": "step 2"},
    ]
    scores = await provider.score_episode(conversation, "gold", score_max=10)
    assert scores == {1: 9, 2: 4}


@pytest.mark.asyncio
async def test_score_episode_clamps_via_parser():
    xml = "<turn><turn_idx>1</turn_idx><score>99</score></turn>"
    provider = _make_provider_with_openai(xml)
    scores = await provider.score_episode(
        [{"role": "user", "content": "Q"}], "g", score_max=10
    )
    assert scores == {1: 10}


@pytest.mark.asyncio
async def test_score_episode_api_error_raises():
    import openai

    provider = _make_provider_with_openai(openai.OpenAIError("boom"))
    with pytest.raises(TeacherServiceError):
        await provider.score_episode(
            [{"role": "user", "content": "Q"}], "g", score_max=10
        )


@pytest.mark.asyncio
async def test_score_episode_empty_content_raises():
    provider = _make_provider_with_openai("")  # empty content
    with pytest.raises(TeacherServiceError):
        await provider.score_episode(
            [{"role": "user", "content": "Q"}], "g", score_max=10
        )

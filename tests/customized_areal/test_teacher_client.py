"""Tests for TeacherClient and TeacherConfig."""

import math
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from customized_areal.tree_search.core.teacher_client import (
    TeacherClient,
    TeacherConfig,
)
from customized_areal.tree_search.core.teacher_provider import (
    EngineTeacherProvider,
    ExternalTeacherProvider,
)

# ---------------------------------------------------------------------------
# TeacherConfig tests
# ---------------------------------------------------------------------------


class TestTeacherConfig:
    """Test TeacherConfig defaults and custom values."""

    def test_defaults(self):
        """Test that default values are set correctly."""
        config = TeacherConfig()
        assert config.teacher_base_url == "http://localhost:8001"
        assert config.teacher_model_name == ""
        assert config.teacher_top_k == 10
        assert config.teacher_max_retries == 3
        assert config.teacher_timeout == 60.0
        assert config.teacher_missing_logprob == pytest.approx(
            math.log(1e-10), abs=0.01
        )

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = TeacherConfig(
            teacher_base_url="http://teacher:9000",
            teacher_model_name="big-model",
            teacher_top_k=20,
            teacher_max_retries=5,
            teacher_timeout=120.0,
            teacher_missing_logprob=-50.0,
        )
        assert config.teacher_base_url == "http://teacher:9000"
        assert config.teacher_model_name == "big-model"
        assert config.teacher_top_k == 20
        assert config.teacher_max_retries == 5
        assert config.teacher_timeout == 120.0
        assert config.teacher_missing_logprob == -50.0

    def test_missing_logprob_default_approx_minus_23(self):
        """Test the default missing logprob is approximately -23.025."""
        config = TeacherConfig()
        assert config.teacher_missing_logprob == pytest.approx(-23.025, abs=0.001)


# ---------------------------------------------------------------------------
# TeacherClient tests
# ---------------------------------------------------------------------------


def _make_teacher_api_response(
    prompt_len: int,
    output_logprobs: list[list[dict]],
) -> dict:
    """Build a mock vLLM/SGLang completions API response.

    Parameters
    ----------
    prompt_len : int
        Number of prompt positions (logprobs echoed for these).
    output_logprobs : list[list[dict]]
        For each output position, a list of dicts like
        ``{"token_id": int, "logprob": float}``.

    Returns
    -------
    dict
        A response dict mimicking the vLLM/SGLang completions API format.
    """
    # Prompt positions get empty logprob entries (echo=True includes them).
    prompt_top_logprobs: list[None] = [None] * prompt_len
    output_top_logprobs: list[list[dict] | None] = output_logprobs

    return {
        "choices": [
            {
                "logprobs": {
                    "top_logprobs": prompt_top_logprobs + output_top_logprobs,
                },
                "text": "generated",
            }
        ]
    }


class TestTeacherClient:
    """Test TeacherClient class."""

    @pytest.fixture
    def config(self):
        """Create a TeacherConfig for testing."""
        return TeacherConfig(
            teacher_base_url="http://localhost:8001",
            teacher_model_name="test-teacher",
            teacher_top_k=5,
            teacher_max_retries=2,
            teacher_timeout=10.0,
            teacher_missing_logprob=-23.0,
        )

    @pytest.fixture
    def client(self, config):
        """Create a TeacherClient for testing."""
        return TeacherClient(config)

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager opens and closes the httpx client."""
        assert client._client is None
        async with client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)
        assert client._client is None

    @pytest.mark.asyncio
    async def test_ensure_client_raises_when_not_open(self, client):
        """Test that calling methods without opening client raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not open"):
            client._ensure_client()

    @pytest.mark.asyncio
    async def test_get_logprobs_for_candidates_success(self, client):
        """Test successful get_logprobs_for_candidates with mocked HTTP."""
        # Setup: 3 prompt tokens, 2 output tokens
        input_ids = [100, 101, 102]
        output_ids = [200, 201]
        candidate_token_ids = [
            [200, 300, 400],  # position 0 candidates
            [201, 500, 600],  # position 1 candidates
        ]

        # Teacher returns top-2 logprobs per position
        mock_response_data = _make_teacher_api_response(
            prompt_len=3,
            output_logprobs=[
                # Position 0: teacher knows token 200 and 300
                [
                    {"token_id": 200, "logprob": -0.5},
                    {"token_id": 300, "logprob": -1.2},
                ],
                # Position 1: teacher knows token 201 and 500
                [
                    {"token_id": 201, "logprob": -0.3},
                    {"token_id": 500, "logprob": -0.8},
                ],
            ],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_http_client
            result = await client.get_logprobs_for_candidates(
                input_ids=input_ids,
                output_ids=output_ids,
                candidate_token_ids=candidate_token_ids,
            )

        # Position 0: 200 and 300 found, 400 missing
        assert result[0][200] == pytest.approx(-0.5)
        assert result[0][300] == pytest.approx(-1.2)
        assert result[0][400] == pytest.approx(-23.0)  # missing_logprob

        # Position 1: 201 and 500 found, 600 missing
        assert result[1][201] == pytest.approx(-0.3)
        assert result[1][500] == pytest.approx(-0.8)
        assert result[1][600] == pytest.approx(-23.0)  # missing_logprob

    @pytest.mark.asyncio
    async def test_get_logprobs_missing_candidates(self, client):
        """Test that candidate tokens not in teacher's top-k get missing_logprob."""
        input_ids = [10]
        output_ids = [20]
        candidate_token_ids = [
            [20, 30, 40, 50],  # Only token 20 is in teacher top-k
        ]

        mock_response_data = _make_teacher_api_response(
            prompt_len=1,
            output_logprobs=[
                [
                    {"token_id": 20, "logprob": -0.1},
                    {"token_id": 99, "logprob": -2.0},  # not a candidate
                ],
            ],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_http_client
            result = await client.get_logprobs_for_candidates(
                input_ids=input_ids,
                output_ids=output_ids,
                candidate_token_ids=candidate_token_ids,
            )

        # Token 20 is found, tokens 30, 40, 50 get missing_logprob
        assert result[0][20] == pytest.approx(-0.1)
        assert result[0][30] == pytest.approx(-23.0)
        assert result[0][40] == pytest.approx(-23.0)
        assert result[0][50] == pytest.approx(-23.0)

    @pytest.mark.asyncio
    async def test_get_logprobs_candidate_token_ids_length_mismatch(self, client):
        """Test that mismatched candidate_token_ids length raises ValueError."""
        input_ids = [10]
        output_ids = [20, 21]
        candidate_token_ids = [[20]]  # only 1 position, but 2 output tokens

        async with client:
            with pytest.raises(ValueError, match="must match output_ids length"):
                await client.get_logprobs_for_candidates(
                    input_ids=input_ids,
                    output_ids=output_ids,
                    candidate_token_ids=candidate_token_ids,
                )

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, client):
        """Test retry logic on transient HTTP failures."""
        input_ids = [10]
        output_ids = [20]
        candidate_token_ids = [[20]]

        # First call fails, second succeeds
        mock_response_data = _make_teacher_api_response(
            prompt_len=1,
            output_logprobs=[
                [{"token_id": 20, "logprob": -0.1}],
            ],
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()
        success_response.json = MagicMock(return_value=mock_response_data)

        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=fail_response,
            )
        )

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(side_effect=[fail_response, success_response])

        async with client:
            client._client = mock_http_client
            result = await client.get_logprobs_for_candidates(
                input_ids=input_ids,
                output_ids=output_ids,
                candidate_token_ids=candidate_token_ids,
            )

        assert result[0][20] == pytest.approx(-0.1)
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, client):
        """Test RuntimeError when all retries are exhausted."""
        input_ids = [10]
        output_ids = [20]
        candidate_token_ids = [[20]]

        # All calls fail
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=fail_response,
            )
        )

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=fail_response)

        async with client:
            client._client = mock_http_client
            with pytest.raises(RuntimeError, match="failed after 2 retries"):
                await client.get_logprobs_for_candidates(
                    input_ids=input_ids,
                    output_ids=output_ids,
                    candidate_token_ids=candidate_token_ids,
                )

    @pytest.mark.asyncio
    async def test_request_payload_format(self, client):
        """Test that the API request payload is constructed correctly."""
        input_ids = [100, 101]
        output_ids = [200]
        candidate_token_ids = [[200]]

        mock_response_data = _make_teacher_api_response(
            prompt_len=2,
            output_logprobs=[[{"token_id": 200, "logprob": -0.1}]],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_http_client
            await client.get_logprobs_for_candidates(
                input_ids=input_ids,
                output_ids=output_ids,
                candidate_token_ids=candidate_token_ids,
            )

        # Verify the POST call was made with the correct payload
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "/v1/completions"
        payload = call_args[1]["json"]
        assert payload["prompt"] == [100, 101]
        assert payload["max_tokens"] == 1
        assert payload["temperature"] == 0.0
        assert payload["logprobs"] == 5  # matches config.teacher_top_k
        assert payload["echo"] is True
        assert payload["model"] == "test-teacher"

    @pytest.mark.asyncio
    async def test_request_payload_no_model_name(self):
        """Test that model is omitted when teacher_model_name is empty."""
        config = TeacherConfig(teacher_model_name="")
        client = TeacherClient(config)

        input_ids = [10]
        output_ids = [20]
        candidate_token_ids = [[20]]

        mock_response_data = _make_teacher_api_response(
            prompt_len=1,
            output_logprobs=[[{"token_id": 20, "logprob": -0.1}]],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_http_client
            await client.get_logprobs_for_candidates(
                input_ids=input_ids,
                output_ids=output_ids,
                candidate_token_ids=candidate_token_ids,
            )

        payload = mock_http_client.post.call_args[1]["json"]
        assert "model" not in payload

    @pytest.mark.asyncio
    async def test_no_choices_raises(self, client):
        """Test that an API response with no choices raises RuntimeError."""
        input_ids = [10]
        output_ids = [20]
        candidate_token_ids = [[20]]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"choices": []})

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        async with client:
            client._client = mock_http_client
            with pytest.raises(RuntimeError, match="no choices"):
                await client.get_logprobs_for_candidates(
                    input_ids=input_ids,
                    output_ids=output_ids,
                    candidate_token_ids=candidate_token_ids,
                )

    @pytest.mark.asyncio
    async def test_retry_on_timeout_exception(self, client):
        """Test retry logic on httpx.TimeoutException."""
        input_ids = [10]
        output_ids = [20]
        candidate_token_ids = [[20]]

        mock_response_data = _make_teacher_api_response(
            prompt_len=1,
            output_logprobs=[[{"token_id": 20, "logprob": -0.1}]],
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()
        success_response.json = MagicMock(return_value=mock_response_data)

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("timed out"),
                success_response,
            ]
        )

        async with client:
            client._client = mock_http_client
            result = await client.get_logprobs_for_candidates(
                input_ids=input_ids,
                output_ids=output_ids,
                candidate_token_ids=candidate_token_ids,
            )

        assert result[0][20] == pytest.approx(-0.1)
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_text_returns_text_and_sends_payload(self, client):
        """Test complete_text returns choice text and sends expected payload."""
        client._post_with_retries = AsyncMock(
            return_value={"choices": [{"text": "diagnosis"}]}
        )

        result = await client.complete_text(
            "episode context",
            model="diagnoser",
            max_tokens=256,
            temperature=0.2,
        )

        assert result == "diagnosis"
        client._post_with_retries.assert_awaited_once_with(
            {
                "prompt": "episode context",
                "max_tokens": 256,
                "temperature": 0.2,
                "model": "diagnoser",
            }
        )

    @pytest.mark.asyncio
    async def test_complete_text_returns_message_content_when_text_absent(self, client):
        """Test complete_text falls back to chat message content."""
        client._post_with_retries = AsyncMock(
            return_value={"choices": [{"message": {"content": "diagnosis"}}]}
        )

        result = await client.complete_text("episode context")

        assert result == "diagnosis"

    @pytest.mark.asyncio
    async def test_complete_text_empty_choices_raises(self, client):
        """Test complete_text raises when the API returns no choices."""
        client._post_with_retries = AsyncMock(return_value={"choices": []})

        with pytest.raises(RuntimeError, match="no completion choices"):
            await client.complete_text("episode context")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("response_data", [[], "bad"])
    async def test_complete_text_malformed_response_type_raises(
        self, client, response_data
    ):
        """Test complete_text raises when the API response is not a mapping."""
        client._post_with_retries = AsyncMock(return_value=response_data)

        with pytest.raises(RuntimeError, match="response must be a mapping"):
            await client.complete_text("episode context")

    @pytest.mark.asyncio
    async def test_complete_text_malformed_choices_type_raises(self, client):
        """Test complete_text raises when choices is not a sequence."""
        client._post_with_retries = AsyncMock(return_value={"choices": {"text": "x"}})

        with pytest.raises(RuntimeError, match="choices must be a sequence"):
            await client.complete_text("episode context")

    @pytest.mark.asyncio
    async def test_complete_text_choice_without_text_raises(self, client):
        """Test complete_text raises when a choice has no completion text."""
        client._post_with_retries = AsyncMock(return_value={"choices": [{}]})

        with pytest.raises(RuntimeError, match="contained no text"):
            await client.complete_text("episode context")

    @pytest.mark.asyncio
    async def test_complete_text_malformed_choice_type_raises(self, client):
        """Test complete_text raises when the choice is not a mapping."""
        client._post_with_retries = AsyncMock(return_value={"choices": ["bad-choice"]})

        with pytest.raises(RuntimeError, match="completion choice"):
            await client.complete_text("episode context")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message", [None, "bad-message"])
    async def test_complete_text_malformed_message_raises(self, client, message):
        """Test complete_text raises when message is present but malformed."""
        client._post_with_retries = AsyncMock(
            return_value={"choices": [{"message": message}]}
        )

        with pytest.raises(RuntimeError, match="completion message"):
            await client.complete_text("episode context")


class FakeTeacherClient:
    def __init__(self):
        self.diagnose_payload = None
        self.logprob_payload = None

    async def complete_text(
        self,
        prompt,
        *,
        model=None,
        max_tokens=1024,
        temperature=0.0,
    ):
        self.diagnose_payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return (
            '{"turns":[{"turn_idx":1,"should_improve":true,"guidance":"Be precise."}]}'
        )

    async def get_logprobs_for_candidates(
        self,
        input_ids,
        output_ids,
        candidate_token_ids,
        tokenizer=None,
    ):
        self.logprob_payload = {
            "input_ids": input_ids,
            "output_ids": output_ids,
            "candidate_token_ids": candidate_token_ids,
            "tokenizer": tokenizer,
        }
        return [{7: -0.25, 8: -1.5}, {9: -0.75}]


@pytest.mark.asyncio
async def test_external_provider_delegates_diagnosis_to_client():
    client = FakeTeacherClient()
    provider = ExternalTeacherProvider(
        client=client,
        diagnose_model_name="qwen-397b",
        diagnose_max_tokens=300,
        diagnose_temperature=0.0,
    )

    text = await provider.diagnose_episode("context", "gold")

    assert "turns" in text
    assert client.diagnose_payload["model"] == "qwen-397b"
    assert client.diagnose_payload["max_tokens"] == 300
    assert client.diagnose_payload["temperature"] == 0.0
    assert "context" in client.diagnose_payload["prompt"]
    assert "gold" in client.diagnose_payload["prompt"]


@pytest.mark.asyncio
async def test_external_provider_delegates_candidate_logprobs_to_client():
    client = FakeTeacherClient()
    provider = ExternalTeacherProvider(client=client)

    result = await provider.get_logprobs_for_prompt(
        prompt_ids=[1, 2],
        generation_ids=[7, 9],
        candidate_token_ids=[[7, 8], [9]],
    )

    assert result == [[-0.25, -1.5], [-0.75]]
    assert client.logprob_payload["input_ids"] == [1, 2]
    assert client.logprob_payload["output_ids"] == [7, 9]
    assert client.logprob_payload["candidate_token_ids"] == [[7, 8], [9]]


def test_engine_provider_fails_early_without_compatible_methods():
    class Engine:
        pass

    with pytest.raises(NotImplementedError, match="engine-backed teacher provider"):
        EngineTeacherProvider(Engine())


def test_engine_provider_fails_early_with_non_callable_logprobs_method():
    class Engine:
        get_logprobs_for_prompt = None

    with pytest.raises(NotImplementedError, match="engine.get_logprobs_for_prompt"):
        EngineTeacherProvider(Engine())


def test_engine_provider_rejects_sync_logprobs_method_at_init():
    class Engine:
        def get_logprobs_for_prompt(
            self,
            prompt_ids,
            generation_ids,
            candidate_token_ids,
        ):
            return [[-0.1]]

    with pytest.raises(NotImplementedError, match="async.*get_logprobs_for_prompt"):
        EngineTeacherProvider(Engine())


@pytest.mark.asyncio
async def test_engine_provider_diagnose_without_engine_method_raises():
    class Engine:
        async def get_logprobs_for_prompt(
            self,
            prompt_ids,
            generation_ids,
            candidate_token_ids,
        ):
            return [[-0.1]]

    provider = EngineTeacherProvider(Engine())

    with pytest.raises(NotImplementedError, match="engine.diagnose_episode"):
        await provider.diagnose_episode("context", "gold")


@pytest.mark.asyncio
async def test_engine_provider_diagnose_with_non_callable_engine_method_raises():
    class Engine:
        async def get_logprobs_for_prompt(
            self,
            prompt_ids,
            generation_ids,
            candidate_token_ids,
        ):
            return [[-0.1]]

        diagnose_episode = None

    provider = EngineTeacherProvider(Engine())

    with pytest.raises(NotImplementedError, match="engine.diagnose_episode"):
        await provider.diagnose_episode("context", "gold")


@pytest.mark.asyncio
async def test_engine_provider_rejects_sync_diagnose_method_at_use():
    class Engine:
        async def get_logprobs_for_prompt(
            self,
            prompt_ids,
            generation_ids,
            candidate_token_ids,
        ):
            return [[-0.1]]

        def diagnose_episode(self, context, gold_answer):
            return "diagnosis"

    provider = EngineTeacherProvider(Engine())

    with pytest.raises(NotImplementedError, match="async.*diagnose_episode"):
        await provider.diagnose_episode("context", "gold")


@pytest.mark.asyncio
async def test_engine_provider_delegates_logprobs_to_engine():
    class Engine:
        def __init__(self):
            self.get_logprobs_for_prompt = AsyncMock(return_value=[[0.1, 0.2]])

    engine = Engine()
    provider = EngineTeacherProvider(engine)

    result = await provider.get_logprobs_for_prompt(
        prompt_ids=[1, 2],
        generation_ids=[3],
        candidate_token_ids=[[4, 5]],
    )

    assert result == [[0.1, 0.2]]
    engine.get_logprobs_for_prompt.assert_awaited_once_with(
        prompt_ids=[1, 2],
        generation_ids=[3],
        candidate_token_ids=[[4, 5]],
    )

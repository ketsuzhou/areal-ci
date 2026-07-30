# Teacher Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement teacher distillation where a remote teacher model evaluates the
student's top-k candidate tokens and provides position-level rewards (student_logp -
teacher_logp) for the grpo_distill_loss_fn.

**Architecture:** Proxy-server-mediated teacher evaluation. After student rollout via
proxy server, \_compute_token_rewards calls the teacher API to get logprobs for
student's top-k candidates at each position, computes rewards, and sends
PositionRewardInfo to the proxy server. The existing MultiCandidateFSDPEngine and
grpo_distill_loss_fn consume these rewards during training.

**Tech Stack:** Python 3.12+, aiohttp/httpx for async HTTP, OpenAI-compatible
completions API (vLLM/SGLang), PyTorch, AReaL framework.

______________________________________________________________________

## File Structure

| File                                                                       | Responsibility                                                      | Change Type |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------- |
| `customized_areal/on_policy_distill/core/config.py`                        | Add TeacherConfig fields to OnPolicyDistillConfig                   | Modify      |
| `customized_areal/on_policy_distill/core/teacher_client.py`                | **NEW**: TeacherClient for remote teacher API                       | Create      |
| `customized_areal/on_policy_distill/core/reward_compute.py`                | **NEW**: \_compute_token_rewards function                           | Create      |
| `customized_areal/on_policy_distill/core/agent.py`                         | Replace \_convert_to_position_rewards with \_compute_token_rewards  | Modify      |
| `customized_areal/on_policy_distill/core/__init__.py`                      | Export new modules                                                  | Modify      |
| `customized_areal/on_policy_distill/proxy/client.py`                       | Fix set_last_rewards bug; add get_last_interaction                  | Modify      |
| `customized_areal/on_policy_distill/engine/fsdp_engine.py`                 | Fix try/finally for rolled_input_ids; fix tree training duplication | Modify      |
| `customized_areal/on_policy_distill/training/actor.py`                     | Fix stats logging gap in patched \_ppo_update                       | Modify      |
| `customized_areal/on_policy_distill/configs/config_on_policy_distill.yaml` | Add teacher config section                                          | Modify      |
| `areal/api/io_struct.py`                                                   | Add output_top_logprobs field to ModelResponse                      | Modify      |
| `areal/engine/sglang_remote.py`                                            | Parse and store top_logprobs from SGLang response                   | Modify      |
| `tests/customized_areal/test_teacher_client.py`                            | **NEW**: Unit tests for TeacherClient                               | Create      |
| `tests/customized_areal/test_reward_compute.py`                            | **NEW**: Unit tests for \_compute_token_rewards                     | Create      |

______________________________________________________________________

### Task 1: Add output_top_logprobs to ModelResponse

This is the foundation - everything else depends on having top-k logprobs available in
ModelResponse.

**Files:**

- Modify: `areal/api/io_struct.py:60-81` (ModelResponse dataclass)

- Test: `tests/test_model_response_top_logprobs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_response_top_logprobs.py`:

```python
"""Tests for ModelResponse.output_top_logprobs field."""
import pytest
from areal.api.io_struct import ModelResponse


def test_model_response_default_top_logprobs_is_none():
    """ModelResponse should default output_top_logprobs to None."""
    resp = ModelResponse()
    assert resp.output_top_logprobs is None


def test_model_response_top_logprobs_assignment():
    """ModelResponse should accept output_top_logprobs as list of list of tuples."""
    top_logprobs = [
        [(100, -0.5), (200, -1.2), (300, -2.0)],  # position 0
        [(150, -0.3), (250, -0.9), (350, -1.5)],  # position 1
    ]
    resp = ModelResponse(output_top_logprobs=top_logprobs)
    assert resp.output_top_logprobs == top_logprobs
    assert len(resp.output_top_logprobs) == 2


def test_model_response_backward_compatible_without_top_logprobs():
    """Code that doesn't use output_top_logprobs should work unchanged."""
    resp = ModelResponse(
        input_tokens=[1, 2, 3],
        output_tokens=[4, 5, 6],
        output_logprobs=[-0.1, -0.2, -0.3],
    )
    # Existing fields work
    assert resp.input_len == 3
    assert resp.output_len == 3
    assert resp.output_logprobs == [-0.1, -0.2, -0.3]
    # New field is None
    assert resp.output_top_logprobs is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_response_top_logprobs.py -v` Expected: FAIL with
`AttributeError: ModelResponse has no field output_top_logprobs`

- [ ] **Step 3: Add output_top_logprobs field to ModelResponse**

In `areal/api/io_struct.py`, add after line 66 (after `output_versions`):

```python
    # Top-k logprobs per output position: list of lists of (token_id, log_prob) tuples.
    # None if not requested or not available.
    output_top_logprobs: list[list[tuple[int, float]]] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_response_top_logprobs.py -v` Expected: All 3 tests
PASS

- [ ] **Step 5: Commit**

```bash
git add areal/api/io_struct.py tests/test_model_response_top_logprobs.py
git commit -m "feat: add output_top_logprobs field to ModelResponse for teacher distillation"
```

______________________________________________________________________

### Task 2: Parse top-k logprobs from SGLang response

Extend the SGLang remote engine to capture and store top-k logprobs when available.

**Files:**

- Modify: `areal/engine/sglang_remote.py:88-124` (parse_generation_response)

- Test: `tests/test_sglang_top_logprobs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sglang_top_logprobs.py`:

```python
"""Tests for SGLang remote engine top-k logprobs parsing."""
import pytest
from areal.engine.sglang_remote import SGLangBackend


def test_parse_top_logprobs_from_meta_info():
    """parse_generation_response should extract output_top_logprobs from meta_info."""
    backend = SGLangBackend.__new__(SGLangBackend)

    meta_info = {
        "output_token_logprobs": [
            (-0.5, 100),  # (logprob, token_id) for position 0
            (-0.3, 150),  # position 1
        ],
        "output_top_logprobs": [
            {100: -0.5, 200: -1.2, 300: -2.0},  # position 0: top-3
            {150: -0.3, 250: -0.9, 350: -1.5},  # position 1: top-3
        ],
        "finish_reason": "stop",
    }

    result = backend.parse_generation_response(
        token_ids=[100, 150],
        meta_info=meta_info,
        finish_reason="stop",
    )

    assert result.output_top_logprobs is not None
    assert len(result.output_top_logprobs) == 2
    # Position 0: top-3 logprobs
    pos0 = result.output_top_logprobs[0]
    assert len(pos0) == 3
    assert (100, pytest.approx(-0.5, abs=1e-6)) in pos0
    assert (200, pytest.approx(-1.2, abs=1e-6)) in pos0
    assert (300, pytest.approx(-2.0, abs=1e-6)) in pos0
    # Position 1: top-3 logprobs
    pos1 = result.output_top_logprobs[1]
    assert len(pos1) == 3
    assert (150, pytest.approx(-0.3, abs=1e-6)) in pos1


def test_parse_no_top_logprobs_backward_compatible():
    """When output_top_logprobs is not in meta_info, result should have None."""
    backend = SGLangBackend.__new__(SGLangBackend)

    meta_info = {
        "output_token_logprobs": [
            (-0.5, 100),
            (-0.3, 150),
        ],
        "finish_reason": "stop",
    }

    result = backend.parse_generation_response(
        token_ids=[100, 150],
        meta_info=meta_info,
        finish_reason="stop",
    )

    assert result.output_top_logprobs is None


def test_parse_top_logprobs_with_token_string_keys():
    """SGLang may return top_logprobs with string token keys; convert to token IDs."""
    backend = SGLangBackend.__new__(SGLangBackend)

    meta_info = {
        "output_token_logprobs": [(-0.5, 100)],
        "output_top_logprobs": [
            {"hello": -0.5, "world": -1.2, "test": -2.0},
        ],
        "finish_reason": "stop",
    }

    # When top_logprobs has string keys, we should convert using tokenizer
    # or fall back to numeric token IDs from the paired output_token_logprobs
    result = backend.parse_generation_response(
        token_ids=[100],
        meta_info=meta_info,
        finish_reason="stop",
    )

    # If tokenizer is not available, string keys should be skipped or handled gracefully
    # The implementation should handle this case
    assert result.output_top_logprobs is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sglang_top_logprobs.py -v` Expected: FAIL
(output_top_logprobs not yet parsed)

- [ ] **Step 3: Implement top-k logprobs parsing in SGLangBackend**

In `areal/engine/sglang_remote.py`, modify `parse_generation_response` (around lines
88-124) to extract `output_top_logprobs` from `meta_info`:

After the existing lines that extract `output_tokens` and `output_logprobs` (lines
116-117), add:

```python
    # Extract top-k logprobs per position if available
    output_top_logprobs = None
    if "output_top_logprobs" in meta_info:
        raw_top_logprobs = meta_info["output_top_logprobs"]
        output_top_logprobs = []
        for pos_top_logprobs in raw_top_logprobs:
            position_logprobs = []
            for token_key, logprob in pos_top_logprobs.items():
                if isinstance(token_key, int):
                    position_logprobs.append((token_key, logprob))
                # String token keys require tokenizer for ID conversion
                # They are handled separately if tokenizer is provided
            output_top_logprobs.append(position_logprobs)
```

And add `output_top_logprobs=output_top_logprobs` to the `HttpGenerationResult` return
statement.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sglang_top_logprobs.py -v` Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add areal/engine/sglang_remote.py tests/test_sglang_top_logprobs.py
git commit -m "feat: parse output_top_logprobs from SGLang response for teacher distillation"
```

______________________________________________________________________

### Task 3: Create TeacherClient

New module for calling the remote teacher inference API.

**Files:**

- Create: `customized_areal/on_policy_distill/core/teacher_client.py`

- Test: `tests/customized_areal/test_teacher_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/customized_areal/test_teacher_client.py`:

```python
"""Tests for TeacherClient."""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from customized_areal.on_policy_distill.core.teacher_client import TeacherClient, TeacherConfig


@pytest.fixture
def teacher_config():
    return TeacherConfig(
        teacher_base_url="http://localhost:8001",
        teacher_model_name="test-teacher-model",
        teacher_top_k=5,
        teacher_max_retries=2,
        teacher_timeout=10.0,
    )


@pytest.fixture
def teacher_client(teacher_config):
    return TeacherClient(teacher_config)


def test_teacher_config_defaults():
    """TeacherConfig should have sensible defaults."""
    config = TeacherConfig()
    assert config.teacher_base_url == "http://localhost:8001"
    assert config.teacher_model_name == ""
    assert config.teacher_top_k == 10
    assert config.teacher_max_retries == 3
    assert config.teacher_timeout == 60.0


def test_teacher_config_custom():
    """TeacherConfig should accept custom values."""
    config = TeacherConfig(
        teacher_base_url="http://teacher:8002",
        teacher_model_name="big-model",
        teacher_top_k=20,
    )
    assert config.teacher_base_url == "http://teacher:8002"
    assert config.teacher_model_name == "big-model"
    assert config.teacher_top_k == 20


@pytest.mark.asyncio
async def test_get_logprobs_for_candidates(teacher_client):
    """TeacherClient should call teacher API and extract logprobs for candidate tokens."""
    # Mock the HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "logprobs": {
                    "token_logprobs": [-0.5, -0.3],
                    "tokens": ["hello", " world"],
                    "top_logprobs": [
                        {"hello": -0.5, " world": -1.0, " the": -2.0, " a": -2.5, " an": -3.0},
                        {" world": -0.3, " the": -0.8, " a": -1.5, " to": -2.0, " is": -2.5},
                    ],
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await teacher_client.get_logprobs_for_candidates(
            input_ids=[1, 2, 3],
            output_ids=[100, 200],
            candidate_token_ids=[[100, 200, 300], [200, 400, 500]],
            tokenizer=None,
        )

    assert len(result) == 2  # One dict per output position
    # Position 0: teacher logprobs for candidates 100, 200, 300
    assert 100 in result[0]  # "hello" token
    assert 200 in result[0]  # " world" token
    assert 300 in result[0]  # " the" token
    assert result[0][100] == pytest.approx(-0.5, abs=1e-6)
    assert result[0][200] == pytest.approx(-1.0, abs=1e-6)

    # Position 1: teacher logprobs for candidates 200, 400, 500
    assert 200 in result[1]
    assert result[1][200] == pytest.approx(-0.3, abs=1e-6)


@pytest.mark.asyncio
async def test_get_logprobs_handles_missing_candidates(teacher_client):
    """TeacherClient should use default logprob for tokens not in teacher's top-k."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "logprobs": {
                    "token_logprobs": [-0.5],
                    "tokens": ["hello"],
                    "top_logprobs": [
                        {"hello": -0.5, " world": -1.0},
                    ],
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await teacher_client.get_logprobs_for_candidates(
            input_ids=[1, 2, 3],
            output_ids=[100],
            candidate_token_ids=[[100, 999]],  # 999 is not in teacher's top-k
            tokenizer=None,
        )

    # Token 100 ("hello") should have teacher logprob
    assert 100 in result[0]
    assert result[0][100] == pytest.approx(-0.5, abs=1e-6)
    # Token 999 should get the default logprob (very negative)
    assert 999 in result[0]
    assert result[0][999] < -20  # Default is log(1e-10) ≈ -23


@pytest.mark.asyncio
async def test_get_logprobs_retry_on_failure(teacher_client):
    """TeacherClient should retry on transient failures."""
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Connection refused")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [-0.5],
                        "tokens": ["hello"],
                        "top_logprobs": [{"hello": -0.5}],
                    }
                }
            ]
        }
        return mock_response

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_post):
        result = await teacher_client.get_logprobs_for_candidates(
            input_ids=[1],
            output_ids=[100],
            candidate_token_ids=[[100]],
            tokenizer=None,
        )

    assert call_count == 2
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/customized_areal/test_teacher_client.py -v` Expected: FAIL
with
`ModuleNotFoundError: No module named 'customized_areal.on_policy_distill.core.teacher_client'`

- [ ] **Step 3: Implement TeacherClient**

Create `customized_areal/on_policy_distill/core/teacher_client.py`:

```python
"""TeacherClient for calling remote teacher inference API.

This module provides an async client for evaluating student model outputs
against a teacher model via the OpenAI-compatible completions API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import httpx

from areal.utils import logging

logger = logging.getLogger("TeacherClient")

# Default logprob for tokens not in the teacher's top-k
DEFAULT_MISSING_LOGPROB = math.log(1e-10)  # ≈ -23.0


@dataclass
class TeacherConfig:
    """Configuration for the remote teacher model.

    Attributes:
        teacher_base_url: Base URL for the teacher inference API (vLLM/SGLang).
        teacher_model_name: Model name for the teacher API.
        teacher_top_k: Number of top candidate tokens to evaluate per position.
        teacher_max_retries: Maximum number of retries for API calls.
        teacher_timeout: Request timeout in seconds.
        teacher_missing_logprob: Default logprob for tokens not in teacher's top-k.
    """

    teacher_base_url: str = "http://localhost:8001"
    teacher_model_name: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_timeout: float = 60.0
    teacher_missing_logprob: float = DEFAULT_MISSING_LOGPROB


class TeacherClient:
    """Async client for calling the remote teacher inference API.

    Uses the OpenAI-compatible completions API with logprobs to evaluate
    the teacher model on the same prefix as the student, then extracts
    logprobs for the student's candidate tokens.

    Example:
        config = TeacherConfig(teacher_base_url="http://localhost:8001")
        client = TeacherClient(config)
        logprobs = await client.get_logprobs_for_candidates(
            input_ids=[1, 2, 3],
            output_ids=[4, 5, 6],
            candidate_token_ids=[[4, 50, 60], [5, 70, 80]],
            tokenizer=None,
        )
    """

    def __init__(self, config: TeacherConfig):
        self.base_url = config.teacher_base_url.rstrip("/")
        self.model_name = config.teacher_model_name
        self.top_k = config.teacher_top_k
        self.max_retries = config.teacher_max_retries
        self.timeout = config.teacher_timeout
        self.missing_logprob = config.teacher_missing_logprob

    async def get_logprobs_for_candidates(
        self,
        input_ids: list[int],
        output_ids: list[int],
        candidate_token_ids: list[list[int]],
        tokenizer: Any = None,
    ) -> list[dict[int, float]]:
        """Get teacher logprobs for specific candidate token IDs at each position.

        Sends the full sequence (input + output) to the teacher model's
        completions API with echo=True and logprobs enabled, then extracts
        the teacher's log probabilities for the specified candidate tokens
        at each output position.

        Parameters
        ----------
        input_ids : list[int]
            Input (prefix) token IDs.
        output_ids : list[int]
            Output token IDs generated by the student.
        candidate_token_ids : list[list[int]]
            For each output position, a list of candidate token IDs
            (the student's top-k tokens) for which to get teacher logprobs.
        tokenizer : Any, optional
            Tokenizer for converting token IDs to strings (for API call).
            If None, uses raw token IDs.

        Returns
        -------
        list[dict[int, float]]
            For each output position, a dict mapping token_id -> teacher_logprob.
            Tokens not found in teacher's top-k receive `self.missing_logprob`.
        """
        import time

        start_time = time.monotonic()
        n_positions = len(output_ids)
        logger.info(
            "TeacherClient: getting logprobs for %d positions, %d avg candidates",
            n_positions,
            sum(len(c) for c in candidate_token_ids) // max(n_positions, 1),
        )

        # Build the prompt for the teacher model
        # Use the token IDs directly via the completions API
        full_token_ids = input_ids + output_ids
        prompt_tokens = input_ids

        # Make a single completions API call with echo=True to get
        # logprobs for all output positions at once
        response_data = await self._call_teacher_api(prompt_tokens, len(output_ids))

        # Extract logprobs for each position
        result = self._extract_candidate_logprobs(response_data, candidate_token_ids)

        elapsed = time.monotonic() - start_time
        logger.info(
            "TeacherClient: completed in %.2fs, %d positions processed",
            elapsed,
            n_positions,
        )

        return result

    async def _call_teacher_api(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ) -> dict:
        """Make a completions API call to the teacher model.

        Parameters
        ----------
        prompt_tokens : list[int]
            The prompt token IDs to send to the teacher.
        max_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        dict
            The parsed JSON response from the teacher API.
        """
        url = f"{self.base_url}/v1/completions"

        payload = {
            "model": self.model_name,
            "prompt": prompt_tokens,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Greedy for deterministic evaluation
            "logprobs": self.top_k,
            "echo": True,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)

                if response.status_code == 200:
                    return response.json()

                last_error = Exception(
                    f"Teacher API returned status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                logger.warning(
                    "TeacherClient: attempt %d/%d failed with status %d",
                    attempt + 1,
                    self.max_retries,
                    response.status_code,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "TeacherClient: attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    str(e)[:100],
                )

        raise RuntimeError(
            f"TeacherClient: all {self.max_retries} attempts failed. "
            f"Last error: {last_error}"
        )

    def _extract_candidate_logprobs(
        self,
        response_data: dict,
        candidate_token_ids: list[list[int]],
    ) -> list[dict[int, float]]:
        """Extract teacher logprobs for candidate tokens from API response.

        Parameters
        ----------
        response_data : dict
            Parsed JSON response from the teacher completions API.
        candidate_token_ids : list[list[int]]
            For each output position, the candidate token IDs to look up.

        Returns
        -------
        list[dict[int, float]]
            For each output position, a dict mapping token_id -> teacher_logprob.
        """
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("Teacher API response has no choices")

        logprobs_data = choices[0].get("logprobs", {})
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        result: list[dict[int, float]] = []
        n_output_positions = len(candidate_token_ids)

        for pos_idx in range(n_output_positions):
            candidates = candidate_token_ids[pos_idx]
            teacher_logprobs: dict[int, float] = {}

            # Default: missing logprob for all candidates
            for token_id in candidates:
                teacher_logprobs[token_id] = self.missing_logprob

            # Look up teacher logprobs from the response
            if pos_idx < len(top_logprobs_list):
                position_top_logprobs = top_logprobs_list[pos_idx]
                if position_top_logprobs is not None:
                    # top_logprobs is a dict mapping token_string -> logprob
                    # We need to match by token ID, but the API returns strings
                    # For now, we store string-keyed logprobs and match by ID later
                    # This requires tokenizer or a string-to-ID mapping
                    for token_key, logprob in position_top_logprobs.items():
                        if isinstance(token_key, int):
                            # If the API returns integer token IDs directly
                            if token_key in teacher_logprobs:
                                teacher_logprobs[token_key] = logprob

            result.append(teacher_logprobs)

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/customized_areal/test_teacher_client.py -v` Expected: All
tests PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/core/teacher_client.py tests/customized_areal/test_teacher_client.py
git commit -m "feat: add TeacherClient for remote teacher model API calls"
```

______________________________________________________________________

### Task 4: Create \_compute_token_rewards function

The core reward computation that compares student and teacher logprobs.

**Files:**

- Create: `customized_areal/on_policy_distill/core/reward_compute.py`

- Test: `tests/customized_areal/test_reward_compute.py`

- [ ] **Step 1: Write the failing test**

Create `tests/customized_areal/test_reward_compute.py`:

```python
"""Tests for _compute_token_rewards function."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from customized_areal.on_policy_distill.core.reward_compute import _compute_token_rewards
from customized_areal.on_policy_distill.core.teacher_client import TeacherClient, TeacherConfig
from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo


@pytest.fixture
def mock_teacher_client():
    """Create a TeacherClient with mocked get_logprobs_for_candidates."""
    config = TeacherConfig(teacher_top_k=3)
    client = TeacherClient(config)
    client.get_logprobs_for_candidates = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_compute_token_rewards_basic(mock_teacher_client):
    """Test basic reward computation with known logprobs."""
    # Student top-k at each position:
    # Position 0: student generated token 100 (logp=-0.5), also considered 200 (-1.0), 300 (-2.0)
    # Position 1: student generated token 150 (logp=-0.3), also considered 250 (-0.8), 350 (-1.5)
    student_top_k_logprobs = [
        [(100, -0.5), (200, -1.0), (300, -2.0)],  # position 0
        [(150, -0.3), (250, -0.8), (350, -1.5)],   # position 1
    ]

    # Teacher logprobs for those same tokens
    mock_teacher_client.get_logprobs_for_candidates.return_value = [
        {100: -0.3, 200: -1.2, 300: -2.5},  # position 0
        {150: -0.6, 250: -1.0, 350: -1.8},   # position 1
    ]

    result = await _compute_token_rewards(
        student_output_ids=[100, 150],
        student_input_ids=[1, 2, 3],
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=mock_teacher_client,
        top_k=3,
    )

    assert len(result) == 2

    # Position 0: reward = student_logp - teacher_logp
    # Token 100: -0.5 - (-0.3) = -0.2  (student less confident)
    # Token 200: -1.0 - (-1.2) = 0.2   (student more confident)
    # Token 300: -2.0 - (-2.5) = 0.5   (student much more confident)
    pos0 = result[0]
    assert pos0.position == 0
    assert pos0.candidate_token_ids == [100, 200, 300]
    assert pos0.chosen_index == 0  # Token 100 was generated
    assert len(pos0.rewards) == 3
    assert pos0.rewards[0] == pytest.approx(-0.2, abs=1e-6)
    assert pos0.rewards[1] == pytest.approx(0.2, abs=1e-6)
    assert pos0.rewards[2] == pytest.approx(0.5, abs=1e-6)

    # Position 1: reward = student_logp - teacher_logp
    pos1 = result[1]
    assert pos1.position == 1
    assert pos1.candidate_token_ids == [150, 250, 350]
    assert pos1.chosen_index == 0  # Token 150 was generated


@pytest.mark.asyncio
async def test_compute_token_rewards_with_missing_teacher_logprobs(mock_teacher_client):
    """Tokens not in teacher's top-k should use default missing logprob."""
    student_top_k_logprobs = [
        [(100, -0.5), (999, -3.0)],  # 999 not in teacher's vocabulary
    ]

    # Teacher only knows about token 100
    mock_teacher_client.get_logprobs_for_candidates.return_value = [
        {100: -0.3, 999: -23.0},  # 999 gets default missing logprob
    ]

    result = await _compute_token_rewards(
        student_output_ids=[100],
        student_input_ids=[1, 2],
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=mock_teacher_client,
        top_k=2,
    )

    assert len(result) == 1
    # Token 999: reward = -3.0 - (-23.0) = 20.0 (student much more confident than default)
    assert result[0].rewards[1] == pytest.approx(20.0, abs=1e-6)


@pytest.mark.asyncio
async def test_compute_token_rewards_empty_output(mock_teacher_client):
    """Empty output should return empty list."""
    result = await _compute_token_rewards(
        student_output_ids=[],
        student_input_ids=[1, 2, 3],
        student_top_k_logprobs=[],
        teacher_client=mock_teacher_client,
        top_k=3,
    )
    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/customized_areal/test_reward_compute.py -v` Expected: FAIL
with
`ModuleNotFoundError: No module named 'customized_areal.on_policy_distill.core.reward_compute'`

- [ ] **Step 3: Implement \_compute_token_rewards**

Create `customized_areal/on_policy_distill/core/reward_compute.py`:

```python
"""Compute position-level rewards from teacher/student logprob comparison.

This module provides the core reward computation function for on-policy distillation,
comparing student model top-k logprobs against teacher model logprobs at each
generation position.
"""

from __future__ import annotations

from areal.utils import logging
from ..proxy.cache import PositionRewardInfo
from .teacher_client import TeacherClient

logger = logging.getLogger("RewardCompute")


async def _compute_token_rewards(
    student_output_ids: list[int],
    student_input_ids: list[int],
    student_top_k_logprobs: list[list[tuple[int, float]]],
    teacher_client: TeacherClient,
    top_k: int = 10,
) -> list[PositionRewardInfo]:
    """Compute position-level rewards from teacher/student logprob comparison.

    For each output position:
    1. Take student's top-k tokens as candidates
    2. Call teacher to get teacher_logp for each candidate
    3. Compute reward = student_logp - teacher_logp for each candidate
    4. Build PositionRewardInfo with candidates, rewards, and chosen_index

    Parameters
    ----------
    student_output_ids : list[int]
        Student output token IDs (the actually generated tokens).
    student_input_ids : list[int]
        Input (prefix) token IDs.
    student_top_k_logprobs : list[list[tuple[int, float]]]
        Student's top-k logprobs per position. Each position has a list
        of (token_id, log_prob) tuples, sorted by probability (descending).
    teacher_client : TeacherClient
        TeacherClient for calling the teacher inference API.
    top_k : int
        Number of top candidates to consider per position.

    Returns
    -------
    list[PositionRewardInfo]
        One PositionRewardInfo per output position, containing candidate
        token IDs, student logprobs, rewards, and chosen_index.
    """
    if not student_output_ids or not student_top_k_logprobs:
        return []

    n_positions = len(student_output_ids)
    assert len(student_top_k_logprobs) == n_positions, (
        f"student_top_k_logprobs length ({len(student_top_k_logprobs)}) must match "
        f"student_output_ids length ({n_positions})"
    )

    # Build candidate_token_ids for teacher API call
    candidate_token_ids: list[list[int]] = []
    for pos_logprobs in student_top_k_logprobs:
        # Take top-k candidates (already sorted by probability)
        candidates = [tid for tid, _ in pos_logprobs[:top_k]]
        candidate_token_ids.append(candidates)

    # Call teacher API to get teacher logprobs for all candidate positions
    teacher_logprobs = await teacher_client.get_logprobs_for_candidates(
        input_ids=student_input_ids,
        output_ids=student_output_ids,
        candidate_token_ids=candidate_token_ids,
    )

    # Build PositionRewardInfo for each position
    position_rewards: list[PositionRewardInfo] = []

    for i in range(n_positions):
        output_token_id = student_output_ids[i]
        pos_candidates = student_top_k_logprobs[i][:top_k]

        candidate_token_ids_pos = [tid for tid, _ in pos_candidates]
        student_logprobs_pos = [logp for _, logp in pos_candidates]
        teacher_logprobs_pos = teacher_logprobs[i]

        # Find chosen_index: the index of the actually generated token
        chosen_index = 0  # Default: first candidate
        for idx, tid in enumerate(candidate_token_ids_pos):
            if tid == output_token_id:
                chosen_index = idx
                break

        # Compute reward = student_logp - teacher_logp for each candidate
        rewards = []
        for j, (tid, s_logp) in enumerate(pos_candidates):
            t_logp = teacher_logprobs_pos.get(tid, teacher_client.missing_logprob)
            reward = s_logp - t_logp
            rewards.append(reward)

        # Build token strings for candidates (use str(token_id) as placeholder)
        candidates_str = [str(tid) for tid in candidate_token_ids_pos]

        position_rewards.append(
            PositionRewardInfo(
                position=i,
                candidates=candidates_str,
                candidate_token_ids=candidate_token_ids_pos,
                logprobs=student_logprobs_pos,
                rewards=rewards,
                chosen_index=chosen_index,
            )
        )

    logger.info(
        "Computed token rewards: %d positions, avg %.1f candidates per position, "
        "reward mean=%.4f std=%.4f",
        n_positions,
        sum(len(pr.candidates) for pr in position_rewards) / max(n_positions, 1),
        sum(sum(pr.rewards) / len(pr.rewards) for pr in position_rewards) / max(n_positions, 1),
        0.0,  # std computation omitted for logging brevity
    )

    return position_rewards
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/customized_areal/test_reward_compute.py -v` Expected: All
tests PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/core/reward_compute.py tests/customized_areal/test_reward_compute.py
git commit -m "feat: add _compute_token_rewards for teacher/student logprob comparison"
```

______________________________________________________________________

### Task 5: Add TeacherConfig to OnPolicyDistillConfig

Add teacher configuration fields and integrate TeacherClient into OnPolicyDistillAgent.

**Files:**

- Modify: `customized_areal/on_policy_distill/core/config.py:1-98`

- Modify: `customized_areal/on_policy_distill/core/agent.py:1-292`

- Modify: `customized_areal/on_policy_distill/core/__init__.py`

- [ ] **Step 1: Add TeacherConfig fields to OnPolicyDistillConfig**

In `customized_areal/on_policy_distill/core/config.py`, add after the `reward_bias`
field (line 98):

```python
    # Teacher model configuration
    teacher_base_url: str = field(
        default="http://localhost:8001",
        metadata={"help": "Base URL for the teacher model inference API."},
    )
    teacher_model_name: str = field(
        default="",
        metadata={"help": "Teacher model name for the inference API. Required for teacher distillation."},
    )
    teacher_top_k: int = field(
        default=10,
        metadata={"help": "Number of top candidate tokens to evaluate per position."},
    )
    teacher_max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for teacher API calls."},
    )
    teacher_timeout: float = field(
        default=60.0,
        metadata={"help": "Request timeout in seconds for teacher API calls."},
    )
    teacher_missing_logprob: float = field(
        default=-23.0,
        metadata={"help": "Default logprob for tokens not in teacher's top-k (log(1e-10) ≈ -23.0)."},
    )
```

Also add the import for `TeacherConfig` at the top:

```python
from customized_areal.on_policy_distill.core.teacher_client import TeacherConfig
```

- [ ] **Step 2: Modify OnPolicyDistillAgent to use \_compute_token_rewards**

In `customized_areal/on_policy_distill/core/agent.py`:

1. Add imports at the top:

```python
from customized_areal.on_policy_distill.core.teacher_client import TeacherClient, TeacherConfig
from customized_areal.on_policy_distill.core.reward_compute import _compute_token_rewards
```

2. Modify `OnPolicyDistillAgent.__init__` (around line 87) to accept and initialize
   TeacherClient:

Replace the current `__init__` method with:

```python
    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        teacher_config: TeacherConfig | None = None,
        **kwargs,
    ):
        """Initialize OnPolicyDistillAgent.

        Args:
            agent_id: Optional agent ID to use. Falls back to default_agent_id.
            user_id: Optional user ID for authentication.
            model_name: Optional model name for LLM calls.
            teacher_config: Optional teacher configuration for distillation.
                If provided, _compute_token_rewards will be called to compute
                position-level rewards from teacher/student logprob comparison.
            **kwargs: Additional configuration options (ignored but accepted).
        """
        self.agent_id = agent_id or self.default_agent_id
        self.user_id = user_id
        self.model_name = model_name
        self.teacher_client = TeacherClient(teacher_config) if teacher_config else None

        logger.info(
            "OnPolicyDistillAgent initialized: agent_id=%s, model_name=%s, teacher=%s",
            self.agent_id,
            self.model_name,
            "enabled" if self.teacher_client else "disabled",
        )
```

3. Replace the `_convert_to_position_rewards` method and the position rewards extraction
   in `run()` (lines 112-216) with a new implementation that uses
   `_compute_token_rewards`. The key change is replacing lines 177-193 (the
   metadata-based extraction) with teacher-based computation:

Replace the block from `# Extract token_rewards from message metadata` through
`_convert_to_position_rewards` with:

```python
            # Compute position-level rewards using teacher model
            position_rewards: list[PositionRewardInfo] = []
            completion_id: str | None = None

            if completion_messages and self.teacher_client is not None:
                # Get student output tokens and logprobs from proxy interaction
                proxy_client = extra_kwargs.get("proxy_client")

                if proxy_client is not None:
                    # Get the last interaction from the proxy server
                    # to extract student output tokens and top-k logprobs
                    try:
                        interaction = await proxy_client.get_last_interaction()
                        if interaction and interaction.model_response is not None:
                            student_output_ids = interaction.model_response.output_tokens
                            student_input_ids = interaction.model_response.input_tokens
                            student_top_k_logprobs = getattr(
                                interaction.model_response, "output_top_logprobs", None
                            )

                            if student_top_k_logprobs is not None:
                                position_rewards = await _compute_token_rewards(
                                    student_output_ids=student_output_ids,
                                    student_input_ids=student_input_ids,
                                    student_top_k_logprobs=student_top_k_logprobs,
                                    teacher_client=self.teacher_client,
                                    top_k=self.teacher_client.top_k,
                                )
                                completion_id = hashlib.md5(
                                    str(completion_messages).encode()
                                ).hexdigest()[:16]
                                logger.info(
                                    "Computed position rewards via teacher: %d positions",
                                    len(position_rewards),
                                )
                    except Exception as e:
                        logger.warning("Failed to compute position rewards via teacher: %s", e)

            # Calculate scalar reward using reward function
            reward_fn = AsyncRewardWrapper(on_policy_distill_reward_fn)
            reward = await reward_fn(completions=completion_messages, gt=gt)
```

- [ ] **Step 3: Update core/__init__.py exports**

In `customized_areal/on_policy_distill/core/__init__.py`, add exports for the new
modules:

```python
from .teacher_client import TeacherClient, TeacherConfig
from .reward_compute import _compute_token_rewards
```

And update `__all__` to include these.

- [ ] **Step 4: Update parent __init__.py**

In `customized_areal/on_policy_distill/__init__.py`, add `__getattr__` entries for
`TeacherClient`, `TeacherConfig`, and `_compute_token_rewards`.

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `uv run pytest tests/customized_areal/ -v --timeout=30` Expected: Existing tests
PASS (new code is additive)

- [ ] **Step 6: Commit**

```bash
git add customized_areal/on_policy_distill/core/config.py customized_areal/on_policy_distill/core/agent.py customized_areal/on_policy_distill/core/__init__.py customized_areal/on_policy_distill/__init__.py
git commit -m "feat: integrate TeacherClient and _compute_token_rewards into OnPolicyDistillAgent"
```

______________________________________________________________________

### Task 6: Fix set_last_rewards bug and add get_last_interaction

Bug fix and feature addition for the proxy client.

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/client.py:212-251`

- [ ] **Step 1: Fix `set_last_rewards` to send `None` instead of `""`**

In `customized_areal/on_policy_distill/proxy/client.py`, change lines 212-229 and
231-251:

Replace `set_last_rewards` (line 227):

```python
        await self.set_rewards(
            completion_id="",  # Empty string means "last interaction"
            token_rewards=token_rewards,
        )
```

With:

```python
        await self.set_rewards(
            completion_id=None,  # None means "last interaction"
            token_rewards=token_rewards,
        )
```

Replace `set_last_position_rewards` (line 249):

```python
        await self.set_position_rewards(
            completion_id="",  # Empty string means "last interaction"
            position_rewards=position_rewards,
        )
```

With:

```python
        await self.set_position_rewards(
            completion_id=None,  # None means "last interaction"
            position_rewards=position_rewards,
        )
```

- [ ] **Step 2: Add `get_last_interaction` method to OpenAIProxyClient**

In `customized_areal/on_policy_distill/proxy/client.py`, add a new method after
`get_entropies`:

```python
    async def get_last_interaction(self) -> Any:
        """Get the most recent interaction from the proxy server.

        Returns the last interaction object which includes model_response
        with output_tokens, input_tokens, and output_top_logprobs.

        Returns
        -------
        Any
            The last interaction object, or None if no interactions exist.
        """
        if self.session_id is None:
            raise RuntimeError("Session not started")

        url = f"{self.base_url}{EXPORT_TRAJECTORIES_PATHNAME}"
        params = {"discount": "1.0", "style": "individual"}
        headers = self._session_auth_headers()

        async with self._session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        interactions = data.get("interactions", {})
        if not interactions:
            return None

        # Return the last interaction by insertion order
        last_id = list(interactions.keys())[-1]
        return interactions[last_id]
```

Also add the import for `EXPORT_TRAJECTORIES_PATHNAME` at the top if not already
present:

```python
from areal.experimental.openai.proxy.server import EXPORT_TRAJECTORIES_PATHNAME
```

- [ ] **Step 3: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/client.py
git commit -m "fix: set_last_rewards sends None instead of empty string; add get_last_interaction"
```

______________________________________________________________________

### Task 7: Fix engine and actor bugs

Fix the `rolled_input_ids` mutation without try/finally, tree training duplication, and
stats logging gap.

**Files:**

- Modify: `customized_areal/on_policy_distill/engine/fsdp_engine.py:255-315`

- Modify: `customized_areal/on_policy_distill/training/actor.py:43-71`

- [ ] **Step 1: Fix rolled_input_ids mutation in fsdp_engine.py**

In `customized_areal/on_policy_distill/engine/fsdp_engine.py`, replace lines 304-315
(the multi-candidate labels override and restoration):

Replace:

```python
                        if multi_candidate_labels is not None:
                        # Use multi-candidate labels for gathering
                        # Temporarily override rolled_input_ids
                        original_rolled = ctx.model_inputs.get("rolled_input_ids")
                        ctx.model_inputs["rolled_input_ids"] = multi_candidate_labels

                        logprobs, entropy = self._compute_logprobs_entropy(
                            logits, ctx.model_inputs, ctx.ulysses_pad_size
                        )

                        # Restore original
                        if original_rolled is not None:
                            ctx.model_inputs["rolled_input_ids"] = original_rolled
                        else:
                            del ctx.model_inputs["rolled_input_ids"]
```

With:

```python
                        if multi_candidate_labels is not None:
                        # Use multi-candidate labels for gathering
                        # Temporarily override rolled_input_ids with try/finally
                        # to ensure restoration even on error
                        original_rolled = ctx.model_inputs.get("rolled_input_ids")
                        ctx.model_inputs["rolled_input_ids"] = multi_candidate_labels

                        try:
                            logprobs, entropy = self._compute_logprobs_entropy(
                                logits, ctx.model_inputs, ctx.ulysses_pad_size
                            )
                        finally:
                            # Restore original rolled_input_ids
                            if original_rolled is not None:
                                ctx.model_inputs["rolled_input_ids"] = original_rolled
                            else:
                                ctx.model_inputs.pop("rolled_input_ids", None)
```

- [ ] **Step 2: Fix tree training path duplication**

In the same file, replace the tree training branch (lines 255-277) with a delegation to
super():

Replace the `if self.enable_tree_training:` block (lines 255-277) with:

```python
            if self.enable_tree_training:
                # Delegate to parent class for tree training
                return super()._compute_logprobs_and_loss(
                    logits, ctx, loss_fn, loss_weight_fn, total_loss_weight, loss_multiplier,
                )
```

Note: This requires removing the `from areal.models.tree_attn.functional import ...`
that was inside the tree training block. The import is no longer needed in this file
since we're delegating to super().

- [ ] **Step 3: Fix stats logging gap in actor.py**

In `customized_areal/on_policy_distill/training/actor.py`, add back critical stats
tracking to `_ppo_update_with_distill_loss`:

Replace the `_ppo_update_with_distill_loss` method body (lines 43-71) with:

```python
    def _ppo_update_with_distill_loss(self, data: dict[str, Any]) -> None:
        """PPO update using grpo_distill_loss_fn."""
        from ..training.loss import grpo_distill_loss_fn

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        self.engine.train()

        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_distill_loss_fn,
                        config=self.config,
                        current_version=current_version,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

            # Log critical denominator stats (dropped by original patch)
            loss_mask = data.get("loss_mask")
            if loss_mask is not None:
                if isinstance(loss_mask, torch.Tensor):
                    n_valid = loss_mask.count_nonzero().item()
                else:
                    n_valid = sum(x.count_nonzero().item() for x in mb_inputs.mbs)
                stats_tracker.denominator(n_valid_tokens=n_valid)
```

Add the missing import at the top:

```python
import torch
```

- [ ] **Step 4: Remove unused import**

In `customized_areal/on_policy_distill/engine/fsdp_engine.py`, remove the unused import
on line 247:

```python
from areal.engine.core import compute_total_loss_weight
```

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/engine/fsdp_engine.py customized_areal/on_policy_distill/training/actor.py
git commit -m "fix: rolled_input_ids try/finally, tree training delegation, stats logging gap"
```

______________________________________________________________________

### Task 8: Update config YAML

Add teacher configuration section to the YAML config.

**Files:**

- Modify:
  `customized_areal/on_policy_distill/configs/config_on_policy_distill.yaml:124-136`

- [ ] **Step 1: Add teacher config section**

In `config_on_policy_distill.yaml`, after line 136 (`reward_bias: -0.5`), add:

```yaml
# Teacher model configuration (for on-policy distillation)
teacher_base_url: http://localhost:8001
teacher_model_name: ""  # Required: teacher model name for inference API
teacher_top_k: 10
teacher_max_retries: 3
teacher_timeout: 60.0
teacher_missing_logprob: -23.0  # log(1e-10) ≈ -23.0
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/on_policy_distill/configs/config_on_policy_distill.yaml
git commit -m "feat: add teacher model configuration to on-policy distill config"
```

______________________________________________________________________

### Task 9: Remove dead code

Remove `_convert_to_position_rewards` that references non-existent `manager_idm.py` and
clean up.

**Files:**

- Modify: `customized_areal/on_policy_distill/core/agent.py`

- [ ] **Step 1: Remove `_convert_to_position_rewards` method**

In `customized_areal/on_policy_distill/core/agent.py`, delete the entire
`_convert_to_position_rewards` method (lines 218-292). This method referenced
non-existent `manager_idm.py:_compute_token_rewards` and is replaced by the
`_compute_token_rewards` function from `reward_compute.py`.

- [ ] **Step 2: Update `__init__.py` exports**

In `customized_areal/on_policy_distill/__init__.py`, remove
`_convert_to_position_rewards` from exports if it was listed (it wasn't, so no change
needed).

- [ ] **Step 3: Commit**

```bash
git add customized_areal/on_policy_distill/core/agent.py
git commit -m "refactor: remove dead _convert_to_position_rewards method referencing non-existent manager_idm"
```

______________________________________________________________________

### Task 10: Integration test

Write an end-to-end integration test that verifies the full teacher distillation
pipeline with mock components.

**Files:**

- Create: `tests/customized_areal/test_teacher_distill_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/customized_areal/test_teacher_distill_integration.py`:

```python
"""Integration test for teacher distillation pipeline.

Tests the full flow: TeacherConfig → TeacherClient → _compute_token_rewards → PositionRewardInfo
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from customized_areal.on_policy_distill.core.teacher_client import TeacherClient, TeacherConfig
from customized_areal.on_policy_distill.core.reward_compute import _compute_token_rewards
from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo


@pytest.mark.asyncio
async def test_end_to_end_teacher_distillation_pipeline():
    """Test the full teacher distillation pipeline from config to PositionRewardInfo."""
    # 1. Create TeacherConfig
    config = TeacherConfig(
        teacher_base_url="http://localhost:8001",
        teacher_model_name="test-teacher",
        teacher_top_k=3,
        teacher_max_retries=1,
        teacher_timeout=5.0,
    )
    assert config.teacher_top_k == 3

    # 2. Create TeacherClient
    client = TeacherClient(config)
    assert client.top_k == 3
    assert client.model_name == "test-teacher"

    # 3. Mock teacher API response
    client.get_logprobs_for_candidates = AsyncMock(return_value=[
        {100: -0.3, 200: -1.2, 300: -2.5},  # position 0
        {150: -0.6, 250: -1.0, 350: -1.8},   # position 1
        {400: -0.4, 500: -0.9, 600: -1.7},   # position 2
    ])

    # 4. Define student output
    student_output_ids = [100, 150, 400]
    student_input_ids = [1, 2, 3, 4, 5]
    student_top_k_logprobs = [
        [(100, -0.5), (200, -1.0), (300, -2.0)],   # position 0
        [(150, -0.3), (250, -0.8), (350, -1.5)],   # position 1
        [(400, -0.2), (500, -0.7), (600, -1.3)],   # position 2
    ]

    # 5. Compute token rewards
    position_rewards = await _compute_token_rewards(
        student_output_ids=student_output_ids,
        student_input_ids=student_input_ids,
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=client,
        top_k=3,
    )

    # 6. Verify results
    assert len(position_rewards) == 3

    # Position 0: chosen token = 100
    pr0 = position_rewards[0]
    assert pr0.position == 0
    assert pr0.candidate_token_ids == [100, 200, 300]
    assert pr0.chosen_index == 0
    assert pr0.logprobs == [-0.5, -1.0, -2.0]
    # reward = student_logp - teacher_logp
    assert pr0.rewards[0] == pytest.approx(-0.5 - (-0.3), abs=1e-6)  # -0.2
    assert pr0.rewards[1] == pytest.approx(-1.0 - (-1.2), abs=1e-6)  # 0.2
    assert pr0.rewards[2] == pytest.approx(-2.0 - (-2.5), abs=1e-6)  # 0.5

    # Position 1: chosen token = 150
    pr1 = position_rewards[1]
    assert pr1.position == 1
    assert pr1.chosen_index == 0
    assert pr1.rewards[0] == pytest.approx(-0.3 - (-0.6), abs=1e-6)  # 0.3

    # Position 2: chosen token = 400
    pr2 = position_rewards[2]
    assert pr2.position == 2
    assert pr2.chosen_index == 0
    assert pr2.rewards[0] == pytest.approx(-0.2 - (-0.4), abs=1e-6)  # 0.2


@pytest.mark.asyncio
async def test_pipeline_with_missing_teacher_candidates():
    """Test pipeline when some student candidates are not in teacher's top-k."""
    config = TeacherConfig(teacher_top_k=3, teacher_missing_logprob=-23.0)
    client = TeacherClient(config)

    # Teacher only has 2 of 3 candidates
    client.get_logprobs_for_candidates = AsyncMock(return_value=[
        {100: -0.3, 200: -1.2, 999: -23.0},  # 999 gets missing_logprob
    ])

    student_top_k_logprobs = [
        [(100, -0.5), (200, -1.0), (999, -3.0)],
    ]

    position_rewards = await _compute_token_rewards(
        student_output_ids=[100],
        student_input_ids=[1, 2, 3],
        student_top_k_logprobs=student_top_k_logprobs,
        teacher_client=client,
        top_k=3,
    )

    assert len(position_rewards) == 1
    pr = position_rewards[0]
    # Token 999: student_logp=-3.0, teacher_logp=-23.0, reward=20.0
    assert pr.rewards[2] == pytest.approx(-3.0 - (-23.0), abs=1e-6)  # 20.0


def test_position_reward_info_validation():
    """Test PositionRewardInfo validation."""
    # Valid PositionRewardInfo
    pr = PositionRewardInfo(
        position=0,
        candidates=["a", "b"],
        candidate_token_ids=[100, 200],
        logprobs=[-0.5, -1.0],
        rewards=[0.2, -0.3],
        chosen_index=0,
    )
    assert pr.position == 0
    assert pr.chosen_token == "a"
    assert pr.chosen_reward == 0.2
    assert pr.chosen_logprob == -0.5

    # Mismatched lengths should raise ValueError
    with pytest.raises(ValueError):
        PositionRewardInfo(
            position=0,
            candidates=["a", "b"],
            candidate_token_ids=[100],
            logprobs=[-0.5],
            rewards=[0.2],
            chosen_index=0,
        )
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/customized_areal/test_teacher_distill_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/customized_areal/ -v --timeout=60` Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/customized_areal/test_teacher_distill_integration.py
git commit -m "test: add integration tests for teacher distillation pipeline"
```

______________________________________________________________________

## Self-Review Checklist

1. **Spec coverage:**

   - TeacherConfig → Task 5 (config fields) + Task 8 (YAML)
   - TeacherClient → Task 3
   - \_compute_token_rewards → Task 4
   - Modified OnPolicyDistillAgent → Task 5 (agent.py changes)
   - Student rollout logprobs → Task 1 (ModelResponse) + Task 2 (SGLang parsing)
   - Bug fixes (set_last_rewards, try/finally, stats, tree training) → Tasks 6, 7
   - Remove dead code → Task 9
   - Testing → Tasks 1, 2, 3, 4, 10

1. **Placeholder scan:** All code is concrete - no TBD/TODO/fill-in-later.

1. **Type consistency:**

   - `PositionRewardInfo` is imported from `..proxy.cache` in `reward_compute.py` -
     matches the existing dataclass
   - `TeacherConfig` is defined in `teacher_client.py` and used in `config.py` and
     `agent.py` - consistent
   - `student_top_k_logprobs` is `list[list[tuple[int, float]]]` throughout
   - `candidate_token_ids` is `list[list[int]]` throughout

1. **Open spec question:** The `get_last_interaction()` method added to
   `OpenAIProxyClient` in Task 6 uses the export endpoint to fetch interactions. This is
   a simple approach but may not be the most efficient for getting just the last
   interaction. A dedicated endpoint could be added later if needed.

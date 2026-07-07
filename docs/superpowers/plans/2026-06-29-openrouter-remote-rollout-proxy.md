# OpenRouter Remote Rollout Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let AReaL proxy users route `/chat/completions` requests to OpenRouter via a
`remote:<provider/model>` model prefix, while still collecting local-tokenized training
trajectories in the session cache.

**Architecture:** A new `RemoteRolloutClient` in
`areal/experimental/openai/proxy/remote_rollout.py` owns the OpenRouter round-trip,
retokenization, and cache storage. `ProxyRolloutServer.chat_completions()` dispatches by
model prefix: `"default"` stays on the existing local path, `"remote:..."` enters the
remote client, anything else returns 404. Shared prompt-tokenization helpers are
extracted into `areal/experimental/openai/_prompt_utils.py` so both paths use them. A
new `PPOActorConfig.enable_remote_rollout` flag drives a trainer-config warning; the
remote client enforces a first-request safety check via `should_compute_prox_logp()`.

**Tech Stack:** Python 3.12+, FastAPI, `openai.AsyncOpenAI` (for the OpenRouter HTTP
client), HuggingFace `transformers` tokenizers, pytest + httpx ASGI transport for tests.

**Spec:** `docs/superpowers/specs/2026-06-29-openrouter-remote-rollout-proxy-design.md`

______________________________________________________________________

## File Structure

- **Create** `areal/experimental/openai/_prompt_utils.py` — shared helpers extracted
  from `client.py`: `_ensure_message_dict_list`, `_extract_images_from_messages`,
  `_DATA_URI_RE`, and a re-export of `apply_chat_template`. Both local and remote paths
  import from here.
- **Create** `areal/experimental/openai/proxy/remote_rollout.py` — `RemoteRolloutClient`
  class with lazy OpenRouter `AsyncOpenAI` client, `create_completion()` method
  implementing the 10-step per-request lifecycle, finish-reason mapper, and recompute
  gate.
- **Modify** `areal/experimental/openai/client.py` — replace the four helper definitions
  (`_ensure_message_dict_list`, `_DATA_URI_RE`, `_extract_images_from_messages`, and the
  `apply_chat_template` import) with imports from `_prompt_utils`. No behavior change.
- **Modify** `areal/api/cli_args.py` — add `PPOActorConfig.enable_remote_rollout: bool`
  field (after `prox_logp_method`, ~line 1517) and a warning block in
  `PPOActorConfig.__post_init__` (after the SAPO block, before `super().__post_init__()`
  at line 1585).
- **Modify** `areal/experimental/openai/proxy/proxy_rollout_server.py` — add
  `_remote_client` global, construct it in `_setup_openai_client()`, add model-prefix
  dispatch in `chat_completions()`.
- **Create** `tests/experimental/openai/test_remote_rollout.py` — all 23 tests from the
  spec's testing plan.

______________________________________________________________________

## Task 1: Extract shared prompt helpers into `_prompt_utils.py`

**Why first:** The remote client needs `_extract_images_from_messages` and
`_ensure_message_dict_list`. Extracting them first, into a shared module, lets both
`client.py` and `remote_rollout.py` import from one place. This is a pure refactor — no
behavior change — so the existing test suite guards it.

**Files:**

- Create: `areal/experimental/openai/_prompt_utils.py`

- Modify: `areal/experimental/openai/client.py:5-60` (imports) and `client.py:80-237`
  (helper definitions)

- [ ] **Step 1: Create `_prompt_utils.py` with the extracted helpers**

Create `areal/experimental/openai/_prompt_utils.py`:

```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from areal.utils.hf_utils import apply_chat_template

__all__ = [
    "apply_chat_template",
    "_DATA_URI_RE",
    "_ensure_message_dict_list",
    "_extract_images_from_messages",
]


def _ensure_message_dict_list(
    name: str,
    value: list[Any],
) -> list[dict[str, Any]]:
    """Validate that ``value`` is a list of dictionaries or BaseModel objects.

    Args:
        name: Name of the argument being validated (for error messages).
        value: The list provided by the caller.

    Returns:
        A list containing only dictionaries. BaseModel objects are
        converted into their dictionary representation with
        `model_dump(exclude_none=True)`; dictionaries are preserved.

    Raises:
        TypeError: If ``value`` is not a list or an element cannot be converted to a dict.
    """

    if not isinstance(value, list):
        raise TypeError(
            f"{name} must be provided as a list, got {type(value).__name__}"
        )

    def _normalize(item: Any):
        # we should convert BaseModel first, because BaseModel is also Iterable
        if isinstance(item, BaseModel):
            return item.model_dump(exclude_none=True)
        elif isinstance(item, Mapping):
            return {k: _normalize(v) for k, v in item.items() if v is not None}
        elif (
            isinstance(item, Iterable)
            and not isinstance(item, str)
            and not isinstance(item, bytes)
            and not isinstance(item, bytearray)
        ):
            return [_normalize(sub_item) for sub_item in item]
        else:
            return item

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict) or isinstance(item, BaseModel):
            normalized.append(_normalize(item))
        else:
            raise TypeError(
                f"{name}[{index}] must be a dict or a BaseModel; got {type(item).__name__}"
            )
    return normalized


# Regex for data URI: data:image/<subtype>;base64,<data>
_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)


def _extract_images_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract image data from OpenAI-format messages.

    Scans message ``content`` lists for ``image_url`` content parts,
    extracts base64 data (or raw URLs), and converts messages to a
    HuggingFace-compatible format for ``apply_chat_template``.

    Args:
        messages: Normalized list of message dicts (OpenAI format).

    Returns:
        A 3-tuple of:

        - **image_data** – list of base64 image strings (no data-URI prefix)
          or raw URL strings for each image found.
        - **messages_for_tokenizer** – deep copy of *messages* where every
          ``{"type": "image_url", ...}`` part is replaced by
          ``{"type": "image"}`` so that HuggingFace VLM tokenizers insert
          the correct image-placeholder tokens.
        - **vision_messages_for_vllm** – deep copy of *messages* where
          ``image_url`` parts retain the ``image_url`` key but the ``url``
          value is replaced with a placeholder (the actual base64 data URI
          is injected later by the vLLM backend from *image_data*).
    """
    image_data: list[str] = []
    messages_for_tokenizer: list[dict[str, Any]] = []
    vision_messages_for_vllm: list[dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            messages_for_tokenizer.append(deepcopy(msg))
            vision_messages_for_vllm.append(deepcopy(msg))
            continue

        tok_parts: list[dict[str, Any]] = []
        vllm_parts: list[dict[str, Any]] = []

        for part in content:
            if not isinstance(part, dict):
                tok_parts.append(part)
                vllm_parts.append(deepcopy(part))
                continue

            if part.get("type") == "image_url":
                image_url_obj = part.get("image_url", {})
                url = (
                    image_url_obj.get("url", "")
                    if isinstance(image_url_obj, dict)
                    else ""
                )

                if not url:
                    raise ValueError(
                        "image_url content part has an empty or missing URL. "
                        "Provide a valid data URI or HTTP(S) URL in "
                        "image_url.url."
                    )

                # Extract base64 payload from data URIs; keep raw URLs as-is.
                m = _DATA_URI_RE.match(url)
                if m:
                    image_data.append(m.group(1))
                else:
                    image_data.append(url)

                tok_parts.append({"type": "image"})

                # vLLM backend injects actual data URI from req.image_data.
                vllm_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "placeholder"},
                    }
                )
            else:
                tok_parts.append(deepcopy(part))
                vllm_parts.append(deepcopy(part))

        tok_msg = {**msg, "content": tok_parts}
        vllm_msg = {**msg, "content": vllm_parts}
        messages_for_tokenizer.append(tok_msg)
        vision_messages_for_vllm.append(vllm_msg)

    return image_data, messages_for_tokenizer, vision_messages_for_vllm
```

- [ ] **Step 2: Update `client.py` imports to pull from `_prompt_utils`**

In `areal/experimental/openai/client.py`, find the import block (lines 1-61). The file
currently imports `apply_chat_template` from `areal.utils.hf_utils` at line 60:

```python
from areal.utils.hf_utils import apply_chat_template
```

Replace that single line with:

```python
from areal.experimental.openai._prompt_utils import (
    apply_chat_template,
    _DATA_URI_RE,
    _ensure_message_dict_list,
    _extract_images_from_messages,
)
```

Also remove now-redundant imports that `_prompt_utils` owns. The file imports `re` at
line 6, `Iterable`/`Mapping` at line 8, `deepcopy` at line 9, and `BaseModel` at line 52
— these are still used elsewhere in `client.py` (e.g. `_find_kth`,
`_convert_tool_output_format`, `_build_messages_list`), so **leave them**. Do not remove
any import that is still referenced.

- [ ] **Step 3: Delete the moved helper definitions from `client.py`**

Delete these three blocks from `areal/experimental/openai/client.py`:

1. The `_ensure_message_dict_list` function (lines 80-128, the block starting with
   `def _ensure_message_dict_list(` and ending at the `return normalized` line).
1. The `_DATA_URI_RE` assignment (line 148: `_DATA_URI_RE = re.compile(...)`).
1. The `_extract_images_from_messages` function (lines 151-237, the block starting with
   `def _extract_images_from_messages(` and ending at
   `return image_data, messages_for_tokenizer, vision_messages_for_vllm`).

After deletion, `client.py` should no longer define these names — it imports them from
`_prompt_utils`. The functions `_find_kth`, `_convert_tool_output_format`,
`_build_messages_list`, and `concat_prompt_token_ids_with_parent` remain in `client.py`
(they are local-generation-specific).

- [ ] **Step 4: Run the existing test suite to confirm no behavior change**

Run:
`uv run pytest tests/experimental/openai/test_concat_prompt.py tests/experimental/openai/test_client.py tests/experimental/openai/test_streaming_chat_completions.py -x -q 2>&1 | tail -20`

Expected: Tests that don't require a GPU/sglang server pass. The `test_client.py` tests
are marked `pytest.mark.sglang` and may skip; that is fine. `test_concat_prompt.py` and
`test_streaming_chat_completions.py` should pass (the latter mocks the client, the
former uses a real tokenizer but no server).

If `test_concat_prompt.py` or `test_streaming_chat_completions.py` fail with an
`ImportError` or `NameError` for `_extract_images_from_messages` /
`_ensure_message_dict_list`, the extraction missed a reference — re-read `client.py` to
find any remaining direct definition or import that conflicts.

- [ ] **Step 5: Commit**

```bash
git add areal/experimental/openai/_prompt_utils.py areal/experimental/openai/client.py
git commit -m "$(cat <<'EOF'
refactor: extract shared prompt helpers into _prompt_utils

Move _ensure_message_dict_list, _DATA_URI_RE, and
_extract_images_from_messages out of client.py into a shared
_prompt_utils module so the upcoming RemoteRolloutClient can reuse
them. client.py re-imports from _prompt_utils — no behavior change.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

______________________________________________________________________

## Task 2: Add `enable_remote_rollout` flag and trainer-config warning

**Why second:** The recompute-gate depends on `should_compute_prox_logp()`, which
already exists. Adding the config flag + warning is a small, isolated change that the
later remote-client tests will exercise. Doing it before the client exists lets us write
the config-warning tests (spec tests 18-20) against a stable API.

**Files:**

- Modify: `areal/api/cli_args.py:1517` (add field after `prox_logp_method`) and
  `areal/api/cli_args.py:1572-1585` (add warning in `__post_init__`)

- [ ] **Step 1: Write the failing tests for the config warning**

Create `tests/experimental/openai/test_remote_rollout.py` with the config-warning tests
(spec tests 18-20). The class is `PPOActorConfig` (not `ActorConfig` — the spec used
shorthand):

```python
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenRouter remote rollout proxy.

Spec: docs/superpowers/specs/2026-06-29-openrouter-remote-rollout-proxy-design.md
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from areal.api.cli_args import PPOActorConfig
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.types import InteractionWithTokenLogpReward


# ---------------------------------------------------------------------------
# Tests: PPOActorConfig enable_remote_rollout warning (spec tests 18-20)
# ---------------------------------------------------------------------------


class TestEnableRemoteRolloutWarning:
    def test_warns_when_recompute_disabled(self, caplog):
        """enable_remote_rollout=True with no recompute path → warning."""
        with caplog.at_level(logging.WARNING, logger="CLIArgs"):
            PPOActorConfig(
                enable_remote_rollout=True,
                recompute_logprob=False,
                use_decoupled_loss=False,
            )
        assert any(
            "enable_remote_rollout" in rec.message and "recompute" in rec.message
            for rec in caplog.records
        ), f"expected recompute warning, got: {[r.message for r in caplog.records]}"

    def test_no_warning_when_recompute_logprob_true(self, caplog):
        """enable_remote_rollout=True + recompute_logprob=True → no warning."""
        with caplog.at_level(logging.WARNING, logger="CLIArgs"):
            PPOActorConfig(
                enable_remote_rollout=True,
                recompute_logprob=True,
                use_decoupled_loss=False,
            )
        assert not any(
            "enable_remote_rollout" in rec.message for rec in caplog.records
        ), f"unexpected warning: {[r.message for r in caplog.records]}"

    def test_no_warning_when_decoupled_loss_true(self, caplog):
        """enable_remote_rollout=True + use_decoupled_loss=True → no warning."""
        with caplog.at_level(logging.WARNING, logger="CLIArgs"):
            PPOActorConfig(
                enable_remote_rollout=True,
                recompute_logprob=False,
                use_decoupled_loss=True,
            )
        assert not any(
            "enable_remote_rollout" in rec.message for rec in caplog.records
        ), f"unexpected warning: {[r.message for r in caplog.records]}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestEnableRemoteRolloutWarning -x -q 2>&1 | tail -15`

Expected: FAIL with
`TypeError: __init__() got an unexpected keyword argument 'enable_remote_rollout'` (the
field does not exist yet).

- [ ] **Step 3: Add the `enable_remote_rollout` field to `PPOActorConfig`**

In `areal/api/cli_args.py`, find the `prox_logp_method` field (ends at line 1517 with
the closing `)`). Immediately after it (before the blank line and the
`# Logging Agent Trajectories` comment at line 1519), insert:

```python
    enable_remote_rollout: bool = field(
        default=False,
        metadata={
            "help": "Enable OpenRouter remote rollout via model='remote:<provider/model>'. "
            "Requires actor.recompute_logprob=true or actor.use_decoupled_loss=true, "
            "because remote completions carry placeholder logprobs that must be recomputed."
        },
    )
```

- [ ] **Step 4: Add the warning to `PPOActorConfig.__post_init__`**

In `areal/api/cli_args.py`, find the SAPO validation block in `__post_init__` (lines
1572-1583, ending with the `use_decoupled_loss` SAPO check). Immediately after that
block and before `super().__post_init__()` (line 1585), insert:

```python
        # Warn if remote rollout is enabled but no recompute path is active.
        # Remote completions carry placeholder logprobs that would corrupt
        # PPO ratios without recompute; see the OpenRouter remote rollout
        # design spec for the full rationale.
        if self.enable_remote_rollout and not self.should_compute_prox_logp():
            logger.warning(
                "enable_remote_rollout=True but neither recompute_logprob nor "
                "use_decoupled_loss is active. Remote completions carry placeholder "
                "logprobs; set actor.recompute_logprob=true or "
                "actor.use_decoupled_loss=true to avoid corrupting PPO ratios."
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestEnableRemoteRolloutWarning -x -q 2>&1 | tail -15`

Expected: PASS (3 tests).

If the `test_warns_when_recompute_disabled` test still fails, check that
`should_compute_prox_logp()` returns False for that config — it depends on
`prox_logp_method` (default `PROX_LOGP_METHOD_RECOMPUTE`) and the
`use_decoupled_loss`/`recompute_logprob` flags. See `areal/api/cli_args.py:1534-1545`.

- [ ] **Step 6: Commit**

```bash
git add areal/api/cli_args.py tests/experimental/openai/test_remote_rollout.py
git commit -m "$(cat <<'EOF'
feat: add enable_remote_rollout flag with recompute warning

PPOActorConfig.enable_remote_rollout (default False) opts in to
OpenRouter remote rollout. When True and no recompute path is active
(recompute_logprob nor use_decoupled_loss), __post_init__ warns that
placeholder remote logprobs would corrupt PPO ratios.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

______________________________________________________________________

## Task 3: Implement `RemoteRolloutClient` core — finish-reason mapper and recompute gate

**Why:** The finish-reason mapper (spec resolved question 3) and the recompute gate
(spec resolved question 1, layer 2) are pure, testable units that don't need a tokenizer
or network. Build and test them first, then layer the network + tokenization logic on
top in Task 4.

**Files:**

- Create: `areal/experimental/openai/proxy/remote_rollout.py`

- Test: `tests/experimental/openai/test_remote_rollout.py` (append)

- [ ] **Step 1: Write the failing tests for the finish-reason mapper and recompute
  gate**

Append to `tests/experimental/openai/test_remote_rollout.py`:

```python
# ---------------------------------------------------------------------------
# Tests: finish-reason mapping (spec test 14)
# ---------------------------------------------------------------------------


class TestMapFinishReason:
    def test_length(self):
        from areal.experimental.openai.proxy.remote_rollout import (
            _map_finish_reason,
        )

        assert _map_finish_reason("length") == "length"

    def test_tool_calls(self):
        from areal.experimental.openai.proxy.remote_rollout import (
            _map_finish_reason,
        )

        assert _map_finish_reason("tool_calls") == "tool_calls"

    def test_stop(self):
        from areal.experimental.openai.proxy.remote_rollout import (
            _map_finish_reason,
        )

        assert _map_finish_reason("stop") == "stop"

    def test_content_filter(self):
        from areal.experimental.openai.proxy.remote_rollout import (
            _map_finish_reason,
        )

        assert _map_finish_reason("content_filter") == "stop"

    def test_none(self):
        from areal.experimental.openai.proxy.remote_rollout import (
            _map_finish_reason,
        )

        assert _map_finish_reason(None) == "stop"

    def test_unknown(self):
        from areal.experimental.openai.proxy.remote_rollout import (
            _map_finish_reason,
        )

        assert _map_finish_reason("some-new-future-reason") == "stop"


# ---------------------------------------------------------------------------
# Tests: recompute gate (spec tests 21-22)
# ---------------------------------------------------------------------------


def _make_remote_client(recompute_enabled: bool):
    """Construct a RemoteRolloutClient with stubbed tokenizer."""
    from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    return RemoteRolloutClient(
        tokenizer=tokenizer,
        chat_template_type="hf",
        recompute_enabled=recompute_enabled,
    )


class TestRecomputeGate:
    @pytest.mark.asyncio
    async def test_first_request_fails_when_recompute_disabled(self, monkeypatch):
        """recompute_enabled=False → first remote request raises 500; gate stays closed."""
        from fastapi import HTTPException

        client = _make_remote_client(recompute_enabled=False)
        cache = InteractionCache()
        request = {"model": "remote:openai/gpt-4o-mini", "messages": []}

        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)

        assert exc_info.value.status_code == 500
        assert "recompute_logprob" in exc_info.value.detail
        # Gate stays closed: a second call also fails.
        with pytest.raises(HTTPException) as exc_info_2:
            await client.create_completion(request, cache)
        assert exc_info_2.value.status_code == 500

    @pytest.mark.asyncio
    async def test_gate_opens_after_first_success(self, monkeypatch):
        """recompute_enabled=True → first request succeeds; gate flips; later requests skip check."""
        client = _make_remote_client(recompute_enabled=True)
        assert client._recompute_verified is False

        # Stub the OpenRouter call to return a minimal completion.
        fake_completion = ChatCompletion(
            id="chatcmpl-remote-1",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                )
            ],
            created=0,
            model="openai/gpt-4o-mini",
            object="chat.completion",
        )
        monkeypatch.setattr(
            client, "_call_openrouter", lambda *a, **kw: _async_return(fake_completion)
        )
        # Stub prompt tokenization to avoid needing a real tokenizer.
        monkeypatch.setattr(
            "areal.experimental.openai.proxy.remote_rollout.apply_chat_template",
            lambda *a, **kw: [1, 2, 3],
        )
        monkeypatch.setattr(client.tokenizer, "encode", lambda *a, **kw: [10, 20])

        cache = InteractionCache()
        request = {"model": "remote:openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        result = await client.create_completion(request, cache)

        assert result.id == "chatcmpl-remote-1"
        assert client._recompute_verified is True


async def _async_return(value):
    """Helper: make a sync lambda usable as an awaitable."""
    return value
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestMapFinishReason tests/experimental/openai/test_remote_rollout.py::TestRecomputeGate -x -q 2>&1 | tail -15`

Expected: FAIL with `ImportError: cannot import name '_map_finish_reason'` /
`RemoteRolloutClient` from `remote_rollout`.

- [ ] **Step 3: Create `remote_rollout.py` with the mapper, gate, and class skeleton**

Create `areal/experimental/openai/proxy/remote_rollout.py`:

```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import ModelResponse
from areal.experimental.openai._prompt_utils import (
    apply_chat_template,
    _ensure_message_dict_list,
    _extract_images_from_messages,
)
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger("RemoteRollout")

_REMOTE_PREFIX = "remote:"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _map_finish_reason(finish_reason: str | None) -> str:
    """Map an OpenRouter/OpenAI finish_reason to a ModelResponse stop_reason.

    See spec "Resolved Implementation Questions > 3. Finish-reason mapping".
    Error completions are never cached (raised as HTTP 502/504 before this
    function is called), so "abort" is never produced here.
    """
    if finish_reason == "length":
        return "length"
    if finish_reason == "tool_calls":
        return "tool_calls"
    # "stop", "content_filter", None, and any future reason collapse to "stop".
    return "stop"


class RemoteRolloutClient:
    """OpenRouter-backed remote rollout client.

    Calls OpenRouter with the stripped model name, retokenizes the remote
    assistant output with the local tokenizer, builds a local ModelResponse
    with placeholder logprobs, and stores an InteractionWithTokenLogpReward
    in the session cache.

    Privacy: prompts, tool schemas, tool outputs, and conversation context
    are sent to OpenRouter and the selected upstream provider. Do not use
    remote rollout with data that must stay local.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        chat_template_type: str,
        engine_max_tokens: int | None = None,
        recompute_enabled: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template_type = chat_template_type
        self.engine_max_tokens = engine_max_tokens
        self.recompute_enabled = recompute_enabled
        self._recompute_verified: bool = False
        self._openrouter_client: AsyncOpenAI | None = None

    def _get_openrouter_client(self) -> AsyncOpenAI:
        """Lazily construct the OpenRouter AsyncOpenAI client from env vars.

        Raises HTTPException(500) if OPENROUTER_API_KEY is missing.
        """
        if self._openrouter_client is not None:
            return self._openrouter_client
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY is not set; cannot call OpenRouter for remote rollout.",
            )
        base_url = os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL)
        self._openrouter_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(
            "Constructed OpenRouter client (base_url=%s). "
            "Remote rollout sends prompts and conversation context to OpenRouter.",
            base_url,
        )
        return self._openrouter_client

    async def _call_openrouter(
        self, provider_model: str, forwarded: dict[str, Any]
    ) -> ChatCompletion:
        """Call OpenRouter and return the non-streaming ChatCompletion.

        Wrapped separately so tests can monkeypatch this method.
        """
        client = self._get_openrouter_client()
        try:
            return await client.chat.completions.create(
                model=provider_model,
                stream=False,
                n=1,
                **forwarded,
            )
        except HTTPException:
            raise
        except Exception as e:
            # Distinguish HTTP errors from network errors for the status code.
            from openai import APIStatusError, APIError

            if isinstance(e, APIStatusError):
                raise HTTPException(
                    status_code=502,
                    detail=f"OpenRouter upstream error: {e.status_code}: {e}",
                )
            if isinstance(e, APIError):
                raise HTTPException(
                    status_code=504,
                    detail=f"OpenRouter network error: {type(e).__name__}: {e}",
                )
            raise HTTPException(
                status_code=504,
                detail=f"OpenRouter request failed: {type(e).__name__}: {e}",
            )

    async def create_completion(
        self,
        request: dict[str, Any],
        session_cache: InteractionCache,
    ) -> ChatCompletion:
        """Process a remote rollout request through OpenRouter.

        Implements the 10-step per-request lifecycle from the design spec.
        """
        raise NotImplementedError("Implemented in Task 4")
```

- [ ] **Step 4: Run the finish-reason mapper tests to verify they pass**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestMapFinishReason -x -q 2>&1 | tail -15`

Expected: PASS (6 tests).

- [ ] **Step 5: Run the recompute gate tests**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestRecomputeGate -x -q 2>&1 | tail -20`

Expected: Both tests FAIL — `test_first_request_fails_when_recompute_disabled` fails
because `create_completion` raises `NotImplementedError` instead of the 500
HTTPException. `test_gate_opens_after_success` fails the same way. This is expected —
Task 4 implements `create_completion`.

- [ ] **Step 6: Commit (mapper and skeleton only; gate tests land red)**

```bash
git add areal/experimental/openai/proxy/remote_rollout.py tests/experimental/openai/test_remote_rollout.py
git commit -m "$(cat <<'EOF'
feat: add RemoteRolloutClient skeleton with finish-reason mapper

_map_finish_reason maps OpenRouter finish_reason to ModelResponse
stop_reason (length/tool_calls preserved; everything else collapses
to stop; abort is never produced since errors raise before mapping).

RemoteRolloutClient owns the lazy OpenRouter AsyncOpenAI client,
the _call_openrouter wrapper (502 for APIStatusError, 504 for
network errors), and the recompute gate state. create_completion
is a stub pending Task 4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

______________________________________________________________________

## Task 4: Implement `RemoteRolloutClient.create_completion` — the full 10-step lifecycle

**Why:** This is the heart of the feature. It wires the validation, prompt tokenization,
OpenRouter call, cache insertion, output tokenization, finish-reason mapping,
ModelResponse construction, and interaction storage into one method. The cleanup
contract (delete cache entry on any failure after step 5) is enforced with a try/except.

**Files:**

- Modify: `areal/experimental/openai/proxy/remote_rollout.py` (replace the
  `NotImplementedError` stub)

- Test: `tests/experimental/openai/test_remote_rollout.py` (append happy-path and
  failure tests)

- [ ] **Step 1: Write the failing happy-path and failure tests**

Append to `tests/experimental/openai/test_remote_rollout.py`. These tests use a real
tokenizer (Qwen3-0.6B) following the `test_concat_prompt.py` pattern, and stub the
OpenRouter call via monkeypatch on `_call_openrouter`.

```python
# ---------------------------------------------------------------------------
# Real tokenizer fixture (same pattern as test_concat_prompt.py)
# ---------------------------------------------------------------------------

from tests.utils import get_model_path
from areal.utils.hf_utils import load_hf_tokenizer

_MODEL_PATH = "Qwen/Qwen3-0.6B"
_LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B"


@pytest.fixture(scope="module")
def real_tokenizer():
    return load_hf_tokenizer(get_model_path(_LOCAL_MODEL_PATH, _MODEL_PATH))


def _make_chat_completion(
    completion_id: str = "chatcmpl-remote-1",
    content: str = "Hello there.",
    finish_reason: str = "stop",
    tool_calls=None,
) -> ChatCompletion:
    """Build a minimal ChatCompletion as OpenRouter would return."""
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return ChatCompletion(
        id=completion_id,
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=message,
            )
        ],
        created=0,
        model="openai/gpt-4o-mini",
        object="chat.completion",
    )


# ---------------------------------------------------------------------------
# Tests: remote client happy path and failures (spec tests 8-13)
# ---------------------------------------------------------------------------


class TestCreateCompletionHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_stores_interaction(self, real_tokenizer, monkeypatch):
        """OpenRouter returns a completion → InteractionWithTokenLogpReward stored correctly."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer,
            chat_template_type="hf",
            recompute_enabled=True,
        )
        fake_completion = _make_chat_completion(
            completion_id="chatcmpl-remote-xyz",
            content="The answer is 42.",
            finish_reason="stop",
        )
        monkeypatch.setattr(
            client,
            "_call_openrouter",
            lambda *a, **kw: _async_return(fake_completion),
        )

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the answer?"}],
        }
        result = await client.create_completion(request, cache)

        # Returned completion is the remote one, unmodified.
        assert result is fake_completion
        # Cache contains the interaction under the remote ID.
        assert "chatcmpl-remote-xyz" in cache
        interaction = cache["chatcmpl-remote-xyz"]
        assert isinstance(interaction, InteractionWithTokenLogpReward)
        assert interaction.completion is fake_completion
        # ModelResponse built with local tokenization.
        assert interaction.model_response is not None
        mr = interaction.model_response
        assert len(mr.input_tokens) > 0
        # Output tokens = local encoding of "The answer is 42." + EOS.
        expected_output = real_tokenizer.encode(
            "The answer is 42.", add_special_tokens=False
        ) + [real_tokenizer.eos_token_id]
        assert mr.output_tokens == expected_output
        # Placeholder logprobs and versions.
        assert mr.output_logprobs == [0.0] * len(mr.output_tokens)
        assert mr.output_versions == [-1] * len(mr.output_tokens)
        assert mr.stop_reason == "stop"
        # Output message list preserved from remote.
        assert interaction.output_message_list == [
            fake_completion.choices[0].message.model_dump(exclude_none=True)
        ]
        # Gate flipped.
        assert client._recompute_verified is True


class TestCreateCompletionFailures:
    @pytest.mark.asyncio
    async def test_empty_remote_output_returns_400(self, real_tokenizer, monkeypatch):
        """Empty content and no tool_calls → 400, no cache entry."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )
        fake_completion = _make_chat_completion(content="", finish_reason="stop")
        monkeypatch.setattr(
            client, "_call_openrouter", lambda *a, **kw: _async_return(fake_completion)
        )
        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 400
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_openrouter_4xx_returns_502_no_cache(self, real_tokenizer, monkeypatch):
        """OpenRouter APIStatusError → 502, no cache entry."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient
        from openai import APIStatusError

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )

        async def _raise(*a, **kw):
            raise APIStatusError(
                message="upstream 429",
                response=None,
                body=None,
            )

        monkeypatch.setattr(client, "_call_openrouter", _raise)
        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 502
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_openrouter_network_error_returns_504_no_cache(
        self, real_tokenizer, monkeypatch
    ):
        """OpenRouter APIError (network) → 504, no cache entry."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient
        from openai import APIConnectionError

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )

        async def _raise(*a, **kw):
            raise APIConnectionError(request=None)

        monkeypatch.setattr(client, "_call_openrouter", _raise)
        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 504
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_output_tokenization_failure_removes_cache_entry(
        self, real_tokenizer, monkeypatch
    ):
        """tokenizer.encode raising after cache insert → 500, cache entry removed."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )
        fake_completion = _make_chat_completion(content="some output", finish_reason="stop")
        monkeypatch.setattr(
            client, "_call_openrouter", lambda *a, **kw: _async_return(fake_completion)
        )
        # Make encode raise to simulate tokenization failure after cache insert.
        call_count = {"n": 0}

        def _flaky_encode(text, add_special_tokens=False):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("simulated encode failure")
            return [1, 2, 3]

        monkeypatch.setattr(real_tokenizer, "encode", _flaky_encode)

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 500
        # Cache entry was inserted at step 5 then removed on failure.
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_500_before_network(
        self, real_tokenizer, monkeypatch
    ):
        """No OPENROUTER_API_KEY → 500 before any network call."""
        from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

        client = RemoteRolloutClient(
            tokenizer=real_tokenizer, chat_template_type="hf", recompute_enabled=True
        )
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Reset the lazy client so _get_openrouter_client re-reads env.
        client._openrouter_client = None

        cache = InteractionCache()
        request = {
            "model": "remote:openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion(request, cache)
        assert exc_info.value.status_code == 500
        assert "OPENROUTER_API_KEY" in exc_info.value.detail
        assert len(cache) == 0


# Re-import HTTPException at module level for the failure tests above.
from fastapi import HTTPException  # noqa: E402
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestCreateCompletionHappyPath tests/experimental/openai/test_remote_rollout.py::TestCreateCompletionFailures -x -q 2>&1 | tail -20`

Expected: FAIL with `NotImplementedError: Implemented in Task 4` (the stub from Task 3).

- [ ] **Step 3: Implement `create_completion` — replace the stub**

In `areal/experimental/openai/proxy/remote_rollout.py`, replace the `create_completion`
method body (the `raise NotImplementedError("Implemented in Task 4")` line and its
docstring) with the full implementation:

```python
    async def create_completion(
        self,
        request: dict[str, Any],
        session_cache: InteractionCache,
    ) -> ChatCompletion:
        """Process a remote rollout request through OpenRouter.

        Implements the 10-step per-request lifecycle from the design spec.
        The cache entry is created at step 5 (after the OpenRouter call
        succeeds and the completion ID is known). Every failure path
        from step 5 onward deletes the cache entry before raising.
        """
        model = request.get("model", "default")

        # --- Step 1: Parse & validate ---
        provider_model = model.removeprefix(_REMOTE_PREFIX)
        if not provider_model:
            raise HTTPException(
                status_code=400, detail="remote: model name is empty"
            )
        if self.chat_template_type != "hf":
            raise HTTPException(
                status_code=400,
                detail="remote rollout MVP supports chat_template_type='hf' only",
            )
        if request.get("stream") is True:
            raise HTTPException(
                status_code=400, detail="remote streaming not supported in MVP"
            )
        if request.get("n", 1) != 1:
            raise HTTPException(
                status_code=400, detail="remote n != 1 not supported"
            )
        if not self.recompute_enabled and not self._recompute_verified:
            raise HTTPException(
                status_code=500,
                detail="remote rollout requires actor.recompute_logprob=true "
                "or actor.use_decoupled_loss=true (placeholder remote logprobs "
                "would corrupt PPO ratios without recompute)",
            )

        messages_list = _ensure_message_dict_list(
            "messages", list(request.get("messages", []))
        )
        if not messages_list:
            raise HTTPException(
                status_code=400, detail="messages cannot be empty"
            )

        # --- Step 2: Tokenize prompt locally (hf mode) ---
        tools_list = None
        if not _is_omitted(request.get("tools")):
            tools_list = list(request["tools"])
        extra_body = request.get("extra_body") or {}
        chat_template_kwargs = extra_body.get("chat_template_kwargs", {})

        image_data, messages_for_tokenizer, _ = _extract_images_from_messages(
            messages_list
        )
        tokenizer_messages = messages_for_tokenizer if image_data else messages_list
        try:
            prompt_token_ids = apply_chat_template(
                self.tokenizer,
                tokenizer_messages,
                tools=tools_list,
                add_generation_prompt=True,
                tokenize=True,
                **chat_template_kwargs,
            )
        except Exception as e:
            # Failure before cache insert — no cleanup needed.
            raise HTTPException(
                status_code=500,
                detail=f"prompt tokenization failed: {type(e).__name__}: {e}",
            )

        # --- Step 3: Call OpenRouter ---
        forwarded = _build_forwarded_params(request)
        remote_completion = await self._call_openrouter(provider_model, forwarded)

        # --- Step 4: Extract assistant output & determine cache key ---
        choice = remote_completion.choices[0]
        output_text = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        completion_id = remote_completion.id
        if not completion_id or completion_id in session_cache:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        if not output_text and not tool_calls:
            raise HTTPException(
                status_code=400,
                detail="remote returned empty completion (no content and no tool_calls)",
            )

        # --- Step 5: Insert interaction into cache ---
        interaction = InteractionWithTokenLogpReward(
            messages=deepcopy(messages_list),
            chat_template_type="hf",
        )
        session_cache[completion_id] = interaction

        try:
            # --- Step 6: Tokenize output locally ---
            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            output_tokens = output_tokens + [self.tokenizer.eos_token_id]
            if not output_tokens:
                raise ValueError("output tokenization produced no tokens")

            # --- Step 7: Map finish reason ---
            stop_reason = _map_finish_reason(choice.finish_reason)

            # --- Step 8: Build local ModelResponse ---
            model_response = ModelResponse(
                input_tokens=prompt_token_ids,
                output_tokens=output_tokens,
                output_logprobs=[0.0] * len(output_tokens),
                output_versions=[-1] * len(output_tokens),
                stop_reason=stop_reason,
                tokenizer=self.tokenizer,
            )

            # --- Step 9: Store interaction fields ---
            interaction.completion = remote_completion
            interaction.model_response = model_response
            interaction.output_message_list = [
                choice.message.model_dump(exclude_none=True)
            ]
        except HTTPException:
            del session_cache[completion_id]
            raise
        except Exception as e:
            del session_cache[completion_id]
            raise HTTPException(
                status_code=500,
                detail=f"remote rollout post-processing failed: {type(e).__name__}: {e}",
            )

        # --- Step 10: Flip the recompute gate and return ---
        self._recompute_verified = True
        return remote_completion


def _is_omitted(value: Any) -> bool:
    """True if value is None, NOT_GIVEN, or an Omit sentinel."""
    if value is None:
        return True
    try:
        from openai import NOT_GIVEN, Omit

        if value is NOT_GIVEN or isinstance(value, Omit):
            return True
    except ImportError:
        pass
    if hasattr(value, "__class__"):
        return value.__class__.__name__ in ("NotGiven", "Omit")
    return False


def _build_forwarded_params(request: dict[str, Any]) -> dict[str, Any]:
    """Build the kwargs dict to forward to OpenRouter.

    Forwards sampling/tool params. Strips model, stream, n, areal_cache, store
    (model is passed explicitly; the others are not OpenRouter params).
    """
    _STRIPPED = {"model", "stream", "n", "areal_cache", "store", "messages", "tools", "extra_body"}
    _FORWARDABLE = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "tool_choice",
        "frequency_penalty",
        "metadata",
        "tools",
    }
    forwarded: dict[str, Any] = {}
    for key, value in request.items():
        if key in _STRIPPED:
            continue
        if key not in _FORWARDABLE:
            continue
        if _is_omitted(value):
            continue
        forwarded[key] = value
    # tools is forwarded separately (re-added after the STRIPPED check above
    # because it appears in both sets — we want it forwarded, not stripped).
    if not _is_omitted(request.get("tools")):
        forwarded["tools"] = list(request["tools"])
    return forwarded
```

Note: `tools` appears in both `_STRIPPED` and `_FORWARDABLE` — this is intentional. The
`_STRIPPED` set prevents it from being picked up by the generic loop (where it would be
confused with the request-level `tools` field handling), and the explicit re-add at the
end forwards it cleanly. If this feels brittle, an alternative is to remove `"tools"`
from `_STRIPPED` and let the generic loop handle it — but then `messages` (also in
`_STRIPPED`) would need the same treatment, and `messages` must never be forwarded (it
is rebuilt from `messages_list`). Keep the explicit re-add for `tools` only.

- [ ] **Step 4: Run the happy-path test**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestCreateCompletionHappyPath -x -q 2>&1 | tail -20`

Expected: PASS (1 test). If it fails with a tokenizer download error, the test
environment lacks network — the `get_model_path` helper should fall back to the local
path; check `/storage/openpsi/models/Qwen__Qwen3-0.6B` exists or that HuggingFace
download works.

- [ ] **Step 5: Run the failure tests**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestCreateCompletionFailures -x -q 2>&1 | tail -20`

Expected: PASS (5 tests).

If `test_output_tokenization_failure_removes_cache_entry` fails because the cache is not
empty, the `del session_cache[completion_id]` in the except block is not running — check
that the failure path catches the right exception type. The flaky encode mock raises
`RuntimeError`, which is caught by the `except Exception` block (not
`except HTTPException`), so the cleanup should run.

- [ ] **Step 6: Run the recompute gate tests from Task 3 (they should now pass)**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestRecomputeGate -x -q 2>&1 | tail -20`

Expected: PASS (2 tests). The `test_gate_opens_after_first_success` test stubs
`_call_openrouter`, `apply_chat_template`, and `tokenizer.encode` — all three are now
called by the real `create_completion`, so the stubs must align. If
`apply_chat_template` stub returns `[1, 2, 3]` but the real code expects it to be called
with keyword args, adjust the lambda signature.

- [ ] **Step 7: Commit**

```bash
git add areal/experimental/openai/proxy/remote_rollout.py tests/experimental/openai/test_remote_rollout.py
git commit -m "$(cat <<'EOF'
feat: implement RemoteRolloutClient.create_completion lifecycle

Implements the 10-step per-request lifecycle: validate model prefix
and MVP constraints, tokenize prompt locally, call OpenRouter, extract
output and determine cache key, insert interaction, tokenize output
with appended EOS, map finish reason, build ModelResponse with
placeholder logprobs and [-1] versions, store interaction fields,
flip the recompute gate.

Cache entry is created at step 5 (after the OpenRouter call) so the
remote completion ID is known. Every failure path from step 5 onward
deletes the cache entry before raising.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

______________________________________________________________________

## Task 5: Wire dispatch in `ProxyRolloutServer.chat_completions`

**Why:** With `RemoteRolloutClient` complete and tested, wire it into the proxy server.
This is the integration point — the model-prefix dispatch and the `_remote_client`
global construction in `_setup_openai_client()`.

**Files:**

- Modify: `areal/experimental/openai/proxy/proxy_rollout_server.py:32` (import), `:96`
  (global), `:262-282` (`_setup_openai_client`), `:605-667` (`chat_completions`)

- Test: `tests/experimental/openai/test_remote_rollout.py` (append routing tests)

- [ ] **Step 1: Write the failing routing tests (spec tests 1-7)**

Append to `tests/experimental/openai/test_remote_rollout.py`. These follow the
`test_streaming_chat_completions.py` pattern — httpx ASGI transport, monkeypatch the
server globals.

```python
# ---------------------------------------------------------------------------
# Tests: routing dispatch in chat_completions (spec tests 1-7)
# ---------------------------------------------------------------------------

httpx = pytest.importorskip("httpx")

_ADMIN_KEY = "test-admin-key-routing"


@pytest.fixture
def _routing_env(monkeypatch):
    """Reset server globals and install a fake session for routing tests."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)
    # Stub _openai_client so chat_completions doesn't 500 on "not initialized".
    mock_local = MagicMock()
    mock_local.chat.completions.create = _fake_local_create
    monkeypatch.setattr(srv, "_openai_client", mock_local)
    # Stub _remote_client so routing tests don't need a real tokenizer.
    mock_remote = MagicMock()
    mock_remote.create_completion = _fake_remote_create
    monkeypatch.setattr(srv, "_remote_client", mock_remote)


_transport_routing = None


def _routing_client():
    global _transport_routing
    if _transport_routing is None:
        _transport_routing = httpx.ASGITransport(app=srv.app)
    return httpx.AsyncClient(transport=_transport_routing, base_url="http://testserver")


def _start_session_for_routing(monkeypatch):
    """Start a session and return (session_id, api_key)."""
    monkeypatch.setattr(srv, "_capacity", 1)
    import asyncio

    async def _start():
        async with _routing_client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "routing-test"},
            )
            return resp.json()["session_id"], resp.json()["api_key"]

    return asyncio.get_event_loop().run_until_complete(_start())


async def _fake_local_create(*, messages=None, stream=None, areal_cache=None, **kwargs):
    return _make_chat_completion(
        completion_id="chatcmpl-local",
        content="local response",
        finish_reason="stop",
    )


async def _fake_remote_create(request, session_cache):
    return _make_chat_completion(
        completion_id="chatcmpl-remote-routed",
        content="remote response",
        finish_reason="stop",
    )


class TestRoutingDispatch:
    @pytest.mark.asyncio
    async def test_default_model_calls_local_client(self, _routing_env, monkeypatch):
        """model='default' → local client, never remote."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]

            # Spy on which client got called.
            local_called = {"v": False}
            remote_called = {"v": False}

            original_local = srv._openai_client.chat.completions.create

            async def _spy_local(*a, **kw):
                local_called["v"] = True
                return await original_local(*a, **kw)

            srv._openai_client.chat.completions.create = _spy_local

            async def _spy_remote(req, cache):
                remote_called["v"] = True
                return await _fake_remote_create(req, cache)

            srv._remote_client.create_completion = _spy_remote

            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        assert local_called["v"] is True
        assert remote_called["v"] is False

    @pytest.mark.asyncio
    async def test_remote_prefix_calls_remote_client(self, _routing_env, monkeypatch):
        """model='remote:openai/gpt-4o-mini' → remote client with stripped model."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]

            captured = {"request": None}

            async def _capture_remote(req, cache):
                captured["request"] = req
                return await _fake_remote_create(req, cache)

            srv._remote_client.create_completion = _capture_remote

            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        assert captured["request"] is not None
        assert captured["request"]["model"] == "remote:openai/gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_empty_remote_model_returns_400(self, _routing_env, monkeypatch):
        """model='remote:' → 400, no remote call, no cache entry."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]
            session_id = start.json()["session_id"]

            remote_called = {"v": False}

            async def _spy_remote(req, cache):
                remote_called["v"] = True
                return await _fake_remote_create(req, cache)

            srv._remote_client.create_completion = _spy_remote

            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "remote:", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 400
        assert remote_called["v"] is False
        # No interaction cached.
        assert len(srv._session_cache[session_id].completions) == 0

    @pytest.mark.asyncio
    async def test_unknown_model_returns_404(self, _routing_env, monkeypatch):
        """model='unknown-local-model' → 404, no local/remote call, no cache."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]
            session_id = start.json()["session_id"]

            local_called = {"v": False}
            remote_called = {"v": False}

            async def _spy_local(*a, **kw):
                local_called["v"] = True
                return await _fake_local_create(**kw)

            async def _spy_remote(req, cache):
                remote_called["v"] = True
                return await _fake_remote_create(req, cache)

            srv._openai_client.chat.completions.create = _spy_local
            srv._remote_client.create_completion = _spy_remote

            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "unknown-local-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 404
        assert local_called["v"] is False
        assert remote_called["v"] is False
        assert len(srv._session_cache[session_id].completions) == 0

    @pytest.mark.asyncio
    async def test_remote_stream_true_returns_400(self, _routing_env, monkeypatch):
        """model='remote:...' + stream=True → 400."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]

            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_remote_n_2_returns_400(self, _routing_env, monkeypatch):
        """model='remote:...' + n=2 → 400."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]

            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "n": 2,
                },
            )
        assert resp.status_code == 400
```

- [ ] **Step 2: Run the routing tests to verify they fail**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestRoutingDispatch -x -q 2>&1 | tail -20`

Expected: FAIL — `srv._remote_client` does not exist yet (AttributeError), or
`chat_completions` routes everything to the local client (so
`test_remote_prefix_calls_remote_client` fails because the remote spy is never called,
and `test_unknown_model_returns_404` fails because the unknown model is sent to the
local client which accepts it).

- [ ] **Step 3: Add the `_remote_client` global and import**

In `areal/experimental/openai/proxy/proxy_rollout_server.py`:

1. Add the import near the existing `ArealOpenAI` import (line 32):

```python
from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient
```

2. Add the global near `_openai_client` (after line 96):

```python
# Remote rollout client (constructed in _setup_openai_client alongside _openai_client)
_remote_client: RemoteRolloutClient | None = None
```

- [ ] **Step 4: Construct `_remote_client` in `_setup_openai_client`**

In `_setup_openai_client` (starts at line 262), the function currently ends at line 282
with the admin-key warning. After the `_session_timeout_seconds` assignment and before
the `with _lock:` block, add construction of `_remote_client`:

Find this block (lines 267-282):

```python
    _openai_client = ArealOpenAI(
        engine=_engine,
        tokenizer=tokenizer,
        tool_call_parser=agent_cfg.tool_call_parser,
        reasoning_parser=agent_cfg.reasoning_parser,
        engine_max_tokens=agent_cfg.engine_max_tokens,
        chat_template_type=agent_cfg.chat_template_type,
    )
    _session_timeout_seconds = agent_cfg.session_timeout_seconds
    with _lock:
        _admin_api_key = agent_cfg.admin_api_key
        if _admin_api_key == DEFAULT_ADMIN_API_KEY:
            logger.warning(
                "Using default admin API key. Change 'admin_api_key' in "
                "AgentConfig for non-local deployments."
            )
```

Add `global _remote_client` at the top of the function (the existing `global` line is
`global _openai_client, _session_timeout_seconds, _admin_api_key` — extend it), and
construct the client after `_openai_client`:

```python
def _setup_openai_client():
    global _openai_client, _remote_client, _session_timeout_seconds, _admin_api_key
    config = _engine.config
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    agent_cfg = config.agent
    _openai_client = ArealOpenAI(
        engine=_engine,
        tokenizer=tokenizer,
        tool_call_parser=agent_cfg.tool_call_parser,
        reasoning_parser=agent_cfg.reasoning_parser,
        engine_max_tokens=agent_cfg.engine_max_tokens,
        chat_template_type=agent_cfg.chat_template_type,
    )
    _remote_client = RemoteRolloutClient(
        tokenizer=tokenizer,
        chat_template_type=agent_cfg.chat_template_type,
        engine_max_tokens=agent_cfg.engine_max_tokens,
        recompute_enabled=agent_cfg.should_compute_prox_logp(),
    )
    _session_timeout_seconds = agent_cfg.session_timeout_seconds
    with _lock:
        _admin_api_key = agent_cfg.admin_api_key
        if _admin_api_key == DEFAULT_ADMIN_API_KEY:
            logger.warning(
                "Using default admin API key. Change 'admin_api_key' in "
                "AgentConfig for non-local deployments."
            )
```

Note: `PPOActorConfig` may not have `enable_remote_rollout` plumbed through `agent_cfg`
depending on how `config.agent` is resolved. The
`recompute_enabled=agent_cfg.should_compute_prox_logp()` call is what matters for the
runtime gate; the `enable_remote_rollout` flag only controls the trainer-config warning
(already added in Task 2). Do not read `enable_remote_rollout` here.

- [ ] **Step 5: Add model-prefix dispatch in `chat_completions`**

In `chat_completions` (starts at line 610), the function currently checks
`_openai_client is None` then branches on `is_streaming`. The dispatch must happen
*before* the streaming check, because a remote `stream=True` request should return 400
(from the remote client), not enter the local streaming path.

Find the current handler (lines 610-667):

```python
async def chat_completions(
    request: CompletionCreateParams, session_id: str = Depends(_require_session_key)
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint. ..."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    # CompletionCreateParams is a TypedDict (dict subclass), so use dict access.
    is_streaming = request.get("stream") is True
    ...
```

Replace the body from the `_openai_client is None` check onward with dispatch logic.
Insert the dispatch between the session-key dependency (already resolved) and the
existing streaming logic:

```python
async def chat_completions(
    request: CompletionCreateParams, session_id: str = Depends(_require_session_key)
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=True) and non-streaming requests.
    For streaming requests, returns a StreamingResponse with Server-Sent Events
    in the OpenAI streaming format (data: {json}\\n\\n ... data: [DONE]\\n\\n).

    Remote rollout: requests with model starting with "remote:" are routed
    to OpenRouter via RemoteRolloutClient. model="default" uses the local
    inference engine. Unknown models return 404.
    """
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    # --- Model-prefix dispatch ---
    model = request.get("model", "default")
    if isinstance(model, str) and model.startswith("remote:"):
        if _remote_client is None:
            raise HTTPException(
                status_code=500,
                detail="Remote rollout client not initialized.",
            )
        # Resolve session cache (same as _call_client_create does).
        with _lock:
            if session_id not in _session_cache:
                raise HTTPException(
                    status_code=410,
                    detail=f"Session {session_id} already ended or expired",
                )
            session_data = _session_cache[session_id]
        session_data.update_last_access()
        return await _remote_client.create_completion(
            dict(request), session_data.completions
        )
    if model != "default":
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {model!r}. Use 'default' for local inference "
            "or 'remote:<provider/model>' for OpenRouter.",
        )

    # --- Existing local path (unchanged below) ---
    # CompletionCreateParams is a TypedDict (dict subclass), so use dict access.
    is_streaming = request.get("stream") is True
    if is_streaming:
        openai_stream = None
        try:
            openai_stream = await _call_client_create(
                create_fn=_openai_client.chat.completions.create,
                request=request,
                session_id=session_id,
                stream=True,
            )
            # ... (rest of the existing streaming logic unchanged)
```

The key insertion is the `model-prefix dispatch` block between the
`_openai_client is None` check and the `is_streaming = request.get("stream") is True`
line. Everything after `is_streaming` stays exactly as it is today.

- [ ] **Step 6: Run the routing tests**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestRoutingDispatch -x -q 2>&1 | tail -25`

Expected: PASS (6 tests — spec tests 1-6). Spec test 7 (`chat_template_type="concat"` +
remote → 400) is covered by the remote client unit tests in Task 4 (the
`chat_template_type != "hf"` check in step 1 of `create_completion`).

If `test_remote_prefix_calls_remote_client` fails with a 500 "Remote rollout client not
initialized", the `_remote_client` global is None because `_setup_openai_client` was
never called in the test — but the `_routing_env` fixture monkeypatches
`srv._remote_client` directly, so this should not happen. Check that the fixture runs
before the test body.

- [ ] **Step 7: Run the existing streaming tests to confirm no regression**

Run:
`uv run pytest tests/experimental/openai/test_streaming_chat_completions.py -x -q 2>&1 | tail -15`

Expected: PASS. The local path is unchanged; only a dispatch block was added before it.
If a streaming test fails, the dispatch block is interfering — check that `model` is
`"default"` or absent in those tests (the existing tests don't set `model`, so
`request.get("model", "default")` returns `"default"` and skips both the `remote:` and
the 404 branches).

- [ ] **Step 8: Commit**

```bash
git add areal/experimental/openai/proxy/proxy_rollout_server.py tests/experimental/openai/test_remote_rollout.py
git commit -m "$(cat <<'EOF'
feat: wire remote rollout dispatch into chat_completions

chat_completions now inspects request.model: 'default' uses the
existing local _openai_client path; 'remote:<provider/model>' enters
RemoteRolloutClient.create_completion; anything else returns 404.

_remote_client is constructed in _setup_openai_client alongside
_openai_client, sharing the tokenizer and chat_template_type, with
recompute_enabled derived from agent_cfg.should_compute_prox_logp().

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

______________________________________________________________________

## Task 6: End-to-end cache/export tests and documentation example

**Why:** The routing and unit tests prove the pieces work in isolation. These tests
prove the full stack — that a remote completion flows through the proxy, lands in the
session cache, is reward-addressable by completion ID, and exports as tensor data. The
documentation example (spec test 23) locks in the user-facing contract.

**Files:**

- Test: `tests/experimental/openai/test_remote_rollout.py` (append)

- [ ] **Step 1: Write the end-to-end cache/export tests (spec tests 15-17)**

Append to `tests/experimental/openai/test_remote_rollout.py`:

```python
# ---------------------------------------------------------------------------
# Tests: end-to-end cache/export (spec tests 15-17)
# ---------------------------------------------------------------------------


@pytest.fixture
def _e2e_env(monkeypatch, real_tokenizer):
    """Full server env with a real-tokenizer RemoteRolloutClient and stubbed OpenRouter."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)
    # Local client stub.
    mock_local = MagicMock()
    mock_local.chat.completions.create = _fake_local_create
    monkeypatch.setattr(srv, "_openai_client", mock_local)
    # Real-tokenizer remote client with stubbed OpenRouter call.
    from areal.experimental.openai.proxy.remote_rollout import RemoteRolloutClient

    remote_client = RemoteRolloutClient(
        tokenizer=real_tokenizer,
        chat_template_type="hf",
        recompute_enabled=True,
    )
    fake_completion = _make_chat_completion(
        completion_id="chatcmpl-e2e-remote",
        content="The answer is 42.",
        finish_reason="stop",
    )
    remote_client._call_openrouter = lambda *a, **kw: _async_return(fake_completion)
    monkeypatch.setattr(srv, "_remote_client", remote_client)


class TestEndToEndCacheExport:
    @pytest.mark.asyncio
    async def test_set_reward_by_remote_completion_id(self, _e2e_env, monkeypatch):
        """Remote completion → /rl/set_reward by remote completion ID succeeds (spec test 15)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]
            session_id = start.json()["session_id"]

            chat = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "What is the answer?"}],
                },
            )
            assert chat.status_code == 200
            completion_id = chat.json()["id"]
            assert completion_id == "chatcmpl-e2e-remote"

            reward = await client.post(
                "/rl/set_reward",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"interaction_id": completion_id, "reward": 1.5},
            )
            assert reward.status_code == 200

        # Verify the reward landed on the cached interaction.
        interaction = srv._session_cache[session_id].completions[completion_id]
        assert interaction.reward == 1.5

    @pytest.mark.asyncio
    async def test_export_returns_tensor_data(self, _e2e_env, monkeypatch):
        """Export returns tensor data with local-tokenized prompt/output and reward (spec test 16)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]
            session_id = start.json()["session_id"]

            await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "What is the answer?"}],
                },
            )
            await client.post(
                "/rl/set_reward",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"interaction_id": "chatcmpl-e2e-remote", "reward": 2.0},
            )
            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            export = await client.post(
                "/export_trajectories",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"session_id": session_id},
            )
            assert export.status_code == 200
            data = export.json()
            interactions = data["interactions"]
            assert "chatcmpl-e2e-remote" in interactions
            entry = interactions["chatcmpl-e2e-remote"]
            assert "tensor_dict" in entry
            assert entry["reward"] == 2.0

    @pytest.mark.asyncio
    async def test_remote_and_local_in_same_session(self, _e2e_env, monkeypatch):
        """Remote + local completion in one session — both stored, both exportable (spec test 17)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "t"},
            )
            api_key = start.json()["api_key"]
            session_id = start.json()["session_id"]

            remote_chat = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "remote q"}],
                },
            )
            local_chat = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "local q"}],
                },
            )
            assert remote_chat.status_code == 200
            assert local_chat.status_code == 200

            completions = srv._session_cache[session_id].completions
            assert "chatcmpl-e2e-remote" in completions
            assert "chatcmpl-local" in completions


# ---------------------------------------------------------------------------
# Tests: documentation example (spec test 23)
# ---------------------------------------------------------------------------


class TestDocumentationExample:
    @pytest.mark.asyncio
    async def test_remote_request_shape_documented(self, _routing_env, monkeypatch):
        """The documented request shape routes to the remote client (no real network call)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _routing_client() as client:
            start = await client.post(
                "/rl/start_session",
                headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
                json={"task_id": "docs"},
            )
            api_key = start.json()["api_key"]

            captured = {"request": None}

            async def _capture(req, cache):
                captured["request"] = req
                return await _fake_remote_create(req, cache)

            srv._remote_client.create_completion = _capture

            # Documented request shape from the spec:
            # {
            #   "model": "remote:anthropic/claude-3.5-sonnet",
            #   "messages": [{"role": "user", "content": "Solve this task."}]
            # }
            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "remote:anthropic/claude-3.5-sonnet",
                    "messages": [{"role": "user", "content": "Solve this task."}],
                },
            )
        assert resp.status_code == 200
        assert captured["request"]["model"] == "remote:anthropic/claude-3.5-sonnet"
```

- [ ] **Step 2: Run the end-to-end tests**

Run:
`uv run pytest tests/experimental/openai/test_remote_rollout.py::TestEndToEndCacheExport tests/experimental/openai/test_remote_rollout.py::TestDocumentationExample -x -q 2>&1 | tail -25`

Expected: PASS (4 tests). These need a real tokenizer (the `_e2e_env` fixture uses
`real_tokenizer`), so they may be slow on first run due to tokenizer download.

If `test_export_returns_tensor_data` fails on the `interaction.has_tensor_data` check
(export returns the string-only `concat_string_interactions` form instead of
`tensor_dict`), the `model_response` was not set on the interaction — check that step 9
in `create_completion` assigns `interaction.model_response = model_response` before
returning.

- [ ] **Step 3: Run the full test file**

Run: `uv run pytest tests/experimental/openai/test_remote_rollout.py -q 2>&1 | tail -25`

Expected: All tests pass (23 tests total: 3 config-warning + 6 finish-reason + 2
recompute-gate + 1 happy-path + 5 failure + 6 routing + 3 e2e + 1 docs — note some
counts differ from the spec because the finish-reason mapper is split into 6 sub-tests
and the recompute gate is 2).

- [ ] **Step 4: Run ruff on the new and modified files**

Run:
`uv run ruff check areal/experimental/openai/_prompt_utils.py areal/experimental/openai/proxy/remote_rollout.py areal/experimental/openai/proxy/proxy_rollout_server.py areal/api/cli_args.py tests/experimental/openai/test_remote_rollout.py 2>&1 | tail -20`

Expected: No errors. If ruff flags unused imports (e.g. `Iterable`/`Mapping` in
`_prompt_utils.py` are used; `re` is used), remove them. If it flags the `HTTPException`
import at the bottom of the test file (the `# noqa: E402` comment suppresses the
late-import warning), keep the noqa.

- [ ] **Step 5: Run ruff format**

Run:
`uv run ruff format areal/experimental/openai/_prompt_utils.py areal/experimental/openai/proxy/remote_rollout.py areal/experimental/openai/proxy/proxy_rollout_server.py areal/api/cli_args.py tests/experimental/openai/test_remote_rollout.py 2>&1 | tail -10`

Expected: Files formatted. Re-run the test suite if formatting changed anything.

- [ ] **Step 6: Commit**

```bash
git add tests/experimental/openai/test_remote_rollout.py
git commit -m "$(cat <<'EOF'
test: add end-to-end and documentation tests for remote rollout

End-to-end tests verify a remote completion flows through the proxy,
lands in the session cache, is reward-addressable by completion ID,
and exports as tensor data with local-tokenized prompt/output. A
mixed remote+local session confirms both paths coexist.

The documentation test locks in the spec's documented request shape
(model='remote:anthropic/claude-3.5-sonnet') as a routing contract.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

______________________________________________________________________

## Self-Review Checklist

After all tasks are complete, verify:

- [ ] **Spec coverage:** Every section of the spec maps to a task.

  - Routing (spec §Routing) → Task 5
  - Per-Request Lifecycle (spec §Implementation Approach > Per-Request Lifecycle) → Task
    4
  - Cleanup Contract → Task 4 (try/except in `create_completion`)
  - Recompute Validation 3 layers → Task 2 (layer 1: config warning) + Task 3/4 (layer
    2: first-request gate) + layer 3 "no proxy-startup check" is implicit (no task
    needed)
  - Tokenization And Tensor Semantics → Task 4 (EOS append, `[-1]` versions, `[0.0]`
    logprobs)
  - Error Handling → Task 3 (`_call_openrouter` 502/504) + Task 4 (cleanup on failure)
  - Security And Privacy → Task 3 (docstring on `RemoteRolloutClient`, log message on
    client construction)
  - Testing Plan (23 tests) → Tasks 2, 3, 4, 5, 6
  - Resolved Implementation Questions 1-3 → Task 3 (mapper), Task 4 (gate, ID
    preservation)

- [ ] **No placeholders:** Search the plan for "TBD", "TODO", "implement later", "fill
  in" — none should remain.

- [ ] **Type consistency:** `RemoteRolloutClient.__init__` signature is
  `(tokenizer, chat_template_type, engine_max_tokens=None, recompute_enabled=False)` in
  Task 3 and Task 5 — match. `_map_finish_reason(finish_reason: str | None) -> str` in
  Task 3 — match. `_call_openrouter(provider_model, forwarded)` in Task 3 and Task 4 —
  match. `_recompute_verified` attribute name in Task 3, Task 4, and tests — match.

- [ ] **Import paths:** `_prompt_utils` is at `areal.experimental.openai._prompt_utils`
  (not `areal.experimental.openai.proxy._prompt_utils`) — it's shared by `client.py`
  (which is in the parent package) and `remote_rollout.py` (in the proxy subpackage).
  Verify the import in `client.py` (Task 1 step 2) uses
  `from areal.experimental.openai._prompt_utils import ...`.

- [ ] **Class name:** The spec says `ActorConfig` but the actual class is
  `PPOActorConfig` (areal/api/cli_args.py:1388). All tasks and tests use
  `PPOActorConfig`.

______________________________________________________________________

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-06-29-openrouter-remote-rollout-proxy.md`. Two execution
options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review
between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch
execution with checkpoints.

Which approach?

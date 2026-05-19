# Selected-Turn Tree Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build selected-turn teacher distillation for `TreeSearchGroupedRolloutWorkflow`, with turn-wise Qwen diagnosis, optional top-k distillation, cache refill, and direct `student_logp - teacher_logp` loss.

**Architecture:** Add a provider layer for diagnosis/logprob calls, defaulting to the existing external OpenAI-compatible teacher client and exposing an engine-provider boundary. Add a selected-turn distill builder that converts episode `Node`s into teacher-scored `PositionRewardInfo` records, then integrate it into the grouped workflow before batching. Update the custom distill loss to use explicit teacher logprobs instead of the current normalized position-level reward objective.

**Tech Stack:** Python 3.12, asyncio, dataclasses, PyTorch, AReaL FSDP actor training, existing `customized_areal.tree_search` modules, pytest.

---

## File Structure

- Create `customized_areal/tree_search/core/selected_turn_distill.py`
  - Owns diagnosis parsing, turn span extraction, teacher prompt assembly, selected-turn episode preparation, and `PositionRewardInfo` construction.
- Create `customized_areal/tree_search/core/teacher_provider.py`
  - Defines `TeacherProvider`, `ExternalTeacherProvider`, and `EngineTeacherProvider`.
- Modify `customized_areal/tree_search/core/teacher_client.py`
  - Add generic completion support for diagnosis and a prompt-plus-target candidate logprob method.
- Modify `customized_areal/tree_search/distill_types.py`
  - Add `teacher_logprobs` to `PositionRewardInfo` and small diagnosis result dataclasses.
- Modify `customized_areal/tree_search/config.py`
  - Add distill provider/config fields to `TreeBackupConfig`.
- Modify `customized_areal/tree_search/tree_search_grouped_workflow.py`
  - Add tokenizer cache, provider construction, selected-turn distill stage, cache refill, failure policy, and `position_rewards` injection.
- Modify `customized_areal/tree_search/training/loss.py`
  - Replace position-level reward objective with direct teacher-logprob KL loss.
- Modify `areal/infra/remote_inf_engine.py`
  - Read new env values and pass them to the workflow.
- Add tests in `tests/test_tree_search/test_selected_turn_distill.py`
  - Unit tests for parser, prompt spans, top-k reuse/recompute, failure policy helpers, and sample index behavior.
- Add/update tests in `tests/customized_areal/test_teacher_client.py`
  - Client/provider request parsing tests.
- Add/update tests in `tests/test_tree_search/test_distill_loss.py`
  - Direct KL loss tests.

---

### Task 1: Distill Types And Config

**Files:**
- Modify: `customized_areal/tree_search/distill_types.py`
- Modify: `customized_areal/tree_search/config.py`
- Test: `tests/test_tree_search/test_selected_turn_distill.py`

- [ ] **Step 1: Write failing tests for diagnosis result types and teacher logprobs**

Add this file:

```python
from customized_areal.tree_search.config import TreeBackupConfig
from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    PositionRewardInfo,
)


def test_position_reward_info_carries_teacher_logprobs():
    info = PositionRewardInfo(
        position=0,
        candidate_token_ids=[11, 12],
        logprobs=[-0.7, -1.3],
        teacher_logprobs=[-0.4, -2.0],
        rewards=[-0.3, 0.7],
        sample_index=3,
    )

    assert info.teacher_logprobs == [-0.4, -2.0]
    assert info.sample_index == 3


def test_diagnosis_turn_requires_guidance_for_selected_turns():
    selected = DiagnosisTurn(turn_idx=2, should_improve=True, guidance="Use the tool.")
    skipped = DiagnosisTurn(turn_idx=3, should_improve=False, guidance="")

    assert selected.is_selected is True
    assert skipped.is_selected is False


def test_tree_backup_config_has_distill_defaults():
    config = TreeBackupConfig()

    assert config.topk_distill is False
    assert config.teacher_provider == "external"
    assert config.teacher_top_k == 10
    assert config.diagnose_temperature == 0.0
    assert config.strict_distill_json is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: FAIL with import errors for `DiagnosisTurn` and missing `teacher_logprobs` or config fields.

- [ ] **Step 3: Add types and config fields**

In `customized_areal/tree_search/distill_types.py`, extend `PositionRewardInfo` and add diagnosis types:

```python
@dataclass
class PositionRewardInfo:
    """Reward information for a single generation position."""

    position: int
    candidates: list[str] = field(default_factory=list)
    candidate_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] | None = None
    teacher_logprobs: list[float] | None = None
    rewards: list[float] = field(default_factory=list)
    chosen_index: int = 0
    sample_index: int = 0


@dataclass(frozen=True)
class DiagnosisTurn:
    """Teacher diagnosis for one assistant turn."""

    turn_idx: int
    should_improve: bool
    guidance: str = ""

    @property
    def is_selected(self) -> bool:
        return self.should_improve and bool(self.guidance.strip())


@dataclass(frozen=True)
class EpisodeDiagnosis:
    """Parsed episode-level diagnosis with turn-wise guidance."""

    turns: tuple[DiagnosisTurn, ...]

    @property
    def selected_turns(self) -> dict[int, str]:
        return {turn.turn_idx: turn.guidance for turn in self.turns if turn.is_selected}
```

In `customized_areal/tree_search/config.py`, add fields to `TreeBackupConfig`:

```python
    topk_distill: bool = False
    teacher_provider: str = "external"
    teacher_base_url: str = "http://localhost:8001"
    teacher_model_name: str = ""
    teacher_top_k: int = 10
    teacher_max_retries: int = 3
    teacher_timeout: float = 60.0
    teacher_missing_logprob: float = -23.0
    diagnose_model_name: str = ""
    diagnose_max_tokens: int = 1024
    diagnose_temperature: float = 0.0
    strict_distill_json: bool = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/distill_types.py customized_areal/tree_search/config.py tests/test_tree_search/test_selected_turn_distill.py
git commit -m "feat(tree-search): add selected-turn distill types"
```

---

### Task 2: Teacher Client And Provider Interface

**Files:**
- Create: `customized_areal/tree_search/core/teacher_provider.py`
- Modify: `customized_areal/tree_search/core/teacher_client.py`
- Modify: `customized_areal/tree_search/core/__init__.py`
- Test: `tests/customized_areal/test_teacher_client.py`

- [ ] **Step 1: Write failing tests for completion and provider behavior**

Append these tests to `tests/customized_areal/test_teacher_client.py`:

```python
import pytest

from customized_areal.tree_search.core.teacher_client import TeacherConfig
from customized_areal.tree_search.core.teacher_provider import (
    EngineTeacherProvider,
    ExternalTeacherProvider,
)


class FakeTeacherClient:
    def __init__(self):
        self.diagnose_payload = None
        self.logprob_payload = None

    async def complete_text(self, prompt, *, model=None, max_tokens=1024, temperature=0.0):
        self.diagnose_payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return '{"turns":[{"turn_idx":1,"should_improve":true,"guidance":"Be precise."}]}'

    async def get_logprobs_for_candidates(self, input_ids, output_ids, candidate_token_ids, tokenizer=None):
        self.logprob_payload = {
            "input_ids": input_ids,
            "output_ids": output_ids,
            "candidate_token_ids": candidate_token_ids,
        }
        return [{7: -0.25}]


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
    assert "context" in client.diagnose_payload["prompt"]
    assert "gold" in client.diagnose_payload["prompt"]


@pytest.mark.asyncio
async def test_external_provider_delegates_candidate_logprobs_to_client():
    client = FakeTeacherClient()
    provider = ExternalTeacherProvider(client=client)

    result = await provider.get_logprobs_for_prompt(
        prompt_ids=[1, 2],
        generation_ids=[7],
        candidate_token_ids=[[7]],
    )

    assert result == [[-0.25]]
    assert client.logprob_payload["input_ids"] == [1, 2]
    assert client.logprob_payload["output_ids"] == [7]


def test_engine_provider_fails_early_without_compatible_methods():
    class Engine:
        pass

    with pytest.raises(NotImplementedError, match="engine-backed teacher provider"):
        EngineTeacherProvider(Engine())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/customized_areal/test_teacher_client.py -q
```

Expected: FAIL because `teacher_provider.py` and `complete_text()` do not exist.

- [ ] **Step 3: Add `complete_text()` to `TeacherClient`**

In `customized_areal/tree_search/core/teacher_client.py`, add this method to `TeacherClient`:

```python
    async def complete_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return a text completion from the teacher API."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        selected_model = model or self.config.teacher_model_name
        if selected_model:
            payload["model"] = selected_model

        response_data = await self._post_with_retries(payload)
        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError("Teacher API returned no completion choices")
        text = choices[0].get("text")
        if text is None:
            message = choices[0].get("message", {})
            text = message.get("content", "")
        return str(text)
```

- [ ] **Step 4: Create provider interface**

Create `customized_areal/tree_search/core/teacher_provider.py`:

```python
from __future__ import annotations

from typing import Protocol

from customized_areal.tree_search.core.teacher_client import TeacherClient


class TeacherProvider(Protocol):
    async def diagnose_episode(self, context: str, gold_answer: str) -> str: ...

    async def get_logprobs_for_prompt(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[list[float]]: ...


class ExternalTeacherProvider:
    def __init__(
        self,
        client: TeacherClient,
        diagnose_model_name: str = "",
        diagnose_max_tokens: int = 1024,
        diagnose_temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.diagnose_model_name = diagnose_model_name
        self.diagnose_max_tokens = diagnose_max_tokens
        self.diagnose_temperature = diagnose_temperature

    async def diagnose_episode(self, context: str, gold_answer: str) -> str:
        prompt = (
            "You are diagnosing a multi-turn assistant trajectory. "
            "Return strict JSON with key 'turns'. Each turn must include "
            "'turn_idx', 'should_improve', and 'guidance'. Select only turns "
            "where the assistant generation can be improved toward the gold answer.\n\n"
            f"Gold answer:\n{gold_answer}\n\n"
            f"Episode context:\n{context}\n"
        )
        return await self.client.complete_text(
            prompt,
            model=self.diagnose_model_name or None,
            max_tokens=self.diagnose_max_tokens,
            temperature=self.diagnose_temperature,
        )

    async def get_logprobs_for_prompt(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[list[float]]:
        maps = await self.client.get_logprobs_for_candidates(
            input_ids=prompt_ids,
            output_ids=generation_ids,
            candidate_token_ids=candidate_token_ids,
        )
        return [
            [pos_map[token_id] for token_id in pos_candidates]
            for pos_map, pos_candidates in zip(maps, candidate_token_ids, strict=True)
        ]


class EngineTeacherProvider:
    def __init__(self, engine) -> None:
        if not hasattr(engine, "get_logprobs_for_prompt"):
            raise NotImplementedError(
                "engine-backed teacher provider requires engine.get_logprobs_for_prompt"
            )
        self.engine = engine

    async def diagnose_episode(self, context: str, gold_answer: str) -> str:
        if not hasattr(self.engine, "diagnose_episode"):
            raise NotImplementedError(
                "engine-backed teacher provider requires engine.diagnose_episode"
            )
        return await self.engine.diagnose_episode(context=context, gold_answer=gold_answer)

    async def get_logprobs_for_prompt(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[list[float]]:
        return await self.engine.get_logprobs_for_prompt(
            prompt_ids=prompt_ids,
            generation_ids=generation_ids,
            candidate_token_ids=candidate_token_ids,
        )
```

Update `customized_areal/tree_search/core/__init__.py`:

```python
from .teacher_provider import (
    EngineTeacherProvider,
    ExternalTeacherProvider,
    TeacherProvider,
)
```

and add these names to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/customized_areal/test_teacher_client.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/core/teacher_client.py customized_areal/tree_search/core/teacher_provider.py customized_areal/tree_search/core/__init__.py tests/customized_areal/test_teacher_client.py
git commit -m "feat(tree-search): add teacher provider interface"
```

---

### Task 3: Selected-Turn Distill Builder

**Files:**
- Create: `customized_areal/tree_search/core/selected_turn_distill.py`
- Test: `tests/test_tree_search/test_selected_turn_distill.py`

- [ ] **Step 1: Write failing tests for parser, spans, and single-candidate scoring**

Append these tests to `tests/test_tree_search/test_selected_turn_distill.py`:

```python
import pytest

from customized_areal.tree_search.core.selected_turn_distill import (
    build_teacher_prompt_ids,
    parse_episode_diagnosis,
    response_token_span,
    selected_turn_to_position_rewards,
)
from customized_areal.tree_search.mcts_tree_store import Node


class FakeTokenizer:
    def __init__(self):
        self.sep = 999

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 251 for ch in text]


class FakeProvider:
    async def get_logprobs_for_prompt(self, prompt_ids, generation_ids, candidate_token_ids):
        assert generation_ids == [30, 31]
        assert candidate_token_ids == [[30], [31]]
        return [[-0.3], [-0.4]]


def test_parse_episode_diagnosis_keeps_selected_turn_guidance():
    raw = '{"turns":[{"turn_idx":1,"should_improve":true,"guidance":"Be exact."},{"turn_idx":2,"should_improve":false,"guidance":"Ignore"}]}'

    diagnosis = parse_episode_diagnosis(raw)

    assert diagnosis.selected_turns == {1: "Be exact."}


def test_response_token_span_returns_current_response_region():
    assert response_token_span([0, 0, 1, 1, 0]) == (2, 4)
    assert response_token_span([0, 1, 1, 0, 0, 1, 1]) == (5, 7)


def test_build_teacher_prompt_ids_keeps_generation_out_of_prefix():
    tokenizer = FakeTokenizer()
    node = Node(
        input_ids=[10, 20, 30, 31],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.7, -0.8],
        versions=[-1, -1, 0, 0],
        turn_idx=1,
    )

    prompt_ids, generation_ids = build_teacher_prompt_ids(
        node=node,
        guidance="Be exact.",
        tokenizer=tokenizer,
    )

    assert generation_ids == [30, 31]
    assert 30 not in prompt_ids[:2]


@pytest.mark.asyncio
async def test_selected_turn_to_position_rewards_single_candidate():
    tokenizer = FakeTokenizer()
    node = Node(
        input_ids=[10, 20, 30, 31],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.7, -0.8],
        versions=[-1, -1, 0, 0],
        turn_idx=1,
    )

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Be exact.",
        tokenizer=tokenizer,
        provider=FakeProvider(),
        sample_index=4,
        topk_distill=False,
        engine=None,
        teacher_top_k=10,
    )

    assert [reward.position for reward in rewards] == [0, 1]
    assert [reward.candidate_token_ids for reward in rewards] == [[30], [31]]
    assert [reward.teacher_logprobs for reward in rewards] == [[-0.3], [-0.4]]
    assert [reward.rewards for reward in rewards] == [[-0.4], [-0.4]]
    assert all(reward.sample_index == 4 for reward in rewards)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: FAIL because `selected_turn_distill.py` does not exist.

- [ ] **Step 3: Implement selected-turn helper module**

Create `customized_areal/tree_search/core/selected_turn_distill.py`:

```python
from __future__ import annotations

import json
from typing import Any

from customized_areal.tree_search.distill_types import (
    DiagnosisTurn,
    EpisodeDiagnosis,
    PositionRewardInfo,
)
from customized_areal.tree_search.mcts_tree_store import Node


def parse_episode_diagnosis(raw_text: str) -> EpisodeDiagnosis:
    data = json.loads(raw_text)
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("diagnosis JSON must contain a list field named 'turns'")

    turns: list[DiagnosisTurn] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            raise ValueError("each diagnosis turn must be an object")
        turns.append(
            DiagnosisTurn(
                turn_idx=int(item["turn_idx"]),
                should_improve=bool(item["should_improve"]),
                guidance=str(item.get("guidance", "")),
            )
        )
    return EpisodeDiagnosis(turns=tuple(turns))


def response_token_span(loss_mask: list[int]) -> tuple[int, int]:
    start: int | None = None
    latest_span: tuple[int, int] | None = None
    for idx, value in enumerate(loss_mask):
        if value == 1 and start is None:
            start = idx
        elif value != 1 and start is not None:
            latest_span = (start, idx)
            start = None
    if start is not None:
        latest_span = (start, len(loss_mask))
    return latest_span or (0, 0)


def build_teacher_prompt_ids(
    node: Node,
    guidance: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    start, end = response_token_span(node.loss_mask)
    prefix_ids = node.input_ids[:start]
    generation_ids = node.input_ids[start:end]
    guidance_ids = tokenizer.encode(
        "\n\nTeacher guidance:\n" + guidance.strip() + "\n\nAssistant generation:\n",
        add_special_tokens=False,
    )
    return prefix_ids + guidance_ids, generation_ids


async def _recompute_student_topk(
    engine: Any,
    node: Node,
    teacher_top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    if engine is None or not hasattr(engine, "get_topk_logprobs"):
        raise NotImplementedError(
            "topk_distill requires engine.get_topk_logprobs for missing student top-k"
        )
    start, end = response_token_span(node.loss_mask)
    topk_ids, topk_logp = await engine.get_topk_logprobs(
        input_ids=node.input_ids,
        loss_mask=node.loss_mask,
        top_k=teacher_top_k,
    )
    node.topk_ids = topk_ids[start:end] if len(topk_ids) == len(node.input_ids) else topk_ids
    node.topk_logp = topk_logp[start:end] if len(topk_logp) == len(node.input_ids) else topk_logp
    return node.topk_ids, node.topk_logp


async def selected_turn_to_position_rewards(
    node: Node,
    guidance: str,
    tokenizer: Any,
    provider: Any,
    sample_index: int,
    topk_distill: bool,
    engine: Any,
    teacher_top_k: int,
) -> list[PositionRewardInfo]:
    prompt_ids, generation_ids = build_teacher_prompt_ids(node, guidance, tokenizer)
    start, end = response_token_span(node.loss_mask)
    student_logprobs = node.logprobs[start:end]

    if topk_distill:
        if node.topk_ids is None or node.topk_logp is None:
            topk_ids, topk_logp = await _recompute_student_topk(engine, node, teacher_top_k)
        else:
            topk_ids, topk_logp = node.topk_ids, node.topk_logp
        candidate_token_ids = []
        candidate_student_logprobs = []
        for generated_id, generated_logp, ids, logps in zip(
            generation_ids, student_logprobs, topk_ids, topk_logp, strict=True
        ):
            ids_without_generated = [token_id for token_id in ids[:teacher_top_k] if token_id != generated_id]
            logp_by_id = {token_id: logp for token_id, logp in zip(ids, logps, strict=False)}
            candidate_token_ids.append([generated_id] + ids_without_generated)
            candidate_student_logprobs.append(
                [generated_logp] + [float(logp_by_id[token_id]) for token_id in ids_without_generated]
            )
    else:
        candidate_token_ids = [[token_id] for token_id in generation_ids]
        candidate_student_logprobs = [[float(logp)] for logp in student_logprobs]

    teacher_logprobs = await provider.get_logprobs_for_prompt(
        prompt_ids=prompt_ids,
        generation_ids=generation_ids,
        candidate_token_ids=candidate_token_ids,
    )

    node.teacher_logp = teacher_logprobs
    node.distill_reward = [
        [student_lp - teacher_lp for student_lp, teacher_lp in zip(student_row, teacher_row, strict=True)]
        for student_row, teacher_row in zip(candidate_student_logprobs, teacher_logprobs, strict=True)
    ]

    return [
        PositionRewardInfo(
            position=position,
            candidates=[str(token_id) for token_id in candidate_row],
            candidate_token_ids=candidate_row,
            logprobs=student_row,
            teacher_logprobs=teacher_row,
            rewards=reward_row,
            chosen_index=0,
            sample_index=sample_index,
        )
        for position, (candidate_row, student_row, teacher_row, reward_row) in enumerate(
            zip(
                candidate_token_ids,
                candidate_student_logprobs,
                teacher_logprobs,
                node.distill_reward,
                strict=True,
            )
        )
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/selected_turn_distill.py tests/test_tree_search/test_selected_turn_distill.py
git commit -m "feat(tree-search): build selected-turn distill rewards"
```

---

### Task 4: Top-K Reuse And Recompute Tests

**Files:**
- Modify: `customized_areal/tree_search/core/selected_turn_distill.py`
- Test: `tests/test_tree_search/test_selected_turn_distill.py`

- [ ] **Step 1: Add failing tests for top-k reuse and recompute**

Append:

```python
@pytest.mark.asyncio
async def test_selected_turn_topk_reuses_cached_student_candidates():
    tokenizer = FakeTokenizer()
    node = Node(
        input_ids=[10, 20, 30],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.7],
        versions=[-1, -1, 0],
        turn_idx=1,
        topk_ids=[[30, 40]],
        topk_logp=[[-0.7, -1.2]],
    )

    class Provider:
        async def get_logprobs_for_prompt(self, prompt_ids, generation_ids, candidate_token_ids):
            assert candidate_token_ids == [[30, 40]]
            return [[-0.3, -1.5]]

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Improve.",
        tokenizer=tokenizer,
        provider=Provider(),
        sample_index=0,
        topk_distill=True,
        engine=None,
        teacher_top_k=2,
    )

    assert rewards[0].candidate_token_ids == [30, 40]
    assert rewards[0].teacher_logprobs == [-0.3, -1.5]
    assert node.topk_ids == [[30, 40]]


@pytest.mark.asyncio
async def test_selected_turn_topk_recomputes_missing_student_candidates():
    tokenizer = FakeTokenizer()
    node = Node(
        input_ids=[10, 20, 30],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.7],
        versions=[-1, -1, 0],
        turn_idx=1,
    )

    class Engine:
        async def get_topk_logprobs(self, input_ids, loss_mask, top_k):
            assert top_k == 2
            return [[30, 41]], [[-0.7, -1.4]]

    class Provider:
        async def get_logprobs_for_prompt(self, prompt_ids, generation_ids, candidate_token_ids):
            return [[-0.2, -1.8]]

    rewards = await selected_turn_to_position_rewards(
        node=node,
        guidance="Improve.",
        tokenizer=tokenizer,
        provider=Provider(),
        sample_index=0,
        topk_distill=True,
        engine=Engine(),
        teacher_top_k=2,
    )

    assert node.topk_ids == [[30, 41]]
    assert node.topk_logp == [[-0.7, -1.4]]
    assert rewards[0].rewards == [-0.49999999999999994, 0.40000000000000013]
```

- [ ] **Step 2: Run tests**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: PASS if Task 3 implementation already covers this; otherwise FAIL points to top-k alignment.

- [ ] **Step 3: Fix top-k alignment if needed**

If the recompute path returns full-sequence top-k rows, keep this alignment in `_recompute_student_topk()`:

```python
    start, end = response_token_span(node.loss_mask)
    if len(topk_ids) == len(node.input_ids):
        node.topk_ids = topk_ids[start:end]
        node.topk_logp = topk_logp[start:end]
    else:
        node.topk_ids = topk_ids
        node.topk_logp = topk_logp
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/selected_turn_distill.py tests/test_tree_search/test_selected_turn_distill.py
git commit -m "feat(tree-search): support top-k distill candidates"
```

---

### Task 5: Direct KL Distill Loss

**Files:**
- Modify: `customized_areal/tree_search/training/loss.py`
- Test: `tests/test_tree_search/test_distill_loss.py`

- [ ] **Step 1: Write failing tests for direct teacher-logprob KL**

Create `tests/test_tree_search/test_distill_loss.py`:

```python
from dataclasses import dataclass

import torch

from customized_areal.tree_search.distill_types import PositionRewardInfo
from customized_areal.tree_search.training.loss import _compute_teacher_kl_loss


@dataclass
class Config:
    eps_clip: float = 0.2
    eps_clip_higher: float | None = None
    c_clip: float | None = None
    behave_imp_weight_cap: float | None = None
    importance_sampling_level: str = "token"
    prox_clip: str = "recompute"


def test_compute_teacher_kl_loss_single_candidate():
    logprobs = torch.tensor([-0.1, -0.2, -0.7, -0.8], dtype=torch.float32)
    loss_mask = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    position_rewards = [
        PositionRewardInfo(position=0, teacher_logprobs=[-0.3], sample_index=0),
        PositionRewardInfo(position=1, teacher_logprobs=[-0.5], sample_index=0),
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[2],
    )

    assert torch.allclose(loss, torch.tensor((-0.7 + 0.3 - 0.8 + 0.5) / 2))


def test_compute_teacher_kl_loss_multi_candidate():
    logprobs = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.7, -1.4],
        ],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor([0, 0, 1], dtype=torch.bool)
    position_rewards = [
        PositionRewardInfo(
            position=0,
            candidate_token_ids=[30, 40],
            teacher_logprobs=[-0.2, -1.8],
            sample_index=0,
        )
    ]

    loss = _compute_teacher_kl_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[2],
    )

    assert torch.allclose(loss, torch.tensor(((-0.7 + 0.2) + (-1.4 + 1.8)) / 2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tree_search/test_distill_loss.py -q
```

Expected: FAIL because `_compute_teacher_kl_loss` does not exist.

- [ ] **Step 3: Implement direct KL helper**

In `customized_areal/tree_search/training/loss.py`, add:

```python
def _compute_teacher_kl_loss(
    position_rewards: list,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int] | int = 0,
) -> torch.Tensor:
    """Compute direct mean(student_logp - teacher_logp) for selected positions."""
    if not position_rewards:
        return torch.tensor(0.0, dtype=torch.float32, device=loss_mask.device)

    device = logprobs.device
    terms: list[torch.Tensor] = []
    for pr in position_rewards:
        teacher_lps = getattr(pr, "teacher_logprobs", None)
        if not teacher_lps:
            continue
        if isinstance(prompt_lens, list):
            prompt_len = prompt_lens[pr.sample_index] if pr.sample_index < len(prompt_lens) else 0
        else:
            prompt_len = prompt_lens
        position = pr.position + prompt_len
        if position < 0 or position >= logprobs.shape[0]:
            continue
        teacher_t = torch.tensor(teacher_lps, dtype=torch.float32, device=device)
        if logprobs.dim() == 1:
            student_t = logprobs[position].reshape(1)
            teacher_t = teacher_t[:1]
        else:
            student_t = logprobs[position, : teacher_t.shape[0]]
        terms.append(student_t - teacher_t)

    if not terms:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.cat([term.reshape(-1) for term in terms]).mean()
```

- [ ] **Step 4: Replace position-level GRPO branch**

In `grpo_distill_loss_fn`, replace the call to `_compute_position_level_grpo_loss(...)` with:

```python
        teacher_kl_loss = _compute_teacher_kl_loss(
            position_rewards=position_rewards,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=prompt_lens,
        )

        loss = rl_loss_weight * loss + distill_loss_weight * teacher_kl_loss
        distill_stat = teacher_kl_loss.detach()
```

Keep `_compute_position_level_grpo_loss()` temporarily if external callers still import it, but do not use it from `grpo_distill_loss_fn`.

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_tree_search/test_distill_loss.py -q
uv run pytest tests/customized_areal/test_reward_compute.py tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/training/loss.py tests/test_tree_search/test_distill_loss.py
git commit -m "feat(tree-search): use teacher logprob KL distill loss"
```

---

### Task 6: Workflow Integration

**Files:**
- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py`
- Test: `tests/test_tree_search/test_selected_turn_distill.py`

- [ ] **Step 1: Add failing tests for episode filtering and sample index assignment**

Append:

```python
from customized_areal.tree_search.config import LossMode
from customized_areal.tree_search.tree_search_grouped_workflow import (
    _filter_distill_episode_failure,
    _set_position_reward_sample_indices,
)


def test_both_mode_keeps_episode_after_distill_failure():
    node = Node(
        input_ids=[1],
        loss_mask=[1],
        logprobs=[-0.1],
        versions=[0],
        episode_id="ep",
    )

    assert _filter_distill_episode_failure([node], LossMode.BOTH) == [node]


def test_distill_mode_drops_episode_after_distill_failure():
    node = Node(
        input_ids=[1],
        loss_mask=[1],
        logprobs=[-0.1],
        versions=[0],
        episode_id="ep",
    )

    assert _filter_distill_episode_failure([node], LossMode.DISTILL) == []


def test_set_position_reward_sample_indices_uses_final_node_order():
    node_a = Node(input_ids=[1], loss_mask=[1], logprobs=[-0.1], versions=[0], node_id="a")
    node_b = Node(input_ids=[2], loss_mask=[1], logprobs=[-0.2], versions=[0], node_id="b")
    rewards_by_node_id = {
        "b": [PositionRewardInfo(position=0, teacher_logprobs=[-0.5])],
    }

    all_rewards = _set_position_reward_sample_indices([node_a, node_b], rewards_by_node_id)

    assert len(all_rewards) == 1
    assert all_rewards[0].sample_index == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: FAIL because workflow helper functions do not exist.

- [ ] **Step 3: Add workflow helper functions**

In `customized_areal/tree_search/tree_search_grouped_workflow.py`, add module helpers:

```python
def _filter_distill_episode_failure(nodes: list[Node], loss_mode: LossMode) -> list[Node]:
    if loss_mode == LossMode.DISTILL:
        return []
    return nodes


def _set_position_reward_sample_indices(
    nodes: list[Node],
    rewards_by_node_id: dict[str, list[Any]],
) -> list[Any]:
    all_rewards: list[Any] = []
    for sample_index, node in enumerate(nodes):
        for reward in rewards_by_node_id.get(node.node_id, []):
            reward.sample_index = sample_index
            all_rewards.append(reward)
    return all_rewards
```

- [ ] **Step 4: Add tokenizer cache and constructor fields**

In `TreeSearchGroupedRolloutWorkflow`, add class attributes:

```python
    _tokenizer_cache: dict[str, Any] = {}
    _tokenizer_lock = asyncio.Lock()
```

Add constructor parameters:

```python
        topk_distill: bool = False,
        teacher_provider: str = "external",
        teacher_base_url: str = "http://localhost:8001",
        teacher_model_name: str = "",
        teacher_top_k: int = 10,
        teacher_max_retries: int = 3,
        teacher_timeout: float = 60.0,
        teacher_missing_logprob: float = -23.0,
        diagnose_model_name: str = "",
        diagnose_max_tokens: int = 1024,
        diagnose_temperature: float = 0.0,
        strict_distill_json: bool = True,
```

Store them on `self`.

Add:

```python
    async def _get_tokenizer(self):
        if self.loss_mode == LossMode.GRPO:
            return None
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path is required when tree-search distillation is enabled")
        async with self._tokenizer_lock:
            tokenizer = self._tokenizer_cache.get(self.tokenizer_path)
            if tokenizer is None:
                from areal.utils.hf_utils import load_hf_tokenizer

                tokenizer = load_hf_tokenizer(self.tokenizer_path)
                self._tokenizer_cache[self.tokenizer_path] = tokenizer
            return tokenizer
```

- [ ] **Step 5: Integrate selected-turn distill stage**

Add methods:

```python
    async def _build_teacher_provider(self, engine):
        from customized_areal.tree_search.core.teacher_client import TeacherConfig, TeacherClient
        from customized_areal.tree_search.core.teacher_provider import (
            EngineTeacherProvider,
            ExternalTeacherProvider,
        )

        if self.teacher_provider == "engine":
            return EngineTeacherProvider(engine), None

        config = TeacherConfig(
            teacher_base_url=self.teacher_base_url,
            teacher_model_name=self.teacher_model_name,
            teacher_top_k=self.teacher_top_k,
            teacher_max_retries=self.teacher_max_retries,
            teacher_timeout=self.teacher_timeout,
            teacher_missing_logprob=self.teacher_missing_logprob,
        )
        client = TeacherClient(config)
        await client.__aenter__()
        provider = ExternalTeacherProvider(
            client=client,
            diagnose_model_name=self.diagnose_model_name or self.teacher_model_name,
            diagnose_max_tokens=self.diagnose_max_tokens,
            diagnose_temperature=self.diagnose_temperature,
        )
        return provider, client
```

Add episode preparation:

```python
    async def _prepare_distill_for_episode(
        self,
        nodes: list[Node],
        data: dict[str, Any],
        engine: Any,
        provider: Any,
        tokenizer: Any,
    ) -> tuple[list[Node], dict[str, list[Any]]]:
        from customized_areal.tree_search.core.selected_turn_distill import (
            parse_episode_diagnosis,
            selected_turn_to_position_rewards,
        )

        if not nodes:
            return nodes, {}
        context = tokenizer.decode(nodes[-1].input_ids, skip_special_tokens=False)
        raw = await provider.diagnose_episode(context, str(data.get("answer", "")))
        diagnosis = parse_episode_diagnosis(raw)
        selected = diagnosis.selected_turns
        if not selected:
            if self.loss_mode == LossMode.DISTILL:
                return [], {}
            return nodes, {}

        rewards_by_node_id: dict[str, list[Any]] = {}
        for node in nodes:
            guidance = selected.get(node.turn_idx)
            if not guidance:
                continue
            rewards_by_node_id[node.node_id] = await selected_turn_to_position_rewards(
                node=node,
                guidance=guidance,
                tokenizer=tokenizer,
                provider=provider,
                sample_index=0,
                topk_distill=self.topk_distill,
                engine=engine,
                teacher_top_k=self.teacher_top_k,
            )
        if self.loss_mode == LossMode.DISTILL and not rewards_by_node_id:
            return [], {}
        return nodes, rewards_by_node_id
```

In `arun_episode()`, after fresh and cached nodes are collected and before insert/advantage, call this stage for each episode group when `loss_mode != LossMode.GRPO`. Collect `rewards_by_node_id`, apply the failure policy per episode, then after `all_nodes` final order is known set:

```python
position_rewards = _set_position_reward_sample_indices(all_nodes, rewards_by_node_id)
if position_rewards:
    result_dict["position_rewards"] = position_rewards
```

Close external client after use:

```python
        provider_client = None
        try:
            provider, provider_client = await self._build_teacher_provider(engine)
            ...
        finally:
            if provider_client is not None:
                await provider_client.__aexit__(None, None, None)
```

- [ ] **Step 6: Run focused workflow helper tests**

Run:

```bash
uv run pytest tests/test_tree_search/test_selected_turn_distill.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py tests/test_tree_search/test_selected_turn_distill.py
git commit -m "feat(tree-search): integrate selected-turn distill workflow"
```

---

### Task 7: Env Wiring

**Files:**
- Modify: `areal/infra/remote_inf_engine.py`
- Test: `tests/test_tree_search/test_config.py`

- [ ] **Step 1: Add failing test for env parsing helper if available**

If `tests/test_tree_search/test_config.py` already tests `TreeBackupConfig`, append:

```python
from customized_areal.tree_search.config import TreeBackupConfig


def test_tree_backup_config_distill_env_defaults_are_representable():
    config = TreeBackupConfig(
        topk_distill=True,
        teacher_provider="external",
        teacher_base_url="http://teacher:8001",
        teacher_model_name="qwen-397b",
        teacher_top_k=5,
        diagnose_model_name="qwen-397b",
    )

    assert config.topk_distill is True
    assert config.teacher_provider == "external"
    assert config.teacher_top_k == 5
```

- [ ] **Step 2: Wire env parsing**

In `areal/infra/remote_inf_engine.py`, near the existing `TREE_SEARCH_*` reads, add:

```python
                topk_distill = (
                    os.getenv("TREE_SEARCH_TOPK_DISTILL", "false").lower() == "true"
                )
                teacher_provider = os.getenv("TREE_SEARCH_TEACHER_PROVIDER", "external")
                teacher_base_url = os.getenv(
                    "TREE_SEARCH_TEACHER_BASE_URL", "http://localhost:8001"
                )
                teacher_model_name = os.getenv("TREE_SEARCH_TEACHER_MODEL_NAME", "")
                teacher_top_k = int(os.getenv("TREE_SEARCH_TEACHER_TOP_K", "10"))
                teacher_max_retries = int(
                    os.getenv("TREE_SEARCH_TEACHER_MAX_RETRIES", "3")
                )
                teacher_timeout = float(
                    os.getenv("TREE_SEARCH_TEACHER_TIMEOUT", "60.0")
                )
                teacher_missing_logprob = float(
                    os.getenv("TREE_SEARCH_TEACHER_MISSING_LOGPROB", "-23.0")
                )
                diagnose_model_name = os.getenv(
                    "TREE_SEARCH_DIAGNOSE_MODEL_NAME", teacher_model_name
                )
                diagnose_max_tokens = int(
                    os.getenv("TREE_SEARCH_DIAGNOSE_MAX_TOKENS", "1024")
                )
                diagnose_temperature = float(
                    os.getenv("TREE_SEARCH_DIAGNOSE_TEMPERATURE", "0.0")
                )
                strict_distill_json = (
                    os.getenv("TREE_SEARCH_STRICT_DISTILL_JSON", "true").lower()
                    == "true"
                )
```

Pass these arguments to `TreeSearchGroupedRolloutWorkflow(...)`:

```python
                    topk_distill=topk_distill,
                    teacher_provider=teacher_provider,
                    teacher_base_url=teacher_base_url,
                    teacher_model_name=teacher_model_name,
                    teacher_top_k=teacher_top_k,
                    teacher_max_retries=teacher_max_retries,
                    teacher_timeout=teacher_timeout,
                    teacher_missing_logprob=teacher_missing_logprob,
                    diagnose_model_name=diagnose_model_name,
                    diagnose_max_tokens=diagnose_max_tokens,
                    diagnose_temperature=diagnose_temperature,
                    strict_distill_json=strict_distill_json,
```

- [ ] **Step 3: Run tests**

Run:

```bash
uv run pytest tests/test_tree_search/test_config.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add areal/infra/remote_inf_engine.py tests/test_tree_search/test_config.py
git commit -m "feat(infra): wire selected-turn distill env config"
```

---

### Task 8: Integration Verification

**Files:**
- Modify tests only if prior tasks reveal broken assumptions.

- [ ] **Step 1: Run tree-search and customized teacher tests**

Run:

```bash
uv run pytest tests/test_tree_search tests/customized_areal/test_teacher_client.py tests/customized_areal/test_reward_compute.py -q
```

Expected: PASS.

- [ ] **Step 2: Run custom distill regression tests**

Run:

```bash
uv run pytest tests/test_distill_bugfixes.py tests/customized_areal/test_teacher_distill_integration.py -q
```

Expected: PASS. If a test asserts the old normalized position-level reward objective, update the assertion to the direct `student_logp - teacher_logp` objective and include the exact expected scalar in the test.

- [ ] **Step 3: Run formatting/lint for changed files**

Run:

```bash
uv run ruff check customized_areal/tree_search tests/test_tree_search tests/customized_areal/test_teacher_client.py
```

Expected: PASS.

- [ ] **Step 4: Run pre-commit on changed files**

Run:

```bash
uv run pre-commit run --files customized_areal/tree_search/distill_types.py customized_areal/tree_search/config.py customized_areal/tree_search/core/teacher_client.py customized_areal/tree_search/core/teacher_provider.py customized_areal/tree_search/core/selected_turn_distill.py customized_areal/tree_search/tree_search_grouped_workflow.py customized_areal/tree_search/training/loss.py areal/infra/remote_inf_engine.py tests/test_tree_search/test_selected_turn_distill.py tests/test_tree_search/test_distill_loss.py tests/test_tree_search/test_config.py tests/customized_areal/test_teacher_client.py
```

Expected: PASS.

- [ ] **Step 5: Commit verification-only test adjustments**

If Step 2 required test assertion updates, commit them:

```bash
git add tests/test_distill_bugfixes.py tests/customized_areal/test_teacher_distill_integration.py
git commit -m "test(tree-search): update distill KL expectations"
```

If no files changed in Task 8, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Class-level tokenizer cache: Task 6.
- External teacher default and engine provider boundary: Task 2.
- Strict diagnosis JSON with turn-wise guidance: Tasks 1 and 3.
- Selected generation span only: Task 3.
- `topk_distill` false and true paths: Tasks 3 and 4.
- Cached top-k/teacher metadata refill: Tasks 4 and 6.
- `position_rewards` and `sample_index` contract: Task 6.
- Direct `student_logp - teacher_logp` KL loss: Task 5.
- Env/config wiring: Task 7.
- Failure policy for `BOTH` and `DISTILL`: Task 6.
- Verification: Task 8.

Red-flag scan: the plan contains no incomplete markers, no open-ended implementation steps, and no unnamed test commands.

Type consistency: `teacher_logprobs`, `DiagnosisTurn`, `EpisodeDiagnosis`, `TeacherProvider`, `ExternalTeacherProvider`, `EngineTeacherProvider`, `selected_turn_to_position_rewards`, and `_compute_teacher_kl_loss` are introduced before later tasks use them.

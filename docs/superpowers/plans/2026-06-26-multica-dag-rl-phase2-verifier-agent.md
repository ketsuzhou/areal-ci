# Phase 2: Verifier Agent — Implementation Plan (Python)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the constant `set_reward(1.0)` with a real verifier that judges task success via objective checks (where available) plus an LLM-judge fallback over `issue.acceptance_criteria`, and assign per-agent + per-step credit at fan-in joins.

**Architecture:** A `Verifier` Protocol with `verify(run) -> VerifierResult` is the seam. `ObjectiveVerifier` runs deterministic checks (test pass/fail, build status, lint) when a check spec is supplied. `LLMJudgeVerifier` is the fallback when objective checks are unavailable or inconclusive, using the existing `verify_answer_llm_simpleqa` / `compute_reward` templates. `CreditAssigner` assigns per-agent credit at fan-in joins explicitly (no fixed sum/mean/max aggregation rule, per design decision 8).

**Tech Stack:** Python 3.12+ · `openai.AsyncOpenAI` (already a dep) · `typing.Protocol` · `pytest` + `pytest-asyncio`

**Design reference:** `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md` §2 decisions 7–8, §4 (verifier.py, credit.py), §5 Phase 2, §6 Testing

**Project rules:** `backend/areal/CLAUDE.md` (Python conventions, no vendor SDK leakage, structured logging), `AGENTS.md` (no premature abstraction)

**Dependencies:** Phase 0 (`ForkableEnvironment`) complete. This phase is pure Python — no Go dependency.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `customized_areal/tree_search/dag/verifier.py` (create) | `Verifier` Protocol, `VerifierResult` dataclass, `ObjectiveVerifier`, `LLMJudgeVerifier` |
| `customized_areal/tree_search/dag/credit.py` (create) | `CreditAssigner` — assigns per-agent + per-step credit at fan-in joins |
| `customized_areal/tree_search/dag/test_verifier.py` (create) | Tests: objective-check success/failure, LLM-judge fallback (mocked), credit at fan-in |
| `customized_areal/tree_search/dag/__init__.py` (modify) | Export verifier + credit types |

Reference patterns (do NOT modify): `customized_areal/tpfc/eval_utils.py::verify_answer_llm_simpleqa` (LLM-judge template), `customized_areal/tpfc/gaia_final_reward.py::compute_reward` (judge-call + fallback pattern).

---

## Task 6: Verifier Protocol + VerifierResult + ObjectiveVerifier

**Files:**
- Create: `customized_areal/tree_search/dag/verifier.py`
- Create: `customized_areal/tree_search/dag/test_verifier.py`

- [ ] **Step 1: Write the failing test for ObjectiveVerifier**

```python
# customized_areal/tree_search/dag/test_verifier.py
"""Tests for the verifier agent.

The verifier judges task success via objective checks where available, with
an LLM-judge fallback over acceptance_criteria. Credit is assigned explicitly
at fan-in joins — no fixed aggregation rule.
"""
from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.verifier import (
    ObjectiveVerifier,
    Verifier,
    VerifierResult,
)


@pytest.mark.asyncio
async def test_objective_verifier_passes_on_success() -> None:
    """When the objective check returns success, the verifier returns a 1.0 reward."""
    verifier = ObjectiveVerifier(
        check=lambda run: True,  # objective check passes
    )
    result = await verifier.verify(run={"task_id": "t1", "check_output": "ok"})
    assert result.success is True
    assert result.reward == 1.0
    assert result.source == "objective"


@pytest.mark.asyncio
async def test_objective_verifier_fails_on_failure() -> None:
    verifier = ObjectiveVerifier(check=lambda run: False)
    result = await verifier.verify(run={"task_id": "t1"})
    assert result.success is False
    assert result.reward == 0.0
    assert result.source == "objective"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/dag/test_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.dag.verifier'`

- [ ] **Step 3: Implement the Protocol + dataclass + ObjectiveVerifier**

```python
# customized_areal/tree_search/dag/verifier.py
"""Verifier agent for multi-agent DAG RL training.

Judges task success via objective checks (deterministic — test pass/fail,
build status, lint) where a check spec is available, with an LLM-judge
fallback over issue.acceptance_criteria when objective checks are
unavailable or inconclusive.

Replaces the constant set_reward(1.0) currently wired into db_bridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from areal.utils import logging

logger = logging.getLogger("Verifier")


@dataclass(frozen=True)
class VerifierResult:
    """Outcome of verifying one agent run.

    ``reward`` is the terminal outcome reward in [0.0, 1.0]; ``success``
    is the boolean interpretation; ``source`` records which path produced
    the result ("objective" or "llm_judge" or "default").
    """

    success: bool
    reward: float
    source: str
    rationale: str = ""
    per_step_signals: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    async def verify(self, run: dict[str, Any]) -> VerifierResult:
        ...


ObjectiveCheck = Callable[[dict[str, Any]], bool]


class ObjectiveVerifier:
    """Runs a deterministic objective check when one is available.

    The check callable receives the run dict and returns True/False. Used
    for cases where task success is decidable by running tests, checking
    build status, etc. — no LLM call needed.
    """

    def __init__(self, *, check: ObjectiveCheck) -> None:
        self._check = check

    async def verify(self, run: dict[str, Any]) -> VerifierResult:
        try:
            ok = bool(self._check(run))
        except Exception as exc:
            logger.warning(
                "objective check raised; treating as failure",
                error=str(exc),
                task_id=run.get("task_id"),
            )
            return VerifierResult(
                success=False, reward=0.0, source="objective",
                rationale=f"check raised: {exc}",
            )
        return VerifierResult(
            success=ok,
            reward=1.0 if ok else 0.0,
            source="objective",
            rationale="objective check" if ok else "objective check failed",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/verifier.py customized_areal/tree_search/dag/test_verifier.py
git commit -m "feat(dag): add Verifier Protocol, VerifierResult, and ObjectiveVerifier"
```

---

## Task 7: LLMJudgeVerifier — fallback over acceptance_criteria

**Files:**
- Modify: `customized_areal/tree_search/dag/verifier.py`
- Modify: `customized_areal/tree_search/dag/test_verifier.py`

- [ ] **Step 1: Write the failing test for LLMJudgeVerifier (with mocked OpenAI client)**

```python
from customized_areal.tree_search.dag.verifier import LLMJudgeVerifier


class _FakeOpenAI:
    """Fake AsyncOpenAI — returns a canned judge response."""

    def __init__(self, *, judge_response: str) -> None:
        self._judge_response = judge_response
        self.calls: list[dict[str, Any]] = []

    class _Completions:
        def __init__(self, fake: "_FakeOpenAI") -> None:
            self._fake = fake

        async def create(self, **kwargs: Any) -> Any:
            self._fake.calls.append(kwargs)
            choice = type("Choice", (), {"message": type("Msg", (), {"content": self._fake._judge_response})()})()
            return type("Resp", (), {"choices": [choice]})()

    @property
    def chat(self) -> Any:
        return type("ChatNamespace", (), {"completions": self._Completions(self)})()


@pytest.mark.asyncio
async def test_llm_judge_verifier_returns_correct_on_judge_yes() -> None:
    """When the LLM judge says 'yes', the verifier returns success."""
    fake = _FakeOpenAI(judge_response="yes")
    verifier = LLMJudgeVerifier(
        openai_client=fake,
        model_name="deepseek/deepseek-v4-flash",
    )
    result = await verifier.verify(
        run={
            "task_id": "t1",
            "question": "What is 2+2?",
            "predicted_answer": "4",
            "acceptance_criteria": ["The answer must be 4"],
        }
    )
    assert result.success is True
    assert result.reward == 1.0
    assert result.source == "llm_judge"


@pytest.mark.asyncio
async def test_llm_judge_verifier_returns_incorrect_on_judge_no() -> None:
    fake = _FakeOpenAI(judge_response="no")
    verifier = LLMJudgeVerifier(
        openai_client=fake,
        model_name="deepseek/deepseek-v4-flash",
    )
    result = await verifier.verify(
        run={
            "task_id": "t1",
            "question": "What is 2+2?",
            "predicted_answer": "5",
            "acceptance_criteria": ["The answer must be 4"],
        }
    )
    assert result.success is False
    assert result.reward == 0.0
    assert result.source == "llm_judge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py -k "llm_judge" -v`
Expected: FAIL — `LLMJudgeVerifier` undefined.

- [ ] **Step 3: Implement LLMJudgeVerifier**

Add to `verifier.py`:

```python
class LLMJudgeVerifier:
    """LLM-judge fallback over issue.acceptance_criteria.

    Used when objective checks are unavailable or inconclusive. Reuses the
    simple-qa judge prompt pattern from
    customized_areal/tpfc/eval_utils.py::verify_answer_llm_simpleqa — a
    yes/no judge that decides if the predicted answer satisfies the
    acceptance_criteria.
    """

    JUDGE_PROMPT = """Judge whether the predicted answer satisfies the acceptance criteria.

Question: {question}
Predicted answer: {predicted_answer}
Acceptance criteria:
{criteria}

Answer 'yes' if the predicted answer satisfies ALL criteria, 'no' otherwise.
Reply with only 'yes' or 'no'."""

    def __init__(
        self,
        *,
        openai_client: Any,
        model_name: str,
        max_completion_tokens: int = 2,
    ) -> None:
        self._client = openai_client
        self._model_name = model_name
        self._max_tokens = max_completion_tokens

    async def verify(self, run: dict[str, Any]) -> VerifierResult:
        question = str(run.get("question", ""))
        predicted = str(run.get("predicted_answer", ""))
        criteria = run.get("acceptance_criteria") or []
        if not isinstance(criteria, list):
            criteria = [str(criteria)]
        criteria_text = "\n".join(f"- {c}" for c in criteria) or "- (no criteria supplied)"

        prompt = self.JUDGE_PROMPT.format(
            question=question,
            predicted_answer=predicted,
            criteria=criteria_text,
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self._max_tokens,
            )
            content = resp.choices[0].message.content
            content_str = content if isinstance(content, str) else ""
            ok = "yes" in content_str.strip().lower()
        except Exception as exc:
            logger.warning(
                "LLM judge call failed; treating as not-attempted",
                error=str(exc),
                task_id=run.get("task_id"),
            )
            return VerifierResult(
                success=False, reward=0.0, source="llm_judge",
                rationale=f"judge call failed: {exc}",
            )
        return VerifierResult(
            success=ok,
            reward=1.0 if ok else 0.0,
            source="llm_judge",
            rationale=f"judge said {'yes' if ok else 'no'}",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py -k "llm_judge" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/verifier.py customized_areal/tree_search/dag/test_verifier.py
git commit -m "feat(dag): add LLMJudgeVerifier fallback over acceptance_criteria"
```

---

## Task 8: Composite verifier (objective → LLM-judge fallback chain)

**Files:**
- Modify: `customized_areal/tree_search/dag/verifier.py`
- Modify: `customized_areal/tree_search/dag/test_verifier.py`

- [ ] **Step 1: Write the failing test for the composite fallback**

```python
from customized_areal.tree_search.dag.verifier import CompositeVerifier


@pytest.mark.asyncio
async def test_composite_verifier_uses_objective_when_available() -> None:
    """When the objective check succeeds, the LLM-judge is never called."""
    objective = ObjectiveVerifier(check=lambda run: True)
    llm_judge_calls: list[bool] = []

    class _NeverCalledLLMJudge:
        async def verify(self, run: dict[str, Any]) -> VerifierResult:
            llm_judge_calls.append(True)
            return VerifierResult(success=True, reward=1.0, source="llm_judge")

    composite = CompositeVerifier(objective=objective, fallback=_NeverCalledLLMJudge())
    result = await composite.verify(run={"task_id": "t1"})
    assert result.source == "objective"
    assert llm_judge_calls == []  # fallback not invoked


@pytest.mark.asyncio
async def test_composite_verifier_falls_back_when_objective_unavailable() -> None:
    """When objective is None (unavailable), the LLM-judge is used."""
    llm_judge = LLMJudgeVerifier(
        openai_client=_FakeOpenAI(judge_response="yes"),
        model_name="deepseek/deepseek-v4-flash",
    )
    composite = CompositeVerifier(objective=None, fallback=llm_judge)
    result = await composite.verify(
        run={
            "task_id": "t1", "question": "q", "predicted_answer": "a",
            "acceptance_criteria": ["c"],
        }
    )
    assert result.source == "llm_judge"
    assert result.success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py -k "composite" -v`
Expected: FAIL — `CompositeVerifier` undefined.

- [ ] **Step 3: Implement CompositeVerifier**

Add to `verifier.py`:

```python
class CompositeVerifier:
    """Try objective first; fall back to LLM-judge when objective is unavailable.

    "Unavailable" means objective is None — when the objective check itself
    raises, ObjectiveVerifier already returns a failure result with
    source="objective"; that does NOT trigger the fallback. The fallback
    fires only when no objective check was supplied at all.
    """

    def __init__(self, *, objective: ObjectiveVerifier | None, fallback: Verifier) -> None:
        self._objective = objective
        self._fallback = fallback

    async def verify(self, run: dict[str, Any]) -> VerifierResult:
        if self._objective is not None:
            return await self._objective.verify(run)
        return await self._fallback.verify(run)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py -k "composite" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/verifier.py customized_areal/tree_search/dag/test_verifier.py
git commit -m "feat(dag): add CompositeVerifier with objective -> LLM-judge fallback chain"
```

---

## Task 9: CreditAssigner — per-agent credit at fan-in joins

**Files:**
- Create: `customized_areal/tree_search/dag/credit.py`
- Create: `customized_areal/tree_search/dag/test_credit.py`

**Rationale:** Per design decision 8, at fan-in joins (where ≥2 upstream runs feed one downstream run), the verifier assigns per-agent credit explicitly — there is no fixed sum/mean/max aggregation rule. The `CreditAssigner` takes the DAG + verifier results and produces per-node credit weights.

- [ ] **Step 1: Write the failing test for credit assignment at a fan-in**

```python
# customized_areal/tree_search/dag/test_credit.py
"""Tests for per-agent credit assignment at fan-in joins.

At a join (>= 2 upstream runs feeding one downstream), credit is assigned
explicitly by the verifier — not summed/averaged/maxed. This test exercises
the simplest case: 2 upstream runs, one contributes more.
"""
from __future__ import annotations

from customized_areal.tree_search.dag.credit import CreditAssigner
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    EdgeType,
    ExecutionDAG,
)
from customized_areal.tree_search.dag.verifier import VerifierResult


def _make_dag_with_join() -> tuple[ExecutionDAG, dict[str, VerifierResult]]:
    """A DAG: root -> {A, B} -> join_node. Two runs fan into the join."""
    dag = ExecutionDAG()
    root = AgentRunNode(node_id="root", agent_id="ag1", issue_id="i1", task_id="t1")
    a = AgentRunNode(node_id="A", agent_id="ag2", issue_id="i2", task_id="t2")
    b = AgentRunNode(node_id="B", agent_id="ag3", issue_id="i3", task_id="t3")
    join = AgentRunNode(node_id="join", agent_id="ag4", issue_id="i4", task_id="t4")
    dag.add_node(root)
    dag.add_node(a)
    dag.add_node(b)
    dag.add_node(join)
    dag.add_edge("root", "A", EdgeType.DELEGATION)
    dag.add_edge("root", "B", EdgeType.DELEGATION)
    dag.add_edge("A", "join", EdgeType.COMPLETION)
    dag.add_edge("B", "join", EdgeType.COMPLETION)

    # Verifier rewards: A succeeded (1.0), B failed (0.0). The join's
    # outcome_reward is the terminal reward (1.0 — the task succeeded).
    verifier_results = {
        "A": VerifierResult(success=True, reward=1.0, source="objective"),
        "B": VerifierResult(success=False, reward=0.0, source="objective"),
        "join": VerifierResult(success=True, reward=1.0, source="objective"),
    }
    return dag, verifier_results


def test_credit_assigner_distributes_terminal_reward_at_join() -> None:
    """The terminal reward (1.0) is distributed across the 2 upstream runs
    according to their verifier rewards — A gets full credit, B gets none.
    No fixed aggregation rule; the verifier's per-run reward is the weight.
    """
    dag, results = _make_dag_with_join()
    assigner = CreditAssigner()
    credit = assigner.assign(dag, results)

    # join's terminal reward is 1.0; A contributed 1.0 of the upstream reward,
    # B contributed 0.0. Distribution: A gets 1.0 * (1.0 / (1.0 + 0.0)) = 1.0.
    assert credit["A"] == pytest.approx(1.0)
    assert credit["B"] == pytest.approx(0.0)
    # The join itself keeps its terminal reward.
    assert credit["join"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_credit.py -v`
Expected: FAIL — `CreditAssigner` undefined.

- [ ] **Step 3: Implement CreditAssigner**

```python
# customized_areal/tree_search/dag/credit.py
"""Per-agent + per-step credit assignment at fan-in joins.

At a join (>= 2 upstream runs feeding one downstream), credit is assigned
explicitly — no fixed sum/mean/max aggregation rule. The verifier's
per-run reward is the weight used to distribute the terminal outcome
reward across the contributing runs.

This implements design decision 8: "Fan-in joins: the verifier assigns
per-agent credit explicitly".
"""
from __future__ import annotations

from typing import Any

from customized_areal.tree_search.dag.execution_dag import ExecutionDAG
from customized_areal.tree_search.dag.verifier import VerifierResult

from areal.utils import logging

logger = logging.getLogger("CreditAssigner")


class CreditAssigner:
    """Distributes terminal outcome reward across contributing runs at joins.

    For each node N with a terminal outcome_reward R_N and in-degree >= 2,
    distributes R_N across the upstream runs weighted by their own verifier
    rewards. If all upstream rewards are zero, distributes uniformly (rare
    edge case — documented but not a fixed aggregation rule, just a
    tie-breaker that avoids division-by-zero).
    """

    def assign(
        self,
        dag: ExecutionDAG,
        verifier_results: dict[str, VerifierResult],
    ) -> dict[str, float]:
        credit: dict[str, float] = {}
        # Walk in topological order so a node's credit is finalized before
        # its downstream consumers need it.
        for node in dag.topological_order():
            terminal = verifier_results.get(node.node_id)
            base = terminal.reward if terminal else 0.0
            # Leaf / non-join nodes keep their terminal reward as-is.
            if dag.in_degree(node.node_id) < 2:
                credit[node.node_id] = base
                continue
            # Fan-in: distribute base across upstream runs by their reward weight.
            upstream = dag.parents(node.node_id)
            weights = [verifier_results.get(p.node_id).reward if verifier_results.get(p.node_id) else 0.0 for p in upstream]
            total = sum(weights)
            if total <= 0.0:
                # All upstream runs failed; distribute uniformly as a tie-breaker.
                share = base / len(upstream) if upstream else 0.0
                for p in upstream:
                    credit[p.node_id] = credit.get(p.node_id, 0.0) + share
            else:
                for p, w in zip(upstream, weights):
                    credit[p.node_id] = credit.get(p.node_id, 0.0) + base * (w / total)
            credit[node.node_id] = base
        return credit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_credit.py -v`
Expected: PASS

- [ ] **Step 5: Add `import pytest` to `test_credit.py`** (needed for `pytest.approx`).

- [ ] **Step 6: Run test again to verify it still passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_credit.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/dag/credit.py customized_areal/tree_search/dag/test_credit.py
git commit -m "feat(dag): add CreditAssigner for per-agent credit at fan-in joins"
```

---

## Task 10: Per-step credit signals (process signals shape intermediate steps)

**Files:**
- Modify: `customized_areal/tree_search/dag/credit.py`
- Modify: `customized_areal/tree_search/dag/test_credit.py`

**Rationale:** Per design decision 6, reward is hybrid — terminal outcome reward + per-node process signals. The terminal reward is distributed at joins (Task 9); process signals shape intermediate steps. This task adds the per-step signal plumbing (a `dict[str, float]` of step_id → signal) to `VerifierResult` consumption.

- [ ] **Step 1: Write the failing test for per-step signals**

```python
def test_credit_assigner_carries_per_step_signals() -> None:
    """per_step_signals from the verifier result land on the credit dict
    as node_id -> signal_value, separate from the terminal reward distribution.
    """
    dag, results = _make_dag_with_join()
    # Override: A has per-step signals.
    results["A"] = VerifierResult(
        success=True, reward=1.0, source="objective",
        per_step_signals={"step_1": 0.5, "step_2": 0.8},
    )
    assigner = CreditAssigner()
    credit = assigner.assign(dag, results)
    # Per-step signals are surfaced under a separate key namespace so the
    # advantage computer can pick them up without conflating with terminal credit.
    assert credit.get("A#step_1") == pytest.approx(0.5)
    assert credit.get("A#step_2") == pytest.approx(0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_credit.py::test_credit_assigner_carries_per_step_signals -v`
Expected: FAIL — `A#step_1` not in credit dict.

- [ ] **Step 3: Extend CreditAssigner to surface per-step signals**

Modify `assign()` — after computing `credit[node.node_id]`, add per-step signal surfacing:

```python
            # Surface per-step signals under "{node_id}#{step_id}" keys so
            # the advantage computer can pick them up without conflating
            # with the terminal reward distribution.
            if terminal is not None and terminal.per_step_signals:
                for step_id, signal in terminal.per_step_signals.items():
                    credit[f"{node.node_id}#{step_id}"] = float(signal)
```

Place this inside the `for node in dag.topological_order():` loop, after the `credit[node.node_id] = base` line.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_credit.py::test_credit_assigner_carries_per_step_signals -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/credit.py customized_areal/tree_search/dag/test_credit.py
git commit -m "feat(dag): surface per-step process signals on CreditAssigner output"
```

---

## Task 11: Export from dag/__init__.py + full suite run

**Files:**
- Modify: `customized_areal/tree_search/dag/__init__.py`

- [ ] **Step 1: Write the failing test for exports**

```python
# customized_areal/tree_search/dag/test_verifier.py (append)
def test_dag_package_exports_verifier_types() -> None:
    from customized_areal.tree_search.dag import (
        CompositeVerifier,
        CreditAssigner,
        LLMJudgeVerifier,
        ObjectiveVerifier,
        Verifier,
        VerifierResult,
    )
    assert all(
        cls is not None
        for cls in [CompositeVerifier, CreditAssigner, LLMJudgeVerifier, ObjectiveVerifier, Verifier, VerifierResult]
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py::test_dag_package_exports_verifier_types -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add the exports**

Modify `customized_areal/tree_search/dag/__init__.py`:

```python
from customized_areal.tree_search.dag.credit import CreditAssigner
from customized_areal.tree_search.dag.verifier import (
    CompositeVerifier,
    LLMJudgeVerifier,
    ObjectiveVerifier,
    Verifier,
    VerifierResult,
)
```

And extend `__all__`:

```python
    # verifier + credit
    "CompositeVerifier",
    "CreditAssigner",
    "LLMJudgeVerifier",
    "ObjectiveVerifier",
    "Verifier",
    "VerifierResult",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_verifier.py::test_dag_package_exports_verifier_types -v`
Expected: PASS

- [ ] **Step 5: Run the full DAG test suite**

Run: `uv run pytest customized_areal/tree_search/dag/ -v`
Expected: All tests PASS — execution_dag, environment, verifier, credit.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/dag/__init__.py
git commit -m "feat(dag): export Verifier, CreditAssigner, and friends from dag package"
```

---

## Task 12: Pre-commit + lint

**Files:** No code changes — verification step.

- [ ] **Step 1: Run ruff check**

Run: `cd /workspaces/leagent/backend/areal && uv run ruff check customized_areal/tree_search/dag/verifier.py customized_areal/tree_search/dag/credit.py customized_areal/tree_search/dag/test_verifier.py customized_areal/tree_search/dag/test_credit.py customized_areal/tree_search/dag/__init__.py`
Expected: No errors.

- [ ] **Step 2: Run ruff format check**

Run: `uv run ruff format --check customized_areal/tree_search/dag/`
Expected: No reformatting needed.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore(dag): ruff fixes for verifier and credit modules"
```

---

## Self-Review Notes

**Spec coverage:**
- Design §2 decision 7 (Verifier = objective + LLM-judge fallback) → Tasks 6, 7, 8
- Design §2 decision 8 (Fan-in joins: verifier assigns per-agent credit explicitly) → Tasks 9, 10
- Design §2 decision 6 (Reward is hybrid — terminal + per-step process signals) → Task 10
- Design §4 (verifier.py, credit.py new files) → all tasks
- Design §5 Phase 2 (Tasks 6, 7, 8) → all tasks
- Design §6 (Verifier tests: objective success/failure, LLM-judge fallback mocked, credit at fan-in) → Tasks 6, 7, 9

**Placeholder scan:** None. Every step has concrete code or commands.

**Type consistency:**
- `VerifierResult(success, reward, source, rationale, per_step_signals)` is used consistently in Tasks 6, 7, 8, 9, 10. The `per_step_signals: dict[str, float]` field is added in Task 6 and consumed in Task 10.
- `Verifier` Protocol method signature `async def verify(self, run: dict[str, Any]) -> VerifierResult` is consistent across ObjectiveVerifier, LLMJudgeVerifier, CompositeVerifier.
- `CreditAssigner.assign(dag, verifier_results) -> dict[str, float]` signature consistent across Tasks 9, 10.

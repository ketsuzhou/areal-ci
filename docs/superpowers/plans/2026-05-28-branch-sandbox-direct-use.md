# Branch Sandbox Direct Use + Subproc Metadata Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix subproc metadata loss for task_id/raw_messages and eliminate redundant sandbox cloning in branch episodes.

**Architecture:** TPFCAgent returns a structured `TPFCAgentResult` dataclass instead of mutating `data`, which survives subprocess pickling. `OpenAIProxyWorkflow.arun_episode` unpacks it and propagates metadata to `data` in the parent process. `build_branch_task` uses `branch_sandbox_id` directly (no clone), and `_cleanup_branch` deletes the sandbox and marks the node after the branch episode finishes.

**Tech Stack:** Python 3.12+, asyncio, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `customized_areal/tpfc/tpfc_agent.py` | Modify | Add `TPFCAgentResult`, return it from `run()` |
| `areal/experimental/openai/proxy/workflow.py` | Modify | Handle `TPFCAgentResult` in `arun_episode` |
| `customized_areal/tree_search/tree_search_grouped_workflow.py` | Modify | Remove clone from `build_branch_task`, add `_cleanup_branch`, update `_run_fresh_episode` |
| `tests/customized_areal/test_tree_search_branch_sampling.py` | Modify | Update existing tests, add new tests |

---

### Task 1: Add `TPFCAgentResult` dataclass and update `TPFCAgent.run()`

**Files:**
- Modify: `customized_areal/tpfc/tpfc_agent.py:1-198`
- Test: `tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing test for TPFCAgentResult pickling**

Add to `tests/customized_areal/test_tree_search_branch_sampling.py`:

```python
import pickle

from customized_areal.tpfc.tpfc_agent import TPFCAgentResult


def test_tpfca_agent_result_is_picklable():
    result = TPFCAgentResult(
        reward=0.75,
        task_id="task-123",
        raw_messages=[{"role": "assistant", "content": "hi"}],
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.reward == 0.75
    assert restored.task_id == "task-123"
    assert restored.raw_messages == [{"role": "assistant", "content": "hi"}]


def test_tpfca_agent_result_default_fields():
    result = TPFCAgentResult(reward=0.5)
    assert result.task_id == ""
    assert result.raw_messages == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_tpfca_agent_result_is_picklable tests/customized_areal/test_tree_search_branch_sampling.py::test_tpfca_agent_result_default_fields -v`
Expected: FAIL — `ImportError: cannot import name 'TPFCAgentResult'`

- [ ] **Step 3: Add TPFCAgentResult dataclass to tpfc_agent.py**

At the top of `customized_areal/tpfc/tpfc_agent.py`, add after the existing imports (after `from areal.utils import logging`):

```python
from dataclasses import dataclass, field


@dataclass
class TPFCAgentResult:
    """Structured result from TPFCAgent.run() that survives subprocess pickling.

    Unlike data dict mutations (which are lost across process boundaries),
    this return value is picklable and propagates task_id and raw_messages
    back to the parent process in subproc mode.
    """

    reward: float
    task_id: str = ""
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
```

Note: `from dataclasses import dataclass` is not yet imported in this file. Add `dataclass, field` to a new import or to the existing typing imports. The file already imports `Any` from `typing` indirectly (via the `pathlib` and other imports). Check that `Any` is available — it's not explicitly imported. Add `from typing import Any` at the top.

Then modify `TPFCAgent.run()` to return `TPFCAgentResult` instead of mutating `data`:

Replace lines 156-157 and the return in `_do_run()`:

```python
# Before (lines 156-157):
data["_backend_run_task_id"] = run_result.task_id
data["_backend_run_raw_messages"] = run_result.raw_messages

# After (delete those two lines, change the return):
```

Change the return at the end of `_do_run()` (currently `return float(reward)`):

```python
return TPFCAgentResult(
    reward=float(reward),
    task_id=run_result.task_id,
    raw_messages=run_result.raw_messages,
)
```

Also update the outer return type and timeout handling in `run()` (currently `-> float`). Change the method signature:

```python
async def run(
    self,
    data: dict[str, Any],
    **extra_kwargs,
) -> TPFCAgentResult:
```

And the timeout wrapper at the bottom of `run()` — currently:

```python
return await asyncio.wait_for(_do_run(), timeout=1500)
```

This already returns whatever `_do_run()` returns, so no change needed there.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_tpfca_agent_result_is_picklable tests/customized_areal/test_tree_search_branch_sampling.py::test_tpfca_agent_result_default_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tpfc/tpfc_agent.py tests/customized_areal/test_tree_search_branch_sampling.py
git commit -m "feat(tpfc): add TPFCAgentResult dataclass, return from run() instead of mutating data"
```

---

### Task 2: Handle `TPFCAgentResult` in `OpenAIProxyWorkflow.arun_episode`

**Files:**
- Modify: `areal/experimental/openai/proxy/workflow.py:215-252`
- Test: `tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing test for TPFCAgentResult handling in arun_episode**

This test verifies that when an agent returns `TPFCAgentResult`, the metadata is propagated to `data` and the reward is correctly assigned. Since `arun_episode` is complex and depends on proxy infrastructure, we test the data-propagation logic via the existing `_run_fresh_episode` / `_with_episode_metadata` path instead.

Add to `tests/customized_areal/test_tree_search_branch_sampling.py`:

```python
from customized_areal.tpfc.tpfc_agent import TPFCAgentResult


def test_with_episode_metadata_extracts_tpfca_result():
    """_with_episode_metadata should read _backend_run_task_id and _backend_run_raw_messages from data."""
    from customized_areal.tree_search.tree_search_grouped_workflow import (
        EpisodeRunResult,
        _with_episode_metadata,
    )

    data = {
        "_backend_run_task_id": "task-xyz",
        "_backend_run_raw_messages": [{"role": "assistant", "content": "hi"}],
    }
    result = _with_episode_metadata("some_result", data)
    assert isinstance(result, EpisodeRunResult)
    assert result.task_id == "task-xyz"
    assert result.raw_messages == [{"role": "assistant", "content": "hi"}]
    assert result.result == "some_result"
```

This test should already pass (existing functionality). The real test is that after we add `TPFCAgentResult` handling in `arun_episode`, the data dict gets populated correctly in both inline and subproc modes. Since testing `arun_episode` end-to-end requires the full proxy server, we add a focused unit test for the data-propagation logic:

```python
def test_tpfca_agent_result_propagates_to_data_dict():
    """Simulate what arun_episode does when it receives TPFCAgentResult."""
    data = {"query_id": "q1"}
    rewards = TPFCAgentResult(
        reward=0.9,
        task_id="task-abc",
        raw_messages=[{"role": "assistant", "content": "response"}],
    )

    # This is the duck-typing logic we'll add to arun_episode:
    if hasattr(rewards, "task_id") and hasattr(rewards, "raw_messages"):
        data["_backend_run_task_id"] = rewards.task_id
        data["_backend_run_raw_messages"] = rewards.raw_messages
        rewards = rewards.reward

    assert rewards == 0.9
    assert data["_backend_run_task_id"] == "task-abc"
    assert data["_backend_run_raw_messages"] == [{"role": "assistant", "content": "response"}]
```

- [ ] **Step 2: Run test to verify it passes (data propagation test should pass since it's self-contained)**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_with_episode_metadata_extracts_tpfca_result tests/customized_areal/test_tree_search_branch_sampling.py::test_tpfca_agent_result_propagates_to_data_dict -v`
Expected: PASS (the propagation test is self-contained; the metadata test uses existing code)

- [ ] **Step 3: Add TPFCAgentResult handling to `arun_episode`**

In `areal/experimental/openai/proxy/workflow.py`, modify the `arun_episode` method. After line 226 (`rewards = await self._run_agent(...)`), add the structured result unpacking.

**Important:** Do NOT import `TPFCAgentResult` from `customized_areal` — `areal` must not depend on `customized_areal`. Instead, use duck-typing: check for the `task_id` and `raw_messages` attributes that `TPFCAgentResult` exposes.

Find this block (lines 225-238):

```python
            # Run the user code.
            try:
                rewards = await self._run_agent(proxy_client.session_api_key, data)
            except Exception:
                logger.warning("Agent task failed. This trajectory will be rejected.")
                raise

            # Assign rewards back according to user code output
            if isinstance(rewards, dict):
```

Replace with:

```python
            # Run the user code.
            try:
                rewards = await self._run_agent(proxy_client.session_api_key, data)
            except Exception:
                logger.warning("Agent task failed. This trajectory will be rejected.")
                raise

            # Unpack structured result (e.g. TPFCAgentResult) so metadata
            # propagates to data dict in the parent process — critical for
            # subproc mode where data mutations are lost across process boundary.
            # Use duck-typing to avoid areal -> customized_areal import dependency.
            if hasattr(rewards, "task_id") and hasattr(rewards, "raw_messages"):
                data["_backend_run_task_id"] = rewards.task_id
                data["_backend_run_raw_messages"] = rewards.raw_messages
                rewards = rewards.reward

            # Assign rewards back according to user code output
            if isinstance(rewards, dict):
```

No new imports needed — `hasattr` is a builtin.

- [ ] **Step 4: Run all tree search branch tests**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add areal/experimental/openai/proxy/workflow.py tests/customized_areal/test_tree_search_branch_sampling.py
git commit -m "feat(proxy): handle TPFCAgentResult in arun_episode for subproc metadata propagation"
```

---

### Task 3: Modify `build_branch_task` to use `branch_sandbox_id` directly (no clone)

**Files:**
- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py:111-156`
- Test: `tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing test for direct sandbox use in build_branch_task**

Add to `tests/customized_areal/test_tree_search_branch_sampling.py`:

```python
@pytest.mark.asyncio
async def test_build_branch_task_uses_sandbox_directly_without_clone(monkeypatch):
    """build_branch_task should bind candidate.branch_sandbox_id directly, not clone it."""
    created = []
    copied = []

    class FakeClient:
        pass

    candidate = _node(2)
    candidate.task_id = "source-task"
    candidate.turn_idx = 2
    candidate.branch_sandbox_id = "direct-sandbox-id"

    raw_messages = [
        {
            "message_id": "u1",
            "role": "user",
            "content": {"role": "user", "content": "q"},
        },
        {
            "message_id": "a1",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step1"},
        },
        {
            "message_id": "a2",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step2"},
        },
    ]

    async def fake_get_raw_messages(client, task_id):
        assert isinstance(client, FakeClient)
        assert task_id == "source-task"
        return raw_messages

    async def fake_create_task(**kwargs):
        created.append(kwargs)
        return "branch-task"

    async def fake_bind_sandbox_to_task(client, *, sandbox_id, task_id, account_id):
        assert isinstance(client, FakeClient)
        copied.append(("bind", sandbox_id, task_id, account_id))

    async def fake_copy_messages_to_task(client, *, task_id, messages):
        assert isinstance(client, FakeClient)
        copied.append(("messages", task_id, messages))

    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow._get_raw_messages_with_client",
        fake_get_raw_messages,
    )
    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.create_task",
        fake_create_task,
    )
    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.bind_sandbox_to_task",
        fake_bind_sandbox_to_task,
    )
    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.copy_messages_to_task",
        fake_copy_messages_to_task,
    )

    branch_task_id = await build_branch_task(
        client=FakeClient(),
        account_id="account",
        agent_id="agent",
        candidate=candidate,
        name="branch",
    )

    assert branch_task_id == "branch-task"
    assert created[0]["account_id"] == "account"
    assert created[0]["agent_id"] == "agent"
    assert created[0]["name"] == "branch"
    # Key assertion: sandbox bound is the DIRECT sandbox, not a clone
    assert copied[0] == ("bind", "direct-sandbox-id", "branch-task", "account")
    assert copied[1] == (
        "messages",
        "branch-task",
        [
            {"role": "user", "content": {"role": "user", "content": "q"}},
            {
                "role": "assistant",
                "content": {"role": "assistant", "content": "step1"},
            },
        ],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_build_branch_task_uses_sandbox_directly_without_clone -v`
Expected: FAIL — the existing code clones the sandbox, so `copied[0]` will have a different sandbox_id than `"direct-sandbox-id"`

- [ ] **Step 3: Modify `build_branch_task` to use sandbox directly**

In `customized_areal/tree_search/tree_search_grouped_workflow.py`, replace the `build_branch_task` function (lines 111-156) with:

```python
async def build_branch_task(
    *,
    client: Any,
    account_id: str,
    agent_id: str,
    candidate: Node,
    name: str | None,
) -> str | None:
    if not candidate.task_id or not candidate.branch_sandbox_id:
        return None

    raw_messages = await _get_raw_messages_with_client(client, candidate.task_id)
    prefix = truncate_messages_before_turn(raw_messages, candidate.turn_idx)
    branch_sandbox_id = candidate.branch_sandbox_id

    branch_task_id = await create_task(
        client=client,
        account_id=account_id,
        agent_id=agent_id,
        name=name,
    )
    await bind_sandbox_to_task(
        client,
        sandbox_id=branch_sandbox_id,
        task_id=branch_task_id,
        account_id=account_id,
    )
    await copy_messages_to_task(client, task_id=branch_task_id, messages=prefix)

    return branch_task_id
```

Key changes:
- Removed `clone_sandbox` call and its fallback
- Removed `try/except` with sandbox cleanup on error (no cloned sandbox to clean up)
- Use `candidate.branch_sandbox_id` directly

Also remove the now-unused `clone_sandbox` import from line 30:

```python
# Before:
from customized_areal.db_service import (
    bind_sandbox_to_task,
    clone_sandbox,
    copy_messages_to_task,
    create_task,
    delete_sandbox,
    truncate_messages_before_turn,
)

# After:
from customized_areal.db_service import (
    bind_sandbox_to_task,
    copy_messages_to_task,
    create_task,
    delete_sandbox,
    truncate_messages_before_turn,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_build_branch_task_uses_sandbox_directly_without_clone -v`
Expected: PASS

- [ ] **Step 5: Update the old clone-based test to match new behavior**

The existing `test_build_branch_task_clones_sandbox_and_copies_prefix` (line 320) now fails because `clone_sandbox` is no longer called. Replace it with the new test above, or update the assertions. Since the new test covers the same scenario with the correct expectations, remove the old test.

Delete `test_build_branch_task_clones_sandbox_and_copies_prefix` from the test file (lines 319-415).

- [ ] **Step 6: Run all branch sampling tests**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py tests/customized_areal/test_tree_search_branch_sampling.py
git commit -m "feat(tree-search): use branch_sandbox_id directly in build_branch_task, remove clone"
```

---

### Task 4: Add `_cleanup_branch` and update `_run_fresh_episode`

**Files:**
- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py:818-892`
- Test: `tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing test for `_cleanup_branch`**

Add to `tests/customized_areal/test_tree_search_branch_sampling.py`:

```python
@pytest.mark.asyncio
async def test_cleanup_branch_deletes_sandbox_and_clears_node_state(monkeypatch):
    deleted_ids = []

    async def fake_delete_sandbox(sandbox_id):
        deleted_ids.append(sandbox_id)

    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.delete_sandbox",
        fake_delete_sandbox,
    )

    workflow = TreeSearchGroupedRolloutWorkflow.__new__(TreeSearchGroupedRolloutWorkflow)
    candidate = _node(2)
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-to-delete"

    await workflow._cleanup_branch(candidate)

    assert deleted_ids == ["sb-to-delete"]
    assert candidate.need_branch is False
    assert candidate.branch_sandbox_id is None


@pytest.mark.asyncio
async def test_cleanup_branch_tolerates_delete_failure(monkeypatch):
    async def failing_delete(sandbox_id):
        raise RuntimeError("sandbox API down")

    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.delete_sandbox",
        failing_delete,
    )

    workflow = TreeSearchGroupedRolloutWorkflow.__new__(TreeSearchGroupedRolloutWorkflow)
    candidate = _node(2)
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-failing"

    await workflow._cleanup_branch(candidate)

    # Node state should still be cleared even if delete fails
    assert candidate.need_branch is False
    assert candidate.branch_sandbox_id is None


@pytest.mark.asyncio
async def test_cleanup_branch_skips_delete_when_no_sandbox_id():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(TreeSearchGroupedRolloutWorkflow)
    candidate = _node(2)
    candidate.need_branch = True
    candidate.branch_sandbox_id = None

    await workflow._cleanup_branch(candidate)

    assert candidate.need_branch is False
    assert candidate.branch_sandbox_id is None


def test_select_branch_candidate_ignores_cleaned_up_node():
    """After _cleanup_branch, node should not be a branch candidate."""
    cleaned = _node(2)
    cleaned.query_id = "q"
    cleaned.task_id = "task-cleaned"
    cleaned.need_branch = False  # cleared by _cleanup_branch
    cleaned.branch_sandbox_id = None  # cleared by _cleanup_branch
    cleaned.entropy_stats = {"max_entropy": 5.0}

    assert select_branch_candidate([cleaned], "q") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_cleanup_branch_deletes_sandbox_and_clears_node_state -v`
Expected: FAIL — `AttributeError: 'TreeSearchGroupedRolloutWorkflow' object has no attribute '_cleanup_branch'`

- [ ] **Step 3: Add `_cleanup_branch` method to `TreeSearchGroupedRolloutWorkflow`**

In `customized_areal/tree_search/tree_search_grouped_workflow.py`, add the method after `_prepare_branch_task` (after line 892):

```python
    async def _cleanup_branch(self, candidate: Node) -> None:
        """Delete branch sandbox and mark node as branched to prevent re-use."""
        if candidate.branch_sandbox_id:
            try:
                await delete_sandbox(candidate.branch_sandbox_id)
            except Exception:
                logger.warning(
                    "Failed to delete branch sandbox_id=%s",
                    candidate.branch_sandbox_id,
                    exc_info=True,
                )
        candidate.need_branch = False
        candidate.branch_sandbox_id = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_cleanup_branch_deletes_sandbox_and_clears_node_state tests/customized_areal/test_tree_search_branch_sampling.py::test_cleanup_branch_tolerates_delete_failure tests/customized_areal/test_tree_search_branch_sampling.py::test_cleanup_branch_skips_delete_when_no_sandbox_id tests/customized_areal/test_tree_search_branch_sampling.py::test_select_branch_candidate_ignores_cleaned_up_node -v`
Expected: PASS

- [ ] **Step 5: Update `_run_fresh_episode` to call `_cleanup_branch` after branch episode**

In `customized_areal/tree_search/tree_search_grouped_workflow.py`, modify `_run_fresh_episode` (lines 834-850). Replace:

```python
        if source == SampleSource.BRANCH and candidate is not None:
            branch_data = dict(episode_data)
            try:
                branch_task_id = await self._prepare_branch_task(branch_data, candidate)
            except Exception as exc:
                logger.warning(
                    "Branch task preparation errored for query_id=%s; "
                    "falling back to scratch: %s",
                    query_id,
                    exc,
                )
                branch_task_id = None
            if branch_task_id:
                branch_data["task_id"] = branch_task_id
                branch_data["seed_messages_already_inserted"] = True
                result = await self._retry_episode(engine, branch_data, group_idx)
                return _with_episode_metadata(result, branch_data)
            logger.warning(
                "Branch task preparation failed for query_id=%s; falling back to scratch",
                query_id,
            )
```

With:

```python
        if source == SampleSource.BRANCH and candidate is not None:
            branch_data = dict(episode_data)
            try:
                branch_task_id = await self._prepare_branch_task(branch_data, candidate)
            except Exception as exc:
                logger.warning(
                    "Branch task preparation errored for query_id=%s; "
                    "falling back to scratch: %s",
                    query_id,
                    exc,
                )
                branch_task_id = None
            if branch_task_id:
                branch_data["task_id"] = branch_task_id
                branch_data["seed_messages_already_inserted"] = True
                result = await self._retry_episode(engine, branch_data, group_idx)
                await self._cleanup_branch(candidate)
                return _with_episode_metadata(result, branch_data)
            logger.warning(
                "Branch task preparation failed for query_id=%s; falling back to scratch",
                query_id,
            )
```

The only change is adding `await self._cleanup_branch(candidate)` before the return on the branch path.

- [ ] **Step 6: Write test verifying _cleanup_branch is called after branch episode**

Add to `tests/customized_areal/test_tree_search_branch_sampling.py`:

```python
@pytest.mark.asyncio
async def test_run_fresh_episode_calls_cleanup_after_branch():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(TreeSearchGroupedRolloutWorkflow)
    workflow.sample_source = SampleSource.BRANCH
    workflow.branch_probability = 1.0

    candidate = _node(2)
    candidate.query_id = "q"
    candidate.task_id = "source-task"
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-branch"
    workflow.tree_store = type("Store", (), {"trajectories": {"q": [candidate]}})()

    cleanup_called = []

    async def fake_prepare_branch_task(data, candidate_arg):
        return "branch-task-id"

    async def fake_retry_episode(engine, episode_data, group_idx):
        return {"branch": True}

    async def fake_cleanup(candidate_arg):
        cleanup_called.append(candidate_arg)

    workflow._prepare_branch_task = fake_prepare_branch_task
    workflow._retry_episode = fake_retry_episode
    workflow._cleanup_branch = fake_cleanup

    data = {"query_id": "q"}
    result = await workflow._run_fresh_episode(None, data, 0, "q")

    assert len(cleanup_called) == 1
    assert cleanup_called[0] is candidate
```

- [ ] **Step 7: Run all branch sampling tests**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py tests/customized_areal/test_tree_search_branch_sampling.py
git commit -m "feat(tree-search): add _cleanup_branch, delete branch sandbox after episode, clear node state"
```

---

### Task 5: Run pre-commit and full test suite

**Files:** None

- [ ] **Step 1: Run pre-commit**

Run: `pre-commit run --all-files`
Expected: PASS (fix any linting/formatting issues)

- [ ] **Step 2: Run the full branch sampling test suite one final time**

Run: `uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit any formatting fixes if needed**

```bash
git add -u
git commit -m "style: pre-commit formatting fixes"
```

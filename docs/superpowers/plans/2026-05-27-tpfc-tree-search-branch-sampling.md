# TPFC Tree Search Branch Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let TPFC tree search sample new episodes from scratch, from a high-entropy intermediate node, or from a mixed policy, while cloning the selected node's sandbox and preserving truncated DB message context.

**Architecture:** Keep Leagent runtime entropy recording unchanged. Add AReaL-side run metadata, branch task/sandbox helpers, node metadata, and `TreeSearchGroupedRolloutWorkflow` sampling logic. Sandbox cloning is behind one helper that returns `None` when Daytona clone/snapshot support is unavailable, causing a logged fallback to scratch.

**Tech Stack:** Python 3.14, Supabase async client, Daytona SDK when available, pytest, AReaL tree-search workflow.

---

## File Structure

- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/backend_run.py`: add `BackendRunResult`, fetch raw messages with metadata, support starting an already-seeded branch task without inserting a prompt, and return named run metadata.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/messages.py`: add raw-message metadata reads and bulk message copy helper.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/sandbox.py`: add clone/snapshot helper and sandbox-row binding helper.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/__init__.py`: export new helpers.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/config.py`: add `SampleSource` and branch sampling fields to `TreeBackupConfig`.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/mcts_tree_store.py`: add optional TPFC metadata fields to `Node`.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/tree_search_grouped_workflow.py`: annotate nodes from run metadata, choose scratch/branch/mixed episodes, create branch data, and insert branch nodes.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/tpfc_agent.py`: consume `BackendRunResult` attributes.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/core/agent.py`: replace its `run_backend(...)` tuple handling with `BackendRunResult.messages`, `.final_answer`, and `.log_path` reads.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/scripts/benchmark_run_base.py`: keep compatibility with the new result object.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/configs/*tree_search*.yaml`: add optional `sample_source` and `branch_probability` fields with current behavior defaulting to scratch.
- Add `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tree_search_branch_sampling.py`: unit tests for metadata mapping, candidate selection, sampling modes, and branch task construction.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tpfc_backend_auth.py`: assert `BackendRunResult` compatibility.
- Modify `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_sandbox_cleanup.py`: add sandbox clone fallback/binding tests.

## Task 1: Add Backend Run Result Contract

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/backend_run.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tpfc_backend_auth.py`

- [ ] **Step 1: Write failing result-contract test**

Add this test near the other `run_backend` tests:

```python
@pytest.mark.asyncio
async def test_run_backend_returns_named_result_and_tuple_compat(monkeypatch):
    class FakeClient:
        pass

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_resolve_agent_id(client, user_id, agent_id):
        return "agent"

    async def fake_create_task(**kwargs):
        return "task-id"

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        return {"agent_run_id": "run-id", "status": "running"}, "token"

    async def fake_wait_for_agent_run(*args, **kwargs):
        return "completed"

    parsed_messages = [{"role": "assistant", "content": "<answer>42</answer>"}]
    raw_messages = [
        {
            "message_id": "m1",
            "role": "assistant",
            "content": {"role": "assistant", "content": "<answer>42</answer>"},
            "metadata": {"entropy_stats": {"max_entropy": 3.2}, "need_branch": True},
        }
    ]

    async def fake_get_messages(client, task_id):
        assert task_id == "task-id"
        return parsed_messages

    async def fake_get_raw_messages(client, task_id):
        assert task_id == "task-id"
        return raw_messages

    async def fake_cleanup(client, task_id):
        return None

    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "_resolve_agent_id", fake_resolve_agent_id)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token)
    monkeypatch.setattr(backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh)
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", fake_wait_for_agent_run)
    monkeypatch.setattr(backend_run, "_get_llm_messages_with_client", fake_get_messages)
    monkeypatch.setattr(backend_run, "_get_raw_messages_with_client", fake_get_raw_messages)
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    result = await backend_run.run_backend("task", [], user_id="user")

    assert result.task_id == "task-id"
    assert result.messages == parsed_messages
    assert result.raw_messages == raw_messages
    assert result.final_answer == "42"
    messages, answer, log_path, trace = result
    assert messages == parsed_messages
    assert answer == "42"
    assert log_path == "./log.json"
    assert trace is None
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tpfc_backend_auth.py::test_run_backend_returns_named_result_and_tuple_compat -q
```

Expected: FAIL because `BackendRunResult` and `_get_raw_messages_with_client` do not exist yet.

- [ ] **Step 3: Implement `BackendRunResult` and raw message fetch**

In `customized_areal/tpfc/backend_run.py`, add imports and dataclass near constants:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackendRunResult:
    messages: list[dict[str, Any]]
    final_answer: str | None
    log_path: str
    task_id: str
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    trace: Any | None = None

    def __iter__(self):
        yield self.messages
        yield self.final_answer
        yield self.log_path
        yield self.trace
```

Add the raw read helper beside `_get_llm_messages_with_client`:

```python
async def _get_raw_messages_with_client(client, task_id: str) -> list[dict[str, Any]]:
    all_messages: list[dict[str, Any]] = []
    batch_size = 1000
    offset = 0

    while True:
        query = (
            client.table("messages")
            .select("message_id, role, content, created_at, updated_at, metadata, is_meta")
            .eq("task_id", task_id)
            .order("created_at", desc=False)
            .range(offset, offset + batch_size - 1)
        )
        result = await _execute_message_query_with_retry(query, task_id=task_id, offset=offset)
        data = getattr(result, "data", None) or []
        all_messages.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    return all_messages
```

In `_do_run()`, replace the final return with:

```python
messages = await _get_llm_messages_with_client(client, task_id)
raw_messages = await _get_raw_messages_with_client(client, task_id)
final_boxed_answer = _extract_final_answer(messages)

return BackendRunResult(
    messages=messages,
    raw_messages=raw_messages,
    final_answer=final_boxed_answer,
    log_path=log_path,
    task_id=task_id,
)
```

- [ ] **Step 4: Run the result-contract test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tpfc_backend_auth.py::test_run_backend_returns_named_result_and_tuple_compat -q
```

Expected: PASS.

## Task 2: Add Message Copy Helpers

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/messages.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/__init__.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing message truncation test**

Create the new test file with:

```python
from customized_areal.db_service.messages import truncate_messages_before_turn


def test_truncate_messages_before_high_entropy_assistant_turn():
    raw_messages = [
        {"message_id": "u1", "role": "user", "content": {"role": "user", "content": "q"}},
        {"message_id": "a1", "role": "assistant", "content": {"role": "assistant", "content": "step 1"}, "metadata": {"entropy_stats": {"max_entropy": 0.1}}},
        {"message_id": "t1", "role": "tool", "content": {"role": "tool", "content": "obs"}},
        {"message_id": "a2", "role": "assistant", "content": {"role": "assistant", "content": "step 2"}, "metadata": {"need_branch": True}},
        {"message_id": "a3", "role": "assistant", "content": {"role": "assistant", "content": "step 3"}},
    ]

    prefix = truncate_messages_before_turn(raw_messages, assistant_turn_idx=2)

    assert [row["message_id"] for row in prefix] == ["u1", "a1", "t1"]
    assert all("message_id" not in row for row in prefix)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_truncate_messages_before_high_entropy_assistant_turn -q
```

Expected: FAIL because `truncate_messages_before_turn` is missing.

- [ ] **Step 3: Implement truncation and bulk insert helpers**

In `customized_areal/db_service/messages.py`, add:

```python
def truncate_messages_before_turn(
    raw_messages: list[dict[str, Any]],
    assistant_turn_idx: int,
) -> list[dict[str, Any]]:
    if assistant_turn_idx < 1:
        raise ValueError("assistant_turn_idx must be >= 1")

    assistant_count = 0
    prefix: list[dict[str, Any]] = []
    for row in raw_messages:
        if row.get("role") == "assistant":
            assistant_count += 1
            if assistant_count == assistant_turn_idx:
                return prefix
        copied = {
            key: value
            for key, value in row.items()
            if key not in {"message_id", "task_id", "created_at", "updated_at"}
        }
        prefix.append(copied)
    raise ValueError(f"assistant turn {assistant_turn_idx} not found")


async def copy_messages_to_task(
    client,
    *,
    task_id: str,
    messages: list[dict[str, Any]],
) -> None:
    if not messages:
        return
    rows = []
    for message in messages:
        row = dict(message)
        row["task_id"] = task_id
        rows.append(row)
    await client.table(TABLE_MESSAGES).insert(rows).execute()
```

Update `__all__` or `customized_areal/db_service/__init__.py` exports:

```python
from .messages import add_message, copy_messages_to_task, get_llm_messages, truncate_messages_before_turn
```

- [ ] **Step 4: Run the truncation test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_truncate_messages_before_high_entropy_assistant_turn -q
```

Expected: PASS.

## Task 3: Add Sandbox Clone And Binding Helpers

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/sandbox.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/db_service/__init__.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_sandbox_cleanup.py`

- [ ] **Step 1: Write failing sandbox clone fallback and binding tests**

Append:

```python
@pytest.mark.asyncio
async def test_clone_sandbox_returns_none_when_sdk_has_no_clone(monkeypatch):
    class FakeDaytona:
        async def get(self, sandbox_id):
            return object()

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())

    assert await sandbox.clone_sandbox("source-sandbox") is None


@pytest.mark.asyncio
async def test_bind_sandbox_to_task_inserts_row():
    inserted = []

    class FakeQuery:
        def insert(self, row):
            inserted.append(row)
            return self

        async def execute(self):
            return None

    class FakeClient:
        def table(self, table_name):
            assert table_name == "sandboxes"
            return FakeQuery()

    await sandbox.bind_sandbox_to_task(
        FakeClient(),
        sandbox_id="sandbox-copy",
        task_id="task-copy",
        account_id="account",
    )

    assert inserted == [{"sandbox_id": "sandbox-copy", "task_id": "task-copy", "account_id": "account"}]
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_sandbox_cleanup.py::test_clone_sandbox_returns_none_when_sdk_has_no_clone tests/customized_areal/test_sandbox_cleanup.py::test_bind_sandbox_to_task_inserts_row -q
```

Expected: FAIL because helpers are missing.

- [ ] **Step 3: Implement helpers**

Add to `customized_areal/db_service/sandbox.py`:

```python
async def clone_sandbox(source_sandbox_id: str) -> str | None:
    daytona = _get_daytona()
    source = await daytona.get(source_sandbox_id)

    if hasattr(daytona, "clone"):
        cloned = await daytona.clone(source)
        return getattr(cloned, "id", None) or (cloned.get("id") if isinstance(cloned, dict) else None)

    if hasattr(source, "create_snapshot") and hasattr(daytona, "create"):
        snapshot = await source.create_snapshot()
        snapshot_id = getattr(snapshot, "id", None) or str(snapshot)
        created = await daytona.create({"snapshot": snapshot_id})
        return getattr(created, "id", None) or (created.get("id") if isinstance(created, dict) else None)

    logger.warning("Daytona sandbox clone/snapshot API unavailable")
    return None


async def bind_sandbox_to_task(
    client,
    *,
    sandbox_id: str,
    task_id: str,
    account_id: str,
) -> None:
    await client.table("sandboxes").insert(
        {"sandbox_id": sandbox_id, "task_id": task_id, "account_id": account_id}
    ).execute()
```

Export from `customized_areal/db_service/__init__.py`:

```python
from .sandbox import bind_sandbox_to_task, cleanup_sandbox_for_task, clone_sandbox, delete_sandbox
```

- [ ] **Step 4: Run sandbox tests**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_sandbox_cleanup.py -q
```

Expected: PASS.

## Task 4: Add Node Metadata And Mapping Helpers

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/mcts_tree_store.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/tree_search_grouped_workflow.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing node annotation test**

Append:

```python
from customized_areal.tree_search.mcts_tree_store import Node
from customized_areal.tree_search.tree_search_grouped_workflow import annotate_nodes_from_run


def _node(turn_idx: int) -> Node:
    return Node(input_ids=[turn_idx], loss_mask=[1], logprobs=[0.0], versions=[0], node_id=f"n{turn_idx}", turn_idx=turn_idx)


def test_annotate_nodes_from_run_copies_entropy_metadata():
    nodes = [_node(1), _node(2)]
    raw_messages = [
        {"role": "assistant", "metadata": {"entropy_stats": {"max_entropy": 0.2}, "need_branch": False}},
        {"role": "assistant", "metadata": {"entropy_stats": {"max_entropy": 3.4}, "need_branch": True, "branch_sandbox_id": "sb2"}},
    ]

    annotate_nodes_from_run(nodes, task_id="task-id", raw_messages=raw_messages)

    assert nodes[0].task_id == "task-id"
    assert nodes[0].entropy_stats == {"max_entropy": 0.2}
    assert nodes[0].need_branch is False
    assert nodes[1].task_id == "task-id"
    assert nodes[1].entropy_stats == {"max_entropy": 3.4}
    assert nodes[1].need_branch is True
    assert nodes[1].branch_sandbox_id == "sb2"
```

- [ ] **Step 2: Run failing test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_annotate_nodes_from_run_copies_entropy_metadata -q
```

Expected: FAIL because new fields/helper are missing.

- [ ] **Step 3: Add node fields**

In `Node`, add:

```python
    task_id: str = ""
    entropy_stats: dict[str, Any] | None = None
    need_branch: bool = False
    branch_sandbox_id: str | None = None
```

- [ ] **Step 4: Implement annotation helper**

In `tree_search_grouped_workflow.py`, add:

```python
def _assistant_metadata(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in raw_messages:
        if row.get("role") != "assistant":
            continue
        metadata = row.get("metadata") or {}
        result.append(metadata if isinstance(metadata, dict) else {})
    return result


def annotate_nodes_from_run(
    nodes: list[Node],
    *,
    task_id: str,
    raw_messages: list[dict[str, Any]],
) -> None:
    assistant_meta = _assistant_metadata(raw_messages)
    for node in nodes:
        node.task_id = task_id
        idx = max(node.turn_idx - 1, 0)
        if idx >= len(assistant_meta):
            continue
        metadata = assistant_meta[idx]
        entropy_stats = metadata.get("entropy_stats")
        node.entropy_stats = entropy_stats if isinstance(entropy_stats, dict) else None
        node.need_branch = bool(metadata.get("need_branch"))
        branch_sandbox_id = metadata.get("branch_sandbox_id")
        node.branch_sandbox_id = branch_sandbox_id if isinstance(branch_sandbox_id, str) and branch_sandbox_id else None
```

- [ ] **Step 5: Run node annotation test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_annotate_nodes_from_run_copies_entropy_metadata -q
```

Expected: PASS.

## Task 5: Add Branch Task Construction In `backend_run`

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/backend_run.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tpfc_backend_auth.py`

- [ ] **Step 1: Write failing branch-start test**

Add:

```python
@pytest.mark.asyncio
async def test_run_backend_existing_seeded_task_starts_without_new_task(monkeypatch):
    created_tasks = []

    class FakeClient:
        pass

    async def fake_create_client():
        return FakeClient()

    async def fake_close_client(client):
        return None

    async def fake_resolve_agent_id(client, user_id, agent_id):
        return "agent"

    async def fake_create_task(**kwargs):
        created_tasks.append(kwargs)
        return "unexpected"

    async def fake_get_valid_token(self):
        return "token"

    async def fake_start_agent_run_with_refresh(**kwargs):
        assert kwargs["form_data"]["task_id"] == "branch-task"
        assert kwargs["form_data"]["prompt"] == ""
        return {"agent_run_id": "run-id", "status": "running"}, "token"

    async def fake_wait_for_agent_run(*args, **kwargs):
        return "completed"

    async def fake_get_messages(client, task_id):
        return [{"role": "assistant", "content": "<answer>x</answer>"}]

    async def fake_get_raw_messages(client, task_id):
        return []

    async def fake_cleanup(client, task_id):
        return None

    monkeypatch.setattr(backend_run, "DEFAULT_USER_ID", "user")
    monkeypatch.setattr(backend_run, "_create_shortlived_db_client", fake_create_client)
    monkeypatch.setattr(backend_run, "_close_db_client", fake_close_client)
    monkeypatch.setattr(backend_run, "_resolve_agent_id", fake_resolve_agent_id)
    monkeypatch.setattr(backend_run, "create_task", fake_create_task)
    monkeypatch.setattr(backend_run.SharedTokenManager, "get_valid_token", fake_get_valid_token)
    monkeypatch.setattr(backend_run, "_start_agent_run_with_refresh", fake_start_agent_run_with_refresh)
    monkeypatch.setattr(backend_run, "_wait_for_agent_run", fake_wait_for_agent_run)
    monkeypatch.setattr(backend_run, "_get_llm_messages_with_client", fake_get_messages)
    monkeypatch.setattr(backend_run, "_get_raw_messages_with_client", fake_get_raw_messages)
    monkeypatch.setattr(backend_run, "cleanup_sandbox_for_task", fake_cleanup)

    result = await backend_run.run_backend(
        task_description="",
        task_file_path=[],
        task_id="branch-task",
        user_id="user",
        seed_messages_already_inserted=True,
    )

    assert result.task_id == "branch-task"
    assert created_tasks == []
```

- [ ] **Step 2: Run failing test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tpfc_backend_auth.py::test_run_backend_existing_seeded_task_starts_without_new_task -q
```

Expected: FAIL because `seed_messages_already_inserted` is missing and `run_backend` always creates a task.

- [ ] **Step 3: Implement seeded task mode**

Add parameter to `run_backend`:

```python
    seed_messages_already_inserted: bool = False,
```

Replace unconditional task creation with:

```python
if seed_messages_already_inserted:
    if not task_id:
        raise ValueError("task_id is required when seed_messages_already_inserted=True")
else:
    task_id = await create_task(
        client=client,
        account_id=user_id,
        agent_id=resolved_agent_id,
        name=task_description[:100] if task_description else None,
    )
    logger.info("Task created: %s", task_id)
```

Ensure `_prepare_form_data()` receives `task_description or ""` so no new prompt text is inserted for seeded branch runs.

- [ ] **Step 4: Run branch-start test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tpfc_backend_auth.py::test_run_backend_existing_seeded_task_starts_without_new_task -q
```

Expected: PASS.

## Task 6: Add Branch Sampling Logic To Tree Workflow

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/config.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/tree_search_grouped_workflow.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing sampling mode tests**

Append:

```python
from customized_areal.tree_search.config import SampleSource
from customized_areal.tree_search.tree_search_grouped_workflow import choose_sample_source, select_branch_candidate


def test_choose_sample_source_modes():
    assert choose_sample_source(SampleSource.SCRATCH, branch_probability=1.0, has_candidate=True, random_value=0.0) == SampleSource.SCRATCH
    assert choose_sample_source(SampleSource.BRANCH, branch_probability=0.0, has_candidate=True, random_value=1.0) == SampleSource.BRANCH
    assert choose_sample_source(SampleSource.BRANCH, branch_probability=1.0, has_candidate=False, random_value=0.0) == SampleSource.SCRATCH
    assert choose_sample_source(SampleSource.MIXED, branch_probability=0.5, has_candidate=True, random_value=0.4) == SampleSource.BRANCH
    assert choose_sample_source(SampleSource.MIXED, branch_probability=0.5, has_candidate=True, random_value=0.6) == SampleSource.SCRATCH


def test_select_branch_candidate_prefers_max_entropy():
    low = _node(1)
    low.query_id = "q"
    low.task_id = "task-low"
    low.need_branch = True
    low.branch_sandbox_id = "sb-low"
    low.entropy_stats = {"max_entropy": 2.5}

    high = _node(2)
    high.query_id = "q"
    high.task_id = "task-high"
    high.need_branch = True
    high.branch_sandbox_id = "sb-high"
    high.entropy_stats = {"max_entropy": 4.0}

    assert select_branch_candidate([low, high], "q") is high
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_choose_sample_source_modes tests/customized_areal/test_tree_search_branch_sampling.py::test_select_branch_candidate_prefers_max_entropy -q
```

Expected: FAIL because `SampleSource` and helpers are missing.

- [ ] **Step 3: Add config fields**

In `tree_search/config.py`:

```python
class SampleSource(str, Enum):
    SCRATCH = "scratch"
    BRANCH = "branch"
    MIXED = "mixed"
```

Add to `TreeBackupConfig`:

```python
    sample_source: SampleSource = SampleSource.SCRATCH
    branch_probability: float = 0.5
```

- [ ] **Step 4: Implement selection helpers**

In `tree_search_grouped_workflow.py`:

```python
from customized_areal.tree_search.config import AdvantageMode, CacheMode, LossMode, SampleSource


def choose_sample_source(
    mode: SampleSource,
    *,
    branch_probability: float,
    has_candidate: bool,
    random_value: float,
) -> SampleSource:
    if mode == SampleSource.SCRATCH or not has_candidate:
        return SampleSource.SCRATCH
    if mode == SampleSource.BRANCH:
        return SampleSource.BRANCH
    if mode == SampleSource.MIXED and random_value < branch_probability:
        return SampleSource.BRANCH
    return SampleSource.SCRATCH


def _max_entropy(node: Node) -> float:
    stats = node.entropy_stats or {}
    value = stats.get("max_entropy") if isinstance(stats, dict) else None
    return float(value) if isinstance(value, int | float) else 0.0


def select_branch_candidate(nodes: list[Node], query_id: str) -> Node | None:
    candidates = [
        node
        for node in nodes
        if node.query_id == query_id
        and node.need_branch
        and bool(node.task_id)
        and bool(node.branch_sandbox_id)
    ]
    if not candidates:
        return None
    return max(candidates, key=_max_entropy)
```

- [ ] **Step 5: Run sampling helper tests**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_choose_sample_source_modes tests/customized_areal/test_tree_search_branch_sampling.py::test_select_branch_candidate_prefers_max_entropy -q
```

Expected: PASS.

## Task 7: Wire Branch Run Orchestration

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/tree_search_grouped_workflow.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/tpfc_agent.py`
- Test: `/dfs/share-groups/letrain/zhoujie/AReaL-main/tests/customized_areal/test_tree_search_branch_sampling.py`

- [ ] **Step 1: Write failing branch data preparation test**

Append:

```python
import pytest

from customized_areal.tree_search.tree_search_grouped_workflow import build_branch_task


@pytest.mark.asyncio
async def test_build_branch_task_clones_sandbox_and_copies_prefix(monkeypatch):
    created = []
    copied = []

    class FakeClient:
        pass

    candidate = _node(2)
    candidate.task_id = "source-task"
    candidate.turn_idx = 2
    candidate.branch_sandbox_id = "source-sandbox"

    raw_messages = [
        {"message_id": "u1", "role": "user", "content": {"role": "user", "content": "q"}},
        {"message_id": "a1", "role": "assistant", "content": {"role": "assistant", "content": "step1"}},
        {"message_id": "a2", "role": "assistant", "content": {"role": "assistant", "content": "step2"}},
    ]

    async def fake_get_raw_messages(task_id, return_raw=True):
        assert task_id == "source-task"
        return raw_messages

    async def fake_create_task(**kwargs):
        created.append(kwargs)
        return "branch-task"

    async def fake_clone_sandbox(sandbox_id):
        assert sandbox_id == "source-sandbox"
        return "cloned-sandbox"

    async def fake_bind_sandbox_to_task(client, *, sandbox_id, task_id, account_id):
        copied.append(("bind", sandbox_id, task_id, account_id))

    async def fake_copy_messages_to_task(client, *, task_id, messages):
        copied.append(("messages", task_id, messages))

    monkeypatch.setattr("customized_areal.tree_search.tree_search_grouped_workflow.get_llm_messages", fake_get_raw_messages)
    monkeypatch.setattr("customized_areal.tree_search.tree_search_grouped_workflow.create_task", fake_create_task)
    monkeypatch.setattr("customized_areal.tree_search.tree_search_grouped_workflow.clone_sandbox", fake_clone_sandbox)
    monkeypatch.setattr("customized_areal.tree_search.tree_search_grouped_workflow.bind_sandbox_to_task", fake_bind_sandbox_to_task)
    monkeypatch.setattr("customized_areal.tree_search.tree_search_grouped_workflow.copy_messages_to_task", fake_copy_messages_to_task)

    branch_task_id = await build_branch_task(
        client=FakeClient(),
        account_id="account",
        agent_id="agent",
        candidate=candidate,
        name="branch",
    )

    assert branch_task_id == "branch-task"
    assert created[0]["account_id"] == "account"
    assert copied[0] == ("bind", "cloned-sandbox", "branch-task", "account")
    assert copied[1][0] == "messages"
    assert copied[1][1] == "branch-task"
    assert copied[1][2] == [{"role": "user", "content": {"role": "user", "content": "q"}}, {"role": "assistant", "content": {"role": "assistant", "content": "step1"}}]
```

- [ ] **Step 2: Run failing test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_build_branch_task_clones_sandbox_and_copies_prefix -q
```

Expected: FAIL because `build_branch_task` is missing.

- [ ] **Step 3: Implement branch task builder**

In `tree_search_grouped_workflow.py`, import helpers:

```python
from customized_areal.db_service import (
    bind_sandbox_to_task,
    clone_sandbox,
    copy_messages_to_task,
    create_task,
    get_llm_messages,
    truncate_messages_before_turn,
)
```

Add:

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

    raw_messages = await get_llm_messages(candidate.task_id, return_raw=True)
    prefix = truncate_messages_before_turn(raw_messages, candidate.turn_idx)
    cloned_sandbox_id = await clone_sandbox(candidate.branch_sandbox_id)
    if not cloned_sandbox_id:
        logger.warning("Branch sandbox clone unavailable; falling back to scratch")
        return None

    branch_task_id = await create_task(
        client=client,
        account_id=account_id,
        agent_id=agent_id,
        name=name,
    )
    try:
        await bind_sandbox_to_task(
            client,
            sandbox_id=cloned_sandbox_id,
            task_id=branch_task_id,
            account_id=account_id,
        )
        await copy_messages_to_task(client, task_id=branch_task_id, messages=prefix)
    except Exception:
        from customized_areal.db_service import delete_sandbox

        await delete_sandbox(cloned_sandbox_id)
        raise
    return branch_task_id
```

- [ ] **Step 4: Run branch builder test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py::test_build_branch_task_clones_sandbox_and_copies_prefix -q
```

Expected: PASS.

- [ ] **Step 5: Wire workflow execution**

In `TreeSearchGroupedRolloutWorkflow.__init__`, accept:

```python
        sample_source: SampleSource = SampleSource.SCRATCH,
        branch_probability: float = 0.5,
```

Normalize strings:

```python
self.sample_source = SampleSource(sample_source)
self.branch_probability = branch_probability
```

Inside fresh generation, replace the single `asyncio.gather(self._retry_episode(...))` with an internal method:

```python
async def _run_fresh_episode(self, engine, data: dict[str, Any], group_idx: int, query_id: str) -> Any:
    all_query_nodes = self.tree_store.trajectories.get(query_id, [])
    candidate = select_branch_candidate(all_query_nodes, query_id)
    source = choose_sample_source(
        self.sample_source,
        branch_probability=self.branch_probability,
        has_candidate=candidate is not None,
        random_value=random.random(),
    )
    if source == SampleSource.BRANCH and candidate is not None:
        branch_data = dict(data)
        branch_task_id = await self._prepare_branch_task(branch_data, candidate)
        if branch_task_id:
            branch_data["task_id"] = branch_task_id
            branch_data["seed_messages_already_inserted"] = True
            return await self._retry_episode(engine, branch_data, group_idx)
    return await self._retry_episode(engine, data, group_idx)
```

Implement `_prepare_branch_task()` with the same identity sources that `run_backend()` already uses:

```python
async def _prepare_branch_task(self, data: dict[str, Any], candidate: Node) -> str | None:
    from customized_areal.tpfc.backend_run import (
        DEFAULT_AGENT_ID,
        DEFAULT_USER_ID,
        _close_db_client,
        _create_shortlived_db_client,
        _resolve_agent_id,
    )

    account_id = str(data.get("user_id") or DEFAULT_USER_ID or "")
    if not account_id:
        logger.warning("Cannot branch TPFC episode without user/account id")
        return None

    client = await _create_shortlived_db_client()
    try:
        agent_id = await _resolve_agent_id(client, account_id, data.get("agent_id") or DEFAULT_AGENT_ID)
        return await build_branch_task(
            client=client,
            account_id=account_id,
            agent_id=agent_id,
            candidate=candidate,
            name=str(data.get("query", ""))[:100] if data.get("query") else None,
        )
    finally:
        await _close_db_client(client)
```

- [ ] **Step 6: Update TPFC agent to pass branch task fields**

In `tpfc_agent.py`, after reading `data`, pass through:

```python
task_id=data.get("task_id", ""),
seed_messages_already_inserted=bool(data.get("seed_messages_already_inserted", False)),
```

Store the result:

```python
run_result = await run_backend(...)
completion_messages = run_result.messages
_final_answer = run_result.final_answer
```

- [ ] **Step 7: Run branch workflow tests**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest tests/customized_areal/test_tree_search_branch_sampling.py -q
```

Expected: PASS.

## Task 8: Update Config Wiring And Callers

**Files:**
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/scripts/train_tpfc_tree_search.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search_v2.yaml`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/tpfc_agent.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tree_search/core/agent.py`
- Modify: `/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/scripts/benchmark_run_base.py`

- [ ] **Step 1: Update workflow construction**

Where `TreeSearchGroupedRolloutWorkflow` is instantiated in trainer code, pass:

```python
sample_source=tree_search_config.sample_source,
branch_probability=tree_search_config.branch_probability,
```

- [ ] **Step 2: Update YAML configs**

Under `tree_search:` add:

```yaml
  sample_source: scratch
  branch_probability: 0.5
```

Keep `scratch` as default to preserve current behavior. Users can set `branch` or `mixed`.

- [ ] **Step 3: Update non-TPFC result consumers**

Replace tuple unpacking where direct attributes are clearer:

```python
result = await run_backend(...)
completion_messages = result.messages
final_answer = result.final_answer
log_path = result.log_path
```

For `benchmark_run_base.py`, if it expects a response string, preserve current behavior by extracting the last assistant content:

```python
run_result = await run_backend(...)
response = ""
for message in reversed(run_result.messages):
    if message.get("role") == "assistant":
        content = message.get("content", "")
        response = content.get("content", "") if isinstance(content, dict) else str(content)
        break
final_boxed_answer = run_result.final_answer or ""
log_file_path = Path(run_result.log_path)
```

- [ ] **Step 4: Run import/config smoke test**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run python - <<'PY'
from customized_areal.tree_search.config import SampleSource, TreeBackupConfig
cfg = TreeBackupConfig(sample_source=SampleSource.MIXED, branch_probability=0.5)
assert cfg.sample_source == SampleSource.MIXED
assert cfg.branch_probability == 0.5
print("ok")
PY
```

Expected: prints `ok`.

## Task 9: Verification

**Files:**
- No source changes unless failures expose bugs.

- [ ] **Step 1: Run focused tests**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run pytest \
  tests/customized_areal/test_tree_search_branch_sampling.py \
  tests/customized_areal/test_tpfc_backend_auth.py \
  tests/customized_areal/test_sandbox_cleanup.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run formatting/lint on touched files**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run ruff check customized_areal/tpfc/backend_run.py customized_areal/db_service/messages.py customized_areal/db_service/sandbox.py customized_areal/tree_search/tree_search_grouped_workflow.py customized_areal/tree_search/mcts_tree_store.py customized_areal/tree_search/config.py tests/customized_areal/test_tree_search_branch_sampling.py tests/customized_areal/test_tpfc_backend_auth.py tests/customized_areal/test_sandbox_cleanup.py
```

Expected: PASS or only pre-existing unrelated findings. Fix touched-file findings before proceeding.

- [ ] **Step 3: Inspect git diff for user changes**

Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git diff -- customized_areal/tpfc/backend_run.py customized_areal/db_service/messages.py customized_areal/db_service/sandbox.py customized_areal/db_service/__init__.py customized_areal/tree_search/config.py customized_areal/tree_search/mcts_tree_store.py customized_areal/tree_search/tree_search_grouped_workflow.py customized_areal/tpfc/tpfc_agent.py customized_areal/tree_search/core/agent.py customized_areal/tpfc/scripts/benchmark_run_base.py tests/customized_areal/test_tree_search_branch_sampling.py tests/customized_areal/test_tpfc_backend_auth.py tests/customized_areal/test_sandbox_cleanup.py
```

Expected: diff contains only branch sampling work and preserves pre-existing local edits.

## Self-Review

- Spec coverage: sampling modes, branch probability, task id, entropy metadata, branch sandbox id, message truncation, sandbox clone, fallback behavior, and tests are covered.
- Placeholder scan: no `TBD`, `TODO`, or vague implementation-only steps remain.
- Type consistency: `SampleSource`, `BackendRunResult`, `task_id`, `raw_messages`, `entropy_stats`, `need_branch`, and `branch_sandbox_id` names match across tasks.

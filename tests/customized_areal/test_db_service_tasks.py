import httpx
import pytest

from customized_areal.db_service import tasks


@pytest.mark.asyncio
async def test_create_task_retries_remote_protocol_error(monkeypatch):
    """Transient transport failures are retried before task creation succeeds."""
    insert_calls = 0
    sleep_calls: list[float] = []

    class FakeQuery:
        def insert(self, row):
            assert row["account_id"] == "account-id"
            assert row["agent_id"] == "agent-id"
            return self

        async def execute(self):
            nonlocal insert_calls
            insert_calls += 1
            if insert_calls == 1:
                raise httpx.RemoteProtocolError(
                    "Server disconnected without sending a response."
                )
            return None

    class FakeClient:
        def table(self, table_name):
            assert table_name == "tasks"
            return FakeQuery()

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(tasks.asyncio, "sleep", fake_sleep)

    task_id = await tasks.create_task(
        client=FakeClient(),
        account_id="account-id",
        agent_id="agent-id",
        name="task-name",
    )

    assert insert_calls == 2
    assert sleep_calls == [tasks._TASK_WRITE_RETRY_DELAY]
    assert isinstance(task_id, str)


@pytest.mark.asyncio
async def test_create_task_raises_after_retry_budget_exhausted(monkeypatch):
    """Task creation surfaces the final transient failure after retries are exhausted."""
    insert_calls = 0
    sleep_calls: list[float] = []

    class FakeQuery:
        def insert(self, row):
            return self

        async def execute(self):
            nonlocal insert_calls
            insert_calls += 1
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            )

    class FakeClient:
        def table(self, table_name):
            assert table_name == "tasks"
            return FakeQuery()

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(tasks.asyncio, "sleep", fake_sleep)

    with pytest.raises(httpx.RemoteProtocolError):
        await tasks.create_task(
            client=FakeClient(),
            account_id="account-id",
            agent_id="agent-id",
        )

    assert insert_calls == tasks._TASK_WRITE_RETRIES + 1
    assert sleep_calls == [
        min(tasks._TASK_WRITE_RETRY_DELAY * (2**attempt), 30.0)
        for attempt in range(tasks._TASK_WRITE_RETRIES)
    ]


@pytest.mark.asyncio
async def test_update_task_status_retries_remote_protocol_error(monkeypatch):
    """Transient transport failures are retried before status update succeeds."""
    update_calls = 0
    sleep_calls: list[float] = []

    class FakeQuery:
        def update(self, row):
            assert row["status"] == tasks.TaskStatus.RUNNING
            assert "started_at" in row
            return self

        def eq(self, column, value):
            assert column == "task_id"
            assert value == "task-id"
            return self

        async def execute(self):
            nonlocal update_calls
            update_calls += 1
            if update_calls == 1:
                raise httpx.RemoteProtocolError(
                    "Server disconnected without sending a response."
                )
            return None

    class FakeClient:
        def table(self, table_name):
            assert table_name == "tasks"
            return FakeQuery()

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(tasks.asyncio, "sleep", fake_sleep)

    await tasks.update_task_status(
        FakeClient(),
        "task-id",
        tasks.TaskStatus.RUNNING,
    )

    assert update_calls == 2
    assert sleep_calls == [tasks._TASK_WRITE_RETRY_DELAY]

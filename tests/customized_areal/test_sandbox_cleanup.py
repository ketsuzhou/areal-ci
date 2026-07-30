import httpx
import pytest

from customized_areal.db_service import sandbox


@pytest.mark.asyncio
async def test_delete_sandbox_retries_transient_failures(monkeypatch):
    """Daytona delete is retried before surfacing a cleanup failure."""
    calls = []

    class FakeDaytona:
        async def get(self, sandbox_id):
            return {"id": sandbox_id}

        async def delete(self, sandbox_obj, timeout):
            calls.append((sandbox_obj["id"], timeout))
            if len(calls) == 1:
                raise TimeoutError("slow delete")

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())
    monkeypatch.setattr(sandbox.asyncio, "sleep", fake_sleep)

    await sandbox.delete_sandbox("sandbox-id", timeout=3, max_retries=1)

    assert calls == [("sandbox-id", 3), ("sandbox-id", 3)]


@pytest.mark.asyncio
async def test_cleanup_sandbox_for_task_swallows_delete_failure(monkeypatch):
    """Best-effort task cleanup does not fail the caller when Daytona times out."""

    class FakeResult:
        data = {"sandbox_id": "sandbox-id"}

    class FakeQuery:
        def select(self, columns):
            return self

        def eq(self, column, value):
            return self

        def maybe_single(self):
            return self

        async def execute(self):
            return FakeResult()

    class FakeClient:
        def table(self, table_name):
            assert table_name == "sandboxes"
            return FakeQuery()

    async def fail_delete_sandbox(sandbox_id):
        raise TimeoutError("slow delete")

    monkeypatch.setattr(sandbox, "delete_sandbox", fail_delete_sandbox)

    await sandbox.cleanup_sandbox_for_task(FakeClient(), "task-id")


@pytest.mark.asyncio
async def test_cleanup_sandbox_for_task_retries_lookup_timeout(monkeypatch):
    """Transient Supabase lookup timeouts are retried before Daytona cleanup."""
    query_calls = 0
    deleted = []

    class FakeResult:
        data = {"sandbox_id": "sandbox-id"}

    class FakeQuery:
        def select(self, columns):
            return self

        def eq(self, column, value):
            return self

        def maybe_single(self):
            return self

        async def execute(self):
            nonlocal query_calls
            query_calls += 1
            if query_calls == 1:
                raise httpx.ConnectTimeout("proxy connect timeout")
            return FakeResult()

    class FakeClient:
        def table(self, table_name):
            assert table_name == "sandboxes"
            return FakeQuery()

    async def fake_sleep(delay):
        return None

    async def fake_delete_sandbox(sandbox_id):
        deleted.append(sandbox_id)

    monkeypatch.setattr(sandbox.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sandbox, "delete_sandbox", fake_delete_sandbox)

    await sandbox.cleanup_sandbox_for_task(FakeClient(), "task-id")

    assert query_calls == 2
    assert deleted == ["sandbox-id"]


@pytest.mark.asyncio
async def test_cleanup_sandbox_for_task_swallows_lookup_timeout(monkeypatch):
    """Cleanup is best effort when sandbox lookup keeps timing out."""
    query_calls = 0

    class FakeQuery:
        def select(self, columns):
            return self

        def eq(self, column, value):
            return self

        def maybe_single(self):
            return self

        async def execute(self):
            nonlocal query_calls
            query_calls += 1
            raise httpx.ConnectTimeout("proxy connect timeout")

    class FakeClient:
        def table(self, table_name):
            assert table_name == "sandboxes"
            return FakeQuery()

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(sandbox.asyncio, "sleep", fake_sleep)

    await sandbox.cleanup_sandbox_for_task(FakeClient(), "task-id")

    assert query_calls == 3


@pytest.mark.asyncio
async def test_clone_sandbox_returns_none_when_sdk_has_no_clone(monkeypatch):
    class FakeDaytona:
        async def get(self, sandbox_id):
            return object()

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())

    assert await sandbox.clone_sandbox("source-sandbox") is None


@pytest.mark.asyncio
async def test_clone_sandbox_uses_daytona_clone(monkeypatch):
    class FakeDaytona:
        async def get(self, sandbox_id):
            assert sandbox_id == "source-sandbox"
            return {"id": sandbox_id}

        async def clone(self, source):
            assert source == {"id": "source-sandbox"}
            return {"id": "sandbox-copy"}

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())

    assert await sandbox.clone_sandbox("source-sandbox") == "sandbox-copy"


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

    assert inserted == [
        {"sandbox_id": "sandbox-copy", "task_id": "task-copy", "account_id": "account"}
    ]


@pytest.mark.asyncio
async def test_clone_sandbox_uses_experimental_fork(monkeypatch):
    class FakeSource:
        async def _experimental_fork(self):
            return {"id": "sandbox-copy"}

    class FakeDaytona:
        async def get(self, sandbox_id):
            assert sandbox_id == "source-sandbox"
            return FakeSource()

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())

    assert await sandbox.clone_sandbox("source-sandbox") == "sandbox-copy"


@pytest.mark.asyncio
async def test_clone_sandbox_uses_public_snapshot_create_fallback(monkeypatch):
    created_params = []

    class FakeSnapshot:
        id = "snapshot-id"

    class FakeSource:
        async def create_snapshot(self):
            return FakeSnapshot()

    class FakeDaytona:
        async def get(self, sandbox_id):
            assert sandbox_id == "source-sandbox"
            return FakeSource()

        async def create(self, params):
            created_params.append(params)
            return {"id": "sandbox-copy"}

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())

    assert await sandbox.clone_sandbox("source-sandbox") == "sandbox-copy"
    assert len(created_params) == 1
    assert created_params[0].snapshot == "snapshot-id"


@pytest.mark.asyncio
async def test_clone_sandbox_uses_experimental_snapshot_create_fallback(monkeypatch):
    created_params = []
    snapshot_names = []

    class FakeSource:
        async def _experimental_create_snapshot(self, name):
            snapshot_names.append(name)

    class FakeDaytona:
        async def get(self, sandbox_id):
            assert sandbox_id == "source-sandbox"
            return FakeSource()

        async def create(self, params):
            created_params.append(params)
            return {"id": "sandbox-copy"}

    monkeypatch.setattr(sandbox, "_get_daytona", lambda: FakeDaytona())

    assert await sandbox.clone_sandbox("source-sandbox") == "sandbox-copy"
    assert len(snapshot_names) == 1
    assert snapshot_names[0].startswith("branch-source-sandbox-")
    assert len(created_params) == 1
    assert created_params[0].snapshot == snapshot_names[0]


def test_sandbox_helpers_exported_from_db_service_package():
    from customized_areal.db_service import bind_sandbox_to_task, clone_sandbox

    assert bind_sandbox_to_task is sandbox.bind_sandbox_to_task
    assert clone_sandbox is sandbox.clone_sandbox

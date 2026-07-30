# Multica Bridge Caller Credentials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forward each AReaL caller's `MULTICA_API_KEY` through the DB bridge to Multica, encrypted at rest, so `BRIDGE_MULTICA_UPSTREAM_API_KEY` is no longer needed.

**Architecture:** `MulticaEnvDispatchClient` and `MulticaDagClient` send the caller PAT as a bearer token to the AReaL-side stub. The stub requires a shared Fernet key, encrypts `Authorization` before enqueue, and the Multica-side executor decrypts and forwards that header while dropping alternate credential headers. `multica_api` rows are always redacted after terminal completion or failure.

**Tech Stack:** Python 3.12, httpx, FastAPI, cryptography/Fernet, pytest, Docker Compose, Supabase RPC bridge.

## Global Constraints

- Follow red-green-refactor: no production change before its focused failing test.
- Never log, include in exceptions, or persist a plaintext Multica PAT.
- `BRIDGE_HEADER_ENCRYPTION_KEY` is mandatory on the AReaL stub and Multica executor because those processes host `multica_api` channels.
- The two bridge processes must use the same valid Fernet key.
- Preserve all non-`multica_api` authentication behavior.
- Preserve unrelated dirty files in both the root repository and the nested `multica/` repository.
- Commit root-repository and nested-`multica/` changes separately, staging only task-owned paths.
- Do not edit historical files under `docs/superpowers/reports/`, old specs, or old plans merely to rewrite their description of the former behavior.
- After code changes, run `graphify update .` from the root repository.

---

## File Structure

- `customized_areal/tree_search/agents/multica_dag_client.py`: resolve and attach the caller PAT during DAG polling.
- `customized_areal/tree_search/tests/test_multica_dag_client.py`: client credential behavior.
- `customized_areal/tree_search/tests/test_env_dispatch_client.py`: regression coverage for the already-supported dispatch credential.
- `multica/db_bridge/relay.py`: remove alternate credentials while preserving `Authorization`.
- `multica/db_bridge/executor.py`: forward caller auth and redact terminal `multica_api` rows.
- `multica/db_bridge/config.py`: remove the executor PAT and support required cipher construction.
- `multica/db_bridge/stub_server.py`: require encryption when serving `multica_api` channels.
- `multica/db_bridge/tests/test_executor.py`: executor pass-through, failure, and redaction tests.
- `multica/db_bridge/tests/test_env_dispatch_channels.py`: encrypted end-to-end POST/DELETE coverage.
- `multica/db_bridge/tests/test_config.py`: configuration removal and required-cipher tests.
- `multica/db_bridge/tests/test_leagent_channels.py`: supply the required cipher to the shared AReaL-side stub test fixture.
- `multica/db_bridge/tests/test_integration_e2e.py`: supply the required cipher to the shared AReaL-side stub test fixture.
- `multica/db_bridge/README.md`, `.env.areal.example`, `.env.multica.example`, and `docker-compose.selfhost.yml`: deployment migration documentation and wiring.
- `customized_areal/tree_search/agents/multica_environment_protocol.md`: describe caller credential pass-through.

---

### Task 1: Authenticate DAG polling with the caller PAT

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py`
- Modify: `customized_areal/tree_search/tests/test_multica_dag_client.py`
- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`

**Interfaces:**
- Consumes: explicit `api_key: str | None` and environment variable `MULTICA_API_KEY`.
- Produces: `MulticaDagClient(..., api_key=None)` and authenticated `GET /api/v1/env-dispatch/{project_id}/dag` polling.

- [ ] **Step 1: Write failing DAG credential tests**

Replace `test_get_dag_reads_bridge_stub_url_from_env_and_sends_no_auth` with environment-derived auth coverage and add explicit-argument precedence:

```python
def test_get_dag_reads_bridge_url_and_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AREAL_BRIDGE_STUB_URL", "http://127.0.0.1:9101")
    monkeypatch.setenv("MULTICA_API_KEY", "mul_env")
    seen: dict = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient(_transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert seen["url"].startswith("http://127.0.0.1:9101")
    assert seen["auth"] == "Bearer mul_env"


def test_get_dag_explicit_api_key_overrides_environment(monkeypatch):
    monkeypatch.setenv("MULTICA_API_KEY", "mul_env")

    def handler(request):
        assert request.headers["authorization"] == "Bearer mul_explicit"
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient(
        "http://stub",
        api_key="mul_explicit",
        _transport=httpx.MockTransport(handler),
    )
    client.get_dag("proj-1", timeout=5.0, interval=0.0)
```

- [ ] **Step 2: Run the DAG tests and verify RED**

Run:

```bash
uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py -q
```

Expected: FAIL because `MulticaDagClient` does not accept `api_key` and sends no `Authorization` header.

- [ ] **Step 3: Implement the minimal DAG client change**

Add the constructor argument and resolved credential:

```python
def __init__(
    self,
    base_url: str | None = None,
    *,
    api_key: str | None = None,
    poll_interval: float = 2.0,
    poll_timeout: float = 300.0,
    poll_backoff: float = 1.5,
    poll_max_interval: float = 10.0,
    http_timeout: float = 10.0,
    _transport: httpx.BaseTransport | None = None,
) -> None:
    ...
    self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
```

Build request headers once per `get_dag` call and use them for every retry:

```python
headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
...
resp = client.get(url, headers=headers)
```

Update the module and class docstrings to state that caller auth passes through the encrypted bridge.

- [ ] **Step 4: Add dispatch-client credential regression coverage**

The dispatch client already resolves and sends the PAT, so add a characterization test without changing its implementation:

```python
def test_create_env_dispatch_sends_environment_api_key(monkeypatch):
    monkeypatch.setenv("MULTICA_API_KEY", "mul_dispatch")

    def handler(request):
        assert request.headers["authorization"] == "Bearer mul_dispatch"
        return httpx.Response(201, json={"project_id": "p1"})

    client = MulticaEnvDispatchClient(
        base_url="http://stub", transport=_transport(handler)
    )
    project_id = asyncio.run(
        client.create_env_dispatch(
            mode="scratch", dispatch_type="issue", agent_id="agent-1"
        )
    )
    assert project_id == "p1"
```

- [ ] **Step 5: Run focused client tests and verify GREEN**

Run:

```bash
uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the root client change**

```bash
git add customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py
git commit -m "feat: authenticate Multica DAG bridge requests"
```

---

### Task 2: Forward caller credentials and always redact Multica rows

**Files:**
- Modify: `multica/db_bridge/relay.py`
- Modify: `multica/db_bridge/executor.py`
- Modify: `multica/db_bridge/tests/test_executor.py`
- Modify: `multica/db_bridge/tests/test_env_dispatch_channels.py`

**Interfaces:**
- Consumes: decrypted request headers containing caller `Authorization`.
- Produces: `relay.strip_alternate_credentials(headers) -> dict[str, str]`; forwarded caller bearer auth; mandatory terminal redaction for `multica_api`.

- [ ] **Step 1: Replace key-injection tests with failing pass-through tests**

In `test_executor.py`, replace the two old injection/stripping tests with:

```python
async def test_multica_api_forwards_caller_authorization_only():
    cfg = _config(BRIDGE_MULTICA_UPSTREAM_URL="http://127.0.0.1:9999")
    db = BridgeDB(cfg, client=FakeSupabaseClient())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["x_api_key"] = request.headers.get("x-api-key")
        captured["x_admin"] = request.headers.get("x-admin-api-key")
        return httpx.Response(200, json={"segments": []})

    ex = Executor(db, "multica", config=cfg, client=_exec_client(handler))
    await db.insert_request(
        ENV_DISPATCH_DAG,
        user_id=USER_ID,
        method="GET",
        path="/api/v1/env-dispatch/proj-1/dag",
        headers={
            "authorization": "Bearer mul_caller",
            "x-api-key": "alternate",
            "x-admin-api-key": "admin",
        },
        content_type=None,
        body=b"",
    )

    assert await ex.process_one(ENV_DISPATCH_DAG, "w0") is True
    assert captured == {
        "auth": "Bearer mul_caller",
        "x_api_key": None,
        "x_admin": None,
    }
    row = db.client.tables[ENV_DISPATCH_DAG.table]
    assert next(iter(row.values()))["request_headers"]["authorization"] == "REDACTED"
```

Add a transport-failure test that asserts the row becomes `error` and its stored authorization becomes `REDACTED`.

- [ ] **Step 2: Update the env-dispatch end-to-end expectation**

Rename `test_env_dispatch_post_relays_rollouts_and_injects_upstream_key` to `test_env_dispatch_post_relays_caller_authorization` and change its final assertion to:

```python
assert seen["auth"] == "Bearer multica-key"
```

Remove `BRIDGE_MULTICA_UPSTREAM_API_KEY` from that test module's `_config` fixture.

- [ ] **Step 3: Run focused bridge tests and verify RED**

Run from the nested repository:

```bash
cd multica/db_bridge
uv run python -m pytest -q tests/test_executor.py tests/test_env_dispatch_channels.py
```

Expected: FAIL because the executor strips caller auth, injects the configured executor key, and only conditionally redacts successful rows.

- [ ] **Step 4: Preserve only the supported caller credential**

Replace the all-credential stripping helper with a helper that removes alternate credentials but keeps bearer auth:

```python
_ALTERNATE_CRED_HEADERS: Final = frozenset({"x-api-key", "x-admin-api-key"})


def strip_alternate_credentials(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop alternate credentials while preserving caller Authorization."""
    return {
        key: value
        for key, value in _normalize(headers)
        if key not in _ALTERNATE_CRED_HEADERS
    }
```

In `Executor._forward`, replace executor-key injection with:

```python
elif channel.group == "multica_api":
    headers = relay.strip_alternate_credentials(headers)
```

- [ ] **Step 5: Centralize mandatory redaction**

Add:

```python
def _must_redact(self, channel: Channel) -> bool:
    return channel.group == "multica_api" or self._config.redact_tokens_after_complete

async def _redact_if_required(self, channel: Channel, row_id: str) -> None:
    if not self._must_redact(channel):
        return
    try:
        await self._db.redact_headers(channel, row_id)
    except Exception:  # noqa: BLE001 -- redaction cannot replace relay outcome
        logger.exception(
            "executor token redaction failed channel=%s id=%s",
            channel.name,
            row_id,
        )
```

Call `_redact_if_required` after a successful `complete` and after a successful `fail`. Do not include exception text from decryption or redaction in logs when it could contain credential material.

- [ ] **Step 6: Run focused bridge tests and verify GREEN**

Run:

```bash
cd multica/db_bridge
uv run python -m pytest -q tests/test_executor.py tests/test_env_dispatch_channels.py tests/test_hardening.py
```

Expected: PASS.

- [ ] **Step 7: Commit the nested bridge behavior change**

```bash
git -C multica add db_bridge/relay.py db_bridge/executor.py db_bridge/tests/test_executor.py db_bridge/tests/test_env_dispatch_channels.py
git -C multica commit -m "feat(db-bridge): forward caller Multica credentials"
```

---

### Task 3: Require encryption for every `multica_api` bridge side

**Files:**
- Modify: `multica/db_bridge/config.py`
- Modify: `multica/db_bridge/stub_server.py`
- Modify: `multica/db_bridge/executor.py`
- Modify: `multica/db_bridge/tests/test_config.py`
- Modify: `multica/db_bridge/tests/test_executor.py`
- Modify: `multica/db_bridge/tests/test_env_dispatch_channels.py`
- Modify: `multica/db_bridge/tests/test_leagent_channels.py`
- Modify: `multica/db_bridge/tests/test_integration_e2e.py`

**Interfaces:**
- Consumes: `BridgeConfig.header_encryption_key` and the channels selected by a process side.
- Produces: `BridgeConfig.build_cipher(required: bool = False)`; startup failure when a process hosts `multica_api` channels without a key.

- [ ] **Step 1: Write failing required-encryption tests**

Add to `test_config.py`:

```python
def test_build_cipher_required_rejects_missing_key():
    cfg = BridgeConfig.from_env(_MINIMAL)
    with pytest.raises(RuntimeError, match="BRIDGE_HEADER_ENCRYPTION_KEY"):
        cfg.build_cipher(required=True)
```

Add lifecycle assertions:

```python
def test_areal_stub_requires_header_encryption_key():
    cfg = _config()
    db = BridgeDB(cfg, client=FakeSupabaseClient())
    with pytest.raises(RuntimeError, match="BRIDGE_HEADER_ENCRYPTION_KEY"):
        create_stub_app(db, "areal", cfg)


def test_multica_executor_requires_header_encryption_key():
    cfg = _config()
    db = BridgeDB(cfg, client=FakeSupabaseClient())
    with pytest.raises(RuntimeError, match="BRIDGE_HEADER_ENCRYPTION_KEY"):
        Executor(db, "multica", config=cfg)
```

- [ ] **Step 2: Run focused lifecycle tests and verify RED**

Run:

```bash
cd multica/db_bridge
uv run python -m pytest -q tests/test_config.py tests/test_entrypoints.py
```

Expected: FAIL because cipher construction is optional.

- [ ] **Step 3: Implement required cipher construction**

Change `BridgeConfig.build_cipher` to:

```python
def build_cipher(self, *, required: bool = False):
    """Build the configured header cipher, optionally requiring its key."""
    if required and not self.header_encryption_key:
        raise RuntimeError(
            "BRIDGE_HEADER_ENCRYPTION_KEY must be set for multica_api channels"
        )
    from . import crypto

    return crypto.build_cipher(self.header_encryption_key)
```

In `create_stub_app`, compute channels before the cipher and require encryption when any served channel has `group == "multica_api"`:

```python
channels = stub_channels(side)
cipher = config.build_cipher(
    required=any(channel.group == "multica_api" for channel in channels)
)
```

In `Executor.__init__`, require encryption when any `executor_channels(side)` entry belongs to `multica_api`.

- [ ] **Step 4: Give affected test harnesses one shared valid key**

In each `_config` helper that creates the AReaL stub or Multica executor, add the deterministic test-only Fernet key:

```python
"BRIDGE_HEADER_ENCRYPTION_KEY": (
    "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
),
```

Update `test_executor.py`, `test_env_dispatch_channels.py`, `test_leagent_channels.py`, and `test_integration_e2e.py`. Keep the missing-key lifecycle tests based on a config that deliberately omits the key.

- [ ] **Step 5: Assert ciphertext is stored during an env-dispatch round trip**

In `test_env_dispatch_channels.py`, inspect the retained row after the request. Because mandatory terminal redaction replaces the encrypted value, capture the pending row before executor completion or wrap `BridgeDB.insert_request` to record the inserted header and assert:

```python
assert captured_stored_auth.startswith("enc:v1:")
assert "multica-key" not in captured_stored_auth
```

Also keep the upstream assertion `Bearer multica-key` to prove decrypt-before-forward.

- [ ] **Step 6: Run all bridge tests and verify GREEN**

Run:

```bash
cd multica/db_bridge
uv run python -m pytest -q
```

Expected: PASS, with live PostgreSQL tests skipped only when `BRIDGE_TEST_PG_DSN` is unset.

- [ ] **Step 7: Commit required encryption in the nested repository**

```bash
git -C multica add db_bridge/config.py db_bridge/stub_server.py db_bridge/executor.py db_bridge/tests/test_config.py db_bridge/tests/test_entrypoints.py db_bridge/tests/test_executor.py db_bridge/tests/test_env_dispatch_channels.py db_bridge/tests/test_leagent_channels.py db_bridge/tests/test_integration_e2e.py
git -C multica commit -m "feat(db-bridge): require encryption for Multica auth"
```

---

### Task 4: Remove the executor PAT configuration and update deployment docs

**Files:**
- Modify: `multica/db_bridge/config.py`
- Modify: `multica/db_bridge/tests/test_config.py`
- Modify: `multica/docker-compose.selfhost.yml`
- Modify: `multica/db_bridge/.env.areal.example`
- Modify: `multica/db_bridge/.env.multica.example`
- Modify: `multica/db_bridge/README.md`
- Modify: `customized_areal/tree_search/agents/multica_environment_protocol.md`

**Interfaces:**
- Consumes: shared `BRIDGE_HEADER_ENCRYPTION_KEY` and caller-side `MULTICA_API_KEY`.
- Produces: deployment configuration with no `BRIDGE_MULTICA_UPSTREAM_API_KEY` support.

- [ ] **Step 1: Write the failing configuration-removal test**

Replace `test_multica_upstream_url_default_and_env` assertions about the old key with:

```python
def test_multica_upstream_url_default_and_env():
    cfg = BridgeConfig.from_env(_MINIMAL)
    assert cfg.multica_upstream_url == "http://127.0.0.1:8080"
    assert not hasattr(cfg, "multica_upstream_api_key")
    cfg2 = BridgeConfig.from_env(
        {**_MINIMAL, "BRIDGE_MULTICA_UPSTREAM_URL": "http://127.0.0.1:9999"}
    )
    assert cfg2.upstream_for_group("multica_api") == "http://127.0.0.1:9999"
```

- [ ] **Step 2: Run the config test and verify RED**

Run:

```bash
cd multica/db_bridge
uv run python -m pytest -q tests/test_config.py
```

Expected: FAIL because `BridgeConfig.multica_upstream_api_key` still exists.

- [ ] **Step 3: Remove the obsolete configuration API**

Delete `ENV_BRIDGE_MULTICA_UPSTREAM_API_KEY`, the `multica_upstream_api_key` dataclass field, and its `BridgeConfig.from_env` assignment. Keep `BRIDGE_MULTICA_UPSTREAM_URL` unchanged.

- [ ] **Step 4: Replace Compose secret wiring**

In `db-bridge-executor-multica.environment`, remove:

```yaml
BRIDGE_MULTICA_UPSTREAM_API_KEY: ${BRIDGE_MULTICA_UPSTREAM_API_KEY:-}
```

and add a required shared encryption key:

```yaml
BRIDGE_HEADER_ENCRYPTION_KEY: ${BRIDGE_HEADER_ENCRYPTION_KEY:?BRIDGE_HEADER_ENCRYPTION_KEY must be set for db-bridge}
```

Rewrite the surrounding comments to say that caller `MULTICA_API_KEY` headers arrive encrypted through Supabase and are decrypted immediately before forwarding.

- [ ] **Step 5: Update active documentation and examples**

Make these exact semantic changes:

- `.env.areal.example`: set `MULTICA_API_KEY=mul_...`, require `BRIDGE_HEADER_ENCRYPTION_KEY`, and state that dispatch and DAG clients send the caller token.
- `.env.multica.example`: remove `BRIDGE_MULTICA_UPSTREAM_API_KEY`; require the same `BRIDGE_HEADER_ENCRYPTION_KEY` value as the AReaL stub.
- `README.md`: change encryption from optional to required for `multica_api`, document Fernet key generation with `python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'`, and describe caller credential pass-through and mandatory redaction.
- `multica_environment_protocol.md`: replace executor-key injection with encrypted caller-token forwarding for POST, DELETE, and DAG GET.

- [ ] **Step 6: Verify obsolete active references are gone**

Run:

```bash
rg -n "BRIDGE_MULTICA_UPSTREAM_API_KEY|multica_upstream_api_key" multica/db_bridge multica/docker-compose.selfhost.yml customized_areal/tree_search/agents/multica_environment_protocol.md
```

Expected: no matches. Historical design/report files outside these active paths may still mention the retired setting.

- [ ] **Step 7: Run config and documentation checks**

Run:

```bash
cd multica/db_bridge
uv run python -m pytest -q tests/test_config.py tests/test_executor.py tests/test_env_dispatch_channels.py
cd ../..
mdformat --check multica/db_bridge/README.md customized_areal/tree_search/agents/multica_environment_protocol.md
```

Expected: tests PASS and Markdown formatting PASS.

- [ ] **Step 8: Commit both repository portions separately**

Nested Multica repository:

```bash
git -C multica add db_bridge/config.py db_bridge/tests/test_config.py db_bridge/.env.areal.example db_bridge/.env.multica.example db_bridge/README.md docker-compose.selfhost.yml
git -C multica commit -m "chore(db-bridge): retire executor Multica PAT"
```

Root repository:

```bash
git add customized_areal/tree_search/agents/multica_environment_protocol.md
git commit -m "docs: document encrypted Multica caller auth"
```

---

### Task 5: Final verification and graph refresh

**Files:**
- Modify: `graphify-out/` generated graph files.

**Interfaces:**
- Consumes: completed client, bridge, and deployment changes.
- Produces: test evidence, formatting evidence, and an updated repository graph.

- [ ] **Step 1: Run root client tests**

```bash
uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_workflow_wiring.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete bridge suite**

```bash
cd multica/db_bridge
uv run python -m pytest -q
```

Expected: PASS; explicitly report live-DB skips caused by missing `BRIDGE_TEST_PG_DSN`.

- [ ] **Step 3: Run repository formatting and lint checks**

From the root repository, activate `.venv` if it exists and run:

```bash
pre-commit run --all-files
```

Expected: PASS. If the environment lacks `pre-commit`, report that as an unavailable verification instead of claiming it passed.

- [ ] **Step 4: Refresh Graphify**

```bash
graphify update .
```

Expected: graph update completes successfully. Dirty generated `graphify-out/` files are expected; do not stage them unless the user explicitly requests graph artifacts in the commits.

- [ ] **Step 5: Audit both worktrees**

```bash
git status --short
git -C multica status --short
git diff --check
git -C multica diff --check
```

Expected: only pre-existing unrelated changes and intentionally unstaged generated graph files remain; no task-owned change is uncommitted and no whitespace errors are reported.


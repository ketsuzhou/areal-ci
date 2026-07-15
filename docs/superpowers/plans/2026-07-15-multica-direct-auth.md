# MultiCA Direct Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit terminal email-code login to AReaL, securely reuse the resulting MultiCA PAT for direct client calls, and remove the obsolete MultiCA-side bridge executor.

**Architecture:** A focused `multica_auth.py` module owns login, persistence, and credential resolution. Both MultiCA clients depend on that module and address `MULTICA_BASE_URL` directly; the self-host Compose file retains only the MultiCA-side stub needed for MultiCA-to-AReaL traffic.

**Tech Stack:** Python 3.12, httpx, argparse, pathlib, pytest, Docker Compose, Bash

## Global Constraints

- Interactive login is explicit and never starts from a training worker.
- Store credentials at `customized_areal/tree_search/agents/credentials.json` with mode `0600` and add that exact path to `.gitignore`.
- Resolve credentials in this order: explicit `api_key`, `MULTICA_API_KEY`, saved PAT.
- Bind saved credentials to the normalized `MULTICA_BASE_URL`.
- Never print, log, or include a JWT, verification code, or PAT in an exception.
- Create 90-day PATs and validate them through `GET /api/me` before persistence.
- Preserve `db-bridge-stub-multica` for MultiCA-to-AReaL calls.
- Preserve all unrelated root and nested-`multica` worktree changes.
- Follow red-green-refactor for every behavior change.
- Run `graphify update .` after code changes.

---

## File Structure

- `customized_areal/tree_search/agents/multica_auth.py`: login command, credential-file IO, URL normalization, and credential resolution.
- `customized_areal/tree_search/tests/test_multica_auth.py`: authentication and persistence unit tests.
- `customized_areal/tree_search/agents/multica_client.py`: asynchronous direct client using shared credential resolution.
- `customized_areal/tree_search/agents/multica_dag_client.py`: synchronous direct DAG client using shared credential resolution.
- `customized_areal/tree_search/tests/test_env_dispatch_client.py`: dispatch credential precedence and 401 guidance.
- `customized_areal/tree_search/tests/test_multica_dag_client.py`: direct URL, authentication, precedence, and 401 guidance.
- `customized_areal/tree_search/agents/multica_environment_protocol.md`: direct AReaL-to-MultiCA topology documentation.
- `.gitignore`: exclude the credential file.
- `multica/docker-compose.selfhost.yml`: remove the obsolete MultiCA executor and correct retained-stub comments.
- `multica/scripts/selfhost-config.test.sh`: assert the retained/removed service topology.

### Task 1: Credential login and secure persistence

**Files:**
- Create: `customized_areal/tree_search/agents/multica_auth.py`
- Create: `customized_areal/tree_search/tests/test_multica_auth.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `DEFAULT_CREDENTIALS_PATH: Path`
- Produces: `MulticaAuthError(RuntimeError)`
- Produces: `normalize_base_url(base_url: str) -> str`
- Produces: `load_saved_api_key(base_url: str, *, credentials_path: Path | None = None) -> str`
- Produces: `resolve_api_key(base_url: str, explicit_api_key: str | None = None, *, credentials_path: Path | None = None) -> str`
- Produces: `login(base_url: str, email: str, code: str, *, credentials_path: Path | None = None, transport: httpx.BaseTransport | None = None) -> dict[str, str]`
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing credential persistence tests**

Create tests that pin URL normalization, URL binding, explicit/environment/saved precedence, malformed-file failure, and owner-only permissions:

```python
def test_resolve_api_key_prefers_explicit_then_environment_then_saved(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080/", "mul_saved", credentials_path=path)
    monkeypatch.setenv("MULTICA_API_KEY", "mul_env")
    assert resolve_api_key("http://multica:8080", "mul_explicit", credentials_path=path) == "mul_explicit"
    assert resolve_api_key("http://multica:8080", credentials_path=path) == "mul_env"
    monkeypatch.delenv("MULTICA_API_KEY")
    assert resolve_api_key("http://multica:8080", credentials_path=path) == "mul_saved"


def test_save_credentials_is_owner_only(tmp_path):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080", "mul_saved", credentials_path=path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
```

- [ ] **Step 2: Run persistence tests and verify RED**

Run:

```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_multica_auth.py -q
```

Expected: collection fails because `multica_auth` does not exist.

- [ ] **Step 3: Implement minimal persistence and resolution**

Implement a versioned JSON payload and atomic writer:

```python
DEFAULT_CREDENTIALS_PATH = Path(__file__).with_name("credentials.json")


def normalize_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise MulticaAuthError("MultiCA base URL must be an absolute HTTP(S) URL")
    return value


def save_credentials(base_url: str, api_key: str, *, credentials_path: Path | None = None) -> None:
    credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
    payload = {"version": 1, "base_url": normalize_base_url(base_url), "api_key": api_key}
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{credentials_path.name}.", dir=credentials_path.parent)
    tmp_path = Path(raw_tmp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, credentials_path)
        os.chmod(credentials_path, 0o600)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
```

Reject symlinks/non-regular credential files before reading and return actionable `MulticaAuthError` messages without embedding file contents or tokens.

- [ ] **Step 4: Run persistence tests and verify GREEN**

Run the Task 1 test command. Expected: persistence and resolution tests pass.

- [ ] **Step 5: Write failing login-flow tests**

Use `httpx.MockTransport` to assert the exact endpoint sequence and authorization transition:

```python
def test_login_exchanges_email_code_for_validated_pat(tmp_path):
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("authorization")))
        if request.url.path == "/auth/send-code":
            return httpx.Response(200, json={"message": "Verification code sent"})
        if request.url.path == "/auth/verify-code":
            return httpx.Response(200, json={"token": "jwt_temp", "user": {"email": "user@example.com"}})
        if request.url.path == "/api/tokens":
            assert json.loads(request.content)["expires_in_days"] == 90
            return httpx.Response(201, json={"token": "mul_created"})
        if request.url.path == "/api/me":
            return httpx.Response(200, json={"email": "user@example.com", "name": "User"})
        raise AssertionError(request.url.path)

    result = login(
        "http://multica:8080",
        "user@example.com",
        "123456",
        credentials_path=tmp_path / "credentials.json",
        transport=httpx.MockTransport(handler),
    )
    assert result["email"] == "user@example.com"
    assert seen == [
        ("/auth/send-code", None),
        ("/auth/verify-code", None),
        ("/api/tokens", "Bearer jwt_temp"),
        ("/api/me", "Bearer mul_created"),
    ]
```

Also assert that a failed verification, PAT creation, or `/api/me` validation does not replace an existing credential file.

- [ ] **Step 6: Run login tests and verify RED**

Expected: FAIL because `login` has not implemented the HTTP exchange.

- [ ] **Step 7: Implement login and explicit CLI**

Use `httpx.Client`, `getpass.getpass("Verification code: ")`, `input("Email: ")`, and `argparse` with a required `login` subcommand. Raise sanitized errors using only the endpoint and status code. Print only `Authenticated as <email>` and the credential path after a successful save.

- [ ] **Step 8: Ignore the credential file and verify GREEN**

Add this exact line to `.gitignore`:

```gitignore
customized_areal/tree_search/agents/credentials.json
```

Run all Task 1 tests. Expected: PASS.

### Task 2: Direct authenticated MultiCA clients

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_client.py`
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py`
- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Modify: `customized_areal/tree_search/tests/test_multica_dag_client.py`

**Interfaces:**
- Consumes: `resolve_api_key(base_url, explicit_api_key) -> str` from Task 1.
- Produces: both clients address MultiCA directly and attach bearer authentication.

- [ ] **Step 1: Write failing dispatch saved-PAT and 401 tests**

Patch `DEFAULT_CREDENTIALS_PATH` to a temporary credential file, clear `MULTICA_API_KEY`, and assert dispatch sends the saved PAT. Add a 401 response test asserting the exception says to run the `multica_auth login` command and does not contain the PAT.

- [ ] **Step 2: Run dispatch tests and verify RED**

Run:

```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -q
```

Expected: saved-PAT test fails because the client only reads `MULTICA_API_KEY`.

- [ ] **Step 3: Integrate shared credentials into the dispatch client**

Resolve the base URL first, then set `self._api_key = resolve_api_key(self._base_url, api_key)`. Centralize non-success errors so HTTP 401 adds the login-command guidance without response headers or credentials.

- [ ] **Step 4: Run dispatch tests and verify GREEN**

Run the Task 2 dispatch command. Expected: PASS.

- [ ] **Step 5: Replace bridge-specific DAG tests with failing direct-call tests**

Replace `test_get_dag_reads_bridge_stub_url_from_env_and_sends_no_auth` with:

```python
def test_get_dag_reads_direct_url_and_saved_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTICA_BASE_URL", "http://multica:8080")
    monkeypatch.delenv("AREAL_BRIDGE_STUB_URL", raising=False)
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080", "mul_saved", credentials_path=path)
    monkeypatch.setattr(multica_auth, "DEFAULT_CREDENTIALS_PATH", path)
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_dag_payload())

    dag = MulticaDagClient(_transport=httpx.MockTransport(handler)).get_dag("proj-1")
    assert isinstance(dag, AssembledDag)
    assert seen["url"].startswith("http://multica:8080")
    assert seen["auth"] == "Bearer mul_saved"
```

Add explicit/environment precedence and sanitized 401 guidance tests.

- [ ] **Step 6: Run DAG tests and verify RED**

Run:

```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_multica_dag_client.py -q
```

Expected: FAIL because the client still requires `AREAL_BRIDGE_STUB_URL` and sends no bearer token.

- [ ] **Step 7: Implement direct DAG authentication**

Add `api_key: str | None = None`, resolve `base_url or MULTICA_BASE_URL`, resolve the shared credential, and pass one bearer-header dictionary to every `client.get` poll. Update module/class docstrings and 401 handling; retain existing 202/502/503/504 polling semantics.

- [ ] **Step 8: Run all client tests and verify GREEN**

Run:

```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_multica_auth.py customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py -q
```

Expected: PASS.

### Task 3: Remove the obsolete Compose executor and document topology

**Files:**
- Modify: `multica/scripts/selfhost-config.test.sh`
- Modify: `multica/docker-compose.selfhost.yml`
- Modify: `customized_areal/tree_search/agents/multica_environment_protocol.md`

**Interfaces:**
- Consumes: direct authenticated client topology from Task 2.
- Produces: self-host configuration with `db-bridge-stub-multica` present and `db-bridge-executor-multica` absent.

- [ ] **Step 1: Add failing Compose topology assertions**

After rendering `config`, assert the retained service exists and the obsolete service and executor-only variables do not:

```bash
require_config "$config" 'db-bridge-stub-multica:'

for obsolete in db-bridge-executor-multica BRIDGE_MULTICA_UPSTREAM_URL BRIDGE_MULTICA_UPSTREAM_API_KEY; do
  if grep -Fq "$obsolete" <<<"$config"; then
    echo "Obsolete AReaL-to-MultiCA bridge setting remains: $obsolete"
    exit 1
  fi
done
```

- [ ] **Step 2: Run the self-host test and verify RED**

Run:

```bash
bash multica/scripts/selfhost-config.test.sh
```

Expected: FAIL because `db-bridge-executor-multica` is still rendered.

- [ ] **Step 3: Remove only the obsolete executor**

Delete the `db-bridge-executor-multica` comment and service block from `multica/docker-compose.selfhost.yml`. Keep `db-bridge-stub-multica`; edit its comments to describe only MultiCA-to-AReaL relay behavior and remove the “matching executor below” wording.

- [ ] **Step 4: Update the protocol document**

Rename “AReaL → db_bridge → Multica API Surface” to “AReaL → Multica API Surface”, document `MULTICA_BASE_URL` plus the explicit login command, and state that direct requests use the saved PAT. Keep the later gateway-group section because it documents the opposite MultiCA-to-AReaL direction.

- [ ] **Step 5: Run topology and focused client tests**

Run the Compose test and the combined Task 2 pytest command. Expected: PASS.

### Task 4: Final verification and graph refresh

**Files:**
- Modify mechanically: `graphify-out/**`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified implementation and current repository graph.

- [ ] **Step 1: Run formatting and lint checks for touched files**

Run the available repository formatter/linter through the active environment. If the configured `.venv` remains unavailable, run the same tools from `.venv-test` when present and record any tooling that cannot start.

- [ ] **Step 2: Run focused tests from a working Python interpreter**

Run the combined authentication/client tests and `bash multica/scripts/selfhost-config.test.sh`. Expected: all pass; if Docker is unavailable, record the exact environmental failure and validate the Compose YAML structurally with the available parser.

- [ ] **Step 3: Refresh Graphify**

Run:

```bash
graphify update .
```

Expected: graph update completes without errors.

- [ ] **Step 4: Review scoped diffs and worktree state**

Use `git diff --check`, root `git status --short`, nested `git -C multica status --short`, and scoped diffs. Confirm no existing dirty files were overwritten and no credential file is tracked.

- [ ] **Step 5: Report verification evidence**

Report each command, pass/fail status, environmental skips, changed files, and the explicit login command. Do not claim unavailable checks passed.

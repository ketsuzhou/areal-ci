---
change: env-dispatch-dag-db-bridge
design-doc: docs/superpowers/specs/2026-07-13-env-dispatch-dag-db-bridge-design.md
base-ref: 853e5cbcda6929e80961ed778194c1402fb7d9ba
multica-base-ref: 2fad5d51e0839a23b95ba0f6f222e520444a6477
---

# env-dispatch-dag-db-bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route AReaL's env-dispatch DAG fetch and multica's `close_segment` call through `db_bridge`, and reconcile the bridge's Side/group naming so env-dispatch channels forward to the multica server.

**Architecture:** Add two db_bridge channels (`env_dispatch_dag` in a new `multica_api` group; `rl_close_segment` in the existing `gateway` group), introduce a `multica` Side with its own `BRIDGE_MULTICA_UPSTREAM_URL`, repoint `MulticaDagClient` at the areal-side stub, and leave the multica `arealrl` client untouched (it already targets the stub). See `docs/superpowers/specs/2026-07-13-env-dispatch-dag-db-bridge-design.md` (sections D1-D6, Risks, Testing) for the full design.

**Tech Stack:** Python 3.12+ (db_bridge: FastAPI + httpx + supabase-py + asyncpg; areal client: httpx); Go (multica arealrl client - read-only verification only); OpenSpec deltas.

## Global Constraints

- db_bridge runs under `uv run` (`python -m db_bridge.*` fails with ModuleNotFoundError) - all run commands use `uv run`.
- Stub servers bind `127.0.0.1` only; the local app is the sole caller.
- `BRIDGE_USER_ID` must match across areal stub and multica executor or rows go unclaimed.
- Channel `path` with `{projectID}` is supported (proven by `env_dispatch_delete`); GET is supported via `channel.method`.
- Gateway channels forward caller `Authorization` end-to-end; `multica_api` channels strip caller auth and inject `MULTICA_UPSTREAM_API_KEY`.
- No wildcard imports. Conventional Commits. Run `pre-commit` / `ruff` before committing (areal: `uvx ruff check`; db_bridge: `uv run pytest`).

## Repo layout & commit strategy

This change spans **two repos** (multica is a nested git repo at `multica/`, on branch `dev`; areal is the outer repo on `master`):

| Repo      | Tasks  | Commit where                          |
|-----------|--------|---------------------------------------|
| multica   | P1-P8, P10 (db_bridge core + arealrl verify + db_bridge tests) | `multica/` repo, branch `dev` (multica-side MR) |
| areal     | P9, P11 (dag client + protocol doc + areal tests) | outer repo (isolation branch) |

- **multica-repo tasks** are committed in `multica/` (cd into `multica/`, commit there). They ship via the multica-side MR (established pattern - see memory `multica_v2_segment_dag_build`).
- **areal-repo tasks** are committed in the outer repo on the chosen isolation branch.
- Task 16 (v2-segment-dag delta spec) is **already complete** - both MODIFIED requirements were written in the open phase and validated (2 deltas). No plan task for it.

## File Structure

**multica repo** (`multica/db_bridge/`):
- `channels.py` - add `multica` Side + `multica_api` group; move `env_dispatch`/`env_dispatch_delete`; add `env_dispatch_dag` + `rl_close_segment`.
- `config.py` - add `multica_upstream_url` + `BRIDGE_MULTICA_UPSTREAM_URL`; extend `upstream_for_group`.
- `executor.py` - forward `multica_api` to `multica_upstream_url` with upstream-key inject + caller-auth strip.
- `run_executor.py` / `entrypoints.py` - accept `--side multica`.
- `schema.sql` - add `rpc_env_dispatch_dag` + `rpc_rl_close_segment`.
- `tests/test_env_dispatch_channels.py` (extend) + `tests/test_executor.py` (extend) + new `tests/test_close_segment_channel.py`.

**multica repo** (`multica/server/internal/arealrl/client.go`): read-only verification (no change).

**areal repo** (`customized_areal/tree_search/agents/`):
- `multica_dag_client.py` - repoint base to `AREAL_BRIDGE_STUB_URL`; drop `MULTICA_API_KEY`; preserve error mapping + 202 re-poll; 504/5xx -> transient retry.
- `multica_environment_protocol.md` - doc: DAG fetch bridged, side labels, close_segment bridged.
- `tests/test_env_dispatch_client.py` (extend) + dag-client wiring test.

---

## Phase A - db_bridge side/group reconciliation (multica repo)

### Task P1: Add `multica` Side + `multica_api` group to channel model

**Files:** Modify `multica/db_bridge/channels.py`

**Interfaces:**
- Produces: `Side = Literal["leagent", "areal", "multica"]`, `Group = Literal["gateway", "leagent_api", "multica_api"]`; `stub_side`/`executor_side` now return three-valued mapping.

- [ ] **Step 1: Write the failing test** in `multica/db_bridge/tests/test_channels.py` (create if absent):

```python
from db_bridge.channels import CHANNELS_BY_NAME, Group, Side


def test_multica_api_group_routing():
    ch = CHANNELS_BY_NAME["env_dispatch"]  # moved in P2; for now assert literal exists
    assert "multica_api" in Group.__args__  # type: ignore[attr-defined]
    assert "multica" in Side.__args__  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run test to verify it fails** - `cd multica && uv run pytest db_bridge/tests/test_channels.py::test_multica_api_group_routing -v` -> FAIL (literal missing).
- [ ] **Step 3: Implement** - extend the literals and the derivation in `channels.py`:

```python
Group = Literal["gateway", "leagent_api", "multica_api"]
Side = Literal["leagent", "areal", "multica"]
```

```python
@property
def stub_side(self) -> Side:
    return "leagent" if self.group == "gateway" else "areal"

@property
def executor_side(self) -> Side:
    if self.group == "gateway":
        return "areal"
    if self.group == "multica_api":
        return "multica"
    return "leagent"
```

- [ ] **Step 4: Run test to verify it passes** -> PASS.
- [ ] **Step 5: Commit** - `cd multica && git add db_bridge/channels.py db_bridge/tests/test_channels.py && git commit -m "feat(db_bridge): add multica side + multica_api group literals"`

### Task P2: Move env_dispatch channels into `multica_api`; add `env_dispatch_dag` + `rl_close_segment`

**Files:** Modify `multica/db_bridge/channels.py` (the `CHANNELS` tuple)

**Interfaces:**
- Produces: `CHANNELS_BY_NAME["env_dispatch"]`, `["env_dispatch_delete"]` now `group="multica_api"`; new `["env_dispatch_dag"]` (`multica_api`, GET) and `["rl_close_segment"]` (`gateway`, POST).

- [ ] **Step 1: Write the failing test** (extend `test_channels.py`):

```python
def test_env_dispatch_channels_in_multica_api():
    for name in ("env_dispatch", "env_dispatch_delete", "env_dispatch_dag"):
        assert CHANNELS_BY_NAME[name].group == "multica_api", name
        assert CHANNELS_BY_NAME[name].stub_side == "areal"
        assert CHANNELS_BY_NAME[name].executor_side == "multica"


def test_rl_close_segment_in_gateway():
    ch = CHANNELS_BY_NAME["rl_close_segment"]
    assert ch.group == "gateway"
    assert ch.method == "POST"
    assert ch.path == "/rl/close_segment"
    assert ch.stub_side == "leagent"
    assert ch.executor_side == "areal"
    assert ch.table == "rpc_rl_close_segment"


def test_env_dispatch_dag_channel():
    ch = CHANNELS_BY_NAME["env_dispatch_dag"]
    assert ch.method == "GET"
    assert ch.path == "/api/v1/env-dispatch/{projectID}/dag"
    assert ch.table == "rpc_env_dispatch_dag"
    assert ch.default_timeout_s == 30.0
```

- [ ] **Step 2: Run -> FAIL** (channels not present / wrong group).
- [ ] **Step 3: Implement** - in `CHANNELS`, change `env_dispatch` and `env_dispatch_delete` `group="leagent_api"` -> `group="multica_api"`. Add after `env_dispatch_delete`:

```python
    Channel(
        name="env_dispatch_dag",
        group="multica_api",
        method="GET",
        path="/api/v1/env-dispatch/{projectID}/dag",
        kind="json",
        default_timeout_s=30.0,
        default_concurrency=4,
    ),
```

Add in the `gateway` group (after `chat_completions`):

```python
    Channel(
        name="rl_close_segment",
        group="gateway",
        method="POST",
        path="/rl/close_segment",
        kind="json",
        default_timeout_s=30.0,
        default_concurrency=4,
    ),
```

- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** - `git commit -m "feat(db_bridge): move env_dispatch to multica_api; add env_dispatch_dag + rl_close_segment channels"`

### Task P3: Add `multica_upstream_url` config + `upstream_for_group` routing

**Files:** Modify `multica/db_bridge/config.py`

**Interfaces:**
- Produces: `BridgeConfig.multica_upstream_url: str`; `upstream_for_group("multica_api")` -> `multica_upstream_url`. Env `BRIDGE_MULTICA_UPSTREAM_URL`.

- [ ] **Step 1: Failing test** in `tests/test_config.py`:

```python
def test_multica_upstream_url_default_and_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://s")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    monkeypatch.delenv("BRIDGE_MULTICA_UPSTREAM_URL", raising=False)
    cfg = BridgeConfig.from_env()  # adjust to actual loader
    assert cfg.multica_upstream_url  # non-empty default
    monkeypatch.setenv("BRIDGE_MULTICA_UPSTREAM_URL", "http://127.0.0.1:9999")
    cfg2 = BridgeConfig.from_env()
    assert cfg2.multica_upstream_url == "http://127.0.0.1:9999"
    assert cfg2.upstream_for_group("multica_api") == "http://127.0.0.1:9999"
```

- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** - add field + default + env read + routing (mirror `gateway_upstream_url`/`leagent_upstream_url` exactly; default `_DEFAULT_MULTICA_UPSTREAM = "http://127.0.0.1:8081"` placeholder - the real port is confirmed in P10/verify):

```python
_DEFAULT_MULTICA_UPSTREAM: Final = "http://127.0.0.1:8081"
# ...
    multica_upstream_url: str = _DEFAULT_MULTICA_UPSTREAM
# in from_env:
            multica_upstream_url=(
                _get(src, "BRIDGE_MULTICA_UPSTREAM_URL") or _DEFAULT_MULTICA_UPSTREAM
            ),
# upstream_for_group:
    def upstream_for_group(self, group: Group) -> str:
        if group == "gateway":
            return self.gateway_upstream_url
        if group == "multica_api":
            return self.multica_upstream_url
        return self.leagent_upstream_url
```

- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit** - `git commit -m "feat(db_bridge): add BRIDGE_MULTICA_UPSTREAM_URL + multica_api routing"`

### Task P4: `run_executor` / `entrypoints` accept `--side multica`

**Files:** Modify `multica/db_bridge/run_executor.py`, `multica/db_bridge/entrypoints.py`

- [ ] **Step 1: Failing test** in `tests/test_entrypoints.py` - assert `--side multica` is accepted and selects `executor_channels("multica")` (the `multica_api` channels).
- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** - extend the `--side` choices to include `"multica"` wherever `leagent`/`areal` are enumerated; `executor_channels("multica")` already works once P2 lands (it filters by `executor_side == side`). Verify the executor process boots for `multica_api` channels.
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit** - `git commit -m "feat(db_bridge): run_executor accepts --side multica"`

---

## Phase B - db_bridge stub / executor / schema (multica repo)

### Task P5: schema.sql - add `rpc_env_dispatch_dag` + `rpc_rl_close_segment`

**Files:** Modify `multica/db_bridge/schema.sql`

- [ ] **Step 1: Failing test** - `tests/test_db.py`: assert both tables exist after applying `schema.sql` to a test DB (mirror the existing table-creation test pattern).
- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** - add two table definitions mirroring `rpc_env_dispatch` / `rpc_rl_set_reward` (id, user_id, status, request cols, response cols, timestamps, `FOR UPDATE SKIP LOCKED` claim support). Idempotent (`CREATE TABLE IF NOT EXISTS`).
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit** - `git commit -m "feat(db_bridge): schema for rpc_env_dispatch_dag + rpc_rl_close_segment"`

### Task P6: stub_server serves the new channels (verify route registration + pass-through)

**Files:** `multica/db_bridge/stub_server.py` (likely no code change - routes auto-register via `add_api_route(channel.path, ...)`)

- [ ] **Step 1: Failing test** in new `tests/test_close_segment_channel.py` + extend `tests/test_env_dispatch_channels.py`:
  - `rl_close_segment`: POST to stub -> row enqueued in `rpc_rl_close_segment`; stub does NOT return 404 for `/rl/close_segment`; response (200 `CloseSegmentResponse`, 400 no-active) passed through.
  - `env_dispatch_dag`: GET to stub with `{projectID}` -> row in `rpc_env_dispatch_dag`; `202`/`200`/`404` pass-through.
- [ ] **Step 2: Run -> FAIL** (404 for close_segment until P2's channel is registered; the test must run against a stub built with the new channels).
- [ ] **Step 3: Implement** - confirm `stub_channels("areal")` includes `env_dispatch_dag` (multica_api, stub_side=areal) and `stub_channels("leagent")` includes `rl_close_segment` (gateway, stub_side=leagent). If the stub's route registration already iterates `stub_channels(side)`, no code change is needed - the test passing is the deliverable. If path-param GET needs a handler tweak, apply it here.
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit** - `git commit -m "test(db_bridge): rl_close_segment + env_dispatch_dag stub pass-through"`

### Task P7: executor forwards `multica_api` to `multica_upstream_url` (key inject + auth strip)

**Files:** Modify `multica/db_bridge/executor.py`

**Interfaces:**
- Consumes: `BridgeConfig.multica_upstream_url`, `upstream_for_group`, `upstream_api_key`/`MULTICA_UPSTREAM_API_KEY`.
- Produces: `multica_api` channel requests forwarded to multica with caller `Authorization` stripped and `MULTICA_UPSTREAM_API_KEY` injected.

- [ ] **Step 1: Failing test** in `tests/test_executor.py`: a `multica_api` row is forwarded to `multica_upstream_url` (not `leagent_upstream_url`/`gateway_upstream_url`), the upstream receives `Authorization: Bearer <MULTICA_UPSTREAM_API_KEY>`, and any caller-supplied `Authorization`/`X-API-Key` is NOT forwarded.
- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** - in the executor's forwarding path, branch on `channel.group` (or use `config.upstream_for_group(channel.group)`); for `multica_api`, set upstream URL to `multica_upstream_url`, strip `_BRIDGE_CRED_HEADERS` from the forwarded request, and inject `upstream_api_key`. Reuse the existing `upstream_api_key` plumbing from `multica_server.py` / `config.py`.
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit** - `git commit -m "feat(db_bridge): executor forwards multica_api to multica upstream with key inject"`

### Task P8: update `.env.*` examples

**Files:** `multica/db_bridge/.env.areal`, `.env.leagent`, `.env.multica.example`

- [ ] **Step 1:** Add `BRIDGE_MULTICA_UPSTREAM_URL` (with a comment pointing at the multica Go server `$PORT`), document the two new channels, and add `--side multica` run notes for the multica host (mirror the leagent/areal two-process blocks).
- [ ] **Step 2:** No test (config docs); verify the examples source cleanly: `set -a && source .env.multica.example && set +a && echo $BRIDGE_MULTICA_UPSTREAM_URL`.
- [ ] **Step 3: Commit** - `git commit -m "docs(db_bridge): env examples for multica side + new channels"`

---

## Phase C - multica arealrl verification (multica repo, read-only)

### Task P9: Verify arealrl `CloseSegment` succeeds once `rl_close_segment` is registered

**Files:** `multica/server/internal/arealrl/client.go` (NO change - verification only)

- [ ] **Step 1:** Re-read `arealrl/client.go` `CloseSegment` + `New(stubBaseURL, ...)`. Confirm it posts to `closeSegmentPath` against `stubBaseURL` with `Bearer <proxyKey>`. No code change expected.
- [ ] **Step 2:** Add/extend an arealrl test (Go) that posts `close_segment` against a stub fake and asserts a non-404 response + `trajectory_id` parse - only if an existing test harness exists; otherwise document the manual verification (the stub no longer 404s after P2/P6).
- [ ] **Step 3: Commit** (test only, if added) - `git commit -m "test(arealrl): close_segment routes through db_bridge stub (no 404)"`

---

## Phase D - areal client + docs (areal repo)

### Task P10: Repoint `MulticaDagClient` to `AREAL_BRIDGE_STUB_URL`

**Files:** Modify `customized_areal/tree_search/agents/multica_dag_client.py`

**Interfaces:**
- Produces: `MulticaDagClient(base_url=None)` reads `AREAL_BRIDGE_STUB_URL` (was `MULTICA_BASE_URL`); no `MULTICA_API_KEY` sent; `202` -> re-poll, `200` -> AssembledDag, `404` -> `DagNotFound`, `403` -> `DagForbidden`, `504`/5xx (bridge timeout) -> transient retry; `DagTimeout` on overall wall-clock.

- [ ] **Step 1: Failing test** in `customized_areal/tree_search/tests/test_env_dispatch_client.py` (or a new `test_multica_dag_client.py`): with `AREAL_BRIDGE_STUB_URL` set and a fake transport that returns 202 then 200, assert the client polls the stub URL (not `MULTICA_BASE_URL`), sends no `Authorization`, and returns the parsed DAG; assert 404 -> `DagNotFound`; assert 504 -> re-poll (not raise).
- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** - in `multica_dag_client.py`:
  - `__init__`: `self._base = (base_url or os.environ.get("AREAL_BRIDGE_STUB_URL") or "").rstrip("/")`; raise `ValueError` if empty (rename message).
  - Drop `MULTICA_API_KEY`/`_api_key` from the DAG path (the stub is loopback; no bearer). Remove `_headers` `Authorization` for the DAG call (keep `Accept`/`Content-Type` if a GET body is ever sent - GET has no body).
  - In the poll loop: treat `504`/`502`/`503` (bridge timeout / transient) as `202`-equivalent (sleep + re-poll), up to the client's existing `DagTimeout` wall-clock.
  - Keep `DagNotFound`(404)/`DagForbidden`(403) mapping intact.
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit** (areal repo) - `git commit -m "feat(dag-client): route env-dispatch DAG fetch through db_bridge stub"`

### Task P11: Update `multica_environment_protocol.md`

**Files:** Modify `customized_areal/tree_search/agents/multica_environment_protocol.md`

- [ ] **Step 1:** Edit the endpoint table + prose: DAG fetch is bridged via `AREAL_BRIDGE_STUB_URL` (not `MULTICA_BASE_URL`); correct the side labels (env-dispatch -> `multica`/`multica_api`, not le-agent); note `close_segment` is bridged via the `gateway`-group `rl_close_segment` channel (multica `arealrl` client already targets the stub).
- [ ] **Step 2:** No test (docs). Sanity-check rendered table alignment.
- [ ] **Step 3: Commit** (areal repo) - `git commit -m "docs(protocol): DAG fetch + close_segment bridged; correct multica side labels"`

---

## Phase E - tests + verify (both repos)

### Task P12: db_bridge test suite green + rename fallout

**Files:** `multica/db_bridge/tests/`

- [ ] **Step 1:** Run `cd multica && uv run pytest db_bridge/tests/ -q`. Fix any test that asserted `env_dispatch` in `leagent_api` or the two-side `leagent|areal` model (e.g. `test_env_dispatch_channels.py`, `test_gateway_channels.py`, `test_leagent_channels.py`).
- [ ] **Step 2:** Add the channel tests from P2/P6/P7 if not already passing.
- [ ] **Step 3: Commit** - `git commit -m "test(db_bridge): green suite after multica side + new channels"`

### Task P13: areal test suite green

**Files:** `customized_areal/tree_search/tests/`

- [ ] **Step 1:** Run the dag-client + env-dispatch tests (`.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_env_dispatch_client.py` per memory `areal_test_invocation`). Fix fallout from the `MULTICA_BASE_URL` -> `AREAL_BRIDGE_STUB_URL` rename (e.g. `test_multica_workflow_wiring.py:44` sets `MULTICA_BASE_URL=http://multica.test` - update or add `AREAL_BRIDGE_STUB_URL`).
- [ ] **Step 2: Commit** (areal repo) - `git commit -m "test(areal): dag-client bridge routing green"`

### Task P14: Verify + deployment note

- [ ] **Step 1:** Confirm the multica Go server's `$PORT` default (grep `multica/server/cmd/server/main.go` + deploy config) and set the real `BRIDGE_MULTICA_UPSTREAM_URL` default in `.env.multica.example` (replace the P3 placeholder `:8081`).
- [ ] **Step 2:** Run both test suites green: `cd multica && uv run pytest db_bridge/tests/ -q` and areal dag/env-dispatch tests.
- [ ] **Step 3:** Write the deployment note (idempotent `schema.sql` apply for both new tables; multica host runs `--side multica` executor; areal stub gains both channels; `rl_close_segment` served by existing gateway executor) into the design doc's "Migration & deployment ordering" section if not already captured (it is - this is a confirmation step).
- [ ] **Step 4: Commit** any final config tweak - `git commit -m "chore(db_bridge): confirm multica upstream port + deployment note"`

---

## Self-Review

**Spec coverage:**
- v2-segment-dag "Polling AssembledDag" transport -> P2 (channel) + P6 (stub) + P10 (client repoint). ✓
- v2-segment-dag "V2 no-reward segment close" transport -> P2 (channel) + P6 (stub no-404) + P9 (arealrl verify). ✓
- Side/group reconciliation -> P1, P2, P3, P4, P7, P8. ✓
- Task 16 (delta spec) -> already complete (open phase). ✓ (no plan task)

**Placeholder scan:** P3's `:8081` default is an explicit placeholder resolved in P14 (flagged inline, not hidden). P6 may need no code change (flagged). P9 may add no test (flagged). No TBD/TODO hidden.

**Type consistency:** `multica_upstream_url` / `BRIDGE_MULTICA_UPSTREAM_URL` / `multica_api` used consistently across P1-P8. `env_dispatch_dag` / `rl_close_segment` channel names consistent across P2, P5, P6.

**Multi-repo note:** P1-P9, P12 commit in `multica/`; P10, P11, P13 commit in the outer areal repo. The plan-ready pause (comet-build Step 2) is the point to decide isolation given this two-repo span.

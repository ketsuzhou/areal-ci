# env-dispatch over db_bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the AReaL runner's two env-dispatch calls (`POST /api/v1/env-dispatch`,
`DELETE /api/v1/env-dispatch/{projectID}`) across the split-host boundary through
db_bridge, with zero AReaL client code change.

**Architecture:** env-dispatch is AReaL→multica traffic, so it joins the existing
`leagent_api` bridge group: a loopback stub on the AReaL host captures each call into a
per-endpoint Supabase table (`rpc_env_dispatch`, `rpc_env_dispatch_delete`), and the
multica-side executor forwards it to the real multica API and relays the response
verbatim. The change is two new `Channel` entries + two new tables; the
`MulticaEnvDispatchClient` is untouched and reaches the bridge purely by pointing
`MULTICA_BASE_URL` at the local stub.

**Tech Stack:** Python 3.12, FastAPI (stub app), httpx (relay + tests),
Supabase/Postgres (channel tables), pytest + `httpx.MockTransport`/`ASGITransport`,
ruff.

## Global Constraints

- Bridge exactly two endpoints: `POST /api/v1/env-dispatch` and
  `DELETE /api/v1/env-dispatch/{projectID}`. Do **not** add `env_create`/`env_delete` or
  the legacy issue-fork endpoint.
- Both channels are in the `leagent_api` group: `stub_side = areal`,
  `executor_side = multica` (derived from the group; do not set explicitly).
- Channel values, verbatim: `env_dispatch` → `method=POST`, `path=/api/v1/env-dispatch`,
  `kind=json`, `default_timeout_s=600.0`, `default_concurrency=8`. `env_dispatch_delete`
  → `method=DELETE`, `path=/api/v1/env-dispatch/{projectID}`, `kind=json`,
  `default_timeout_s=60.0`, `default_concurrency=4`.
- Timeout invariant: executor forward-timeout ≥ channel `default_timeout_s` ≥ client
  httpx timeout.
- No change to `MulticaEnvDispatchClient`
  (`customized_areal/tree_search/agents/swe_lego_client.py`). Bridging is a
  deployment/config concern.
- Preserve the env-dispatch HTTP contract end to end: status codes, JSON bodies
  (`rollouts[]`), and the `Authorization` header all relay unchanged.
- No wildcard imports. Run `ruff` on changed Python before each commit.
- All db_bridge commands run with `uv run` from the `multica/db_bridge/` directory.

______________________________________________________________________

### Task 1: Add the two env-dispatch channels

**Files:**

- Modify: `multica/db_bridge/channels.py` (append two `Channel` entries to the
  `CHANNELS` tuple, inside/after the `leagent_api` group)
- Test: `multica/db_bridge/tests/test_env_dispatch_channels.py` (create)

**Interfaces:**

- Consumes: `db_bridge.channels.Channel` (fields
  `name, group, method, path, kind, default_timeout_s, default_concurrency`);
  `CHANNELS_BY_NAME`; the stub/executor harness pattern from
  `tests/test_leagent_channels.py`.

- Produces: `CHANNELS_BY_NAME["env_dispatch"]` and
  `CHANNELS_BY_NAME["env_dispatch_delete"]` with `.table == "rpc_env_dispatch"` /
  `"rpc_env_dispatch_delete"`, `.stub_side == "areal"`, `.executor_side == "leagent"`.

- [ ] **Step 1: Write the failing tests**

Create `multica/db_bridge/tests/test_env_dispatch_channels.py`:

```python
"""env-dispatch leagent_api channel tests: POST /api/v1/env-dispatch and
DELETE /api/v1/env-dispatch/{projectID}.

Stub runs on the AReaL side; executor on the multica side (forwarding to the
real multica API). Verifies the env-dispatch contract relays end to end:
status code, rollouts[] JSON body, path params, and the Authorization header.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import httpx

from db_bridge.channels import CHANNELS_BY_NAME
from db_bridge.config import BridgeConfig
from db_bridge.db import BridgeDB
from db_bridge.executor import Executor
from db_bridge.stub_server import create_stub_app

from _fakes import FakeSupabaseClient

USER_ID = "00000000-0000-0000-0000-00000000000a"


def _config(**overrides: str) -> BridgeConfig:
    env = {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "k",
        "BRIDGE_POLL_INTERVAL": "0.01",
        "BRIDGE_USER_ID": USER_ID,
        **overrides,
    }
    return BridgeConfig.from_env(env)


@contextlib.asynccontextmanager
async def areal_harness(handler, **cfg_overrides):
    cfg = _config(**cfg_overrides)
    db = BridgeDB(cfg, client=FakeSupabaseClient())
    ex = Executor(
        db,
        "leagent",
        config=cfg,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    run_task = asyncio.create_task(ex.run())
    stub = create_stub_app(db, "areal", cfg)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=stub), base_url="http://stub"
        ) as client:
            yield client, db
    finally:
        ex.stop()
        await run_task


def test_channels_registered_in_leagent_api_group():
    disp = CHANNELS_BY_NAME["env_dispatch"]
    dele = CHANNELS_BY_NAME["env_dispatch_delete"]
    assert disp.group == "leagent_api"
    assert disp.method == "POST"
    assert disp.path == "/api/v1/env-dispatch"
    assert disp.table == "rpc_env_dispatch"
    assert disp.stub_side == "areal"
    assert disp.executor_side == "leagent"
    assert disp.default_timeout_s == 600.0
    assert dele.method == "DELETE"
    assert dele.path == "/api/v1/env-dispatch/{projectID}"
    assert dele.table == "rpc_env_dispatch_delete"


def test_env_dispatch_post_relays_rollouts_and_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "rollouts": [
                    {"env_id": "e1", "project_id": "p1", "issue_id": "i1",
                     "agent_run_id": "r1"},
                ]
            },
        )

    async def run():
        async with areal_harness(handler) as (client, _db):
            return await client.post(
                "/api/v1/env-dispatch",
                headers={"Authorization": "Bearer multica-key"},
                json={
                    "mode": "scratch", "env_id": "base", "domain": "swe_lego",
                    "dispatch_type": "issue", "group_size": 1, "agent_id": "ag",
                    "issue": {"title": "t"},
                },
            )

    resp = asyncio.run(run())
    assert resp.status_code == 201
    assert resp.json()["rollouts"][0]["agent_run_id"] == "r1"
    assert seen["auth"] == "Bearer multica-key"
    assert seen["path"] == "/api/v1/env-dispatch"
    assert seen["body"]["mode"] == "scratch"


def test_env_dispatch_delete_relays_path_param():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        return httpx.Response(204)

    async def run():
        async with areal_harness(handler) as (client, _db):
            return await client.delete(
                "/api/v1/env-dispatch/proj-123",
                headers={"Authorization": "Bearer multica-key"},
            )

    resp = asyncio.run(run())
    assert resp.status_code == 204
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/v1/env-dispatch/proj-123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/db_bridge && uv run pytest tests/test_env_dispatch_channels.py -v`
Expected: FAIL — `KeyError: 'env_dispatch'` in `CHANNELS_BY_NAME` (channels not yet
defined).

- [ ] **Step 3: Add the two channels**

In `multica/db_bridge/channels.py`, append these two entries to the `CHANNELS` tuple,
immediately after the `agent_start_branch` `Channel(...)` (keep them in the
`leagent_api` group section):

```python
    Channel(
        name="env_dispatch",
        group="leagent_api",
        method="POST",
        path="/api/v1/env-dispatch",
        kind="json",
        default_timeout_s=600.0,
        default_concurrency=8,
    ),
    Channel(
        name="env_dispatch_delete",
        group="leagent_api",
        method="DELETE",
        path="/api/v1/env-dispatch/{projectID}",
        kind="json",
        default_timeout_s=60.0,
        default_concurrency=4,
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/db_bridge && uv run pytest tests/test_env_dispatch_channels.py -v`
Expected: PASS (3 tests). If `test_env_dispatch_delete_relays_path_param` fails to
route, confirm the stub registers `channel.path` with FastAPI templating (`{projectID}`)
— no code change needed, the path string is the route.

- [ ] **Step 5: Lint and commit**

```bash
cd multica/db_bridge
uv run ruff check channels.py tests/test_env_dispatch_channels.py
git add channels.py tests/test_env_dispatch_channels.py
git commit -m "feat(db_bridge): add env-dispatch channels (env_dispatch, env_dispatch_delete)"
```

______________________________________________________________________

### Task 2: Create the two channel tables in schema.sql

**Files:**

- Modify: `multica/db_bridge/schema.sql` (add two names to the `tables text[]` array)
- Test: `multica/db_bridge/tests/test_env_dispatch_schema.py` (create)

**Interfaces:**

- Consumes: the existing idempotent create-tables
  `do $bridge$ … tables text[] := array[ … ] … foreach` block in `schema.sql`.

- Produces: `create table if not exists public.rpc_env_dispatch` and
  `public.rpc_env_dispatch_delete` in `schema.sql` (with the generic per-channel columns
  and RLS applied by the same loop).

- [ ] **Step 1: Write the failing test**

Create `multica/db_bridge/tests/test_env_dispatch_schema.py`:

```python
"""Static structural check: env-dispatch channel tables exist in schema.sql."""

from __future__ import annotations

from pathlib import Path

_SCHEMA = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()


def test_env_dispatch_tables_in_array():
    assert "'rpc_env_dispatch'" in _SCHEMA
    assert "'rpc_env_dispatch_delete'" in _SCHEMA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/db_bridge && uv run pytest tests/test_env_dispatch_schema.py -v`
Expected: FAIL — the two table names are not yet in `schema.sql`.

- [ ] **Step 3: Add the two table names to the array**

In `multica/db_bridge/schema.sql`, extend the `tables text[] := array[ … ]` literal
(currently ending with `'rpc_agent_start_branch'`) so it reads:

```sql
    tables text[] := array[
        'rpc_rl_start_session',
        'rpc_rl_set_reward',
        'rpc_rl_end_session',
        'rpc_chat_completions',
        'rpc_agent_start',
        'rpc_agent_start_branch',
        'rpc_env_dispatch',
        'rpc_env_dispatch_delete'
    ];
```

Do not add any other DDL — the existing `foreach tbl in array tables loop` block creates
the table, `user_id` column/constraint, and RLS for each new name automatically.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/db_bridge && uv run pytest tests/test_env_dispatch_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica/db_bridge
git add schema.sql tests/test_env_dispatch_schema.py
git commit -m "feat(db_bridge): create env-dispatch channel tables in schema.sql"
```

______________________________________________________________________

### Task 3: Document the bridged deployment config

**Files:**

- Modify: `multica/db_bridge/.env.areal.example` (document pointing env-dispatch at the
  local stub)
- Modify: `multica/db_bridge/README.md` (add the two channels to the channel table)

**Interfaces:**

- Consumes: nothing at runtime — this task records the deployment invariant that makes
  the (unchanged) `MulticaEnvDispatchClient` use the bridge.

- Produces: operator-facing documentation only. No code, no test.

- [ ] **Step 1: Add the two channels to the README channel table**

In `multica/db_bridge/README.md`, in the "How it works" channel table (the one listing
`agent_start` / `agent_start_branch`), add two rows:

```markdown
| `env_dispatch`        | `/api/v1/env-dispatch`             | `leagent_api` | AReaL     | le-agent      |
| `env_dispatch_delete` | `/api/v1/env-dispatch/{projectID}` | `leagent_api` | AReaL     | le-agent      |
```

- [ ] **Step 2: Document the AReaL-side pointing + timeout in `.env.areal.example`**

In `multica/db_bridge/.env.areal.example`, add a comment block explaining the
env-dispatch bridging invariant:

```bash
# env-dispatch (AReaL -> multica) is bridged via the leagent_api channels
# `env_dispatch` / `env_dispatch_delete`. Point the AReaL env-dispatch client at
# the local AReaL stub so its calls traverse the bridge:
#
#   MULTICA_BASE_URL=http://127.0.0.1:9101   # the local AReaL stub
#   MULTICA_API_KEY=<multica key>            # relayed as Authorization: Bearer
#
# Keep the client's httpx timeout <= the env_dispatch channel timeout (600s) so
# a slow group_size=N dispatch surfaces multica's real response, not a 504.
```

- [ ] **Step 3: Verify the docs render and commit**

Run: `cd multica/db_bridge && grep -n "env_dispatch" README.md .env.areal.example`
Expected: the new rows/comments are present.

```bash
git add README.md .env.areal.example
git commit -m "docs(db_bridge): document bridged env-dispatch deployment config"
```

______________________________________________________________________

### Task 4: Full-suite regression check

**Files:**

- None (verification only)

**Interfaces:**

- Consumes: all db_bridge tests.

- Produces: confirmation that the two new channels/tables did not break existing
  channels.

- [ ] **Step 1: Run the whole db_bridge test suite**

Run: `cd multica/db_bridge && uv run pytest -q` Expected: PASS, including the
pre-existing `tests/test_leagent_channels.py`, `tests/test_gateway_channels.py`,
`tests/test_integration_e2e.py`, and the two new test modules. No new failures.

- [ ] **Step 2: Confirm the AReaL client is unchanged**

Run:
`cd /workspaces/leagent/backend/areal && git status --short customized_areal/tree_search/agents/swe_lego_client.py`
Expected: empty output (no modification) — validates the config-only /
zero-client-change goal.

______________________________________________________________________

## Self-Review

**Spec coverage:**

- §5.1 channels → Task 1. §5.2 schema tables → Task 2. §5.3 deployment config / zero
  client change → Task 3 (+ Task 4 Step 2 asserts no client change). §5.4 timeout
  invariant → Global Constraints + Task 1 channel values + Task 3 `.env` note. §6 data
  flow → exercised by Task 1 relay tests. §7 error handling → contract preserved (relay
  verbatim), covered by the 201/204 relay assertions; bridge-level 504/502/413 are
  existing bridge behavior, not re-implemented. §8 tests → Task 1 (channel round-trip
  incl. path-param DELETE + Authorization), Task 2 (schema), and Task 4
  (client-unchanged assertion, standing in for the "zero-change round-trip" since the
  client wrapper itself is already covered by `tests/test_env_dispatch_client.py`). §9
  out-of-scope respected (no env_create/env_delete/legacy fork).

**Placeholder scan:** No TBD/TODO; every code/DDL step shows the literal content.

**Type consistency:** `env_dispatch` / `env_dispatch_delete` names, table names
(`rpc_env_dispatch`, `rpc_env_dispatch_delete`), paths, and the 600.0/60.0 & 8/4 values
are identical across the spec, Global Constraints, Task 1 code, and Task 2 DDL.
`stub_side=areal` / `executor_side=leagent` are asserted in Task 1 Step 1 and derive
from `group="leagent_api"`.

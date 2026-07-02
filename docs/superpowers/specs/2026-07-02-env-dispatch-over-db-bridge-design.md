# env-dispatch over db_bridge — design

**Date:** 2026-07-02
**Status:** Draft
**Scope:** `db_bridge/` (channel registry + schema), AReaL deployment config. No AReaL client code change.

## 1. Motivation

AReaL's tree-search runners call multica's unified env-dispatch API
(`POST /api/v1/env-dispatch`, `DELETE /api/v1/env-dispatch/{projectID}`) over
**direct HTTP** (`MulticaEnvDispatchClient`, httpx against `MULTICA_BASE_URL`).
In the target deployment, the AReaL host and the multica host cannot reach each
other directly; they share one Supabase database. Every cross-service call must
therefore travel through `db_bridge` (the Supabase-as-springboard RPC bridge),
exactly as `agent_start` / `agent_start_branch` and the `rl_*` gateway calls
already do. env-dispatch is the one remaining AReaL→multica call still on direct
HTTP.

This sub-project moves the env-dispatch calls the runner actually makes onto
db_bridge. It is the transport foundation for the later sub-projects (default
env, squad routing, training-agent session lifecycle, entropy/critic save
callback), each of which will add its own channel(s) on the same mechanism.

## 2. Goals

- Route the two env-dispatch calls the AReaL runner makes across the split-host
  boundary through db_bridge channels (`leagent_api` group).
- Zero change to `MulticaEnvDispatchClient`: bridging is a deployment concern
  (point `MULTICA_BASE_URL` at the local stub).
- Preserve the env-dispatch HTTP contract end to end — status codes, JSON
  bodies (`rollouts[]`), and `Authorization` all relay unchanged.
- Tolerate slow `POST /api/v1/env-dispatch` calls (forking N sandboxes for
  `group_size=N`) without spurious bridge timeouts.

## 3. Non-goals

- Bridging `POST /api/v1/env` (`create_base_env`) or `DELETE /api/v1/env/{envID}`
  (`delete_env`). The runners do not call these; base-env provisioning is
  out-of-band / multica-side (the docker image is built out-of-band, and base
  env creation is a provisioning step, not part of the training hot path). If a
  future need arises to provision base envs *from the AReaL host*, adding those
  channels is a trivial follow-up on the same mechanism.
- Bridging the legacy `POST /api/issues/{id}/fork` endpoint (removed with the
  BranchMaterializer sub-project).
- Any functional/feature change to env-dispatch itself: `env_id=None` default
  env, `squad_id`, `training_agent`, and the entropy/critic save callback are
  separate sub-projects (B–E).
- Changing db_bridge's routing core, stub/executor process model, or schema
  shape. This sub-project only adds channels + tables using the existing
  patterns.

## 4. Background — how db_bridge carries an endpoint

A **channel** is one bridged HTTP endpoint backed by one Supabase table
(`rpc_<name>`). For each channel:

- The **stub server** (loopback, on the *caller's* host) registers the channel
  path as a FastAPI route, captures the request (method, concrete path+query,
  filtered headers, raw body), writes a `pending` row, polls it for the
  response, and returns the response verbatim.
- The **executor worker** (on the *callee's* host) claims the pending row
  (`FOR UPDATE SKIP LOCKED`), forwards it to the real local service over
  loopback, and writes the response row back.

env-dispatch is AReaL→multica, i.e. the `leagent_api` group: **stub on the
AReaL host** (already bound to `127.0.0.1:9101`, serving `/api/agent/*`),
**executor on the multica host** forwarding to the real multica API. `run_stub`
/ `run_executor` auto-serve every channel registered for their side, so no new
process wiring is needed — only new channel entries and their tables.

Verified mechanics that make this a transparent drop-in:

- **Path params work.** The stub registers `channel.path` via
  `app.add_api_route(channel.path, ..., methods=[channel.method])`, so a
  templated path like `/api/v1/env-dispatch/{projectID}` matches, and
  `_full_path(request)` stores the *concrete* path (real UUID + query) that the
  executor forwards.
- **Status/body relay verbatim.** The stub returns
  `Response(content=result.body, status_code=result.response_status or 200, …)`,
  so env-dispatch's `201` / `409` / `500` / `503` and JSON bodies pass through
  unchanged. Bridge-level failures use the bridge's own codes: `504` (timeout),
  `502` (executor/relay error), `413` (oversized body).
- **Auth passes through.** `relay.filter_request_headers` preserves
  `Authorization` (and `Content-Type`); the executor replays it to multica.

## 5. Design

### 5.1 Channels (`db_bridge/channels.py`)

Add two `Channel` entries to the `leagent_api` group (stub side = AReaL,
executor side = multica):

| name                  | method | path                                  | kind | default_timeout_s | default_concurrency |
|-----------------------|--------|---------------------------------------|------|-------------------|---------------------|
| `env_dispatch`        | POST   | `/api/v1/env-dispatch`                | json | `600.0`           | `8`                 |
| `env_dispatch_delete` | DELETE | `/api/v1/env-dispatch/{projectID}`    | json | `60.0`            | `4`                 |

- The two paths are distinct keys (one carries a `{projectID}` suffix), so
  `CHANNELS_BY_PATH` has no collision and the stub registers two independent
  FastAPI routes.
- `env_dispatch` gets a large `default_timeout_s` (600s) because a
  `group_size=N` dispatch forks N sandboxes on the multica side; the delete is
  fast (60s).
- `default_concurrency` sets the executor worker-coroutine count for that
  channel's table.

### 5.2 Schema (`db_bridge/schema.sql`)

Add the two table names to the `tables text[]` array in the idempotent
create-tables block:

```
'rpc_env_dispatch',
'rpc_env_dispatch_delete'
```

The generic per-channel table shape (with `request_path`, `request_headers`,
`response_status`, `response_body`, …) is unchanged and already stores the
concrete path and arbitrary status/body. Re-applying `schema.sql` is idempotent.

### 5.3 AReaL client + deployment config

`MulticaEnvDispatchClient` (`customized_areal/tree_search/agents/swe_lego_client.py`)
is **not modified**. In the bridged deployment:

- Set `MULTICA_BASE_URL=http://127.0.0.1:9101` (the local AReaL stub).
- Keep `MULTICA_API_KEY`; it is sent as `Authorization: Bearer …` and relayed
  by the executor to multica unchanged.
- Direct mode remains available for local dev/tests purely by pointing
  `MULTICA_BASE_URL` at a real multica instance — no code branch, no flag.

The only client-adjacent change is a **config** one: the client's default httpx
timeout (currently 120s) is aligned to the env-dispatch channel timeout so a
legitimately slow dispatch is not cut off before the bridge returns. This is
done via the client's existing `timeout` constructor argument at the
call/launch site, not by changing default behavior in a way that affects the
direct path unexpectedly.

### 5.4 Timeout invariant

To ensure a slow-but-successful env-dispatch surfaces multica's real response
(not a bridge `504`):

```
executor forward-timeout  ≥  channel default_timeout_s  ≥  client httpx timeout
```

For `env_dispatch`: executor forward-timeout ≥ 600s, channel `default_timeout_s`
= 600s, client timeout ≤ 600s. The executor forward-timeout comes from
`config.timeout_for(channel)`, which is driven by the channel's
`default_timeout_s`, so this holds by construction once the channel value is
set; the client timeout is the only value to keep at or below it.

## 6. Data flow (POST /api/v1/env-dispatch)

1. Runner calls `MulticaEnvDispatchClient.create_env_dispatch(...)` → httpx
   `POST http://127.0.0.1:9101/api/v1/env-dispatch` (local AReaL stub).
2. Stub writes a `pending` row to `rpc_env_dispatch` (method, path, filtered
   headers incl. `Authorization`, raw JSON body), then polls.
3. Multica-side executor claims the row, `POST`s the real multica
   `/api/v1/env-dispatch` over loopback, writes the response row (status +
   `rollouts[]` body + headers).
4. Stub relays the response verbatim; the client parses `rollouts[]` exactly as
   today.

`DELETE /api/v1/env-dispatch/{projectID}` follows the same path on
`rpc_env_dispatch_delete`; the concrete `projectID` is in the stored
`request_path`.

## 7. Error handling

Unchanged env-dispatch contract; multica's own `4xx`/`5xx` + JSON bodies pass
through. Additional bridge-level failures the client may now observe:

| Situation                                   | Status | Source |
|---------------------------------------------|--------|--------|
| Dispatch exceeded the channel timeout        | 504    | stub (bridge) |
| Executor could not reach multica / relay error | 502  | stub (bridge) |
| Request body over the bridge size limit      | 413    | stub (bridge) |

The runner's existing error handling already raises on non-`201` from
`create_env_dispatch` and tolerates `200/204/404` on cleanup, so these
bridge-level codes are surfaced as errors without new client logic.

## 8. Testing strategy

- **Channel unit tests** (`db_bridge/tests/`, mirroring
  `test_leagent_channels.py`): drive the stub app over `httpx.ASGITransport`
  with a fake executor/backend, for both channels:
  - `env_dispatch` — a `POST` enqueues on `rpc_env_dispatch`; a stubbed multica
    `201` with a `rollouts[]` body relays back verbatim; `Authorization` is
    preserved end to end.
  - `env_dispatch_delete` — a `DELETE /api/v1/env-dispatch/<uuid>` enqueues on
    `rpc_env_dispatch_delete` with the concrete `projectID` in `request_path`;
    a `204` relays back.
- **Schema test** (mirroring `test_remote_shell_schema.py`): assert
  `schema.sql` creates `rpc_env_dispatch` and `rpc_env_dispatch_delete`.
- **AReaL client round-trip test**: point `MulticaEnvDispatchClient` at the
  in-process stub app and confirm `create_env_dispatch` / `cleanup_env_dispatch`
  work with **zero client changes** (validates the config-only approach).

## 9. Out of scope / follow-ups

- `env_create` / `env_delete` channels — add only if base-env provisioning ever
  runs from the AReaL host.
- Sub-projects B–E (default env for `env_id=None`, `squad_id` routing,
  `training_agent` session lifecycle with `proxy_url`/`proxy_key`, entropy +
  critic-agent env-save callback) each add their own channel(s) on this same
  mechanism.
- Removing BranchMaterializer and reimplementing branch via env-dispatch
  (sub-project C).

---
comet_change: env-dispatch-dag-db-bridge
role: technical-design
canonical_spec: openspec
---

# Design - env-dispatch-dag-db-bridge

> OpenSpec artifacts (`openspec/changes/env-dispatch-dag-db-bridge/`: proposal.md,
> design.md, tasks.md, `specs/v2-segment-dag/spec.md`) are the canonical source of truth
> for requirements. This document is the deep technical design: implementation approach,
> risks, testing, boundary conditions. It does not redefine requirements.

## Context

`db_bridge` (in `multica/db_bridge/`) relays cross-service HTTP through per-endpoint
Supabase tables: a loopback **stub** on the caller's host enqueues a row; an **executor** on
the callee's host claims it (`FOR UPDATE SKIP LOCKED`), forwards to the real service, writes
the response back. Three parties are involved - AReaL (Python), multica (Go server +
`multica_server.py` public surface), le-agent (FastAPI) - and the bridge is the only path
between network-isolated hosts sharing one Supabase.

This change closes two bridge gaps on the v2-segment-dag surface and corrects a mislabel:

1. `MulticaDagClient` fetches `GET /api/v1/env-dispatch/{projectID}/dag` over direct `httpx`
   to `MULTICA_BASE_URL`, bypassing the bridge.
2. multica's `arealrl` Go client calls `POST /rl/close_segment` on the db_bridge stub, but
   the stub has no `rl_close_segment` route registered -> it 404s.
3. The `env_dispatch`/`env_dispatch_delete` channels are labeled `leagent_api` / `leagent`
   side (forwarded to le-agent `:8000`), but the endpoint lives on the multica Go server.

## Resolved topology

- **le-agent host**: runs a `gateway`-group stub (caller of `/rl/*`, `/chat/completions`)
  + a `leagent_api` executor (forwards `agent_start` to le-agent).
- **areal host**: runs a `leagent_api`/`multica_api` stub (areal is the caller) + a
  `gateway` executor (forwards `/rl/*`, `/chat/completions` to the real AReaL gateway at
  `gateway_upstream_url`).
- **multica host**: runs `multica_server.py` (public LLM + remote-shell surface, port
  `9200`, DB-routed - already bridged) and the multica Go server on `$PORT`. **New:** a
  `--side multica` executor that forwards `multica_api` channels to the multica Go server
  at `BRIDGE_MULTICA_UPSTREAM_URL` (loopback `$PORT`).

The `arealrl` client's "multica's in-network db_bridge stub" is the gateway-group stub
serving `/rl/*`; it gains `rl_close_segment` once the channel is registered. No third
gateway stub is added - the gateway group's `stub_side` is `leagent` and multica runs an
instance of that side's stub for its `/rl/*` calls (same `rpc_rl_*` tables, claimed by the
areal-side executor).

## Implementation approach

### `multica/db_bridge/channels.py`

- Extend literals: `Side = Literal["leagent", "areal", "multica"]`,
  `Group = Literal["gateway", "leagent_api", "multica_api"]`.
- Extend `stub_side`/`executor_side` derivation: `multica_api` -> stub `areal`,
  executor `multica`.
- Move `env_dispatch`, `env_dispatch_delete` from `leagent_api` to `multica_api`
  (`agent_start` stays in `leagent_api`).
- Add:
  - `env_dispatch_dag` - `GET /api/v1/env-dispatch/{projectID}/dag`, `multica_api`,
    `kind=json`, `default_timeout_s=30`, `default_concurrency=4`.
  - `rl_close_segment` - `POST /rl/close_segment`, `gateway`, `kind=json`,
    `default_timeout_s=30`, `default_concurrency=4` (mirror `rl_set_reward`).

Path params (`{projectID}`) and GET method are already supported - `env_dispatch_delete`
proves path-param routing; `add_api_route(channel.path, ..., methods=[channel.method])`
handles GET.

### `multica/db_bridge/config.py`

- Add `multica_upstream_url` field + `BRIDGE_MULTICA_UPSTREAM_URL` env (default to the
  multica Go server loopback `$PORT` - exact default confirmed at deploy, Task 19).
- `upstream_for_group("multica_api")` -> `multica_upstream_url`.

### `multica/db_bridge/executor.py`

- Forward `multica_api` channels to `multica_upstream_url`, injecting
  `BRIDGE_MULTICA_UPSTREAM_API_KEY` (read into `BridgeConfig.multica_upstream_api_key`)
  as `Authorization: Bearer <key>` and stripping caller-supplied
  `Authorization` / `x-api-key` / `x-admin-api-key` via a new `relay.strip_credentials`
  helper. The env var is deliberately distinct from `MulticaConfig`'s
  `MULTICA_UPSTREAM_API_KEY` (which authenticates the multica_server LLM relay to the
  AReaL gateway) since the executor forwards to the multica Go server -- a different
  upstream needing its own token.
- `gateway` group unchanged - `rl_close_segment` is forwarded to `gateway_upstream_url`
  with the session-key `Authorization` passed through end-to-end (mirrors `rl_set_reward`).
  **No executor change for `rl_close_segment`.**

### `multica/db_bridge/run_executor.py` / `entrypoints.py`

- Accept `--side multica`; run the executor for `multica_api` channels.

### `multica/db_bridge/stub_server.py`

- Routes auto-register via `add_api_route(channel.path, ...)` per channel; adding the two
  channels to the registry is sufficient - `rl_close_segment` no longer 404s,
  `env_dispatch_dag` is served on the areal side with `202`/`200`/`404` pass-through.

### `multica/db_bridge/schema.sql`

- Add `rpc_env_dispatch_dag` and `rpc_rl_close_segment` (idempotent), mirroring the other
  `multica_api` / `gateway` tables (`id`, `user_id`, `status`, request/response columns,
  `FOR UPDATE SKIP LOCKED` claim).

### `customized_areal/tree_search/agents/multica_dag_client.py`

- Base URL -> `AREAL_BRIDGE_STUB_URL` (constructor-overridable for tests); stop sending
  `MULTICA_API_KEY` on the bridged call (stub is loopback; executor injects the upstream
  key). Preserve `DagNotFound` (404) / `DagForbidden` (403) / `DagTimeout` and the `202`
  re-poll loop. Keep the `httpx.BaseTransport` seam for test fakes.

### `multica/server/internal/arealrl/client.go`

- **No change.** Already targets the db_bridge stub (`stubBaseURL`); `CloseSegment` succeeds
  once the `rl_close_segment` route is registered.

## Data flow

**DAG fetch (areal -> multica, via bridge):**
```
MulticaDagClient (areal) -- GET .../dag, base=AREAL_BRIDGE_STUB_URL -->
areal stub -- insert --> rpc_env_dispatch_dag -- claim --> multica executor
  -- GET .../dag + BRIDGE_MULTICA_UPSTREAM_API_KEY --> multica Go server (GetDag)
  -- 202/200/404 --> (response written back) --> client
client: 202 -> re-poll (new row); 200 -> AssembledDag; 404 -> DagNotFound
```

**Close segment (multica -> areal, via bridge):**
```
arealrl.CloseSegment (multica) -- POST /rl/close_segment, Bearer <proxy_key> -->
multica gateway stub -- insert --> rpc_rl_close_segment -- claim --> areal executor
  -- POST /rl/close_segment + Bearer <proxy_key> --> real AReaL gateway
  -- 200 CloseSegmentResponse / 400 (no active completions) --> client
```

## Risks & mitigations

1. **DAG poll creates N bridge rows** (pass-through-pending). Risk: row bloat if cleanup
   lags. Mitigation: existing `BRIDGE_CLEANUP_INTERVAL` / `BRIDGE_ROW_RETENTION_SECONDS`;
   DAG-fetch volume is low.
2. **Bridge per-row timeout (30s) vs. multica slowness** -> stub returns 504. The DAG
   client MUST treat 504/5xx as transient (re-poll), not raise `DagTimeout`. *Build action:
   confirm `MulticaDagClient`'s status mapping; add a test.*
3. **`BRIDGE_USER_ID` mismatch** across areal stub and multica executor -> rows unclaimed.
   Mitigation: deploy invariant (already required for existing channels); document in
   `.env.multica.example`.
4. **`--side multica` executor is new infra.** Ordering: deploy multica executor + apply
   schema -> then repoint `MulticaDagClient`. The dormant `env_dispatch`/`env_dispatch_delete`
   relabel can happen anytime (see Migration below).
5. **Group move is dormant-safe.** `MulticaEnvDispatchClient` calls `MULTICA_BASE_URL`
   directly (not the stub), so `env_dispatch`/`env_dispatch_delete` carry no bridge traffic
   today. Moving them to `multica_api` is a pure relabel - no traffic migration. (If a
   deployment sets `MULTICA_BASE_URL` to the stub, this changes - flag at build.)

## Migration & deployment ordering

1. Apply `schema.sql` (idempotent) - adds `rpc_env_dispatch_dag`, `rpc_rl_close_segment`.
2. Deploy updated db_bridge code to all hosts; start the new `--side multica` executor on
   the multica host.
3. `rl_close_segment` becomes functional immediately (arealrl client already targets the
   stub; areal-side executor already handles `gateway` channels).
4. Repoint `MulticaDagClient` to `AREAL_BRIDGE_STUB_URL` (env/config change on the areal
   host) - `env_dispatch_dag` becomes functional.
5. The `env_dispatch`/`env_dispatch_delete` client repoint is **out of scope** (follow-up);
   the relabeled channels remain dormant until then.

## Testing strategy

- **Channel registry (unit):** `env_dispatch_dag` in `multica_api` with correct
  `stub_side`/`executor_side`; `rl_close_segment` in `gateway`; `env_dispatch`/
  `env_dispatch_delete` moved to `multica_api`; `agent_start` still in `leagent_api`.
- **Stub:** `rl_close_segment` route registered (no 404); `env_dispatch_dag` `202`/`200`/
  `404` pass-through; GET + path-param routing.
- **Executor:** `multica_api` forwarding to `multica_upstream_url` with
  `MULTICA_UPSTREAM_API_KEY` injected and caller auth stripped; `gateway` passthrough of
  session-key auth for `rl_close_segment`.
- **Client:** `MulticaDagClient` routes to `AREAL_BRIDGE_STUB_URL` (not `MULTICA_BASE_URL`);
  `DagNotFound`/`DagForbidden`/`DagTimeout` preserved; 202 re-poll; 504 treated as
  transient retry.
- **E2E:** close_segment through the bridge (arealrl -> stub -> executor -> gateway);
  DAG fetch through the bridge end-to-end.
- **Rename fallout:** update tests asserting the old `leagent_api` membership / two-side
  `leagent|areal` model.

## Boundary conditions

- **Concurrent DAG polls** for one project: multiple rows, each claimed independently;
  multica returns the same `202`/`200` - idempotent.
- **`close_segment` 400** (no active completions): passed through as 400; `arealrl.checkStatus`
  maps non-2xx to error - correct (closing an empty segment is an error).
- **504 on bridge timeout**: transient retry (risk #2).
- **`BRIDGE_USER_ID` mismatch**: unclaimed rows (risk #3).
- **Long DAG assembly**: bounded by the client's `DagTimeout` (wall-clock), not the bridge
  per-row timeout; each poll is short-lived.

## Deferred to build

- Confirm the multica Go server's `$PORT` default for `BRIDGE_MULTICA_UPSTREAM_URL`
  (Task 19).
- Confirm `X-Bridge-User-Id` need on DAG calls (single `BRIDGE_USER_ID` vs. multi-user stub)
  - match whatever `env_dispatch` will use when its client is migrated (Task 19 / open).
- Confirm `MulticaDagClient` 504/5xx handling (risk #2) and add a test.

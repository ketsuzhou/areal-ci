# Design - env-dispatch-dag-db-bridge

## Context

`db_bridge` relays cross-service HTTP through per-endpoint Supabase tables: a loopback
**stub** on the caller's host captures the request, enqueues a row, polls for the response;
an **executor** on the callee's host claims the row (`FOR UPDATE SKIP LOCKED`), forwards to
the real service over loopback, and writes the response back. Two sides exist today -
`leagent` and `areal` - with two groups: `gateway` (le-agent->AReaL: `rl_*`, chat) and
`leagent_api` (AReaL->le-agent: `agent_start`, `env_dispatch`, `env_dispatch_delete`).

The DAG fetch (`GET /api/v1/env-dispatch/{projectID}/dag`) is called by areal's
`MulticaDagClient`, which polls until `200` (assembled) or `404` (unknown project), with
`202` while in progress. Today it does this over direct `httpx` to `MULTICA_BASE_URL`.

Two defects to fix together:
1. The DAG fetch has **no db_bridge channel** - it bypasses the bridge entirely.
2. The env-dispatch channels are **mislabeled**: group `leagent_api` forwards to
   `BRIDGE_LEAGENT_UPSTREAM_URL` (`127.0.0.1:8000`, le-agent), but the endpoint is served by
   the **multica Go server** (`multica/server/internal/handler/env_dispatch.go`), a separate
   process/port from le-agent. Adding the DAG channel on top of a mislabeled group would
   propagate the bug, so the side/group is reconciled first.

## Selected approach

### D1. Add a `multica` Side and `multica_api` group

Extend the type literals and move the env-dispatch surface into the new group:

```
Side  = Literal["leagent", "areal", "multica"]
Group = Literal["gateway", "leagent_api", "multica_api"]

# multica_api: AReaL -> multica server
#   stub_side      = areal    (areal calls the loopback stub)
#   executor_side  = multica  (forwards to the multica Go server)
```

Channels moved into `multica_api`: `env_dispatch`, `env_dispatch_delete`, and the new
`env_dispatch_dag`. `agent_start` stays in `leagent_api` (it really does forward to
le-agent). `gateway` is unchanged.

`stub_side` / `executor_side` derivation in `channels.py` is extended to map
`multica_api` -> stub `areal`, executor `multica`.

**Why a third side, not a rename of `leagent`:** le-agent (FastAPI, `backend/core`,
typically `:8000`) and the multica server (Go, `multica/server/`, separate port) are
distinct processes even when co-located on one host. The current
`BRIDGE_LEAGENT_UPSTREAM_URL=127.0.0.1:8000` cannot serve both. A dedicated `multica` side
with its own `BRIDGE_MULTICA_UPSTREAM_URL` decouples env-dispatch forwarding from le-agent's
port and makes the deployment topology explicit.

**Alternative considered (rejected):** keep two sides and add per-channel upstream URLs.
Rejected because it spreads routing config across channels instead of grouping by callee,
and the group concept already exists to encode "which host runs the executor."

### D2. New channel: `env_dispatch_dag`

```
Channel(
    name="env_dispatch_dag",
    group="multica_api",
    method="GET",
    path="/api/v1/env-dispatch/{projectID}/dag",
    kind="json",
    default_timeout_s=30.0,      # each poll is a quick GET; client re-polls on 202
    default_concurrency=4,
)
```

Schema: add `rpc_env_dispatch_dag` to `schema.sql` (idempotent), mirroring the other
`multica_api`/`leagent_api` tables (`id`, `user_id`, `status`, request/response columns,
`FOR UPDATE SKIP LOCKED` claim).

### D3. Polling semantics - pass-through-pending, not block-in-stub

The `MulticaDagClient` already polls (202 -> sleep -> re-poll; 200 -> done; 404 ->
`DagNotFound`). Each client poll becomes **one bridged request**: stub enqueues a row,
executor forwards the GET to multica, multica returns `202`/`200`/`404`, response is written
back, stub returns it. The client decides whether to re-poll.

Chosen over **block-in-stub** (one long-lived row where the stub blocks until multica
returns `200`):

| Approach            | Bridge rows        | Executor occupancy         | Faithful to client? |
|---------------------|--------------------|----------------------------|---------------------|
| Pass-through-pending| one per poll       | held only for the GET      | yes (loop unchanged)|
| Block-in-stub       | one per DAG        | held for whole assembly    | no (loop removed)   |

Pass-through preserves the existing client semantics and `DagTimeout` behavior, and avoids
tying up an executor coroutine for the full DAG assembly (which can run minutes). The cost
is N short-lived rows per DAG; acceptable given low DAG-fetch volume and the existing row
retention/cleanup (`BRIDGE_CLEANUP_INTERVAL`, `BRIDGE_ROW_RETENTION_SECONDS`).

`default_timeout_s=30` bounds a single poll; the client's own `DagTimeout` bounds the
overall wait. `202` carries a status body, passed through untouched.

### D4. Auth / header handling

The multica server requires `Authorization: Bearer <MULTICA_API_KEY>`. Under the bridge:

- The **stub** is loopback-only (`127.0.0.1`); the areal client needs no multica credential.
  If the stub serves multiple users, the client sends `X-Bridge-User-Id` (the stub already
  supports this via `BRIDGE_USER_ID`).
- The **executor** attaches the upstream credential when forwarding to multica. db_bridge
  already has this pattern (`upstream_api_key` in `config.py` and credential stripping in
  `multica_server.py`). Reuse it: the multica-side executor injects
  `BRIDGE_MULTICA_UPSTREAM_API_KEY` (read into `BridgeConfig.multica_upstream_api_key`) and
  strips caller auth via `relay.strip_credentials` (`authorization` / `x-api-key` /
  `x-admin-api-key`). The env var is deliberately distinct from
  `MULTICA_UPSTREAM_API_KEY` (which authenticates the multica_server LLM relay to the AReaL
  gateway) to avoid the config-constant collision discovered in build.

So `MulticaDagClient` is changed to read `AREAL_BRIDGE_STUB_URL` (not `MULTICA_BASE_URL`)
for its base, and to stop sending `MULTICA_API_KEY` on the bridged call. `MULTICA_BASE_URL`
/ `MULTICA_API_KEY` remain in use by the **out-of-scope** `MulticaEnvDispatchClient` and
`MulticaSweLegoProvider` until those are migrated in a follow-up.

### D5. Client repoint

`multica_dag_client.py` currently builds `f"{self._base}/api/v1/env-dispatch/{project_id}/dag"`
from `MULTICA_BASE_URL`. Change the base to `AREAL_BRIDGE_STUB_URL` (default) with
constructor override for tests. Keep the existing error mapping:

- `200` -> parse `AssembledDag`, return.
- `202` -> sleep, re-poll. Bridge transients `502`/`503`/`504` (relay error / stub timeout)
  are also re-polled up to the wall-clock deadline; the `200`/`404`/`403` mapping is
  unchanged.
- `404` -> `DagNotFound` (unchanged).
- `403` -> `DagForbidden` (unchanged).
- timeout -> `DagTimeout` (unchanged; now driven by client wall-clock, not bridge per-row
  timeout).

Constructor takes an `httpx.BaseTransport` already (for test fakes) - preserved.

### D6. New channel: `rl_close_segment` (gateway group)

`POST /rl/close_segment` is served by AReaL's inference_service gateway alongside
`/rl/start_session`, `/rl/set_reward`, `/rl/end_session` (session-key auth, mirroring
`set_reward`). The multica `arealrl` Go client (`multica/server/internal/arealrl/client.go`)
is **already wired to route `/rl/*` through the db_bridge stub** - `Client.stubBaseURL` is
the stub URL and `CloseSegment` posts to `closeSegmentPath`. But the stub registers a route
only for channels in the `CHANNELS` registry (`app.add_api_route(channel.path, ...)`,
`stub_server.py:348`), and `rl_close_segment` is absent - so the call currently receives a
FastAPI `404`.

```
Channel(
    name="rl_close_segment",
    group="gateway",                 # le-agent/multica -> AReaL gateway (unchanged group)
    method="POST",
    path="/rl/close_segment",
    kind="json",
    default_timeout_s=30.0,          # mirror rl_set_reward / rl_end_session
    default_concurrency=4,           # mirror rl_set_reward
)
```

**Why no caller repoint (unlike the DAG):** the `arealrl` client already points at the
stub. The gap is purely the missing channel definition + `rpc_rl_close_segment` table + the
stub route registration. Once added, the existing gateway-group executor on the areal host
forwards the call to `gateway_upstream_url` (the real AReaL gateway) automatically - no new
executor side, no `multica_api` involvement. The session-key `Authorization` header is
forwarded end-to-end (gateway channels forward caller auth to the real gateway, unlike
`multica_api` which strips and re-injects).

**Schema:** add `rpc_rl_close_segment` to `schema.sql` (idempotent), mirroring the other
gateway tables (`rl_start_session`, `rl_set_reward`, `rl_end_session`).

## Data flow

```
MulticaDagClient (areal, customized_areal)
  │  GET /api/v1/env-dispatch/{projectID}/dag
  │  base = AREAL_BRIDGE_STUB_URL  (127.0.0.1:9101)
  ▼
areal-side stub  ── insert ──▶  rpc_env_dispatch_dag  (Supabase, shared)
  │                                      │
  │  poll row for response               │  claim (FOR UPDATE SKIP LOCKED)
  ▼                                      ▼
multica-side executor  ── GET ──▶  multica Go server
  │  + BRIDGE_MULTICA_UPSTREAM_API_KEY   (env_dispatch.go: GetDag)
  │  to BRIDGE_MULTICA_UPSTREAM_URL      │
  │                                      ▼
  ◀── write response (202 / 200 / 404) ──
  │
  ▼
MulticaDagClient receives:
   202 -> sleep, re-poll (new bridge row)
   200 -> return AssembledDag
   404 -> DagNotFound
   403 -> DagForbidden
```

## Transition state (intentional)

After this change, the env-dispatch surface is partially bridged:

| Endpoint                          | Channel            | Client routed via bridge? |
|-----------------------------------|--------------------|---------------------------|
| `GET .../dag`                     | `env_dispatch_dag` | **yes** (this change)     |
| `POST /api/v1/env-dispatch`       | `env_dispatch`     | no (client still direct)  |
| `DELETE /api/v1/env-dispatch/{id}`| `env_dispatch_delete` | no (client still direct)|

The channels for POST/DELETE are relabeled to `multica_api` (correct forwarding target) but
their clients are not repointed in this change - explicitly scoped out. This is acceptable
because the bridge channels are transparent drop-ins: repointing the clients later is a
config-only change once the `multica` side executor is deployed.

## Open questions / build-phase verification

1. **Deployment topology**: confirm the multica server's loopback port (distinct from
   le-agent `:8000`) so `BRIDGE_MULTICA_UPSTREAM_URL` defaults correctly. If multica and
   le-agent are co-located, the multica-side executor still forwards to the multica port,
   not `:8000`.
2. **`X-Bridge-User-Id` on DAG calls**: confirm whether the areal-side DAG stub serves a
   single `BRIDGE_USER_ID` (no header needed) or multiple users (client must send the
   header). Match whatever `env_dispatch` will use when its client is later migrated.
3. **Schema apply**: `schema.sql` is idempotent; coordinate the `rpc_env_dispatch_dag`
   table apply with the shared Supabase (deployment note in tasks).

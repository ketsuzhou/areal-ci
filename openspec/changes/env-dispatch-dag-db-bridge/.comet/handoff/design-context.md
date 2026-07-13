# Comet Design Handoff

- Change: env-dispatch-dag-db-bridge
- Phase: design
- Mode: compact
- Context hash: a8a80465935ae065bffa2d62fca7017f96719b1a311a69aee6c62882b50bd91e

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-dispatch-dag-db-bridge/proposal.md

- Source: openspec/changes/env-dispatch-dag-db-bridge/proposal.md
- Lines: 1-85
- SHA256: bfd5c68448b9b4f6a442c7590923b2892f90724bc172f1a4e0abfa385fd85e97

[TRUNCATED]

```md
## Why

AReaL's `MulticaDagClient` fetches the assembled rollout DAG over direct `httpx` to
`MULTICA_BASE_URL` (`GET /api/v1/env-dispatch/{projectID}/dag`), bypassing `db_bridge`.
This contradicts the deployment invariant `db_bridge` exists to enforce - AReaL and
multica run on hosts that cannot reach each other directly and share only a Supabase
database - and contradicts `multica_environment_protocol.md`, which states verbatim
*"AReaL never calls Multica directly for env-dispatch."* In the network-isolated
production topology the DAG fetch fails entirely. A second defect compounds this: the
existing `env_dispatch` / `env_dispatch_delete` db_bridge channels are labeled
`leagent_api` / `leagent` side and forwarded to `BRIDGE_LEAGENT_UPSTREAM_URL`
(`127.0.0.1:8000`, "the real le-agent API"), but the endpoint is actually served by the
**multica Go server** (`multica/server/internal/handler/env_dispatch.go`). So even where
a channel exists, the bridge's forwarding target is ambiguous and likely wrong. A third
gap: multica's `arealrl` Go client already calls `POST /rl/close_segment` against the
db_bridge stub (it is wired to route `/rl/*` through the bridge), but the stub registers
no `close_segment` channel - so the call receives a `404`. The bridge's `/rl/*` contract
is incomplete.

## What Changes

- Add an `env_dispatch_dag` db_bridge channel for `GET /api/v1/env-dispatch/{projectID}/dag`
  (new `rpc_env_dispatch_dag` table + schema row), with `202`/`200`/`404` pass-through so
  the existing client polling semantics are preserved.
- Repoint `MulticaDagClient` (`customized_areal/tree_search/agents/multica_dag_client.py`)
  to fetch through the local areal-side db_bridge stub (`AREAL_BRIDGE_STUB_URL`) instead of
  direct `MULTICA_BASE_URL`. The client keeps its poll-until-`200`/`404` loop; each poll is
  one bridged request.
- **Reconcile the db_bridge Side/group naming.** Introduce a `multica` Side and a
  `multica_api` group; move `env_dispatch`, `env_dispatch_delete`, and the new
  `env_dispatch_dag` channels into `multica_api` (stub on the areal host, executor on the
  multica host). Add a distinct `BRIDGE_MULTICA_UPSTREAM_URL` config so env-dispatch
  traffic forwards to the multica server explicitly, not to le-agent's port. Update
  `Side`/`Group` literals, `stub_side`/`executor_side` derivation, `upstream_for_group`,
  `.env.*` examples, and `run_executor --side` to support `multica`.
- Add an `rl_close_segment` db_bridge channel for `POST /rl/close_segment` (gateway group,
  `rpc_rl_close_segment` table) so the multica `arealrl` client's existing stub-routed
  `close_segment` calls are bridged to the AReaL gateway instead of receiving a stub `404`.
  No caller repoint is needed - the `arealrl` client already targets the db_bridge stub.
- Update `multica_environment_protocol.md` to document the DAG fetch as bridged and correct
  the side labels.

### Non-goals

- Repointing the `env_dispatch` / `env_dispatch_delete` **clients**
  (`MulticaEnvDispatchClient`) - the channels are relabeled but the POST/DELETE clients
  keep their current direct-HTTP wiring for now (tracked as follow-up).
- The env-checkpoint endpoints (`POST /api/v1/env-checkpoints`,
  `GET /api/v1/projects/{id}/env-checkpoints`) - no db_bridge channel yet.
- The `MulticaSweLegoProvider` sandboxes/* calls and the TS `verifier-rl` extension.

## Capabilities

### New Capabilities

_(none)_

### Modified Capabilities

- `v2-segment-dag`: two requirements gain transport contracts. "Polling AssembledDag
  return at task completion" - AReaL SHALL fetch the assembled DAG through the db_bridge
  stub (`AREAL_BRIDGE_STUB_URL`), not via direct HTTP to multica (`202`/`200`/`404`
  semantics unchanged). "V2 no-reward segment close" - `POST /rl/close_segment` SHALL be
  reachable through the db_bridge stub (registered route + `rpc_rl_close_segment` table),
  with session-key auth forwarded end-to-end; the stub MUST NOT return `404` for the path.

## Impact

- `multica/db_bridge/`: `channels.py` (new `env_dispatch_dag` + `rl_close_segment`
  channels; `multica` side / `multica_api` group), `config.py`
  (`BRIDGE_MULTICA_UPSTREAM_URL`, `upstream_for_group`), `entrypoints.py` /
  `run_executor.py` (`--side multica`), `stub_server.py` + `executor.py` (serve/forward the
  new channels; `rl_close_segment` reuses the existing gateway-group forwarding to
  `gateway_upstream_url`), `schema.sql` (`rpc_env_dispatch_dag`, `rpc_rl_close_segment`),
  `.env.areal` / `.env.leagent` / `.env.multica` examples.
- `customized_areal/tree_search/agents/multica_dag_client.py`: repoint to the bridge stub;
  preserve `DagNotFound`/`DagForbidden`/`DagTimeout` error mapping.
- `customized_areal/tree_search/agents/multica_environment_protocol.md`: doc update.
- `multica/server/internal/arealrl/client.go`: **no change** - already targets the db_bridge
  stub; gains a working `close_segment` route once the channel is registered.
```

Full source: openspec/changes/env-dispatch-dag-db-bridge/proposal.md

## openspec/changes/env-dispatch-dag-db-bridge/design.md

- Source: openspec/changes/env-dispatch-dag-db-bridge/design.md
- Lines: 1-215
- SHA256: db78b265831d48d2b166ef39e2e406fc07266cc413fa71ebdd883d419329a7da

[TRUNCATED]

```md
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
```

Full source: openspec/changes/env-dispatch-dag-db-bridge/design.md

## openspec/changes/env-dispatch-dag-db-bridge/tasks.md

- Source: openspec/changes/env-dispatch-dag-db-bridge/tasks.md
- Lines: 1-95
- SHA256: 5872733b4a942bb07de38e76570e23d77decb1a845e3c6e2fbb82786d7f93e7c

[TRUNCATED]

```md
# Tasks - env-dispatch-dag-db-bridge

## db_bridge - side/group reconciliation

- [ ] Task 1: Extend `Side`/`Group` literals in `multica/db_bridge/channels.py` to add
      `multica` and `multica_api`; update `stub_side`/`executor_side` derivation so
      `multica_api` -> stub `areal`, executor `multica`.
- [ ] Task 2: Move `env_dispatch` and `env_dispatch_delete` from `leagent_api` into
      `multica_api` (they forward to the multica server, not le-agent). `agent_start` stays
      in `leagent_api`.
- [ ] Task 3: Add the `env_dispatch_dag` channel (`GET /api/v1/env-dispatch/{projectID}/dag`,
      group `multica_api`, `kind=json`, `default_timeout_s=30`, `default_concurrency=4`) to
      the `CHANNELS` registry.
- [ ] Task 4: `multica/db_bridge/config.py` - add `multica_upstream_url` field +
      `BRIDGE_MULTICA_UPSTREAM_URL` env (default to the multica server loopback port, TBD in
      Task 18); extend `upstream_for_group` so `multica_api` -> `multica_upstream_url`.
- [ ] Task 5: `multica/db_bridge/run_executor.py` / `entrypoints.py` - accept `--side multica`
      and run the executor for `multica_api` channels.

## db_bridge - rl_close_segment gateway channel

- [ ] Task 6: Add the `rl_close_segment` channel to `CHANNELS` in `channels.py`
      (`POST /rl/close_segment`, group `gateway`, `kind=json`, `default_timeout_s=30`,
      `default_concurrency=4` - mirror `rl_set_reward` / `rl_end_session`).
- [ ] Task 7: `multica/db_bridge/schema.sql` - add the `rpc_rl_close_segment` table
      (idempotent), mirroring `rpc_rl_set_reward` / `rpc_rl_end_session`.
- [ ] Task 8: `multica/db_bridge/stub_server.py` - confirm the `POST /rl/close_segment`
      route auto-registers via `add_api_route(channel.path, ...)` for the new channel and
      that the response (`CloseSegmentResponse`, typed 400 on no-active-completions) is
      passed through untouched; the stub MUST NOT return `404` for the path. Executor
      forwarding needs no change (gateway group already forwards to `gateway_upstream_url`,
      session-key `Authorization` forwarded end-to-end).

## db_bridge - env_dispatch_dag stub / executor / schema

- [ ] Task 9: `multica/db_bridge/schema.sql` - add the `rpc_env_dispatch_dag` table
      (idempotent), mirroring the other `multica_api` tables incl. `user_id`,
      `FOR UPDATE SKIP LOCKED` claim support.
- [ ] Task 10: `multica/db_bridge/stub_server.py` - serve `env_dispatch_dag` on the areal
      side (path routing, `202`/`200`/`404` pass-through; body untouched).
- [ ] Task 11: `multica/db_bridge/executor.py` - forward `multica_api` channels to
      `multica_upstream_url`, injecting `MULTICA_UPSTREAM_API_KEY` and stripping
      caller-supplied auth (reuse the existing `upstream_api_key` /
      `_BRIDGE_CRED_HEADERS` pattern).
- [ ] Task 12: Update `.env.areal` / `.env.leagent` / `.env.multica.example` with
      `BRIDGE_MULTICA_UPSTREAM_URL`, both new channels, and `--side multica` run notes.

## areal - client repoint + docs

- [ ] Task 13: `customized_areal/tree_search/agents/multica_dag_client.py` - repoint base to
      `AREAL_BRIDGE_STUB_URL` (constructor-overridable for tests); stop sending
      `MULTICA_API_KEY` on the bridged call; preserve `DagNotFound` (404) /
      `DagForbidden` (403) / `DagTimeout` mapping and the `202` re-poll loop.
- [ ] Task 14: Update `customized_areal/tree_search/agents/multica_environment_protocol.md` -
      document the DAG fetch as bridged via `AREAL_BRIDGE_STUB_URL`; correct the side labels
      (env-dispatch -> multica, not le-agent); note `close_segment` is bridged via the
      gateway-group `rl_close_segment` channel.
- [ ] Task 15: Confirm `multica/server/internal/arealrl/client.go` needs **no change** - it
      already targets the db_bridge stub (`stubBaseURL`); verify `CloseSegment` succeeds
      once the `rl_close_segment` route is registered (currently 404).

## specs

- [ ] Task 16: `v2-segment-dag` delta spec - extend two requirements with transport
      contracts: (a) "Polling AssembledDag return at task completion" - AReaL SHALL fetch
      the DAG through the db_bridge stub (`AREAL_BRIDGE_STUB_URL`); (b) "V2 no-reward
      segment close" - `POST /rl/close_segment` SHALL be reachable through the db_bridge
      stub (registered route + `rpc_rl_close_segment`), session-key auth forwarded
      end-to-end, stub MUST NOT 404. Add scenarios asserting the bridged paths and that
      `202`/`200`/`404` (DAG) and reward-less-close (segment) semantics are preserved
      end-to-end.

## tests

- [ ] Task 17: `multica/db_bridge/tests/` - add channel tests for `env_dispatch_dag`
      (`202`/`200`/`404` pass-through + `multica_api` side/group) and `rl_close_segment`
      (route registered, reward-less `CloseSegmentResponse` pass-through, 400 on
      no-active-completions); update existing tests that assume `env_dispatch` is in
      `leagent_api` / the two-side `leagent|areal` model.
- [ ] Task 18: `customized_areal/tree_search/tests/` - update `test_env_dispatch_client.py`
```

Full source: openspec/changes/env-dispatch-dag-db-bridge/tasks.md

## openspec/changes/env-dispatch-dag-db-bridge/specs/v2-segment-dag/spec.md

- Source: openspec/changes/env-dispatch-dag-db-bridge/specs/v2-segment-dag/spec.md
- Lines: 1-93
- SHA256: 9db7236a08bb9e7b3c6855f58f47d75a7df0c0149234c63db354669dfa4620dd

[TRUNCATED]

```md
## MODIFIED Requirements

### Requirement: Polling AssembledDag return at task completion

Multica SHALL expose `GET /api/v1/env-dispatch/{projectID}/dag` that returns `202` with a
status body while the root task is in progress and `200` with the assembled `AssembledDag`
when the root task completes (success or failed). The endpoint reuses the existing
env-dispatch path with `project_id` as the root handle.

AReaL SHALL fetch the assembled DAG through the `db_bridge` stub (`AREAL_BRIDGE_STUB_URL`),
not via direct HTTP to multica. The bridge is a transparent transport: the `202`/`200`/`404`
response semantics defined above are preserved end-to-end. Each client poll is one bridged
request - the stub enqueues a row in `rpc_env_dispatch_dag`, the multica-side executor
forwards the `GET` to the multica server (attaching the upstream credential and stripping
caller auth), and the stub returns multica's response to the client. AReaL's poll loop
(re-poll on `202`, return on `200`, raise `DagNotFound` on `404`) is unchanged.

The env-dispatch channels (`env_dispatch`, `env_dispatch_delete`, `env_dispatch_dag`)
belong to the `multica_api` bridge group: the stub runs on the AReaL host and the executor
runs on the multica host forwarding to the multica server's loopback URL
(`BRIDGE_MULTICA_UPSTREAM_URL`), distinct from the le-agent upstream.

#### Scenario: In-progress poll

- **WHEN** AReaL polls the endpoint before the root task completes
- **THEN** the system returns `202` with a status body indicating the task is in progress

#### Scenario: Completed poll returns AssembledDag

- **WHEN** the root task has completed
- **THEN** the system returns `200` with an `AssembledDag` that AReaL resolves and
  reconstructs losslessly into the `ExecutionDAG`

#### Scenario: Unknown project rejected

- **WHEN** a caller polls a `project_id` that does not exist
- **THEN** the system returns `404`

#### Scenario: DAG fetch is routed through db_bridge

- **WHEN** AReaL fetches the assembled DAG for a project
- **THEN** the request is sent to the local `db_bridge` stub (`AREAL_BRIDGE_STUB_URL`) and
  forwarded to the multica server by the multica-side executor; AReaL does not open a
  direct HTTP connection to `MULTICA_BASE_URL` for the DAG fetch

#### Scenario: Bridged poll preserves response semantics

- **WHEN** AReaL polls the DAG through the bridge while the task is in progress, then again
  after it completes
- **THEN** the first poll returns `202` (AReaL re-polls) and the subsequent poll returns
  `200` with the `AssembledDag`, identical to the direct-HTTP contract

#### Scenario: Bridged unknown project returns 404

- **WHEN** AReaL polls a non-existent `project_id` through the bridge
- **THEN** the bridge returns `404` and AReaL raises `DagNotFound`, identical to the
  direct-HTTP contract

### Requirement: V2 no-reward segment close

The v2 inference_service SHALL provide a `close_segment` operation (`POST /rl/close_segment`)
that snapshots a session's active completions into a ready trajectory **without setting a
reward**, returning the new `trajectory_id`. Closing a segment MUST NOT require a prior
`set_reward` and MUST NOT assign any reward value. The session MUST remain live for further
turns and further `close_segment` calls after a close.

The `close_segment` operation SHALL be reachable through the `db_bridge` stub: the stub
registers a route for `POST /rl/close_segment` and relays it via the `rpc_rl_close_segment`
table to the AReaL-side executor, which forwards it to the real AReaL proxy gateway. The
session-key auth (`Authorization: Bearer <proxy_key>`) is forwarded end-to-end, mirroring
`rl_set_reward` / `rl_end_session`. The stub MUST NOT return `404` for `/rl/close_segment`.

#### Scenario: Close produces a reward-less trajectory
- **WHEN** Multica calls `close_segment` on a session that has active completions
- **THEN** the system moves the active completions into a ready trajectory, returns its
  `trajectory_id`, and assigns no reward to any interaction in that trajectory

#### Scenario: Session stays live after close
- **WHEN** a segment is closed on a session
- **THEN** subsequent `/chat/completions` turns on the same session are captured into a new
```

Full source: openspec/changes/env-dispatch-dag-db-bridge/specs/v2-segment-dag/spec.md


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
- Tests: `multica/db_bridge/tests/` (new channels + side/group rename), `customized_areal/
  tree_search/tests/` (`test_env_dispatch_client.py`, dag-client wiring).
- Deployment: the multica host must run a `--side multica` executor forwarding env-dispatch
  channels to the multica server's loopback port; the areal host stub gains the new
  channels; `rl_close_segment` is served by the existing gateway-group stub/executor.

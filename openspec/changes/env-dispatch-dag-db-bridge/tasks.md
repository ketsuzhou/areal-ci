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
      and dag-client wiring tests to assert the DAG fetch routes through the bridge stub
      (not `MULTICA_BASE_URL`), with `202`/`200`/`404` intact. Add a close_segment
      end-to-end test asserting the arealrl path through the stub no longer 404s.

## verify

- [ ] Task 19: Confirm deployment topology - the multica server's loopback port (distinct
      from le-agent `:8000`) so `BRIDGE_MULTICA_UPSTREAM_URL` defaults correctly; record in
      `.env.multica.example`.
- [ ] Task 20: Run `multica/db_bridge/tests/` green and `customized_areal/tree_search/tests/`
      green (env_dispatch, dag-client, close_segment, side/group tests).
- [ ] Task 21: Deployment note - apply `rpc_env_dispatch_dag` + `rpc_rl_close_segment`
      tables against the shared Supabase (idempotent `schema.sql`); multica host runs
      `--side multica` executor; areal host stub gains both new channels;
      `rl_close_segment` is served by the existing gateway-group executor.

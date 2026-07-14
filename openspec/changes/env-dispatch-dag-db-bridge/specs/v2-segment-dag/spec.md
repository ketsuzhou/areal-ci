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
  active segment that can be closed independently

#### Scenario: Close without active completions is rejected
- **WHEN** `close_segment` is called on a session with no active completions
- **THEN** the system returns a typed error and produces no trajectory

#### Scenario: Close segment routed through db_bridge
- **WHEN** Multica's `arealrl` client calls `close_segment` (it is wired to route `/rl/*`
  through the db_bridge stub)
- **THEN** the stub serves `POST /rl/close_segment` (registered route), relays it via
  `rpc_rl_close_segment`, the AReaL-side executor forwards it to the real gateway with the
  session-key auth, and the `CloseSegmentResponse` is returned to the client - the stub does
  not return `404` for the path

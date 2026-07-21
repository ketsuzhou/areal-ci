## 1. Required training-mode request contract

- [ ] 1.1 Write failing handler and service tests: omitted `training_mode` returns
  HTTP 400, `false` plus training IDs fails validation, `true` without
  `train_agent_id` fails, and the two valid forms reach the service with the exact
  boolean.
- [ ] 1.2 Verify RED: `go test ./server/internal/handler ./server/internal/service
  -run 'EnvDispatch.*TrainingMode' -count=1` fails because no explicit
  training-mode contract exists.
- [ ] 1.3 Add `EnvDispatchRequest.TrainingMode *bool` at the HTTP boundary; reject
  nil before constructing `EnvDispatchInput`; pass the dereferenced value as
  `EnvDispatchInput.TrainingMode bool`. In service validation enforce
  `!TrainingMode && (TrainAgentID != "" || CriticAgentID != "")` failure and
  `TrainingMode && TrainAgentID == ""` failure.
- [ ] 1.4 Verify GREEN: the Step 1.2 command passes.

## 2. Durable dispatch root and readiness

- [ ] 2.1 Write failing persistence and readiness tests: every successful rollout
  persists mode and leader task, `/dag` returns 202 for queued/running roots, and
  returns assembled data for a completed non-training root without any
  `training_dispatch` row.
- [ ] 2.2 Verify RED: `go test ./server/internal/handler ./server/internal/service
  -run 'EnvDispatch.*(Root|Readiness|Dag)' -count=1` fails because readiness still
  joins `training_dispatch`.
- [ ] 2.3 Create migration `204_env_dispatch_run` (`env_dispatch_run` keyed by
  project with workspace, training mode, nullable root task) and add create,
  root-bind, and workspace-scoped status queries; update sqlc output via the
  existing generation workflow without adding tools.
- [ ] 2.4 Wire dispatch persistence: create the dispatch row once the project
  exists, bind `LeaderRunID` after enqueue, and make `/dag` exclusively query the
  new root status. Preserve 202, failed-density, and successful-DAG response
  shapes.
- [ ] 2.5 Verify GREEN: the Step 2.2 command passes.

## 3. Dual-source segment persistence

- [ ] 3.1 Write failing local-segment tests: insert task messages at known sequence
  numbers and assert the local recorder upserts session `multica:<task-id>`,
  snapshots only the requested sequence range in order, sets
  `trajectory_source=task_messages`, sets `trainable=false`, leaves AReaL fields
  null, repeated close is idempotent, and runtime provider secrets never enter the
  serialized trajectory.
- [ ] 3.2 Verify RED: `go test ./server/internal/service -run 'InteractionDAG.*Local'
  -count=1` fails because local segment recording does not exist.
- [ ] 3.3 Create migration `205_interaction_dag_local_trajectory`: backfill existing
  rows, make `trajectory_id`/`tensor_ref` nullable, add `trajectory_source`
  (default `areal_tensor`), `trainable` (default true), and `trajectory` (default
  `[]`); add checks requiring non-null AReaL fields only for trainable tensor
  segments and null AReaL fields for task-message segments.
- [ ] 3.4 Implement `RecordLocalSegmentForEvent` and assembly: serialize an
  allowlisted message-event shape (sequence, type, tool, content, input, output)
  from persisted rows, compute start/end using existing sequence queries, atomically
  insert the segment and environment snapshot, and emit `TrajectorySource`,
  `Trainable`, and `Trajectory` from both source types (AReaL-only fields nullable).
- [ ] 3.5 Verify GREEN: the Step 3.2 command plus
  `go test ./server/internal/service -run InteractionDAG -count=1` pass.

## 4. Record every env-dispatch agent at event seams

- [ ] 4.1 Replace the old non-trained no-op test with failing behavior tests: a
  non-trained issue task and channel task record local segments, a mixed
  trained/non-trained pair records both sources with an edge, ordinary non-env-
  dispatch tasks remain no-ops, and the non-training path makes zero fake AReaL
  client calls.
- [ ] 4.2 Verify RED: `go test ./server/internal/service -run
  'InteractionDAG.*(NonTrain|Mixed|Channel)' -count=1` fails at the current
  `extractArealProxyConfig` early return.
- [ ] 4.3 Implement unified project and trajectory-source routing: resolve project
  from issue or chat session, gate local recording on an `env_dispatch_run` lookup,
  keep the current bridge close/export order for proxy tasks, and otherwise use
  deterministic local session/run mapping plus local segment recording. Reuse
  existing edge and one-segment guards.
- [ ] 4.4 Verify GREEN: the Step 4.2 command and
  `go test ./server/internal/service -count=1` pass.

## 5. Parse mixed DAGs safely in AReaL

- [ ] 5.1 Write failing Python contract and resolver tests: build a mixed DAG with
  one `areal_tensor` and one `task_messages` segment; assert strict parsing
  succeeds, topology retains both segments, only the trainable tensor ref is
  resolved, and cleanup never receives the local segment.
- [ ] 5.2 Verify RED: `uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py
  customized_areal/tree_search/tests/test_assembler_ref_resolve.py
  customized_areal/tree_search/tests/test_segment_dag_training_path.py -q` fails
  because current dataclasses require tensor fields and the assembler resolves every
  segment.
- [ ] 5.3 Implement the Python dual-source contract: add typed defaults for
  `trajectory_source`, `trainable`, and `trajectory`; make `trajectory_id`/
  `tensor_ref` optional; validate trainable segments strictly and skip
  resolution/cleanup for non-trainable segments while retaining their DAG identity
  and metadata.
- [ ] 5.4 Verify GREEN: the Step 5.2 command passes.

## 6. Cross-layer regression verification

- [ ] 6.1 Run focused Go tests: `go test ./server/internal/service
  ./server/internal/handler -count=1` passes.
- [ ] 6.2 Run Go static checks: `go vet ./server/internal/service
  ./server/internal/handler` exits 0.
- [ ] 6.3 Run focused Python tests: the Task 5 Step 2 command passes.
- [ ] 6.4 Update the repository graph: `graphify update .` completes successfully;
  dirty graph outputs are retained.
- [ ] 6.5 Review secret and AReaL-call boundaries: confirm provider API-key fixtures
  occur only in request/setup fixtures, never in serialized DAG assertions, errors,
  or structured log fields, and that non-training fake AReaL client call counts
  remain zero.

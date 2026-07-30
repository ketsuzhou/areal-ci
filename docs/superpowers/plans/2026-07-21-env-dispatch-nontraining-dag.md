---
change: env-dispatch-nontraining-dag
design-doc: docs/superpowers/specs/2026-07-21-env-dispatch-nontraining-dag-design.md
base-ref: 1dfc83c64835df6fa5fa5585f662ecb9114ad263
---

# Env Dispatch Non-Training DAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require an explicit env-dispatch training mode and return a complete dual-source DAG containing both trained AReaL tensor trajectories and non-trained Multica task-message trajectories.

**Architecture:** Persist dispatch readiness independently in `env_dispatch_run`, then record segments through one terminal/delegation seam that selects `areal_tensor` when an AReaL proxy exists and `task_messages` otherwise. Extend the server/Python DAG contract with explicit source and trainability so mixed DAGs preserve topology while AReaL resolves only trainable tensors.

**Tech Stack:** Go 1.26, PostgreSQL migrations and sqlc queries, testify, Python 3.12, dataclasses, pytest.

## Global Constraints

- `training_mode` is required; no backward-compatible inference is allowed.
- `training_mode=false` must make zero AReaL session/trajectory lifecycle calls.
- Only `train_agent_id` is trainable in `training_mode=true`; all other agents still enter the DAG as non-trainable nodes.
- Provider API keys must never appear in DAG data, responses, errors, or logs.
- Preserve current one-segment-per-task and best-effort recording behavior.
- Do not add dependencies or install Go.

---

### Task 1: Required training-mode request contract

**Files:**
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Test: `multica/server/internal/handler/env_dispatch_test.go`
- Test: `multica/server/internal/service/env_dispatch_test.go`

**Interfaces:**
- Produces: `EnvDispatchRequest.TrainingMode *bool` at the HTTP boundary and `EnvDispatchInput.TrainingMode bool` after validation.
- Produces: validation that false forbids training IDs and true requires `TrainAgentID`.

- [x] **Task 1 / Step 1: Write failing handler and service tests**

Add table cases asserting omitted `training_mode` returns HTTP 400, false plus
training IDs fails validation, true without `train_agent_id` fails, and the two
valid forms reach the service with the exact boolean.

- [x] **Task 1 / Step 2: Run tests and verify RED**

Run: `go test ./server/internal/handler ./server/internal/service -run 'EnvDispatch.*TrainingMode' -count=1`

Expected: FAIL because the request/input has no explicit training-mode contract.

- [x] **Task 1 / Step 3: Implement the minimal request and validation changes**

Use a pointer only in the handler to distinguish an omitted JSON field:

```go
TrainingMode *bool `json:"training_mode"`
```

Reject nil before constructing `EnvDispatchInput`; pass the dereferenced value
as `TrainingMode bool`. In service validation enforce:

```go
if !in.TrainingMode && (in.TrainAgentID != "" || in.CriticAgentID != "") { ... }
if in.TrainingMode && in.TrainAgentID == "" { ... }
```

- [x] **Task 1 / Step 4: Run tests and verify GREEN**

Run the Step 2 command; expected PASS.

### Task 2: Durable dispatch root and readiness

**Files:**
- Create: `multica/server/migrations/204_env_dispatch_run.up.sql`
- Create: `multica/server/migrations/204_env_dispatch_run.down.sql`
- Modify: `multica/server/pkg/db/queries/environment.sql`
- Modify: `multica/server/pkg/db/generated/environment.sql.go`
- Modify: `multica/server/pkg/db/models.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Test: `multica/server/internal/service/env_dispatch_test.go`
- Test: `multica/server/internal/handler/env_dispatch_test.go`

**Interfaces:**
- Produces: `CreateEnvDispatchRun(projectID, workspaceID, trainingMode)`, `BindEnvDispatchRootTask(projectID, rootTaskID)`, and `GetEnvDispatchRootTaskStatus(projectID, workspaceID)` dependency/query seams.
- Consumes: `EnvRollout.LeaderRunID` after the leader task is enqueued.

- [x] **Task 2 / Step 1: Write failing persistence and readiness tests**

Assert every successful rollout persists mode and leader task, `/dag` returns
202 for queued/running roots, and returns assembled data for a completed
non-training root without any `training_dispatch` row.

- [x] **Task 2 / Step 2: Run tests and verify RED**

Run: `go test ./server/internal/handler ./server/internal/service -run 'EnvDispatch.*(Root|Readiness|Dag)' -count=1`

Expected: FAIL because readiness still joins `training_dispatch`.

- [x] **Task 2 / Step 3: Add schema and queries**

Create one row per project:

```sql
CREATE TABLE env_dispatch_run (
  project_id uuid PRIMARY KEY REFERENCES project(id) ON DELETE CASCADE,
  workspace_id uuid NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
  training_mode boolean NOT NULL,
  root_task_id uuid REFERENCES agent_task_queue(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
```

Add create, root-bind, and workspace-scoped status queries; update sqlc output
using the repository's existing generation workflow, without adding tools.

- [x] **Task 2 / Step 4: Wire dispatch persistence and replace readiness lookup**

Create the dispatch row once the project exists, bind `LeaderRunID` after
enqueue, and make `/dag` exclusively query the new root status. Preserve 202,
failed-density, and successful-DAG response shapes.

- [x] **Task 2 / Step 5: Run tests and verify GREEN**

Run the Step 2 command; expected PASS.

### Task 3: Dual-source segment persistence

**Files:**
- Create: `multica/server/migrations/205_interaction_dag_local_trajectory.up.sql`
- Create: `multica/server/migrations/205_interaction_dag_local_trajectory.down.sql`
- Modify: `multica/server/pkg/db/queries/interaction_dag.sql`
- Modify: `multica/server/pkg/db/queries/task_message.sql`
- Modify: `multica/server/pkg/db/generated/interaction_dag.sql.go`
- Modify: `multica/server/pkg/db/generated/task_message.sql.go`
- Modify: `multica/server/pkg/db/models.go`
- Modify: `multica/server/internal/service/interaction_dag.go`
- Test: `multica/server/internal/service/interaction_dag_test.go`

**Interfaces:**
- Produces: `RecordLocalSegmentForEvent(ctx, projectID, agentRunID, issueID, closingEvent, envSnapshot) (string, error)`.
- Produces: `AssembledSegment` fields `TrajectorySource`, `Trainable`, and `Trajectory`; AReaL-only fields are nullable.

- [x] **Task 3 / Step 1: Write failing local-segment tests**

Insert task messages at known sequence numbers and assert the local recorder
upserts session `multica:<task-id>`, snapshots only the requested sequence
range in order, sets `trajectory_source=task_messages`, sets
`trainable=false`, and leaves AReaL fields null. Assert repeated close is
idempotent and runtime provider secrets never enter serialized trajectory.

- [x] **Task 3 / Step 2: Run tests and verify RED**

Run: `go test ./server/internal/service -run 'InteractionDAG.*Local' -count=1`

Expected: FAIL because local segment recording does not exist.

- [x] **Task 3 / Step 3: Add the dual-source schema**

Backfill existing rows, constrain the source, and enforce source-specific
validity:

```sql
ALTER TABLE interaction_dag_segment
  ALTER COLUMN trajectory_id DROP NOT NULL,
  ALTER COLUMN tensor_ref DROP NOT NULL,
  ADD COLUMN trajectory_source text NOT NULL DEFAULT 'areal_tensor',
  ADD COLUMN trainable boolean NOT NULL DEFAULT true,
  ADD COLUMN trajectory jsonb NOT NULL DEFAULT '[]'::jsonb;
```

Add checks requiring non-null AReaL fields only for trainable tensor segments
and null AReaL fields for task-message segments.

- [x] **Task 3 / Step 4: Implement local snapshot recording and assembly**

Serialize an allowlisted message event shape containing sequence, type, tool,
content, input, and output from persisted rows. Compute start/end using the
existing sequence queries, atomically insert the segment and environment
snapshot, and emit the three new contract fields from both source types.

- [x] **Task 3 / Step 5: Run tests and verify GREEN**

Run the Step 2 command plus `go test ./server/internal/service -run InteractionDAG -count=1`; expected PASS.

### Task 4: Record every env-dispatch agent at event seams

**Files:**
- Modify: `multica/server/internal/service/interaction_dag_seams.go`
- Modify: `multica/server/internal/service/task.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Test: `multica/server/internal/service/interaction_dag_gating_test.go`
- Test: `multica/server/internal/service/interaction_dag_seams_test.go`
- Test: `multica/server/internal/service/task_test.go`

**Interfaces:**
- Consumes: env-dispatch project membership and `RecordLocalSegmentForEvent`.
- Produces: a unified close seam choosing AReaL only when `areal_proxy` exists; otherwise local recording for env-dispatch tasks.

- [x] **Task 4 / Step 1: Replace the old non-trained no-op test with failing behavior tests**

Prove a non-trained issue task and channel task record local segments, a mixed
trained/non-trained pair records both sources with an edge, ordinary
non-env-dispatch tasks remain no-ops, and the non-training path makes zero fake
AReaL client calls.

- [x] **Task 4 / Step 2: Run tests and verify RED**

Run: `go test ./server/internal/service -run 'InteractionDAG.*(NonTrain|Mixed|Channel)' -count=1`

Expected: FAIL at the current `extractArealProxyConfig` early return.

- [x] **Task 4 / Step 3: Implement unified project and trajectory-source routing**

Resolve project from issue or chat session. Gate local recording on an
`env_dispatch_run` lookup. Keep the current bridge close/export order for proxy
tasks; use deterministic local session/run mapping and local segment recording
otherwise. Reuse existing edge and one-segment guards.

- [x] **Task 4 / Step 4: Run tests and verify GREEN**

Run the Step 2 command and `go test ./server/internal/service -count=1`; expected PASS.

### Task 5: Parse mixed DAGs safely in AReaL

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py`
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py`
- Modify: `customized_areal/tree_search/agents/segment_dag_trainer.py`
- Test: `customized_areal/tree_search/tests/test_multica_dag_client.py`
- Test: `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`
- Test: `customized_areal/tree_search/tests/test_segment_dag_training_path.py`

**Interfaces:**
- Consumes: server `SegmentSpec` dual-source JSON.
- Produces: nullable tensor fields and explicit local trajectory fields; only `trainable=true` segments reach tensor resolution and shard cleanup.

- [x] **Task 5 / Step 1: Write failing Python contract and resolver tests**

Build a mixed DAG containing one `areal_tensor` and one `task_messages`
segment. Assert strict parsing succeeds, topology retains both segments, only
the trainable tensor ref is resolved, and cleanup never receives the local
segment.

- [x] **Task 5 / Step 2: Run tests and verify RED**

Run: `uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_assembler_ref_resolve.py customized_areal/tree_search/tests/test_segment_dag_training_path.py -q`

Expected: FAIL because current dataclasses require tensor fields and the assembler resolves every segment.

- [x] **Task 5 / Step 3: Implement the Python dual-source contract**

Add typed defaults for `trajectory_source`, `trainable`, and `trajectory`; make
`trajectory_id`/`tensor_ref` optional. Validate trainable segments strictly and
skip resolution/cleanup for non-trainable segments while retaining their DAG
identity and metadata.

- [x] **Task 5 / Step 4: Run tests and verify GREEN**

Run the Step 2 command; expected PASS.

### Task 6: Cross-layer regression verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-21-env-dispatch-nontraining-dag.md` (check completed steps)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: reproducible verification evidence.

- [x] **Task 6 / Step 1: Run focused Go tests**

Run: `go test ./server/internal/service ./server/internal/handler -count=1`

Expected: PASS.

- [x] **Task 6 / Step 2: Run Go static checks**

Run: `go vet ./server/internal/service ./server/internal/handler`

Expected: exit 0.

- [x] **Task 6 / Step 3: Run focused Python tests**

Run the Task 5 Step 2 command; expected PASS.

- [x] **Task 6 / Step 4: Update the repository graph**

Run: `graphify update .`

Expected: graph update completes successfully; dirty graph outputs are retained.

- [x] **Task 6 / Step 5: Review secret and AReaL-call boundaries**

Search changed code and test output for provider API-key fixtures and verify
they occur only in request/setup fixtures, never serialized DAG assertions,
errors, or structured log fields. Verify the non-training fake AReaL client
call counts remain zero.

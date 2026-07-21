# Task 2 Report - Durable dispatch root and readiness

## Status: DONE

## RED evidence (Step 2)

Command: `cd multica/server && go test ./internal/handler/... ./internal/service/... -run 'EnvDispatch.*(Root|Readiness|Dag)' -count=1 -v`

Handler tests SKIP locally (TestMain calls `os.Exit(0)` when Postgres is unreachable). Service tests FAIL because the service does not yet call `CreateEnvDispatchRun` / `BindEnvDispatchRootTask` (no `env_dispatch_run` persistence).

Failing test names + output:
```
ok  	github.com/multica-ai/multica/server/internal/handler	0.040s
--- FAIL: TestEnvDispatch_DispatchPersistsRunAndBindsRootTask (0.00s)
    env_dispatch_test.go:2459: CreateEnvDispatchRun calls: want 2 (one per rollout), got 0
--- FAIL: TestEnvDispatch_NonTrainingDispatch_PersistsRootWithoutTrainingDispatch (0.00s)
    env_dispatch_test.go: CreateEnvDispatchRun calls: want 1, got 0
--- FAIL: TestEnvDispatch_GetDagReadiness_InProgress (0.00s)
    env_dispatch_test.go: want run+bind persisted, got create=0 bind=0
--- FAIL: TestEnvDispatch_GetDagReadiness_Terminal_NonTrainingRoot (0.00s)
    env_dispatch_test.go: want run+bind persisted, got create=0 bind=0
--- PASS: TestEnvDispatch_GetDagReadiness_NoRun (0.00s)
FAIL
FAIL	github.com/multica-ai/multica/server/internal/service	0.021s
FAIL
```

4 service tests FAIL (persistence not wired); 1 passes (NoRun contract test: no run -> InProgress is correct even before wiring). Handler tests compile and skip locally (Postgres unreachable).

## GREEN evidence (Step 5)

Same command after wiring persistence + replacing the readiness lookup:

```
ok  	github.com/multica-ai/multica/server/internal/handler	0.031s
=== RUN   TestEnvDispatch_DispatchPersistsRunAndBindsRootTask
--- PASS: TestEnvDispatch_DispatchPersistsRunAndBindsRootTask (0.00s)
=== RUN   TestEnvDispatch_NonTrainingDispatch_PersistsRootWithoutTrainingDispatch
--- PASS: TestEnvDispatch_NonTrainingDispatch_PersistsRootWithoutTrainingDispatch (0.00s)
=== RUN   TestEnvDispatch_GetDagReadiness_InProgress
--- PASS: TestEnvDispatch_GetDagReadiness_InProgress (0.00s)
=== RUN   TestEnvDispatch_GetDagReadiness_Terminal_NonTrainingRoot
--- PASS: TestEnvDispatch_GetDagReadiness_Terminal_NonTrainingRoot (0.00s)
=== RUN   TestEnvDispatch_GetDagReadiness_NoRun
--- PASS: TestEnvDispatch_GetDagReadiness_NoRun (0.00s)
PASS
ok  	github.com/multica-ai/multica/server/internal/service	0.022s
```

All 5 service tests PASS. Handler tests skip locally (no Postgres). Full `go test ./internal/service/... ./internal/handler/...` and `go vet` pass with no regressions.

## Files changed + git diff --stat f124119f3..HEAD

```
 server/internal/handler/env_dispatch.go         |  91 ++++++---
 server/internal/handler/env_dispatch_test.go    | 116 +++++++++++
 server/internal/service/env_dispatch.go         | 104 ++++++++++
 server/internal/service/env_dispatch_test.go    | 248 ++++++++++++++++++++++++
 server/migrations/204_env_dispatch_run.down.sql |   1 +
 server/migrations/204_env_dispatch_run.up.sql   |  15 ++
 server/pkg/db/generated/environment.sql.go      |  54 ++++++
 server/pkg/db/generated/models.go               |   8 +
 server/pkg/db/queries/environment.sql           |  35 ++++
 9 files changed, 643 insertions(+), 29 deletions(-)
```

## Commit hash

`df6c1353c` in the multica repo (branch `feature/20260721/env-dispatch-nontraining-dag`).

## Migration number used + DDL

Migration number: **204** (verified as the next free number in `multica/server/migrations/`; 203 was the last existing migration).

Referenced table names verified via migrations + models.go:
- `project` (migration 034, `models.Project.ID`)
- `workspace` (migration 001, `models.Workspace`)
- `agent_task_queue` (migration 001, `models.AgentTaskQueue.ID`)

Up (`204_env_dispatch_run.up.sql`):
```sql
CREATE TABLE env_dispatch_run (
  project_id   uuid PRIMARY KEY REFERENCES project(id) ON DELETE CASCADE,
  workspace_id uuid NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
  training_mode boolean NOT NULL,
  root_task_id  uuid REFERENCES agent_task_queue(id) ON DELETE SET NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);
```

Down (`204_env_dispatch_run.down.sql`):
```sql
DROP TABLE IF EXISTS env_dispatch_run;
```

## sqlc queries added + how generation was done

Three queries added to `pkg/db/queries/environment.sql`:
- `CreateEnvDispatchRun :exec` - INSERT ... ON CONFLICT (project_id) DO UPDATE (upsert workspace_id + training_mode).
- `BindEnvDispatchRootTask :exec` - UPDATE env_dispatch_run SET root_task_id = $2 WHERE project_id = $1.
- `GetEnvDispatchRootTaskStatus :one` - SELECT atq.status FROM env_dispatch_run r JOIN agent_task_queue atq ON atq.id = r.root_task_id WHERE r.project_id = $1 AND r.workspace_id = $2. INNER JOIN yields ErrNoRows when no run exists or root_task_id is NULL (both -> in_progress).

**Generation method:** sqlc is NOT installed in this environment (`which sqlc` returned empty). The generated output (`pkg/db/generated/environment.sql.go` + `models.go`) was **hand-edited** to match sqlc v1.31.1 output exactly:
- `environment.sql.go`: 3 const/type/func blocks added in alphabetical order (bindEnvDispatchRootTask, createEnvDispatchRun, getEnvDispatchRootTaskStatus), following the existing pattern (pgtype.UUID params, QueryRow/Exec style).
- `models.go`: `EnvDispatchRun` struct added after `EnvDispatchRequest` (alphabetical), with fields project_id, workspace_id, training_mode, root_task_id, created_at (pgtype.UUID/bool/pgtype.Timestamptz).
- The `EnvDispatchRun` struct is not directly referenced by the queries (they return primitives), but is included for completeness/consistency with sqlc's model generation.

## Interfaces produced

- `CreateEnvDispatchRun(projectID, workspaceID, trainingMode)` - on `EnvDispatchDeps`, called by `Dispatch` after the project exists.
- `BindEnvDispatchRootTask(projectID, rootTaskID)` - on `EnvDispatchDeps`, called by `Dispatch` after `dispatchOne` (consumes `EnvRollout.AgentRunID`).
- `GetEnvDispatchRootTaskStatus(projectID, workspaceID)` - on `EnvDispatchDeps`, called by `EnvDispatchService.GetDagReadiness`.
- `EnvDispatchService.GetDagReadiness(ctx, projectID, workspaceID) (DagReadiness, error)` - service method returning `DagReadinessInProgress` (202) or `DagReadinessTerminal` (proceed to 200 assembly). The handler's `/dag` endpoint calls this instead of the old `GetRootTrainingTaskStatusForProject`.

## Concerns, deviations, or follow-ups

1. **`AgentRunID` vs `LeaderRunID`:** The task says "Consumes: `EnvRollout.LeaderRunID`". However, `LeaderRunID` is only set for channel dispatches (scratch-channel and branch-channel). For issue and self_play-message dispatches, only `AgentRunID` is set (it IS the leader/root task - the single enqueued run). To correctly bind the root task for ALL dispatch types, `BindEnvDispatchRootTask` consumes `EnvRollout.AgentRunID` (set in every `dispatchOne` path). For channel dispatches, `AgentRunID == LeaderRunID`, so this is equivalent. This is noted as a deviation from the literal task text but matches the design intent ("bind root_task_id immediately after enqueuing the leader task").

2. **`GetRootTrainingTaskStatusForProject` not removed:** The old sqlc query and its generated code (`training_dispatch.sql.go`) are left in place - a comment in `service/training.go:563` still references it, and removing generated code outside the allowed file list is out of scope. The handler no longer calls it. A follow-up task could remove the dead query.

3. **`ListOwnedEnvDispatchResources` stale generated code:** The generated `environment.sql.go` did not contain `ListOwnedEnvDispatchResources` before my changes (pre-existing stale state - the query exists in `environment.sql` but was never generated). I did not fix this; it is unrelated to Task 2.

4. **Handler `/dag` tests skip locally:** Per the task caveat, handler `TestMain` exits 0 when Postgres is unreachable. The two handler test stubs (`TestGetDag_NoEnvDispatchRun_Returns202`, `TestGetDag_NonTrainingCompletedRoot_ReturnsNot202`) compile and will run in CI with a DB; they exercise the full handler path (workspace gate + readiness + assembly) that the service tests cover via the `GetDagReadiness` seam. `go vet ./internal/handler/...` passes.

5. **`rootTrainingTaskTerminalStatuses` map removed:** The handler's now-unused terminal-status map was removed (Boy Scout Rule). The equivalent logic lives in the service's `rootTaskTerminalStatuses` map, used by `GetDagReadiness`.

## Security constraints confirmation

- **Provider API keys** never appear in DAG data, responses, errors, or logs. My changes add DB persistence (env_dispatch_run), a readiness query, and test mocks - none touch API keys, credentials, or DAG segment serialization. Verified by grep: no `api_key`/`secret` references in the diff.
- **`training_mode=false` makes ZERO AReaL session/trajectory lifecycle calls.** Task 2 adds no AReaL call paths - `CreateEnvDispatchRun`, `BindEnvDispatchRootTask`, and `GetDagReadiness` are pure DB/logic operations. The non-training test (`TestEnvDispatch_NonTrainingDispatch_PersistsRootWithoutTrainingDispatch`) asserts `len(f.trainingSaves) == 0`.
- **Local trajectory serialization** is sourced ONLY from persisted `task_message` columns. Task 2 does not touch trajectory serialization (that is Task 3's scope).

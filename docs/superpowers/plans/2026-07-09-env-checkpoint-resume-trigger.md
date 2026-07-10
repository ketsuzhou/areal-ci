# env-checkpoint-resume-trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---
change: env-checkpoint-resume-trigger
design-doc: docs/superpowers/specs/2026-07-09-env-checkpoint-resume-trigger-design.md
base-ref: f2256a7aa2cf4f1aad1e410e5519994098901611
multica-base-ref: eb24a5b82 (branch feature/multica-v2-segment-dag-training)
---

**Goal:** Make checkpoint resume re-engage the agent runtime (not just the sandbox container) so a resumed rollout continues its in-flight task from the checkpointed state.

**Architecture:** Capture a `resume_trigger` descriptor (server-side resolved from the project's in-flight task) on `env_checkpoint` at create time; on `ResumeFromCheckpoint`, after resuming sandboxes, execute the trigger via a `ResumeAgentRunner` seam that resets the existing in-flight task row to `queued` and wakes the resumed daemon. Decoupled via small interfaces (mirrors the existing `SandboxInstanceResumer`/`ProjectSnapshotReader` seams) so all logic is unit-testable with fakes (no DB).

**Tech Stack:** Go 1.26 (multica/server) + sqlc v1.31.1 + Python 3.14 (areal). Two repos: multica (nested, Go) and areal (Python + openspec/docs).

## Global Constraints

- TDD: failing test first, then implement, per task. Do not trust GREEN self-reports (project memory).
- multica Go: `cd multica/server && go build ./...` and `go test ./internal/service ./internal/handler -count=1`. sqlc codegen: `cd multica/server && sqlc generate` (Makefile `sqlc` target).
- AReaL Python: `.venv-test/bin/python -m pytest` + `uvx ruff check` (broken `.venv/uv`; use `.venv-test`).
- Commits: multica changes commit in the multica repo (branch `feature/multica-v2-segment-dag-training`); areal changes commit in the areal repo. Conventional Commits, ~72-char subject.
- Boy Scout rule; match existing patterns; no unrelated refactoring; no wildcard imports.
- Design decisions D1-D6 (design doc) are fixed - do not re-litigate.
- `multica/` is untracked from areal's view (nested repo); Go changes are contract+test consistency, not deployed from areal.

## File Structure (multica unless noted)

| File | Responsibility |
|------|----------------|
| `server/migrations/155_env_checkpoint_resume_trigger.up.sql` (+ `.down.sql`) | add `resume_trigger` JSONB column |
| `server/pkg/db/queries/env_checkpoint.sql` | add `resume_trigger` to `CreateEnvCheckpoint` |
| `server/pkg/db/queries/agent.sql` | add `ResetInFlightTaskForResume`, `ListInFlightTasksForProject` |
| `server/internal/service/env_checkpoint.go` | `ResumeTrigger`, `TriggerStatus`, `ResumeAgentRunner` + `InFlightTaskResolver` seams, `Create` resolution, `ResumeFromCheckpoint` execution |
| `server/internal/service/task_resume_runner.go` (new) | `taskResumeRunner` adapter implementing `ResumeAgentRunner` |
| `server/internal/service/env_checkpoint_test.go` | service-level tests (fakes) |
| `server/internal/service/task_resume_runner_test.go` (new) | primitive tests (fakes) |
| `server/internal/handler/env_checkpoint.go` | `ResumeFromCheckpointResponse.TriggerStatus`; surface `resume_trigger` in create response |
| `customized_areal/tree_search/agents/swe_lego_client.py` (areal) | resume surfaces trigger status |
| `customized_areal/tree_search/tests/test_env_dispatch_client.py` (areal) | client test |

Investigation (tasks.md #1) is already answered by the design doc (state machine: `queued->dispatched->running->terminal`; running tasks are not auto-reclaimed, only failed on `running_timeout_secs`; `runtime_id` stable across resume via `ON CONFLICT (workspace_id, daemon_id, provider)`). No new investigation tasks.

---

### Task 1: Migration - resume_trigger column (multica)

**Files:**
- Create: `multica/server/migrations/155_env_checkpoint_resume_trigger.up.sql`
- Create: `multica/server/migrations/155_env_checkpoint_resume_trigger.down.sql`

- [ ] **Step 1: Write the up migration**

`multica/server/migrations/155_env_checkpoint_resume_trigger.up.sql`:
```sql
-- Adds the resume-trigger descriptor captured at checkpoint-create time and
-- executed by ResumeFromCheckpoint to re-engage the agent runtime. Nullable so
-- pre-change checkpoints (NULL) degrade to sandbox-only resume (legacy).
ALTER TABLE env_checkpoint
    ADD COLUMN IF NOT EXISTS resume_trigger jsonb;
```

- [ ] **Step 2: Write the down migration**

`multica/server/migrations/155_env_checkpoint_resume_trigger.down.sql`:
```sql
ALTER TABLE env_checkpoint
    DROP COLUMN IF EXISTS resume_trigger;
```

- [ ] **Step 3: Verify migrations are well-formed**

Run: `cd multica/server && go build ./...`
Expected: builds (migrations are embedded via the existing migration loader; no SQL syntax check at build, but the loader must parse them - if the loader runs in a build-time test, it must pass).

- [ ] **Step 4: Commit**

```bash
cd multica && git add server/migrations/155_env_checkpoint_resume_trigger.up.sql server/migrations/155_env_checkpoint_resume_trigger.down.sql
git commit -m "feat(env-checkpoint-resume-trigger): add resume_trigger column migration"
```

---

### Task 2: sqlc queries - resume_trigger + reset + resolve (multica)

**Files:**
- Modify: `multica/server/pkg/db/queries/env_checkpoint.sql` (`CreateEnvCheckpoint`)
- Modify: `multica/server/pkg/db/queries/agent.sql` (append two queries)
- Regenerate: `multica/server/pkg/db/generated/` via `sqlc generate`

**Interfaces:**
- Produces: `(*Queries).ResetInFlightTaskForResume(ctx, ResetInFlightTaskForResumeParams) (AgentTaskQueue, error)`, `(*Queries).ListInFlightTasksForProject(ctx, ListInFlightTasksForProjectParams) ([]AgentTaskQueue, error)`; `CreateEnvCheckpoint` gains a `ResumeTrigger` param.

- [ ] **Step 1: Add resume_trigger to CreateEnvCheckpoint**

In `multica/server/pkg/db/queries/env_checkpoint.sql`, replace the `CreateEnvCheckpoint` query:
```sql
-- name: CreateEnvCheckpoint :one
INSERT INTO env_checkpoint (
    workspace_id, project_id, event_ref, checkpoint_kind,
    env_id_map, sandbox_refs, db_snapshot, entropy_score,
    save_timeout_ms, save_status, save_error, resume_trigger
) VALUES (
    @workspace_id, @project_id, @event_ref, @checkpoint_kind,
    @env_id_map, @sandbox_refs, @db_snapshot, @entropy_score,
    @save_timeout_ms, @save_status, @save_error, sqlc.narg(resume_trigger)
)
RETURNING *;
```
(`GetEnvCheckpointForWorkspace` / `ListEnvCheckpointsForProject` use `SELECT *` so they auto-include the column; `UpdateEnvCheckpointSaveStatus` is unchanged.)

- [ ] **Step 2: Add ResetInFlightTaskForResume to agent.sql**

Append to `multica/server/pkg/db/queries/agent.sql`:
```sql
-- name: ResetInFlightTaskForResume :one
-- Re-activates a specific in-flight task for resume-from-checkpoint by
-- returning it to `queued` so the resumed runtime's claim loop re-claims it.
-- Only non-terminal (running/dispatched) tasks bound to the given runtime are
-- eligible; a terminal task or a runtime mismatch returns no rows (caller
-- treats that as a stale/unresumable trigger). Preserves context/runtime_id/
-- issue_id/chat_session_id - this is the SAME task row, not a new one.
UPDATE agent_task_queue
SET status = 'queued', started_at = NULL, dispatched_at = NULL, updated_at = now()
WHERE id = @task_id
  AND runtime_id = @runtime_id
  AND status IN ('running', 'dispatched')
RETURNING *;

-- name: ListInFlightTasksForProject :many
-- Resolves a project's in-flight (running/dispatched) agent tasks for
-- resume-trigger capture at checkpoint-create time. Joins via issue or
-- chat_session to the project. project_id is workspace-scoped by nature.
SELECT atq.* FROM agent_task_queue atq
LEFT JOIN issue i ON atq.issue_id = i.id
LEFT JOIN chat_session cs ON atq.chat_session_id = cs.id
WHERE (i.project_id = @project_id OR cs.project_id = @project_id)
  AND atq.status IN ('running', 'dispatched')
ORDER BY atq.created_at ASC;
```

- [ ] **Step 3: Regenerate sqlc**

Run: `cd multica/server && sqlc generate`
Expected: no errors; `pkg/db/generated/env_checkpoint.sql.go` and `agent.sql.go` updated with the new methods/params.

- [ ] **Step 4: Verify build**

Run: `cd multica/server && go build ./...`
Expected: PASS (compiles with new generated code).

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/pkg/db/queries/env_checkpoint.sql server/pkg/db/queries/agent.sql server/pkg/db/generated/
git commit -m "feat(env-checkpoint-resume-trigger): sqlc queries for resume_trigger, reset, resolve"
```

---

### Task 3: Service types + seams + storage round-trip (multica)

**Files:**
- Modify: `multica/server/internal/service/env_checkpoint.go`
- Modify: `multica/server/internal/service/env_checkpoint_test.go` (11 `NewEnvCheckpointService` call sites + fakes + new test)

**Interfaces:**
- Produces: `ResumeTrigger` struct, `TriggerStatus` type, `ResumeAgentRunner` interface, `InFlightTaskResolver` interface; `EnvCheckpointCreateInput.ResumeTrigger`, `EnvCheckpoint.ResumeTrigger`, `ResumeFromCheckpointResult.TriggerStatus`; `NewEnvCheckpointService(..., inFlight InFlightTaskResolver, resumeAgent ResumeAgentRunner)`.

- [ ] **Step 1: Write the failing round-trip test**

In `env_checkpoint_test.go`, add a fake resolver + assert `Create` persists `resume_trigger`:
```go
type fakeInFlightResolver struct {
	triggers []ResumeTrigger
	err      error
}
func (f *fakeInFlightResolver) ListInFlightTasksForProject(_ context.Context, _, _ string) ([]ResumeTrigger, error) {
	return f.triggers, f.err
}

func TestEnvCheckpointCreateResolvesResumeTriggerFromInFlightTask(t *testing.T) {
	repo := newFakeCheckpointRepo()
	saver := &fakeCheckpointSaver{}
	resumer := &fakeCheckpointResumer{}
	snapshot := &fakeProjectSnapshotReader{}
	trigger := ResumeTrigger{TaskID: "t-1", RuntimeID: "r-1", AgentID: "a-1", IssueID: "i-1", ProjectID: "p-1", Kind: "issue"}
	svc := NewEnvCheckpointService(repo, saver, resumer, snapshot, &fakeInFlightResolver{triggers: []ResumeTrigger{trigger}}, nil)

	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "p-1", EventRef: "e", Kind: "always",
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws", NodeID: "n-1"}},
		SaveTimeout: time.Second,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if cp.ResumeTrigger == nil || cp.ResumeTrigger.TaskID != "t-1" {
		t.Fatalf("resume_trigger not resolved: %+v", cp.ResumeTrigger)
	}
	if repo.createCalls[0].in.ResumeTrigger == nil {
		t.Fatal("repo did not receive resume_trigger")
	}
}

func TestEnvCheckpointCreateEmptyResumeTriggerWhenNoInFlightTask(t *testing.T) {
	repo := newFakeCheckpointRepo()
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{}, &fakeProjectSnapshotReader{}, &fakeInFlightResolver{triggers: nil}, nil)
	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "p-1", EventRef: "e", Kind: "always",
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws", NodeID: "n-1"}}, SaveTimeout: time.Second,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if cp.ResumeTrigger != nil {
		t.Fatalf("expected nil resume_trigger, got %+v", cp.ResumeTrigger)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service -run TestEnvCheckpointCreateResumes -count=1`
Expected: FAIL (undefined `ResumeTrigger`, `InFlightTaskResolver`, `NewEnvCheckpointService` arity).

- [ ] **Step 3: Add types + seams to env_checkpoint.go**

Add after the `EnvCheckpointStatus` block:
```go
// ResumeTrigger names the in-flight agent task/runtime to re-engage on resume.
type ResumeTrigger struct {
	TaskID        string `json:"task_id"`
	RuntimeID     string `json:"runtime_id"`
	AgentID       string `json:"agent_id"`
	IssueID       string `json:"issue_id,omitempty"`
	ChatSessionID string `json:"chat_session_id,omitempty"`
	ProjectID     string `json:"project_id"`
	Kind          string `json:"kind"` // "issue" | "chat"
}

// TriggerStatus reports whether ResumeFromCheckpoint re-engaged the agent runtime.
type TriggerStatus string

const (
	TriggerExecuted      TriggerStatus = "executed"
	TriggerSkippedLegacy TriggerStatus = "skipped_legacy"
	TriggerFailed        TriggerStatus = "failed"
)

// InFlightTaskResolver resolves a project's in-flight (running/dispatched)
// agent tasks at checkpoint-create time so the resume-trigger descriptor can be
// populated server-side (the caller does not know multica-internal task ids).
type InFlightTaskResolver interface {
	ListInFlightTasksForProject(ctx context.Context, workspaceID, projectID string) ([]ResumeTrigger, error)
}

// ResumeAgentRunner re-activates an existing in-flight task against its resumed
// agent_runtime. Mirrors SandboxInstanceResumer injection. A nil runner is a
// loud error for non-empty triggers (resume without a runner would no-op).
type ResumeAgentRunner interface {
	ResumeAgentRun(ctx context.Context, trigger ResumeTrigger) error
}
```

Add `ResumeTrigger json.RawMessage` to `EnvCheckpointCreateInput` and `EnvCheckpoint` (use `json.RawMessage`/`[]byte` to match how `DBSnapshot` is handled - nullable; nil when empty).

Add `TriggerStatus TriggerStatus` to `ResumeFromCheckpointResult`.

Add `inFlight InFlightTaskResolver` and `resumeAgent ResumeAgentRunner` fields to `EnvCheckpointService`; update `NewEnvCheckpointService`:
```go
func NewEnvCheckpointService(repo EnvCheckpointRepository, saver SandboxInstanceSaver, resumer SandboxInstanceResumer, snapshot ProjectSnapshotReader, inFlight InFlightTaskResolver, resumeAgent ResumeAgentRunner) *EnvCheckpointService {
	return &EnvCheckpointService{repo: repo, saver: saver, resumer: resumer, snapshot: snapshot, inFlight: inFlight, resumeAgent: resumeAgent}
}
```

Update the `EnvCheckpointRepository.CreateCheckpoint`/fake to carry `ResumeTrigger` (add `ResumeTrigger json.RawMessage` to `EnvCheckpointCreateInput`; the fake `CreateCheckpoint` copies `in.ResumeTrigger` onto the `EnvCheckpoint` and the stored row).

- [ ] **Step 4: Resolve resume_trigger in Create (D5)**

In `Create`, after `in.DBSnapshot = snapshot` and before `s.repo.CreateCheckpoint`, add:
```go
	if s.inFlight != nil {
		triggers, err := s.inFlight.ListInFlightTasksForProject(ctx, in.WorkspaceID, in.ProjectID)
		if err != nil {
			return EnvCheckpoint{}, fmt.Errorf("resolve in-flight tasks: %w", err)
		}
		if len(triggers) > 0 {
			raw, err := json.Marshal(triggers[0]) // v1: single descriptor (group_size=1)
			if err != nil {
				return EnvCheckpoint{}, fmt.Errorf("marshal resume_trigger: %w", err)
			}
			in.ResumeTrigger = raw
		}
	}
```

- [ ] **Step 5: Update all 11 NewEnvCheckpointService call sites in env_checkpoint_test.go**

Add `&fakeInFlightResolver{}, nil` (or a configured resolver) to each existing `NewEnvCheckpointService(repo, saver, resumer, snapshot)` call. Add a `fakeInFlightResolver` (Step 1) + ensure `fakeCheckpointRepo.CreateCheckpoint` copies `in.ResumeTrigger`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/service -run TestEnvCheckpoint -count=1`
Expected: PASS (new round-trip tests + existing tests still pass).

- [ ] **Step 7: Commit**

```bash
cd multica && git add server/internal/service/env_checkpoint.go server/internal/service/env_checkpoint_test.go
git commit -m "feat(env-checkpoint-resume-trigger): resume_trigger types, seams, server-side resolution"
```

---

### Task 4: ResumeAgentRun primitive (multica)

**Files:**
- Create: `multica/server/internal/service/task_resume_runner.go`
- Create: `multica/server/internal/service/task_resume_runner_test.go`

**Interfaces:**
- Consumes: `db.ResetInFlightTaskForResumeParams`/`(*Queries).ResetInFlightTaskForResume` (Task 2), `TaskWakeupNotifier.NotifyTaskAvailable(runtimeID, taskID string)` (task.go:60), `ResumeTrigger` (Task 3).
- Produces: `taskResumeRunner` satisfying `ResumeAgentRunner`; constructor `NewTaskResumeRunner(resetter InFlightTaskResetter, waker TaskWakeupNotifier) ResumeAgentRunner`.

- [ ] **Step 1: Write failing tests**

`multica/server/internal/service/task_resume_runner_test.go`:
```go
package service

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
	db "github.com/multica-ai/multica/server/pkg/db/generated"
)

type fakeInFlightResetter struct {
	task db.AgentTaskQueue
	err  error
	called bool
}
func (f *fakeInFlightResetter) ResetInFlightTaskForResume(_ context.Context, _ db.ResetInFlightTaskForResumeParams) (db.AgentTaskQueue, error) {
	f.called = true
	return f.task, f.err
}
type fakeWaker struct{ notified []string }
func (f *fakeWaker) NotifyTaskAvailable(runtimeID, taskID string) { f.notified = append(f.notified, runtimeID+":"+taskID) }

func TestResumeAgentRunReactivatesExistingTask(t *testing.T) {
	resetter := &fakeInFlightResetter{task: db.AgentTaskQueue{ID: [16]byte{1}, RuntimeID: [16]byte{2}}}
	waker := &fakeWaker{}
	runner := NewTaskResumeRunner(resetter, waker)
	if err := runner.ResumeAgentRun(context.Background(), ResumeTrigger{TaskID: "t", RuntimeID: "r"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !resetter.called {
		t.Fatal("reset not called")
	}
	if len(waker.notified) != 1 {
		t.Fatalf("expected 1 wake, got %d", len(waker.notified))
	}
}

func TestResumeAgentRunRejectsTerminalTask(t *testing.T) {
	resetter := &fakeInFlightResetter{err: pgx.ErrNoRows}
	runner := NewTaskResumeRunner(resetter, &fakeWaker{})
	err := runner.ResumeAgentRun(context.Background(), ResumeTrigger{TaskID: "t", RuntimeID: "r"})
	if !errors.Is(err, ErrTriggerTaskNotResumable) {
		t.Fatalf("expected ErrTriggerTaskNotResumable, got %v", err)
	}
}
```
(Note: exact `db.AgentTaskQueue.ID`/`RuntimeID` zero-value construction - use `pgtype.UUID` if the generated type uses pgtype; adjust to match `generated/agent.sql.go`. The `util.UUIDToString` helper serializes them.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service -run TestResumeAgentRun -count=1`
Expected: FAIL (undefined `NewTaskResumeRunner`, `InFlightTaskResetter`, `ErrTriggerTaskNotResumable`).

- [ ] **Step 3: Implement the primitive**

`multica/server/internal/service/task_resume_runner.go`:
```go
package service

import (
	"context"
	"errors"
	"fmt"

	db "github.com/multica-ai/multica/server/pkg/db/generated"
	"github.com/multica-ai/multica/server/internal/util"
)

// ErrTriggerTaskNotResumable means the trigger's task is terminal, missing, or
// bound to a different runtime - resume must not double-run it.
var ErrTriggerTaskNotResumable = errors.New("trigger_task_not_resumable")

// InFlightTaskResetter is the sqlc seam the primitive resets a task through.
type InFlightTaskResetter interface {
	ResetInFlightTaskForResume(ctx context.Context, arg db.ResetInFlightTaskForResumeParams) (db.AgentTaskQueue, error)
}

// taskResumeRunner implements ResumeAgentRunner by resetting the existing
// in-flight task row to `queued` and waking the resumed daemon. It does NOT
// create a new task row (continuity preserved) and does NOT send a chat message.
type taskResumeRunner struct {
	resetter InFlightTaskResetter
	waker    TaskWakeupNotifier
}

func NewTaskResumeRunner(resetter InFlightTaskResetter, waker TaskWakeupNotifier) ResumeAgentRunner {
	return &taskResumeRunner{resetter: resetter, waker: waker}
}

func (r *taskResumeRunner) ResumeAgentRun(ctx context.Context, trigger ResumeTrigger) error {
	taskID, err := util.ParseUUID(trigger.TaskID) // use the existing util UUID parse helper
	if err != nil {
		return fmt.Errorf("invalid task_id: %w", err)
	}
	runtimeID, err := util.ParseUUID(trigger.RuntimeID)
	if err != nil {
		return fmt.Errorf("invalid runtime_id: %w", err)
	}
	task, err := r.resetter.ResetInFlightTaskForResume(ctx, db.ResetInFlightTaskForResumeParams{
		TaskID:    taskID,
		RuntimeID: runtimeID,
	})
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrTriggerTaskNotResumable // terminal, missing, or runtime mismatch
	}
	if err != nil {
		return fmt.Errorf("reset in-flight task: %w", err)
	}
	r.waker.NotifyTaskAvailable(util.UUIDToString(task.RuntimeID), util.UUIDToString(task.ID))
	return nil
}
```
(Adjust UUID helpers to match `util` package exports - verify `util.ParseUUID`/`util.UUIDToString` exist; if `pgtype.UUID` is used, use its scan/Value. Import `github.com/jackc/pgx/v5` and `github.com/multica-ai/multica/server/internal/util`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/service -run TestResumeAgentRun -count=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task_resume_runner.go server/internal/service/task_resume_runner_test.go
git commit -m "feat(env-checkpoint-resume-trigger): resume-agent-run primitive"
```

---

### Task 5: ResumeFromCheckpoint trigger execution (multica)

**Files:**
- Modify: `multica/server/internal/service/env_checkpoint.go` (`ResumeFromCheckpoint`)
- Modify: `multica/server/internal/service/env_checkpoint_test.go`

**Interfaces:**
- Consumes: `ResumeAgentRunner` (Task 3/4), `ResumeTrigger`, `TriggerStatus` (Task 3), `ErrTriggerTaskNotResumable` (Task 4).

- [ ] **Step 1: Write failing tests**

In `env_checkpoint_test.go`:
```go
type fakeResumeAgentRunner struct {
	calledWith *ResumeTrigger
	err        error
}
func (f *fakeResumeAgentRunner) ResumeAgentRun(_ context.Context, t ResumeTrigger) error {
	f.calledWith = &t
	return f.err
}

func TestResumeFromCheckpointExecutesTriggerAfterSandboxResume(t *testing.T) {
	repo := newFakeCheckpointRepo()
	triggerJSON := []byte(`{"task_id":"t-1","runtime_id":"r-1","agent_id":"a-1","project_id":"p-1","kind":"issue"}`)
	repo.checkpoints["cp-1"] = EnvCheckpoint{ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws", NodeID: "n-1"}}, ResumeTrigger: triggerJSON}
	runner := &fakeResumeAgentRunner{}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{}, &fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, runner)
	res, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u")
	if err != nil { t.Fatalf("resume: %v", err) }
	if runner.calledWith == nil || runner.calledWith.TaskID != "t-1" { t.Fatal("trigger not executed") }
	if res.TriggerStatus != TriggerExecuted { t.Fatalf("status=%v want executed", res.TriggerStatus) }
}

func TestResumeFromCheckpointSkipsLegacyEmptyTrigger(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws", NodeID: "n-1"}}, ResumeTrigger: nil}
	runner := &fakeResumeAgentRunner{}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{}, &fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, runner)
	res, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u")
	if err != nil { t.Fatalf("resume: %v", err) }
	if runner.calledWith != nil { t.Fatal("legacy trigger should not execute") }
	if res.TriggerStatus != TriggerSkippedLegacy { t.Fatalf("status=%v want skipped_legacy", res.TriggerStatus) }
}

func TestResumeFromCheckpointTriggerFailureIsPartialResume(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws", NodeID: "n-1"}}, ResumeTrigger: []byte(`{"task_id":"t-1","runtime_id":"r-1","kind":"issue"}`)}
	runner := &fakeResumeAgentRunner{err: ErrTriggerTaskNotResumable}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{}, &fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, runner)
	res, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u")
	if err == nil { t.Fatal("expected partial-resume error") }
	if res.TriggerStatus != TriggerFailed { t.Fatalf("status=%v want failed", res.TriggerStatus) }
}

func TestResumeFromCheckpointRejectsTriggerWithoutRunner(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws", NodeID: "n-1"}}, ResumeTrigger: []byte(`{"task_id":"t-1","kind":"issue"}`)}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{}, &fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, nil)
	if _, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u"); err == nil {
		t.Fatal("expected error for non-empty trigger with nil runner")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service -run TestResumeFromCheckpoint -count=1`
Expected: FAIL (TriggerStatus not set; no trigger execution).

- [ ] **Step 3: Implement trigger execution in ResumeFromCheckpoint**

In `env_checkpoint.go`, update `ResumeFromCheckpoint` to execute the trigger after the sandbox resume loop and set `TriggerStatus`. After the `for _, ref := range cp.SandboxRefs { ... s.resumer.Resume ... }` loop, before the return:
```go
	result := ResumeFromCheckpointResult{
		CheckpointID: cp.ID, ProjectID: cp.ProjectID, EnvIDMap: cp.EnvIDMap,
		SandboxRefs: cp.SandboxRefs, RolloutHandle: fmt.Sprintf("resume:%s", cp.ID),
	}
	if len(cp.ResumeTrigger) == 0 {
		result.TriggerStatus = TriggerSkippedLegacy
		return result, nil
	}
	if s.resumeAgent == nil {
		return ResumeFromCheckpoint{}, fmt.Errorf("validation_failed: non-empty resume_trigger but no resume agent runner configured")
	}
	var trigger ResumeTrigger
	if err := json.Unmarshal(cp.ResumeTrigger, &trigger); err != nil {
		result.TriggerStatus = TriggerFailed
		return result, fmt.Errorf("unmarshal resume_trigger: %w", err)
	}
	if err := s.resumeAgent.ResumeAgentRun(ctx, trigger); err != nil {
		result.TriggerStatus = TriggerFailed
		return result, fmt.Errorf("resume agent run: %w", err)
	}
	result.TriggerStatus = TriggerExecuted
	return result, nil
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/service -run TestResumeFromCheckpoint -count=1`
Expected: PASS (all 4 new + existing resume tests).

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/env_checkpoint.go server/internal/service/env_checkpoint_test.go
git commit -m "feat(env-checkpoint-resume-trigger): execute resume trigger in ResumeFromCheckpoint"
```

---

### Task 6: Handler surfaces TriggerStatus + resume_trigger (multica)

**Files:**
- Modify: `multica/server/internal/handler/env_checkpoint.go`
- Modify: `multica/server/internal/handler/env_checkpoint_test.go` (if it asserts response shape)

**Interfaces:**
- Consumes: `ResumeFromCheckpointResult.TriggerStatus` (Task 5), `EnvCheckpoint.ResumeTrigger` (Task 3).

- [ ] **Step 1: Add TriggerStatus to the resume response**

In `handler/env_checkpoint.go`, add to `ResumeFromCheckpointResponse`:
```go
	TriggerStatus string `json:"trigger_status"`
```
and set it in `ResumeEnvCheckpoint` from `res.TriggerStatus`.

- [ ] **Step 2: Surface resume_trigger in the create/get/list response**

`EnvCheckpointResponse` already has `DBSnapshot`; add `ResumeTrigger json.RawMessage \`json:"resume_trigger,omitempty"\`` and set it in `mapEnvCheckpointResponse` from `cp.ResumeTrigger`. (Create request does NOT accept resume_trigger - it is server-resolved.)

- [ ] **Step 3: Verify handler builds + tests**

Run: `cd multica/server && go build ./... && go test ./internal/handler -run EnvCheckpoint -count=1`
Expected: PASS. If a handler test asserts exact JSON body, update it; the existing tests check status, not exact body (per plan-env-dispatch notes), so they should be unaffected.

- [ ] **Step 4: Commit**

```bash
cd multica && git add server/internal/handler/env_checkpoint.go server/internal/handler/env_checkpoint_test.go
git commit -m "feat(env-checkpoint-resume-trigger): surface trigger_status + resume_trigger in API"
```

---

### Task 7: AReaL client surfaces trigger status (areal)

**Files:**
- Modify: `customized_areal/tree_search/agents/swe_lego_client.py`
- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`

**Interfaces:**
- Consumes: the existing `resume_from_checkpoint` client helper; the multica `ResumeFromCheckpointResponse` now includes `trigger_status`.

- [ ] **Step 1: Write the failing test**

In `test_env_dispatch_client.py`, add (mirror the existing resume-from-checkpoint test's fake transport):
```python
def test_resume_from_checkpoint_returns_trigger_status(fake_dispatch_http):
    # fake_dispatch_http returns {"rollout_handle": "resume:cp-1", "trigger_status": "executed", ...}
    result = client.resume_from_checkpoint(checkpoint_id="cp-1")
    assert result.rollout_handle == "resume:cp-1"
    assert result.trigger_status == "executed"
```
(Use the existing fake/transport pattern in this test file; the exact helper name follows the file's conventions.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -k resume_from_checkpoint -x`
Expected: FAIL (`trigger_status` not parsed/returned).

- [ ] **Step 3: Implement - parse trigger_status**

In `swe_lego_client.py`, update the `resume_from_checkpoint` return type to carry `trigger_status` (parse `body["trigger_status"]`; default `""`/`None` when absent for backward compat). Preserve existing `rollout_handle` behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -k resume_from_checkpoint -x`
Expected: PASS.

- [ ] **Step 5: Lint + commit (areal repo)**

```bash
uvx ruff check customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py
git add customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py
git commit -m "feat(env-checkpoint-resume-trigger): areal client surfaces trigger_status"
```

---

### Task 8: Check off tasks.md + final verification (both repos)

**Files:**
- Modify: `openspec/changes/env-checkpoint-resume-trigger/tasks.md` (areal) - check off all groups.
- Modify: `openspec/changes/env-checkpoint-resume-trigger/specs/env-checkpoint-resume-trigger/spec.md` - already patched (D5 scenarios); re-validate.

- [ ] **Step 1: Check off completed tasks in tasks.md**

Mark groups 1-7 (1 investigation is design-doc-answered; 2 storage; 3 primitive; 4 capture+execution; 5 areal client; 6 specs; 7 verification) `- [ ]` -> `- [x]` as completed.

- [ ] **Step 2: Run full multica scoped tests**

Run: `cd multica/server && go build ./... && go test ./internal/service ./internal/handler -count=1`
Expected: PASS. (Note any pre-existing DB-dependent `cmd/server` failures as unrelated.)

- [ ] **Step 3: Run AReaL scoped tests + lint**

Run: `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_env_dispatch_client.py` and `uvx ruff check customized_areal/tree_search/`
Expected: PASS.

- [ ] **Step 4: Validate OpenSpec**

Run: `openspec validate env-checkpoint-resume-trigger --strict`
Expected: valid.

- [ ] **Step 5: Commit tasks.md checkoff (areal)**

```bash
git add openspec/changes/env-checkpoint-resume-trigger/tasks.md
git commit -m "docs(env-checkpoint-resume-trigger): check off completed tasks"
```

---

## Self-Review

**Spec coverage:** "Resume-trigger captured at checkpoint create" -> Tasks 1-3 (column, query, Create resolution). "Resume-agent-run primitive re-engages in-flight task" -> Task 4 (+ terminal/mismatch rejection). "Resume-from-checkpoint executes trigger after sandbox resume" -> Task 5 (+ partial-resume). "Legacy checkpoint without trigger degrades" -> Task 5 (skipped_legacy). "Trigger descriptor resolved server-side" + "No in-flight task yields empty trigger" -> Task 3 (fakeInFlightResolver tests). AReaL client integration -> Task 7.

**Type consistency:** `ResumeTrigger`, `TriggerStatus`, `ResumeAgentRunner`, `InFlightTaskResolver`, `InFlightTaskResetter`, `ErrTriggerTaskNotResumable` defined once and reused. `NewEnvCheckpointService` arity updated everywhere (11 test sites in Task 3 Step 5). `ResetInFlightTaskForResume`/`ListInFlightTasksForProject` from Task 2 consumed in Tasks 3/4.

**Open verifications (engineer must confirm during impl):** `util.ParseUUID`/`util.UUIDToString` exact names; whether `db.AgentTaskQueue` ID/RuntimeID are `pgtype.UUID` or `[16]byte` (adjust zero-values in Task 4 test); `chat_session.project_id` column exists for the `ListInFlightTasksForProject` join; `env_checkpoint_test.go` existing tests' `NewEnvCheckpointService` arity (all 11). All resolved via `go build`/`go test` in-task.

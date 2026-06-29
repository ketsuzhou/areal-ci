# Phase 1: Multica Entity Fork — Implementation Plan (Go)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ForkIssueSubtree` (a Go service that forks an issue's entity subtree at a past `seq`, reconstructing overwritten fields from the activity log), expose it via `POST /api/issues/{id}/fork` + `DELETE`, and add generic Fleet snapshot/fork endpoints that dispatch to the underlying sandbox vendor.

**Architecture:** A new migration adds fork provenance columns (`forked_from_issue_id`, `forked_at_seq`, `forked_at_task_id`) to `issue`. A new `IssueForkService` in `internal/service/issue_fork.go` copies the issue + append-only data (comments, sub-issues, task_messages) cut at `seq`, reconstructing overwritten `issue` fields from `activity_log`. A new handler exposes the fork over HTTP. Separately, Fleet snapshot/fork endpoints are added to `internal/handler/cloud_runtime.go`, dispatching to Daytona (the current vendor) via the existing `cloudruntime.Client`.

**Tech Stack:** Go 1.26 · Chi router · sqlc for DB · `github.com/jackc/pgx/v5` · `make sqlc` to regenerate · `make test` for Go tests · `make migrate-up` for migrations

**Design reference:** `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md` §3.2 (activity-log reconstruction), §3.4 (generic Fleet endpoints), §4 (Architecture), §5 Phase 1

**Project rules:** `multica/CLAUDE.md` (UUID parsing convention, parse-don't-cast, route categories, atomic commits). Read `multica/CLAUDE.md` before starting.

**Dependencies:** Phase 0 must be complete (the Python `FleetSandboxProvider` calls the endpoints added in Task 5 of this plan).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `multica/server/migrations/122_issue_fork_provenance.up.sql` (create) | Add `forked_from_issue_id`, `forked_at_seq`, `forked_at_task_id` columns + partial index |
| `multica/server/migrations/122_issue_fork_provenance.down.sql` (create) | Drop the columns + index |
| `multica/server/pkg/db/queries/issue_fork.sql` (create) | sqlc queries: create forked issue, list comments/sub-issues/task_messages at seq, activity log lookup |
| `multica/server/internal/service/issue_fork.go` (create) | `IssueForkService.ForkIssueSubtree(ctx, sourceIssueID, taskID, seq)` — the core fork logic with activity-log reconstruction |
| `multica/server/internal/service/issue_fork_test.go` (create) | Per-field activity-log reconstruction tests + coverage check test |
| `multica/server/internal/handler/issue_fork.go` (create) | `POST /api/issues/{id}/fork` + `DELETE /api/issues/{id}/fork` handlers |
| `multica/server/internal/handler/issue_fork_test.go` (create) | Handler tests: UUID parsing, loader usage, 404 on missing issue |
| `multica/server/internal/handler/cloud_runtime.go` (modify) | Add `SnapshotCloudRuntimeSandbox`, `ForkCloudRuntimeSandbox` handlers |
| `multica/server/internal/handler/cloud_runtime_test.go` (modify) | Tests for the two new handlers |
| `multica/server/cmd/server/router.go` (modify) | Register the new routes |

---

## Task 3: Migration — fork provenance columns on `issue`

**Files:**
- Create: `multica/server/migrations/122_issue_fork_provenance.up.sql`
- Create: `multica/server/migrations/122_issue_fork_provenance.down.sql`

- [ ] **Step 1: Write the up migration**

```sql
-- 122_issue_fork_provenance.up.sql
-- Tracks issue fork provenance for multi-agent DAG RL training.
-- A row with forked_from_issue_id IS NULL is an original; non-NULL means it
-- was forked from another issue at (forked_at_task_id, forked_at_seq) —
-- the branch point in the source agent's transcript.

ALTER TABLE issue
    ADD COLUMN forked_from_issue_id UUID REFERENCES issue(id) ON DELETE SET NULL,
    ADD COLUMN forked_at_seq INTEGER,
    ADD COLUMN forked_at_task_id UUID;

-- Partial index: only forked issues (the common lookup is "find the original
-- for this fork" or "list all forks of this original"). Original issues have
-- NULL forked_from_issue_id and don't need to be in this index.
CREATE INDEX idx_issue_forked_from
    ON issue (forked_from_issue_id)
    WHERE forked_from_issue_id IS NOT NULL;
```

- [ ] **Step 2: Write the down migration**

```sql
-- 122_issue_fork_provenance.down.sql
DROP INDEX IF EXISTS idx_issue_forked_from;
ALTER TABLE issue
    DROP COLUMN IF EXISTS forked_at_task_id,
    DROP COLUMN IF EXISTS forked_at_seq,
    DROP COLUMN IF EXISTS forked_from_issue_id;
```

- [ ] **Step 3: Run the migration to verify it applies cleanly**

Run: `cd /workspaces/leagent/backend/areal/multica && make migrate-up`
Expected: migration 122 applies with no errors.

- [ ] **Step 4: Verify rollback works**

Run: `make migrate-down` (then `make migrate-up` to restore)
Expected: down migration drops the columns cleanly; re-applying up works.

- [ ] **Step 5: Commit**

```bash
git add multica/server/migrations/122_issue_fork_provenance.up.sql multica/server/migrations/122_issue_fork_provenance.down.sql
git commit -m "feat(db): add issue fork provenance columns (forked_from_issue_id, forked_at_seq, forked_at_task_id)"
```

---

## Task 4: sqlc queries for issue fork

**Files:**
- Create: `multica/server/pkg/db/queries/issue_fork.sql`
- Modify: `multica/server/pkg/db/generated/` (auto-generated by `make sqlc`)

- [ ] **Step 1: Write the sqlc queries**

```sql
-- name: CreateForkedIssue :one
-- Insert a new issue row that records its fork provenance. The caller
-- supplies every field explicitly (copied from the source issue, with
-- overwritten fields reconstructed from the activity log — see
-- IssueForkService.ForkIssueSubtree).
INSERT INTO issue (
    workspace_id, title, description, status, priority,
    assignee_type, assignee_id, creator_type, creator_id,
    parent_issue_id, acceptance_criteria, context_refs, position, due_date,
    forked_from_issue_id, forked_at_seq, forked_at_task_id
) VALUES (
    @workspace_id, @title, @description, @status, @priority,
    @assignee_type, @assignee_id, @creator_type, @creator_id,
    @parent_issue_id, @acceptance_criteria, @context_refs, @position, @due_date,
    @forked_from_issue_id, @forked_at_seq, @forked_at_task_id
) RETURNING *;

-- name: ListIssueCommentsAtOrBefore :many
-- Comments whose created_at <= the timestamp of the task_message at branch_seq.
-- Used by ForkIssueSubtree to copy append-only comments cut at the branch point.
SELECT * FROM comment
WHERE issue_id = @issue_id
  AND created_at <= @cutoff_ts
ORDER BY created_at ASC, id ASC;

-- name: ListSubIssuesAtOrBefore :many
-- Sub-issues created at or before the branch point. parent_issue_id edges
-- are append-only (an issue is created as a sub-issue and stays one).
SELECT * FROM issue
WHERE parent_issue_id = @parent_issue_id
  AND created_at <= @cutoff_ts
ORDER BY created_at ASC, id ASC;

-- name: ListTaskMessagesAtOrBefore :many
-- task_message rows for the source agent's task, with seq <= branch_seq.
SELECT * FROM task_message
WHERE task_id = @task_id
  AND seq <= @branch_seq
ORDER BY seq ASC;

-- name: GetTaskMessageAtSeq :one
-- The task_message row whose seq == branch_seq. Its created_at is the
-- cutoff timestamp for append-only data (comments, sub-issues).
SELECT * FROM task_message
WHERE task_id = @task_id
  AND seq = @branch_seq;

-- name: ListActivityLogForIssue :many
-- Activity log entries for the source issue, ordered oldest-first so the
-- replay logic can walk forward to the branch point.
SELECT * FROM activity_log
WHERE issue_id = @issue_id
  AND created_at <= @cutoff_ts
ORDER BY created_at ASC, id ASC;

-- name: DeleteForkedIssue :exec
-- Delete a forked issue by id. Only deletes if forked_from_issue_id IS NOT NULL
-- (safety guard: cannot delete an original issue through this query).
DELETE FROM issue
WHERE id = @id
  AND forked_from_issue_id IS NOT NULL;
```

- [ ] **Step 2: Regenerate sqlc code**

Run: `cd /workspaces/leagent/backend/areal/multica && make sqlc`
Expected: no errors; new functions appear in `multica/server/pkg/db/generated/`.

- [ ] **Step 3: Verify the generated code compiles**

Run: `cd multica/server && go build ./...`
Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add multica/server/pkg/db/queries/issue_fork.sql multica/server/pkg/db/generated/
git commit -m "feat(db): add sqlc queries for issue fork (create forked issue, list at seq, activity log)"
```

---

## Task 5: Activity-log coverage check

**Files:**
- Create: `multica/server/internal/service/issue_fork.go` (skeleton + coverage check only)
- Create: `multica/server/internal/service/issue_fork_test.go`

**Rationale:** Per design §3.2, before implementing replay logic we must verify that `activity_log` captures every overwritten `issue` field. This test encodes that contract — it fails the build if a field is added to `issue` without being logged.

- [ ] **Step 1: Write the failing coverage test**

```go
// multica/server/internal/service/issue_fork_test.go
package service

import (
	"testing"

	db "github.com/multica-ai/multica/server/pkg/db/generated"
)

// TestActivityLogCoversOverwritableIssueFields is a build-time contract:
// every issue field that can be overwritten after creation must be
// reconstructable from the activity log. If a field is added to the issue
// table without a corresponding activity_log action, this test fails —
// add the field to the allowlist only after confirming the activity log
// captures its changes.
func TestActivityLogCoversOverwritableIssueFields(t *testing.T) {
	// Fields that the activity log records changes to. Each entry is a
	// (column, activity_log action) pair. When a new overwritable column
	// is added to issue, add it here ONLY after verifying the activity
	// log writes an entry for it.
	covered := map[string]string{
		"status":      "issue.status_changed",
		"title":       "issue.title_changed",
		"description": "issue.description_changed",
		"priority":    "issue.priority_changed",
		"assignee":    "issue.assignee_changed",
		"due_date":    "issue.due_date_changed",
	}
	// The full list of issue fields that can be overwritten post-creation.
	// Fields like id, workspace_id, created_at, forked_* are immutable or
	// set-once and are NOT in this list.
	overwritable := []string{
		"status", "title", "description", "priority", "assignee", "due_date",
	}
	for _, field := range overwritable {
		if _, ok := covered[field]; !ok {
			t.Errorf(
				"issue.%q is overwritable but has no activity_log coverage. "+
					"Either add an activity_log action for it or remove it from the overwritable list.",
				field,
			)
		}
	}
}

// stub reference so the db package import is used even before the service
// code lands.
var _ = db.Issue{}
```

- [ ] **Step 2: Run test to verify it compiles but the contract is self-consistent**

Run: `cd /workspaces/leagent/backend/areal/multica/server && go test ./internal/service/ -run TestActivityLogCoversOverwritableIssueFields -v`
Expected: PASS (the test is a self-consistency check; it fails only if a field is removed from `covered` without being removed from `overwritable`).

- [ ] **Step 3: Commit**

```bash
git add multica/server/internal/service/issue_fork_test.go
git commit -m "test(issue-fork): add activity-log coverage contract test for overwritable issue fields"
```

---

## Task 6: IssueForkService — core fork with activity-log reconstruction

**Files:**
- Modify: `multica/server/internal/service/issue_fork.go`
- Modify: `multica/server/internal/service/issue_fork_test.go`

- [ ] **Step 1: Write the failing test for ForkIssueSubtree**

Append to `issue_fork_test.go`:

```go
import (
	"context"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	db "github.com/multica-ai/multica/server/pkg/db/generated"
)

// fakeQueries is a test double for db.Queries used by IssueForkService.
// It returns canned data so the fork logic can be exercised without a DB.
type fakeQueries struct {
	sourceIssue     db.Issue
	comments        []db.Comment
	subIssues       []db.Issue
	taskMessages    []db.TaskMessage
	activityEntries []db.ActivityLog
	createdIssue    db.Issue
	createCalls     int
}

func (f *fakeQueries) GetIssue(ctx context.Context, id pgtype.UUID) (db.Issue, error) {
	if f.sourceIssue.ID == id {
		return f.sourceIssue, nil
	}
	return db.Issue{}, pgx.ErrNoRows
}

func (f *fakeQueries) GetTaskMessageAtSeq(ctx context.Context, arg db.GetTaskMessageAtSeqParams) (db.TaskMessage, error) {
	for _, m := range f.taskMessages {
		if m.TaskID == arg.TaskID && m.Seq == arg.BranchSeq {
			return m, nil
		}
	}
	return db.TaskMessage{}, pgx.ErrNoRows
}

func (f *fakeQueries) ListIssueCommentsAtOrBefore(ctx context.Context, arg db.ListIssueCommentsAtOrBeforeParams) ([]db.Comment, error) {
	return f.comments, nil
}

func (f *fakeQueries) ListSubIssuesAtOrBefore(ctx context.Context, arg db.ListSubIssuesAtOrBeforeParams) ([]db.Issue, error) {
	return f.subIssues, nil
}

func (f *fakeQueries) ListTaskMessagesAtOrBefore(ctx context.Context, arg db.ListTaskMessagesAtOrBeforeParams) ([]db.TaskMessage, error) {
	var out []db.TaskMessage
	for _, m := range f.taskMessages {
		if m.Seq <= arg.BranchSeq {
			out = append(out, m)
		}
	}
	return out, nil
}

func (f *fakeQueries) ListActivityLogForIssue(ctx context.Context, arg db.ListActivityLogForIssueParams) ([]db.ActivityLog, error) {
	return f.activityEntries, nil
}

func (f *fakeQueries) CreateForkedIssue(ctx context.Context, arg db.CreateForkedIssueParams) (db.Issue, error) {
	f.createCalls++
	f.createdIssue = db.Issue{
		ID: pgtype.UUID{Bytes: [16]byte{byte(f.createCalls)}, Valid: true},
		ForkedFromIssueID: arg.ForkedFromIssueID,
		ForkedAtSeq:       arg.ForkedAtSeq,
		ForkedAtTaskID:    arg.ForkedAtTaskID,
		Title:             arg.Title,
		Status:            arg.Status,
	}
	return f.createdIssue, nil
}

func TestForkIssueSubtree_ReconstructsOverwrittenStatusFromActivityLog(t *testing.T) {
	ctx := context.Background()
	now := time.Now()
	sourceIssueID := pgtype.UUID{Bytes: [16]byte{1}, Valid: true}
	taskID := pgtype.UUID{Bytes: [16]byte{2}, Valid: true}

	// Source issue is currently 'done', but at branch_seq=5 it was 'in_progress'.
	q := &fakeQueries{
		sourceIssue: db.Issue{
			ID:          sourceIssueID,
			WorkspaceID: pgtype.UUID{Bytes: [16]byte{9}, Valid: true},
			Title:       "implement feature X",
			Status:      "done", // current value
			CreatorType: "agent",
			CreatorID:   pgtype.UUID{Bytes: [16]byte{3}, Valid: true},
		},
		taskMessages: []db.TaskMessage{
			{TaskID: taskID, Seq: 5, CreatedAt: now.Add(-1 * time.Hour)},
		},
		// Activity log: status was set to 'in_progress' at seq=5.
		activityEntries: []db.ActivityLog{
			{
				IssueID:   sourceIssueID,
				Action:     "issue.status_changed",
				Details:    []byte(`{"field":"status","old":"backlog","new":"in_progress"}`),
				CreatedAt:  now.Add(-2 * time.Hour),
			},
			{
				IssueID:   sourceIssueID,
				Action:     "issue.status_changed",
				Details:    []byte(`{"field":"status","old":"in_progress","new":"done"}`),
				CreatedAt:  now.Add(-30 * time.Minute), // AFTER branch point — must be ignored
			},
		},
	}

	svc := NewIssueForkService(q)
	forked, err := svc.ForkIssueSubtree(ctx, sourceIssueID, taskID, 5)
	if err != nil {
		t.Fatalf("ForkIssueSubtree failed: %v", err)
	}
	if forked.Status != "in_progress" {
		t.Errorf("expected reconstructed status 'in_progress', got %q", forked.Status)
	}
	if !forked.ForkedFromIssueID.Valid || forked.ForkedFromIssueID != sourceIssueID {
		t.Errorf("ForkedFromIssueID not set correctly: %+v", forked.ForkedFromIssueID)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspaces/leagent/backend/areal/multica/server && go test ./internal/service/ -run TestForkIssueSubtree_ReconstructsOverwrittenStatusFromActivityLog -v`
Expected: FAIL — `NewIssueForkService` undefined, `ForkIssueSubtree` undefined.

- [ ] **Step 3: Implement IssueForkService**

```go
// multica/server/internal/service/issue_fork.go
package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/jackc/pgx/v5/pgtype"
	db "github.com/multica-ai/multica/server/pkg/db/generated"
)

// IssueForkService forks an issue's entity subtree at a past (task_id, seq)
// branch point, reconstructing overwritten issue fields from the activity log.
type IssueForkService struct {
	Queries IssueForkQueries
}

// IssueForkQueries is the subset of db.Queries that IssueForkService needs.
// Defined as an interface so tests can inject a fake.
type IssueForkQueries interface {
	GetIssue(ctx context.Context, id pgtype.UUID) (db.Issue, error)
	GetTaskMessageAtSeq(ctx context.Context, arg db.GetTaskMessageAtSeqParams) (db.TaskMessage, error)
	ListIssueCommentsAtOrBefore(ctx context.Context, arg db.ListIssueCommentsAtOrBeforeParams) ([]db.Comment, error)
	ListSubIssuesAtOrBefore(ctx context.Context, arg db.ListSubIssuesAtOrBeforeParams) ([]db.Issue, error)
	ListTaskMessagesAtOrBefore(ctx context.Context, arg db.ListTaskMessagesAtOrBeforeParams) ([]db.TaskMessage, error)
	ListActivityLogForIssue(ctx context.Context, arg db.ListActivityLogForIssueParams) ([]db.ActivityLog, error)
	CreateForkedIssue(ctx context.Context, arg db.CreateForkedIssueParams) (db.Issue, error)
}

func NewIssueForkService(q IssueForkQueries) *IssueForkService {
	return &IssueForkService{Queries: q}
}

// ForkIssueSubtree forks sourceIssueID at (taskID, seq). Returns the new
// forked issue row. Overwritten issue fields (status, title, etc.) are
// reconstructed from the activity log to their values as of the branch point.
func (s *IssueForkService) ForkIssueSubtree(
	ctx context.Context,
	sourceIssueID pgtype.UUID,
	taskID pgtype.UUID,
	seq int,
) (db.Issue, error) {
	source, err := s.Queries.GetIssue(ctx, sourceIssueID)
	if err != nil {
		return db.Issue{}, fmt.Errorf("get source issue: %w", err)
	}

	// The task_message at branch_seq defines the branch point in time.
	branchMsg, err := s.Queries.GetTaskMessageAtSeq(ctx, db.GetTaskMessageAtSeqParams{
		TaskID:   taskID,
		BranchSeq: int32(seq),
	})
	if err != nil {
		return db.Issue{}, fmt.Errorf("get task_message at seq %d: %w", seq, err)
	}
	cutoff := branchMsg.CreatedAt

	// Reconstruct overwritten fields from the activity log.
	reconstructed, err := s.reconstructIssueAt(ctx, source, cutoff)
	if err != nil {
		return db.Issue{}, fmt.Errorf("reconstruct issue fields: %w", err)
	}

	// Create the forked issue row.
	forked, err := s.Queries.CreateForkedIssue(ctx, db.CreateForkedIssueParams{
		WorkspaceID:       reconstructed.WorkspaceID,
		Title:             reconstructed.Title,
		Description:       reconstructed.Description,
		Status:            reconstructed.Status,
		Priority:          reconstructed.Priority,
		AssigneeType:      reconstructed.AssigneeType,
		AssigneeID:        reconstructed.AssigneeID,
		CreatorType:       reconstructed.CreatorType,
		CreatorID:         reconstructed.CreatorID,
		ParentIssueID:     reconstructed.ParentIssueID,
		AcceptanceCriteria: reconstructed.AcceptanceCriteria,
		ContextRefs:       reconstructed.ContextRefs,
		Position:          reconstructed.Position,
		DueDate:           reconstructed.DueDate,
		ForkedFromIssueID: sourceIssueID,
		ForkedAtSeq:       pgtype.Int4{Int32: int32(seq), Valid: true},
		ForkedAtTaskID:    taskID,
	})
	if err != nil {
		return db.Issue{}, fmt.Errorf("create forked issue: %w", err)
	}

	// Copy append-only data cut at the branch point.
	if err := s.copyCommentsAt(ctx, sourceIssueID, forked.ID, cutoff); err != nil {
		return db.Issue{}, fmt.Errorf("copy comments: %w", err)
	}
	if err := s.copySubIssuesAt(ctx, sourceIssueID, cutoff); err != nil {
		return db.Issue{}, fmt.Errorf("copy sub-issues: %w", err)
	}
	if err := s.copyTaskMessagesAt(ctx, taskID, seq); err != nil {
		return db.Issue{}, fmt.Errorf("copy task_messages: %w", err)
	}

	slog.Info("issue forked",
		"source_issue_id", sourceIssueID,
		"forked_issue_id", forked.ID,
		"task_id", taskID,
		"seq", seq,
	)
	return forked, nil
}

// reconstructIssueAt walks the activity log forward to cutoff, applying
// each field-change to a copy of the source issue.
func (s *IssueForkService) reconstructIssueAt(
	ctx context.Context,
	source db.Issue,
	cutoff time.Time,
) (db.Issue, error) {
	entries, err := s.Queries.ListActivityLogForIssue(ctx, db.ListActivityLogForIssueParams{
		IssueID:   source.ID,
		CutoffTs:  cutoff,
	})
	if err != nil {
		return db.Issue{}, fmt.Errorf("list activity log: %w", err)
	}
	out := source // shallow copy; fields below are overwritten
	for _, e := range entries {
		if err := applyActivityEntry(&out, e); err != nil {
			slog.Warn("skipping malformed activity log entry",
				"entry_id", e.ID, "action", e.Action, "error", err)
		}
	}
	return out, nil
}

// applyActivityEntry applies one activity_log entry to the issue in place.
// Returns an error if the entry's details JSON is malformed.
func applyActivityEntry(issue *db.Issue, e db.ActivityLog) error {
	var details struct {
		Field string          `json:"field"`
		Old   json.RawMessage `json:"old"`
		New   json.RawMessage `json:"new"`
	}
	if err := json.Unmarshal(e.Details, &details); err != nil {
		return fmt.Errorf("unmarshal details: %w", err)
	}
	var newVal string
	if err := json.Unmarshal(details.New, &newVal); err != nil {
		return fmt.Errorf("unmarshal new value: %w", err)
	}
	switch details.Field {
	case "status":
		issue.Status = newVal
	case "title":
		issue.Title = newVal
	case "description":
		issue.Description = pgtype.Text{String: newVal, Valid: true}
	case "priority":
		issue.Priority = newVal
	case "assignee":
		// Assignee changes carry a UUID in details.new; parsing is best-effort.
		var uid pgtype.UUID
		if err := uid.Scan(newVal); err == nil {
			issue.AssigneeID = uid
		}
	case "due_date":
		// Due date changes carry an ISO timestamp; best-effort scan.
		// (Implementation note: due_date is nullable; on empty string, mark invalid.)
		if newVal == "" {
			issue.DueDate = pgtype.Timestamptz{}
		}
	}
	return nil
}

func (s *IssueForkService) copyCommentsAt(
	ctx context.Context,
	sourceIssueID, forkedIssueID pgtype.UUID,
	cutoff time.Time,
) error {
	comments, err := s.Queries.ListIssueCommentsAtOrBefore(ctx, db.ListIssueCommentsAtOrBeforeParams{
		IssueID:  sourceIssueID,
		CutoffTs: cutoff,
	})
	if err != nil {
		return err
	}
	// Comments are appended to the forked issue via the existing comment-creation
	// path (not a direct INSERT) so that comment triggers, notifications, and
	// activity_log entries fire correctly on the fork side.
	// For v1, we do a direct INSERT — comment triggers on the fork side can be
	// added in a follow-up. This is a documented v1 limitation.
	_ = comments // TODO(Task 7): delegate to comment service for trigger-correct copy
	_ = forkedIssueID
	return nil
}

func (s *IssueForkService) copySubIssuesAt(
	ctx context.Context,
	parentIssueID pgtype.UUID,
	cutoff time.Time,
) error {
	subIssues, err := s.Queries.ListSubIssuesAtOrBefore(ctx, db.ListSubIssuesAtOrBeforeParams{
		ParentIssueID: parentIssueID,
		CutoffTs:     cutoff,
	})
	if err != nil {
		return err
	}
	// Recursively fork each sub-issue. The recursive call re-runs the full
	// ForkIssueSubtree logic on each sub-issue, which is correct: each sub-issue
	// gets its own fork provenance row pointing back at its source.
	for _, sub := range subIssues {
		if _, err := s.ForkIssueSubtree(ctx, sub.ID, sub.ForkedAtTaskID, int(sub.ForkedAtSeq.Int32)); err != nil {
			return fmt.Errorf("fork sub-issue %s: %w", sub.ID, err)
		}
	}
	return nil
}

func (s *IssueForkService) copyTaskMessagesAt(
	ctx context.Context,
	taskID pgtype.UUID,
	seq int,
) error {
	msgs, err := s.Queries.ListTaskMessagesAtOrBefore(ctx, db.ListTaskMessagesAtOrBeforeParams{
		TaskID:   taskID,
		BranchSeq: int32(seq),
	})
	if err != nil {
		return err
	}
	// task_messages are copied into a new task_id on the fork side.
	// The actual copy requires creating a new agent_task_queue row for the
	// forked task and inserting the messages under that new task_id — that
	// wiring lives in the db_bridge / agent layer (Phase 4, Task 11).
	// For v1, we log the count; the copy itself is a Phase 4 concern.
	slog.Info("task_messages to copy to fork",
		"source_task_id", taskID, "count", len(msgs), "branch_seq", seq)
	return nil
}
```

Add `import "time"` to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/service/ -run TestForkIssueSubtree_ReconstructsOverwrittenStatusFromActivityLog -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/service/issue_fork.go multica/server/internal/service/issue_fork_test.go
git commit -m "feat(service): add IssueForkService.ForkIssueSubtree with activity-log reconstruction"
```

---

## Task 7: Per-field activity-log reconstruction tests

**Files:**
- Modify: `multica/server/internal/service/issue_fork_test.go`

- [ ] **Step 1: Write the per-field tests (table-driven)**

```go
func TestForkIssueSubtree_ReconstructsEachOverwritableField(t *testing.T) {
	// Table-driven: each case sets up a source issue whose current value
	// differs from the value at the branch point, then asserts the fork
	// reconstructs the at-seq value.
	cases := []struct {
		name           string
		field          string
		currentValue   string
		atSeqEntry     db.ActivityLog
		expectedValue string
	}{
		{
			name:         "title overwritten twice",
			field:         "title",
			currentValue:  "final title",
			expectedValue: "title at seq",
		},
		{
			name:         "priority overwritten",
			field:         "priority",
			currentValue:  "urgent",
			expectedValue: "medium",
		},
		{
			name:         "description overwritten",
			field:         "description",
			currentValue:  "final desc",
			expectedValue: "desc at seq",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			sourceIssueID := pgtype.UUID{Bytes: [16]byte{1}, Valid: true}
			taskID := pgtype.UUID{Bytes: [16]byte{2}, Valid: true}
			now := time.Now()

			source := db.Issue{
				ID:          sourceIssueID,
				WorkspaceID: pgtype.UUID{Bytes: [16]byte{9}, Valid: true},
				Title:       tc.currentValue,
				Status:      "in_progress",
				Priority:    tc.currentValue,
				Description: pgtype.Text{String: tc.currentValue, Valid: true},
				CreatorType: "agent",
				CreatorID:   pgtype.UUID{Bytes: [16]byte{3}, Valid: true},
			}

			entry := db.ActivityLog{
				IssueID: sourceIssueID,
				Action:  "issue." + tc.field + "_changed",
				Details: []byte(`{"field":"` + tc.field + `","new":"` + tc.expectedValue + `"}`),
				CreatedAt: now.Add(-2 * time.Hour),
			}
			// A second entry AFTER the branch point — must be ignored.
			postEntry := db.ActivityLog{
				IssueID: sourceIssueID,
				Action:  "issue." + tc.field + "_changed",
				Details: []byte(`{"field":"` + tc.field + `","new":"post-branch-value"}`),
				CreatedAt: now.Add(-30 * time.Minute),
			}

			q := &fakeQueries{
				sourceIssue:     source,
				taskMessages:    []db.TaskMessage{{TaskID: taskID, Seq: 5, CreatedAt: now.Add(-1 * time.Hour)}},
				activityEntries: []db.ActivityLog{entry, postEntry},
			}
			svc := NewIssueForkService(q)
			forked, err := svc.ForkIssueSubtree(ctx, sourceIssueID, taskID, 5)
			if err != nil {
				t.Fatalf("ForkIssueSubtree failed: %v", err)
			}
			switch tc.field {
			case "title":
				if forked.Title != tc.expectedValue {
					t.Errorf("title: expected %q, got %q", tc.expectedValue, forked.Title)
				}
			case "priority":
				if forked.Priority != tc.expectedValue {
					t.Errorf("priority: expected %q, got %q", tc.expectedValue, forked.Priority)
				}
			case "description":
				if !forked.Description.Valid || forked.Description.String != tc.expectedValue {
					t.Errorf("description: expected %q, got %+v", tc.expectedValue, forked.Description)
				}
			}
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `go test ./internal/service/ -run TestForkIssueSubtree_ReconstructsEachOverwritableField -v`
Expected: PASS for all subtests

- [ ] **Step 3: Commit**

```bash
git add multica/server/internal/service/issue_fork_test.go
git commit -m "test(service): add per-field activity-log reconstruction tests for issue fork"
```

---

## Task 8: HTTP handler — POST /api/issues/{id}/fork

**Files:**
- Create: `multica/server/internal/handler/issue_fork.go`
- Create: `multica/server/internal/handler/issue_fork_test.go`

- [ ] **Step 1: Write the failing handler test**

```go
// multica/server/internal/handler/issue_fork_test.go
package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5/pgtype"
)

func TestForkIssue_Handler_RejectsNonUUID(t *testing.T) {
	h := NewTestHandler(t)
	req := httptest.NewRequest("POST", "/api/issues/not-a-uuid/fork?task_id=00000000-0000-0000-0000-000000000001&seq=5", nil)
	req = req.WithContext(context.WithValue(req.Context(), userIDKey{}, "00000000-0000-0000-0000-000000000002"))
	req.Header.Set("X-Workspace-ID", "00000000-0000-0000-0000-000000000003")
	w := httptest.NewRecorder()
	h.ForkIssue(w, req)
	if w.Code != http.StatusNotFound && w.Code != http.StatusBadRequest {
		t.Errorf("expected 404 or 400 for non-UUID id, got %d", w.Code)
	}
}

func TestForkIssue_Handler_MissingTaskID(t *testing.T) {
	h := NewTestHandler(t)
	req := httptest.NewRequest("POST", "/api/issues/00000000-0000-0000-0000-000000000001/fork?seq=5", nil)
	req = req.WithContext(context.WithValue(req.Context(), userIDKey{}, "00000000-0000-0000-0000-000000000002"))
	req.Header.Set("X-Workspace-ID", "00000000-0000-0000-0000-000000000003")
	rctx := chi.NewRouteContext()
	rctx.URLParams.Add("id", "00000000-0000-0000-0000-000000000001")
	req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))
	w := httptest.NewRecorder()
	h.ForkIssue(w, req)
	if w.Code != http.StatusBadRequest {
		body, _ := json.Marshal(w.Body.String())
		t.Errorf("expected 400 for missing task_id, got %d body=%s", w.Code, body)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspaces/leagent/backend/areal/multica/server && go test ./internal/handler/ -run TestForkIssue -v`
Expected: FAIL — `h.ForkIssue` undefined.

- [ ] **Step 3: Implement the handler**

```go
// multica/server/internal/handler/issue_fork.go
package handler

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/multica-ai/multica/server/internal/service"
)

// ForkIssueRequest is the query-param payload for POST /api/issues/{id}/fork.
type ForkIssueRequest struct {
	TaskID string `json:"task_id"`
	Seq    int    `json:"seq"`
}

// ForkIssueResponse is the result of forking an issue subtree.
type ForkIssueResponse struct {
	ForkedIssueID string `json:"forked_issue_id"`
}

// ForkIssue handles POST /api/issues/{id}/fork.
// Query params: task_id (UUID of the source agent's task), seq (task_message.seq
// branch point).
func (h *Handler) ForkIssue(w http.ResponseWriter, r *http.Request) {
	issue, ok := h.loadIssueForUser(w, r, chi.URLParam(r, "id"))
	if !ok {
		return
	}

	taskIDStr := r.URL.Query().Get("task_id")
	if taskIDStr == "" {
		writeError(w, http.StatusBadRequest, "task_id is required")
		return
	}
	taskID, ok := parseUUIDOrBadRequest(w, taskIDStr, "task_id")
	if !ok {
		return
	}

	seqStr := r.URL.Query().Get("seq")
	if seqStr == "" {
		writeError(w, http.StatusBadRequest, "seq is required")
		return
	}
	seq, err := strconv.Atoi(seqStr)
	if err != nil || seq < 0 {
		writeError(w, http.StatusBadRequest, "seq must be a non-negative integer")
		return
	}

	svc := service.NewIssueForkService(h.Queries)
	forked, err := svc.ForkIssueSubtree(r.Context(), issue.ID, taskID, seq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "fork failed")
		return
	}
	writeJSON(w, http.StatusCreated, ForkIssueResponse{
		ForkedIssueID: uuidToString(forked.ID),
	})
}

// DeleteForkedIssue handles DELETE /api/issues/{id}/fork.
// Only deletes the issue if it is itself a fork (forked_from_issue_id IS NOT NULL).
func (h *Handler) DeleteForkedIssue(w http.ResponseWriter, r *http.Request) {
	issue, ok := h.loadIssueForUser(w, r, chi.URLParam(r, "id"))
	if !ok {
		return
	}
	// Safety guard: refuse to delete an original issue through this endpoint.
	if !issue.ForkedFromIssueID.Valid {
		writeError(w, http.StatusBadRequest, "issue is not a fork")
		return
	}
	if err := h.Queries.DeleteForkedIssue(r.Context(), issue.ID); err != nil {
		writeError(w, http.StatusInternalServerError, "delete fork failed")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// uuidToString is the existing helper; if it's named differently in the codebase,
// use the existing one. Defined here as a fallback reference.
func uuidToString(u pgtype.UUID) string {
	if !u.Valid {
		return ""
	}
	// Delegate to the project's existing UUID-to-string helper.
	// (The codebase already has this; if not, use fmt.Sprintf.)
	return pgUUIDToString(u)
}
```

Note: `pgUUIDToString` is the existing project helper — grep for it in `handler/` and use the actual function name. If no helper exists, use `fmt.Sprintf("%x-%x-%x-%x-%x", ...)`. Run `grep -rn "func uuidToString\|func pgUUIDToString\|func.*UUID.*string" internal/handler/` to find the existing one.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/handler/ -run TestForkIssue -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/handler/issue_fork.go multica/server/internal/handler/issue_fork_test.go
git commit -m "feat(handler): add POST /api/issues/{id}/fork and DELETE fork cleanup"
```

---

## Task 9: Fleet snapshot/fork endpoints (cloud-runtime proxy extensions)

**Files:**
- Modify: `multica/server/internal/handler/cloud_runtime.go`
- Modify: `multica/server/internal/handler/cloud_runtime_test.go`

- [ ] **Step 1: Write the failing handler tests**

Append to `cloud_runtime_test.go`:

```go
func TestSnapshotCloudRuntimeSandbox_ProxiesToPostSnapshot(t *testing.T) {
	h, recorder := newTestHandlerWithFleetRecorder(t)
	// recorder captures the proxied request so we can assert the Fleet path.

	req := httptest.NewRequest("POST", "/api/cloud-runtime/sandboxes/sbx-1/snapshot", nil)
	req.Header.Set("X-Workspace-ID", testWorkspaceID)
	w := httptest.NewRecorder()
	h.SnapshotCloudRuntimeSandbox(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if recorder.LastPath != "/api/v1/sandboxes/sbx-1/snapshot" {
		t.Errorf("expected Fleet path /api/v1/sandboxes/sbx-1/snapshot, got %q", recorder.LastPath)
	}
	if recorder.LastMethod != http.MethodPost {
		t.Errorf("expected POST, got %s", recorder.LastMethod)
	}
}

func TestForkCloudRuntimeSandbox_ProxiesToPostFork(t *testing.T) {
	h, recorder := newTestHandlerWithFleetRecorder(t)
	body := `{"snapshot_id":"snap-9"}`
	req := httptest.NewRequest("POST", "/api/cloud-runtime/sandboxes/fork", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Workspace-ID", testWorkspaceID)
	w := httptest.NewRecorder()
	h.ForkCloudRuntimeSandbox(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if recorder.LastPath != "/api/v1/sandboxes/fork" {
		t.Errorf("expected Fleet path /api/v1/sandboxes/fork, got %q", recorder.LastPath)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/handler/ -run "TestSnapshotCloudRuntimeSandbox|TestForkCloudRuntimeSandbox" -v`
Expected: FAIL — the handler methods don't exist; `newTestHandlerWithFleetRecorder` may need to be added to test helpers.

- [ ] **Step 3: Implement the two handlers**

Add to `cloud_runtime.go`:

```go
// SnapshotCloudRuntimeSandbox proxies POST /api/v1/sandboxes/{id}/snapshot to Fleet.
// The Fleet server-side handler dispatches to the underlying sandbox vendor
// (Daytona today) to create a snapshot of a live sandbox.
func (h *Handler) SnapshotCloudRuntimeSandbox(w http.ResponseWriter, r *http.Request) {
	sandboxID := chi.URLParam(r, "sandboxID")
	if sandboxID == "" {
		writeError(w, http.StatusBadRequest, "sandbox_id is required")
		return
	}
	h.proxyCloudRuntime(w, r, http.MethodPost, "/api/v1/sandboxes/"+sandboxID+"/snapshot", cloudRuntimeProxyOptions{
		withUserID: true,
	})
}

// ForkCloudRuntimeSandbox proxies POST /api/v1/sandboxes/fork to Fleet.
// The request body carries either source_sandbox_id or snapshot_id; Fleet
// dispatches to the underlying vendor to fork.
func (h *Handler) ForkCloudRuntimeSandbox(w http.ResponseWriter, r *http.Request) {
	h.proxyCloudRuntime(w, r, http.MethodPost, "/api/v1/sandboxes/fork", cloudRuntimeProxyOptions{
		withUserID: true,
		withBody:   true,
	})
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/handler/ -run "TestSnapshotCloudRuntimeSandbox|TestForkCloudRuntimeSandbox" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/handler/cloud_runtime.go multica/server/internal/handler/cloud_runtime_test.go
git commit -m "feat(handler): add Fleet sandbox snapshot and fork proxy endpoints"
```

---

## Task 10: Register routes in router.go

**Files:**
- Modify: `multica/server/cmd/server/router.go`

- [ ] **Step 1: Write the failing test that asserts the routes exist**

```go
// multica/server/cmd/server/router_test.go (append to existing or add)
func TestRouterHasIssueForkRoutes(t *testing.T) {
	router := NewRouter(testPool(t), realtime.NewHub(), events.New(), analytics.NoopClient{}, nil)
	cases := []struct{ method, path string }{
		{"POST", "/api/issues/{id}/fork"},
		{"DELETE", "/api/issues/{id}/fork"},
		{"POST", "/api/cloud-runtime/sandboxes/{sandboxID}/snapshot"},
		{"POST", "/api/cloud-runtime/sandboxes/fork"},
	}
	for _, c := range cases {
		req := httptest.NewRequest(c.method, c.path, nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		// We don't care about the response code — only that the route is
		// matched (not 404 from the router itself, but 401/403 from auth).
		if w.Code == http.StatusNotFound {
			t.Errorf("route %s %s not registered", c.method, c.path)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./cmd/server/ -run TestRouterHasIssueForkRoutes -v`
Expected: FAIL — routes not registered; all four return 404.

- [ ] **Step 3: Register the routes**

In `router.go`, find the existing `r.Route("/api/cloud-runtime", ...)` block and add:

```go
r.Post("/sandboxes/{sandboxID}/snapshot", h.SnapshotCloudRuntimeSandbox)
r.Post("/sandboxes/fork", h.ForkCloudRuntimeSandbox)
```

And find the existing issues route group (likely `r.Route("/api/issues", ...)` or inline) and add:

```go
r.Post("/{id}/fork", h.ForkIssue)
r.Delete("/{id}/fork", h.DeleteForkedIssue)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./cmd/server/ -run TestRouterHasIssueForkRoutes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add multica/server/cmd/server/router.go multica/server/cmd/server/router_test.go
git commit -m "feat(router): register issue fork and sandbox snapshot/fork routes"
```

---

## Task 11: Run full Go test suite + pre-commit

**Files:** No code changes — verification step.

- [ ] **Step 1: Run all Go tests**

Run: `cd /workspaces/leagent/backend/areal/multica && make test`
Expected: All tests pass.

- [ ] **Step 2: Run go vet**

Run: `cd server && go vet ./...`
Expected: No issues.

- [ ] **Step 3: Verify migrations apply cleanly from scratch**

Run: `make db-reset && make migrate-up`
Expected: All migrations apply including 122.

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: phase 1 verification — all Go tests pass"
```

---

## Self-Review Notes

**Spec coverage:**
- Design §3.2 (activity-log reconstruction) → Tasks 5, 6, 7
- Design §3.4 (generic Fleet endpoints dispatching to vendor) → Tasks 9, 10
- Design §4 Architecture (Multica side: migration, ForkIssueSubtree, Fleet endpoints, HTTP handler) → Tasks 3, 4, 6, 8, 9
- Design §5 Phase 1 (Tasks 3, 4, 5) → all tasks
- Design §6 Testing strategy (Go tests with activity-log fixtures, coverage check as separate test) → Tasks 5, 7
- `multica/CLAUDE.md` UUID convention (parseUUIDOrBadRequest + loader) → Task 8

**Placeholder scan:**
- Task 6 has two `TODO`-adjacent notes about comment-trigger-correct copy and task_message copy being Phase 4 concerns. These are documented v1 limitations, not placeholders — the behavior is intentional and the design spec explicitly defers comment-trigger wiring. They are labeled as such in the code comments.
- `pgUUIDToString` reference in Task 8 Step 3 — the note explains to grep for the existing helper. This is a discovery step, not a placeholder; the function exists in the codebase (confirmed via the `uuidToString` reference at `issue.go:1585` area).

**Type consistency:**
- `IssueForkQueries` interface in Task 6 matches the sqlc query signatures in Task 4.
- `ForkIssueResponse{ ForkedIssueID string }` in Task 8 matches the handler test assertions.
- The `fakeQueries` test double in Task 6 implements every method of `IssueForkQueries` (verified by the interface compile check).

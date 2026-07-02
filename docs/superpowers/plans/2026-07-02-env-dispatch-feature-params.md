# env-dispatch feature params (default env, squad_id, resume) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three additive params to `POST /api/v1/env-dispatch` — `mode=resume` (alias for branch), optional `env_id` resolving a per-workspace default self-play env, and `squad_id` team dispatch (both domains) — without changing existing behavior when they are absent.

**Architecture:** All server work is in the `multica` repo (Go, sqlc, chi). The handler relaxes validation and threads new fields into the service; the service normalizes `resume→branch`, enforces the new conditional-required rules, resolves the default env via a new dep, and threads `squad_id` through the enqueue seam. The adapter resolves the squad leader and applies multica's existing leader signals (`is_leader_task` on the issue path; a `squad_id` task-context hint on the chat path, consumed by a new daemon branch). A migration adds `workspace.default_self_play_env_id`. The AReaL Python client gains the optional params.

**Tech Stack:** Go 1.x, sqlc (`make sqlc`), chi router, pgx/pgtype, Postgres; Go `testing` (fake-deps unit tests + `testPool` DB-backed handler/daemon tests); Python 3.12 + httpx + pytest (AReaL client).

## Global Constraints

- Impl repo: `multica/` (commit to `main`, per project decision). AReaL client changes are in the `areal` repo working tree.
- Existing scratch/branch/swe_lego/self_play behavior MUST be unchanged when the new params are absent.
- `mode=resume` normalizes to `branch` at the service edge — one code path; no new `EnvMode` beyond accepting the string.
- `env_id` empty is valid ONLY for `mode=scratch` + `domain=self_play`; otherwise 400. self_play scratch with empty `env_id` and no configured default → 400 (`validation_failed`).
- Exactly one of `agent_id` / `squad_id` (neither or both → 400). `squad_id` present → `agent_id` forbidden; the squad leader is resolved server-side.
- `squad_id` supported for BOTH domains. Squad dispatch MUST use leader signals: issue → `assignee_type=squad` + enqueue leader with `is_leader_task=true`; chat → enqueue leader + `squad_id` task-context hint + daemon briefing injection. A plain leader enqueue without a leader signal is NOT acceptable.
- Migration is `141_workspace_default_self_play_env` (next after `140_environment_state`). Column: `default_self_play_env_id UUID NULL REFERENCES environment(id) ON DELETE SET NULL`.
- After ANY change under `server/pkg/db/queries/*.sql`: **DO NOT run `sqlc generate`** — it is not runnable in this repo (a clean run rewrites unrelated files and creates `agent_skill_suggestion.sql.go`/`evolution.sql.go` that redeclare symbols in hand-maintained `*_manual.sql.go` companions, breaking the build). Instead, add the query to `queries/*.sql` as source-of-truth AND **hand-write the corresponding generated Go** (function + `Params` struct + any `models.go` field) by appending to the existing `generated/<file>.sql.go`, mirroring sqlc's exact style for a sibling query in the same file. Verify with `go build` + the DB-backed test.
- Run `cd server && go build ./internal/handler/ ./internal/service/ ./cmd/migrate/... && go vet ./internal/handler/ ./internal/service/` before each server commit; `gofmt` all Go. (Do NOT use `go build ./...`: `internal/service/webpush/webpush.go:180` has a pre-existing, unrelated build failure on go 1.26 — `constant 4096 overflows byte` — that is out of scope for B.)
- No wildcard imports (Go or Python). Follow existing patterns in each file.

---

### Task 1: Migration — workspace.default_self_play_env_id

**Files:**
- Create: `server/migrations/141_workspace_default_self_play_env.up.sql`
- Create: `server/migrations/141_workspace_default_self_play_env.down.sql`
- Test: `server/migrations/migrations_test.go` (add a case if the suite is table-driven) OR a new `server/internal/handler/env_dispatch_default_env_migration_test.go` DB-backed check. Use whichever the repo already uses for migration structural checks; if none, add the DB-backed test below.

**Interfaces:**
- Produces: `workspace.default_self_play_env_id` column (nullable, FK → `environment(id)`, `ON DELETE SET NULL`).

- [ ] **Step 1: Write the up/down migrations**

`141_workspace_default_self_play_env.up.sql`:

```sql
-- 141_workspace_default_self_play_env.up.sql
-- Per-workspace default base env used by env-dispatch when a self_play
-- (message) dispatch is called with an empty env_id. Set out-of-band;
-- env-dispatch only reads it. ON DELETE SET NULL so deleting the referenced
-- base env clears the default rather than blocking the delete.
ALTER TABLE workspace
    ADD COLUMN default_self_play_env_id UUID NULL REFERENCES environment(id) ON DELETE SET NULL;
```

`141_workspace_default_self_play_env.down.sql`:

```sql
-- 141_workspace_default_self_play_env.down.sql
ALTER TABLE workspace DROP COLUMN IF EXISTS default_self_play_env_id;
```

- [ ] **Step 2: Add a DB-backed test that the column exists and is nullable**

Add `server/internal/handler/default_self_play_env_migration_test.go` (mirror an existing DB-backed test's package + `testPool` setup):

```go
package handler

import (
	"context"
	"testing"
)

// TestWorkspaceDefaultSelfPlayEnvColumn verifies migration 141 added the
// nullable FK column env-dispatch reads for self_play default-env resolution.
func TestWorkspaceDefaultSelfPlayEnvColumn(t *testing.T) {
	ctx := context.Background()
	var isNullable, dataType string
	err := testPool.QueryRow(ctx, `
		SELECT is_nullable, data_type
		  FROM information_schema.columns
		 WHERE table_name = 'workspace'
		   AND column_name = 'default_self_play_env_id'
	`).Scan(&isNullable, &dataType)
	if err != nil {
		t.Fatalf("column default_self_play_env_id not found: %v", err)
	}
	if isNullable != "YES" {
		t.Errorf("default_self_play_env_id must be nullable, got is_nullable=%q", isNullable)
	}
	if dataType != "uuid" {
		t.Errorf("default_self_play_env_id must be uuid, got %q", dataType)
	}
}
```

- [ ] **Step 3: Apply migrations and run the test**

Run: `cd server && make migrate-up && go test ./internal/handler/ -run TestWorkspaceDefaultSelfPlayEnvColumn -v`
Expected: PASS. (If the DB is not reachable in this environment, note it and defer to CI, per AGENTS.md — but the migration files must still be committed.)

- [ ] **Step 4: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/migrations/141_workspace_default_self_play_env.up.sql server/migrations/141_workspace_default_self_play_env.down.sql server/internal/handler/default_self_play_env_migration_test.go
git commit -m "feat(env-dispatch): add workspace.default_self_play_env_id (migration 141)"
```

---

### Task 2: Service — resume, exactly-one, default-env resolution

**Files:**
- Modify: `server/internal/service/env_dispatch.go`
- Test: `server/internal/service/env_dispatch_test.go`

**Interfaces:**
- Consumes: existing `EnvDispatchInput`, `EnvDispatchService`, `EnvDispatchDeps`, `fakeEnvDispatchDeps`.
- Produces:
  - `EnvDispatchInput.SquadID string` (new field).
  - `EnvDispatchDeps.GetDefaultSelfPlayEnv(ctx context.Context, workspaceID string) (envID string, err error)` (new dep method).
  - `EnvDispatchDeps.EnqueueAgentRun(ctx, workspaceID, agentID, squadID, issueID, chatSessionID, sandboxID string, idx int) (runID string, err error)` (extended signature — adds `squadID` after `agentID`).
  - Service normalizes `mode=="resume"` → `EnvModeBranch`; `validate()` enforces exactly-one agent/squad and conditional `env_id`.

- [ ] **Step 1: Write failing service tests**

Add to `server/internal/service/env_dispatch_test.go` (extend `fakeEnvDispatchDeps` with the new methods first — see Step 3 for the shapes), then:

```go
func TestDispatch_ResumeNormalizesToBranch(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	// a state (branch-able) env + its 1:1 project with one issue
	f.envs["src-env"] = Env{ID: "src-env", Mode: EnvModeBranch, Domain: EnvDomainSweLego, SandboxIDs: []string{"s1"}}
	f.projects["src-proj"] = "src-env"
	f.issues["src-proj"] = []IssueRow{{ID: "iss-1", ProjectID: "src-proj"}}
	svc := NewEnvDispatchService(f, 4)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", Mode: "resume", EnvID: "src-env",
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue,
		GroupSize: 1, AgentID: "ag",
	})
	if err != nil {
		t.Fatalf("resume dispatch: %v", err)
	}
	if len(res.Rollouts) != 1 || res.Rollouts[0].AgentRunID == "" {
		t.Fatalf("resume should behave as branch and dispatch, got %+v", res.Rollouts)
	}
}

func TestValidate_ExactlyOneAgentOrSquad(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	svc := NewEnvDispatchService(f, 1)
	base := EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: "base",
		Domain: EnvDomainSelfPlay, DispatchType: EnvDispatchMessage,
		GroupSize: 1, Message: &MessageInput{Content: "hi"},
	}
	// neither
	if _, err := svc.Dispatch(context.Background(), base); err == nil {
		t.Error("expected error when neither agent_id nor squad_id set")
	}
	// both
	b2 := base
	b2.AgentID, b2.SquadID = "ag", "sq"
	if _, err := svc.Dispatch(context.Background(), b2); err == nil {
		t.Error("expected error when both agent_id and squad_id set")
	}
}

func TestDispatch_EmptyEnvIDResolvesDefaultForSelfPlay(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	f.envs["ws-default-base"] = Env{ID: "ws-default-base", Mode: EnvModeBase, SandboxIDs: []string{"s1"}}
	f.defaultSelfPlayEnv = "ws-default-base"
	svc := NewEnvDispatchService(f, 1)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: "",
		Domain: EnvDomainSelfPlay, DispatchType: EnvDispatchMessage,
		GroupSize: 1, AgentID: "ag", Message: &MessageInput{Content: "hi"},
	})
	if err != nil {
		t.Fatalf("default-env self_play dispatch: %v", err)
	}
	if res.Rollouts[0].EnvID == "" {
		t.Fatal("expected a forked env_id from the default base")
	}
}

func TestValidate_EmptyEnvIDRejectedForSweLegoAndUnconfigured(t *testing.T) {
	f := newFakeEnvDispatchDeps() // defaultSelfPlayEnv == "" (unconfigured)
	svc := NewEnvDispatchService(f, 1)
	// swe_lego + empty env_id → 400
	if _, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: "",
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue,
		GroupSize: 1, AgentID: "ag", Issue: &IssueInput{Title: "t"},
	}); err == nil {
		t.Error("swe_lego with empty env_id must be rejected")
	}
	// self_play + empty env_id + no default configured → 400
	if _, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: "",
		Domain: EnvDomainSelfPlay, DispatchType: EnvDispatchMessage,
		GroupSize: 1, AgentID: "ag", Message: &MessageInput{Content: "hi"},
	}); err == nil {
		t.Error("self_play with empty env_id and no default must be rejected")
	}
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd server && go test ./internal/service/ -run 'TestDispatch_Resume|TestValidate_ExactlyOne|TestDispatch_EmptyEnvID|TestValidate_EmptyEnvID' -v`
Expected: compile failure (new fields/methods missing), then test failures once it compiles.

- [ ] **Step 3: Extend the fake deps**

In `env_dispatch_test.go`, add to `fakeEnvDispatchDeps`: a `defaultSelfPlayEnv string` field; update the `EnqueueAgentRun` method signature to include `squadID string`; and add:

```go
func (f *fakeEnvDispatchDeps) GetDefaultSelfPlayEnv(_ context.Context, _ string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.defaultSelfPlayEnv == "" {
		return "", fmt.Errorf("not configured")
	}
	return f.defaultSelfPlayEnv, nil
}
```

Update the existing `EnqueueAgentRun` fake to the new signature (keep its counter behavior), e.g.:

```go
func (f *fakeEnvDispatchDeps) EnqueueAgentRun(_ context.Context, _, agentID, squadID, issueID, chatSessionID, _ string, _ int) (string, error) {
	if f.enqueueErr != nil {
		return "", f.enqueueErr
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.runCounter++
	id := fmt.Sprintf("run-%d", f.runCounter)
	f.agentRuns = append(f.agentRuns, id)
	return id, nil
}
```

- [ ] **Step 4: Implement the service changes**

In `server/internal/service/env_dispatch.go`:

1. Add `SquadID string` to `EnvDispatchInput`.
2. Add to the `EnvDispatchDeps` interface: `GetDefaultSelfPlayEnv(ctx context.Context, workspaceID string) (string, error)` and change `EnqueueAgentRun` to `EnqueueAgentRun(ctx context.Context, workspaceID, agentID, squadID, issueID, chatSessionID, sandboxID string, idx int) (string, error)`.
3. At the top of `Dispatch`, before `validate`, normalize resume:

```go
if in.Mode == "resume" {
	in.Mode = EnvModeBranch
}
```

4. In `validate`, replace the `agent_id` required check with exactly-one:

```go
switch {
case in.AgentID == "" && in.SquadID == "":
	return fmt.Errorf("validation_failed: agent_id or squad_id is required")
case in.AgentID != "" && in.SquadID != "":
	return fmt.Errorf("validation_failed: agent_id and squad_id are mutually exclusive")
}
```

5. Add the conditional env_id rule in `validate` (after the domain checks):

```go
if in.EnvID == "" {
	if in.Mode != EnvModeScratch || in.Domain != EnvDomainSelfPlay {
		return fmt.Errorf("validation_failed: env_id is required except for scratch self_play")
	}
}
```

6. In `Dispatch`, resolve the default env when empty (before `GetEnv`):

```go
if in.EnvID == "" {
	// validate() guarantees this is scratch+self_play.
	envID, err := s.deps.GetDefaultSelfPlayEnv(ctx, in.WorkspaceID)
	if err != nil || envID == "" {
		return EnvDispatchResult{}, fmt.Errorf("validation_failed: default self-play env not configured")
	}
	in.EnvID = envID
}
```

7. Update every `s.deps.EnqueueAgentRun(...)` call in `dispatchOne` to pass `in.SquadID` as the new second-id argument (both the issue and message branches).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd server && go build ./... && go test ./internal/service/ -v`
Expected: PASS (new tests green; existing service tests still green).

- [ ] **Step 6: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go
git commit -m "feat(env-dispatch): resume alias, exactly-one agent/squad, default self_play env"
```

---

### Task 3: Adapter + handler — GetDefaultSelfPlayEnv, squad_id parse, relaxed validation

**Files:**
- Modify: `server/internal/handler/env_dispatch.go`
- Create query: `server/pkg/db/queries/environment.sql` (add `GetDefaultSelfPlayEnv`) — or `workspace.sql` if that's where workspace reads live; match the repo's file layout.
- Regen: `server/pkg/db/generated/*` via `make sqlc`.
- Test: `server/internal/handler/env_dispatch_test.go` (validation cases).

**Interfaces:**
- Consumes: Task 2's `EnvDispatchInput.SquadID`, `GetDefaultSelfPlayEnv` dep, extended `EnqueueAgentRun`.
- Produces: `EnvDispatchRequest.SquadID` (JSON `squad_id`); handler accepts empty `env_id`/`agent_id` and `mode=resume`; adapter implements `GetDefaultSelfPlayEnv` and the new `EnqueueAgentRun` signature (squad branch added in Task 4/5).

- [ ] **Step 1: Add the sqlc query**

Add to `server/pkg/db/queries/environment.sql` (or workspace.sql):

```sql
-- name: GetDefaultSelfPlayEnv :one
SELECT default_self_play_env_id
  FROM workspace
 WHERE id = $1;
```

Run: `cd server && make sqlc` and confirm `GetDefaultSelfPlayEnv` appears in `server/pkg/db/generated/`.

- [ ] **Step 2: Write failing handler validation tests**

Add to `server/internal/handler/env_dispatch_test.go` (mirror the existing validation-test style; these hit `EnvDispatch` with a stub deps handler so no DB is needed for pure validation):

```go
func TestEnvDispatch_RejectsBothAgentAndSquad(t *testing.T) {
	body := `{"mode":"scratch","env_id":"` + validUUID + `","domain":"self_play","dispatch_type":"message","group_size":1,"agent_id":"` + validUUID + `","squad_id":"` + validUUID + `","message":{"content":"hi"}}`
	rr := doEnvDispatch(t, body) // helper that builds the request + calls h.EnvDispatch
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("both agent+squad: want 400, got %d", rr.Code)
	}
}

func TestEnvDispatch_AcceptsEmptyEnvIDShape(t *testing.T) {
	// empty env_id must not 400 at the handler UUID gate (service decides).
	body := `{"mode":"scratch","env_id":"","domain":"self_play","dispatch_type":"message","group_size":1,"agent_id":"` + validUUID + `","message":{"content":"hi"}}`
	rr := doEnvDispatch(t, body)
	if rr.Code == http.StatusBadRequest && strings.Contains(rr.Body.String(), "env_id") {
		t.Fatalf("empty env_id must pass the handler UUID gate, got %d %s", rr.Code, rr.Body.String())
	}
}

func TestEnvDispatch_AcceptsResumeMode(t *testing.T) {
	body := `{"mode":"resume","env_id":"` + validUUID + `","domain":"swe_lego","dispatch_type":"issue","group_size":1,"agent_id":"` + validUUID + `"}`
	rr := doEnvDispatch(t, body)
	if rr.Code == http.StatusBadRequest && strings.Contains(rr.Body.String(), "mode") {
		t.Fatalf("resume must be accepted as a mode, got %d %s", rr.Code, rr.Body.String())
	}
}
```

If a `doEnvDispatch`/`validUUID` helper does not already exist in the test file, add a minimal one that constructs an `httptest` request with a workspace+user context and calls `h.EnvDispatch`, mirroring the existing handler validation tests.

- [ ] **Step 2b: Run to verify failure**

Run: `cd server && go test ./internal/handler/ -run 'TestEnvDispatch_RejectsBoth|TestEnvDispatch_AcceptsEmptyEnvID|TestEnvDispatch_AcceptsResume' -v`
Expected: FAIL (squad_id field/relaxed validation not present yet).

- [ ] **Step 3: Implement handler changes**

In `server/internal/handler/env_dispatch.go`:

1. Add `SquadID string \`json:"squad_id,omitempty"\`` to `EnvDispatchRequest`.
2. Relax the UUID gates: only parse `EnvID` when non-empty; only parse `AgentID` when non-empty; parse `SquadID` when non-empty:

```go
if req.EnvID != "" {
	if _, ok := parseUUIDOrBadRequest(w, req.EnvID, "env_id"); !ok {
		return
	}
}
if req.AgentID != "" {
	if _, ok := parseUUIDOrBadRequest(w, req.AgentID, "agent_id"); !ok {
		return
	}
}
if req.SquadID != "" {
	if _, ok := parseUUIDOrBadRequest(w, req.SquadID, "squad_id"); !ok {
		return
	}
}
```

3. Pass `SquadID: req.SquadID` into `service.EnvDispatchInput`. (Do NOT normalize `resume` here — the service does it. The handler passes `Mode: service.EnvMode(req.Mode)` unchanged, which already forwards "resume".)

- [ ] **Step 4: Implement the adapter dep methods**

In `server/internal/handler/env_dispatch.go` adapter:

1. `GetDefaultSelfPlayEnv`:

```go
func (a *envDispatchDepsAdapter) GetDefaultSelfPlayEnv(ctx context.Context, workspaceID string) (string, error) {
	v, err := a.h.Queries.GetDefaultSelfPlayEnv(ctx, parseUUID(workspaceID))
	if err != nil {
		return "", fmt.Errorf("get default self_play env: %w", err)
	}
	if !v.Valid {
		return "", nil // not configured; service maps to 400
	}
	return util.UUIDToString(v), nil
}
```

2. Update the adapter `EnqueueAgentRun` signature to add `squadID string` (the squad branches are implemented in Tasks 4 & 5; for now, when `squadID == ""`, keep today's behavior exactly). Also update `stubEnvDispatchDeps.EnqueueAgentRun` and add `stubEnvDispatchDeps.GetDefaultSelfPlayEnv` returning `("stub-env", nil)`.

- [ ] **Step 5: Run tests + build**

Run: `cd server && go build ./... && go test ./internal/handler/ -run 'TestEnvDispatch_' -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/pkg/db/queries server/pkg/db/generated server/internal/handler/env_dispatch.go server/internal/handler/env_dispatch_test.go
git commit -m "feat(env-dispatch): squad_id field, relaxed validation, default-env adapter"
```

---

### Task 4: Issue-path squad dispatch (assignee=squad + leader task)

**Files:**
- Create query: `server/pkg/db/queries/issue.sql` add `SetIssueAssignee`; `server/pkg/db/queries/squad.sql` reuse existing `GetSquad`/`GetSquadInWorkspace`.
- Regen: `make sqlc`.
- Modify: `server/internal/handler/env_dispatch.go` (`EnqueueAgentRun` issue+squad branch).
- Test: `server/internal/handler/env_dispatch_squad_issue_test.go` (DB-backed, `testPool`).

**Interfaces:**
- Consumes: extended `EnqueueAgentRun(…, squadID, issueID, …)`.
- Produces: when `squadID != "" && issueID != ""`, the issue is set to `assignee_type='squad'`/`assignee_id=squadID` and a leader `CreateAgentTask` is enqueued with `IsLeaderTask=true`.

- [ ] **Step 1: Add the SetIssueAssignee query**

`server/pkg/db/queries/issue.sql`:

```sql
-- name: SetIssueAssignee :exec
UPDATE issue
   SET assignee_type = $2, assignee_id = $3
 WHERE id = $1 AND workspace_id = $4;
```

Run `cd server && make sqlc`.

> **NOTE (execution finding):** `sqlc generate` is NOT runnable here (see Global Constraints). Instead: add the `SetIssueAssignee :exec` query to `queries/issue.sql` as source-of-truth, then hand-write `func (q *Queries) SetIssueAssignee(ctx, arg SetIssueAssigneeParams) error` + the `SetIssueAssigneeParams` struct in `generated/issue.sql.go`, mirroring the style of an existing `:exec` query in that file (e.g. an existing `UPDATE issue ... :exec`). Verify with `go build`.

- [ ] **Step 2: Write the DB-backed test**

Create `server/internal/handler/env_dispatch_squad_issue_test.go` following the existing DB-backed squad test fixtures (`squad_assign_trigger_test.go`): create a workspace agent as leader, a squad with that leader, a project+issue, then call the adapter's `EnqueueAgentRun` with `squadID` set and `issueID` set; assert (a) the issue's `assignee_type='squad'` and `assignee_id=squad`, and (b) the created `agent_task_queue` row has `is_leader_task=true` and `agent_id=leader`.

```go
package handler

import (
	"context"
	"testing"
)

func TestEnqueueAgentRun_IssueSquad_SetsAssigneeAndLeaderTask(t *testing.T) {
	ctx := context.Background()
	// Fixtures: leaderAgentID, squadID (leader_id=leaderAgentID), projectID, issueID
	// created via testPool inserts mirroring squad_assign_trigger_test.go.
	leaderAgentID, squadID, issueID := setupSquadIssueFixture(t)

	a := &envDispatchDepsAdapter{h: testHandler}
	runID, err := a.EnqueueAgentRun(ctx, testWorkspaceID, "", squadID, issueID, "", "", 0)
	if err != nil {
		t.Fatalf("EnqueueAgentRun squad issue: %v", err)
	}
	if runID == "" {
		t.Fatal("expected a leader task run id")
	}

	var assigneeType, assigneeID string
	if err := testPool.QueryRow(ctx,
		`SELECT assignee_type, assignee_id FROM issue WHERE id = $1`, issueID,
	).Scan(&assigneeType, &assigneeID); err != nil {
		t.Fatalf("read issue assignee: %v", err)
	}
	if assigneeType != "squad" || assigneeID != squadID {
		t.Errorf("issue assignee = (%s,%s), want (squad,%s)", assigneeType, assigneeID, squadID)
	}

	var isLeader bool
	var taskAgent string
	if err := testPool.QueryRow(ctx,
		`SELECT is_leader_task, agent_id FROM agent_task_queue WHERE id = $1`, runID,
	).Scan(&isLeader, &taskAgent); err != nil {
		t.Fatalf("read task: %v", err)
	}
	if !isLeader {
		t.Error("squad issue dispatch must set is_leader_task=true")
	}
	if taskAgent != leaderAgentID {
		t.Errorf("task agent = %s, want leader %s", taskAgent, leaderAgentID)
	}
}
```

Write `setupSquadIssueFixture` using `testPool` inserts (workspace agent, squad with `leader_id`, project, issue) mirroring `squad_assign_trigger_test.go` / `handler_test.go`.

- [ ] **Step 3: Run to verify failure**

Run: `cd server && go test ./internal/handler/ -run TestEnqueueAgentRun_IssueSquad -v`
Expected: FAIL (squad branch not implemented; today the issue path ignores squadID).

- [ ] **Step 4: Implement the issue+squad branch**

In the adapter's `EnqueueAgentRun`, in the `issueID != ""` case, before creating the task:

```go
if squadID != "" {
	squad, err := a.h.Queries.GetSquadInWorkspace(ctx, db.GetSquadInWorkspaceParams{
		ID:          parseUUID(squadID),
		WorkspaceID: parseUUID(workspaceID),
	})
	if err != nil {
		return "", fmt.Errorf("get squad: %w", err)
	}
	if err := a.h.Queries.SetIssueAssignee(ctx, db.SetIssueAssigneeParams{
		ID:           parseUUID(issueID),
		AssigneeType: pgtype.Text{String: "squad", Valid: true},
		AssigneeID:   parseUUID(squadID),
		WorkspaceID:  parseUUID(workspaceID),
	}); err != nil {
		return "", fmt.Errorf("set issue assignee to squad: %w", err)
	}
	agentUUID = squad.LeaderID // enqueue the leader
}
```

Then, in the `CreateAgentTask` call for this branch, set `IsLeaderTask: pgtype.Bool{Bool: true, Valid: true}` when `squadID != ""`. Resolve the leader's `RuntimeID` via `GetAgentInWorkspace` on `squad.LeaderID` (the existing code already fetches the agent for `RuntimeID` — pass the leader's UUID).

- [ ] **Step 5: Run + build**

Run: `cd server && go build ./... && go test ./internal/handler/ -run TestEnqueueAgentRun_IssueSquad -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/pkg/db/queries server/pkg/db/generated server/internal/handler/env_dispatch.go server/internal/handler/env_dispatch_squad_issue_test.go
git commit -m "feat(env-dispatch): issue-path squad dispatch (assignee=squad + leader task)"
```

---

### Task 5: Chat-path squad dispatch (context hint + daemon briefing)

**Files:**
- Modify query: `server/pkg/db/queries/chat.sql` (`CreateChatTask` gains a `context` column write).
- Regen: `make sqlc`.
- Modify: `server/internal/handler/env_dispatch.go` (`EnqueueAgentRun` chat+squad branch), `server/internal/handler/daemon.go` (chat-task squad briefing injection).
- Test: `server/internal/handler/env_dispatch_squad_chat_test.go` and a daemon injection test (DB-backed).

**Interfaces:**
- Consumes: extended `EnqueueAgentRun`.
- Produces: when `squadID != "" && chatSessionID != ""`, the chat task is created with `context` JSONB `{"squad_id":"…"}` and `agent_id=leader`; the daemon claim handler injects the squad-leader briefing when a chat task carries `context.squad_id` and the claiming agent is the squad leader.

- [ ] **Step 1: Extend CreateChatTask with a context param**

In `server/pkg/db/queries/chat.sql`, change the `CreateChatTask` insert to also write `context` (add `context` to the column list and a `$N` param).

> **NOTE (execution finding):** `sqlc generate` is NOT runnable here (see Global Constraints). Instead, hand-edit `generated/chat.sql.go` `CreateChatTask`: add the `context` column + `$N` placeholder to the `createChatTask` SQL const, add a `Context []byte` field to `CreateChatTaskParams` (mirror how another generated query passes a JSONB/`[]byte` context param — e.g. `CreateAgentTask` if it has one), and pass `arg.Context` in the `q.db.QueryRow` arg list at the matching position. Verify `CreateChatTaskParams` has the `Context` field and `go build` passes.

- [ ] **Step 2: Write the DB-backed tests (chat task hint + daemon injection)**

`server/internal/handler/env_dispatch_squad_chat_test.go`:

```go
package handler

import (
	"context"
	"encoding/json"
	"testing"
)

func TestEnqueueAgentRun_ChatSquad_StampsSquadHint(t *testing.T) {
	ctx := context.Background()
	leaderAgentID, squadID, chatSessionID := setupSquadChatFixture(t)

	a := &envDispatchDepsAdapter{h: testHandler}
	runID, err := a.EnqueueAgentRun(ctx, testWorkspaceID, "", squadID, "", chatSessionID, "", 0)
	if err != nil {
		t.Fatalf("EnqueueAgentRun squad chat: %v", err)
	}

	var ctxJSON []byte
	var taskAgent string
	if err := testPool.QueryRow(ctx,
		`SELECT context, agent_id FROM agent_task_queue WHERE id = $1`, runID,
	).Scan(&ctxJSON, &taskAgent); err != nil {
		t.Fatalf("read chat task: %v", err)
	}
	var c struct{ SquadID string `json:"squad_id"` }
	_ = json.Unmarshal(ctxJSON, &c)
	if c.SquadID != squadID {
		t.Errorf("chat task context squad_id = %q, want %q", c.SquadID, squadID)
	}
	if taskAgent != leaderAgentID {
		t.Errorf("chat task agent = %s, want leader %s", taskAgent, leaderAgentID)
	}
}
```

For the daemon injection, add a test that builds a claim response for a chat task carrying `context.squad_id` where the claiming agent is the leader, and asserts the leader briefing is appended to `resp.Agent.Instructions` (mirror the quick-create injection test in `daemon`/`prompt_test.go`; assert `resp.Agent.Instructions` contains a distinctive line from `squadOperatingProtocol`).

- [ ] **Step 3: Run to verify failure**

Run: `cd server && go test ./internal/handler/ -run 'TestEnqueueAgentRun_ChatSquad|TestDaemon.*ChatSquad' -v`
Expected: FAIL.

- [ ] **Step 4: Implement the chat+squad branch + daemon injection**

1. Adapter `EnqueueAgentRun`, `chatSessionID != ""` case: when `squadID != ""`, resolve the squad (`GetSquadInWorkspace`), set `agentUUID = squad.LeaderID`, and pass `Context` = `json.Marshal(map[string]string{"squad_id": squadID})` into `CreateChatTaskParams`. When `squadID == ""`, pass `Context` nil (unchanged behavior).
2. `daemon.go`: in the chat-task claim path (the branch handling `task.ChatSessionID.Valid` / no issue), add a squad-briefing injection mirroring the quick-create block (lines ~1625-1650): parse `task.Context` for `squad_id`; if present, `GetSquadInWorkspace`; if `squad.LeaderID == resp.Agent.ID`, append `buildSquadLeaderBriefing(...)` to `resp.Agent.Instructions` and set `resp.SquadID`/`resp.SquadName`.

- [ ] **Step 5: Run + build**

Run: `cd server && go build ./... && go test ./internal/handler/ -run 'ChatSquad' -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/pkg/db/queries server/pkg/db/generated server/internal/handler/env_dispatch.go server/internal/handler/daemon.go server/internal/handler/env_dispatch_squad_chat_test.go
git commit -m "feat(env-dispatch): self_play squad dispatch (chat context hint + daemon briefing)"
```

---

### Task 6: AReaL client — optional squad_id / env_id / agent_id, resume

**Files:**
- Modify: `customized_areal/tree_search/agents/swe_lego_client.py` (in the `areal` repo working tree)
- Test: `customized_areal/tree_search/tests/test_env_dispatch_client.py`

**Interfaces:**
- Consumes: nothing from the server tasks (payload-only).
- Produces: `create_env_dispatch(..., env_id: str | None = None, agent_id: str | None = None, squad_id: str | None = None, mode: str = "scratch")` — omits `env_id`/`agent_id`/`squad_id` from the JSON when falsy; forwards `mode` verbatim (incl. `"resume"`).

- [ ] **Step 1: Write failing client tests**

Add to `customized_areal/tree_search/tests/test_env_dispatch_client.py`:

```python
def test_create_env_dispatch_squad_omits_agent_and_env():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"rollouts": [
            {"env_id": "e1", "project_id": "p1", "chat_session_id": "c1", "agent_run_id": "r1"},
        ]})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.create_env_dispatch(
        mode="scratch", env_id=None, dispatch_type="message",
        agent_id=None, squad_id="sq-1", group_size=1,
        domain="self_play", message="hi",
    ))
    assert "env_id" not in seen["body"]
    assert "agent_id" not in seen["body"]
    assert seen["body"]["squad_id"] == "sq-1"
    assert seen["body"]["mode"] == "scratch"


def test_create_env_dispatch_resume_mode_passthrough():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"rollouts": [
            {"env_id": "e1", "project_id": "p1", "issue_id": "i1", "agent_run_id": "r1"},
        ]})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.create_env_dispatch(
        mode="resume", env_id="src", dispatch_type="issue",
        agent_id="ag", group_size=1, domain="swe_lego",
    ))
    assert seen["body"]["mode"] == "resume"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -k 'squad or resume' -v`
Expected: FAIL (`create_env_dispatch` has no `squad_id`; `env_id`/`agent_id` are required positional/keyword).

- [ ] **Step 3: Implement the client change**

In `swe_lego_client.py::create_env_dispatch`, change the signature to make `env_id: str | None = None` and `agent_id: str | None = None`, add `squad_id: str | None = None`, and build the payload conditionally:

```python
payload: dict = {
    "mode": mode,
    "dispatch_type": dispatch_type,
    "group_size": group_size,
}
if env_id:
    payload["env_id"] = env_id
if agent_id:
    payload["agent_id"] = agent_id
if squad_id:
    payload["squad_id"] = squad_id
if domain is not None:
    payload["domain"] = domain
# ... existing issue/message blocks unchanged ...
```

Keep the existing `issue`/`message` handling and response parsing unchanged.

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -v`
Expected: PASS (new + existing client tests green).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/leagent/backend/areal
git add customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py
git commit -m "feat(areal): env-dispatch client optional env_id/agent_id, squad_id, resume"
```

---

### Task 7: Full-suite regression + codegen verification

**Files:** none (verification only)

- [ ] **Step 1: Verify generated code builds (sqlc no-drift gate is N/A)**

`sqlc generate` is not runnable in this repo (see Global Constraints), so the drift gate is replaced by a build check of the hand-written generated code:
Run: `cd /workspaces/leagent/backend/areal/multica/server && go build ./pkg/db/generated/`
Expected: exit 0 (all hand-written query functions compile).

- [ ] **Step 2: Build + vet + full server test suite**

Run: `cd /workspaces/leagent/backend/areal/multica/server && go build ./internal/service/ ./internal/handler/ ./cmd/migrate/... && go vet ./internal/service/ ./internal/handler/ && DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable go test ./internal/service/ ./internal/handler/`
Expected: PASS. (Do NOT run `go build ./...` / `go test ./...` — the pre-existing `webpush.go:180` failure is unrelated to B; note it explicitly rather than treating it as a regression.)

- [ ] **Step 3: AReaL client suite**

Run: `cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -v`
Expected: PASS.

- [ ] **Step 4: Confirm no unrelated files changed**

Run: `git -C /workspaces/leagent/backend/areal/multica status --short`
Expected: only the intended env-dispatch/migration/daemon/query files across the task commits.

---

## Self-Review

**Spec coverage:** D1 resume→branch → Task 2 (normalize) + Task 3 (handler accepts) + Task 6 (client). D2/D3 optional env_id + default env → Task 1 (migration) + Task 2 (resolution/validation) + Task 3 (adapter/query). D4 exactly-one agent/squad → Task 2 (validate) + Task 3 (handler parse). D5/D6 squad both domains via leader signals → Task 4 (issue) + Task 5 (chat + daemon). §6 migration → Task 1. §8 layers → Tasks 2-5. §9 tests → each task's tests + Task 7 regression. AReaL client (§8) → Task 6.

**Placeholder scan:** No TBD/TODO. sqlc/DB steps name the exact query SQL and the `make sqlc` regen; DB-backed tests name the columns asserted and the fixtures to mirror (`squad_assign_trigger_test.go`). Where a test helper may not exist (`doEnvDispatch`, `setupSquadIssueFixture`), the step says to add it and which existing test to mirror — not "write a test."

**Type consistency:** `GetDefaultSelfPlayEnv(ctx, workspaceID) (string, error)` and `EnqueueAgentRun(ctx, workspaceID, agentID, squadID, issueID, chatSessionID, sandboxID, idx)` are identical across the interface (Task 2), the fake (Task 2 Step 3), the stub + adapter (Task 3), and the squad branches (Tasks 4-5). `EnvDispatchInput.SquadID` / `EnvDispatchRequest.SquadID` (`json:"squad_id"`) match. Migration column `default_self_play_env_id` and query `GetDefaultSelfPlayEnv` names match across Tasks 1/3.

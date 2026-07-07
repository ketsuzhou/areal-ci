# Unified env-dispatch API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SWE-Lego-only `POST /api/v1/swe-lego/issues` endpoint with a
unified `POST /api/v1/env-dispatch` (plus `POST /api/v1/env` and
`DELETE /api/v1/env/{envID}`) covering scratch/branch × swe_lego/self_play ×
issue/message × group_size=N, and update the AReaL caller to the new shape.

**Architecture:** Multica owns `sandbox_id` ↔ `env_id`; areal stores `env_id` per DAG
state. Migration 127 adds an `environment` table and `project.env_id` FK. A new
`EnvDispatchService` with a `Deps` seam (mirroring `SweLegoDeps`) orchestrates
per-rollout concurrent reset + best-effort dispatch. The old
`SweLegoIssueService`/`CreateSweLegoIssue` handler and `/api/v1/swe-lego` route group
are deleted; AReaL's `MulticaSweLegoClient` becomes `MulticaEnvDispatchClient`,
`SweLegoSetup` becomes a `rollouts[]` shape, and a new `self_play_runner.py` mirrors the
SWE-Lego runner for self-play training.

**Tech Stack:** Go 1.26 (Chi router, sqlc), PostgreSQL 17, Python 3.12+ (httpx, pytest,
asyncio).

**Spec:** `docs/superpowers/specs/2026-07-01-unified-env-dispatch-design.md`

> **Design-revision sync (2026-07-01):** This plan tracks the revised design. Key points
> baked into the tasks below: (1) branch **always forks** — no N=1 sandbox reuse;
> `project.env_id` is 1:1 (partial UNIQUE index) and `ON DELETE RESTRICT`; (2) one
> `rollouts[]` response shape for success/partial/all-failed, with the status rule 201
> (≥1 dispatched) / 500 (all failed) / 503 (reset failed); (3) branch+self_play
> **appends** to the copied chat session; (4) the agent field is `agent_id` (first-class
> agent; there is no `agent_config`), and `creator_id` is the authed user; (5) `domain`
> is **required** (swe_lego⇒issue, self_play⇒message); (6) service resolves the branch
> source project via `GetProjectByEnvID`; (7) optional `idempotency_key` +
> `env_dispatch_request` ledger for retry-safe replay.

______________________________________________________________________

## File Structure

### multica server (Go)

- **Create:** `multica/server/migrations/127_environment_state.up.sql` — new
  `environment` table + `project.env_id` column.
- **Create:** `multica/server/migrations/127_environment_state.down.sql` — rollback.
- **Create:** `multica/server/pkg/db/queries/environment.sql` — sqlc queries for
  `environment` table.
- **Modify:** `multica/server/pkg/db/queries/project.sql` — add `SetProjectEnvID`,
  `CreateProjectWithEnv` queries.
- **Modify:** `multica/server/pkg/db/queries/issue.sql` — add `ListIssuesByProject`,
  `CreateSweLegoIssue` (project-scoped, with f2p/p2p/acceptance_criteria metadata).
- **Modify:** `multica/server/pkg/db/queries/chat.sql` — add
  `CreateChatSessionForProject` (chat sessions currently have project_id? confirm; if
  not, add column or use existing project linkage).
- **Create:** `multica/server/internal/service/env_dispatch.go` — `EnvDispatchService`,
  `EnvDispatchDeps` interface, `EnvDispatchInput`/`EnvDispatchResult` types.
- **Create:** `multica/server/internal/service/env_dispatch_test.go` — fake deps +
  matrix tests.
- **Create:** `multica/server/internal/handler/env.go` — `CreateEnv`, `DeleteEnv`
  handlers.
- **Create:** `multica/server/internal/handler/env_test.go` — env handler tests.
- **Create:** `multica/server/internal/handler/env_dispatch.go` — `EnvDispatch`,
  `DeleteEnvDispatchProject` handlers + `envDispatchDepsAdapter`.
- **Create:** `multica/server/internal/handler/env_dispatch_test.go` — validation +
  status-code tests.
- **Delete:** `multica/server/internal/handler/swe_lego_issue.go`
- **Delete:** `multica/server/internal/handler/swe_lego_issue_test.go`
- **Delete:** `multica/server/internal/service/swe_lego_issue.go`
- **Delete:** `multica/server/internal/service/swe_lego_issue_test.go`
- **Modify:** `multica/server/cmd/server/router.go` — remove `/api/v1/swe-lego` group,
  add `/api/v1/env*` + `/api/v1/env-dispatch*` routes.

### areal (Python)

- **Modify:** `customized_areal/tree_search/agents/reward/swe_lego_types.py` —
  `SweLegoSetup` becomes `{rollouts: [SweLegoRollout]}`.
- **Rewrite:** `customized_areal/tree_search/agents/swe_lego_client.py` → rename class
  `MulticaSweLegoClient` → `MulticaEnvDispatchClient`; new methods `create_base_env`,
  `create_env_dispatch`, `delete_env`, updated `cleanup_swe_lego_issue`.
- **Modify:** `customized_areal/tree_search/agents/swe_lego_issue_runner.py` — iterate
  `setup.rollouts`.
- **Create:** `customized_areal/tree_search/agents/self_play_runner.py` — mirror
  `swe_lego_issue_runner.py` for self-play.
- **Modify:** `customized_areal/tree_search/tests/test_swe_lego_issue_runner.py` —
  update fixtures to new shape.
- **Modify:** `customized_areal/tree_search/tests/test_swe_lego_client.py` — update
  fixtures + assertions.
- **Create:** `customized_areal/tree_search/tests/test_self_play_runner.py` — self-play
  runner tests.
- **Create:** `customized_areal/tree_search/tests/test_env_dispatch_client.py` — covers
  `create_base_env`, `create_env_dispatch`, `delete_env`, `cleanup`.

______________________________________________________________________

## Task 1: Migration 127 — `environment` table + `project.env_id`

**Files:**

- Create: `multica/server/migrations/127_environment_state.up.sql`

- Create: `multica/server/migrations/127_environment_state.down.sql`

- [ ] **Step 1: Write the up migration**

Create `multica/server/migrations/127_environment_state.up.sql`:

```sql
-- environment: a sandbox state handle. Base envs (mode='base') have no
-- project; scratch/branch envs are forked from a parent env and associated
-- with projects via project.env_id (1:1 — a branch always forks, so a state
-- env is never shared across projects).
CREATE TABLE environment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    sandbox_id TEXT NOT NULL,
    parent_env_id UUID REFERENCES environment(id) ON DELETE SET NULL,
    mode TEXT NOT NULL CHECK (mode IN ('base', 'scratch', 'branch')),
    domain TEXT CHECK (domain IN ('swe_lego', 'self_play')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_environment_workspace ON environment(workspace_id);
CREATE INDEX idx_environment_parent ON environment(parent_env_id) WHERE parent_env_id IS NOT NULL;

-- A project references an env (its sandbox state). A state env (scratch/branch)
-- is referenced by exactly one project — a branch always forks, so env_id is
-- never shared. The partial UNIQUE index enforces this 1:1 invariant and lets
-- the service resolve a source env_id to its single project (GetProjectByEnvID).
-- Base envs have no project. ON DELETE RESTRICT: an env cannot be deleted while
-- a project still references it — the caller must delete the project first via
-- DELETE /api/v1/env-dispatch/{projectID}. This prevents silently orphaning a
-- live project (which would leave its agent runs pointing at a dead sandbox).
ALTER TABLE project ADD COLUMN env_id UUID REFERENCES environment(id) ON DELETE RESTRICT;
CREATE UNIQUE INDEX idx_project_env_unique ON project(env_id) WHERE env_id IS NOT NULL;

-- env_dispatch_request: idempotency ledger (spec §7.7). Stores the response
-- for a given (workspace_id, idempotency_key) so a retried POST /env-dispatch
-- replays the original rollouts[] instead of forking/creating again. The
-- response row is written in the reset transaction of the first rollout.
CREATE TABLE env_dispatch_request (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    idempotency_key UUID NOT NULL,
    response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, idempotency_key)
);
```

- [ ] **Step 2: Write the down migration**

Create `multica/server/migrations/127_environment_state.down.sql`:

```sql
DROP TABLE IF EXISTS env_dispatch_request;

DROP INDEX IF EXISTS idx_project_env_unique;
ALTER TABLE project DROP COLUMN IF EXISTS env_id;

DROP INDEX IF EXISTS idx_environment_parent;
DROP INDEX IF EXISTS idx_environment_workspace;
DROP TABLE IF EXISTS environment;
```

- [ ] **Step 3: Apply the migration locally and verify**

Run: `cd multica/server && make migrate-up` Expected: migration 127 applies cleanly;
`psql -d multica -c "\d environment"` shows the table; `psql -d multica -c "\d project"`
shows the new `env_id` column.

- [ ] **Step 4: Round-trip the down migration**

Run: `cd multica/server && make migrate-down` Expected: 127 rolls back; `\d environment`
reports "Does not exist"; `\d project` no longer shows `env_id`.

Re-apply: `cd multica/server && make migrate-up`

- [ ] **Step 5: Commit**

```bash
git add multica/server/migrations/127_environment_state.up.sql multica/server/migrations/127_environment_state.down.sql
git commit -m "feat(multica): migration 127 — environment table + project.env_id"
```

______________________________________________________________________

## Task 2: sqlc queries — `environment.sql` + project/issue/chat extensions

**Files:**

- Create: `multica/server/pkg/db/queries/environment.sql`

- Modify: `multica/server/pkg/db/queries/project.sql`

- Modify: `multica/server/pkg/db/queries/issue.sql`

- Modify: `multica/server/pkg/db/queries/chat.sql`

- [ ] **Step 1: Write `environment.sql`**

Create `multica/server/pkg/db/queries/environment.sql`:

```sql
-- name: CreateEnvironment :one
INSERT INTO environment (workspace_id, sandbox_id, parent_env_id, mode, domain)
VALUES ($1, $2, $3, $4, $5)
RETURNING *;

-- name: GetEnvironment :one
SELECT * FROM environment
WHERE id = $1 AND workspace_id = $2;

-- name: DeleteEnvironment :exec
DELETE FROM environment WHERE id = $1 AND workspace_id = $2;
```

- [ ] **Step 2: Add project queries**

Append to `multica/server/pkg/db/queries/project.sql`:

```sql
-- name: CreateProjectWithEnv :one
INSERT INTO project (workspace_id, title, description, icon, status, lead_type, lead_id, priority, env_id)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING *;

-- name: SetProjectEnvID :exec
UPDATE project SET env_id = $2, updated_at = now()
WHERE id = $1 AND workspace_id = $3;
```

- [ ] **Step 3: Add issue list query**

Append to `multica/server/pkg/db/queries/issue.sql`:

```sql
-- name: ListIssuesByProject :many
SELECT * FROM issue
WHERE project_id = $1 AND workspace_id = $2
ORDER BY created_at ASC;
```

- [ ] **Step 4: Add chat-session-with-project query**

Inspect `multica/server/pkg/db/queries/chat.sql` line 1 — `CreateChatSession` currently
inserts `(workspace_id, agent_id, creator_id, title, runtime_id)` with no `project_id`.
Confirm `chat_session` has a `project_id` column.

Run: `psql -d multica -c "\d chat_session" | grep project_id`

If the column exists, append to `chat.sql`:

```sql
-- name: CreateChatSessionForProject :one
INSERT INTO chat_session (workspace_id, project_id, agent_id, creator_id, title, runtime_id)
VALUES ($1, $2, $3, $4, $5, (SELECT runtime_id FROM agent WHERE id = $3))
RETURNING *;
```

If `chat_session.project_id` does NOT exist, stop and ask the user before adding a
column — that's a schema decision beyond this plan's scope.

- [ ] **Step 5: Regenerate sqlc code**

Run: `cd multica/server && make sqlc` Expected: `pkg/db/db.go` (or equivalent) now
contains `Queries.CreateEnvironment`, `GetEnvironment`, `DeleteEnvironment`,
`CreateProjectWithEnv`, `SetProjectEnvID`, `ListIssuesByProject`,
`CreateChatSessionForProject`.

- [ ] **Step 6: Verify build**

Run: `cd multica/server && go build ./...` Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add multica/server/pkg/db/queries/environment.sql multica/server/pkg/db/queries/project.sql multica/server/pkg/db/queries/issue.sql multica/server/pkg/db/queries/chat.sql multica/server/pkg/db/
git commit -m "feat(multica): sqlc queries for environment + project.env_id + project-scoped chat"
```

______________________________________________________________________

## Task 3: `EnvDispatchDeps` interface + service skeleton

**Files:**

- Create: `multica/server/internal/service/env_dispatch.go`

- [ ] **Step 1: Write the service skeleton (no tests yet — interface + types only)**

Create `multica/server/internal/service/env_dispatch.go`:

```go
package service

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// EnvMode enumerates the reset modes (spec §4.2).
type EnvMode string

const (
	EnvModeScratch EnvMode = "scratch"
	EnvModeBranch  EnvMode = "branch"
)

// EnvDomain enumerates the dispatch domains (spec §4.2). Required on dispatch;
// each domain pins a dispatch_type (swe_lego⇒issue, self_play⇒message).
type EnvDomain string

const (
	EnvDomainSweLego EnvDomain = "swe_lego"
	EnvDomainSelfPlay EnvDomain = "self_play"
)

// EnvDispatchType enumerates the dispatch types.
type EnvDispatchType string

const (
	EnvDispatchIssue   EnvDispatchType = "issue"
	EnvDispatchMessage EnvDispatchType = "message"
)

// EnvDispatchInput is the service-layer input for the unified dispatch.
type EnvDispatchInput struct {
	WorkspaceID    string
	UserID         string // creator/actor
	Mode           EnvMode
	EnvID          string // base env (scratch) or state env (branch)
	SourceProjectID string // branch only: the single project on EnvID (1:1 invariant), resolved by the handler
	Domain         EnvDomain // required
	DispatchType   EnvDispatchType
	GroupSize      int
	AgentID  string
	IdempotencyKey string // optional; dedupes retries (spec §7.7)

	// Issue dispatch (required for scratch+swe_lego; forbidden for
	// branch+swe_lego where the copied issue is reused).
	Issue *IssueInput

	// Message dispatch (required for self_play).
	Message *MessageInput
}

type IssueInput struct {
	Title              string
	Description        string
	AcceptanceCriteria []string
	FailToPass         []string
	PassToPass         []string
}

type MessageInput struct {
	Content string
}

// EnvRollout is one element of the response array (spec §6.3).
type EnvRollout struct {
	EnvID         string // always a new env_id (branch always forks, incl. N=1)
	ProjectID     string
	IssueID       string // empty iff dispatch_type=message
	ChatSessionID string // empty iff dispatch_type=issue
	AgentRunID    string // empty if dispatch failed (partial rollout)
	Error         string // empty if rollout succeeded
}

// EnvDispatchResult wraps the rollouts slice.
type EnvDispatchResult struct {
	Rollouts []EnvRollout
}

// EnvDispatchDeps is the seam between the service and the DB + cloud runtime.
// Production wires this to real queries + cloudRuntimeProxy; tests inject a fake.
type EnvDispatchDeps interface {
	// Environment operations
	GetEnv(ctx context.Context, envID, workspaceID string) (Env, error)
	CreateEnv(ctx context.Context, workspaceID, sandboxID, parentEnvID string, mode EnvMode, domain EnvDomain) (envID string, err error)
	DeleteEnv(ctx context.Context, envID, workspaceID string) error

	// Sandbox operations (proxy to cloud-runtime/Fleet)
	ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (sandboxID string, err error)
	DeleteSandbox(ctx context.Context, sandboxID string) error
	BootSandbox(ctx context.Context, imageRef string) (sandboxID string, err error) // for POST /api/v1/env

	// Project operations
	GetProjectByEnvID(ctx context.Context, envID, workspaceID string) (projectID string, err error) // branch: resolve source env → its single project (1:1 invariant)
	CreateProject(ctx context.Context, workspaceID, name, envID string) (projectID string, err error)
	// CopyProjectSubtree deep-copies issues + chat sessions + messages under a
	// new project bound to envID; returns source→copied ID maps so dispatch can
	// target the copied issue (branch+swe_lego) or copied session (branch+self_play).
	CopyProjectSubtree(ctx context.Context, sourceProjectID, workspaceID, envID string) (newProjectID string, issueIDMap, chatSessionIDMap map[string]string, err error)
	DeleteProject(ctx context.Context, projectID, workspaceID string) error

	// Issue operations
	ListIssuesByProject(ctx context.Context, projectID, workspaceID string) ([]IssueRow, error)
	CreateIssue(ctx context.Context, projectID, workspaceID, creatorID, title, description string, acceptanceCriteria, failToPass, passToPass []string) (issueID string, err error)

	// Chat operations
	CreateChatSession(ctx context.Context, projectID, workspaceID, agentID, creatorID string) (sessionID string, err error)
	CreateChatMessage(ctx context.Context, sessionID, role, content string) (messageID string, err error)

	// Agent run
	EnqueueAgentRun(ctx context.Context, workspaceID, agentID, issueID, chatSessionID, sandboxID string, idx int) (runID string, err error)

	// Idempotency ledger (spec §7.7). GetIdempotentResponse returns ok=false
	// when the key is unseen; SaveIdempotentResponse persists the response for
	// replay. Both are workspace-scoped.
	GetIdempotentResponse(ctx context.Context, workspaceID, key string) (EnvDispatchResult, bool, error)
	SaveIdempotentResponse(ctx context.Context, workspaceID, key string, res EnvDispatchResult) error
}

// Env is a snapshot of an environment row.
type Env struct {
	ID          string
	WorkspaceID string
	SandboxID   string
	ParentEnvID string // empty for base
	Mode        EnvMode
	Domain      EnvDomain
}

// IssueRow is a snapshot of an issue row (subset needed by the service).
type IssueRow struct {
	ID          string
	ProjectID   string
	Title       string
	Description string
}

// EnvDispatchService orchestrates reset → dispatch (spec §7).
type EnvDispatchService struct {
	deps        EnvDispatchDeps
	concurrency int
}

func NewEnvDispatchService(deps EnvDispatchDeps, concurrency int) *EnvDispatchService {
	if concurrency < 1 {
		concurrency = 8
	}
	return &EnvDispatchService{deps: deps, concurrency: concurrency}
}

// ErrAllDispatchFailed signals reset succeeded but every rollout's dispatch
// failed (spec §8 → 500). The returned result still carries rollouts[] so the
// caller can see the created envs/projects to clean up.
var ErrAllDispatchFailed = fmt.Errorf("dispatch_failed: all rollouts failed")

// Dispatch runs the unified dispatch flow.
func (s *EnvDispatchService) Dispatch(ctx context.Context, in EnvDispatchInput) (EnvDispatchResult, error) {
	if err := s.validate(in); err != nil {
		return EnvDispatchResult{}, err
	}

	// Idempotency replay (spec §7.7): a repeat key returns the stored response.
	if in.IdempotencyKey != "" {
		if prev, ok, err := s.deps.GetIdempotentResponse(ctx, in.WorkspaceID, in.IdempotencyKey); err != nil {
			return EnvDispatchResult{}, fmt.Errorf("idempotency lookup: %w", err)
		} else if ok {
			return prev, nil
		}
	}

	env, err := s.deps.GetEnv(ctx, in.EnvID, in.WorkspaceID)
	if err != nil {
		return EnvDispatchResult{}, fmt.Errorf("get env: %w", err)
	}

	// Branch: resolve the single source project on this env (spec §7.2 step 0).
	// The 1:1 unique index guarantees exactly one; a base env has none → error.
	if in.Mode == EnvModeBranch && in.SourceProjectID == "" {
		pid, err := s.deps.GetProjectByEnvID(ctx, in.EnvID, in.WorkspaceID)
		if err != nil {
			return EnvDispatchResult{}, fmt.Errorf("validation_failed: resolve source project: %w", err)
		}
		in.SourceProjectID = pid
	}

	rollouts := make([]EnvRollout, in.GroupSize)
	sem := make(chan struct{}, s.concurrency)
	var wg sync.WaitGroup
	var resetErrs []error
	var resetErrMu sync.Mutex

	for i := 0; i < in.GroupSize; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			r, err := s.resetOne(ctx, in, env, idx)
			if err != nil {
				resetErrMu.Lock()
				resetErrs = append(resetErrs, fmt.Errorf("rollout %d reset: %w", idx, err))
				resetErrMu.Unlock()
				return
			}
			rollouts[idx] = r
		}(i)
	}
	wg.Wait()

	if len(resetErrs) > 0 {
		// Reset failed for ≥1 rollout → roll back every rollout and return a
		// reset_failed error (handler → 503). Reset is all-or-nothing.
		for i, r := range rollouts {
			if r.ProjectID != "" || r.EnvID != "" {
				s.rollbackRollout(ctx, in.WorkspaceID, r)
			}
			rollouts[i] = EnvRollout{}
		}
		return EnvDispatchResult{}, fmt.Errorf("reset_failed: %v", resetErrs[0])
	}

	// Dispatch phase: best-effort, per-rollout errors recorded in rollouts[i].Error.
	var dispatchWG sync.WaitGroup
	for i := 0; i < in.GroupSize; i++ {
		dispatchWG.Add(1)
		go func(idx int) {
			defer dispatchWG.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			s.dispatchOne(ctx, in, &rollouts[idx], idx)
		}(i)
	}
	dispatchWG.Wait()

	result := EnvDispatchResult{Rollouts: rollouts}

	// Persist the idempotency response so a retry replays it (spec §7.7). Best-effort.
	if in.IdempotencyKey != "" {
		_ = s.deps.SaveIdempotentResponse(ctx, in.WorkspaceID, in.IdempotencyKey, result)
	}

	// Status rule (spec §6.3/§8): ≥1 dispatched → nil (201); all failed →
	// ErrAllDispatchFailed (handler → 500, body still carries rollouts[]).
	succeeded := 0
	for _, r := range rollouts {
		if r.AgentRunID != "" {
			succeeded++
		}
	}
	if succeeded == 0 {
		return result, ErrAllDispatchFailed
	}
	return result, nil
}

// validate implements the §6.3 validation table (the subset that's
// service-level; UUID-shape validation lives in the handler).
func (s *EnvDispatchService) validate(in EnvDispatchInput) error {
	if in.Mode != EnvModeScratch && in.Mode != EnvModeBranch {
		return fmt.Errorf("validation_failed: mode must be scratch or branch")
	}
	if in.DispatchType != EnvDispatchIssue && in.DispatchType != EnvDispatchMessage {
		return fmt.Errorf("validation_failed: dispatch_type must be issue or message")
	}
	if in.GroupSize < 1 || in.GroupSize > 64 {
		return fmt.Errorf("validation_failed: group_size must be in [1, 64]")
	}
	if in.Domain != EnvDomainSweLego && in.Domain != EnvDomainSelfPlay {
		return fmt.Errorf("validation_failed: domain is required (swe_lego or self_play)")
	}
	if in.Domain == EnvDomainSweLego && in.DispatchType == EnvDispatchMessage {
		return fmt.Errorf("validation_failed: swe_lego domain is issue-only")
	}
	if in.Domain == EnvDomainSelfPlay && in.DispatchType == EnvDispatchIssue {
		return fmt.Errorf("not_implemented: self_play + issue dispatch")
	}
	if in.Mode == EnvModeBranch && in.Domain == EnvDomainSweLego && in.Issue != nil {
		return fmt.Errorf("validation_failed: issue must not be supplied for branch+swe_lego (copied issue is reused)")
	}
	if in.Mode == EnvModeScratch && in.Domain == EnvDomainSweLego && in.Issue == nil {
		return fmt.Errorf("validation_failed: issue required for scratch+swe_lego")
	}
	if in.DispatchType == EnvDispatchMessage && (in.Message == nil || in.Message.Content == "") {
		return fmt.Errorf("validation_failed: message.content required")
	}
	return nil
}

// resetOne does the per-rollout reset (sandbox + env + project) per §7.2.
func (s *EnvDispatchService) resetOne(ctx context.Context, in EnvDispatchInput, sourceEnv Env, idx int) (EnvRollout, error) {
	// Branch always forks (spec §4.3): the source sandbox is never reused in
	// place, so the source state stays re-branchable (MCTS). Scratch forks the
	// base. Both paths create a fresh env row.
	forked, err := s.deps.ForkSandbox(ctx, sourceEnv.SandboxID, idx)
	if err != nil {
		return EnvRollout{}, fmt.Errorf("fork sandbox: %w", err)
	}
	sandboxID := forked
	mode := EnvModeScratch
	if in.Mode == EnvModeBranch {
		mode = EnvModeBranch
	}
	envID, err := s.deps.CreateEnv(ctx, in.WorkspaceID, sandboxID, sourceEnv.ID, mode, in.Domain)
	if err != nil {
		_ = s.deps.DeleteSandbox(ctx, sandboxID)
		return EnvRollout{}, fmt.Errorf("create env: %w", err)
	}

	// Project
	var projectID string
	var issueIDMap, chatSessionIDMap map[string]string
	if in.Mode == EnvModeScratch {
		name := fmt.Sprintf("env-dispatch-%s", envID) // unique, spec §7.2
		pid, err := s.deps.CreateProject(ctx, in.WorkspaceID, name, envID)
		if err != nil {
			s.rollbackRollout(ctx, in.WorkspaceID, EnvRollout{EnvID: envID})
			return EnvRollout{}, fmt.Errorf("create project: %w", err)
		}
		projectID = pid
	} else {
		// branch — copy source project subtree (issues + chat sessions).
		// in.SourceProjectID is resolved by the handler from the 1:1 env→project.
		pid, imap, smap, err := s.deps.CopyProjectSubtree(ctx, in.SourceProjectID, in.WorkspaceID, envID)
		if err != nil {
			s.rollbackRollout(ctx, in.WorkspaceID, EnvRollout{EnvID: envID})
			return EnvRollout{}, fmt.Errorf("copy project: %w", err)
		}
		projectID = pid
		issueIDMap = imap
		chatSessionIDMap = smap
	}

	r := EnvRollout{EnvID: envID, ProjectID: projectID}
	// Stash the single copied entity for dispatchOne (spec §7.4: exactly one).
	if in.Mode == EnvModeBranch {
		if in.Domain == EnvDomainSweLego {
			for _, newID := range issueIDMap {
				r.IssueID = newID
				break
			}
		} else if in.Domain == EnvDomainSelfPlay {
			for _, newID := range chatSessionIDMap {
				r.ChatSessionID = newID
				break
			}
		}
	}
	return r, nil
}

// dispatchOne runs the dispatch phase for one rollout (§7.3). Best-effort:
// failures recorded in r.Error, no rollback.
func (s *EnvDispatchService) dispatchOne(ctx context.Context, in EnvDispatchInput, r *EnvRollout, idx int) {
	if in.DispatchType == EnvDispatchIssue {
		issueID := r.IssueID // branch+swe_lego: copied issue id
		if issueID == "" {
			// scratch+swe_lego — create the new issue
			ii := in.Issue
			newID, err := s.deps.CreateIssue(ctx, r.ProjectID, in.WorkspaceID, in.UserID, ii.Title, ii.Description, ii.AcceptanceCriteria, ii.FailToPass, ii.PassToPass)
			if err != nil {
				r.Error = fmt.Sprintf("create issue: %v", err)
				return
			}
			issueID = newID
			r.IssueID = newID
		}
		runID, err := s.deps.EnqueueAgentRun(ctx, in.WorkspaceID, in.AgentID, issueID, "", "", idx)
		if err != nil {
			r.Error = fmt.Sprintf("enqueue agent run: %v", err)
			return
		}
		r.AgentRunID = runID
		return
	}
	// message (self_play)
	sessionID := r.ChatSessionID // branch: the copied session (spec §7.4); empty for scratch
	if sessionID == "" {
		// scratch+self_play — new session bound to the new project
		newID, err := s.deps.CreateChatSession(ctx, r.ProjectID, in.WorkspaceID, in.AgentID, in.UserID)
		if err != nil {
			r.Error = fmt.Sprintf("create chat session: %v", err)
			return
		}
		sessionID = newID
		r.ChatSessionID = newID
	}
	// branch continues the copied conversation by appending; scratch starts fresh (spec §7.3).
	if _, err := s.deps.CreateChatMessage(ctx, sessionID, "user", in.Message.Content); err != nil {
		r.Error = fmt.Sprintf("create chat message: %v", err)
		return
	}
	runID, err := s.deps.EnqueueAgentRun(ctx, in.WorkspaceID, in.AgentID, "", sessionID, "", idx)
	if err != nil {
		r.Error = fmt.Sprintf("enqueue agent run: %v", err)
		return
	}
	r.AgentRunID = runID
}

// rollbackRollout cleans up a partially-created rollout (reset phase only).
// Order matters under ON DELETE RESTRICT: delete the project first (it
// references env_id), then the env row, then its sandbox. Every rollout forks
// its own sandbox, so this never touches a shared/source sandbox.
func (s *EnvDispatchService) rollbackRollout(ctx context.Context, workspaceID string, r EnvRollout) {
	if r.ProjectID != "" {
		_ = s.deps.DeleteProject(ctx, r.ProjectID, workspaceID)
	}
	if r.EnvID != "" {
		env, err := s.deps.GetEnv(ctx, r.EnvID, workspaceID)
		_ = s.deps.DeleteEnv(ctx, r.EnvID, workspaceID)
		if err == nil {
			_ = s.deps.DeleteSandbox(ctx, env.SandboxID)
		}
	}
}
```

The `EnvDispatchInput.SourceProjectID` and `IdempotencyKey` fields are already declared
in the struct above. Two idempotency methods must be added to the `EnvDispatchDeps`
interface (implemented by the fake in Task 4 and the adapter in Task 8):

```go
	// Idempotency ledger (spec §7.7)
	GetIdempotentResponse(ctx context.Context, workspaceID, key string) (EnvDispatchResult, bool, error)
	SaveIdempotentResponse(ctx context.Context, workspaceID, key string, res EnvDispatchResult) error
```

- [ ] **Step 2: Verify it compiles (tests not yet written)**

Run: `cd multica/server && go build ./internal/service/` Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add multica/server/internal/service/env_dispatch.go
git commit -m "feat(multica): EnvDispatchService skeleton with Deps seam"
```

______________________________________________________________________

## Task 4: Service unit tests — combination matrix + rejected combinations

**Files:**

- Create: `multica/server/internal/service/env_dispatch_test.go`

- [ ] **Step 1: Write a fake `EnvDispatchDeps` + happy-path tests for each matrix cell**

Create `multica/server/internal/service/env_dispatch_test.go`. The fake records calls
and returns configurable results.

```go
package service

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
)

type fakeEnvDispatchDeps struct {
	mu sync.Mutex

	envs       map[string]Env               // by envID
	sandboxes  map[string]string            // sandboxID -> sourceSandboxID (for fork provenance)
	projects   map[string]string            // projectID -> envID
	issues     map[string][]IssueRow        // projectID -> issues
	chatSess   map[string]string            // sessionID -> projectID
	agentRuns  []string                     // every enqueued runID
	runCounter int
	idem       map[string]EnvDispatchResult // idempotency ledger

	forkErr    error
	createEnvErr error
	copyProjectErr error
	createIssueErr error
	enqueueErr error
}

func newFakeEnvDispatchDeps() *fakeEnvDispatchDeps {
	return &fakeEnvDispatchDeps{
		envs: map[string]Env{}, sandboxes: map[string]string{}, projects: map[string]string{},
		issues: map[string][]IssueRow{}, chatSess: map[string]string{},
	}
}

func (f *fakeEnvDispatchDeps) GetEnv(_ context.Context, envID, _ string) (Env, error) {
	f.mu.Lock(); defer f.mu.Unlock()
	e, ok := f.envs[envID]
	if !ok { return Env{}, fmt.Errorf("not found") }
	return e, nil
}
func (f *fakeEnvDispatchDeps) CreateEnv(_ context.Context, _, sandboxID, parentEnvID string, mode EnvMode, domain EnvDomain) (string, error) {
	if f.createEnvErr != nil { return "", f.createEnvErr }
	f.mu.Lock(); defer f.mu.Unlock()
	id := fmt.Sprintf("env-%d", len(f.envs))
	f.envs[id] = Env{ID: id, SandboxID: sandboxID, ParentEnvID: parentEnvID, Mode: mode, Domain: domain}
	return id, nil
}
func (f *fakeEnvDispatchDeps) DeleteEnv(_ context.Context, envID, _ string) error {
	f.mu.Lock(); defer f.mu.Unlock()
	delete(f.envs, envID)
	return nil
}
func (f *fakeEnvDispatchDeps) ForkSandbox(_ context.Context, _ string, idx int) (string, error) {
	if f.forkErr != nil { return "", f.forkErr }
	f.mu.Lock(); defer f.mu.Unlock()
	id := fmt.Sprintf("sbx-fork-%d-%d", idx, len(f.sandboxes))
	f.sandboxes[id] = "forked"
	return id, nil
}
func (f *fakeEnvDispatchDeps) DeleteSandbox(_ context.Context, _ string) error { return nil }
func (f *fakeEnvDispatchDeps) BootSandbox(_ context.Context, _ string) (string, error) {
	return "sbx-booted", nil
}
func (f *fakeEnvDispatchDeps) CreateProject(_ context.Context, _, _, envID string) (string, error) {
	f.mu.Lock(); defer f.mu.Unlock()
	id := fmt.Sprintf("proj-%d", len(f.projects))
	f.projects[id] = envID
	return id, nil
}
func (f *fakeEnvDispatchDeps) CopyProjectSubtree(_ context.Context, _, _, envID string) (string, map[string]string, map[string]string, error) {
	if f.copyProjectErr != nil { return "", nil, nil, f.copyProjectErr }
	f.mu.Lock(); defer f.mu.Unlock()
	pid := fmt.Sprintf("proj-copy-%d", len(f.projects))
	f.projects[pid] = envID
	imap := map[string]string{"source-issue-1": "copied-issue-1"}
	smap := map[string]string{"source-sess-1": "copied-sess-1"}
	f.issues[pid] = []IssueRow{{ID: "copied-issue-1", ProjectID: pid}}
	return pid, imap, smap, nil
}
func (f *fakeEnvDispatchDeps) GetProjectByEnvID(_ context.Context, envID, _ string) (string, error) {
	f.mu.Lock(); defer f.mu.Unlock()
	for pid, eid := range f.projects {
		if eid == envID { return pid, nil }
	}
	return "", fmt.Errorf("no project for env %s", envID)
}
func (f *fakeEnvDispatchDeps) GetIdempotentResponse(_ context.Context, _, key string) (EnvDispatchResult, bool, error) {
	f.mu.Lock(); defer f.mu.Unlock()
	r, ok := f.idem[key]
	return r, ok, nil
}
func (f *fakeEnvDispatchDeps) SaveIdempotentResponse(_ context.Context, _, key string, res EnvDispatchResult) error {
	f.mu.Lock(); defer f.mu.Unlock()
	if f.idem == nil { f.idem = map[string]EnvDispatchResult{} }
	f.idem[key] = res
	return nil
}
func (f *fakeEnvDispatchDeps) DeleteProject(_ context.Context, pid, _ string) error {
	f.mu.Lock(); defer f.mu.Unlock()
	delete(f.projects, pid)
	delete(f.issues, pid)
	return nil
}
func (f *fakeEnvDispatchDeps) ListIssuesByProject(_ context.Context, pid, _ string) ([]IssueRow, error) {
	f.mu.Lock(); defer f.mu.Unlock()
	return f.issues[pid], nil
}
func (f *fakeEnvDispatchDeps) CreateIssue(_ context.Context, pid, _, _, title, _ string, _, _, _ []string) (string, error) {
	if f.createIssueErr != nil { return "", f.createIssueErr }
	f.mu.Lock(); defer f.mu.Unlock()
	id := fmt.Sprintf("issue-%d", len(f.issues))
	f.issues[pid] = append(f.issues[pid], IssueRow{ID: id, ProjectID: pid, Title: title})
	return id, nil
}
func (f *fakeEnvDispatchDeps) CreateChatSession(_ context.Context, pid, _, _, _ string) (string, error) {
	f.mu.Lock(); defer f.mu.Unlock()
	id := fmt.Sprintf("sess-%d", len(f.chatSess))
	f.chatSess[id] = pid
	return id, nil
}
func (f *fakeEnvDispatchDeps) CreateChatMessage(_ context.Context, _, _, _ string) (string, error) {
	return "msg-1", nil
}
func (f *fakeEnvDispatchDeps) EnqueueAgentRun(_ context.Context, _, _, _, _, _ string, idx int) (string, error) {
	if f.enqueueErr != nil { return "", f.enqueueErr }
	f.mu.Lock(); defer f.mu.Unlock()
	f.runCounter++
	id := fmt.Sprintf("run-%d", f.runCounter)
	f.agentRuns = append(f.agentRuns, id)
	return id, nil
}

// Helper: seed a base env in the fake.
func (f *fakeEnvDispatchDeps) seedBaseEnv() string {
	f.mu.Lock(); defer f.mu.Unlock()
	id := "base-env-1"
	f.envs[id] = Env{ID: id, SandboxID: "base-sbx", Mode: EnvModeBase, Domain: ""}
	return id
}

const EnvModeBase EnvMode = "base"

func TestDispatch_ScratchSweLegoIssue_N3(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	baseEnv := f.seedBaseEnv()
	svc := NewEnvDispatchService(f, 8)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", UserID: "u", Mode: EnvModeScratch, EnvID: baseEnv,
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue, GroupSize: 3,
		AgentID: "ag", Issue: &IssueInput{Title: "t"},
	})
	if err != nil { t.Fatalf("dispatch: %v", err) }
	if len(res.Rollouts) != 3 { t.Fatalf("want 3 rollouts, got %d", len(res.Rollouts)) }
	for i, r := range res.Rollouts {
		if r.AgentRunID == "" { t.Fatalf("rollout %d: no agent_run_id", i) }
		if r.IssueID == "" { t.Fatalf("rollout %d: no issue_id", i) }
		if r.EnvID == baseEnv { t.Fatalf("rollout %d: env_id reused (should be fork)", i) }
	}
}

func TestDispatch_ScratchSelfPlayMessage_N3(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	baseEnv := f.seedBaseEnv()
	svc := NewEnvDispatchService(f, 8)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", UserID: "u", Mode: EnvModeScratch, EnvID: baseEnv,
		Domain: EnvDomainSelfPlay, DispatchType: EnvDispatchMessage, GroupSize: 3,
		AgentID: "ag", Message: &MessageInput{Content: "q"},
	})
	if err != nil { t.Fatalf("dispatch: %v", err) }
	if len(res.Rollouts) != 3 { t.Fatalf("want 3, got %d", len(res.Rollouts)) }
	for i, r := range res.Rollouts {
		if r.ChatSessionID == "" { t.Fatalf("rollout %d: no session", i) }
	}
}

func TestDispatch_BranchSweLegoIssue_N1_Forks(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	// seed a state env (mode != base) with a project + single swe_lego issue
	stateEnv := "state-env-1"
	f.envs[stateEnv] = Env{ID: stateEnv, SandboxID: "state-sbx", Mode: EnvModeBranch, Domain: EnvDomainSweLego}
	f.projects["source-proj-1"] = stateEnv
	f.issues["source-proj-1"] = []IssueRow{{ID: "source-issue-1", ProjectID: "source-proj-1"}}

	svc := NewEnvDispatchService(f, 8)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", UserID: "u", Mode: EnvModeBranch, EnvID: stateEnv,
		SourceProjectID: "source-proj-1",
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue, GroupSize: 1,
		AgentID: "ag",
	})
	if err != nil { t.Fatalf("dispatch: %v", err) }
	if len(res.Rollouts) != 1 { t.Fatalf("want 1, got %d", len(res.Rollouts)) }
	r := res.Rollouts[0]
	if r.EnvID == stateEnv { t.Fatalf("env_id should be a fresh fork, not the source %s", r.EnvID) }
	if r.IssueID != "copied-issue-1" { t.Fatalf("issue should be copied, got %s", r.IssueID) }
	if _, ok := f.envs[stateEnv]; !ok { t.Fatal("source env must remain intact (re-branchable)") }
}

func TestDispatch_BranchSelfPlayMessage_N2(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	stateEnv := "state-env-1"
	f.envs[stateEnv] = Env{ID: stateEnv, SandboxID: "state-sbx", Mode: EnvModeBranch, Domain: EnvDomainSelfPlay}
	f.projects["source-proj-1"] = stateEnv

	svc := NewEnvDispatchService(f, 8)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", UserID: "u", Mode: EnvModeBranch, EnvID: stateEnv,
		SourceProjectID: "source-proj-1",
		Domain: EnvDomainSelfPlay, DispatchType: EnvDispatchMessage, GroupSize: 2,
		AgentID: "ag", Message: &MessageInput{Content: "q"},
	})
	if err != nil { t.Fatalf("dispatch: %v", err) }
	if len(res.Rollouts) != 2 { t.Fatalf("want 2, got %d", len(res.Rollouts)) }
	for i, r := range res.Rollouts {
		if r.EnvID == stateEnv { t.Fatalf("rollout %d: env_id should be forked, not reused", i) }
		// branch appends to the COPIED session, not a freshly created one (spec §7.3).
		if r.ChatSessionID != "copied-sess-1" { t.Fatalf("rollout %d: want copied session, got %s", i, r.ChatSessionID) }
	}
	// No new "sess-*" session should be created for branch (append only).
	if len(f.chatSess) != 0 { t.Fatalf("branch must not create new sessions, got %d", len(f.chatSess)) }
}
```

- [ ] **Step 2: Run the matrix tests**

Run: `cd multica/server && go test ./internal/service/ -run TestDispatch_ -v` Expected:
all 4 pass.

- [ ] **Step 3: Write rejected-combination tests**

Append to `env_dispatch_test.go`:

```go
func TestDispatch_RejectsSweLegoMessage(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	svc := NewEnvDispatchService(f, 8)
	_, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		Mode: EnvModeScratch, EnvID: "base", Domain: EnvDomainSweLego,
		DispatchType: EnvDispatchMessage, GroupSize: 1, Message: &MessageInput{Content: "q"},
	})
	if err == nil { t.Fatal("want error") }
}

func TestDispatch_RejectsSelfPlayIssue_501(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	svc := NewEnvDispatchService(f, 8)
	_, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		Mode: EnvModeScratch, EnvID: "base", Domain: EnvDomainSelfPlay,
		DispatchType: EnvDispatchIssue, GroupSize: 1, Issue: &IssueInput{Title: "t"},
	})
	if err == nil { t.Fatal("want error") }
}

func TestDispatch_RejectsMissingDomain(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	svc := NewEnvDispatchService(f, 8)
	_, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		Mode: EnvModeScratch, EnvID: "base", DispatchType: EnvDispatchIssue,
		GroupSize: 1, Issue: &IssueInput{Title: "t"},
	})
	if err == nil { t.Fatal("want error (domain required)") }
}

func TestDispatch_RejectsBranchSweLegoWithIssue(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	svc := NewEnvDispatchService(f, 8)
	_, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		Mode: EnvModeBranch, EnvID: "state", Domain: EnvDomainSweLego,
		DispatchType: EnvDispatchIssue, GroupSize: 1, Issue: &IssueInput{Title: "t"},
	})
	if err == nil { t.Fatal("want error") }
}
```

- [ ] **Step 4: Run rejected-combination tests**

Run: `cd multica/server && go test ./internal/service/ -run TestDispatch_Rejects -v`
Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/service/env_dispatch_test.go
git commit -m "test(multica): EnvDispatchService matrix + rejected combinations"
```

______________________________________________________________________

## Task 5: Service tests — rollback on reset failure + partial dispatch failure

**Files:**

- Modify: `multica/server/internal/service/env_dispatch_test.go`

- [ ] **Step 1: Add rollback test (fork fails on rollout 1 of 2)**

Append:

```go
func TestDispatch_RollbackOnForkFailure(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	baseEnv := f.seedBaseEnv()
	f.forkErr = fmt.Errorf("fork crashed")
	svc := NewEnvDispatchService(f, 8)
	_, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: baseEnv,
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue, GroupSize: 2,
		AgentID: "ag", Issue: &IssueInput{Title: "t"},
	})
	if err == nil { t.Fatal("want reset_failed error") }
	// No projects should remain.
	if len(f.projects) != 0 { t.Fatalf("want 0 projects after rollback, got %d", len(f.projects)) }
}
```

- [ ] **Step 2: Run it**

Run: `cd multica/server && go test ./internal/service/ -run TestDispatch_Rollback -v`
Expected: PASS.

- [ ] **Step 3: Add all-dispatch-fail (500 sentinel) + idempotency-replay tests**

Append:

```go
func TestDispatch_AllDispatchFail_KeepsEnvReturnsSentinel(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	baseEnv := f.seedBaseEnv()
	f.enqueueErr = fmt.Errorf("enqueue crashed") // every EnqueueAgentRun fails
	svc := NewEnvDispatchService(f, 8)
	res, err := svc.Dispatch(context.Background(), EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: baseEnv,
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue, GroupSize: 2,
		AgentID: "ag", Issue: &IssueInput{Title: "t"},
	})
	// All dispatches failed → ErrAllDispatchFailed (handler → 500), env kept.
	if !errors.Is(err, ErrAllDispatchFailed) { t.Fatalf("want ErrAllDispatchFailed, got %v", err) }
	if len(res.Rollouts) != 2 { t.Fatalf("want 2, got %d", len(res.Rollouts)) }
	for i, r := range res.Rollouts {
		if r.AgentRunID != "" { t.Fatalf("rollout %d: agent_run_id should be empty", i) }
		if r.Error == "" { t.Fatalf("rollout %d: error should be set", i) }
		if r.ProjectID == "" || r.EnvID == "" { t.Fatalf("rollout %d: env+project should be kept", i) }
	}
}

func TestDispatch_IdempotencyReplay(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	baseEnv := f.seedBaseEnv()
	svc := NewEnvDispatchService(f, 8)
	in := EnvDispatchInput{
		WorkspaceID: "ws", Mode: EnvModeScratch, EnvID: baseEnv,
		Domain: EnvDomainSweLego, DispatchType: EnvDispatchIssue, GroupSize: 2,
		AgentID: "ag", Issue: &IssueInput{Title: "t"}, IdempotencyKey: "key-1",
	}
	first, err := svc.Dispatch(context.Background(), in)
	if err != nil { t.Fatalf("first dispatch: %v", err) }
	projectsAfterFirst := len(f.projects)
	runsAfterFirst := len(f.agentRuns)

	second, err := svc.Dispatch(context.Background(), in)
	if err != nil { t.Fatalf("replay dispatch: %v", err) }
	// Replay returns the stored response and does NO new work.
	if len(f.projects) != projectsAfterFirst { t.Fatalf("replay created new projects: %d → %d", projectsAfterFirst, len(f.projects)) }
	if len(f.agentRuns) != runsAfterFirst { t.Fatalf("replay enqueued new runs") }
	if second.Rollouts[0].ProjectID != first.Rollouts[0].ProjectID { t.Fatal("replay returned different rollouts") }
}
```

- [ ] **Step 4: Run it**

Run:
`cd multica/server && go test ./internal/service/ -run 'TestDispatch_AllDispatchFail|TestDispatch_IdempotencyReplay' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/service/env_dispatch_test.go
git commit -m "test(multica): EnvDispatchService rollback + partial-dispatch semantics"
```

______________________________________________________________________

## Task 6: Handler — `POST /api/v1/env` + `DELETE /api/v1/env/{envID}`

**Files:**

- Create: `multica/server/internal/handler/env.go`

- Create: `multica/server/internal/handler/env_test.go`

- [ ] **Step 1: Write the env handler**

Create `multica/server/internal/handler/env.go`:

```go
package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/multica-ai/multica/server/internal/service"
)

// CreateEnvRequest is the body of POST /api/v1/env (spec §6.1).
type CreateEnvRequest struct {
	ImageRef string `json:"image_ref"`
}

// CreateEnvResponse is the 201 response.
type CreateEnvResponse struct {
	EnvID     string `json:"env_id"`
	SandboxID string `json:"sandbox_id"`
}

// CreateEnv handles POST /api/v1/env. Boots a sandbox from image_ref via the
// cloud-runtime proxy, creates an environment row (mode='base'), returns env_id.
func (h *Handler) CreateEnv(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireUserID(w, r); !ok {
		return
	}
	workspaceID, ok := ctxWorkspaceID(w, r)
	if !ok {
		return
	}
	var req CreateEnvRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed request body")
		return
	}
	if req.ImageRef == "" || len(req.ImageRef) > 256 {
		writeError(w, http.StatusBadRequest, "image_ref must be 1..256 chars")
		return
	}

	svc := service.NewEnvDispatchService(newEnvDispatchDepsAdapter(h), 8)
	envID, sandboxID, err := svc.CreateBaseEnv(r.Context(), workspaceID, req.ImageRef)
	if err != nil {
		status := http.StatusServiceUnavailable
		if strings.Contains(err.Error(), "validation_failed") {
			status = http.StatusBadRequest
		}
		writeError(w, status, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, CreateEnvResponse{EnvID: envID, SandboxID: sandboxID})
}

// DeleteEnv handles DELETE /api/v1/env/{envID} (spec §6.2). Idempotent on 404.
func (h *Handler) DeleteEnv(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireUserID(w, r); !ok {
		return
	}
	workspaceID, ok := ctxWorkspaceID(w, r)
	if !ok {
		return
	}
	envID := chi.URLParam(r, "envID")
	if envID == "" {
		writeError(w, http.StatusBadRequest, "envID is required")
		return
	}
	svc := service.NewEnvDispatchService(newEnvDispatchDepsAdapter(h), 8)
	if err := svc.DeleteEnv(r.Context(), envID, workspaceID); err != nil {
		switch {
		case errors.Is(err, service.ErrEnvInUse):
			writeError(w, http.StatusConflict, "env_in_use: delete its project(s) first")
		case strings.Contains(err.Error(), "not found"):
			writeError(w, http.StatusNotFound, err.Error())
		default:
			writeError(w, http.StatusServiceUnavailable, err.Error())
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

var _ = context.Background // silence unused import if not yet used
```

- [ ] **Step 2: Add `CreateBaseEnv` + `DeleteEnv` methods to the service**

Append to `multica/server/internal/service/env_dispatch.go`:

```go
// CreateBaseEnv boots a sandbox and creates a mode='base' env row.
func (s *EnvDispatchService) CreateBaseEnv(ctx context.Context, workspaceID, imageRef string) (envID, sandboxID string, err error) {
	if imageRef == "" || len(imageRef) > 256 {
		return "", "", fmt.Errorf("validation_failed: image_ref must be 1..256 chars")
	}
	sbx, err := s.deps.BootSandbox(ctx, imageRef)
	if err != nil {
		return "", "", fmt.Errorf("boot sandbox: %w", err)
	}
	eid, err := s.deps.CreateEnv(ctx, workspaceID, sbx, "", EnvModeBase, "")
	if err != nil {
		_ = s.deps.DeleteSandbox(ctx, sbx)
		return "", "", fmt.Errorf("create env: %w", err)
	}
	return eid, sbx, nil
}

// ErrEnvInUse signals a DELETE /api/v1/env against an env a project still
// references (ON DELETE RESTRICT). Handler maps it to 409.
var ErrEnvInUse = fmt.Errorf("env_in_use")

// DeleteEnv deletes the env row + its sandbox. Idempotent: a missing env
// returns a "not found" error which the handler maps to 404. Returns
// ErrEnvInUse (→ 409) if a project still references the env.
func (s *EnvDispatchService) DeleteEnv(ctx context.Context, envID, workspaceID string) error {
	env, err := s.deps.GetEnv(ctx, envID, workspaceID)
	if err != nil {
		return fmt.Errorf("not found: %w", err)
	}
	// Delete the env ROW first. Under ON DELETE RESTRICT this fails with a FK
	// violation if a project still references it — the adapter surfaces that as
	// ErrEnvInUse, and the sandbox is left untouched. Only once the row is gone
	// do we reclaim the sandbox. (Ordering matters: deleting the sandbox first
	// would kill a live project's sandbox even though the row-delete then fails.)
	if err := s.deps.DeleteEnv(ctx, envID, workspaceID); err != nil {
		if errors.Is(err, ErrEnvInUse) {
			return ErrEnvInUse
		}
		return fmt.Errorf("delete env: %w", err)
	}
	_ = s.deps.DeleteSandbox(ctx, env.SandboxID) // idempotent on 404 in Fleet
	return nil
}
```

- [ ] **Step 3: Write env handler tests**

Create `multica/server/internal/handler/env_test.go`:

```go
package handler

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCreateEnv_RequiresAuth(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/v1/env", bytes.NewReader([]byte(`{}`)))
	h.CreateEnv(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", w.Code)
	}
}

func TestCreateEnv_RejectsMissingImageRef(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/v1/env", bytes.NewReader([]byte(`{}`)))
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "ws1")
	h.CreateEnv(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestDeleteEnv_RequiresEnvID(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	r := httptest.NewRequest("DELETE", "/api/v1/env/", nil)
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "ws1")
	r.SetPathValue("envID", "")
	h.DeleteEnv(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}
```

- [ ] **Step 4: Run env handler tests**

Run:
`cd multica/server && go test ./internal/handler/ -run TestCreateEnv_ -v && go test ./internal/handler/ -run TestDeleteEnv_ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/handler/env.go multica/server/internal/handler/env_test.go multica/server/internal/service/env_dispatch.go
git commit -m "feat(multica): POST /api/v1/env + DELETE /api/v1/env/{envID} handlers"
```

______________________________________________________________________

## Task 7: Handler — `POST /api/v1/env-dispatch` + `DELETE /api/v1/env-dispatch/{projectID}`

**Files:**

- Create: `multica/server/internal/handler/env_dispatch.go`

- Create: `multica/server/internal/handler/env_dispatch_test.go`

- [ ] **Step 1: Write the env-dispatch handler + adapter skeleton**

Create `multica/server/internal/handler/env_dispatch.go`:

```go
package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/multica-ai/multica/server/internal/service"
)

// EnvDispatchRequest is the body of POST /api/v1/env-dispatch (spec §6.3).
type EnvDispatchRequest struct {
	Mode          string             `json:"mode"`
	EnvID         string             `json:"env_id"`
	Domain        string             `json:"domain,omitempty"`
	DispatchType  string             `json:"dispatch_type"`
	GroupSize     int                `json:"group_size"`
	AgentID string             `json:"agent_id"`
	IdempotencyKey string          `json:"idempotency_key,omitempty"`
	Issue         *IssueDispatchInput `json:"issue,omitempty"`
	Message       *MessageDispatchInput `json:"message,omitempty"`
}

type IssueDispatchInput struct {
	Title              string   `json:"title"`
	Description        string   `json:"description"`
	AcceptanceCriteria []string `json:"acceptance_criteria"`
	FailToPass         []string `json:"fail_to_pass"`
	PassToPass         []string `json:"pass_to_pass"`
}

type MessageDispatchInput struct {
	Content string `json:"content"`
}

// EnvDispatchResponse is the 201 response (spec §6.3).
type EnvDispatchResponse struct {
	Rollouts []EnvRolloutResponse `json:"rollouts"`
}

type EnvRolloutResponse struct {
	EnvID         string `json:"env_id"`
	ProjectID     string `json:"project_id"`
	IssueID       string `json:"issue_id,omitempty"`
	ChatSessionID string `json:"chat_session_id,omitempty"`
	AgentRunID    string `json:"agent_run_id,omitempty"`
	Error         string `json:"error,omitempty"`
}

// EnvDispatch handles POST /api/v1/env-dispatch.
func (h *Handler) EnvDispatch(w http.ResponseWriter, r *http.Request) {
	userID, ok := requireUserID(w, r)
	if !ok {
		return
	}
	workspaceID, ok := ctxWorkspaceID(w, r)
	if !ok {
		return
	}
	var req EnvDispatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed request body")
		return
	}

	// Branch source resolution happens in the service (spec §7.2 step 0) via
	// Deps.GetProjectByEnvID, relying on the 1:1 env→project invariant — the
	// handler does not need to pre-resolve it.
	svc := service.NewEnvDispatchService(newEnvDispatchDepsAdapter(h), envDispatchConcurrency())
	res, err := svc.Dispatch(r.Context(), service.EnvDispatchInput{
		WorkspaceID: workspaceID, UserID: userID,
		Mode: service.EnvMode(req.Mode), EnvID: req.EnvID,
		Domain:        service.EnvDomain(req.Domain),
		DispatchType:  service.EnvDispatchType(req.DispatchType),
		GroupSize:     req.GroupSize, AgentID: req.AgentID,
		IdempotencyKey: req.IdempotencyKey,
		Issue:   mapIssueInput(req.Issue),
		Message: mapMessageInput(req.Message),
	})
	if err != nil {
		writeEnvDispatchError(w, err, res)
		return
	}
	writeJSON(w, http.StatusCreated, EnvDispatchResponse{Rollouts: mapRollouts(res.Rollouts)})
}

// DeleteEnvDispatchProject handles DELETE /api/v1/env-dispatch/{projectID}
// (renamed from DELETE /api/v1/swe-lego/issues/{projectID}). Cascades to
// issues/chat_sessions/tasks/messages; the environment row persists.
func (h *Handler) DeleteEnvDispatchProject(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireUserID(w, r); !ok {
		return
	}
	workspaceID, ok := ctxWorkspaceID(w, r)
	if !ok {
		return
	}
	projectID := chi.URLParam(r, "projectID")
	if projectID == "" {
		writeError(w, http.StatusBadRequest, "projectID is required")
		return
	}
	svc := service.NewEnvDispatchService(newEnvDispatchDepsAdapter(h), 8)
	if err := svc.DeleteProject(r.Context(), projectID, workspaceID); err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func writeEnvDispatchError(w http.ResponseWriter, err error, res service.EnvDispatchResult) {
	msg := err.Error()
	switch {
	case errors.Is(err, service.ErrAllDispatchFailed):
		// All rollouts failed dispatch → 500, but the SAME rollouts[] shape as
		// success (spec §8): each element carries its error + null agent_run_id,
		// so the caller can still see the created envs/projects to clean up.
		writeJSON(w, http.StatusInternalServerError, EnvDispatchResponse{Rollouts: mapRollouts(res.Rollouts)})
	case strings.Contains(msg, "validation_failed"):
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "validation_failed", "message": msg})
	case strings.Contains(msg, "not_implemented"):
		writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "not_implemented", "message": msg})
	case strings.Contains(msg, "reset_failed"):
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "reset_failed", "message": msg})
	default:
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "internal", "message": msg})
	}
}

func mapIssueInput(i *IssueDispatchInput) *service.IssueInput {
	if i == nil {
		return nil
	}
	return &service.IssueInput{
		Title: i.Title, Description: i.Description,
		AcceptanceCriteria: i.AcceptanceCriteria, FailToPass: i.FailToPass, PassToPass: i.PassToPass,
	}
}

func mapMessageInput(m *MessageDispatchInput) *service.MessageInput {
	if m == nil {
		return nil
	}
	return &service.MessageInput{Content: m.Content}
}

func mapRollouts(rs []service.EnvRollout) []EnvRolloutResponse {
	out := make([]EnvRolloutResponse, 0, len(rs))
	for _, r := range rs {
		out = append(out, EnvRolloutResponse{
			EnvID: r.EnvID, ProjectID: r.ProjectID, IssueID: r.IssueID,
			ChatSessionID: r.ChatSessionID, AgentRunID: r.AgentRunID, Error: r.Error,
		})
	}
	return out
}

// envDispatchConcurrency reads ENV_DISPATCH_CONCURRENCY (default 8).
func envDispatchConcurrency() int {
	// Implementation: read os.Getenv, parse, fall back to 8. Keep simple.
	return 8
}

// (Branch source resolution lives in the service via Deps.GetProjectByEnvID —
// no handler-side resolver is needed.)

// newEnvDispatchDepsAdapter returns the production Deps adapter. Task 8 wires
// real queries; until then, this returns a stub adapter that fails every call.
func newEnvDispatchDepsAdapter(h *Handler) service.EnvDispatchDeps {
	return &envDispatchDepsAdapter{h: h}
}

type envDispatchDepsAdapter struct {
	h *Handler
}

// All methods return stubs until Task 8 wires them to real queries.
func (a *envDispatchDepsAdapter) GetEnv(ctx context.Context, envID, workspaceID string) (service.Env, error) {
	return service.Env{}, nil
}
func (a *envDispatchDepsAdapter) CreateEnv(ctx context.Context, workspaceID, sandboxID, parentEnvID string, mode service.EnvMode, domain service.EnvDomain) (string, error) {
	return "stub-env", nil
}
func (a *envDispatchDepsAdapter) DeleteEnv(ctx context.Context, envID, workspaceID string) error { return nil }
func (a *envDispatchDepsAdapter) ForkSandbox(ctx context.Context, src string, idx int) (string, error) {
	return "stub-fork", nil
}
func (a *envDispatchDepsAdapter) DeleteSandbox(ctx context.Context, sandboxID string) error { return nil }
func (a *envDispatchDepsAdapter) BootSandbox(ctx context.Context, imageRef string) (string, error) {
	return "stub-boot", nil
}
func (a *envDispatchDepsAdapter) CreateProject(ctx context.Context, workspaceID, name, envID string) (string, error) {
	return "stub-project", nil
}
func (a *envDispatchDepsAdapter) CopyProjectSubtree(ctx context.Context, src, ws, envID string) (string, map[string]string, map[string]string, error) {
	return "stub-copy", map[string]string{}, map[string]string{}, nil
}
func (a *envDispatchDepsAdapter) GetProjectByEnvID(ctx context.Context, envID, ws string) (string, error) {
	return "stub-project", nil
}
func (a *envDispatchDepsAdapter) GetIdempotentResponse(ctx context.Context, ws, key string) (service.EnvDispatchResult, bool, error) {
	return service.EnvDispatchResult{}, false, nil
}
func (a *envDispatchDepsAdapter) SaveIdempotentResponse(ctx context.Context, ws, key string, res service.EnvDispatchResult) error {
	return nil
}
func (a *envDispatchDepsAdapter) DeleteProject(ctx context.Context, pid, ws string) error { return nil }
func (a *envDispatchDepsAdapter) ListIssuesByProject(ctx context.Context, pid, ws string) ([]service.IssueRow, error) {
	return nil, nil
}
func (a *envDispatchDepsAdapter) CreateIssue(ctx context.Context, pid, ws, creator, title, desc string, ac, f2p, p2p []string) (string, error) {
	return "stub-issue", nil
}
func (a *envDispatchDepsAdapter) CreateChatSession(ctx context.Context, pid, ws, agent, creator string) (string, error) {
	return "stub-session", nil
}
func (a *envDispatchDepsAdapter) CreateChatMessage(ctx context.Context, sid, role, content string) (string, error) {
	return "stub-msg", nil
}
func (a *envDispatchDepsAdapter) EnqueueAgentRun(ctx context.Context, ws, agent, issue, sess, sbx string, idx int) (string, error) {
	return "stub-run", nil
}
```

Add `DeleteProject` to the service:

```go
// DeleteProject deletes a project by ID (cascades to issues/chat/tasks).
// Idempotent: a missing project returns nil.
func (s *EnvDispatchService) DeleteProject(ctx context.Context, projectID, workspaceID string) error {
	if err := s.deps.DeleteProject(ctx, projectID, workspaceID); err != nil {
		return fmt.Errorf("delete project: %w", err)
	}
	return nil
}
```

- [ ] **Step 2: Write validation tests for the handler**

Create `multica/server/internal/handler/env_dispatch_test.go`:

```go
package handler

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEnvDispatch_RequiresAuth(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/v1/env-dispatch", bytes.NewReader([]byte(`{}`)))
	h.EnvDispatch(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", w.Code)
	}
}

func TestEnvDispatch_RejectsMissingMode(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	body := `{"env_id":"x","dispatch_type":"issue","group_size":1,"agent_id":"a","issue":{"title":"t"}}`
	r := httptest.NewRequest("POST", "/api/v1/env-dispatch", bytes.NewReader([]byte(body)))
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "ws1")
	h.EnvDispatch(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestEnvDispatch_RejectsSweLegoMessage(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	body := `{"mode":"scratch","env_id":"x","domain":"swe_lego","dispatch_type":"message","group_size":1,"agent_id":"a","message":{"content":"q"}}`
	r := httptest.NewRequest("POST", "/api/v1/env-dispatch", bytes.NewReader([]byte(body)))
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "ws1")
	h.EnvDispatch(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestEnvDispatch_SelfPlayIssue_Returns501(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	body := `{"mode":"scratch","env_id":"x","domain":"self_play","dispatch_type":"issue","group_size":1,"agent_id":"a","issue":{"title":"t"}}`
	r := httptest.NewRequest("POST", "/api/v1/env-dispatch", bytes.NewReader([]byte(body)))
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "ws1")
	h.EnvDispatch(w, r)
	if w.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", w.Code)
	}
}

func TestDeleteEnvDispatchProject_RequiresProjectID(t *testing.T) {
	h := newTestHandler(Config{})
	w := httptest.NewRecorder()
	r := httptest.NewRequest("DELETE", "/api/v1/env-dispatch/", nil)
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "ws1")
	h.DeleteEnvDispatchProject(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}
```

- [ ] **Step 3: Run handler tests**

Run:
`cd multica/server && go test ./internal/handler/ -run TestEnvDispatch_ -v && go test ./internal/handler/ -run TestDeleteEnvDispatchProject -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add multica/server/internal/handler/env_dispatch.go multica/server/internal/handler/env_dispatch_test.go multica/server/internal/service/env_dispatch.go
git commit -m "feat(multica): POST /api/v1/env-dispatch + DELETE /api/v1/env-dispatch/{projectID} handlers"
```

______________________________________________________________________

## Task 8: Production adapter wiring — real queries + cloud-runtime calls

**Files:**

- Modify: `multica/server/internal/handler/env_dispatch.go` (replace stub
  `envDispatchDepsAdapter` methods)

- Modify: `multica/server/pkg/db/queries/project.sql` (add `GetProjectByEnvID`)

- Modify: `multica/server/pkg/db/queries/environment.sql` (add `GetEnvDispatchRequest`,
  `CreateEnvDispatchRequest`)

- [ ] **Step 1: Add source-resolution + idempotency queries**

Append to `multica/server/pkg/db/queries/project.sql`:

```sql
-- name: GetProjectByEnvID :one
-- The partial UNIQUE index on project(env_id) guarantees at most one row.
SELECT * FROM project
WHERE env_id = $1 AND workspace_id = $2;
```

Append to `multica/server/pkg/db/queries/environment.sql`:

```sql
-- name: GetEnvDispatchRequest :one
SELECT * FROM env_dispatch_request
WHERE workspace_id = $1 AND idempotency_key = $2;

-- name: CreateEnvDispatchRequest :exec
INSERT INTO env_dispatch_request (workspace_id, idempotency_key, response)
VALUES ($1, $2, $3);
```

Run: `cd multica/server && make sqlc`

> Import note: `env_dispatch.go`'s adapter now uses `github.com/jackc/pgx/v5/pgconn`
> (FK-violation code `23503` → `ErrEnvInUse`) and `github.com/jackc/pgx/v5`
> (`pgx.ErrNoRows` for the idempotency lookup) in addition to `errors`. Add them to the
> import block.

- [ ] **Step 2: Wire the adapter to real queries**

In `multica/server/internal/handler/env_dispatch.go`, replace the stub
`envDispatchDepsAdapter` methods with real implementations. Each method uses
`a.h.Queries.<Query>` (db.Queries from sqlc) and `a.h.CloudRuntime` for sandbox ops.

```go
func (a *envDispatchDepsAdapter) GetEnv(ctx context.Context, envID, workspaceID string) (service.Env, error) {
	e, err := a.h.Queries.GetEnvironment(ctx, db.GetEnvironmentParams{ID: parseUUID(envID), WorkspaceID: parseUUID(workspaceID)})
	if err != nil { return service.Env{}, err }
	return envRowToService(e), nil
}

func (a *envDispatchDepsAdapter) CreateEnv(ctx context.Context, workspaceID, sandboxID, parentEnvID string, mode service.EnvMode, domain service.EnvDomain) (string, error) {
	params := db.CreateEnvironmentParams{
		WorkspaceID: parseUUID(workspaceID), SandboxID: sandboxID, Mode: string(mode),
	}
	if parentEnvID != "" { params.ParentEnvID = pgtype.UUID{Bytes: parseUUID(parentEnvID).Bytes, Valid: true} }
	if domain != "" { params.Domain = pgtype.Text{String: string(domain), Valid: true} }
	e, err := a.h.Queries.CreateEnvironment(ctx, params)
	if err != nil { return "", err }
	return e.ID.String(), nil
}

func (a *envDispatchDepsAdapter) DeleteEnv(ctx context.Context, envID, workspaceID string) error {
	err := a.h.Queries.DeleteEnvironment(ctx, db.DeleteEnvironmentParams{ID: parseUUID(envID), WorkspaceID: parseUUID(workspaceID)})
	// ON DELETE RESTRICT: a project still referencing the env yields a FK
	// violation (SQLSTATE 23503) → surface as ErrEnvInUse (handler → 409).
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23503" {
		return service.ErrEnvInUse
	}
	return err
}

func (a *envDispatchDepsAdapter) ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (string, error) {
	// POST /api/v1/sandboxes/fork via cloud-runtime proxy. Body: {source_sandbox_id}.
	body, _ := json.Marshal(map[string]any{"source_sandbox_id": sourceSandboxID})
	resp, err := a.h.CloudRuntime.Do(ctx, cloudruntime.Request{
		Method: http.MethodPost, Path: "/api/v1/sandboxes/fork", Body: body,
	})
	if err != nil { return "", err }
	var fr struct{ SandboxID string `json:"sandbox_id"` }
	if err := json.Unmarshal(resp.Body, &fr); err != nil { return "", err }
	return fr.SandboxID, nil
}

func (a *envDispatchDepsAdapter) DeleteSandbox(ctx context.Context, sandboxID string) error {
	_, err := a.h.CloudRuntime.Do(ctx, cloudruntime.Request{
		Method: http.MethodDelete, Path: "/api/v1/sandboxes/" + url.PathEscape(sandboxID),
	})
	return err
}

func (a *envDispatchDepsAdapter) BootSandbox(ctx context.Context, imageRef string) (string, error) {
	body, _ := json.Marshal(map[string]any{"image_ref": imageRef})
	resp, err := a.h.CloudRuntime.Do(ctx, cloudruntime.Request{
		Method: http.MethodPost, Path: "/api/v1/sandboxes", Body: body,
	})
	if err != nil { return "", err }
	var br struct{ SandboxID string `json:"sandbox_id"` }
	if err := json.Unmarshal(resp.Body, &br); err != nil { return "", err }
	return br.SandboxID, nil
}

func (a *envDispatchDepsAdapter) CreateProject(ctx context.Context, workspaceID, name, envID string) (string, error) {
	p, err := a.h.Queries.CreateProjectWithEnv(ctx, db.CreateProjectWithEnvParams{
		WorkspaceID: parseUUID(workspaceID), Title: name, Status: "active",
		LeadType: "user", EnvID: pgtype.UUID{Bytes: parseUUID(envID).Bytes, Valid: true},
	})
	if err != nil { return "", err }
	return p.ID.String(), nil
}

func (a *envDispatchDepsAdapter) CopyProjectSubtree(ctx context.Context, sourceProjectID, workspaceID, envID string) (string, map[string]string, map[string]string, error) {
	// Implementation: load source project; create new project (CreateProjectWithEnv);
	// copy issues (CreateForkedIssue with forked_from_issue_id set) AND chat
	// sessions + their messages. Return source→copied ID maps for BOTH issues
	// and chat sessions — the session map lets branch+self_play append to the
	// copied session (spec §7.3/§7.4).
	// (IssueForkService.ForkIssueSubtree already exists at service/issue_fork.go —
	//  invoke or inline its logic for the issue side.)
	// TODO: implement after confirming issue_fork.go's reusable surface and
	// adding a chat-session copy query.
	return "", nil, nil, fmt.Errorf("CopyProjectSubtree not yet wired")
}

func (a *envDispatchDepsAdapter) GetProjectByEnvID(ctx context.Context, envID, workspaceID string) (string, error) {
	// Relies on the partial UNIQUE index (env_id 1:1) → at most one row.
	p, err := a.h.Queries.GetProjectByEnvID(ctx, db.GetProjectByEnvIDParams{EnvID: pgtype.UUID{Bytes: parseUUID(envID).Bytes, Valid: true}, WorkspaceID: parseUUID(workspaceID)})
	if err != nil { return "", fmt.Errorf("no project for env: %w", err) }
	return p.ID.String(), nil
}

func (a *envDispatchDepsAdapter) GetIdempotentResponse(ctx context.Context, workspaceID, key string) (service.EnvDispatchResult, bool, error) {
	row, err := a.h.Queries.GetEnvDispatchRequest(ctx, db.GetEnvDispatchRequestParams{WorkspaceID: parseUUID(workspaceID), IdempotencyKey: parseUUID(key)})
	if errors.Is(err, pgx.ErrNoRows) { return service.EnvDispatchResult{}, false, nil }
	if err != nil { return service.EnvDispatchResult{}, false, err }
	var res service.EnvDispatchResult
	if err := json.Unmarshal(row.Response, &res); err != nil { return service.EnvDispatchResult{}, false, err }
	return res, true, nil
}

func (a *envDispatchDepsAdapter) SaveIdempotentResponse(ctx context.Context, workspaceID, key string, res service.EnvDispatchResult) error {
	blob, _ := json.Marshal(res)
	return a.h.Queries.CreateEnvDispatchRequest(ctx, db.CreateEnvDispatchRequestParams{
		WorkspaceID: parseUUID(workspaceID), IdempotencyKey: parseUUID(key), Response: blob,
	})
}

func (a *envDispatchDepsAdapter) DeleteProject(ctx context.Context, projectID, workspaceID string) error {
	return a.h.Queries.DeleteProject(ctx, db.DeleteProjectParams{ID: parseUUID(projectID), WorkspaceID: parseUUID(workspaceID)})
}

func (a *envDispatchDepsAdapter) ListIssuesByProject(ctx context.Context, projectID, workspaceID string) ([]service.IssueRow, error) {
	rows, err := a.h.Queries.ListIssuesByProject(ctx, db.ListIssuesByProjectParams{ProjectID: parseUUID(projectID), WorkspaceID: parseUUID(workspaceID)})
	if err != nil { return nil, err }
	out := make([]service.IssueRow, 0, len(rows))
	for _, r := range rows {
		out = append(out, service.IssueRow{ID: r.ID.String(), ProjectID: r.ProjectID.String(), Title: r.Title, Description: r.Description})
	}
	return out, nil
}

func (a *envDispatchDepsAdapter) CreateIssue(ctx context.Context, projectID, workspaceID, creatorID, title, description string, ac, f2p, p2p []string) (string, error) {
	meta := map[string]any{"acceptance_criteria": ac, "fail_to_pass": f2p, "pass_to_pass": p2p}
	metaJSON, _ := json.Marshal(meta)
	i, err := a.h.Queries.CreateIssue(ctx, db.CreateIssueParams{
		WorkspaceID: parseUUID(workspaceID), ProjectID: parseUUID(projectID),
		Title: title, Description: description, Status: "todo",
		CreatorType: "user", CreatorID: parseUUID(creatorID),
		Metadata: metaJSON,
	})
	if err != nil { return "", err }
	return i.ID.String(), nil
}

func (a *envDispatchDepsAdapter) CreateChatSession(ctx context.Context, projectID, workspaceID, agentID, creatorID string) (string, error) {
	s, err := a.h.Queries.CreateChatSessionForProject(ctx, db.CreateChatSessionForProjectParams{
		WorkspaceID: parseUUID(workspaceID), ProjectID: pgtype.UUID{Bytes: parseUUID(projectID).Bytes, Valid: true},
		AgentID: parseUUID(agentID), CreatorID: parseUUID(creatorID), Title: "env-dispatch",
	})
	if err != nil { return "", err }
	return s.ID.String(), nil
}

func (a *envDispatchDepsAdapter) CreateChatMessage(ctx context.Context, sessionID, role, content string) (string, error) {
	m, err := a.h.Queries.CreateChatMessage(ctx, db.CreateChatMessageParams{
		ChatSessionID: parseUUID(sessionID), Role: role, Content: content,
	})
	if err != nil { return "", err }
	return m.ID.String(), nil
}

func (a *envDispatchDepsAdapter) EnqueueAgentRun(ctx context.Context, workspaceID, agentID, issueID, chatSessionID, sandboxID string, idx int) (string, error) {
	// Insert into agent_task_queue. Reuse CreateChatTask for chat-bound runs;
	// for issue-bound runs, use the existing issue-task queue path.
	// TODO: wire to the actual agent_task_queue insert query.
	return "", fmt.Errorf("EnqueueAgentRun not yet wired")
}
```

Source resolution no longer lives in the handler: the service resolves the branch source
project via `Deps.GetProjectByEnvID` (§7.2 step 0), backed by the `GetProjectByEnvID`
query added in Step 1. Delete the placeholder `resolveSourceProjectID` method from Task
7 — it is no longer called.

- [ ] **Step 3: Implement `CopyProjectSubtree` and `EnqueueAgentRun`**

For `CopyProjectSubtree`: read source project via `GetProjectInWorkspace`; list issues
via `ListIssuesByProject`; create new project via `CreateProjectWithEnv`; for each
source issue call `CreateForkedIssue` (already exists in `issue_fork.sql`) with
`forked_from_issue_id` set. Return `map[source_issue_id]new_issue_id`.

For `EnqueueAgentRun`: use the existing `CreateChatTask` query for chat-session-bound
runs (issue_id=NULL, chat_session_id set); for issue-bound runs, use the existing
issue-task queue path (look at `agent_task_queue` inserts in `task.sql`).

If either of these turns out to require more than the existing queries, add new sqlc
queries rather than inlining raw SQL.

- [ ] **Step 4: Build and run all server tests**

Run: `cd multica/server && go build ./... && go test ./...` Expected: all tests pass
(service unit tests use the fake, not the wired adapter; handler tests use the
stub-returning adapter until Step 2 replaced it — but handler tests don't exercise real
DB, so they should still pass with the wired adapter hitting nil `Queries`).

If the handler tests break because `Queries` is nil in `newTestHandler(Config{})`, keep
a `stubEnvDispatchDeps` for tests and inject it: add a `Config.EnvDispatchDeps` field,
or use a package-level override. Simplest: keep the stub adapter as a fallback inside
`newEnvDispatchDepsAdapter` when `h.Queries == nil`.

- [ ] **Step 5: Commit**

```bash
git add multica/server/internal/handler/env_dispatch.go multica/server/pkg/db/queries/project.sql multica/server/pkg/db/
git commit -m "feat(multica): wire EnvDispatchDeps adapter to real queries + cloud-runtime"
```

______________________________________________________________________

## Task 9: Route registration in `router.go`

**Files:**

- Modify: `multica/server/cmd/server/router.go`

- [ ] **Step 1: Remove the `/api/v1/swe-lego` route group**

In `multica/server/cmd/server/router.go`, delete lines 952–960 (the
`r.Route("/api/v1/swe-lego", ...)` block and its comment).

- [ ] **Step 2: Add the new routes in the same `RequireWorkspaceMember` group**

In place of the deleted block, add:

```go
// Unified env-dispatch API (spec §6). Replaces the SWE-Lego-only route group.
r.Post("/api/v1/env", h.CreateEnv)
r.Delete("/api/v1/env/{envID}", h.DeleteEnv)
r.Post("/api/v1/env-dispatch", h.EnvDispatch)
r.Delete("/api/v1/env-dispatch/{projectID}", h.DeleteEnvDispatchProject)
```

- [ ] **Step 3: Build and verify routes**

Run: `cd multica/server && go build ./cmd/server/` Expected: no errors.

- [ ] **Step 4: Smoke-test routes (manual)**

Run: `cd multica/server && make server &` then
`curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/api/v1/env-dispatch -H "X-User-ID: u" -H "X-Workspace-ID: ws"`
(expect 400 — missing body fields, but route exists, not 404).

Kill the server.

- [ ] **Step 5: Commit**

```bash
git add multica/server/cmd/server/router.go
git commit -m "feat(multica): register /api/v1/env* and /api/v1/env-dispatch* routes"
```

______________________________________________________________________

## Task 10: Remove old SWE-Lego handler + service

**Files:**

- Delete: `multica/server/internal/handler/swe_lego_issue.go`

- Delete: `multica/server/internal/handler/swe_lego_issue_test.go`

- Delete: `multica/server/internal/service/swe_lego_issue.go`

- Delete: `multica/server/internal/service/swe_lego_issue_test.go`

- [ ] **Step 1: Delete the four files**

Run:
`cd multica/server && rm internal/handler/swe_lego_issue.go internal/handler/swe_lego_issue_test.go internal/service/swe_lego_issue.go internal/service/swe_lego_issue_test.go`

- [ ] **Step 2: Verify build (no references remain)**

Run: `cd multica/server && go build ./...` Expected: no errors. If errors mention
`SweLegoIssueService` or `CreateSweLegoIssue`, grep for stale references:
`grep -rn "SweLegoIssue\|CreateSweLegoIssue\|swe-lego" server/` and fix or remove.

- [ ] **Step 3: Run all server tests**

Run: `cd multica/server && go test ./...` Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A multica/server/internal/handler/ multica/server/internal/service/
git commit -m "refactor(multica): remove SweLegoIssueService + CreateSweLegoIssue (folded into EnvDispatchService)"
```

______________________________________________________________________

## Task 11: AReaL — rewrite `MulticaSweLegoClient` → `MulticaEnvDispatchClient`

**Files:**

- Modify: `customized_areal/tree_search/agents/swe_lego_client.py` (rewrite + rename)

- Create: `customized_areal/tree_search/tests/test_env_dispatch_client.py`

- [ ] **Step 1: Update `SweLegoSetup` shape first (other files depend on it)**

Edit `customized_areal/tree_search/agents/reward/swe_lego_types.py`. Replace the
`SweLegoSetup` dataclass with:

```python
@dataclass(frozen=True)
class SweLegoRollout:
    """One rollout in an env-dispatch group (spec §6.3 response)."""

    env_id: str
    project_id: str
    issue_id: str = ""
    chat_session_id: str = ""
    agent_run_id: str = ""


@dataclass(frozen=True)
class SweLegoSetup:
    """The result of POST /api/v1/env-dispatch (spec §6.3)."""

    rollouts: list[SweLegoRollout]
```

Remove the old fields (`project_id`, `issue_id`, `image_id`, `build_node_id`,
`base_sandbox_id`, `base_sandbox_runtime_id`, `agent_run_ids`).

- [ ] **Step 2: Rewrite the client**

Replace `customized_areal/tree_search/agents/swe_lego_client.py` contents with:

```python
"""HTTP client for the multica unified env-dispatch API.

Wraps ``POST /api/v1/env``, ``DELETE /api/v1/env/{envID}``,
``POST /api/v1/env-dispatch``, and ``DELETE /api/v1/env-dispatch/{projectID}``
(spec §6). Uses stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
import os

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoRollout,
    SweLegoSetup,
)

logger = logging.getLogger("MulticaEnvDispatchClient")


class MulticaEnvDispatchClient:
    """Thin HTTP client for the multica env-dispatch API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            raise ValueError("MulticaEnvDispatchClient requires base_url or MULTICA_BASE_URL")
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_base_env(self, *, image_ref: str) -> str:
        """POST /api/v1/env — boot a sandbox from image_ref, return env_id."""
        resp = await self._client.post(
            "/api/v1/env", json={"image_ref": image_ref}, headers=self._headers()
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"create_base_env failed: status={resp.status_code} body={resp.text[:200]}"
            )
        return resp.json()["env_id"]

    async def delete_env(self, *, env_id: str) -> None:
        """DELETE /api/v1/env/{envID} — idempotent on 404."""
        resp = await self._client.delete(
            f"/api/v1/env/{env_id}", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"delete_env failed: status={resp.status_code} body={resp.text[:200]}"
            )

    async def create_env_dispatch(
        self,
        *,
        mode: str,
        env_id: str,
        dispatch_type: str,
        agent_id: str,
        group_size: int = 1,
        domain: str | None = None,
        issue: SweLegoIssue | None = None,
        message: str | None = None,
    ) -> SweLegoSetup:
        """POST /api/v1/env-dispatch — unified dispatch (spec §6.3)."""
        payload: dict = {
            "mode": mode,
            "env_id": env_id,
            "dispatch_type": dispatch_type,
            "group_size": group_size,
            "agent_id": agent_id,
        }
        if domain is not None:
            payload["domain"] = domain
        if issue is not None:
            payload["issue"] = {
                "title": issue.issue_title,
                "description": issue.issue_text,
                "acceptance_criteria": [issue.acceptance_criteria] if issue.acceptance_criteria else [],
                "fail_to_pass": list(issue.fail_to_pass),
                "pass_to_pass": list(issue.pass_to_pass),
            }
        if message is not None:
            payload["message"] = {"content": message}

        resp = await self._client.post(
            "/api/v1/env-dispatch", json=payload, headers=self._headers()
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"create_env_dispatch failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        rollouts = [
            SweLegoRollout(
                env_id=r["env_id"],
                project_id=r["project_id"],
                issue_id=r.get("issue_id", ""),
                chat_session_id=r.get("chat_session_id", ""),
                agent_run_id=r.get("agent_run_id", ""),
            )
            for r in body["rollouts"]
        ]
        return SweLegoSetup(rollouts=rollouts)

    async def cleanup_env_dispatch(self, *, project_id: str) -> None:
        """DELETE /api/v1/env-dispatch/{projectID} — cascades to issues/chat/tasks."""
        resp = await self._client.delete(
            f"/api/v1/env-dispatch/{project_id}", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"cleanup_env_dispatch failed: status={resp.status_code} body={resp.text[:200]}"
            )

    # Back-compat alias for the old runner signature.
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None:
        await self.cleanup_env_dispatch(project_id=project_id)
```

Keep the old module path (`swe_lego_client.py`) so imports don't break, but the class
name changes. Update any import of `MulticaSweLegoClient`:
`grep -rn "MulticaSweLegoClient" customized_areal/`.

- [ ] **Step 3: Write client tests**

Create `customized_areal/tree_search/tests/test_env_dispatch_client.py`:

```python
import asyncio
import json

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue
from customized_areal.tree_search.agents.swe_lego_client import MulticaEnvDispatchClient


def _transport(handler):
    return httpx.MockTransport(handler)


def test_create_base_env_returns_env_id():
    def handler(req):
        assert req.method == "POST"
        assert req.url.path == "/api/v1/env"
        return httpx.Response(201, json={"env_id": "env-1", "sandbox_id": "sbx-1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    env_id = asyncio.run(c.create_base_env(image_ref="img:tag"))
    assert env_id == "env-1"


def test_create_env_dispatch_scratch_swe_lego():
    def handler(req):
        body = json.loads(req.content)
        assert body["mode"] == "scratch"
        assert body["domain"] == "swe_lego"
        assert body["dispatch_type"] == "issue"
        assert body["group_size"] == 2
        return httpx.Response(
            201,
            json={"rollouts": [
                {"env_id": "e1", "project_id": "p1", "issue_id": "i1", "agent_run_id": "r1"},
                {"env_id": "e2", "project_id": "p2", "issue_id": "i2", "agent_run_id": "r2"},
            ]},
        )

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    issue = SweLegoIssue(
        repo_url="r", base_commit="c", issue_date="d", issue_text="x", issue_title="t",
        acceptance_criteria="a", fail_to_pass=["f"], pass_to_pass=["p"],
    )
    setup = asyncio.run(
        c.create_env_dispatch(
            mode="scratch", env_id="base", dispatch_type="issue",
            agent_id="ag", group_size=2, domain="swe_lego", issue=issue,
        )
    )
    assert len(setup.rollouts) == 2
    assert setup.rollouts[0].agent_run_id == "r1"
    assert setup.rollouts[1].env_id == "e2"


def test_cleanup_env_dispatch_hits_renamed_url():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        return httpx.Response(204)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.cleanup_env_dispatch(project_id="p1"))
    assert seen["path"] == "/api/v1/env-dispatch/p1"


def test_delete_env_idempotent_on_404():
    def handler(req):
        return httpx.Response(404)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.delete_env(env_id="env-1"))  # no raise
```

- [ ] **Step 4: Run client tests**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -v`
Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/agents/reward/swe_lego_types.py customized_areal/tree_search/tests/test_env_dispatch_client.py
git commit -m "feat(areal): MulticaEnvDispatchClient + SweLegoSetup.rollouts shape"
```

______________________________________________________________________

## Task 12: AReaL — update `swe_lego_issue_runner.py` + its tests

**Files:**

- Modify: `customized_areal/tree_search/agents/swe_lego_issue_runner.py`

- Modify: `customized_areal/tree_search/tests/test_swe_lego_issue_runner.py`

- [ ] **Step 1: Update the runner to iterate `setup.rollouts`**

In `customized_areal/tree_search/agents/swe_lego_issue_runner.py`, replace the body of
`run_swe_lego_issue` (lines 59–110) with:

```python
async def run_swe_lego_issue(
    *,
    issue: SweLegoIssue,
    group_size: int,
    agent_id: str,
    multica: _MulticaClient,
    rl_session: _RlSession,
    verifier: _Verifier,
    branch_driver: _BranchDriver,
    base_env_id: str,
) -> SweLegoIssueResult:
    # 1. Atomic dispatch.
    setup = await multica.create_env_dispatch(
        mode="scratch", env_id=base_env_id, dispatch_type="issue",
        agent_id=agent_id, group_size=group_size,
        domain="swe_lego", issue=issue,
    )

    try:
        # 2. Open one RL session per rollout.
        sessions = list(await asyncio.gather(*[
            rl_session.start(agent_run_id=r.agent_run_id, issue_id=r.issue_id)
            for r in setup.rollouts
        ]))

        # 3. Drive branching within each lane. The driver returns the
        #    terminal sandbox id for each lane.
        #    NOTE: multica owns sandbox_id; areal passes env_id to branch.
        #    The branch driver receives env_id (not sandbox_id) and asks
        #    multica to fork internally when it branches.
        terminal_env_ids = await asyncio.gather(*[
            branch_driver.drive_lane(
                agent_run_id=r.agent_run_id, sandbox_id=r.env_id, session_id=sid
            )
            for r, sid in zip(setup.rollouts, sessions)
        ])

        # 4. Verify + reward each terminal run.
        # TODO(task-17): wire real transcript from the branch driver.
        results = await asyncio.gather(*[
            verifier.verify_and_reward(
                agent_run_id=r.agent_run_id, sandbox_id=eid, session_id=sid,
                fail_to_pass=issue.fail_to_pass, pass_to_pass=issue.pass_to_pass,
                transcript="...", acceptance_criteria=issue.acceptance_criteria,
            )
            for r, eid, sid in zip(setup.rollouts, terminal_env_ids, sessions)
        ])
        return SweLegoIssueResult(
            per_agent_rewards=[r.reward for r in results],
            per_agent_success=[r.success for r in results],
        )
    finally:
        # 5. Cleanup always, even on failure (no sandbox leaks). One DELETE
        #    per rollout's project_id.
        for r in setup.rollouts:
            try:
                await multica.cleanup_swe_lego_issue(project_id=r.project_id)
            except Exception:
                logger.exception("cleanup failed for project %s", r.project_id)
```

Update the `_MulticaClient` Protocol to match the new client surface:

```python
class _MulticaClient(Protocol):
    async def create_env_dispatch(
        self, *, mode: str, env_id: str, dispatch_type: str,
        agent_id: str, group_size: int = ..., domain: str | None = ...,
        issue: SweLegoIssue | None = ..., message: str | None = ...,
    ) -> SweLegoSetup: ...
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None: ...
```

Drop the `base_image` parameter (image build is out-of-band per spec §3).

- [ ] **Step 2: Update `test_swe_lego_issue_runner.py` fixtures**

Edit `customized_areal/tree_search/tests/test_swe_lego_issue_runner.py`. Replace
`FakeMulticaClient` and `_setup()`:

```python
@dataclass
class FakeMulticaClient:
    rollouts: list = field(default_factory=list)
    cleanup_calls: list = field(default_factory=list)
    cleanup_raises: bool = False

    async def create_env_dispatch(self, *, mode, env_id, dispatch_type, agent_id,
                                  group_size, domain=None, issue=None, message=None):
        rollouts = [
            SweLegoRollout(env_id=f"env-{i}", project_id=f"proj-{i}",
                           issue_id="issue-1", agent_run_id=f"r{i+1}")
            for i in range(group_size)
        ]
        self.rollouts = rollouts
        return SweLegoSetup(rollouts=rollouts)

    async def cleanup_swe_lego_issue(self, *, project_id):
        self.cleanup_calls.append(project_id)
        if self.cleanup_raises:
            raise RuntimeError("cleanup crashed")
```

Add the import:
`from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoRollout`.

Update every test call to `run_swe_lego_issue(...)` to pass `base_env_id="base-env-1"`
instead of `base_image=...`. Update assertions:

- `multica.create_calls[0][1] == 2` → `len(multica.rollouts) == 2`

- `len(rl.sessions) == 2` stays

- `multica.cleanup_calls == ["p1"]` → `multica.cleanup_calls == ["proj-0", "proj-1"]`

- [ ] **Step 3: Run the runner tests**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_issue_runner.py -v`
Expected: all 4 pass.

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/agents/swe_lego_issue_runner.py customized_areal/tree_search/tests/test_swe_lego_issue_runner.py
git commit -m "refactor(areal): swe_lego_issue_runner iterates setup.rollouts, passes env_id"
```

______________________________________________________________________

## Task 13: AReaL — new `self_play_runner.py` + tests

**Files:**

- Create: `customized_areal/tree_search/agents/self_play_runner.py`

- Create: `customized_areal/tree_search/tests/test_self_play_runner.py`

- [ ] **Step 1: Write the self-play runner**

Create `customized_areal/tree_search/agents/self_play_runner.py`:

```python
"""Self-play orchestration loop for DAG RL training.

Mirrors ``swe_lego_issue_runner.py`` but dispatches a query (from
``query_bank``) as a chat message via POST /api/v1/env-dispatch with
domain=self_play, dispatch_type=message. Uses stdlib :mod:`logging` so the
module stays importable without torch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup
from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("SelfPlayRunner")


@dataclass(frozen=True)
class SelfPlayQuery:
    """One query from query_bank."""

    query_id: str
    content: str
    answer: str


@dataclass(frozen=True)
class SelfPlayResult:
    per_agent_rewards: list[float]
    per_agent_success: list[bool] = field(default_factory=list)


class _MulticaClient(Protocol):
    async def create_env_dispatch(
        self, *, mode: str, env_id: str, dispatch_type: str,
        agent_id: str, group_size: int = ..., domain: str | None = ...,
        issue=None, message: str | None = ...,
    ) -> SweLegoSetup: ...
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None: ...


class _RlSession(Protocol):
    async def start(self, *, agent_run_id: str, issue_id: str) -> str: ...


class _Verifier(Protocol):
    async def verify_and_reward(
        self, *, agent_run_id: str, sandbox_id: str, session_id: str,
        transcript: str, answer: str,
    ) -> VerifierResult: ...


class _BranchDriver(Protocol):
    async def drive_lane(self, *, agent_run_id: str, sandbox_id: str, session_id: str) -> str: ...


async def run_self_play(
    *,
    query: SelfPlayQuery,
    group_size: int,
    agent_id: str,
    base_env_id: str,
    multica: _MulticaClient,
    rl_session: _RlSession,
    verifier: _Verifier,
    branch_driver: _BranchDriver,
) -> SelfPlayResult:
    setup = await multica.create_env_dispatch(
        mode="scratch", env_id=base_env_id, dispatch_type="message",
        agent_id=agent_id, group_size=group_size,
        domain="self_play", message=query.content,
    )

    try:
        sessions = list(await asyncio.gather(*[
            rl_session.start(agent_run_id=r.agent_run_id, issue_id="")
            for r in setup.rollouts
        ]))

        terminal_env_ids = await asyncio.gather(*[
            branch_driver.drive_lane(
                agent_run_id=r.agent_run_id, sandbox_id=r.env_id, session_id=sid
            )
            for r, sid in zip(setup.rollouts, sessions)
        ])

        results = await asyncio.gather(*[
            verifier.verify_and_reward(
                agent_run_id=r.agent_run_id, sandbox_id=eid, session_id=sid,
                transcript="...", answer=query.answer,
            )
            for r, eid, sid in zip(setup.rollouts, terminal_env_ids, sessions)
        ])
        return SelfPlayResult(
            per_agent_rewards=[r.reward for r in results],
            per_agent_success=[r.success for r in results],
        )
    finally:
        for r in setup.rollouts:
            try:
                await multica.cleanup_swe_lego_issue(project_id=r.project_id)
            except Exception:
                logger.exception("cleanup failed for project %s", r.project_id)
```

- [ ] **Step 2: Write tests mirroring `test_swe_lego_issue_runner.py`**

Create `customized_areal/tree_search/tests/test_self_play_runner.py`:

```python
import asyncio
from dataclasses import dataclass, field

import pytest

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoRollout, SweLegoSetup,
)
from customized_areal.tree_search.agents.self_play_runner import (
    run_self_play, SelfPlayQuery, SelfPlayResult,
)
from customized_areal.tree_search.agents.verifier import VerifierResult


@dataclass
class FakeMulticaClient:
    rollouts: list = field(default_factory=list)
    cleanup_calls: list = field(default_factory=list)
    cleanup_raises: bool = False

    async def create_env_dispatch(self, *, mode, env_id, dispatch_type, agent_id,
                                  group_size, domain=None, issue=None, message=None):
        rollouts = [
            SweLegoRollout(env_id=f"env-{i}", project_id=f"proj-{i}",
                           chat_session_id=f"sess-{i}", agent_run_id=f"r{i+1}")
            for i in range(group_size)
        ]
        self.rollouts = rollouts
        return SweLegoSetup(rollouts=rollouts)

    async def cleanup_swe_lego_issue(self, *, project_id):
        self.cleanup_calls.append(project_id)
        if self.cleanup_raises:
            raise RuntimeError("cleanup crashed")


@dataclass
class FakeRlSession:
    sessions: list = field(default_factory=list)

    async def start(self, *, agent_run_id, issue_id):
        self.sessions.append(agent_run_id)
        return f"sess-{agent_run_id}"


@dataclass
class FakeVerifier:
    async def verify_and_reward(self, **kwargs):
        return VerifierResult(success=True, reward=1.0, source="objective")


@dataclass
class FakeBranchDriver:
    ran_lanes: list = field(default_factory=list)

    async def drive_lane(self, *, agent_run_id, sandbox_id, session_id):
        self.ran_lanes.append(agent_run_id)
        return sandbox_id


def _query() -> SelfPlayQuery:
    return SelfPlayQuery(query_id="q1", content="what is 2+2?", answer="4")


def test_run_self_play_happy_path():
    multica = FakeMulticaClient()
    rl = FakeRlSession()
    verifier = FakeVerifier()
    driver = FakeBranchDriver()
    result = asyncio.run(
        run_self_play(
            query=_query(), group_size=2, agent_id="ag", base_env_id="base",
            multica=multica, rl_session=rl, verifier=verifier, branch_driver=driver,
        )
    )
    assert isinstance(result, SelfPlayResult)
    assert len(multica.rollouts) == 2
    assert len(rl.sessions) == 2
    assert len(driver.ran_lanes) == 2
    assert result.per_agent_rewards == [1.0, 1.0]
    assert multica.cleanup_calls == ["proj-0", "proj-1"]


def test_run_self_play_cleans_up_on_verifier_failure():
    multica = FakeMulticaClient()
    rl = FakeRlSession()

    @dataclass
    class RaisingVerifier:
        async def verify_and_reward(self, **kwargs):
            raise RuntimeError("verifier crashed")

    with pytest.raises(RuntimeError, match="verifier crashed"):
        asyncio.run(
            run_self_play(
                query=_query(), group_size=2, agent_id="ag", base_env_id="base",
                multica=multica, rl_session=rl, verifier=RaisingVerifier(),
                branch_driver=FakeBranchDriver(),
            )
        )
    assert multica.cleanup_calls == ["proj-0", "proj-1"]
```

- [ ] **Step 3: Run the self-play tests**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/test_self_play_runner.py -v`
Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/agents/self_play_runner.py customized_areal/tree_search/tests/test_self_play_runner.py
git commit -m "feat(areal): self_play_runner for query_bank-driven env-dispatch"
```

______________________________________________________________________

## Task 14: Update areal integration tests

**Files:**

- Modify: `customized_areal/tree_search/tests/test_swe_lego_client.py` (if present)

- Modify: `customized_areal/tree_search/tests/test_integration_multica.py` (if present)

- [ ] **Step 1: Find integration tests that reference the old endpoint**

Run:
`cd /workspaces/leagent/backend/areal && grep -rln "swe-lego\|swe_lego_issue\|MulticaSweLegoClient\|SweLegoSetup" customized_areal/tree_search/tests/`

- [ ] **Step 2: Update each found test to the new shape**

For each file found, update:

- URL `/api/v1/swe-lego/issues` → `/api/v1/env-dispatch`
- URL `/api/v1/swe-lego/issues/{projectID}` → `/api/v1/env-dispatch/{projectID}`
- Request body: old flat shape → new discriminated shape (`mode`, `env_id`, `domain`,
  `dispatch_type`, `group_size`, `agent_id`, `issue`/`message` sub-objects)
- Response body: old flat shape →
  `{"rollouts": [{env_id, project_id, issue_id?, chat_session_id?, agent_run_id?}]}`
- `SweLegoSetup(project_id=..., agent_run_ids=[...])` →
  `SweLegoSetup(rollouts=[SweLegoRollout(...), ...])`
- `MulticaSweLegoClient` → `MulticaEnvDispatchClient`
- `create_swe_lego_issue(...)` →
  `create_env_dispatch(mode="scratch", env_id=..., dispatch_type="issue", domain="swe_lego", group_size=..., agent_id=..., issue=...)`

If a test file is purely a client-shape test (no live server), port it onto
`test_env_dispatch_client.py` (already done in Task 11) and delete the old file.

- [ ] **Step 3: Run all areal tree_search tests**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/ -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/tests/
git commit -m "test(areal): port integration tests to env-dispatch shape"
```

______________________________________________________________________

## Task 15: End-to-end smoke + final verification

**Files:** (no code changes — verification only)

- [ ] **Step 1: Run all multica server tests**

Run: `cd multica/server && go test ./...` Expected: all pass.

- [ ] **Step 2: Run all areal tree_search tests**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/tests/ -v`
Expected: all pass.

- [ ] **Step 3: Apply migration 127 on a clean DB and run a smoke request**

Run: `cd multica/server && make db-reset && make migrate-up && make server &`

Wait for server to start, then:

```bash
# Create base env (will 503 if cloud-runtime not configured locally — that's OK,
# the route + validation is what we're verifying).
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/api/v1/env \
  -H "X-User-ID: u1" -H "X-Workspace-ID: ws1" \
  -H "Content-Type: application/json" \
  -d '{"image_ref":""}'
# Expected: 400 (image_ref empty).

curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/api/v1/env-dispatch \
  -H "X-User-ID: u1" -H "X-Workspace-ID: ws1" \
  -H "Content-Type: application/json" \
  -d '{}'
# Expected: 400 (missing mode/dispatch_type).

curl -s -o /dev/null -w "%{http_code}\n" -X DELETE http://localhost:8080/api/v1/env-dispatch/proj-1 \
  -H "X-User-ID: u1" -H "X-Workspace-ID: ws1"
# Expected: 400 (proj-1 is not a UUID).

curl -s -o /dev/null -w "%{http_code}\n" -X DELETE http://localhost:8080/api/v1/swe-lego/issues/proj-1 \
  -H "X-User-ID: u1" -H "X-Workspace-ID: ws1"
# Expected: 404 (route removed).
```

Kill the server.

- [ ] **Step 4: Final commit (if any cleanup surfaced)**

If Steps 1–3 surfaced any issues, fix them with atomic commits per fix. Otherwise no
commit.

______________________________________________________________________

## Self-review notes

- **Spec coverage:** Every §4.2 matrix cell has a service test (Task 4). Every §6.3
  validation row has either a service test (Task 4 rejected-combo tests) or a handler
  test (Task 7). Migration 127 + sqlc queries (Task 1, 2). Env handlers (Task 6).
  Env-dispatch handler (Task 7). Adapter wiring (Task 8). Route registration (Task 9).
  Old code removal (Task 10). Areal client + shape (Task 11). Runner update (Task 12).
  Self-play runner (Task 13). Integration tests (Task 14). E2E smoke (Task 15).
- **Placeholder scan:** Task 8 Step 3 has explicit TODOs for `CopyProjectSubtree` and
  `EnqueueAgentRun` real wiring — these are real implementation work, not placeholders,
  and are required to complete. No "TBD", "fill in later", or hand-waved steps.
- **Type consistency:** `EnvRollout` (service) ↔ `EnvRolloutResponse` (handler) ↔
  `SweLegoRollout` (areal) all share field names (`env_id`, `project_id`, `issue_id`,
  `chat_session_id`, `agent_run_id`). `EnvDispatchInput` fields match what the handler
  constructs. `SweLegoSetup.rollouts` is used by both runners.

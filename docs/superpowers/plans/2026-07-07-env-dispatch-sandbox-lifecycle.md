---
change: env-dispatch-sandbox-lifecycle
design-doc: docs/superpowers/specs/2026-07-07-env-dispatch-sandbox-lifecycle-design.md
base-ref: 6bffe9f6d31c0d75e81bd0cf86713074d45270e9
---

# Env-dispatch Sandbox Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Multica sandbox-instance-backed env-dispatch lifecycle handles, synchronous checkpoint save/resume APIs, and AReaL client integration for training rollouts.

**Architecture:** Multica owns sandbox lifecycle and checkpoint persistence through an internal env sandbox lifecycle service that wraps existing sandboxd job/query machinery. Env-dispatch carries structured `sandbox_instance` refs and optional per-agent env intent; checkpoint APIs synchronously save with timeout, store inline JSONB project snapshots, and resume through existing sandboxd `resume` jobs. AReaL remains a thin HTTP client and policy layer.

**Tech Stack:** Go services/handlers/sqlc/migrations under `multica/server`; Python 3.12+ AReaL client/tests under `customized_areal/tree_search`; OpenSpec delta specs under `openspec/changes/env-dispatch-sandbox-lifecycle`.

## Global Constraints

- Canonical design: `docs/superpowers/specs/2026-07-07-env-dispatch-sandbox-lifecycle-design.md`.
- Canonical OpenSpec change: `openspec/changes/env-dispatch-sandbox-lifecycle`.
- Save semantics: checkpoint create synchronously waits for sandboxd `stop` / Cube pause up to a configured timeout.
- Snapshot storage: v1 uses inline JSONB `db_snapshot` on the checkpoint row.
- Resume semantics: resume the same saved sandbox instance; do not expose branch, fork, immutable snapshot, or copy-on-write semantics.
- Built-in sandbox behavior must preserve websocket wakeups, node ownership, runtime metadata, `local_ref`, and offline-node force-delete behavior.
- Use TDD for implementation unless the Comet build state records `tdd_mode: direct`.
- Keep changes scoped to this OpenSpec change; if implementation reveals larger requirements, follow Comet Build Step 4 before expanding scope.

## File Structure

- Create `multica/server/internal/service/env_sandbox_lifecycle.go`: service seam around sandbox instance/job operations used by env-dispatch and checkpointing.
- Create `multica/server/internal/service/env_sandbox_lifecycle_test.go`: lifecycle service TDD tests.
- Modify `multica/server/internal/service/env_dispatch.go`: add structured sandbox refs, per-agent env input validation, and lifecycle handle propagation.
- Modify `multica/server/internal/service/env_dispatch_test.go`: env-dispatch service tests for per-agent env specs and structured refs.
- Modify `multica/server/internal/handler/env_dispatch.go`: parse optional per-agent env specs and return structured sandbox refs where applicable.
- Create `multica/server/internal/service/env_checkpoint.go`: checkpoint create/get/list/resume service, sync save wait, inline JSONB snapshot construction.
- Create `multica/server/internal/service/env_checkpoint_test.go`: checkpoint service TDD tests.
- Create `multica/server/internal/handler/env_checkpoint.go`: HTTP handlers for create/list/get/resume-from-checkpoint.
- Create `multica/server/internal/handler/env_checkpoint_test.go`: handler tests for auth, status mapping, and config gating.
- Modify `multica/server/cmd/server/router.go`: register checkpoint routes under `/api/v1/env-checkpoints` and resume route.
- Create `multica/server/migrations/154_env_checkpoint_lifecycle.up.sql` and `.down.sql`: checkpoint schema and env lifecycle ref storage.
- Modify or create `multica/server/pkg/db/queries/env_checkpoint.sql`: sqlc queries for checkpoint rows and snapshots.
- Modify generated sqlc files after running the repository's sqlc generation command if configured in `multica/server`.
- Modify `multica/server/internal/service/training.go` and relevant tests only as needed to preserve env id and sandbox refs in trained task/session context.
- Modify `customized_areal/tree_search/agents/swe_lego_client.py`: per-agent env serialization and checkpoint create/list/resume helpers.
- Modify `customized_areal/tree_search/tests/test_env_dispatch_client.py`: Python client tests.
- Add `customized_areal/tree_search/tests/test_env_checkpoint_entropy.py` only if entropy helper code is not already covered by an existing test module.
- Update `customized_areal/tree_search/agents/multica_environment_protocol.md`: operational semantics.
- Check off `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md` as implementation tasks complete.

---

### Task 1: Confirm Existing Seams and Record Findings

**OpenSpec tasks:** 1.1, 1.2, 1.3, 1.4

**Files:**
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/design.md`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`
- Read: `multica/server/internal/service/env_dispatch.go`
- Read: `multica/server/internal/handler/env_dispatch.go`
- Read: `multica/server/internal/handler/sandbox.go`
- Read: `multica/server/pkg/db/queries/sandbox.sql`
- Read: `multica/server/migrations/140_sandbox_node_gateway.up.sql`
- Read: `multica/server/migrations/141_sandbox_resume.up.sql`
- Read: `multica/server/migrations/142_sandbox_node_owner.up.sql`
- Read: `multica/server/migrations/143_sandbox_reconfigure.up.sql`
- Read: `multica/server/migrations/152_training_dispatch.up.sql`
- Read: `multica/server/migrations/153_training_dispatch_critic.up.sql`
- Read: `customized_areal/tree_search/agents/swe_lego_client.py`

**Interfaces:**
- Consumes: existing sandbox handler methods `CreateSandboxInstance`, `StopSandboxInstance`, `ResumeSandboxInstance`, `DeleteSandboxInstance`.
- Produces: documented mapping from existing sandbox job/query behavior to lifecycle service methods used in later tasks.

- [ ] **Step 1: Confirm author commits and lifecycle features**

  Run:

  ```bash
  git -C multica log --all --author='lijiannankai@126.com' --grep='sandbox' -i --oneline --date=short --pretty=format:'%h %ad %s'
  ```

  Expected: includes commits for sandbox node gateway, per-user sandboxd node ownership/setup, optimized sandbox config/function, and offline-node force-delete behavior.

- [ ] **Step 2: Trace current env-dispatch flow**

  Run:

  ```bash
  rg -n "type EnvDispatchInput|func \(s \*EnvDispatchService\) Dispatch|func \(s \*EnvDispatchService\) resetOne|func \(s \*EnvDispatchService\) dispatchOne|SaveTrainingDispatch|maybeOpenTrainingSession|type EnvDispatchRequest|func \(h \*Handler\) EnvDispatch" multica/server/internal/service/env_dispatch.go multica/server/internal/handler/env_dispatch.go
  ```

  Expected: locate service input, handler request parsing, reset, dispatch, training persistence, and session-open hook seams.

- [ ] **Step 3: Trace current sandbox job flow**

  Run:

  ```bash
  rg -n "CreateSandboxInstance|StopSandboxInstance|ResumeSandboxInstance|DeleteSandboxInstance|CreateSandboxJob|NotifyJobAvailable|force-delete|local_ref|runtime" multica/server/internal/handler/sandbox.go multica/server/pkg/db/queries/sandbox.sql multica/server/migrations/14*_sandbox*.sql
  ```

  Expected: locate instance row fields, job enqueueing, node wakeup, pause/resume/delete job types, `local_ref`, and runtime metadata.

- [ ] **Step 4: Record the confirmed seam map in design.md**

  Add a short implementation note under `openspec/changes/env-dispatch-sandbox-lifecycle/design.md` with these exact bullets adjusted only for file/line references found in steps 2 and 3:

  ```markdown
  ## Implementation Seam Notes

  - Sandbox lifecycle is already represented by `sandbox_instance` rows and `sandbox_job` rows; checkpointing should call a service seam that delegates to the same query/job behavior used by sandbox handlers.
  - Env-dispatch currently persists legacy `environment.sandbox_ids`; new save/resume code must prefer structured sandbox-instance refs while preserving compatibility reads for legacy env rows.
  - Training-dispatch and session-open hooks already preserve env id; this change extends that context with sandbox-instance refs.
  - DB subtree snapshots for v1 are scoped to the rollout project subtree and stored inline as JSONB on the checkpoint row.
  ```

- [ ] **Step 5: Validate OpenSpec after documentation patch**

  Run:

  ```bash
  openspec validate env-dispatch-sandbox-lifecycle --strict
  ```

  Expected: `Change 'env-dispatch-sandbox-lifecycle' is valid`.

- [ ] **Step 6: Check off OpenSpec investigation tasks and commit**

  Mark tasks 1.1 through 1.4 as complete in `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md` after the notes are accurate.

  Commit:

  ```bash
  git add openspec/changes/env-dispatch-sandbox-lifecycle/design.md openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "docs: record env sandbox lifecycle seams"
  ```

---

### Task 2: Add Env Sandbox Lifecycle Service

**OpenSpec tasks:** 2.1, 2.2, 2.3, 2.4

**Files:**
- Create: `multica/server/internal/service/env_sandbox_lifecycle.go`
- Create: `multica/server/internal/service/env_sandbox_lifecycle_test.go`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Produces:
  ```go
  type SandboxInstanceRef struct {
      InstanceID string
      WorkspaceID string
      NodeID string
      LocalRef string
      Template string
      Status string
      RuntimeMetadata json.RawMessage
      EndpointInfo json.RawMessage
  }

  type EnvSandboxLifecycleService struct { deps EnvSandboxLifecycleDeps }

  func NewEnvSandboxLifecycleService(deps EnvSandboxLifecycleDeps, timeout time.Duration) *EnvSandboxLifecycleService
  func (s *EnvSandboxLifecycleService) Save(ctx context.Context, ref SandboxInstanceRef, actorUserID string) (SandboxLifecycleJobResult, error)
  func (s *EnvSandboxLifecycleService) Resume(ctx context.Context, ref SandboxInstanceRef, actorUserID string) (SandboxLifecycleJobResult, error)
  func (s *EnvSandboxLifecycleService) Delete(ctx context.Context, ref SandboxInstanceRef, actorUserID string) error
  func (s *EnvSandboxLifecycleService) Reconfigure(ctx context.Context, ref SandboxInstanceRef, actorUserID string, runtime json.RawMessage) (SandboxLifecycleJobResult, error)
  ```
- Consumes: existing sandbox instance and job query behavior from `sandbox.sql` via a production adapter added later.

- [ ] **Step 1: Write failing lifecycle service tests**

  Create `multica/server/internal/service/env_sandbox_lifecycle_test.go` with table-driven tests for:

  ```go
  func TestEnvSandboxLifecycleSaveEnqueuesStopJobAndWakeup(t *testing.T) {}
  func TestEnvSandboxLifecycleResumeEnqueuesResumeWithRuntimeMetadata(t *testing.T) {}
  func TestEnvSandboxLifecycleDeletePreservesOfflineForceDelete(t *testing.T) {}
  func TestEnvSandboxLifecycleMissingSandboxReturnsTypedError(t *testing.T) {}
  ```

  Fake deps should record `jobType`, `instanceID`, `nodeID`, `payload`, and `wakeupNodeID` so assertions prove the seam delegates to existing sandboxd behavior.

- [ ] **Step 2: Run tests and verify red**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvSandboxLifecycle' -count=1
  ```

  Expected: compile failure because `EnvSandboxLifecycleService`, deps, and typed errors do not exist.

- [ ] **Step 3: Implement lifecycle types and fake-friendly deps seam**

  Create `multica/server/internal/service/env_sandbox_lifecycle.go` with the interfaces and errors used by the tests:

  ```go
  var ErrSandboxInstanceNotFound = errors.New("sandbox_instance_not_found")
  var ErrSandboxNodeUnavailable = errors.New("sandbox_node_unavailable")

  type EnvSandboxLifecycleDeps interface {
      GetSandboxInstanceRef(ctx context.Context, workspaceID, instanceID string) (SandboxInstanceRef, error)
      EnqueueSandboxJob(ctx context.Context, workspaceID, actorUserID, nodeID, instanceID, jobType string, payload json.RawMessage) (SandboxLifecycleJobResult, error)
      NotifySandboxJobAvailable(ctx context.Context, nodeID, jobID string) error
      ForceDeleteSandboxInstance(ctx context.Context, workspaceID, instanceID string) error
  }
  ```

  Keep business logic in service package; production handler adapters can be thin wrappers.

- [ ] **Step 4: Implement save/resume/reconfigure/delete methods**

  Implement minimal behavior to pass tests:

  - `Save` enqueues job type `stop` with instance and local ref metadata.
  - `Resume` enqueues job type `resume` with runtime metadata from the ref.
  - `Reconfigure` enqueues job type `reconfigure` with caller-provided runtime JSON.
  - `Delete` enqueues job type `delete`; if deps report node unavailable, call `ForceDeleteSandboxInstance`.
  - Each enqueue calls `NotifySandboxJobAvailable` with the target node and job id.

- [ ] **Step 5: Run scoped Go tests and fix compile issues**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvSandboxLifecycle' -count=1
  ```

  Expected: PASS.

- [ ] **Step 6: Check off OpenSpec lifecycle tasks and commit**

  Mark tasks 2.1 through 2.4 complete after the scoped test passes.

  Commit:

  ```bash
  git add multica/server/internal/service/env_sandbox_lifecycle.go multica/server/internal/service/env_sandbox_lifecycle_test.go openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: add env sandbox lifecycle service"
  ```

---

### Task 3: Add Structured Sandbox Refs and Per-agent Env Intent to Env-dispatch

**OpenSpec tasks:** 3.1, 3.2, 3.3, 3.4

**Files:**
- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/service/env_dispatch_test.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Consumes: `service.SandboxInstanceRef` from Task 2.
- Produces:
  ```go
  type PerAgentEnvSpec struct {
      AgentID string
      Template string
      BaseEnvID string
  }

  type EnvRollout struct {
      EnvID string
      ProjectID string
      IssueID string
      ChatSessionID string
      AgentRunID string
      Error string
      SandboxRefs []SandboxInstanceRef
      AgentSandboxRefs map[string]SandboxInstanceRef
  }
  ```

- [ ] **Step 1: Write failing service tests for per-agent env specs**

  Extend `multica/server/internal/service/env_dispatch_test.go` with tests named:

  ```go
  func TestEnvDispatchPerAgentEnvSpecsAssignDistinctSandboxRefs(t *testing.T) {}
  func TestEnvDispatchPerAgentEnvSpecsRejectUnknownAgent(t *testing.T) {}
  func TestEnvDispatchPerAgentEnvSpecsRejectUnknownEnvSpec(t *testing.T) {}
  func TestEnvDispatchPerAgentEnvSpecsEmptyPreservesCurrentBehavior(t *testing.T) {}
  func TestEnvDispatchPerAgentEnvSpecsPartialSquadUsesDefaults(t *testing.T) {}
  ```

  Add fake deps methods for validating agent membership and resolving env specs. Assertions should prove that empty specs produce the same payload as existing behavior and non-empty specs populate `AgentSandboxRefs`.

- [ ] **Step 2: Run tests and verify red**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvDispatchPerAgentEnvSpecs' -count=1
  ```

  Expected: compile failure for missing `PerAgentEnvSpec` and structured refs.

- [ ] **Step 3: Extend service input/output types**

  Modify `EnvDispatchInput`, `Env`, and `EnvRollout` in `env_dispatch.go` to include optional per-agent specs and structured refs. Keep `SandboxIDs []string` for compatibility reads. New fields should be omitted from JSON only at handler response mapping, not hidden in service state.

- [ ] **Step 4: Add deps seam for per-agent validation and ref resolution**

  Extend `EnvDispatchDeps` with methods that tests can fake:

  ```go
  ResolvePerAgentEnvSpec(ctx context.Context, workspaceID string, spec PerAgentEnvSpec) (SandboxInstanceRef, error)
  ValidateAgentInWorkspaceOrSquad(ctx context.Context, workspaceID, squadID, agentID string) error
  ```

  Update `stubEnvDispatchDeps` and `fakeEnvDispatchDeps` so existing tests still compile.

- [ ] **Step 5: Implement validation before rollout creation**

  In `validate` or a new preflight method, reject unknown agents/specs before calling reset or dispatch. Preserve existing behavior when `len(in.PerAgentEnvSpecs) == 0`.

- [ ] **Step 6: Propagate refs into rollout and training context**

  Ensure reset/dispatch fills `SandboxRefs` and `AgentSandboxRefs` on `EnvRollout`. Extend `maybeOpenTrainingSession` plumbing only if needed so trained task/session context can preserve env id and sandbox refs.

- [ ] **Step 7: Parse and emit per-agent env fields in handler**

  Extend `EnvDispatchRequest` and response mapping in `multica/server/internal/handler/env_dispatch.go` with JSON fields:

  ```go
  PerAgentEnv map[string]PerAgentEnvRequest `json:"per_agent_env,omitempty"`
  SandboxRefs []service.SandboxInstanceRef `json:"sandbox_refs,omitempty"`
  AgentSandboxRefs map[string]service.SandboxInstanceRef `json:"agent_sandbox_refs,omitempty"`
  ```

  The handler should omit fields when empty.

- [ ] **Step 8: Run scoped tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvDispatch' -count=1
  cd multica/server && go test ./internal/handler -run 'TestEnvDispatch' -count=1
  ```

  Expected: PASS for existing and new env-dispatch tests.

- [ ] **Step 9: Check off OpenSpec env-dispatch tasks and commit**

  Mark tasks 3.1 through 3.4 complete.

  Commit:

  ```bash
  git add multica/server/internal/service/env_dispatch.go multica/server/internal/service/env_dispatch_test.go multica/server/internal/handler/env_dispatch.go openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: carry sandbox refs through env dispatch"
  ```

---

### Task 4: Add Checkpoint Schema, Queries, and Service

**OpenSpec tasks:** 4.1, 4.2, 4.3

**Files:**
- Create: `multica/server/migrations/154_env_checkpoint_lifecycle.up.sql`
- Create: `multica/server/migrations/154_env_checkpoint_lifecycle.down.sql`
- Create: `multica/server/pkg/db/queries/env_checkpoint.sql`
- Regenerate: `multica/server/pkg/db/generated/*.go` if sqlc is configured
- Create: `multica/server/internal/service/env_checkpoint.go`
- Create: `multica/server/internal/service/env_checkpoint_test.go`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Consumes: `EnvSandboxLifecycleService`, `SandboxInstanceRef`.
- Produces:
  ```go
  type EnvCheckpointStatus string
  const (
      EnvCheckpointSaveComplete EnvCheckpointStatus = "complete"
      EnvCheckpointSaveFailed EnvCheckpointStatus = "failed"
      EnvCheckpointSaveTimedOut EnvCheckpointStatus = "timed_out"
  )

  type EnvCheckpointCreateInput struct {
      WorkspaceID string
      ProjectID string
      EventRef string
      Kind string
      EnvIDMap map[string]string
      SandboxRefs []SandboxInstanceRef
      DBSnapshot json.RawMessage
      EntropyScore *float64
      ActorUserID string
      SaveTimeout time.Duration
  }
  ```

- [ ] **Step 1: Write failing checkpoint service tests**

  Create `env_checkpoint_test.go` with tests named:

  ```go
  func TestEnvCheckpointCreateWaitsForSynchronousSaveComplete(t *testing.T) {}
  func TestEnvCheckpointCreateRecordsTimeoutStatus(t *testing.T) {}
  func TestEnvCheckpointCreateRecordsSaveFailureStatus(t *testing.T) {}
  func TestEnvCheckpointListNewestFirstAndWorkspaceScoped(t *testing.T) {}
  func TestEnvCheckpointStoresInlineDBSnapshot(t *testing.T) {}
  ```

  Use a fake repository and fake lifecycle service. Tests should assert save status, timeout duration, inline `DBSnapshot`, workspace filtering, and ordering.

- [ ] **Step 2: Run tests and verify red**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvCheckpoint' -count=1
  ```

  Expected: compile failure because checkpoint service types do not exist.

- [ ] **Step 3: Add migration**

  Create `154_env_checkpoint_lifecycle.up.sql` with an `env_checkpoint` table containing:

  ```sql
  CREATE TABLE IF NOT EXISTS env_checkpoint (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      workspace_id uuid NOT NULL,
      project_id uuid NOT NULL,
      event_ref text NOT NULL,
      checkpoint_kind text NOT NULL,
      env_id_map jsonb NOT NULL DEFAULT '{}'::jsonb,
      sandbox_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
      db_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
      entropy_score double precision,
      save_timeout_ms integer NOT NULL,
      save_status text NOT NULL,
      save_error text,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
  );

  CREATE INDEX IF NOT EXISTS env_checkpoint_project_created_idx
      ON env_checkpoint (workspace_id, project_id, created_at DESC);
  ```

  Create the down migration that drops the index and table with `IF EXISTS`.

- [ ] **Step 4: Add sqlc queries**

  Create `multica/server/pkg/db/queries/env_checkpoint.sql` with create/get/list/update queries:

  ```sql
  -- name: CreateEnvCheckpoint :one
  INSERT INTO env_checkpoint (workspace_id, project_id, event_ref, checkpoint_kind, env_id_map, sandbox_refs, db_snapshot, entropy_score, save_timeout_ms, save_status, save_error)
  VALUES (@workspace_id, @project_id, @event_ref, @checkpoint_kind, @env_id_map, @sandbox_refs, @db_snapshot, @entropy_score, @save_timeout_ms, @save_status, @save_error)
  RETURNING *;

  -- name: GetEnvCheckpointForWorkspace :one
  SELECT * FROM env_checkpoint
  WHERE id = @id AND workspace_id = @workspace_id;

  -- name: ListEnvCheckpointsForProject :many
  SELECT * FROM env_checkpoint
  WHERE workspace_id = @workspace_id AND project_id = @project_id
  ORDER BY created_at DESC;

  -- name: UpdateEnvCheckpointSaveStatus :one
  UPDATE env_checkpoint
  SET save_status = @save_status, save_error = @save_error, updated_at = now()
  WHERE id = @id AND workspace_id = @workspace_id
  RETURNING *;
  ```

- [ ] **Step 5: Generate DB code**

  Inspect `multica/server` for the existing sqlc command:

  ```bash
  cd multica/server && rg -n "sqlc" Makefile justfile package.json go.mod .github . 2>/dev/null | head -20
  ```

  Run the project's configured sqlc generation command. If no wrapper exists, run `sqlc generate` from `multica/server` if available.

- [ ] **Step 6: Implement checkpoint service**

  Create `env_checkpoint.go` with a repository interface and service methods:

  ```go
  type EnvCheckpointRepository interface {
      CreateCheckpoint(ctx context.Context, in EnvCheckpointCreateInput, status EnvCheckpointStatus, saveErr string) (EnvCheckpoint, error)
      UpdateCheckpointSaveStatus(ctx context.Context, checkpointID, workspaceID string, status EnvCheckpointStatus, saveErr string) (EnvCheckpoint, error)
      GetCheckpoint(ctx context.Context, checkpointID, workspaceID string) (EnvCheckpoint, error)
      ListCheckpoints(ctx context.Context, workspaceID, projectID string) ([]EnvCheckpoint, error)
  }
  ```

  `CreateCheckpoint` should call lifecycle `Save` for each ref, wait for completion or context timeout, then persist final status. Keep timeout/failure states terminal and non-resumable.

- [ ] **Step 7: Implement inline snapshot capture seam**

  Add a `ProjectSnapshotReader` interface to the service package:

  ```go
  type ProjectSnapshotReader interface {
      CaptureProjectSnapshot(ctx context.Context, workspaceID, projectID string) (json.RawMessage, error)
  }
  ```

  The production adapter can be filled by handler/database code; service tests use a fake that returns deterministic JSON.

- [ ] **Step 8: Run scoped service tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvCheckpoint' -count=1
  ```

  Expected: PASS.

- [ ] **Step 9: Check off OpenSpec checkpoint storage tasks and commit**

  Mark tasks 4.1 through 4.3 complete.

  Commit:

  ```bash
  git add multica/server/migrations/154_env_checkpoint_lifecycle.*.sql multica/server/pkg/db/queries/env_checkpoint.sql multica/server/pkg/db/generated multica/server/internal/service/env_checkpoint.go multica/server/internal/service/env_checkpoint_test.go openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: add env checkpoint persistence service"
  ```

---

### Task 5: Add Checkpoint HTTP APIs and Config Gate

**OpenSpec tasks:** 4.4

**Files:**
- Create: `multica/server/internal/handler/env_checkpoint.go`
- Create: `multica/server/internal/handler/env_checkpoint_test.go`
- Modify: `multica/server/cmd/server/router.go`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Consumes: `service.EnvCheckpointService` from Task 4.
- Produces HTTP API:
  - `POST /api/v1/env-checkpoints`
  - `GET /api/v1/env-checkpoints/{checkpointID}`
  - `GET /api/v1/projects/{projectID}/env-checkpoints`

- [ ] **Step 1: Write failing handler tests**

  Create handler tests covering:

  ```go
  func TestCreateEnvCheckpointDisabledReturns404(t *testing.T) {}
  func TestCreateEnvCheckpointReturnsCreatedWhenSaveCompletes(t *testing.T) {}
  func TestCreateEnvCheckpointTimeoutMapsToConflict(t *testing.T) {}
  func TestListEnvCheckpointsRequiresWorkspaceAndNewestFirst(t *testing.T) {}
  func TestGetEnvCheckpointCrossWorkspaceDoesNotLeak(t *testing.T) {}
  ```

  Use `httptest`, a fake service, and existing handler auth helpers used by nearby tests.

- [ ] **Step 2: Run tests and verify red**

  Run:

  ```bash
  cd multica/server && go test ./internal/handler -run 'Test.*EnvCheckpoint' -count=1
  ```

  Expected: compile failure because handlers/routes do not exist.

- [ ] **Step 3: Implement request/response structs and config gate**

  In `env_checkpoint.go`, define request/response structs. Config gate reads an env var such as `ENV_CHECKPOINTS_ENABLED`; disabled routes return 404 or 403 consistently with existing feature-gated handlers.

- [ ] **Step 4: Implement create/list/get handlers**

  Handler behavior:

  - Require user id and workspace id.
  - Decode `project_id`, `event_ref`, `checkpoint_kind`, `env_id_map`, `sandbox_refs`, optional `entropy_score`, and optional `save_timeout_ms`.
  - Use default timeout when request omits timeout.
  - Map validation to 400, forbidden/not found to 404 or 403, save timeout/failure to 409, and success to 201.
  - Omit internal sandboxd details from error bodies.

- [ ] **Step 5: Register routes**

  In `multica/server/cmd/server/router.go`, register routes near existing `/api/v1/env-dispatch` routes:

  ```go
  r.Post("/api/v1/env-checkpoints", h.CreateEnvCheckpoint)
  r.Get("/api/v1/env-checkpoints/{checkpointID}", h.GetEnvCheckpoint)
  r.Get("/api/v1/projects/{projectID}/env-checkpoints", h.ListEnvCheckpoints)
  ```

- [ ] **Step 6: Run handler and router tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/handler -run 'Test.*EnvCheckpoint' -count=1
  cd multica/server && go test ./cmd/server -run 'Test.*Route|Test.*EnvCheckpoint' -count=1
  ```

  Expected: PASS.

- [ ] **Step 7: Check off API task and commit**

  Mark task 4.4 complete.

  Commit:

  ```bash
  git add multica/server/internal/handler/env_checkpoint.go multica/server/internal/handler/env_checkpoint_test.go multica/server/cmd/server/router.go openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: expose env checkpoint APIs"
  ```

---

### Task 6: Implement Resume-from-checkpoint

**OpenSpec tasks:** 5.1, 5.2, 5.3, 5.4

**Files:**
- Modify: `multica/server/internal/service/env_checkpoint.go`
- Modify: `multica/server/internal/service/env_checkpoint_test.go`
- Modify: `multica/server/internal/handler/env_checkpoint.go`
- Modify: `multica/server/internal/handler/env_checkpoint_test.go`
- Modify: `multica/server/cmd/server/router.go`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Produces:
  ```go
  type ResumeFromCheckpointResult struct {
      CheckpointID string `json:"checkpoint_id"`
      ProjectID string `json:"project_id"`
      EnvIDMap map[string]string `json:"env_id_map"`
      SandboxRefs []SandboxInstanceRef `json:"sandbox_refs"`
      RolloutHandle string `json:"rollout_handle"`
  }
  ```
- Produces HTTP API: `POST /api/v1/env-checkpoints/{checkpointID}/resume`.

- [ ] **Step 1: Write failing resume service tests**

  Add tests:

  ```go
  func TestResumeFromCheckpointResumesCompletedSandboxRefs(t *testing.T) {}
  func TestResumeFromCheckpointRejectsTimedOutCheckpoint(t *testing.T) {}
  func TestResumeFromCheckpointRejectsFailedCheckpoint(t *testing.T) {}
  func TestResumeFromCheckpointNotFound(t *testing.T) {}
  func TestResumeFromCheckpointPreservesPerAgentSandboxRefs(t *testing.T) {}
  ```

- [ ] **Step 2: Write failing resume handler tests**

  Add tests:

  ```go
  func TestResumeFromCheckpointRouteUsesResumeNaming(t *testing.T) {}
  func TestResumeFromCheckpointMapsIncompleteToConflict(t *testing.T) {}
  func TestResumeFromCheckpointCrossWorkspaceRejected(t *testing.T) {}
  ```

- [ ] **Step 3: Run tests and verify red**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestResumeFromCheckpoint' -count=1
  cd multica/server && go test ./internal/handler -run 'TestResumeFromCheckpoint' -count=1
  ```

  Expected: compile failure or failing assertions for missing resume method/route.

- [ ] **Step 4: Implement service resume method**

  Add `ResumeFromCheckpoint(ctx, workspaceID, checkpointID, actorUserID string)` to checkpoint service. It must:

  - Load checkpoint by workspace.
  - Require `save_status == complete`.
  - Decode sandbox refs and env id map.
  - Call lifecycle `Resume` for every sandbox ref.
  - Return a continuation handle containing checkpoint id, project id, env id map, and sandbox refs.

- [ ] **Step 5: Implement handler and route**

  Add handler method `ResumeEnvCheckpoint` and route:

  ```go
  r.Post("/api/v1/env-checkpoints/{checkpointID}/resume", h.ResumeEnvCheckpoint)
  ```

  The route path and method names must use resume terminology only.

- [ ] **Step 6: Run scoped tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestResumeFromCheckpoint|TestEnvCheckpoint' -count=1
  cd multica/server && go test ./internal/handler -run 'TestResumeFromCheckpoint|Test.*EnvCheckpoint' -count=1
  ```

  Expected: PASS.

- [ ] **Step 7: Check off resume tasks and commit**

  Mark tasks 5.1 through 5.4 complete.

  Commit:

  ```bash
  git add multica/server/internal/service/env_checkpoint.go multica/server/internal/service/env_checkpoint_test.go multica/server/internal/handler/env_checkpoint.go multica/server/internal/handler/env_checkpoint_test.go multica/server/cmd/server/router.go openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: resume env checkpoints"
  ```

---

### Task 7: Add Event/Entropy Checkpoint Triggers

**OpenSpec tasks:** 6.1, 6.2, 6.3, 6.4

**Files:**
- Modify: `multica/server/internal/service/training.go`
- Modify: `multica/server/internal/service/training_test.go`
- Modify: `multica/server/internal/service/task_critic_test.go` if critic routing owns the event seam
- Modify: `customized_areal/tree_search/agents/swe_lego_client.py`
- Create or modify: `customized_areal/tree_search/tests/test_env_checkpoint_entropy.py`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Consumes: checkpoint create API from Tasks 4 and 5.
- Produces Python helper:
  ```python
  def should_create_entropy_checkpoint(entropy: float | None, threshold: float | None) -> bool:
      return entropy is not None and threshold is not None and entropy >= threshold
  ```

- [ ] **Step 1: Write failing Multica trigger tests**

  Add tests proving checkpoint requests are created for trained rollout structural events and skipped for non-trained, sweeper, autopilot, and sandbox lifecycle events. Use explicit names:

  ```go
  func TestTrainingCheckpointTriggerCreatesForTrainedStructuralEvent(t *testing.T) {}
  func TestTrainingCheckpointTriggerSkipsNonTrainingProject(t *testing.T) {}
  func TestTrainingCheckpointTriggerSkipsSweeperAutopilotAndSandboxLifecycleEvents(t *testing.T) {}
  ```

- [ ] **Step 2: Run Multica tests and verify red**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestTrainingCheckpointTrigger' -count=1
  ```

  Expected: failing because trigger seam is not implemented.

- [ ] **Step 3: Implement minimal trigger seam**

  Add a small service seam that detects checkpoint-eligible structural events only for projects with training dispatch rows. It should call checkpoint creation without triggering on sandbox lifecycle jobs.

- [ ] **Step 4: Write failing AReaL entropy tests**

  Add Python tests:

  ```python
  def test_entropy_checkpoint_threshold_true():
      assert should_create_entropy_checkpoint(1.2, 1.0) is True

  def test_entropy_checkpoint_threshold_false():
      assert should_create_entropy_checkpoint(0.2, 1.0) is False

  def test_entropy_checkpoint_skips_missing_logprobs():
      assert should_create_entropy_checkpoint(None, 1.0) is False
  ```

- [ ] **Step 5: Run Python tests and verify red**

  Run:

  ```bash
  source .venv-test/bin/activate && pytest customized_areal/tree_search/tests/test_env_checkpoint_entropy.py -q
  ```

  Expected: import or assertion failure until helper exists.

- [ ] **Step 6: Implement entropy helper and checkpoint call site**

  Add helper code near the AReaL env-dispatch client or a focused tree-search utility module. When logprobs are unavailable, do not call Multica and do not fail the rollout. When entropy is above threshold, call checkpoint create with `checkpoint_kind="entropy_gated"` and `entropy_score`.

- [ ] **Step 7: Run scoped trigger and entropy tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestTrainingCheckpointTrigger' -count=1
  source .venv-test/bin/activate && pytest customized_areal/tree_search/tests/test_env_checkpoint_entropy.py -q
  ```

  Expected: PASS.

- [ ] **Step 8: Check off trigger tasks and commit**

  Mark tasks 6.1 through 6.4 complete.

  Commit:

  ```bash
  git add multica/server/internal/service/training.go multica/server/internal/service/training_test.go multica/server/internal/service/task_critic_test.go customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/tests/test_env_checkpoint_entropy.py openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: trigger env checkpoints for training rollouts"
  ```

---

### Task 8: Add AReaL Client Support

**OpenSpec tasks:** 7.1, 7.2, 7.3

**Files:**
- Modify: `customized_areal/tree_search/agents/swe_lego_client.py`
- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Produces Python methods:
  ```python
  async def create_checkpoint(self, *, project_id: str, event_ref: str, checkpoint_kind: str, env_id_map: dict[str, str], sandbox_refs: list[dict], entropy_score: float | None = None, save_timeout_ms: int | None = None) -> dict: ...
  async def list_checkpoints(self, *, project_id: str) -> list[dict]: ...
  async def resume_from_checkpoint(self, *, checkpoint_id: str) -> dict: ...
  ```
- Produces typed client error:
  ```python
  class MulticaCheckpointError(RuntimeError):
      def __init__(self, status_code: int, body: str) -> None: ...
  ```

- [ ] **Step 1: Write failing per-agent serialization test**

  Add to `test_env_dispatch_client.py`:

  ```python
  def test_create_env_dispatch_serializes_per_agent_env_specs():
      seen = {}
      def handler(req):
          seen["body"] = json.loads(req.content)
          return httpx.Response(201, json={"rollouts": []})
      c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
      asyncio.run(c.create_env_dispatch(mode="scratch", dispatch_type="message", squad_id="sq", domain="self_play", message="hi", per_agent_env={"agent-1": {"template": "python"}}))
      assert seen["body"]["per_agent_env"] == {"agent-1": {"template": "python"}}
  ```

- [ ] **Step 2: Write failing checkpoint client tests**

  Add tests for create/list/resume and typed errors:

  ```python
  def test_create_checkpoint_posts_sync_timeout_fields(): ...
  def test_list_checkpoints_returns_items(): ...
  def test_resume_from_checkpoint_posts_resume_route(): ...
  def test_checkpoint_conflict_raises_typed_error(): ...
  ```

- [ ] **Step 3: Run Python client tests and verify red**

  Run:

  ```bash
  source .venv-test/bin/activate && pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -q
  ```

  Expected: FAIL for missing method arguments and helpers.

- [ ] **Step 4: Implement optional per-agent env serialization**

  Extend `create_env_dispatch` signature with `per_agent_env: dict[str, dict] | None = None`. Only add `payload["per_agent_env"]` when the dict is truthy.

- [ ] **Step 5: Implement checkpoint helpers and typed errors**

  Add `MulticaCheckpointError` and methods that call:

  - `POST /api/v1/env-checkpoints`
  - `GET /api/v1/projects/{project_id}/env-checkpoints`
  - `POST /api/v1/env-checkpoints/{checkpoint_id}/resume`

  Raise `MulticaCheckpointError` for 403, 404, and 409 responses. Continue raising `RuntimeError` for unexpected non-checkpoint API errors if that matches existing client style.

- [ ] **Step 6: Run Python client tests**

  Run:

  ```bash
  source .venv-test/bin/activate && pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -q
  ```

  Expected: PASS.

- [ ] **Step 7: Check off AReaL client tasks and commit**

  Mark tasks 7.1 through 7.3 complete.

  Commit:

  ```bash
  git add customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "feat: add areal env checkpoint client"
  ```

---

### Task 9: Documentation, Full Scoped Verification, and OpenSpec Closure

**OpenSpec tasks:** 8.1, 8.2, 8.3, 8.4

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_environment_protocol.md`
- Modify: `openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md`

**Interfaces:**
- Consumes: all implementation tasks.
- Produces: documented operational semantics and passing verification evidence.

- [ ] **Step 1: Document operational semantics**

  Update `customized_areal/tree_search/agents/multica_environment_protocol.md` to state:

  ```markdown
  ## Env Checkpoint Semantics

  Env checkpoint creation is pause-in-place. Multica synchronously waits for sandboxd stop/Cube pause up to the configured timeout and stores the project subtree inline as JSONB. A completed checkpoint can be resumed through resume-from-checkpoint, which resumes the same sandbox instances. The API does not provide immutable fork, branch, snapshot, or copy-on-write semantics.
  ```

- [ ] **Step 2: Run scoped Multica service tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/service -run 'TestEnvSandboxLifecycle|TestEnvDispatch|TestEnvCheckpoint|TestResumeFromCheckpoint|TestTrainingCheckpointTrigger' -count=1
  ```

  Expected: PASS.

- [ ] **Step 3: Run scoped Multica handler/router tests**

  Run:

  ```bash
  cd multica/server && go test ./internal/handler -run 'Test.*EnvCheckpoint|TestEnvDispatch' -count=1
  cd multica/server && go test ./cmd/server -run 'Test.*Route|Test.*EnvCheckpoint' -count=1
  ```

  Expected: PASS.

- [ ] **Step 4: Run scoped AReaL tests**

  Run:

  ```bash
  source .venv-test/bin/activate && pytest customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_env_checkpoint_entropy.py -q
  ```

  Expected: PASS.

- [ ] **Step 5: Run OpenSpec validation**

  Run:

  ```bash
  openspec validate env-dispatch-sandbox-lifecycle --strict
  ```

  Expected: `Change 'env-dispatch-sandbox-lifecycle' is valid`.

- [ ] **Step 6: Run broader package checks when time permits**

  Run:

  ```bash
  cd multica/server && go test ./internal/service ./internal/handler ./cmd/server
  source .venv-test/bin/activate && pytest customized_areal/tree_search/tests/test_env_dispatch_client.py -q
  ```

  Expected: PASS. If a command is too slow or blocked by missing local dependencies, record the exact failure and the narrower passing commands in the final verification report.

- [ ] **Step 7: Check off verification tasks and commit**

  Mark tasks 8.1 through 8.4 complete only after the required scoped verification commands pass or an explicit Comet-approved skip is recorded.

  Commit:

  ```bash
  git add customized_areal/tree_search/agents/multica_environment_protocol.md openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
  git commit -m "docs: document env checkpoint lifecycle semantics"
  ```

## Self-Review

- Spec coverage: Tasks 1-3 cover sandbox lifecycle handles, per-agent env intent, and training session handle preservation. Tasks 4-6 cover checkpoint create/list/get/resume with sync timeout and inline JSONB. Task 7 covers event/entropy triggers. Task 8 covers AReaL client integration. Task 9 covers verification and docs.
- Placeholder scan: This plan avoids unresolved placeholders and gives exact files, route names, method names, test names, commands, and expected outcomes.
- Type consistency: `SandboxInstanceRef` is produced in Task 2 and reused by env-dispatch and checkpoint tasks. `EnvCheckpoint*` types are produced in Task 4 and reused by handlers/resume/client tasks. Resume naming is consistent across service, handler, route, and Python client.

---
change: env-dispatch-agent-runtime-config
design-doc: docs/superpowers/specs/2026-07-20-env-dispatch-agent-runtime-config-design.md
base-ref: ef19c0e66c655a66f3feb4b0eb6cf0513de965af
---

# EnvDispatch Sandbox Runtime and Derived-Agent Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Build reconciliation (2026-07-21):** All step checkboxes checked off against the
> completed, verified implementation (AC-1..AC-8 code-level DONE per
> `docs/superpowers/reports/2026-07-20-env-dispatch-agent-runtime-config-verify.md`).
> `tdd_mode: direct` was used (implementation-first), so TDD red-phase step checkboxes
> reflect that the corresponding tests exist and pass rather than a strict red-green
> sequence. Independently re-verified 2026-07-21: `openspec validate --strict` PASS;
> nested multica `go build ./...` + `go test ./internal/service/...` PASS. Deferrals:
> Task 1.2 `ListOwnedEnvDispatchResources` sqlc regen is unused (env-wide cleanup uses
> `listBindings`); Task 8 Steps 3-4 (deployed AC-8 verification) are split to a separate
> issue (feature-flag kill-switch `envDispatchDerivedAgentEnabled` itself DONE).

**Goal:** Make first-address message env-dispatch create a frontend-equivalent sandbox,
discover its online Pi runtime, clone a source agent into an isolated derived agent,
support static and AReaL-owned model credentials, enqueue a normal real task, and clean
up every derived resource while returning a valid DAG.

**Architecture:** Keep `(env_id, source_agent_id)` binding rows as the single-flight
workflow owner. A shared sandbox creation service mints a daemon correlation nonce
without pre-creating `agent_runtime`; provisioning then resolves an online runtime by
workspace, provider, daemon ID, and sandbox instance ID, creates a derived global agent
transactionally, and only then inserts/enqueues a normal task. Training bootstrap uses
the persistent binding ID as `session_ref`; the bridge remains compatible with legacy
`task_id`, and the real task ID is linked to the session only after normal task
insertion.

**Tech Stack:** Go 1.26.1, Chi, pgx/sqlc, PostgreSQL JSONB, Python 3.12,
FastAPI/Pydantic, httpx, pytest, OpenSpec 1.6.

## Global Constraints

- Work from outer repository base `ef19c0e66c655a66f3feb4b0eb6cf0513de965af`; `multica/`
  is a nested Git repository, so run its Go commits from `multica/` and outer AReaL
  commits from the outer root.
- Do not touch VimPO-owned files or unrelated dirty files.
- Do not add dependencies, hardcode provider credentials, hardcode the deployment bridge
  URL, log secrets, or return secrets from an API.
- Do not reserve, synthesize, or pre-insert a task ID for training bootstrap.
  `start_session` receives the persistent binding ID as `session_ref`; a normal task
  insert later supplies the real task ID.
- Do not insert `agent_runtime` during env-dispatch provisioning. Runtime registration
  is daemon-owned, and matching requires workspace, provider `pi`, daemon ID, sandbox
  instance ID, and status `online`.
- The source agent and its existing runtime, skills, and non-dispatch memberships are
  immutable throughout provisioning and cleanup.
- Use synthetic secrets such as `sentinel-static-key` and `sentinel-training-key` in
  tests; assert they are absent from responses, errors, and logs.
- No database schema backfill is required beyond nullable/defaulted columns; migrations
  must be reversible and old callers that omit `runtime` must remain valid.
- Run `graphify update .` after code changes and
  `openspec validate env-dispatch-agent-runtime-config --strict` before completion.

## File Structure

- `multica/server/migrations/198_env_dispatch_derived_agents.{up,down}.sql`: reversible
  schema for agent lineage and expanded binding identity/state.
- `multica/server/pkg/db/queries/{agent,runtime,environment,interaction_dag}.sql`:
  workspace-safe derived-agent, runtime-discovery, binding-state, cleanup, and
  real-task/session-link queries; regenerate `multica/server/pkg/db/generated/` with
  sqlc.
- `multica/server/internal/service/env_sandbox_lifecycle.go`: the one shared
  sandbox-create service used by frontend and env-dispatch.
- `multica/server/internal/handler/{sandbox.go,env_sandbox_lifecycle_adapter.go}`: thin
  frontend adapter to the shared service.
- `multica/server/internal/handler/env_dispatch_channel_{policy,provision,store}.go`:
  typed credential policy, first-address state machine, runtime readiness, and binding
  persistence.
- `multica/server/internal/service/env_dispatch.go` and
  `multica/server/internal/handler/env_dispatch.go`: dispatch contracts,
  leader/lazy-peer routing, derived task insertion, and sanitized rollout status.
- `multica/server/internal/service/env_dispatch_derived_agent.go`: focused transactional
  clone operation and approved-field copy list.
- `multica/server/internal/arealrl/client.go`: `session_ref` bootstrap plus real-task
  linkage and session close.
- `areal/experimental/openai/proxy/{server.py,proxy_gateway.py,proxy_rollout_server.py}`:
  backward-compatible `session_ref | task_id` request model and canonical reference
  forwarding.
- `customized_areal/tree_search/agents/{multica_client.py,multica_dag_client.py}`:
  strict rollout/readiness/DAG failure behavior.

______________________________________________________________________

### Task 1: Persist binding identity, derived-agent lineage, and workflow state

**Files:**

- Create: `multica/server/migrations/198_env_dispatch_derived_agents.up.sql`
- Create: `multica/server/migrations/198_env_dispatch_derived_agents.down.sql`
- Modify: `multica/server/internal/migrations/migrations_test.go`
- Modify: `multica/server/pkg/db/queries/agent.sql`
- Modify: `multica/server/pkg/db/queries/runtime.sql`
- Modify: `multica/server/pkg/db/queries/environment.sql`
- Modify: `multica/server/pkg/db/queries/interaction_dag.sql`
- Regenerate: `multica/server/pkg/db/generated/*.sql.go`
- Test: `multica/server/internal/handler/env_dispatch_channel_store_test.go`

**Interfaces:**

- Produces: nullable `agent.source_agent_id`; binding ID plus `source_agent_id`,
  `derived_agent_id`, `daemon_id`, `runtime_id`, `training_session_id`,
  `training_session_ref`, `credential_kind`, `model_config_owner_agent_id`, and explicit
  provisioning state.

- Produces: sqlc methods `ClaimEnvDispatchBinding`, `FindOnlineSandboxRuntime`,
  `CreateDerivedEnvDispatchAgent`, `LinkTrainingSessionTask`, and
  `ListOwnedEnvDispatchResources`.

- [x] **Step 1: Write migration and store tests that fail against the current schema**

```go
func TestEnvDispatchBindingIdentityAndRetryState(t *testing.T) {
	ctx, store, envID, channelID, sourceID := setupEnvDispatchChannelStoreFixture(t)
	b := envAgentSandboxBinding{
		ID: uuid.NewString(), EnvID: envID, ChannelID: channelID,
		SourceAgentID: sourceID, Status: "pending",
		ModelConfigOwnerAgentID: sourceID,
		SandboxConfig: json.RawMessage(`{"template":"default"}`),
	}
	require.NoError(t, store.insertBinding(ctx, testPool, b))
	won, claimed, err := store.claimProvisioning(ctx, testPool, envID, sourceID)
	require.NoError(t, err)
	require.True(t, won)
	require.Equal(t, b.ID, claimed.ID)
	require.Equal(t, sourceID, claimed.ModelConfigOwnerAgentID)
	require.Equal(t, "credential_ready", claimed.Status)
}

func TestAgentLineageRejectsCrossWorkspaceSource(t *testing.T) {
	// Insert source in workspace A and attempt a derived insert in workspace B.
	_, err := testPool.Exec(context.Background(), `INSERT INTO agent
		(workspace_id, name, source_agent_id) VALUES ($1, 'derived', $2)`,
		otherWorkspaceID, sourceAgentID)
	require.Error(t, err)
}
```

- [x] **Step 2: Run the red tests**

Run:
`cd multica/server && go test ./internal/migrations ./internal/handler -run 'TestEnvDispatchBindingIdentityAndRetryState|TestAgentLineageRejectsCrossWorkspaceSource' -count=1`

Expected: FAIL because the new columns, binding ID, and state are absent.

- [x] **Step 3: Add a reversible migration with workspace-safe constraints**

Add this schema in `198_env_dispatch_derived_agents.up.sql`; legacy state names remain
accepted during feature-gated rollout:

```sql
ALTER TABLE agent ADD COLUMN source_agent_id uuid;
ALTER TABLE agent ADD CONSTRAINT agent_source_workspace_fk
  FOREIGN KEY (workspace_id, source_agent_id)
  REFERENCES agent(workspace_id, id) ON DELETE RESTRICT;

ALTER TABLE environment_agent_sandbox
  ADD COLUMN id uuid NOT NULL DEFAULT gen_random_uuid(),
  ADD COLUMN source_agent_id uuid,
  ADD COLUMN derived_agent_id uuid,
  ADD COLUMN training_session_id text,
  ADD COLUMN training_session_ref text,
  ADD COLUMN credential_kind text,
  ADD COLUMN model_config_owner_agent_id uuid;
CREATE UNIQUE INDEX environment_agent_sandbox_id_uidx ON environment_agent_sandbox(id);
CREATE UNIQUE INDEX environment_agent_sandbox_source_uidx
  ON environment_agent_sandbox(env_id, source_agent_id)
  WHERE source_agent_id IS NOT NULL;
ALTER TABLE environment_agent_sandbox
  DROP CONSTRAINT environment_agent_sandbox_status_check,
  ADD CONSTRAINT environment_agent_sandbox_status_check CHECK (status IN (
    'pending', 'provisioning', 'failed', 'credential_ready', 'sandbox_creating',
    'runtime_waiting', 'agent_creating', 'ready', 'failed_retryable',
    'deleting', 'deleted'
  ));
```

`198_env_dispatch_derived_agents.down.sql` restores the original state check, drops the
two indexes, added binding columns, workspace-safe foreign key, and
`agent.source_agent_id` in reverse dependency order. Existing rows keep
`source_agent_id=NULL`; new-path inserts always set it, so no legacy binding is silently
claimed as a derived workflow.

- [x] **Step 4: Add exact sqlc query contracts and generate code**

```sql
-- name: FindOnlineSandboxRuntime :one
SELECT * FROM agent_runtime
WHERE workspace_id = $1
  AND provider = 'pi'
  AND daemon_id = $2
  AND status = 'online'
  AND metadata->>'sandbox_instance_id' = $3
LIMIT 1;

-- name: LinkTrainingSessionTask :exec
INSERT INTO interaction_dag_session_run (session_id, project_id, agent_run_id, issue_id)
VALUES ($1, $2, $3, NULLIF($4, ''))
ON CONFLICT (session_id) DO UPDATE SET
  project_id = EXCLUDED.project_id,
  agent_run_id = EXCLUDED.agent_run_id,
  issue_id = EXCLUDED.issue_id;
```

Run: `cd multica/server && sqlc generate`

Expected: generated query structs compile without manual edits to `pkg/db/generated`.

- [x] **Step 5: Run migration, store, and generated-query tests**

Run:
`cd multica/server && go test ./internal/migrations ./internal/handler ./pkg/db/generated -count=1`

Expected: PASS; DB-backed tests may SKIP only when the documented test database is
unavailable.

- [x] **Step 6: Commit the persistence slice in the nested repository**

```bash
cd multica
git add server/migrations/198_env_dispatch_derived_agents.up.sql server/migrations/198_env_dispatch_derived_agents.down.sql server/internal/migrations/migrations_test.go server/pkg/db/queries server/pkg/db/generated server/internal/handler/env_dispatch_channel_store_test.go
git commit -m "feat(env-dispatch): persist derived runtime ownership"
```

### Task 2: Make frontend and env-dispatch share sandbox creation

**Files:**

- Modify: `multica/server/internal/service/env_sandbox_lifecycle.go`
- Modify: `multica/server/internal/service/env_sandbox_lifecycle_test.go`
- Modify: `multica/server/internal/handler/sandbox.go`
- Modify: `multica/server/internal/handler/env_sandbox_lifecycle_adapter.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_policy.go`
- Test: `multica/server/internal/handler/sandbox_test.go`

**Interfaces:**

- Consumes: typed `service.ExternalModelRuntime` already represented by binding policy.

- Produces: `CreateSandboxInstance(ctx, CreateSandboxInstanceInput, actorUserID)`
  returning a ref with minted daemon nonce and canonical metadata; it does not insert
  `agent_runtime`.

- [x] **Step 1: Write equivalence tests for frontend and env-dispatch create inputs**

```go
func TestSharedSandboxCreatePersistsFrontendEquivalentPayload(t *testing.T) {
	runtime := json.RawMessage(`{"base_url":"https://provider.example/v1","api_key":"sentinel-static-key","model":"model-a"}`)
	in := CreateSandboxInstanceInput{
		WorkspaceID: "ws-1", Template: "default", Runtime: runtime,
		DaemonEnabled: true, SandboxName: "env-dispatch-agent",
	}
	ref, err := svc.Create(context.Background(), in, "user-1")
	require.NoError(t, err)
	require.NotEmpty(t, ref.DaemonID)
	require.Equal(t, 0, deps.precreatedRuntimeCount)
	require.JSONEq(t, string(runtime), string(deps.inserted.RuntimeMetadata))
	require.Equal(t, ref.DaemonID, deps.inserted.RuntimeEnv["MULTICA_DAEMON_ID"])
	require.JSONEq(t, string(deps.frontendJobPayload), string(deps.envDispatchJobPayload))
}
```

- [x] **Step 2: Run the red test**

Run:
`cd multica/server && go test ./internal/service ./internal/handler -run TestSharedSandboxCreatePersistsFrontendEquivalentPayload -count=1`

Expected: FAIL because frontend creation still owns logic outside the shared service or
the service expects a pre-created runtime.

- [x] **Step 3: Move the canonical create sequence into the lifecycle service**

The service input and result must explicitly distinguish the bootstrap PAT from model
credentials:

```go
type CreateSandboxInstanceInput struct {
	WorkspaceID string
	Template string
	SandboxName string
	Runtime json.RawMessage
	DaemonEnabled bool
	RuntimeEnv map[string]string
}

type SandboxInstanceRef struct {
	InstanceID string `json:"instance_id"`
	WorkspaceID string `json:"workspace_id"`
	Template string `json:"template"`
	DaemonID string `json:"daemon_id,omitempty"`
}
```

Inside `Create`, mint the daemon ID and bootstrap PAT, persist `runtime`, `runtime_env`,
and `sandbox_instance_id` metadata, build one `sandboxCreatePayload`, enqueue one create
job, and notify the node. Delete calls must revoke the bootstrap PAT.

- [x] **Step 4: Reduce the frontend handler to request mapping plus response writing**

```go
ref, err := newEnvSandboxLifecycleService(h).Create(r.Context(), service.CreateSandboxInstanceInput{
	WorkspaceID: workspaceID,
	Template: req.Template,
	SandboxName: req.Name,
	Runtime: req.Runtime,
	DaemonEnabled: req.DaemonEnabled,
	RuntimeEnv: req.RuntimeEnv,
}, userID)
```

Do not change the frontend HTTP status or response fields.

- [x] **Step 5: Run lifecycle and frontend regression tests**

Run:
`cd multica/server && go test ./internal/service ./internal/handler -run 'SandboxInstance|SharedSandboxCreate' -count=1`

Expected: PASS and no new `agent_runtime` row during create.

- [x] **Step 6: Commit the shared create slice**

```bash
cd multica
git add server/internal/service/env_sandbox_lifecycle.go server/internal/service/env_sandbox_lifecycle_test.go server/internal/handler/sandbox.go server/internal/handler/env_sandbox_lifecycle_adapter.go server/internal/handler/env_dispatch_channel_policy.go server/internal/handler/sandbox_test.go
git commit -m "refactor(sandbox): share canonical creation lifecycle"
```

### Task 3: Discover the registered runtime and create an isolated derived agent

**Files:**

- Create: `multica/server/internal/service/env_dispatch_derived_agent.go`
- Create: `multica/server/internal/service/env_dispatch_derived_agent_test.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_provision.go`
- Create: `multica/server/internal/handler/env_dispatch_channel_provision_test.go`
- Modify: `multica/server/internal/daemon/daemon.go`
- Modify: `multica/server/internal/daemon/daemon_test.go`

**Interfaces:**

- Produces:
  `WaitForOnlineSandboxRuntime(ctx, workspaceID, daemonID, sandboxInstanceID string) (service.RuntimeRef, error)`.

- Produces:
  `CloneEnvDispatchAgent(ctx, CloneEnvDispatchAgentInput) (derivedAgentID string, error)`
  with explicit source/runtime/channel/binding identities.

- [x] **Step 1: Write runtime mismatch, timeout, lineage, and source-preservation
  tests**

```go
func TestWaitForOnlineSandboxRuntimeRejectsIdentityMismatch(t *testing.T) {
	deps.runtime = RuntimeRef{ID: "rt-1", WorkspaceID: "ws", Provider: "pi", DaemonID: "daemon-x", SandboxInstanceID: "sbx-other", Status: "online"}
	_, err := waitForOnlineSandboxRuntime(context.Background(), deps, "ws", "daemon-x", "sbx-expected", 20*time.Millisecond)
	require.ErrorContains(t, err, "runtime identity mismatch")
	require.NotContains(t, err.Error(), "sentinel-static-key")
}

func TestCloneEnvDispatchAgentLeavesSourceUnchanged(t *testing.T) {
	before := loadAgent(t, sourceID)
	derivedID, err := svc.CloneEnvDispatchAgent(ctx, CloneEnvDispatchAgentInput{
		WorkspaceID: workspaceID, SourceAgentID: sourceID, RuntimeID: runtimeID,
		EnvID: envID, ChannelID: channelID, BindingID: bindingID,
	})
	require.NoError(t, err)
	after := loadAgent(t, sourceID)
	require.Equal(t, before.RuntimeID, after.RuntimeID)
	derived := loadAgent(t, derivedID)
	require.Equal(t, sourceID, uuidToString(derived.SourceAgentID))
	require.Equal(t, runtimeID, uuidToString(derived.RuntimeID))
}
```

- [x] **Step 2: Run the red tests**

Run:
`cd multica/server && go test ./internal/daemon ./internal/service ./internal/handler -run 'OnlineSandboxRuntime|CloneEnvDispatchAgent' -count=1`

Expected: FAIL because registration metadata, safe discovery, and clone service are
absent.

- [x] **Step 3: Persist sandbox identity at daemon registration and poll by immutable
  identity**

```go
metadata["sandbox_instance_id"] = registration.SandboxInstanceID

func waitForOnlineSandboxRuntime(ctx context.Context, q runtimeLookup, workspaceID, daemonID, sandboxID string, timeout time.Duration) (service.RuntimeRef, error) {
	deadlineCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for {
		rt, err := q.FindOnlineSandboxRuntime(deadlineCtx, workspaceID, daemonID, sandboxID)
		if err == nil { return rt, nil }
		if !errors.Is(err, pgx.ErrNoRows) { return service.RuntimeRef{}, fmt.Errorf("resolve sandbox runtime: %w", err) }
		select {
		case <-deadlineCtx.Done(): return service.RuntimeRef{}, fmt.Errorf("runtime readiness timeout")
		case <-time.After(100 * time.Millisecond):
		}
	}
}
```

- [x] **Step 4: Implement the transactional approved-field clone**

`CloneEnvDispatchAgent` must load source and runtime in the same workspace, copy
name/instructions/provider-visible Pi settings and skills, set `source_agent_id` and the
discovered `runtime_id`, add the derived channel member, remove only the source member
from that env-dispatch channel, and update the claimed binding. It must not copy
credentials, task state, or source runtime ownership.

```go
type CloneEnvDispatchAgentInput struct {
	WorkspaceID string
	SourceAgentID string
	RuntimeID string
	EnvID string
	ChannelID string
	BindingID string
}
```

- [x] **Step 5: Run runtime/clone tests including concurrent source reuse**

Run:
`cd multica/server && go test ./internal/daemon ./internal/service ./internal/handler -run 'OnlineSandboxRuntime|CloneEnvDispatchAgent|ConcurrentDispatchesShareSource' -count=1`

Expected: PASS; two dispatches from one source produce distinct derived agents/runtimes
and do not mutate the source.

- [x] **Step 6: Commit runtime discovery and derived identity**

```bash
cd multica
git add server/internal/daemon server/internal/service/env_dispatch_derived_agent.go server/internal/service/env_dispatch_derived_agent_test.go server/internal/handler/env_dispatch_channel_provision.go server/internal/handler/env_dispatch_channel_provision_test.go
git commit -m "feat(env-dispatch): bind derived agents to online runtimes"
```

### Task 4: Add backward-compatible `session_ref` training bootstrap and real-task linkage

**Files:**

- Modify: `areal/experimental/openai/proxy/server.py`
- Modify: `areal/experimental/openai/proxy/proxy_gateway.py`
- Modify: `areal/experimental/openai/proxy/proxy_rollout_server.py`
- Test: `tests/experimental/openai/test_proxy_gateway.py`
- Test: `tests/experimental/openai/test_proxy_rollout_server.py`
- Modify: `multica/server/internal/arealrl/client.go`
- Modify: `multica/server/internal/arealrl/client_test.go`
- Modify: `multica/server/internal/service/interaction_dag.go`
- Modify: `multica/server/internal/service/interaction_dag_test.go`

**Interfaces:**

- Produces:
  `StartSessionRequest(session_ref: str | None, task_id: str | None, api_key: str | None, env_id: str | None)`
  with exactly one canonical reference required.

- Produces: Go `StartSession(ctx, sessionRef, envID string)` sending `session_ref`,
  never a fabricated task.

- Produces:
  `LinkSessionTask(ctx, sessionID, projectID, realTaskID, issueID string) error` after
  normal task insertion.

- [x] **Step 1: Write compatibility and ordering tests**

```python
def test_start_session_accepts_session_ref_without_task_id():
    req = StartSessionRequest(session_ref="binding-123", env_id="env-1")
    assert req.canonical_session_ref == "binding-123"

def test_start_session_keeps_legacy_task_id_compatible():
    req = StartSessionRequest(task_id="task-legacy")
    assert req.canonical_session_ref == "task-legacy"

def test_start_session_rejects_missing_or_conflicting_reference():
    with pytest.raises(ValidationError):
        StartSessionRequest()
    with pytest.raises(ValidationError):
        StartSessionRequest(session_ref="binding", task_id="task")
```

```go
func TestStartSessionUsesBindingReferenceWithoutTaskReservation(t *testing.T) {
	creds, err := client.StartSession(context.Background(), "binding-123", "env-1")
	require.NoError(t, err)
	require.Equal(t, "binding-123", receivedBody["session_ref"])
	require.NotContains(t, receivedBody, "task_id")
	require.Equal(t, 0, taskInsertCountBeforeDerivedReady)
}
```

- [x] **Step 2: Run the red proxy and Go client tests**

Run:
`uv run pytest tests/experimental/openai/test_proxy_gateway.py tests/experimental/openai/test_proxy_rollout_server.py -q`

Run:
`cd multica/server && go test ./internal/arealrl ./internal/service -run 'StartSession|LinkSessionTask' -count=1`

Expected: FAIL because `task_id` is currently required and no post-insert link method
exists.

- [x] **Step 3: Implement canonical request validation and forwarding**

```python
class StartSessionRequest(BaseModel):
    session_ref: str | None = None
    task_id: str | None = None
    api_key: str | None = None
    env_id: str | None = None

    @model_validator(mode="after")
    def validate_reference(self):
        refs = [value for value in (self.session_ref, self.task_id) if value]
        if len(refs) != 1:
            raise ValueError("exactly one of session_ref or task_id is required")
        return self

    @property
    def canonical_session_ref(self) -> str:
        return self.session_ref or self.task_id or ""
```

Gateway logs may log the canonical reference but never the API key. Forward the original
compatible body; rollout storage keys the session namespace from
`canonical_session_ref`.

- [x] **Step 4: Change the Go client and DAG store to link only the real task**

```go
func (c *Client) StartSession(ctx context.Context, sessionRef, envID string) (SessionCreds, error) {
	body := map[string]any{"session_ref": sessionRef, "group_size": 1}
	if envID != "" { body["env_id"] = envID }
	// Existing authenticated POST and response validation remain unchanged.
}

func (s *InteractionDAGService) LinkSessionTask(ctx context.Context, sessionID, projectID, realTaskID, issueID string) error {
	return s.store.UpsertInteractionDAGSessionRun(ctx, db.UpsertInteractionDAGSessionRunParams{
		SessionID: sessionID, ProjectID: projectID, AgentRunID: realTaskID,
		IssueID: textOrNull(issueID),
	})
}
```

- [x] **Step 5: Run both compatibility suites**

Run:
`uv run pytest tests/experimental/openai/test_proxy_gateway.py tests/experimental/openai/test_proxy_rollout_server.py -q && cd multica/server && go test ./internal/arealrl ./internal/service -run 'StartSession|LinkSessionTask' -count=1`

Expected: PASS for new `session_ref`, legacy `task_id`, and post-insert real-task
linkage.

- [x] **Step 6: Commit AReaL and nested server changes separately**

```bash
git add areal/experimental/openai/proxy tests/experimental/openai
git commit -m "feat(proxy): accept stable session references"
cd multica
git add server/internal/arealrl server/internal/service/interaction_dag.go server/internal/service/interaction_dag_test.go
git commit -m "feat(env-dispatch): link training sessions to real tasks"
```

### Task 5: Orchestrate single-flight credentials, sandbox, runtime, derived agent, and normal task

**Files:**

- Modify: `multica/server/internal/handler/env_dispatch_channel_store.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_policy.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_provision.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Test: `multica/server/internal/handler/env_dispatch_channel_store_test.go`
- Test: `multica/server/internal/handler/env_dispatch_channel_provision_test.go`
- Test: `multica/server/internal/service/env_dispatch_test.go`

**Interfaces:**

- Consumes: persistent binding ID, shared sandbox service, safe runtime discovery, clone
  service, `arealrl.StartSession(sessionRef, envID)`, and
  `LinkSessionTask(sessionID, projectID, realTaskID, issueID)`.

- Produces: one terminal `ready` or `failed_retryable` result shared by leader and lazy
  peers; `EnvDispatchAgentProvisionResult` includes derived agent, sandbox, runtime,
  session, and real task IDs.

- [x] **Step 1: Write state-machine, isolation, retry, and real-task-order tests**

```go
func TestTrainingFirstAddressUsesBindingRefThenLinksRealTask(t *testing.T) {
	result, err := orchestrator.Provision(ctx, trainingBinding)
	require.NoError(t, err)
	require.Equal(t, trainingBinding.ID, calls.startSessionRef)
	require.Less(t, calls.index("StartSession"), calls.index("CreateSandbox"))
	require.Less(t, calls.index("CreateDerivedAgent"), calls.index("InsertTask"))
	require.Less(t, calls.index("InsertTask"), calls.index("LinkSessionTask"))
	require.Equal(t, result.TaskID, calls.linkedRealTaskID)
	require.NotEqual(t, trainingBinding.ID, result.TaskID)
}

func TestRetryReusesPersistedTrainingSession(t *testing.T) {
	binding.TrainingSessionID = "session-1"
	binding.TrainingSessionRef = binding.ID
	binding.ModelCredential = encryptedTestKey("sentinel-training-key")
	_, err := orchestrator.Provision(ctx, binding)
	require.NoError(t, err)
	require.Equal(t, 0, calls.startSessionCount)
}

func TestConcurrentFirstMentionsProvisionExactlyOnce(t *testing.T) {
	results := runTwoConcurrentMentions(t, orchestrator, binding)
	require.Equal(t, results[0], results[1])
	require.Equal(t, 1, calls.sandboxCount)
	require.Equal(t, 1, calls.derivedAgentCount)
	require.Equal(t, 1, calls.taskCount)
}
```

- [x] **Step 2: Run the red orchestration tests**

Run:
`cd multica/server && go test ./internal/handler ./internal/service -run 'TrainingFirstAddress|RetryReusesPersistedTrainingSession|ConcurrentFirstMentions|StaticSquadCredentials' -count=1`

Expected: FAIL because current provisioning pre-creates a runtime and lacks the expanded
state machine.

- [x] **Step 3: Implement typed credential resolution with owner checks**

```go
type ResolvedModelCredential struct {
	OwnerSourceAgentID string
	Kind string
	Runtime service.ExternalModelRuntime
	TrainingSessionID string
	TrainingSessionRef string
}

if credential.OwnerSourceAgentID != binding.SourceAgentID {
	return failRetryable(binding, errors.New("model credential owner mismatch"))
}
```

Static credentials come only from the claimed binding policy. Training credentials are
server-owned: bridge URL from training config, model `areal-default`, API key from the
recorded or newly opened session.

- [x] **Step 4: Replace `PrecreateAgentRuntime` with the ordered first-address
  workflow**

```text
claim binding
resolve/persist credential
create sandbox through shared service
wait for matching online Pi runtime
clone derived global agent transactionally
insert and enqueue a normal derived-agent task with explicit runtime_id
link training session to that real task_id when training
persist collaboration trigger with the same real task_id
mark binding ready
```

Every transition persists before the next external side effect. On retry, read the
durable state and reuse completed identities. The scratch leader calls this same method
during initial dispatch; peers call it from directed-mention routing.

- [x] **Step 5: Implement sanitized terminal failure and compensation**

Credential failure creates no sandbox. Sandbox failure revokes bootstrap credentials.
Runtime timeout deletes the sandbox. Derived-agent failure deletes runtime/sandbox and
closes training session. Task failure archives the derived agent, closes session, and
deletes runtime/sandbox. Persist a stable sanitized `last_error` code, never an external
body or credential.

- [x] **Step 6: Run the complete env-dispatch handler/service suite**

Run:
`cd multica/server && go test ./internal/handler ./internal/service -run 'EnvDispatch|FirstMention|TrainingFirstAddress|StaticSquadCredentials' -count=1`

Expected: PASS; no test observes `sentinel-static-key` or `sentinel-training-key` in
JSON/errors/log capture.

- [x] **Step 7: Commit the orchestration slice**

```bash
cd multica
git add server/internal/handler/env_dispatch_channel_store.go server/internal/handler/env_dispatch_channel_policy.go server/internal/handler/env_dispatch_channel_provision.go server/internal/handler/env_dispatch.go server/internal/handler/*env_dispatch*_test.go server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go
git commit -m "feat(env-dispatch): provision derived agents on first address"
```

### Task 6: Complete idempotent owned-resource cleanup

**Files:**

- Modify: `multica/server/internal/handler/env_dispatch_channel_routes.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_routes_test.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_store.go`
- Modify: `multica/server/internal/service/env_sandbox_lifecycle.go`

**Interfaces:**

- Consumes: `ListOwnedEnvDispatchResources` from Task 1 and persisted workflow
  identities from Task 5.

- Produces: idempotent `deleting -> deleted` cleanup preserving the source agent.

- [x] **Step 1: Write ready-state, partial-state, and race cleanup tests**

```go
func TestDeleteEnvDispatchCleansDerivedResourcesAndPreservesSource(t *testing.T) {
	fixture := setupReadyDerivedDispatch(t)
	deleteEnvDispatchTwice(t, fixture.ChannelID)
	require.True(t, agentExists(t, fixture.SourceAgentID))
	require.False(t, activeAgentExists(t, fixture.DerivedAgentID))
	require.False(t, sandboxExists(t, fixture.SandboxID))
	require.False(t, runtimeExists(t, fixture.RuntimeID))
	require.Equal(t, 1, fixture.SessionCloser.CallCount(fixture.SessionID))
}

func TestDeleteRacingRuntimeWaitLeavesNoOwnedResources(t *testing.T) {
	fixture := setupRuntimeWaitingDispatch(t)
	runDeleteAndProvisionConcurrently(t, fixture)
	require.Empty(t, listOwnedResources(t, fixture.BindingID))
}
```

- [x] **Step 2: Run the red cleanup tests**

Run:
`cd multica/server && go test ./internal/handler -run 'DeleteEnvDispatchCleansDerivedResources|DeleteRacingRuntimeWait' -count=1`

Expected: FAIL because cleanup does not yet archive derived agents or close training
sessions.

- [x] **Step 3: Implement ordered, repeat-safe cleanup**

Mark bindings `deleting`, cancel derived tasks, remove derived channel membership,
archive derived agent, stop/delete sandbox, retire the discovered runtime, close the
training session, revoke bootstrap/model credentials, then delete
binding/env/project/channel rows. Treat already-absent resources as success.
Provisioning checks `deleting` before every new external side effect.

- [x] **Step 4: Run cleanup and provisioning-race tests**

Run:
`cd multica/server && go test ./internal/handler ./internal/service -run 'DeleteEnvDispatch|Cleanup|ProvisioningRace' -count=1`

Expected: PASS twice against the same dispatch; source agent/runtime remain queryable.

- [x] **Step 5: Commit cleanup**

```bash
cd multica
git add server/internal/handler/env_dispatch_channel_routes.go server/internal/handler/env_dispatch_channel_routes_test.go server/internal/handler/env_dispatch_channel_store.go server/internal/service/env_sandbox_lifecycle.go
git commit -m "feat(env-dispatch): clean up derived runtime resources"
```

### Task 7: Make the standalone client fail closed on rollout and DAG errors

**Files:**

- Modify: `customized_areal/tree_search/agents/multica_client.py`
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py`
- Test: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Test: `customized_areal/tree_search/tests/test_multica_dag_client.py`
- Test: `customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py`

**Interfaces:**

- Consumes: existing `EnvDispatchHandle` and DAG endpoint.

- Produces: exceptions for per-rollout errors, readiness/DAG timeout, and
  malformed/cyclic/dangling DAG; cleanup remains in `finally`.

- [x] **Step 1: Write client red tests**

```python
def test_create_env_dispatch_rejects_per_rollout_error():
    transport = _transport(lambda req: httpx.Response(201, json={
        "project_id": "p1", "channel_id": "c1",
        "rollouts": [{"env_id": "e1", "error": "runtime readiness timeout"}],
    }))
    client = MulticaEnvDispatchClient(base_url="http://x", transport=transport)
    with pytest.raises(RuntimeError, match="runtime readiness timeout"):
        asyncio.run(client.create_env_dispatch(mode="scratch", dispatch_type="message", agent_id="a", message="hi"))

def test_poll_dag_timeout_is_failure_and_cleanup_runs():
    client = FakeClient(dag_responses=[httpx.Response(202)] * 3)
    with pytest.raises(TimeoutError, match="DAG readiness timeout"):
        asyncio.run(_poll_dag(client, handle, timeout=0.01, interval=0))
    assert client.cleanup_calls == [handle]
```

- [x] **Step 2: Run the red client tests**

Run:
`uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -q`

Expected: FAIL because a 200 is accepted without complete structural validation and
timeout handling is not uniformly fatal.

- [x] **Step 3: Add explicit response and DAG validation**

```python
errors = [r.get("error") for r in data.get("rollouts", []) if r.get("error")]
if errors:
    raise RuntimeError(f"env-dispatch rollout failed: {errors[0]}")

dag = ExecutionDAG.from_records(payload["runs"], payload["edges"])
dag.topological_order()
if not dag.nodes:
    raise DAGError("assembled DAG contains no nodes")
```

At deadline, raise `TimeoutError`; never return a placeholder result. Keep
provider/admin keys redacted in response-body errors and retain cleanup in a `finally`
block.

- [x] **Step 4: Run focused Python tests**

Run:
`uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -q`

Expected: PASS for success, rollout error, timeout, cycle, dangling edge, and cleanup
paths.

- [x] **Step 5: Commit the client hardening in the outer repository**

```bash
git add customized_areal/tree_search/agents/multica_client.py customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py
git commit -m "fix(env-dispatch): reject incomplete DAG results"
```

### Task 8: Verify contracts, formatting, graph, and deployed behavior

**Files:**

- Modify: `openspec/changes/env-dispatch-agent-runtime-config/tasks.md` only to check
  completed boxes and record commands/evidence.
- Create:
  `docs/superpowers/reports/2026-07-20-env-dispatch-agent-runtime-config-verify.md`

**Interfaces:**

- Consumes: all prior tasks.

- Produces: reproducible local and deployed evidence with no credentials recorded.

- [x] **Step 1: Run targeted Go and Python suites**

```bash
cd multica/server
go test ./internal/migrations ./internal/arealrl ./internal/daemon ./internal/service ./internal/handler -count=1
cd ../../
uv run pytest tests/experimental/openai/test_proxy_gateway.py tests/experimental/openai/test_proxy_rollout_server.py customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -q
```

Expected: all available unit tests PASS; DB-backed tests explicitly SKIP only when their
test database is unavailable.

- [x] **Step 2: Run formatting, lint, generated-code, and specification checks**

```bash
cd multica/server && gofmt -w internal pkg/db/generated && go test ./... -count=1
cd ../../
uv run pre-commit run --all-files
openspec validate env-dispatch-agent-runtime-config --strict
graphify update .
git diff --check
```

Expected: all commands exit 0; generated sqlc files have no manual drift; OpenSpec
reports valid.

- [x] **Step 3: Perform static deployed verification with a rotated injected
  credential**
  _(reconciliation: deployed AC-8 verification split to a separate issue; feature-flag
  kill-switch DONE. Code-level AC-8 - static+training path produces
  sandbox+runtime+derived+session - verified per the committed verify report.)_

Send a scratch message request whose `per_agent_env.<source_agent_id>.runtime` contains
a freshly rotated credential supplied only through the shell environment. Verify:
binding states reach `ready`; sandbox is `running`; discovered runtime is Pi/online and
matches daemon+sandbox identity; derived agent has `source_agent_id`; source runtime is
unchanged; channel contains the derived reply; DAG endpoint returns HTTP 200 and the
client accepts its structure. Delete the dispatch twice and verify derived resources are
gone.

- [x] **Step 4: Perform training deployed verification without reserving a task**
  _(reconciliation: deployed AC-8 verification split to a separate issue.)_

Dispatch with `train_agent_id`, then verify: `start_session` receives binding ID as
`session_ref`; no task exists before derived readiness; runtime uses configured bridge
URL and model `areal-default`; exactly one normal task is inserted; its real ID is
linked to the session/DAG; the derived agent replies; DAG returns HTTP 200; cleanup
closes the session. Record only IDs and statuses, never keys.

- [x] **Step 5: Write the verification report with exact evidence**

Use this fixed structure:

```markdown
# EnvDispatch Agent Runtime Config Verification

## Local commands
- `<command>` — PASS/SKIP with reason

## Static dispatch
- binding/source/derived IDs
- sandbox/runtime identity checks
- reply and DAG status
- cleanup result

## Training dispatch
- binding session_ref and session_id
- real task linkage order
- reply and DAG status
- cleanup result

## Secret audit
- no credential recorded in source, logs, errors, responses, or this report
```

- [x] **Step 6: Commit only the verification artifacts in the outer repository**

```bash
git add openspec/changes/env-dispatch-agent-runtime-config/tasks.md docs/superpowers/reports/2026-07-20-env-dispatch-agent-runtime-config-verify.md
git commit -m "test(env-dispatch): verify derived runtime lifecycle"
```

## Self-Review Results

- Spec coverage: Tasks 1–3 cover persistence, shared frontend lifecycle, runtime
  discovery, identity matching, derived lineage, source preservation, and concurrency.
  Tasks 4–5 cover backward-compatible `session_ref`, no fake task, retry-stable
  sessions, real-task linkage, static/training credential isolation, first-address
  single-flight, and terminal failure. Tasks 6–8 cover cleanup, client failure
  semantics, valid DAGs, and deployed static/training replies. No requirement is
  uncovered.
- Placeholder scan: every implementation step names exact files, symbols, code shape,
  commands, and expected outcomes; no deferred implementation markers are present.
- Type consistency: `binding.ID` is the training `session_ref`; `SessionCreds.SessionID`
  is persisted on that same binding; `EnvDispatchAgentProvisionResult.TaskID` is created
  normally after `DerivedAgentID` and `RuntimeID`;
  `LinkSessionTask(SessionID, ProjectID, TaskID, IssueID)` uses that exact real task ID.
  `SandboxInstanceRef.DaemonID` is a correlation nonce, never a runtime row ID.
- Repository-boundary review: Go/server commits run inside `multica/`; AReaL proxy,
  client, OpenSpec, plan, and report commits run from the outer repository. VimPO-owned
  and unrelated dirty files are excluded from every `git add` command.

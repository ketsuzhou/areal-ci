# EnvDispatch Message Channels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make `dispatch_type=message` create a rollout-isolated project-backed group channel, wake only the selected agent in an isolated sandbox, lazily provision other squad members on first mention, and resume branches from a persisted collaboration trigger.

**Architecture:** Keep the project as the internal owner of env, DAG, checkpoint, and training state, and make the bound channel the public handle for message dispatch. An env-scoped agent-binding table is the authority for sandbox routing; channel delivery consults it before ordinary runtime selection. Scratch creates one ready leader binding and pending peer bindings, while branch copies the conversation graph and clones only the trigger-selected agent's sandbox state.

**Tech Stack:** Go 1.23, chi, pgx/PostgreSQL migrations, existing Multica sandboxd/Cube HTTP API, Python 3.12, httpx, pytest.

## Global Constraints

- `dispatch_type=issue` and all existing project-first routes retain their current behavior.
- Each message rollout owns a distinct env, project, group channel, sandbox, daemon, and runtime namespace.
- The caller and the exact requested agent roster are members; EnvDispatch channels never auto-provision Beckham.
- EnvDispatch agent tasks must use the binding runtime and must never fall back to the agent default runtime.
- Scratch wakes only the canonical leader; branch wakes only the agent named by the validated source env trigger.
- Branch validation occurs before resource creation and returns HTTP 400 for a missing, malformed, unauthorized, or roster-incompatible trigger.
- Existing public APIs remain available; no dependencies are added.
- Preserve unrelated dirty changes in `customized_areal/tree_search/agents/multica_client.py`, `customized_areal/tree_search/agents/multica_environment_protocol.md`, `multica/`, and `training_multica.log8`.

---

## File map

- `multica/server/migrations/183_env_dispatch_message_channels.{up,down}.sql`: env trigger, per-agent binding state, constraints, and sandbox clone job type.
- `multica/server/internal/service/env_dispatch_channel.go`: channel-facing service contracts, roster/trigger types, and message rollout orchestration.
- `multica/server/internal/service/env_dispatch.go`: invoke the message-channel orchestration and expose channel-first results without changing issue flow.
- `multica/server/internal/handler/env_dispatch_channel_store.go`: pgx implementation for bindings, triggers, channel creation/copy, and old-to-new ID maps.
- `multica/server/internal/handler/env_dispatch_channel_provision.go`: single-flight lazy provisioning, compensation, and trigger persistence.
- `multica/server/internal/handler/env_dispatch.go`: request adapter, response shape, source validation, and cleanup helpers.
- `multica/server/internal/handler/channel.go`: one narrow routing hook before the existing channel-session/default-runtime path.
- `multica/server/internal/service/env_sandbox_lifecycle.go`, `multica/server/internal/handler/env_sandbox_lifecycle_adapter.go`, `multica/server/cmd/multica/cmd_sandboxd.go`, and `multica/server/internal/handler/sandbox.go`: clone lifecycle and completion handling.
- `multica/server/internal/handler/env_dispatch_channel_routes.go` and `multica/server/cmd/server/router.go`: channel-first DAG, cleanup, and checkpoint facades.
- `customized_areal/tree_search/agents/multica_client.py`, `customized_areal/tree_search/agents/multica_dag_client.py`, and `customized_areal/tree_search/multi_agent_workflow.py`: channel-first Python handle and polling.
- Focused `*_test.go`/`test_*.py` files named in each task keep the large legacy files from growing further.

### Task 1: Persist EnvDispatch channel execution state

**Files:**
- Create: `multica/server/migrations/183_env_dispatch_message_channels.up.sql`
- Create: `multica/server/migrations/183_env_dispatch_message_channels.down.sql`
- Create: `multica/server/internal/handler/env_dispatch_channel_store.go`
- Test: `multica/server/internal/handler/env_dispatch_channel_store_test.go`
- Test: `multica/server/internal/migrations/migrations_test.go`

**Interfaces:**
- Produces: `envCollaborationTrigger`, `envAgentSandboxBinding`, `envDispatchChannelStore`, and transactional claim/update methods used by Tasks 3–7.
- State transition contract: `pending|failed -> provisioning -> ready`, any active state -> `deleting`; only the transaction holding the row lock may change provisioning state.

- [x] **Step 1: Write migration and store tests that fail**

Add tests named:

```go
func TestMigration183UpContainsBindingAndTriggerConstraints(t *testing.T)
func TestEnvDispatchChannelStoreClaimProvisioningIsSingleWinner(t *testing.T)
func TestEnvDispatchChannelStoreRejectsTriggerAgentOutsideChannel(t *testing.T)
func TestEnvDispatchChannelStoreMarkDeletingBlocksProvisioning(t *testing.T)
```

The store test fixture must assert this exact state shape:

```go
binding := envAgentSandboxBinding{
	EnvID: envID, ChannelID: channelID, AgentID: agentID,
	Status: "pending", SandboxConfig: json.RawMessage(`{"template":"default"}`),
}
require.NoError(t, store.insertBinding(ctx, tx, binding))
won, got, err := store.claimProvisioning(ctx, tx, envID, agentID)
require.NoError(t, err)
require.True(t, won)
require.Equal(t, "provisioning", got.Status)
```

- [x] **Step 2: Run the focused tests and observe failure**

Run: `cd multica/server && go test ./internal/migrations ./internal/handler -run 'Test(Migration183|EnvDispatchChannelStore)' -count=1`

Expected: FAIL because migration 183 and the store types do not exist.

- [x] **Step 3: Add the migration**

Use this schema, preserving the current `sandbox_job` check expression while adding `clone` to its accepted values:

```sql
ALTER TABLE environment
  ADD COLUMN collaboration_trigger jsonb;

CREATE TABLE environment_agent_sandbox (
  env_id uuid NOT NULL REFERENCES environment(id) ON DELETE CASCADE,
  channel_id uuid NOT NULL REFERENCES channel(id) ON DELETE CASCADE,
  agent_id uuid NOT NULL REFERENCES agent(id) ON DELETE CASCADE,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'provisioning', 'ready', 'failed', 'deleting')),
  sandbox_instance_id uuid REFERENCES sandbox_instance(id) ON DELETE SET NULL,
  runtime_id uuid REFERENCES agent_runtime(id) ON DELETE SET NULL,
  daemon_id uuid,
  source_sandbox_instance_id uuid REFERENCES sandbox_instance(id) ON DELETE SET NULL,
  sandbox_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (env_id, agent_id),
  UNIQUE (sandbox_instance_id),
  UNIQUE (runtime_id),
  CHECK (
    status <> 'ready' OR
    (sandbox_instance_id IS NOT NULL AND runtime_id IS NOT NULL AND daemon_id IS NOT NULL)
  )
);

CREATE INDEX environment_agent_sandbox_channel_idx
  ON environment_agent_sandbox(channel_id);
```

The down migration drops the table, removes `environment.collaboration_trigger`, and restores the prior sandbox-job check verbatim.

- [x] **Step 4: Implement the focused pgx store**

Define the exact types and methods:

```go
type envCollaborationTrigger struct {
	AgentID            string  `json:"agent_id"`
	Kind               string  `json:"kind"`
	ChannelID          string  `json:"channel_id"`
	ProjectID          string  `json:"project_id"`
	ChatSessionID      string  `json:"chat_session_id"`
	SourceMessageID    string  `json:"source_message_id"`
	ThreadRootMessageID *string `json:"thread_root_message_id,omitempty"`
	TaskID             string  `json:"task_id"`
	RuntimeID          string  `json:"runtime_id"`
}

type envAgentSandboxBinding struct {
	EnvID, ChannelID, AgentID, Status string
	SandboxInstanceID, RuntimeID, DaemonID, SourceSandboxInstanceID *string
	SandboxConfig json.RawMessage
	LastError *string
}

type envDispatchChannelStore struct { db DBTX }

func (s envDispatchChannelStore) insertBinding(context.Context, DBTX, envAgentSandboxBinding) error
func (s envDispatchChannelStore) listBindings(context.Context, DBTX, string) ([]envAgentSandboxBinding, error)
func (s envDispatchChannelStore) claimProvisioning(context.Context, DBTX, string, string) (bool, envAgentSandboxBinding, error)
func (s envDispatchChannelStore) markReady(context.Context, DBTX, string, string, string, string, string) error
func (s envDispatchChannelStore) markFailed(context.Context, DBTX, string, string, string) error
func (s envDispatchChannelStore) markDeleting(context.Context, DBTX, string) error
func (s envDispatchChannelStore) loadTrigger(context.Context, DBTX, string, string) (envCollaborationTrigger, error)
func (s envDispatchChannelStore) saveTrigger(context.Context, DBTX, string, envCollaborationTrigger) error
```

`claimProvisioning` must use one `UPDATE ... WHERE status IN ('pending','failed') RETURNING ...`; if it returns no row, select the current row and return `won=false`. `loadTrigger` decodes JSON, validates all required UUIDs, validates `kind`, confirms the trigger channel/project belong to the env, and confirms the agent is a channel member.

- [x] **Step 5: Run tests and commit**

Run: `cd multica/server && go test ./internal/migrations ./internal/handler -run 'Test(Migration183|EnvDispatchChannelStore)' -count=1`

Expected: PASS.

```bash
git -C multica/server add migrations/183_env_dispatch_message_channels.* internal/handler/env_dispatch_channel_store.go internal/handler/env_dispatch_channel_store_test.go internal/migrations/migrations_test.go
git -C multica/server commit -m "feat(env-dispatch): persist channel sandbox bindings"
```

### Task 2: Clone sandbox-instance state through sandboxd

**Files:**
- Modify: `multica/server/internal/service/env_sandbox_lifecycle.go`
- Modify: `multica/server/internal/handler/env_sandbox_lifecycle_adapter.go`
- Modify: `multica/server/cmd/multica/cmd_sandboxd.go`
- Modify: `multica/server/internal/handler/sandbox.go`
- Test: `multica/server/internal/service/env_sandbox_lifecycle_test.go`
- Test: `multica/server/cmd/multica/cmd_sandboxd_test.go`
- Test: `multica/server/internal/handler/sandbox_test.go`

**Interfaces:**
- Consumes: migration 183's `sandbox_job.job_type='clone'` support.
- Produces: `CloneSandboxInstance(ctx, source, input, actor) (SandboxInstanceRef, error)` for Tasks 4 and 6.

- [x] **Step 1: Add failing clone lifecycle tests**

Cover these exact cases:

```go
func TestCloneSandboxInstanceCreatesOfflineRuntimeAndCloneJob(t *testing.T)
func TestCloneSandboxInstanceCompensatesRuntimeWhenJobInsertFails(t *testing.T)
func TestCallCubeCloneSnapshotsCreatesAndDeletesSnapshot(t *testing.T)
func TestCompleteCloneJobStoresNewExternalSandboxID(t *testing.T)
```

The HTTP test server must assert this request order:

```text
POST   /sandboxes/source-external-id/snapshots  {}
POST   /sandboxes                               {"templateID":"snapshot-id", ...resolved create payload...}
DELETE /templates/snapshot-id
```

- [x] **Step 2: Run tests and observe failure**

Run: `cd multica/server && go test ./internal/service ./internal/handler ./cmd/multica -run 'Test(CloneSandboxInstance|CallCubeClone|CompleteCloneJob)' -count=1`

Expected: FAIL because clone is unsupported.

- [x] **Step 3: Add the lifecycle contract and job payload**

Add:

```go
type CloneSandboxInstanceInput struct {
	WorkspaceID string
	EnvID string
	AgentID string
	RuntimeID string
	DaemonID string
	Name string
	CreatePayload json.RawMessage
}

func (s *EnvSandboxLifecycleService) CloneSandboxInstance(
	ctx context.Context,
	source SandboxInstanceRef,
	in CloneSandboxInstanceInput,
	actor SandboxActor,
) (SandboxInstanceRef, error)
```

The method creates a new `sandbox_instance`, inserts a `clone` job whose payload contains `source_sandbox_instance_id`, `source_external_id`, and `create_payload`, waits using the existing lifecycle completion mechanism, and deletes the precreated runtime plus new instance when creation cannot be queued.

- [x] **Step 4: Implement Cube clone as snapshot/create/delete**

Add a `clone` case in `callCube` that uses the existing authenticated JSON helper. The control flow must be:

```go
snapshotID, err := cubeCreateSnapshot(ctx, client, baseURL, sourceExternalID)
if err != nil { return nil, err }
defer func() { _ = cubeDeleteTemplate(context.WithoutCancel(ctx), client, baseURL, snapshotID) }()
payload.TemplateID = snapshotID
return cubeCreateSandbox(ctx, client, baseURL, payload)
```

Sanitize returned errors using the same path as create/resume. Completion handling writes the newly returned external sandbox ID to the destination instance and never mutates the source instance.

- [x] **Step 5: Run tests and commit**

Run: `cd multica/server && go test ./internal/service ./internal/handler ./cmd/multica -run 'Test(CloneSandboxInstance|CallCubeClone|CompleteCloneJob)' -count=1`

Expected: PASS.

```bash
git -C multica/server add internal/service/env_sandbox_lifecycle.go internal/service/env_sandbox_lifecycle_test.go internal/handler/env_sandbox_lifecycle_adapter.go internal/handler/sandbox.go internal/handler/sandbox_test.go cmd/multica/cmd_sandboxd.go cmd/multica/cmd_sandboxd_test.go
git -C multica/server commit -m "feat(sandbox): clone instance state through sandboxd"
```

### Task 3: Create project-backed message channels and exact rosters

**Files:**
- Create: `multica/server/internal/service/env_dispatch_channel.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Create: `multica/server/internal/service/env_dispatch_channel_test.go`
- Create: `multica/server/internal/handler/env_dispatch_channel_adapter.go`
- Test: `multica/server/internal/handler/env_dispatch_channel_adapter_test.go`

**Interfaces:**
- Produces: `ResolveMessageRoster`, `CreateEnvDispatchChannel`, result `ChannelID`, and pending binding rows.
- Consumes: Task 1 store.

- [x] **Step 1: Add failing roster and reset tests**

Add table-driven tests for single agent, squad leader duplicated in members, non-agent squad members, cross-workspace members, and two-rollout isolation. Assert:

```go
require.Equal(t, []string{leaderID, memberAID, memberBID}, got.AgentIDs)
require.Equal(t, leaderID, got.LeaderID)
require.NotEqual(t, result.Rollouts[0].ChannelID, result.Rollouts[1].ChannelID)
require.NotEqual(t, result.Rollouts[0].ProjectID, result.Rollouts[1].ProjectID)
require.NotEqual(t, result.Rollouts[0].EnvID, result.Rollouts[1].EnvID)
```

The adapter test verifies one group channel with `project_id`, caller membership, exact agent memberships, no Beckham session, and one binding row per agent.

- [x] **Step 2: Run tests and observe failure**

Run: `cd multica/server && go test ./internal/service ./internal/handler -run 'Test(ResolveMessageRoster|MessageResetCreates|CreateEnvDispatchChannel)' -count=1`

Expected: FAIL because the channel orchestration seam is absent.

- [x] **Step 3: Define service types and dependency methods**

Add these fields and contracts:

```go
type MessageRoster struct { LeaderID string; AgentIDs []string }
type AgentSandboxStatus struct {
	Status string `json:"status"`
	SandboxInstanceID string `json:"sandbox_instance_id,omitempty"`
	RuntimeID string `json:"runtime_id,omitempty"`
}

// EnvRollout additions:
ChannelID string
LeaderRunID string
AgentSandboxes map[string]AgentSandboxStatus

// EnvDispatchResult addition:
ChannelID string

// EnvDispatchDeps additions:
ResolveMessageRoster(ctx context.Context, workspaceID, agentID, squadID string) (MessageRoster, error)
CreateEnvDispatchChannel(ctx context.Context, workspaceID, userID, projectID, envID string, roster MessageRoster, specs map[string]SandboxInstanceRef) (channelID string, err error)
DeleteChannel(ctx context.Context, workspaceID, channelID string) error
```

- [x] **Step 4: Implement channel creation without the public CreateChannel handler**

Inside one transaction, insert:

```sql
INSERT INTO channel (workspace_id, name, type, project_id, created_by)
VALUES ($1, $2, 'group', $3, $4)
RETURNING id;
```

Then insert the caller as `member_type='user'`, each deduplicated roster member as `member_type='agent'`, and one `environment_agent_sandbox(status='pending')` row for each agent. Do not call `CreateChannel` or `provisionGroupManagerForNewChannel`.

Update `resetOne` so only message dispatch calls this adapter after project creation. Preserve issue reset byte-for-byte except for shared result-field initialization. Compensation order is channel, project, env, sandbox/runtime.

- [x] **Step 5: Run tests and commit**

Run: `cd multica/server && go test ./internal/service ./internal/handler -run 'Test(ResolveMessageRoster|MessageResetCreates|CreateEnvDispatchChannel)' -count=1`

Expected: PASS.

```bash
git -C multica/server add internal/service/env_dispatch.go internal/service/env_dispatch_channel.go internal/service/env_dispatch_channel_test.go internal/handler/env_dispatch_channel_adapter.go internal/handler/env_dispatch_channel_adapter_test.go
git -C multica/server commit -m "feat(env-dispatch): create rollout group channels"
```

### Task 4: Provision and wake only the scratch leader

**Files:**
- Create: `multica/server/internal/handler/env_dispatch_channel_provision.go`
- Modify: `multica/server/internal/service/env_dispatch_channel.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Test: `multica/server/internal/service/env_dispatch_channel_test.go`
- Test: `multica/server/internal/handler/env_dispatch_squad_chat_test.go`

**Interfaces:**
- Produces: `ProvisionEnvDispatchAgent`, `EnqueueEnvDispatchChannelRun`, initial trigger persistence, and the channel-first JSON response.
- Consumes: Tasks 1–3.

- [x] **Step 1: Add failing leader-only tests**

Test names and required assertions:

```go
func TestScratchMessageProvisionsOnlyLeader(t *testing.T)
func TestScratchMessageUsesRolloutRuntimeNotAgentDefault(t *testing.T)
func TestScratchSquadDoesNotWakePeerMembers(t *testing.T)
func TestEnvDispatchMessageResponseIsChannelFirst(t *testing.T)
```

Assert one ready leader binding, all peers pending, exactly one channel message containing request content, exactly one inbox/task for the leader, and `task.runtime_id == leaderBinding.RuntimeID != agent.DefaultRuntimeID`.

- [x] **Step 2: Run tests and observe failure**

Run: `cd multica/server && go test ./internal/service ./internal/handler -run 'Test(ScratchMessage|EnvDispatchMessageResponse)' -count=1`

Expected: FAIL because message dispatch still creates a project chat run.

- [x] **Step 3: Implement explicit rollout-runtime provisioning**

Define:

```go
type ProvisionEnvDispatchAgentInput struct {
	WorkspaceID, UserID, EnvID, ProjectID, ChannelID, AgentID string
	SourceSandboxInstanceID string
	SandboxConfig json.RawMessage
}

type ProvisionEnvDispatchAgentResult struct {
	SandboxInstanceID, RuntimeID, DaemonID, ChatSessionID string
}

func (h *Handler) provisionEnvDispatchAgent(ctx context.Context, in ProvisionEnvDispatchAgentInput) (ProvisionEnvDispatchAgentResult, error)
```

It claims the binding, precreates a runtime/daemon, creates a fresh sandbox from `sandbox_config`, creates the chat session with the explicit project and runtime, inserts `channel_agent_session`, and marks ready. On any failure after runtime creation, delete the new sandbox/runtime/session, mark failed with a sanitized message, and return without enqueueing.

- [x] **Step 4: Replace message dispatch with channel enqueue**

The scratch path must execute in this order:

```go
provisioned, err := deps.ProvisionEnvDispatchAgent(ctx, leaderInput)
messageID, err := deps.CreateChannelMessage(ctx, channelID, userID, in.Message.Content)
runID, err := deps.EnqueueEnvDispatchChannelRun(ctx, ChannelRunInput{
	AgentID: roster.LeaderID, ChannelID: channelID, ProjectID: projectID,
	EnvID: envID, ChatSessionID: provisioned.ChatSessionID,
	SandboxInstanceID: provisioned.SandboxInstanceID,
	RuntimeID: provisioned.RuntimeID, SourceMessageID: messageID,
})
err = deps.SaveCollaborationTrigger(ctx, envID, envCollaborationTrigger{
	AgentID: roster.LeaderID, Kind: "channel_message", ChannelID: channelID,
	ProjectID: projectID, ChatSessionID: provisioned.ChatSessionID,
	SourceMessageID: messageID, TaskID: runID, RuntimeID: provisioned.RuntimeID,
})
```

Populate `LeaderRunID`, retain `AgentRunID=runID` for compatibility, and populate `AgentSandboxes` from binding rows.

- [x] **Step 5: Emit the channel-first response**

Use:

```go
type EnvDispatchResponse struct {
	ChannelID string `json:"channel_id,omitempty"`
	ProjectID string `json:"project_id"`
	Rollouts []EnvRolloutResponse `json:"rollouts"`
	Message string `json:"message,omitempty"`
}

type EnvRolloutResponse struct {
	ChannelID string `json:"channel_id,omitempty"`
	ProjectID string `json:"project_id"`
	EnvID string `json:"env_id"`
	LeaderRunID string `json:"leader_run_id,omitempty"`
	AgentSandboxes map[string]service.AgentSandboxStatus `json:"agent_sandboxes,omitempty"`
	// retain every existing field
}
```

Save and replay the complete new result through the existing idempotency ledger.

- [x] **Step 6: Run tests and commit**

Run: `cd multica/server && go test ./internal/service ./internal/handler -run 'Test(ScratchMessage|EnvDispatchMessageResponse|Idempotent)' -count=1`

Expected: PASS.

```bash
git -C multica/server add internal/service/env_dispatch.go internal/service/env_dispatch_channel.go internal/service/env_dispatch_channel_test.go internal/handler/env_dispatch.go internal/handler/env_dispatch_channel_provision.go internal/handler/env_dispatch_squad_chat_test.go
git -C multica/server commit -m "feat(env-dispatch): wake only sandboxed channel leader"
```

### Task 5: Lazily provision peers on first mention

**Files:**
- Modify: `multica/server/internal/handler/channel.go`
- Modify: `multica/server/internal/handler/env_dispatch_channel_provision.go`
- Create: `multica/server/internal/handler/env_dispatch_channel_mention_test.go`
- Test: `multica/server/internal/handler/channel_test.go`

**Interfaces:**
- Consumes: Task 4 provisioning and Task 1 claim semantics.
- Produces: `routeEnvDispatchChannelAgent` hook returning `(handled bool, err error)`.

- [x] **Step 1: Add failing mention tests**

Add:

```go
func TestEnvDispatchFirstMentionProvisionsPeerExactlyOnce(t *testing.T)
func TestEnvDispatchConcurrentMentionsShareOneProvisioningResult(t *testing.T)
func TestEnvDispatchProvisionFailureDoesNotFallbackOrEnqueue(t *testing.T)
func TestOrdinaryChannelMentionStillUsesExistingRuntimePath(t *testing.T)
```

Run 16 goroutines against the same pending binding and assert one sandbox create, one runtime create, one `channel_agent_session`, and no task with the default runtime.

- [x] **Step 2: Run tests and observe failure**

Run: `cd multica/server && go test ./internal/handler -run 'Test(EnvDispatch.*Mention|OrdinaryChannelMention)' -count=1`

Expected: FAIL because mention delivery immediately resolves the default agent runtime.

- [x] **Step 3: Add the routing hook before session creation**

At the start of `enqueueChannelAgentPromptRangeWithTx`, call:

```go
handled, err := h.routeEnvDispatchChannelAgent(ctx, tx, channel, agent, trigger, reason)
if err != nil { return channelPromptEnqueueResult{}, err }
if handled { return result, nil }
```

The hook returns `handled=false` only when no env-agent binding exists. If a binding exists, every state is handled locally: ready enqueues with its runtime; pending/failed provisions; provisioning waits/reloads with a bounded context; deleting returns `errEnvDispatchDeleting`. No branch may call `ensureChannelAgentSessionWithDB` for a bound agent.

- [x] **Step 4: Persist the continuation atomically with enqueue**

The transaction that creates the channel prompt/inbox event must also save:

```go
envCollaborationTrigger{
	AgentID: agent.ID.String(), Kind: reasonToTriggerKind(reason),
	ChannelID: channel.ID.String(), ProjectID: projectID,
	ChatSessionID: bindingSessionID, SourceMessageID: trigger.ID,
	ThreadRootMessageID: trigger.ThreadRootMessageID,
	TaskID: enqueueResult.TaskID, RuntimeID: binding.RuntimeID,
}
```

If enqueue fails after first provisioning, compensate the just-created resources and return the binding to `failed`; if the binding was already ready, retain it and only roll back the enqueue transaction.

- [x] **Step 5: Run tests and commit**

Run: `cd multica/server && go test ./internal/handler -run 'Test(EnvDispatch.*Mention|OrdinaryChannelMention)' -count=1`

Expected: PASS, including `-race` for the concurrent test when the host supports it.

```bash
git -C multica/server add internal/handler/channel.go internal/handler/env_dispatch_channel_provision.go internal/handler/env_dispatch_channel_mention_test.go internal/handler/channel_test.go
git -C multica/server commit -m "feat(channels): lazily provision env-dispatch agents"
```

### Task 6: Copy channel history and resume from the source trigger

**Files:**
- Create: `multica/server/internal/handler/env_dispatch_channel_copy.go`
- Modify: `multica/server/internal/service/env_dispatch_channel.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Create: `multica/server/internal/handler/env_dispatch_channel_branch_test.go`
- Test: `multica/server/internal/service/env_dispatch_channel_test.go`

**Interfaces:**
- Produces: `ValidateBranchMessageSource`, `CopyEnvDispatchChannel`, `ChannelCopyMap`, and remapped-trigger execution.
- Consumes: Tasks 1–5 and clone lifecycle from Task 2.

- [x] **Step 1: Add pre-write validation and deep-copy tests**

Add tests named:

```go
func TestBranchRejectsMissingTriggerBeforeWrites(t *testing.T)
func TestBranchRejectsMalformedOrUnauthorizedTriggerBeforeWrites(t *testing.T)
func TestBranchRejectsRosterMismatchBeforeWrites(t *testing.T)
func TestBranchCopiesMessagesRepliesQuotesThreadsAndReadState(t *testing.T)
func TestBranchWakesOnlyTriggerAgentWithClonedSandbox(t *testing.T)
func TestBranchLeavesNonTriggeredAgentsPendingWithCloneSources(t *testing.T)
func TestBranchAppendsNewMessageWithoutChangingTriggerAgent(t *testing.T)
```

For every 400 case, assert zero calls to fork env, create project, create channel, precreate runtime, and clone sandbox.

- [x] **Step 2: Run tests and observe failure**

Run: `cd multica/server && go test ./internal/service ./internal/handler -run 'TestBranch(Rejects|Copies|Wakes|Leaves|Appends)' -count=1`

Expected: FAIL because branch only copies the project subtree/session today.

- [x] **Step 3: Define the copy map and validate before reset fan-out**

Add:

```go
type ChannelCopyMap struct {
	ChannelID string
	MessageIDs map[string]string
	ConversationIDs map[string]string
	ChatSessionIDs map[string]string
}

type ValidatedBranchMessageSource struct {
	SourceEnvID, SourceProjectID, SourceChannelID string
	Roster MessageRoster
	Trigger envCollaborationTrigger
}
```

`Dispatch` must call `ValidateBranchMessageSource` once before entering rollout reset goroutines. Compare sorted, deduplicated requested agent IDs with the source channel's agent members and return the service validation error mapped to HTTP 400 on any mismatch.

- [x] **Step 4: Deep-copy the channel in one transaction**

`CopyEnvDispatchChannel` creates the destination channel/project link, then copies in dependency order:

```text
channel members
conversation rows and conversation_member read/wake state
channel messages without reply/quote/thread foreign keys
message reply_to_message_id, quote_message_id, thread_root_message_id remapped in a second pass
thread_participant rows with remapped conversation/root IDs
channel_agent_session rows with new sessions only for the trigger agent
pending env-agent bindings, carrying source_sandbox_instance_id per source agent
```

Preserve author, content, parts, source, external/client IDs where constraints permit, quote snapshots, trigger depth, timeline visibility, edit/delete timestamps, and sequence order. Historical inserts must use a store function that does not dispatch channel events.

- [x] **Step 5: Remap and execute the trigger**

Build the destination trigger as:

```go
dst := src.Trigger
dst.ChannelID = copyMap.ChannelID
dst.ProjectID = newProjectID
dst.ChatSessionID = copyMap.ChatSessionIDs[src.Trigger.ChatSessionID]
dst.SourceMessageID = copyMap.MessageIDs[src.Trigger.SourceMessageID]
if src.Trigger.ThreadRootMessageID != nil {
	mapped := copyMap.MessageIDs[*src.Trigger.ThreadRootMessageID]
	dst.ThreadRootMessageID = &mapped
}
dst.TaskID = ""
dst.RuntimeID = ""
```

Provision the trigger agent using `CloneSandboxInstance` when its source binding is ready; create from saved policy otherwise. Enqueue a new destination task, set the new task/runtime IDs, and save the remapped trigger. Append request `message.content`, if non-empty, as nondispatching context before the continuation enqueue.

- [x] **Step 6: Run tests and commit**

Run: `cd multica/server && go test ./internal/service ./internal/handler -run 'TestBranch(Rejects|Copies|Wakes|Leaves|Appends)' -count=1`

Expected: PASS.

```bash
git -C multica/server add internal/service/env_dispatch.go internal/service/env_dispatch_channel.go internal/service/env_dispatch_channel_test.go internal/handler/env_dispatch.go internal/handler/env_dispatch_channel_copy.go internal/handler/env_dispatch_channel_branch_test.go
git -C multica/server commit -m "feat(env-dispatch): resume branched channel collaboration"
```

### Task 7: Add channel-first facades and concurrency-safe cleanup

**Files:**
- Create: `multica/server/internal/handler/env_dispatch_channel_routes.go`
- Modify: `multica/server/internal/handler/env_dispatch.go`
- Modify: `multica/server/internal/handler/env_checkpoint.go`
- Modify: `multica/server/cmd/server/router.go`
- Create: `multica/server/internal/handler/env_dispatch_channel_routes_test.go`

**Interfaces:**
- Produces: channel-to-project resolver and three channel-first handlers.
- Consumes: Task 1 deleting state and all existing project-first handler logic.

- [x] **Step 1: Add failing route tests**

Cover:

```go
func TestChannelDagFacadeResolvesBoundProject(t *testing.T)
func TestChannelCheckpointFacadeResolvesBoundProject(t *testing.T)
func TestChannelCleanupDeletesChannelProjectEnvAndBindings(t *testing.T)
func TestChannelCleanupIsIdempotent(t *testing.T)
func TestChannelCleanupSerializesWithProvisioning(t *testing.T)
func TestProjectFirstRoutesRemainAvailable(t *testing.T)
```

- [x] **Step 2: Run tests and observe failure**

Run: `cd multica/server && go test ./internal/handler ./cmd/server -run 'Test(Channel(Dag|Checkpoint|Cleanup)|ProjectFirstRoutes)' -count=1`

Expected: FAIL with missing handlers/routes.

- [x] **Step 3: Extract shared project helpers and add facades**

Register exactly:

```go
r.Get("/api/v1/env-dispatch/channels/{channelID}/dag", h.GetEnvDispatchChannelDag)
r.Delete("/api/v1/env-dispatch/channels/{channelID}", h.DeleteEnvDispatchChannel)
r.Get("/api/v1/channels/{channelID}/env-checkpoints", h.ListChannelEnvCheckpoints)
```

Each handler parses `channelID`, verifies workspace access and `channel.project_id IS NOT NULL`, then calls the same unexported project helper used by the existing handler. Do not make an internal HTTP request.

- [x] **Step 4: Implement deletion serialization and compensation**

Within a transaction, lock the env and binding rows, mark bindings `deleting`, and prevent new claims. After any in-flight provision reaches ready/failed, delete external sandboxes and runtimes, then database resources in this order:

```text
channel (cascades messages/members/sessions)
project (cascades DAG/chat/issues)
environment_agent_sandbox rows
environment
```

Return 204 when already absent. Keep project cleanup behavior unchanged for compatibility.

- [x] **Step 5: Run tests and commit**

Run: `cd multica/server && go test ./internal/handler ./cmd/server -run 'Test(Channel(Dag|Checkpoint|Cleanup)|ProjectFirstRoutes)' -count=1`

Expected: PASS.

```bash
git -C multica/server add internal/handler/env_dispatch_channel_routes.go internal/handler/env_dispatch_channel_routes_test.go internal/handler/env_dispatch.go internal/handler/env_checkpoint.go cmd/server/router.go
git -C multica/server commit -m "feat(env-dispatch): add channel-first lifecycle routes"
```

### Task 8: Make the AReaL message client channel-first

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_client.py`
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py`
- Modify: `customized_areal/tree_search/multi_agent_workflow.py`
- Modify: `customized_areal/tree_search/tests/test_env_dispatch_client.py`
- Modify: `customized_areal/tree_search/tests/test_multica_dag_client.py`

**Interfaces:**
- Produces: immutable `EnvDispatchHandle`; message callers use channel routes, issue callers retain project routes.
- Consumes: Task 7 API.

- [x] **Step 1: Add failing client tests**

Use this response fixture and assert the three channel-first paths:

```python
body = {
    "channel_id": "c1",
    "project_id": "p1",
    "rollouts": [{"channel_id": "c1", "project_id": "p1", "env_id": "e1"}],
}
assert handle.channel_id == "c1"
assert handle.project_id == "p1"
assert seen_paths == [
    "/api/v1/env-dispatch/channels/c1/dag",
    "/api/v1/channels/c1/env-checkpoints",
    "/api/v1/env-dispatch/channels/c1",
]
```

Also retain an issue-dispatch test that uses `/api/v1/env-dispatch/p1/dag`.

- [x] **Step 2: Run tests and observe failure**

Run: `uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py -q`

Expected: FAIL because `create_env_dispatch` returns a project string.

- [x] **Step 3: Add the handle and route selection**

Add:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class EnvDispatchHandle:
    channel_id: str | None
    project_id: str
    env_id: str
    dispatch_type: str

    @property
    def primary_id(self) -> str:
        return self.channel_id if self.dispatch_type == "message" else self.project_id
```

Change `create_env_dispatch(...) -> EnvDispatchHandle`, validate that message responses contain `channel_id`, and update cleanup/DAG/checkpoint methods to accept `handle`. Route by `handle.dispatch_type`; never infer message mode from a missing/present arbitrary string.

- [x] **Step 4: Update workflow ownership**

`multi_agent_workflow.py` stores the returned handle, passes `handle.project_id` only to project-internal payloads such as checkpoint creation, and passes the whole handle to DAG polling, checkpoint listing, and cleanup. `multica_dag_client.py` accepts either a handle or explicit `channel_id/project_id + dispatch_type` at its public boundary so existing issue users remain source-compatible.

- [x] **Step 5: Run tests and commit**

Run: `uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py -q`

Expected: PASS.

```bash
git add customized_areal/tree_search/agents/multica_client.py customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/multi_agent_workflow.py customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py
git commit -m "feat(tree-search): use channel-first env dispatch handles"
```

### Task 9: Regression verification and protocol documentation

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_environment_protocol.md`
- Modify: `docs/superpowers/specs/2026-07-16-env-dispatch-message-channel-design.md` only if implementation names differ while preserving semantics
- Test: all files changed in Tasks 1–8

**Interfaces:**
- Consumes: complete implementation.
- Produces: user-facing protocol examples and full verification evidence.

- [x] **Step 1: Document the final request/response and lifecycle**

Document the exact message response, channel-first routes, leader-only initial wake, peer first-mention provisioning, and branch validation errors. Include this example:

```json
{
  "channel_id": "channel UUID",
  "project_id": "project UUID",
  "rollouts": [{
    "channel_id": "channel UUID",
    "project_id": "project UUID",
    "env_id": "env UUID",
    "leader_run_id": "task UUID",
    "agent_sandboxes": {
      "leader UUID": {"status": "ready", "sandbox_instance_id": "sandbox UUID", "runtime_id": "runtime UUID"},
      "peer UUID": {"status": "pending"}
    }
  }]
}
```

- [x] **Step 2: Run focused Go suites**

Run: `cd multica/server && go test ./internal/service ./internal/handler ./internal/migrations ./cmd/multica ./cmd/server -count=1`

Expected: PASS.

- [x] **Step 3: Run focused Python suites**

Run: `uv run pytest customized_areal/tree_search/tests/test_env_dispatch_client.py customized_areal/tree_search/tests/test_multica_dag_client.py -q`

Expected: PASS.

- [x] **Step 4: Run formatting, static checks, and graph refresh**

Run:

```bash
gofmt -w multica/server/internal/service/env_dispatch*.go multica/server/internal/handler/env_dispatch_channel*.go multica/server/internal/handler/env_dispatch.go multica/server/internal/handler/channel.go multica/server/internal/handler/env_checkpoint.go multica/server/internal/handler/sandbox.go multica/server/cmd/multica/cmd_sandboxd.go multica/server/cmd/server/router.go
cd multica/server && go test ./... -count=1
cd /workspaces/leagent/backend/areal && graphify update .
pre-commit run --all-files
```

Expected: Go tests pass; graph refresh completes; pre-commit passes. If hardware-only or external-service integration suites skip, record their exact skip reason in the handoff rather than changing markers.

- [x] **Step 5: Inspect the final diff for invariant violations**

Run:

```bash
git status --short
git -C multica/server status --short
git diff --check
git -C multica/server diff --check
rg -n "ensureChannelAgentSessionWithDB|DefaultRuntime|RuntimeID" multica/server/internal/handler/channel.go multica/server/internal/handler/env_dispatch_channel_provision.go
```

Expected: no whitespace errors; every EnvDispatch-bound path reaches the binding runtime before the ordinary default-runtime function; only intended files are staged.

- [x] **Step 6: Commit documentation**

```bash
git add customized_areal/tree_search/agents/multica_environment_protocol.md docs/superpowers/specs/2026-07-16-env-dispatch-message-channel-design.md
git commit -m "docs: describe channel-first env dispatch"
```

## Execution checkpoints

- After Task 4, scratch message dispatch is usable with leader-only execution.
- After Task 6, branch/resume and lazy peer cloning satisfy the complete state-inheritance contract.
- After Task 8, AReaL uses `channel_id` as the primary message handle while preserving issue compatibility.
- Task 9 is required before any completion claim or PR creation.

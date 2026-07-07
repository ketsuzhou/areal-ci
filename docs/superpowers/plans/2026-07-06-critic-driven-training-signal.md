---
change: sub-project-e-critic-reward-entropy-env
design-doc: docs/superpowers/specs/2026-07-06-critic-driven-training-signal-design.md
base-ref: 48d49aba673f34dfbe0059387e1a6f609cc7bf8c
archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

# Sub-project E — critic-driven reward + entropy + env_id (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace D's placeholder reward with a real signal from a critic agent, and enrich the session with env_id + entropy so AReaL can actually train on captured trajectories.

**Architecture:** multica auto-spawns a critic task on trained-task-terminal (replacing D's immediate close); the critic's JSON output is parsed for a scalar reward; the deferred close hook fires on critic-terminal with that reward. AReaL's proxy gains an `env_id` field on `StartSessionRequest` and injects `logprobs=true` on forwarded calls for entropy.

**Tech Stack:** Go (multica server), Python 3.12+ (AReaL), Postgres (multica migrations), OpenAI-compatible proxy (AReaL experimental).

## Global Constraints

- **Cross-repo**: multica Go (primary, commits to multica `main`); AReaL Python (contract changes, commits to areal `master`). areal repo holds OpenSpec artifacts + design doc + plan; multica repo holds Go code; `.superpowers/sdd/progress.md` (in areal) tracks the cross-repo ledger.
- **Depends on sub-project D**: D's T7-T9 (session-close hook, config, regression) MUST be complete before E's T7/T8 (critic spawn + deferred close) can be implemented. D is currently paused at `build_pause=plan-ready`. T1 (investigation), T2-T6 (contract + env_id + RL client), T9 (AReaL logprobs), T4 (AReaL env_id) can proceed in parallel with D's remaining work.
- **Codegen rule (multica)**: do NOT run `sqlc generate` repo-wide (creates colliding `agent_skill_suggestion.sql.go`/`evolution.sql.go`, breaks build). For any new query: add to `queries/*.sql` AND hand-write the generated Go in `generated/<file>.sql.go` mirroring a sibling.
- **Scope build/test (multica)**: `go build ./...` fails on pre-existing `internal/service/webpush/webpush.go:180` "constant 4096 overflows byte" (go 1.26). Scope to touched packages. 16 pre-existing `internal/handler` ON CONFLICT (42P10) failures are baseline.
- **AReaL Python**: `uv run pytest` from areal root; `pre-commit run --files <touched>` before commit. Many tests require GPU — explain skips when unavailable.
- **Commit each task**: to multica `main` (Go) or areal `master` (Python); record commit hashes in `.superpowers/sdd/progress.md`. Commits local-only unless the user says push.
- **Reward range**: fixed `[0.0, 1.0]` for E. Out-of-range → fallback to `TRAINING_DEFAULT_REWARD`.
- **No `parent_task_id` on critic task**: `parent_task_id` (migration 055) is the retry back-pointer set by `CreateRetryTask` — do NOT overload for critique linkage. Use `context.critic_of` only.

## File Structure

### multica (Go)

- `server/migrations/153_training_dispatch_critic.up.sql` / `.down.sql` — new, add `critic_agent_id UUID NULL` to `training_dispatch`.
- `server/pkg/db/queries/training_dispatch.sql` — modify, extend `CreateTrainingDispatch` + `GetTrainingDispatchByProject` for `critic_agent_id`.
- `server/pkg/db/generated/training_dispatch.sql.go` — modify (hand-written, mirror sibling).
- `internal/handler/env_dispatch.go` + `internal/service/env_dispatch.go` — modify, add `critic_agent_id` to request/input + validation.
- `internal/handler/env_dispatch_test.go` + `internal/service/env_dispatch_test.go` — modify, tests.
- `internal/arealrl/client.go` — modify, add `envID` parameter to `StartSession`.
- `internal/arealrl/client_test.go` — modify, tests.
- `internal/service/task.go` — modify: `maybeOpenTrainingSession` passes `env_id`; new `maybeSpawnCriticTask`; new `maybeCloseTrainingSessionFromCritic`; routing in `CompleteTask`/`FailTask`/`CancelTask`.
- `internal/service/task_test.go` (or new `internal/service/task_critic_test.go`) — new tests.

### AReaL (Python)

- `areal/experimental/openai/proxy/server.py` — modify, `StartSessionRequest` gains `env_id`; `SessionData` persists it.
- `areal/experimental/openai/proxy/proxy_rollout_server.py` — modify, inject `logprobs=True` in `_call_client_create`; graceful fallback.
- `areal/experimental/openai/proxy/server_test.py` (or new) — new tests.

### areal (docs/ledger)

- `.superpowers/sdd/progress.md` — modify, append E task ledger.

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 1: Investigation — confirm seams for critic dispatch + entropy capture

No production code. Produce `docs/superpowers/notes/2026-07-06-E-seams.md` answering:

- [ ] **1a. Critic auto-spawn seam.** Confirm `CompleteTask`/`FailTask`/`CancelTask` in `internal/service/task.go` are the right chokepoints for `maybeSpawnCriticTask`. D's `maybeCloseTrainingSession` attaches there — the spawn hook attaches at the same points, replacing the close when critic is configured. Note the exact line numbers.
- [ ] **1b. Critic task → trained session linkage.** Confirm `context` JSONB (migration 003) is the right home for `critic_of = {trained_task_id, proxy_key, session_id, project_id}`. Confirm no existing field collides. Confirm the close hook can read `context.critic_of` from the critic task without a join.
- [ ] **1c. Critic reward result shape.** Confirm where the critic's output text lives (e.g. `agent_task_queue.output` or similar). Confirm multica can parse the last line as JSON `{"reward": <float>}`. Note the exact column/field name.
- [ ] **1d. env_id availability at session-open.** Confirm `env_id` is on `training_dispatch` (or derivable from `env_dispatch`) at the time D's `maybeOpenTrainingSession` fires. If `training_dispatch` doesn't have `env_id`, confirm it's on `env_dispatch.EnvID` and thread through.
- [ ] **1e. AReaL proxy logprobs path.** Confirm `_call_client_create` in `proxy_rollout_server.py` is the single chokepoint for all chat-completions forwarding. Confirm `InteractionWithTokenLogpReward` (in `areal/experimental/openai/types.py`) can hold the captured logprobs. Confirm the existing `should_compute_prox_logp()` flag is orthogonal (recompute path, not capture path).
- [ ] **1f. StartSessionRequest env_id field.** Confirm `StartSessionRequest` in `server.py` can gain `env_id: str | None = None` additively (old callers still work). Confirm `SessionData` can persist it without breaking existing serialization.

**STOP-and-report** (begin `BLOCKED:`) if: the critic auto-spawn cannot be injected at the same chokepoints as D's close hook (e.g. the terminal transition is in raw SQL bypassing `FailTask`), OR AReaL's proxy cannot transparently inject `logprobs=True` (e.g. `_call_client_create` is not the single chokepoint). Otherwise begin `DONE:` with the mapping. Commit the note.

**Files:**
- Create: `docs/superpowers/notes/2026-07-06-E-seams.md`

- [ ] **Step 1: Write the investigation note** answering 1a-1f with file:line references.
- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/notes/2026-07-06-E-seams.md
git commit -m "docs(E): record Task 1 seams investigation (critic spawn + logprobs + env_id)"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 2: Contract — `critic_agent_id` on env_dispatch (TDD, multica)

**Files:**
- Modify: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`
- Test: `internal/handler/env_dispatch_test.go`, `internal/service/env_dispatch_test.go`

**Interfaces:**
- Consumes: D's `EnvDispatchRequest` + `service.EnvDispatchInput` (existing).
- Produces: `EnvDispatchRequest.CriticAgentID string` (json `critic_agent_id,omitempty`); `service.EnvDispatchInput.CriticAgentID string`. Validation rule: allowed with `squad_id` + `train_agent_id`; `critic_agent_id == train_agent_id` rejected; `critic_agent_id == agent_id` (single-agent critique) rejected; empty ⇒ unchanged behavior.

- [ ] **Step 1: Write failing tests**

In `internal/service/env_dispatch_test.go`:
```go
func TestEnvDispatchInput_Validate_CriticAgentID(t *testing.T) {
    tests := []struct {
        name    string
        in      EnvDispatchInput
        wantErr string
    }{
        {"empty critic ok", EnvDispatchInput{SquadID: sid, TrainAgentID: tid, AgentID: aid}, ""},
        {"critic with squad+train ok", EnvDispatchInput{SquadID: sid, TrainAgentID: tid, CriticAgentID: cid, AgentID: aid}, ""},
        {"critic without train rejected", EnvDispatchInput{SquadID: sid, CriticAgentID: cid}, "critic_agent_id requires train_agent_id"},
        {"critic == train rejected", EnvDispatchInput{SquadID: sid, TrainAgentID: tid, CriticAgentID: tid}, "critic_agent_id must differ from train_agent_id"},
        {"critic == agent rejected", EnvDispatchInput{AgentID: aid, TrainAgentID: tid, CriticAgentID: aid}, "critic_agent_id must differ from agent_id"},
    }
    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            err := tc.in.Validate()
            if tc.wantErr == "" {
                require.NoError(t, err)
            } else {
                require.ErrorContains(t, err, tc.wantErr)
            }
        })
    }
}
```

In `internal/handler/env_dispatch_test.go`:
```go
func TestEnvDispatchHandler_CriticAgentID_ShapeValidation(t *testing.T) {
    // 400 on malformed UUID
    req := httptest.NewRequest("POST", "/api/v1/env-dispatch", strings.NewReader(`{"squad_id":"`+sid+`","train_agent_id":"`+tid+`","critic_agent_id":"not-a-uuid"}`))
    req.Header.Set("Content-Type", "application/json")
    rr := httptest.NewRecorder()
    handler.ServeHTTP(rr, req)
    assert.Equal(t, 400, rr.Code)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/handler/ ./internal/service/ -run 'EnvDispatch|Dispatch' -v`
Expected: FAIL (field doesn't exist / validation rule missing).

- [ ] **Step 3: Implement**

In `internal/service/env_dispatch.go`, add to `EnvDispatchInput`:
```go
type EnvDispatchInput struct {
    // ... existing fields ...
    TrainAgentID  string
    CriticAgentID string // NEW: optional critic for trained agent (sub-project E)
    // ...
}
```

Add to `Validate()`:
```go
if in.CriticAgentID != "" {
    if in.TrainAgentID == "" {
        return fmt.Errorf("validation_failed: critic_agent_id requires train_agent_id")
    }
    if in.CriticAgentID == in.TrainAgentID {
        return fmt.Errorf("validation_failed: critic_agent_id must differ from train_agent_id")
    }
    if in.CriticAgentID == in.AgentID {
        return fmt.Errorf("validation_failed: critic_agent_id must differ from agent_id")
    }
}
```

In `internal/handler/env_dispatch.go`, add to `EnvDispatchRequest`:
```go
type EnvDispatchRequest struct {
    // ... existing fields ...
    TrainAgentID  string `json:"train_agent_id,omitempty"`
    CriticAgentID string `json:"critic_agent_id,omitempty"` // NEW (sub-project E)
    // ...
}
```

Add UUID shape-check in the handler (mirror `train_agent_id` pattern):
```go
if req.CriticAgentID != "" {
    if _, err := util.ParseUUID(req.CriticAgentID); err != nil {
        writeJSONError(w, http.StatusBadRequest, "critic_agent_id must be a valid UUID")
        return
    }
}
```

Thread `CriticAgentID` through the handler→service mapping.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/handler/ ./internal/service/ -run 'EnvDispatch|Dispatch' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/internal/handler/env_dispatch.go server/internal/handler/env_dispatch_test.go server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go
git commit -m "feat(env-dispatch): accept critic_agent_id (critic for trained agent)"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 3: Persist critic intent — migration 153 (TDD, multica)

**Files:**
- Create: `server/migrations/153_training_dispatch_critic.up.sql`, `153_training_dispatch_critic.down.sql`
- Modify: `server/pkg/db/queries/training_dispatch.sql`, `server/pkg/db/generated/training_dispatch.sql.go`
- Test: `internal/service/env_dispatch_test.go` (extend)

**Interfaces:**
- Consumes: D's `training_dispatch` table (migration 152) + `CreateTrainingDispatch` / `GetTrainingDispatchByProject` queries.
- Produces: `training_dispatch.critic_agent_id UUID NULL` column; `CreateTrainingDispatch` accepts `critic_agent_id`; `GetTrainingDispatchByProject` returns it.

- [ ] **Step 1: Write failing test** (in `internal/service/env_dispatch_test.go`)

```go
func TestEnvDispatch_PersistsCriticAgentID(t *testing.T) {
    fake := newFakeEnvDispatchDeps(t)
    fake.trainingDispatchRows = []trainingDispatchRow{} // track inserts

    in := EnvDispatchInput{
        WorkspaceID: wid, SquadID: sid, TrainAgentID: tid, CriticAgentID: cid, AgentID: aid,
        // ... other required fields ...
    }
    _, err := fake.service.Dispatch(context.Background(), in)
    require.NoError(t, err)

    require.Len(t, fake.trainingDispatchRows, 1)
    assert.Equal(t, cid, fake.trainingDispatchRows[0].CriticAgentID)
}

func TestEnvDispatch_NoCritic_PersistsNull(t *testing.T) {
    fake := newFakeEnvDispatchDeps(t)
    in := EnvDispatchInput{WorkspaceID: wid, SquadID: sid, TrainAgentID: tid, AgentID: aid}
    _, err := fake.service.Dispatch(context.Background(), in)
    require.NoError(t, err)
    require.Len(t, fake.trainingDispatchRows, 1)
    assert.Empty(t, fake.trainingDispatchRows[0].CriticAgentID)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service/ -run 'PersistsCritic|PersistsCriticAgentID' -v`
Expected: FAIL (column doesn't exist).

- [ ] **Step 3: Write migration**

`server/migrations/153_training_dispatch_critic.up.sql`:
```sql
-- sub-project E: persist critic_agent_id on training_dispatch so the
-- auto-spawn hook (T7) and deferred close hook (T8) can resolve the critic
-- by project_id. Nullable: empty critic_agent_id ⇒ D's behavior (no critic).
ALTER TABLE training_dispatch
    ADD COLUMN critic_agent_id UUID NULL;
```

`server/migrations/153_training_dispatch_critic.down.sql`:
```sql
ALTER TABLE training_dispatch DROP COLUMN critic_agent_id;
```

- [ ] **Step 4: Extend queries + hand-write generated Go**

In `server/pkg/db/queries/training_dispatch.sql`, extend `CreateTrainingDispatch`:
```sql
-- name: CreateTrainingDispatch :one
INSERT INTO training_dispatch (project_id, workspace_id, train_agent_id, critic_agent_id, default_reward)
VALUES ($1, $2, $3, $4, COALESCE($5, 1.0))
RETURNING *;
```

Extend `GetTrainingDispatchByProject` (it already does `SELECT *` so no SQL change needed, but the generated struct must include the new column).

In `server/pkg/db/generated/training_dispatch.sql.go`, add `CriticAgentID pgtype.UUID` to the struct (mirror `TrainAgentID`). Update `CreateTrainingDispatch` Go signature to accept `CriticAgentID pgtype.UUID`.

- [ ] **Step 5: Wire service to pass critic_agent_id**

In `internal/service/env_dispatch.go`, where `CreateTrainingDispatch` is called (in `resetOne`/`dispatchOne`), pass `in.CriticAgentID`. Update the `EnvDispatchDeps` interface + adapter + fake to include `CriticAgentID` in the row.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd multica/server && go build ./pkg/db/generated/ ./internal/service/ && go test ./internal/service/ -run 'PersistsCritic|EnvDispatch' -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd multica && git add server/migrations/153_training_dispatch_critic.up.sql server/migrations/153_training_dispatch_critic.down.sql server/pkg/db/queries/training_dispatch.sql server/pkg/db/generated/training_dispatch.sql.go server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go
git commit -m "feat(training): persist critic_agent_id on training_dispatch (migration 153)"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 4: AReaL contract — env_id on StartSessionRequest (TDD, areal Python)

**Files:**
- Modify: `areal/experimental/openai/proxy/server.py` (`StartSessionRequest`, `SessionData`)
- Test: `areal/experimental/openai/proxy/server_test.py` (or new)

**Interfaces:**
- Consumes: existing `StartSessionRequest` (`task_id`, `api_key`).
- Produces: `StartSessionRequest.env_id: str | None = None`; `SessionData.env_id: str | None` persisted.

- [ ] **Step 1: Write failing test**

```python
def test_start_session_accepts_env_id():
    req = StartSessionRequest(task_id="t1", env_id="env_abc")
    assert req.env_id == "env_abc"

def test_start_session_env_id_optional():
    req = StartSessionRequest(task_id="t1")
    assert req.env_id is None

def test_session_data_persists_env_id():
    # After start_session with env_id, the SessionData should carry it
    # (verify via the session cache or export)
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_proxy_server.py -k 'env_id' -v` (path may vary)
Expected: FAIL (`env_id` field doesn't exist).

- [ ] **Step 3: Implement**

In `areal/experimental/openai/proxy/server.py`:
```python
class StartSessionRequest(BaseModel):
    """Request to start a new RL session."""
    task_id: str
    api_key: str | None = None
    env_id: str | None = None  # NEW (sub-project E): environment attribution
```

In `SessionData.__init__`:
```python
def __init__(self, session_id: str, prefix_matcher=None, env_id: str | None = None):
    self.session_id = session_id
    self.env_id = env_id  # NEW
    # ... rest unchanged ...
```

In `start_session` handler (in `proxy_rollout_server.py`), pass `env_id` to `SessionData`:
```python
_session_cache[session_id] = SessionData(
    session_id=session_id,
    prefix_matcher=...,
    env_id=request.env_id,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_proxy_server.py -k 'env_id' -v`
Expected: PASS.

- [ ] **Step 5: Commit (in areal repo)**

```bash
git add areal/experimental/openai/proxy/server.py areal/experimental/openai/proxy/proxy_rollout_server.py areal/experimental/openai/proxy/server_test.py
git commit -m "feat(proxy): accept env_id on start_session for trajectory attribution"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 5: RL bridge client — env_id in StartSession (TDD, multica Go)

**Files:**
- Modify: `internal/arealrl/client.go`
- Test: `internal/arealrl/client_test.go`

**Interfaces:**
- Consumes: D's `Client.StartSession(ctx, taskID string) (SessionCreds, error)`.
- Produces: `Client.StartSession(ctx, taskID, envID string) (SessionCreds, error)` — `envID` empty ⇒ omit from body.

- [ ] **Step 1: Write failing tests** (in `internal/arealrl/client_test.go`)

```go
func TestStartSession_IncludesEnvIDWhenNonEmpty(t *testing.T) {
    var gotBody map[string]any
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        b, _ := io.ReadAll(r.Body)
        json.Unmarshal(b, &gotBody)
        json.NewEncoder(w).Encode(map[string]any{"session_id": "s1", "api_key": "k1"})
    }))
    defer srv.Close()
    c := New(srv.URL, "admin-key")
    _, err := c.StartSession(context.Background(), "task-1", "env_abc")
    require.NoError(t, err)
    assert.Equal(t, "env_abc", gotBody["env_id"])
}

func TestStartSession_OmitsEnvIDWhenEmpty(t *testing.T) {
    var gotBody map[string]any
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        b, _ := io.ReadAll(r.Body)
        json.Unmarshal(b, &gotBody)
        json.NewEncoder(w).Encode(map[string]any{"session_id": "s1", "api_key": "k1"})
    }))
    defer srv.Close()
    c := New(srv.URL, "admin-key")
    _, err := c.StartSession(context.Background(), "task-1", "")
    require.NoError(t, err)
    _, hasEnvID := gotBody["env_id"]
    assert.False(t, hasEnvID, "env_id should be omitted when empty")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/arealrl/ -run 'StartSession' -v`
Expected: FAIL (signature mismatch — too few args).

- [ ] **Step 3: Implement**

In `internal/arealrl/client.go`, change `StartSession`:
```go
func (c *Client) StartSession(ctx context.Context, taskID, envID string) (SessionCreds, error) {
    body := map[string]any{
        "task_id":    taskID,
        "group_size": 1,
    }
    if envID != "" {
        body["env_id"] = envID
    }
    resp, err := c.doJSON(ctx, startSessionPath, c.adminKey, body)
    // ... rest unchanged ...
}
```

- [ ] **Step 4: Update callers**

D's `maybeOpenTrainingSession` calls `StartSession(ctx, taskID)`. Update to `StartSession(ctx, taskID, envID)` where `envID` is threaded from `training_dispatch` (Task 6). For now, pass `""` to keep D's behavior unchanged until Task 6 wires the real value.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/arealrl/ -run 'StartSession' -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd multica && git add server/internal/arealrl/client.go server/internal/arealrl/client_test.go server/internal/service/task.go
git commit -m "feat(arealrl): pass env_id to start_session"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 6: Session-open hook — pass env_id (TDD, multica)

**Files:**
- Modify: `internal/service/task.go` (D's `maybeOpenTrainingSession`)
- Test: `internal/service/task_test.go` or `internal/service/task_critic_test.go`

**Interfaces:**
- Consumes: `training_dispatch.env_id` (or `env_dispatch.EnvID`) + `arealrl.Client.StartSession(ctx, taskID, envID)`.
- Produces: `start_session` called with env_id when available.

- [ ] **Step 1: Write failing test**

```go
func TestMaybeOpenTrainingSession_PassesEnvID(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.trainingDispatch = &trainingDispatchRow{
        ProjectID: pid, TrainAgentID: tid, CriticAgentID: "", DefaultReward: 1.0,
        EnvID: "env_abc", // if env_id is on training_dispatch (T1/1d confirms)
    }
    fake.rlClient = &fakeRLClient{}
    // ... create a task for the trained agent ...
    err := maybeOpenTrainingSession(ctx, task, fake.deps)
    require.NoError(t, err)
    require.Len(t, fake.rlClient.startCalls, 1)
    assert.Equal(t, "env_abc", fake.rlClient.startCalls[0].EnvID)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service/ -run 'MaybeOpenTrainingSession|SessionOpen' -v`
Expected: FAIL (env_id not threaded).

- [ ] **Step 3: Implement**

In `internal/service/task.go`, `maybeOpenTrainingSession`: read `envID` from `training_dispatch` (if T1/1d confirms env_id is stored there) or from `env_dispatch.EnvID`. Pass to `arealrl.Client.StartSession(ctx, taskID, envID)`.

If T1/1d finds env_id is NOT on `training_dispatch`, thread it from `env_dispatch` via the dispatch lookup (add `env_id` to the dispatch row read by the open hook).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/service/ -run 'MaybeOpenTrainingSession|SessionOpen' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/internal/service/task_test.go
git commit -m "feat(training): pass env_id when opening RL session"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 7: Critic auto-spawn on trained-terminal (TDD, multica)

**Files:**
- Modify: `internal/service/task.go` (new `maybeSpawnCriticTask`; routing in `CompleteTask`/`FailTask`/`CancelTask`)
- Test: `internal/service/task_critic_test.go` (new) or extend `task_test.go`

**Interfaces:**
- Consumes: `training_dispatch.critic_agent_id`; `trainedTask.context.areal_proxy` (D's injected proxy config); trained task's output.
- Produces: a new critic task with `context.critic_of = {trained_task_id, proxy_key, session_id, project_id}`; trained session NOT closed.

- [ ] **Step 1: Write failing tests**

```go
func TestMaybeSpawnCriticTask_CreatesCriticTask(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.trainingDispatch = &trainingDispatchRow{
        ProjectID: pid, TrainAgentID: tid, CriticAgentID: cid, DefaultReward: 1.0,
    }
    trainedTask := &agentTask{
        ID: tid, AgentID: tid, ProjectID: pid,
        Context: json.RawMessage(`{"areal_proxy":{"session_id":"s1","api_key":"pk1","provider":"areal","model":"areal-default","base_url":"http://stub:9100/v1"}}`),
        Output: "trained agent's final output text",
    }
    err := maybeSpawnCriticTask(ctx, trainedTask, fake.trainingDispatch, fake.deps)
    require.NoError(t, err)
    require.Len(t, fake.createdTasks, 1)
    critic := fake.createdTasks[0]
    assert.Equal(t, cid, critic.AgentID)
    // parent_task_id NOT set
    assert.False(t, critic.ParentTaskID.Valid)
    // context.critic_of populated
    var cof struct {
        TrainedTaskID string `json:"trained_task_id"`
        ProxyKey      string `json:"proxy_key"`
        SessionID     string `json:"session_id"`
        ProjectID     string `json:"project_id"`
    }
    json.Unmarshal(critic.Context, &struct {
        CriticOf any `json:"critic_of"`
    }{CriticOf: &cof})
    assert.Equal(t, tid, cof.TrainedTaskID)
    assert.Equal(t, "pk1", cof.ProxyKey)
    assert.Equal(t, "s1", cof.SessionID)
}

func TestMaybeSpawnCriticTask_NoCritic_NoSpawn(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.trainingDispatch = &trainingDispatchRow{ProjectID: pid, TrainAgentID: tid, CriticAgentID: ""}
    err := maybeSpawnCriticTask(ctx, trainedTask, fake.trainingDispatch, fake.deps)
    require.NoError(t, err)
    assert.Empty(t, fake.createdTasks)
}

func TestMaybeSpawnCriticTask_Idempotent(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.existingCriticForTrained = tid // simulate prior spawn
    err := maybeSpawnCriticTask(ctx, trainedTask, fake.trainingDispatch, fake.deps)
    require.NoError(t, err)
    assert.Empty(t, fake.createdTasks) // no duplicate spawn
}

func TestMaybeSpawnCriticTask_SpawnFails_FallsBackToD(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.createTaskErr = errors.New("db down")
    fake.rlClient = &fakeRLClient{}
    err := maybeSpawnCriticTask(ctx, trainedTask, fake.trainingDispatch, fake.deps)
    require.NoError(t, err) // error swallowed, fallback fired
    require.Len(t, fake.rlClient.setRewardCalls, 1)
    require.Len(t, fake.rlClient.endSessionCalls, 1)
    // default reward used
    assert.Equal(t, 1.0, fake.rlClient.setRewardCalls[0].Reward)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service/ -run 'CriticSpawn' -v`
Expected: FAIL (function doesn't exist).

- [ ] **Step 3: Implement**

In `internal/service/task.go`:
```go
// maybeSpawnCriticTask creates a critic task for the trained agent's output
// when training_dispatch has a critic_agent_id. Replaces D's close hook for
// trained tasks when a critic is configured. Idempotent. On spawn failure,
// falls back to D's close (SetReward(default) → EndSession).
func (s *TaskService) maybeSpawnCriticTask(ctx context.Context, trained *db.AgentTaskQueue, td trainingDispatchRow) error {
    if td.CriticAgentID == "" {
        return nil // no critic → D's close hook fires
    }
    // Idempotency: skip if a critic task already exists for this trained task
    existing, err := s.Queries.FindCriticTaskForTrained(ctx, trained.ID)
    if err != nil && !errors.Is(err, pg.ErrNoRows) {
        return err
    }
    if existing.ID.Valid {
        return nil // already spawned
    }
    // Read trained session creds from context.areal_proxy
    var proxy struct {
        SessionID string `json:"session_id"`
        APIKey    string `json:"api_key"`
    }
    var ctxPayload struct{ ArealProxy json.RawMessage `json:"areal_proxy"` }
    json.Unmarshal(trained.Context, &ctxPayload)
    json.Unmarshal(ctxPayload.ArealProxy, &proxy)
    // Build critic_of context
    criticOf, _ := json.Marshal(map[string]string{
        "trained_task_id": util.UUIDToString(trained.ID),
        "proxy_key":       proxy.APIKey,
        "session_id":      proxy.SessionID,
        "project_id":      util.UUIDToString(trained.ProjectID),
    })
    criticCtx, _ := json.Marshal(map[string]json.RawMessage{"critic_of": criticOf})
    // Create critic task
    criticAgentUUID, _ := util.ParseUUID(td.CriticAgentID)
    _, err = s.Queries.CreateAgentTask(ctx, db.CreateAgentTaskParams{
        AgentID:   criticAgentUUID,
        ProjectID: trained.ProjectID,
        Context:   criticCtx,
        Input:     trained.Output, // literal text
        // ... other required fields ...
    })
    if err != nil {
        // Spawn failed — fall back to D's close
        slog.Warn("critic spawn failed; closing with default reward",
            "trained_task_id", util.UUIDToString(trained.ID), "error", err)
        return s.maybeCloseTrainingSession(ctx, trained) // D's close
    }
    return nil
}
```

Add `FindCriticTaskForTrained` query (hand-write generated Go):
```sql
-- name: FindCriticTaskForTrained :one
SELECT * FROM agent_task_queue
WHERE context->'critic_of'->>'trained_task_id' = $1::text
LIMIT 1;
```

**Routing in `CompleteTask`/`FailTask`/`CancelTask`**: replace D's direct call to `maybeCloseTrainingSession` with:
```go
if td, err := s.lookupTrainingDispatch(ctx, task); err == nil && td.CriticAgentID != "" {
    if err := s.maybeSpawnCriticTask(ctx, &task, td); err != nil {
        slog.Warn("critic spawn routing failed", "error", err)
    }
    // Do NOT call D's close — deferred to critic-terminal
} else {
    // No critic OR lookup failed → D's close fires
    s.maybeCloseTrainingSession(ctx, task)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/service/ -run 'CriticSpawn|TrainingClose' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/internal/service/task_critic_test.go server/pkg/db/queries/task.sql server/pkg/db/generated/task.sql.go
git commit -m "feat(training): auto-spawn critic task on trained-task terminal"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 8: Deferred close hook on critic-terminal (TDD, multica)

**Files:**
- Modify: `internal/service/task.go` (new `maybeCloseTrainingSessionFromCritic`; routing)
- Test: `internal/service/task_critic_test.go`

**Interfaces:**
- Consumes: `criticTask.context.critic_of` (`proxy_key`, `session_id`); critic's output text (JSON `{"reward": <float>}`).
- Produces: `SetReward(proxy_key, parsedReward)` then `EndSession(proxy_key)` on the trained session.

- [ ] **Step 1: Write failing tests**

```go
func TestMaybeCloseFromCritic_ParsesRewardAndCloses(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.rlClient = &fakeRLClient{}
    criticTask := &agentTask{
        ID: cid, AgentID: cid,
        Context: json.RawMessage(`{"critic_of":{"trained_task_id":"`+tid+`","proxy_key":"pk1","session_id":"s1","project_id":"`+pid+`"}}`),
        Output: `Some critique text...\n{"reward": 0.85}`,
    }
    err := maybeCloseTrainingSessionFromCritic(ctx, criticTask, fake.deps)
    require.NoError(t, err)
    require.Len(t, fake.rlClient.setRewardCalls, 1)
    assert.Equal(t, "pk1", fake.rlClient.setRewardCalls[0].ProxyKey)
    assert.InDelta(t, 0.85, fake.rlClient.setRewardCalls[0].Reward, 0.001)
    require.Len(t, fake.rlClient.endSessionCalls, 1)
    assert.Equal(t, "pk1", fake.rlClient.endSessionCalls[0].ProxyKey)
    // Order: setReward BEFORE endSession
    assert.True(t, fake.rlClient.setRewardCalls[0].Time.Before(fake.rlClient.endSessionCalls[0].Time))
}

func TestMaybeCloseFromCritic_Unparseable_FallbackDefault(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.rlClient = &fakeRLClient{}
    criticTask := &agentTask{
        Context: json.RawMessage(`{"critic_of":{"proxy_key":"pk1","session_id":"s1","project_id":"`+pid+`"}}`),
        Output: "I couldn't decide on a score",
    }
    err := maybeCloseTrainingSessionFromCritic(ctx, criticTask, fake.deps)
    require.NoError(t, err)
    require.Len(t, fake.rlClient.setRewardCalls, 1)
    assert.InDelta(t, 1.0, fake.rlClient.setRewardCalls[0].Reward, 0.001) // default
}

func TestMaybeCloseFromCritic_OutOfRange_FallbackDefault(t *testing.T) {
    // reward 1.5 → fallback to default 1.0
    ...
}

func TestMaybeCloseFromCritic_NonCriticTask_NoOp(t *testing.T) {
    fake := newFakeTrainingDeps(t)
    fake.rlClient = &fakeRLClient{}
    regularTask := &agentTask{Context: json.RawMessage(`{}`)} // no critic_of
    err := maybeCloseTrainingSessionFromCritic(ctx, regularTask, fake.deps)
    require.NoError(t, err)
    assert.Empty(t, fake.rlClient.setRewardCalls)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd multica/server && go test ./internal/service/ -run 'CriticClose|CloseFromCritic' -v`
Expected: FAIL (function doesn't exist).

- [ ] **Step 3: Implement**

In `internal/service/task.go`:
```go
// maybeCloseTrainingSessionFromCritic closes the trained session when a
// critic task reaches terminal. Reads proxy_key from critic.context.critic_of
// and reward from parsing critic.Output as JSON {"reward": <float>}.
// Fallback to TRAINING_DEFAULT_REWARD on parse failure / out-of-range.
// No-op if the task has no context.critic_of (not a critic task).
func (s *TaskService) maybeCloseTrainingSessionFromCritic(ctx context.Context, critic db.AgentTaskQueue) error {
    var cof struct {
        TrainedTaskID string `json:"trained_task_id"`
        ProxyKey      string `json:"proxy_key"`
        SessionID     string `json:"session_id"`
        ProjectID     string `json:"project_id"`
    }
    var payload struct{ CriticOf json.RawMessage `json:"critic_of"` }
    if err := json.Unmarshal(critic.Context, &payload); err != nil || len(payload.CriticOf) == 0 {
        return nil // not a critic task
    }
    if err := json.Unmarshal(payload.CriticOf, &cof); err != nil {
        return nil // malformed context — skip
    }
    reward := s.parseCriticReward(critic.Output) // returns default_reward on failure
    if err := s.rlClient.SetReward(ctx, cof.ProxyKey, reward); err != nil {
        slog.Warn("critic close: SetReward failed", "error", err)
    }
    if err := s.rlClient.EndSession(ctx, cof.ProxyKey); err != nil {
        slog.Warn("critic close: EndSession failed", "error", err)
    }
    return nil
}

// parseCriticReward extracts {"reward": <float>} from the last line of output.
// Returns defaultReward on any failure. Range-checked to [0.0, 1.0].
func (s *TaskService) parseCriticReward(output string) float64 {
    defaultReward := s.config.TrainingDefaultReward // from TRAINING_DEFAULT_REWARD
    lines := strings.Split(strings.TrimSpace(output), "\n")
    if len(lines) == 0 {
        return defaultReward
    }
    var parsed struct{ Reward float64 `json:"reward"` }
    if err := json.Unmarshal([]byte(lines[len(lines)-1]), &parsed); err != nil {
        return defaultReward
    }
    if parsed.Reward < 0.0 || parsed.Reward > 1.0 {
        return defaultReward
    }
    return parsed.Reward
}
```

**Routing**: in `CompleteTask`/`FailTask`/`CancelTask`, before D's `maybeCloseTrainingSession`, call:
```go
// If this is a critic task, close the trained session from it.
if closed, err := s.maybeCloseTrainingSessionFromCritic(ctx, task); err != nil {
    slog.Warn("critic close failed", "error", err)
} else if closed {
    return nil // closed via critic; skip D's close
}
// Otherwise, D's close hook fires (trained task, no critic, etc.)
```

(Adjust based on whether `maybeCloseTrainingSessionFromCritic` returns a bool or just an error — the test asserts no-op on non-critic tasks, so a bool "did close" is cleaner.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd multica/server && go test ./internal/service/ -run 'CriticClose|CloseFromCritic|TrainingClose' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica && git add server/internal/service/task.go server/internal/service/task_critic_test.go
git commit -m "feat(training): deferred close hook on critic-terminal with critic reward"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 9: AReaL proxy — logprobs capture for entropy (TDD, areal Python)

**Files:**
- Modify: `areal/experimental/openai/proxy/proxy_rollout_server.py` (`_call_client_create`)
- Test: `areal/experimental/openai/proxy/proxy_rollout_server_test.py` (or new)

**Interfaces:**
- Consumes: existing `_call_client_create` forwarding path.
- Produces: forwarded requests include `logprobs=True`; response logprobs stored in `InteractionWithTokenLogpReward`; graceful fallback if upstream rejects.

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_call_client_create_injects_logprobs(monkeypatch):
    captured_kwargs = {}
    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        # return a minimal chat completion with logprobs
        ...
    # wire fake_create as _openai_client.chat.completions.create
    await _call_client_create(create_fn=fake_create, request={...}, session_id="s1")
    assert captured_kwargs.get("logprobs") is True

@pytest.mark.asyncio
async def test_call_client_create_fallback_on_logprobs_error(monkeypatch):
    call_count = 0
    captured_kwargs = []
    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        captured_kwargs.append(kwargs)
        if call_count == 1 and kwargs.get("logprobs") is True:
            raise OpenAIError("logprobs not supported")
        return minimal_completion()
    await _call_client_create(create_fn=fake_create, request={...}, session_id="s1")
    assert call_count == 2  # retried without logprobs
    assert "logprobs" not in captured_kwargs[1] or captured_kwargs[1]["logprobs"] is not True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_proxy_rollout_server.py -k 'logprobs' -v`
Expected: FAIL (logprobs not injected).

- [ ] **Step 3: Implement**

In `areal/experimental/openai/proxy/proxy_rollout_server.py`, modify `_call_client_create`:
```python
async def _call_client_create(
    create_fn,
    request: dict[str, Any] | BaseModel,
    session_id: str,
    extra_ignored_args: list[str] | None = None,
    stream: bool = False,
) -> ChatCompletion | Response | AsyncGenerator[ChatCompletionChunk, None]:
    # ... existing setup ...
    
    # sub-project E: inject logprobs=True for entropy capture.
    # Graceful fallback if upstream rejects.
    forwarded_kwargs = build_forwarded_kwargs(request, extra_ignored_args)
    forwarded_kwargs["logprobs"] = True
    try:
        return await create_fn(**forwarded_kwargs)
    except Exception as e:
        if _is_logprobs_unsupported(e):
            logger.warning("upstream rejected logprobs=True; retrying without (session %s)", session_id)
            forwarded_kwargs.pop("logprobs", None)
            return await create_fn(**forwarded_kwargs)
        raise
```

(`_is_logprobs_unsupported` detects the upstream error — e.g. checks for "logprobs" in the error message or a specific status code. Implementation detail — T1/1e confirms the exact error shape.)

For storing logprobs in the response: the existing `InteractionCache` / `InteractionWithTokenLogpReward` path should pick them up from the response. T1/1e confirms where to wire this.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_proxy_rollout_server.py -k 'logprobs' -v`
Expected: PASS.

- [ ] **Step 5: Commit (in areal repo)**

```bash
git add areal/experimental/openai/proxy/proxy_rollout_server.py areal/experimental/openai/proxy/proxy_rollout_server_test.py
git commit -m "feat(proxy): capture logprobs for entropy computation"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 10: Config + production wiring (TDD-light, multica)

**Files:**
- Modify: `internal/daemon/config.go` (or server config) — confirm `TRAINING_DEFAULT_REWARD` from D is wired.
- Modify: handler/service construction — wire the new critic spawn + deferred close into the task service.
- Modify: `.env.example` — add `CRITIC_AGENT_DEFAULT_*` if needed (likely no new config).
- Modify: `multica/db_bridge/README.md` or protocol doc — note env_id in /rl/start_session body.

- [ ] **Step 1: Confirm D's `TRAINING_DEFAULT_REWARD` is the fallback** in `parseCriticReward` (Task 8). If D's config plumbing doesn't expose it to the task service, add it now.
- [ ] **Step 2: Verify the task service construction** injects the RL client + training dispatch deps into `maybeSpawnCriticTask` and `maybeCloseTrainingSessionFromCritic`. No new config needed for critic (critic_agent_id comes from env_dispatch).
- [ ] **Step 3: Update `.env.example`** if any new config (likely none — confirm and skip if so).
- [ ] **Step 4: Note env_id in db_bridge protocol doc** (one-line addition: "`/rl/start_session` body may include `env_id` (string, optional, sub-project E)").
- [ ] **Step 5: Build touched packages**

Run: `cd multica/server && go build ./internal/handler/ ./internal/service/ ./internal/arealrl/ ./pkg/db/generated/`
Expected: clean build.

- [ ] **Step 6: Commit**

```bash
cd multica && git add server/internal/daemon/config.go server/internal/service/task.go .env.example multica/db_bridge/README.md
git commit -m "chore(training): wire critic spawn + deferred close; note env_id in db_bridge"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

### Task 11: Full regression + cross-repo verification + grep sweep

- [ ] **Step 1: Scoped multica Go build + vet + test**

Run:
```bash
cd multica/server && \
  go build ./internal/handler/ ./internal/service/ ./internal/arealrl/ ./pkg/db/generated/ && \
  go vet ./internal/handler/ ./internal/service/ ./internal/arealrl/ ./pkg/db/generated/ && \
  go test ./internal/handler/ ./internal/service/ ./internal/arealrl/ ./pkg/db/generated/
```
Expected: only pre-existing failures (16 ON CONFLICT handler fails; 0 new).

- [ ] **Step 2: `gofmt -l` clean on touched files**

Run: `cd multica/server && gofmt -l internal/service/task.go internal/handler/env_dispatch.go internal/arealrl/client.go pkg/db/generated/training_dispatch.sql.go`
Expected: empty output.

- [ ] **Step 3: AReaL Python tests + pre-commit**

Run:
```bash
uv run pytest tests/test_proxy_server.py tests/test_proxy_rollout_server.py -v
pre-commit run --files areal/experimental/openai/proxy/server.py areal/experimental/openai/proxy/proxy_rollout_server.py
```
Expected: PASS.

- [ ] **Step 4: db_bridge smoke** (if touched)

Run: `cd multica/db_bridge && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Cross-repo E2E** (if feasible)

Manual or scripted: dispatch a team with `train_agent_id` + `critic_agent_id` + `env_id`; let the trained task complete; verify the critic task spawns and completes; verify AReaL's `export_trajectories` shows the critic's reward + env_id + entropy.

- [ ] **Step 6: grep sweep**

Run:
```bash
grep -rn 'critic_agent_id\|critic_of\|maybeSpawnCriticTask\|maybeCloseTrainingSessionFromCritic\|parseCriticReward' multica/server/internal/ multica/server/pkg/db/
grep -rn 'env_id' areal/experimental/openai/proxy/
grep -rn 'logprobs=True\|logprobs=True' areal/experimental/openai/proxy/
```
Expected: all hits resolve to intended code only (no stray references).

- [ ] **Step 7: Final whole-branch review**

Review the full diff (areal + multica). Check: routing table from design doc §4.4 is correctly implemented; no `parent_task_id` set on critic tasks; reward range `[0.0, 1.0]` enforced; fallback paths all log + continue.

Mark: READY TO MERGE / NEEDS_CHANGES.

- [ ] **Step 8: Update ledger**

Append to `.superpowers/sdd/progress.md`:
```
Sub-project E (critic-driven-training-signal): T1-T11 complete
  areal master: <commit-range>
  multica main: <commit-range>
  Review: <READY TO MERGE / NEEDS_CHANGES>
```

Commit:
```bash
git add .superpowers/sdd/progress.md
git commit -m "chore(sdd): record sub-project E completion in ledger"
```

archived-with: 2026-07-07-sub-project-e-critic-reward-entropy-env
---

## Self-review

**Spec coverage**: all 6 ADDED Requirements in `specs/critic-driven-training-signal/spec.md` are covered:
- "Critic agent squad role" → T2 (contract), T3 (persist), T7 (auto-spawn)
- "Deferred session-close on critic-completion" → T8
- "env_id emission at session-open" → T4 (AReaL), T5 (RL client), T6 (open hook)
- "Entropy recording from proxied LLM traffic" → T9
- "Critic agent configuration" → T2
- "Persisted critic intent" → T3

**Placeholder scan**: no TBDs in task bodies. T1 is intentionally an investigation (produces a note, not code) — its steps are concrete (write note answering 1a-1f, commit). T10 has "if needed" / "likely none" for config — this is accurate (T1 confirms whether new config is needed) and the step says "confirm and skip if so", which is concrete.

**Type consistency**: `StartSession(ctx, taskID, envID string)` signature is consistent across T5 (definition) and T6 (caller). `critic_of` JSON shape is consistent across T7 (write) and T8 (read). `CriticAgentID string` on `EnvDispatchInput` is consistent across T2 (definition), T3 (persist), T7 (read). `trainingDispatchRow` shape is consistent across T3, T7, T8.

**Cross-repo ordering**: T4 (AReaL env_id) and T9 (AReaL logprobs) are independent of multica tasks and can be done in parallel. T5 depends on T4 (client sends env_id, AReaL must accept it). T7/T8 depend on D's T7-T9 (session-close hook) being complete. T1 should run first to de-risk.

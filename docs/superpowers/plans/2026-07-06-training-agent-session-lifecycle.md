# Sub-project D — training_agent session lifecycle (implementation plan)

Spec: `docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md`
Repo: multica `main` (primary) — commit directly to `main`. AReaL `master` confirm-only.
Execution: subagent-driven (implementer → reviewer per task).

## Test runners / constraints
- multica Go: `DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable`.
  **Scope build/test to touched packages** — `go build ./...` fails on the
  pre-existing `internal/service/webpush/webpush.go:180` "constant 4096 overflows
  byte" (go 1.26). Pre-existing: 16 `internal/handler` ON CONFLICT (42P10)
  daemon/claim failures; `cmd/server` does not compile here (imports webpush) —
  validate router/handler edits by `go build ./internal/handler/` + inspection.
- **Codegen rule:** do NOT run `sqlc generate` repo-wide (creates colliding
  `agent_skill_suggestion.sql.go`/`evolution.sql.go`, breaks build). For any new
  query: add to `queries/*.sql` (source of truth) AND hand-write the generated Go
  in `generated/<file>.sql.go` mirroring a sibling.
- db_bridge (if touched): `cd multica/db_bridge && uv run pytest -q`.
- Commit each task to multica `main`; record commit hash in the ledger.

---

### Task 1: Read-and-document the seams (investigation — STOP-if-broken)

No production code. Produce a short markdown note
(`docs/superpowers/notes/2026-07-06-D-seams.md`) answering spec §6:

- [ ] **1a. Trained-member task-creation chokepoint.** Trace how a squad
  teammate's `agent_task_queue` row is created when the leader delegates:
  (i) @mention delegation (issue/channel comment → task) and (ii)
  `/api/agent/start` with `parent_task_id`. Identify the **single server-side
  function** (or the minimal set) where a new task's `(agent_id, project_id,
  context)` is known before it is claimed. Files to read: `internal/handler/agent.go`
  (start), `internal/handler/channel.go` (`dispatchChannel*`), mention→task
  path, `internal/service/task.go`, `CreateAgentTask`/`CreateChatTask`.
- [ ] **1b. Completion path.** Find where a task transitions to
  `completed`/`failed`/`cancelled` (daemon completion report handler +
  `internal/service/task.go`), i.e. where the close hook attaches. Note
  failed/timeout/cancel transitions too.
- [ ] **1c. execenv provider override.** Determine whether `internal/daemon/execenv`
  already supports a per-task `provider`/`base_url`/`api_key` override sourced
  from `task.context` (grep `provider`, `base_url`, `InjectRuntimeConfig`,
  `writeContextFiles`). Record: minimal (existing field) vs needs new field.
- [ ] **1d. `start_session` task_id semantics.** Confirm what AReaL's
  `StartSessionRequest.task_id` is used for and decide what multica passes
  (multica `task.id`? agent_run_id?). Read `areal/v2/inference_service/data_proxy/session.py`.
- [ ] **1e. project_id availability.** Confirm the trained teammate task can be
  joined to its rollout `project_id` (via issue→project or chat_session→project)
  so the `training_dispatch` lookup in the open-hook works.

**STOP-and-report** (begin report `BLOCKED:`) if: there is no single/あ small set
of server-side chokepoints for teammate task creation (e.g. teammates are created
only inside the daemon/runtime, invisible to the server) — that would break
Approach A's server-side open-hook and require re-brainstorming. Otherwise begin
`DONE:` with the mapping. Commit the note.

---

### Task 2: Contract — `train_agent_id` on env_dispatch (TDD)

**Files:** `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/handler/env_dispatch_test.go`, `internal/service/env_dispatch_test.go`.

- [ ] Write failing tests: request with `train_agent_id` shape-validated (400 on
  malformed UUID); service accepts it; validation — allowed with `squad_id`, or
  equal to `agent_id`; empty ⇒ unchanged behavior.
- [ ] Add `TrainAgentID string` to `EnvDispatchRequest` (json `train_agent_id,omitempty`)
  and `service.EnvDispatchInput`; thread through the handler→service mapping.
- [ ] Handler UUID shape-check when present; service `validate()` rule.
- [ ] Run: `go test ./internal/handler/ ./internal/service/ -run 'EnvDispatch|Dispatch'`.
- [ ] Commit: `feat(env-dispatch): accept train_agent_id (training target)`.

---

### Task 3: Persist training intent — `training_dispatch` (TDD)

**Files:** `server/migrations/152_training_dispatch.up.sql`/`.down.sql`,
`server/pkg/db/queries/training_dispatch.sql`,
`server/pkg/db/generated/training_dispatch.sql.go` (hand-written),
`internal/service/env_dispatch.go` (+ deps method + adapter + fake), tests.

- [ ] Migration: `training_dispatch(project_id UUID PK/UNIQUE, workspace_id UUID,
  train_agent_id UUID, default_reward double precision NOT NULL DEFAULT 1.0,
  created_at timestamptz default now())`.
- [ ] Queries (source + hand-written generated, mirror a sibling): `CreateTrainingDispatch`,
  `GetTrainingDispatchByProject`.
- [ ] Service: when `train_agent_id != ""`, after a rollout's project is created
  in `resetOne`/`dispatchOne`, persist a `training_dispatch` row per rollout
  project (add `EnvDispatchDeps.SaveTrainingDispatch`; wire adapter + fake +
  stub). Failing tests first (fake asserts one row per rollout when train set;
  none when empty).
- [ ] Verify generated code compiles: `go build ./pkg/db/generated/ ./internal/service/`.
- [ ] Commit: `feat(training): persist training_dispatch per rollout project (migration 152)`.

---

### Task 4: RL bridge client (Go) (TDD)

**Files:** new `internal/arealrl/client.go` (+ `client_test.go`).

- [ ] Define `Client` with `StartSession(ctx, taskID string, groupSize int) (SessionCreds, error)`
  (`SessionCreds{SessionID, ProxyKey}`), `SetReward(ctx, sessionID string, reward float64) error`,
  `EndSession(ctx, sessionID string) error`. POST JSON to `<stubBaseURL>/rl/start_session`,
  `/rl/set_reward`, `/rl/end_session` with admin key header. Decode
  `StartSessionResponse{group_id, sessions:[{session_id, session_api_key}]}` →
  first session's `session_api_key` = ProxyKey.
- [ ] Tests with `httptest.Server`: asserts paths/method/body; maps
  session_api_key→ProxyKey; error on non-2xx; empty-sessions error.
- [ ] Run: `go test ./internal/arealrl/`.
- [ ] Commit: `feat(arealrl): Go client for /rl start/set_reward/end_session bridge channels`.

---

### Task 5: Session-open hook at trained-member task creation (TDD)

**Files:** the chokepoint(s) from Task 1a (`internal/service/task.go` and/or
`internal/handler/agent.go`), a new `internal/service` seam, queries to store
`session_id` on the task + set `context`, tests.

- [ ] Failing tests (fake RL client + fake queries): when a new task's project has
  a `training_dispatch` row AND `task.agent_id == train_agent_id` AND no session:
  `StartSession(group_size=1)` is called once, `session_id` stored on the task,
  `context.areal_proxy = {provider:areal, model:areal-default, api_key:proxy_key,
  base_url:proxy_url}` injected. Idempotent: task with a session is skipped.
  Non-trained task: no RL call, context untouched.
- [ ] Implement the hook at the chokepoint; add query to set task session_id +
  merge context (hand-written generated). `proxy_url` from config (§4.7).
- [ ] Run: `go test ./internal/service/ -run 'Training|SessionOpen'` (+ handler if hooked there).
- [ ] Commit: `feat(training): open RL session + inject areal proxy config on trained task creation`.

---

### Task 6: Runtime provider wiring in execenv (TDD) — size per Task 1c

**Files:** `internal/daemon/execenv/*` (+ tests). If Task 1c found an existing
per-task provider override via context, this task just maps `areal_proxy` onto it;
else add a minimal field.

- [ ] Failing test: a task whose `context.areal_proxy` is set produces a runtime
  config with `provider=areal, model=areal-default, api_key=<proxy_key>,
  base_url=<base_url>` (the `pi --provider areal --model areal-default --api-key`
  invocation / provider config file).
- [ ] Implement minimal mapping.
- [ ] Run: `go test ./internal/daemon/execenv/ -run 'Provider|ArealProxy'`.
- [ ] Commit: `feat(execenv): wire areal proxy provider config from task context`.

---

### Task 7: Session-close hook on completion (TDD)

**Files:** the completion path from Task 1b (`internal/service/task.go` /
completion handler), tests.

- [ ] Failing tests (fake RL client): a task with a training `session_id`
  transitioning to `completed` → `SetReward(session_id, default_reward)` THEN
  `EndSession(session_id)` (order asserted); also fires on `failed`/`cancelled`.
  A task without a session → no RL calls. RL errors are logged, not fatal.
- [ ] Implement the close hook; read `default_reward` from `training_dispatch`
  (fallback config `TRAINING_DEFAULT_REWARD`).
- [ ] Run: `go test ./internal/service/ -run 'TrainingClose|SessionClose'`.
- [ ] Commit: `feat(training): default reward + end_session on trained task completion`.

---

### Task 8: Config + production wiring (TDD-light)

**Files:** config loader (`internal/daemon/config.go` or server config),
handler/service construction, `.env.example`, docs.

- [ ] Add `AREAL_PROXY_URL` (default `http://db_bridge_stub:9100/v1`),
  `AREAL_BRIDGE_STUB_URL`, `AREAL_ADMIN_API_KEY`, `TRAINING_DEFAULT_REWARD`
  (default 1.0). Construct the `arealrl.Client` and inject into the service.
- [ ] Guard: if training is requested but `AREAL_BRIDGE_STUB_URL`/admin key are
  unset, fail the open-hook loudly (don't silently run un-proxied).
- [ ] `.env.example` entries + short note in db_bridge/README or protocol doc.
- [ ] Build touched packages; commit: `chore(training): config + wire arealrl client`.

---

### Task 9: Full regression + AReaL confirm + grep sweep

- [ ] Scoped Go: `go build ./internal/handler/ ./internal/service/ ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/` + `go vet` same + `go test` same (confirm only pre-existing 16 ON CONFLICT handler fails; 0 new).
- [ ] `gofmt -l` clean on touched files.
- [ ] db_bridge smoke (if touched): `cd multica/db_bridge && uv run pytest -q`.
- [ ] AReaL confirm-only: verify `start_session`/`set_reward`/`end_session` exist
  and the bridge routes `/rl/*` to the gateway (no code change expected).
- [ ] grep: `train_agent_id`, `training_dispatch`, `areal_proxy`, `arealrl`
  resolve to intended code only.
- [ ] Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.

---

## Task ledger (track in `.superpowers/sdd/progress.md`)
T1 (read-doc) → T2 (contract) → T3 (persist) → T4 (rl client) → T5 (open hook)
→ T6 (execenv) → T7 (close hook) → T8 (config) → T9 (regression+review).
Bases: multica `main` @ `7969187a` (T7 of sub-project C). Commits local-only
unless the user says push.

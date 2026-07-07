---
change: sub-project-d-session-lifecycle
design-doc: docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md
base-ref: b48c9ab222428efb022fc772e0526cefc055e9aa
archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

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

archived-with: 2026-07-07-sub-project-d-session-lifecycle
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

archived-with: 2026-07-07-sub-project-d-session-lifecycle
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

archived-with: 2026-07-07-sub-project-d-session-lifecycle
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

archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

### Task 4: RL bridge client (Go) (TDD)

**Files:** new `internal/arealrl/client.go` (+ `client_test.go`).

Contract = **experimental openai-proxy stack** (T1/1d, user-approved):
- [ ] `StartSession(ctx, taskID string) (SessionCreds, error)` — POST
  `<stubBaseURL>/rl/start_session`, **admin-key** auth (Bearer/`x-api-key` per
  proxy_rollout_server), body `{task_id, group_size:1}`, decode **flat**
  `{session_id, api_key}` → `SessionCreds{SessionID, ProxyKey}` (api_key = ProxyKey).
- [ ] `SetReward(ctx, proxyKey string, reward float64) error` — POST
  `/rl/set_reward`, **session-key** auth (Bearer `<proxyKey>`), body `{reward}`
  (NO session_id).
- [ ] `EndSession(ctx, proxyKey string) error` — POST `/rl/end_session`,
  **session-key** auth (Bearer `<proxyKey>`), NO body session_id.
- [ ] Client ctor takes stub base URL + admin key.
- [ ] Tests with `httptest.Server`: assert paths/method/auth header (admin on
  start, session-key on reward/end); map api_key→ProxyKey; error on non-2xx;
  missing-key error. First confirm the exact auth header names against
  `areal/experimental/openai/proxy/proxy_rollout_server.py`
  (`_require_session_key` / admin key extraction).
- [ ] Run: `go test ./internal/arealrl/`.
- [ ] Commit: `feat(arealrl): Go client for experimental /rl start/set_reward/end_session`.

archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

### Task 5: Session-open hook at trained-member task creation (TDD)

**Files:** `internal/service/task.go` (the `Enqueue*` family: `EnqueueTaskForIssue`,
`EnqueueTaskForMention`/`EnqueueTaskForSquadLeader`, `EnqueueChatTask`,
`EnqueueQuickCreateTask`) **and** `handler/env_dispatch.go` `EnqueueAgentRun`
(both funnel into `CreateAgentTask`/`CreateChatTask`). NO `/api/agent/start`
(does not exist here — T1/1a). New query to store `session_id`+`proxy_key`+merge
`context` on the task (hand-written generated). Tests.

- [ ] Factor a single server-side helper `maybeOpenTrainingSession(ctx, taskID,
  agentID, projectID, ...)` invoked right after each task insert, so all
  chokepoints share one implementation.
- [ ] Failing tests (fake RL client + fake queries): project has a
  `training_dispatch` row AND `agent_id == train_agent_id` AND no session →
  `StartSession(taskID)` called once, `session_id`+`proxy_key` stored,
  `context.areal_proxy = {provider:areal, model:areal-default, api_key:proxy_key,
  base_url:proxy_url}` injected. Idempotent (has session → skip). Non-trained →
  no RL call, context untouched. Missing bridge config → loud error (Task 8 guard).
- [ ] Implement the helper. Persist the RL session state **inside
  `context.areal_proxy`** (`{provider, model, api_key:proxy_key, base_url,
  session_id}`) via a `MergeTaskContext`/`SetTaskContext` query (hand-written
  generated) — NO new task column (the existing `agent_task.session_id` is the
  runtime/chat session, do not reuse it). `proxy_url` from config (§4.7).
  Resolve project via Issue.ProjectID / ChatSession.ProjectID (T1/1e).
- [ ] Run: `go test ./internal/service/ ./internal/handler/ -run 'Training|SessionOpen|EnvDispatch'`.
- [ ] Commit: `feat(training): open RL session + inject areal proxy config on trained task creation`.

archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

### Task 6: Runtime provider wiring in execenv (TDD) — NEEDS-NEW-FIELD (T1/1c)

**Files:** `handler/daemon.go` (`ClaimTaskByRuntime` ~:1066 — populate a new
field from `task.context.areal_proxy`), `internal/daemon/types.go` (Task /
AgentData struct — new field), `internal/daemon/daemon.go` (ExecOptions build
~:3039), `pkg/agent/pi.go` (confirm pi arg/env mapping), tests.

T1 confirmed there is NO existing per-task, context-sourced provider override
(provider is per-runtime; base_url/api_key only via agent-scoped CustomEnv).

- [ ] First confirm `pkg/agent/pi.go` `buildPiArgs` — how Model→`--provider`/
  `--model` maps and the env-var names pi reads for api-key/base-url.
- [ ] Failing test: a claim response whose `context.areal_proxy` is set yields a
  daemon Task carrying the override, and ExecOptions produce
  `pi --provider areal --model areal-default --api-key <proxy_key>` with
  `base_url=<base_url>` (via the confirmed env vars).
- [ ] Implement: new field on claim response + daemon Task, populated at
  ClaimTaskByRuntime from context, consumed at ExecOptions.
- [ ] Run: `go test ./internal/daemon/... ./internal/handler/ -run 'Provider|ArealProxy|Claim'`.
- [ ] Commit: `feat(execenv): wire areal proxy provider config from task context at claim`.

archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

### Task 7: Session-close hook on completion (TDD)

**Files:** `internal/service/task.go` (`CompleteTask` :1285, `FailTask` :1468,
`CancelTask`/`CancelTaskWithResult` :905/:915), tests.

- [ ] Failing tests (fake RL client): a task whose `context.areal_proxy` carries
  `session_id`+`api_key`(proxy_key) reaching `completed` → `SetReward(proxy_key,
  default_reward)` THEN `EndSession(proxy_key)` (order asserted, session-key
  auth); also fires on `failed`/`cancelled`. Task without `areal_proxy` → no RL
  calls. RL errors logged, not fatal.
- [ ] Implement a shared `maybeCloseTrainingSession(ctx, task)` called from the
  three terminal transitions; read the proxy_key from `task.context.areal_proxy`
  and `default_reward` from `training_dispatch` (fallback config
  `TRAINING_DEFAULT_REWARD`).
- [ ] NOTE (deferred, document only): `runtime_sweeper.FailStaleTasks` (raw SQL)
  bypasses `FailTask`, so timeout tasks won't auto-close — a reaper is future
  hardening, out of D scope.
- [ ] Run: `go test ./internal/service/ -run 'TrainingClose|SessionClose|Complete|Fail|Cancel'`.
- [ ] Commit: `feat(training): default reward + end_session on trained task completion`.

archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

### Task 8: Config + production wiring (TDD-light)

**Files:** config loader (`internal/daemon/config.go` or server config),
handler/service construction, `.env.example`, docs.

- [ ] Add `AREAL_PROXY_URL` (default `http://db_bridge_stub:9100/v1`),
  `AREAL_BRIDGE_STUB_URL`, `AREAL_ADMIN_API_KEY`, `TRAINING_DEFAULT_REWARD`
  (default 1.0). Construct the `arealrl.Client` and inject into the service.
- [ ] **From T6:** pi has no base-url flag — T6 injects the proxy base URL as env
  `AREAL_PROXY_BASE_URL`. Wire the `areal` provider entry in pi's `models.json`
  (or the daemon's provider config) so its base URL reads `$AREAL_PROXY_BASE_URL`,
  so the trained pi actually routes to the bridge stub. Confirm the exact
  models.json/provider-config seam and add a test.
- [ ] Guard: if training is requested but `AREAL_BRIDGE_STUB_URL`/admin key are
  unset, fail the open-hook loudly (don't silently run un-proxied).
- [ ] `.env.example` entries + short note in db_bridge/README or protocol doc.
- [ ] Build touched packages; commit: `chore(training): config + wire arealrl client`.

archived-with: 2026-07-07-sub-project-d-session-lifecycle
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

archived-with: 2026-07-07-sub-project-d-session-lifecycle
---

## Task ledger (track in `.superpowers/sdd/progress.md`)
T1 (read-doc) → T2 (contract) → T3 (persist) → T4 (rl client) → T5 (open hook)
→ T6 (execenv) → T7 (close hook) → T8 (config) → T9 (regression+review).
Bases: areal `master` @ `b48c9ab2` (OpenSpec change creation point);
multica `main` @ `816d1e86c` (T6 tip — code-side base for T7-T9).
T1-T6 complete on multica main (`7969187a..816d1e86c`). This OpenSpec change
tracks T7-T9 only. Commits local-only unless the user says push.

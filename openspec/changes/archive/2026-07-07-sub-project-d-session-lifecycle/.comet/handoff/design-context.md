# Comet Design Handoff

- Change: sub-project-d-session-lifecycle
- Phase: design
- Mode: compact
- Context hash: bc1d7b7213a2f44848f55797ab224538f0554849706063e0c645b404878da81d

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic,
source-traceable context pack, not an agent-authored summary.

## openspec/changes/sub-project-d-session-lifecycle/proposal.md

- Source: openspec/changes/sub-project-d-session-lifecycle/proposal.md
- Lines: 1-60
- SHA256: 9f01e503274a5ccc64badc8f643219213e511b57f05a0883428af8e6066bb2eb

```md
## Why

Sub-project D (training-agent session lifecycle) is mid-implementation: T1-T6 are
committed on multica `main` (7969187a..816d1e86c), but the session-close hook
(T7), config + pi `models.json` wiring (T8), and full regression (T9) remain.
Without T7-T9, trained-agent runs won't auto-close RL sessions (no default
reward + `end_session` on completion) and the proxy routing won't actually work
end-to-end — T6 found that pi has no base-url flag, so the `areal` provider
entry in `models.json` must read `$AREAL_PROXY_BASE_URL` for the trained pi to
route to the bridge stub. This change closes out D so the trained-agent flow is
complete end-to-end with a default-placeholder reward, deferring real reward to
sub-project E.

## What Changes

- **T7 — Session-close hook on task completion.** Shared `maybeCloseTrainingSession`
  helper called from `CompleteTask` / `FailTask` / `CancelTask` in
  `internal/service/task.go`; reads `proxy_key` + `session_id` from
  `task.context.areal_proxy`; calls `SetReward(default_reward)` then
  `EndSession` (session-key Bearer auth, order asserted); RL errors logged, not
  fatal. Sweeper timeout gap (`FailStaleTasks` raw SQL bypasses `FailTask`)
  documented as deferred.
- **T8 — Config + pi `models.json` production wiring.** Env vars
  `AREAL_PROXY_URL`, `AREAL_BRIDGE_STUB_URL`, `AREAL_ADMIN_API_KEY`,
  `TRAINING_DEFAULT_REWARD` (default 1.0); pi `models.json` `areal` provider
  entry with `baseURL=$AREAL_PROXY_BASE_URL` (closes the T6 finding); guard
  fails loud if training requested but bridge config missing.
- **T9 — Full regression + AReaL confirm + grep sweep.** Scoped Go build/test
  (webpush pre-existing failure excluded), db_bridge smoke, AReaL confirm-only
  (no code change expected), grep sweep for `train_agent_id` /
  `training_dispatch` / `areal_proxy` / `arealrl`.

## Capabilities

### New Capabilities
- `training-session-lifecycle`: Server-side lifecycle for RL training sessions —
  opens an AReaL proxy session when a trained-member task is created, routes the
  trained agent's LLM traffic through the proxy, and closes the session with a
  default placeholder reward on task completion.

### Modified Capabilities
(none — `openspec/specs/` is empty; this change creates the first capability
spec in this repo.)

## Impact

- **Code**: multica Go server (`internal/service/task.go`, `internal/arealrl/`,
  config loader, pi `models.json`); no AReaL code change expected (confirm-only).
- **DB**: migration 152 (`training_dispatch`) already landed in T3; no new
  migration in T7-T9.
- **External**: AReaL experimental openai-proxy stack
  (`areal/experimental/openai/proxy/proxy_gateway.py` + `proxy_rollout_server.py`)
  must be running; db_bridge stub channels `/rl/*` already exist (T4 confirmed).
- **Prior work**: T1-T6 on multica main (7969187a..816d1e86c) — full technical
  design in areal `docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md`;
  implementation plan in `docs/superpowers/plans/2026-07-06-training-agent-session-lifecycle.md`;
  SDD ledger in `.superpowers/sdd/progress.md`.
- **Deferred (non-goals)**: real reward / entropy / critic env-save (sub-project
  E); D6 env-dispatch rewire of the wired loop (needs external le-agent
  materialize endpoint); sweeper timeout session reaper (future hardening).
```

## openspec/changes/sub-project-d-session-lifecycle/design.md

- Source: openspec/changes/sub-project-d-session-lifecycle/design.md
- Lines: 1-146
- SHA256: f7fa270ca2a6cead51e57019cb7bf2da2d99062d1ab6ec8d69241f6dae5574b9

\[TRUNCATED\]

```md
# Design — sub-project-d-session-lifecycle

## Context

This OpenSpec change closes out sub-project D (T7-T9). The full technical design
lives in `docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md`;
this document summarizes the high-level architecture decisions and points to the
Superpowers design doc for implementation details.

## Prior work (T1-T6, on multica main 7969187a..816d1e86c)

- **T1** investigation: confirmed Approach A (server-side lifecycle) is feasible;
  chokepoints mapped (see `docs/superpowers/notes/2026-07-06-D-seams.md`).
- **T2** contract: `train_agent_id` on `EnvDispatchRequest` +
  `service.EnvDispatchInput`; validation rule (requires `squad_id` or == `agent_id`).
- **T3** persistence: migration 152 `training_dispatch(project_id PK FK ON DELETE
  CASCADE, workspace_id, train_agent_id, default_reward dp default 1.0, created_at)`;
  `SaveTrainingDispatch` on Deps + adapter + stub + fake.
- **T4** RL client: `internal/arealrl.Client` — `StartSession` (admin-key, flat
  `{session_id, api_key}` response), `SetReward` / `EndSession` (session-key
  Bearer); experimental contract confirmed against `proxy_rollout_server.py`.
- **T5** session-open hook: `maybeOpenTrainingSession` at all `Enqueue*`
  chokepoints (`service/task.go`) + env_dispatch `EnqueueAgentRun`; merges
  `context.areal_proxy = {provider, model, api_key:proxy_key, base_url, session_id}`;
  idempotent; loud error if training target but bridge dep missing.
- **T6** execenv wiring: `ClaimTaskByRuntime` parses `context.areal_proxy` →
  daemon Task/AgentData omitempty fields → ExecOptions (`pi --provider areal
  --model areal-default --api-key <proxy_key>`, `AREAL_PROXY_BASE_URL` env).
  **Finding**: pi has NO base-url flag → T8 must wire `models.json` provider entry.

## Locked decisions (D1-D6, from Superpowers design doc §3)

- **D1 Ownership** — AReaL hosts the LLM proxy + mints session key (`proxy_key`);
  multica orchestrates.
- **D2 Trigger** — external driver (AReaL) tells multica to launch the team;
  `env_dispatch` payload marks the training target via `train_agent_id`.
- **D3 `proxy_url`** — multica deployment config (env var, default
  `http://db_bridge_stub:9100/v1`); `start_session` does NOT return it.
- **D4 Reward boundary** — D writes default placeholder reward on close; real
  reward / entropy / critic env-save = sub-project E.
- **D5 Lifecycle owner** — server-side (Approach A); multica Go owns start/close;
  daemon stays a thin consumer of injected config.
- **D6 Team-member creation** — leader task only at dispatch; teammates spawned
  later (leader @mention delegation). Session-open hook fires at trained-member
  task creation, not at `env_dispatch`.

## RL contract (experimental openai-proxy stack)

Served ONLY by `areal/experimental/openai/proxy/proxy_gateway.py` +
`proxy_rollout_server.py` (the v2 `inference_service` gateway lacks `end_session`).

| Endpoint | Auth | Body | Returns |
|---|---|---|---|
| `start_session` | admin-key (Bearer/`x-api-key`) | `{task_id, group_size:1}` | flat `{session_id, api_key}` (api_key = `proxy_key`) |
| `set_reward` | session-key (Bearer `proxy_key`) | `{reward}` (no `session_id`) | — |
| `end_session` | session-key (Bearer `proxy_key`) | (no body) | — |

`task_id` = multica `agent_task_queue.id`.

## T7-T9 architecture

### T7: Session-close hook

```

task terminal transition (service/task.go)
┌────────────────────────────────────────────────┐ │ CompleteTask │ FailTask │
CancelTask │ └────────────────┴───────────┴───────────────────┘ │ ▼
maybeCloseTrainingSession(ctx, task) │ ┌───────────────┼────────────────────────┐ ▼ ▼ ▼
has context. default_reward SetReward(proxy_key, default) areal_proxy? from
training_dispatch │ │ (fallback config) ▼ │ │ EndSession(proxy_key) ▼ │ │ skip ◀── no
──────┘ ▼

````

Full source: openspec/changes/sub-project-d-session-lifecycle/design.md

## openspec/changes/sub-project-d-session-lifecycle/tasks.md

- Source: openspec/changes/sub-project-d-session-lifecycle/tasks.md
- Lines: 1-93
- SHA256: 7001ea0beb2ff897752102b62bb6a53999282dadfec72198e147167a00fb6001

[TRUNCATED]

```md
# Tasks — sub-project-d-session-lifecycle

## Prior work (T1-T6, already on multica main 7969187a..816d1e86c)

T1 investigation, T2 contract (`train_agent_id`), T3 `training_dispatch` +
migration 152, T4 Go RL client (`internal/arealrl`), T5 session-open hook,
T6 execenv provider wiring — all complete. See `.superpowers/sdd/progress.md`
for the full ledger. This change tracks only the remaining T7-T9.

## Remaining work (this change)

### Task 7: Session-close hook on completion (TDD)

**Files**: `internal/service/task.go` (`CompleteTask` :1285, `FailTask` :1468,
`CancelTask`/`CancelTaskWithResult` :905/:915), tests.

- [ ] Failing tests (fake RL client): a task whose `context.areal_proxy` carries
  `session_id`+`api_key` (proxy_key) reaching `completed` →
  `SetReward(proxy_key, default_reward)` THEN `EndSession(proxy_key)` (order
  asserted, session-key auth); also fires on `failed` / `cancelled`. Task
  without `areal_proxy` → no RL calls. RL errors logged, not fatal.
- [ ] Implement shared `maybeCloseTrainingSession(ctx, task)` called from the
  three terminal transitions; read `proxy_key` from `task.context.areal_proxy`
  and `default_reward` from `training_dispatch` (fallback config
  `TRAINING_DEFAULT_REWARD`).
- [ ] NOTE (deferred, document only): `runtime_sweeper.FailStaleTasks` (raw SQL)
  bypasses `FailTask`, so timeout tasks won't auto-close — a reaper is future
  hardening, out of D scope.
- [ ] Run: `go test ./internal/service/ -run 'TrainingClose|SessionClose|Complete|Fail|Cancel'`.
- [ ] Commit: `feat(training): default reward + end_session on trained task completion`.

### Task 8: Config + production wiring (TDD-light)

**Files**: config loader (`internal/daemon/config.go` or server config),
handler/service construction, `.env.example`, pi `models.json` (or daemon
provider config), docs.

- [ ] Add `AREAL_PROXY_URL` (default `http://db_bridge_stub:9100/v1`),
  `AREAL_BRIDGE_STUB_URL`, `AREAL_ADMIN_API_KEY`, `TRAINING_DEFAULT_REWARD`
  (default 1.0). Construct the `arealrl.Client` and inject into the service.
- [ ] **From T6:** pi has no base-url flag — T6 injects the proxy base URL as
  env `AREAL_PROXY_BASE_URL`. Wire the `areal` provider entry in pi's
  `models.json` so its base URL reads `$AREAL_PROXY_BASE_URL`, so the trained
  pi actually routes to the bridge stub. Confirm the exact `models.json` /
  provider-config seam and add a test.
- [ ] Guard: if training is requested but `AREAL_BRIDGE_STUB_URL` / admin key
  are unset, fail the open-hook loudly (don't silently run un-proxied).
- [ ] `.env.example` entries + short note in db_bridge/README or protocol doc.
- [ ] Build touched packages; commit: `chore(training): config + wire arealrl client`.

### Task 9: Full regression + AReaL confirm + grep sweep

- [ ] Scoped Go: `go build ./internal/handler/ ./internal/service/
  ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/` +
  `go vet` same + `go test` same (confirm only pre-existing 16 ON CONFLICT
  handler fails; 0 new).
- [ ] `gofmt -l` clean on touched files.
- [ ] db_bridge smoke (if touched): `cd multica/db_bridge && uv run pytest -q`.
- [ ] AReaL confirm-only: verify `start_session` / `set_reward` / `end_session`
  exist and the bridge routes `/rl/*` to the gateway (no code change expected).
- [ ] grep: `train_agent_id`, `training_dispatch`, `areal_proxy`, `arealrl`
  resolve to intended code only.
- [ ] Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.

## Test runners / constraints

- multica Go: `DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable`.
  **Scope build/test to touched packages** — `go build ./...` fails on the
  pre-existing `internal/service/webpush/webpush.go:180` "constant 4096
  overflows byte" (go 1.26). Pre-existing: 16 `internal/handler` ON CONFLICT
  (42P10) daemon/claim failures; `cmd/server` does not compile here (imports
  webpush) — validate router/handler edits by `go build ./internal/handler/` +
  inspection.
- **Codegen rule:** do NOT run `sqlc generate` repo-wide (creates colliding
  `agent_skill_suggestion.sql.go` / `evolution.sql.go`, breaks build). For any
  new query: add to `queries/*.sql` AND hand-write the generated Go in
  `generated/<file>.sql.go` mirroring a sibling. T7-T9 are not expected to add
  queries.
- db_bridge (if touched): `cd multica/db_bridge && uv run pytest -q`.
- Commit each task to multica `main`; record commit hash in
````

Full source: openspec/changes/sub-project-d-session-lifecycle/tasks.md

## openspec/changes/sub-project-d-session-lifecycle/specs/training-session-lifecycle/spec.md

- Source:
  openspec/changes/sub-project-d-session-lifecycle/specs/training-session-lifecycle/spec.md
- Lines: 1-132
- SHA256: 0845592fda0130ee90ce339e17ff5a841451b7f9f281a7338890ba11793f0053

\[TRUNCATED\]

```md
## ADDED Requirements

### Requirement: Session-open on trained-member task creation

The system SHALL open an AReaL RL proxy session when a task is created
server-side for an agent marked as a training target (via `env_dispatch`
`train_agent_id`), before the task is claimable by the daemon.

The session-open hook SHALL:
- fire at every server-side task-creation chokepoint that produces a trained
  teammate task (the `Enqueue*` family in `internal/service/task.go` and
  `env_dispatch` `EnqueueAgentRun`);
- call `start_session(task_id=agent_task_queue.id, group_size=1)` with
  admin-key auth against the experimental openai-proxy stack;
- store `session_id` and `proxy_key` (the returned `api_key`) into
  `task.context.areal_proxy`;
- inject `provider=areal`, `model=areal-default`, `api_key=proxy_key`,
  `base_url=proxy_url` into `task.context.areal_proxy`;
- be idempotent — a task that already has `context.areal_proxy` is skipped;
- leave non-trained tasks (no `training_dispatch` row, or `agent_id` !=
  `train_agent_id`) untouched, with no RL call and no `context.areal_proxy`.

#### Scenario: Trained teammate task created via leader @mention delegation
- **WHEN** a squad leader delegates a task to the trained member via @mention
  (the trained member's `agent_id` matches a `training_dispatch.train_agent_id`
  for the task's project)
- **THEN** the system calls `start_session` with the new task's id and
  `group_size=1`, stores `session_id` + `proxy_key` into
  `task.context.areal_proxy`, and injects the areal proxy provider config

#### Scenario: Idempotent retry on already-sessioned task
- **WHEN** the session-open hook fires on a task whose `context.areal_proxy` is
  already populated
- **THEN** the system skips `start_session` (no duplicate session) and leaves
  `context.areal_proxy` unchanged

#### Scenario: Non-trained task is untouched
- **WHEN** a task is created for an agent that is NOT a training target (no
  `training_dispatch` row for the project, or `agent_id` != `train_agent_id`)
- **THEN** the system makes no RL call and does not set `context.areal_proxy`

### Requirement: Trained-agent LLM routing through AReaL proxy

When a trained-member task is claimed by the daemon, the runtime SHALL route
the trained agent's LLM traffic through the AReaL proxy configured in
`task.context.areal_proxy`.

The runtime SHALL:
- read `provider`, `model`, `api_key`, `base_url` from
  `task.context.areal_proxy` at `ClaimTaskByRuntime`;
- configure `pi` with `--provider areal --model areal-default --api-key
  <proxy_key>`;
- inject `AREAL_PROXY_BASE_URL` as an env var on the pi process;
- wire the `areal` provider entry in pi's `models.json` (or daemon provider
  config) so its base URL reads `$AREAL_PROXY_BASE_URL`.

#### Scenario: Trained task claimed by daemon routes through proxy
- **WHEN** a task with `context.areal_proxy` set is claimed by the daemon
- **THEN** the daemon's ExecOptions produce `pi --provider areal --model
  areal-default --api-key <proxy_key>` with `AREAL_PROXY_BASE_URL` set, and the
  pi `models.json` `areal` provider entry resolves its base URL from that env var

#### Scenario: Non-trained task claimed by daemon uses normal provider
- **WHEN** a task without `context.areal_proxy` is claimed by the daemon
- **THEN** the daemon uses the agent's normal provider config (no areal proxy
  override)

### Requirement: Session-close on task completion

The system SHALL close the RL session when a trained-member task reaches a
terminal state via the standard service paths.

The session-close hook SHALL:
- attach to `CompleteTask`, `FailTask`, and `CancelTask` /
  `CancelTaskWithResult` in `internal/service/task.go`;
- call `SetReward(proxy_key, default_reward)` THEN `EndSession(proxy_key)`,
  both with session-key Bearer auth;
- read `proxy_key` and `session_id` from `task.context.areal_proxy`;
- read `default_reward` from `training_dispatch.default_reward` (fallback to
  `TRAINING_DEFAULT_REWARD` config, default 1.0);
```

Full source:
openspec/changes/sub-project-d-session-lifecycle/specs/training-session-lifecycle/spec.md

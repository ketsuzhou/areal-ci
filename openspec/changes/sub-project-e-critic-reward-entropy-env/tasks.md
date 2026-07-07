# Tasks — sub-project-e-critic-reward-entropy-env

## Prior work (depends on sub-project D — COMPLETE)

Sub-project D (session lifecycle) is complete and archived
(`openspec/changes/archive/2026-07-07-sub-project-d-session-lifecycle/`).
D's multica `main` commits `816d1e86c..0b68f606d` (T7-T10: close hook,
config guard, wiring) are local-only. E's T7/T8/T10/T11 are unblocked.

T1-T6 of the overall training-agent effort are on multica `main`
(`7969187a..816d1e86c`). E's T1-T9 are complete (commits below); T10,
T11 remain.

## Remaining work (this change)

### Task 1: Investigation — confirm seams for critic dispatch + entropy capture

No production code. Produce a short markdown note
(`docs/superpowers/notes/2026-07-06-E-seams.md`) answering:

- **1a. Critic auto-spawn seam.** Confirm where to inject the critic task
  creation on trained-task-terminal. D's close hook attaches to
  `CompleteTask`/`FailTask`/`CancelTask` in `internal/service/task.go` —
  can the critic-spawn hook attach at the same chokepoints (before D's
  close logic)? Or does it need a separate transition?
- **1b. Critic task → trained session linkage.** How does the deferred
  close hook (on critic-terminal) find the trained session's `proxy_key`?
  New DB column on the critic task? Context field? Join via
  `training_dispatch`?
- **1c. Critic reward result shape.** Where does the critic's scalar reward
  live in its task result? New field on `agent_task`? Parsed from output?
  Stored in `context`?
- **1d. env_id availability at session-open.** Confirm `env_id` is
  available on `training_dispatch` (or the task) at the time D's open hook
  fires. If not, where to thread it from.
- **1e. AReaL proxy logprobs path.** Confirm
  `areal/experimental/openai/proxy/proxy_rollout_server.py` /
  `proxy_gateway.py` can request `logprobs=true` on proxied calls and
  persist logprobs per interaction. Identify the exact insertion points.
- **1f. StartSessionRequest env_id field.** Confirm
  `areal/experimental/openai/proxy/server.py` `StartSessionRequest` can
  gain an optional `env_id` field without breaking existing callers
  (additive change).

**STOP-and-report** (begin report `BLOCKED:`) if: the critic auto-spawn
cannot be injected at the same chokepoints as D's close hook (e.g. the
terminal transition is in raw SQL like the runtime sweeper, bypassing
`FailTask`), OR AReaL's proxy cannot transparently capture logprobs (e.g.
the upstream LLM SDK doesn't expose logprobs). Otherwise begin `DONE:` with
the mapping. Commit the note.

### Task 2: Contract — `critic_agent_id` on env_dispatch (TDD)

**Files**: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/handler/env_dispatch_test.go`, `internal/service/env_dispatch_test.go`.

- [x] Failing tests: request with `critic_agent_id` shape-validated (400 on
  malformed UUID); service accepts it; validation — allowed with `squad_id`
  + `train_agent_id`; equal to `agent_id` rejected (can't critique yourself);
  empty ⇒ unchanged behavior.
- [x] Add `CriticAgentID string` to `EnvDispatchRequest` (json
  `critic_agent_id,omitempty`) and `service.EnvDispatchInput`; thread
  through the handler→service mapping.
- [x] Handler UUID shape-check when present; service `validate()` rule.
- [x] Run: `go test ./internal/handler/ ./internal/service/ -run 'EnvDispatch|Dispatch'`.
- [x] Commit: `feat(env-dispatch): accept critic_agent_id (critic for trained agent)`.

### Task 3: Persist critic intent — extend `training_dispatch` (TDD)

**Files**: `server/migrations/153_training_dispatch_critic.up.sql`/`.down.sql`,
`server/pkg/db/queries/training_dispatch.sql` (extend),
`server/pkg/db/generated/training_dispatch.sql.go` (hand-written, mirror
sibling), `internal/service/env_dispatch.go` (+ deps method + adapter +
fake), tests.

- [x] Migration: `ALTER TABLE training_dispatch ADD COLUMN critic_agent_id
  UUID NULL`.
- [x] Queries: extend `CreateTrainingDispatch` to accept `critic_agent_id`;
  extend `GetTrainingDispatchByProject` to return it.
- [x] Service: persist `critic_agent_id` when set. Failing tests first
  (fake asserts critic_agent_id is stored when set; NULL when empty).
- [x] Verify generated code compiles: `go build ./pkg/db/generated/ ./internal/service/`.
- [x] Commit: `feat(training): persist critic_agent_id on training_dispatch (migration 153)`.

### Task 4: AReaL contract — env_id on StartSessionRequest (TDD)

**Files**: `areal/experimental/openai/proxy/server.py`,
`areal/experimental/openai/proxy/proxy_rollout_server.py`,
`areal/experimental/openai/proxy/proxy_gateway.py`, tests.

- [x] Add `env_id: str | None = None` to `StartSessionRequest`.
- [x] Persist `env_id` on the session (extend session data structure).
- [x] Tests: `start_session` accepts and persists `env_id`; old requests
  without `env_id` still work (additive).
- [x] Commit (in areal repo): `feat(proxy): accept env_id on start_session for trajectory attribution`.

### Task 5: RL bridge client — env_id in StartSession (TDD, multica Go)

**Files**: `internal/arealrl/client.go`, `internal/arealrl/client_test.go`.

- [x] Failing tests: `StartSession(ctx, taskID, envID string)` includes
  `env_id` in the request body when non-empty; omits when empty.
- [x] Implement: add `envID` parameter; marshal into request body
  conditionally.
- [x] Run: `go test ./internal/arealrl/`.
- [x] Commit: `feat(arealrl): pass env_id to start_session`.

### Task 6: Session-open hook — pass env_id (TDD, multica)

**Files**: `internal/service/task.go` (D's `maybeOpenTrainingSession`),
tests.

- [x] Failing tests: when `training_dispatch` has `env_id`, the open hook
  passes it to `arealrl.Client.StartSession`. When no `env_id`, omitted.
- [x] Implement: read `env_id` from `training_dispatch` (or env_dispatch
  input) and pass to the RL client.
- [x] Run: `go test ./internal/service/ -run 'Training|SessionOpen|EnvDispatch'`.
- [x] Commit: `feat(training): pass env_id when opening RL session`.

### Task 7: Critic auto-spawn on trained-terminal (TDD, multica)

**Files**: `internal/service/task.go` (new `maybeSpawnCriticTask`),
tests.

- [x] Failing tests: trained task terminal + `critic_agent_id` set +
  session open → critic task created with trained agent's output as input;
  trained session NOT closed. Trained task terminal + no critic → no spawn
  (D's behavior). Idempotent (don't spawn twice).
- [x] Implement: `maybeSpawnCriticTask` called from the same chokepoints as
  D's close hook (but BEFORE the close — the close is deferred to
  critic-terminal). Link the critic task to the trained session (T1/1b
  decides the linkage mechanism).
- [x] Run: `go test ./internal/service/ -run 'CriticSpawn|TrainingClose'`.
- [x] Commit: `feat(training): auto-spawn critic task on trained-task terminal`.

### Task 8: Deferred close hook on critic-terminal (TDD, multica)

**Files**: `internal/service/task.go` (extend `maybeCloseTrainingSession`
or add `maybeCloseTrainingSessionFromCritic`), tests.

- [x] Failing tests: critic task terminal + linked trained session open →
  `SetReward(critic_reward)` then `EndSession` on the trained session.
  Critic produced no reward → `SetReward(default)` fallback. No linked
  session → no-op. RL errors logged, not fatal.
- [x] Implement: read critic reward from critic task result (T1/1c decides
  shape); read linked trained session's `proxy_key` (T1/1b); call
  `SetReward` + `EndSession`.
- [x] Run: `go test ./internal/service/ -run 'TrainingClose|CriticClose'`.
- [x] Commit: `feat(training): deferred close hook on critic-terminal with critic reward`.

### Task 9: AReaL proxy — logprobs capture for entropy (TDD, areal Python)

**Files**: `areal/experimental/openai/proxy/proxy_rollout_server.py`,
`areal/experimental/openai/proxy/proxy_gateway.py`, tests.

- [x] Failing tests: proxied LLM calls include `logprobs=true` in the
  request; logprobs in the response are persisted per interaction. Upstream
  LLM error on logprobs → logged, interaction still recorded.
- [x] Implement: inject `logprobs=true` in the proxy's forwarded request;
  persist logprobs in the session's interaction data.
- [x] Run: `uv run pytest tests/test_proxy_*.py` (or equivalent).
- [x] Commit (in areal repo): `feat(proxy): capture logprobs for entropy computation`.

### Task 10: Config + production wiring (TDD-light, multica)

**Files**: config loader, handler/service construction, `.env.example`,
docs.

- [ ] No new config required for critic (critic_agent_id comes from
  env_dispatch). Confirm `TRAINING_DEFAULT_REWARD` (from D) is the fallback.
- [ ] Wire the critic auto-spawn + deferred close into the task service
  construction.
- [ ] `.env.example` entries (if any new config) + short note in
  db_bridge/README or protocol doc.
- [ ] Build touched packages; commit: `chore(training): wire critic auto-spawn + deferred close`.

### Task 11: Full regression + cross-repo verification + grep sweep

- [ ] Scoped multica Go: `go build ./internal/handler/ ./internal/service/
  ./internal/arealrl/ ./pkg/db/generated/` + `go vet` same + `go test` same
  (confirm only pre-existing failures; 0 new).
- [ ] `gofmt -l` clean on touched files.
- [ ] AReaL Python: `uv run pytest` on proxy tests + `pre-commit run
  --files areal/experimental/openai/proxy/`.
- [ ] db_bridge smoke: `cd multica/db_bridge && uv run pytest -q`.
- [ ] Cross-repo E2E (if feasible): a trained session with critic produces
  a non-default reward + env_id + entropy in AReaL's trajectory export.
- [ ] grep: `critic_agent_id`, `critic-driven-training-signal`, `env_id`
  (in arealrl/proxy), `logprobs` resolve to intended code only.
- [ ] Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.

## Test runners / constraints

- multica Go: `DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable`.
  **Scope build/test to touched packages** — `go build ./...` fails on the
  pre-existing `internal/service/webpush/webpush.go:180` "constant 4096
  overflows byte" (go 1.26). Pre-existing: 16 `internal/handler` ON CONFLICT
  (42P10) daemon/claim failures; `cmd/server` does not compile here (imports
  webpush). Validate router/handler edits by `go build ./internal/handler/`
  + inspection.
- **Codegen rule:** do NOT run `sqlc generate` repo-wide (creates colliding
  `agent_skill_suggestion.sql.go`/`evolution.sql.go`, breaks build). For any
  new query: add to `queries/*.sql` (source of truth) AND hand-write the
  generated Go in `generated/<file>.sql.go` mirroring a sibling.
- AReaL Python: `uv run pytest` from areal root; `pre-commit run --files
  <touched>` before commit. Many tests require GPU — explain skips when
  unavailable.
- db_bridge (if touched): `cd multica/db_bridge && uv run pytest -q`.
- Commit each task to multica `main` (multica-side) or areal `master`
  (areal-side); record commit hashes in `.superpowers/sdd/progress.md`.

## Task ledger

(append "Task N: complete (commits <base7>..<head7>, review clean)" as tasks
finish)

T1: complete (areal `052020fb` — seams investigation note)
T2: complete (multica `0fb6c2644` — critic_agent_id on env_dispatch)
T3: complete (multica `9e3fa6f0b` — migration 153 + training_dispatch.critic_agent_id)
T4: complete (areal `4fb458ea` — env_id on StartSessionRequest)
T5: complete (multica `eeac55e62` + `343215231` — arealrl Go client + env_id)
T6: complete (multica `fb40610c7` — session-open hook passes env_id)
T7: complete (multica `0b68f606d..f43a9ab66` + `b77577bfa` review-fix; spec ✅, code quality Approved. maybeSpawnCriticTask + RouteTerminalTrainingTask + FindCriticTaskForTrained/CreateCriticTask queries; 4 new tests + 8 existing close tests pass. Minor findings: dead MaybeCloseTrainingSession public method kept for T8/T10; brief test-regex was wrong (SpawnCritic not CriticSpawn).)
T8: complete (multica `b77577bfa..de2ac7aa9`; spec ✅, code quality Approved. maybeCloseTrainingSessionFromCritic + parseCriticReward + RouteTerminalTrainingTask critic-task check; 6 new tests + 4 T7 + 7 D-close pass. Minor: defaultReward==0 fallback is pre-existing pattern from D; T8-2 fixture missing trained_task_id in critic_of — optional.)
T9: complete (areal `277d6f7b` — logprobs capture in proxy)
T10: pending (D-blocked — now unblocked)
T11: pending

Bases: areal `master` @ `48d49aba` (D's OpenSpec change creation); multica
`main` @ `816d1e86c` (D's T6 tip). D complete (archived
`2026-07-07-sub-project-d-session-lifecycle`). Commits local-only unless the
user says push.

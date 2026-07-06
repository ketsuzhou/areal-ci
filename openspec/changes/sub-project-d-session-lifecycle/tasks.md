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
  `.superpowers/sdd/progress.md`.

## Task ledger

(append "Task N: complete (commits <base7>..<head7>, review clean)" as tasks
finish)

T7: _(dispatched from prior SDD; pending close-out under this OpenSpec change)_
T8: pending
T9: pending

Bases: multica `main` @ 816d1e86c (T6 tip). Commits local-only unless the user
says push.

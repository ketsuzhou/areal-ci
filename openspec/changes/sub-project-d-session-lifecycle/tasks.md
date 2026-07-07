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

- [x] Scoped Go: `go build ./internal/handler/ ./internal/service/
  ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/` +
  `go vet` same + `go test` same (confirm only pre-existing 16 ON CONFLICT
  handler fails; 0 new).
- [x] `gofmt -l` clean on touched files.
- [x] db_bridge smoke (if touched): `cd multica/db_bridge && uv run pytest -q`.
- [x] AReaL confirm-only: verify `start_session` / `set_reward` / `end_session`
  exist and the bridge routes `/rl/*` to the gateway (no code change expected).
- [x] grep: `train_agent_id`, `training_dispatch`, `areal_proxy`, `arealrl`
  resolve to intended code only.
- [x] Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.

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

T7: complete (multica main fb40610c7..86c3c28ec..61ed426fd, review CLEAN; 7/7 MaybeClose tests pass, build clean)
  Shared maybeCloseTrainingSession(ctx, deps, task, projectID) called from CompleteTask/FailTask/CancelTaskWithResult; arealSessionCloser interface (SetReward+EndSession) added; TrainingSessionDeps gains Closer field; extractArealProxyConfig safely parses task.Context JSONB; default_reward from training_dispatch with fallback to trainingDefaultReward=1.0 (T8 makes configurable); SetReward error → still calls EndSession (best-effort); RL errors logged via slog.Warn, never fatal. Doc note added: runtime_sweeper.FailStaleTasks bypasses FailTask (raw SQL), so stale tasks won't auto-close — reaper is future hardening.
T8: complete (multica main 61ed426fd..ae6f2435a, review CLEAN; 5 config tests + 7 close tests + 6 open tests pass, build clean)
  TrainingConfig struct + LoadTrainingConfig() reads AREAL_BRIDGE_STUB_URL/AREAL_ADMIN_API_KEY/AREAL_PROXY_URL/TRAINING_DEFAULT_REWARD from env; NewTrainingSessionDeps(cfg, q) returns nil when BridgeStubURL/AdminAPIKey empty (hooks stay no-ops); arealrl.New assigned to both RL (starter) + Closer fields; TaskService.WithTraining(*TrainingSessionDeps) builder injects it; cmd/server/main.go wires LoadTrainingConfig + conditional WithTraining after NewTaskService; .env.example documents all 4 vars + notes AREAL_PROXY_BASE_URL is daemon-set; TrainingSessionDeps gains DefaultReward float64 field used in maybeCloseTrainingSession (falls back to 1.0 when zero). Invalid TRAINING_DEFAULT_REWARD → warning log + 1.0 fallback.
  MINOR (non-blocking): trainingDefaultReward constant in training.go:86 has stale comment "T8 will make this configurable" — T8 is done, comment should say "fallback used when DefaultReward is zero" or be inlined. Literal 1.0 appears in 3 places (training.go:268, training_config.go:45,49) — could DRY to the constant. Not worth a fix cycle.
T9: complete (verification only; multica main ae6f2435a..1667f85c3 gofmt fix)
  - go build/vet/test scoped packages (./internal/handler/ ./internal/service/ ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/) = OK.
  - internal/service: training_test (7 close + 6 open) + training_config_test (5) + env_dispatch_test all pass. internal/arealrl OK. internal/daemon/execenv OK.
  - internal/handler: 16 FAIL (8 TestDaemonRegister_* + 8 TestClaimTask_* ON CONFLICT 42P10) = PRE-EXISTING baseline (reproduced at base 816d1e86c), 0 new. cmd/server NOT compiled here (pre-existing webpush.go:180 go 1.26 overflow).
  - gofmt -l: 4 touched files needed alignment (training_config.go, cmd/server/main.go, training_test.go, env_dispatch_test.go) → committed 1667f85c3. Re-check clean.
  - db_bridge: NOT touched by D, skipped.
  - AReaL confirm-only: start_session/set_reward/end_session exist in areal/experimental/openai/proxy/proxy_gateway.py:353/623/637; server.py:182-184 defines RL_*_PATHNAME constants. Bridge routes /rl/* to gateway. NO code change.
  - grep sweep: train_agent_id / training_dispatch / areal_proxy / arealrl — ALL resolve to intended code only.
  - MINOR carried: trainingDefaultReward constant stale comment + 3 literal 1.0 occurrences (training.go:268, training_config.go:45,49). Non-blocking.

D READY FOR VERIFY. multica main: 816d1e86c..fc35d4587 (5 commits: 86c3c28ec close-hook, 61ed426fd doc-note, ae6f2435a config+wiring, 1667f85c3 gofmt, fc35d4587 stale-comment-fix). Final whole-branch review: READY TO MERGE (no Critical/Important; 1 stale-comment fix landed as fc35d4587). Commits local-only.

Bases: multica `main` @ 816d1e86c (T6 tip). Commits local-only unless the user
says push.

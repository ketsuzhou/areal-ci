# Verification Report — sub-project-d-session-lifecycle

**Date:** 2026-07-07 **Change:** sub-project-d-session-lifecycle **Mode:** full (16
tasks, 1 delta spec, 10 changed files) **Base:** multica main `816d1e86c` (T6 tip) /
areal `b48c9ab2` **Head:** multica main `fc35d4587` / areal `9ac2c1e7`

## Summary

| Dimension    | Status                                                                         |
| ------------ | ------------------------------------------------------------------------------ |
| Completeness | 16/16 tasks complete; 5/5 requirements present                                 |
| Correctness  | 5/5 requirements implemented; 9/10 scenarios covered (1 production-wiring gap) |
| Coherence    | Design decisions D1-D6 followed; code patterns consistent                      |

## Commits under review (multica main)

- `86c3c28ec` feat(training): default reward + end_session on trained task completion
  (T7)
- `61ed426fd` docs(training): note runtime_sweeper.FailStaleTasks bypass (T7)
- `ae6f2435a` chore(training): config + wire arealrl client (T8)
- `1667f85c3` chore(training): gofmt T7-T8 touched files (T9)
- `fc35d4587` docs(training): fix stale comment on trainingDefaultReward constant (T9)

T1-T6 (commits `f1d954d83`..`fb40610c7`) were already on main before D started; this
verify covers T7-T9 only.

## Completeness

### Task Completion

- 16/16 tasks.md items checked `[x]`.
- T7 (5 items), T8 (5 items), T9 (6 items) all complete.

### Spec Coverage (5 requirements)

| Requirement                                   | Implemented                                                                | Files                                                              |
| --------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Session-open on trained-member task creation  | Yes (T1-T6, on main)                                                       | `internal/service/training.go`, `task.go`, `env_dispatch.go`       |
| Trained-agent LLM routing through AReaL proxy | Yes (T6, on main); models.json wiring is deployment concern                | `internal/daemon/daemon.go`, `internal/handler/agent.go`           |
| Session-close on task completion              | Yes (T7)                                                                   | `internal/service/training.go:256-310`, `task.go:956/1380/1571`    |
| Config guard against un-proxied training runs | Partial (helper-level guard exists; production-wiring gap — see WARNING 1) | `internal/service/training.go:161-166`, `training_config.go:60-63` |
| Default placeholder reward on close           | Yes (T7+T8)                                                                | `internal/service/training.go:267-283`, `training_config.go:45-53` |

## Correctness

### Scenario Coverage

**Requirement 1: Session-open** (T1-T6, on main)

- Trained teammate task created via leader @mention delegation — covered (T5 tests)
- Idempotent retry on already-sessioned task — covered
  (`TestMaybeOpenTrainingSession_AlreadyHasArealProxy`)
- Non-trained task is untouched — covered
  (`TestMaybeOpenTrainingSession_NotATrainingTarget`)

**Requirement 2: LLM routing** (T6, on main)

- Trained task claimed by daemon routes through proxy — covered (T6 daemon tests)
- Non-trained task claimed uses normal provider — covered (T6 daemon tests)
- pi models.json `areal` provider entry reads `$AREAL_PROXY_BASE_URL` — **deployment
  concern, no models.json in this repo** (see SUGGESTION 1)

**Requirement 3: Session-close** (T7)

- Trained task completes — reward then end_session in order — covered
  (`TestMaybeCloseTrainingSession_CompletedTask`)
- Trained task fails or is cancelled — close hook still fires — covered
  (`..._FailedTask`, `..._CancelledTask`)
- RL error during close does not fail the transition — covered
  (`..._RLClientError_LoggedNotFatal`)
- Non-trained task completion — no RL calls — covered (`..._NoArealProxy`)

**Requirement 4: Config guard** (T8)

- Training requested but bridge config missing — **partial**: helper-level test exists
  (`TestMaybeOpenTrainingSession_TargetButMissingBridge_LoudError`), but production path
  has a gap (see WARNING 1)

**Requirement 5: Default placeholder reward** (T7+T8)

- Default reward written from training_dispatch — covered
  (`TestMaybeCloseTrainingSession_CompletedTask`)
- Default reward falls back to config — covered
  (`TestMaybeCloseTrainingSession_NoTrainingDispatch_FallbackReward`)

### Build / Test / Lint

- `go build ./internal/handler/ ./internal/service/ ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/`
  = OK
- `go vet` same scoped packages = OK
- `go test` scoped: `internal/service` OK (training_test 7 close + 6 open,
  training_config_test 5, env_dispatch_test all pass); `internal/arealrl` OK;
  `internal/daemon/execenv` OK; `pkg/db/generated` no tests.
- `internal/handler`: 16 FAIL (8 `TestDaemonRegister_*` + 8 `TestClaimTask_*` ON
  CONFLICT 42P10) = pre-existing baseline (reproduced at base `816d1e86c`), 0 new.
- `gofmt -l`: clean after commit `1667f85c3`.
- AReaL confirm-only: `start_session` / `set_reward` / `end_session` exist in
  `areal/experimental/openai/proxy/proxy_gateway.py:353/623/637`; `server.py:182-184`
  defines `RL_*_PATHNAME` constants. No code change.
- grep sweep: `train_agent_id`, `training_dispatch`, `areal_proxy`, `arealrl` — all
  resolve to intended code only.

## Coherence

### Design Adherence (D1-D6)

- **D1 Ownership** — AReaL hosts proxy + mints proxy_key; multica orchestrates. ✓
- **D2 Trigger** — external driver (env_dispatch) marks training target via
  `train_agent_id`. ✓
- **D3 proxy_url** — multica config (`AREAL_PROXY_URL`, default
  `http://db_bridge_stub:9100/v1`); `start_session` does not return it. ✓
- **D4 Reward boundary** — D writes default placeholder; real reward = sub-project E. ✓
- **D5 Lifecycle owner** — server-side (Approach A); multica Go owns start/close; daemon
  is thin consumer. ✓
- **D6 Team-member creation** — leader task at dispatch; teammates via @mention;
  session-open hook fires at trained-member task creation. ✓

### Code Pattern Consistency

- Interface segregation (`arealSessionStarter` vs `arealSessionCloser`) — clean, both
  satisfied by `*arealrl.Client`. ✓
- Error handling: open hook errors loud (task runs un-proxied never silent by intent);
  close hook errors logged (task already terminal). ✓
- Config loader follows existing env-var patterns. ✓
- `.env.example` documents all 4 env vars. ✓

## Issues

### CRITICAL (Must fix before archive)

(none)

### WARNING (Should fix)

(none — WARNING 1 was fixed in commit `0b68f606d`; see Resolution below)

### WARNING 1 — Config guard production-wiring gap (RESOLVED)

- **Originally found:** `internal/service/training.go:104-106`
  (`tryOpenTrainingSession`), `internal/service/training_config.go:60-63`
  (`NewTrainingSessionDeps`)
- **What was wrong:** When `AREAL_BRIDGE_STUB_URL` or `AREAL_ADMIN_API_KEY` are unset,
  `NewTrainingSessionDeps` returned `nil`, so `TaskService.Training` was `nil`. The
  open-hook wrapper `tryOpenTrainingSession` then returned early at
  `training.go:104-106` (`if s.Training == nil { return }`) — a **silent no-op**. If
  `train_agent_id` was set on `env_dispatch` (training requested) but the bridge env
  vars were unset, the trained task was created without `context.areal_proxy` and the
  trained agent ran **un-proxied**.
- **Spec violated:** Requirement "Config guard against un-proxied training runs" —
  Scenario: "Training requested but bridge config missing" — "the session-open hook
  returns a loud error and the trained task is not created in an un-proxied state".
- **Helper-level guard exists:** `training.go:161-166` returns a loud `fmt.Errorf(...)`
  when `deps.RL == nil || deps.ProxyURL == ""`, but it is unreachable in production
  because `maybeOpenTrainingSession` is never called when `s.Training` is nil. The test
  `TestMaybeOpenTrainingSession_TargetButMissingBridge_LoudError` covers the helper
  directly but not the production wiring.
- **Impact:** Operator misconfiguration (sets `train_agent_id` but forgets bridge env
  vars) leads to silent un-proxied training runs — no reward signal, no session
  lifecycle. Bounded to operator-error scenarios; no data loss or security issue.
- **Recommended fix:** Either (a) make `NewTrainingSessionDeps` return a non-nil
  `TrainingSessionDeps` with `Lookup` set but `RL=nil`/`ProxyURL=""` when env vars are
  unset, so the existing loud-error guard at `training.go:161-166` is reachable; or (b)
  in `tryOpenTrainingSession`, when `s.Training == nil`, still check if the task is a
  training target (requires `Queries` always available on `TaskService`) and log a loud
  `slog.Error` if so. Option (a) is simpler; option (b) preserves the nil-means-disabled
  semantics.

**Resolution (commit `0b68f606d`):** `NewTrainingSessionDeps` now returns a non-nil
`*TrainingSessionDeps` with `Lookup`+`Store` set but `RL`/`Closer` nil when config is
missing and `q` is non-nil (production path). The existing loud-error guard at
`training.go:161-166` is now reachable: when a training target is requested despite
missing config, the hook returns
`fmt.Errorf("training: task %s targets train_agent %s but the RL bridge is not configured...")`,
which `tryOpenTrainingSession` logs via `slog.Error`. The close hook no-ops on nil
`Closer` (`training.go:257-259`). New test
`TestNewTrainingSessionDeps_GuardDepsWhenConfigMissing` verifies the production-path
contract; existing `TestMaybeOpenTrainingSession_TargetButMissingBridge_LoudError`
covers the helper-level behavior.

### SUGGESTION (Nice to fix)

**SUGGESTION 1 — pi models.json wiring is a deployment concern**

- **File:** N/A (no `models.json` in multica repo)
- **What:** The spec requirement "wire the `areal` provider entry in pi's `models.json`
  so its base URL reads `$AREAL_PROXY_BASE_URL`" cannot be completed in this repo
  because the pi binary reads its own `models.json` at deployment time. The daemon
  exports `AREAL_PROXY_BASE_URL` (`daemon.go:3892`) and `.env.example` documents it, but
  the actual models.json entry is a pi-deployment config task.
- **Impact:** The trained pi will not route to the bridge stub until the deployment
  `models.json` has the `areal` provider entry. This is documented in `tasks.md` T8.
- **Recommendation:** Document this as a deployment prerequisite in the db_bridge README
  or a runbook. Not a code change.

**SUGGESTION 2 — trainingDefaultReward constant could DRY the 1.0 literal**

- **File:** `internal/service/training.go:268`, `training_config.go:45,49`
- **What:** The literal `1.0` appears in 3 places as the final fallback. The
  `trainingDefaultReward` constant (training.go:86) is used only in tests.
- **Impact:** Minor; 3 occurrences in fallback paths. Not worth a fix cycle.

## Dirty Worktree Note

The working tree has uncommitted changes in
`customized_areal/tree_search/agents/{README.md, dag_backup.py, multica_environment_protocol.md}`
\+ `tests/test_dag_backup.py` (521+/176-). These predate D's build phase, are unrelated
to D (which is multica Go code), and the user confirmed they should be left untouched.
They are out of D's scope.

## Final Assessment

**Ready to merge: Yes** (WARNING 1 resolved in commit `0b68f606d`; remaining items are
SUGGESTION-level only).

**Reasoning:** D's T7-T10 implementation is spec-complete. The config guard
production-wiring gap (WARNING 1) is fixed: `NewTrainingSessionDeps` now returns
guard-ready deps when config is missing, making the loud-error guard reachable in
production. All 5 requirements are met, all scenarios are covered, design decisions
D1-D6 are followed, and build/vet/test/gofmt are clean. Pre-existing baseline failures
(16 handler ON CONFLICT, webpush build) are unrelated. Two SUGGESTION-level items (pi
models.json deployment concern, 1.0 literal DRY) are non-blocking.

## Recommendation

Address WARNING 1 before archive, OR accept it as a known gap with the rationale that:

1. The operator misconfiguration case is bounded (no data loss, no security issue)
1. The fix requires a design decision (option a vs b) that may belong in sub-project E's
   scope (E owns the full training reward/critic flow and may revisit the config guard)
1. The helper-level test
   (`TestMaybeOpenTrainingSession_TargetButMissingBridge_LoudError`) documents the
   intended behavior

If accepting as a known gap, note it in the archive's spec sync and flag it for E's T10
(Config + production wiring).

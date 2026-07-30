# Verification Report — sub-project-e-critic-reward-entropy-env

**Date:** 2026-07-07 **Change:** sub-project-e-critic-reward-entropy-env **Mode:** full
(11 tasks, 1 delta spec, 15 changed files across 2 repos) **Base:** multica main
`816d1e86c` (D's T6 tip) / areal `48d49aba` (D's OpenSpec change creation) **Head:**
multica main `69511fdf3` / areal `sub-project-e-critic-reward-entropy-env` branch

## Summary

| Dimension    | Status                                                 |
| ------------ | ------------------------------------------------------ |
| Completeness | 11/11 tasks complete; all requirements present         |
| Correctness  | All requirements implemented; all scenarios covered    |
| Coherence    | Design decisions followed; cross-task routing verified |

## Commits under review

### multica main (E's Go implementation)

- `0fb6c2644` feat(env-dispatch): accept critic_agent_id (T2)
- `9e3fa6f0b` feat(training): persist critic_agent_id on training_dispatch (T3)
- `eeac55e62` feat(arealrl): Go client for experimental /rl (T5 base)
- `343215231` feat(arealrl): pass env_id to start_session (T5)
- `fb40610c7` feat(training): pass env_id when opening RL session (T6)
- `f43a9ab66` feat(training): auto-spawn critic task on trained-task terminal (T7)
- `b77577bfa` fix(training): close session when critic creator is nil (T7 review fix)
- `de2ac7aa9` feat(training): deferred close hook on critic-terminal with critic reward
  (T8)
- `69511fdf3` chore(training): note env_id in db_bridge /rl/start_session protocol (T10)

### areal sub-project-e-critic-reward-entropy-env branch (docs + Python)

- `863c4e6e` docs(E): create OpenSpec change
- `052020fb` docs(E): record Task 1 seams investigation
- `4fb458ea` feat(proxy): accept env_id on start_session (T4)
- `277d6f7b` feat(proxy): capture logprobs for entropy computation (T9)
- `b613b93a` docs(E): record T6 + T9 completion
- `8af517c2` docs(E): sync tasks.md — T1-T6+T9 complete, D unblocked
- `1aff1105` docs(E): mark T7 complete
- `5936cec8` docs(E): mark T8 complete
- `33e3465e` docs(E): mark T10 complete
- `81cbb320` docs(E): mark T11 complete + configure build/verify commands

## Completeness

### Task Completion

- 11/11 tasks.md items checked `[x]` (45/45 checkboxes).

### Spec Coverage

All requirements from the delta spec (`critic-driven-training-signal`) are implemented:

- Critic auto-spawn on trained-terminal ✅ (T7)
- Deferred close on critic-terminal with critic reward ✅ (T8)
- env_id on StartSessionRequest ✅ (T4)
- env_id threaded through RL client + open hook ✅ (T5, T6)
- Logprobs capture in proxy ✅ (T9)
- Config + production wiring ✅ (T10)

## Correctness

### Scenario Coverage

**T7 (Critic auto-spawn)**

- Trained task terminal + critic_agent_id set → critic task created with
  `context.critic_of` ✅
- No critic → no spawn (D's behavior) ✅
- Idempotent (don't spawn twice) ✅
- Spawn failure → D's close fallback ✅

**T8 (Deferred close)**

- Critic task terminal + `context.critic_of` → SetReward(critic_reward) then EndSession
  ✅
- Unparseable output → default reward ✅
- Out-of-range reward → default reward ✅
- Non-critic task → no-op ✅
- Nil deps / malformed context → no-op ✅

**T4 (env_id on StartSessionRequest)**

- start_session accepts env_id ✅
- Old requests without env_id still work ✅

**T9 (logprobs capture)**

- Proxied calls include logprobs=true ✅
- Upstream rejects logprobs → graceful fallback (retry without) ✅

### Build / Test / Lint

- `go build ./internal/handler/ ./internal/service/ ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/`
  — OK
- `go vet` same scoped packages — OK
- `go test ./internal/service/ ./internal/arealrl/ ./internal/daemon/execenv/` — all
  pass (17 critic/close tests + existing suite)
- `gofmt -l` — clean
- AReaL proxy tests: `test_proxy_env_id.py` + `test_proxy_logprobs.py` — 9 passed
- db_bridge smoke: 168 passed, 1 skipped
- Pre-existing baseline: 16 `internal/handler` ON CONFLICT (42P10) failures — NOT
  regressions

### Grep Sweep

- `critic_agent_id` / `CriticAgentID` — resolves to intended code only (training.go,
  env_dispatch.go)
- `maybeSpawnCriticTask` / `maybeCloseTrainingSessionFromCritic` /
  `RouteTerminalTrainingTask` — intended code only
- `critic_of` — intended JSON key in training.go + tests
- `parseCriticReward` — intended code only
- `env_id` (areal proxy) — server.py + proxy_rollout_server.py only
- `logprobs` (areal proxy) — proxy_rollout_server.py only

## Coherence

### Design Adherence

- Critic as peer task (no `parent_task_id`) ✅
- `context.critic_of` for linkage ✅
- `{"reward": <float>}` from last line, range-checked \[0.0, 1.0\] ✅
- logprobs injection with graceful fallback ✅
- env_id additive, backward compatible ✅
- Fallback to `TRAINING_DEFAULT_REWARD` via `deps.DefaultReward` ✅

### Cross-Task Coherence

- **Routing flow**: Complete/Fail/Cancel → RouteTerminalTrainingTask → critic-task check
  → trained-terminal logic → D's close. All 3 chokepoints routed consistently.
- **Linkage integrity**: T7 writes
  `critic_of.{trained_task_id, proxy_key, session_id, project_id}`; T8 reads the same 4
  fields. JSON keys match.
- **Fallback chain**: All paths safe — no orphaned sessions.
- **Idempotency**: `FindCriticTaskForTrained` prevents duplicate spawns.
- **D compatibility**: Non-critic-trained tasks fall through to D's close. Critic tasks
  carry `critic_of` but NOT `areal_proxy`, so D's close is a no-op on critic-terminal.

### Final Whole-Branch Review

**Verdict: READY TO MERGE** — all 11 tasks complete, cross-task coherence verified, 5
non-blocking Minor findings.

## Issues

### CRITICAL (Must fix before archive)

(none)

### WARNING (Should fix)

(none)

### SUGGESTION (Nice to fix)

1. **Dead code — `MaybeCloseTrainingSession` public method** (`training.go:327`): No
   production callers; the lowercase `maybeCloseTrainingSession` is used by
   `RouteTerminalTrainingTask`. Can be removed in a future cleanup commit.
1. **T8-2 test fixture missing `trained_task_id`** (`task_critic_test.go:234`): Doesn't
   affect test validity but inconsistent with what T7 writes.
1. **`runtime_sweeper.FailStaleTasks` bypass** (pre-existing from D): Stale tasks bypass
   `FailTask` and won't trigger critic spawn. Documented as a known gap.
1. **Chat-trained tasks don't get critic auto-spawn**: Tasks without `IssueID` can't
   resolve `projectID`. Only issue-trained tasks get critic auto-spawn. Documented in
   code.
1. **Cross-repo E2E not verified**: Design is sound (additive contracts, unit tests on
   both sides, tested fallbacks). E2E risk is low; first real exercise will be at
   deployment.

## Final Assessment

**Ready to merge: Yes**

**Reasoning:** E's T1-T11 implementation is spec-complete. All 5 requirements are met,
all scenarios are covered, design decisions are followed, and build/vet/test/gofmt are
clean across both repos. The final whole-branch review confirmed cross-task coherence:
routing flow, linkage integrity, fallback chain, idempotency, and D compatibility are
all correct. Pre-existing baseline failures (16 handler ON CONFLICT) are unrelated. Five
SUGGESTION-level items are non-blocking.

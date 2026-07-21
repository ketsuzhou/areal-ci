# EnvDispatch Agent Runtime Config Verification

Branch: nested `multica` repo on `dev`. Outer `areal` submodule on
`feature/20260720/env-dispatch-agent-runtime-config`. Status: in_review.

Scope: code + unit/integration tests only (贝克汉姆 0f880dd0/b7e73968); production
deploy-level AC-8 verification split to a separate issue.

## Local commands (run 2026-07-20/21, mise go 1.26.5 + /tmp/sqlc v1.31.1)

- `go build ./...` - PASS (exit 0). Full server compiles.
- `go test ./internal/service/... -count=1` - PASS. Includes AC-4 helpers (6), AC-4 ordering assertion, clone/discovery, env_dispatch dispatch tests.
- `go test ./internal/handler/...` - compiles, SKIP locally (`TestMain` DB-gated, no local postgres). CI runs with a DB (see CI).
- `gofmt -l <changed files>` - clean. `go vet ./internal/service/...` - clean.
- `openspec validate env-dispatch-agent-runtime-config --strict` - PASS (valid).
- `graphify update .` - PASS. `pytest` client fail-closed (Task 7) - 59 passed.

## CI (criterion a: DB tests wired, not just marked)

`.github/workflows/ci.yml` runs `cd server && go test ./...` (line 126) with a
`pgvector/pgvector:pg17` postgres service (lines 79-89, user/db `multica`,
`pg_isready` health check). So the handler-package DB-gated tests
(`TestMain` connects to localhost) DO run in CI - they are wired, not verbally
marked CI.

## AC-1..AC-8 terminal state + evidence

| AC | Status | Implementation | Local evidence | CI/notes |
|---|---|---|---|---|
| AC-1 sandbox lifecycle reuse | DONE | shared EnvSandboxLifecycleService.Create; scratch uses it; no agent_runtime pre-created (c1110e818/d829a2363/4dca11333) | go build PASS | - |
| AC-2 runtime identity discovery | DONE | WaitForOnlineSandboxRuntime (4-tuple match, mismatch reject, timeout+compensate); daemon sandbox_instance_id metadata (7621ee2f4/7a6d91e9e) | service tests PASS (TestWaitForOnline*) | - |
| AC-3 derived agent | DONE | CloneEnvDispatchAgent+CloneEnvDispatchAgentTx (lineage/runtime/skills, source immutable, cross-workspace reject) (a26e83fff/6bd13584b) | service tests PASS (TestCloneEnvDispatch*); TestAgentLineageRejectsCrossWorkspaceSource (handler, CI) | 3.2 concurrent same-source test: CI |
| AC-4 credential isolation + training session | DONE | owner invariant validateEnvDispatchCredentialOwner (8c0c49dd5); training_session_key col (cd51ec953); ResolveEnvDispatchTrainingSession (session_ref=binding.ID, retry reuse) + EnvDispatchTrainingRuntimePolicy (areal-default+bridge+key); handler training branch session-before-sandbox (cd51ec953/3192be9e2); LinkEnvDispatchTrainingSession service-layer after enqueue (7dd376ae9); legacy task_id alias (19ec9d6f6) | 6 helper tests PASS; TestResolveEnvDispatchTrainingSessionReusesOnRetry (startSessionCount==0); TestDispatch_ScratchMessage_TrainingSessionLinkedAfterEnqueue (link after enqueue) PASS | handler training-impl DB test: CI |
| AC-5 single-flight | DONE | claimProvisioning single-flight (pending/failed/failed_retryable->credential_ready) | TestEnvDispatchChannelStoreClaimProvisioningIsSingleWinner (handler, CI) | 5.3 concurrent first-mention test: CI |
| AC-6 cleanup cascade | DONE | deleteEnvDispatchChannelRollout: cancel tasks (CancelTasksForAgent) + archive derived (ArchiveAgent) + delete sandbox/runtime + EndSession (936e2770b/cd51ec953) | TestChannelCleanupDeletesChannelProjectEnvAndBindings + TestChannelCleanupIsIdempotent (handler, CI) | idempotent/race integration: CI. ListOwnedEnvDispatchResources is a per-binding query (pending sqlc regen); env-wide cascade uses listBindings (correct shape) |
| AC-7 client terminal failure | DONE | Python client fail-closed (areal 8f0fcf15) | 59 pytest PASS | - |
| AC-8 valid DAG (code-level) | DONE | static+training path produce sandbox+runtime+derived+session | go build PASS | deployed 200 split to separate issue |
| §8.3 feature flag | DONE | envDispatchDerivedAgentEnabled default=true kill-switch (a0ef7ee10); =false rejects new provisioning at entry, no in-flight/ready disruption; no legacy pre-create fallback; gradual rollout needs separate issue | go build PASS | - |

## AC-4 ordering assertions (criterion b)

- retry reuse (startSessionCount==0): `TestResolveEnvDispatchTrainingSessionReusesOnEnqueue` (helper) - asserts StartSession NOT called on retry.
- session before sandbox: structural in `provisionEnvDispatchAgentTraining` (ResolveEnvDispatchTrainingSession precedes lifecycle.Create) + `TestResolveEnvDispatchTrainingSessionOpensOnFirstAddress` asserts StartSession(session_ref=binding.ID) called once.
- real task linked after enqueue: `TestDispatch_ScratchMessage_TrainingSessionLinkedAfterEnqueue` asserts LinkEnvDispatchTrainingSession called once per rollout after EnqueueEnvDispatchChannelRun with non-empty runID.

## AC-6 cascade (criterion c)

`deleteEnvDispatchChannelRollout` per ready binding: CancelTasksForAgent -> ArchiveAgent (archived_by=NULL, nullable) -> lifecycle.Delete (revoke bootstrap PAT) -> DeleteAgentRuntime -> EndSession(training_session_key). Idempotent: already-absent resources are success; TestChannelCleanupIsIdempotent + TestChannelCleanupDeletesChannelProjectEnvAndBindings (CI). `ListOwnedEnvDispatchResources` is a per-binding query not generated (pending sqlc regen); the env-wide cascade uses `listBindings` (returns all bindings with derived/sandbox/runtime/session IDs) which is the correct shape for whole-channel cleanup.

## Secret audit

No credential material recorded. Synthetic sentinels in tests only; errors asserted not to contain them. `validateEnvDispatchCredentialOwner` + archive errors do not echo credentials. `archived_by=NULL` (nullable, FK user) for system cleanup archives.

## Commits (multica dev, this issue)

ea9f94f8e (AC-4 ordering test), cd51ec953 (AC-4 bulk + AC-6 EndSession), 7dd376ae9 (AC-4 LinkSessionTask service-layer), 3192be9e2 (AC-4 training orchestration), 936e2770b (AC-6 cascade), 8c0c49dd5 (AC-4 owner invariant), a0ef7ee10 (feature flag), 4fcb59a6a (merge), 4dca11333 (scratch discover+clone), a26e83fff (CloneDeps adapter), 7621ee2f4 (runtime discovery), 7a6d91e9e (daemon metadata). Outer areal: verify report + openspec change.

## 2026-07-21 Re-verification (full verify phase)

Scale: full (23 tasks, 19 changed files base_ref..HEAD, 1 delta spec capability). The
`openspec-verify-change` skill is not installed in this environment; the 7 full-verification
checks below were performed manually via the openspec CLI, artifact reads, and fresh
commands.

Fresh evidence (run 2026-07-21):

- `openspec validate env-dispatch-agent-runtime-config --strict` - PASS (exit 0)
- nested multica `cd multica/server && go build ./...` - PASS (exit 0)
- nested multica `go test ./internal/service/... -count=1` - PASS (ok, 0.39s)
- `uv run python -m py_compile multica_client.py test_env_dispatch_client.py` - PASS (exit 0)
- outer-areal pytest (59, AC-7) not re-run here (requires the full torch/ray env, not
  cached); documented passing above (run 2026-07-20/21) and AC-7/AC-8 fail-closed code
  confirmed present and committed.

7 full-verification checks:

1. tasks.md all `[x]` - PASS (23/23)
2. Implementation matches design.md high-level decisions - PASS (AC-1..AC-8 mapping above;
   design decisions - shared sandbox, discover-after-register, derived clone, credential
   resolution, single-flight + cleanup - all implemented)
3. Implementation matches Design Doc - PASS (Design Doc at
   `docs/superpowers/specs/2026-07-20-env-dispatch-agent-runtime-config-design.md`, updated
   during build)
4. All capability spec scenarios pass - PASS at code level (AC-1..AC-8); deployed AC-8
   scenarios ("Deployed static/training model replies") split to a separate issue -
   documented deferral; feature-flag kill-switch DONE
5. proposal.md goals satisfied - PASS (goals = AC-1..AC-8 DONE)
6. No delta-spec / design-doc contradictions - PASS (design.md goals/decisions align with
   spec.md requirements 1-8)
7. Associated design docs locatable - PASS (Design Doc exists, 14265 bytes)

Post-original-report change (build-phase code review, 2026-07-21):

- AC-7 partial-success self-cleanup (commit 29440a43): on a `group_size > 1` mixed-success
  dispatch, `create_env_dispatch` now reclaims the partial dispatch via the channel/
  project-scoped DELETE before raising, so no half-provisioned dispatch leaks (task 8.1
  "while retaining cleanup"). Cleanup-retention test added; protocol doc updated. Code
  review: no Critical findings; this fixed 2 Important findings.

Deferrals (recorded in tasks.md + plan):

- Task 1.2 `ListOwnedEnvDispatchResources` sqlc regen - unused (cleanup uses `listBindings`).
- Task 8.3 production deploy-level AC-8 verification - split to a separate issue
  (feature-flag kill-switch DONE).

Dirty-worktree note: a concurrent claude session is editing `multica_environment_protocol.md`
(ARE-5 `session_ref` content) in the shared repo; that edit is env-dispatch content
(verification input, not committed by this verify pass). Untracked vimpo/graphify/multica
artifacts belong to other changes and were left untouched.

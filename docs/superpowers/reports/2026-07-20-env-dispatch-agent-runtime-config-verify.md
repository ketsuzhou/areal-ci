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

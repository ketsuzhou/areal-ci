# EnvDispatch Agent Runtime Config Verification

Branch: nested `multica` repo on `dev` (env-dispatch derived-agent runtime config
integrated via 4fcb59a6a + feature flag a0ef7ee10 + credential owner-check
8c0c49dd5). Outer `areal` submodule on `feature/20260720/env-dispatch-agent-runtime-config`.

Scope note: per 贝克汉姆 0f880dd0 ruling, this issue delivers code + unit/integration
tests only; production deploy-level verification (AC-8 deployed 200) is split to a
separate issue.

## Local commands (run 2026-07-20)

- `mise exec go@1.26.5 -- go build ./internal/handler/... ./internal/service/... ./internal/daemon/... ./internal/arealrl/...` — PASS (exit 0). Toolchain available via mise (go 1.26.5) + /tmp/sqlc v1.31.1; the prior "areal-algo has no go/sqlc" premise was stale.
- `mise exec go@1.26.5 -- gofmt -l <changed files>` — clean (no diffs).
- `mise exec go@1.26.5 -- go vet ./internal/handler/...` — PASS (exit 0).
- `mise exec go@1.26.5 -- go test ./internal/service/... -run 'EnvDispatch|Clone|SandboxRuntime|WaitForOnline' -count=1` — PASS (clone lineage/runtime, workspace-mismatch, runtime discovery mismatch/timeout, etc.).
- `mise exec go@1.26.5 -- go test ./internal/handler/... -run 'EnvDispatch|Clone|ValidateEnvDispatch' -count=1` — SKIP locally. The `internal/handler` package has a DB-gated `TestMain` that skips the whole package when no test database is reachable (`testPool == nil`). Handler tests (including the new `TestValidateEnvDispatchCredentialOwner` and DB-backed provisioning/cleanup/single-flight tests) run in CI with a database. Build/gofmt/vet confirm they compile and are well-formed.
- `openspec validate env-dispatch-agent-runtime-config --strict` — PASS ("Change is valid").
- `./.venv-test/bin/pytest customized_areal/tree_search/tests/test_env_dispatch_client.py test_multica_dag_client.py test_multi_agent_env_dispatch.py -q` — PASS (59 passed). Client fail-closed (Task 7, debug domain) verified green.

## Implemented / verified present (AC mapping)

- AC-1 (frontend sandbox lifecycle reuse): shared `EnvSandboxLifecycleService.Create` mints daemon nonce; scratch path uses it; no `agent_runtime` pre-created for scratch. (commits c1110e818, d829a2363, 4dca11333)
- AC-2 (runtime identity safe discovery): `WaitForOnlineSandboxRuntime` matches workspace+provider=pi+daemon_id+sandbox_instance_id+online; rejects mismatch ("runtime identity mismatch"); timeout ("runtime readiness timeout") + compensation. (7621ee2f4; daemon metadata 7a6d91e9e)
- AC-3 (derived global agent): `CloneEnvDispatchAgent` logic + `CloneEnvDispatchAgentTx` transactional adapter; copies name/instructions/approvedConfig/skills; records `source_agent_id` lineage; binds `runtime_id`; source immutable; cross-workspace rejected. (a26e83fff, 6bd13584b)
- AC-4 (credential isolation): `model_config_owner_agent_id` persisted on binding; `validateEnvDispatchCredentialOwner` enforces owner==source (fail closed, retryable) wired after the single-flight claim. (8c0c49dd5 this turn) **Gap: training session orchestration (start_session with binding.ID as session_ref before sandbox + LinkSessionTask after real task) — see Remaining.**
- AC-5 (single-flight): `claimProvisioning` single-flight claim (pending/failed/failed_retryable -> credential_ready); concurrent callers observe winner. **Gap: explicit concurrent first-mention integration test (DB-backed) not yet added.**
- AC-6 (cleanup): `deleteEnvDispatchChannelRollout` marks deleting, waits for in-flight, deletes sandbox+runtime+channel/project/env. **Gap: does not archive derived agent, cancel derived tasks, close training session, or revoke model credentials; `ListOwnedEnvDispatchResources` unused.**
- AC-7 (client terminal failure): Python client rejects per-rollout errors, DAG timeout, malformed/cyclic/dangling DAG; cleanup in finally. (areal 8f0fcf15; 59 tests PASS)
- AC-8 (valid DAG, code-level): static path produces sandbox+runtime+derived+session; deployed 200 verification split to separate issue per 贝克汉姆 ruling.
- §8.3 feature flag: `envDispatchDerivedAgentEnabled()` reads `ENV_DISPATCH_DERIVED_AGENT` (default false); gates the scratch derived-agent path at provision.go entry. (a0ef7ee10)

## Remaining gaps (not yet complete; why #5 is not in_review)

1. **AC-6 cleanup cascade**: extend `deleteEnvDispatchChannelRollout` to archive derived global agents, cancel derived tasks, close training sessions, revoke bootstrap/model credentials; adopt `ListOwnedEnvDispatchResources`; add idempotent + provisioning-race integration tests. DB-backed (verifies in CI).
2. **AC-4 training session orchestration**: spec requires `start_session(session_ref=binding.ID, env_id)` BEFORE sandbox creation (model `areal-default`, bridge URL from `AREAL_BRIDGE_URL`/`areal_bridge_url` config), then insert the real task after derived readiness and `LinkSessionTask`. Current code opens the session with `taskID` as session_ref when the task runs (training.go). This rearchitects the training dispatch path — a training-main-chain change. Per areal-algo Escalation rules ("影响训练主链路的改动，先提方案再合"), a plan must be proposed and confirmed (zhoujie22/贝克汉姆) before implementation.
3. **AC-3/AC-5 concurrent tests (3.2/5.3)**: concurrent same-source dispatch -> distinct derived; concurrent first-mention -> single provision. DB-backed (verifies in CI).
4. **Local verification limitation**: `internal/handler` package tests skip locally (DB-gated `TestMain`); they run in CI. Non-DB unit tests (service package: clone, discovery) pass locally.

## Secret audit

No credential material recorded in this report. Synthetic sentinels (`sentinel-static-key`, `sentinel-training-key`, `synthetic-secret-for-tests`) are used only in tests; error messages are asserted not to contain them. `validateEnvDispatchCredentialOwner` error does not echo credential values.

## Correction of prior breakdowns

My earlier comments (cdbd7d6c triage, 76763875 breakdown) undercounted completed work: they stated Task 5 "NOT done" and the derived-agent clone "missing." Both were inaccurate — the scratch first-address state machine (4dca11333) and the transactional clone adapter (a26e83fff) are implemented, and the feature flag (a0ef7ee10) was added by a prior run on `dev`. This report reflects the audited true state.

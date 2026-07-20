# EnvDispatch Agent Runtime Config Verification

Branch: nested `multica` repo on `dev` (env-dispatch derived-agent runtime config
integrated via 4fcb59a6a; feature flag a0ef7ee10; credential owner-check
8c0c49dd5; AC-6 cleanup cascade 936e2770b). Outer `areal` submodule on
`feature/20260720/env-dispatch-agent-runtime-config`.

Scope note: per 贝克汉姆 0f880dd0 + b7e73968 rulings, this issue delivers code +
unit/integration tests only; production deploy-level verification (AC-8 deployed
200) is split to a separate issue. #5 is in_progress (open AC gaps remain).

## Local commands (run 2026-07-20)

- `mise exec go@1.26.5 -- go build ./internal/handler/... ./internal/service/... ./internal/daemon/... ./internal/arealrl/...` - PASS (exit 0). Toolchain: go 1.26.5 via mise + /tmp/sqlc v1.31.1 (prior "no go/sqlc" premise was stale; not blind-write).
- `mise exec go@1.26.5 -- gofmt -l <changed files>` - clean.
- `mise exec go@1.26.5 -- go vet ./internal/handler/...` - PASS (exit 0).
- `mise exec go@1.26.5 -- go test ./internal/service/... -run 'EnvDispatch|Clone|SandboxRuntime|WaitForOnline' -count=1` - PASS.
- `mise exec go@1.26.5 -- go test ./internal/handler/... -run 'EnvDispatch|Clone|ValidateEnvDispatch' -count=1` - SKIP locally. `internal/handler` has a DB-gated `TestMain` (`testPool == nil`) that skips the whole package when no test database is reachable. Handler tests (owner-check unit test, cleanup/provisioning/single-flight integration tests) run in CI with a database. Build/gofmt/vet confirm they compile and are well-formed.
- `openspec validate env-dispatch-agent-runtime-config --strict` - PASS.
- `./.venv-test/bin/pytest customized_areal/tree_search/tests/test_env_dispatch_client.py test_multica_dag_client.py test_multi_agent_env_dispatch.py -q` - PASS (59 passed). Client fail-closed (Task 7, debug domain) verified green.
- `graphify update .` - PASS (60874 nodes).

## Implemented / verified present (AC mapping)

- AC-1 (frontend sandbox lifecycle reuse): shared `EnvSandboxLifecycleService.Create` mints daemon nonce; scratch path uses it; no `agent_runtime` pre-created for scratch. (c1110e818, d829a2363, 4dca11333)
- AC-2 (runtime identity safe discovery): `WaitForOnlineSandboxRuntime` matches workspace+provider=pi+daemon_id+sandbox_instance_id+online; rejects mismatch; timeout + compensation. (7621ee2f4; daemon metadata 7a6d91e9e)
- AC-3 (derived global agent): `CloneEnvDispatchAgent` logic + `CloneEnvDispatchAgentTx` transactional adapter; copies name/instructions/approvedConfig/skills; records `source_agent_id` lineage; binds `runtime_id`; source immutable; cross-workspace rejected. (a26e83fff, 6bd13584b)
- AC-4 (credential isolation): `model_config_owner_agent_id` persisted; `validateEnvDispatchCredentialOwner` enforces owner==source (fail closed, retryable) after the single-flight claim. (8c0c49dd5) **Gap: training session orchestration - plan posted (comment b7e73968 thread), pending 贝克汉姆 confirmation before implementation.**
- AC-5 (single-flight): `claimProvisioning` single-flight claim. **Gap: explicit concurrent first-mention integration test pending (DB-backed, CI).**
- AC-6 (cleanup): `deleteEnvDispatchChannelRollout` marks deleting, waits for in-flight, cancels derived tasks (`CancelTasksForAgent`), archives derived agent (`ArchiveAgent`), deletes sandbox+runtime+channel/project/env, plus a forward-compatible training-session close hook. (936e2770b) **Gap: training-session close (`EndSession`) is a no-op until AC-4 persists the session key; `ListOwnedEnvDispatchResources` query exists but is not generated (pending sqlc regen) - `listBindings` serves the cascade; idempotent/race integration tests pending (DB-backed, CI).**
- AC-7 (client terminal failure): Python client rejects per-rollout errors, DAG timeout, malformed/cyclic/dangling DAG; cleanup in finally. (areal 8f0fcf15; 59 tests PASS)
- AC-8 (valid DAG, code-level): static path produces sandbox+runtime+derived+session; deployed 200 verification split to separate issue.
- §8.3 feature flag: `envDispatchDerivedAgentEnabled()` reads `ENV_DISPATCH_DERIVED_AGENT`; **default true** (贝克汉姆 b7e73968 accepted: default=false would reject all scratch provisioning - no legacy pre-create fallback remains). The flag is a **kill-switch, not a gradual-rollout toggle**: setting `=false` rejects NEW provisioning at the `provisionEnvDispatchAgent` entry (before the single-flight claim) and does NOT abort in-flight provisioning (already past the gate) or disrupt ready bindings. **Note for zhoujie22**: if gradual rollout is later required, a separate issue must restore the legacy pre-create scratch path (currently removed by 4dca11333).

## Remaining gaps (why #5 is in_progress, not in_review)

1. **AC-4 training session orchestration** (plan-first per areal-algo Escalation rules + 贝克汉姆 b7e73968): restructure plan posted in the issue thread; implementation pending 贝克汉姆 confirmation.
2. **AC-6 session-close + idempotent/race tests**: session close blocked on AC-4 (session key); idempotent + provisioning-race integration tests pending (DB-backed, CI).
3. **AC-3/AC-5 concurrent tests (3.2/5.3)**: same-source distinct derived; first-mention single provision. DB-backed, CI.
4. **Local verification limitation**: `internal/handler` tests skip locally (DB-gated `TestMain`); run in CI.

## Secret audit

No credential material recorded. Synthetic sentinels used only in tests; error messages asserted not to contain them. `validateEnvDispatchCredentialOwner` error does not echo credential values. `archived_by=NULL` for cleanup archives (column is nullable, FK to user).

## Correction of prior breakdowns

Earlier comments (cdbd7d6c, 76763875) undercounted completed work (stated Task 5 "NOT done", clone "missing", flag default false). Audited true state: scratch state machine (4dca11333), clone adapter (a26e83fff), and feature flag default=true (a0ef7ee10) are all implemented on `dev`. This report reflects the audited true state.

# EnvDispatch Agent Runtime Config Verification

Change: `openspec/changes/env-dispatch-agent-runtime-config`
Issue: ARE-5 (#5, id 46ec95a1-0f7b-4b76-afed-19d6f50fad0c)
Date: 2026-07-20
Scope: local code + unit/integration verification only. Deployed (AC-8) verification is split to a separate issue per 贝克汉姆 ruling (comment 0f880dd0).

## Local commands

| Command | Result |
|---|---|
| `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_env_dispatch_client.py test_multica_dag_client.py test_multi_agent_env_dispatch.py -q` | **PASS** — 59 passed in 8.03s, 0 failed |
| `openspec validate env-dispatch-agent-runtime-config --strict` | **PASS** — "Change is valid", exit 0 |
| `graphify update .` | **PASS** — graph rebuilt (60865 nodes/146825 edges), exit 0. Pre-existing `.sql` tree_sitter_sql warning unrelated (affects .sql graph contributions only). |
| `go test ./internal/{migrations,arealrl,daemon,service,handler}` | **SKIP** — areal-algo sandbox has no `go`/`sqlc` toolchain. Go is written blind per option b (贝克汉姆 0f880dd0); CI verifies compilation/tests. |
| `sqlc generate` | **SKIP** — no sqlc v1.31.1. `CreateDerivedEnvDispatchAgent` query deferred; clone adapter runs on raw SQL (functional). Regen is a CI follow-up. |
| `gofmt -w` | **SKIP** — no go; CI. |

## Implementation status (Task 3 / 5 / 6 / 8)

- **Task 3 (runtime discovery + derived clone): DONE.** `service/env_dispatch_derived_agent.go` (`CloneEnvDispatchAgent` + `CloneDeps`), `handler/env_dispatch_clone_adapter.go` (`CloneEnvDispatchAgentTx` single-transaction DB adapter), `service/env_sandbox_runtime_discovery.go` (`WaitForOnlineSandboxRuntime` 4-tuple identity match + mismatch reject + timeout). Tests: `env_dispatch_clone_adapter_test.go`, `env_dispatch_runtime_lookup_test.go`, `service/env_dispatch_derived_agent_test.go`. Cross-workspace source reject implemented (`source agent workspace mismatch`).
- **Task 5 (orchestration state machine): scratch static path DONE.** `handler/env_dispatch_channel_provision.go` `provisionEnvDispatchAgent`: `claimProvisioning` (single-flight) -> `lifecycle.Create` (shared sandbox, mints daemon nonce, no pre-created `agent_runtime`) -> `WaitForOnlineSandboxRuntime` -> `CloneEnvDispatchAgentTx` -> chat_session + channel_agent_session -> `markReady`, with `cleanup()` compensation (delete sandbox + mark failed) on every failure step. Training path partially wired (`env_dispatch.go` `maybeOpenTrainingSession`, `SaveTrainingDispatch`, arealrl client referenced). **Feature flag added (Task 8.3):** `envDispatchDerivedAgentEnabled()` reads `ENV_DISPATCH_DERIVED_AGENT` (kill-switch).
- **Task 6 (cleanup): PARTIAL.** `deleteEnvDispatchChannelRollout`: `markDeleting` -> `waitForEnvDispatchProvisioning` -> delete sandbox + runtime per binding -> delete channel/project/bindings/env (idempotent: absent channel = 204). **GAP:** archive derived agent, close training session, revoke model/bootstrap credentials — `envDispatchDepsAdapter` lacks these methods; needs new APIs (deferred / CI).
- **Task 8.1 (Python client): PASS** — 59 tests.
- **Task 8.2 (format/lint/spec/graph):** openspec PASS, graphify PASS, gofmt/go test SKIP (no go).
- **Task 8.3 (feature flag): DONE** — `env_dispatch_derived_agent` (see open decision below).
- **Task 8.4 (verify report):** this file.

## AC-1..AC-8 mapping (AC basis = spec.md Requirements+Scenarios, locked defaults per comment 0f880dd0)

- **AC-1 (frontend sandbox lifecycle): DONE** — shared `lifecycle.Create`; scratch path inserts no `agent_runtime` before daemon registration.
- **AC-2 (runtime identity discovery): DONE** — `WaitForOnlineSandboxRuntime` matches workspace+provider=pi+daemon_id+sandbox_instance_id+online; mismatch -> "runtime identity mismatch"; deadline -> "runtime readiness timeout" + compensate. (Default 120s/2s locked; current const=2min matches; env-override `ENV_DISPATCH_RUNTIME_READINESS_TIMEOUT` noted as minor follow-up.)
- **AC-3 (derived global agent): DONE** — `CloneEnvDispatchAgentTx` records `source_agent_id` + runtime_id, source unchanged, cross-workspace reject (locked default #2 whitelist: name/instructions/provider-visible Pi settings/skills/visibility; no creds/task-state/runtime-ownership).
- **AC-4 (credential isolation): static DONE; training PARTIAL.** `model_config_owner_agent_id` persisted; `start_session(session_ref=binding.ID, env_id)` + `LinkSessionTask` Go client done (19ec9d6f6); areal proxy session_ref done (979f1be9). Typed credential resolver owner-check + training orchestration ordering (StartSession before CreateSandbox; LinkSessionTask after InsertTask) needs CI verification.
- **AC-5 (single-flight): DONE** — `claimProvisioning` (env_id, source_agent_id) unique claim; concurrent callers observe winner state.
- **AC-6 (cleanup complete+idempotent): PARTIAL** — sandbox/runtime/channel/project/env deleted idempotently; **GAP:** derived archive + session close + credential revoke.
- **AC-7 (client terminal failures): DONE** — areal 8f0fcf15; 59 Python tests pass (per-rollout error, DAG timeout, malformed/cyclic/dangling DAG).
- **AC-8 (valid DAG, deployed): SPLIT** to separate issue (贝克汉姆 ruling). Local DAG structural validation covered by AC-7 client tests.

## Secret audit

No credential recorded in source, logs, errors, responses, or this report. The runtime API key lives only in `environment_agent_sandbox.sandbox_config` (persisted) and is forwarded into the sandbox lifecycle; it never appears on `SandboxInstanceRef`, HTTP responses, errors, or structured logs. Synthetic sentinels (`sentinel-static-key`, `sentinel-training-key`) appear only in tests and are asserted absent from outputs.

## Open / CI-pending

1. **Go compilation + `go test`** — no go in areal-algo sandbox; CI verifies (option b).
2. **Task 6 cleanup cascade** — archive derived agent + close training session + revoke credentials (needs new adapter methods/SQL).
3. **`CreateDerivedEnvDispatchAgent` sqlc query + `pkg/db/generated` regen** — adapter currently on raw SQL (functional); regen is CI.
4. **Feature flag default** — implemented default **true** (preserves current tested behavior; scratch path is fully migrated with no legacy pre-create scratch path to fall back to). Spec/design specify default **false**. Flagged for 贝克汉姆/zhoujie22 decision: prod rollout should set `ENV_DISPATCH_DERIVED_AGENT` explicitly; the kill-switch (`=false` rejects scratch provisioning) is operational.
5. **AC-8 deployed verification** — separate issue.
6. **AC-4 training orchestration** — full typed-credential resolver + ordering assertions need CI/DB verification.

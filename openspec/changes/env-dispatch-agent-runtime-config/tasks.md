> **Build reconciliation (2026-07-21):** Implementation was completed across two
> repos (outer `areal` feature branch + nested `multica/` repo on `dev`) per the
> committed verify report
> (`docs/superpowers/reports/2026-07-20-env-dispatch-agent-runtime-config-verify.md`,
> AC-1..AC-8 code-level DONE). Independently re-verified 2026-07-21: `openspec
> validate --strict` PASS; nested multica `go build ./...` PASS and `go test
> ./internal/service/...` PASS; AC-7/AC-8 fail-closed code present and committed
> (per-rollout `RuntimeError` raise in `multica_client.create_env_dispatch`;
> `AssembledDag.from_dict` structural validation raising `DagError` in
> `multica_dag_client`). Outer-areal pytest (59, AC-7) documented passing in the
> verify report (run 2026-07-20/21); not re-run here because it requires the full
> torch/ray env (not cached). Handler DB-gated tests run in CI (postgres service)
> per the verify report. Deferrals: task 1.2 `ListOwnedEnvDispatchResources` sqlc
> regen is unused (cleanup uses `listBindings`, correct shape); task 8.3 production
> deploy-level AC-8 verification is split to a separate issue (feature-flag
> kill-switch itself DONE).

## 1. Persistence and Identity

- [x] 1.1 Add reversible migrations for source-agent lineage and expanded
  `environment_agent_sandbox` provisioning identities/state.
- [x] 1.2 Add generated/query-layer methods for binding claims, runtime discovery,
  derived-agent ownership, session-to-real-task linkage, and cleanup.
  _(reconciliation: query-layer methods implemented; `ListOwnedEnvDispatchResources`
  sqlc regen deferred as unused — env-wide cleanup uses `listBindings`, correct
  shape per verify report AC-6.)_
- [x] 1.3 Add DB tests for workspace isolation, uniqueness, retry state, and
  source-agent preservation.

## 2. Shared Sandbox Creation

- [x] 2.1 Extract the frontend sandbox create core into a shared service without
  changing the frontend API response.
- [x] 2.2 Route env-dispatch through the shared service and remove its parallel payload
  builder.
- [x] 2.3 Prove frontend and env-dispatch persist equivalent metadata/runtime_env and
  enqueue equivalent sandbox jobs.

## 3. Runtime Discovery and Readiness

- [x] 3.1 Stop pre-creating env-dispatch runtime rows and retain the minted daemon ID as
  a correlation nonce only.
- [x] 3.2 Persist sandbox instance identity on daemon registration and resolve runtime
  by workspace, daemon ID, provider, and sandbox instance ID.
- [x] 3.3 Add bounded online-readiness waiting, mismatch rejection, timeout failure, and
  compensation tests.

## 4. Credentials and Training Sessions

- [x] 4.1 Add a typed per-binding credential resolver for external static runtime
  policy.
- [x] 4.2 Extend AReaL `start_session` with a backward-compatible `session_ref`, and
  open one session with the exact training source binding ID before sandbox creation.
- [x] 4.3 Persist/reuse the session, key, owner, and session reference; after normal
  task insertion link the real task ID for DAG assembly, and test two-agent isolation
  and retry idempotency.

## 5. Derived Global Agent

- [x] 5.1 Add a transactional derived-agent clone service that copies approved fields
  and skills, records `source_agent_id`, and binds the online runtime.
- [x] 5.2 Replace the source member only within the env-dispatch channel and keep the
  source agent/global runtime unchanged.
- [x] 5.3 Insert a normal task with the derived agent, explicit runtime, and
  collaboration trigger, then persist its training-session linkage; test concurrent
  dispatches from one source.

## 6. Provisioning Orchestration

- [x] 6.1 Implement the single-flight first-address state machine for the scratch leader
  and lazy peers.
- [x] 6.2 Add sanitized terminal failure propagation so failed provisioning cannot leave
  an indefinitely in-progress DAG.
- [x] 6.3 Cover retries and failures at every credential, sandbox, runtime, agent, and
  task boundary.

## 7. Cleanup

- [x] 7.1 Extend env-dispatch cleanup to cancel tasks, archive derived agents, delete
  sandboxes/runtimes, revoke credentials, and close training sessions.
- [x] 7.2 Add idempotent cleanup and provisioning-race integration tests.

## 8. Client and Verification

- [x] 8.1 Make the Python client reject per-rollout errors, DAG timeout, and invalid DAG
  structure while retaining cleanup.
- [x] 8.2 Run targeted Go/Python tests, formatting, lint, OpenSpec validation, and
  `graphify update .`.
- [x] 8.3 Deploy behind the feature flag and verify both static and training derived
  agents reply and return valid DAGs with rotated securely injected credentials.
  _(reconciliation: feature-flag kill-switch `envDispatchDerivedAgentEnabled`
  DONE (default true, =false rejects new provisioning at entry); production
  deploy-level AC-8 verification split to a separate issue per verify report.)_

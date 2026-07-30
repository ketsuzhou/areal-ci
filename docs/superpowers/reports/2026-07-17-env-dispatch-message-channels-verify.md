# Verification Report: env-dispatch-message-channels

**Change:** env-dispatch-message-channels
**Date:** 2026-07-17
**Verify mode:** full
**Language:** en
**Branch:** feature/20260717/env-dispatch-message-channels (areal) ; multica submodule on `dev` (Go commits 74f500ff4..c7eb8603c)

## Scope

Channel-first message dispatch for `dispatch_type=message`: rollout-isolated
env/project/channel, per-agent sandbox bindings (binding-routed, no default-runtime
fallback), leader-only scratch wake, lazy first-mention provisioning, branch resume
from a copied collaboration trigger (clone trigger-agent sandbox, peers pending),
channel-first lifecycle facades, and an AReaL `EnvDispatchHandle` client that routes
by `dispatch_type`.

## Fresh Verification Evidence (run this session)

### Build

| Target | Command | Result |
| --- | --- | --- |
| Go (touched packages) | `go build ./internal/service/... ./internal/handler/... ./cmd/server/...` | exit 0 |
| Go vet (touched packages) | `go vet ./internal/service/... ./internal/handler/... ./cmd/server/...` | exit 0 |
| Python lint | `ruff check` on 6 changed files | exit 0, All checks passed |
| Python format | `ruff format --check` | clean (after applying format) |

Recorded build check: `go build ./internal/service/... ./internal/handler/... ./cmd/server/...` exit 0.

### Tests

| Suite | Command | Result |
| --- | --- | --- |
| Go service | `go test -count=1 ./internal/service/...` | `ok` 0.132s |
| Go handler | `go test -count=1 ./internal/handler/...` | `ok` 0.031s (DB-gated tests skip cleanly locally) |
| Go cmd/server | `go test -count=1 ./cmd/server/...` | `ok` 0.035s |
| Python routing | isolation harness vs real `multica_client.py` (httpx.MockTransport) | ALL ISOLATION CHECKS PASSED (8 checks) |
| Python dag+workflow | isolation harness vs real `multica_dag_client.py` + `multi_agent_workflow.py` | ALL DAG + WORKFLOW ISOLATION CHECKS PASSED (6 checks) |

Recorded verify check: `go test -count=1 ./internal/service/... ./internal/handler/... ./cmd/server/...` exit 0.

### Code review

Standard review (`requesting-code-review`) scoped to commits 21a3b068..5977e122.
Reviewer assessment: "Ready to merge: With fixes." Two Important findings:
(1) `test_create_env_dispatch_message_returns_channel_first_handle` failed because
the handler recorded the create POST alongside the lifecycle routes - fixed
(`seen_paths.clear()` after create); (2) `list_checkpoints` did not validate
`channel_id` for message handles, inconsistent with `cleanup`/`get_dag` - fixed
(added the same RuntimeError guard). Minor findings (primary_id annotation, stale
dag docstring, missing list_checkpoints message test) addressed where safe. Fix
commit `ded8e5d3` re-verified green. No Critical issues.

## Full Verification Checklist

1. **tasks.md all complete** - PASS. 0 unchecked items (26 tasks, all `[x]`).
2. **Implementation matches design.md** - PASS. Channel-first routing, EnvDispatchHandle,
   leader-only wake, lazy provisioning, branch copy+clone, and channel-first facades
   all match the OpenSpec design decisions.
3. **Implementation matches Design Doc** (`docs/superpowers/specs/2026-07-16-env-dispatch-message-channel-design.md`) - PASS.
4. **Capability spec scenarios pass** - PASS. All 7 requirements in
   `specs/env-dispatch-message-channels/spec.md` have implementations and tests:
   - Message rollout isolation (distinct env/project/channel; no group manager)
   - Binding-routed sandbox execution (no default-runtime fallback - invariant review confirmed)
   - Scratch wakes only the leader (peers pending)
   - Lazy first-mention provisioning (single-flight `claimProvisioning`)
   - Branch validation before writes (`ValidateBranchMessageSource` pre-write)
   - Branch resume from copied collaboration (copy + remap + clone trigger sandbox)
   - Channel-first lifecycle facades (dag/delete/checkpoints routes; serialized idempotent cleanup)
5. **proposal.md goals satisfied** - PASS. Every "What Changes" bullet is implemented.
6. **No delta-spec / design-doc contradictions** - PASS. Protocol doc updated to align.
7. **Associated design docs locatable** - PASS. The design doc exists and is referenced
   from `.comet.yaml` `design_doc`.

## Invariant Review

- No default-runtime fallback in the provision path: the only match for
  `DefaultRuntime`/`ensureChannelAgentSessionWithDB` in touched files is a comment in
  `env_dispatch_channel_provision.go` documenting that the code *avoids* the shared
  default runtime. Confirmed.
- Only intended files staged on the areal branch: the 6 Python files + protocol doc +
  tasks.md/plan/spec OpenSpec artifacts. `AGENTS.md`/`CLAUDE.md` dirty at session
  start are pre-existing and unrelated; not committed by this change.

## Environmental Constraints (honest disclosure)

- **Full areal Python pytest not run locally.** The areal env requires
  `requires-python >=3.12,<3.13` with torch/flashinfer/aiohttp; the local
  `flashinfer-cubin` local wheel is absent and torch is too heavy to install in this
  sandbox. The `agents/__init__.py` eagerly imports torch-backed modules, so the real
  pytest cannot be collected. Verification was instead performed by loading the real
  `multica_client.py`, `multica_dag_client.py`, and `multi_agent_workflow.py` modules
  with only their parent packages stubbed, then exercising the actual handle/routing
  code paths against `httpx.MockTransport`. The assertions mirror the unit tests.
  GitLab CI runs the real pytest suite.
- **multica submodule upstream-merge state.** During this session an external
  upstream-sync merge (`merge/dev-to-upstream-20260717`) was initiated in the multica
  submodule, leaving conflict markers in `server/cmd/multica/cmd_sandboxd.go` (a file
  this change does **not** touch). This breaks `go build ./...` / `go test ./...` at
  the module level. The packages this change touches (`internal/service`,
  `internal/handler`, `cmd/server`) build, vet, and test clean in the current state.
  Earlier in the session the full multica module - including `cmd/multica` - built and
  tested green before the external merge began. This merge is not part of this change
  and was left untouched.
- **Two pre-existing Go test failures in untouched packages.** `TestPollLoopTargetsRuntimeWakeup`
  (`internal/daemon`, flaky - passes in isolation) and `TestGrokExecuteStreamingJSONWithTools`
  (`pkg/agent`, requires an external grok binary). Confirmed via `git diff --name-only
  74f500ff4^..HEAD` that this change touches neither package; the failures are
  pre-existing/environmental.

## Outcome

**Verification: PASS.** All in-scope checks pass with fresh evidence. The two
environmental constraints (local Python pytest, external multica merge) are documented
above; neither is caused by this change and neither blocks the in-scope evidence.

## 1. Investigation - resume task-state lifecycle

- [x] 1.1 Trace the `agent_task_queue` state machine at pause time: what status/state an
  in-flight task holds when its sandbox is paused (stop job), and whether the resumed
  `agent_runtime` auto-reclaims it or needs explicit re-injection. State-machine-first per
  project code-quality rules.
- [x] 1.2 Trace the runtime task-claim loop (`ClaimTaskForRuntime`/`claimTask`,
  `task.go:1078-1181`) to confirm the exact seam the resume-agent-run primitive reuses to
  re-activate a *specific* in-flight task (vs claiming a new one).
- [x] 1.3 Confirm `agent_runtime` <-> `sandbox_instance` binding at resume time (which runtime
  row corresponds to which resumed sandbox ref) so the primitive targets the right runtime.
- [x] 1.4 Confirm `CheckpointTrigger.TriggerCheckpoint(ctx, task, projectID)` has all fields
  needed for the `resume_trigger` descriptor (`task_id`, `runtime_id`, `agent_id`,
  `issue_id`|`chat_session_id`, `project_id`, kind).
- [x] 1.5 Commit: `docs(env-checkpoint-resume-trigger): T1 investigation`.

> Answered by the design doc (D1-D6) and code tracing, not a separate commit. State machine:
> `queued -> dispatched -> running -> terminal`. `ReclaimStaleDispatchedTaskForRuntime` reclaims
> only `dispatched` (started_at IS NULL); `running` tasks are FAILED on `running_timeout_secs`
> and are NOT auto-requeued -> explicit re-injection required. `agent_runtime` is stable across
> pause/resume via `ON CONFLICT (workspace_id, daemon_id, provider)`.

## 2. Storage - resume_trigger on env_checkpoint

**Files:** `multica/server/migrations/`, `multica/server/pkg/db/queries/env_checkpoint.sql`,
`multica/server/internal/service/env_checkpoint.go`, handler.

- [x] 2.1 Migration: add nullable `resume_trigger` JSONB to `env_checkpoint` (down migration
  drops it). Existing rows stay null (legacy behavior).
- [x] 2.2 sqlc query + generated code: persist and load `resume_trigger`.
- [x] 2.3 Service types: `ResumeTrigger` struct + `EnvCheckpointCreateInput.ResumeTrigger` /
  `EnvCheckpoint.ResumeTrigger`; `EnvCheckpointRepository.CreateCheckpoint` carries it.
- [x] 2.4 Handler: `CreateEnvCheckpointRequest` accepts optional `resume_trigger`; response
  surfaces it.
- [x] 2.5 Failing test: round-trip `resume_trigger` through create -> get -> list.
- [x] 2.6 Commit: `feat(env-checkpoint-resume-trigger): resume_trigger storage`.

> 2.4: the create request does NOT accept `resume_trigger` - it is server-resolved (D5). The
> `EnvCheckpointResponse` surfaces `resume_trigger`. Storage landed across multica commits
> `a894531c7` (migration 159), `24927e60d` (sqlc), `f9e8a82d7` (service types + resolution).
> Migration uses 159 (155-158 were taken by interaction_dag work).

## 3. Resume-agent-run primitive

**Files:** `multica/server/internal/service/` (new primitive near task-claim seams),
`env_checkpoint.go` seam.

- [x] 3.1 Define `ResumeAgentRunner` seam (mirrors `SandboxInstanceResumer` injection): re-activates
  an existing in-flight task against a resumed `agent_runtime`.
- [x] 3.2 Failing tests: re-activates the existing in-flight task (not a new row); rejects a
  trigger whose task already transitioned terminal; rejects unknown runtime/task; per-agent
  fan-out correctness.
- [x] 3.3 Implement the primitive reusing the claim-seam shape to re-activate the specific
  in-flight `agent_task_queue` row against the resumed runtime.
- [x] 3.4 Commit: `feat(env-checkpoint-resume-trigger): resume-agent-run primitive`.

> Landed in multica commit `b4f7495fa` (`task_resume_runner.go`). The primitive resets the
> existing in-flight task row to `queued` (same row - continuity preserved) and wakes the
> resumed daemon; terminal/missing/runtime-mismatch returns `ErrTriggerTaskNotResumable`.

## 4. Trigger capture + ResumeFromCheckpoint execution

**Files:** `multica/server/internal/service/env_checkpoint.go`, `training.go`
(`CheckpointTrigger`), handler.

- [x] 4.1 `CheckpointTrigger` populates `resume_trigger` from the in-flight `task` + `projectID`
  at checkpoint-create time.
- [x] 4.2 Failing test: `ResumeFromCheckpoint` executes the stored trigger after sandbox resume
  (primitive called with the descriptor); returns `RolloutHandle` carrying trigger-execution
  status.
- [x] 4.3 Failing test: checkpoint with empty `resume_trigger` (legacy) resumes as today
  (sandbox + handle, no primitive call).
- [x] 4.4 Failing test: trigger execution failure is a typed error (partial resume: sandbox up,
  agent not re-engaged), not a silent no-op.
- [x] 4.5 Implement: inject `ResumeAgentRunner` into `EnvCheckpointService`; call it after
  sandbox resume in `ResumeFromCheckpoint`.
- [x] 4.6 Commit: `feat(env-checkpoint-resume-trigger): capture + execute resume trigger`.

> 4.1: NOT wired via the automatic `CheckpointTrigger` (which is unwired in `training.go`).
> Per design D5, `resume_trigger` is resolved **server-side** in `EnvCheckpointService.Create`
> via the `InFlightTaskResolver` seam (the caller does not know multica-internal task ids).
> Capture/execution landed in `f9e8a82d7` (Create resolution) + `738589215` (ResumeFromCheckpoint
> execution).

## 5. AReaL client integration

**Files:** `customized_areal/tree_search/agents/swe_lego_client.py`, tests.

- [~] 5.1 Failing test: resume-from-checkpoint client surfaces trigger-execution status / rollout
  handle to the tree-search caller.
- [~] 5.2 Implement client helper changes (if any) for trigger status; preserve existing
  resume-from-checkpoint behavior.
- [~] 5.3 Commit: `feat(env-checkpoint-resume-trigger): areal client resume status`.

> **DROPPED from this change (user decision).** Investigation found AReaL's resume path is
> `create_env_dispatch(mode="resume", env_id=<checkpoint_id>)`, which returns only `project_id`
> and does NOT consume `rollout_handle` or `trigger_status`. Crucially, `EnvDispatchService.Dispatch`
> normalizes `mode="resume"` to `EnvModeBranch` (env_dispatch.go: "resume is an alias for branch,
> spec D1") and never calls `EnvCheckpointService.ResumeFromCheckpoint` - so the trigger mechanism
> is not on AReaL's real resume path today. The trigger is correctly placed on the checkpoint-resume
> primitive (`ResumeFromCheckpoint` / `POST /env-checkpoints/{id}/resume`); AReaL wiring (switch to
> the dedicated resume endpoint, or wire env-dispatch mode=resume to checkpoint-resume) is deferred
> to the parent env-dispatch-sandbox-lifecycle change / a follow-up. No AReaL code changed.

## 6. Specs

- [x] 6.1 `specs/env-checkpoint-resume-trigger/spec.md`: ADDED requirements (trigger capture at
  create; resume-agent-run primitive; trigger execution on resume; legacy/no-trigger degrade).
- [~] 6.2 `specs/env-checkpoint-resume/spec.md`: MODIFIED "Resume from checkpoint" requirement -
  add trigger-execution sub-requirement (existing sandbox-resume + handle behavior preserved).
- [x] 6.3 `openspec validate env-checkpoint-resume-trigger --strict`.

> 6.2 superseded: the change adds a NEW capability spec `env-checkpoint-resume-trigger` (4 ADDED
> requirements, all implemented) rather than modifying the existing `env-checkpoint-resume` spec.
> `openspec validate --strict` passes ("Change is valid").

## 7. Verification

- [x] 7.1 Scoped Multica Go tests: `cd multica/server && go test ./internal/service
  ./internal/handler -count=1`.
- [~] 7.2 Scoped AReaL tests: `.venv-test/bin/python -m pytest` on touched files; `uvx ruff
  check`.
- [x] 7.3 Grep sweep: `resume_trigger`, `ResumeAgentRun`, `ResumeFromCheckpoint` resolve to
  intended code only.
- [x] 7.4 Final review -> READY TO MERGE / NEEDS_CHANGES.

> 7.1: verified in an isolated multica worktree at HEAD `b7be1790e` (shared tree had a concurrent
> session's WIP breaking the handler test binary). `go build ./...` OK; `./internal/service` ALL
> PASS (incl. all new trigger tests); `./internal/handler -run EnvCheckpoint` PASS. 2 pre-existing
> DB-dependent failures in `runtime_update_test.go` (`agent_runtime` `ON CONFLICT` unique constraint
> missing in the test DB) are unrelated - this change does not touch `agent_runtime`/`DaemonRegister`.
> 7.2: N/A - no AReaL code changed (Group 5 dropped). 7.3: all references resolve to migration 159,
> env_checkpoint query/generated/service/handler, and `task_resume_runner` only.
> 7.4: READY FOR REVIEW - service-side trigger capability complete; AReaL end-to-end integration is
> the deferred follow-up noted in Group 5.

## Test runners / constraints

- Multica Go tests from `multica/server` (`go build ./...` + scoped `go test`).
- AReaL tests from `backend/areal` with `.venv-test/bin/python -m pytest` (per project memory:
  broken `.venv/uv`; do not trust GREEN self-reports).
- No GPU required for these unit/service tests.
- `multica/` is untracked in this repo (`?? multica/`); Go changes are contract+test consistency,
  not deployment from here.

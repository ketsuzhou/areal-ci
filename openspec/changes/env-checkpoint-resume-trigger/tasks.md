## 1. Investigation - resume task-state lifecycle

- [ ] 1.1 Trace the `agent_task_queue` state machine at pause time: what status/state an
  in-flight task holds when its sandbox is paused (stop job), and whether the resumed
  `agent_runtime` auto-reclaims it or needs explicit re-injection. State-machine-first per
  project code-quality rules.
- [ ] 1.2 Trace the runtime task-claim loop (`ClaimTaskForRuntime`/`claimTask`,
  `task.go:1078-1181`) to confirm the exact seam the resume-agent-run primitive reuses to
  re-activate a *specific* in-flight task (vs claiming a new one).
- [ ] 1.3 Confirm `agent_runtime` <-> `sandbox_instance` binding at resume time (which runtime
  row corresponds to which resumed sandbox ref) so the primitive targets the right runtime.
- [ ] 1.4 Confirm `CheckpointTrigger.TriggerCheckpoint(ctx, task, projectID)` has all fields
  needed for the `resume_trigger` descriptor (`task_id`, `runtime_id`, `agent_id`,
  `issue_id`|`chat_session_id`, `project_id`, kind).
- [ ] 1.5 Commit: `docs(env-checkpoint-resume-trigger): T1 investigation`.

## 2. Storage - resume_trigger on env_checkpoint

**Files:** `multica/server/migrations/`, `multica/server/pkg/db/queries/env_checkpoint.sql`,
`multica/server/internal/service/env_checkpoint.go`, handler.

- [ ] 2.1 Migration: add nullable `resume_trigger` JSONB to `env_checkpoint` (down migration
  drops it). Existing rows stay null (legacy behavior).
- [ ] 2.2 sqlc query + generated code: persist and load `resume_trigger`.
- [ ] 2.3 Service types: `ResumeTrigger` struct + `EnvCheckpointCreateInput.ResumeTrigger` /
  `EnvCheckpoint.ResumeTrigger`; `EnvCheckpointRepository.CreateCheckpoint` carries it.
- [ ] 2.4 Handler: `CreateEnvCheckpointRequest` accepts optional `resume_trigger`; response
  surfaces it.
- [ ] 2.5 Failing test: round-trip `resume_trigger` through create -> get -> list.
- [ ] 2.6 Commit: `feat(env-checkpoint-resume-trigger): resume_trigger storage`.

## 3. Resume-agent-run primitive

**Files:** `multica/server/internal/service/` (new primitive near task-claim seams),
`env_checkpoint.go` seam.

- [ ] 3.1 Define `ResumeAgentRunner` seam (mirrors `SandboxInstanceResumer` injection): re-activates
  an existing in-flight task against a resumed `agent_runtime`.
- [ ] 3.2 Failing tests: re-activates the existing in-flight task (not a new row); rejects a
  trigger whose task already transitioned terminal; rejects unknown runtime/task; per-agent
  fan-out correctness.
- [ ] 3.3 Implement the primitive reusing the claim-seam shape to re-activate the specific
  in-flight `agent_task_queue` row against the resumed runtime.
- [ ] 3.4 Commit: `feat(env-checkpoint-resume-trigger): resume-agent-run primitive`.

## 4. Trigger capture + ResumeFromCheckpoint execution

**Files:** `multica/server/internal/service/env_checkpoint.go`, `training.go`
(`CheckpointTrigger`), handler.

- [ ] 4.1 `CheckpointTrigger` populates `resume_trigger` from the in-flight `task` + `projectID`
  at checkpoint-create time.
- [ ] 4.2 Failing test: `ResumeFromCheckpoint` executes the stored trigger after sandbox resume
  (primitive called with the descriptor); returns `RolloutHandle` carrying trigger-execution
  status.
- [ ] 4.3 Failing test: checkpoint with empty `resume_trigger` (legacy) resumes as today
  (sandbox + handle, no primitive call).
- [ ] 4.4 Failing test: trigger execution failure is a typed error (partial resume: sandbox up,
  agent not re-engaged), not a silent no-op.
- [ ] 4.5 Implement: inject `ResumeAgentRunner` into `EnvCheckpointService`; call it after
  sandbox resume in `ResumeFromCheckpoint`.
- [ ] 4.6 Commit: `feat(env-checkpoint-resume-trigger): capture + execute resume trigger`.

## 5. AReaL client integration

**Files:** `customized_areal/tree_search/agents/swe_lego_client.py`, tests.

- [ ] 5.1 Failing test: resume-from-checkpoint client surfaces trigger-execution status / rollout
  handle to the tree-search caller.
- [ ] 5.2 Implement client helper changes (if any) for trigger status; preserve existing
  resume-from-checkpoint behavior.
- [ ] 5.3 Commit: `feat(env-checkpoint-resume-trigger): areal client resume status`.

## 6. Specs

- [ ] 6.1 `specs/env-checkpoint-resume-trigger/spec.md`: ADDED requirements (trigger capture at
  create; resume-agent-run primitive; trigger execution on resume; legacy/no-trigger degrade).
- [ ] 6.2 `specs/env-checkpoint-resume/spec.md`: MODIFIED "Resume from checkpoint" requirement -
  add trigger-execution sub-requirement (existing sandbox-resume + handle behavior preserved).
- [ ] 6.3 `openspec validate env-checkpoint-resume-trigger --strict`.

## 7. Verification

- [ ] 7.1 Scoped Multica Go tests: `cd multica/server && go test ./internal/service
  ./internal/handler -count=1`.
- [ ] 7.2 Scoped AReaL tests: `.venv-test/bin/python -m pytest` on touched files; `uvx ruff
  check`.
- [ ] 7.3 Grep sweep: `resume_trigger`, `ResumeAgentRun`, `ResumeFromCheckpoint` resolve to
  intended code only.
- [ ] 7.4 Final review -> READY TO MERGE / NEEDS_CHANGES.

## Test runners / constraints

- Multica Go tests from `multica/server` (`go build ./...` + scoped `go test`).
- AReaL tests from `backend/areal` with `.venv-test/bin/python -m pytest` (per project memory:
  broken `.venv/uv`; do not trust GREEN self-reports).
- No GPU required for these unit/service tests.
- `multica/` is untracked in this repo (`?? multica/`); Go changes are contract+test consistency,
  not deployment from here.

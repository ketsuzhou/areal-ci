# Subagent Progress

- Change: env-dispatch-sandbox-lifecycle
- Mode: subagent-driven-development
- TDD mode: tdd (user-selected; Comet state script lacks tdd_mode field)

## Current Task

- Plan task: Task 1: Confirm Existing Seams and Record Findings
- Mapped OpenSpec tasks: 1.1 Confirm sandbox lifecycle commits; 1.2 Trace env-dispatch state flow; 1.3 Document legacy raw sandbox ids to structured refs transition; 1.4 Confirm DB subtree snapshot query scope.
- Stage: implementing
- Brief: .superpowers/sdd/task-1-brief.md
- Report: .superpowers/sdd/task-1-report.md
- Review-fix round: 0
- Implementation commit: pending
- RED evidence: pending (documentation/seam-confirmation task; implementer must still use TDD skill and explain any documentation-only exception if no executable failing test is applicable)
- GREEN evidence: pending

## Dispatch Event

- Stage: implementing
- Task 1 implementer run `125dec1a-d875-4ae6-a043-f3ac24ebdc67` failed before writing a result; no child session persisted.
- Action: re-dispatch Task 1 with narrower context.

## Dispatch Event

- Stage: blocked
- Task 1 implementer run `5be1e213-39cd-4145-a0c8-d824f3c3f2f4` also failed before writing a result; no child session persisted.
- Blocker: current background subagent runner is failing immediately for implementer dispatches, so `subagent-driven-development` cannot proceed safely.
- Required user decision: switch build mode to `executing-plans` for inline execution, or debug/fix subagent runtime and then resume subagent-driven execution.

## Mode Switch

- User selected: switch to `executing-plans` and continue inline with TDD.
- Comet state updated: `build_mode=executing-plans`.

## Task 3 Blocker (architectural gap)

- Stage: blocked
- Finding: env-dispatch rollouts today create sandboxes via the cloud-runtime/Fleet proxy (`ForkSandbox`/`BootSandbox` -> `/api/v1/sandboxes/...`), producing opaque `sandbox_id` strings stored in `environment.sandbox_ids`.
- The Task 2 lifecycle service (and the checkpoint save/resume design D2/D3) operates on `sandbox_instance` rows + sandboxd `stop`/`resume` jobs from the sandbox node gateway — a DIFFERENT sandbox system that Fleet sandboxes are not part of.
- Consequence: Task 3's "persist/return structured sandbox-instance refs for rollout environments" cannot be truthfully implemented without first deciding how rollout sandboxes become `sandbox_instance`-backed. Populating `SandboxRefs` from Fleet sandbox ids would be meaningless for pause/resume.
- Note: Fleet also exposes snapshot/fork routes (`SnapshotCloudRuntimeSandbox`, `ForkCloudRuntimeSandbox`), but the confirmed design (D2) explicitly chose sandbox_instance pause-in-place over Fleet snapshots/forks.
- Required user decision: see options presented in chat.

## Build Progress (executing-plans + TDD, inline)

- Task 1 (investigation): DONE — commit afa8b95c (outer repo). Seam notes in design.md.
- Task 2 (lifecycle service): DONE — multica commit 152538cbc. Save/Resume/Delete/Reconfigure + typed errors.
- Task 2b (lifecycle Create): DONE — multica commit da0ad0b50. Create inserts pending row, enqueues create job, wakes node.
- Task 3 slice A (per-agent types + shape validation + ref fields): DONE — multica commit a4f821dd2.
- Task 3 slice B (bridge seam): DONE — multica commit dbe5934ae. WithSandboxLifecycle; trained rollouts create sandbox_instances; non-trained preserves Fleet.
- Bridge design pivot: confirmed by user (option 1 bridge + option 2 branch-from-template); design doc + delta spec + tasks + plan updated — outer commit 3cb250f8.

## Remaining
- Task 3: handler per_agent_env JSON parsing/response; production adapter wiring (node selection via PickAvailableSandboxNodeForWorkspace, template resolution, branch-from-source-template); DB membership validation (unknown agent/env spec); trained task/session context ref preservation (3.6).
- Tasks 4-9: checkpoint schema/queries/service, checkpoint APIs, resume-from-checkpoint, entropy/event triggers, AReaL client integration, verification/docs.

## Verification evidence
- multica: `go test ./internal/service -count=1` -> ok; `go build ./...` -> BUILD_OK; `go test ./internal/handler -run TestEnvDispatch` -> ok.
- openspec: `openspec validate env-dispatch-sandbox-lifecycle --strict` -> valid.

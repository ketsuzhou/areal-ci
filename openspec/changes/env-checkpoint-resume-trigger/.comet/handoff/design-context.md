# Comet Design Handoff

- Change: env-checkpoint-resume-trigger
- Phase: design
- Mode: compact
- Context hash: d2619b9ec16f6183d1d6310fecd46265540c03ce8c937cd2aa041b1cb493c019

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-checkpoint-resume-trigger/proposal.md

- Source: openspec/changes/env-checkpoint-resume-trigger/proposal.md
- Lines: 1-66
- SHA256: 7edd7571ba5013e9bf9e565bc95a0ca6e9e745267ce0b8e5520e3d648fa5fc62

```md
## Why

`env-dispatch-sandbox-lifecycle` (task-complete, at verify, not yet archived) shipped env
checkpoint save/resume, but its resume path has a gap that exists in **both code and spec**:
`ResumeFromCheckpoint` (`multica/server/internal/service/env_checkpoint.go:171`) only resumes
the sandbox container (sandboxd `resume` job -> Cube `/resume`, `runtime_restarted: true`) and
returns a `RolloutHandle` token. It never re-engages the **agent runtime**, so the resumed
sandbox has a fresh agent process that never picks up the paused in-flight task - the rollout
cannot actually continue from the checkpointed intermediate state. The `env-checkpoint-resume`
delta spec's "Resume from checkpoint" requirement has the same blind spot: it only specifies
sandbox resume + handle return, with no requirement that the agent runtime be re-engaged.

## What Changes

- **Capture a resume-trigger at checkpoint-create time.** Add a `resume_trigger` JSONB
  descriptor to `env_checkpoint` (new migration + service/handler types). It is populated where
  checkpoints are already created - `CheckpointTrigger.TriggerCheckpoint(ctx, task, projectID)`
  (`training.go:94`, fired from `maybeTriggerCheckpoint` at trained structural events) - which
  already has the full in-flight `task` (`task_id`, `runtime_id`, `agent_id`,
  `issue_id`|`chat_session_id`) + `project_id`. The descriptor names the in-flight
  agent_runtime/task to re-engage on resume.
- **Add a new multica "resume agent run" primitive** that re-activates the **existing** in-flight
  task (`agent_task_queue` row) against the resumed `agent_runtime` - **not** a fresh
  `TaskService.EnqueueTaskForIssue` (which would create a new task row and lose mid-task
  continuity), **not** a literal chat message. It reuses the existing claim-seam shape
  (`ClaimTaskForRuntime` / `claimTask`, `task.go:1078-1181`) but re-activates the specific
  in-flight task rather than claiming a new one.
- **Execute the trigger in `ResumeFromCheckpoint`** after the sandboxes are resumed, so the agent
  runtime actually continues the rollout from the checkpointed state. A checkpoint with no
  trigger (legacy/pre-change) degrades to today's behavior (sandbox resume + handle only).
- **AReaL client + multica service/handler tests** for trigger capture, primitive execution, and
  resume continuation.

## Capabilities

### New Capabilities

- `env-checkpoint-resume-trigger`: Checkpoint resume re-engages the agent runtime (a
  resume-agent-run primitive) so a resumed rollout continues its in-flight task from the
  checkpointed state, not just the sandbox container. Captures a resume-trigger descriptor at
  checkpoint-create time and executes it at resume time. This layers on top of the
  `env-checkpoint-resume` capability introduced (not yet archived) by
  `env-dispatch-sandbox-lifecycle`: that capability's "resume sandbox + return rollout handle"
  requirement stays valid; this change adds the trigger-capture + agent-runtime re-engagement
  requirements as a new capability rather than formally modifying the not-yet-archived
  `env-checkpoint-resume` delta (avoids an archive-ordering conflict).

### Modified Capabilities

<!-- None. env-checkpoint-resume is not yet in the archived base (only a delta in
     env-dispatch-sandbox-lifecycle), so this change adds the trigger behavior as a new
     capability instead of formally modifying it. -->

## Impact

- **multica (Go, primary)**: new migration adding `resume_trigger` JSONB to `env_checkpoint`;
  `env_checkpoint` service/handler types + sqlc query; new resume-agent-run primitive (service,
  near the `task.go` claim seams / `agent_runtime`); `ResumeFromCheckpoint` trigger execution;
  `CheckpointTrigger` populates the descriptor; multica service/handler tests.
- **AReaL (Python)**: `customized_areal/tree_search/agents/swe_lego_client.py` resume helper may
  surface trigger/resume status; AReaL tests.
- **Dependencies**: extends the resume path of `env-dispatch-sandbox-lifecycle` (at verify, not
  archived) and layers on `training-session-lifecycle`. Does **not** reopen the parent change.
- **Out of scope**: immutable sandbox fork/branch (still deferred); changing checkpoint save
  semantics (pause-in-place via sandboxd stop is unchanged); per-interaction reward; AReaL
  tree-search frontier selection policy.
```

## openspec/changes/env-checkpoint-resume-trigger/design.md

- Source: openspec/changes/env-checkpoint-resume-trigger/design.md
- Lines: 1-125
- SHA256: d0742091b179db025991cd1d1bd79705a9934c51e4789fa99f6d5ef71b837508

[TRUNCATED]

```md
## Context

`env-dispatch-sandbox-lifecycle` implemented env checkpoint save/resume. Save = pause-in-place
via sandboxd `stop` (Cube `/pause`). Resume = `ResumeFromCheckpoint`
(`env_checkpoint.go:171`): load checkpoint, require `save_status == complete`, resume each
sandbox instance (sandboxd `resume` -> Cube `/resume`, `runtime_restarted: true`), return
`RolloutHandle = "resume:<checkpointID>"`.

The gap: resuming the **container** does not re-engage the **agent runtime**. A cloud
`agent_runtime` (migration `004_agent_runtime_loop`) runs inside the sandbox; its in-flight
`agent_task_queue` task is bound by `runtime_id`. When the sandbox pauses, the runtime goes
offline mid-task; when it resumes, the runtime process restarts fresh and nothing tells it to
re-claim/continue the in-flight task. So the resumed rollout stalls - the sandbox is up, but the
agent is not working. The `env-checkpoint-resume` spec has the same blind spot (resume =
sandbox + handle only).

This change closes the gap by capturing, at checkpoint-create time, a **resume-trigger**
describing the in-flight agent_runtime/task, and executing it at resume time via a new
**resume-agent-run** primitive.

## Goals / Non-Goals

**Goals:**
- Capture a resume-trigger descriptor at checkpoint-create time (where `maybeTriggerCheckpoint`
  already has the in-flight `task`).
- Add a resume-agent-run primitive that re-activates the existing in-flight task against the
  resumed runtime (continuity preserved, not a new task).
- Execute the trigger in `ResumeFromCheckpoint` after sandbox resume.
- Degrade gracefully for pre-change checkpoints (no trigger -> today's behavior).

**Non-Goals:**
- Immutable sandbox fork/branch (deferred by parent change).
- Changing save/pause semantics.
- AReaL tree-search frontier/resume selection policy (AReaL already receives a `RolloutHandle`).
- Full deep design (this is the open-phase high-level design; comet-design produces the deep
  design with alternatives, sequence diagrams, and edge-case analysis).

## Data Flow

```
SAVE (checkpoint create)                              RESUME
─────────────────────                                 ─────────────────────
maybeTriggerCheckpoint(task, projectID)               ResumeFromCheckpoint(cp)
        │                                                     │
        ▼                                                     ▼
CheckpointTrigger.TriggerCheckpoint                         resume each sandbox
        │                                                   (sandboxd resume/Cube /resume)
        ▼                                                     │
EnvCheckpointService.Create                                 ▼
  - capture db_snapshot                                      execute resume_trigger
  - capture resume_trigger ◄── new                          │
      {task_id, runtime_id,                                  ▼
       agent_id, issue_id|chat_session_id,          ResumeAgentRun(trigger)
       project_id}                                    - re-activate EXISTING in-flight
  - save sandboxes (stop)                              agent_task_queue task against
  - persist row w/ resume_trigger                       resumed agent_runtime
        │                                              - (NOT EnqueueTaskForIssue; NOT
        ▼                                                a new task row; NOT a chat msg)
env_checkpoint row                                     │
  + resume_trigger JSONB ◄── new (migration)           ▼
                                                      agent runtime continues the
                                                      in-flight task from checkpoint
                                                      │
                                                      ▼
                                                      return RolloutHandle
                                                      (carry trigger-execution status)
```

## Preliminary Decisions (to be deepened in comet-design)

- **D1 - Trigger descriptor shape**: a JSONB `resume_trigger` on `env_checkpoint` carrying
  `{task_id, runtime_id, agent_id, issue_id|chat_session_id, project_id, kind}`. `kind` lets the
  primitive dispatch issue-task vs chat-task resume. Populated by `CheckpointTrigger` from the
  in-flight `task`.
- **D2 - New resume-agent-run primitive, not re-enqueue**: re-enqueue (`EnqueueTaskForIssue`)
  creates a new `agent_task_queue` row and loses mid-task continuity / the existing task's
  context/state. The primitive re-activates the **existing** in-flight task row against the
  resumed runtime (claim-seam shape, `ClaimTaskForRuntime`/`claimTask`), so the agent continues
  the same task. Confirmed with user over re-enqueue and literal-chat-message alternatives.
- **D3 - Primitive placement**: a new service method near the task-claim seams (`task.go`) /
```

Full source: openspec/changes/env-checkpoint-resume-trigger/design.md

## openspec/changes/env-checkpoint-resume-trigger/tasks.md

- Source: openspec/changes/env-checkpoint-resume-trigger/tasks.md
- Lines: 1-99
- SHA256: d216700367a2419c85231c00578788ba41706d7dd3344ddee7a9e72f040678c0

[TRUNCATED]

```md
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
```

Full source: openspec/changes/env-checkpoint-resume-trigger/tasks.md

## openspec/changes/env-checkpoint-resume-trigger/specs/env-checkpoint-resume-trigger/spec.md

- Source: openspec/changes/env-checkpoint-resume-trigger/specs/env-checkpoint-resume-trigger/spec.md
- Lines: 1-106
- SHA256: f6d2e6293cc18a83f2a1732539c71f0d298b0c028c6e43a155fe7739708985af

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Resume-trigger captured at checkpoint create

The system SHALL capture a resume-trigger descriptor at checkpoint-create time that names the
in-flight agent runtime and task to re-engage on resume. The descriptor SHALL be persisted on
the `env_checkpoint` row as a `resume_trigger` JSONB field and SHALL identify the in-flight
`agent_task_queue` task (`task_id`), the bound `agent_runtime` (`runtime_id`), the `agent_id`,
the owning `issue_id` or `chat_session_id`, the `project_id`, and a `kind` distinguishing
issue-task resume from chat-task resume.

The descriptor SHALL be resolved server-side at checkpoint-create time from the project's
in-flight `agent_task_queue` task (status `running` or `dispatched`), so the caller is not
required to supply multica-internal task or runtime identifiers. If no in-flight task exists for
the project at create time, the descriptor SHALL be empty and resume degrades to sandbox-only.

#### Scenario: Checkpoint records resume-trigger from in-flight task

- **WHEN** a trained rollout creates a checkpoint at a structural event with an in-flight task
- **THEN** the checkpoint row stores a `resume_trigger` descriptor carrying the in-flight
  `task_id`, `runtime_id`, `agent_id`, `issue_id` or `chat_session_id`, `project_id`, and `kind`

#### Scenario: Trigger descriptor resolved server-side

- **WHEN** a checkpoint is created for a project that has an in-flight (running or dispatched)
  task
- **THEN** the system resolves the in-flight task server-side and stores a `resume_trigger`
  descriptor carrying its `task_id`, `runtime_id`, `agent_id`, `issue_id` or `chat_session_id`,
  `project_id`, and `kind`, without requiring the caller to supply those identifiers

#### Scenario: No in-flight task yields empty trigger

- **WHEN** a checkpoint is created for a project that has no in-flight (running or dispatched)
  task at create time
- **THEN** the checkpoint stores an empty `resume_trigger` and a later resume re-engages no task
  (sandbox-only resume)

#### Scenario: Resume-trigger round-trips through storage

- **WHEN** a checkpoint with a `resume_trigger` is created and later retrieved via get or list
- **THEN** the returned checkpoint carries the same `resume_trigger` descriptor

### Requirement: Resume-agent-run primitive re-engages in-flight task

The system SHALL provide a resume-agent-run primitive that re-activates the **existing** in-flight
`agent_task_queue` task against the resumed `agent_runtime`, so the agent continues the same
task from the checkpointed intermediate state. The primitive MUST NOT create a new task row
(unlike `EnqueueTaskForIssue`) and MUST NOT be implemented as a literal chat message.

The primitive SHALL validate that the referenced task is still resumable (not terminal) and that
the referenced runtime corresponds to a resumed sandbox; a task that has transitioned terminal
between checkpoint create and resume SHALL be rejected with a typed error rather than
double-running.

#### Scenario: Primitive re-activates existing in-flight task

- **WHEN** the resume-agent-run primitive is invoked with a valid resume-trigger after the
  sandbox resumed
- **THEN** the existing in-flight task is re-activated against the resumed runtime and no new
  task row is created

#### Scenario: Terminal task is not double-run

- **WHEN** the primitive is invoked for a task that has already transitioned terminal between
  checkpoint create and resume
- **THEN** the primitive rejects the trigger with a typed error and does not re-run the task

#### Scenario: Unknown runtime or task rejected

- **WHEN** the primitive is invoked with a trigger referencing an unknown runtime or task
- **THEN** the primitive returns a typed error and performs no re-activation

### Requirement: Resume-from-checkpoint executes trigger after sandbox resume

`ResumeFromCheckpoint` SHALL execute the stored resume-trigger via the resume-agent-run primitive
after the checkpoint's sandbox instances have been resumed, so the agent runtime actually
continues the rollout from the checkpointed state. The result returned to the caller SHALL
report whether trigger execution succeeded.

A trigger-execution failure (sandbox resumed but agent not re-engaged) SHALL be surfaced as a
```

Full source: openspec/changes/env-checkpoint-resume-trigger/specs/env-checkpoint-resume-trigger/spec.md


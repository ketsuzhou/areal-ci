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
  `agent_runtime`, injected into `EnvCheckpointService` as a `ResumeAgentRunner` seam (mirrors
  the existing `SandboxInstanceResumer` injection pattern), so checkpointing stays decoupled
  from task-service internals.
- **D4 - Trigger execution timing**: after all sandbox resume jobs are enqueued/successful,
  before returning the `RolloutHandle`. A trigger-execution failure is a typed error (resume
  partially succeeded: sandbox up, agent not re-engaged) surfaced in the result, not a silent
  no-op.
- **D5 - Legacy/no-trigger checkpoints**: a checkpoint with empty `resume_trigger` resumes as
  today (sandbox + handle). This preserves backward compatibility with checkpoints created
  before this change and with non-trained rollouts.
- **D6 - What "re-activate the in-flight task" means concretely**: the in-flight task's
  queue/lease state at pause time, and whether the resumed runtime re-claims it automatically or
  needs an explicit re-injection, is the central design question for comet-design (requires
  tracing the runtime task-claim loop and pause-time task state).

## Risks / Trade-offs

- **Trigger descriptor stale by resume time**: the in-flight task may have transitioned (e.g.,
  critic spawned, session closed) between checkpoint create and resume. Mitigation: primitive
  validates the task is still resumable; typed error if not.
- **Sandbox resumed but agent re-engagement fails**: partial resume. Mitigation: D4 surfaces it
  as a typed status, not silent.
- **Re-activating vs double-running a task**: risk of the task running twice (once pre-pause,
  once post-resume) if pause-time state is not correctly marked. Mitigation: comet-design must
  define the pause/resume task-state transition precisely (state-machine-first, per project
  code-quality rules).
- **Coupling checkpointing to task service**: mitigated by the `ResumeAgentRunner` seam (D3).

## Migration Plan

1. Migration adds `resume_trigger` JSONB (nullable) to `env_checkpoint`; existing rows get null
   (D5 legacy behavior).
2. `CheckpointTrigger` populates the descriptor; `EnvCheckpointService.Create` persists it.
3. Resume-agent-run primitive + `ResumeFromCheckpoint` execution, behind the existing
   `ENV_CHECKPOINTS_ENABLED` gate.
4. AReaL client + tests.
5. Rollback: disable trigger execution by config; resume falls back to sandbox + handle (D5).

## Open Questions (for comet-design)

- Exact pause-time and resume-time `agent_task_queue` state transitions (D6).
- Whether the resumed runtime auto-reclaims its in-flight task (making the primitive a
  no-op/confirmation) or requires explicit re-injection.
- Whether `RolloutHandle` should encode trigger-execution status for AReaL.
- Per-agent (squad) trigger fan-out: one trigger per sandbox ref, or one aggregate trigger?

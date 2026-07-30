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

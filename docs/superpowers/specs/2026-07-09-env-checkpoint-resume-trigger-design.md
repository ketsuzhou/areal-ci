---
comet_change: env-checkpoint-resume-trigger
role: technical-design
canonical_spec: openspec
---

# Design: env-checkpoint-resume-trigger

OpenSpec canonical spec: `openspec/changes/env-checkpoint-resume-trigger/specs/env-checkpoint-resume-trigger/spec.md`.
This document is the deep technical design; requirements/scenarios live in the OpenSpec delta.

## Problem

`env-dispatch-sandbox-lifecycle` shipped env checkpoint save/resume. `ResumeFromCheckpoint`
(`multica/server/internal/service/env_checkpoint.go:171`) resumes the **sandbox container**
(sandboxd `resume` job -> Cube `/resume`, `runtime_restarted: true`) and returns a
`RolloutHandle = "resume:<checkpointID>"`. It never re-engages the **agent runtime**, so the
resumed sandbox's fresh agent process never picks up the paused in-flight task - the rollout
cannot continue from the checkpointed intermediate state.

## Investigation findings (code-grounded)

### Task state machine

`agent_task_queue.status`: `queued -> dispatched -> running -> terminal`
(`completed`/`failed`/`cancelled`). Transitions (from `pkg/db/queries/agent.sql`):
- `ClaimAgentTask` / `ClaimAgentChatTask`: `queued -> dispatched` (sets `dispatched_at`).
- `StartAgentTask`: `dispatched|waiting_local_directory -> running` (sets `started_at`).
- Terminal: `completed`/`failed`/`cancelled` (sets `completed_at`).

### Central question: does the resumed runtime auto-reclaim a running task? NO.

`ClaimTaskForRuntime` (`task.go:1181`, called by the daemon at `handler/daemon.go:1167`) first
runs `ReclaimStaleDispatchedTaskForRuntime`, which reclaims only **dispatched** tasks
(`status = 'dispatched' AND started_at IS NULL AND dispatched_at < now() - claim_recovery_secs`).
A **running** task is not reclaimed. The only running-task expiry is
`status = 'running' AND started_at < now() - running_timeout_secs`, which **fails** the task
(terminal) - it does not requeue it. **Conclusion: explicit re-injection is required.**

### Enablers confirmed

- **`runtime_id` is stable across pause/resume**: `agent_runtime` upsert is
  `ON CONFLICT (workspace_id, daemon_id, provider) DO UPDATE` (`runtime.sql:57`), so the daemon
  re-registers with the same `daemon_id` and reuses the same `agent_runtime.id`. The task's
  `runtime_id` binding stays valid.
- **Daemon wake notifier exists**: `TaskWakeupNotifier.NotifyTaskAvailable(runtimeID, taskID)`
  (`task.go:60`), called by `notifyTaskAvailable` on enqueue (`task.go:2255`). The primitive
  reuses it to wake the resumed daemon immediately.
- **Claim respects `runtime_id`**: `ListQueuedClaimCandidatesByRuntime` filters candidates by
  `runtime_id`, so a reset-to-`queued` task is only claimable by its bound (resumed) runtime.

### CheckpointTrigger is unwired (shapes the capture design)

`CheckpointTrigger` (`training.go:94`) is an interface; `maybeTriggerCheckpoint`
(`training.go:390`) calls `deps.CheckpointTrigger.TriggerCheckpoint(...)` but
`NewTrainingSessionDeps` (`training_config.go:70-106`) **never sets `CheckpointTrigger`**, so it
is `nil` in production and `maybeTriggerCheckpoint` no-ops (`training.go:391`). Checkpoints are
created only via the HTTP `CreateEnvCheckpoint` API by AReaL, which passes `project_id` but not
`task_id`/`runtime_id` (multica-internal). **AReaL cannot populate the trigger descriptor** ->
multica must resolve it **server-side at checkpoint-create**.

## Design decisions

### D1 - Trigger descriptor (single, server-side resolved, v1)

`resume_trigger` JSONB on `env_checkpoint`:
```json
{"task_id","runtime_id","agent_id","issue_id"|"chat_session_id","project_id","kind"}
```
`kind` ∈ {`issue`, `chat`} dispatches the resume path. Resolved server-side in
`EnvCheckpointService.Create` from the project's in-flight task (see D5). v1 = single object
(`group_size=1`); JSONB is shaped to allow an array for future squad fan-out.

### D2 - Primitive: reset-to-claimable + wake (not re-enqueue, not chat message)

`ResumeAgentRun(trigger)`:
1. **Validate**: load `agent_task_queue` by `task_id`; reject if terminal
   (`completed`/`failed`/`cancelled`) - no double-run; reject if `runtime_id` does not match the
   descriptor (stale/bound elsewhere).
2. **Reset the same row** to `queued` via a new `ResetInFlightTaskForResume(task_id, runtime_id)`
   query: `status='queued'`, `started_at=NULL`, `dispatched_at=NULL`, preserve
   `context`/`runtime_id`/`issue_id`/`chat_session_id`/`priority`. **No new task row** ->
   continuity preserved (task context + Multica DB conversation + paused sandbox disk carry the
   intermediate state).
3. **Wake**: `NotifyTaskAvailable(runtime_id, task_id)` so the resumed daemon claims promptly.
4. The resumed daemon's `ClaimTaskForRuntime -> ClaimTask -> StartTask` re-runs the task; the
   agent continues the issue/chat_session from preserved state.

Rejected:
- **A0 lease auto-requeue**: running tasks are *failed* on timeout, not requeued - would require
  changing lease semantics for all tasks. Rejected.
- **A2 reset-to-dispatched**: the daemon only reclaims *stale* dispatched (`dispatched_at < now()
  - recovery_window`); a fresh `dispatched_at` is not promptly claimed, and faking an old
  timestamp is hacky. Rejected.
- **A3 new `resumed` status + claim-loop edit**: heavier, touches the hot claim path. Rejected.
- **Re-enqueue (`EnqueueTaskForIssue`)**: creates a new task row, loses the in-flight row's
  context/state. Rejected (user-confirmed).
- **Literal chat message**: does not re-engage the task/claim path. Rejected (user-confirmed).

### D3 - `ResumeAgentRunner` seam + partial-resume status

A new interface injected into `EnvCheckpointService` (mirrors `SandboxInstanceResumer`):
```go
type ResumeAgentRunner interface {
    ResumeAgentRun(ctx context.Context, trigger ResumeTrigger) error
}
```
`ResumeFromCheckpoint` calls it **after** all sandbox resume jobs succeed.
`ResumeFromCheckpointResult` gains `TriggerStatus` ∈ {`executed`, `skipped_legacy`, `failed`}.
- Empty `resume_trigger` (legacy/pre-change) -> `skipped_legacy`, no primitive call (today's
  behavior).
- Primitive failure (sandbox up, agent not re-engaged) -> `failed` typed partial-resume, **not**
  a silent no-op; surfaced to the caller.

A `nil` `ResumeAgentRunner` is a loud error for non-empty triggers (resume without a runner would
silently no-op), mirroring the existing `nil`-resumer guard on `ResumeFromCheckpoint`.

### D4 - Execution timing

`ResumeFromCheckpoint` order: (1) load checkpoint, require `save_status == complete`; (2) resume
each sandbox; (3) if `resume_trigger` non-empty, call `ResumeAgentRun`; (4) return
`RolloutHandle` + `TriggerStatus`. The handle stays `"resume:<checkpointID>"` (AReaL's
tree-search contract unchanged); trigger status is an additional field.

### D5 - Server-side trigger resolution at checkpoint-create

`EnvCheckpointService.Create` resolves the in-flight task for the project and populates
`resume_trigger` before persisting:
- Query: `agent_task_queue` rows with `status IN ('running','dispatched')` whose `issue_id` or
  `chat_session_id` belongs to `projectID` (issue/chat_session -> project_id), scoped to the
  workspace.
- v1: take the single in-flight task (`group_size=1`). If none (checkpoint taken between tasks),
  `resume_trigger` is empty -> resume degrades to sandbox-only (D3 `skipped_legacy`-equivalent:
  nothing to re-engage; the next task enqueues normally).
- `kind` = `issue` if `issue_id` set, else `chat`.

This works for the current HTTP create path **and** any future wired automatic trigger (both
route through `Create`). **Wiring the automatic `CheckpointTrigger` is out of scope** for this
change (user-confirmed: server-side resolution only).

### D6 - Migration

Add nullable `resume_trigger` JSONB to `env_checkpoint` (migration `155`-ish, after `154`).
Existing rows stay `NULL` (legacy). Down migration drops the column.

## Sequence

```
SAVE                                     RESUME
────                                     ─────
POST /api/v1/env-checkpoints             POST /api/v1/env-checkpoints/{id}/resume
  (project_id, sandbox_refs, ...)          │
        │                                  ▼
        ▼                          ResumeFromCheckpoint
EnvCheckpointService.Create          ├─ require save_status==complete
  ├─ CaptureProjectSnapshot          ├─ for each sandbox_ref: resumer.Resume
  ├─ resolve in-flight task ◄── D5   │     (sandboxd resume -> Cube /resume, runtime_restarted)
  │   -> resume_trigger              ├─ if resume_trigger non-empty:
  ├─ saver.Save each sandbox (stop)  │     resumeRunner.ResumeAgentRun(trigger)
  └─ persist row + resume_trigger         ├─ validate task non-terminal + runtime_id match
        │                                 ├─ ResetInFlightTaskForResume (same row -> queued)
        ▼                                 └─ NotifyTaskAvailable(runtime_id, task_id)
env_checkpoint row                 └─ return RolloutHandle + TriggerStatus
  + resume_trigger                          │
                                            ▼
                              resumed daemon: ClaimTaskForRuntime -> ClaimTask
                              -> StartTask -> running (continues from preserved state)
```

## Edge cases

- **Task terminal between create and resume**: primitive rejects (`failed` partial-resume); no
  double-run. AReaL sees `TriggerStatus=failed` and can select a different checkpoint.
- **No in-flight task at create**: `resume_trigger` empty; resume is sandbox-only (next task
  enqueues normally).
- **`runtime_id` no longer online at resume**: primitive still resets the task to `queued` and
  wakes; if the runtime never comes back, `running_timeout_secs` eventually fails it (existing
  behavior). The primitive does not block on runtime liveness.
- **Squad/multi-sandbox (future)**: v1 single descriptor; array shape reserved. Multi-runtime
  resolution + per-sandbox trigger matching deferred.
- **Legacy checkpoints (pre-change)**: `resume_trigger` NULL -> `skipped_legacy`, today's
  behavior.
- **Concurrent resume of same checkpoint**: `ResumeFromCheckpoint` has no idempotency guard
  today; re-running the primitive twice would reset an already-queued task (idempotent-ish) but
  could double-wake. v1 accepts single-caller resume; a resume-lock is future hardening.

## Testing strategy

- **Service tests** (fake `ResumeAgentRunner` + fake `EnvCheckpointRepository`):
  - primitive re-activates the existing task row (no new row); rejects terminal task; rejects
    `runtime_id` mismatch; rejects unknown task.
  - `ResumeFromCheckpoint` executes the trigger after sandbox resume; returns `TriggerStatus=executed`.
  - empty `resume_trigger` -> `skipped_legacy`, no primitive call.
  - primitive failure -> `TriggerStatus=failed` (partial resume), not silent.
  - server-side resolution: `Create` populates `resume_trigger` from the project's in-flight task;
    no in-flight task -> empty trigger.
- **sqlc query test**: `ResetInFlightTaskForResume` transitions `running`/`dispatched` -> `queued`,
  clears timestamps, preserves context/runtime_id; rejects terminal (WHERE status IN
  non-terminal).
- **State-machine round-trip test**: `running -> [save] -> [resume+primitive] -> queued ->
  dispatched -> running` via the real query + fake daemon claim.
- **AReaL client test**: `resume_from_checkpoint` surfaces `TriggerStatus`.
- `multica/` is untracked here (`?? multica/`); Go changes are contract+test consistency, not
  deployment from this repo. AReaL tests via `.venv-test/bin/python -m pytest` (per project
  memory: broken `.venv/uv`).

## Risks / trade-offs

- **Task re-runs from claim**: acceptable - conversation state is in Multica DB (preserved) and
  sandbox disk is preserved by pause-in-place; the agent continues the issue/chat, not restarts
  cold.
- **Re-claim race**: mitigated by `runtime_id` binding (`ListQueuedClaimCandidatesByRuntime`
  filters by runtime; only the resumed runtime sees the reset task).
- **Coupling checkpointing to task service**: mitigated by the `ResumeAgentRunner` seam (D3);
  checkpointing stays decoupled from `TaskService` internals.
- **Partial resume ambiguity**: `TriggerStatus=failed` is surfaced; AReaL policy decides next
  action (this change does not impose a policy).

## Out of scope

- Wiring the automatic `CheckpointTrigger` (server-side resolution covers the current path).
- Immutable sandbox fork/branch (deferred by parent change).
- Changing save/pause semantics.
- AReaL tree-search frontier/resume selection policy.
- Squad/multi-sandbox trigger fan-out (v1 single descriptor).
- Resume idempotency/locking.

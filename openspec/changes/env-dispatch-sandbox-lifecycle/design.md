## Context

Env-dispatch currently orchestrates rollout reset and task dispatch, while sandbox lifecycle work lives in Multica's sandbox node gateway. The relevant `lijiannankai@126.com` commits added `sandbox_instance` persistence, sandboxd job execution, Cube create/pause/resume/delete calls, runtime reconfiguration, per-user node ownership, websocket job wakeups, and unreachable-node force-delete behavior.

Training work from earlier changes opens AReaL sessions for trained member tasks and captures critic/logprob signals. The missing piece is a stable environment lifecycle contract connecting env-dispatch, sandbox instances, checkpoint storage, and AReaL resume calls.

## Goals / Non-Goals

**Goals:**

- Treat `sandbox_instance` as the canonical environment lifecycle handle for env-dispatch save/resume paths.
- Reuse existing sandboxd `stop`, `resume`, `delete`, and `reconfigure` jobs instead of creating a parallel sandbox provider.
- Add checkpoint create/list/resume semantics for training rollouts.
- Preserve runtime metadata needed to resume a paused sandbox and continue the rollout.
- Extend AReaL client code with per-agent env intent and checkpoint resume helpers.

**Non-Goals:**

- Immutable sandbox snapshots, forks, copy-on-write, or concurrent branches from one live rollout.
- Replacing sandboxd/Cube lifecycle internals.
- Changing critic reward semantics or the core training session open/close contract.
- Implementing production garbage collection for old checkpoints beyond list/get visibility.

## Decisions

### D1: Use `sandbox_instance` refs as env-dispatch lifecycle handles

Env-dispatch will record structured sandbox refs that include `sandbox_instance.id`, `local_ref`, node id, template, status, endpoint/runtime metadata, and ownership/workspace context. Existing raw sandbox id fields remain compatibility data, but new save/resume code should consume the structured refs.

Alternative considered: add a new AReaL/Fleet/Daytona checkpoint provider. This is rejected for v1 because Multica already owns sandbox lifecycle and has production job/status semantics.

### D2: Save means synchronous pause-in-place via sandboxd `stop`

Checkpoint creation saves an environment by enqueueing the existing sandboxd `stop` job for each affected sandbox instance, then waiting up to the configured checkpoint save timeout for completion. Sandboxd calls Cube `/pause`, updates status through the existing stopped state, and records save status on the checkpoint. A completed save is resumable; a failed or timed-out save is recorded as a terminal non-resumable checkpoint status and returned as a typed failure.

Alternative considered: snapshot/fork the sandbox without interrupting the active rollout. This is deferred because the current sandbox node gateway exposes pause/resume/delete, not immutable fork semantics.

### D3: Resume means resume-from-checkpoint, not branch-from-checkpoint

The public API and client naming should use `resume_from_checkpoint` or `resume-from-checkpoint`. It resumes the saved sandbox instance and restores/uses the saved Multica DB subtree reference. It must not advertise branch/fork semantics.

### D4: Wrap existing sandbox handler/job logic behind a service seam

Add an internal lifecycle service used by checkpointing and env-dispatch. It should delegate to existing query/job code for create, save/stop, resume, delete, runtime restart metadata, websocket wakeup, and force-delete fallback. This avoids copying handler logic into checkpoint code.

### D5: Checkpoints capture sandbox refs and inline Multica state JSONB

A checkpoint stores sandbox instance refs plus an event reference and an inline JSONB DB subtree snapshot for the issue/project state. The snapshot should cover issue, sub-issues, tasks, messages, comments, and event ordering data so resume can re-enter the same Multica entity state. Inline JSONB is the v1 storage shape; object refs or normalized checkpoint tables are deferred until snapshot size or query requirements justify them.

### D6: Per-agent env intent is explicit and optional

Env-dispatch accepts optional per-agent environment specs. Specified agents resolve to their own sandbox instance/template; unspecified agents use the existing default/shared behavior. All agents still share the same Multica entity subtree for the rollout.

### D7: Bridge env-dispatch to sandbox_instance for save/resume-capable rollouts

Env-dispatch creates sandbox_instance-backed environments through the env sandbox lifecycle service `Create` operation (mirroring the existing `CreateSandboxInstance` handler) for save/resume-capable rollouts, and populates structured `SandboxInstanceRef`s. The existing Fleet fork/boot path stays for non-checkpointed rollouts. Scratch creates fresh sandbox_instances from a template; branch creates fresh sandbox_instances from the source env's template (not a live fork), relying on the copied Multica DB subtree to carry trajectory state. Checkpoint save/resume only operates on sandbox_instance refs and returns a typed error against Fleet-only envs. True live-state fork of a sandbox_instance is deferred.

Alternative considered: checkpoint via Fleet snapshot/fork. Rejected because the confirmed save semantic (D2) is pause-in-place via sandboxd stop, which Fleet sandboxes do not support.

## Risks / Trade-offs

- Pause-in-place interrupts the active rollout -> restrict automatic saves to training flows that expect a pause and name APIs as resume, not branch.
- Synchronous checkpoint creation can block or time out -> keep the save timeout configurable and return typed timeout/failure responses.
- Sandbox pause and DB subtree serialization are not perfectly atomic -> store event refs, timestamps, and save status so resume can validate ordering and incomplete saves.
- Inline JSONB snapshots are simple but can grow large -> bound the captured subtree and leave room for a future object-ref migration.
- Legacy `environment.sandbox_ids` may diverge from structured refs -> keep a compatibility read path while new save/resume code prefers `sandbox_instance` refs.
- Offline nodes can make delete/resume unreliable -> preserve the existing force-delete behavior for deletes and return typed errors for resume when the node cannot be reached.
- Entropy logprobs may be unavailable -> skip entropy-gated checkpoints while preserving always-event checkpoint behavior.

## Migration Plan

1. Add schema for env checkpoint records and, if needed, per-agent env/sandbox refs.
2. Backfill or compatibility-read existing sandbox id fields only where needed; do not require existing rollouts to have checkpoint refs.
3. Add lifecycle service tests around existing sandbox job behavior before wiring checkpointing.
4. Add checkpoint APIs and AReaL client methods behind config gates.
5. Roll out save/resume triggers for trained env-dispatch sessions after scoped service tests pass.

Rollback is to disable env checkpointing by config and keep env-dispatch using existing raw sandbox behavior. The sandbox node gateway migrations remain independently useful and should not be rolled back as part of checkpoint rollback.

## Implementation Seam Notes

- Sandbox lifecycle is already represented by `sandbox_instance` rows and `sandbox_job` rows; checkpointing should call a service seam that delegates to the same query/job behavior used by sandbox handlers.
- Env-dispatch currently persists legacy `environment.sandbox_ids`; new save/resume code must prefer structured sandbox-instance refs while preserving compatibility reads for legacy env rows.
- Training-dispatch and session-open hooks already preserve env id through `SaveTrainingDispatch` and `maybeOpenTrainingSession`; this change extends that context with sandbox-instance refs.
- DB subtree snapshots for v1 are scoped to the rollout project subtree and stored inline as JSONB on the checkpoint row.

## Open Questions

- Which AReaL loop owns checkpoint selection policy: tree search frontier selection, critic post-processing, or a separate replay controller?

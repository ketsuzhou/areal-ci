## Why

Three savepoint-shaped mechanisms now sit on the same Cube primitive
(`POST /sandboxes/{id}/snapshots`) with three different lifecycles, and the concern of
"re-engage the agent once its environment is back" is implemented twice in two unrelated
places. The cost is concrete: a tree-search frontier cannot be expanded twice, and
expanding it into N children pays N snapshots of the identical source state instead of
one.

- `env_checkpoint` save/resume pauses **in place** and holds no savepoint at all, so
  resume is structurally 1:1 — it only resumes the instances already listed in
  `sandbox_refs`.
- `env_dispatch(mode="branch")` snapshots the source into a **new** instance but treats
  the snapshot as disposable (`defer` template delete), so nothing survives for a later
  re-expansion of the same frontier.
- `sandbox_snapshot` (`create_template` / `delete_template`) already models an immutable
  savepoint correctly — `cube_snapshot_id`, a `creating/ready/failed/deleting` status
  machine, workspace scoping — but neither consumer above uses it as one.

Both prior designs deferred exactly this work: `env-dispatch-sandbox-lifecycle` deferred
"true live-state fork of a sandbox_instance", and `env-checkpoint-resume-trigger`
deferred "immutable sandbox fork/branch". `channel-message-env-dispatch-source` then
shipped a live source clone as a **third** parallel mechanism rather than filling either
deferral. This change closes both.

A prerequisite experiment against the Cube host settled the design's open questions
empirically: a snapshot completes in ~1.2s, leaves the source `running` and its
processes undisturbed, and restores into a clone with live processes intact (same PID,
same in-progress log file). Cube snapshots are memory-level checkpoint/restore, not
filesystem copies. That result removed three items from the original plan — an assumed
source-pause bug, a parent re-engagement primitive, and asynchronous saving — and
confirmed that the `pkill` of the snapshot-restored daemon in
`buildStartRuntimeInCubeCode` is load-bearing correctness rather than hygiene.

## What Changes

- `env_checkpoint` gains `save_mode` (`pause_in_place` | `snapshot`) and owns the
  savepoints it creates. `pause_in_place` keeps today's behavior byte-for-byte and stays
  a first-class mode for suspend-to-reclaim-resources; `snapshot` records an immutable
  savepoint owned by a `sandbox_snapshot` row and leaves the source running.
- Resume accepts `group_size` and an explicit per-lane idempotency key. `snapshot` mode
  materializes N lanes from one savepoint; `pause_in_place` rejects `group_size > 1`.
  Re-resuming with the same lane key returns the existing lane instead of creating
  another.
- `env_dispatch(mode="branch")` is routed internally to
  `resume(checkpoint, group_size=N)`. The AReaL-facing contract is unchanged; branch
  stops being a separate provisioning path.
- The continuation concern is unified behind the existing `ResumeAgentRunner` seam with
  two named strategies: same-runtime task resume (`pause_in_place`) and forked-runtime
  task enqueue (`snapshot` lanes). The re-enqueue logic currently inline in
  `provisionEnvDispatchAgentBranch` moves behind the seam.
- **BREAKING** (internal only): the `clone` sandbox job type is retired in favor of
  `create_template` + `create`, so N lanes share one snapshot and the savepoint outlives
  the first use.
- Each savepoint is owned by exactly one checkpoint and released when that checkpoint
  goes away, replacing today's use-once-and-delete behavior.
- `pause_in_place` resume continues the interrupted CLI session instead of starting
  cold, by resolving the prior session from the resumed task row itself. Forked lanes
  stay cold on purpose.

This change is kept as a single OpenSpec change even though the session-continuation
item is independently deliverable, because the reviewer needs the savepoint and
continuation decisions in one place to judge the per-mode session policy.

## Capabilities

### New Capabilities

- `env-savepoint-fanout`: immutable savepoint ownership for env checkpoints, the two
  save modes, fan-out resume with per-lane idempotency and interruption recovery,
  branch-as-resume routing, and savepoint reclamation.
- `agent-continuation-seam`: a single seam for re-engaging an agent after its
  environment returns, with one strategy per save mode, and the per-mode CLI session
  continuation policy.

### Modified Capabilities

The capabilities this change touches exist only as delta specs in sibling changes that
are not yet archived into `openspec/specs/`, so there is no main spec to write a
`## MODIFIED Requirements` block against. Reconciled below instead, one row per sibling
requirement. Note that `env-checkpoint-resume` is not its own change: its spec lives
under `openspec/changes/env-dispatch-sandbox-lifecycle/specs/env-checkpoint-resume/`.

**Two requirements are contradicted outright** and must be corrected when those siblings
archive, because both explicitly ruled out the primitive this change adds:

| Sibling requirement                                                             | Statement now false                                                                                                                                                              | Replaced by                                                                                                                                                  |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `env-checkpoint-resume` → Resume from checkpoint                                | "Resume ... MUST not expose immutable branch/fork semantics"                                                                                                                     | `env-savepoint-fanout`: snapshot-mode checkpoints own immutable savepoints, and resume materializes lanes from them                                          |
| `env-dispatch-sandbox-lifecycle` → Env-dispatch sandbox_instance backend bridge | "Branch ... SHALL create fresh sandbox_instances from the source env's template rather than a live fork" and "True live-state fork of a sandbox_instance is out of scope for v1" | `env-savepoint-fanout`: branch dispatch is served by savepoint-backed create, so a lane does carry the source's live state (Cube snapshots are memory-level) |

The rest are preserved or extended rather than replaced:

| Sibling requirement                                                                            | Disposition                                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env-checkpoint-resume` → Pause-in-place checkpoint creation                                   | **Extended.** Still the default and still a synchronous wait, but the wait is now per save mode: `pause_in_place` waits for sandboxd stop, `snapshot` waits for each savepoint to reach ready and leaves the source running. Pre-existing rows resolve to `pause_in_place`. |
| `env-checkpoint-resume` → Checkpoint listing and retrieval                                     | **Preserved**, including the cross-workspace refusal. Deletion is added alongside it; the sibling specified none.                                                                                                                                                           |
| `env-checkpoint-resume` → Resume from checkpoint (naming scenario)                             | **Preserved.** The API is still resume-from-checkpoint; branch dispatch is routed *through* it internally rather than renamed.                                                                                                                                              |
| `env-checkpoint-resume` → AReaL checkpoint client integration                                  | **Extended.** `lane_count` and `lane_key` are optional and omitted when absent, so a caller that sends no body still gets its single-lane resume.                                                                                                                           |
| `env-checkpoint-resume-trigger` → Resume-trigger captured at checkpoint create                 | **Preserved** unchanged, descriptor and server-side resolution included.                                                                                                                                                                                                    |
| `env-checkpoint-resume-trigger` → Resume-agent-run primitive re-engages in-flight task         | **Preserved, moved behind a seam.** It is now the `pause_in_place` strategy of the continuation seam. Its guarantees are unchanged: same task row, no new row, terminal tasks rejected rather than double-run.                                                              |
| `env-checkpoint-resume-trigger` → Resume-from-checkpoint executes trigger after sandbox resume | **Extended.** The typed partial-resume result now also covers a fan-out where some lanes materialized and others failed, rather than only "sandbox up, agent not re-engaged".                                                                                               |
| `env-checkpoint-resume-trigger` → Legacy checkpoint without trigger degrades gracefully        | **Preserved.** An empty trigger still resumes sandbox-only.                                                                                                                                                                                                                 |
| `env-dispatch-sandbox-lifecycle` → Sandbox lifecycle service reuse                             | **Narrowed, not contradicted.** The requirement enumerates create, save, resume, delete, and reconfigure; `CloneSandboxInstance` was never among them, so retiring it leaves the requirement intact.                                                                        |
| `env-dispatch-sandbox-lifecycle` → Env-dispatch sandbox-instance lifecycle handles             | **Untouched.**                                                                                                                                                                                                                                                              |
| `env-dispatch-sandbox-lifecycle` → Per-agent environment intent                                | **Untouched.**                                                                                                                                                                                                                                                              |
| `critic-driven-training-signal` → Entropy recording from proxied LLM traffic                   | **Untouched.**                                                                                                                                                                                                                                                              |
| `training-session-lifecycle` → Session-open on trained-member task creation                    | **Untouched.** A lane's session is opened by its own task creation on the existing path.                                                                                                                                                                                    |

## Impact

- **Schema**: `env_checkpoint` (`save_mode`); `sandbox_snapshot` (owning checkpoint); a
  new lane table backing per-lane idempotency and interruption recovery;
  `sandbox_job_type_check` (drop `clone`). Existing `env_checkpoint` rows default to
  `pause_in_place` and own no savepoint, so their behavior is unchanged.
- **Server**: `EnvCheckpointService` (`Create`, `ResumeFromCheckpoint`),
  `EnvSandboxLifecycleService` (`CloneSandboxInstance` retired), env-dispatch branch
  provisioning, `taskResumeRunner`, and the agent task claim path for prior-session
  resolution.
- **sandboxd**: the `clone` job handler is removed; `create_template` and `create` cover
  it.
- **AReaL**: no client contract change. `create_checkpoint` stays synchronous and
  returns a terminal save status; slow saves keep degrading through the existing
  `save_timeout_ms` → `timed_out` → resume-rejected path that `MulticaCheckpointError`
  already handles.
- **Not affected**: Fleet-backed envs remain non-checkpointable; the automatic
  `CheckpointTrigger` stays unwired; AReaL's frontier selection policy is untouched.

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

<!-- None. The capabilities this change touches (`env-checkpoint-resume`,
     `env-checkpoint-resume-trigger`, `env-dispatch-sandbox-lifecycle`) exist only as delta
     specs in sibling changes that are not yet archived into `openspec/specs/`, so their
     requirement changes are stated in the new capabilities above and reconciled with those
     deltas at archive time. -->

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

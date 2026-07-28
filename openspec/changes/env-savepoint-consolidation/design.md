## Context

Three mechanisms wrap the same Cube primitive (`POST /sandboxes/{id}/snapshots`) with
three lifecycles, and the "re-engage the agent after its environment returns" concern
has two unrelated implementations.

| Mechanism                                 | Sandbox semantics                                          | Fan-out                | Source after the op | Agent re-engagement                                                   |
| ----------------------------------------- | ---------------------------------------------------------- | ---------------------- | ------------------- | --------------------------------------------------------------------- |
| `env_checkpoint` save/resume              | pause in place, **same** instance                          | 1:1 only               | paused              | reset the same task row, wake the **same** runtime                    |
| `env_dispatch(mode="branch")`             | snapshot into a **new** instance, template deleted on exit | one child per dispatch | keeps running       | pre-create runtime, copy channel subtree, re-enqueue, **new** runtime |
| `sandbox_snapshot` create/delete template | snapshot into a reusable template with a DB status machine | n/a                    | keeps running       | none (infrastructure)                                                 |

Checkpoint-resume and branch are not two spellings of one operation.
`ResumeFromCheckpoint` only calls `Resume` over the instances already in `sandbox_refs`,
there is no `group_size` on either API, and `env_checkpoint` has no savepoint column —
so there is nothing to materialize a second child from. Resuming the same checkpoint
twice cannot diverge either: both resumes target one instance and one task row, and
`ResetInFlightTaskForResume` requires `status='draining'`, so the second call matches no
rows. Conversely branch cannot continue the *same* task on the *same* instance, because
it necessarily mints a new runtime, subtree, and task row. The duplication is therefore
not at the operation level; it is the savepoint primitive and the continuation logic
underneath.

### Prerequisite experiment (settled the design's unknowns)

Run directly against the Cube API on the project's Cube host, creating and deleting two
sandboxes and one template:

- A snapshot completed in **1.2s** and left the source `state=running`, immediately
  executable.
- The source's probe process was undisturbed across the snapshot: consecutive ticks at T
  and T+1s, same PID.
- A sandbox created from the resulting template came up with the probe **still running**
  — same PID, same append-in-progress log file, counter continuing from the snapshot
  point with roughly 3s of missing ticks (the restore window).

Cube snapshots are therefore memory-level checkpoint/restore, not filesystem copies.
This matches the `cube.master.runtime.snapshot.id` metadata observed on live sandboxes.

Consequences, all of which shrink the change:

- The suspected "snapshot leaves the source paused" bug does not exist; the defensive
  comment in `createCubeSnapshotTemplate` does not describe current Cube behavior. No
  fix needed.
- No parent re-engagement primitive is needed, since the parent's in-flight turn is
  never interrupted.
- Asynchronous saving is unnecessary at this latency.
- A memory image is application-consistent by construction, so the usual
  crash-consistency caveat about a source agent mid-write does not apply.
- The `pkill` of the snapshot-restored daemon in `buildStartRuntimeInCubeCode` is
  load-bearing correctness: without it, every lane would come up running the source's
  daemon, sharing one `daemon.id` and one duplicated in-flight model request.

## Goals / Non-Goals

**Goals:**

- One savepoint primitive, owned by `sandbox_snapshot`, shared by checkpoint and branch.
- One continuation seam with one strategy per save mode.
- A frontier can be expanded more than once, and expanding into N lanes costs one
  snapshot.
- `pause_in_place` behavior is preserved exactly, including its shipped resume-trigger
  path.

**Non-Goals:**

- Changing the AReaL client contract.
- Checkpointing Fleet-backed envs (still a typed error).
- Wiring the automatic `CheckpointTrigger`.
- AReaL's frontier selection policy.
- Forking inside a turn (see Open Questions).
- Squad / multi-runtime trigger fan-out beyond the existing single-descriptor shape.

## Decisions

### D1 — Separate savepoint from continuation; branch becomes a fan-out of resume

A checkpoint records an immutable savepoint, the project subtree it already captures,
and a continuation descriptor. Resume materializes `group_size` lanes from the
savepoint, and `mode="branch"` is routed to `resume(checkpoint, group_size=N)`
server-side.

*Alternative rejected*: keep two operations and only deduplicate the Cube call. That
leaves two continuation implementations and still cannot re-expand a frontier, which is
the actual requirement.

### D2 — Both save modes are first-class

`pause_in_place` is retained, not deprecated: it is the only mode that continues the
same task row on the same instance, and the only cheap suspend-to-reclaim-resources
path. `snapshot` records a savepoint and leaves the source running.

*Alternative rejected*: make `snapshot` the only mode. Suspending to reclaim resources
would then require creating a savepoint and destroying the instance, which is strictly
more work and loses same-task continuity.

### D3 — Live filesystem/memory state is carried for every branch

All domains branch with live state. The experiment showed the cost is ~1s, so there is
no reason to offer a state-less "logical branch" mode, and the earlier assumption that
trajectory state lives entirely in the Multica DB does not hold for repository-editing
rollouts.

*Alternative rejected*: per-domain choice (live for `swe_lego`, logical for
`self_play`). Two branch semantics for one operation is the kind of divergence this
change exists to remove, and the measured cost does not justify it.

### D4 — One continuation seam, two named strategies

`ResumeAgentRunner` already is the seam. Give it a same-runtime strategy (reset the
existing task row, wake the same runtime — today's `taskResumeRunner`) and a
forked-runtime strategy (enqueue on the copied subtree bound to the lane's new runtime —
today's inline logic in `provisionEnvDispatchAgentBranch`). Both share the existing
`ResumeTrigger` descriptor and `TriggerStatus` contract.

### D5 — Retire the `clone` job type

`clone` fuses snapshot, create, and template delete, owning none durably. Splitting it
into `create_template` (a `sandbox_snapshot` row) plus one `create` per lane makes the
savepoint durable, lets N lanes share one snapshot, and removes the fused job whose type
check already churned across two migrations.

### D6 — Per-mode CLI session continuation

`pause_in_place` resume continues the interrupted session, resolved from the resumed
task row's own pinned session. Today it starts cold because prior-session lookup only
matches terminal tasks, even though the session transcript is on the preserved disk — a
plain waste. Forked lanes stay cold: N lanes must not resume one mutable session file
under N runtimes, and for tree search a cold start preserves the divergence that
branching exists to create.

### D7 — Savepoint lifetime is owned by the checkpoint

A savepoint is released when its owning checkpoint is deleted or expires, not on first
use. Reference counting prevents reclaiming a savepoint that another checkpoint still
references. This is the inverse of today's delete-on-exit and is what makes re-expansion
possible.

### D8 — Saving stays synchronous

`create_checkpoint` keeps returning a terminal save status, so the client is unchanged.
A slow save degrades through the existing `save_timeout_ms` → `timed_out` →
resume-rejected path, which `MulticaCheckpointError` already handles, so no asynchronous
channel is needed even if a large sandbox snapshots more slowly than the measured 1.2s.

### D9 — Resume idempotency via an explicit per-lane key

Once resume fans out, "resume this checkpoint again" is ambiguous between a retry and a
request for more lanes. The caller supplies a lane key: the same key returns the
existing lane, a new key opens a new one. This also makes expanding a frontier in
batches well-defined.

*Alternatives rejected*: cumulative semantics (every resume opens N more lanes) pushes
idempotency onto the caller and makes retries destructive; single-shot (409 on
re-resume) gives up frontier re-expansion, which is a primary goal.

## Risks / Trade-offs

- Snapshot latency at real SWE scale is unmeasured (the experiment used a 2 GB sandbox
  with a small working set) → D8's synchronous path already degrades to `timed_out` +
  resume rejection, so a slow save costs a frontier, not correctness.
- Savepoint storage and Cube template quota grow, since savepoints now outlive first use
  → D7 checkpoint-owned reclamation; partly offset by N lanes sharing one snapshot
  instead of taking N.
- Routing branch through resume touches a path that is live for tree-search rollouts →
  phase the change so the seam extraction and the schema work land before the routing
  switch, each behavior-preserving on its own.
- `pause_in_place` is the mode with a shipped, tested resume-trigger path → it must stay
  on exactly that code path; the schema work defaults existing rows to it.
- Retiring `clone` changes a sandboxd job contract → sandboxd and server deploy
  together, and the replacement jobs (`create_template`, `create`) already exist and are
  exercised.

## Migration Plan

Schema: add `save_mode` (defaulting to `pause_in_place`) to `env_checkpoint`; give
`sandbox_snapshot` an owning checkpoint that cascades on deletion; add a lane table
backing per-lane idempotency and interruption recovery (shape in the Design Doc); drop
`clone` from `sandbox_job_type_check`. Existing `env_checkpoint` rows keep today's
behavior with no backfill. Down migrations drop the added columns and table and restore
the job type value.

Phasing, each step shippable and behavior-preserving except where noted:

1. Extract the continuation seam — move branch's inline re-enqueue behind the
   forked-runtime strategy. Pure refactor.
1. Schema, `save_mode=snapshot`, `group_size`, lane keys, and checkpoint-owned
   savepoints. Branch still uses its own provisioning path.
1. Route `mode="branch"` to `resume(checkpoint, group_size=N)`. This is the step that
   removes the duplication and the one to stage carefully.
1. Retire the `clone` job type.
1. Warm session continuation for `pause_in_place`. Independent; can land any time after
   2\.

Rollback: steps 1, 4, and 5 revert independently. Step 3 reverts by restoring the direct
branch provisioning path, which step 2 leaves intact. Step 2's columns are additive and
default to current behavior, so reverting code without dropping columns is safe.

## Open Questions

- Snapshot duration for a realistic SWE sandbox (large repository, larger memory). Does
  not block the design because of D8, but it should be measured before relying on branch
  latency in a training budget.
- Forking inside a turn is now known to be technically possible, since a restored clone
  carries live processes including a mid-turn agent. It is out of scope here: it
  requires solving runtime identity across N clones sharing one `daemon.id` and N
  duplicated in-flight model requests, which is precisely what the existing `pkill`
  avoids. Worth a separate exploration if intra-turn branch points prove valuable for
  tree search.
- Whether a `group_size=1` branch — provably the only consumer of its savepoint — should
  be allowed to warm-resume the cloned session as an exception to D6.

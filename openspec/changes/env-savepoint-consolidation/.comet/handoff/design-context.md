# Comet Design Handoff

- Change: env-savepoint-consolidation
- Phase: design
- Mode: compact
- Context hash: a05d1c1f73d5ae1495026c2956e6a06ed31244fb0597f12cfd45e397121a6f01

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-savepoint-consolidation/proposal.md

- Source: openspec/changes/env-savepoint-consolidation/proposal.md
- Lines: 1-101
- SHA256: b3d691ab6b2c636412d8f8d1f705b0397792663ac8668d0278725a8d14e3dc5e

[TRUNCATED]

```md
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

```

Full source: openspec/changes/env-savepoint-consolidation/proposal.md

## openspec/changes/env-savepoint-consolidation/design.md

- Source: openspec/changes/env-savepoint-consolidation/design.md
- Lines: 1-237
- SHA256: ec256f0a00bb07bb19bc41cd9f104be5a775dcfac7272ac681ea3927e24d1dd9

[TRUNCATED]

```md
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

```

Full source: openspec/changes/env-savepoint-consolidation/design.md

## openspec/changes/env-savepoint-consolidation/tasks.md

- Source: openspec/changes/env-savepoint-consolidation/tasks.md
- Lines: 1-300
- SHA256: 1496ab6d67bc2d967b3d58bad43e89124f62c6707a7a2fbf969e7d7c44df5832

[TRUNCATED]

```md
## 0. Ground rules

- [x] 0.1 Confirm `multica/` is excluded from this repository, so Go implementation
  lands in the multica repo while this change owns the contract, specs, and test
  expectations
- [x] 0.2 Confirm each phase below is landed and verified independently; only phase 4
  changes externally observable branch behavior

Build configuration for this change: `isolation: worktree` (multica worktree on branch
`feature/20260728/env-savepoint-consolidation`, based on `upstream/dev` at `cf7016107`),
`build_mode: executing-plans`, `tdd_mode: direct`, `review_mode: off`. Automatic code
review is skipped at the user's explicit request; the per-phase Go tests named in each
task below remain the acceptance evidence.

Verification limit, decided explicitly: Postgres is **not** installed in the build
environment. The `*_Integration` query tests are written but self-skip without
`DATABASE_URL` and must not be reported as passing. Migrations 244/245/246, the
`UNIQUE (checkpoint_id, lane_key)` claim race, and the `ON DELETE CASCADE` reclamation
stay unverified until run against a real database — this is the primary open risk to
carry into verification.

The cost is larger than skipped integration tests, and silently so: `internal/handler`,
`cmd/server`, `internal/workgraph`, and `pkg/agent` each have a `TestMain` that calls
`os.Exit(0)` when Postgres is unreachable, so `go test` on those packages prints `ok`
while executing **no tests at all**. Any `ok` from them in this environment is
worthless; only compilation was checked. Every HTTP-boundary test in this change is
therefore unverified. `internal/service` and `internal/migrations` have no such gate and
do really run, so they carry the real evidence.

`sqlc` **is** usable (v1.31.1, matching the version that generated the checked-in code;
it reads the schema from `migrations/` statically and needs no database), so column
order in generated code is derived rather than guessed. It cannot simply be run, though:
the checked-in generated code is hand-maintained — `sandbox.sql.go` routes all
`sandbox_snapshot` reads through one hand-written scan helper, and files are laid out in
query-file order instead of the alphabetical order sqlc v1.31.1 emits — so a wholesale
regeneration rewrites roughly 20 unrelated files. The working method is to regenerate,
transplant only the hunks belonging to this change, and revert the rest. Because a
scan/column misalignment there is invisible to the compiler and to every non-database
test, `TestGeneratedSnapshotScanMatchesSelectedColumns` guards it and was
mutation-checked.

## 1. Continuation seam extraction (behavior-preserving)

- [x] 1.1 Introduce two named strategies behind the existing `ResumeAgentRunner` seam: a
  same-runtime strategy wrapping today's `taskResumeRunner`, and a forked-runtime
  strategy interface
- [x] 1.2 Select the strategy from the checkpoint's save mode at resume time, defaulting
  to same-runtime for existing rows
- [x] 1.3 Move the per-lane task enqueue currently inline in
  `provisionEnvDispatchAgentBranch` behind the forked-runtime strategy without changing
  its behavior (the enqueue is actually in `dispatchBranchChannelMessage`, not
  `provisionEnvDispatchAgentBranch`, which never enqueues)
- [x] 1.4 Report the continuation outcome (executed, skipped, failed) uniformly from
  both strategies, keeping a failed continuation after a successful restore visible as a
  partial resume
- [x] 1.5 Service tests: strategy selection by save mode; branch continuation routed
  through the seam; terminal-task and runtime-mismatch rejections still hold; skipped
  outcome when no continuation descriptor exists

## 2. Savepoint schema and snapshot save mode

- [x] 2.1 Migration: add `save_mode` (default `pause_in_place`) to `env_checkpoint`;
  give `sandbox_snapshot` a single owning checkpoint reference that cascades on
  checkpoint deletion; write the matching down migration
- [x] 2.2 Verify existing `env_checkpoint` rows resolve to `pause_in_place` with no
  owned savepoint and need no backfill — guaranteed by the DDL itself
  (`ADD COLUMN ... NOT NULL DEFAULT 'pause_in_place'` fills every existing row, and a
  new nullable `checkpoint_id` leaves every existing snapshot unowned), and asserted by
  `TestMigration244AddsSaveModeAndCheckpointOwnedSavepoints`, which fails if any
  backfill statement appears. Applying the migration against a live database is part of
  the deferred verification above.
- [x] 2.3 Queries: read/write `save_mode`; attach a savepoint to its owning checkpoint;
  list a checkpoint's savepoints
- [x] 2.4 Checkpoint create in `snapshot` mode: create one savepoint per sandbox ref
  through the existing `create_template` job, wait for the snapshot record to reach
  ready, and leave every source instance running — the service drives this through the
  `SavepointCreator` seam; the production adapter that actually enqueues
  `create_template` arrives with Phase 6, so snapshot mode is refused as unconfigured
  until then rather than downgraded
- [x] 2.5 Fail the checkpoint save when a savepoint's snapshot record reaches a failed

```

Full source: openspec/changes/env-savepoint-consolidation/tasks.md

## openspec/changes/env-savepoint-consolidation/specs/agent-continuation-seam/spec.md

- Source: openspec/changes/env-savepoint-consolidation/specs/agent-continuation-seam/spec.md
- Lines: 1-119
- SHA256: f391d216be9e8c951ae0e77ca23e9cba8acdf6a0c9afeac9e78f3b153fdfebcf

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Agent continuation goes through a single seam

Re-engaging an agent after its environment returns SHALL be performed through one
continuation seam with exactly one strategy selected by the checkpoint's save mode. No
code path outside that seam SHALL enqueue or re-activate an agent task as part of
environment restoration.

#### Scenario: Save mode selects the strategy

- **WHEN** a checkpoint is resumed
- **THEN** the continuation seam is invoked with the checkpoint's continuation
  descriptor
- **AND** the same-runtime strategy is used for `pause_in_place`
- **AND** the forked-runtime strategy is used for `snapshot`

#### Scenario: Branch continuation uses the seam

- **WHEN** a branch dispatch materializes a lane
- **THEN** that lane's agent is re-engaged through the continuation seam
- **AND** not through provisioning-local enqueue logic

### Requirement: Same-runtime continuation reuses the existing task

The same-runtime strategy SHALL re-activate the existing in-flight task row bound to the
resumed runtime, without creating a new task row, and SHALL wake that runtime so it
claims the task promptly. It SHALL reject a task that has reached a terminal state and
SHALL reject a task whose bound runtime does not match the continuation descriptor.

#### Scenario: Interrupted task is re-activated in place

- **WHEN** a `pause_in_place` checkpoint carrying a continuation descriptor is resumed
- **THEN** the descriptor's existing task row becomes claimable again by its bound
  runtime
- **AND** no new task row is created
- **AND** the task's preserved context, issue or chat association, and runtime binding
  are unchanged

#### Scenario: Terminal task is rejected

- **WHEN** the descriptor's task reached a terminal state between checkpoint creation
  and resume
- **THEN** the continuation is rejected
- **AND** the task is not run a second time

#### Scenario: Runtime mismatch is rejected

- **WHEN** the descriptor's task is bound to a runtime other than the one named in the
  descriptor
- **THEN** the continuation is rejected

### Requirement: Forked-runtime continuation enqueues per lane

The forked-runtime strategy SHALL enqueue a task for each materialized lane against that
lane's copied project subtree and its own agent runtime. Lanes SHALL NOT share a task
row, an agent runtime, or a runtime identity with each other or with the source
environment.

#### Scenario: Each lane gets its own task and runtime

- **WHEN** a `snapshot` checkpoint is resumed into three lanes
- **THEN** three distinct task rows are enqueued, one per lane
- **AND** each is bound to that lane's own agent runtime
- **AND** the source environment's task and runtime are unaffected

#### Scenario: Lane runtime identity is reset

- **WHEN** a lane is materialized from a savepoint that captured a running agent daemon
- **THEN** the daemon inherited from the savepoint is stopped before the lane's runtime
  starts
- **AND** the lane registers under its own runtime identity rather than the source's

### Requirement: Continuation outcome is reported, not silently dropped

A resume SHALL report the outcome of continuation as a distinct status covering
executed, skipped because no continuation descriptor was recorded, and failed. A failed
continuation after a successful environment restore SHALL be reported as a partial
resume rather than as success.


```

Full source: openspec/changes/env-savepoint-consolidation/specs/agent-continuation-seam/spec.md

## openspec/changes/env-savepoint-consolidation/specs/env-savepoint-fanout/spec.md

- Source: openspec/changes/env-savepoint-consolidation/specs/env-savepoint-fanout/spec.md
- Lines: 1-190
- SHA256: 96872a6c51d098815d4d3d9db2e899810d15d45a9c484df218ddb669744b450c

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Env checkpoints declare a save mode

An env checkpoint SHALL declare a save mode of either `pause_in_place` or `snapshot`.
`pause_in_place` SHALL suspend the source sandbox instances and record no savepoint.
`snapshot` SHALL record an immutable savepoint per source instance and SHALL leave the
source instances running. Checkpoints created without an explicit save mode SHALL
default to `pause_in_place`.

#### Scenario: Snapshot mode leaves the source running

- **WHEN** a checkpoint is created with save mode `snapshot` against a live env whose
  agent is mid-task
- **THEN** the checkpoint records one savepoint reference per source sandbox instance
- **AND** every source instance remains in a running state
- **AND** the source agent's in-flight task remains in its pre-checkpoint state

#### Scenario: Pause-in-place mode suspends the source and records no savepoint

- **WHEN** a checkpoint is created with save mode `pause_in_place`
- **THEN** every source sandbox instance is suspended
- **AND** the checkpoint's savepoint references are empty

#### Scenario: Existing checkpoints keep their behavior

- **WHEN** a checkpoint row created before this capability is read
- **THEN** its save mode resolves to `pause_in_place`
- **AND** its savepoint references are empty
- **AND** resuming it behaves exactly as it did before this capability

### Requirement: Savepoints are immutable and independently owned

A savepoint created for a checkpoint SHALL be represented by a durable snapshot record
that reaches a ready state before the checkpoint is reported complete. The savepoint
SHALL NOT be reclaimed when a lane is materialized from it.

#### Scenario: Savepoint is ready before the checkpoint completes

- **WHEN** a `snapshot` mode checkpoint is created
- **THEN** each savepoint's snapshot record is in a ready state
- **AND** only then is the checkpoint's save status reported complete

#### Scenario: Savepoint survives lane materialization

- **WHEN** a lane has been materialized from a checkpoint's savepoint
- **THEN** the savepoint's snapshot record is still ready
- **AND** it can materialize a further lane

#### Scenario: Savepoint creation failure fails the save

- **WHEN** a savepoint's snapshot record reaches a failed state during checkpoint
  creation
- **THEN** the checkpoint's save status is reported failed
- **AND** the checkpoint is not resumable

### Requirement: Resume materializes a requested number of lanes

Resuming a `snapshot` mode checkpoint SHALL accept a requested lane count and SHALL
materialize that many sandbox instances from the checkpoint's savepoints, taking one
snapshot per source instance rather than one per lane. Each lane SHALL receive its own
copy of the captured project subtree and its own agent runtime.

#### Scenario: Three lanes from one savepoint

- **WHEN** a `snapshot` mode checkpoint is resumed with a requested lane count of three
- **THEN** three sandbox instances are created from the checkpoint's savepoint
- **AND** each lane has its own copied project subtree and its own agent runtime
- **AND** no additional snapshot of the source is taken

#### Scenario: Pause-in-place rejects fan-out

- **WHEN** a `pause_in_place` checkpoint is resumed with a requested lane count greater
  than one
- **THEN** the resume is rejected with a typed error
- **AND** no sandbox instance is created

#### Scenario: Pause-in-place resumes the same instance

- **WHEN** a `pause_in_place` checkpoint is resumed with a requested lane count of one

```

Full source: openspec/changes/env-savepoint-consolidation/specs/env-savepoint-fanout/spec.md

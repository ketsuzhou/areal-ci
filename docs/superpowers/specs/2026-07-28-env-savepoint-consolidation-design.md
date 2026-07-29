---
comet_change: env-savepoint-consolidation
role: technical-design
canonical_spec: openspec
---

# Design: env-savepoint-consolidation

OpenSpec canonical specs:
`openspec/changes/env-savepoint-consolidation/specs/env-savepoint-fanout/spec.md` and
`openspec/changes/env-savepoint-consolidation/specs/agent-continuation-seam/spec.md`.
This document is the deep technical design; requirements and scenarios live in those
deltas, and the high-level framework lives in the change's `design.md`.

## Prerequisite experiment (settled three assumptions)

Run directly against the Cube API on the project's Cube host, creating and deleting two
sandboxes and one template. A probe process appended a counter and a unix timestamp to a
file every second, which distinguishes "frozen then restored" (continuous counter, gap
in timestamps) from "killed" (counter restarts or process absent).

| Question                                                       | Result                                                                                                                                              |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Does `POST /sandboxes/{id}/snapshots` leave the source paused? | No. `state=running` immediately after, exec available.                                                                                              |
| How long does it take?                                         | **1.2s** (not the "several minutes" the code comments assume).                                                                                      |
| Does the source's running process survive?                     | Yes, undisturbed: consecutive ticks at T and T+1s, same PID 305.                                                                                    |
| Does the produced template carry live processes into a clone?  | **Yes.** Same PID 305, same append-in-progress log file, counter continuing from the snapshot point with ~3s of missing ticks (the restore window). |

Cube snapshots are memory-level checkpoint/restore, not filesystem copies. This matches
the `cube.master.runtime.snapshot.id` metadata observed on live sandboxes.

Four consequences, three of which removed planned work:

- The suspected "snapshot leaves the source paused" bug does not exist. The defensive
  resume in `createCubeSnapshotTemplate` does not describe current Cube behavior, and
  `cloneCubeSandbox` omitting it is harmless. **No fix needed** — this had been scoped
  as the lowest-risk first phase.
- No parent re-engagement primitive is needed. This matters because a `draining` task is
  reclaimed by nothing (`ReclaimStaleDispatchedTaskForRuntime` only takes `dispatched`,
  and the only `running` expiry *fails* the task), so a frozen parent would have
  silently died. The parent is never frozen, so the gap never opens.
- Asynchronous saving is unnecessary at this latency; saving stays synchronous and the
  client contract is untouched.
- A memory image is application-consistent by construction, so the usual
  crash-consistency caveat about a source agent mid-write does not apply.

The fourth consequence adds a constraint rather than removing work: the `pkill` of the
snapshot-restored daemon in `buildStartRuntimeInCubeCode` is **load-bearing
correctness**, not hygiene. Its comment ("possibly a live daemon in memory") is
literally true. Without it every lane comes up running the source's daemon, sharing one
`daemon.id` and one duplicated in-flight model request.

## D1 — Lane records: one table serves idempotency and crash recovery

Lane idempotency and lane-provisioning crash recovery are the same problem. A single
table solves both:

```
env_checkpoint_lane
  id, checkpoint_id -> env_checkpoint(id) ON DELETE CASCADE
  lane_key text
  status text CHECK (status IN ('provisioning','ready','failed'))
  instance_id, project_id, runtime_id, task_id   -- filled as materialization advances
  error text
  created_at, updated_at
  UNIQUE (checkpoint_id, lane_key)
```

Resume inserts the lane row with `ON CONFLICT DO NOTHING`:

- **Insert wins** → this caller owns the lane and materializes it.
- **Insert conflicts** → a lane already exists; its status decides the response. `ready`
  returns the existing lane, a stale `provisioning` continues materialization, `failed`
  surfaces the failure.

The unique index **is** the idempotency mechanism, so concurrent resume needs no
application lock. The status column **is** the crash-recovery mechanism, so a sandbox
orphaned mid-lane has a discoverable owner and can be swept.

Per-lane materialization order: create instance from the savepoint, copy the project
subtree, mint the runtime, enqueue the task. Each step records its id on the lane row,
so a resumed `provisioning` lane continues from the first unfilled step instead of
restarting.

Rejected:

- **JSONB lane array on `env_checkpoint` + row lock.** Serializes every resume of a
  checkpoint, and lane state buried in JSONB is hard to query or sweep.
- **Non-persistent idempotency keys (Redis/in-memory).** Lost on restart, and a partial
  failure silently leaks a sandbox and Cube template quota on a shared cluster.

## D2 — Savepoint ownership is 1:1, reclaimed by cascade

A savepoint is owned by exactly one checkpoint and reclaimed when that checkpoint is
deleted. No reference counting, and no `snapshot_refs` JSONB (the open-phase sketch).

The sharing case is unreachable. Branch reuse is keyed on `env_id`, so one frontier has
one snapshot-mode checkpoint; once the source advances, the next branch gets a new
`env_id`, a new snapshot, and a new savepoint. Reference counting would pay for that
unreachable case with a real hazard: an integer count drifts silently when a crash lands
between checkpoint deletion and decrement, either leaking a Cube template or — worse —
deleting a savepoint that a still- resumable checkpoint needs.

**Consistency with the re-expansion goal**: re-expanding a frontier means a *new lane
key on the same checkpoint*, not a second checkpoint. This is what makes 1:1 ownership
compatible with unbounded expansion, and it promotes the branch reuse lookup (`env_id` →
the existing `save_mode='snapshot'`, `save_status='complete'` checkpoint for that
workspace and project) to a load-bearing path rather than an optimization.

If savepoint deduplication is ever wanted, adding a join table turns the count into a
`NOT EXISTS` query. That is additive, not a rewrite.

## D3 — Lane keys derive from a retry-stable anchor

Branch dispatch supplies only `env_id` and its contract does not change, so the server
mints lane keys. They must derive from an anchor stable across retries of the same
dispatch — the dispatch record id, or the caller's `event_ref` plus an ordinal.

This is the single point where outer-retry safety is decided. If the anchor is not
stable, a retried branch dispatch doubles the lanes and D1's idempotency protects
nothing, because each retry presents fresh keys. The anchor choice must be pinned during
implementation against whatever stable id the dispatch record actually carries.

## D4 — Checkpoint deletion is refused while lanes are provisioning

Deleting a checkpoint with `provisioning` lanes would cascade away the lane rows and
leave orphaned sandboxes with no owner — exactly the situation D1's status column exists
to prevent. Deletion is refused until no lane is `provisioning`. This closes the orphan
window instead of creating a permanent orphan-reaping obligation, at the cost of
deletion being briefly unavailable.

## Revision after build feedback (D5–D9)

D1–D4 settled where lane state lives, how long a savepoint lives, how lane keys are
derived, and when deletion is refused. They did not say what each seam *receives*, and
implementation hit four missing inputs in a row: the lane binding had no conversation,
the branch path already owned the rows a lane was supposed to create, phase 5 turned out
to supply nothing phase 4 needed, and fan-out had no way to find the source
conversation. A fifth gap sat above all of them — nothing constructed the service that
owns any of it.

The decisions below close those gaps. They are recorded here rather than patched
task-by-task because each one is a fork in the design, not a plan typo.

### D5 — A lane's conversation is minted with its runtime, and recorded on the lane

`ResumeTrigger` identifies the agent to re-engage but carries no conversation, while the
enqueue path (`EnqueueEnvDispatchChannelRun`) requires a channel, a chat session and a
source message. Lanes cannot share the source's: two lanes posting into one channel
would not be independent continuations, which is the whole point of fanning out.

So the runtime step produces the lane's execution identity *and* its conversation
together, as one `LaneBinding` (runtime, daemon, agent, channel, chat session, source
message), and all three conversation ids are recorded on the lane row alongside the
per-step ids from D1.

Recording rather than deriving is what makes interruption safe: a lane that died between
copying its channel and starting its run would otherwise copy a second channel on
recovery. Splitting the runtime and the conversation into separate steps was rejected
for the same reason — it adds a crash window whose only cure is another recorded id.

### D6 — On the branch path the rollout owns env/project/channel; the lane row records them

Branch dispatch already builds most of a lane before dispatch begins. Its reset phase
creates the env (deliberately with no sandboxes, for message dispatch), copies the
project subtree, and creates the channel with the copied conversation. Only the sandbox,
runtime and chat session are built during dispatch, and the sandbox is built by a live
filesystem clone.

Ownership stays there. The lane row is claimed and *pre-seeded* with the ids the reset
phase already produced, and materialization then runs only the steps whose ids are still
empty — which is exactly D1's "continue from the first unfilled step" rule, not an
exception to it.

Moving ownership into lane materialization was rejected:

- `resetOne` is shared with scratch dispatch, so carving branch out of it puts the live
  scratch path at risk for no gain.
- `rollbackRollout` is written against reset owning those rows; ownership cannot move
  without moving unwind too.
- The dispatch response contract (pinned before any of this changed) reports the env,
  project and channel that reset created, and the dispatch phase is best-effort by
  design — per-rollout errors, no rollback. A lane that owned those rows would have to
  unwind rows the response already promised.

What actually changes on this path is the sandbox: a live clone becomes a create from
the checkpoint's savepoint. That is where the design's benefit lives (a durable
savepoint, and N lanes sharing one snapshot instead of N fused clones), and it is the
only part that needs to move.

### D7 — Phases 4 and 5 are one server change, and `create_template` already exists

Phase 5 was written as though it introduced the savepoint-creating capability. It does
not: `create_template` is already a sandboxd capability, an allowed `sandbox_job` type,
and a server path that persists a `sandbox_snapshot` row and fills in its
`cube_snapshot_id` on completion. Phase 5's actual content is retiring `clone`.

That inverts the dependency the plan assumed. `CloneSandboxInstance` has exactly one
production caller — the branch trigger provisioning path — so removing `clone` without
switching that caller in the same change breaks the only live branch path. The two land
together.

It also means the rollout is ordered, not simultaneous:

1. The server stops enqueueing `clone` (branch provisioning switches to savepoint-backed
   create).
1. Migration 248 drops `'clone'` from the `sandbox_job` type CHECK. Landing this first
   would make a still-enqueueing server fail its inserts.
1. sandboxd drops the `clone` handler and capability. A sandboxd that still advertises
   `clone` while the server no longer sends it is harmless, so sandboxd goes last rather
   than in lockstep.

### D8 — A checkpoint records its source conversation

Fan-out from a checkpoint id alone has no way to find the conversation to continue.
`CopyProjectSubtree` copies issues and chat sessions but not channels — channels are
created separately with a roster — so a lane cannot derive one from the copied subtree.

The checkpoint therefore records the source conversation (channel plus the roster needed
to rebuild per-lane bindings), alongside the `env_id_map`, `sandbox_refs`, `db_snapshot`
and `resume_trigger` it already records. Requiring the caller to re-supply it was
rejected: a checkpoint that cannot be materialized from its id alone is not a savepoint
that outlives first use, which contradicts D2 and the re-expansion goal it exists to
serve.

Under D6 the live branch path does not need this, since its conversation comes from the
dispatch. So this is what standalone fan-out (`lane_count > 1` against a bare
checkpoint) requires, and it is scheduled with that capability rather than with the
branch wiring — the migration and the capability land together or not at all, so no
release advertises a fan-out it cannot serve.

### D9 — Production wiring is part of this change

The change as planned would have finished with every seam implemented, fully covered by
fakes, and unreachable: `EnvCheckpointService` was constructed nowhere outside tests,
the routes were gated off, and the handler field was never assigned. "Wiring the
automatic `CheckpointTrigger`" was already out of scope, which made it easy to miss that
*all* the wiring was.

The adapters are in scope: checkpoint repository, savepoint creator (over the existing
`create_template` path), savepoint reader, lane repository, lane materializer, and the
construction that assembles them. They live in `internal/service` behind narrow
generated-query interfaces, following the `diagnosisStateQueries` precedent, so they are
covered by unit tests in a package whose tests actually run — rather than in
`internal/handler`, whose `TestMain` exits successfully without a database and would
leave adapter tests silently unexecuted.

## Edge cases

| Case                                                           | Behavior                                                                                                                                           |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Savepoint `ready` but its Cube template was deleted externally | Lane → `failed` with a typed error; resume reports partial; the savepoint is marked failed so later resumes fail fast instead of retrying forever  |
| Source instance deleted between save and resume                | Irrelevant in snapshot mode — that is the point. `sandbox_snapshot.instance_id ON DELETE SET NULL` already decouples the savepoint from the source |
| Requested lane count of zero                                   | Rejected as invalid input, not a silent no-op                                                                                                      |
| Concurrent resume with the same lane key                       | Unique index decides; the loser reads the winner's row                                                                                             |
| Lane materializes but task enqueue fails                       | Lane is `ready` while the agent is not engaged → reported as a per-lane failed continuation, never as success                                      |
| All N lanes fail                                               | Resume returns a failed result, not an empty success                                                                                               |
| `pause_in_place` resumed twice                                 | Existing `draining` guard already rejects the second (`ErrTriggerTaskNotResumable`). Unchanged                                                     |
| Snapshot-mode create when the source instance is already gone  | Typed error at create; no partial checkpoint row left behind                                                                                       |
| Checkpoint deleted while a lane is `provisioning`              | Refused (D4)                                                                                                                                       |
| Branch dispatch retried after its rollout rows exist           | The lane claim loses, the existing row is read, and the pre-seeded ids (D6) mean no second env, project or channel is created                      |
| Lane interrupted between minting its channel and its run       | The recorded conversation ids (D5) are reused; recovery never copies a second channel                                                              |
| Fan-out requested against a checkpoint captured before D8      | Refused with a typed error rather than served with the source's own conversation, which would make lanes share a channel                           |

## Testing strategy

Service tests reuse the existing fake seams (`EnvCheckpointRepository`,
`SandboxInstanceSaver` and `Resumer`, `ResumeAgentRunner`, `InFlightTaskResolver`) plus
two new fakes: a savepoint creator and a lane repository.

- **Load-bearing invariant**: resuming into N lanes triggers exactly one snapshot *per
  source instance*, asserted on the savepoint creator's call count rather than on N.
  This is the single regression guard for the whole change.
- **Idempotency** is proven by a sqlc query test against the real unique index.
  In-memory fakes cannot demonstrate a database constraint, so a service-level test here
  would be vacuous.
- **Query tests**: lane insert `ON CONFLICT DO NOTHING` semantics; lane status
  transitions; cascade removal of lanes and savepoint ownership on checkpoint deletion;
  refusal to delete while a lane is `provisioning`.
- **State-machine round trip**: `running` → snapshot save → `complete` → resume N → N
  `ready` lanes with a fake daemon claim, mirroring the round-trip style established in
  `env-checkpoint-resume-trigger`.
- **Phase 4 guard**: a serialization/contract test pinning the branch dispatch response
  shape, so routing branch through resume cannot drift the AReaL-facing contract.
- **Warm session**: a task carrying a mid-flight recorded session continues it after a
  `pause_in_place` resume; lanes each start fresh and no two continue the same session.
- **Limitations, stated explicitly**: CI has no Cube cluster, so Cube-facing semantics
  rest on the prerequisite experiment plus fakes at the sandboxd job boundary.
  `multica/` is excluded from this repository, so Go tests run in the multica repo while
  this change owns the expectations.

## Risks / trade-offs

- **Lane-key anchor stability** → D3 is the only place outer-retry safety is decided;
  pin the anchor against a real dispatch id during implementation and cover it with a
  retry test.
- **Snapshot latency at real SWE scale is unmeasured** (the experiment used a 2 GB
  sandbox with a small working set) → the synchronous path degrades through the existing
  `save_timeout_ms` → `timed_out` → resume-rejected route, so a slow save costs a
  frontier, not correctness.
- **Cube template quota growth**, since savepoints now outlive first use → D2 cascade
  reclamation, partly offset by N lanes sharing one snapshot instead of taking N.
- **Phase 4 touches a live tree-search path** → phases 1 through 3 are
  behavior-preserving and land first, so the routing switch is the only behavioral step.
  It cannot be staged entirely alone, though: D7 binds it to the `clone` retirement, so
  the smallest behavioral unit is "branch provisioning switches to a savepoint *and*
  `clone` goes away", released in the order D7 sets out.
- **Adapter behavior is unverifiable in the build environment** (no Postgres, and the
  packages that would host DB-backed tests exit successfully without one) → the adapters
  are written behind narrow query interfaces so their logic is unit-tested with fakes,
  but every statement's interaction with the real schema is deferred to an environment
  with a database. This is the change's largest open risk and belongs in the
  verification record.
- **One extra table plus a sweeper** for stale `provisioning` lanes → accepted in
  exchange for removing an application lock and making orphans discoverable.

## Out of scope

- Forking inside a turn. Now known to be technically possible, since a restored clone
  carries live processes including a mid-turn agent, but it requires solving runtime
  identity across N clones sharing one `daemon.id` plus N duplicated in-flight model
  requests — precisely what the existing `pkill` avoids. Worth separate exploration if
  intra-turn branch points prove valuable.
- Savepoint deduplication across checkpoints (D2 migration path is additive).
- Wiring the automatic `CheckpointTrigger`; it remains `nil` in production. Note that
  everything *else* about wiring is in scope — see D9, which exists because this line
  was once the only mention of wiring and made a much larger gap easy to miss.
- Fleet-backed env checkpointing (still a typed error).
- AReaL frontier selection policy.
- Squad / multi-runtime trigger fan-out beyond the existing single-descriptor shape.

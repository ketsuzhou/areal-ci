# Brainstorm Summary

- Change: env-savepoint-consolidation
- Date: 2026-07-28

## Confirmed Technical Approach

### Lane records: one table serves idempotency and crash recovery

Lane idempotency (D9) and lane-provisioning crash recovery are the same problem, solved
by `env_checkpoint_lane` with `UNIQUE (checkpoint_id, lane_key)` and a status column
(`provisioning` / `ready` / `failed`).

- Resume inserts the lane row with `ON CONFLICT DO NOTHING`. Winning the insert means
  owning the lane and materializing it; losing means a lane already exists and its
  status decides the response (`ready` returns it, stale `provisioning` continues
  materialization, `failed` surfaces the failure).
- The unique index is the idempotency mechanism, so concurrent resume needs no
  application lock.
- The status column is the crash-recovery mechanism, so a sandbox orphaned by a mid-lane
  failure has a discoverable owner and can be swept.
- Per-lane materialization order: create instance from savepoint, copy project subtree,
  mint runtime, enqueue task.

Rejected: a JSONB lane array on `env_checkpoint` guarded by a row lock (serializes every
resume of a checkpoint, and lane state becomes hard to query or sweep); non-persistent
idempotency keys in Redis (lost on restart, and partial failures silently leak sandboxes
and Cube template quota on a shared cluster).

### Savepoint ownership is 1:1, reclaimed by cascade

A savepoint is owned by exactly one checkpoint and reclaimed when that checkpoint is
deleted. No reference counting and no `snapshot_refs` JSONB.

Rationale: two checkpoints never share a savepoint. Branch reuse is keyed on `env_id`,
so one frontier has one snapshot-mode checkpoint; once the source advances, the next
branch gets a new `env_id`, a new snapshot, and a new savepoint. Reference counting
would pay for an unreachable case — an integer count drifts silently on a crash between
checkpoint deletion and decrement, either leaking a Cube template or, worse, deleting a
savepoint a resumable checkpoint still needs.

Consistency with the re-expansion goal: **re-expanding a frontier means a new lane key
on the same checkpoint, not a second checkpoint.** This makes the branch reuse lookup
(`env_id` → the existing complete snapshot-mode checkpoint) load-bearing for
re-expansion.

Migration path if savepoint sharing is ever needed: add a join table so the reference
count becomes a `NOT EXISTS` query. That is an additive change, not a rewrite.

### Lane keys must derive from a stable per-dispatch anchor

Branch dispatch supplies only `env_id`, and its contract does not change, so the server
mints lane keys. They must be derived from an anchor that is stable across retries of
the same dispatch (the dispatch record id, or the caller's `event_ref` plus an ordinal).
Otherwise a retried branch dispatch doubles the lanes and D9's idempotency does not
actually protect the outer retry.

### Checkpoint deletion is refused while lanes are provisioning

Deleting a checkpoint whose lanes are still `provisioning` would cascade away the lane
rows and leave orphaned sandboxes with no owner. Deletion is refused until no lane is
`provisioning`, which removes the orphan window entirely rather than relying on a
sweeper to close it.

## Key Trade-offs and Risks

- One extra table plus a sweeper for stale `provisioning` lanes, accepted in exchange
  for removing an application-level lock and making orphans discoverable.
- 1:1 savepoint ownership forgoes savepoint deduplication; accepted because the sharing
  case is unreachable under `env_id`-keyed reuse and the migration path is additive.
- Refusing checkpoint deletion during provisioning makes deletion conditionally
  unavailable for a short window; accepted over a permanent orphan-reaping obligation.
- Lane-key derivation is the single point where outer-retry safety is decided; if the
  chosen anchor is not stable, fan-out silently doubles.
- CI has no Cube cluster, so Cube-facing semantics rest on the one-off experiment plus
  fakes at the sandboxd job boundary.

## Testing Strategy

- Service tests reuse the existing fake seams (`EnvCheckpointRepository`,
  `SandboxInstanceSaver` / `Resumer`, `ResumeAgentRunner`, `InFlightTaskResolver`) plus
  two new fakes: a savepoint creator and a lane repository.
- Load-bearing invariant: resuming into N lanes triggers **exactly one** snapshot per
  source instance, asserted by the savepoint creator's call count rather than N.
- Idempotency is proven by a sqlc query test against the real unique index; in-memory
  fakes cannot demonstrate a database constraint.
- Query tests: lane insert `ON CONFLICT DO NOTHING` semantics, lane status transitions,
  cascade removal of lanes and savepoint ownership on checkpoint deletion, and refusal
  to delete while a lane is `provisioning`.
- State-machine round trip: `running` → snapshot save → `complete` → resume N → N
  `ready` lanes, with a fake daemon claim, mirroring the round-trip test style from
  `env-checkpoint-resume-trigger`.
- Phase 4 guard: a serialization/contract test pinning the branch dispatch response
  shape.
- Warm session: a task carrying a mid-flight recorded session continues it after a
  `pause_in_place` resume.
- Explicit limitation: no Cube in CI, and `multica/` is excluded from this repository,
  so Go tests run in the multica repo while this change owns the expectations.

## Spec Patches

To write back into the delta specs:

1. `env-savepoint-fanout` — replace reference-counted reclamation with 1:1 savepoint
   ownership, and remove the "A shared savepoint is retained" scenario.
1. `env-savepoint-fanout` — add lane crash recovery: a stale `provisioning` lane is
   continued or failed, never duplicated.
1. `env-savepoint-fanout` — add rejection of a requested lane count of zero.
1. `env-savepoint-fanout` — add typed lane failure when the underlying savepoint
   template is gone, and fast-fail of subsequent resumes.
1. `env-savepoint-fanout` — add refusal to delete a checkpoint while a lane is
   `provisioning`.
1. `env-savepoint-fanout` — state re-expansion explicitly as a new lane key on the same
   checkpoint, and lane-key derivation from a retry-stable anchor.
1. `agent-continuation-seam` — add per-lane failed continuation when the environment is
   ready but the task enqueue fails.

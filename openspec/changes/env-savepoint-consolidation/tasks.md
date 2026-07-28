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
  state; keep the existing `save_timeout_ms` to timed-out path intact
- [x] 2.6 Keep `pause_in_place` create on exactly its current code path
- [x] 2.7 Query tests for the new column and savepoint ownership; service tests for
  snapshot-mode create (savepoint owned and ready, source still running),
  failed-savepoint handling, and unchanged pause-in-place create — service tests really
  run; the query tests are the skipped integration test noted above

## 3. Fan-out resume

- [ ] 3.1 Add a requested lane count and a lane key to the resume request, service
  signature, and result
- [ ] 3.2 Materialize one sandbox instance per lane from the checkpoint's savepoint,
  taking no additional snapshot of the source
- [ ] 3.3 Give each lane its own copy of the captured project subtree and its own agent
  runtime
- [ ] 3.4 Reject a requested lane count greater than one for `pause_in_place`, and keep
  its single-instance resume unchanged
- [x] 3.5 Migration and queries for `env_checkpoint_lane`:
  `UNIQUE (checkpoint_id, lane_key)`, a `provisioning` / `ready` / `failed` status,
  per-step ids (instance, project, runtime, task), and cascade on checkpoint deletion —
  the claim derives `workspace_id` from the checkpoint rather than accepting it, so a
  lane cannot escape its checkpoint's workspace
- [ ] 3.6 Claim a lane by inserting with `ON CONFLICT DO NOTHING`, and branch on the
  existing row's status when the insert loses: return a `ready` lane, continue a stale
  `provisioning` lane from its first incomplete step, surface a `failed` lane
- [ ] 3.7 Derive lane keys from an anchor that is stable across retries of the same
  branch request, and pin the chosen anchor against the dispatch record's actual stable
  id
- [ ] 3.8 Reject a requested lane count of zero as invalid input
- [ ] 3.9 Reject a checkpoint whose save status is not complete with a typed
  non-resumable error distinguishable from transient errors
- [ ] 3.10 Fail a lane with a typed error when its savepoint's underlying snapshot is
  gone, and mark the savepoint failed so later resumes fail fast
- [ ] 3.11 Report failure when every requested lane fails, rather than success with an
  empty lane set
- [ ] 3.12 Add a sweeper for lanes stuck in `provisioning`
- [ ] 3.13 Service tests: three lanes trigger exactly one snapshot per source instance
  (assert the savepoint creator's call count, not the lane count); a new lane key
  re-expands the frontier without creating a second checkpoint; pause-in-place fan-out
  rejected; timed-out checkpoint rejected; zero lane count rejected; all-lanes-failed
  reported as failure
- [ ] 3.14 Query tests against the real unique index: concurrent claims of one lane key
  create one lane; an interrupted lane is continued rather than duplicated — written as
  `TestEnvCheckpointLaneUniqueIndex_Integration`, but left unchecked because it has
  never run: without Postgres it self-skips, and a claim race is precisely what no fake
  can demonstrate

## 4. Route branch dispatch through resume

- [ ] 4.1 Serve branch-mode env dispatch by creating or reusing a `snapshot` checkpoint
  at the requested env and resuming it with the requested lane count
- [ ] 4.2 Keep the dispatch request and response contract, including rollout handles,
  byte-compatible with the pre-existing branch contract
- [ ] 4.3 Remove the now-dead direct branch provisioning path
- [ ] 4.4 Tests: branch dispatch contract unchanged; source env keeps running with its
  task undisturbed; each lane has its own runtime and subtree
- [ ] 4.5 Confirm no AReaL client change is required, and that `create_checkpoint` still
  returns a terminal save status synchronously

## 5. Retire the `clone` job type

- [ ] 5.1 Replace the sandboxd `clone` handler with `create_template` plus one `create`
  per lane
- [ ] 5.2 Remove `CloneSandboxInstance` and its remaining callers
- [ ] 5.3 Migration: drop `clone` from `sandbox_job_type_check`, with a down migration
  restoring it
- [ ] 5.4 Tests: lane creation from a savepoint template; no `clone` job is enqueued by
  any path
- [ ] 5.5 Note in the deployment plan that sandboxd and server must roll out together
  for this phase

## 6. Savepoint reclamation

- [ ] 6.1 Release a savepoint through the existing template deletion job when its owning
  checkpoint is deleted or expires
- [ ] 6.2 Refuse to delete a checkpoint while any of its lanes is still `provisioning`,
  with a typed error
- [ ] 6.3 Tests: deleting the owning checkpoint schedules savepoint deletion and removes
  its lane records; deletion is refused while a lane is `provisioning` and leaves the
  savepoint, lane, and sandbox intact

## 7. Session continuation policy

- [ ] 7.1 Resolve the prior session for a `pause_in_place` resume from the resumed task
  row's own recorded session, so an interrupted session continues instead of starting
  cold
- [ ] 7.2 Keep forked lanes on fresh sessions, and keep the snapshot-restored daemon
  stop and runtime identity reset that makes lane identity correct
- [ ] 7.3 Tests: a task with a mid-flight recorded session continues that session after
  pause-in-place resume; multiple lanes each start a fresh session and no two continue
  the same recorded session

## 8. Verification and documentation

- [ ] 8.1 Measure snapshot duration for a realistic SWE sandbox (large repository,
  larger memory) and record the result against the ~1.2s figure from the prerequisite
  experiment
- [ ] 8.2 Reconcile this change's capabilities with the unarchived sibling delta specs
  (`env-checkpoint-resume`, `env-checkpoint-resume-trigger`,
  `env-dispatch-sandbox-lifecycle`) before archive
- [ ] 8.3 Update the multica environment protocol document so branch is described as a
  fan-out of checkpoint resume, and correct its statement that the API provides no
  snapshot or fork semantics
- [ ] 8.4 Record the intra-turn fork finding (a restored clone carries live processes)
  as explicitly out of scope, with the runtime-identity and duplicated-request reasons

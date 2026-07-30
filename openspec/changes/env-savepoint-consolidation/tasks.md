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

Migration numbers were 244/245/246 until a rebase onto `upstream/dev` brought in
`244_research_fleet` and `245_research_fleet_agent_indexes`. Git does not report that as
a conflict, since the filenames differ, so a colliding version number survives a clean
rebase and only surfaces when the runner sees two migrations claiming one version. Any
further rebase during this change has to re-check the highest number and, because
`248_sandbox_job_retire_clone` rewrites `sandbox_job_type_check`, whether an incoming
migration redefined that constraint — dropping values another migration added is the
mistake 186 made and 187 repaired.

Verification limit, decided explicitly: Postgres is **not** installed in the build
environment. The `*_Integration` query tests are written but self-skip without
`DATABASE_URL` and must not be reported as passing. Migrations 246/247/248, the
`UNIQUE (checkpoint_id, lane_key)` claim race, and the `ON DELETE CASCADE` reclamation
stay unverified until run against a real database.

That risk is **discharged** as of run 30422501039 on commit `cd704761b` (branch
`feature/20260728/env-savepoint-consolidation`, draft PR #1370): the `backend` job
applied `246_env_checkpoint_save_mode`, `247_env_checkpoint_lane` and
`248_sandbox_job_retire_clone` to a real Postgres and passed `go test -p 1 ./...`.
`DATABASE_URL` is job-level env, so the `*_Integration` tests ran rather than
self-skipping, and `internal/handler` took 40s — locally it exits immediately without a
database, so that duration is itself the evidence the test step had one.

The mechanism: `multica`'s `.github/workflows/ci.yml` runs a `backend` job with real
Postgres and Redis that applies the migrations (`cmd/migrate up`) and runs
`go test -p 1 ./...`, so the four packages that execute nothing locally do execute
there. It triggers on `pull_request` to `dev` only — pushing to `dev` reaches the deploy
workflow, not this one — so verification runs on a draft PR against `dev` while the
change is incomplete. Opened as LRM-Teams/multica#1370, from a branch on `lrm` rather
than a fork, since the frontend job fetches `origin dev`.

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
  `TestMigration246AddsSaveModeAndCheckpointOwnedSavepoints`, which fails if any
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

- [x] 3.1 Add a requested lane count and a lane key to the resume request, service
  signature, and result — the HTTP body is optional, so a caller sending none still gets
  one lane anchored on the checkpoint id, and `lanes` is `omitempty` so a pause-in-place
  response is unchanged

- [x] 3.2 Materialize one sandbox instance per lane from the checkpoint's savepoint,
  taking no additional snapshot of the source

- [x] 3.3 Give each lane its own copy of the captured project subtree and its own agent
  runtime — and its own conversation (channel, chat session, source message), because
  the enqueue path requires all three and lanes sharing a channel would not be
  independent; migration 247 gained those columns so an interrupted lane does not copy a
  second channel on recovery

- [x] 3.4 Reject a requested lane count greater than one for `pause_in_place`, and keep
  its single-instance resume unchanged — a checkpoint with an empty `save_mode` is a
  pre-change row and is refused fan-out on the same grounds

- [x] 3.5 Migration and queries for `env_checkpoint_lane`:
  `UNIQUE (checkpoint_id, lane_key)`, a `provisioning` / `ready` / `failed` status,
  per-step ids (instance, project, runtime, task), and cascade on checkpoint deletion —
  the claim derives `workspace_id` from the checkpoint rather than accepting it, so a
  lane cannot escape its checkpoint's workspace

- [x] 3.6 Claim a lane by inserting with `ON CONFLICT DO NOTHING`, and branch on the
  existing row's status when the insert loses: return a `ready` lane, continue a stale
  `provisioning` lane from its first incomplete step, surface a `failed` lane

- [x] 3.7 Derive lane keys from an anchor that is stable across retries of the same
  branch request, and pin the chosen anchor against the dispatch record's actual stable
  id — anchor is `env_dispatch_request.idempotency_key` (a request-body field the
  handler already validates as a UUID when present). Nothing else in the request
  qualifies: server-minted ids are fresh per attempt, and the `env_dispatch_request` row
  is written only after the dispatch completes, so a retry arriving mid-flight has no
  prior row to recover the anchor from. **Enforcement is deferred to 4.1, by user
  decision.** The AReaL client sends no idempotency key at all (verified: the field
  appears nowhere in `customized_areal/`), and lane keys do not govern branch dispatch
  until the branch path is served by checkpoint resume, so requiring it now would reject
  live branch dispatches for a property nothing yet relies on — and `proposal.md`
  promises no client contract change. Two ordering constraints carry forward: the client
  must send the key **before** the server requires it, and making it required is a
  client-visible contract change that must be reflected in `proposal.md`, task 4.5
  (which currently asserts no AReaL client change is required), and the protocol doc
  (task 8.3) when it lands. `TestBranchDispatchStillAcceptsAKeylessRequest` is the
  tripwire guarding the deferral; it was mutation-checked and fails the moment the
  validation is added

- [x] 3.8 Reject a requested lane count of zero as invalid input — validated before the
  checkpoint is loaded, so a bad count cannot have a side effect

- [x] 3.9 Reject a checkpoint whose save status is not complete with a typed
  non-resumable error distinguishable from transient errors —
  `ErrCheckpointNotResumable` maps to 409 and `ErrLaneCountInvalid` to 400, since one is
  permanent and the other is worth retrying with a corrected request

- [x] 3.10 Fail a lane with a typed error when its savepoint's underlying snapshot is
  gone, and mark the savepoint failed so later resumes fail fast

- [x] 3.11 Report failure when every requested lane fails, rather than success with an
  empty lane set — the first lane's cause is wrapped so a typed failure stays
  recognizable through the summary error

- [x] 3.12 Add a sweeper for lanes stuck in `provisioning` — global scheduler job on a
  5-minute cadence, failing lanes 15 minutes past their last progress. Deviates from the
  plan in one respect: the plan's `ListStaleProvisioningEnvCheckpointLanes` +
  per-row-mark shape was replaced by a single `UPDATE ... RETURNING`
  (`SweepStaleProvisioningEnvCheckpointLanes`), because `MarkEnvCheckpointLaneFailed`
  carries no status guard, so list-then-mark would fail a lane its owner drove to
  `ready` in between. No fake can reproduce that race, so
  `TestLaneSweepFailsStaleLanesInOneStatement` asserts the atomicity statically instead.
  The staleness cutoff derives from the scheduler's plan time (floored from the database
  clock) rather than the process clock, so it cannot drift with app/DB clock skew and
  errs toward sweeping late. Sweeping only fails the row; it does not release sandboxes,
  and the recorded error says so — reclamation is 6.x's job. Note `sqlc` statically
  caught an ambiguous `updated_at` between the UPDATE target and its candidate subquery,
  which is the class of error this environment otherwise could not catch

- [x] 3.13 Service tests: three lanes trigger exactly one snapshot per source instance
  (assert the savepoint creator's call count, not the lane count); a new lane key
  re-expands the frontier without creating a second checkpoint; pause-in-place fan-out
  rejected; timed-out checkpoint rejected; zero lane count rejected; all-lanes-failed
  reported as failure. Closed out with `TestSnapshotCheckpointRoundTripRunningToLanes`,
  which is the only one of these that walks capture and resume together: it feeds resume
  the savepoints capture actually produced, so a disagreement over ids or over
  `save_mode` fails here instead of in production (mutation-checked by making capture
  drop `save_mode`)

- [x] 3.14 Query tests against the real unique index: concurrent claims of one lane key
  create one lane; an interrupted lane is continued rather than duplicated — written as
  `TestEnvCheckpointLaneUniqueIndex_Integration`

  Ticked on CI evidence, and the evidence is decisive rather than inferred. A green run
  only proves the package compiled and nothing failed -- a self-skipping test also
  reports `ok` -- so the test body was confirmed to execute by breaking one assertion on
  purpose (`require.Len(t, lanes, 1)` -> `99`) and pushing it. CI run 30425009029 failed
  with `--- FAIL: TestEnvCheckpointLaneUniqueIndex_Integration` at
  `env_checkpoint_lane_query_test.go:82`, which is inside the transaction after the two
  competing claims. The probe commit was then removed from the branch.

  So the claim race, the `UNIQUE (checkpoint_id, lane_key)` index, and the
  interrupted-lane continuation are verified against a real Postgres, not asserted.

## 3b. Production wiring

Added during build, by user decision, after discovering the plan never wired any of
phases 1–3 into production. Every `NewEnvCheckpointService` call in the plan is a test;
`Handler.EnvCheckpointService` is assigned nowhere outside `env_checkpoint_test.go`, and
the routes are gated off by `ENV_CHECKPOINTS_ENABLED`. Without this phase the change
lands as fully-tested unreachable machinery, and phase 5's `clone` retirement would
break the only live branch path, whose replacement sits behind the unwired service.

Everything here is DB-backed adapter code, so in this environment it is verifiable only
by compilation, interface satisfaction, and pure-helper unit tests —
`internal/handler`'s `TestMain` exits 0 without Postgres. This is the phase whose
deferred verification carries the most risk.

The adapters live in `internal/service` behind narrow generated-query interfaces (the
`diagnosisStateQueries` precedent), not in `internal/handler`, so their logic is covered
by tests that actually execute here. Design D9/D14.

- [x] 3b.1 `EnvCheckpointRepository` adapter over the checkpoint queries — `save_mode`
  survives the write, a legacy row without one resolves to `pause_in_place` (both
  mutation-checked), every query is workspace-scoped, and a malformed id is refused
  before it can reach the database as a zero UUID

- [x] 3b.2 `SavepointCreator` adapter reusing the existing `create_template` path
  (`CreateSandboxSnapshotTemplate` already persists a `sandbox_snapshot` row, enqueues
  the job, and has its `cube_snapshot_id` filled in on completion). Binds the snapshot
  to its owning checkpoint before the job runs, leaves the source running, waits for a
  terminal status, and fails a row whose job was lost rather than leaving it `creating`
  forever. The ownership binding and the wait are both mutation-checked

- [x] 3b.3 `SavepointReader` and `EnvCheckpointLaneRepository` adapters over the queries
  from 2.x/3.5. Two properties the interfaces only describe are pinned by
  mutation-checked tests: a lost claim is an ordinary outcome rather than an error, and
  an unrecorded step reaches the database as NULL so `COALESCE` keeps what an earlier
  step wrote

- [x] 3b.4 `LaneMaterializer` adapter. Blocked during the first attempt: the interface
  gave `ProvisionLaneAgent` no way to find the source conversation, and
  `CopyProjectSubtree` copies issues and chat sessions but not channels, so it cannot be
  derived. Resolved by design D6 (the branch path keeps owning env/project/channel and
  the lane row is pre-seeded with them) and D8 (a checkpoint records its source
  conversation, shipping with standalone fan-out). This task now covers only the
  branch-path shape: create the lane instance from the savepoint's Cube template and
  provision the lane's runtime and chat session, reusing the existing env-dispatch
  provisioning helpers.

  Landed as `internal/service/lane_materializer.go` behind a four-method
  `LaneMaterializerDeps`. Two guards carry the weight and were mutation-checked:
  creating with an empty Cube template would succeed against the node default and hand
  back a healthy-looking sandbox that lost the captured state, so an empty or non-ready
  savepoint is refused as `ErrSavepointGone`; and a lane with no channel is refused as
  `ErrLaneConversationUnavailable` rather than reusing the source's, which would
  collapse independent continuations into one thread. `LaneRuntimeInput` gained the
  lane's recorded project/env/channel so the runtime step acts on the pre-seeded
  conversation (D6) instead of minting a second one.

  Two things the adapter deliberately does not solve, both landing with D8's fan-out
  columns rather than here: `CopyLaneProjectSubtree` creates the lane env with no parent
  and in the self-play domain because a checkpoint does not record the env it was taken
  from (branch dispatch never reaches this step, so only standalone fan-out is
  affected); and `ProvisionLaneAgentRuntime` needs a per-lane env-dispatch binding row
  plus a lane channel, so the handler-side implementation is a typed refusal for now,
  which 3b.6 makes unreachable from the API

- [x] 3b.5 Construct the service in the handler (per-request, matching how
  `EnvDispatchService` and the lifecycle service are built) while keeping the injected
  fake as the test escape hatch, and record what the feature flag now gates.

  Most seams turned out to already have production implementations, which is why this
  landed smaller than feared: `*db.Queries` satisfies `InFlightTaskResetter`,
  `envDispatchDepsAdapter` satisfies `ForkedRuntimeEnqueuer`, the daemon hub satisfies
  `TaskWakeupNotifier`, `ListInFlightTasksForProject` already existed as a query, and
  `envSandboxLifecycleDepsAdapter` already enqueues sandbox jobs. Two seams were
  genuinely new and are tested in `internal/service`: `NewProjectSnapshotReader`
  (captures the same subtree `CopyProjectSubtree` copies -- issues, chat sessions, and
  each session's messages -- in a versioned envelope whose collections are `[]` not
  `null`, since the snapshot is handed straight back over the API) and
  `NewInFlightTaskResolver` (drops rows that could never be triggered, because
  continuation resets by task+runtime and a captured trigger missing either only fails
  once someone tries to resume it). Thin saver/resumer adapters over the lifecycle
  service propagate the error that records a checkpoint failed or timed out.

  Two traps worth naming. A nil `*daemonws.Hub` stored in a `TaskWakeupNotifier` would
  be a non-nil interface and the wake fast-path would call through a nil receiver, so
  the hub is only installed when present. And `ENV_CHECKPOINTS_ENABLED` now gates a
  service that is actually reachable -- before this the endpoints were dead even with
  the flag on, because nothing ever set the field. No existing test asserts the "not
  configured" 503 through `handler.New`, so nothing regresses on that path.

  Verification: `go build ./...`, `go vet ./internal/... ./cmd/...`, and the new
  `internal/service` tests with mutation checks on the workspace filter, the message
  capture, and the untriggerable-row skip. The construction itself is unverifiable here
  -- `internal/handler` has no runnable tests without Postgres

- [x] 3b.6 Refuse `lane_count > 1` against a checkpoint that recorded no source
  conversation, with a typed error. Design D8/D13: the service-level fan-out from phase
  3 is complete, but a bare checkpoint cannot name the conversation its lanes should
  continue, and serving it from the source's own channel would make the lanes share one
  — the exact opposite of independent continuations. The migration that records the
  source conversation ships with the standalone fan-out capability, so no release
  advertises a fan-out it cannot serve.

  Implemented in `resumeSnapshotLanes` as `ErrCheckpointNotResumable` (409): the request
  is fine, the checkpoint is what cannot serve it. A single lane is exempt because its
  caller supplies the conversation -- branch dispatch pre-seeds the lane row (D6) -- so
  the branch path is unaffected.

  The rule reads `cp.SourceChannelID`, a field with no column yet, rather than an
  unconditional refusal. Every checkpoint written today therefore reports none and
  fan-out is refused, and the capability arrives by populating the field when D8's
  migration lands, with no temporary guard for someone to remember to remove. The
  round-trip test sets the field on its stored checkpoint to keep exercising the state
  machine behind the boundary, which is also the one place the dependency is visible.
  Mutation-checked by removing the refusal

## 4. Route branch dispatch through resume

Re-scoped by design D6/D11 and D7/D12. As originally written, 4.1 would have overwritten
the env and project the reset phase already created, orphaning both and leaving the
copied channel attached to the abandoned project — the copied conversation is the entire
point of branch+message, so that is not a reroute but a duplication. And 4.3's deletion
of the direct path removes the only production caller of `CloneSandboxInstance`, which
phase 5 retires, so 4.x and 5.x are one change released server → migration 248 →
sandboxd.

- [x] 4.1 Serve branch-mode env dispatch from a `snapshot` checkpoint at the requested
  env: claim a lane, pre-seed it with the env, project and channel the reset phase
  created, and build the sandbox from the checkpoint's savepoint instead of a live
  filesystem clone. Creating or reusing the checkpoint is keyed on `env_id`, so
  re-expansion is a new lane key on the same checkpoint (D2). Capture runs once for the
  whole group **before** the reset fan-out, not per rollout: rollouts reset and dispatch
  concurrently (`sem`/`wg` in `Dispatch`), so per-rollout capture would snapshot the
  same source once per rollout and race on creating the owning checkpoint, and capture
  is the only step that fails for the whole group at once, so failing early avoids
  rolling back N envs/projects/channels. The seam is required, not optional — without it
  branch dispatch refuses rather than falling back to the clone that 5.x removes
- [x] 4.2 Keep the dispatch request and response contract, including rollout handles,
  byte-compatible with the pre-existing branch contract — the pin now exists
  (`internal/apicontract`, byte-level plus the four fields the AReaL client hard-depends
  on: `project_id`, `channel_id`, `rollouts[0].env_id`, and the absence of
  `rollouts[].error` on success; mutation-checked). Left unticked because the pin only
  proves compatibility once 4.1 has actually rerouted the path. Note the pin
  deliberately does **not** live in `internal/handler`, whose `TestMain` exits 0 without
  Postgres and would leave it silently unexecuted here. Ticked now that 4.1 reroutes the
  path: the pin still passes byte-for-byte
- [x] 4.3 Remove the now-dead direct branch provisioning path (`CloneSandboxInstance`,
  its input type and tests; `provisionEnvDispatchAgentBranch` now calls
  `lifecycle.Create` with the savepoint template)
- [x] 4.4 Tests: branch dispatch contract unchanged; source env keeps running with its
  task undisturbed; each lane has its own runtime and subtree. In
  `env_dispatch_branch_savepoint_test.go` and `branch_savepoint_provider_test.go`,
  mutation-checked: dropping the template pass-through, the fail-closed guard, the lane
  pre-seed, the failure settle, moving capture after the fan-out, skipping the channel's
  peers, reusing an unfinished capture, and treating a missing savepoint as an empty
  template all fail a test
- [x] 4.5 Confirm no AReaL client change is required, and that `create_checkpoint` still
  returns a terminal save status synchronously. Confirmed: the request/response pin in
  `internal/apicontract` is unchanged, and `EnvCheckpointService.Create` still blocks to
  a terminal `save_status`, which is what the branch capture path depends on to know the
  savepoint is usable. Server-side `idempotency_key` enforcement stays deferred (task
  3.7) — the tripwire test remains

## 5. Retire the `clone` job type

- [x] 5.1 Replace the sandboxd `clone` handler with `create_template` plus one `create`
  per lane. Per D7/D12 `create_template` already existed end to end, so this removed
  `cloneCubeSandbox`, its dispatch case, the `clone` capability, and the
  `source_external_id`/`create_payload` payload fields only clone used
- [x] 5.2 Remove `CloneSandboxInstance` and its remaining callers. The second caller the
  plan had missed: copying a branch channel copies every roster member's binding with
  its source sandbox (`env_dispatch_channel_copy.go`), so a peer mentioned later in the
  branch also cloned — on the mention path, under a 5s `waitCtx`, which cannot fit a
  snapshot. Resolved by capturing every ready sandbox in the source channel at dispatch
  time and having the mention path look one up. Also removed a stale
  `case "create", "clone"` job-completion branch in `internal/handler/sandbox.go`
- [x] 5.3 Migration: drop `clone` from `sandbox_job_type_check`, with a down migration
  restoring it (`248_sandbox_job_retire_clone`). The up migration keeps
  `create_template`/`delete_template`/`exec`/`message`, which is the mistake migration
  187 existed to repair, and the test asserts that
- [x] 5.4 Tests: lane creation from a savepoint template; no `clone` job is enqueued by
  any path. `TestNoPathEnqueuesOrHandlesACloneJob` walks the server tree for the two
  shapes that carry a job type (the enqueue argument and sandboxd's dispatch switch)
  rather than the bare word, which also appears in `git clone`; it found the stale
  handler branch above, and a mutation restoring the sandboxd case fails it
- [x] 5.5 Note the release order in the deployment plan. Corrected by design D7/D12: the
  replacement (`create_template`) already exists end to end, so this phase only removes
  `clone`, and the order is server (stops enqueueing) → migration 248 (drops the CHECK
  value) → sandboxd (drops the handler and capability). Lockstep is not required;
  landing the migration or sandboxd first is what breaks

## 6. Savepoint reclamation

- [x] 6.1 Release a savepoint through the existing template deletion job when its owning
  checkpoint is deleted or expires

  Scope addition: there was no deletion path to hook into. `env_checkpoint.go` exposed
  only create, get, list, and resume, and `env_checkpoint.sql` had no `DELETE`, so this
  task creates `DeleteEnvCheckpoint` (query, repository method, `Delete` on the service,
  and `DELETE /api/v1/env-checkpoints/{checkpointID}`). Expiry is still absent --
  nothing schedules deletion on a clock, so this covers the explicit delete only.

  Savepoints are released *before* the row is deleted. The row is the only record that
  the templates exist, so deleting it first would leak them on a failed release with
  nothing left to retry from. Releasing reuses the `delete_template` job that
  `DeleteSandboxSnapshot` already enqueues, extracted into
  `Handler.scheduleSnapshotTemplateDeletion` rather than duplicated: a savepoint is an
  ordinary `sandbox_snapshot` with an owning checkpoint, so there is one reclamation
  path. `savepointReleaserAdapter` handles the states that path can find a savepoint in
  -- already gone succeeds (failing would pin the checkpoint on a savepoint that no
  longer exists), already `deleting` is not re-queued, `creating` is refused rather than
  raced, and one that never reached Cube has its row dropped instead of queuing a job
  for a template that was never made.

  A snapshot checkpoint that owns savepoints and has no releaser installed is refused,
  not deleted, since deleting it would leak every template it owns. `pause_in_place`
  owns none and needs no releaser.

- [x] 6.2 Refuse to delete a checkpoint while any of its lanes is still `provisioning`,
  with a typed error

  `ErrCheckpointHasProvisioningLanes`, mapped to 409 rather than 4xx-permanent: retrying
  once the lanes settle succeeds. Terminal lanes do not block -- a ready lane's sandbox
  belongs to its env and a failed lane's was already reclaimed.

- [x] 6.3 Tests: deleting the owning checkpoint schedules savepoint deletion and removes
  its lane records; deletion is refused while a lane is `provisioning` and leaves the
  savepoint, lane, and sandbox intact

  `internal/service/env_checkpoint_delete_test.go` runs locally and carries four
  mutation-checked assertions: skipping the provisioning-lane guard, deleting the row
  before releasing, silently deleting without a releaser, and treating terminal lanes as
  blocking all fail their test. `env_checkpoint_repo_test.go` pins that the delete
  carries the workspace, so a checkpoint id from another workspace cannot cascade
  someone else's lanes away.

  The endpoint tests (`internal/handler/env_checkpoint_test.go`) and the releaser-state
  tests (`internal/handler/env_checkpoint_release_test.go`) sit in `internal/handler`,
  whose `TestMain` exits without Postgres, so they are CI-verified rather than locally
  verified -- see the verification note above.

## 7. Session continuation policy

- [x] 7.1 Resolve the prior session for a `pause_in_place` resume from the resumed task
  row's own recorded session, so an interrupted session continues instead of starting
  cold

  Root cause confirmed against the schema: `ResetInFlightTaskForResume` moves the row
  `draining` -> `pending` and clears only `started_at`/`dispatched_at`/`claimed_at`, so
  `session_id` survives on the same row; `GetLastTaskSession` and
  `GetLastChatTaskSession` match only terminal rows (`acked`, or `suppressed` with
  `followup_interrupt`), so neither can see it.

  `service.OwnPinnedSession` reads it, consulted ahead of the cross-task fallbacks in
  both `populateAgentInboxWorkContext` and `populateAgentInboxChatContext`.

  Deviation: the planned `GetResumedTaskOwnSession` query was not needed. The claim path
  already holds the `agent_inbox_event` row, so this is a pure function over it -- no
  new query, no sqlc transplant, and the policy lives in `internal/service` where tests
  actually run. The chat path had to take the claiming runtime as a parameter, since it
  only had the row's own runtime and comparing that against itself would have made the
  runtime guard a no-op.

  Three guards, each mutation-checked: terminal rows are left to the filtered lookups
  (which exclude poisoned outcomes such as `iteration_limit` and `api_invalid_request`
  -- a filter this must not bypass), the claiming runtime must match, and
  `force_fresh_session` still wins.

- [x] 7.2 Keep forked lanes on fresh sessions, and keep the snapshot-restored daemon
  stop and runtime identity reset that makes lane identity correct

  Lanes stay cold by construction: a lane's task row is new with no `session_id` and its
  own runtime, so neither the own-row read nor the cross-task lookups can hand it the
  source's session. `buildStartRuntimeInCubeCode`'s `pkill` plus `daemon.id` rewrite is
  unchanged; its comment now records that fan-out sharpens it, since several lanes
  restore the same frozen `daemon.id` from one savepoint and would otherwise register as
  one runtime.

- [x] 7.3 Tests: a task with a mid-flight recorded session continues that session after
  pause-in-place resume; multiple lanes each start a fresh session and no two continue
  the same recorded session

  `internal/service/session_continuation_test.go`, which runs locally. Dropping any of
  the three guards fails a test. The "no two lanes share a session" property is covered
  as the cross-runtime refusal plus the empty-session lane case rather than by spinning
  up N rows, since the runtime guard is what makes it true.

## 8. Verification and documentation

- [ ] 8.1 Measure snapshot duration for a realistic SWE sandbox (large repository,
  larger memory) and record the result against the ~1.2s figure from the prerequisite
  experiment

  **Left unticked deliberately: needs a human operator with Cube access.** This
  environment has no Cube base URL or credential and cannot reach the Cube API, so there
  is no honest way to produce the number. `.comet/snapshot-latency.md` carries the
  procedure, a table to fill in, and what to do if it comes back slow.

  What the investigation *did* settle is the threshold, which the plan did not state.
  The binding limit is not the server's 15-minute `branchSavepointSaveTimeout`; it is
  the AReaL client's 120s `httpx` timeout (`MulticaEnvDispatchClient` defaults
  `timeout=120.0`, and the server sets no `WriteTimeout`), combined with capture being
  serial -- `EnvCheckpointService.Create` loops over `SandboxRefs` one at a time, and
  eager-all captures the whole roster. So the passing condition is
  `per_snapshot_duration × roster_size < 120s`, and exceeding it produces a dispatch
  that looks failed to AReaL while the server still creates the checkpoint and lanes.

  Parallelizing the capture loop is the first mitigation if the measurement is slow, and
  is deliberately not implemented ahead of it: at the measured 1.2s it buys nothing and
  adds concurrent snapshot load plus cross-lane error aggregation.

- [x] 8.2 Reconcile this change's capabilities with the unarchived sibling delta specs
  (`env-checkpoint-resume`, `env-checkpoint-resume-trigger`,
  `env-dispatch-sandbox-lifecycle`) before archive

  Table in `proposal.md`, one row per sibling requirement. Correction to the task's
  premise: `env-checkpoint-resume` is not its own change -- its spec lives under
  `env-dispatch-sandbox-lifecycle/specs/env-checkpoint-resume/`.

  Two sibling requirements are contradicted outright rather than extended, and both need
  correcting when those siblings archive: "Resume ... MUST not expose immutable
  branch/fork semantics" and the bridge requirement's "Branch ... rather than a live
  fork" plus "True live-state fork ... out of scope for v1". This change adds exactly
  the primitive both ruled out.

  One near-miss worth recording: retiring `CloneSandboxInstance` does *not* contradict
  "Sandbox lifecycle service reuse", which enumerates create, save, resume, delete, and
  reconfigure -- clone was never in it.

- [x] 8.3 Update the multica environment protocol document so branch is described as a
  fan-out of checkpoint resume, and correct its statement that the API provides no
  snapshot or fork semantics. The `idempotency_key` row records the client contract
  honestly: lane keys derive from it, but the server does not yet reject a branch
  dispatch without one (task 3.7), so the row says so rather than promising enforcement
  that is not there

- [x] 8.4 Record the intra-turn fork finding (a restored clone carries live processes)
  as explicitly out of scope, with the runtime-identity and duplicated-request reasons,
  and note that the `pkill` in `buildStartRuntimeInCubeCode` is therefore load-bearing
  correctness rather than hygiene

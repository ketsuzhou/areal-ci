---
change: env-savepoint-consolidation
design-doc: docs/superpowers/specs/2026-07-28-env-savepoint-consolidation-design.md
base-ref: 2529da6cd28c34d2adbf446a7179d736db53aa92
---

# Env Savepoint Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one immutable savepoint primitive serve both env-checkpoint resume and
env-dispatch branch, so a tree-search frontier can be expanded repeatedly and expanding
into N lanes costs one Cube snapshot instead of N.

**Architecture:** `env_checkpoint` gains a `save_mode` (`pause_in_place` | `snapshot`);
a `snapshot` checkpoint owns `sandbox_snapshot` rows as savepoints and leaves the source
running. Resume fans out into N lanes recorded in a new `env_checkpoint_lane` table
whose `UNIQUE (checkpoint_id, lane_key)` index *is* the idempotency mechanism and whose
`status` column *is* the crash-recovery mechanism. Agent re-engagement moves behind the
existing `ResumeAgentRunner` seam with one strategy per save mode.
`env_dispatch(mode="branch")` is then routed internally to
`resume(checkpoint, lane_count=N)` with no change to the AReaL-facing contract.

Rationale, the prerequisite Cube experiment, decisions D1–D4, edge cases, and the
testing strategy live in the Design Doc
(`docs/superpowers/specs/2026-07-28-env-savepoint-consolidation-design.md`) and the
change's `design.md`. This plan does not restate them.

**Tech Stack:** Go 1.26 (chi, pgx/v5, sqlc), PostgreSQL, Cube sandbox API via
`sandboxd`, Python AReaL client (read-only in this change).

______________________________________________________________________

## Global Constraints

### CRITICAL: this change spans two separate git repositories

Every task below is prefixed `[multica]` or `[areal]`. Getting this wrong will produce
commits that cannot land.

- **`[areal]`** — the repository at `/workspaces/leagent/backend/areal`, HEAD
  `2529da6cd28c34d2adbf446a7179d736db53aa92`. Owns the OpenSpec change, delta specs, the
  Design Doc, this plan, the AReaL client, and
  `customized_areal/tree_search/agents/multica_environment_protocol.md`.
- **`[multica]`** — a **different git repository** checked out at
  `/workspaces/leagent/backend/areal/multica`. It is deliberately excluded from the
  areal repo: areal is itself a submodule of the `leagent` superproject, so its git dir
  is `/workspaces/leagent/.git/modules/backend/areal`, and line 7 of that directory's
  `info/exclude` holds `/multica/`. Confirm with `git check-ignore -v multica`;
  `git ls-files multica` in the areal repo returns nothing. All Go source, SQL
  migrations, sqlc queries, and Go tests land **here**.
- **Work happens in a dedicated worktree, not the main multica checkout:**
  `/workspaces/leagent/backend/areal/multica/.worktrees/env-savepoint-consolidation`, on
  branch `feature/20260728/env-savepoint-consolidation`, based on `upstream/dev` at
  `cf7016107`. Every `[multica]` command below runs there.
- **Watch the remotes.** `origin` is a personal fork (`ketsuzhou/multica`) whose `dev`
  is over a thousand commits stale; `upstream` and `lrm` are both `LRM-Teams/multica`.
  The base for this work is **`upstream/dev`**, never `origin/dev`. `upstream/dev` also
  moved during this session, so re-check it before rebasing.
- Per `tasks.md` 0.1: the Go implementation lands in the multica repo while this
  OpenSpec change owns the contract, specs, and test expectations.
- **The main multica checkout has unrelated uncommitted work** (`db_bridge/*`,
  `server/internal/service/interaction_dag_seams.go`,
  `server/internal/service/training.go`,
  `server/pkg/db/generated/training_tasks_manual.go`, plus two untracked `*.md` notes).
  The worktree above already isolates this change from it, so that work must be left
  exactly as it is — do not stash, reset, or commit it. Never `git add -A` in either
  checkout.
- Never commit across both repos in one command. Each task's commit step names its repo
  explicitly.

### Migration conventions `[multica]`

- Path: `multica/server/migrations/NNN_snake_case_name.up.sql` + a matching `.down.sql`.
  Both files are required.
- **Highest existing number at the working base `cf7016107` is 243**, with 242 unused.
  Numbers 230, 231, and 232 each have multiple files, so collisions are tolerated
  historically — but this plan uses fresh sequential numbers above the highest to keep
  ordering unambiguous:
  - **244** — `244_env_checkpoint_save_mode.{up,down}.sql` (Phase 2)
  - **245** — `245_env_checkpoint_lane.{up,down}.sql` (Phase 3)
  - **246** — `246_sandbox_job_drop_clone.{up,down}.sql` (Phase 5)
- Before writing each migration, re-run
  `ls server/migrations | grep -oP '^\d+' | sort -n -u | tail -5` in the worktree —
  `upstream/dev` moves during the day and another branch may have taken 244.
- Use `TIMESTAMPTZ`, and `IF EXISTS` / `IF NOT EXISTS` where it does not hide a real
  error.

### sqlc regeneration `[multica]`

**Yes, regeneration is required after editing any file in
`multica/server/pkg/db/queries/`.**

- Canonical command: `make sqlc` from `/workspaces/leagent/backend/areal/multica`, which
  runs `cd server && sqlc generate` (see `multica/Makefile:326`).
- `sqlc` **is usable here**, contrary to this plan's original assumption. It reads the
  schema from `migrations/` statically (see `sqlc.yaml`) and needs no database, so it
  works even with Postgres declined. Install with
  `go install github.com/sqlc-dev/sqlc/cmd/sqlc@v1.31.1` — pin that version, because it
  is what generated the checked-in code, and v1.29 produces a far larger diff. `mise`
  swaps the Go toolchain per directory, so `sqlc` on `PATH` inside the multica worktree
  may resolve to a different build; invoke the pinned binary by absolute path
  (`$(go env GOBIN)/sqlc`) and confirm `sqlc version` before trusting output.
- **`sqlc generate` cannot simply be committed.** The checked-in generated code has been
  hand-maintained and no longer matches its inputs: `sandbox.sql.go` routes every
  `sandbox_snapshot` read through one hand-written `scanSandboxSnapshot` helper instead
  of inline scans, files are laid out in query-file order rather than the alphabetical
  order v1.31.1 emits, and some files carry
  `// Hand-written (sqlc generate is broken in this repo)` (e.g.
  `pkg/db/generated/interaction_dag.sql.go:357`). A wholesale regeneration rewrites
  roughly 20 unrelated files and adds 3 new ones.
- **Rule for this change:** regenerate, then keep only the hunks belonging to this
  change and revert everything else (save the regenerated files aside first, so the
  correct text can be transplanted). Match the surrounding local style, including the
  shared scan helper, rather than sqlc's inline scans. A new column lands **last** in
  every expansion, in physical table order. `go build ./...` must pass.
- Because a column/scan misalignment in that transplant is invisible to both the
  compiler and every non-database test, assert it:
  `TestGeneratedSnapshotScanMatchesSelectedColumns` in
  `server/internal/migrations/migrations_test.go` counts column-list occurrences and
  pins the scan order. Mutation-check any such guard by breaking the invariant once and
  confirming the test fails.

### Verification capability limits in this environment

Established by direct check, not assumption: `psql` is not installed, Docker is not
available, and the `DATABASE_URL` configured in `multica/.env` points at
`localhost:5432`, which refuses connections. Installing Postgres was explicitly declined
for this change. (`sqlc` was also assumed missing, but is installable and needs no
database — see the rule above.)

Consequences that every schema and query task below must respect:

- The SQL text is machine-checked against the real schema by `sqlc generate`, which
  parses `migrations/`. That catches typos, unknown columns, and wrong parameter counts,
  but says nothing about runtime behavior.
- The `*_Integration` query tests this plan specifies are still **written**, and they
  self-skip when `DATABASE_URL` is unset, exactly like
  `TestInteractionDAGQueries_Integration`. They are **not executed here**. Do not report
  them as passing.
- **Whole packages silently run nothing.** `internal/handler`, `cmd/server`,
  `internal/workgraph`, and `pkg/agent` each have a `TestMain` that prints
  `Skipping tests: database not reachable` and calls `os.Exit(0)`, so `go test` on them
  prints `ok` with zero tests executed. Never cite `ok` from those packages as evidence
  here; only their compilation was checked. `internal/service` and `internal/migrations`
  have no such gate, so put the load-bearing tests there whenever there is a choice.
- Therefore the acceptance evidence available in this environment is: `sqlc generate`,
  `go build ./...`, `go vet ./...`, the fake-injected unit tests, and file-content
  assertions on migrations and queries in the style of
  `server/internal/migrations/migrations_test.go`. Actually applying migrations,
  unique-index claim races, and cascade behavior remain **unverified until run against a
  real Postgres**, and each affected task says so at its verification step.
- Phases 2, 3, and 6 carry the most exposure, since their core invariants (the
  `UNIQUE (checkpoint_id, lane_key)` claim race and the `ON DELETE CASCADE` reclamation)
  live in the database rather than in Go.

### Go test conventions `[multica]`

Read `multica/server/internal/service/env_checkpoint_test.go` and
`task_resume_runner_test.go` before writing tests. The house style, which every new test
in this plan follows:

- Plain `testing` with hand-rolled fakes in the same `package service` file — **not**
  table-driven, **not** `testify` in these two files. Fakes are structs recording calls:
  `fakeCheckpointRepo` (with `createCalls`/`updateCalls` slices and a `sync.Mutex`),
  `fakeCheckpointSaver`, `fakeCheckpointResumer`, `fakeProjectSnapshotReader`,
  `fakeInFlightResolver`, `fakeResumeAgentRunner`, `fakeInFlightResetter`, `fakeWaker`,
  `fakeEnvSandboxLifecycleDeps`.
- Assertions are `if got != want { t.Fatalf(...) }`. Test names are long and
  behavioural: `TestEnvCheckpointCreateRecordsTimeoutStatus`,
  `TestResumeFromCheckpointTriggerFailureIsPartialResume`.
- DB-backed query tests follow `TestInteractionDAGQueries_Integration` in
  `internal/service/interaction_dag_test.go:926`: a `*_Integration` suffix, a
  `t.Skip("integration test requires Postgres at DATABASE_URL")` helper when
  `DATABASE_URL` is unset, `pool.Begin` + `defer tx.Rollback(ctx)` so the test is
  hermetic, and `testify`'s `require`/`assert` (that file does import testify).
- Run one test: `cd multica/server && go test ./internal/service/ -run TestName -v`
- Run a package: `cd multica/server && go test ./internal/service/`
- Whole Go suite: `cd multica && make test`

### Names this plan introduces (use these exact spellings)

| Name                                                                                      | Package   | Meaning                                                         |
| ----------------------------------------------------------------------------------------- | --------- | --------------------------------------------------------------- |
| `EnvCheckpointSaveMode`                                                                   | `service` | `SaveModePauseInPlace` \| `SaveModeSnapshot`                    |
| `Savepoint` / `SavepointCreator`                                                          | `service` | one immutable snapshot record owned by a checkpoint             |
| `ErrSavepointFailed` / `ErrSavepointGone`                                                 | `service` | typed savepoint errors                                          |
| `ContinuationRequest` / `ContinuationOutcome`                                             | `service` | uniform input/result of the continuation seam                   |
| `ContinuationRegistry`                                                                    | `service` | `{SameRuntime, Forked ResumeAgentRunner}`                       |
| `EnvCheckpointLane` / `EnvCheckpointLaneRepository`                                       | `service` | lane row + persistence seam                                     |
| `LaneStatusProvisioning` / `LaneStatusReady` / `LaneStatusFailed`                         | `service` | lane status constants                                           |
| `LaneMaterializer`                                                                        | `service` | per-step lane materialization seam                              |
| `ResumeFromCheckpointInput`                                                               | `service` | struct-form resume input carrying `LaneCount` + `LaneKeyAnchor` |
| `ErrCheckpointNotResumable` / `ErrLaneCountInvalid` / `ErrCheckpointHasProvisioningLanes` | `service` | typed rejections                                                |

### Phasing intent (`tasks.md` 0.2)

Phases 1, 2, 3, 5, 6, 7 are behavior-preserving or additive and land independently.
**Only Phase 4 changes externally observable branch behavior.** Phase 5 is the only
cross-process breaking change: sandboxd and server must deploy together.

______________________________________________________________________

## Task 0: Ground rules and repository setup

**tasks.md:** 0.1, 0.2

**Files:**

- Modify: `openspec/changes/env-savepoint-consolidation/tasks.md` (check 0.1, 0.2)
  `[areal]`

- [ ] **Step 1: `[areal]` Confirm multica is excluded from this repository**

```bash
cd /workspaces/leagent/backend/areal
git ls-files multica | head -5     # expect: empty output
git check-ignore -v multica        # expect: .../modules/backend/areal/info/exclude:7:/multica/
cd multica && git rev-parse --abbrev-ref HEAD    # expect: dev
```

Expected: `git ls-files multica` prints nothing, and the exclusion rule resolves. This
is the evidence for `tasks.md` 0.1.

- [ ] **Step 2: `[multica]` Leave the main checkout's uncommitted work alone**

The main multica checkout carries unrelated in-progress work. Isolation comes from the
worktree, not from stashing, so nothing needs to be parked and nothing may be discarded.

```bash
cd /workspaces/leagent/backend/areal/multica
git status --porcelain    # record it; it must look the same when this change is done
```

- [ ] **Step 3: `[multica]` Verify the isolated worktree and its baseline**

The worktree already exists on `upstream/dev`. Confirm it, and confirm the baseline is
green before any edit.

```bash
cd /workspaces/leagent/backend/areal/multica/.worktrees/env-savepoint-consolidation
git branch --show-current                      # expect: feature/20260728/env-savepoint-consolidation
git rev-list --left-right --count HEAD...upstream/dev   # expect: 0 0 before the first commit
cd server && go build ./... && go test ./internal/service/... ./internal/handler/...
```

Expected: build exits 0; `internal/service`, `internal/service/voicecall`, and
`internal/handler` all report `ok`.

- [ ] **Step 4: `[areal]` Record the phasing invariant and tick the boxes**

Edit `openspec/changes/env-savepoint-consolidation/tasks.md`: change `- [ ] 0.1` →
`- [x] 0.1` and `- [ ] 0.2` → `- [x] 0.2`.

- [ ] **Step 5: `[areal]` Commit**

```bash
cd /workspaces/leagent/backend/areal
git add openspec/changes/env-savepoint-consolidation/tasks.md
git commit -m "chore(openspec): confirm env-savepoint-consolidation ground rules"
```

______________________________________________________________________

# Phase 1 — Continuation seam extraction (behavior-preserving)

## Task 1: `[multica]` Two named strategies behind the `ResumeAgentRunner` seam

**tasks.md:** 1.1, 1.2, 1.4

**Files:**

- Modify: `multica/server/internal/service/env_checkpoint.go` (the `ResumeAgentRunner`,
  `TriggerStatus`, and `EnvCheckpointService` region, lines 36–58 and 135–147, plus
  `ResumeFromCheckpoint` at 243–288)
- Modify: `multica/server/internal/service/task_resume_runner.go`
  (`taskResumeRunner.ResumeAgentRun`, lines 45–69)
- Modify: `multica/server/internal/service/env_checkpoint_test.go` (all 17
  `NewEnvCheckpointService(...)` call sites)
- Modify: `multica/server/internal/service/task_resume_runner_test.go`

**Interfaces:**

- Consumes: existing `ResumeTrigger`, `TriggerStatus` (`TriggerExecuted`,
  `TriggerSkippedLegacy`, `TriggerFailed`), `ErrTriggerTaskNotResumable`.
- Produces:

```go
type EnvCheckpointSaveMode string

const (
    SaveModePauseInPlace EnvCheckpointSaveMode = "pause_in_place"
    SaveModeSnapshot     EnvCheckpointSaveMode = "snapshot"
)

// LaneRef identifies one materialized lane for forked continuation. Zero value
// means "no lane" (same-runtime continuation).
type LaneRef struct {
    LaneKey         string
    InstanceID      string
    ProjectID       string
    RuntimeID       string
    AgentID         string
    ChannelID       string
    ChatSessionID   string
    SourceMessageID string
}

type ContinuationRequest struct {
    Trigger     ResumeTrigger
    Lane        LaneRef
    WorkspaceID string
    ActorUserID string
}

type ContinuationOutcome struct {
    Status  TriggerStatus
    TaskID  string
    LaneKey string
}

// ResumeAgentRunner is the single continuation seam. Exactly one strategy is
// selected per resume by the checkpoint's save mode.
type ResumeAgentRunner interface {
    Mode() EnvCheckpointSaveMode
    ResumeAgentRun(ctx context.Context, req ContinuationRequest) (ContinuationOutcome, error)
}

type ContinuationRegistry struct {
    SameRuntime ResumeAgentRunner
    Forked      ResumeAgentRunner
}

func (r ContinuationRegistry) For(mode EnvCheckpointSaveMode) ResumeAgentRunner
func NewSameRuntimeContinuation(resetter InFlightTaskResetter, waker TaskWakeupNotifier) ResumeAgentRunner
```

- [ ] **Step 1: Write the failing tests for strategy selection**

Append to `multica/server/internal/service/env_checkpoint_test.go`. Replace the existing
`fakeResumeAgentRunner` with a mode-aware version (its old two-field shape no longer
satisfies the widened interface):

```go
type fakeContinuationStrategy struct {
	mode       EnvCheckpointSaveMode
	calls      []ContinuationRequest
	outcome    ContinuationOutcome
	err        error
}

func (f *fakeContinuationStrategy) Mode() EnvCheckpointSaveMode { return f.mode }

func (f *fakeContinuationStrategy) ResumeAgentRun(_ context.Context, req ContinuationRequest) (ContinuationOutcome, error) {
	f.calls = append(f.calls, req)
	if f.err != nil {
		return ContinuationOutcome{Status: TriggerFailed}, f.err
	}
	if f.outcome.Status == "" {
		return ContinuationOutcome{Status: TriggerExecuted, TaskID: req.Trigger.TaskID}, nil
	}
	return f.outcome, nil
}

func TestResumeSelectsSameRuntimeStrategyForPauseInPlace(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SaveMode:      SaveModePauseInPlace,
		SandboxRefs:   []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws"}},
		ResumeTrigger: json.RawMessage(`{"task_id":"t-1","runtime_id":"r-1","kind":"issue"}`),
	}
	same := &fakeContinuationStrategy{mode: SaveModePauseInPlace}
	forked := &fakeContinuationStrategy{mode: SaveModeSnapshot}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{},
		ContinuationRegistry{SameRuntime: same, Forked: forked})

	res, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u")
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(same.calls) != 1 {
		t.Fatalf("same-runtime strategy calls = %d, want 1", len(same.calls))
	}
	if len(forked.calls) != 0 {
		t.Fatalf("forked strategy must not be invoked for pause_in_place, got %d", len(forked.calls))
	}
	if res.TriggerStatus != TriggerExecuted {
		t.Fatalf("trigger status = %s, want executed", res.TriggerStatus)
	}
}

func TestResumeDefaultsToSameRuntimeStrategyForLegacyRowsWithoutSaveMode(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SaveMode:      "", // pre-change row
		SandboxRefs:   []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws"}},
		ResumeTrigger: json.RawMessage(`{"task_id":"t-1","runtime_id":"r-1","kind":"issue"}`),
	}
	same := &fakeContinuationStrategy{mode: SaveModePauseInPlace}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{},
		ContinuationRegistry{SameRuntime: same})

	if _, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u"); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(same.calls) != 1 {
		t.Fatalf("legacy row must use same-runtime strategy, calls = %d", len(same.calls))
	}
}

func TestResumeReportsSkippedWhenNoContinuationDescriptor(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", SaveStatus: EnvCheckpointSaveComplete,
		SaveMode:    SaveModePauseInPlace,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "s-1", WorkspaceID: "ws"}},
	}
	same := &fakeContinuationStrategy{mode: SaveModePauseInPlace}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{},
		ContinuationRegistry{SameRuntime: same})

	res, err := svc.ResumeFromCheckpoint(context.Background(), "ws", "cp-1", "u")
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if res.TriggerStatus != TriggerSkippedLegacy {
		t.Fatalf("status = %s, want skipped_legacy", res.TriggerStatus)
	}
	if len(same.calls) != 0 {
		t.Fatalf("no strategy may be invoked without a descriptor, got %d", len(same.calls))
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestResumeSelects|TestResumeDefaultsToSameRuntime|TestResumeReportsSkipped' -v
```

Expected: FAIL to compile — `EnvCheckpoint.SaveMode` undefined, `ContinuationRegistry`
undefined, `NewEnvCheckpointService` arity/type mismatch.

- [ ] **Step 3: Add the save-mode and continuation types**

In `multica/server/internal/service/env_checkpoint.go`, after the `EnvCheckpointStatus`
constants (line 19), insert the `EnvCheckpointSaveMode` type and constants, and after
`TriggerStatus` (line 43) insert `LaneRef`, `ContinuationRequest`,
`ContinuationOutcome`, and the widened `ResumeAgentRunner` plus `ContinuationRegistry`
exactly as given in the **Produces** block above, then:

```go
// For selects the single strategy for a save mode. An empty mode is a
// pre-change row and resolves to pause_in_place (spec: "Existing checkpoints
// keep their behavior").
func (r ContinuationRegistry) For(mode EnvCheckpointSaveMode) ResumeAgentRunner {
	if mode == SaveModeSnapshot {
		return r.Forked
	}
	return r.SameRuntime
}
```

Add `SaveMode EnvCheckpointSaveMode` to both `EnvCheckpointCreateInput` (after `Kind`)
and `EnvCheckpoint` (after `Kind`).

- [ ] **Step 4: Swap the service field to the registry**

Replace the `resumeAgent ResumeAgentRunner` field and constructor parameter:

```go
type EnvCheckpointService struct {
	repo          EnvCheckpointRepository
	saver         SandboxInstanceSaver
	resumer       SandboxInstanceResumer
	snapshot      ProjectSnapshotReader
	inFlight      InFlightTaskResolver
	continuations ContinuationRegistry
}

func NewEnvCheckpointService(repo EnvCheckpointRepository, saver SandboxInstanceSaver, resumer SandboxInstanceResumer, snapshot ProjectSnapshotReader, inFlight InFlightTaskResolver, continuations ContinuationRegistry) *EnvCheckpointService {
	return &EnvCheckpointService{repo: repo, saver: saver, resumer: resumer, snapshot: snapshot, inFlight: inFlight, continuations: continuations}
}
```

Replace the trigger block at the end of `ResumeFromCheckpoint` (lines 266–287) with the
uniform-outcome version:

```go
	// No descriptor recorded (pre-change row or no in-flight task at create
	// time): the environment is restored and continuation is skipped.
	if len(cp.ResumeTrigger) == 0 {
		result.TriggerStatus = TriggerSkippedLegacy
		return result, nil
	}
	strategy := s.continuations.For(cp.SaveMode)
	if strategy == nil {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: non-empty resume_trigger but no continuation strategy configured for save_mode %q", cp.SaveMode)
	}
	var trigger ResumeTrigger
	if err := json.Unmarshal(cp.ResumeTrigger, &trigger); err != nil {
		result.TriggerStatus = TriggerFailed
		return result, fmt.Errorf("unmarshal resume_trigger: %w", err)
	}
	outcome, err := strategy.ResumeAgentRun(ctx, ContinuationRequest{
		Trigger:     trigger,
		WorkspaceID: workspaceID,
		ActorUserID: actorUserID,
	})
	if err != nil {
		result.TriggerStatus = TriggerFailed
		return result, fmt.Errorf("resume agent run: %w", err)
	}
	result.TriggerStatus = outcome.Status
	return result, nil
```

- [ ] **Step 5: Rename the existing runner to the same-runtime strategy**

In `multica/server/internal/service/task_resume_runner.go`, rename `NewTaskResumeRunner`
→ `NewSameRuntimeContinuation`, rename the struct `taskResumeRunner` →
`sameRuntimeContinuation`, add the `Mode` method, and widen the signature. The body's
reset + wake logic is unchanged:

```go
func NewSameRuntimeContinuation(resetter InFlightTaskResetter, waker TaskWakeupNotifier) ResumeAgentRunner {
	return &sameRuntimeContinuation{resetter: resetter, waker: waker}
}

func (r *sameRuntimeContinuation) Mode() EnvCheckpointSaveMode { return SaveModePauseInPlace }

func (r *sameRuntimeContinuation) ResumeAgentRun(ctx context.Context, req ContinuationRequest) (ContinuationOutcome, error) {
	taskID, err := util.ParseUUID(req.Trigger.TaskID)
	if err != nil {
		return ContinuationOutcome{Status: TriggerFailed}, fmt.Errorf("invalid task_id: %w", err)
	}
	runtimeID, err := util.ParseUUID(req.Trigger.RuntimeID)
	if err != nil {
		return ContinuationOutcome{Status: TriggerFailed}, fmt.Errorf("invalid runtime_id: %w", err)
	}
	task, err := r.resetter.ResetInFlightTaskForResume(ctx, db.ResetInFlightTaskForResumeParams{
		TaskID:    taskID,
		RuntimeID: runtimeID,
	})
	if errors.Is(err, pgx.ErrNoRows) {
		// Terminal, missing, or bound to a different runtime - do not wake.
		return ContinuationOutcome{Status: TriggerFailed}, ErrTriggerTaskNotResumable
	}
	if err != nil {
		return ContinuationOutcome{Status: TriggerFailed}, fmt.Errorf("reset in-flight task: %w", err)
	}
	if r.waker != nil {
		r.waker.NotifyTaskAvailable(util.UUIDToString(task.RuntimeID), util.UUIDToString(task.ID))
	}
	return ContinuationOutcome{Status: TriggerExecuted, TaskID: util.UUIDToString(task.ID)}, nil
}
```

- [ ] **Step 6: Mechanically update every existing test call site**

In `env_checkpoint_test.go`, delete the old `fakeResumeAgentRunner` type (lines 161–169)
and rewrite the 17 `NewEnvCheckpointService(...)` calls: replace a trailing `nil` with
`ContinuationRegistry{}` and a trailing `runner` with
`ContinuationRegistry{SameRuntime: runner}` where
`runner := &fakeContinuationStrategy{mode: SaveModePauseInPlace}`. Tests that assert on
`runner.calledWith` become `runner.calls[0].Trigger`. In
`TestResumeFromCheckpointRejectsTriggerWithoutRunner`, keep `ContinuationRegistry{}` and
keep asserting an error.

In `task_resume_runner_test.go`, change the three `NewTaskResumeRunner(...)` calls to
`NewSameRuntimeContinuation(...)` and the three
`runner.ResumeAgentRun(context.Background(), ResumeTrigger{...})` calls to
`runner.ResumeAgentRun(context.Background(), ContinuationRequest{Trigger: ResumeTrigger{...}})`,
discarding the new first return value with `_,`.

- [ ] **Step 7: Run the whole service package**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go build ./... && go test ./internal/service/
```

Expected: PASS. Every pre-existing checkpoint/resume test still passes — this task
changes no behavior.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/env_checkpoint.go server/internal/service/task_resume_runner.go server/internal/service/env_checkpoint_test.go server/internal/service/task_resume_runner_test.go
git commit -m "refactor(env-checkpoint): select continuation strategy by save mode"
```

______________________________________________________________________

## Task 2: `[multica]` Move branch continuation behind the forked-runtime strategy

**tasks.md:** 1.3, 1.5

> **Discrepancy to respect (see the plan's closing note):** `tasks.md` 1.3 says the
> per-lane task enqueue is "currently inline in `provisionEnvDispatchAgentBranch`". It
> is not. `provisionEnvDispatchAgentBranch`
> (`multica/server/internal/handler/env_dispatch_channel_provision.go:330-382`) clones
> the sandbox, inserts `chat_session` + `channel_agent_session`, and marks the binding
> ready — it never enqueues a task. The task enqueue for a branch lane is
> `s.deps.EnqueueEnvDispatchChannelRun(...)` in
> `EnvDispatchService.dispatchBranchChannelMessage`
> (`multica/server/internal/service/env_dispatch.go:1506`). This task moves **that**
> call behind the seam, which is what 1.3 intends.

**Files:**

- Create: `multica/server/internal/service/forked_runtime_continuation.go`
- Create: `multica/server/internal/service/forked_runtime_continuation_test.go`
- Modify: `multica/server/internal/service/env_dispatch.go`
  (`dispatchBranchChannelMessage`, lines 1438–1541)
- Modify: `multica/server/internal/service/env_dispatch_test.go`

**Interfaces:**

- Consumes: `ContinuationRequest`, `ContinuationOutcome`, `LaneRef`, `ResumeAgentRunner`
  (Task 1); existing `ChannelRunInput` and
  `EnvDispatchDeps.EnqueueEnvDispatchChannelRun` (`env_dispatch.go:307` region).
- Produces:

```go
// ForkedRuntimeEnqueuer is the seam the forked strategy enqueues through.
// *EnvDispatchService's deps adapter satisfies it in production.
type ForkedRuntimeEnqueuer interface {
    EnqueueEnvDispatchChannelRun(ctx context.Context, workspaceID, actorUserID string, in ChannelRunInput, idx int) (string, error)
}

func NewForkedRuntimeContinuation(enqueuer ForkedRuntimeEnqueuer) ResumeAgentRunner
```

- [ ] **Step 1: Write the failing test for the forked strategy**

Create `multica/server/internal/service/forked_runtime_continuation_test.go`:

```go
package service

import (
	"context"
	"fmt"
	"testing"
)

type fakeForkedEnqueuer struct {
	calls []ChannelRunInput
	err   error
}

func (f *fakeForkedEnqueuer) EnqueueEnvDispatchChannelRun(_ context.Context, _, _ string, in ChannelRunInput, _ int) (string, error) {
	f.calls = append(f.calls, in)
	if f.err != nil {
		return "", f.err
	}
	return fmt.Sprintf("run-%d", len(f.calls)), nil
}

func TestForkedRuntimeContinuationEnqueuesAgainstLaneRuntime(t *testing.T) {
	enq := &fakeForkedEnqueuer{}
	strategy := NewForkedRuntimeContinuation(enq)

	if strategy.Mode() != SaveModeSnapshot {
		t.Fatalf("mode = %s, want snapshot", strategy.Mode())
	}
	out, err := strategy.ResumeAgentRun(context.Background(), ContinuationRequest{
		WorkspaceID: "ws",
		ActorUserID: "u",
		Trigger:     ResumeTrigger{AgentID: "a-src", Kind: "chat"},
		Lane: LaneRef{
			LaneKey: "lane-0", InstanceID: "inst-0", ProjectID: "proj-0",
			RuntimeID: "rt-0", AgentID: "a-0", ChannelID: "ch-0",
			ChatSessionID: "cs-0", SourceMessageID: "msg-0",
		},
	})
	if err != nil {
		t.Fatalf("continue: %v", err)
	}
	if out.Status != TriggerExecuted || out.TaskID != "run-1" || out.LaneKey != "lane-0" {
		t.Fatalf("outcome = %+v", out)
	}
	if len(enq.calls) != 1 {
		t.Fatalf("enqueue calls = %d, want 1", len(enq.calls))
	}
	got := enq.calls[0]
	if got.RuntimeID != "rt-0" || got.SandboxInstanceID != "inst-0" || got.AgentID != "a-0" {
		t.Fatalf("lane binding not used: %+v", got)
	}
}

func TestForkedRuntimeContinuationRejectsMissingLaneRuntime(t *testing.T) {
	strategy := NewForkedRuntimeContinuation(&fakeForkedEnqueuer{})
	out, err := strategy.ResumeAgentRun(context.Background(), ContinuationRequest{
		WorkspaceID: "ws",
		Lane:        LaneRef{LaneKey: "lane-0", InstanceID: "inst-0"},
	})
	if err == nil {
		t.Fatal("expected error when the lane has no runtime")
	}
	if out.Status != TriggerFailed {
		t.Fatalf("status = %s, want failed", out.Status)
	}
}

func TestForkedRuntimeContinuationReportsEnqueueFailureAsFailed(t *testing.T) {
	enq := &fakeForkedEnqueuer{err: fmt.Errorf("queue down")}
	strategy := NewForkedRuntimeContinuation(enq)
	out, err := strategy.ResumeAgentRun(context.Background(), ContinuationRequest{
		WorkspaceID: "ws",
		Lane:        LaneRef{LaneKey: "lane-0", InstanceID: "i", RuntimeID: "rt", AgentID: "a", ChannelID: "ch"},
	})
	if err == nil {
		t.Fatal("expected enqueue error")
	}
	if out.Status != TriggerFailed || out.LaneKey != "lane-0" {
		t.Fatalf("outcome = %+v", out)
	}
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run TestForkedRuntimeContinuation -v
```

Expected: FAIL to compile — `NewForkedRuntimeContinuation` undefined.

- [ ] **Step 3: Implement the forked strategy**

Create `multica/server/internal/service/forked_runtime_continuation.go`:

```go
package service

import (
	"context"
	"fmt"
)

// ForkedRuntimeEnqueuer is the seam the forked-runtime strategy enqueues
// through. In production this is EnvDispatchService's deps adapter, which owns
// the channel-run insert; the strategy holds no SQL of its own.
type ForkedRuntimeEnqueuer interface {
	EnqueueEnvDispatchChannelRun(ctx context.Context, workspaceID, actorUserID string, in ChannelRunInput, idx int) (string, error)
}

// forkedRuntimeContinuation enqueues one task per materialized lane, bound to
// that lane's own runtime, sandbox instance, and copied project subtree. Lanes
// never share a task row or a runtime identity with each other or the source.
type forkedRuntimeContinuation struct {
	enqueuer ForkedRuntimeEnqueuer
}

func NewForkedRuntimeContinuation(enqueuer ForkedRuntimeEnqueuer) ResumeAgentRunner {
	return &forkedRuntimeContinuation{enqueuer: enqueuer}
}

func (f *forkedRuntimeContinuation) Mode() EnvCheckpointSaveMode { return SaveModeSnapshot }

func (f *forkedRuntimeContinuation) ResumeAgentRun(ctx context.Context, req ContinuationRequest) (ContinuationOutcome, error) {
	failed := ContinuationOutcome{Status: TriggerFailed, LaneKey: req.Lane.LaneKey}
	if f.enqueuer == nil {
		return failed, fmt.Errorf("validation_failed: forked continuation has no enqueuer")
	}
	if req.Lane.RuntimeID == "" || req.Lane.InstanceID == "" || req.Lane.AgentID == "" {
		return failed, fmt.Errorf("validation_failed: lane %q missing runtime/instance/agent binding", req.Lane.LaneKey)
	}
	runID, err := f.enqueuer.EnqueueEnvDispatchChannelRun(ctx, req.WorkspaceID, req.ActorUserID, ChannelRunInput{
		AgentID:           req.Lane.AgentID,
		ChannelID:         req.Lane.ChannelID,
		ProjectID:         req.Lane.ProjectID,
		EnvID:             req.Lane.LaneKey,
		ChatSessionID:     req.Lane.ChatSessionID,
		SandboxInstanceID: req.Lane.InstanceID,
		RuntimeID:         req.Lane.RuntimeID,
		SourceMessageID:   req.Lane.SourceMessageID,
	}, 0)
	if err != nil {
		return failed, fmt.Errorf("enqueue lane %q: %w", req.Lane.LaneKey, err)
	}
	return ContinuationOutcome{Status: TriggerExecuted, TaskID: runID, LaneKey: req.Lane.LaneKey}, nil
}
```

> `ChannelRunInput.EnvID` is set from the lane key here as a placeholder; Task 13
> replaces it with the lane's real `env_id` once branch dispatch owns lane provisioning.
> Add `LaneEnvID string` to `LaneRef` in Task 8 and switch this line to
> `req.Lane.LaneEnvID` there.

- [ ] **Step 4: Route branch dispatch's enqueue through the strategy**

In `env_dispatch.go`, add a field and setter next to `WithSandboxLifecycle` (line 518):

```go
// WithForkedContinuation injects the forked-runtime continuation strategy so
// branch dispatch re-engages its lane agents through the single continuation
// seam instead of provisioning-local enqueue logic.
func (s *EnvDispatchService) WithForkedContinuation(strategy ResumeAgentRunner) *EnvDispatchService {
	s.forkedContinuation = strategy
	return s
}
```

In `dispatchBranchChannelMessage`, replace the direct enqueue at line 1506 with:

```go
	strategy := s.forkedContinuation
	if strategy == nil {
		strategy = NewForkedRuntimeContinuation(s.deps)
	}
	outcome, err := strategy.ResumeAgentRun(ctx, ContinuationRequest{
		WorkspaceID: in.WorkspaceID,
		ActorUserID: in.UserID,
		Trigger:     src,
		Lane: LaneRef{
			LaneKey:         r.EnvID,
			InstanceID:      provisioned.SandboxInstanceID,
			ProjectID:       r.ProjectID,
			RuntimeID:       provisioned.RuntimeID,
			AgentID:         provisioned.AgentID,
			ChannelID:       r.ChannelID,
			ChatSessionID:   provisioned.ChatSessionID,
			SourceMessageID: dst.SourceMessageID,
		},
	})
	if err != nil {
		r.Error = fmt.Sprintf("enqueue branch trigger: %v", err)
		r.Stack = stackerr.StackOf(err)
		return
	}
	runID := outcome.TaskID
```

Everything after (`dst.TaskID = runID`, `SaveCollaborationTrigger`, `r.LeaderRunID`,
`r.AgentSandboxes`) is unchanged. `s.deps` already satisfies `ForkedRuntimeEnqueuer`
because `EnvDispatchDeps` declares `EnqueueEnvDispatchChannelRun` with the same
signature.

- [ ] **Step 5: Add the seam-routing assertion to the dispatch tests**

Append to `multica/server/internal/service/env_dispatch_test.go`:

```go
func TestBranchContinuationGoesThroughTheSeam(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	seam := &fakeContinuationStrategy{mode: SaveModeSnapshot}
	svc := newBranchMessageDispatchServiceForTest(t, f).WithForkedContinuation(seam)

	if _, err := svc.Dispatch(context.Background(), branchMessageDispatchInputForTest()); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if len(seam.calls) != 1 {
		t.Fatalf("branch continuation must go through the seam, calls = %d", len(seam.calls))
	}
	if seam.calls[0].Lane.RuntimeID == "" {
		t.Fatalf("seam must receive the lane runtime binding: %+v", seam.calls[0].Lane)
	}
}
```

Build `newBranchMessageDispatchServiceForTest` / `branchMessageDispatchInputForTest` by
extracting the existing setup in `TestBranchWakesOnlyTriggerAgentWithClonedSandbox`
(`env_dispatch_test.go:817`) into helpers, and make that existing test call them too so
the setup is defined once.

- [ ] **Step 6: Run the tests**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestForkedRuntimeContinuation|TestBranch' -v
go test ./internal/service/ ./internal/handler/
```

Expected: PASS, including the pre-existing
`TestBranchWakesOnlyTriggerAgentWithClonedSandbox` and
`TestBranchLeavesNonTriggeredAgentsPendingWithCloneSources` — branch behavior is
unchanged.

- [ ] **Step 7: Commit and tick tasks.md**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/forked_runtime_continuation.go server/internal/service/forked_runtime_continuation_test.go server/internal/service/env_dispatch.go server/internal/service/env_dispatch_test.go
git commit -m "refactor(env-dispatch): route branch continuation through the resume seam"
```

Then in `[areal]`, tick `tasks.md` 1.1–1.5 and commit separately.

______________________________________________________________________

# Phase 2 — Savepoint schema and snapshot save mode

## Task 3: `[multica]` Migration 244 — `save_mode` and checkpoint-owned savepoints

**tasks.md:** 2.1, 2.2

**Files:**

- Create: `multica/server/migrations/244_env_checkpoint_save_mode.up.sql`
- Create: `multica/server/migrations/244_env_checkpoint_save_mode.down.sql`

**Interfaces:**

- Consumes: `env_checkpoint` (created in `154_env_checkpoint_lifecycle.up.sql`),
  `sandbox_snapshot` (created in `182_sandbox_snapshot.up.sql`).

- Produces: `env_checkpoint.save_mode`, `sandbox_snapshot.checkpoint_id`.

- [ ] **Step 1: Re-check the next free migration number**

```bash
cd /workspaces/leagent/backend/areal/multica/server/migrations
ls | sort -V | tail -5
```

Expected: nothing numbered ≥ 244. If there is, use the next free number consistently for
the rest of this plan.

- [ ] **Step 2: Write the up migration**

`multica/server/migrations/244_env_checkpoint_save_mode.up.sql`:

```sql
-- save_mode distinguishes the two first-class checkpoint modes.
-- pause_in_place (the default, and what every pre-existing row resolves to)
-- suspends the source instances and records no savepoint. snapshot records an
-- immutable savepoint per source instance and leaves the source running.
ALTER TABLE env_checkpoint
    ADD COLUMN IF NOT EXISTS save_mode TEXT NOT NULL DEFAULT 'pause_in_place';

ALTER TABLE env_checkpoint
    DROP CONSTRAINT IF EXISTS env_checkpoint_save_mode_check,
    ADD CONSTRAINT env_checkpoint_save_mode_check
        CHECK (save_mode IN ('pause_in_place', 'snapshot'));

-- A savepoint is owned by exactly one checkpoint (design D2: no reference
-- counting). Deleting the checkpoint cascades the ownership row away; the Cube
-- template itself is released by the delete_template job (Phase 6).
ALTER TABLE sandbox_snapshot
    ADD COLUMN IF NOT EXISTS checkpoint_id UUID
        REFERENCES env_checkpoint(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS sandbox_snapshot_checkpoint_idx
    ON sandbox_snapshot (checkpoint_id)
    WHERE checkpoint_id IS NOT NULL;
```

- [ ] **Step 3: Write the down migration**

`multica/server/migrations/244_env_checkpoint_save_mode.down.sql`:

```sql
DROP INDEX IF EXISTS sandbox_snapshot_checkpoint_idx;

ALTER TABLE sandbox_snapshot
    DROP COLUMN IF EXISTS checkpoint_id;

ALTER TABLE env_checkpoint
    DROP CONSTRAINT IF EXISTS env_checkpoint_save_mode_check;

ALTER TABLE env_checkpoint
    DROP COLUMN IF EXISTS save_mode;
```

- [ ] **Step 4: Apply, verify the no-backfill claim, and roll back**

```bash
cd /workspaces/leagent/backend/areal/multica
make migrate-up
psql "$DATABASE_URL" -c "\d env_checkpoint" | grep save_mode
psql "$DATABASE_URL" -c "SELECT save_mode, count(*) FROM env_checkpoint GROUP BY 1;"
psql "$DATABASE_URL" -c "SELECT count(*) FROM sandbox_snapshot WHERE checkpoint_id IS NOT NULL;"
```

Expected: `save_mode | text | not null default 'pause_in_place'`; every existing row
groups under `pause_in_place`; zero savepoints owned by a checkpoint. That is the
evidence for `tasks.md` 2.2 — no backfill is required.

Then verify the down migration is real:

```bash
make migrate-down && make migrate-up
```

Expected: both succeed with no error.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/migrations/244_env_checkpoint_save_mode.up.sql server/migrations/244_env_checkpoint_save_mode.down.sql
git commit -m "feat(db): add env_checkpoint.save_mode and checkpoint-owned savepoints"
```

______________________________________________________________________

## Task 4: `[multica]` Queries for `save_mode` and savepoint ownership

**tasks.md:** 2.3

**Files:**

- Modify: `multica/server/pkg/db/queries/env_checkpoint.sql`
- Modify: `multica/server/pkg/db/queries/sandbox.sql` (savepoint section, lines 290–348)
- Modify (generated): `multica/server/pkg/db/generated/env_checkpoint.sql.go`,
  `multica/server/pkg/db/generated/sandbox.sql.go`,
  `multica/server/pkg/db/generated/models.go`

**Interfaces:**

- Produces: `db.CreateEnvCheckpointParams.SaveMode`, `db.EnvCheckpoint.SaveMode`,
  `db.SandboxSnapshot.CheckpointID`, `Queries.AttachSandboxSnapshotToCheckpoint`,
  `Queries.ListSandboxSnapshotsForCheckpoint`, `Queries.UpdateEnvCheckpointSaveMode`.

- [ ] **Step 1: Add `save_mode` to the checkpoint queries**

In `multica/server/pkg/db/queries/env_checkpoint.sql`, extend `CreateEnvCheckpoint`'s
column and value lists with `save_mode` / `@save_mode`, and append:

```sql
-- name: UpdateEnvCheckpointSaveMode :one
UPDATE env_checkpoint
SET save_mode = @save_mode, updated_at = now()
WHERE id = @id AND workspace_id = @workspace_id
RETURNING *;
```

- [ ] **Step 2: Add the savepoint ownership queries**

Append to `multica/server/pkg/db/queries/sandbox.sql`:

```sql
-- name: AttachSandboxSnapshotToCheckpoint :one
-- Binds a savepoint to its single owning checkpoint (design D2). Idempotent for
-- the same owner so a retried checkpoint create does not fail.
UPDATE sandbox_snapshot
SET checkpoint_id = @checkpoint_id, updated_at = now()
WHERE id = @id AND workspace_id = @workspace_id
  AND (checkpoint_id IS NULL OR checkpoint_id = @checkpoint_id)
RETURNING *;

-- name: ListSandboxSnapshotsForCheckpoint :many
SELECT *
FROM sandbox_snapshot
WHERE checkpoint_id = @checkpoint_id AND workspace_id = @workspace_id
ORDER BY created_at ASC;
```

- [ ] **Step 3: Regenerate**

```bash
cd /workspaces/leagent/backend/areal/multica
make sqlc && cd server && go build ./...
```

Expected: `db.EnvCheckpoint` gains `SaveMode string`, `db.SandboxSnapshot` gains
`CheckpointID pgtype.UUID`, and the two new `Queries` methods exist. Transplant only
those hunks per the **sqlc regeneration** constraint above, revert the unrelated
regeneration, and re-run `go build ./...`.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/pkg/db/queries/env_checkpoint.sql server/pkg/db/queries/sandbox.sql server/pkg/db/generated
git commit -m "feat(db): queries for checkpoint save mode and savepoint ownership"
```

______________________________________________________________________

## Task 5: `[multica]` Snapshot-mode checkpoint create

**tasks.md:** 2.4, 2.5, 2.6, 2.7

**Files:**

- Modify: `multica/server/internal/service/env_checkpoint.go` (`Create`, lines 155–219)
- Modify: `multica/server/internal/service/env_checkpoint_test.go`
- Modify: `multica/server/internal/handler/env_checkpoint.go`
  (`CreateEnvCheckpointRequest`, `mapEnvCheckpointResponse`)

**Interfaces:**

- Consumes: `EnvCheckpointSaveMode` (Task 1), `SandboxInstanceRef`, the existing
  `create_template` job path (`Handler.CreateSandboxSnapshotTemplate`,
  `handler/sandbox.go:1258`, whose completion hook `completeCreateTemplateSnapshot` at
  `handler/sandbox.go:1797` marks the `sandbox_snapshot` row ready and returns the
  instance to `running`).
- Produces:

```go
// Savepoint is one immutable snapshot record owned by a checkpoint.
type Savepoint struct {
    SnapshotID     string
    CubeSnapshotID string
    InstanceID     string
    Status         string // creating | ready | failed | deleting
}

// SavepointCreator creates a savepoint from a live sandbox instance through the
// existing create_template job and blocks until the snapshot record reaches a
// terminal state. The source instance is left running.
type SavepointCreator interface {
    CreateSavepoint(ctx context.Context, ref SandboxInstanceRef, checkpointID, actorUserID string) (Savepoint, error)
}

var ErrSavepointFailed = errors.New("savepoint_failed")
```

- [ ] **Step 1: Write the failing tests**

Append to `env_checkpoint_test.go`:

```go
type fakeSavepointCreator struct {
	calls     []SandboxInstanceRef
	checkpts  []string
	status    string // defaults to "ready"
	err       error
}

func (f *fakeSavepointCreator) CreateSavepoint(_ context.Context, ref SandboxInstanceRef, checkpointID, _ string) (Savepoint, error) {
	f.calls = append(f.calls, ref)
	f.checkpts = append(f.checkpts, checkpointID)
	if f.err != nil {
		return Savepoint{}, f.err
	}
	status := f.status
	if status == "" {
		status = "ready"
	}
	return Savepoint{
		SnapshotID:     fmt.Sprintf("snap-%d", len(f.calls)),
		CubeSnapshotID: fmt.Sprintf("cube-%d", len(f.calls)),
		InstanceID:     ref.InstanceID,
		Status:         status,
	}, nil
}

func TestSnapshotModeCreateOwnsReadySavepointAndLeavesSourceRunning(t *testing.T) {
	repo := newFakeCheckpointRepo()
	saver := &fakeCheckpointSaver{}
	creator := &fakeSavepointCreator{}
	svc := NewEnvCheckpointService(repo, saver, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{snapshot: json.RawMessage(`{}`)},
		&fakeInFlightResolver{}, ContinuationRegistry{}).WithSavepointCreator(creator)

	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "proj", SaveMode: SaveModeSnapshot,
		SandboxRefs: []SandboxInstanceRef{
			{InstanceID: "inst-1", WorkspaceID: "ws"},
			{InstanceID: "inst-2", WorkspaceID: "ws"},
		},
		ActorUserID: "u", SaveTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if cp.SaveStatus != EnvCheckpointSaveComplete {
		t.Fatalf("status = %s, want complete", cp.SaveStatus)
	}
	if len(creator.calls) != 2 {
		t.Fatalf("want one savepoint per source instance (2), got %d", len(creator.calls))
	}
	if len(saver.calls) != 0 {
		t.Fatalf("snapshot mode must not stop any source instance, got %d stops", len(saver.calls))
	}
	for _, id := range creator.checkpts {
		if id != cp.ID {
			t.Fatalf("savepoint owner = %q, want %q", id, cp.ID)
		}
	}
}

func TestSnapshotModeCreateFailsWhenSavepointReachesFailed(t *testing.T) {
	repo := newFakeCheckpointRepo()
	creator := &fakeSavepointCreator{status: "failed"}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{snapshot: json.RawMessage(`{}`)},
		&fakeInFlightResolver{}, ContinuationRegistry{}).WithSavepointCreator(creator)

	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "proj", SaveMode: SaveModeSnapshot,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "inst-1", WorkspaceID: "ws"}},
		ActorUserID: "u", SaveTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("create should record the failure, not return it: %v", err)
	}
	if cp.SaveStatus != EnvCheckpointSaveFailed {
		t.Fatalf("status = %s, want failed", cp.SaveStatus)
	}
}

func TestSnapshotModeCreateRecordsTimeoutStatus(t *testing.T) {
	repo := newFakeCheckpointRepo()
	creator := &fakeSavepointCreator{err: context.DeadlineExceeded}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{snapshot: json.RawMessage(`{}`)},
		&fakeInFlightResolver{}, ContinuationRegistry{}).WithSavepointCreator(creator)

	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "proj", SaveMode: SaveModeSnapshot,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "inst-1", WorkspaceID: "ws"}},
		ActorUserID: "u", SaveTimeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if cp.SaveStatus != EnvCheckpointSaveTimedOut {
		t.Fatalf("status = %s, want timed_out", cp.SaveStatus)
	}
}

func TestPauseInPlaceCreateStaysOnTheStopPath(t *testing.T) {
	repo := newFakeCheckpointRepo()
	saver := &fakeCheckpointSaver{}
	creator := &fakeSavepointCreator{}
	svc := NewEnvCheckpointService(repo, saver, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{snapshot: json.RawMessage(`{}`)},
		&fakeInFlightResolver{}, ContinuationRegistry{}).WithSavepointCreator(creator)

	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "proj", // SaveMode omitted -> pause_in_place
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "inst-1", WorkspaceID: "ws"}},
		ActorUserID: "u", SaveTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if cp.SaveMode != SaveModePauseInPlace {
		t.Fatalf("save mode = %q, want pause_in_place", cp.SaveMode)
	}
	if len(saver.calls) != 1 {
		t.Fatalf("pause_in_place must stop the source, stops = %d", len(saver.calls))
	}
	if len(creator.calls) != 0 {
		t.Fatalf("pause_in_place must take no savepoint, got %d", len(creator.calls))
	}
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestSnapshotModeCreate|TestPauseInPlaceCreate' -v
```

Expected: FAIL to compile — `WithSavepointCreator` and `Savepoint` undefined.

- [ ] **Step 3: Add the savepoint seam and split `Create` by mode**

In `env_checkpoint.go`, add the `Savepoint` / `SavepointCreator` / `ErrSavepointFailed`
declarations from **Produces**, a `savepoints SavepointCreator` field, and:

```go
// WithSavepointCreator injects the snapshot-mode savepoint seam. A nil creator
// leaves snapshot-mode create rejected as unconfigured; pause_in_place is
// unaffected.
func (s *EnvCheckpointService) WithSavepointCreator(c SavepointCreator) *EnvCheckpointService {
	s.savepoints = c
	return s
}
```

Normalize the mode right after the existing validation block in `Create` (after the
Fleet-only check at line 166):

```go
	if in.SaveMode == "" {
		in.SaveMode = SaveModePauseInPlace
	}
	switch in.SaveMode {
	case SaveModePauseInPlace, SaveModeSnapshot:
	default:
		return EnvCheckpoint{}, fmt.Errorf("validation_failed: save_mode must be pause_in_place or snapshot")
	}
	if in.SaveMode == SaveModeSnapshot && s.savepoints == nil {
		return EnvCheckpoint{}, fmt.Errorf("validation_failed: snapshot save_mode requires a savepoint creator")
	}
```

Replace the save loop (lines 200–212) with a mode switch. `pause_in_place` keeps the
existing loop byte-for-byte (`tasks.md` 2.6); `snapshot` takes one savepoint per source
ref and never calls `s.saver`:

```go
	status := EnvCheckpointSaveComplete
	var saveErr string
	for _, ref := range in.SandboxRefs {
		var err error
		if in.SaveMode == SaveModeSnapshot {
			var sp Savepoint
			sp, err = s.savepoints.CreateSavepoint(saveCtx, ref, cp.ID, in.ActorUserID)
			if err == nil && sp.Status != "ready" {
				err = fmt.Errorf("%w: savepoint %s status %s", ErrSavepointFailed, sp.SnapshotID, sp.Status)
			}
		} else {
			err = s.saver.Save(saveCtx, ref, in.ActorUserID)
		}
		if err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				status = EnvCheckpointSaveTimedOut
			} else {
				status = EnvCheckpointSaveFailed
			}
			saveErr = err.Error()
			break
		}
	}
```

- [ ] **Step 4: Run the tests**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestSnapshotModeCreate|TestPauseInPlaceCreate|TestEnvCheckpointCreate' -v
```

Expected: PASS, including all pre-existing `TestEnvCheckpointCreate*` tests.

- [ ] **Step 5: Surface `save_mode` on the HTTP boundary**

In `handler/env_checkpoint.go`: add
`SaveMode string \`json:"save_mode,omitempty"\``to`CreateEnvCheckpointRequest`(after`Kind`) and to `EnvCheckpointResponse`(after`Kind`); pass `SaveMode:
service.EnvCheckpointSaveMode(req.SaveMode)`in the`Create`call at line 103; map`SaveMode:
string(cp.SaveMode)`in`mapEnvCheckpointResponse`. Omitting the field keeps today's `pause_in_place\`
behavior, so the AReaL client is unaffected.

- [ ] **Step 6: Add the query integration test for the new column and ownership**

Append to `env_checkpoint_test.go` (imports `os`, `pgxpool`, `require` as in
`interaction_dag_test.go`):

```go
func envCheckpointTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("integration test requires Postgres at DATABASE_URL")
	}
	pool, err := pgxpool.New(context.Background(), dbURL)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	return pool
}

// TestEnvCheckpointSaveModeQueries_Integration proves migration 244's column,
// its CHECK constraint, and savepoint ownership cascade against a real
// Postgres. Runs inside a rolled-back transaction so it is hermetic.
func TestEnvCheckpointSaveModeQueries_Integration(t *testing.T) {
	pool := envCheckpointTestPool(t)
	defer pool.Close()
	ctx := context.Background()
	tx, err := pool.Begin(ctx)
	require.NoError(t, err)
	defer tx.Rollback(ctx)

	var cpID string
	require.NoError(t, tx.QueryRow(ctx, `
		INSERT INTO env_checkpoint (workspace_id, project_id, event_ref, checkpoint_kind, save_timeout_ms, save_status, save_mode)
		VALUES (gen_random_uuid(), gen_random_uuid(), 'evt', 'always', 30000, 'complete', 'snapshot')
		RETURNING id::text`).Scan(&cpID))

	// The CHECK constraint rejects an unknown mode.
	_, err = tx.Exec(ctx, `UPDATE env_checkpoint SET save_mode = 'bogus' WHERE id::text = $1`, cpID)
	require.Error(t, err, "save_mode CHECK must reject unknown values")
}
```

- [ ] **Step 7: Run everything and commit**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ ./internal/handler/
DATABASE_URL="$DATABASE_URL" go test ./internal/service/ -run TestEnvCheckpointSaveModeQueries_Integration -v
cd .. && git add server/internal/service/env_checkpoint.go server/internal/service/env_checkpoint_test.go server/internal/handler/env_checkpoint.go
git commit -m "feat(env-checkpoint): add snapshot save mode with checkpoint-owned savepoints"
```

Then in `[areal]`, tick `tasks.md` 2.1–2.7.

______________________________________________________________________

# Phase 3 — Fan-out resume

## Task 6: `[multica]` Migration 245 and queries for `env_checkpoint_lane`

**tasks.md:** 3.5

**Files:**

- Create: `multica/server/migrations/245_env_checkpoint_lane.up.sql`
- Create: `multica/server/migrations/245_env_checkpoint_lane.down.sql`
- Create: `multica/server/pkg/db/queries/env_checkpoint_lane.sql`
- Modify (generated): `multica/server/pkg/db/generated/env_checkpoint_lane.sql.go`,
  `models.go`

**Interfaces:**

- Produces: table `env_checkpoint_lane`; `Queries.ClaimEnvCheckpointLane`,
  `GetEnvCheckpointLane`, `ListEnvCheckpointLanes`, `UpdateEnvCheckpointLaneStep`,
  `MarkEnvCheckpointLaneReady`, `MarkEnvCheckpointLaneFailed`,
  `ListStaleProvisioningEnvCheckpointLanes`, `CountProvisioningEnvCheckpointLanes`.

- [ ] **Step 1: Write the up migration**

`multica/server/migrations/245_env_checkpoint_lane.up.sql`:

```sql
-- env_checkpoint_lane serves both per-lane idempotency and lane-provisioning
-- crash recovery (design D1). The UNIQUE index IS the idempotency mechanism, so
-- concurrent resume needs no application lock; the status column IS the
-- crash-recovery mechanism, so a sandbox orphaned mid-lane has a discoverable
-- owner. Per-step ids are filled as materialization advances, letting a resumed
-- provisioning lane continue from its first unfilled step.
CREATE TABLE IF NOT EXISTS env_checkpoint_lane (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_id UUID NOT NULL REFERENCES env_checkpoint(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL,
    lane_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'provisioning'
        CHECK (status IN ('provisioning', 'ready', 'failed')),
    instance_id UUID,
    project_id UUID,
    runtime_id UUID,
    task_id UUID,
    env_id UUID,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (checkpoint_id, lane_key)
);

CREATE INDEX IF NOT EXISTS env_checkpoint_lane_provisioning_idx
    ON env_checkpoint_lane (updated_at)
    WHERE status = 'provisioning';
```

- [ ] **Step 2: Write the down migration**

`multica/server/migrations/245_env_checkpoint_lane.down.sql`:

```sql
DROP INDEX IF EXISTS env_checkpoint_lane_provisioning_idx;
DROP TABLE IF EXISTS env_checkpoint_lane;
```

- [ ] **Step 3: Write the queries**

`multica/server/pkg/db/queries/env_checkpoint_lane.sql`:

```sql
-- name: ClaimEnvCheckpointLane :one
-- Claims a lane by inserting with ON CONFLICT DO NOTHING. The insert winning
-- means this caller owns materialization; losing returns no rows and the caller
-- reads the existing row with GetEnvCheckpointLane to branch on its status.
INSERT INTO env_checkpoint_lane (checkpoint_id, workspace_id, lane_key, status)
VALUES (@checkpoint_id, @workspace_id, @lane_key, 'provisioning')
ON CONFLICT (checkpoint_id, lane_key) DO NOTHING
RETURNING *;

-- name: GetEnvCheckpointLane :one
SELECT * FROM env_checkpoint_lane
WHERE checkpoint_id = @checkpoint_id AND lane_key = @lane_key;

-- name: ListEnvCheckpointLanes :many
SELECT * FROM env_checkpoint_lane
WHERE checkpoint_id = @checkpoint_id
ORDER BY created_at ASC;

-- name: UpdateEnvCheckpointLaneStep :one
-- Records one materialization step's id. COALESCE keeps already-filled steps so
-- a continued lane never regresses.
UPDATE env_checkpoint_lane
SET instance_id = COALESCE(sqlc.narg(instance_id), instance_id),
    project_id  = COALESCE(sqlc.narg(project_id), project_id),
    runtime_id  = COALESCE(sqlc.narg(runtime_id), runtime_id),
    task_id     = COALESCE(sqlc.narg(task_id), task_id),
    env_id      = COALESCE(sqlc.narg(env_id), env_id),
    updated_at  = now()
WHERE id = @id AND workspace_id = @workspace_id
RETURNING *;

-- name: MarkEnvCheckpointLaneReady :one
UPDATE env_checkpoint_lane
SET status = 'ready', error = NULL, updated_at = now()
WHERE id = @id AND workspace_id = @workspace_id
RETURNING *;

-- name: MarkEnvCheckpointLaneFailed :one
UPDATE env_checkpoint_lane
SET status = 'failed', error = @error, updated_at = now()
WHERE id = @id AND workspace_id = @workspace_id
RETURNING *;

-- name: CountProvisioningEnvCheckpointLanes :one
-- Deleting a checkpoint with provisioning lanes would cascade the lane rows away
-- and orphan their sandboxes (design D4), so deletion consults this first.
SELECT count(*) FROM env_checkpoint_lane
WHERE checkpoint_id = @checkpoint_id AND status = 'provisioning';

-- name: ListStaleProvisioningEnvCheckpointLanes :many
SELECT * FROM env_checkpoint_lane
WHERE status = 'provisioning' AND updated_at < @stale_before
ORDER BY updated_at ASC
LIMIT @row_limit;
```

- [ ] **Step 4: Migrate, regenerate, build**

```bash
cd /workspaces/leagent/backend/areal/multica
make migrate-up && make sqlc && cd server && go build ./...
psql "$DATABASE_URL" -c "\d env_checkpoint_lane"
```

Expected: the table exists with `env_checkpoint_lane_checkpoint_id_lane_key_key` unique
constraint; build is green. Apply the hand-written fallback if `sqlc` is unavailable.

- [ ] **Step 5: Write the unique-index integration test**

`tasks.md` 3.14 — this is the only proof of idempotency; in-memory fakes cannot
demonstrate a database constraint. Create
`multica/server/internal/service/env_checkpoint_lane_query_test.go`:

```go
package service

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/require"
)

func laneTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("integration test requires Postgres at DATABASE_URL")
	}
	pool, err := pgxpool.New(context.Background(), dbURL)
	require.NoError(t, err)
	return pool
}

func insertLaneTestCheckpoint(t *testing.T, ctx context.Context, tx pgx.Tx) string {
	t.Helper()
	var id string
	require.NoError(t, tx.QueryRow(ctx, `
		INSERT INTO env_checkpoint (workspace_id, project_id, event_ref, checkpoint_kind, save_timeout_ms, save_status, save_mode)
		VALUES (gen_random_uuid(), gen_random_uuid(), 'evt', 'always', 30000, 'complete', 'snapshot')
		RETURNING id::text`).Scan(&id))
	return id
}

// TestEnvCheckpointLaneUniqueIndex_Integration proves that two claims of one
// lane key create exactly one lane (the losing claim returns no rows) and that
// an interrupted lane is continued rather than duplicated.
func TestEnvCheckpointLaneUniqueIndex_Integration(t *testing.T) {
	pool := laneTestPool(t)
	defer pool.Close()
	ctx := context.Background()
	tx, err := pool.Begin(ctx)
	require.NoError(t, err)
	defer tx.Rollback(ctx)

	cpID := insertLaneTestCheckpoint(t, ctx, tx)

	claim := func() (string, bool) {
		var id string
		err := tx.QueryRow(ctx, `
			INSERT INTO env_checkpoint_lane (checkpoint_id, workspace_id, lane_key, status)
			SELECT $1::uuid, workspace_id, 'lane-0', 'provisioning' FROM env_checkpoint WHERE id = $1::uuid
			ON CONFLICT (checkpoint_id, lane_key) DO NOTHING
			RETURNING id::text`, cpID).Scan(&id)
		if err != nil {
			return "", false
		}
		return id, true
	}

	first, won := claim()
	require.True(t, won, "first claim must win")
	_, wonAgain := claim()
	require.False(t, wonAgain, "second claim of the same lane key must lose")

	var laneCount int
	require.NoError(t, tx.QueryRow(ctx, `SELECT count(*) FROM env_checkpoint_lane WHERE checkpoint_id = $1::uuid`, cpID).Scan(&laneCount))
	require.Equal(t, 1, laneCount, "one lane key must yield exactly one lane")

	// Interrupted lane: instance recorded, task not. A continuation fills the
	// remaining steps on the SAME row.
	_, err = tx.Exec(ctx, `UPDATE env_checkpoint_lane SET instance_id = gen_random_uuid() WHERE id = $1::uuid`, first)
	require.NoError(t, err)
	var instanceBefore, instanceAfter string
	require.NoError(t, tx.QueryRow(ctx, `SELECT instance_id::text FROM env_checkpoint_lane WHERE id = $1::uuid`, first).Scan(&instanceBefore))
	_, wonThird := claim()
	require.False(t, wonThird, "claiming an interrupted lane must not create a second row")
	require.NoError(t, tx.QueryRow(ctx, `SELECT instance_id::text FROM env_checkpoint_lane WHERE id = $1::uuid`, first).Scan(&instanceAfter))
	require.Equal(t, instanceBefore, instanceAfter, "the interrupted lane keeps its sandbox instance")

	// A new lane key re-expands the frontier on the same checkpoint.
	var second string
	require.NoError(t, tx.QueryRow(ctx, `
		INSERT INTO env_checkpoint_lane (checkpoint_id, workspace_id, lane_key, status)
		SELECT $1::uuid, workspace_id, 'lane-1', 'provisioning' FROM env_checkpoint WHERE id = $1::uuid
		RETURNING id::text`, cpID).Scan(&second))
	require.NotEqual(t, first, second)

	// Deleting the checkpoint cascades the lanes away (design D2/D4 basis).
	_, err = tx.Exec(ctx, `DELETE FROM env_checkpoint WHERE id = $1::uuid`, cpID)
	require.NoError(t, err)
	require.NoError(t, tx.QueryRow(ctx, `SELECT count(*) FROM env_checkpoint_lane WHERE checkpoint_id = $1::uuid`, cpID).Scan(&laneCount))
	require.Equal(t, 0, laneCount, "checkpoint deletion cascades its lanes")
}
```

- [ ] **Step 6: Run and commit**

```bash
cd /workspaces/leagent/backend/areal/multica/server
DATABASE_URL="$DATABASE_URL" go test ./internal/service/ -run TestEnvCheckpointLaneUniqueIndex_Integration -v
cd .. && git add server/migrations/245_env_checkpoint_lane.up.sql server/migrations/245_env_checkpoint_lane.down.sql server/pkg/db/queries/env_checkpoint_lane.sql server/pkg/db/generated server/internal/service/env_checkpoint_lane_query_test.go
git commit -m "feat(db): add env_checkpoint_lane for per-lane idempotency and recovery"
```

______________________________________________________________________

## Task 7: `[multica]` Resume request shape, lane-count validation, typed rejections

**tasks.md:** 3.1, 3.4, 3.8, 3.9

**Files:**

- Modify: `multica/server/internal/service/env_checkpoint.go`
  (`ResumeFromCheckpointResult`, `ResumeFromCheckpoint`)
- Modify: `multica/server/internal/handler/env_checkpoint.go`
  (`EnvCheckpointServiceAPI`, `ResumeEnvCheckpoint`, `ResumeFromCheckpointResponse`)
- Modify: `multica/server/internal/service/env_checkpoint_test.go`
- Modify: `multica/server/internal/handler/env_checkpoint_test.go`

**Interfaces:**

- Produces:

```go
type ResumeFromCheckpointInput struct {
    WorkspaceID   string
    CheckpointID  string
    ActorUserID   string
    LaneCount     int    // 0 means 1 for backward compatibility at the HTTP edge
    LaneKeyAnchor string // retry-stable anchor; lane keys are anchor + ordinal
}

type ResumeLane struct {
    LaneKey       string
    Status        string // provisioning | ready | failed
    InstanceID    string
    ProjectID     string
    RuntimeID     string
    TaskID        string
    EnvID         string
    ChatSessionID string
    TriggerStatus TriggerStatus
    Error         string
}

// ResumeFromCheckpointResult keeps every pre-existing field; Lanes is additive.
type ResumeFromCheckpointResult struct {
    CheckpointID  string
    ProjectID     string
    EnvIDMap      map[string]string
    SandboxRefs   []SandboxInstanceRef
    RolloutHandle string
    TriggerStatus TriggerStatus
    Lanes         []ResumeLane
}

var (
    ErrCheckpointNotResumable = errors.New("checkpoint_not_resumable")
    ErrLaneCountInvalid       = errors.New("lane_count_invalid")
)

func (s *EnvCheckpointService) ResumeFromCheckpoint(ctx context.Context, in ResumeFromCheckpointInput) (ResumeFromCheckpointResult, error)
```

- [ ] **Step 1: Write the failing validation tests**

Add `"errors"` to `env_checkpoint_test.go`'s import block first — the file currently
imports only `context`, `encoding/json`, `fmt`, `sort`, `strings`, `sync`, `testing`,
`time`, and these tests use `errors.Is`. Then append:

```go
func newSnapshotCheckpointForResume(repo *fakeCheckpointRepo, status EnvCheckpointStatus) {
	repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", ProjectID: "proj",
		SaveMode: SaveModeSnapshot, SaveStatus: status,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "inst-1", WorkspaceID: "ws"}},
	}
}

func TestResumeRejectsZeroLaneCount(t *testing.T) {
	repo := newFakeCheckpointRepo()
	newSnapshotCheckpointForResume(repo, EnvCheckpointSaveComplete)
	resumer := &fakeCheckpointResumer{}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, resumer,
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, ContinuationRegistry{})

	_, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 0, LaneKeyAnchor: "anchor",
	})
	if !errors.Is(err, ErrLaneCountInvalid) {
		t.Fatalf("expected ErrLaneCountInvalid, got %v", err)
	}
	if len(resumer.calls) != 0 {
		t.Fatalf("zero lane count must create nothing, resume calls = %d", len(resumer.calls))
	}
}

func TestResumeRejectsFanOutForPauseInPlace(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", SaveMode: SaveModePauseInPlace,
		SaveStatus:  EnvCheckpointSaveComplete,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "inst-1", WorkspaceID: "ws"}},
	}
	resumer := &fakeCheckpointResumer{}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, resumer,
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, ContinuationRegistry{})

	_, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 3, LaneKeyAnchor: "anchor",
	})
	if !errors.Is(err, ErrLaneCountInvalid) {
		t.Fatalf("expected ErrLaneCountInvalid for pause_in_place fan-out, got %v", err)
	}
	if len(resumer.calls) != 0 {
		t.Fatalf("rejected fan-out must not resume anything, got %d", len(resumer.calls))
	}
}

func TestResumeRejectsTimedOutCheckpointWithTypedError(t *testing.T) {
	repo := newFakeCheckpointRepo()
	newSnapshotCheckpointForResume(repo, EnvCheckpointSaveTimedOut)
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, ContinuationRegistry{})

	_, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u", LaneCount: 1, LaneKeyAnchor: "a",
	})
	if !errors.Is(err, ErrCheckpointNotResumable) {
		t.Fatalf("expected ErrCheckpointNotResumable, got %v", err)
	}
}

func TestPauseInPlaceLaneCountOneResumesSameInstances(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", SaveMode: SaveModePauseInPlace,
		SaveStatus: EnvCheckpointSaveComplete,
		SandboxRefs: []SandboxInstanceRef{
			{InstanceID: "inst-1", WorkspaceID: "ws"},
			{InstanceID: "inst-2", WorkspaceID: "ws"},
		},
	}
	resumer := &fakeCheckpointResumer{}
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, resumer,
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, ContinuationRegistry{})

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u", LaneCount: 1, LaneKeyAnchor: "a",
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(resumer.calls) != 2 {
		t.Fatalf("pause_in_place must resume both instances, got %d", len(resumer.calls))
	}
	if len(res.Lanes) != 0 {
		t.Fatalf("pause_in_place resume must report no lanes, got %d", len(res.Lanes))
	}
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestResumeRejects|TestPauseInPlaceLaneCount' -v
```

Expected: FAIL to compile — `ResumeFromCheckpointInput` undefined and
`ResumeFromCheckpoint` has the old positional signature.

- [ ] **Step 3: Convert the signature and add validation**

In `env_checkpoint.go`, add the `ResumeFromCheckpointInput`, `ResumeLane`, and typed
errors from **Produces**, add `Lanes []ResumeLane` to `ResumeFromCheckpointResult`, then
change `ResumeFromCheckpoint` to take the struct and open with:

```go
func (s *EnvCheckpointService) ResumeFromCheckpoint(ctx context.Context, in ResumeFromCheckpointInput) (ResumeFromCheckpointResult, error) {
	if s.resumer == nil {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: resume is not configured (no sandbox resumer)")
	}
	if in.LaneCount < 1 {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: %w: lane_count must be at least 1", ErrLaneCountInvalid)
	}
	cp, err := s.repo.GetCheckpoint(ctx, in.CheckpointID, in.WorkspaceID)
	if err != nil {
		return ResumeFromCheckpointResult{}, fmt.Errorf("not found: %w", err)
	}
	if cp.SaveStatus != EnvCheckpointSaveComplete {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: %w: save_status is %s, must be complete to resume", ErrCheckpointNotResumable, cp.SaveStatus)
	}
	mode := cp.SaveMode
	if mode == "" {
		mode = SaveModePauseInPlace
	}
	if mode == SaveModePauseInPlace && in.LaneCount > 1 {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: %w: pause_in_place cannot fan out (lane_count=%d)", ErrLaneCountInvalid, in.LaneCount)
	}
	if mode == SaveModeSnapshot {
		return s.resumeSnapshotLanes(ctx, cp, in)
	}
	return s.resumePauseInPlace(ctx, cp, in)
```

Move the existing body (resume each `cp.SandboxRefs` entry, build the result, run the
continuation) verbatim into `resumePauseInPlace` so its behavior is byte-identical, and
stub `resumeSnapshotLanes` to return
`fmt.Errorf("not_implemented: snapshot fan-out resume")` — Task 8 fills it in.

- [ ] **Step 4: Update the HTTP boundary**

In `handler/env_checkpoint.go`:

- Change `EnvCheckpointServiceAPI.ResumeFromCheckpoint` to
  `(ctx context.Context, in service.ResumeFromCheckpointInput) (service.ResumeFromCheckpointResult, error)`.
- Add a request body to `ResumeEnvCheckpoint` (it currently reads none):

```go
	var req struct {
		LaneCount int    `json:"lane_count,omitempty"`
		LaneKey   string `json:"lane_key,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		writeError(w, http.StatusBadRequest, "malformed request body")
		return
	}
	if req.LaneCount == 0 {
		req.LaneCount = 1
	}
	if req.LaneKey == "" {
		req.LaneKey = checkpointID
	}
	res, err := h.EnvCheckpointService.ResumeFromCheckpoint(r.Context(), service.ResumeFromCheckpointInput{
		WorkspaceID:   workspaceID,
		CheckpointID:  checkpointID,
		ActorUserID:   userID,
		LaneCount:     req.LaneCount,
		LaneKeyAnchor: req.LaneKey,
	})
```

- Map `ErrLaneCountInvalid` → 400 and `ErrCheckpointNotResumable` → 409, replacing the
  current `strings.Contains(msg, "validation_failed")` → 409 branch:

```go
	if err != nil {
		switch {
		case errors.Is(err, service.ErrLaneCountInvalid):
			writeError(w, http.StatusBadRequest, err.Error())
		case errors.Is(err, service.ErrCheckpointNotResumable):
			writeError(w, http.StatusConflict, "checkpoint is not resumable")
		case strings.Contains(err.Error(), "not found"):
			writeError(w, http.StatusNotFound, "checkpoint not found")
		default:
			writeError(w, http.StatusInternalServerError, "resume failed")
		}
		return
	}
```

- Add
  `Lanes []LaneResponse \`json:"lanes,omitempty"\``to`ResumeFromCheckpointResponse`with a`LaneResponse`mirroring`service.ResumeLane`in`snake_case`. An empty-body resume of a `pause_in_place`checkpoint therefore serializes exactly as before, since`lanes`is`omitempty\`.

- [ ] **Step 5: Run all affected tests**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go build ./... && go test ./internal/service/ ./internal/handler/
```

Expected: PASS. Update the four `ResumeEnvCheckpoint` cases in
`handler/env_checkpoint_test.go` (lines ~203–243) and the pre-existing service resume
tests to the struct signature — mechanical.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/env_checkpoint.go server/internal/handler/env_checkpoint.go server/internal/service/env_checkpoint_test.go server/internal/handler/env_checkpoint_test.go
git commit -m "feat(env-checkpoint): accept lane count and lane key on resume"
```

______________________________________________________________________

## Task 8: `[multica]` Lane claim, materialization, and interruption recovery

**tasks.md:** 3.2, 3.3, 3.6, 3.10, 3.11

**Files:**

- Create: `multica/server/internal/service/env_checkpoint_lane.go`
- Create: `multica/server/internal/service/env_checkpoint_lane_test.go`
- Modify: `multica/server/internal/service/env_checkpoint.go` (`resumeSnapshotLanes`)

**Interfaces:**

- Consumes: `Savepoint`, `ContinuationRegistry`, `ResumeAgentRunner`, `LaneRef`, the
  Task 6 queries.
- Produces:

```go
const (
    LaneStatusProvisioning = "provisioning"
    LaneStatusReady        = "ready"
    LaneStatusFailed       = "failed"
)

var ErrSavepointGone = errors.New("savepoint_gone")

type EnvCheckpointLane struct {
    ID, CheckpointID, WorkspaceID, LaneKey, Status string
    InstanceID, ProjectID, RuntimeID, TaskID, EnvID string
    Error string
}

// EnvCheckpointLaneRepository is the persistence seam for lane records.
// ClaimLane returns won=false when the unique index rejected the insert; the
// caller then reads the existing row and branches on its status.
type EnvCheckpointLaneRepository interface {
    ClaimLane(ctx context.Context, checkpointID, workspaceID, laneKey string) (lane EnvCheckpointLane, won bool, err error)
    GetLane(ctx context.Context, checkpointID, laneKey string) (EnvCheckpointLane, error)
    ListLanes(ctx context.Context, checkpointID string) ([]EnvCheckpointLane, error)
    RecordLaneStep(ctx context.Context, laneID, workspaceID string, step LaneStep) (EnvCheckpointLane, error)
    MarkLaneReady(ctx context.Context, laneID, workspaceID string) (EnvCheckpointLane, error)
    MarkLaneFailed(ctx context.Context, laneID, workspaceID, reason string) (EnvCheckpointLane, error)
    CountProvisioningLanes(ctx context.Context, checkpointID string) (int, error)
}

// LaneStep carries one filled materialization id. Exactly one field is set per
// call; empty fields leave the stored value untouched.
type LaneStep struct {
    InstanceID, ProjectID, RuntimeID, TaskID, EnvID string
}

// LaneMaterializer performs the four materialization steps, in this order:
// instance from the savepoint, project subtree copy, runtime mint, then the
// task enqueue (which goes through the continuation seam, not this interface).
type LaneMaterializer interface {
    CreateLaneInstance(ctx context.Context, in LaneInstanceInput) (SandboxInstanceRef, error)
    CopyLaneProjectSubtree(ctx context.Context, in LaneProjectInput) (projectID, envID string, err error)
    MintLaneRuntime(ctx context.Context, in LaneRuntimeInput) (runtimeID, daemonID string, err error)
}

type LaneInstanceInput struct {
    WorkspaceID, ActorUserID, LaneKey string
    Savepoint Savepoint
}
type LaneProjectInput struct {
    WorkspaceID, ActorUserID, LaneKey, SourceProjectID string
}
type LaneRuntimeInput struct {
    WorkspaceID, ActorUserID, LaneKey, AgentID, InstanceID string
}

func (s *EnvCheckpointService) WithLanes(repo EnvCheckpointLaneRepository, mat LaneMaterializer, savepoints SavepointReader) *EnvCheckpointService

// SavepointReader lists a checkpoint's savepoints and marks one failed when its
// underlying snapshot is gone.
type SavepointReader interface {
    ListSavepoints(ctx context.Context, checkpointID, workspaceID string) ([]Savepoint, error)
    MarkSavepointFailed(ctx context.Context, snapshotID, workspaceID, reason string) error
}
```

Add `LaneEnvID string` to `LaneRef` (Task 1) and switch `forkedRuntimeContinuation` to
use it for `ChannelRunInput.EnvID`.

- [ ] **Step 1: Write the failing load-bearing regression test**

This is the single regression guard for the whole change (Design Doc "Testing
strategy"). Create `multica/server/internal/service/env_checkpoint_lane_test.go` with
the `newFanoutFixture` helper defined below the tests, plus:

```go
func TestThreeLanesTriggerOneSnapshotPerSourceInstance(t *testing.T) {
	svc, deps := newFanoutFixture(t)
	creator, savepoints, mat, forked := deps.creator, deps.savepoints, deps.mat, deps.forked

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 3, LaneKeyAnchor: "dispatch-abc",
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	// THE invariant: one snapshot per source instance, not one per lane.
	if len(creator.calls) != 0 {
		t.Fatalf("resume must take no new snapshot, got %d", len(creator.calls))
	}
	if savepoints.listCalls != 1 {
		t.Fatalf("resume must read the checkpoint's savepoints once, got %d", savepoints.listCalls)
	}
	if len(res.Lanes) != 3 {
		t.Fatalf("lanes = %d, want 3", len(res.Lanes))
	}
	if len(mat.instanceCalls) != 3 {
		t.Fatalf("want 3 lane instances, got %d", len(mat.instanceCalls))
	}
	for _, c := range mat.instanceCalls {
		if c.Savepoint.CubeSnapshotID != "cube-1" {
			t.Fatalf("lane instance must come from the single savepoint, got %q", c.Savepoint.CubeSnapshotID)
		}
	}
	if len(mat.projectCalls) != 3 || len(mat.runtimeCalls) != 3 {
		t.Fatalf("each lane needs its own subtree and runtime: subtrees=%d runtimes=%d", len(mat.projectCalls), len(mat.runtimeCalls))
	}
	seen := map[string]bool{}
	for _, c := range forked.calls {
		if seen[c.Lane.RuntimeID] {
			t.Fatalf("lanes must not share a runtime: %q", c.Lane.RuntimeID)
		}
		seen[c.Lane.RuntimeID] = true
	}
	if len(forked.calls) != 3 {
		t.Fatalf("want 3 continuations, got %d", len(forked.calls))
	}
}

func TestRepeatedLaneKeyReturnsExistingLane(t *testing.T) {
	svc, deps := newFanoutFixture(t)

	first, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 1, LaneKeyAnchor: "same",
	})
	if err != nil {
		t.Fatalf("first resume: %v", err)
	}
	second, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 1, LaneKeyAnchor: "same",
	})
	if err != nil {
		t.Fatalf("second resume: %v", err)
	}
	if len(deps.mat.instanceCalls) != 1 {
		t.Fatalf("retry must not materialize a second sandbox, instances = %d", len(deps.mat.instanceCalls))
	}
	if first.Lanes[0].LaneKey != second.Lanes[0].LaneKey {
		t.Fatalf("lane key not stable: %q vs %q", first.Lanes[0].LaneKey, second.Lanes[0].LaneKey)
	}
	if first.Lanes[0].InstanceID != second.Lanes[0].InstanceID {
		t.Fatalf("retry must return the existing instance: %q vs %q", first.Lanes[0].InstanceID, second.Lanes[0].InstanceID)
	}
	if len(deps.forked.calls) != 1 {
		t.Fatalf("retry must not re-enqueue the continuation, calls = %d", len(deps.forked.calls))
	}
}

func TestNewLaneKeyReExpandsWithoutASecondCheckpoint(t *testing.T) {
	svc, deps := newFanoutFixture(t)

	for _, anchor := range []string{"anchor-a", "anchor-b"} {
		if _, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
			WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
			LaneCount: 1, LaneKeyAnchor: anchor,
		}); err != nil {
			t.Fatalf("resume %s: %v", anchor, err)
		}
	}
	all, err := deps.lanes.ListLanes(context.Background(), "cp-1")
	if err != nil {
		t.Fatalf("list lanes: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("a new anchor must add a lane, got %d", len(all))
	}
	if len(deps.repo.createCalls) != 0 {
		t.Fatalf("re-expansion must not create a second checkpoint, got %d", len(deps.repo.createCalls))
	}
	if len(deps.creator.calls) != 0 {
		t.Fatalf("re-expansion must reuse the savepoint, snapshots taken = %d", len(deps.creator.calls))
	}
}

func TestInterruptedLaneContinuesFromFirstIncompleteStep(t *testing.T) {
	svc, deps := newFanoutFixture(t)
	deps.lanes.seed(EnvCheckpointLane{
		ID: "l-0", CheckpointID: "cp-1", WorkspaceID: "ws",
		LaneKey: laneKey("dispatch-abc", 0), Status: LaneStatusProvisioning,
		InstanceID: "inst-recovered", // crashed after the sandbox, before the subtree
	})

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 1, LaneKeyAnchor: "dispatch-abc",
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(deps.mat.instanceCalls) != 0 {
		t.Fatalf("completed step must not re-run, instance calls = %d", len(deps.mat.instanceCalls))
	}
	if len(deps.mat.projectCalls) != 1 || len(deps.mat.runtimeCalls) != 1 {
		t.Fatalf("remaining steps must run once: subtrees=%d runtimes=%d", len(deps.mat.projectCalls), len(deps.mat.runtimeCalls))
	}
	if res.Lanes[0].InstanceID != "inst-recovered" {
		t.Fatalf("recovered lane must keep its sandbox, got %q", res.Lanes[0].InstanceID)
	}
	if res.Lanes[0].Status != LaneStatusReady {
		t.Fatalf("recovered lane status = %q, want ready", res.Lanes[0].Status)
	}
}

func TestLaneWithMissingSavepointFailsTypedAndMarksSavepointFailed(t *testing.T) {
	svc, deps := newFanoutFixture(t)
	deps.mat.instanceErr = ErrSavepointGone

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 1, LaneKeyAnchor: "dispatch-abc",
	})
	if !errors.Is(err, ErrSavepointGone) {
		t.Fatalf("expected typed ErrSavepointGone, got %v", err)
	}
	if res.Lanes[0].Status != LaneStatusFailed {
		t.Fatalf("lane status = %q, want failed", res.Lanes[0].Status)
	}
	if len(deps.savepoints.failed) != 1 || deps.savepoints.failed[0] != "snap-1" {
		t.Fatalf("savepoint must be marked failed so later lanes fail fast: %v", deps.savepoints.failed)
	}
}

func TestAllLanesFailedIsReportedAsFailure(t *testing.T) {
	svc, deps := newFanoutFixture(t)
	deps.mat.instanceErr = fmt.Errorf("cube unavailable")

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 3, LaneKeyAnchor: "dispatch-abc",
	})
	if err == nil {
		t.Fatal("all lanes failing must surface as an error, not a partial success")
	}
	if len(res.Lanes) != 3 {
		t.Fatalf("failure must still report every lane, got %d", len(res.Lanes))
	}
	for i, lane := range res.Lanes {
		if lane.Status != LaneStatusFailed {
			t.Fatalf("lane %d status = %q, want failed", i, lane.Status)
		}
		if lane.Error == "" {
			t.Fatalf("lane %d must carry a diagnosable error", i)
		}
	}
}

func TestLaneTaskEnqueueFailureIsPerLanePartial(t *testing.T) {
	svc, deps := newFanoutFixture(t)
	deps.forked.errOnCall = map[int]error{1: fmt.Errorf("enqueue rejected")} // 0-indexed: 2nd lane

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: "cp-1", ActorUserID: "u",
		LaneCount: 3, LaneKeyAnchor: "dispatch-abc",
	})
	if err != nil {
		t.Fatalf("a single trigger failure is partial, not fatal: %v", err)
	}
	want := []TriggerStatus{TriggerExecuted, TriggerFailed, TriggerExecuted}
	for i, lane := range res.Lanes {
		if lane.TriggerStatus != want[i] {
			t.Fatalf("lane %d trigger = %q, want %q", i, lane.TriggerStatus, want[i])
		}
		// The sandbox exists either way -- only the continuation failed.
		if lane.InstanceID == "" {
			t.Fatalf("lane %d lost its sandbox on trigger failure", i)
		}
	}
	if res.Status != ResumePartial {
		t.Fatalf("result status = %q, want partial", res.Status)
	}
}
```

`newFanoutFixture` is the shared arrange helper for this file; add it above the tests so
each case only states its deviation:

```go
type fanoutDeps struct {
	repo       *fakeCheckpointRepo
	creator    *fakeSavepointCreator
	savepoints *fakeSavepointReader
	lanes      *fakeLaneRepo
	mat        *fakeLaneMaterializer
	forked     *fakeContinuationStrategy
}

// newFanoutFixture builds a snapshot-mode checkpoint with exactly one ready
// savepoint over one source instance -- the shape every fan-out case starts from.
func newFanoutFixture(t *testing.T) (*EnvCheckpointService, *fanoutDeps) {
	t.Helper()
	d := &fanoutDeps{
		repo:    newFakeCheckpointRepo(),
		creator: &fakeSavepointCreator{},
		savepoints: &fakeSavepointReader{savepoints: []Savepoint{
			{SnapshotID: "snap-1", CubeSnapshotID: "cube-1", InstanceID: "src-1", Status: "ready"},
		}},
		lanes:  newFakeLaneRepo(),
		mat:    &fakeLaneMaterializer{},
		forked: &fakeContinuationStrategy{mode: SaveModeSnapshot},
	}
	d.repo.checkpoints["cp-1"] = EnvCheckpoint{
		ID: "cp-1", WorkspaceID: "ws", ProjectID: "proj",
		SaveMode: SaveModeSnapshot, SaveStatus: EnvCheckpointSaveComplete,
		SandboxRefs:   []SandboxInstanceRef{{InstanceID: "src-1", WorkspaceID: "ws"}},
		ResumeTrigger: json.RawMessage(`{"agent_id":"a-1","project_id":"proj","kind":"chat"}`),
	}
	svc := NewEnvCheckpointService(d.repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{},
		ContinuationRegistry{Forked: d.forked}).
		WithSavepointCreator(d.creator).
		WithLanes(d.lanes, d.mat, d.savepoints)
	return svc, d
}
```

`fakeContinuationStrategy` needs an `errOnCall map[int]error` field (keyed by call
index) in addition to the `err` field from Task 3; `fakeSavepointReader` needs a
`failed []string` recorder for `MarkSavepointFailed`. Extend both rather than adding
parallel fakes.

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestThreeLanes|TestRepeatedLaneKey|TestNewLaneKey|TestInterruptedLane|TestLaneWith|TestAllLanes|TestLaneTaskEnqueue' -v
```

Expected: FAIL to compile — `WithLanes`, `fakeLaneRepo`, `fakeLaneMaterializer`,
`fakeSavepointReader` undefined.

- [ ] **Step 3: Implement the lane types and fakes**

Create `multica/server/internal/service/env_checkpoint_lane.go` with the declarations
from **Produces**, and write the three fakes in the test file: `fakeLaneRepo` (a
`map[string]EnvCheckpointLane` keyed `checkpointID+"/"+laneKey` guarded by a
`sync.Mutex`, `ClaimLane` returning `won=false` on an existing key),
`fakeLaneMaterializer` (recording `instanceCalls []LaneInstanceInput`, `projectCalls`,
`runtimeCalls`, with per-step error fields), and `fakeSavepointReader` (canned
`savepoints`, a `listCalls` counter, and a `failed []string` slice).

- [ ] **Step 4: Implement `resumeSnapshotLanes`**

In `env_checkpoint.go`:

```go
func (s *EnvCheckpointService) resumeSnapshotLanes(ctx context.Context, cp EnvCheckpoint, in ResumeFromCheckpointInput) (ResumeFromCheckpointResult, error) {
	if s.lanes == nil || s.materializer == nil || s.savepointReader == nil {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: snapshot fan-out resume is not configured")
	}
	savepoints, err := s.savepointReader.ListSavepoints(ctx, cp.ID, in.WorkspaceID)
	if err != nil {
		return ResumeFromCheckpointResult{}, fmt.Errorf("list savepoints: %w", err)
	}
	if len(savepoints) == 0 {
		return ResumeFromCheckpointResult{}, fmt.Errorf("validation_failed: %w: checkpoint owns no savepoint", ErrCheckpointNotResumable)
	}
	strategy := s.continuations.For(SaveModeSnapshot)
	result := ResumeFromCheckpointResult{
		CheckpointID:  cp.ID,
		ProjectID:     cp.ProjectID,
		EnvIDMap:      cp.EnvIDMap,
		SandboxRefs:   cp.SandboxRefs,
		RolloutHandle: fmt.Sprintf("resume:%s", cp.ID),
	}
	var trigger ResumeTrigger
	hasTrigger := len(cp.ResumeTrigger) > 0
	if hasTrigger {
		if err := json.Unmarshal(cp.ResumeTrigger, &trigger); err != nil {
			return result, fmt.Errorf("unmarshal resume_trigger: %w", err)
		}
	}
	ready := 0
	for i := 0; i < in.LaneCount; i++ {
		laneKey := laneKeyForOrdinal(in.LaneKeyAnchor, i)
		lane, err := s.materializeLane(ctx, cp, in, laneKey, savepoints[0], trigger, hasTrigger, strategy)
		result.Lanes = append(result.Lanes, lane)
		if err == nil && lane.Status == LaneStatusReady {
			ready++
		}
	}
	if ready == 0 {
		result.TriggerStatus = TriggerFailed
		return result, fmt.Errorf("resume: all %d requested lanes failed", in.LaneCount)
	}
	if ready < len(result.Lanes) || !hasTrigger {
		// Partial fan-out or no descriptor: never reported as fully successful.
		result.TriggerStatus = TriggerFailed
		if !hasTrigger {
			result.TriggerStatus = TriggerSkippedLegacy
		}
		return result, nil
	}
	result.TriggerStatus = TriggerExecuted
	return result, nil
}
```

`materializeLane` claims the lane, then runs the four steps guarded by "skip if the lane
row already records this step's id", recording each id with `RecordLaneStep` before the
next step. On the claim losing, it reads the existing row: `LaneStatusReady` returns it
as-is, `LaneStatusFailed` returns it as failed, `LaneStatusProvisioning` continues from
the first unfilled step. On `errors.Is(err, ErrSavepointGone)` it calls
`s.savepointReader.MarkSavepointFailed(ctx, savepoint.SnapshotID, in.WorkspaceID, err.Error())`
before `MarkLaneFailed`. Finally it calls `strategy.ResumeAgentRun` with a `LaneRef`
built from the recorded step ids and `MarkLaneReady`.

Add
`laneKeyForOrdinal(anchor string, ordinal int) string { return fmt.Sprintf("%s#%d", anchor, ordinal) }`
— Task 9 pins the anchor itself.

- [ ] **Step 5: Run the tests**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -v
```

Expected: PASS, all lane tests plus every pre-existing service test.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/env_checkpoint_lane.go server/internal/service/env_checkpoint_lane_test.go server/internal/service/env_checkpoint.go server/internal/service/forked_runtime_continuation.go
git commit -m "feat(env-checkpoint): materialize N lanes from one savepoint"
```

______________________________________________________________________

## Task 9: `[multica]` Pin the lane-key anchor against a real stable dispatch id

**tasks.md:** 3.7

> Design D3 calls this "the single point where outer-retry safety is decided". The
> grounded facts: `EnvDispatchInput.IdempotencyKey` (`env_dispatch.go:65`) is an
> **optional** caller-supplied UUID, validated at `handler/env_dispatch.go:165`, and
> persisted in `env_dispatch_request (workspace_id, idempotency_key)` (unique,
> `migrations/127_environment_state.up.sql:40`) — but the ledger row is written only
> *after* the dispatch completes (`env_dispatch.go:750`), so a mid-flight retry finds no
> row. There is **no** `event_ref` on `EnvDispatchInput` (`event_ref` lives on
> `EnvCheckpointCreateInput`), so the Design Doc's "caller's `event_ref` plus an
> ordinal" option does not exist on the branch path.
>
> **Decision to implement:** the anchor is `EnvDispatchInput.IdempotencyKey`, and branch
> mode **requires** it. A branch dispatch without an idempotency key is rejected at
> validation rather than silently made retry-unsafe.

> **SUPERSEDED at execution time (user decision).** The anchor choice above stands and
> is now documented on `laneKeyForOrdinal`, but **the requirement is not enforced here**
> — it moves to Task 13, where branch dispatch actually starts deriving lane keys.
>
> Two facts forced the split. First, the AReaL client sends no idempotency key at all:
> `create_env_dispatch`'s payload (`multica_client.py:216`) carries mode, dispatch_type,
> group_size, training_mode and a few optional ids, and the string `idempotency` appears
> nowhere in `customized_areal/`. So the validation in Step 3 would reject every branch
> dispatch the current client can make — and `proposal.md:96` promises "no client
> contract change". Second, lane keys do not govern branch dispatch until Task 13
> reroutes it through resume, so enforcing now buys a retry-safety property that nothing
> yet consumes.
>
> **Do in this task:** Step 1's retry-stability test (kept, as
> `TestRetriedRequestDerivesTheSameLaneKeys` in `env_checkpoint_lane_test.go`, since
> `laneKeyForOrdinal` lives there) and Step 3's documentation half. **Skip** Step 3's
> validation and Step 5's client change.
>
> **Carry into Task 13:** (a) add the validation there, (b) the client must send the key
> *before* the server requires it — a server-first rollout fails every branch dispatch
> in the gap, and (c) it is a client-visible contract change, so `proposal.md:96`,
> tasks.md 4.5 ("Confirm no AReaL client change is required") and the protocol doc (Task
> 20\) all have to be corrected. `TestBranchDispatchStillAcceptsAKeylessRequest` is the
> tripwire holding the deferral in place; it was mutation-checked and fails as soon as
> the validation lands, which is also the signal to update those three documents.

**Files:**

- Modify: `multica/server/internal/service/env_dispatch.go` (`validate`, ~line 822)

- Modify: `multica/server/internal/service/env_checkpoint_lane.go` (`laneKeyForOrdinal`
  doc)

- Modify: `multica/server/internal/service/env_dispatch_test.go`

- [ ] **Step 1: Write the failing retry test**

```go
func TestBranchDispatchRequiresIdempotencyKeyAsLaneKeyAnchor(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	svc := newBranchMessageDispatchServiceForTest(t, f)
	in := branchMessageDispatchInputForTest()
	in.IdempotencyKey = ""

	_, err := svc.Dispatch(context.Background(), in)
	if err == nil || !strings.Contains(err.Error(), "validation_failed") {
		t.Fatalf("branch without an idempotency key must be rejected, got %v", err)
	}
}

func TestRetriedBranchRequestDerivesTheSameLaneKeys(t *testing.T) {
	anchor := "11111111-1111-1111-1111-111111111111"
	first := []string{}
	second := []string{}
	for i := 0; i < 3; i++ {
		first = append(first, laneKeyForOrdinal(anchor, i))
		second = append(second, laneKeyForOrdinal(anchor, i))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("lane key %d not retry-stable: %q vs %q", i, first[i], second[i])
		}
	}
	if laneKeyForOrdinal("22222222-2222-2222-2222-222222222222", 0) == first[0] {
		t.Fatal("a different anchor must produce a different lane key")
	}
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestBranchDispatchRequiresIdempotency|TestRetriedBranchRequest' -v
```

Expected: FAIL — the validation does not exist yet.

- [ ] **Step 3: Add the validation**

In `EnvDispatchService.validate`, after the existing `GroupSize` bound check:

```go
	// Lane keys for branch fan-out derive from the dispatch's idempotency key
	// (design D3). Without a retry-stable anchor a retried branch would present
	// fresh lane keys and double the lanes, so it is rejected rather than
	// silently made retry-unsafe.
	if in.Mode == EnvModeBranch && in.IdempotencyKey == "" {
		return fmt.Errorf("validation_failed: idempotency_key is required for branch dispatch (lane key anchor)")
	}
```

Document the anchor in `laneKeyForOrdinal`'s comment, naming
`env_dispatch_request.idempotency_key` as the source.

- [ ] **Step 4: Run and commit**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ ./internal/handler/
cd .. && git add server/internal/service/env_dispatch.go server/internal/service/env_checkpoint_lane.go server/internal/service/env_dispatch_test.go
git commit -m "feat(env-dispatch): require idempotency key as the branch lane-key anchor"
```

- [ ] **Step 5: `[areal]` Confirm the AReaL client already sends an idempotency key for
  branch**

```bash
cd /workspaces/leagent/backend/areal
rg -n "idempotency_key" customized_areal/tree_search/agents/multica_client.py
```

If branch dispatches do not currently send one, add it in
`multica_client.create_env_dispatch` (a `uuid4()` per logical branch request, reused
across retries of that request) and note it in the protocol doc in Task 20. This is the
only AReaL-side code change in the whole plan; flag it to the user before making it,
since the change's `proposal.md` states "no client contract change".

______________________________________________________________________

## Task 10: `[multica]` Sweeper for lanes stuck in `provisioning`

**tasks.md:** 3.12

**Files:**

- Create: `multica/server/internal/scheduler/jobs_env_checkpoint_lane_sweep.go`
- Create: `multica/server/internal/scheduler/jobs_env_checkpoint_lane_sweep_test.go`
- Modify: `multica/server/cmd/server/main.go` (registration block, ~lines 441–476)

**Interfaces:**

- Consumes: `scheduler.JobSpec`, `scheduler.Handler`,
  `scheduler.StaticScopes(ScopeGlobal)`,
  `Queries.ListStaleProvisioningEnvCheckpointLanes` / `MarkEnvCheckpointLaneFailed`
  (Task 6).

- Produces: `scheduler.EnvCheckpointLaneSweepJob(pool *pgxpool.Pool) JobSpec`,
  `const JobNameEnvCheckpointLaneSweep = "env_checkpoint_lane_sweep"`.

- [ ] **Step 1: Write the failing test**

Follow `internal/scheduler/jobs_memory_curation_test.go` for shape. Assert that
`EnvCheckpointLaneSweepJob(nil).Handler` returns
`HandlerResult{Result: map[string]any{"skipped": true, "reason": "database_unavailable"}}`
(mirroring `makeMemoryCurationIntentHandler`'s nil-pool branch) and that the spec's
`Name` is `JobNameEnvCheckpointLaneSweep` with `Scopes` equal to
`StaticScopes(ScopeGlobal)`.

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/scheduler/ -run TestEnvCheckpointLaneSweep -v
```

Expected: FAIL to compile — `EnvCheckpointLaneSweepJob` undefined.

- [ ] **Step 3: Implement the job**

```go
package scheduler

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const JobNameEnvCheckpointLaneSweep = "env_checkpoint_lane_sweep"

// envCheckpointLaneStaleAfter is how long a lane may sit in `provisioning`
// before the sweeper declares it abandoned. Lane materialization is a handful
// of sandbox jobs; 15 minutes is well past the sandboxd create timeout, so a
// lane still provisioning after that lost its owner to a crash.
const envCheckpointLaneStaleAfter = 15 * time.Minute

// EnvCheckpointLaneSweepJob fails lanes abandoned mid-materialization so their
// sandboxes stop being invisible and checkpoint deletion (design D4) is not
// blocked forever by a dead owner.
func EnvCheckpointLaneSweepJob(pool *pgxpool.Pool) JobSpec {
	return JobSpec{
		Name:              JobNameEnvCheckpointLaneSweep,
		Cadence:           5 * time.Minute,
		CatchUpMode:       CatchUpLatestOnly,
		MaxPlansPerTick:   1,
		RunTimeout:        2 * time.Minute,
		StaleTimeout:      5 * time.Minute,
		HeartbeatInterval: 30 * time.Second,
		AllowStaleReentry: true,
		MaxAttempts:       3,
		RetryBackoff:      []time.Duration{time.Minute, 2 * time.Minute, 5 * time.Minute},
		Scopes:            StaticScopes(ScopeGlobal),
		Handler: func(ctx context.Context, in HandlerInput) (HandlerResult, error) {
			if pool == nil {
				return HandlerResult{Result: map[string]any{"skipped": true, "reason": "database_unavailable"}}, nil
			}
			tag, err := pool.Exec(ctx, `
				UPDATE env_checkpoint_lane
				SET status = 'failed',
				    error = 'lane materialization abandoned; swept',
				    updated_at = now()
				WHERE status = 'provisioning' AND updated_at < now() - $1::interval
			`, envCheckpointLaneStaleAfter.String())
			if err != nil {
				return HandlerResult{}, err
			}
			swept := tag.RowsAffected()
			return HandlerResult{RowsAffected: swept, Result: map[string]any{"lanes_swept": swept}}, nil
		},
	}
}
```

- [ ] **Step 4: Register it**

In `multica/server/cmd/server/main.go`, next to the existing
`schedulerMgr.Register(...)` calls:

```go
	if err := schedulerMgr.Register(scheduler.EnvCheckpointLaneSweepJob(pool)); err != nil {
		return fmt.Errorf("register env checkpoint lane sweep job: %w", err)
	}
```

Match the surrounding error-handling shape exactly (read lines 441–476 first — some use
`log.Fatal`, some return).

- [ ] **Step 5: Run and commit**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go build ./... && go test ./internal/scheduler/
cd .. && git add server/internal/scheduler/jobs_env_checkpoint_lane_sweep.go server/internal/scheduler/jobs_env_checkpoint_lane_sweep_test.go server/cmd/server/main.go
git commit -m "feat(scheduler): sweep env checkpoint lanes stuck provisioning"
```

______________________________________________________________________

## Task 11: `[multica]` Phase 3 state-machine round trip

**tasks.md:** 3.13, 3.14 (3.14 was delivered in Task 6 Step 5)

**Files:**

- Modify: `multica/server/internal/service/env_checkpoint_lane_test.go`

- [ ] **Step 1: Write the round-trip test**

Mirror the round-trip style established in the `env-checkpoint-resume-trigger` change:

```go
func TestSnapshotCheckpointRoundTripRunningToLanes(t *testing.T) {
	// running -> snapshot save -> complete -> resume 3 -> 3 ready lanes,
	// each claimed by its own fake daemon runtime.
	repo := newFakeCheckpointRepo()
	creator := &fakeSavepointCreator{}
	savepoints := &fakeSavepointReader{}
	lanes := newFakeLaneRepo()
	mat := &fakeLaneMaterializer{}
	forked := &fakeContinuationStrategy{mode: SaveModeSnapshot}
	saver := &fakeCheckpointSaver{}
	svc := NewEnvCheckpointService(repo, saver, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{snapshot: json.RawMessage(`{}`)},
		&fakeInFlightResolver{triggers: []ResumeTrigger{{TaskID: "t-1", RuntimeID: "r-1", AgentID: "a-1", ProjectID: "proj", Kind: "chat"}}},
		ContinuationRegistry{Forked: forked}).
		WithSavepointCreator(creator).
		WithLanes(lanes, mat, savepoints)

	cp, err := svc.Create(context.Background(), EnvCheckpointCreateInput{
		WorkspaceID: "ws", ProjectID: "proj", SaveMode: SaveModeSnapshot,
		SandboxRefs: []SandboxInstanceRef{{InstanceID: "src-1", WorkspaceID: "ws"}},
		ActorUserID: "u", SaveTimeout: 5 * time.Second,
	})
	if err != nil || cp.SaveStatus != EnvCheckpointSaveComplete {
		t.Fatalf("create: cp=%+v err=%v", cp, err)
	}
	// The savepoint the create produced is what resume reads back.
	savepoints.savepoints = []Savepoint{{SnapshotID: "snap-1", CubeSnapshotID: "cube-1", InstanceID: "src-1", Status: "ready"}}

	res, err := svc.ResumeFromCheckpoint(context.Background(), ResumeFromCheckpointInput{
		WorkspaceID: "ws", CheckpointID: cp.ID, ActorUserID: "u",
		LaneCount: 3, LaneKeyAnchor: "dispatch-abc",
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(res.Lanes) != 3 {
		t.Fatalf("lanes = %d, want 3", len(res.Lanes))
	}
	for _, lane := range res.Lanes {
		if lane.Status != LaneStatusReady {
			t.Fatalf("lane %s status = %s, want ready", lane.LaneKey, lane.Status)
		}
		if lane.TriggerStatus != TriggerExecuted {
			t.Fatalf("lane %s continuation = %s, want executed", lane.LaneKey, lane.TriggerStatus)
		}
	}
	if len(creator.calls) != 1 {
		t.Fatalf("the whole round trip must take exactly one snapshot, got %d", len(creator.calls))
	}
}
```

- [ ] **Step 2: Run the full Phase 3 test set**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -v
DATABASE_URL="$DATABASE_URL" go test ./internal/service/ -run '_Integration' -v
```

Expected: PASS.

- [ ] **Step 3: Commit and tick tasks.md**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/env_checkpoint_lane_test.go
git commit -m "test(env-checkpoint): snapshot save to N ready lanes round trip"
```

Then in `[areal]`, tick `tasks.md` 3.1–3.14.

______________________________________________________________________

# Phase 4 — Route branch dispatch through resume

**This is the only phase that changes externally observable branch behavior. Land phases
1–3 first, and stage this one alone.**

## Task 12: `[multica]` Pin the branch dispatch contract before changing it

**tasks.md:** 4.2 (guard), 4.4 (first half)

**Files:**

- Create: `multica/server/internal/handler/env_dispatch_branch_contract_test.go`

- [ ] **Step 1: Write the contract-pinning serialization test**

This is the Design Doc's "Phase 4 guard". Capture the response bytes **before** the
routing switch so drift is detectable:

```go
package handler

import (
	"encoding/json"
	"testing"

	"github.com/multica-ai/multica/server/internal/service"
)

// TestBranchDispatchResponseShapeIsPinned freezes the AReaL-facing branch
// contract. Routing branch through checkpoint resume (phase 4) must not change
// these bytes. A failure here means the client contract moved.
func TestBranchDispatchResponseShapeIsPinned(t *testing.T) {
	resp := EnvDispatchResponse{
		ChannelID: "ch-1",
		ProjectID: "proj-1",
		Rollouts: []EnvRolloutResponse{{
			ChannelID:      "ch-1",
			LeaderRunID:    "run-1",
			AgentSandboxes: map[string]service.AgentSandboxStatus{"a-1": {Status: "ready", SandboxInstanceID: "inst-1", RuntimeID: "rt-1"}},
			EnvID:          "env-child-1",
			ProjectID:      "proj-1",
			ChatSessionID:  "cs-1",
			AgentRunID:     "run-1",
			SandboxRefs:    []service.SandboxInstanceRef{{InstanceID: "inst-1", WorkspaceID: "ws-1", NodeID: "node-1"}},
		}},
	}
	got, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	const want = `{"channel_id":"ch-1","project_id":"proj-1","rollouts":[{"channel_id":"ch-1","leader_run_id":"run-1","agent_sandboxes":{"a-1":{"status":"ready","sandbox_instance_id":"inst-1","runtime_id":"rt-1"}},"env_id":"env-child-1","project_id":"proj-1","chat_session_id":"cs-1","agent_run_id":"run-1","sandbox_refs":[{"instance_id":"inst-1","workspace_id":"ws-1","node_id":"node-1"}]}]}`
	if string(got) != want {
		t.Fatalf("branch dispatch contract drifted.\n got: %s\nwant: %s", got, want)
	}
}
```

- [ ] **Step 2: Run it and correct `want` from the actual output**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/handler/ -run TestBranchDispatchResponseShapeIsPinned -v
```

Expected: it may FAIL first with the real bytes in `got` (`AgentSandboxStatus`'s and
`SandboxInstanceRef`'s json tags decide the exact field order). Paste the real `got`
into `want` and re-run until PASS. Do **not** relax the assertion to a field-subset
check — the whole point is byte-level pinning.

- [ ] **Step 3: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/handler/env_dispatch_branch_contract_test.go
git commit -m "test(env-dispatch): pin the branch dispatch response contract"
```

______________________________________________________________________

## Task 13: `[multica]` Serve branch dispatch by checkpoint resume

**tasks.md:** 4.1, 4.3, 4.4, 4.5

> **Carried in from Task 9 (do not skip).** This is where branch dispatch first derives
> lane keys, so it is where the anchor requirement belongs. Add the validation Task 9
> deferred — `in.Mode == EnvModeBranch && in.IdempotencyKey == ""` rejected in
> `EnvDispatchService.validate`, right after the `GroupSize` bound check — and delete
> the tripwire `TestBranchDispatchStillAcceptsAKeylessRequest` in the same commit,
> inverting it into a rejection test.
>
> Doing so is a client-visible contract change with a rollout order: **the AReaL client
> must send `idempotency_key` for branch before the server requires it** (add it in
> `multica_client.create_env_dispatch` as one `uuid4()` per logical branch request,
> reused across retries of that request — it must be minted by the caller, not per HTTP
> attempt, or retries present fresh anchors and double the lanes). A server-first
> rollout rejects every branch dispatch in the gap. Also correct `proposal.md:96` ("no
> client contract change") and tasks.md 4.5, which currently asserts the opposite, and
> note it in the protocol doc (Task 20).

**Files:**

- Modify: `multica/server/internal/service/env_dispatch.go`
  (`dispatchBranchChannelMessage` lines 1438–1541; `EnvDispatchDeps` around line 289)
- Modify: `multica/server/internal/handler/env_dispatch.go` (deps adapter)
- Delete: `provisionEnvDispatchAgentBranch` from
  `multica/server/internal/handler/env_dispatch_channel_provision.go` (lines 326–382)
  plus its call site at line 243
- Modify: `multica/server/internal/service/env_dispatch_test.go`

**Interfaces:**

- Consumes: `EnvCheckpointService.Create` (snapshot mode),
  `EnvCheckpointService.ResumeFromCheckpoint`, `ResumeLane`.
- Produces:

```go
// BranchCheckpointResumer serves branch dispatch by creating or reusing a
// snapshot checkpoint at the requested env and resuming it into N lanes.
type BranchCheckpointResumer interface {
    ResumeBranchLanes(ctx context.Context, in BranchResumeInput) ([]ResumeLane, error)
}

type BranchResumeInput struct {
    WorkspaceID, ActorUserID string
    SourceEnvID, SourceProjectID string
    LaneCount int
    LaneKeyAnchor string
}

func (s *EnvDispatchService) WithBranchResumer(r BranchCheckpointResumer) *EnvDispatchService
```

- [ ] **Step 1: Write the failing behavior tests**

```go
func TestBranchDispatchMaterializesLanesThroughCheckpointResume(t *testing.T) {
	f := newFakeEnvDispatchDeps()
	resumer := &fakeBranchResumer{lanes: []ResumeLane{{
		LaneKey: "anchor#0", Status: LaneStatusReady,
		InstanceID: "inst-0", ProjectID: "proj-0", RuntimeID: "rt-0",
		TaskID: "run-0", EnvID: "env-0", TriggerStatus: TriggerExecuted,
	}}}
	svc := newBranchMessageDispatchServiceForTest(t, f).WithBranchResumer(resumer)

	res, err := svc.Dispatch(context.Background(), branchMessageDispatchInputForTest())
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if len(resumer.calls) != 1 {
		t.Fatalf("branch must go through checkpoint resume, calls = %d", len(resumer.calls))
	}
	if res.Rollouts[0].AgentRunID != "run-0" || res.Rollouts[0].EnvID != "env-0" {
		t.Fatalf("rollout not built from the lane: %+v", res.Rollouts[0])
	}
	if len(f.cloneCalls) != 0 {
		t.Fatalf("the direct branch provisioning path must be gone, clone calls = %d", len(f.cloneCalls))
	}
}

func TestBranchDispatchLeavesSourceEnvRunning(t *testing.T) {
	f := newBranchDispatchFake(t)
	resumer := &fakeBranchResumer{result: ResumeFromCheckpointResult{Lanes: []ResumeLane{{
		LaneKey: "anchor#0", Status: LaneStatusReady, InstanceID: "inst-0",
		ProjectID: "proj-0", RuntimeID: "rt-0", TaskID: "run-0", EnvID: "env-0",
		TriggerStatus: TriggerExecuted,
	}}}}
	svc := newBranchMessageDispatchServiceForTest(t, f).WithBranchResumer(resumer)

	if _, err := svc.Dispatch(context.Background(), branchMessageDispatchInputForTest()); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if len(f.saveCalls) != 0 {
		t.Fatalf("snapshot mode must not stop the source, stop jobs = %d", len(f.saveCalls))
	}
	if len(f.taskResets) != 0 {
		t.Fatalf("branch must not reset the source task, resets = %v", f.taskResets)
	}
	if resumer.calls[0].SaveMode != SaveModeSnapshot {
		t.Fatalf("branch must resume a snapshot checkpoint, got %q", resumer.calls[0].SaveMode)
	}
}

func TestBranchDispatchGivesEachLaneItsOwnRuntimeAndSubtree(t *testing.T) {
	f := newBranchDispatchFake(t)
	lanes := make([]ResumeLane, 3)
	for i := range lanes {
		lanes[i] = ResumeLane{
			LaneKey: fmt.Sprintf("anchor#%d", i), Status: LaneStatusReady,
			InstanceID: fmt.Sprintf("inst-%d", i),
			ProjectID:  fmt.Sprintf("proj-%d", i),
			RuntimeID:  fmt.Sprintf("rt-%d", i),
			TaskID:     fmt.Sprintf("run-%d", i),
			EnvID:      fmt.Sprintf("env-%d", i),
			TriggerStatus: TriggerExecuted,
		}
	}
	svc := newBranchMessageDispatchServiceForTest(t, f).
		WithBranchResumer(&fakeBranchResumer{result: ResumeFromCheckpointResult{Lanes: lanes}})

	res, err := svc.Dispatch(context.Background(), branchMessageDispatchInputForTest())
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if len(res.Rollouts) != 3 {
		t.Fatalf("rollouts = %d, want 3", len(res.Rollouts))
	}
	seenRuntime, seenProject := map[string]bool{}, map[string]bool{}
	for i, r := range res.Rollouts {
		if seenRuntime[r.RuntimeID] {
			t.Fatalf("rollout %d shares runtime %q", i, r.RuntimeID)
		}
		if seenProject[r.ProjectID] {
			t.Fatalf("rollout %d shares project subtree %q", i, r.ProjectID)
		}
		seenRuntime[r.RuntimeID], seenProject[r.ProjectID] = true, true
	}
}
```

`newBranchDispatchFake` extends the existing `env_dispatch_test.go` fake with
`cloneCalls`, `saveCalls`, and `taskResets` recorders; `fakeBranchResumer` records
`calls []EnvCheckpointCreateInput` and returns a canned `ResumeFromCheckpointResult`.

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run TestBranchDispatch -v
```

Expected: FAIL to compile — `WithBranchResumer` / `fakeBranchResumer` undefined.

- [ ] **Step 3: Add the resumer seam and rewrite `dispatchBranchChannelMessage`**

Add the `BranchCheckpointResumer` / `BranchResumeInput` declarations and
`WithBranchResumer`. Rewrite `dispatchBranchChannelMessage` so that, after the existing
trigger remap and the optional `CreateChannelMessage` context append, it calls:

```go
	lanes, err := s.branchResumer.ResumeBranchLanes(ctx, BranchResumeInput{
		WorkspaceID:     in.WorkspaceID,
		ActorUserID:     in.UserID,
		SourceEnvID:     in.EnvID,
		SourceProjectID: in.SourceProjectID,
		LaneCount:       1, // one rollout per resetOne iteration; group_size fans out above
		LaneKeyAnchor:   in.IdempotencyKey,
	})
	if err != nil {
		r.Error = fmt.Sprintf("resume branch lanes: %v", err)
		r.Stack = stackerr.StackOf(err)
		return
	}
	if len(lanes) == 0 || lanes[0].Status != LaneStatusReady {
		r.Error = "resume branch lanes: no ready lane"
		return
	}
	lane := lanes[0]
	r.EnvID = lane.EnvID
	r.ProjectID = lane.ProjectID
	r.ChatSessionID = lane.ChatSessionID
	r.LeaderRunID = lane.TaskID
	r.AgentRunID = lane.TaskID
```

then keep the existing `SaveCollaborationTrigger` and `r.AgentSandboxes` population,
sourcing `SandboxInstanceID` / `RuntimeID` from `lane`. The `ProvisionEnvDispatchAgent`
call and its `DeleteAgentRuntime` defer are removed — the lane materializer owns
provisioning now.

`ResumeLane.ChatSessionID` was declared in Task 7; confirm `materializeLane` (Task 8)
records it from the lane's copied subtree, since this is its only consumer.

- [ ] **Step 4: Wire the production resumer and delete the dead branch path**

In `handler/env_dispatch.go`'s deps adapter, implement `ResumeBranchLanes` as: look up
the existing `save_mode='snapshot'`, `save_status='complete'` checkpoint for
`(SourceEnvID, SourceProjectID)`; if none, `EnvCheckpointService.Create` one in snapshot
mode; then `ResumeFromCheckpoint` with the requested `LaneCount` and `LaneKeyAnchor`,
returning `res.Lanes`. Reusing the existing checkpoint keyed on `env_id` is what makes
re-expansion a new lane key on the same checkpoint rather than a second checkpoint
(design D2).

Delete `provisionEnvDispatchAgentBranch` (`env_dispatch_channel_provision.go:326-382`)
and the `if sourceID != "" { return h.provisionEnvDispatchAgentBranch(...) }` dispatch
at line 243. Remove the now-stale comment at lines 143–147 ("The branch path (source
sandbox filesystem clone) still uses the legacy pre-create flow…").

- [ ] **Step 5: Verify the contract test still passes and confirm the AReaL side**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go build ./... && go test ./internal/handler/ -run TestBranchDispatchResponseShapeIsPinned -v
go test ./internal/service/ ./internal/handler/
```

Expected: the pinned contract test PASSES unchanged — that is the `tasks.md` 4.2
evidence.

For `tasks.md` 4.5, confirm in `[areal]` that no client change is required:

```bash
cd /workspaces/leagent/backend/areal
rg -n "def create_checkpoint" -A 45 customized_areal/tree_search/agents/multica_client.py
```

Expected: `create_checkpoint` still POSTs to `/api/v1/env-checkpoints` and returns the
checkpoint dict on `201`, i.e. a terminal save status synchronously. Record that in the
verify notes.

- [ ] **Step 6: Commit and tick tasks.md**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/internal/service/env_dispatch.go server/internal/handler/env_dispatch.go server/internal/handler/env_dispatch_channel_provision.go server/internal/service/env_dispatch_test.go
git commit -m "feat(env-dispatch): serve branch mode by fanning out checkpoint resume"
```

Then in `[areal]`, tick `tasks.md` 4.1–4.5.

______________________________________________________________________

# Phase 5 — Retire the `clone` job type

**Cross-repo/cross-process coordination point:** this phase changes the sandboxd job
contract. Server and sandboxd must roll out together (`tasks.md` 5.5).

## Task 14: `[multica]` Replace `clone` with `create_template` + `create`

**tasks.md:** 5.1, 5.2, 5.3, 5.4

**Files:**

- Modify: `multica/server/cmd/multica/cmd_sandboxd.go` (capabilities list line 201;
  `callCube` case line 493; delete `cloneCubeSandbox` lines 512–530)

- Modify: `multica/server/internal/service/env_sandbox_lifecycle.go` (delete
  `CloneSandboxInstanceInput` lines 86–98 and `CloneSandboxInstance` lines 215–267)

- Modify: `multica/server/internal/service/env_sandbox_lifecycle_test.go` (delete
  `TestCloneSandboxInstanceCreatesOfflineRuntimeAndCloneJob` line 600 and
  `TestCloneSandboxInstanceCompensatesRuntimeWhenJobInsertFails` line 628)

- Modify: `multica/server/internal/handler/sandbox.go` (`CompleteSandboxJob` case
  `"create", "clone"` line 1697)

- Create: `multica/server/migrations/246_sandbox_job_drop_clone.up.sql` / `.down.sql`

- [ ] **Step 1: Write the failing "no clone job is enqueued" test**

Append to `multica/server/internal/service/env_sandbox_lifecycle_test.go`:

```go
func TestNoPathEnqueuesACloneJob(t *testing.T) {
	deps := newFakeEnvSandboxLifecycleDepsWithSource(t)
	svc := NewEnvSandboxLifecycleService(deps, time.Second)
	ctx := context.Background()

	// Every lifecycle entry point the service exposes.
	_, _ = svc.Create(ctx, CreateSandboxInstanceInput{WorkspaceID: "ws", Template: "cube-1", DaemonEnabled: true}, "u")
	_, _ = svc.Save(ctx, SandboxInstanceRef{WorkspaceID: "ws", InstanceID: "inst-1"}, "u")
	_, _ = svc.Resume(ctx, SandboxInstanceRef{WorkspaceID: "ws", InstanceID: "inst-1"}, "u")
	_ = svc.Delete(ctx, SandboxInstanceRef{WorkspaceID: "ws", InstanceID: "inst-1"}, "u")

	for _, job := range deps.jobs {
		if job.JobType == "clone" {
			t.Fatalf("clone job enqueued by %s path: %+v", job.JobType, job)
		}
	}
}

func TestLaneCreateUsesTheSavepointTemplate(t *testing.T) {
	deps := &fakeEnvSandboxLifecycleDeps{refs: map[string]SandboxInstanceRef{}}
	svc := NewEnvSandboxLifecycleService(deps, time.Second)

	// A lane is created from the savepoint's Cube template id, exactly like any
	// other create -- no fused snapshot+create+delete job.
	if _, err := svc.Create(context.Background(), CreateSandboxInstanceInput{
		WorkspaceID: "ws", Template: "cube-snapshot-abc", DaemonEnabled: true,
		RuntimeEnv: map[string]string{"MULTICA_DAEMON_ID": "d-0"},
	}, "u"); err != nil {
		t.Fatalf("create: %v", err)
	}
	if len(deps.jobs) != 1 || deps.jobs[0].JobType != "create" {
		t.Fatalf("jobs = %+v, want a single create job", deps.jobs)
	}
	var payload map[string]any
	if err := json.Unmarshal(deps.jobs[0].Payload, &payload); err != nil {
		t.Fatalf("decode create payload: %v", err)
	}
	if payload["template"] != "cube-snapshot-abc" {
		t.Fatalf("lane create template = %v, want the savepoint template", payload["template"])
	}
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run 'TestNoPathEnqueuesAClone|TestLaneCreateUsesTheSavepoint' -v
```

Expected: `TestLaneCreateUsesTheSavepointTemplate` may already pass (create already
honours an explicit template — see `createCubeSandbox`'s "Explicit job template wins
over the node default"); `TestNoPathEnqueuesACloneJob` fails to compile until the helper
exists.

- [ ] **Step 3: Remove the sandboxd `clone` handler**

In `cmd_sandboxd.go`: drop `"clone"` from the capabilities slice at line 201; delete the
`case "clone": return c.cloneCubeSandbox(...)` arm; delete the whole `cloneCubeSandbox`
function. Leave `createCubeSnapshotTemplate` and `createCubeSandbox` untouched —
together they are the replacement.

Also delete the now-false comment in `createCubeSnapshotTemplate`: line 603 says
"Snapshots can take several minutes while Cube pauses the sandbox" and line 616–617 says
"Snapshot may leave the sandbox paused; resume so Multica runtime stays usable". The
prerequisite experiment established that a snapshot takes ~1.2s and leaves the source
`running`. Keep the 10-minute client timeout (harmless headroom for a large SWE sandbox)
but replace the comment with the measured behavior; **keep the defensive resume call**,
since removing it is not required and the Design Doc explicitly says no fix is needed.

- [ ] **Step 4: Remove `CloneSandboxInstance` and its callers**

Delete `CloneSandboxInstanceInput` and `CloneSandboxInstance` from
`env_sandbox_lifecycle.go`, and the two clone tests from
`env_sandbox_lifecycle_test.go`. Task 13 already deleted the only production caller
(`provisionEnvDispatchAgentBranch`); confirm:

```bash
cd /workspaces/leagent/backend/areal/multica/server
grep -rn "CloneSandboxInstance\|\"clone\"" --include=*.go . | grep -v _test.go
```

Expected: no output. In `handler/sandbox.go`, change `case "create", "clone":` to
`case "create":` in `CompleteSandboxJob`.

- [ ] **Step 5: Write migration 246**

`246_sandbox_job_drop_clone.up.sql`:

```sql
-- 'clone' fused snapshot + create + template delete and owned none durably.
-- create_template (a durable sandbox_snapshot row) plus one create per lane
-- replaces it, so N lanes share one snapshot and the savepoint outlives first
-- use. Existing 'clone' rows are historical job records; the CHECK only
-- constrains new inserts, so no data migration is required.
ALTER TABLE sandbox_job
    DROP CONSTRAINT IF EXISTS sandbox_job_type_check,
    ADD CONSTRAINT sandbox_job_type_check CHECK (type IN (
        'create', 'stop', 'resume', 'delete', 'reconfigure',
        'create_template', 'delete_template', 'exec', 'message'
    ));
```

`246_sandbox_job_drop_clone.down.sql` restores the value list from
`187_sandbox_job_restore_template_types.up.sql`:

```sql
ALTER TABLE sandbox_job
    DROP CONSTRAINT IF EXISTS sandbox_job_type_check,
    ADD CONSTRAINT sandbox_job_type_check CHECK (type IN (
        'create', 'stop', 'resume', 'delete', 'reconfigure', 'clone',
        'create_template', 'delete_template', 'exec', 'message'
    ));
```

> Before applying, check for live `clone` rows:
> `psql "$DATABASE_URL" -c "SELECT status, count(*) FROM sandbox_job WHERE type='clone' GROUP BY 1;"`.
> `ADD CONSTRAINT` validates existing rows by default, so any historical `clone` row
> will make the migration fail. If rows exist, either add `NOT VALID` to the new
> constraint or delete/retype terminal `clone` rows in the same migration — decide from
> the actual query output, and record the choice in the migration comment.

- [ ] **Step 6: Run everything**

```bash
cd /workspaces/leagent/backend/areal/multica
make migrate-up && make migrate-down && make migrate-up
cd server && go build ./... && go test ./internal/service/ ./internal/handler/ ./cmd/...
```

Expected: all green, no `clone` references remaining.

- [ ] **Step 7: Commit**

```bash
cd /workspaces/leagent/backend/areal/multica
git add server/cmd/multica/cmd_sandboxd.go server/internal/service/env_sandbox_lifecycle.go server/internal/service/env_sandbox_lifecycle_test.go server/internal/handler/sandbox.go server/migrations/246_sandbox_job_drop_clone.up.sql server/migrations/246_sandbox_job_drop_clone.down.sql
git commit -m "feat!: retire the clone sandbox job type in favor of create_template + create"
```

______________________________________________________________________

## Task 15: `[areal]` Record the joint sandboxd/server rollout requirement

**tasks.md:** 5.5

**Files:**

- Modify: `openspec/changes/env-savepoint-consolidation/tasks.md` (tick 5.1–5.5)

- Modify: `openspec/changes/env-savepoint-consolidation/design.md` (Migration Plan
  section, after the "Phasing" list)

- [ ] **Step 1: Add the deployment note**

In `design.md`'s **Migration Plan**, after the phasing list, add:

```markdown
### Deployment ordering for the `clone` retirement

Retiring the `clone` job type is the only step that breaks a cross-process
contract, so `sandboxd` and the server must roll out together:

1. Deploy the server build that no longer enqueues `clone` jobs (phase 4 must
   already be live, since it removed the last caller).
2. Drain any in-flight `clone` jobs — check
   `SELECT status, count(*) FROM sandbox_job WHERE type='clone' GROUP BY 1;`
   and wait for no `queued`/`dispatched`/`running` rows.
3. Apply migration 246, which drops `clone` from `sandbox_job_type_check`.
4. Roll out the `sandboxd` build whose capability list no longer advertises
   `clone`.

Rolling out `sandboxd` first leaves a window where a server still enqueues a job
type no node can execute; applying migration 246 before step 2 rejects the
insert for an in-flight retry. The down migration restores the `clone` value, so
a rollback needs the old `sandboxd` build back as well.
```

- [ ] **Step 2: Tick and commit**

```bash
cd /workspaces/leagent/backend/areal
git add openspec/changes/env-savepoint-consolidation/tasks.md openspec/changes/env-savepoint-consolidation/design.md
git commit -m "docs(openspec): record joint sandboxd/server rollout for clone retirement"
```

______________________________________________________________________

# Phase 6 — Savepoint reclamation

## Task 16: `[multica]` Checkpoint deletion releases savepoints and is refused while lanes provision

**tasks.md:** 6.1, 6.2, 6.3

> **Discrepancy to respect:** `tasks.md` 6.1/6.2 assume a checkpoint deletion/expiry
> path exists. It does not. `multica/server/internal/handler/env_checkpoint.go` exposes
> only create, get, list, and resume; `pkg/db/queries/env_checkpoint.sql` has no
> `DELETE`; `grep -rn "DeleteEnvCheckpoint"` returns nothing. This task therefore
> **creates** the deletion path, which is a scope addition the plan makes explicit
> rather than smoothing over.

**Files:**

- Modify: `multica/server/pkg/db/queries/env_checkpoint.sql`
- Modify: `multica/server/internal/service/env_checkpoint.go`
- Modify: `multica/server/internal/handler/env_checkpoint.go`
- Modify: `multica/server/internal/handler/router.go` (route registration, near the
  other `env-checkpoints` routes)
- Modify: `multica/server/internal/service/env_checkpoint_test.go`

**Interfaces:**

- Produces: `Queries.DeleteEnvCheckpoint`; `EnvCheckpointRepository.DeleteCheckpoint`;
  `SavepointReleaser` interface with
  `ReleaseSavepoint(ctx, snapshotID, workspaceID, actorUserID string) error`;
  `(*EnvCheckpointService).Delete(ctx, workspaceID, checkpointID, actorUserID string) error`;
  `ErrCheckpointHasProvisioningLanes`; `Handler.DeleteEnvCheckpoint` on
  `DELETE /api/v1/env-checkpoints/{checkpointID}`.

- [ ] **Step 1: Write the failing tests**

```go
type fakeSavepointReleaser struct {
	released []string
	err      error
}

func (f *fakeSavepointReleaser) ReleaseSavepoint(_ context.Context, snapshotID, _, _ string) error {
	if f.err != nil {
		return f.err
	}
	f.released = append(f.released, snapshotID)
	return nil
}

func TestDeleteCheckpointSchedulesSavepointDeletionAndRemovesLanes(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{ID: "cp-1", WorkspaceID: "ws", SaveMode: SaveModeSnapshot, SaveStatus: EnvCheckpointSaveComplete}
	savepoints := &fakeSavepointReader{savepoints: []Savepoint{{SnapshotID: "snap-1", Status: "ready"}}}
	releaser := &fakeSavepointReleaser{}
	lanes := newFakeLaneRepo()
	lanes.seed(EnvCheckpointLane{ID: "l-1", CheckpointID: "cp-1", WorkspaceID: "ws", LaneKey: "k0", Status: LaneStatusReady})
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, ContinuationRegistry{}).
		WithLanes(lanes, &fakeLaneMaterializer{}, savepoints).
		WithSavepointReleaser(releaser)

	if err := svc.Delete(context.Background(), "ws", "cp-1", "u"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if len(releaser.released) != 1 || releaser.released[0] != "snap-1" {
		t.Fatalf("savepoint not scheduled for deletion: %v", releaser.released)
	}
	if _, err := repo.GetCheckpoint(context.Background(), "cp-1", "ws"); err == nil {
		t.Fatal("checkpoint row must be deleted (lanes cascade with it)")
	}
}

func TestDeleteCheckpointRefusedWhileLaneProvisioning(t *testing.T) {
	repo := newFakeCheckpointRepo()
	repo.checkpoints["cp-1"] = EnvCheckpoint{ID: "cp-1", WorkspaceID: "ws", SaveMode: SaveModeSnapshot, SaveStatus: EnvCheckpointSaveComplete}
	savepoints := &fakeSavepointReader{savepoints: []Savepoint{{SnapshotID: "snap-1", Status: "ready"}}}
	releaser := &fakeSavepointReleaser{}
	lanes := newFakeLaneRepo()
	lanes.seed(EnvCheckpointLane{ID: "l-1", CheckpointID: "cp-1", WorkspaceID: "ws", LaneKey: "k0", Status: LaneStatusProvisioning, InstanceID: "inst-0"})
	svc := NewEnvCheckpointService(repo, &fakeCheckpointSaver{}, &fakeCheckpointResumer{},
		&fakeProjectSnapshotReader{}, &fakeInFlightResolver{}, ContinuationRegistry{}).
		WithLanes(lanes, &fakeLaneMaterializer{}, savepoints).
		WithSavepointReleaser(releaser)

	err := svc.Delete(context.Background(), "ws", "cp-1", "u")
	if !errors.Is(err, ErrCheckpointHasProvisioningLanes) {
		t.Fatalf("expected ErrCheckpointHasProvisioningLanes, got %v", err)
	}
	if len(releaser.released) != 0 {
		t.Fatalf("refused deletion must retain the savepoint, released = %v", releaser.released)
	}
	if lane, _ := lanes.GetLane(context.Background(), "cp-1", "k0"); lane.InstanceID != "inst-0" {
		t.Fatal("refused deletion must retain the lane and its sandbox")
	}
	if _, err := repo.GetCheckpoint(context.Background(), "cp-1", "ws"); err != nil {
		t.Fatal("refused deletion must retain the checkpoint row")
	}
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go test ./internal/service/ -run TestDeleteCheckpoint -v
```

Expected: FAIL to compile — `Delete`, `WithSavepointReleaser`,
`ErrCheckpointHasProvisioningLanes` undefined.

- [ ] **Step 3: Add the query**

Append to `pkg/db/queries/env_checkpoint.sql`:

```sql
-- name: DeleteEnvCheckpoint :exec
-- Cascades sandbox_snapshot.checkpoint_id ownership rows (migration 244) and
-- env_checkpoint_lane rows (migration 245). The Cube template itself is
-- released by a delete_template job before this runs.
DELETE FROM env_checkpoint
WHERE id = @id AND workspace_id = @workspace_id;
```

Regenerate (`make sqlc`, or hand-write per the constraint above).

- [ ] **Step 4: Implement the service delete**

```go
var ErrCheckpointHasProvisioningLanes = errors.New("checkpoint_has_provisioning_lanes")

// SavepointReleaser schedules a savepoint's Cube template for deletion through
// the existing delete_template job.
type SavepointReleaser interface {
	ReleaseSavepoint(ctx context.Context, snapshotID, workspaceID, actorUserID string) error
}

func (s *EnvCheckpointService) WithSavepointReleaser(r SavepointReleaser) *EnvCheckpointService {
	s.savepointReleaser = r
	return s
}

// Delete releases the checkpoint's savepoints and removes the row (lanes and
// savepoint ownership cascade). Deletion is refused while any lane is still
// provisioning: cascading those lane rows away would leave their sandboxes with
// no owning record, which is exactly what the lane status column exists to
// prevent (design D4).
func (s *EnvCheckpointService) Delete(ctx context.Context, workspaceID, checkpointID, actorUserID string) error {
	cp, err := s.repo.GetCheckpoint(ctx, checkpointID, workspaceID)
	if err != nil {
		return fmt.Errorf("not found: %w", err)
	}
	if s.lanes != nil {
		provisioning, err := s.lanes.CountProvisioningLanes(ctx, cp.ID)
		if err != nil {
			return fmt.Errorf("count provisioning lanes: %w", err)
		}
		if provisioning > 0 {
			return fmt.Errorf("%w: %d lane(s) still materializing", ErrCheckpointHasProvisioningLanes, provisioning)
		}
	}
	if s.savepointReader != nil && s.savepointReleaser != nil {
		savepoints, err := s.savepointReader.ListSavepoints(ctx, cp.ID, workspaceID)
		if err != nil {
			return fmt.Errorf("list savepoints: %w", err)
		}
		for _, sp := range savepoints {
			if err := s.savepointReleaser.ReleaseSavepoint(ctx, sp.SnapshotID, workspaceID, actorUserID); err != nil {
				return fmt.Errorf("release savepoint %s: %w", sp.SnapshotID, err)
			}
		}
	}
	return s.repo.DeleteCheckpoint(ctx, cp.ID, workspaceID)
}
```

Add `DeleteCheckpoint(ctx context.Context, checkpointID, workspaceID string) error` to
`EnvCheckpointRepository` and to `fakeCheckpointRepo`.

- [ ] **Step 5: Add the HTTP route**

Add `Delete(ctx context.Context, workspaceID, checkpointID, actorUserID string) error`
to `EnvCheckpointServiceAPI`, a `Handler.DeleteEnvCheckpoint` following the shape of
`ResumeEnvCheckpoint` (feature gate → `requireUserID` → workspace → nil-service 503 →
`parseUUIDOrBadRequest`), mapping `ErrCheckpointHasProvisioningLanes` → 409 and success
→ `w.WriteHeader(http.StatusNoContent)`. Register
`r.Delete("/{checkpointID}", h.DeleteEnvCheckpoint)` alongside the existing
`env-checkpoints` routes in `router.go`.

The production `SavepointReleaser` reuses the existing `delete_template` machinery:
`MarkSandboxSnapshotDeleting` →
`CreateSandboxJob{Type: "delete_template", Payload: {snapshot_id, cube_snapshot_id, local_ref}}`
→ `SandboxHub.NotifyJobAvailable`, exactly as `Handler.DeleteSandboxSnapshot` does at
`handler/sandbox.go:1444-1477`. Factor that block into a shared helper rather than
duplicating it.

- [ ] **Step 6: Run and commit**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go build ./... && go test ./internal/service/ ./internal/handler/
cd .. && git add server/pkg/db/queries/env_checkpoint.sql server/pkg/db/generated server/internal/service/env_checkpoint.go server/internal/service/env_checkpoint_test.go server/internal/handler/env_checkpoint.go server/internal/handler/sandbox.go server/internal/handler/router.go
git commit -m "feat(env-checkpoint): release savepoints on delete and refuse while lanes provision"
```

Then in `[areal]`, tick `tasks.md` 6.1–6.3.

______________________________________________________________________

# Phase 7 — Session continuation policy

## Task 17: `[multica]` Warm session for `pause_in_place`, cold sessions for lanes

**tasks.md:** 7.1, 7.2, 7.3

> **Grounded root cause (design D6).** `GetLastTaskSession`
> (`pkg/db/queries/agent.sql:435`) and `GetLastChatTaskSession`
> (`pkg/db/queries/chat.sql:152`) only match *terminal* tasks (`status='acked'`, or
> `suppressed` with `failure_reason='followup_interrupt'`). A `pause_in_place` resume
> re-activates the **same** task row via `ResetInFlightTaskForResume`, which preserves
> `session_id` (it only clears `started_at`/`dispatched_at`/`claimed_at`). So the
> resumed task's own pinned session is sitting right there on its own row and both
> lookups ignore it, because they look at *other* tasks. The fix is to read the resumed
> task row's own `session_id`.

**Files:**

- Modify: `multica/server/pkg/db/queries/agent.sql`
- Modify: `multica/server/internal/handler/agent_inbox.go` (issue path ~lines 2275–2287;
  chat path ~lines 2412–2427)
- Create: `multica/server/internal/handler/agent_inbox_resumed_session_test.go`

**Interfaces:**

- Produces:
  `Queries.GetResumedTaskOwnSession(ctx, taskID pgtype.UUID) (GetResumedTaskOwnSessionRow, error)`
  returning `session_id`, `work_dir`, `runtime_id`.

- [ ] **Step 1: Write the failing tests**

Create `multica/server/internal/handler/agent_inbox_resumed_session_test.go` following
the `t.Skip("database not available")` pattern used throughout
`internal/handler/handler_test.go`:

```go
// TestResumedTaskContinuesItsOwnRecordedSession proves the D6 fix: a task that
// pinned a session mid-flight, was checkpointed with pause_in_place, and was
// re-activated by ResetInFlightTaskForResume resumes that session instead of
// starting cold. GetLastTaskSession cannot supply it -- that query only matches
// terminal tasks, and this is the same non-terminal row.
func TestResumedTaskContinuesItsOwnRecordedSession(t *testing.T) {
	// Arrange: insert an agent_inbox_event with status='draining', a pinned
	// session_id, and force_fresh_session=false; then run
	// ResetInFlightTaskForResume against it.
	// Act: claim the task through the inbox claim path.
	// Assert: resp.PriorSessionID equals the pinned session_id and
	//         resp.PriorWorkDir equals the pinned work_dir.
}

// TestForkedLanesEachStartFreshSessions proves lanes stay cold on purpose: N
// lanes must not resume one mutable session file under N runtimes.
func TestForkedLanesEachStartFreshSessions(t *testing.T) {
	// Arrange: three lane task rows enqueued by the forked strategy, each with
	// its own runtime_id and no session_id, all descended from one savepoint
	// whose source task had session_id 'sess-src'.
	// Assert: every claimed lane task has PriorSessionID == "" and no two lanes
	//         report the same PriorSessionID.
}
```

Fill both bodies using the fixture helpers already present in
`internal/handler/handler_test.go` (it builds real rows against `DATABASE_URL`); do not
invent a new harness.

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspaces/leagent/backend/areal/multica/server
DATABASE_URL="$DATABASE_URL" go test ./internal/handler/ -run 'TestResumedTaskContinues|TestForkedLanesEachStart' -v
```

Expected: FAIL — the resumed task reports an empty `PriorSessionID`.

- [ ] **Step 3: Add the query**

Append to `pkg/db/queries/agent.sql`:

```sql
-- name: GetResumedTaskOwnSession :one
-- Returns the task row's OWN pinned resume pointer. Used by the claim path for
-- a checkpoint-resumed task: ResetInFlightTaskForResume re-activates the same
-- row and preserves session_id, but GetLastTaskSession / GetLastChatTaskSession
-- only match terminal tasks, so neither can see it. Without this a
-- pause_in_place resume starts cold even though the session transcript is
-- still on the preserved disk.
SELECT session_id, work_dir, runtime_id FROM agent_inbox_event
WHERE id = @task_id AND session_id IS NOT NULL;
```

Regenerate.

- [ ] **Step 4: Consult it first on both claim paths**

In `agent_inbox.go`'s issue path, inside the existing `if !event.ForceFreshSession {`
block at line 2275, put the task's own session ahead of the cross-task lookup:

```go
	if !event.ForceFreshSession {
		// The task's own pinned session wins: a checkpoint-resumed task is the
		// SAME row, so its session is the interrupted one to continue.
		if own, err := h.Queries.GetResumedTaskOwnSession(ctx, event.ID); err == nil {
			if own.RuntimeID == runtime.ID && own.SessionID.Valid {
				resp.PriorSessionID = own.SessionID.String
			}
			if own.WorkDir.Valid {
				resp.PriorWorkDir = own.WorkDir.String
			}
		}
		if resp.PriorSessionID == "" {
			if prior, err := h.Queries.GetLastTaskSession(ctx, db.GetLastTaskSessionParams{
				AgentID: event.AgentID,
				IssueID: event.IssueID,
			}); err == nil {
				if prior.RuntimeID == runtime.ID && prior.SessionID.Valid {
					resp.PriorSessionID = prior.SessionID.String
				}
				if prior.WorkDir.Valid && resp.PriorWorkDir == "" {
					resp.PriorWorkDir = prior.WorkDir.String
				}
			}
		}
	}
```

Apply the same "own session first" precedence in `populateAgentInboxChatContext`'s
`if !event.ForceFreshSession {` block (line 2412), before the `chat_session.session_id`
and `GetLastChatTaskSession` fallbacks.

The runtime-match guard is what keeps forked lanes cold: a lane's task row is new, has
no `session_id`, and is bound to the lane's own runtime, so neither lookup can hand it
the source's session. `buildStartRuntimeInCubeCode`'s `pkill` plus the `daemon.id`
rewrite (`cmd_sandboxd.go:960-991`) is the other half of that guarantee — leave it
exactly as is and add a one-line comment above the `pkill` noting it is load-bearing
correctness for lane runtime identity, not hygiene.

- [ ] **Step 5: Run and commit**

```bash
cd /workspaces/leagent/backend/areal/multica/server
go build ./... && go test ./internal/handler/
DATABASE_URL="$DATABASE_URL" go test ./internal/handler/ -run 'TestResumedTaskContinues|TestForkedLanesEachStart' -v
cd .. && git add server/pkg/db/queries/agent.sql server/pkg/db/generated server/internal/handler/agent_inbox.go server/internal/handler/agent_inbox_resumed_session_test.go server/cmd/multica/cmd_sandboxd.go
git commit -m "feat(agent-inbox): continue a resumed task's own session on pause-in-place"
```

Then in `[areal]`, tick `tasks.md` 7.1–7.3.

______________________________________________________________________

# Phase 8 — Verification and documentation

## Task 18: `[multica]` Measure snapshot duration at realistic SWE scale

**tasks.md:** 8.1

**Files:**

- Create: `openspec/changes/env-savepoint-consolidation/.comet/snapshot-latency.md`
  `[areal]`

- [ ] **Step 1: Run the measurement against the project's Cube host**

Reuse the prerequisite experiment's method (Design Doc, "Prerequisite experiment"): a
probe process appending a counter plus a unix timestamp every second distinguishes
"frozen then restored" from "killed".

```bash
# From a host that can reach the Cube API. Substitute the real Cube base URL.
SBX=$(curl -sS -X POST "$CUBE_URL/sandboxes" -H 'content-type: application/json' \
  -d '{"templateID":"<swe-lego-template>","timeout":3600}' | jq -r .sandboxID)
# Clone a large repository and warm a working set inside the sandbox, then:
time curl -sS -X POST "$CUBE_URL/sandboxes/$SBX/snapshots" -H 'content-type: application/json' -d '{}'
curl -sS "$CUBE_URL/sandboxes/$SBX" | jq -r .state    # expect: running
```

- [ ] **Step 2: Record the result against the ~1.2s baseline**

Write `snapshot-latency.md` with: sandbox memory size, repository size, working-set
size, wall-clock snapshot duration, the source's post-snapshot `state`, and whether the
probe process survived. State plainly whether the measurement supports or contradicts
the synchronous-save decision (design D8). If the sandbox cannot be reached from this
environment, say so explicitly and mark 8.1 as requiring a human operator with Cube
access — do not fabricate a number.

- [ ] **Step 3: Commit `[areal]`**

```bash
cd /workspaces/leagent/backend/areal
git add openspec/changes/env-savepoint-consolidation/.comet/snapshot-latency.md openspec/changes/env-savepoint-consolidation/tasks.md
git commit -m "docs(openspec): record SWE-scale snapshot latency measurement"
```

______________________________________________________________________

## Task 19: `[areal]` Reconcile with the unarchived sibling delta specs

**tasks.md:** 8.2

**Files:**

- Modify: `openspec/changes/env-savepoint-consolidation/proposal.md` (the commented-out
  **Modified Capabilities** block, lines 73–77)

- Read: `openspec/changes/env-checkpoint-resume/specs/**`,
  `openspec/changes/env-checkpoint-resume-trigger/specs/**`,
  `openspec/changes/env-dispatch-sandbox-lifecycle/specs/**`

- [ ] **Step 1: List the requirements this change supersedes**

```bash
cd /workspaces/leagent/backend/areal
rg -n "^### Requirement:" openspec/changes/env-checkpoint-resume/specs openspec/changes/env-checkpoint-resume-trigger/specs openspec/changes/env-dispatch-sandbox-lifecycle/specs
```

- [ ] **Step 2: Write the reconciliation table into `proposal.md`**

Replace the commented-out **Modified Capabilities** block with a real table: one row per
sibling requirement, and for each, whether this change **supersedes** it (with the
`env-savepoint-fanout` / `agent-continuation-seam` requirement that replaces it),
**preserves** it unchanged, or **leaves it untouched**. Cover at minimum: 1:1 resume
over `sandbox_refs` (superseded by "Resume materializes a requested number of lanes"),
the resume-trigger descriptor and `TriggerStatus` (preserved, now behind the seam), and
`CloneSandboxInstance`'s live-fork deferral (fulfilled by this change).

- [ ] **Step 3: Commit**

```bash
cd /workspaces/leagent/backend/areal
git add openspec/changes/env-savepoint-consolidation/proposal.md openspec/changes/env-savepoint-consolidation/tasks.md
git commit -m "docs(openspec): reconcile savepoint consolidation with sibling delta specs"
```

______________________________________________________________________

## Task 20: `[areal]` Update the multica environment protocol document

**tasks.md:** 8.3, 8.4

**Files:**

- Modify: `customized_areal/tree_search/agents/multica_environment_protocol.md`

- [ ] **Step 1: Correct the "no snapshot semantics" statement**

The **Env Checkpoint Semantics** section (lines 832–838) currently ends:

> Env checkpoint creation is pause-in-place. Multica synchronously waits for sandboxd
> stop/Cube pause up to the configured timeout and stores the project subtree inline as
> JSONB. A completed checkpoint can be resumed through resume-from-checkpoint, which
> resumes the same sandbox instances. The API does not provide immutable fork, branch,
> snapshot, or copy-on-write semantics.

Replace it with:

```markdown
Env checkpoint creation has two save modes.

`pause_in_place` (the default, and what every pre-existing checkpoint resolves
to) suspends the source sandbox instances and records no savepoint. Multica
synchronously waits for sandboxd stop / Cube pause up to the configured timeout
and stores the project subtree inline as JSONB. Resume returns the same sandbox
instances and re-activates the same task row, continuing the interrupted CLI
session rather than starting cold. `pause_in_place` rejects a lane count greater
than one.

`snapshot` records an immutable savepoint per source instance (a
`sandbox_snapshot` row backed by a Cube snapshot template) and leaves every
source instance **running**, with its in-flight task undisturbed. Resume accepts
a lane count and materializes that many sandbox instances from the checkpoint's
savepoint — **one snapshot per source instance, not one per lane** — each with
its own copied project subtree, its own agent runtime, and a fresh CLI session.
Every resume carries a lane key: repeating a key returns the existing lane,
while a new key expands the same frontier again without creating a second
checkpoint.

A savepoint is owned by exactly one checkpoint and is released when that
checkpoint is deleted, so it outlives its first use. Cube snapshots are
memory-level checkpoint/restore, not filesystem copies: a snapshot completes in
~1.2s, leaves the source running, and a sandbox created from the resulting
template comes up with the source's processes still live.
```

- [ ] **Step 2: Correct the branch-mode description**

Update these three places so branch is described as a fan-out of checkpoint resume:

- Line 43–46 ("Fresh rollout state is created by…"): `mode="branch"` is served
  internally by creating or reusing a `snapshot` checkpoint at the source env and
  resuming it into lanes; `mode="resume"` still means resume the named checkpoint, and
  for a `pause_in_place` checkpoint that remains an in-place resume of the same
  instances.
- Line 516 and 521 (the `mode` and `group_size` request-parameter rows): note that
  `group_size` for a branch dispatch is the requested lane count and that
  `idempotency_key` is **required** for branch mode because lane keys are derived from
  it (Task 9).
- Line 731–732 ("Server-side owns sandbox snapshot/fork…"): keep the statement but add
  that the fork is now an immutable savepoint shared by N lanes rather than a
  per-dispatch disposable snapshot.

Leave the `ForkableEnvironment` seam description (lines 696–725) as-is: it is still not
on the live branch path.

- [ ] **Step 3: Record the intra-turn fork finding as out of scope**

Add a subsection at the end of **Env Checkpoint Semantics**:

```markdown
### Out of scope: forking inside a turn

A sandbox restored from a Cube snapshot carries live processes, including a
mid-turn agent — the prerequisite experiment observed the same PID and the same
append-in-progress log file continuing on the clone. Intra-turn forking is
therefore technically possible but deliberately out of scope, for two reasons:

- **Runtime identity.** N clones would share one `daemon.id` frozen into
  `~/.multica/daemon.id`, so each lane could re-register as the source
  sandbox's runtime and steal its row.
- **Duplicated in-flight requests.** Each clone would resume the source's
  in-flight model request, producing N duplicated calls for one logical turn.

Both are exactly what the `pkill` of the snapshot-restored daemon in
`buildStartRuntimeInCubeCode` avoids, which makes that `pkill` load-bearing
correctness rather than hygiene. Revisit only if intra-turn branch points prove
valuable for tree search.
```

- [ ] **Step 4: Check formatting and commit**

```bash
cd /workspaces/leagent/backend/areal
source .venv/bin/activate 2>/dev/null; pre-commit run --files customized_areal/tree_search/agents/multica_environment_protocol.md
git add customized_areal/tree_search/agents/multica_environment_protocol.md openspec/changes/env-savepoint-consolidation/tasks.md
git commit -m "docs(protocol): describe branch as a fan-out of checkpoint resume"
```

______________________________________________________________________

## Final verification

- [ ] **Step 1: `[multica]` Everything this environment can actually verify**

```bash
cd /workspaces/leagent/backend/areal/multica/.worktrees/env-savepoint-consolidation/server
go build ./...
go vet ./...
go test ./internal/service/... ./internal/handler/...
```

Expected: all green. This is the complete acceptance evidence available here.

- [ ] **Step 1b: `[multica]` Record the database verification that is still owed**

`make test` needs Postgres, which is unavailable in this environment and was explicitly
not installed for this change. The `*_Integration` tests are written and self-skip, so
they must **not** be counted as passing. On a host with a real database, the outstanding
run is:

```bash
cd /workspaces/leagent/backend/areal/multica/.worktrees/env-savepoint-consolidation
make test
cd server && DATABASE_URL="$DATABASE_URL" go test ./internal/service/ ./internal/handler/ -run '_Integration' -v
```

Until that runs, migrations 244/245/246, the `UNIQUE (checkpoint_id, lane_key)` claim
race, the `ON DELETE CASCADE` reclamation, and every hand-written SQL string remain
unverified. Carry this forward as the primary open risk in the verification report.

- [ ] **Step 2: `[multica]` Confirm nothing unrelated crept in**

```bash
cd /workspaces/leagent/backend/areal/multica/.worktrees/env-savepoint-consolidation
git log --oneline upstream/dev..HEAD
git diff --stat upstream/dev..HEAD
cd /workspaces/leagent/backend/areal/multica
git status --porcelain   # the main checkout's unrelated work must be untouched
```

- [ ] **Step 3: `[areal]` Confirm the areal-side diff is docs and specs only**

```bash
cd /workspaces/leagent/backend/areal
git diff --stat 2529da6cd28c34d2adbf446a7179d736db53aa92..HEAD -- openspec docs customized_areal
```

Expected: only `openspec/changes/env-savepoint-consolidation/**`, `docs/superpowers/**`,
and `customized_areal/tree_search/agents/multica_environment_protocol.md` (plus
`multica_client.py` only if Task 9 Step 5 required the idempotency-key change, which
must have been approved).

- [ ] **Step 4: `[areal]` Confirm every `tasks.md` checkbox is ticked**

```bash
cd /workspaces/leagent/backend/areal
grep -c '^\- \[ \]' openspec/changes/env-savepoint-consolidation/tasks.md
```

Expected: `0`.

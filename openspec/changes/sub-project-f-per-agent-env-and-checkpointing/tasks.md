# Tasks — sub-project-f-per-agent-env-and-checkpointing

## Task 1: Investigation — seams confirmation

**Files**: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/service/task.go` (delegation/mention/completion/failure seams), daemon/pi
tool-call boundary, Fleet sandbox snapshot API, AReaL proxy logprob capture point.

- [ ] Confirm per-agent env dispatch seam: `EnvDispatchRequest` /
  `service.EnvDispatchInput` struct shape, validation site, dispatch loop.
- [ ] Confirm always-event hook seams: exact call sites for
  `EnqueueTaskForIssue`/`EnqueueTaskForMention`/`EnqueueTaskForSquadLeader` (delegation
  \+ mention), `CompleteTask` (LLM path vs sweeper/autopilot discriminator), `FailTask`
  (same discriminator), squad leader briefing generation site.
- [ ] Confirm tool-call boundary in pi/daemon: where the daemon receives tool-call
  completion + the AReaL proxy logprob capture point (E's seam).
- [ ] Confirm Fleet sandbox snapshot API: create snapshot, fork/restore snapshot (used
  by C's lazy-snapshot path).
- [ ] Confirm DB subtree serialization approach: issue + sub-issues + tasks + messages +
  comments query shape.
- [ ] Confirm AReaL client seam: `swe_lego_client.py` `create_env_dispatch` signature +
  where E will attach logprobs.
- [ ] Document findings in design doc updates (T1 section or inline).
- [ ] Commit: `docs(F): T1 investigation — seam confirmation`.

## Task 2: Per-agent env contract (multica handler + service) (TDD)

**Files**: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/service/env_dispatch_test.go`.

- [ ] Failing tests:
  - `TestEnvDispatch_PerAgentEnvs_Valid` — per_agent_envs map with valid agent_id →
    base_env_id → each agent gets its own sandbox.
  - `TestEnvDispatch_PerAgentEnvs_UnknownAgent` — unknown agent_id → 400.
  - `TestEnvDispatch_PerAgentEnvs_UnknownBaseEnv` — unknown base_env_id → 400.
  - `TestEnvDispatch_PerAgentEnvs_MixedWithTopLevel` — per_agent_envs + top-level env_id
    for same agent → 400.
  - `TestEnvDispatch_PerAgentEnvs_Empty` — empty per_agent_envs → today's behavior
    (backward compat).
  - `TestEnvDispatch_PerAgentEnvs_PartialSquad` — some agents have custom envs, others
    use default → validated.
- [ ] Add `per_agent_envs map[string]string` to `EnvDispatchRequest` (handler) and
  `EnvDispatchInput` (service).
- [ ] Handler validation: resolve each agent_id → real agent in workspace; resolve each
  base_env_id → real env; reject mixing with top-level env_id for the same agent.
- [ ] Service: when per_agent_envs is set, dispatch each agent to its own sandbox from
  its specified base env. Squad shares one multica entity subtree.
- [ ] Run: `go test ./internal/service/ -run 'EnvDispatch_PerAgent'`.
- [ ] Commit: `feat(env-dispatch): per-agent env customization contract`.

## Task 3: Per-agent env persistence + AReaL client (TDD)

**Files**: multica migration NNN, `internal/service/training_dispatch.go`,
`internal/service/training_dispatch_test.go`,
`customized_areal/tree_search/agents/swe_lego_client.py`.

- [ ] Failing tests (multica):
  - `TestCreateTrainingDispatch_WithPerAgentEnvs` — per_agent_envs persisted and
    round-tripped.
  - `TestGetTrainingDispatchByProject_WithPerAgentEnvs` — returned with per_agent_envs.
  - `TestCreateTrainingDispatch_NoPerAgentEnvs` — field is null when not set (backward
    compat).
- [ ] Migration: `ALTER TABLE training_dispatch ADD COLUMN per_agent_envs   JSONB NULL`.
  Hand-write generated Go (mirror sibling — do NOT run `sqlc generate` repo-wide).
- [ ] Extend `CreateTrainingDispatch` / `GetTrainingDispatchByProject` queries to handle
  `per_agent_envs`.
- [ ] Failing tests (AReaL):
  - `test_create_env_dispatch_with_per_agent_envs` — per_agent_envs dict passed in
    request body.
  - `test_create_env_dispatch_without_per_agent_envs` — field omitted when empty.
- [ ] Extend `create_env_dispatch` to accept optional `per_agent_envs` dict.
- [ ] Run: `go test ./internal/service/ -run 'TrainingDispatch.*PerAgent'` +
  `uv run pytest customized_areal/tree_search/tests/ -k 'per_agent_env'`.
- [ ] Commit: `feat(env-dispatch): persist per-agent envs + AReaL client`.

## Task 4: Checkpoint storage (multica migration + service) (TDD)

**Files**: migration NNN+1, new `internal/service/env_checkpoint.go`,
`internal/service/env_checkpoint_test.go`, hand-written generated Go.

- [ ] Failing tests:
  - `TestCreateCheckpoint_Always` — creates checkpoint with kind=always, all fields
    populated (env_id_map, db_snapshot_ref, sandbox_snapshot_refs, event_ref,
    timestamp).
  - `TestCreateCheckpoint_EntropyGated` — creates checkpoint with kind=entropy_gated,
    entropy_score populated.
  - `TestGetCheckpoint` — round-trips a created checkpoint.
  - `TestListCheckpointsForProject` — returns checkpoints for a project ordered by
    created_at DESC.
  - `TestCreateCheckpoint_SnapshotFailure` — sandbox snapshot API fails → error logged,
    checkpoint NOT created (best-effort, not blocking).
- [ ] Migration: new `env_checkpoint` table. Columns: id (UUID PK), workspace_id,
  project_id, issue_id, event_ref, env_id_map (JSONB), db_snapshot_ref (JSONB),
  sandbox_snapshot_refs (JSONB), entropy_score (nullable), checkpoint_kind
  (always|entropy_gated), created_at (TIMESTAMPTZ). Indexes on project_id, workspace_id.
  Hand-write generated Go.
- [ ] New `internal/service/env_checkpoint.go`:
  - `CreateCheckpoint(ctx, params)` → calls Fleet sandbox snapshot API eagerly per
    env_id; serializes DB subtree (issue + sub-issues + tasks + messages + comments) as
    JSONB; inserts row.
  - `GetCheckpoint(ctx, id)` → checkpoint row.
  - `ListCheckpointsForProject(ctx, projectID)` → \[\]Checkpoint.
- [ ] Run: `go test ./internal/service/ -run 'Checkpoint|EnvCheckpoint'`.
- [ ] Commit: `feat(env-checkpoint): checkpoint storage table + service`.

## Task 5: Always-event checkpoint triggers (multica service) (TDD)

**Files**: `internal/service/task.go`, `internal/service/env_checkpoint.go`,
`internal/service/task_test.go` (or new
`internal/service/env_checkpoint_trigger_test.go`).

- [ ] Failing tests:
  - `TestMaybeCheckpoint_Delegation` — delegation enqueue → checkpoint created with
    kind=always, correct event_ref.
  - `TestMaybeCheckpoint_Mention` — mention enqueue → checkpoint.
  - `TestMaybeCheckpoint_CompletionLLMPath` — CompleteTask via daemon LLM path →
    checkpoint.
  - `TestMaybeCheckpoint_FailureLLMPath` — FailTask via daemon LLM path → checkpoint.
  - `TestMaybeCheckpoint_SquadBriefing` — squad leader briefing generation → checkpoint.
  - `TestMaybeCheckpoint_NonTrainedRollout` — no training_dispatch row → no checkpoint.
  - `TestMaybeCheckpoint_SweeperFail` — sweeper FailTask → no checkpoint.
  - `TestMaybeCheckpoint_AutopilotFail` — autopilot FailTask → no checkpoint.
  - `TestMaybeCheckpoint_CreationFailure` — checkpoint create fails → error logged,
    rollout continues.
- [ ] Implement `maybeCheckpoint(ctx, task, eventKind)`:
  - Check: is this a trained rollout (training_dispatch exists for project)? Skip if
    not.
  - If yes, call `CreateCheckpoint` with correct kind=always.
  - On error: log (slog.Warn), continue — best-effort like D's RL error handling.
- [ ] Wire hook calls at: delegation enqueue sites
  (`EnqueueTaskForIssue`/`EnqueueTaskForMention`/`EnqueueTaskForSquadLeader`), mention
  enqueue, CompleteTask (daemon LLM path only), FailTask (daemon LLM path only), squad
  leader briefing generation site.
- [ ] Run: `go test ./internal/service/ -run 'MaybeCheckpoint|Checkpoint_Always'`.
- [ ] Commit: `feat(env-checkpoint): always-event checkpoint triggers`.

## Task 6: Entropy-gated tool-call checkpoints (AReaL + multica) (TDD)

**Files**: new `customized_areal/tree_search/agents/env_checkpoint.py` (or extension),
`internal/handler/env_checkpoint.go`, `internal/service/env_checkpoint.go`.

- [ ] Failing tests (AReaL):
  - `test_compute_tool_call_entropy` — sample logprobs → expected float (known-variance
    distribution → reasonable entropy value).
  - `test_compute_tool_call_entropy_deterministic` — logprobs with 1.0 probability →
    entropy ≈ 0.
  - `test_compute_tool_call_entropy_uniform` — uniform logprobs → max entropy.
  - `test_maybe_create_checkpoint_above_threshold` — entropy > threshold → POST to
    multica `/api/v1/env-checkpoint`.
  - `test_maybe_create_checkpoint_below_threshold` — entropy ≤ threshold → no call.
  - `test_maybe_create_checkpoint_logprobs_unavailable` — E graceful degradation → skip
    (no error).
- [ ] Implement `compute_tool_call_entropy(logprobs) → float`: aggregate logprob entropy
  over tool-choice decision span.
- [ ] Implement `maybe_create_checkpoint(...)` — calls multica `/api/v1/env-checkpoint`
  with kind=entropy_gated + entropy_score.
- [ ] Failing tests (multica):
  - `TestCreateCheckpoint_EntropyGatedEndpoint` — POST `/api/v1/env-checkpoint` with
    kind=entropy_gated → checkpoint created.
  - `TestCreateCheckpoint_EntropyGated_Disabled` — ENV_CHECKPOINT_ENABLED=false → 503 or
    skip.
- [ ] New `POST /api/v1/env-checkpoint` handler (multica): validates workspace,
  delegates to `CreateCheckpoint`.
- [ ] Wire the entropy check at the tool-call boundary in AReaL proxy (T1 confirms exact
  hook point — Option A from design §4.4).
- [ ] Run: `uv run pytest customized_areal/tree_search/tests/ -k 'entropy|checkpoint'` +
  `go test ./internal/service/ -run 'EntropyGated'`.
- [ ] Commit (AReaL): `feat(env-checkpoint): entropy-gated tool-call checkpoints`.
  Commit (multica): `feat(env-checkpoint): entropy-gated checkpoint endpoint`.

## Task 7: Branch-from-checkpoint (multica service + handler) (TDD)

**Files**: `internal/handler/env_checkpoint.go` (or `env_dispatch.go`),
`internal/service/env_checkpoint.go`, `internal/service/env_checkpoint_test.go`.

- [ ] Failing tests:
  - `TestBranchFromCheckpoint_Success` — valid checkpoint → new sandbox per agent
    (forked from snapshot) + DB subtree restored under new project_id + returns rollout
    handle.
  - `TestBranchFromCheckpoint_PerAgentEnvs` — checkpoint with per_agent_envs → branched
    rollout preserves per-agent env assignments.
  - `TestBranchFromCheckpoint_CorruptSnapshot` — sandbox fork fails → error returned.
  - `TestBranchFromCheckpoint_CheckpointNotFound` → 404.
  - `TestBranchFromCheckpoint_CrossWorkspace` → 403.
  - `TestBranchFromCheckpoint_CreatesNewRLSession` — branched rollout triggers D's open
    hook (session-open on trained project creation).
- [ ] Implement `BranchFromCheckpoint(ctx, checkpointID)`:
  1. Read checkpoint row (validate workspace ownership).
  1. Fork each sandbox snapshot → new sandbox per agent (Fleet fork endpoint).
  1. Restore DB subtree under new project_id (copy issue + sub-issues + tasks
     - messages + comments from snapshot).
  1. Create new env_dispatch with mode=branch-from-checkpoint + restored envs + DB
     state.
  1. Return new rollout handle (same shape as mode=branch).
- [ ] New `POST /api/v1/env-dispatch` handler: accept
  `mode=branch-from-checkpoint, checkpoint_id=<id>`.
- [ ] Run: `go test ./internal/service/ -run 'BranchFromCheckpoint'`.
- [ ] Commit: `feat(env-checkpoint): branch-from-checkpoint operation`.

## Task 8: AReaL client branch-from-checkpoint (TDD)

**Files**: `customized_areal/tree_search/agents/swe_lego_client.py`,
`customized_areal/tree_search/tests/test_swe_lego_client.py` (or new
`test_env_checkpoint.py`).

- [ ] Failing tests:
  - `test_branch_from_checkpoint_success` — returns rollout handle.
  - `test_branch_from_checkpoint_not_found` — 404 → raises.
  - `test_branch_from_checkpoint_per_agent_envs` — per_agent_envs passed through.
- [ ] Implement `branch_from_checkpoint(checkpoint_id)` method on client.
- [ ] Run:
  `uv run pytest customized_areal/tree_search/tests/ -k 'branch_from_checkpoint'`.
- [ ] Commit: `feat(areal-client): branch-from-checkpoint API`.

## Task 9: Config + production wiring (multica)

**Files**: `internal/daemon/config.go` (or server config), `.env.example`,
handler/service construction.

- [ ] Add config vars:
  - `ENV_CHECKPOINT_ENABLED` (default true) — master toggle.
  - `ENV_CHECKPOINT_ENTROPY_THRESHOLD` (default TBD — calibrate from sample rollouts;
    document).
  - `ENV_CHECKPOINT_ALWAYS_EVENTS` (default: delegation,mention,completion,
    failure,squad_briefing) — configurable list.
- [ ] Wire: if `ENV_CHECKPOINT_ENABLED=false`, `maybeCheckpoint` always skips (both
  always and entropy-gated); `/api/v1/env-checkpoint` returns 503.
- [ ] Guard: if checkpointing is enabled but Fleet sandbox snapshot API is unreachable
  at startup → log warning (not fatal — best-effort at runtime).
- [ ] `.env.example` entries for all 3 vars.
- [ ] Build touched packages; commit:
  `chore(env-checkpoint): config + production wiring`.

## Task 10: Full regression + db_bridge smoke + E2E

- [ ] Scoped Go:
  `go build ./internal/handler/ ./internal/service/ ./internal/daemon/execenv/` +
  `go vet` same + `go test` same (confirm pre-existing 16 ON CONFLICT handler failures +
  webpush build failure are unchanged; 0 new).
- [ ] `gofmt -l` clean on touched files.
- [ ] db_bridge smoke: verify `/api/v1/env-dispatch` carries `per_agent_envs` in JSON
  body; new `/api/v1/env-checkpoint` channel (create/list/branch-from) routes correctly.
- [ ] AReaL confirm-only: `create_env_dispatch` passes per_agent_envs;
  `branch_from_checkpoint` returns rollout handle; entropy helper computes correct
  values for sample logprobs.
- [ ] Cross-repo E2E (if feasible): trained rollout with per-agent envs → checkpoints at
  structural events → AReaL branches from checkpoint → branched rollout inherits sandbox
  \+ DB state.
- [ ] grep: `per_agent_envs`, `env_checkpoint`, `maybeCheckpoint`,
  `branch_from_checkpoint`, `checkpoint_kind` resolve to intended code only.
- [ ] Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.
- [ ] Commit: `docs(F): T10 full regression + E2E + grep sweep`.

## Test runners / constraints

- multica Go:
  `DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable`.
  **Scope build/test to touched packages** — `go build ./...` fails on the pre-existing
  `internal/service/webpush/webpush.go:180` "constant 4096 overflows byte" (go 1.26).
  Pre-existing: 16 `internal/handler` ON CONFLICT (42P10) daemon/claim failures;
  `cmd/server` does not compile here (imports webpush) — validate router/handler edits
  by `go build ./internal/handler/` + inspection.
- **Codegen rule:** do NOT run `sqlc generate` repo-wide (creates colliding
  `agent_skill_suggestion.sql.go` / `evolution.sql.go`, breaks build). For any new
  query: add to `queries/*.sql` AND hand-write the generated Go in
  `generated/<file>.sql.go` mirroring a sibling.
- **Migration rule:** use the `create-migration` skill for DDL changes. Include
  deployment/apply notes.
- AReaL Python: `uv run pytest customized_areal/tree_search/tests/ -k '<filter>'`.
  `pre-commit run --files` before commit.
- db_bridge (if touched): `cd multica/db_bridge && uv run pytest -q`.
- Commit each task to multica `main` (multica-side) or areal `master` (areal-side);
  record commit hash in task ledger below.

## Dependencies

- **Sub-project E** (critic-driven training signal): F's T6 (entropy-gated checkpoints)
  requires E's logprobs. T1-T5, T7-T10 are independent of E and can ship first. E is at
  build phase (0/0 tasks).
- **Sub-project D** (session lifecycle): F's T7 (branch-from-checkpoint) triggers D's
  session-open hook. D is at verify-fail (WARNING 1 pending). F layers on D; no contract
  change to D.
- **Sub-project C** (branch via env-dispatch): F's T7 extends C's `mode=branch`. C is
  complete on multica main.

## Task ledger

(append "Task N: complete (commits <base7>..<head7>, review CLEAN)" as tasks finish)

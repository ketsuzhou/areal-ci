# Comet Design Handoff

- Change: multica-v2-segment-dag-recording-assembly
- Phase: design
- Mode: compact
- Context hash: 33ed0b5706d26b36ae279f9d0584d986da3644fdbf2c9ea19187665122526e59

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/multica-v2-segment-dag-recording-assembly/proposal.md

- Source: openspec/changes/multica-v2-segment-dag-recording-assembly/proposal.md
- Lines: 1-69
- SHA256: 505a84061a9f162f680a110e11765b5befe257342f540c632f59b8920fb1f4dc

```md
## Why

Change 1 (`multica-v2-segment-dag-training`) landed the areal-side data path on master
(`close_segment`, per-segment export, `MulticaDagClient`, `SuperNodeAssembler.assemble_from_refs`,
minimal training + tensor lifecycle; U1–U5) plus the multica building blocks on
`feature/multica-v2-segment-dag-training` (U6 arealrl `CloseSegment`/`ExportTrajectory` client,
U9 migration 155–157, U7.1 `InteractionDAGService` recorder + sqlc). But the multica side is
**unwired**: no code calls the recorder at communication events, retries inherit the parent's
areal RL session (so retry-child segments dangle at assembly), and there is no `AssembledDag`
assembly or `/dag` polling endpoint. Until these land, AReaL cannot poll a complete
`AssembledDag` and the segment-DAG training path is open-ended. This change finishes the
multica side (U7.2 -> U7.3 -> U8 -> U10) so a trained rollout round-trips end to end.

## What Changes

- **U7.2 - Wire the recorder (Multica)**: call `RecordSessionAgentRun` inside
  `maybeOpenTrainingSession` right after `StartSession` succeeds (D10 chokepoint), and call
  `CloseSegmentForEvent` + `AddEdge` at the delegation / mention / completion / squad-briefing
  seams in `task.go` - trained rollouts only, `INTERACTION_DAG_ENABLED` composing with the
  existing `s.Training` gate. Integration tests cover each event type + fan-out.
- **U7.3 - Fresh areal RL session per retry attempt (D9, Multica)**: `CreateRetryTask` strips
  `areal_proxy` from the child's context (keeping the chat `session_id`/`work_dir` resume
  CASE-WHEN); `MaybeRetryFailedTask` calls `tryOpenTrainingSession(child)` before
  `NotifyTaskEnqueued`; the child's `agent_run_id` (= `task.ID`) is recorded. The parent
  session is closed (`EndSession`) before the child opens its own. Behavior change (internal):
  retry children no longer inherit the parent's areal RL session.
- **U8 - AssembledDag assembly + /dag endpoint (Multica)**: `AssembleAssembledDag(project_id)`
  reads recorded segments/edges/env-snapshots/session-runs and returns `{segments, edges,
  session_to_agent_run}`; `GET /api/v1/env-dispatch/{projectID}/dag` serves `202` in-progress,
  `200` + `AssembledDag` done, `404` unknown project, `403` cross-workspace; an
  incomplete/failed rollout yields a `failed` status, not a partial `AssembledDag`.
- **U10 - Config + E2E + grep (both repos)**: `INTERACTION_DAG_ENABLED` default + areal polling
  config (interval/timeout/backoff); cross-repo E2E (3-agent `mode=scratch` rollout ->
  segments -> `AssembledDag` -> AReaL resolve -> `ExecutionDAG` -> minimal training ->
  cleanup); grep sweep (`close_segment`/`AssembledDag`/`tensor_ref` resolve to intended code;
  no `start_turn_idx`/`end_turn_idx`).

## Capabilities

### New Capabilities

- `v2-segment-dag-recording`: Multica-side segment recording at communication events
  (trained-rollout gated), retry-attempt areal RL session lifecycle, `session_to_agent_run`
  recording at session open, and `AssembledDag` assembly + `/dag` polling-endpoint serving
  (202/200/404/403 + failed-status). Complements the parent change's `v2-segment-dag`
  capability, which owns the `AssembledDag` contract shape, the areal ref-resolve consumer,
  and the basic 202/200/404 polling. The `/dag` 403 + failed-status requirements here refine
  the parent's polling requirement - reconcile on apply if both changes ship together.

### Modified Capabilities

<!-- None shipped. `v2-segment-dag` is proposed by the in-flight parent change (not yet in
     openspec/specs/), so this change adds sibling requirements under a new capability rather
     than modifying it. -->

## Impact

- **Multica server (Go, primary)**: `internal/service/interaction_dag.go` (assembly +
  U7.2/U7.3 wiring), `internal/service/training.go` (`RecordSessionAgentRun` at open, retry
  session handling), `internal/service/task.go` (event seams, `CreateRetryTask`,
  `MaybeRetryFailedTask`), `internal/handler/env_dispatch.go` (`/dag` handler). New
  `AssembleAssembledDag` query + hand-written sqlc. Builds on U7.1's `InteractionDAGService`
  (unmerged, on `feature/multica-v2-segment-dag-training`).
- **AReaL (Python, secondary)**: polling config for `MulticaDagClient`; U10 E2E + grep. No
  new areal runtime code - U1-U5 + tree-search branching already consume `AssembledDag`.
- **Dual-repo**: multica Go continues on `feature/multica-v2-segment-dag-training`; areal
  bits branch off `f60c86bb` (master tip). OpenSpec change is tracked in the areal repo.
- **Out of scope**: judge / process-reward + V + GAE (change 2); tree search / branching
  (change 3, already landed); full sandbox snapshot/fork (Sub-project F).
```

## openspec/changes/multica-v2-segment-dag-recording-assembly/design.md

- Source: openspec/changes/multica-v2-segment-dag-recording-assembly/design.md
- Lines: 1-153
- SHA256: 352bcc473a7fec252ab47f435991e6f01d5fcf9bfa76a0b0fc9d900497b1ae2a

[TRUNCATED]

```md
## Context

This change is the multica-side completion of the `v2-segment-dag` data path (change 1). The
areal consumer (U1-U5, on master) already polls `GET .../dag`, resolves `tensor_ref`s, builds
`SuperNode`/`ExecutionDAG`, runs a placeholder-reward training step, and cleans up. The multica
building blocks exist on `feature/multica-v2-segment-dag-training`: the arealrl
`CloseSegment`/`ExportTrajectory` client (U6), migration 155-157 (U9), and the
`InteractionDAGService` recorder + sqlc (U7.1). What is missing is the wiring that makes a real
rollout produce a complete `AssembledDag`: event hooks, retry-session correctness, assembly,
and the polling endpoint.

Locked decisions D8-D10 (from the parent's pre-build seam trace) govern identity and session
semantics; this change implements them and adds wiring/assembly decisions D11-D15.

Canonical capability spec: `openspec/changes/multica-v2-segment-dag-recording-assembly/specs/v2-segment-dag-recording/spec.md`.
This document deepens the HOW; it does not restate requirements.

## Goals / Non-Goals

**Goals**:

- Wire `InteractionDAGService` into the trained-rollout path so each communication event
  records a segment + edge.
- Correct retry-session lifecycle (D9): each attempt opens its own areal RL session.
- `AssembleAssembledDag` + `GET .../dag` serving 202/200/404/403 + failed-status.
- End-to-end round-trip: rollout -> segments -> `AssembledDag` -> AReaL resolve -> train ->
  cleanup.

**Non-Goals**:

- Judge / process reward + V + GAE (change 2).
- Tree search / branching (change 3, landed).
- Full sandbox snapshot/fork in env snapshots (Sub-project F).
- Re-implementing areal U1-U5 (on master) or multica U6/U9/U7.1 (on the feature branch).

## Decisions

### D8 - `agent_run_id` = `task.ID` (attempt-level); no separate runs table [carried]

`EnqueueAgentRun` returns `task.ID` as the runID; `EnvRollout.AgentRunID = task.ID`. AReal
consumes `agent_run_id` as `SuperNode.agent_id` and as the key to reverse-lookup `session_id`
from `session_to_agent_run`. The segment-table `task_id` column is redundant traceability (v2
`SegmentSpec` dropped it; `SuperNode.task_id=""`). Retries create a NEW `task.ID` (child via
`parent_task_id`), so `agent_run_id` is per-attempt, not per-logical-task.

### D9 - Fresh areal RL session per retry attempt (NOT inherited) [carried]

Today `CreateRetryTask` copies `p.context` (which holds `areal_proxy` = areal
`SessionID`+`ProxyKey`), so the child inherits the parent's areal session and
`maybeOpenTrainingSession`'s `hasArealProxyContext` guard no-ops - no fresh `StartSession`, no
`RecordSessionAgentRun`. A retry child's segments then carry the child's `task.ID` as
`agent_run_id`, which is absent from `session_to_agent_run` -> dangling at assembly. Decision:
each attempt opens its OWN areal session via three changes: (1) `CreateRetryTask` strips
`areal_proxy` from the child's context [keep the chat `session_id`/`work_dir` resume CASE-WHEN -
that is the multica chat session, separate]; (2) `MaybeRetryFailedTask` calls
`tryOpenTrainingSession(child, projectID, envID)` BEFORE `NotifyTaskEnqueued` (mirror
`enqueueMentionTask`: open at task.go:614, notify at :618); (3) `RecordSessionAgentRun` (D10)
then fires for the child. Close ordering supports it: `FailTask` -> `RouteTerminalTrainingTask`
closes S_A (`EndSession`) -> `MaybeRetryFailedTask` creates B -> B opens fresh S_B. Only the 4
retryable reasons produce a child (`runtime_offline`/`runtime_recovery`/`timeout`/
`codex_semantic_inactivity`); non-retryable failures are terminal. Scope: change (1) touches
pre-existing `CreateRetryTask` (migration 055) - in-scope dependency. The sweeper path
(`runtime_sweeper.FailStaleTasks`) bypasses `FailTask` (orphaned session, no child) - pre-existing
gap, coverage boundary.

### D10 - `RecordSessionAgentRun` call site [carried]

Inside `maybeOpenTrainingSession`, immediately after `StartSession` succeeds (after the persist,
before the slog). Records `{projectID, sessionID=creds.SessionID, agentRunID=taskID}`. Single
idempotent chokepoint: both the `task.go` `Enqueue*` paths (via `tryOpenTrainingSession`) and
the env_dispatch path (via the adapter -> `MaybeOpenTrainingSession`) funnel through it; placed
after the `hasArealProxyContext` guard it fires only on real first-open. It builds the
`session_to_agent_run: {session_id -> agent_run_id(=task.ID)}` map AReal's assembler consumes.
Conflation trap: `agentRunID` = `taskID` (the run), NOT `agentID` (the agent); AReal stores
`agent_run_id` in a field named `agent_id`, which makes this confusion easy.

### D11 - Recording gated to trained rollouts behind `INTERACTION_DAG_ENABLED` [new]

The recorder hooks fire only when `s.Training != nil` (trained rollout) AND
`INTERACTION_DAG_ENABLED` is set (default on for trained rollouts). Non-trained rollouts incur
```

Full source: openspec/changes/multica-v2-segment-dag-recording-assembly/design.md

## openspec/changes/multica-v2-segment-dag-recording-assembly/tasks.md

- Source: openspec/changes/multica-v2-segment-dag-recording-assembly/tasks.md
- Lines: 1-91
- SHA256: 49f295543378b45cc91a014649d21e7302e05b4c9a85081729e69b5566ae37ec

[TRUNCATED]

```md
## 1. U7.2 - Wire recorder into the trained-rollout path (TDD)

**Files:** `server/internal/service/training.go`, `server/internal/service/task.go`,
`server/internal/service/interaction_dag.go`, integration tests under
`server/internal/service/`.

- [ ] 1.1 Failing tests: `RecordSessionAgentRun` fires inside `maybeOpenTrainingSession`
  immediately after `StartSession` succeeds (D10), recording `{projectID, sessionID,
  agentRunID=taskID}`; idempotent across repeated open attempts.
- [ ] 1.2 Failing tests: at a delegation event the parent run calls `close_segment` + export,
  records a segment with `trajectory_id` + `tensor_ref` + `closing_event`, opens a child
  segment, and records a `delegation` edge.
- [ ] 1.3 Failing tests: a mention records a `mention` edge without closing a segment.
- [ ] 1.4 Failing tests: a completion calls `close_segment` + export on the child run and
  records a `completion` edge.
- [ ] 1.5 Failing tests: a squad briefing closes a segment via `close_segment` + export.
- [ ] 1.6 Failing tests: a leaf run (no communication event) yields exactly one leaf segment
  with `closing_event = None`.
- [ ] 1.7 Failing tests: concurrent fan-out delegation produces multiple `delegation` edges and
  a deterministic, acyclic segment set.
- [ ] 1.8 Failing tests: recording is gated to trained rollouts (`INTERACTION_DAG_ENABLED` AND
  `s.Training != nil`); a non-trained rollout records nothing and makes no `close_segment`/
  export calls.
- [ ] 1.9 Failing tests: a recording error is logged and the run continues (best-effort).
- [ ] 1.10 Implement the D10 chokepoint + the delegation/mention/completion/squad seams behind
  the flag.
- [ ] 1.11 Commit: `feat(v2-segment-dag-recording): wire recorder into trained-rollout path (U7.2)`.

## 2. U7.3 - Fresh areal RL session per retry (D9) (TDD)

**Files:** `server/internal/service/task.go` (`CreateRetryTask`, `MaybeRetryFailedTask`),
`server/internal/service/training.go`, tests.

- [ ] 2.1 Failing tests: `CreateRetryTask` strips `areal_proxy` from the child context while
  keeping the chat `session_id`/`work_dir` resume CASE-WHEN.
- [ ] 2.2 Failing tests: `MaybeRetryFailedTask` calls `tryOpenTrainingSession(child,
  projectID, envID)` BEFORE `NotifyTaskEnqueued` (mirror `enqueueMentionTask` ordering).
- [ ] 2.3 Failing tests: the child opens its own session (`StartSession` fires) and the child's
  `agent_run_id` (= child `task.ID`) is recorded via D10.
- [ ] 2.4 Failing tests: the parent session is closed (`EndSession`) before the child opens.
- [ ] 2.5 Failing tests: a non-retryable failure is terminal (session closed, no child).
- [ ] 2.6 Implement the retry-session lifecycle (D9).
- [ ] 2.7 Commit: `feat(v2-segment-dag-recording): fresh areal RL session per retry (U7.3, D9)`.

## 3. U8 - AssembledDag assembly + /dag endpoint (TDD)

**Files:** `server/internal/service/interaction_dag.go`,
`server/internal/handler/env_dispatch.go`, `server/internal/handler/env_dispatch_test.go`,
hand-written sqlc for `AssembleAssembledDag`.

- [ ] 3.1 Failing tests: `AssembleAssembledDag(project_id)` returns `segments`, `edges`,
  `session_to_agent_run` with each segment carrying `trajectory_id` + `tensor_ref` +
  `closing_event` + `env_snapshot` (no scores, no turn indices, no text).
- [ ] 3.2 Failing tests: edges carry `src` / `dst` / `type` = `delegation` / `mention` /
  `completion`; the assembled DAG is acyclic.
- [ ] 3.3 Failing tests: `GET /api/v1/env-dispatch/{projectID}/dag` returns `202` in progress,
  `200` + `AssembledDag` done, `404` unknown project, `403` cross-workspace.
- [ ] 3.4 Failing tests: an incomplete/failed rollout yields a `failed` status, not a partial
  `AssembledDag`; a densely-covered failed run returns `200` + `AssembledDag`.
- [ ] 3.5 Implement the read-only assembly from recorded rows + the polling handler.
- [ ] 3.6 Commit: `feat(v2-segment-dag-recording): assemble AssembledDag + /dag endpoint (U8)`.

## 4. U10 - Config + E2E + grep sweep (both repos)

- [ ] 4.1 Add `INTERACTION_DAG_ENABLED` default (on for trained rollouts) + areal polling
  config (interval, timeout, backoff) for `MulticaDagClient`.
- [ ] 4.2 Scoped multica Go build/test/vet for touched packages; `gofmt -l` clean.
- [ ] 4.3 AReaL: `.venv-test/bin/python -m pytest areal/v2/inference_service/tests/` and
  `customized_areal/tree_search/tests/ -k 'segment_dag or supernode or env_dispatch'`.
- [ ] 4.4 Cross-repo E2E if feasible: 3-agent `mode=scratch` rollout -> `close_segment` +
  export per event -> poll `GET .../dag` -> AReaL resolves refs and reconstructs the
  `ExecutionDAG` losslessly -> minimal training step -> cleanup.
- [ ] 4.5 Verify env snapshots are refs-only (no sandbox pause/fork) - F-independence.
- [ ] 4.6 grep sweep: `close_segment`, `AssembledDag`, `tensor_ref`, `v2-segment-dag-recording`,
  `env-dispatch/{projectID}/dag` resolve to intended code only; no `start_turn_idx` /
  `end_turn_idx` remain in the new tables/code.
- [ ] 4.7 Final whole-branch review -> READY TO MERGE / NEEDS_CHANGES.
- [ ] 4.8 Commit: `docs(v2-segment-dag-recording): U10 full regression + E2E + grep sweep`.

## Test runners / constraints
```

Full source: openspec/changes/multica-v2-segment-dag-recording-assembly/tasks.md

## openspec/changes/multica-v2-segment-dag-recording-assembly/specs/v2-segment-dag-recording/spec.md

- Source: openspec/changes/multica-v2-segment-dag-recording-assembly/specs/v2-segment-dag-recording/spec.md
- Lines: 1-138
- SHA256: dbaa81e2f73a74279a9d4ae602611661da062ffd7ca4af922fcd27767b439f42

[TRUNCATED]

```md
# v2-segment-dag-recording

## ADDED Requirements

### Requirement: Trained-rollout segment recording gating

Multica SHALL record interaction-DAG segments only for trained rollouts (a `TrainAgentID` is
set on the env dispatch) and only while `INTERACTION_DAG_ENABLED` is on. For each communication
event (delegation, mention, completion, squad briefing) on a trained rollout, Multica SHALL
call `close_segment` + per-segment export on the relevant run and record a segment carrying
`trajectory_id`, `tensor_ref`, `closing_event`, and a ref-only `env_snapshot`. A leaf run with
no communication event SHALL yield exactly one leaf segment with `closing_event = None`.
Non-trained rollouts SHALL record nothing and SHALL incur no recording overhead.

#### Scenario: Trained rollout records a segment per communication event
- **WHEN** a trained rollout fires a delegation, mention, completion, or squad-briefing event
- **THEN** Multica calls `close_segment` + export and records a segment with `trajectory_id` +
  `tensor_ref` + `closing_event` + `env_snapshot` for that event

#### Scenario: Leaf run yields one leaf segment
- **WHEN** a trained run completes with no communication event
- **THEN** exactly one leaf segment is recorded with `closing_event = None`

#### Scenario: Non-trained rollouts record nothing
- **WHEN** a rollout has no `TrainAgentID` or `INTERACTION_DAG_ENABLED` is off
- **THEN** no segment or edge is recorded and no `close_segment`/export call is made

#### Scenario: Concurrent fan-out produces a deterministic acyclic edge set
- **WHEN** a delegation fans out to multiple children concurrently
- **THEN** one `delegation` edge per child is recorded, the segment set is deterministic, and
  the resulting directed graph is acyclic

### Requirement: Retry-attempt areal RL session lifecycle

Each retry attempt SHALL open its own fresh areal RL session. `CreateRetryTask` SHALL strip
`areal_proxy` from the child task's context (preserving the chat `session_id`/`work_dir` resume
state), and `MaybeRetryFailedTask` SHALL open a fresh session for the child via
`tryOpenTrainingSession` before `NotifyTaskEnqueued`. The parent's areal session SHALL be closed
(`EndSession`) before the child opens its own. The child's `agent_run_id` (= `task.ID`) SHALL
be recorded. Inheriting the parent's `areal_proxy` context into a retry child is forbidden. A
non-retryable failure is terminal: the session is closed and no child is created.

#### Scenario: Retry child opens its own fresh session
- **WHEN** a retryable failure produces a retry child
- **THEN** the child's context has no `areal_proxy`, the parent session is closed, the child
  opens a new areal session, and the child's `agent_run_id` is recorded

#### Scenario: Parent session closed before child opens
- **WHEN** a retry is created
- **THEN** the parent's areal session is ended before the child's `StartSession` is called

#### Scenario: Non-retryable failure is terminal
- **WHEN** a non-retryable failure occurs
- **THEN** the areal session is closed and no retry child is created

### Requirement: Session-to-agent-run recording at session open

Multica SHALL record the `session_id` <-> `agent_run_id` mapping at the moment a training
session is opened, as a single idempotent chokepoint shared by the `Enqueue*` task paths and
the env-dispatch path. `agent_run_id` SHALL equal `task.ID` (the run), not the agent ID. The
recorded mapping SHALL populate the `session_to_agent_run` map consumed by `AssembledDag`.

#### Scenario: Mapping recorded on first session open
- **WHEN** a trained run opens its areal session for the first time
- **THEN** `{session_id, agent_run_id=task.ID}` is recorded once and is idempotent across
  repeated open attempts

#### Scenario: Both enqueue and env-dispatch paths funnel through the chokepoint
- **WHEN** a trained run is opened via either an `Enqueue*` path or the env-dispatch adapter
- **THEN** the same `session_to_agent_run` recording fires, keyed by `task.ID`

### Requirement: AssembledDag assembly from recorded rows

At root-task completion Multica SHALL assemble `AssembledDag = {segments, edges,
session_to_agent_run}` by reading the recorded `interaction_dag_segment`, `interaction_dag_edge`,
`interaction_dag_env_snapshot`, and `interaction_dag_session_run` rows for the project. Each
segment SHALL carry `segment_id`, `agent_run_id`, `issue_id`, `trajectory_id`, `tensor_ref`,
`closing_event`, and a ref-only `env_snapshot`. Edges SHALL carry `src`, `dst`, and `type` =
`delegation` / `mention` / `completion`. The assembled DAG SHALL carry no scores, no turn
indices, and no message text. Assembly SHALL NOT re-derive structure; it is a read-only
```

Full source: openspec/changes/multica-v2-segment-dag-recording-assembly/specs/v2-segment-dag-recording/spec.md


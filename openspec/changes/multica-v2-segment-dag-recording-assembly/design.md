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
zero recording overhead. The gate composes at each seam: `maybeOpenTrainingSession` already
gates on training; the event hooks additionally check the flag before calling
`CloseSegmentForEvent`/`AddEdge`.

### D12 - Hook ordering: record-before-notify [new]

`RecordSessionAgentRun` fires inside `maybeOpenTrainingSession` after `StartSession` succeeds
(D10). For retry children (D9), `tryOpenTrainingSession(child)` is called BEFORE
`NotifyTaskEnqueued`, mirroring `enqueueMentionTask` (open at task.go:614, notify at :618), so
the `session_to_agent_run` map is populated before the child can be claimed or execute.

### D13 - `AssembledDag` assembly is read-only over recorded rows [new]

`AssembleAssembledDag(project_id)` reads `interaction_dag_segment` + `_edge` +
`_env_snapshot` + `interaction_dag_session_run` (U7.1's migration 158) and assembles
`{segments, edges, session_to_agent_run}`. No re-derivation of structure; acyclicity is AReal's
concern (the assembler validates topological order). The `/dag` endpoint serves this directly.

### D14 - `/dag` failed-status vs partial DAG [new, refines parent polling]

A root task that completes with failure returns `200` + `AssembledDag` only if the recorded
segments densely cover the run; an incomplete/failed rollout (missing segments, unrecorded
events) returns `200` with a `failed` status body, NOT a partial `AssembledDag`. This refines
the parent's "200 on success or failed": the distinguishing axis is dense coverage, not terminal
state. Reconcile with the parent's polling requirement on apply. AReal's assembler also
validates dense coverage (U5 gap-check) as a second line of defense.

### D15 - `/dag` cross-workspace `403` [new]

The `/dag` endpoint enforces workspace isolation: a caller requesting a `project_id` outside
their workspace gets `403`, distinct from `404` (unknown project). Mirrors the existing
env-dispatch workspace gate.

## Risks / Trade-offs

- **D9 touches pre-existing `CreateRetryTask` (migration 055)** - in-scope dependency, blast
  radius on the retry path. Mitigation: strip only `areal_proxy`; keep chat `session_id`/
  `work_dir`; integration tests assert child opens a fresh session.
- **Hook ordering bugs (record-after-notify)** -> dangling segments at assembly. Mitigation:
  D12 ordering; integration tests assert the map is populated before claim.
- **`/dag` serving a partial DAG on failure** -> AReal trains on incomplete data. Mitigation:
  D14 failed-status; AReal's assembler validates dense coverage (U5 gap-check).
- **Concurrent fan-out** -> edge ordering nondeterminism. Mitigation: deterministic insert
  order; acyclicity validated downstream.
- **Sweeper path bypasses `FailTask`** -> orphaned session, no child (pre-existing gap).
  Coverage boundary; not expanded here.

## Open Questions

- `envID` source for a retry child's `StartSession` (enqueue paths pass `""`; env_dispatch
  passes real envID; retry child is not env-dispatched). Resolve during U7.3.
- Exact `tensor_ref` shape `ExportTrajectory` returns (U6) - `CloseSegmentForEvent` must decode
  it from the raw traj JSON. Resolve during U7.2.
- `envSnapshot` source for `CloseSegmentForEvent` (refs-only, no sandbox pause/fork). Resolve
  during U7.2.

## Dual-repo / Build

- **Multica Go**: continues on `feature/multica-v2-segment-dag-training` (U7.1 tip `157284045`);
  U7.2/U7.3/U8 commit there.
- **AReaL**: branch off `f60c86bb` (master); only U10 config/E2E/grep.
- **Env (areal)**: `.venv-test/bin/python -m pytest <path>` (project `uv`/`.venv` broken);
  `uvx ruff check <paths>`. Do not trust implementer GREEN self-reports - re-run tests.
- **Env (multica)**: `DATABASE_URL=... go run ./cmd/migrate up`; scoped `go build`/`go test`.
  Pre-existing (NOT ours, do not gate): `go build ./...` webpush.go:180; 16 handler `ON CONFLICT`
  daemon/claim failures.
- **sqlc generate is broken** in the multica repo; hand-write generated Go mirroring a sibling
  query (mirrors U7.1's approach).

## Out of Scope (change 2 / 3 / F)

Judge + process-reward scoring + actor-model V critic + GAE (change 2); tree search / branching
/ `env-dispatch mode=branch` fork (change 3, landed); full sandbox snapshot/fork (Sub-project F).

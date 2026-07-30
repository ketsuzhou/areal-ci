---
comet_change: multica-v2-segment-dag-recording-assembly
role: technical-design
canonical_spec: openspec
---

# Design: Multica v2 Segment-DAG Recording + Assembly (Change 1 tail)

Canonical capability spec: `openspec/changes/multica-v2-segment-dag-recording-assembly/specs/v2-segment-dag-recording/spec.md`.
OpenSpec artifacts (proposal/design/tasks) are the upstream source of truth. This document
deepens the HOW (implementation approach, risks, testing, boundary conditions) and does not
restate requirements.

## Context

This change is the multica-side completion of the `v2-segment-dag` data path (change 1). The
areal consumer (U1-U5, on master) polls `GET .../dag`, resolves `tensor_ref`s, builds
`SuperNode`/`ExecutionDAG`, runs a placeholder-reward training step, and cleans up. The multica
building blocks exist on `feature/multica-v2-segment-dag-training`: the arealrl
`CloseSegment`/`ExportTrajectory` client (U6), migration 155-158 (U9 + U7.1's session_run), and
the `InteractionDAGService` recorder + sqlc (U7.1). What is missing is the wiring that makes a
real rollout produce a complete `AssembledDag`: event hooks, retry-session correctness, assembly,
the polling endpoint - and a `tensor_ref` contract fix the wiring surfaces.

Code grounding (live seam trace, `multica/server` + areal master):

- `InteractionDAGService` (`internal/service/interaction_dag.go`): `RecordSessionAgentRun(ctx,
  projectID, sessionID, agentRunID, issueID)` (`:91`, upsert ON CONFLICT session_id);
  `CloseSegmentForEvent(ctx, projectID, sessionID, proxyKey, closingEvent, envSnapshot)
  (segmentID, err)` (`:141`: GetSessionRun -> `CloseSegment` -> `ExportTrajectory` ->
  `decodeTensorRef` -> atomic `InsertInteractionDAGSegmentWithSnapshot`); `AddEdge(...)`
  (`:209`); `SegmentIDForAgentRun(agentRunID)` (`:112`, DELEGATION parent lookup);
  `Enabled()` (`:82`). Gated by `INTERACTION_DAG_ENABLED` + `TrainingSessionDeps.DAG` nil-check
  (`training.go:115`).
- Migrations 155-158 exist: `interaction_dag_segment` (text PK, `tensor_ref jsonb NOT NULL`),
  `_edge` (bigserial, CHECK type), `_env_snapshot` (segment_id PK FK CASCADE, `sandbox_ids
  jsonb`, `issue_snapshot_id`, `env_state jsonb default '{}'`), `_session_run` (session_id PK).
- `/dag` route mounts flat on chi at `cmd/server/router.go:1035-1036` alongside the existing
  env-dispatch POST/DELETE; `GetDag` is a new method on `*Handler`.
- Retry: D9 strip point is `agent.sql:186` (`p.context` copied verbatim); chat
  `session_id`/`work_dir` CASE-WHEN at `:187-188`. `MaybeRetryFailedTask` (`task.go:1753`)
  creates the child at `:1780`, then `NotifyTaskEnqueued` at `:1800`. Call order
  `FailTask` (`:1588`) -> `RouteTerminalTrainingTask` (`:1663`, closes parent via `EndSession`)
  -> `MaybeRetryFailedTask` (`:1668`) already satisfies "close parent before child opens".

## Approach

### U7.2 - Wire the recorder into the trained-rollout path

- **D10 chokepoint**: call `RecordSessionAgentRun(ctx, projectID, creds.SessionID, taskID,
  issueID)` inside `maybeOpenTrainingSession` (`training.go:~226`) right after `StartSession`
  succeeds and persists, before the slog. `agentRunID = taskID` (NOT `agentID`). Idempotent via
  the upsert. This is the single chokepoint for both `Enqueue*` paths (via
  `tryOpenTrainingSession`, `task.go:510/614/750/834`) and the env_dispatch adapter
  (`handler/env_dispatch.go:713` -> `MaybeOpenTrainingSession`).
- **Event seams** (trained rollouts only, `INTERACTION_DAG_ENABLED ∧ s.Training`):
  - delegation: at `enqueueMentionTask` (`task.go:581`), after creating the child task,
    `CloseSegmentForEvent` on the parent (closing_event=`delegation`) + `AddEdge(parentSeg,
    childSeg, "delegation")`. Parent segment resolved via `SegmentIDForAgentRun(parentTaskID)`.
  - mention: `AddEdge(..., "mention")` without closing a segment.
  - completion: at `RouteTerminalTrainingTask` (`task.go:1425`),
    `CloseSegmentForEvent` on the completing run (closing_event=`completion`) + `AddEdge(...,
    "completion")`.
  - squad briefing: at the daemon claim path (env_dispatch), where `SandboxRefs` are in scope.
  - leaf: `CompleteTask` with `closing_event=""` -> one leaf segment.
- **envSnapshot** (lean/refs-only): `sandbox_ids` from `project.env_id -> environment.sandbox_ids`
  (one DB hop); `issue_snapshot_id` NULL (no snapshot capture - F-independence); `env_state`
  `{}`. At the task.go seams `SandboxRefs` are not in hand, so this is the available shape.

### U7.3 - Fresh areal RL session per retry (D9)

- `CreateRetryTask` (`agent.sql:186`): strip `areal_proxy` from the child's `context` JSONB
  (keep the chat `session_id`/`work_dir` resume CASE-WHEN at `:187-188`). Implemented as a
  targeted context rewrite in the Go caller before/after the sqlc call (the query copies
  `p.context` verbatim, so the strip is applied to the parent context blob read for the child -
  or the child row is updated post-insert; pick the lower-risk of the two at impl time and
  document it).
- `MaybeRetryFailedTask` (`task.go:1753`): before `NotifyTaskEnqueued` (`:1800`), call
  `tryOpenTrainingSession(child, projectID, envID)` mirroring `enqueueMentionTask` (open `:614`,
  notify `:618`). `envID` from `project.env_id` resolved via the child's inherited `issue_id`
  (`GetIssue -> issue.ProjectID -> GetProject -> project.EnvID`), exactly as
  `RouteTerminalTrainingTask` does (`training.go:446-451`). Non-env-dispatched parents have NULL
  `project.env_id` -> `envID=""` (matches the Enqueue seams).
- `RecordSessionAgentRun` (D10) then fires for the child, mapping the child's fresh session to
  the child's `task.ID`. Only the 4 retryable reasons (`runtime_offline`/`runtime_recovery`/
  `timeout`/`codex_semantic_inactivity`, `task.go:1725`) produce a child.

### U8 - AssembledDag assembly + /dag endpoint

- `AssembleAssembledDag(project_id)`: read-only projection over `interaction_dag_segment` +
  `_edge` + `_env_snapshot` + `_session_run` into `{segments, edges, session_to_agent_run}`.
  No re-derivation; acyclicity is AReal's concern (`topological_order()` raises `DAGError`).
  Hand-write the sqlc (sqlc generate is broken in this repo).
- `GetDag` handler (`router.go:1035`): `202` in-progress, `200` + `AssembledDag` done, `404`
  unknown project, `403` cross-workspace (mirror the existing workspace gate). Failed/incomplete
  rollout (segments do not densely cover the run) -> `200` + `failed` status, NOT a partial
  `AssembledDag` (D14).

### tensor_ref multi-shard contract (Option B) - the cross-repo fix

The real areal `/export_trajectories` (`data_proxy/app.py:780-781`) returns
`traj = RTensor.remotize(concat_padded_tensors([...]))` - a dict of **N per-field `RTensor`
shards** (`input_ids`/`seq_tokens`, `attention_mask`, `logprobs`, `loss_mask`, `versions`), each
with its own `shard_id` (`rtensor.py:391-444`). `/data/<shard_id>` returns **one** tensor
(`data_blueprint.py:78-100`). U7.1's `decodeTensorRef` + tests assumed a single
`{"shard_id":str}`/`{"tensor_ref":{"shard_id":...}}` shape the real endpoint never produces, so
`decodeTensorRef` silently stores the whole traj dict and the U5 consumer `KeyError`s at
resolve.

Fix (Option B - keep export as-is, multi-shard contract):
- **Multica**: `decodeTensorRef` extracts the field->`{shard_id, node_addr}` map from the real
  traj dict (keys are the tensor field names). Store that map as the segment's `tensor_ref`
  jsonb. Update U7.1's fake `ExportTrajectory` + tests to the real shape.
- **AReal** (master, branch off `f60c86bb`): `SegmentSpec.tensor_ref` becomes the field->shard
  map; `DataProxyTensorResolver.resolve` fetches each field's shard via `/data/<shard_id>` and
  reassembles the field->tensor dict; `assemble_from_refs` stamps the reassembled dict into
  `metadata["tensors"]` as before.

This expands this change's areal scope (the proposal's "no new areal runtime code" line is
corrected): the consumer fix is required for the path to round-trip.

## Data Flow (single trained rollout, with retry)

```
Enqueue* / env_dispatch -> maybeOpenTrainingSession -> StartSession -> RecordSessionAgentRun(D10)
turns via /chat/completions (v2 captures, no reward)
[delegation] CloseSegmentForEvent(parent) + AddEdge(delegation) -> child task
[mention]    AddEdge(mention)
[completion] CloseSegmentForEvent(child) + AddEdge(completion)
[leaf]       CompleteTask -> CloseSegmentForEvent(closing_event="")
root completion -> AssembleAssembledDag(project_id)
AReal polls GET .../dag (202 -> 200 + AssembledDag) -> resolve multi-shard tensor_refs -> SuperNode/ExecutionDAG -> train (placeholder) -> /data/clear + remove_session

[retryable failure] FailTask -> RouteTerminalTrainingTask (EndSession parent) ->
  MaybeRetryFailedTask: CreateRetryTask strips areal_proxy -> tryOpenTrainingSession(child, envID=project.env_id) BEFORE NotifyTaskEnqueued -> child opens fresh S_B -> RecordSessionAgentRun(child task.ID)
```

## Boundary Conditions / Error Handling

- Empty segment (`close_segment` on no active) -> v2 `ValueError` -> 400; `CloseSegmentForEvent`
  logs and the run continues (best-effort recording, D11).
- `tensor_ref` decode: a field missing `shard_id` -> resolver raises a typed error for that
  segment; no silent `None` masking (project rule: absence stays distinguishable at boundaries).
- Ref lifetime: `/data/clear` only after training resolves refs; `remove_session` only after the
  session's last segment is consumed (unchanged from U5).
- Concurrent fan-out delegation -> multiple `delegation` edges, deterministic insert order;
  acyclicity validated downstream.
- Missing segment (gap in coverage) -> `/dag` failed-status (D14); AReal assembler also raises
  `DAGError` (U5 gap-check) as a second line.
- Retry child envID: `project.env_id` NULL -> `""` (matches Enqueue seams); areal `SessionID`
  is NOT envID.
- Sweeper path (`runtime_sweeper.FailStaleTasks`, raw SQL) bypasses `FailTask` -> orphaned
  session, no child. Pre-existing gap; D9 coverage boundary is the `FailTask` path.

## Testing Strategy

- **Multica Go** (hermetic integration on real Postgres, mirroring U7.1's `interaction_dag_test.go`):
  D10 fires once + idempotent; each event seam records the right segment/edge; leaf; concurrent
  fan-out (deterministic acyclic); non-trained records nothing; recording error is best-effort.
  D9: child context has no `areal_proxy`; child opens fresh session; parent closed first;
  non-retryable is terminal. U8: assembly shape (no scores/turn-idx/text, acyclic); `/dag`
  202/200/404/403; failed-status vs partial DAG.
- **AReal** (`.venv-test/bin/python -m pytest`, NOT `uv run`): `DataProxyTensorResolver`
  multi-shard resolve (fetch N shards -> reassembled dict; missing `shard_id` raises); a
  real-shape 3-segment planner->worker->synthesizer round-trip through `assemble_from_refs`;
  regression on U3/U4/e2e. `uvx ruff check` clean.
- **U10 E2E** (if feasible): 3-agent `mode=scratch` rollout -> segments -> `AssembledDag` ->
  resolve -> `ExecutionDAG` -> minimal training -> cleanup. Hardware-gated skip documented if
  unavailable.
- Scope Go builds to touched packages (pre-existing `webpush.go:180` + 16 handler `ON CONFLICT`
  failures are NOT ours). Hand-write sqlc (sqlc generate broken). Re-run tests ourselves; do not
  trust implementer GREEN self-reports.

## Decisions

D8 `agent_run_id` = `task.ID` (attempt-level); D9 fresh areal session per retry; D10
`RecordSessionAgentRun` chokepoint post-`StartSession`; D11 recording gated to trained rollouts
(`INTERACTION_DAG_ENABLED ∧ s.Training`); D12 record-before-notify; D13 read-only assembly;
D14 `/dag` failed-status vs partial DAG; D15 `/dag` 403 cross-workspace.

**D16 (new) - Multi-shard `tensor_ref` contract (Option B).** `tensor_ref` is a field->shard map
matching the real `/export_trajectories` per-field `RTensor.remotize` output; the areal resolver
fetches all shards and reassembles. Chosen over Option A (areal single-shard merge) because it
works with the existing remotize/fetch model and localizes the change to the segment-DAG path
rather than altering the on-master export used by other consumers.

## Open Items (resolve during impl)

- Exact Go mechanism for the `areal_proxy` strip in `CreateRetryTask` (pre-insert context
  rewrite vs post-insert update) - pick lower-risk at U7.3.
- `issue_snapshot_id`: confirmed NULL for this change (no snapshot capture); revisit if a future
  change adds issue snapshots.
- U10 E2E hardware availability.

## Out of Scope (change 2 / 3 / F)

Judge + process-reward scoring + actor-model V critic + GAE (change 2); tree search / branching
(change 3, landed); full sandbox snapshot/fork (Sub-project F).

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

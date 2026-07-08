## 0. Supersede sub-project-g

- [x] 0.1 Mark `sub-project-g-multica-interaction-dag` superseded: set `archived: true` in
  its `.comet.yaml`, add a `Superseded by: multica-v2-segment-dag-training` note + reason
  (contract changed: turn-index -> close_segment + tensor_ref; scores moved to AReaL judge)
  to its `proposal.md`. Do not move/delete its files.
- [ ] 0.2 Commit: `gov(multica-v2-segment-dag): supersede sub-project-g interaction-dag`.

## 1. Investigation - v2 surface and Multica seams

- [ ] 1.1 Confirm v2 `SessionData` (`areal/v2/inference_service/data_proxy/session.py`):
  `set_reward` close path (`_mark_active_trajectory_ready_locked`,
  `_last_reward_interaction_id`), `ready_trajectories` OrderedDict, `export_trajectory` pop
  semantics. Define the no-reward close seam for `close_segment`.
- [ ] 1.2 Confirm v2 `/export_trajectories` (`data_proxy/app.py`): current multi-session
  merge + `RTensor.remotize` path; what it takes to export a single `trajectory_id` with
  `remove_session=False`.
- [ ] 1.3 Confirm v2 data_proxy `/data/<shard_id>`, `/data/batch`, `/data/clear` surface
  (add what is missing).
- [ ] 1.4 Confirm v2 gateway routes (`gateway/app.py`) to add `/rl/close_segment` and
  per-trajectory export.
- [ ] 1.5 Confirm Multica driving seam (proxy `/chat/completions` boundary) and where
  `close_segment` + export are called per communication event (delegation / mention /
  completion / squad briefing).
- [ ] 1.6 Confirm `session_id <-> agent_run_id` capture at `/rl/start_session`
  (`internal/arealrl/client.go`) and the root-task completion signal that flips the DAG
  endpoint `202` -> `200`.
- [ ] 1.7 Decide `close_segment` on empty active (error vs empty trajectory) and record in
  design.
- [ ] 1.8 Commit: `docs(v2-segment-dag): T1 v2 + multica seam confirmation`.

## 2. Migration + DB queries (Multica)

- [ ] 2.1 Add migration: `interaction_dag_segment` (segment_id, project_id, agent_run_id,
  issue_id, task_id, trajectory_id, tensor_ref, closing_event, closing_event_target,
  created_at), `interaction_dag_edge` (src_segment_id, dst_segment_id, type),
  `interaction_dag_env_snapshot` (segment_id, sandbox_ids, issue_snapshot_id, env_state).
  No `start_turn_idx` / `end_turn_idx` columns.
- [ ] 2.2 Generate / add DB query files: `CreateSegment`, `CloseSegment` (set
  trajectory_id + tensor_ref + closing_event), `AddEdge`, `CaptureEnvSnapshot`,
  `RecordSessionAgentRun`, `ListSegmentsForProject`, `ListEdgesForProject`,
  `ListEnvSnapshotsForProject`, `GetDagStatus`.
- [ ] 2.3 Ensure `DELETE /api/v1/env-dispatch/{projectID}` cascades to the new tables.
- [ ] 2.4 Commit: `feat(v2-segment-dag): migration + queries for segments edges snapshots`.

## 3. V2 close_segment (TDD)

**Files:** `areal/v2/inference_service/data_proxy/session.py`,
`areal/v2/inference_service/data_proxy/app.py`, `areal/v2/inference_service/gateway/app.py`,
tests under `areal/v2/inference_service/tests/`.

- [x] 3.1 Failing tests: `close_segment` moves active completions into a ready trajectory
  and returns its `trajectory_id`, assigning no reward.
- [x] 3.2 Failing tests: session stays live after `close_segment` - further turns capture
  into a new active segment that can be closed independently.
- [x] 3.3 Failing tests: `close_segment` on a session with no active completions returns a
  typed error and produces no trajectory.
- [x] 3.4 Failing tests: `close_segment` does not require a prior `set_reward` (no
  `_last_reward_interaction_id` dependency).
- [x] 3.5 Implement `SessionData.close_segment()` + data_proxy `/rl/close_segment` + gateway
  route. (commits d77fab31 + 632e4949 + 6bde9872; 7/7 tests green, ruff clean)
- [x] 3.6 Commit: `feat(v2-segment-dag): close_segment no-reward active->ready close`.

## 4. V2 per-segment tensor-ref export + data_proxy resolve (TDD)

**Files:** `areal/v2/inference_service/data_proxy/app.py`,
`areal/v2/inference_service/gateway/app.py`, tests under
`areal/v2/inference_service/tests/`.

- [ ] 4.1 Failing tests: `/export_trajectories` by `trajectory_id` with
  `remove_session=False` returns `RTensor` refs for that segment and leaves other
  trajectories ready.
- [ ] 4.2 Failing tests: exported payload is refs-only - no message text (only `input_ids`,
  `loss_mask`, `logprobs`, `versions`, `attention_mask`).
- [ ] 4.3 Failing tests: exporting an unknown `trajectory_id` returns a typed error.
- [ ] 4.4 Failing tests: `/data/<shard_id>` and `/data/batch` resolve refs to tensor bytes;
  `DELETE /data/clear` frees shards.
- [ ] 4.5 Implement per-trajectory export + `/data/*` resolve + `/data/clear` (add missing
  endpoints).
- [ ] 4.6 Commit: `feat(v2-segment-dag): per-segment tensor-ref export + data resolve`.

## 5. Multica incremental segment recording (TDD)

**Files:** `internal/service/interaction_dag.go`, `internal/service/interaction_dag_test.go`.

- [ ] 5.1 Failing tests: at a delegation event Multica calls `close_segment` + export on the
  parent run, records a segment with `trajectory_id` + `tensor_ref` + `closing_event`, opens
  a child segment, records a `DELEGATION` edge.
- [ ] 5.2 Failing tests: mention records a `MENTION` edge without closing a segment.
- [ ] 5.3 Failing tests: completion calls `close_segment` + export on the child run, records
  a `COMPLETION` edge.
- [ ] 5.4 Failing tests: squad briefing closes a segment via `close_segment` + export.
- [ ] 5.5 Failing tests: leaf run (no communication event) yields one leaf segment with
  `closing_event = None`.
- [ ] 5.6 Failing tests: concurrent fan-out delegation produces multiple `DELEGATION` edges
  and a deterministic, acyclic segment set.
- [ ] 5.7 Failing tests: `RecordSessionAgentRun` captures `session_id <-> agent_run_id`.
- [ ] 5.8 Failing tests: a recording error is logged and the run continues (best-effort).
- [ ] 5.9 Implement `InteractionDAGService` (record turn, close segment + export on event,
  add edge, capture ref-only snapshot) behind a feature flag.
- [ ] 5.10 Wire hooks from task / mention / completion / squad-briefing seams for trained
  rollouts only.
- [ ] 5.11 Commit: `feat(v2-segment-dag): incremental segment + edge recording`.

## 6. AssembledDag assembly + polling endpoint (TDD)

**Files:** `internal/service/interaction_dag.go`, `internal/handler/env_dispatch.go`,
`internal/handler/env_dispatch_test.go`.

- [ ] 6.1 Failing tests: `AssembleAssembledDag(project_id)` returns `segments`, `edges`,
  `session_to_agent_run` with each segment carrying `trajectory_id` + `tensor_ref` +
  `closing_event` + `env_snapshot` (no scores, no turn indices, no text).
- [ ] 6.2 Failing tests: edges carry `src` / `dst` / `type` = `delegation` / `mention` /
  `completion`; assembled DAG is acyclic.
- [ ] 6.3 Failing tests: `GET /api/v1/env-dispatch/{projectID}/dag` returns `202` in
  progress, `200` + `AssembledDag` done, `404` unknown project, `403` cross-workspace.
- [ ] 6.4 Failing tests: incomplete/failed rollout yields a `failed` status, not a partial
  `AssembledDag`.
- [ ] 6.5 Implement assembly from recorded rows + the polling handler.
- [ ] 6.6 Commit: `feat(v2-segment-dag): assemble AssembledDag + polling endpoint`.

## 7. AReaL client adapter + ref-resolve assembler (TDD)

**Files:** `customized_areal/tree_search/agents/multica_dag_client.py` (or extend
`MulticaEnvDispatchClient`), `customized_areal/tree_search/agents/supernode_assembler.py`,
tests under `customized_areal/tree_search/tests/`.

- [ ] 7.1 Failing tests: `get_env_dispatch_dag(project_id)` polls until `200` and returns an
  `AssembledDag`; `202` retried with backoff; `404`/`403` raise typed errors.
- [ ] 7.2 Failing tests: assembler resolves each segment's `tensor_ref` via `/data/*` to
  tensors and builds a `SuperNode(payload=tensors, metadata=segment)` - no turn-index slice.
- [ ] 7.3 Failing tests: edges build an acyclic `ExecutionDAG` reconstructing
  delegation / mention / completion topology.
- [ ] 7.4 Failing tests: a session whose segments do not densely cover the run raises
  `DAGError` (no partial DAG).
- [ ] 7.5 Failing tests: a synthetic 3-segment planner -> worker -> synthesizer
  `AssembledDag` round-trips losslessly into the `ExecutionDAG`.
- [ ] 7.6 Implement the adapter + the ref-resolve assembler path.
- [ ] 7.7 Commit: `feat(v2-segment-dag): areal client + ref-resolve assembler`.

## 8. Minimal training plumbing + tensor lifecycle (TDD)

**Files:** `customized_areal/tree_search/agents/` (training entry), tests under
`customized_areal/tree_search/tests/`.

- [ ] 8.1 Failing tests: the built DAG is consumed by a minimal training step using a
  placeholder reward (terminal `set_reward` or zero) - end-to-end path exercises without a
  judge.
- [ ] 8.2 Failing tests: after training, `DELETE /data/clear` frees the resolved shards and
  the session is revoked once its last segment is consumed.
- [ ] 8.3 Failing tests: refs are not cleared before training resolves them.
- [ ] 8.4 Implement the minimal training hook + cleanup sequence.
- [ ] 8.5 Commit: `feat(v2-segment-dag): minimal training plumbing + tensor lifecycle`.

## 9. Config + production wiring

- [ ] 9.1 Add `INTERACTION_DAG_ENABLED` (default on for trained rollouts) and v2
  `close_segment` / per-segment export feature flags.
- [ ] 9.2 Add polling config on the AReaL side (interval, timeout, backoff).
- [ ] 9.3 Wire the recording service + v2 ops into construction behind the flags.
- [ ] 9.4 Document the `AssembledDag` contract + v2 ops in
  `customized_areal/tree_search/agents/multica_environment_protocol.md`.
- [ ] 9.5 Commit: `chore(v2-segment-dag): config + production wiring`.

## 10. Full regression + E2E + grep sweep

- [ ] 10.1 Scoped Multica Go build/test/vet for touched packages; `gofmt -l` clean.
- [ ] 10.2 AReaL: `uv run pytest areal/v2/inference_service/tests/` and
  `uv run pytest customized_areal/tree_search/tests/ -k 'segment_dag or supernode or env_dispatch'`.
- [ ] 10.3 Cross-repo E2E if feasible: `mode=scratch` rollout with a 3-agent team ->
  `close_segment` + export per event -> poll `GET .../dag` -> AReaL resolves refs and
  reconstructs the `ExecutionDAG` losslessly -> minimal training step runs.
- [ ] 10.4 Verify env snapshots are refs-only (no sandbox pause/fork) - F-independence.
- [ ] 10.5 grep sweep: `close_segment`, `AssembledDag`, `tensor_ref`, `v2-segment-dag`,
  `env-dispatch/{projectID}/dag` resolve to intended code only; no `start_turn_idx` /
  `end_turn_idx` remain in the new tables/code.
- [ ] 10.6 Final whole-branch review -> READY TO MERGE / NEEDS_CHANGES.
- [ ] 10.7 Commit: `docs(v2-segment-dag): T10 full regression + E2E + grep sweep`.

## Test runners / constraints

- AReaL v2 tests run from `backend/areal` with `uv run pytest areal/v2/inference_service/tests/`.
- AReaL consumer tests run from `backend/areal` with
  `uv run pytest customized_areal/tree_search/tests/ -k '<focused expression>'`.
- Multica Go tests are scoped to touched packages; avoid repo-wide commands if unrelated
  failures remain.
- Cross-repo E2E requires Multica services + the v2 inference_service; if unavailable,
  document the skipped prerequisites.
- This change uses a placeholder reward; real judge / V / GAE arrive in change 2 and tree
  search in change 3 - do not implement them here.

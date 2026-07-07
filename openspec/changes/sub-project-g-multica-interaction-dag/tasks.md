## 1. Investigation — interaction seams and turn-index source

- [ ] 1.1 Confirm where Multica drives each assistant turn (proxy `/chat/completions`
  boundary) and define the per-`agent_run_id` turn-counter seam for `start_turn_idx` /
  `end_turn_idx`.
- [ ] 1.2 Confirm the delegation seam (`issue.parent_issue_id`, `internal/service/task.go`)
  emits a `DELEGATION` edge source and closes the parent run's current segment.
- [ ] 1.3 Confirm the mention seam (`internal/mention`) emits a `MENTION` peer edge without
  closing a segment.
- [ ] 1.4 Confirm the completion / notify-parent seam emits a `COMPLETION` edge and closes
  the child run's current segment.
- [ ] 1.5 Confirm the squad-briefing seam (`handler/squad_briefing.go`) closes a segment.
- [ ] 1.6 Confirm `session_id ↔ agent_run_id` is captured at `/rl/start_session` time
  (`internal/arealrl/client.go`).
- [ ] 1.7 Confirm the root-task completion signal (agent-run terminal status vs. a
  rollout-complete marker) used to flip the DAG endpoint from `202` to `200`.
- [ ] 1.8 Document findings; commit: `docs(G): T1 interaction-DAG seam confirmation`.

## 2. Migration + DB queries

- [ ] 2.1 Add migration: `interaction_dag_segment` (segment_id, project_id, agent_run_id,
  issue_id, task_id, start_turn_idx, end_turn_idx, closing_event, closing_event_target,
  created_at), `interaction_dag_edge` (src_segment_id, dst_segment_id, type), and
  `interaction_dag_env_snapshot` (segment_id, sandbox_ids, issue_snapshot_id, env_state).
- [ ] 2.2 Generate / add DB query files: `CreateSegment`, `CloseSegment` (set
  end_turn_idx + closing_event), `AddEdge`, `CaptureEnvSnapshot`, `RecordSessionAgentRun`,
  `ListSegmentsForProject`, `ListEdgesForProject`, `ListEnvSnapshotsForProject`,
  `GetDagStatus` (project_id → in_progress|done|failed).
- [ ] 2.3 Ensure `DELETE /api/v1/env-dispatch/{projectID}` cascades to the new tables.
- [ ] 2.4 Commit: `feat(interaction-dag): migration + queries for segments edges snapshots`.

## 3. Incremental recording service (TDD)

**Files:** `internal/service/interaction_dag.go`,
`internal/service/interaction_dag_test.go`.

- [ ] 3.1 Failing tests: delegation closes parent segment + opens child + records
  `DELEGATION` edge with correct turn indices.
- [ ] 3.2 Failing tests: mention records `MENTION` edge without closing a segment.
- [ ] 3.3 Failing tests: completion closes child segment + records `COMPLETION` edge.
- [ ] 3.4 Failing tests: squad briefing closes a segment.
- [ ] 3.5 Failing tests: leaf run (no communication event) yields one leaf segment with
  `closing_event = None`.
- [ ] 3.6 Failing tests: concurrent fan-out delegation (planner → multiple workers)
  produces multiple `DELEGATION` edges and a deterministic, acyclic segment set.
- [ ] 3.7 Failing tests: `RecordSessionAgentRun` captures `session_id ↔ agent_run_id`.
- [ ] 3.8 Failing tests: a recording error is logged and the run continues (best-effort).
- [ ] 3.9 Implement `InteractionDAGService` (start run, record turn, close segment on
  event, add edge, capture snapshot) behind a feature flag.
- [ ] 3.10 Wire hooks from task/mention/completion/squad-briefing seams to the service for
  trained rollouts only.
- [ ] 3.11 Commit: `feat(interaction-dag): incremental segment + edge recording`.

## 4. Lightweight env snapshot capture (TDD)

**Files:** `internal/service/interaction_dag.go`,
`internal/service/interaction_dag_test.go`.

- [ ] 4.1 Failing tests: `CaptureEnvSnapshot` records `sandbox_ids` (sandbox_instance ids
  per team agent) + `issue_snapshot_id` + minimal `env_state` at the closing event.
- [ ] 4.2 Failing tests: snapshot is ref-only — no sandbox pause/fork is invoked.
- [ ] 4.3 Failing tests: missing snapshot for a segment is detectable (assembler
  dense-coverage path).
- [ ] 4.4 Implement snapshot capture reusing existing sandbox_instance refs; no F
  dependency.
- [ ] 4.5 Commit: `feat(interaction-dag): ref-only team env snapshots`.

## 5. DagResult assembly (TDD)

**Files:** `internal/service/interaction_dag.go`,
`internal/service/interaction_dag_test.go`.

- [ ] 5.1 Failing tests: `AssembleDagResult(project_id)` returns `session_ids`,
  `session_to_agent_run`, `segments`, `edges`, `env_snapshots` exactly matching the AReaL
  `DagResult` shape.
- [ ] 5.2 Failing tests: segments carry 1-based inclusive `start_turn_idx` / `end_turn_idx`
  with terminal = closing-event turn.
- [ ] 5.3 Failing tests: edges carry `src` / `dst` / `type` with `EdgeType` string values
  `delegation` / `mention` / `completion`.
- [ ] 5.4 Failing tests: assembled DAG is acyclic (topological order valid).
- [ ] 5.5 Failing tests: incomplete/failed rollout yields a `failed` status, not a partial
  `DagResult`.
- [ ] 5.6 Implement assembly from recorded rows; round-trip against AReaL
  `SuperNode.from_dict` / `ExecutionDAG.from_records` expectations.
- [ ] 5.7 Commit: `feat(interaction-dag): assemble DagResult for areal`.

## 6. Polling return endpoint (TDD)

**Files:** `internal/handler/env_dispatch.go`, `internal/handler/env_dispatch_test.go`,
router registration.

- [ ] 6.1 Failing tests: `GET /api/v1/env-dispatch/{projectID}/dag` returns `202` + status
  body while the root task is in progress.
- [ ] 6.2 Failing tests: returns `200` + `DagResult` when the root task is done.
- [ ] 6.3 Failing tests: returns `404` for an unknown project_id.
- [ ] 6.4 Failing tests: returns `403` for a cross-workspace caller.
- [ ] 6.5 Failing tests: feature-flag disabled → `503` or documented skip.
- [ ] 6.6 Implement the handler delegating to `InteractionDAGService` status + assembly.
- [ ] 6.7 Commit: `feat(interaction-dag): polling DAG-result endpoint on env-dispatch`.

## 7. AReaL client adapter (TDD)

**Files:** `customized_areal/tree_search/agents/swe_lego_client.py` (or new
`multica_dag_client.py`), tests under `customized_areal/tree_search/tests/`.

- [ ] 7.1 Failing tests: `get_env_dispatch_dag(project_id)` polls until `200` and returns a
  `DagResult` (`session_ids`, `session_to_agent_run`, `segments`, `edges`,
  `env_snapshots`).
- [ ] 7.2 Failing tests: `202` responses are retried with backoff up to a timeout.
- [ ] 7.3 Failing tests: `404` / `403` raise typed errors.
- [ ] 7.4 Failing tests: returned `DagResult` is consumed losslessly by
  `SuperNodeAssembler.assemble` on a synthetic 3-segment planner→worker→synthesizer DAG.
- [ ] 7.5 Implement the adapter (extend `MulticaEnvDispatchClient` or add
  `MulticaDagClient` Protocol + HTTP impl).
- [ ] 7.6 Commit: `feat(areal-client): poll interaction-DAG result from multica`.

## 8. Config + production wiring

- [ ] 8.1 Add `INTERACTION_DAG_ENABLED` (default on for trained rollouts).
- [ ] 8.2 Add polling config on the AReaL side (interval, timeout, backoff).
- [ ] 8.3 Wire the recording service into handler/service construction behind the flag.
- [ ] 8.4 Document the endpoint + polling contract in
  `customized_areal/tree_search/agents/multica_environment_protocol.md`.
- [ ] 8.5 Commit: `chore(interaction-dag): config + production wiring`.

## 9. Full regression + E2E + grep sweep

- [ ] 9.1 Scoped Multica Go build/test/vet for touched packages; `gofmt -l` clean.
- [ ] 9.2 AReaL: `uv run pytest customized_areal/tree_search/tests/ -k 'dag or env_dispatch or supernode'`.
- [ ] 9.3 Cross-repo E2E if feasible: `mode=scratch` rollout with a 3-agent team → record
  segments/edges → poll `GET .../dag` → `SuperNodeAssembler.assemble` reconstructs the DAG
  losslessly.
- [ ] 9.4 Verify env snapshots are refs-only (no sandbox pause/fork) — F-independence.
- [ ] 9.5 grep sweep: `interaction_dag`, `DagResult`, `SegmentSpec`, `EdgeSpec`,
  `env-dispatch/{projectID}/dag` resolve to intended code only.
- [ ] 9.6 Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.
- [ ] 9.7 Commit: `docs(G): T9 full regression + E2E + grep sweep`.

## Test runners / constraints

- Multica Go tests are scoped to touched packages; avoid repo-wide commands if unrelated
  failures remain.
- AReaL tests run from `backend/areal` with
  `uv run pytest customized_areal/tree_search/tests/ -k '<focused expression>'`.
- Cross-repo E2E requires Multica services + the AReaL proxy; if unavailable, document the
  skipped prerequisites.

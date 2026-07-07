# Interaction-DAG Return — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---
change: sub-project-g-multica-interaction-dag
design-doc: docs/superpowers/specs/2026-07-07-interaction-dag-return-design.md
base-ref: 1117158676e77268810fc4f7c5a6652c4be28428
---

**Goal:** Implement the deferred Multica-side producer of the `DagResult` contract: record agent interactions as a communication-bounded segment DAG during task execution and return the assembled `DagResult` to AReaL at task completion via a polling env-dispatch endpoint.

**Architecture:** Multica (Go) records segments + typed edges + per-run turn indices + ref-only env snapshots incrementally at communication events, assembles `DagResult` at root-task completion, and serves it via `GET /api/v1/env-dispatch/{projectID}/dag` (`202`/`200`). AReaL (Python) adds a thin client adapter that polls the endpoint into the existing `DagResult` dataclasses; `SuperNodeAssembler`/`ExecutionDAG` are unchanged consumers.

**Tech Stack:** Go (Multica: service + handler + sqlc-generated queries + migration), Python (AReaL: httpx client adapter + pytest).

**Repos:** This change spans two repos.
- **areal** (`master`, base-ref above): OpenSpec artifacts, design doc, this plan, and the AReaL Python client adapter under `customized_areal/tree_search/agents/`.
- **multica** (`dev`, nested at `multica/`): the Go implementation — service, handler, migration, generated queries. Has its own git history; the bulk of the work lands here.

## Global Constraints

- Produce exactly the `DagResult` AReaL already consumes (`session_ids`, `session_to_agent_run`, `segments: list[SegmentSpec]`, `edges: list[EdgeSpec]`, `env_snapshots: dict[segment_id, TeamEnvSnapshot]`) — no consumer-side changes.
- `EdgeType` values are the strings `delegation` / `mention` / `completion` (match AReaL `EdgeType`).
- Segment turn indices are 1-based inclusive; terminal turn = closing-event turn; dense `[1, len(nodes)]` coverage per run.
- Env snapshots are **ref-only** (sandbox_instance ids + issue-subtree ref); no sandbox pause/fork — independent of Sub-project F.
- Return is **polling**: `202` in-progress / `200`+DagResult done / `404` unknown project / `403` cross-workspace.
- Reuse the existing env-dispatch path; `project_id` is the root handle. `DELETE /env-dispatch/{projectID}` cascades to new tables.
- Recording is trained-rollout-only, gated by `INTERACTION_DAG_ENABLED`, best-effort (errors logged, run continues).
- TDD: failing test → implement → pass → commit, per task.

---

## Task 1: Investigation — interaction seams and turn-index source

**Files (multica):** `internal/service/task.go`, `internal/mention/`, `internal/handler/squad_briefing.go`, `internal/arealrl/client.go`, `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`.

- [ ] 1.1 Confirm the assistant-turn driving seam (proxy `/chat/completions` boundary) and where to maintain the per-`agent_run_id` turn counter. Record the exact call site.
- [ ] 1.2 Confirm delegation seam (`issue.parent_issue_id` creation in `service/task.go`) — the hook point that closes the parent segment + opens a child + records a `DELEGATION` edge.
- [ ] 1.3 Confirm mention seam (`internal/mention`) — hook point for a `MENTION` edge (no segment close).
- [ ] 1.4 Confirm completion/notify-parent seam — hook point that closes the child segment + records a `COMPLETION` edge.
- [ ] 1.5 Confirm squad-briefing seam (`handler/squad_briefing.go`) — hook point that closes a segment.
- [ ] 1.6 Confirm `session_id ↔ agent_run_id` is captured at `/rl/start_session` time in `internal/arealrl/client.go`.
- [ ] 1.7 Confirm the root-task completion signal (agent-run terminal status vs. a rollout-complete marker) that flips the DAG endpoint `202`→`200`.
- [ ] 1.8 Write findings into design.md Open Questions resolution; commit (areal): `docs(G): T1 interaction-DAG seam confirmation`.

## Task 2: Migration + DB queries (multica)

**Files (multica):** new migration under `server/pkg/db/migrations/` (or project migration dir), generated query files under `server/pkg/db/generated/`, `server/pkg/db/queries/*.sql`.

- [ ] 2.1 Write SQL queries: `CreateSegment`, `CloseSegment`, `AddEdge`, `CaptureEnvSnapshot`, `RecordSessionAgentRun`, `SetDagStatus`, `ListSegmentsForProject`, `ListEdgesForProject`, `ListEnvSnapshotsForProject`, `GetDagStatus`.
- [ ] 2.2 Generate sqlc code (`make` / project codegen command).
- [ ] 2.3 Add migration creating `interaction_dag_segment`, `interaction_dag_edge`, `interaction_dag_env_snapshot`, `interaction_dag_session`, `interaction_dag_status` (idempotent `IF NOT EXISTS`).
- [ ] 2.4 Ensure `DELETE /api/v1/env-dispatch/{projectID}` cleanup cascades to the new tables (FK `ON DELETE CASCADE` or explicit delete).
- [ ] 2.5 Verify migration applies + queries compile; commit (multica): `feat(interaction-dag): migration + queries for segments edges snapshots`.

## Task 3: Incremental recording service — TDD (multica)

**Files (multica):** `internal/service/interaction_dag.go`, `internal/service/interaction_dag_test.go`.

**Interfaces:**
- Produces: `InteractionDAGService` with `StartRun`, `RecordTurn`, `CloseSegment`, `AddEdge`, `CaptureEnvSnapshot`, `AssembleDagResult`, `Status`.

- [ ] 3.1 **Red:** `TestInteractionDAG_Delegation_ClosesParentOpensChild` — delegation closes parent segment at the delegating turn, opens child at next turn, records `DELEGATION` edge, turn indices correct.
- [ ] 3.2 **Red:** `TestInteractionDAG_Mention_AddsEdgeNoClose` — `MENTION` edge recorded, no segment closed.
- [ ] 3.3 **Red:** `TestInteractionDAG_Completion_ClosesChild` — `COMPLETION` edge, child segment closed at completing turn.
- [ ] 3.4 **Red:** `TestInteractionDAG_SquadBriefing_ClosesSegment`.
- [ ] 3.5 **Red:** `TestInteractionDAG_LeafRun_OneSegmentNoClosing` — leaf segment `closing_event=nil`, no edges.
- [ ] 3.6 **Red:** `TestInteractionDAG_FanOutDelegation_AcyclicDeterministic` — planner → N workers yields N `DELEGATION` edges, acyclic.
- [ ] 3.7 **Red:** `TestInteractionDAG_RecordSessionAgentRun` — `session_id↔agent_run_id` captured.
- [ ] 3.8 **Red:** `TestInteractionDAG_RecordingError_LogsAndContinues` — best-effort; run not broken.
- [ ] 3.9 **Green:** implement `InteractionDAGService` behind `INTERACTION_DAG_ENABLED`.
- [ ] 3.10 Wire hooks from task/mention/completion/squad-briefing seams to the service for trained rollouts only.
- [ ] 3.11 Run: `cd multica/server && go test ./internal/service/ -run InteractionDAG`; commit (multica): `feat(interaction-dag): incremental segment + edge recording`.

## Task 4: Lightweight env snapshot capture — TDD (multica)

**Files (multica):** `internal/service/interaction_dag.go`, `internal/service/interaction_dag_test.go`.

- [ ] 4.1 **Red:** `TestCaptureEnvSnapshot_RecordsRefsOnly` — `sandbox_ids` (sandbox_instance ids per team agent) + `issue_snapshot_id` + minimal `env_state` at close time.
- [ ] 4.2 **Red:** `TestCaptureEnvSnapshot_NoSandboxPauseOrFork` — no pause/fork/snapshot op invoked (F-independence).
- [ ] 4.3 **Red:** `TestCaptureEnvSnapshot_MissingSnapshotDetectable` — assembler dense-coverage path can detect a missing snapshot.
- [ ] 4.4 **Green:** implement snapshot capture reusing existing sandbox_instance refs.
- [ ] 4.5 Run: `go test ./internal/service/ -run CaptureEnvSnapshot`; commit (multica): `feat(interaction-dag): ref-only team env snapshots`.

## Task 5: DagResult assembly — TDD (multica)

**Files (multica):** `internal/service/interaction_dag.go`, `internal/service/interaction_dag_test.go`.

- [ ] 5.1 **Red:** `TestAssembleDagResult_ShapeMatchesContract` — `session_ids`, `session_to_agent_run`, `segments`, `edges`, `env_snapshots` exactly match the AReaL `DagResult` shape.
- [ ] 5.2 **Red:** `TestAssembleDagResult_TurnRangesOneBasedInclusive` — terminal = closing-event turn.
- [ ] 5.3 **Red:** `TestAssembleDagResult_EdgeTypesMatchEdgeType` — `delegation`/`mention`/`completion`.
- [ ] 5.4 **Red:** `TestAssembleDagResult_Acyclic` — topological order valid.
- [ ] 5.5 **Red:** `TestAssembleDagResult_FailedRollout_NoPartialResult` — `failed` status, not a partial `DagResult`.
- [ ] 5.6 **Green:** implement `AssembleDagResult` reading recorded rows; cross-check JSON against AReaL `SuperNode.from_dict`/`ExecutionDAG.from_records` expectations.
- [ ] 5.7 Run: `go test ./internal/service/ -run AssembleDagResult`; commit (multica): `feat(interaction-dag): assemble DagResult for areal`.

## Task 6: Polling return endpoint — TDD (multica)

**Files (multica):** `internal/handler/env_dispatch.go`, `internal/handler/env_dispatch_test.go`, router registration.

- [ ] 6.1 **Red:** `TestGetDag_InProgress_Returns202` — `202` + status body while root task running.
- [ ] 6.2 **Red:** `TestGetDag_Done_Returns200DagResult`.
- [ ] 6.3 **Red:** `TestGetDag_UnknownProject_Returns404`.
- [ ] 6.4 **Red:** `TestGetDag_CrossWorkspace_Returns403`.
- [ ] 6.5 **Red:** `TestGetDag_Disabled_Returns503`.
- [ ] 6.6 **Green:** implement handler delegating to `InteractionDAGService.Status` + `AssembleDagResult`; register `GET /api/v1/env-dispatch/{projectID}/dag`.
- [ ] 6.7 Run: `go test ./internal/handler/ -run GetDag`; commit (multica): `feat(interaction-dag): polling DAG-result endpoint on env-dispatch`.

## Task 7: AReaL client adapter — TDD (areal)

**Files (areal):** `customized_areal/tree_search/agents/swe_lego_client.py` (extend) or new `multica_dag_client.py`; tests under `customized_areal/tree_search/tests/`.

**Interfaces:**
- Produces: `MulticaDagClient.poll_dag(project_id, *, timeout, interval) -> DagResult` (Protocol + HTTP impl), decoding into the existing `DagResult`/`SegmentSpec`/`EdgeSpec`/`TeamEnvSnapshot` from `supernode_assembler.py`.
- Consumes: the `GET /api/v1/env-dispatch/{projectID}/dag` endpoint (Task 6).

- [ ] 7.1 **Red:** `test_poll_dag_returns_dagresult` — polls until `200`, returns `DagResult`.
- [ ] 7.2 **Red:** `test_poll_dag_retries_202_with_backoff` — `202` retried up to timeout.
- [ ] 7.3 **Red:** `test_poll_dag_404_raises_typed_error` / `test_poll_dag_403_raises_typed_error`.
- [ ] 7.4 **Red:** `test_poll_dag_assembles_losslessly` — returned `DagResult` consumed by `SuperNodeAssembler.assemble` on a synthetic 3-segment planner→worker→synthesizer DAG (incl. fan-in join → `extra_parent_node_ids`).
- [ ] 7.5 **Green:** implement the adapter (extend `MulticaEnvDispatchClient` or add `MulticaDagClient`).
- [ ] 7.6 Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/ -k 'poll_dag or dag_client'`; commit (areal): `feat(areal-client): poll interaction-DAG result from multica`.

## Task 8: Config + production wiring

**Files (multica):** server config, `.env.example`, handler/service construction. **Files (areal):** polling config.

- [ ] 8.1 Add `INTERACTION_DAG_ENABLED` (default on for trained rollouts) in multica.
- [ ] 8.2 Add AReaL polling config (interval, timeout, backoff).
- [ ] 8.3 Wire `InteractionDAGService` into multica handler/service construction behind the flag.
- [ ] 8.4 Document endpoint + polling contract in `customized_areal/tree_search/agents/multica_environment_protocol.md`.
- [ ] 8.5 Commit (multica): `chore(interaction-dag): config + production wiring`; (areal): `docs(interaction-dag): update protocol with polling DAG endpoint`.

## Task 9: Full regression + E2E + grep sweep

- [ ] 9.1 multica: `cd multica/server && go build ./... && go vet ./... && gofmt -l .` clean; scoped `go test ./internal/...`.
- [ ] 9.2 areal: `uv run pytest customized_areal/tree_search/tests/ -k 'dag or env_dispatch or supernode'`.
- [ ] 9.3 Cross-repo E2E if feasible: `mode=scratch` 3-agent team → record → poll `GET .../dag` → `SuperNodeAssembler.assemble` reconstructs losslessly.
- [ ] 9.4 Verify env snapshots refs-only (no pause/fork) — F-independence.
- [ ] 9.5 grep sweep: `interaction_dag`, `DagResult`, `SegmentSpec`, `EdgeSpec`, `env-dispatch/{projectID}/dag` resolve to intended code only.
- [ ] 9.6 Final whole-branch review → READY TO MERGE / NEEDS_CHANGES.
- [ ] 9.7 Commit: `docs(G): T9 full regression + E2E + grep sweep`.

## Test runners / constraints

- multica Go tests scoped to touched packages (`internal/service`, `internal/handler`); avoid repo-wide runs if unrelated failures exist. The multica repo is on `dev` with 5 pre-existing staged files — do not disturb them; commit G work on a feature branch.
- areal tests run from `backend/areal` with `uv run pytest customized_areal/tree_search/tests/ -k '<expr>'`.
- Cross-repo E2E requires Multica services + the AReaL proxy; if unavailable, document skipped prerequisites.
- Two repos ⇒ two merge targets: multica PR targets upstream `dev`; areal MR targets upstream `master`/`dev` per project convention.

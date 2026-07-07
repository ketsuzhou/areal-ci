---
comet_change: sub-project-g-multica-interaction-dag
role: technical-design
canonical_spec: openspec
---

# Multica Interaction-DAG Producer Design

## Context

AReaL already implements the **consumer** of a multi-agent interaction DAG:
`customized_areal/tree_search/agents/execution_dag.py` (`ExecutionDAG`, `SuperNode`,
`EdgeType`) and `supernode_assembler.py` (`SuperNodeAssembler.assemble`). The assembler
consumes a Multica-produced `DagResult`. The contract is locked by
`docs/superpowers/specs/2026-06-30-supernode-multica-dag-rollout-design.md` (decision D14:
"Multica provides segment specs at task completion"), which explicitly defers the
Multica-side producer (§10 Out of scope; risk: "Multica-side endpoints do not exist yet").

Multica already originates every edge source — delegation creates sub-issues
(`issue.parent_issue_id`), mention triggers (`internal/mention`), completion/notify-parent,
squad briefing (`handler/squad_briefing.go`) — and already calls `/rl/start_session` per
agent run (`internal/arealrl/client.go`). This design adds the missing producer: record
those interactions as a segment DAG during execution and return the assembled `DagResult`
to AReaL at task completion.

This change is independent of Sub-project F (env checkpointing): env snapshots are
reference-only.

## The Contract (unchanged AReaL consumer)

Multica must produce a `DagResult` exactly matching what `SuperNodeAssembler` consumes:

```
DagResult:
  session_ids: list[str]                         # one per agent run
  session_to_agent_run: dict[str, str]           # session_id -> agent_run_id
  segments: list[SegmentSpec]
  edges: list[EdgeSpec]
  env_snapshots: dict[segment_id, TeamEnvSnapshot]

SegmentSpec:
  segment_id, agent_run_id, issue_id, task_id
  closing_event: "delegation" | "mention" | "completion" | None
  closing_event_target_segment: str | None
  start_turn_idx: int   # 1-based inclusive
  end_turn_idx: int     # 1-based inclusive (the closing-event turn)

EdgeSpec:
  src_segment_id, dst_segment_id
  type: "delegation" | "mention" | "completion"

TeamEnvSnapshot:
  sandbox_ids: list[str]          # sandbox_instance ids, one per team agent
  issue_snapshot_id: str | None
  env_state: dict
```

AReaL maps each agent's `list[Node]` into segments by the turn range, sets the unified
`parent_node_id` chain across agents/segments, stamps the env snapshot, and builds the
`ExecutionDAG`. The assembler validates: dense non-overlapping turn coverage per run,
symmetric edges, topological (acyclic) invariant, and exactly one DAG sink. Multica's
producer must satisfy all of these.

## Architecture

```
create_env_dispatch(mode=scratch)  ── creates project/issue/agent_run, calls /rl/start_session
        │
        ▼
InteractionDAGService (per trained rollout, keyed by project_id)
        │  records as communication events fire:
        │
        ├── delegation (parent issue → sub-issue)
        │     close parent run's current segment at the delegating turn
        │     open child segment for the new run
        │     add DELEGATION edge (parent_segment → child_segment)
        │     capture TeamEnvSnapshot for the parent segment
        │
        ├── mention (peer trigger)
        │     add MENTION edge (no segment close)
        │
        ├── completion (child → parent notify)
        │     close child run's current segment at the completing turn
        │     add COMPLETION edge (child_segment → parent continuation segment)
        │     capture TeamEnvSnapshot for the child segment
        │
        └── squad briefing
              close the briefing run's current segment
              capture TeamEnvSnapshot
        │
        ▼
root task completes (success | failed)
        │
        ▼
AssembleDagResult(project_id)  ── reads recorded segments/edges/snapshots/session map
        │
        ▼
GET /api/v1/env-dispatch/{projectID}/dag
   202 + status  (in progress)
   200 + DagResult (done)
        │
        ▼
AReaL MulticaDagClient.poll_dag(project_id) -> DagResult
        │
        ▼
SuperNodeAssembler.assemble(sessions_nodes, dag_result)  [unchanged consumer]
```

## Components (Multica Go)

### InteractionDAGService

`internal/service/interaction_dag.go`. Per-rollout recorder, keyed by `project_id`. API:

- `StartRun(project_id, agent_run_id, session_id)` — record `session_to_agent_run`, open
  the run's first segment at `start_turn_idx=1`.
- `RecordTurn(agent_run_id)` — increment the per-run turn counter.
- `CloseSegment(agent_run_id, closing_event, target_segment_id)` — stamp `end_turn_idx`
  on the run's open segment, set `closing_event`/`closing_event_target_segment`, capture
  the env snapshot, and open the next segment starting at the next turn.
- `AddEdge(src_segment_id, dst_segment_id, type)` — idempotent typed edge.
- `CaptureEnvSnapshot(segment_id)` — ref-only: current sandbox_instance ids per team agent
  + issue-subtree ref + minimal env_state. No sandbox pause/fork.
- `AssembleDagResult(project_id)` — read all recorded rows; emit `DagResult`.
- `Status(project_id)` — `in_progress` | `done` | `failed`.

All recording is best-effort: a recording error is logged and the run continues; a
`DagResult` with a missing segment is detectable by AReaL's assembler dense-coverage check.

### Turn-index tracking

Multica drives each agent turn through the AReaL proxy (`/chat/completions` per assistant
turn). One assistant turn = one turn. The service increments a per-`agent_run_id` counter
at the driving boundary. The closing communication event's turn index becomes the
segment's `end_turn_idx`. Because AReaL's proxy interaction cache orders interactions by
execution order, AReaL's `list[Node]` 1-based positions align with these indices (the
assembler validates and raises `DAGError` on any mismatch).

### Event hooks

Wire the existing seams to the service for **trained rollouts only** (gated by
`INTERACTION_DAG_ENABLED` + rollout training flag):

- delegation: `internal/service/task.go` sub-issue creation → `CloseSegment` (parent) +
  `AddEdge(DELEGATION)` + open child segment.
- mention: `internal/mention` → `AddEdge(MENTION)` (no close).
- completion: task completion/notify-parent → `CloseSegment` (child) +
  `AddEdge(COMPLETION)`.
- squad briefing: `internal/handler/squad_briefing.go` → `CloseSegment` + snapshot.

Non-trained rollouts, claiming, cascade cancellation, sweeper timeouts, autopilot retry,
and sandbox-lifecycle events do not record (no policy signal).

### Polling endpoint

`GET /api/v1/env-dispatch/{projectID}/dag` in `internal/handler/env_dispatch.go`:

- `202` + `{"status": "in_progress"}` while the root task is running.
- `200` + `DagResult` when the root task is done (success or failed).
- `404` for an unknown project_id.
- `403` for a cross-workspace caller (reuse existing env-dispatch authorization).
- `503` (or documented skip) when the feature flag is disabled.

`DELETE /api/v1/env-dispatch/{projectID}` cascades to the new segment/edge/snapshot tables
so cleanup stays consistent.

## Data Model (migration)

Additive tables (idempotent `IF NOT EXISTS`):

- `interaction_dag_segment`: `segment_id`, `project_id`, `agent_run_id`, `issue_id`,
  `task_id`, `start_turn_idx`, `end_turn_idx`, `closing_event`, `closing_event_target`,
  `created_at`.
- `interaction_dag_edge`: `src_segment_id`, `dst_segment_id`, `type`.
- `interaction_dag_env_snapshot`: `segment_id`, `sandbox_ids` (jsonb),
  `issue_snapshot_id`, `env_state` (jsonb).
- `interaction_dag_session`: `project_id`, `session_id`, `agent_run_id` (for
  `session_to_agent_run`).
- `interaction_dag_status`: `project_id`, `status` (`in_progress`/`done`/`failed`).

Generated DB query files: `CreateSegment`, `CloseSegment`, `AddEdge`,
`CaptureEnvSnapshot`, `RecordSessionAgentRun`, `SetDagStatus`, `ListSegmentsForProject`,
`ListEdgesForProject`, `ListEnvSnapshotsForProject`, `GetDagStatus`.

## AReaL Client Adapter (Python)

Extend `customized_areal/tree_search/agents/swe_lego_client.py` (`MulticaEnvDispatchClient`)
or add a thin `multica_dag_client.py` exposing a `MulticaDagClient` Protocol:

- `poll_dag(project_id, *, timeout, interval) -> DagResult` — poll
  `GET /api/v1/env-dispatch/{projectID}/dag`; retry `202` with backoff; surface `404`/`403`
  as typed errors; on `200` decode into the existing `DagResult`/`SegmentSpec`/`EdgeSpec`/
  `TeamEnvSnapshot` dataclasses (already defined in `supernode_assembler.py`).

`SuperNodeAssembler` and `ExecutionDAG` are unchanged. The 2026-06-30 design's
`TeamRolloutCoordinator.collect_result(root_task_id)` maps to
`poll_dag(project_id)` once env-dispatch is the entry point.

## Testing Strategy

- **Multica Go (TDD):**
  - Delegation closes parent segment + opens child + DELEGATION edge + correct turn
    indices.
  - Mention adds MENTION edge without closing.
  - Completion closes child + COMPLETION edge.
  - Squad briefing closes a segment.
  - Leaf run: one segment, `closing_event=None`, no edges.
  - Concurrent fan-out: multiple DELEGATION edges, deterministic, acyclic.
  - `RecordSessionAgentRun` captures `session_id↔agent_run_id`.
  - Recording error is logged and the run continues.
  - `AssembleDagResult` emits the exact `DagResult` shape; acyclic; failed rollout →
    `failed` status (no partial result).
  - Endpoint: `202` in-progress, `200`+DagResult done, `404`, `403`, disabled→`503`.
- **AReaL Python (TDD):**
  - `poll_dag` returns `DagResult`; `202` retried with backoff; `404`/`403` typed.
  - Returned `DagResult` consumed losslessly by `SuperNodeAssembler.assemble` on a
    synthetic 3-segment planner→worker→synthesizer DAG (including a fan-in join →
    `extra_parent_node_ids`).
- **E2E (if feasible):** `mode=scratch` 3-agent team → record → poll `GET .../dag` →
  assemble losslessly; verify env snapshots are refs-only (no sandbox pause/fork).

## Risks and Trade-offs

- **Turn-index drift** → assembler validates ranges and raises `DAGError`; Multica counts
  exactly one turn per assistant completion it drives; boundary tests.
- **Fan-out ordering** → deterministic edge recording; acyclic by parent→child flow.
- **Polling cost** → cheap status read until completion; acceptable vs held connections.
- **Ref-only snapshot fidelity** → documented v1 limitation; F enriches later; assembler
  stamps without interpreting.
- **Recording failures** → best-effort; logged; missing segments detectable by assembler.
- **Coupled to project cleanup** → `DELETE /env-dispatch/{projectID}` cascades to new
  tables; read-only derived result.

## Out of Scope

- Full sandbox snapshot/fork in env snapshots (F).
- Verifier, reward backup, GAE/critic (E / 2026-06-30 Phase 2).
- Immutable branching / fork-from-checkpoint (2026-06-26 phase1).
- Any change to `SuperNodeAssembler` / `ExecutionDAG` / the AReaL proxy interaction cache.

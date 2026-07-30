# Comet Design Handoff

- Change: sub-project-g-multica-interaction-dag
- Phase: design
- Mode: compact
- Context hash: 056ceaab30dd8f5d0665a883a8b485c870e4656cb45688a20d92431db3280093

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/sub-project-g-multica-interaction-dag/proposal.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/proposal.md
- Lines: 1-66
- SHA256: 58cf502f13ea25dac09d42cdb21bbb4ed5735bafc656d3d4175e729c3e55a6bd

```md
## Why

AReaL already designs and implements the **consumer** side of a multi-agent interaction
DAG (`ExecutionDAG` / `SuperNode` / `SuperNodeAssembler` in
`customized_areal/tree_search/agents/`), and
`docs/superpowers/specs/2026-06-30-supernode-multica-dag-rollout-design.md` locks the
`DagResult` contract Multica must produce at task completion (decision D14: "Multica
provides segment specs at task completion"). But the **Multica-side producer is explicitly
deferred and unimplemented** (design §10 Out of scope; risk row: "Multica-side endpoints do
not exist yet"). Without it, AReaL cannot run collaborative multi-agent DAG rollouts —
there is no source for the segments, typed edges, per-run turn indices, and env snapshots
the assembler consumes. This change implements that deferred producer: Multica converts its
own agent interactions into the DAG during execution and returns it to AReaL when the task
finishes.

## What Changes

- **Incremental interaction-DAG recording (Multica Go)**: as communication events fire
  (delegation → sub-issue, mention, completion → notify-parent, squad briefing), Multica
  records communication-bounded segments with per-run turn indices and typed edges
  (`DELEGATION` / `MENTION` / `COMPLETION`), matching AReaL's `EdgeType`. Segments are
  recorded during execution, not inferred afterward.
- **Lightweight team env snapshots per segment**: each segment records a ref-only
  `TeamEnvSnapshot` (sandbox_instance ids + issue-subtree ref). No sandbox pause/fork —
  deliberately independent of Sub-project F.
- **`DagResult` assembly + polling return endpoint**: at root-task completion Multica
  assembles `DagResult` (`session_ids`, `session_to_agent_run`, `segments`, `edges`,
  `env_snapshots`) and serves it via `GET /api/v1/env-dispatch/{projectID}/dag` — `202`
  in-progress, `200` + `DagResult` when done. Reuses the existing env-dispatch path
  (`project_id` as the root handle).
- **AReaL client adapter**: extend `MulticaEnvDispatchClient` (or add a thin
  `MulticaDagClient`) to poll the endpoint and return `DagResult`. `SuperNodeAssembler` and
  `ExecutionDAG` are unchanged consumers.
- **Migration**: new tables for interaction-DAG segments, edges, and per-segment env
  snapshots, keyed by `project_id` / `agent_run_id`.

## Capabilities

### New Capabilities

- `interaction-dag-return`: Multica records agent interactions as a communication-bounded
  segment DAG during task execution and returns the assembled `DagResult` to AReaL at task
  completion via a polling env-dispatch endpoint.

### Modified Capabilities

<!-- None. AReaL's SuperNodeAssembler / ExecutionDAG consumer is unchanged; this change
     only adds the Multica producer plus a thin AReaL client adapter. -->

## Impact

- **Multica (Go, primary)**: new service (`internal/service/interaction_dag.go`), handler
  (`GET .../env-dispatch/{projectID}/dag`), per-run turn-index tracking at the agent-run
  driving seam, env-snapshot capture at communication events, migration + generated DB
  queries. Touches `internal/service/task.go` (delegation/mention/completion hooks),
  `internal/handler/env_dispatch.go`, `internal/handler/squad_briefing.go`,
  `internal/arealrl/client.go` (session↔agent_run mapping).
- **AReaL (Python, secondary)**: thin client adapter in
  `customized_areal/tree_search/agents/` (poll endpoint → `DagResult`);
  `SuperNodeAssembler` / `ExecutionDAG` unchanged.
- **Depends on**: the locked `DagResult` / `SegmentSpec` / `EdgeSpec` / `TeamEnvSnapshot`
  contract from `2026-06-30-supernode-multica-dag-rollout-design.md`. Independent of
  Sub-project F (env snapshots are refs-only; no sandbox pause/fork).
- **Out of scope**: full sandbox snapshot/fork (F), verifier / reward backup
  (E / 2026-06-30 Phase 2), immutable branching (2026-06-26 phase1), AReaL consumer
  changes.
```

## openspec/changes/sub-project-g-multica-interaction-dag/design.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/design.md
- Lines: 1-189
- SHA256: 8c589558c6562a7d8d627e1801f28597a029244f386d541efb04cb7aa1fafe2f

[TRUNCATED]

```md
## Context

`customized_areal/tree_search/agents/` already implements the AReaL **consumer** of a
multi-agent interaction DAG:

- `execution_dag.py` — torch-free `ExecutionDAG` of `SuperNode`s with typed `EdgeType`
  (`DELEGATION` / `MENTION` / `COMPLETION`) and `to_records()` / `from_records()`.
- `supernode_assembler.py` — `SuperNodeAssembler.assemble(sessions_nodes, dag_result)`
  consumes a Multica-produced `DagResult`: `session_ids`, `session_to_agent_run`,
  `segments: list[SegmentSpec]` (1-based inclusive `start_turn_idx` / `end_turn_idx` +
  `closing_event` + `closing_event_target_segment`), `edges: list[EdgeSpec]`, and
  `env_snapshots: dict[segment_id, TeamEnvSnapshot]`.
- `multica_environment_protocol.md` + `2026-06-30-supernode-multica-dag-rollout-design.md`
  lock the contract (D14: "Multica provides segment specs at task completion") and
  explicitly defer the Multica-side producer (§10 Out of scope; risk: "Multica-side
  endpoints do not exist yet").

Multica already originates every edge source — delegation creates sub-issues
(`issue.parent_issue_id`), mention triggers (`internal/mention`), completion/notify-parent,
squad briefing (`handler/squad_briefing.go`) — and already calls `/rl/start_session` per
agent run (`internal/arealrl/client.go`). What is missing is recording these as
segment + edge structure with per-run turn indices during execution and returning the
assembled `DagResult` to AReaL at task completion.

Constraints: this change must produce exactly the `DagResult` AReaL already consumes (no
consumer-side changes), must reuse the existing env-dispatch path, must use a polling
return, and must stay independent of Sub-project F (env snapshots are refs-only).

## Goals / Non-Goals

**Goals:**

- Implement the deferred Multica-side producer of the locked `DagResult` contract.
- Incrementally record communication-bounded segments + typed edges + per-run turn indices
  as communication events fire during execution.
- Capture lightweight (ref-only) team env snapshots per segment.
- Assemble and return `DagResult` at root-task completion via a polling env-dispatch
  endpoint.
- Add a thin AReaL client adapter to poll the endpoint; leave `SuperNodeAssembler` /
  `ExecutionDAG` unchanged.

**Non-Goals:**

- Full sandbox snapshot/fork in env snapshots (deferred to F's pause-in-place checkpoints).
- Verifier, reward backup, GAE/critic (E / 2026-06-30 Phase 2).
- Immutable branching / fork-from-checkpoint (2026-06-26 phase1).
- Any change to `SuperNodeAssembler` / `ExecutionDAG` / the per-turn proxy interaction
  cache on the AReaL side.

## Decisions

### D1 — Incremental segment recording at communication events

Record segments + edges as communication events fire during execution, not reconstruct them
at completion.

**Rationale:** `SegmentSpec` requires 1-based inclusive `start_turn_idx` / `end_turn_idx`
whose terminal turn is the closing communication event; turn indices can only be captured
as the agent runs. The 2026-06-30 design states "Multica decides segment boundaries at
communication events during execution" and "Multica tracks turn indices as the agent runs."

**Alternative considered:** reconstruct the DAG at completion by querying sub-issues /
mentions / completions — rejected because per-run turn indices and exact segment-close
ordering are not recoverable after the fact, and the contract demands turn ranges.

### D2 — Reuse the env-dispatch path + `GET /api/v1/env-dispatch/{projectID}/dag`

Reuse `create_env_dispatch(mode="scratch")` to start the rollout (it already creates
project / issue / agent_run and calls `/rl/start_session`), and add a polling
`GET /api/v1/env-dispatch/{projectID}/dag` that returns `202` in-progress or
`200` + `DagResult` when the root task completes. `project_id` is the root handle.

**Rationale:** env-dispatch already owns the project/issue/agent_run/session machinery;
adding a result endpoint avoids a parallel `submit_root_task` / `collect_result`
orchestration path and fits the existing protocol.

**Alternative considered:** a separate `MulticaDagClient` with `submit_root_task` /
`collect_result` endpoints (assumed by the 2026-06-30 design, precedent
`/api/v1/swe-lego/issues`) — rejected for v1 to avoid duplicating orchestration; the AReaL
client can still expose a `MulticaDagClient` Protocol as a thin adapter over
```

Full source: openspec/changes/sub-project-g-multica-interaction-dag/design.md

## openspec/changes/sub-project-g-multica-interaction-dag/tasks.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/tasks.md
- Lines: 1-146
- SHA256: 555fbb3db6570a2463c76df6d36d94f2780f895997be3ed8cac35d9b0ccb1272

[TRUNCATED]

```md
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
```

Full source: openspec/changes/sub-project-g-multica-interaction-dag/tasks.md

## openspec/changes/sub-project-g-multica-interaction-dag/specs/interaction-dag-return/spec.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/specs/interaction-dag-return/spec.md
- Lines: 1-72
- SHA256: 8847a9201d6a1e86478719511fce6db3e3936f04566eaa829085726bbcb118a4

```md
## ADDED Requirements

### Requirement: Incremental interaction-DAG recording at communication events
The system SHALL record agent interactions as a communication-bounded segment DAG during task execution. A segment MUST span the turns since the previous communication event up to and including the closing-event turn, with 1-based inclusive `start_turn_idx` / `end_turn_idx` whose terminal turn is the closing communication event. Typed edges MUST match AReaL's `EdgeType`: `DELEGATION` (parent-issue run → child sub-issue run), `MENTION` (peer trigger, topology-only), and `COMPLETION` (child run → parent run continuation). Recording MUST happen during execution, not reconstructed after completion.

#### Scenario: Delegation closes parent segment and opens child
- **WHEN** a trained agent run delegates by creating a sub-issue
- **THEN** the system closes the delegating run's current segment at the delegating turn, opens a child segment for the new run, and records a `DELEGATION` edge from the parent segment to the child segment

#### Scenario: Mention records a peer edge without closing a segment
- **WHEN** a trained agent run mentions or triggers another agent
- **THEN** the system records a `MENTION` edge between the runs without closing the mentioning run's current segment

#### Scenario: Completion closes child segment and links to parent
- **WHEN** a child run completes and notifies its parent
- **THEN** the system closes the child run's current segment at the completing turn and records a `COMPLETION` edge from the child segment to the parent's continuation segment

#### Scenario: Leaf run has no closing event
- **WHEN** a trained run ends with no communication event
- **THEN** the system records one leaf segment with `closing_event` set to null and no outgoing edges

#### Scenario: Recorded DAG is acyclic
- **WHEN** the system records segments and edges
- **THEN** the resulting directed graph MUST be acyclic, with every edge flowing from a cause segment to an effect segment in topological order

### Requirement: Per-run turn-index tracking
The system SHALL maintain a per-`agent_run_id` turn counter as it drives each assistant turn through the inference proxy. One assistant turn MUST equal one turn. The closing communication event's turn index MUST be stamped as the segment's `end_turn_idx`.

#### Scenario: Turn indices align with areal node positions
- **WHEN** Multica records a segment's `start_turn_idx` / `end_turn_idx`
- **THEN** the indices MUST densely cover `[1, len(nodes)]` for that run with no gaps or overlaps, matching the 1-based positions of the `list[Node]` AReaL builds from the proxy interaction cache

#### Scenario: Concurrent fan-out delegation stays acyclic
- **WHEN** a planner delegates to multiple workers at once
- **THEN** the system records multiple `DELEGATION` edges from the planner's closing segment to each child segment deterministically, and the DAG remains acyclic

### Requirement: Lightweight ref-only team env snapshots
The system SHALL capture a `TeamEnvSnapshot` per segment at the closing communication event containing `sandbox_ids` (current sandbox_instance ids per team agent), `issue_snapshot_id` (issue-subtree ref), and minimal `env_state`. The snapshot MUST be reference-only: the system MUST NOT pause, snapshot, or fork any sandbox to produce it.

#### Scenario: Snapshot records refs only
- **WHEN** a segment's closing event fires
- **THEN** the system records the team's current sandbox_instance ids and issue-subtree ref without invoking any sandbox pause or fork operation

#### Scenario: Independent of environment checkpointing
- **WHEN** Sub-project F environment checkpointing is not available
- **THEN** the system still captures ref-only team env snapshots for every segment

### Requirement: Polling DagResult return at task completion
The system SHALL expose `GET /api/v1/env-dispatch/{projectID}/dag` that returns `202` with a status body while the root task is in progress and `200` with an assembled `DagResult` when the root task completes (success or failed). The `DagResult` MUST contain `session_ids`, `session_to_agent_run`, `segments`, `edges`, and `env_snapshots` matching the contract AReaL's `SuperNodeAssembler` consumes.

#### Scenario: In-progress poll
- **WHEN** AReaL polls the endpoint before the root task completes
- **THEN** the system returns `202` with a status body indicating the task is in progress

#### Scenario: Completed poll returns DagResult
- **WHEN** the root task has completed
- **THEN** the system returns `200` with a `DagResult` that `SuperNodeAssembler.assemble` reconstructs losslessly into the `ExecutionDAG`

#### Scenario: Unknown project rejected
- **WHEN** a caller polls a project_id that does not exist
- **THEN** the system returns `404`

#### Scenario: Cross-workspace access rejected
- **WHEN** a caller polls a project_id outside its workspace
- **THEN** the system returns `403`

### Requirement: Session-to-agent-run mapping
The system SHALL record the `session_id` to `agent_run_id` mapping at `/rl/start_session` time for every agent run in a trained rollout and include it in the returned `DagResult`.

#### Scenario: Mapping captured at session start
- **WHEN** Multica starts an RL session for an agent run
- **THEN** the system records the `session_id ↔ agent_run_id` pair and emits it in `session_to_agent_run` of the `DagResult`
```


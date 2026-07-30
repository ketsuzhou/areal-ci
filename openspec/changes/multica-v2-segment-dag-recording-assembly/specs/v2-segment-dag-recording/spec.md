# v2-segment-dag-recording

## ADDED Requirements

### Requirement: Trained-rollout segment recording gating

Multica SHALL record interaction-DAG segments only for trained rollouts (a `TrainAgentID` is
set on the env dispatch) and only while `INTERACTION_DAG_ENABLED` is on. For each communication
event (delegation, mention, completion, squad briefing) on a trained rollout, Multica SHALL
call `close_segment` + per-segment export on the producer/parent run whose session has emitted
the handoff/completion, then record a segment carrying `trajectory_id`, `tensor_ref`,
`closing_event`, and a ref-only `env_snapshot`. For squad-context handoff, the edge remains a
`delegation` edge while the producer segment records `closing_event = "squad_briefing"`; Multica
MUST NOT close the receiver/child session before it has emitted its own model turn. A leaf run with
no communication event SHALL yield exactly one leaf segment with `closing_event = None`.
Non-trained rollouts SHALL record nothing and SHALL incur no recording overhead.

#### Scenario: Trained rollout records a segment per communication event
- **WHEN** a trained rollout fires a delegation, mention, completion, or squad-briefing event
- **THEN** Multica calls `close_segment` + export on the producer session and records a segment
  with `trajectory_id` + `tensor_ref` + `closing_event` + `env_snapshot` for that event
- **AND** squad-context handoff records `closing_event = "squad_briefing"` on the producer
  segment while preserving the structural edge type as `delegation`

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
projection of recorded rows.

#### Scenario: Assembly projects recorded rows into AssembledDag
- **WHEN** Multica assembles the DAG for a completed root task
- **THEN** the returned `AssembledDag` contains the segments, typed edges, and
  `session_to_agent_run` recorded for that project, with no scores, turn indices, or text

#### Scenario: Edges are typed and the graph is acyclic
- **WHEN** edges are assembled
- **THEN** every edge type is `delegation`, `mention`, or `completion`, each flows from a cause
  segment to an effect segment, and the directed graph is acyclic

### Requirement: Multi-shard tensor-ref contract

Each segment's `tensor_ref` SHALL be a map from tensor field name (e.g. `input_ids`,
`attention_mask`, `logprobs`, `loss_mask`, `versions`) to that field's shard reference
(`shard_id` + `node_addr`) as produced by the v2 `/export_trajectories` per-field
`RTensor.remotize`. `tensor_ref` SHALL NOT be a single `shard_id`. Multica SHALL decode this map
from the export response; AReaL SHALL resolve each field's shard via `/data/<shard_id>` and
reassemble the field-to-tensor dict for the segment.

#### Scenario: tensor_ref is a field-to-shard map
- **WHEN** Multica records a segment from a real export response
- **THEN** the segment's `tensor_ref` is a map of tensor field name to shard reference, one
  entry per tensor field the export produced, and is not a single `shard_id`

#### Scenario: Resolver fetches all shards and reassembles
- **WHEN** AReaL resolves a segment's `tensor_ref`
- **THEN** it fetches each field's shard via `/data/<shard_id>` and reassembles the
  field-to-tensor dict for that segment

#### Scenario: Missing shard is surfaced, not masked
- **WHEN** a field's shard reference lacks a `shard_id`
- **THEN** the resolver raises a typed error for that segment and does not substitute a default

### Requirement: /dag endpoint cross-workspace rejection

`GET /api/v1/env-dispatch/{projectID}/dag` SHALL return `403` when the caller requests a
`project_id` outside their workspace, distinct from `404` for an unknown project.

#### Scenario: Cross-workspace request is rejected
- **WHEN** a caller polls `/dag` for a `project_id` in another workspace
- **THEN** the endpoint returns `403`

### Requirement: /dag endpoint failed-rollout status

The `/dag` endpoint SHALL return `200` with a `failed` status body and no `AssembledDag`
when the root task's rollout is incomplete or failed such that recorded segments do not densely
cover the run. A densely-covered run (success or failed terminal state) SHALL return `200` +
`AssembledDag`.

#### Scenario: Incomplete rollout yields a failed status, not a partial DAG
- **WHEN** a rollout is incomplete (missing segments or unrecorded events)
- **THEN** the endpoint returns `200` with a `failed` status and no `AssembledDag`

#### Scenario: Densely-covered failed run returns its AssembledDag
- **WHEN** a failed root task's segments densely cover the run
- **THEN** the endpoint returns `200` with the assembled `AssembledDag`

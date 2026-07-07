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

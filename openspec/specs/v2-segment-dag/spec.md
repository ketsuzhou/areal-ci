# v2-segment-dag

## ADDED Requirements

### Requirement: V2 no-reward segment close

The v2 inference_service SHALL provide a `close_segment` operation (`POST /rl/close_segment`)
that snapshots a session's active completions into a ready trajectory **without setting a
reward**, returning the new `trajectory_id`. Closing a segment MUST NOT require a prior
`set_reward` and MUST NOT assign any reward value. The session MUST remain live for further
turns and further `close_segment` calls after a close.

#### Scenario: Close produces a reward-less trajectory
- **WHEN** Multica calls `close_segment` on a session that has active completions
- **THEN** the system moves the active completions into a ready trajectory, returns its
  `trajectory_id`, and assigns no reward to any interaction in that trajectory

#### Scenario: Session stays live after close
- **WHEN** a segment is closed on a session
- **THEN** subsequent `/chat/completions` turns on the same session are captured into a new
  active segment that can be closed independently

#### Scenario: Close without active completions is rejected
- **WHEN** `close_segment` is called on a session with no active completions
- **THEN** the system returns a typed error and produces no trajectory

### Requirement: Per-segment tensor-ref export

The v2 inference_service SHALL export a single ready trajectory by `trajectory_id` via
`/export_trajectories`, storing its tokens/logprobs on the data_proxy and returning `RTensor`
references (not raw bytes). Export with `remove_session=False` MUST keep the session and its
remaining trajectories intact for further segment exports. Export MUST NOT include message
text - only tensor data (`input_ids`, `loss_mask`, `logprobs`, `versions`,
`attention_mask`).

#### Scenario: Export returns refs for one segment
- **WHEN** Multica exports a single `trajectory_id` with `remove_session=False`
- **THEN** the system returns `RTensor` refs for that segment's tensors and leaves the
  session's other trajectories ready for later export

#### Scenario: Export is refs-only, no text
- **WHEN** a trajectory is exported
- **THEN** the returned payload contains tensor refs and no message text

#### Scenario: Exporting an unknown trajectory fails
- **WHEN** Multica exports a `trajectory_id` not present in the session's ready trajectories
- **THEN** the system returns a typed error and exports nothing

### Requirement: AssembledDag contract carries structure, refs, and env only

Multica SHALL assemble an `AssembledDag` at root-task completion containing `segments`,
`edges`, and `session_to_agent_run`. Each segment MUST carry `segment_id`, `agent_run_id`,
`issue_id`, `trajectory_id`, `tensor_ref`, `closing_event`, and a ref-only `env_snapshot`.
Edges MUST carry `src_segment_id`, `dst_segment_id`, and `type` matching AReaL's `EdgeType`
(`delegation` fan-out / `mention` peer / `completion` fan-in). `AssembledDag` MUST NOT carry
scores, turn indices, or message text.

#### Scenario: Segment records refs and structure, not scores or turn indices
- **WHEN** Multica records a segment at a communication event
- **THEN** the segment carries `trajectory_id` + `tensor_ref` + `closing_event` +
  `env_snapshot` and carries no `judge_scores` and no `start_turn_idx` / `end_turn_idx`

#### Scenario: Edges match EdgeType and keep the DAG acyclic
- **WHEN** Multica assembles edges
- **THEN** each edge type is `delegation`, `mention`, or `completion`, every edge flows from
  a cause segment to an effect segment, and the resulting directed graph is acyclic

#### Scenario: AssembledDag contains no text
- **WHEN** `AssembledDag` is assembled
- **THEN** no segment or edge carries message text

### Requirement: AReaL resolves refs and builds the DAG

AReaL SHALL resolve each segment's `tensor_ref` to tokens/logprobs via the v2 data_proxy
(`/data/<shard_id>`, `/data/batch`), build a `SuperNode` whose payload is the resolved
tensors and whose metadata is the `AssembledDag` segment, and construct the `ExecutionDAG`
from the assembled edges - by ref-resolution, not by slicing a node list on turn indices.
The assembler MUST validate that the assembled DAG is acyclic and that each session's
segments cover the run without gaps.

#### Scenario: Refs resolve to tensors and build a SuperNode
- **WHEN** AReaL consumes an `AssembledDag`
- **THEN** each segment's `tensor_ref` resolves to its tokens/logprobs and a `SuperNode` is
  built with that tensor payload and the segment metadata

#### Scenario: Edges build an acyclic ExecutionDAG
- **WHEN** AReaL builds the `ExecutionDAG` from assembled edges
- **THEN** the graph is acyclic and reconstructs the delegation / mention / completion
  topology Multica recorded

#### Scenario: Missing segment is detected
- **WHEN** a session's segments do not densely cover the run (gap or missing trajectory)
- **THEN** the assembler raises a typed `DAGError` and does not produce a partial DAG

### Requirement: Tensor lifecycle cleanup

After training consumes a segment's tensors, AReaL SHALL release the v2 storage by calling
`DELETE /data/clear` for the resolved shards and `remove_session` (or session revoke) once
the session's last segment is consumed. Refs MUST NOT be cleared before training resolves
them.

#### Scenario: Refs freed after training
- **WHEN** training has consumed a segment's tensors
- **THEN** AReaL clears the corresponding data_proxy shards

#### Scenario: Session released after last segment
- **WHEN** the session's last segment is consumed and cleared
- **THEN** AReaL revokes the session, releasing v2 session state

### Requirement: Polling AssembledDag return at task completion

Multica SHALL expose `GET /api/v1/env-dispatch/{projectID}/dag` that returns `202` with a
status body while the root task is in progress and `200` with the assembled `AssembledDag`
when the root task completes (success or failed). The endpoint reuses the existing
env-dispatch path with `project_id` as the root handle.

#### Scenario: In-progress poll
- **WHEN** AReaL polls the endpoint before the root task completes
- **THEN** the system returns `202` with a status body indicating the task is in progress

#### Scenario: Completed poll returns AssembledDag
- **WHEN** the root task has completed
- **THEN** the system returns `200` with an `AssembledDag` that AReaL resolves and
  reconstructs losslessly into the `ExecutionDAG`

#### Scenario: Unknown project rejected
- **WHEN** a caller polls a `project_id` that does not exist
- **THEN** the system returns `404`

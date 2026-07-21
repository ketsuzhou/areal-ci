## MODIFIED Requirements

### Requirement: AssembledDag contract carries structure, refs, and env only

Multica SHALL assemble an `AssembledDag` at root-task completion containing
`segments`, `edges`, and `session_to_agent_run`. Each segment MUST carry
`segment_id`, `agent_run_id`, `issue_id`, `closing_event`, a ref-only
`env_snapshot`, and an explicit `trajectory_source` (`areal_tensor` or
`task_messages`), `trainable` boolean, and `trajectory` list. A trainable
`areal_tensor` segment MUST additionally carry `trajectory_id` and `tensor_ref`
and its `trajectory` MUST be empty. A non-trainable `task_messages` segment MUST
have null `trajectory_id` and `tensor_ref` and its `trajectory` MUST be the
allowlisted message-event snapshot for its sequence range. Edges MUST carry
`src_segment_id`, `dst_segment_id`, and `type` matching AReaL's `EdgeType`
(`delegation` fan-out / `mention` peer / `completion` fan-in). `AssembledDag` MUST
NOT carry scores, turn indices, or provider API keys; only non-trainable
`task_messages` segments carry message-derived trajectory content.

#### Scenario: Trainable segment records refs and structure, not text

- **WHEN** Multica records a trainable segment at a communication event
- **THEN** the segment carries `trajectory_id` + `tensor_ref` + `closing_event` +
  `env_snapshot` with `trajectory_source=areal_tensor`, `trainable=true`, an empty
  `trajectory`, and no `judge_scores` and no `start_turn_idx` / `end_turn_idx`

#### Scenario: Non-trainable segment records a local trajectory

- **WHEN** Multica records a non-trainable segment at a terminal or delegation seam
- **THEN** the segment carries null `trajectory_id` and `tensor_ref`,
  `trajectory_source=task_messages`, `trainable=false`, and a `trajectory` snapshot
  sourced only from persisted `task_message` rows in its sequence range

#### Scenario: Edges match EdgeType and keep the DAG acyclic

- **WHEN** Multica assembles edges
- **THEN** each edge type is `delegation`, `mention`, or `completion`, every edge
  flows from a cause segment to an effect segment, and the resulting directed graph
  is acyclic

#### Scenario: AssembledDag contains no secrets

- **WHEN** `AssembledDag` is assembled
- **THEN** no segment or edge carries a provider API key

### Requirement: AReaL resolves refs and builds the DAG

AReaL SHALL resolve each `trainable=true` segment's `tensor_ref` to
tokens/logprobs via the v2 data_proxy (`/data/<shard_id>`, `/data/batch`), build a
`SuperNode` whose payload is the resolved tensors and whose metadata is the
`AssembledDag` segment, and construct the `ExecutionDAG` from the assembled edges -
by ref-resolution, not by slicing a node list on turn indices. Non-trainable
(`trainable=false`) segments SHALL be preserved in the `ExecutionDAG` topology with
their identity, local trajectory, environment snapshot, and edges, but MUST NOT be
resolved to tensors. The assembler MUST validate that the assembled DAG is acyclic
and that each session's segments cover the run without gaps.

#### Scenario: Trainable refs resolve to tensors and build a SuperNode

- **WHEN** AReaL consumes an `AssembledDag` with a trainable segment
- **THEN** that segment's `tensor_ref` resolves to its tokens/logprobs and a
  `SuperNode` is built with that tensor payload and the segment metadata

#### Scenario: Non-trainable segments are preserved without resolution

- **WHEN** AReaL consumes an `AssembledDag` containing a `task_messages` segment
- **THEN** the segment retains its identity, local trajectory, environment snapshot,
  and edges in the `ExecutionDAG`, and no tensor resolution is attempted for it

#### Scenario: Edges build an acyclic ExecutionDAG

- **WHEN** AReaL builds the `ExecutionDAG` from assembled edges
- **THEN** the graph is acyclic and reconstructs the delegation / mention /
  completion topology Multica recorded, including non-trainable segments

#### Scenario: Missing segment is detected

- **WHEN** a session's segments do not densely cover the run (gap or missing
  trajectory)
- **THEN** the assembler raises a typed `DAGError` and does not produce a partial DAG

### Requirement: Tensor lifecycle cleanup

After training consumes a `trainable=true` segment's tensors, AReaL SHALL release
the v2 storage by calling `DELETE /data/clear` for the resolved shards and
`remove_session` (or session revoke) once the session's last trainable segment is
consumed. Non-trainable segments have no v2 shards and MUST NOT be cleared or
resolved. Refs MUST NOT be cleared before training resolves them.

#### Scenario: Trainable refs freed after training

- **WHEN** training has consumed a trainable segment's tensors
- **THEN** AReaL clears the corresponding data_proxy shards

#### Scenario: Non-trainable segments are never cleared

- **WHEN** a `task_messages` segment is present in the DAG
- **THEN** AReaL never calls `DELETE /data/clear` or `remove_session` on its behalf

#### Scenario: Session released after last trainable segment

- **WHEN** the session's last trainable segment is consumed and cleared
- **THEN** AReaL revokes the session, releasing v2 session state

### Requirement: Polling AssembledDag return at task completion

Multica SHALL expose `GET /api/v1/env-dispatch/{projectID}/dag` that resolves
readiness and completeness through the durable `env_dispatch_run` root task,
independent of any `training_dispatch` row. While the root task is absent, queued,
or running, the endpoint SHALL return `202` with a status body. When the root task
is terminal with incomplete session coverage, the endpoint SHALL return `200` with a
failed status body. When the root task is terminal with dense coverage, the endpoint
SHALL return `200` with the assembled `AssembledDag`. The endpoint reuses the
existing env-dispatch path with `project_id` as the root handle.

#### Scenario: In-progress poll

- **WHEN** AReaL polls the endpoint while the root task is absent, queued, or running
- **THEN** the system returns `202` with a status body indicating the task is in
  progress

#### Scenario: Failed poll on incomplete coverage

- **WHEN** the root task is terminal but session coverage is incomplete
- **THEN** the system returns `200` with a failed status body

#### Scenario: Completed poll returns AssembledDag

- **WHEN** the root task is terminal with dense coverage
- **THEN** the system returns `200` with an `AssembledDag` that AReaL resolves and
  reconstructs losslessly into the `ExecutionDAG`

#### Scenario: Unknown project rejected

- **WHEN** a caller polls a `project_id` that does not exist
- **THEN** the system returns `404`

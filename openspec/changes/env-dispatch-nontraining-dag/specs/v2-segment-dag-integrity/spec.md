## ADDED Requirements

### Requirement: Segment source and trainability are consistent

The system SHALL enforce that every `interaction_dag_segment` carries a
`trajectory_source` of `areal_tensor` or `task_messages` and a `trainable` boolean
consistent with its source. An `areal_tensor` segment MUST be `trainable=true`, carry
non-null `trajectory_id` and `tensor_ref`, and have an empty `trajectory`. A
`task_messages` segment MUST be `trainable=false`, have null `trajectory_id` and
`tensor_ref`, and carry a non-empty `trajectory` sourced from persisted
`task_message` rows.

#### Scenario: Trainable tensor segment is well-formed

- **WHEN** a segment is recorded with `trajectory_source=areal_tensor`
- **THEN** `trainable=true`, `trajectory_id` and `tensor_ref` are non-null, and
  `trajectory` is empty

#### Scenario: Non-trainable task-message segment is well-formed

- **WHEN** a segment is recorded with `trajectory_source=task_messages`
- **THEN** `trainable=false`, `trajectory_id` and `tensor_ref` are null, and
  `trajectory` is the persisted message-range snapshot

#### Scenario: Existing rows backfill as trainable tensor segments

- **WHEN** the dual-source migration runs against pre-existing rows
- **THEN** each row is backfilled with `trajectory_source=areal_tensor` and
  `trainable=true` and retains its existing `trajectory_id` and `tensor_ref`

### Requirement: Mixed DAG topology preserves non-trainable segments

A mixed `AssembledDag` containing both `areal_tensor` and `task_messages` segments
SHALL preserve every segment and every edge. Non-trainable segments MUST retain
their DAG identity, local trajectory, environment snapshot, and edges, and MUST NOT
be silently dropped during assembly, serialization, or deserialization.

#### Scenario: Mixed DAG retains both segment sources

- **WHEN** an `AssembledDag` contains one `areal_tensor` and one `task_messages`
  segment joined by an edge
- **THEN** strict parsing preserves both segments, the edge between them, and each
  segment's source-specific fields

#### Scenario: Non-trainable segment survives a round-trip

- **WHEN** an `AssembledDag` with a `task_messages` segment is serialized and
  restored
- **THEN** the restored segment retains `trajectory_source=task_messages`,
  `trainable=false`, its `trajectory`, and null tensor fields

### Requirement: Only trainable segments reach tensor resolution and cleanup

The AReaL DAG consumer SHALL resolve tensor references and schedule shard cleanup
only for `trainable=true` segments. A `task_messages` segment MUST NOT be passed to
tensor resolution or shard cleanup. A malformed trainable segment (missing
`trajectory_id` or `tensor_ref`) SHALL be a hard DAG-consumer error, while a
malformed `task_messages` message SHALL be a recording warning, not a task failure.

#### Scenario: Only the trainable tensor ref is resolved

- **WHEN** a mixed DAG is consumed
- **THEN** only the `areal_tensor` segment's `tensor_ref` is resolved and only its
  shards are cleared; the `task_messages` segment is never resolved or cleared

#### Scenario: Malformed trainable segment is a hard error

- **WHEN** a `trainable=true` segment is missing its `trajectory_id` or `tensor_ref`
- **THEN** the DAG consumer raises a typed error and does not produce a partial DAG

#### Scenario: Malformed task message is a warning

- **WHEN** a `task_messages` segment contains a malformed local message
- **THEN** the system records a recording warning and does not fail the task

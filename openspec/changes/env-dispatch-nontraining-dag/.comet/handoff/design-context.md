# Comet Design Handoff

- Change: env-dispatch-nontraining-dag
- Phase: design
- Mode: compact
- Context hash: c3d195a2c7a97e597fb4ae638cd2cacb4d9cda2ee7c1460302537b6cdffe06f9

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-dispatch-nontraining-dag/proposal.md

- Source: openspec/changes/env-dispatch-nontraining-dag/proposal.md
- Lines: 1-70
- SHA256: 73cd91ca1e3838fffd2a1b35a44d054173bc7f7f9df42a2bc3270678a1fdf3d0

```md
## Why

Env-dispatch readiness and DAG completeness today depend on the `training_dispatch`
row and on AReaL tensor trajectories. Non-training dispatches (no train agent) and
mixed squads (a trained leader plus non-trained peers) cannot produce a complete
interaction DAG: non-trained agents record no segments, and `GET /dag` readiness
cannot resolve without a training dispatch. There is also no explicit
training/non-training mode, so callers must infer intent from `train_agent_id`,
which is unsafe for a path that must make zero AReaL lifecycle calls when it is not
training.

## What Changes

- **BREAKING**: Make `training_mode` a required JSON boolean on
  `POST /api/v1/env-dispatch`. Its absence is a validation error; there is no
  inference from `train_agent_id`.
- `training_mode=false` rejects `train_agent_id` and `critic_agent_id`, and makes
  zero AReaL session/trajectory lifecycle calls.
- `training_mode=true` requires `train_agent_id`. Only that agent opens an AReaL
  session and exports tensors; other squad agents remain non-trained and use their
  configured external runtime.
- Introduce `env_dispatch_run`, keyed by project, as the durable dispatch identity
  carrying workspace, training mode, and a nullable root task. `GET /dag` resolves
  readiness and completeness through it, removing the dependency on
  `training_dispatch`.
- Extend `interaction_dag_segment` with `trajectory_source`, `trainable`, and
  `trajectory`; make the AReaL-only `trajectory_id` and `tensor_ref` nullable.
  Existing rows backfill as `areal_tensor` and trainable.
- Record every env-dispatch agent (trained and non-trained) at terminal/delegation
  seams: use the existing AReaL close/export path when an `areal_proxy` exists,
  otherwise upsert a deterministic `multica:<task-id>` local session and record a
  `task_messages` segment from the durable message range.
- Extend the server `SegmentSpec` and the Python `SegmentSpec`/assembler with the
  three dual-source fields. The assembler preserves all segments and edges but
  resolves and clears tensor shards only for `trainable=true` segments.

## Capabilities

### New Capabilities

- `env-dispatch-nontraining-dag`: explicit `training_mode` dispatch contract,
  durable `env_dispatch_run` identity and `/dag` readiness, and non-training local
  trajectory recording that produces a complete DAG without any AReaL lifecycle
  calls.

### Modified Capabilities

- `v2-segment-dag`: the interaction segment gains `trajectory_source`
  (`areal_tensor` | `task_messages`), `trainable`, and `trajectory`; AReaL-only
  columns become nullable; mixed DAGs carry both trainable tensor segments and
  non-trainable task-message segments with shared topology.
- `v2-segment-dag-integrity`: integrity rules split by source — trainable
  `areal_tensor` segments require `trajectory_id`/`tensor_ref` and an empty
  `trajectory`; non-trainable `task_messages` segments require null AReaL fields
  and a persisted message-range `trajectory`.

## Impact

- Go server (`multica/server`): env_dispatch handler/service validation, new
  migration `204_env_dispatch_run`, dual-source migration
  `205_interaction_dag_local_trajectory`, sqlc queries and generated code, the
  interaction_dag service and seams, and task-service terminal/delegation routing.
- Python (`customized_areal/tree_search`): `multica_dag_client.SegmentSpec`,
  `supernode_assembler`, and `segment_dag_trainer` parse mixed DAGs and resolve
  only trainable tensors.
- API: `POST /api/v1/env-dispatch` gains a required `training_mode`;
  `SegmentSpec` gains three fields and nullable tensor fields. No new dependencies
  are added and Go is not installed by this change.
- Security: provider API keys must never appear in DAG data, responses, errors, or
  logs; the local trajectory is sourced only from persisted `task_message` columns.

```

## openspec/changes/env-dispatch-nontraining-dag/design.md

- Source: openspec/changes/env-dispatch-nontraining-dag/design.md
- Lines: 1-104
- SHA256: 470a9256e474552e94c06d980c8b214fc8c15b9ede3a118ba3c16540116420a9

[TRUNCATED]

```md
## Context

The env-dispatch DAG path assumes a training dispatch: `GET /dag` readiness joins
`training_dispatch`, and segment recording only happens for agents that carry an
AReaL proxy. Non-training dispatches (no train agent) and mixed squads (a trained
leader plus non-trained peers) therefore cannot produce a complete interaction DAG,
and there is no explicit mode to distinguish a path that must make zero AReaL
lifecycle calls. The deep Design Doc
(`docs/superpowers/specs/2026-07-21-env-dispatch-nontraining-dag-design.md`) and
implementation plan
(`docs/superpowers/plans/2026-07-21-env-dispatch-nontraining-dag.md`) specify the
full contract; this document records the high-level architecture decisions.

## Goals / Non-Goals

**Goals:**

- Make training vs non-training an explicit, required dispatch choice.
- Make `/dag` ready and complete in both modes through a durable dispatch identity
  independent of `training_dispatch`.
- Represent every env-dispatch agent in the DAG - trained or not - with an explicit
  trajectory source and trainability.
- Keep AReaL tensor resolution and cleanup scoped to trainable segments only.

**Non-Goals:**

- Backward-compatible inference of training mode from `train_agent_id`.
- Changing AReaL training algorithms or the tensor format.
- Adding dependencies or installing Go.
- Changing one-segment-per-task or best-effort recording semantics.

## Decisions

### Required `training_mode` at the HTTP boundary

Use a `*bool` pointer in the handler to distinguish an omitted JSON field from
`false`, dereference it before constructing `EnvDispatchInput`, and validate at the
service layer: `false` forbids `train_agent_id`/`critic_agent_id`; `true` requires
`train_agent_id`. Inferring mode from `train_agent_id` was rejected because a
non-training path must provably make zero AReaL calls.

### Durable `env_dispatch_run` identity

Introduce a row keyed by project, carrying workspace ID, `training_mode`, and a
nullable `root_task_id`. The dispatch service creates it after the project exists
and binds `root_task_id` immediately after enqueuing the leader task. `/dag` reads
readiness and completeness exclusively through this row. Keeping the
`training_dispatch` join was rejected because non-training dispatches have no such
row, so readiness could never resolve.

### Dual-source segment model

Add `trajectory_source`, `trainable`, and `trajectory` to `interaction_dag_segment`
and make the AReaL-only `trajectory_id` and `tensor_ref` nullable; backfill existing
rows as `areal_tensor` with `trainable=true`. Source-specific DB checks enforce that
trainable segments carry AReaL fields and an empty trajectory, while task-message
segments carry null AReaL fields and a persisted message-range trajectory. A
separate table for local segments was rejected to preserve one-segment-per-task and
shared topology/edges.

### One terminal/delegation seam selects the source

At a close or delegation seam, use the existing AReaL close/export path when
`context.areal_proxy` exists; otherwise upsert a deterministic `multica:<task-id>`
session/run mapping and record a `task_messages` segment from the persisted message
range. Reuse the existing edge and one-segment-per-task guards so mixed
trained/non-trained dispatches share one DAG topology.

### Assembler preserves topology, resolves only trainable

`AssembledDag` keeps all segments and edges. `SegmentSpec` gains the three
dual-source fields with nullable tensor fields. The assembler resolves tensor
references and clears shards only for `trainable=true` segments; non-trainable
segments retain their identity, local trajectory, environment snapshot, and edges,
and contribute no policy-training tensors.

## Risks / Trade-offs

- [Backward-incompatible `training_mode`] -> Mitigation: it is a required field with
  no inference; callers must update. Documented as BREAKING in the proposal.

```

Full source: openspec/changes/env-dispatch-nontraining-dag/design.md

## openspec/changes/env-dispatch-nontraining-dag/tasks.md

- Source: openspec/changes/env-dispatch-nontraining-dag/tasks.md
- Lines: 1-107
- SHA256: ce72176f7fcf27139af00c7534c0c11d72c2ed9afad6f75f405de81c0058f23d

[TRUNCATED]

```md
## 1. Required training-mode request contract

- [ ] 1.1 Write failing handler and service tests: omitted `training_mode` returns
  HTTP 400, `false` plus training IDs fails validation, `true` without
  `train_agent_id` fails, and the two valid forms reach the service with the exact
  boolean.
- [ ] 1.2 Verify RED: `go test ./server/internal/handler ./server/internal/service
  -run 'EnvDispatch.*TrainingMode' -count=1` fails because no explicit
  training-mode contract exists.
- [ ] 1.3 Add `EnvDispatchRequest.TrainingMode *bool` at the HTTP boundary; reject
  nil before constructing `EnvDispatchInput`; pass the dereferenced value as
  `EnvDispatchInput.TrainingMode bool`. In service validation enforce
  `!TrainingMode && (TrainAgentID != "" || CriticAgentID != "")` failure and
  `TrainingMode && TrainAgentID == ""` failure.
- [ ] 1.4 Verify GREEN: the Step 1.2 command passes.

## 2. Durable dispatch root and readiness

- [ ] 2.1 Write failing persistence and readiness tests: every successful rollout
  persists mode and leader task, `/dag` returns 202 for queued/running roots, and
  returns assembled data for a completed non-training root without any
  `training_dispatch` row.
- [ ] 2.2 Verify RED: `go test ./server/internal/handler ./server/internal/service
  -run 'EnvDispatch.*(Root|Readiness|Dag)' -count=1` fails because readiness still
  joins `training_dispatch`.
- [ ] 2.3 Create migration `204_env_dispatch_run` (`env_dispatch_run` keyed by
  project with workspace, training mode, nullable root task) and add create,
  root-bind, and workspace-scoped status queries; update sqlc output via the
  existing generation workflow without adding tools.
- [ ] 2.4 Wire dispatch persistence: create the dispatch row once the project
  exists, bind `LeaderRunID` after enqueue, and make `/dag` exclusively query the
  new root status. Preserve 202, failed-density, and successful-DAG response
  shapes.
- [ ] 2.5 Verify GREEN: the Step 2.2 command passes.

## 3. Dual-source segment persistence

- [ ] 3.1 Write failing local-segment tests: insert task messages at known sequence
  numbers and assert the local recorder upserts session `multica:<task-id>`,
  snapshots only the requested sequence range in order, sets
  `trajectory_source=task_messages`, sets `trainable=false`, leaves AReaL fields
  null, repeated close is idempotent, and runtime provider secrets never enter the
  serialized trajectory.
- [ ] 3.2 Verify RED: `go test ./server/internal/service -run 'InteractionDAG.*Local'
  -count=1` fails because local segment recording does not exist.
- [ ] 3.3 Create migration `205_interaction_dag_local_trajectory`: backfill existing
  rows, make `trajectory_id`/`tensor_ref` nullable, add `trajectory_source`
  (default `areal_tensor`), `trainable` (default true), and `trajectory` (default
  `[]`); add checks requiring non-null AReaL fields only for trainable tensor
  segments and null AReaL fields for task-message segments.
- [ ] 3.4 Implement `RecordLocalSegmentForEvent` and assembly: serialize an
  allowlisted message-event shape (sequence, type, tool, content, input, output)
  from persisted rows, compute start/end using existing sequence queries, atomically
  insert the segment and environment snapshot, and emit `TrajectorySource`,
  `Trainable`, and `Trajectory` from both source types (AReaL-only fields nullable).
- [ ] 3.5 Verify GREEN: the Step 3.2 command plus
  `go test ./server/internal/service -run InteractionDAG -count=1` pass.

## 4. Record every env-dispatch agent at event seams

- [ ] 4.1 Replace the old non-trained no-op test with failing behavior tests: a
  non-trained issue task and channel task record local segments, a mixed
  trained/non-trained pair records both sources with an edge, ordinary non-env-
  dispatch tasks remain no-ops, and the non-training path makes zero fake AReaL
  client calls.
- [ ] 4.2 Verify RED: `go test ./server/internal/service -run
  'InteractionDAG.*(NonTrain|Mixed|Channel)' -count=1` fails at the current
  `extractArealProxyConfig` early return.
- [ ] 4.3 Implement unified project and trajectory-source routing: resolve project
  from issue or chat session, gate local recording on an `env_dispatch_run` lookup,
  keep the current bridge close/export order for proxy tasks, and otherwise use
  deterministic local session/run mapping plus local segment recording. Reuse
  existing edge and one-segment guards.
- [ ] 4.4 Verify GREEN: the Step 4.2 command and
  `go test ./server/internal/service -count=1` pass.

## 5. Parse mixed DAGs safely in AReaL

- [ ] 5.1 Write failing Python contract and resolver tests: build a mixed DAG with
  one `areal_tensor` and one `task_messages` segment; assert strict parsing

```

Full source: openspec/changes/env-dispatch-nontraining-dag/tasks.md

## openspec/changes/env-dispatch-nontraining-dag/specs/env-dispatch-nontraining-dag/spec.md

- Source: openspec/changes/env-dispatch-nontraining-dag/specs/env-dispatch-nontraining-dag/spec.md
- Lines: 1-96
- SHA256: d802101c3a83d2c4e8bac460fda66b1a9f7545338655f2aa5ea61d01c0d10166

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Required training-mode dispatch contract

`POST /api/v1/env-dispatch` SHALL require a `training_mode` JSON boolean. Its
absence SHALL be rejected as a validation error with no dispatch started.
`training_mode=false` MUST reject `train_agent_id` and `critic_agent_id` and MUST
make zero AReaL session or trajectory lifecycle calls. `training_mode=true` MUST
require `train_agent_id`; only that agent opens an AReaL session and exports
tensors, while every other squad agent remains non-trained and uses its configured
external runtime.

#### Scenario: Omitted training mode is rejected

- **WHEN** a caller POSTs env-dispatch without `training_mode`
- **THEN** the system returns HTTP 400 and starts no dispatch

#### Scenario: Non-training mode forbids training IDs and AReaL calls

- **WHEN** `training_mode=false` is sent with a `train_agent_id` or `critic_agent_id`
- **THEN** the system rejects the request and makes zero AReaL lifecycle calls

#### Scenario: Training mode requires a train agent

- **WHEN** `training_mode=true` is sent without `train_agent_id`
- **THEN** the system rejects the request

#### Scenario: Valid modes reach the service with the exact boolean

- **WHEN** a valid `training_mode=true` or `training_mode=false` request is sent
- **THEN** the service receives the exact boolean value and proceeds

### Requirement: Durable dispatch identity independent of training_dispatch

The dispatch service SHALL persist an `env_dispatch_run` row keyed by project ID,
carrying workspace ID, `training_mode`, and a nullable `root_task_id`, after the
project exists. It SHALL bind `root_task_id` immediately after enqueuing the leader
task. The dispatch identity and its readiness MUST NOT depend on the existence of a
`training_dispatch` row.

#### Scenario: Dispatch run is created and the root task is bound

- **WHEN** a successful rollout creates the project and enqueues the leader task
- **THEN** an `env_dispatch_run` row exists with the persisted `training_mode` and
  its `root_task_id` bound to the leader task

#### Scenario: Non-training dispatch needs no training_dispatch row

- **WHEN** a `training_mode=false` dispatch completes
- **THEN** `/dag` returns the assembled DAG without any `training_dispatch` row
  existing

#### Scenario: Training dispatch still records its training_dispatch row

- **WHEN** a `training_mode=true` dispatch runs
- **THEN** the existing `training_dispatch` lifecycle is unaffected and the
  `env_dispatch_run` row coexists with it

### Requirement: Non-training local trajectory recording

For an env-dispatch task whose context has no `areal_proxy`, the task service SHALL
upsert a deterministic `multica:<task-id>` session/run mapping and record a
`task_messages` segment sourced only from persisted `task_message` rows in the
segment's sequence range. Recording SHALL be best-effort: a recording failure MUST
NOT change the task's terminal result. Provider API keys MUST NOT appear in the
serialized trajectory, the response, errors, or structured log fields.

#### Scenario: Non-trained task records a local segment

- **WHEN** a non-trained env-dispatch task terminates or delegates
- **THEN** the system records a `task_messages` segment with `trainable=false`, null
  AReaL fields, and a trajectory snapshot of only the requested sequence range in
  order

#### Scenario: Local session ID is deterministic and never an AReaL call

- **WHEN** a non-trained task is recorded
- **THEN** its session ID is `multica:<task-id>`, which is never used as an AReaL
  credential and never leaves Multica as an AReaL API call


```

Full source: openspec/changes/env-dispatch-nontraining-dag/specs/env-dispatch-nontraining-dag/spec.md

## openspec/changes/env-dispatch-nontraining-dag/specs/v2-segment-dag/spec.md

- Source: openspec/changes/env-dispatch-nontraining-dag/specs/v2-segment-dag/spec.md
- Lines: 1-135
- SHA256: be9ee5eebe3f4f26b967b7ff82ec77e751b6700faebe29e9693d65d252065381

[TRUNCATED]

```md
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


```

Full source: openspec/changes/env-dispatch-nontraining-dag/specs/v2-segment-dag/spec.md

## openspec/changes/env-dispatch-nontraining-dag/specs/v2-segment-dag-integrity/spec.md

- Source: openspec/changes/env-dispatch-nontraining-dag/specs/v2-segment-dag-integrity/spec.md
- Lines: 1-74
- SHA256: d31fa7f8d5ea6ea50db440b9a8d0b64cf5be7d62ab1e8f2dd957ab13cdee986a

```md
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

```

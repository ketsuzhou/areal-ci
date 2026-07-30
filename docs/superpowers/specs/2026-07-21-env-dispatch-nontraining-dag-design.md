---
comet_change: env-dispatch-nontraining-dag
role: technical-design
canonical_spec: openspec
---

# Env Dispatch Non-Training DAG Design

## Goal

Make `POST /api/v1/env-dispatch` explicitly choose training or non-training
execution, and make `GET /api/v1/env-dispatch/{projectID}/dag` become ready and
return a complete DAG in both modes. A mixed training dispatch must include
non-trained agents in the same DAG without calling AReaL for those agents.

## Contract

`training_mode` is a required JSON boolean. Its absence is a validation error;
there is no compatibility inference from `train_agent_id`.

- `training_mode=false` rejects `train_agent_id` and `critic_agent_id`. No AReaL
  lifecycle endpoint is called.
- `training_mode=true` requires `train_agent_id`. Only that agent opens an AReaL
  session and exports AReaL tensors. Other squad agents remain non-trained and
  use their configured external runtime.
- Every env-dispatch task that participates in the interaction is represented
  by a DAG session/run and at least one segment when it terminates or delegates.

## Dispatch identity and readiness

Introduce `env_dispatch_run`, keyed by project ID, with workspace ID,
`training_mode`, and nullable `root_task_id`. This is the durable identity of a
dispatch independent of whether it has a `training_dispatch` row.

The dispatch service creates the row after creating the project and binds
`root_task_id` immediately after enqueuing the leader task. `GET /dag` reads the
root task through this record:

- no run or no root task yet: `202 {"status":"in_progress"}`;
- non-terminal root task: `202`;
- terminal root task with incomplete session coverage: `200
  {"status":"failed"}`;
- terminal root task with dense coverage: `200` assembled DAG.

This removes readiness' dependency on `training_dispatch`.

## Dual trajectory model

An interaction segment has an explicit source and trainability:

- `areal_tensor`: `trainable=true`, with AReaL `trajectory_id` and
  `tensor_ref`; `trajectory` is empty.
- `task_messages`: `trainable=false`, with a JSON snapshot of Multica
  `task_message` rows in the segment's `start_seq..end_seq` range;
  `trajectory_id` and `tensor_ref` are null.

The database migration adds `trajectory_source`, `trainable`, and `trajectory`,
and makes the two AReaL-only columns nullable. Existing rows are backfilled as
`areal_tensor` and remain trainable.

Non-training session IDs are deterministic (`multica:<task-id>`). They are not
AReaL credentials and never leave Multica as API calls. At a delegation or
terminal seam, the task service:

1. Resolves the task's project for issue and channel tasks.
2. Verifies the project belongs to an env-dispatch run.
3. Uses the existing AReaL close/export path when `context.areal_proxy` exists.
4. Otherwise upserts the deterministic local session/run mapping and records a
   `task_messages` segment from the durable message range.
5. Adds edges through the existing one-segment-per-task logic.

The recording remains best-effort so an observability failure cannot change the
task's terminal result.

## API and AReaL consumer

`SegmentSpec` gains exactly three fields:

- `trajectory_source: Literal["areal_tensor", "task_messages"]`;
- `trainable: bool`;
- `trajectory: list[dict[str, Any]]`.

`trajectory_id` and `tensor_ref` become nullable/optional. The Multica service
and Python dataclass change together so strict `SegmentSpec(**payload)` parsing
continues to work.

The AReaL assembler preserves all segments and edges in `AssembledDag`, but
only resolves tensor references for `trainable=true` segments. Non-trainable
segments retain their identity, local trajectory, environment snapshot, and
edges; they contribute no policy-training tensors and their shards are never
resolved or cleared.

## Security and failure behavior

Local trajectory serialization is sourced only from persisted `task_message`
columns and never from sandbox runtime configuration. Provider `api_key` values
must not appear in the segment, response, errors, or logs. A malformed local
message becomes a recording warning, not a task failure. A malformed trainable
segment remains a hard DAG-consumer error because training without a valid
tensor reference is unsafe.

## Testing

Server tests cover required `training_mode`, contradictory training fields,
zero AReaL calls in non-training mode, mixed trained/non-trained recording,
local trajectory ranges, readiness via root task, dense coverage, delegation
edges, idempotent close, and secret non-disclosure. Python tests cover parsing
both segment types and proving only trainable tensor references are resolved
and cleared.

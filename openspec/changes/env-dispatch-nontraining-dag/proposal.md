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

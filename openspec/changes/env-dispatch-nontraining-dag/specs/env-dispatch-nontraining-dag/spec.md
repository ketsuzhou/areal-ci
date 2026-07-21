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

#### Scenario: Idempotent close

- **WHEN** a non-trained task's close seam fires more than once
- **THEN** the segment is recorded once and repeated closes are idempotent

#### Scenario: Recording failure does not fail the task

- **WHEN** local segment recording fails
- **THEN** the task's terminal result is unchanged and the failure is recorded as an
  observability warning

#### Scenario: Secrets never enter the trajectory

- **WHEN** a local trajectory is serialized
- **THEN** no provider API key appears in the segment, the response, errors, or
  structured log fields

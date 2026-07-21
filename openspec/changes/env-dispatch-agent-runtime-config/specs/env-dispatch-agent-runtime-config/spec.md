## ADDED Requirements

### Requirement: Env-dispatch uses the frontend sandbox lifecycle

Frontend sandbox creation and env-dispatch provisioning SHALL call the same sandbox
creation service and SHALL produce the same canonical instance metadata, daemon
bootstrap environment, runtime object, and sandbox job payload.

#### Scenario: Static source agent starts a frontend-equivalent sandbox

- **WHEN** a non-training source agent is first addressed with a valid external runtime
- **THEN** env-dispatch creates its sandbox through the shared frontend service with the
  source binding's exact base URL, API key, and model

#### Scenario: Runtime rows are not pre-created

- **WHEN** provisioning begins
- **THEN** no `agent_runtime` row is inserted until the sandbox daemon registers itself

### Requirement: Runtime identity is discovered safely

The system SHALL resolve a sandbox runtime only when workspace, provider, daemon ID, and
sandbox instance ID match and the provider reports online. It MUST NOT bind by runtime
display name.

#### Scenario: Matching online Pi runtime becomes ready

- **WHEN** the sandbox daemon registers an online Pi runtime with the binding's daemon
  ID and sandbox instance ID
- **THEN** provisioning records that runtime ID and advances to derived-agent creation

#### Scenario: Runtime identity mismatch fails closed

- **WHEN** a runtime name matches but daemon ID or sandbox instance ID differs
- **THEN** the runtime is rejected and no agent or task is bound to it

#### Scenario: Runtime readiness times out

- **WHEN** the sandbox is running but no matching Pi runtime becomes online before the
  configured deadline
- **THEN** the rollout fails with a sanitized retryable error and compensates owned
  resources instead of leaving the DAG indefinitely in progress

### Requirement: Addressed source agent produces a derived global agent

The system SHALL create a new global agent derived from each addressed source agent,
copy approved executable configuration and skills, record `source_agent_id`, and bind
the derived agent permanently to the discovered runtime for the dispatch lifetime.

#### Scenario: Derived agent records lineage and runtime

- **WHEN** runtime readiness succeeds
- **THEN** one derived agent is created with the source agent's approved configuration,
  the source-agent lineage, and the discovered runtime ID while the source agent remains
  unchanged

#### Scenario: Channel execution uses the derived agent

- **WHEN** the derived agent transaction commits
- **THEN** the env-dispatch channel routes the original address to the derived agent and
  the task explicitly uses the derived agent and binding runtime

#### Scenario: Concurrent dispatches share a source safely

- **WHEN** two dispatches concurrently address the same source agent
- **THEN** each dispatch creates its own sandbox, runtime, and derived agent without
  changing the source agent's global runtime

### Requirement: Credentials are isolated by source binding

Each binding SHALL record its source agent as the model-configuration owner. A model
credential or training session MUST NOT be used when its owner does not equal the
binding source agent.

#### Scenario: Static squad credentials do not cross

- **WHEN** two source agents have distinct external runtime objects
- **THEN** each derived sandbox receives only its source binding's credential and model

#### Scenario: Training target opens one session before sandbox creation

- **WHEN** the training source agent is first addressed
- **THEN** env-dispatch calls `start_session(session_ref, env_id)` once with the
  persistent source binding ID and configures the sandbox with the returned key, model
  `areal-default`, and configured bridge URL without creating or reserving a task

#### Scenario: Training retry reuses session identity

- **WHEN** provisioning retries after `start_session` succeeded
- **THEN** it reuses the recorded session ID, key, and session reference without opening
  another session

#### Scenario: Real task is linked after session bootstrap

- **WHEN** the training derived agent becomes ready
- **THEN** env-dispatch inserts a normal task and records its real ID against the
  existing session so DAG assembly maps the session to the actual derived-agent run

#### Scenario: Existing task-based session clients remain compatible

- **WHEN** an existing client calls `start_session` with `task_id`
- **THEN** the bridge uses that value as the session reference without requiring the
  client to migrate immediately

### Requirement: First-address provisioning is single-flight

The scratch leader's initial message and a peer's first directed mention SHALL use one
binding state machine. Concurrent addresses of the same source binding SHALL create at
most one credential/session, sandbox, runtime association, derived agent, and task.

#### Scenario: Concurrent mentions provision once

- **WHEN** two directed mentions concurrently target one pending source binding
- **THEN** one claimant provisions and both callers observe the same ready or failed
  result

### Requirement: Derived dispatch cleanup is complete and idempotent

Deleting env-dispatch resources SHALL cancel derived-agent tasks, archive the derived
agent, delete its sandbox and registered runtime, revoke bootstrap/model credentials,
close any training session, and preserve the source agent.

#### Scenario: Successful dispatch cleanup

- **WHEN** a ready derived dispatch is deleted
- **THEN** every derived resource is removed or archived exactly once and the source
  agent and its original runtime remain usable

#### Scenario: Partial provisioning cleanup

- **WHEN** deletion races any intermediate provisioning state
- **THEN** cleanup reaches a terminal deleted state without leaking a session, sandbox,
  runtime, derived agent, or credential

### Requirement: Client reports terminal failures

The standalone Python client SHALL fail on per-rollout errors, runtime-readiness
failure, DAG timeout, and structurally invalid DAG responses.

#### Scenario: DAG timeout is not success

- **WHEN** DAG polling reaches its deadline without a valid assembled DAG
- **THEN** the client returns a non-zero result after cleanup

### Requirement: Derived agent produces a valid DAG

A configured non-training or training derived agent SHALL be able to reply in its
env-dispatch channel and produce a structurally valid assembled DAG.

#### Scenario: Deployed static model replies

- **WHEN** a source agent is addressed with a valid rotated external credential
- **THEN** its derived agent replies through the online sandbox runtime and the DAG
  endpoint returns a valid `200` payload

#### Scenario: Deployed training model replies

- **WHEN** a training source agent obtains a valid AReaL session
- **THEN** its derived agent replies through `areal-default`, and the DAG maps the
  session ID to the reserved derived-agent run ID

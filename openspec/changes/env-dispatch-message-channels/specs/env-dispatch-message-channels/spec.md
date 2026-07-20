## ADDED Requirements

### Requirement: Message rollout isolation

Every `dispatch_type=message` rollout SHALL own a distinct env, project, and
group channel. The EnvDispatch caller and the exact requested agent roster
(squad leader plus agent members, deduplicated) SHALL be the only channel
members. EnvDispatch channels SHALL NOT auto-provision a group manager.

#### Scenario: Two rollouts are fully isolated
- **WHEN** a message dispatch requests group_size > 1
- **THEN** every rollout has a distinct env_id, project_id, channel_id, sandbox, daemon, and runtime namespace

#### Scenario: No automatic group manager
- **WHEN** an EnvDispatch group channel is created
- **THEN** no Beckham group-manager agent session is provisioned for that channel

### Requirement: Binding-routed sandbox execution

An agent task created for an EnvDispatch channel SHALL use that rollout's binding
runtime and SHALL NOT fall back to the agent's shared default runtime. The same
agent in two rollouts SHALL NOT share a sandbox, daemon, or runtime.

#### Scenario: Task uses the binding runtime, never the default
- **WHEN** the leader task is enqueued for a message rollout
- **THEN** the task's runtime_id equals the leader binding's runtime_id and differs from the agent's default runtime_id

### Requirement: Scratch wakes only the leader

A scratch message dispatch SHALL provision and wake only the canonical roster
leader. All other roster members SHALL remain pending bindings, provisioned on
their first directed mention.

#### Scenario: Peers are not woken on scratch
- **WHEN** a squad scratch message dispatch runs
- **THEN** exactly one leader binding is ready, every peer binding is pending, and exactly one leader task is enqueued

### Requirement: Lazy first-mention provisioning

A directed mention of a pending EnvDispatch agent SHALL provision that agent
exactly once under concurrency, using the binding runtime. Provisioning failure
SHALL record a retryable failed binding and SHALL NOT enqueue a task or fall
back to the shared runtime.

#### Scenario: Concurrent mentions provision once
- **WHEN** multiple concurrent mentions target the same pending binding
- **THEN** exactly one sandbox, one runtime, and one channel-agent session are created, and no task uses the default runtime

### Requirement: Branch validation before writes

A branch SHALL validate that the source env has a decodable trigger whose target
agent belongs to its EnvDispatch channel and that the requested roster matches
the source channel roster. A missing, malformed, unauthorized, or
roster-incompatible trigger SHALL return an error before any resource is
created.

#### Scenario: Invalid trigger creates no resources
- **WHEN** a branch is requested against an env with a missing or unauthorized trigger
- **THEN** no env fork, project, channel, runtime, or sandbox is created and the request fails with a validation error

### Requirement: Branch resume from copied collaboration

A branch SHALL copy the source channel's members, messages, replies, quotes,
threads, and read state, remap the persisted trigger to the copied entities, and
wake only the trigger-selected agent by cloning its source sandbox. Non-triggered
agents SHALL remain pending with their source sandbox recorded as a lazy clone
source. Request `message.content`, when non-empty, SHALL be appended as
nondispatching context without changing the trigger-selected agent.

#### Scenario: Only the trigger agent wakes with a cloned sandbox
- **WHEN** a branch resumes a collaboration whose trigger selects agent A
- **THEN** only agent A is provisioned (via sandbox clone from its source binding), its task is enqueued on the new runtime, peers remain pending, and the remapped trigger is saved on the new env

### Requirement: Channel-first lifecycle facades

Channel-first routes SHALL resolve the bound project internally:
`GET /api/v1/env-dispatch/channels/{channelID}/dag`,
`DELETE /api/v1/env-dispatch/channels/{channelID}`, and
`GET /api/v1/channels/{channelID}/env-checkpoints`. Cleanup SHALL be idempotent
and SHALL serialize with concurrent provisioning via the env-agent binding
deleting state. Existing project-first routes and issue dispatch behavior SHALL
remain available.

#### Scenario: Cleanup is idempotent and serializes with provisioning
- **WHEN** a channel cleanup runs concurrently with a first-mention provisioning
- **THEN** cleanup marks bindings deleting, prevents new claims, waits for or compensates in-flight provisioning, and removes channel/project/env/bindings in foreign-key-safe order; a repeated cleanup returns success

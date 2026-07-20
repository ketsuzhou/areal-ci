______________________________________________________________________

## comet_change: env-dispatch-agent-runtime-config role: technical-design canonical_spec: openspec

# EnvDispatch Sandbox Runtime and Derived-Agent Provisioning

## Objective

Make message env-dispatch follow the same proven sandbox/runtime lifecycle as the
frontend. When a roster agent is first addressed, the server obtains that agent's model
credential, creates a sandbox through the shared frontend sandbox service, waits for the
sandbox daemon to register an online runtime, creates a new global agent derived from
the addressed source agent, permanently binds the derived agent to that runtime for the
lifetime of the dispatch, and routes the message to the derived agent.

The design supports both external static models and AReaL training sessions. It also
records source-agent lineage and guarantees that credentials, sessions, sandboxes,
runtimes, and derived agents cannot cross between roster members.

## Evidence and Problem Statement

The deployed frontend flow is healthy: a frontend-created sandbox reaches `running`, its
daemon registers a Pi runtime using the sandbox identity, and the runtime remains
online. A frontend-created global agent bound to that runtime can answer in a channel.

The current env-dispatch flow follows a different lifecycle. It pre-creates an offline
`agent_runtime` row with a selected daemon ID, creates a sandbox, and queues a task
against the pre-created runtime before proving provider readiness. In deployed tests the
sandbox reached `running` and the daemon adopted the expected runtime row, but Pi
reported `offline`. The task stayed queued and the DAG endpoint remained
`202 in_progress`.

Container readiness, daemon registration, provider readiness, task completion, and DAG
completion are distinct states. A successful sandbox create is not sufficient evidence
that an agent can run.

## Selected Architecture

### Reuse one sandbox creation service

Extract the frontend `CreateSandboxInstance` core into a shared service and use it from
both the frontend handler and env-dispatch. The service owns:

- node and template selection;
- sandbox instance creation;
- model runtime metadata;
- daemon bootstrap PAT minting;
- runtime environment construction;
- sandbox job payload construction and notification;
- persistence of bootstrap metadata needed for resume and diagnosis.

Env-dispatch must not maintain a second implementation that merely attempts to mirror
the frontend payload.

### Do not pre-create an agent runtime row

Env-dispatch no longer inserts an offline `agent_runtime` before sandbox creation. The
sandbox daemon registers its runtime in the ordinary frontend way. The runtime display
name may follow the sandbox name, but display names are not identifiers and must never
be used as the binding key.

The sandbox service still mints a unique daemon ID as a correlation nonce. This does not
pre-create a runtime row. After daemon registration, env-dispatch resolves the runtime
by the workspace-scoped daemon ID and provider. The runtime registration should
additionally retain `sandbox_instance_id` in trusted metadata so the server can verify
both identifiers before binding.

Success requires all of the following:

```text
sandbox.status = running
runtime.daemon_id = binding.daemon_id
runtime.metadata.sandbox_instance_id = binding.sandbox_instance_id
runtime.provider = pi
runtime.status = online
```

### Create a derived global agent

The roster contains source agents selected by the caller. A source agent is a template
for execution; it is never rebound to an ephemeral dispatch runtime. When a source
agent's sandbox runtime becomes online, env-dispatch creates a new global derived agent
that copies the source agent's executable configuration, including instructions, skills,
visibility-compatible settings, and supported Pi execution settings.

The derived agent records:

```text
source_agent_id
env_dispatch_env_id
sandbox_instance_id
runtime_id
```

The derived agent is permanently bound to that runtime while the dispatch exists. The
current env-dispatch channel replaces the source roster member with the derived member
for execution and authorship. Other channels and the source agent's global runtime
binding remain unchanged.

Creating a derived agent rather than mutating `source_agent.runtime_id` avoids
cross-dispatch races when the same source agent is addressed concurrently in multiple
environments.

## First-Address Provisioning State Machine

Each `environment_agent_sandbox` binding is keyed by `(env_id, source_agent_id)` and is
the single-flight owner for provisioning. The states are:

```text
pending
  -> credential_ready
  -> sandbox_creating
  -> runtime_waiting
  -> agent_creating
  -> ready

Any non-terminal step -> failed_retryable
Cleanup from any step -> deleting -> deleted
```

The first directed mention claims `pending` or `failed_retryable`. Concurrent mentions
observe the winner and wait for the same terminal result. The scratch leader's initial
dispatch counts as its first directed address and uses this same state machine; peers
remain pending until explicitly addressed.

The binding stores separate source and derived identities:

```text
source_agent_id
derived_agent_id
sandbox_instance_id
daemon_id
runtime_id
training_session_id
training_session_ref
credential_kind
model_config_owner_agent_id
```

`model_config_owner_agent_id` must equal `source_agent_id`. This invariant, together
with the binding primary key, prevents one agent's credential from being applied to
another agent's sandbox.

## Credential Resolution

### Non-training agent

The caller supplies a complete per-source-agent runtime object:

```json
{
  "base_url": "https://provider.example/v1",
  "api_key": "<secret>",
  "model": "provider-model"
}
```

The validated policy is stored write-only with that source agent's pending binding. On
first address, only the claimed binding can read the policy and pass it to the shared
sandbox creation service. Squad members never inherit another member's policy.

### Training agent

The training target ignores caller-supplied external runtime credentials. On its first
address, env-dispatch calls `start_session(session_ref, env_id)` with the persistent
env-agent binding ID as `session_ref` and receives a unique `session_id` and model API
key. The AReaL bridge accepts `session_ref` as the session namespace while retaining
legacy `task_id` request compatibility; a session no longer requires a synthetic or
not-yet-inserted Multica task. After the derived agent and runtime are ready,
env-dispatch creates a normal task and records the real task ID against the existing
session for DAG assembly. It builds a server-owned runtime policy:

```text
base_url = configured AReaL bridge URL
api_key  = start_session response for this source agent
model    = areal-default
```

The deployment config supplies the requested bridge URL; production code does not
hardcode a deployment endpoint. The returned `session_id`, session key, and
`training_session_ref` are persisted on the same binding as the source agent, then
injected through the training runtime contract required by the bridge. Retry reuses the
recorded session and never calls `start_session` twice.

The AReaL model key is distinct from the sandbox bootstrap PAT. The bootstrap PAT
authenticates the daemon to MultiCA; the session key authenticates model inference to
the bridge. They must use different typed fields and must never be substituted for one
another.

### One-to-one credential invariant

Before sandbox creation, the server verifies:

```text
claimed binding source_agent_id
  == credential model_config_owner_agent_id
  == start_session source agent (training only)
```

Before task enqueue, it verifies:

```text
binding.derived_agent_id.runtime_id
  == binding.runtime_id
binding.runtime_id.sandbox_instance_id
  == binding.sandbox_instance_id
```

Any mismatch fails closed and triggers compensation. No fallback to a shared or
source-agent runtime is permitted.

## Derived-Agent Creation

Derived-agent creation uses a server-side clone operation rather than replaying a
frontend request. The clone copies only approved agent fields and creates new
ownership/runtime fields atomically. Secrets or runtime-specific state from the source
agent are not copied.

A nullable self-reference such as `agent.source_agent_id` records direct lineage.
Dispatch ownership fields may live in a dedicated mapping table if the agent schema
should remain general, but the lineage must be queryable from the global agent resource
and protected by workspace foreign keys.

The database transaction must:

1. verify source agent and runtime belong to the workspace;
1. create the derived agent bound to the online runtime;
1. copy approved skills and agent settings;
1. record source-agent and dispatch ownership;
1. replace/add the derived member in the env-dispatch channel;
1. persist `derived_agent_id` on the binding.

After commit, the task is created normally with a newly generated real task ID, the
derived agent ID, and explicit runtime ID. The server then records
`session_id -> task_id` for training DAG assembly. Task enqueue and
collaboration-trigger idempotency use that real task identity only.

## Failure and Compensation

- Credential validation or `start_session` failure creates no sandbox or derived agent
  and leaves a retryable binding without exposing a key.
- Sandbox job failure deletes the partial sandbox and revokes its bootstrap PAT.
- Sandbox `running` with no matching online Pi runtime before timeout is a provisioning
  failure, not a successful dispatch.
- Runtime identity mismatch fails closed; the server never binds by runtime display
  name.
- Derived-agent creation failure deletes the sandbox/runtime and closes the training
  session when applicable.
- Task enqueue failure archives the derived agent, deletes the sandbox/runtime, and
  closes the training session.
- Rollout responses include a sanitized per-rollout error. The DAG must not stay
  indefinitely `in_progress` after terminal provisioning failure.
- The Python client treats per-rollout errors, readiness timeout, DAG timeout, and
  invalid DAG structure as failures.

## Cleanup Lifecycle

`DELETE /api/v1/env-dispatch/...` owns every derived resource created by that dispatch.
Cleanup is idempotent and performs:

1. stop/cancel tasks for each derived agent;
1. remove derived channel memberships and archive the derived global agent;
1. stop and delete the sandbox;
1. delete or retire its registered runtime;
1. close/revoke the training session when present;
1. revoke sandbox bootstrap PATs and delete retained model credentials;
1. delete the env-agent binding and remaining dispatch resources.

The source agent, its original runtime, instructions, skills, and memberships outside
the env-dispatch channel are never modified.

## Public API and Response Contract

The non-training request keeps the existing per-agent runtime shape. Training
configuration remains server-owned and is selected by `train_agent_id`.

Responses may expose non-secret provenance and readiness fields:

```text
source_agent_id
derived_agent_id
sandbox_instance_id
runtime_id
runtime_status
```

They must never expose provider API keys, sandbox PATs, training session keys, or stored
runtime policy objects. `session_id` should remain internal unless an existing training
API contract explicitly requires it.

## Test Strategy

### Unit and service tests

- Prove frontend and env-dispatch use the same sandbox create service and canonical
  payload builder.
- Prove no `agent_runtime` row is inserted before daemon registration.
- Exercise runtime discovery by daemon ID plus sandbox instance ID; reject name matching
  and identity mismatch.
- Verify source-agent cloning copies approved instructions/skills and records lineage
  while leaving the source unchanged.
- Verify concurrent first mentions create one sandbox, runtime binding, derived agent,
  and task.
- Verify two source agents receive different credentials, sessions, sandboxes, runtimes,
  and derived agents.
- Verify training calls `start_session` once with the claimed binding ID as
  `session_ref`, later links the resulting session to the normally inserted
  derived-agent task, and uses the returned key only with that agent's sandbox.
- Verify all compensation paths and idempotent cleanup.

### Security tests

- Use distinct sentinel keys for two agents and scan sandbox payloads, tasks, responses,
  errors, and logs for cross-over or disclosure.
- Distinguish bootstrap PAT fields from model credential fields at compile-time and in
  serialized payload assertions.
- Reject cross-workspace source agents, runtimes, and lineage references.

### Deployed verification

For a non-training source agent, provide a rotated credential through a secure
environment or secret injector, address the source agent, and assert:

- a frontend-equivalent sandbox reaches `running`;
- its Pi runtime reaches `online`;
- a derived global agent records the correct source agent and runtime;
- the derived agent replies in the channel;
- the DAG returns `200` and passes structural validation;
- cleanup archives the derived agent and removes the sandbox/runtime.

For a training source agent, additionally assert `start_session` is called once with the
binding's `session_ref`, the returned session/key pair stays on that source binding, the
real task ID is linked after enqueue, the derived agent uses `areal-default`, and
cleanup closes the session.

## Rollout and Migration

This design changes persistence and runtime ownership, so it requires a database
migration for source-agent lineage and/or the expanded binding mapping. Deploy schema,
shared sandbox creation service, provisioning state machine, and cleanup support
together behind an env-dispatch feature flag. Existing pending bindings created by the
pre-created-runtime implementation should be rejected or drained rather than silently
reinterpreted.

Rollback disables new derived provisioning, drains active derived dispatches, and
preserves source agents. It must not restore the old behavior for bindings that already
own a derived global agent.

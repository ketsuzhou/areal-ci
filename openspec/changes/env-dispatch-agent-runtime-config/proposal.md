## Why

The first external-runtime implementation successfully propagates model configuration,
but deployed env-dispatch sandboxes can still register Pi as offline while the frontend
sandbox flow produces an online runtime. Env-dispatch also binds the source agent
directly to a pre-created runtime, which cannot safely support concurrent isolated
dispatches or per-agent training sessions.

Env-dispatch must use the proven frontend sandbox lifecycle, discover the runtime only
after daemon registration, and create a derived global agent for each source-agent
sandbox pair.

## What Changes

- Replace the parallel env-dispatch sandbox-create path with the shared frontend sandbox
  creation service.
- Stop pre-creating `agent_runtime` rows; correlate and discover the daemon-registered
  runtime by daemon ID plus sandbox instance ID.
- On first address, resolve the source agent's external model credential or open its
  AReaL training session before sandbox creation.
- Open training `start_session` with the persistent source binding ID as `session_ref`,
  then link the normally inserted real task to that session after the derived agent is
  ready.
- Clone the addressed source agent into a new global derived agent permanently bound to
  the sandbox runtime, with queryable `source_agent_id` lineage.
- Store source/derived agent, sandbox, runtime, session reference, and credential-owner
  identities on the env-agent binding with single-flight provisioning.
- Route channel execution through the derived agent without changing the source agent's
  global runtime or other memberships.
- Archive derived agents and delete their sandbox/runtime/session during env-dispatch
  cleanup.
- Treat rollout errors, runtime-readiness timeouts, DAG timeouts, and invalid DAGs as
  client failures.

## Capabilities

### New Capabilities

- `env-dispatch-derived-agent-runtime`: frontend-equivalent sandbox provisioning,
  runtime discovery, derived-agent lineage, and per-agent credential/session isolation.

### Modified Capabilities

- `env-dispatch-agent-runtime-config`: caller-provided static runtime configuration is
  consumed by derived-agent provisioning rather than a pre-created runtime.
- `env-dispatch-training`: the training target obtains a unique `start_session`
  credential before sandbox creation through a backward-compatible `session_ref` API,
  links the real task after enqueue, and uses `areal-default` through the configured
  bridge URL.
- `env-dispatch-cleanup`: derived global agents and their owned runtime resources are
  included in idempotent cascade cleanup.

## Impact

- Database migrations for agent lineage and expanded env-agent binding state.
- Shared sandbox creation code used by frontend and env-dispatch.
- Env-dispatch provisioning, training-session, channel membership, task enqueue, and
  cleanup paths.
- Agent response DTOs may gain non-secret lineage fields.
- Python debug/client failure handling and deployed DAG verification.
- No new dependency. Provider, session, and bootstrap secrets remain write-only.

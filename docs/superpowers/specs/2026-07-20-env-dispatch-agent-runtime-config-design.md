---
comet_change: env-dispatch-agent-runtime-config
role: technical-design
canonical_spec: openspec
---

# EnvDispatch Per-Agent External Runtime Configuration

## Objective

Make a non-training agent created by scratch message env-dispatch able to answer
through a caller-selected external model. The request supplies model connection
settings per agent, the server retains them through lazy sandbox provisioning,
and sandboxd receives the same runtime object used by the working frontend
sandbox creation flow.

This delivery does not implement the training-agent `start_session` path. It
establishes a secret-safe internal policy boundary that the later training work
can extend without allowing caller credentials to override an AReaL session.

## Current Flow and Failure

Scratch message dispatch reserves an env and project, creates a group channel,
and inserts one `environment_agent_sandbox` binding per roster member. The
leader is provisioned immediately. Peers remain `pending` and are provisioned by
`routeEnvDispatchChannelAgent` on their first directed mention.

The binding currently retains only:

```json
{"template":"default"}
```

`provisionEnvDispatchAgent` consequently creates a daemon-enabled sandbox with
`MULTICA_DAEMON_ID`, but leaves `CreateSandboxInstanceInput.Runtime` empty. The
default sandbox template supplies the filesystem and programs; it does not
supply an external provider URL, credential, or model. Sandboxd can start the
daemon, but the agent has no usable inference backend.

The frontend path succeeds because its sandbox create request separately sends:

```json
{
  "runtime": {
    "base_url": "https://provider.example/v1",
    "api_key": "<secret>",
    "model": "provider-model"
  }
}
```

Sandboxd already understands these keys. The server-side gap is policy capture
and propagation, not a new sandboxd integration.

## Public Request Contract

Caller-supplied runtime configuration is accepted only for scratch message
dispatch and is owned by one agent:

```json
{
  "mode": "scratch",
  "dispatch_type": "message",
  "agent_id": "<agent-id>",
  "message": {"content": "..."},
  "per_agent_env": {
    "<agent-id>": {
      "runtime": {
        "base_url": "https://provider.example/v1",
        "api_key": "<secret>",
        "model": "provider-model"
      }
    }
  }
}
```

A runtime-only scratch entry is valid and resolves to template `default`.
Existing explicit `template` and `base_env_id` selection remains compatible.

Branch source selection uses top-level `env_id`. The branch copies the source
channel and `sandbox_config`, then clones the trigger-selected source sandbox.
Therefore branch inherits the source binding's provider configuration and does
not need a per-agent `base_env_id` or repeated runtime credential. A branch
request containing a new runtime object is rejected instead of silently ignored.

Runtime configuration for `train_agent_id` is also rejected. Training proxy
configuration belongs to the later server-minted `start_session` path.

## Internal Types and Secret Boundary

Introduce a typed runtime value at the service boundary:

```go
type ExternalModelRuntime struct {
    BaseURL string `json:"base_url"`
    APIKey  string `json:"api_key"`
    Model   string `json:"model"`
}
```

`PerAgentEnvSpec` carries an optional pointer to this value so omission is
distinguishable from an explicitly supplied empty object.

Template resolution produces a separate internal policy:

```go
type ResolvedPerAgentSandboxPolicy struct {
    Template string
    Runtime  *ExternalModelRuntime
}
```

This policy must not be represented by `SandboxInstanceRef`. That type is part
of `EnvRollout` and is serialized as `sandbox_refs` or `agent_sandbox_refs` in
the HTTP response. Reusing its `RuntimeMetadata` field would make the API key
reachable through a public response DTO.

The env-dispatch dependency seam should pass resolved policy maps to channel
creation. Only the channel adapter serializes the internal policy into binding
JSON. Public response assembly never receives this object.

## Validation and Normalization

Validation occurs before DB-backed rollout creation:

1. Detect whether a runtime object was supplied.
2. Require `mode=scratch` and `dispatch_type=message`.
3. Reject the policy when its map key equals `train_agent_id`.
4. Trim all three values and require each to be non-empty.
5. Parse `base_url`; require an absolute URL whose scheme is `http` or `https`
   and whose host is non-empty.
6. Apply the existing agent workspace/squad membership validation.
7. Resolve an explicit template/base env, or select `default` when runtime is the
   only sandbox policy input.

Validation errors identify the agent and invalid field but never include a
field value. Any validation failure occurs before env, project, channel,
binding, runtime, sandbox, or task creation.

An entry that contains none of `template`, `base_env_id`, or `runtime` remains
invalid. An explicitly present but empty `runtime: {}` is invalid rather than
being treated as omission.

## Persistence and Provisioning Data Flow

After normalization, channel binding creation writes canonical JSON:

```json
{
  "template": "default",
  "runtime": {
    "base_url": "https://provider.example/v1",
    "api_key": "<secret>",
    "model": "provider-model"
  }
}
```

The existing `environment_agent_sandbox.sandbox_config` JSONB column is used;
no schema migration is needed. It is the correct persistence point because a
pending peer may be addressed after the originating HTTP request or a server
restart.

Both immediate leader provisioning and first-mention peer provisioning already
converge on `provisionEnvDispatchAgent`. That function must:

1. Claim the binding using the existing single-flight state transition.
2. Strictly decode the stored policy.
3. Fail the binding as retryable if policy decoding or runtime validation fails;
   it must not fall back to an unconfigured/default agent runtime.
4. Marshal only the canonical runtime object into
   `CreateSandboxInstanceInput.Runtime`.
5. Use the same create input in both the ordinary create path and the clone
   `CreatePayload` path.

The lifecycle adapter already persists runtime metadata on the sandbox instance
and emits it in the sandbox create job. Sandboxd already merges the three runtime
keys before starting the daemon. No sandboxd code path or key naming changes.

Branch channel copy already copies `sandbox_config` verbatim into destination
pending bindings and records the source sandbox instance. The existing lazy
clone therefore inherits the external model policy once provisioning consumes
the stored runtime object.

## Failure, Retry, and Cleanup Semantics

- Invalid request policy returns a validation error before writes.
- Invalid stored policy marks provisioning `failed`; a later mention can retry
  after the policy is repaired, but never routes to the shared default runtime.
- Sandbox or runtime creation failure follows current compensation: delete the
  pre-created runtime/sandbox where applicable and retain a retryable failed
  binding.
- Concurrent first mentions continue to use the binding claim as the
  serialization point; exactly one winner reads and applies the policy.
- Env/channel cleanup deletes the binding and therefore removes its retained
  inline provider credential under existing lifecycle rules.
- Idempotency replay returns the original env-dispatch response and never needs
  to reconstruct or expose runtime policy.

## Credential Handling

The API key is write-only at the env-dispatch public boundary:

- never attach runtime policy to `SandboxInstanceRef` or another response DTO;
- never include stored configuration in binding status responses;
- never format runtime values into validation/provisioning errors;
- never log request or binding policy objects;
- use synthetic values in tests and scan serialized responses/errors for the
  sentinel secret.

The key remains in binding JSON and existing sandbox runtime metadata/job
payloads because lazy provisioning and sandbox startup require it. This matches
the current frontend sandbox security model and trusted database/job access
boundary. Introducing an encrypted credential-reference service is valuable
hardening but requires a separate migration and lifecycle design.

## Test Strategy

### Request and service validation

- Parse and map a complete nested runtime object.
- Accept runtime-only scratch policy and resolve template `default`.
- Preserve explicit template and base-env behavior.
- Reject missing/blank fields, relative URLs, non-HTTP schemes, and empty hosts.
- Reject runtime for `train_agent_id`, branch, and non-message dispatch.
- Assert validation failures leave all resource counters unchanged.

### Binding and provisioning

- Persist two squad members' distinct canonical policies without cross-over.
- Prove the leader create input contains its exact runtime object.
- Complete the original request, then first-mention a pending peer and prove its
  create input comes from stored binding JSON.
- Exercise concurrent mentions and assert one sandbox/runtime/session.
- Copy a branch channel and prove the inherited policy reaches clone create
  payload without being resupplied.
- Corrupt stored binding JSON and assert retryable failure with no default
  runtime fallback.

### Non-disclosure and compatibility

- Search success response JSON, validation response JSON, provisioning errors,
  and captured structured logs for a synthetic sentinel key.
- Confirm callers without `runtime` retain existing behavior and response shape.
- Confirm the standalone Python client's generic `per_agent_env` serialization
  preserves the nested object; change it only if the test reveals a gap.

### Deployed verification

After deployment, use a fresh non-training agent and a rotated provider
credential. Dispatch a scratch message with runtime-only per-agent policy, wait
for the sandbox daemon and agent response, then fetch the channel DAG. Success
requires a visible agent reply, no shared default-runtime routing, and a
structurally valid DAG. Never reuse or record a credential previously exposed in
chat or source control.

## Rollout and Rollback

Deploy request parsing, validation, binding serialization, and provisioning
consumption together. The request change is additive; callers that omit runtime
remain compatible. No database migration or backfill is required.

Rollback can restore the previous server. Older code ignores the extra runtime
field in binding JSON and continues reading the template. Any external-model
scratch agents created through the new request would lose model propagation
after rollback, which is an expected capability rollback rather than data
corruption.

## Deferred Training Extension

The later training change will recognize `train_agent_id` on first task creation,
call `start_session`, and construct a server-owned runtime policy with model
`areal-default`, base URL `http://db_bridge_stub:9100/v1`, and the returned
session API key. It should reuse the internal policy-to-binding/provisioning
boundary introduced here, while keeping caller-supplied runtime forbidden for
the training target.

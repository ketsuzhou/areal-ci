# Comet Design Handoff

- Change: env-dispatch-agent-runtime-config
- Phase: design
- Mode: compact
- Context hash: 88c0a1368f007aa8d59bee17b32d4a50ba1353056eef0da061c9d2f7b75f76bd

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-dispatch-agent-runtime-config/proposal.md

- Source: openspec/changes/env-dispatch-agent-runtime-config/proposal.md
- Lines: 1-50
- SHA256: 01b74731bb85e7b22403d6245f673dd71b5dcdb086720edb962e336f299f94ef

```md
## Why

`POST /api/v1/env-dispatch` can select a per-agent sandbox template, but it
cannot carry the external model configuration that the frontend sandbox flow
already uses. As a result, an isolated non-training agent runtime starts without
`base_url`, `api_key`, or `model` and cannot answer the initial dispatch or a
later directed mention.

## What Changes

- Extend each `per_agent_env.<agent_id>` entry with an optional `runtime` object
  containing `base_url`, `api_key`, and `model`.
- Validate the external runtime configuration as one complete, non-empty set and
  reject its use for the current `train_agent_id` until the training-session
  path is implemented.
- Persist the per-agent runtime configuration with the env-agent sandbox binding
  so leader provisioning and lazy first-mention peer provisioning use the same
  policy.
- Pass the runtime configuration into sandbox-instance create and clone payloads
  without returning or logging the API key.
- Preserve existing behavior when `runtime` is omitted.

## Capabilities

### New Capabilities

- `env-dispatch-agent-runtime-config`: Per-agent external model configuration for
  env-dispatch sandbox provisioning, including validation, lazy provisioning,
  compatibility, and secret non-disclosure requirements.

### Modified Capabilities

<!-- None. The related message-channel and sandbox-lifecycle changes have not
     been synced into the repository's main specs, so this change defines a
     focused capability without depending on active change-local specs. -->

## Impact

- Public API: additive `per_agent_env.<agent_id>.runtime` request object on
  `POST /api/v1/env-dispatch`; response schemas remain unchanged.
- `multica/server`: env-dispatch request mapping and validation, per-agent binding
  policy persistence, and sandbox create/clone provisioning.
- Tests: handler parsing/validation, service policy propagation, first-mention
  single-flight provisioning, and API-key non-disclosure.
- `customized_areal/tree_search`: the existing generic `per_agent_env` request
  serialization can carry the nested object; only focused compatibility tests or
  examples are expected unless implementation reveals a client gap.
- No new dependency or database migration is expected.
- Training-agent `start_session` integration is explicitly deferred to a
  separate change.

```

## openspec/changes/env-dispatch-agent-runtime-config/design.md

- Source: openspec/changes/env-dispatch-agent-runtime-config/design.md
- Lines: 1-152
- SHA256: 25725618f22b30b929daa1ce0479b3792e87cffd5ac514c6a814f2dbe4a85d65

[TRUNCATED]

```md
## Context

The frontend's sandbox creation flow already sends a `runtime` object with
`base_url`, `api_key`, and `model`. Sandboxd merges those values into the
runtime environment before starting the in-sandbox daemon. Env-dispatch uses
the same sandbox lifecycle service, but its per-agent policy currently preserves
only a template in `environment_agent_sandbox.sandbox_config`; provisioning
therefore creates a daemon-enabled sandbox without model configuration.

Message dispatch also has two provisioning times. The roster leader is
provisioned during the initial dispatch, while peers remain pending and are
provisioned on their first directed mention. Any runtime policy must therefore
survive beyond the HTTP request and be consumed by the existing single-flight
provisioner.

## Goals / Non-Goals

**Goals:**

- Accept a typed per-agent external model configuration under
  `per_agent_env.<agent_id>.runtime`.
- Apply exactly the same policy to immediate leader provisioning and lazy peer
  provisioning, including the clone path.
- Preserve existing env-dispatch behavior when no runtime object is supplied.
- Prevent API-key disclosure through responses, errors, and structured logs.
- Prove that a non-training agent can answer through the configured external
  model in a deployed end-to-end test.

**Non-Goals:**

- Calling AReaL `start_session` or provisioning the `areal-default` proxy for a
  training target.
- Changing ordinary, non-env-dispatch channel runtime selection.
- Introducing a new credential vault, encryption scheme, dependency, or database
  migration.
- Changing sandboxd's existing runtime key names or process startup behavior.
- Extending this first delivery beyond `dispatch_type=message` interactions.
- Overriding a copied source binding's model configuration during branch.

## Decisions

### Use a nested per-agent runtime object

The request shape is:

```json
{
  "per_agent_env": {
    "<agent_id>": {
      "template": "default",
      "runtime": {
        "base_url": "https://provider.example/v1",
        "api_key": "<secret>",
        "model": "provider-model"
      }
    }
  }
}
```

This matches the sandbox API's runtime vocabulary and allows squad members to
use different providers. A top-level runtime object was rejected because its
ownership is ambiguous for squads and would require inheritance/override rules.

### Validate and canonicalize before creating rollout state

The handler/service boundary will use a typed runtime policy rather than passing
unvalidated arbitrary JSON. For scratch message dispatch, when `runtime` is
present, all three trimmed values must be non-empty and `base_url` must be an
absolute HTTP(S) URL. A runtime-only per-agent entry resolves to template
`default`; explicit template and base-env selection remain supported. A
malformed or partial object fails with a validation error before an env, project,
channel, binding, runtime, or sandbox is created.

Branch source selection remains the top-level `env_id`. Branch copies the source
channel and binding policy, so it inherits runtime configuration and rejects a
caller-supplied runtime override rather than accepting an unused credential.

Runtime configuration for `train_agent_id` is rejected in this change. Silently
ignoring it could run a training target against the wrong model; accepting it

```

Full source: openspec/changes/env-dispatch-agent-runtime-config/design.md

## openspec/changes/env-dispatch-agent-runtime-config/tasks.md

- Source: openspec/changes/env-dispatch-agent-runtime-config/tasks.md
- Lines: 1-52
- SHA256: 2e792420608c10a07bed9e797e9e33c773532492251bc0abdeb02e2c9dab0076

```md
## 1. Request Contract and Validation Tests

- [ ] 1.1 Add handler tests that parse
  `per_agent_env.<agent_id>.runtime` and map `base_url`, `api_key`, and `model`
  to the service input without exposing them in the response.
- [ ] 1.2 Add service tests for complete runtime configuration, missing/blank
  fields, invalid URL schemes, runtime on `train_agent_id`, and no-resource
  side effects on validation failure.
- [ ] 1.3 Add backward-compatibility tests for per-agent environment entries that
  omit `runtime`.

## 2. Runtime Policy Propagation Tests

- [ ] 2.1 Add channel-binding tests proving each agent's canonical runtime policy
  is persisted with its template and remains isolated from other squad members.
- [ ] 2.2 Add provisioning tests proving immediate leader create and lazy peer
  first-mention create pass the stored runtime object to the sandbox lifecycle.
- [ ] 2.3 Add clone/retry and concurrent first-mention tests proving one canonical
  runtime policy is preserved by the existing single-flight path.
- [ ] 2.4 Add failure-path assertions that responses, errors, and structured logs do
  not contain the synthetic API key.

## 3. Server Implementation

- [ ] 3.1 Add typed external runtime request/service structures and atomic
  validation for complete values, absolute HTTP(S) URL, and non-training target.
- [ ] 3.2 Carry the validated runtime policy through per-agent environment
  resolution and serialize it into the existing env-agent sandbox binding
  configuration.
- [ ] 3.3 Decode the stored policy in `provisionEnvDispatchAgent` and populate
  `CreateSandboxInstanceInput.Runtime` for create and clone payloads.
- [ ] 3.4 Audit env-dispatch response/error/log paths and prevent runtime secret
  disclosure without changing response schemas.

## 4. Client Compatibility and Documentation

- [ ] 4.1 Verify the standalone MultiCA client's generic `per_agent_env`
  serialization carries the nested runtime object unchanged; add a focused test
  or minimal client adjustment only if needed.
- [ ] 4.2 Add a secret-free request example documenting per-agent external model
  configuration and the current rejection of caller runtime for
  `train_agent_id`.

## 5. Verification

- [ ] 5.1 Run targeted Go handler/service/provisioning tests and the relevant
  Python client tests; record any unavailable toolchain or integration suite.
- [ ] 5.2 Run repository formatting/lint checks appropriate to the changed files
  and update the code graph with `graphify update .`.
- [ ] 5.3 After deployment, dispatch to a fresh non-training agent with a rotated
  external provider credential, confirm the agent replies in the channel, and
  validate that the returned channel DAG is structurally valid.

```

## openspec/changes/env-dispatch-agent-runtime-config/specs/env-dispatch-agent-runtime-config/spec.md

- Source: openspec/changes/env-dispatch-agent-runtime-config/specs/env-dispatch-agent-runtime-config/spec.md
- Lines: 1-132
- SHA256: 3d0c36a22dd41034a01a7fa8ff334721170b07e61d16fe2d9596b0638692b47d

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Per-agent external runtime request contract

For scratch `dispatch_type=message`, `POST /api/v1/env-dispatch` SHALL accept an
optional `runtime` object within each `per_agent_env.<agent_id>` entry. The
object SHALL contain `base_url`, `api_key`, and `model`, and SHALL apply only to
the agent identified by that map key. A complete runtime object SHALL be a valid
per-agent scratch policy without `template` or `base_env_id` and SHALL resolve
to template `default`.

#### Scenario: A valid external runtime is accepted

- **WHEN** a non-training agent's scratch per-agent environment entry contains a
  complete runtime object
- **THEN** env-dispatch accepts the request and associates that runtime policy
  only with the identified agent, using template `default` when neither an
  explicit template nor base environment is present

#### Scenario: Different squad agents use different runtime policies

- **WHEN** two non-training squad members have different runtime objects
- **THEN** each member's sandbox is provisioned with its own configured base URL,
  API key, and model without inheriting the other member's values

### Requirement: External runtime validation is atomic

When a runtime object is present, the system MUST require non-empty
`base_url`, `api_key`, and `model` values after trimming whitespace, and
`base_url` MUST be an absolute HTTP(S) URL. Invalid runtime configuration SHALL
fail before any rollout resource is created.

#### Scenario: A partial runtime object is rejected without side effects

- **WHEN** a per-agent runtime object omits or empties any required field
- **THEN** the request fails with a validation error and creates no env, project,
  channel, binding, runtime, sandbox, or agent task

#### Scenario: An invalid base URL is rejected without side effects

- **WHEN** a per-agent runtime object contains a relative or non-HTTP(S) base URL
- **THEN** the request fails with a validation error before rollout creation

#### Scenario: Caller runtime is rejected for a training target

- **WHEN** a runtime object is supplied for the agent identified by
  `train_agent_id`
- **THEN** the request fails with a validation error and does not substitute the
  caller's external model for the future AReaL training proxy flow

#### Scenario: Caller runtime is rejected for branch

- **WHEN** a branch message dispatch supplies a per-agent runtime object
- **THEN** the request fails before rollout creation because branch inherits the
  source binding runtime policy and runtime override is out of scope

### Requirement: Runtime policy survives lazy provisioning

The system SHALL persist each validated runtime policy with the agent's
env-dispatch sandbox binding and SHALL use that policy whenever the binding is
provisioned. The initial leader and a pending peer's first directed mention
SHALL use the same provisioning path and sandbox runtime payload contract.

#### Scenario: Initial leader sandbox receives external model configuration

- **WHEN** a scratch message dispatch provisions its leader during initial
  dispatch
- **THEN** the sandbox create payload contains that leader's exact validated
  `base_url`, `api_key`, and `model`, and the resulting task uses the binding
  runtime

#### Scenario: First mention provisions a pending peer with stored configuration

- **WHEN** a pending peer is first directly mentioned after the original HTTP
  request has completed
- **THEN** exactly one sandbox is created with the peer's stored runtime policy,
  the peer task uses the binding runtime, and the shared default runtime is not
  used

#### Scenario: Concurrent first mentions provision once

```

Full source: openspec/changes/env-dispatch-agent-runtime-config/specs/env-dispatch-agent-runtime-config/spec.md

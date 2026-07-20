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

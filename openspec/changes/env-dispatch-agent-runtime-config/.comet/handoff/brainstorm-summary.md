# Brainstorm Summary

- Change: env-dispatch-agent-runtime-config
- Date: 2026-07-20

## Confirmed Technical Approach

- The public request uses `per_agent_env.<agent_id>.runtime` with `base_url`,
  `api_key`, and `model`.
- This change supports non-training `dispatch_type=message` agents only and
  rejects caller runtime configuration for `train_agent_id`.
- Runtime policy must survive the original HTTP request so both immediate leader
  provisioning and lazy first-mention peer provisioning can consume it.
- Introduce a typed internal resolved sandbox policy
  separate from `SandboxInstanceRef`, persist its canonical JSON in the existing
  env-agent binding, and pass only the runtime JSON to sandbox lifecycle create
  and clone payloads. This prevents secrets from entering public rollout DTOs.
- Alternative: overload `SandboxInstanceRef.RuntimeMetadata`. Rejected as unsafe
  because that type is serialized in env-dispatch responses.
- Alternative: introduce an encrypted credential-reference table. Deferred as a
  separate security capability because it requires a migration and lifecycle
  design beyond this focused change.
- A scratch runtime-bearing per-agent entry may omit `template` and
  `base_env_id`; provisioning then uses template `default`. Existing explicit
  template/base-env selection remains compatible.
- Branch source ownership remains top-level `env_id`; branch copies the source
  channel/bindings and clones the trigger agent's source sandbox, so a per-agent
  `base_env_id` is not required for branch.
- Scratch external-model agents explicitly provide the runtime policy in the
  request. The default template does not imply provider configuration.
- Branch inherits runtime policy from copied source bindings. A branch request
  that supplies a new runtime policy is rejected rather than ignored; runtime
  override during branch is out of scope.

## Key Trade-offs and Risks

- The API key must remain available in binding JSON until a pending agent is
  provisioned; use existing DB access controls and prohibit response/error/log
  disclosure.
- A distinct internal policy type requires a small interface change but creates
  a reliable secret boundary.
- The lifecycle adapter already persists sandbox runtime metadata and sandboxd
  already consumes the three expected keys, so no sandboxd behavior change is
  required.

## Testing Strategy

- TDD at handler/service boundaries for parsing, validation, atomic rejection,
  backward compatibility, and training-target rejection.
- Binding/provisioning tests for per-agent isolation, leader and first-mention
  peer runtime payloads, clone/retry, and concurrent single-flight behavior.
- Negative assertions using synthetic secrets across JSON responses and errors.
- Deployed end-to-end test with a rotated provider key: agent reply plus valid
  channel DAG.

## Spec Patches

- Applied: allow a scratch per-agent entry containing a complete runtime object without
  `template` or `base_env_id`, and normatively default its template to `default`.
- Applied: clarify that branch source selection uses top-level `env_id`; per-agent
  `base_env_id` is not a branch requirement.
- Applied: limit caller-supplied runtime configuration to scratch message dispatch and
  reject it on branch, where the copied source binding is authoritative.

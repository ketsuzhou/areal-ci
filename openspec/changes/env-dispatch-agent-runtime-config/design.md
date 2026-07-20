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
would pre-empt the later `start_session` design.

### Persist the canonical runtime policy in the env-agent binding

The canonical policy is stored alongside `template` in the existing
`environment_agent_sandbox.sandbox_config` JSONB value. The provisioning claim
then reads the stored policy and sets `CreateSandboxInstanceInput.Runtime` for
both create and clone payloads. This reuses the binding as the established
single source of truth and preserves retry/idempotency behavior.

Keeping the policy only in request memory was rejected because pending peers may
be mentioned minutes later or after a server restart. Creating every peer
sandbox eagerly was rejected because it breaks the leader-only wake contract.

### Reuse existing sandboxd runtime behavior

No new environment-variable mapping is introduced. The lifecycle job carries
the runtime object, and sandboxd's existing merge logic supplies `api_key`,
`base_url`, and `model` when it starts the daemon. This keeps env-dispatch
equivalent to the already-working frontend sandbox creation flow.

### Treat the API key as write-only at the public API boundary

The runtime policy is needed at rest until lazy provisioning, so this change uses
the existing database protection model for sandbox runtime metadata and binding
JSON. The key must not be included in env-dispatch responses, binding status
responses, errors, or structured logs. Tests use synthetic credentials and
assert non-disclosure.

Adding a new encrypted secret store was rejected for this focused delivery; it
would be a separate security capability and migration. Operational access to the
database and sandbox job payloads remains trusted as it is for frontend-created
sandbox runtime metadata today.

## Risks / Trade-offs

- [The binding must retain a provider credential until a pending peer is
  provisioned] -> Reuse current database access controls, never expose the value
  through API/logging, and delete it with the env-agent binding during normal
  cleanup. A dedicated encrypted credential reference can replace the inline
  value in a later change.
- [A retry could lose or alter runtime policy] -> Persist one canonical policy
  before provisioning and always load it from the claimed binding.
- [Leader and peer behavior could diverge] -> Route both through the existing
  `provisionEnvDispatchAgent` function and test both timing paths.
- [Runtime configuration could accidentally bypass training proxy setup] ->
  Reject a runtime object for `train_agent_id` until the training change owns
  that branch.
- [A valid-looking provider configuration may still be unreachable] -> Keep
  request validation structural; surface sandbox/task failure without echoing
  credentials, and require a deployed end-to-end reply test.

## Migration Plan

1. Deploy the additive request parsing, validation, binding persistence, and
   provisioning changes together.
2. Existing callers that omit `runtime` continue unchanged; no data backfill or
   database migration is required.
3. Update the standalone caller/example to send the nested per-agent object if a
   client-side gap is found.
4. Verify against a fresh non-training agent and synthetic/rotated provider
   credential in the deployed service.
5. Roll back the server code if needed. Existing binding JSON remains readable
   because unknown JSON fields are ignored by the old provisioner.

## Open Questions

- The later training-agent change must decide how `start_session` credentials
  are represented without accepting caller-supplied runtime values for the
  training target.
- Moving provider credentials from binding JSON to an encrypted secret reference
  is a separate hardening decision.

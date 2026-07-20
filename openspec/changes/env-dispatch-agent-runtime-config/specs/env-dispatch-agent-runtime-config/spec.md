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

- **WHEN** concurrent directed mentions target the same pending agent
- **THEN** the existing single-flight binding claim creates one sandbox, runtime,
  and channel-agent session using one canonical runtime policy

#### Scenario: Branch clone preserves configured runtime policy

- **WHEN** a branch lazily clones an agent sandbox from a copied source binding
  that has a configured external runtime policy
- **THEN** the clone create payload carries the inherited binding runtime policy,
  with the branch source selected by top-level `env_id`

### Requirement: External runtime credentials are not publicly disclosed

The system MUST treat the supplied API key as write-only at the env-dispatch API
boundary. It MUST NOT include the key in success responses, binding status
responses, validation errors, task errors, or structured logs.

#### Scenario: Successful dispatch does not echo the API key

- **WHEN** env-dispatch succeeds with an external runtime configuration
- **THEN** no response field contains the supplied API key

#### Scenario: Failed provisioning does not echo the API key

- **WHEN** sandbox or task provisioning fails after the runtime policy is stored
- **THEN** the client-visible error and structured server logs omit the supplied
  API key

### Requirement: Omitted runtime configuration remains backward compatible

The system SHALL preserve existing env-dispatch request and provisioning behavior
when a per-agent runtime object is omitted.

#### Scenario: Existing caller omits runtime

- **WHEN** an existing caller sends a valid per-agent environment spec without a
  runtime object
- **THEN** request validation, sandbox policy resolution, and response structure
  behave as before this change

### Requirement: Configured non-training agent can reply

A non-training env-dispatch agent whose external runtime is valid and reachable
SHALL be able to process the dispatched channel task using the configured model.

#### Scenario: Deployed agent responds through the external model

- **WHEN** a deployed message dispatch addresses a non-training agent with valid
  rotated external provider credentials and the sandbox daemon becomes ready
- **THEN** the agent posts a response in the env-dispatch channel and the channel
  DAG remains valid

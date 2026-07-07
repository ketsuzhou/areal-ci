## ADDED Requirements

### Requirement: Critic agent squad role

The system SHALL support a critic agent as a squad member, specified via
`env_dispatch` `critic_agent_id`, that evaluates the trained agent's output
and produces a scalar reward used as the real training signal.

The critic agent SHALL:
- be auto-spawned as a teammate when the trained agent's task reaches a
  terminal state (NOT pre-dispatched at env_dispatch time);
- receive the trained agent's output as input;
- produce a scalar reward in its completion result;
- be a regular agent (no special training behavior — the critic is a fixed
  LLM judge, not itself a training target).

#### Scenario: Critic auto-spawned on trained-task terminal

- **WHEN** a trained task reaches a terminal state AND `training_dispatch`
  has a `critic_agent_id` set AND the trained session is still open
- **THEN** the system creates a critic task with the trained agent's output
  as input, and the trained session is NOT closed

#### Scenario: No critic configured — D's behavior preserved

- **WHEN** `training_dispatch` has no `critic_agent_id`
- **THEN** the system does NOT auto-spawn a critic task and D's close hook
  fires on the trained task's terminal with `TRAINING_DEFAULT_REWARD`

### Requirement: Deferred session-close on critic-completion

When a critic is configured, the session-close hook SHALL fire on the
critic task's terminal transition (NOT the trained task's). The close hook
SHALL:
- read the critic's scalar reward from the critic task's result;
- call `SetReward(proxy_key, critic_reward)` then `EndSession(proxy_key)`,
  both with session-key Bearer auth;
- fall back to `TRAINING_DEFAULT_REWARD` if the critic produced no reward
  or failed;
- log RL errors without failing the close (best-effort, same as D).

#### Scenario: Critic produces reward — close fires on critic terminal

- **WHEN** a critic task reaches a terminal state AND its result carries a
  scalar reward
- **THEN** the system calls `SetReward` with the critic's reward, then
  `EndSession`, on the trained session (in that order, session-key auth)

#### Scenario: Critic fails — fallback to default reward

- **WHEN** a critic task reaches a terminal state but produced no reward
  (or the critic task itself failed)
- **THEN** the system calls `SetReward` with `TRAINING_DEFAULT_REWARD`, then
  `EndSession`, on the trained session

#### Scenario: Trained agent crashes before critic runs

- **WHEN** the trained task reaches a terminal state via failure/cancel AND
  a critic is configured
- **THEN** the system still auto-spawns the critic (with whatever output is
  available), and the close hook fires on the critic's terminal

### Requirement: env_id emission at session-open

The RL bridge client SHALL pass `env_id` to `start_session` when one is
available from `env_dispatch`. AReaL's `StartSessionRequest` SHALL accept
an optional `env_id` field and persist it on the session for trajectory
attribution.

#### Scenario: env_id passed at session-open

- **WHEN** a trained member task is created with a `training_dispatch` that
  carries an `env_id`
- **THEN** the system calls `start_session(task_id, env_id)` and AReaL
  associates the session with that environment

#### Scenario: env_id missing — session still opens

- **WHEN** no `env_id` is available (e.g. `training_dispatch.env_id` is
  empty)
- **THEN** the system calls `start_session(task_id)` without env_id, and
  the session proceeds without environment attribution

### Requirement: Entropy recording from proxied LLM traffic

AReaL's experimental openai-proxy SHALL capture logprobs from proxied LLM
traffic by requesting `logprobs=true` on all calls, so entropy is computable
at session end. This SHALL be transparent to multica — multica's pi runtime
does not need to request logprobs or know about entropy.

If the upstream LLM does not support logprobs, the proxy SHALL log the
absence and continue (entropy is enrichment, not critical to trajectory
validity).

#### Scenario: Logprobs captured for entropy

- **WHEN** a trained session's LLM traffic flows through the proxy
- **THEN** the proxy requests `logprobs=true` on each call and stores the
  logprobs for later entropy computation

#### Scenario: Upstream LLM doesn't support logprobs

- **WHEN** the upstream LLM returns an error or empty result for
  `logprobs=true`
- **THEN** the proxy logs the absence and continues without entropy for
  that interaction (the trajectory remains valid)

### Requirement: Critic agent configuration

The `env_dispatch` payload SHALL accept an optional `critic_agent_id`
(UUID). When set, it MUST resolve to a real agent in the workspace. The
`critic_agent_id` is allowed with `squad_id` + `train_agent_id` (a team
member evaluating another team member).

#### Scenario: critic_agent_id validated

- **WHEN** `env_dispatch` carries a `critic_agent_id`
- **THEN** the system shape-validates the UUID and confirms it resolves to
  a real agent in the workspace; otherwise rejects with 400

#### Scenario: critic_agent_id empty

- **WHEN** `env_dispatch` has no `critic_agent_id`
- **THEN** the system proceeds with D's default-reward behavior (no critic,
  no auto-spawn, close on trained-terminal)

### Requirement: Persisted critic intent

The system SHALL persist `critic_agent_id` on `training_dispatch` (alongside
`train_agent_id` and `default_reward`) so the auto-spawn and deferred close
hooks can look it up by project_id when the trained task terminates.

#### Scenario: critic_agent_id persisted per rollout project

- **WHEN** `env_dispatch` carries `train_agent_id` + `critic_agent_id`
- **THEN** the system persists both on the `training_dispatch` row for each
  rollout project, and the auto-spawn hook reads `critic_agent_id` from
  there when the trained task terminates

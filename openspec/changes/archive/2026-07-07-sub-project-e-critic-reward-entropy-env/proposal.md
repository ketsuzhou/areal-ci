## Why

Sub-project D established the training-agent session lifecycle but explicitly deferred
four items to E (design doc §5): real reward computation, critic-agent squad membership,
entropy recording, and env_id emission back to AReaL. D emits a placeholder reward
(`TRAINING_DEFAULT_REWARD`, default 1.0) and sends no env_id, so every captured
trajectory is "valid" but uninformative — AReaL cannot actually train on the data. E
closes the loop: a critic agent in the squad produces the real reward, AReaL captures
entropy from proxied LLM traffic, and multica emits env_id at session-open so
trajectories are attributable to environments.

## What Changes

- **Critic agent squad role (multica)**: New squad role specified via `env_dispatch`
  `critic_agent_id`. The critic is auto-spawned as a teammate after the trained agent's
  task reaches a terminal state, evaluates the trained agent's output, and produces a
  scalar reward.
- **Deferred session-close hook (multica)**: D's close hook fires on the trained task's
  terminal state. When a critic is configured, E defers close until the critic task
  completes, so the critic's reward is the one written via `set_reward`.
- **Real reward from critic (multica)**: The close hook reads the critic's evaluation
  result and passes it to `set_reward` instead of D's placeholder. Falls back to the
  placeholder if the critic fails or produces no reward.
- **env_id contract extension (AReaL + multica)** **BREAKING** (additive):
  `StartSessionRequest` gains an optional `env_id` field. multica passes the
  env_dispatch `env_id` at session-open. AReaL persists env_id on the session for
  trajectory attribution. Additive — old clients without env_id still work.
- **Entropy recording (AReaL)**: AReaL's experimental openai-proxy requests
  `logprobs=true` on all proxied LLM calls (transparent to multica) so entropy is
  computable at session end without multica's pi runtime opting in.
- **Failure-mode fallback**: Critic failure / missing reward / missing env_id / missing
  logprobs all degrade gracefully — the trajectory remains valid with whatever signal is
  available.

## Capabilities

### New Capabilities

- `critic-driven-training-signal`: Critic agent squad role + critic-computed reward +
  env_id emission + entropy recording. Closes the RL training loop by replacing D's
  placeholder reward with a real signal and enriching the session with environment
  attribution and entropy metadata.

### Modified Capabilities

- `training-session-lifecycle`: E's close hook defers session-close from
  trained-task-terminal to critic-task-terminal when a critic is configured. D's
  "Default placeholder reward on close" requirement becomes the fallback path (critic
  failure / no critic configured); when a critic is configured and succeeds, the
  critic's reward is written instead. (D is not yet archived — this delta notes the
  supersession; both deltas will land in the main spec when D and E are archived.)

## Impact

- **multica (Go, primary)**: New squad role (critic), critic auto-spawn on
  trained-terminal, deferred close hook on critic-terminal, env_id passing to RL client.
  Touches `internal/service/task.go` (close hook + critic spawn),
  `internal/handler/env_dispatch.go` + `internal/service/env_dispatch.go`
  (`critic_agent_id`), `internal/arealrl/client.go` (env_id in StartSession),
  `pkg/db/queries/training_dispatch.sql` + generated (persist critic_agent_id).
- **AReaL (Python, first contract change in the series)**:
  `areal/experimental/openai/proxy/server.py` `StartSessionRequest` gains `env_id`
  field; `proxy_rollout_server.py` / `proxy_gateway.py` request `logprobs=true` on
  proxied calls and persist logprobs per interaction. Sub-projects A/B/C/D were
  multica-only or AReaL-confirm-only — E is the first to require AReaL code changes.
- **db_bridge**: No new channels; existing `/rl/start_session` carries the new `env_id`
  field transparently as part of the JSON body.
- **Depends on**: Sub-project D (session lifecycle). D's T7-T9 must be complete before
  E's close-hook changes can be implemented. D is currently paused at
  `build_pause=plan-ready`.
- **Out of scope**: Critic agent training (the critic is a fixed LLM judge, not itself a
  training target); reward shaping beyond a single scalar; multi-critic ensembles;
  reward from external signals (test pass/fail, human eval) — these are sub-project F or
  later.

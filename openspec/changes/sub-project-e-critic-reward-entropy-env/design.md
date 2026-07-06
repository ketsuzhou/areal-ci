# Sub-project E — critic-driven reward + entropy + env_id (design)

Status: DRAFT (high-level — detailed technical design TBD in comet-design phase)
Date: 2026-07-06
Repos: multica `main` (primary); AReaL `master` (first contract change in series)

## 1. Goal

Replace D's placeholder reward with a real signal from a critic agent, and
enrich the session with env_id (environment attribution) and entropy
(from captured logprobs) so AReaL can actually train on the captured
trajectories.

## 2. Actors / flow

1. AReaL (external driver) → multica `POST /api/v1/env-dispatch` with
   `train_agent_id` AND `critic_agent_id` AND `env_id`.
2. multica dispatches the team as today (D's behavior). The critic is NOT
   pre-dispatched — it spawns later.
3. Trained member task created → D's open hook + E's env_id extension →
   `start_session(task_id, env_id)`.
4. Trained agent works → LLM traffic via AReaL proxy → E's proxy captures
   logprobs per interaction (transparent to multica).
5. Trained task reaches terminal state. **D would close here. E does NOT.**
6. E's auto-spawn: multica creates a critic task with the trained agent's
   output as input.
7. Critic agent runs, evaluates, produces a scalar reward in its result.
8. Critic task reaches terminal state → E's deferred close hook fires:
   `set_reward(critic_reward)` then `end_session`.
9. AReaL assembles trajectory `{reward: critic_reward, entropy: from
   logprobs, env_id: from start_session}` for training.

## 3. Decisions (locked in explore phase, 2026-07-06)

- **E1 Ownership** — multica owns critic dispatch + reward wiring + env_id
  passing; AReaL owns entropy capture (logprobs) + env_id persistence.
  (Mirrors D1.)
- **E2 Critic trigger** — auto-spawn on trained-task-terminal (deterministic,
  decouples from leader behavior). NOT pre-dispatched (avoids idle resource
  waste); NOT leader-delegates (non-deterministic for RL reproducibility).
- **E3 Critic identity** — `critic_agent_id` on `env_dispatch`, parallel to
  `train_agent_id`. The critic is a regular agent, not a special role.
- **E4 Reward granularity** — critic produces ONE scalar reward for the
  whole trajectory, applied to the last interaction via `set_reward`
  (matching D's per-session-close behavior). Per-interaction reward is out
  of scope (sub-project F or later).
- **E5 Close hook timing** — when a critic is configured, the trained task's
  terminal transition does NOT close the session. The critic task's terminal
  transition closes the trained session (with the critic's reward).
- **E6 env_id contract** — add optional `env_id` field to
  `StartSessionRequest`. multica passes env_dispatch `env_id` at
  session-open. No new endpoint. Additive — old clients without env_id still
  work.
- **E7 Entropy capture** — AReaL proxy requests `logprobs=true` on all
  proxied LLM calls (transparent to multica). If upstream LLM doesn't
  support logprobs, entropy is missing — logged, not fatal.
- **E8 Failure modes** —
  - Critic fails or produces no reward → fall back to
    `TRAINING_DEFAULT_REWARD`.
  - Trained agent crashes before critic runs → close with default reward
    (no critic output to use).
  - Entropy capture fails → log, continue (enrichment, not critical).
  - env_id missing → log, continue (attribution, not critical).

## 4. Layers / changes (high-level — detailed in comet-design)

### 4.1 Contract (multica handler + service)
- `EnvDispatchRequest` + `service.EnvDispatchInput` gain optional
  `critic_agent_id string` (UUID). Validation: allowed with `squad_id` +
  `train_agent_id`; must resolve to a real agent. Empty ⇒ D's behavior.

### 4.2 Persist critic intent (multica, DB)
- Extend `training_dispatch` (migration 153) with `critic_agent_id UUID`
  column (nullable). Persist when set.

### 4.3 Critic auto-spawn (multica service)
- On trained task terminal transition (in `CompleteTask`/`FailTask`/
  `CancelTask`), if `training_dispatch.critic_agent_id` is set AND the
  session is still open, create a critic task with the trained agent's
  output as input. Do NOT close the session here.
- The critic task carries a reference to the trained session (so the close
  hook knows which session to close).

### 4.4 Deferred close hook (multica service)
- On critic task terminal transition, read the critic's result for a scalar
  reward. Call `set_reward(critic_reward)` then `end_session` on the trained
  session.
- Fallback to `TRAINING_DEFAULT_REWARD` if the critic produced no reward or
  failed.
- If no critic was configured, D's close hook fires as before on the trained
  task's terminal.

### 4.5 env_id in RL client (multica Go)
- `arealrl.Client.StartSession` gains an `envID string` parameter. The
  request body includes `env_id` when non-empty.

### 4.6 AReaL proxy contract (AReaL Python)
- `StartSessionRequest` gains `env_id: str | None = None`. Persisted on the
  session.
- `proxy_rollout_server.py` / `proxy_gateway.py` request `logprobs=true` on
  proxied LLM calls. Logprobs stored per interaction for entropy computation
  at session end.

### 4.7 Config (multica)
- No new config required for critic (critic_agent_id comes from
  env_dispatch). `TRAINING_DEFAULT_REWARD` (from D) remains the fallback.

## 5. Out of scope (⇒ sub-project F or later)

- Critic agent training (critic is a fixed LLM judge, not a training target)
- Per-interaction reward (E writes one scalar on the last interaction)
- Multi-critic ensembles
- Reward from external signals (test pass/fail, human eval)
- Reward shaping / penalty terms / KL divergence regularization

## 6. Open items (to resolve in comet-design brainstorming)

1. **Critic task input shape**: how does the critic task receive the trained
   agent's output? As a context field? A linked task ID? A literal output
   string?
2. **Critic reward result shape**: where in the critic task's result does
   the scalar reward live? A new field? Parsed from output text?
3. **Critic task → trained session linkage**: how does the close hook find
   the trained session from the critic task? New DB column? Context field?
4. **Failure-mode for critic auto-spawn**: if the critic task creation
   itself fails, what happens? Close with default? Retry?
5. **AReaL logprobs performance**: does forcing `logprobs=true` on all
   calls materially increase proxy overhead or break any upstream LLM?
6. **env_id format**: is multica's env_id a UUID? A string? Does AReaL need
   to validate it?

## 7. Test strategy (high-level — detailed in comet-build Plan)

- multica Go unit tests: critic_agent_id validation; critic auto-spawn on
  trained-terminal; deferred close on critic-terminal; fallback on critic
  failure; env_id passed to RL client.
- AReaL Python tests: StartSessionRequest accepts env_id; proxy requests
  logprobs; logprobs persisted per interaction.
- db_bridge: smoke test that env_id flows through /rl/start_session.
- Cross-repo E2E: a trained session with critic produces a non-default
  reward + env_id + entropy in AReaL's trajectory export.

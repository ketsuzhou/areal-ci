# Comet Design Handoff

- Change: sub-project-e-critic-reward-entropy-env
- Phase: design
- Mode: compact
- Context hash: f2d21f33de5e97bf455857bb757c2d8350184ae2b8f6512c3064035750d95c50

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/sub-project-e-critic-reward-entropy-env/proposal.md

- Source: openspec/changes/sub-project-e-critic-reward-entropy-env/proposal.md
- Lines: 1-79
- SHA256: 952ac639fdd039e8e0cbb194710d175e09c3a80db081ba89f6364f3d6d900235

```md
## Why

Sub-project D established the training-agent session lifecycle but explicitly
deferred four items to E (design doc §5): real reward computation, critic-agent
squad membership, entropy recording, and env_id emission back to AReaL. D emits
a placeholder reward (`TRAINING_DEFAULT_REWARD`, default 1.0) and sends no
env_id, so every captured trajectory is "valid" but uninformative — AReaL
cannot actually train on the data. E closes the loop: a critic agent in the
squad produces the real reward, AReaL captures entropy from proxied LLM
traffic, and multica emits env_id at session-open so trajectories are
attributable to environments.

## What Changes

- **Critic agent squad role (multica)**: New squad role specified via
  `env_dispatch` `critic_agent_id`. The critic is auto-spawned as a teammate
  after the trained agent's task reaches a terminal state, evaluates the
  trained agent's output, and produces a scalar reward.
- **Deferred session-close hook (multica)**: D's close hook fires on the
  trained task's terminal state. When a critic is configured, E defers close
  until the critic task completes, so the critic's reward is the one written
  via `set_reward`.
- **Real reward from critic (multica)**: The close hook reads the critic's
  evaluation result and passes it to `set_reward` instead of D's placeholder.
  Falls back to the placeholder if the critic fails or produces no reward.
- **env_id contract extension (AReaL + multica)** **BREAKING** (additive):
  `StartSessionRequest` gains an optional `env_id` field. multica passes the
  env_dispatch `env_id` at session-open. AReaL persists env_id on the session
  for trajectory attribution. Additive — old clients without env_id still
  work.
- **Entropy recording (AReaL)**: AReaL's experimental openai-proxy requests
  `logprobs=true` on all proxied LLM calls (transparent to multica) so
  entropy is computable at session end without multica's pi runtime opting in.
- **Failure-mode fallback**: Critic failure / missing reward / missing env_id
  / missing logprobs all degrade gracefully — the trajectory remains valid
  with whatever signal is available.

## Capabilities

### New Capabilities

- `critic-driven-training-signal`: Critic agent squad role + critic-computed
  reward + env_id emission + entropy recording. Closes the RL training loop
  by replacing D's placeholder reward with a real signal and enriching the
  session with environment attribution and entropy metadata.

### Modified Capabilities

- `training-session-lifecycle`: E's close hook defers session-close from
  trained-task-terminal to critic-task-terminal when a critic is configured.
  D's "Default placeholder reward on close" requirement becomes the fallback
  path (critic failure / no critic configured); when a critic is configured
  and succeeds, the critic's reward is written instead. (D is not yet
  archived — this delta notes the supersession; both deltas will land in the
  main spec when D and E are archived.)

## Impact

- **multica (Go, primary)**: New squad role (critic), critic auto-spawn on
  trained-terminal, deferred close hook on critic-terminal, env_id passing to
  RL client. Touches `internal/service/task.go` (close hook + critic spawn),
  `internal/handler/env_dispatch.go` + `internal/service/env_dispatch.go`
  (`critic_agent_id`), `internal/arealrl/client.go` (env_id in StartSession),
  `pkg/db/queries/training_dispatch.sql` + generated (persist critic_agent_id).
- **AReaL (Python, first contract change in the series)**:
  `areal/experimental/openai/proxy/server.py` `StartSessionRequest` gains
  `env_id` field; `proxy_rollout_server.py` / `proxy_gateway.py` request
  `logprobs=true` on proxied calls and persist logprobs per interaction.
  Sub-projects A/B/C/D were multica-only or AReaL-confirm-only — E is the
  first to require AReaL code changes.
- **db_bridge**: No new channels; existing `/rl/start_session` carries the
  new `env_id` field transparently as part of the JSON body.
- **Depends on**: Sub-project D (session lifecycle). D's T7-T9 must be
  complete before E's close-hook changes can be implemented. D is currently
  paused at `build_pause=plan-ready`.
- **Out of scope**: Critic agent training (the critic is a fixed LLM judge,
  not itself a training target); reward shaping beyond a single scalar;
  multi-critic ensembles; reward from external signals (test pass/fail,
  human eval) — these are sub-project F or later.
```

## openspec/changes/sub-project-e-critic-reward-entropy-env/design.md

- Source: openspec/changes/sub-project-e-critic-reward-entropy-env/design.md
- Lines: 1-141
- SHA256: 808112c7efa5f78a0da9b6d0cde67a069e55643e9106ccecb41a29b3c2abe6cc

[TRUNCATED]

```md
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
```

Full source: openspec/changes/sub-project-e-critic-reward-entropy-env/design.md

## openspec/changes/sub-project-e-critic-reward-entropy-env/tasks.md

- Source: openspec/changes/sub-project-e-critic-reward-entropy-env/tasks.md
- Lines: 1-233
- SHA256: 473d6c94733e84d7018de115b6fbc4cbfabf0e52c6064888ad3919fcfd9e5d75

[TRUNCATED]

```md
# Tasks — sub-project-e-critic-reward-entropy-env

## Prior work (depends on sub-project D)

Sub-project D (session lifecycle) MUST be complete (T7-T9 done) before E's
close-hook and critic-spawn changes can be implemented. D is currently
paused at `build_pause=plan-ready` in
`openspec/changes/sub-project-d-session-lifecycle/`. See
`.superpowers/sdd/progress.md` for the cross-repo ledger.

T1-T6 of the overall training-agent effort are on multica `main`
(`7969187a..816d1e86c`). E's implementation will branch from D's tip once
D's T7-T9 land on multica `main`.

## Remaining work (this change)

### Task 1: Investigation — confirm seams for critic dispatch + entropy capture

No production code. Produce a short markdown note
(`docs/superpowers/notes/2026-07-06-E-seams.md`) answering:

- **1a. Critic auto-spawn seam.** Confirm where to inject the critic task
  creation on trained-task-terminal. D's close hook attaches to
  `CompleteTask`/`FailTask`/`CancelTask` in `internal/service/task.go` —
  can the critic-spawn hook attach at the same chokepoints (before D's
  close logic)? Or does it need a separate transition?
- **1b. Critic task → trained session linkage.** How does the deferred
  close hook (on critic-terminal) find the trained session's `proxy_key`?
  New DB column on the critic task? Context field? Join via
  `training_dispatch`?
- **1c. Critic reward result shape.** Where does the critic's scalar reward
  live in its task result? New field on `agent_task`? Parsed from output?
  Stored in `context`?
- **1d. env_id availability at session-open.** Confirm `env_id` is
  available on `training_dispatch` (or the task) at the time D's open hook
  fires. If not, where to thread it from.
- **1e. AReaL proxy logprobs path.** Confirm
  `areal/experimental/openai/proxy/proxy_rollout_server.py` /
  `proxy_gateway.py` can request `logprobs=true` on proxied calls and
  persist logprobs per interaction. Identify the exact insertion points.
- **1f. StartSessionRequest env_id field.** Confirm
  `areal/experimental/openai/proxy/server.py` `StartSessionRequest` can
  gain an optional `env_id` field without breaking existing callers
  (additive change).

**STOP-and-report** (begin report `BLOCKED:`) if: the critic auto-spawn
cannot be injected at the same chokepoints as D's close hook (e.g. the
terminal transition is in raw SQL like the runtime sweeper, bypassing
`FailTask`), OR AReaL's proxy cannot transparently capture logprobs (e.g.
the upstream LLM SDK doesn't expose logprobs). Otherwise begin `DONE:` with
the mapping. Commit the note.

### Task 2: Contract — `critic_agent_id` on env_dispatch (TDD)

**Files**: `internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`,
`internal/handler/env_dispatch_test.go`, `internal/service/env_dispatch_test.go`.

- [ ] Failing tests: request with `critic_agent_id` shape-validated (400 on
  malformed UUID); service accepts it; validation — allowed with `squad_id`
  + `train_agent_id`; equal to `agent_id` rejected (can't critique yourself);
  empty ⇒ unchanged behavior.
- [ ] Add `CriticAgentID string` to `EnvDispatchRequest` (json
  `critic_agent_id,omitempty`) and `service.EnvDispatchInput`; thread
  through the handler→service mapping.
- [ ] Handler UUID shape-check when present; service `validate()` rule.
- [ ] Run: `go test ./internal/handler/ ./internal/service/ -run 'EnvDispatch|Dispatch'`.
- [ ] Commit: `feat(env-dispatch): accept critic_agent_id (critic for trained agent)`.

### Task 3: Persist critic intent — extend `training_dispatch` (TDD)

**Files**: `server/migrations/153_training_dispatch_critic.up.sql`/`.down.sql`,
`server/pkg/db/queries/training_dispatch.sql` (extend),
`server/pkg/db/generated/training_dispatch.sql.go` (hand-written, mirror
sibling), `internal/service/env_dispatch.go` (+ deps method + adapter +
fake), tests.

- [ ] Migration: `ALTER TABLE training_dispatch ADD COLUMN critic_agent_id
  UUID NULL`.
- [ ] Queries: extend `CreateTrainingDispatch` to accept `critic_agent_id`;
  extend `GetTrainingDispatchByProject` to return it.
```

Full source: openspec/changes/sub-project-e-critic-reward-entropy-env/tasks.md

## openspec/changes/sub-project-e-critic-reward-entropy-env/specs/critic-driven-training-signal/spec.md

- Source: openspec/changes/sub-project-e-critic-reward-entropy-env/specs/critic-driven-training-signal/spec.md
- Lines: 1-138
- SHA256: 7b4f814afe3af40fca7661fae1cd7a13aacd8fe0003b7eb6b043fce452a319e7

[TRUNCATED]

```md
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
```

Full source: openspec/changes/sub-project-e-critic-reward-entropy-env/specs/critic-driven-training-signal/spec.md


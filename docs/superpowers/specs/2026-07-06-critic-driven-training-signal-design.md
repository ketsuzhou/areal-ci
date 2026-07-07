---
comet_change: sub-project-e-critic-reward-entropy-env
role: technical-design
canonical_spec: openspec
---

# Sub-project E — critic-driven reward + entropy + env_id (design)

Status: APPROVED (2026-07-06)
Date: 2026-07-06
Repos: multica `main` (primary); AReaL `master` (first contract change in series)

## 1. Goal

Replace D's placeholder reward with a real signal from a critic agent, and
enrich the session with env_id (environment attribution) and entropy
(from captured logprobs) so AReaL can actually train on the captured
trajectories. D emitted `TRAINING_DEFAULT_REWARD` (1.0) and sent no env_id;
every trajectory was "valid" but uninformative. E closes the loop.

## 2. Actors / flow

```
env_dispatch (train_agent_id + critic_agent_id + env_id)
    │
    ▼
[multica] dispatch squad: leader + trained + ... (critic NOT yet dispatched)
    │
    ▼  D's open hook + E's env_id extension
trained member task created → start_session(task_id, env_id)
    │                          ▲
    │              E6: env_id extension on StartSessionRequest
    ▼
trained agent works → LLM traffic via AReaL proxy
    │                          ▲
    │              E7: proxy injects logprobs=true (transparent to multica)
    ▼
trained task reaches terminal state (CompleteTask/FailTask/CancelTask)
    │
    │  D would close here. E does NOT.
    ▼  E2: maybeSpawnCriticTask fires (replaces D's close)
[multica] create critic task:
    agent_id = training_dispatch.critic_agent_id
    context.critic_of = {trained_task_id, proxy_key, session_id, project_id}
    prompt/input = includes trained agent's output (literal text)
    │
    ▼
critic agent runs, produces output text containing {"reward": <float>}
    │
    ▼  E5: deferred close hook fires on CRITIC terminal
maybeCloseTrainingSessionFromCritic:
    1. parse critic output → reward (fallback: TRAINING_DEFAULT_REWARD)
    2. read proxy_key from critic.context.critic_of
    3. SetReward(proxy_key, reward) → EndSession(proxy_key)
    │
    ▼
AReaL assembles trajectory {reward, entropy (from logprobs), env_id}
```

## 3. Decisions (locked)

### From explore phase (E1-E8)

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
- **E7 Entropy capture** — AReaL proxy injects `logprobs=true` on all
  proxied LLM calls (transparent to multica). If upstream LLM doesn't
  support logprobs, retry without logprobs and log the absence (graceful
  degradation).
- **E8 Failure modes** —
  - Critic fails or produces no reward → fall back to
    `TRAINING_DEFAULT_REWARD`.
  - Trained agent crashes before critic runs → still auto-spawn critic
    (with whatever output is available); close on critic terminal.
  - Critic auto-spawn itself fails → close trained session with default
    reward (don't let it hang).
  - Entropy capture fails → log, continue (enrichment, not critical).
  - env_id missing → log, continue (attribution, not critical).

### From brainstorming (6 open items resolved)

- **Item 1 — Critic task input shape**: The critic task receives the trained
  agent's output as **literal text in its prompt/input**. The critic is a
  regular agent — its prompt template includes the trained output. No new
  context field needed for input. Linkage to the trained task is via
  `context.critic_of.trained_task_id` (see item 3) — `parent_task_id` is
  NOT set, because `parent_task_id` (migration 055) is specifically the
  "retry back-pointer" set by `CreateRetryTask`, and overloading it for
  critique linkage would risk confusing retry logic.

- **Item 2 — Critic reward result shape**: The critic's prompt instructs it
  to output JSON `{"reward": <float 0.0-1.0>}` on the last line. multica
  parses the critic's output text to extract the reward. **Fallback**: if
  JSON parse fails or reward is out of range, use `TRAINING_DEFAULT_REWARD`.
  Rationale: keeps the critic a regular agent (no special tools), gives a
  clean extraction path, graceful degradation.

- **Item 3 — Critic task → trained session linkage**: Store
  `critic_of = {trained_task_id, proxy_key, session_id, project_id}` in the
  **critic task's `context` JSONB** (migration 003, no new column). The
  close hook reads `proxy_key` and `session_id` from there. `parent_task_id`
  is NOT set on the critic task (see item 1 — avoids overloading retry
  semantics).

- **Item 4 — Critic auto-spawn failure**: If `maybeSpawnCriticTask` fails
  (DB error, invalid agent_id, etc.), fall back to D's behavior:
  `SetReward(default) → EndSession` on the trained session, log the spawn
  failure. The trained session must not hang.

- **Item 5 — AReaL logprobs performance**: **Option A** — inject
  `logprobs=true` into the forwarded request in `_call_client_create`
  (`areal/experimental/openai/proxy/proxy_rollout_server.py`). The proxy
  already has `InteractionWithTokenLogpReward` infrastructure; this wires
  the chat-completions path to populate it. If the upstream LLM rejects
  `logprobs=true` (e.g. Anthropic-native without gateway), catch the error,
  retry without logprobs, log the absence. **Risk**: providers without
  logprobs support lose entropy for that interaction — acceptable (entropy
  is enrichment, not critical).

- **Item 6 — env_id format**: `env_id` is a **string** (matches multica's
  existing `EnvID` type). AReaL accepts `str | None = None` — **no
  validation** (AReaL is agnostic to the format). If `env_id` is empty,
  omit from request body (not send empty string). Additive — old clients
  without `env_id` still work. **No multica migration needed** — env_id is
  already on `env_dispatch.EnvID` and stored on `training_dispatch` (which
  D created); E just threads it through to the RL client.

## 4. Layers / changes

### 4.1 Contract (multica handler + service)

- `EnvDispatchRequest` + `service.EnvDispatchInput` gain optional
  `critic_agent_id string` (UUID). Handler shape-validates when present.
  Validation: allowed with `squad_id` + `train_agent_id`; must resolve to a
  real agent in the workspace; `critic_agent_id == train_agent_id` rejected
  (can't critique yourself); `critic_agent_id == agent_id` (single-agent
  critique) also rejected (critique requires a separate trained target).
  Empty ⇒ D's behavior exactly (no critic).

### 4.2 Persist critic intent (multica, DB)

- Migration 153: `ALTER TABLE training_dispatch ADD COLUMN critic_agent_id
  UUID NULL`.
- Queries: extend `CreateTrainingDispatch` and `GetTrainingDispatchByProject`
  to handle `critic_agent_id`. Hand-write generated Go (mirror sibling) —
  do NOT run `sqlc generate` repo-wide (D's codegen rule).

### 4.3 Critic auto-spawn (multica service, `internal/service/task.go`)

New `maybeSpawnCriticTask(ctx, trainedTask, trainingDispatch)` invoked from
the same chokepoints as D's close hook (`CompleteTask`/`FailTask`/
`CancelTask`), **replacing** D's close when a critic is configured. When
`trainingDispatch.CriticAgentID` is set AND the trained session is still
open:

1. Read the trained agent's output (from the task's result/output field).
2. Create a critic task:
   - `agent_id = trainingDispatch.CriticAgentID`
   - `parent_task_id` NOT set (see §3 item 1)
   - `context.critic_of = {trained_task_id, proxy_key, session_id,
     project_id}` (read from `trainedTask.context.areal_proxy`)
   - `prompt/input` includes the trained agent's output as literal text
3. **Do NOT close the trained session** — the close is deferred to the
   critic's terminal transition (§4.4). D's close hook is skipped for this
   trained task.

If spawn fails, fall back to D's close (`SetReward(default) → EndSession`)
on the trained session and log the failure. Idempotent: if a critic task
already exists for this trained task (check via a `critic_of.trained_task_id
= <trainedTask.ID>` lookup on existing tasks), skip.

When `trainingDispatch.CriticAgentID` is empty, D's original close hook
fires unchanged (no critic, no spawn, close on trained-terminal with
default reward).

### 4.4 Deferred close hook (multica service, `internal/service/task.go`)

New `maybeCloseTrainingSessionFromCritic(ctx, criticTask)` invoked from the
same chokepoints (`CompleteTask`/`FailTask`/`CancelTask`) when the
terminating task is a critic task (detected via `context.critic_of`
presence). Steps:

1. Read `proxy_key`, `session_id` from `criticTask.context.critic_of`.
2. Read critic's output text; parse JSON `{"reward": <float>}` from the
   last line. **Fallback**: `TRAINING_DEFAULT_REWARD` if parse fails or
   reward is out of `[0.0, 1.0]` range (or whatever range is configured).
3. `SetReward(proxy_key, reward)` then `EndSession(proxy_key)` (session-key
   Bearer auth, same as D).
4. RL errors logged, not fatal (best-effort, same as D).

**Routing summary** (which close hook fires when):

| Terminating task type | `critic_agent_id` configured | Hook that fires |
|-----------------------|------------------------------|-----------------|
| trained task          | not configured               | D's `maybeCloseTrainingSession` (default reward) |
| trained task          | configured, spawn succeeds   | `maybeSpawnCriticTask` (no close; deferred) |
| trained task          | configured, spawn fails      | D's `maybeCloseTrainingSession` (default reward, fallback) |
| critic task           | n/a (has `context.critic_of`) | `maybeCloseTrainingSessionFromCritic` (critic reward) |
| other task            | n/a                           | neither (no `context.areal_proxy`, no `context.critic_of`) |

The critic task does NOT carry `context.areal_proxy` (it's not a trained
task), so D's `maybeCloseTrainingSession` is a no-op on critic-terminal —
only `maybeCloseTrainingSessionFromCritic` fires.

### 4.5 env_id in RL client (multica Go, `internal/arealrl/client.go`)

- `StartSession(ctx, taskID, envID string) (SessionCreds, error)` — add
  `envID` parameter. Marshal `env_id` into request body when non-empty;
  omit when empty.

### 4.6 Session-open hook — pass env_id (multica, `internal/service/task.go`)

D's `maybeOpenTrainingSession` reads `env_id` from `training_dispatch` (or
`env_dispatch` input) and passes it to `arealrl.Client.StartSession`. When
`env_id` is empty, omit (D's current behavior).

### 4.7 AReaL proxy contract (AReaL Python)

Two changes in `areal/experimental/openai/proxy/`:

1. **`server.py`** — `StartSessionRequest` gains `env_id: str | None = None`.
   Persisted on `SessionData` for trajectory attribution.

2. **`proxy_rollout_server.py`** — in `_call_client_create`, inject
   `logprobs=True` into the forwarded `chat.completions.create` call. The
   response logprobs are stored in `InteractionWithTokenLogpReward` (type
   already exists). If the upstream returns an error on `logprobs=True`,
   retry without logprobs and log the absence.

### 4.8 Config (multica)

- No new config required for critic (`critic_agent_id` comes from
  `env_dispatch`).
- `TRAINING_DEFAULT_REWARD` (from D) remains the fallback for critic
  failure / no critic / parse failure.
- Reward validation range is fixed at `[0.0, 1.0]` for E. If a future
  change needs a different range, add `CRITIC_REWARD_RANGE_MIN`/`MAX`
  config then — out of scope for E.

## 5. Out of scope (⇒ sub-project F or later)

- Critic agent training (the critic is a fixed LLM judge, not itself a
  training target)
- Per-interaction reward (E writes one scalar on the last interaction)
- Multi-critic ensembles
- Reward from external signals (test pass/fail, human eval)
- Reward shaping / penalty terms / KL divergence regularization
- Critic agent tooling (the critic uses JSON output, not a structured tool)
- Reaper for orphaned trained sessions (if critic task is lost, the session
  hangs until AReaL's session timeout — future hardening)

## 6. Test strategy

### multica Go unit tests (TDD)

- `env_dispatch` `critic_agent_id` validation (shape, allowed combos,
  self-critique rejected, empty ⇒ D behavior).
- `training_dispatch` persists `critic_agent_id` (migration 153).
- `maybeSpawnCriticTask`:
  - trained-terminal + critic set + session open → critic task created
    with `parent_task_id` + `context.critic_of` populated.
  - no critic configured → no spawn (D's close fires).
  - idempotent (second terminal transition doesn't spawn again).
  - spawn fails → D's close fires with default reward, failure logged.
- `maybeCloseTrainingSessionFromCritic`:
  - critic-terminal + `context.critic_of` present → `SetReward(parsed
    reward)` then `EndSession`, session-key auth, order asserted.
  - critic output unparseable → `SetReward(default)` fallback.
  - reward out of range → fallback.
  - RL error → logged, transition still completes.
  - non-critic task terminal → D's close hook fires (no `context.critic_of`).
- `arealrl.Client.StartSession` includes `env_id` when non-empty; omits
  when empty.
- `maybeOpenTrainingSession` passes `env_id` from `training_dispatch`.

### AReaL Python tests (TDD)

- `StartSessionRequest` accepts `env_id`; old requests without it still
  work (additive).
- `SessionData` persists `env_id`.
- `_call_client_create` injects `logprobs=True` in forwarded request.
- Upstream error on `logprobs=True` → retry without logprobs, log absence.
- Logprobs in response → stored in `InteractionWithTokenLogpReward`.

### db_bridge smoke

- `/rl/start_session` carries `env_id` in JSON body (no new channel).

### Cross-repo E2E (if feasible)

- A trained session with critic produces a non-default reward + env_id +
  entropy in AReaL's `export_trajectories` output.

### Test runners / constraints (from D, still apply)

- multica Go: scope build/test to touched packages (pre-existing webpush
  `./...` failure; 16 pre-existing ON CONFLICT handler failures).
- Codegen rule: do NOT run `sqlc generate` repo-wide.
- AReaL Python: `uv run pytest` on proxy tests; `pre-commit run --files`
  before commit.
- Commit each task to multica `main` (multica-side) or areal `master`
  (areal-side); record commit hashes in `.superpowers/sdd/progress.md`.

## 7. Dependencies

- **Sub-project D** (session lifecycle) MUST be complete (T7-T9 done)
  before E's close-hook and critic-spawn changes can be implemented. D is
  currently paused at `build_pause=plan-ready`.
- T1-T6 of the overall training-agent effort are on multica `main`
  (`7969187a..816d1e86c`). E's implementation will branch from D's tip once
  D's T7-T9 land on multica `main`.
- AReaL-side changes (§4.7) are independent of D and can be developed in
  parallel, but must land before E's cross-repo E2E test.

## 8. Risks

- **Critic output parsing fragility**: LLM may not output clean JSON.
  Mitigation: fallback to default reward; future hardening could add a
  `submit_reward` tool to the critic agent (out of scope for E).
- **Upstream logprobs support**: Anthropic-native APIs don't support
  `logprobs`. Mitigation: graceful fallback (retry without logprobs);
  entropy is enrichment, not critical.
- **Orphaned trained sessions**: if the critic task is lost (e.g. DB
  corruption), the trained session hangs until AReaL's session timeout.
  Mitigation: future reaper (out of scope for E, noted in §5).
- **Critic latency**: the trained session stays open longer (until critic
  completes). May approach AReaL's session timeout for long critic runs.
  Mitigation: monitor; future work could add a max-critic-duration config.

---
comet_change: sub-project-d-session-lifecycle
role: technical-design
canonical_spec: openspec
---

# Sub-project D — training_agent session lifecycle (design)

Status: DRAFT (awaiting hard-gate approval)
Date: 2026-07-06
Repos: multica `main` (primary); AReaL `master` (confirm-only, likely no code change)

## 1. Goal

When AReaL dispatches an agent **team** with a designated training target,
multica opens an AReaL RL proxy session for that agent's run, launches its
runtime as `pi -p --provider areal --model areal-default --api-key <proxy_key>`
(base URL = configured `proxy_url`), and closes the session on completion by
writing a **default placeholder reward** then `end_session`. The trained
agent's LLM traffic routes through `http://db_bridge_stub:9100/v1` (multica's
in-network bridge stub) → AReaL gateway, where the trajectory is captured.

Teammates that are NOT the training target run on their normal provider,
untouched.

## 2. Actors / flow

1. AReaL (external driver) → multica `POST /api/v1/env-dispatch` with a
   **`train_agent_id`** marking the training target squad member.
2. multica dispatches the team as today (squad → leader task created; the
   leader spawns teammates later — **the trained member's task does not exist
   at dispatch time**).
3. When the trained member's task is later created (leader @mention delegation
   or `/api/agent/start` with `parent_task_id`), a multica **session-open
   hook** calls `start_session` (via the bridge) → `{session_id,
   session_api_key=proxy_key}` (group_size=1), stores `session_id` on the task,
   and injects the AReaL proxy config into the task `context`.
4. The daemon `execenv` reads that context and configures the runtime:
   `provider=areal`, `model=areal-default`, `api_key=proxy_key`,
   `base_url=proxy_url` → `pi` routes `/chat/completions` through the stub.
5. When the trained task completes/fails, a multica **session-close hook**
   writes `set_reward(session_id, DEFAULT_REWARD)` then `end_session(session_id)`.

## 3. Decisions (locked in brainstorm)

- **D1 Ownership** — AReaL hosts the LLM proxy and mints the session key
  (`session_api_key` = `proxy_key`). multica is the orchestrator. (brainstorm Q1=a)
- **D2 Trigger** — external driver (AReaL) tells multica to launch the team;
  the `env_dispatch` payload marks the training target. (Q trigger=c + identify=a)
- **D3 `proxy_url`** — multica **deployment config** (env var, default
  `http://db_bridge_stub:9100/v1`). `start_session` is NOT extended to return
  it; no AReaL-side contract change for the URL. (Q proxy_url=a)
- **D4 Reward boundary** — D writes a **default placeholder reward** on close
  (like the existing idle-finalizer `set_reward(1.0)`) so every trajectory is
  valid before sub-project E lands. Real reward / entropy / critic env-save =
  **E**. (Q reward=b)
- **D5 Lifecycle owner** — **server-side (Approach A)**. multica Go owns
  start/close; the daemon stays a thin consumer of injected config. (Q approach=A)
- **D6 Team-member creation** — leader task only at dispatch; teammates spawned
  later by leader/runtime. Therefore the **session-open hook fires at
  trained-member task creation, not at env_dispatch**. (Q member-launch=b)

## 4. Layers / changes

### 4.1 Contract (multica handler)
- `EnvDispatchRequest` + `service.EnvDispatchInput` gain optional
  `train_agent_id string` (UUID). Handler shape-validates when present.
- Validation: `train_agent_id` is allowed with `squad_id` (a team member) or
  when it equals a single `agent_id`; when set it must resolve to a real agent
  in the workspace. Empty ⇒ today's behavior exactly (no session).

### 4.2 Persist training intent (multica, DB)
Because the trained task is created later, `env_dispatch` must persist the
intent keyed to the dispatched **project** (each rollout lane = one project):
`{ project_id, workspace_id, train_agent_id, default_reward }`.
- Proposed: new table `training_dispatch` (migration 152) with a UNIQUE
  `project_id`. (Alt: stash on the leader task `context`; rejected — the
  trained teammate task can't cheaply find the leader's context, and project is
  the natural join key for later task creation.)

### 4.3 Session-open hook (multica service, task creation)
The hook fires wherever the trained member's task is created server-side,
pre-claim. Per T1/1a the chokepoints funnel into `CreateAgentTask` /
`CreateChatTask` reached from the `Enqueue*` family in `internal/service/task.go`
(`EnqueueTaskForIssue`, `EnqueueTaskForMention`/`EnqueueTaskForSquadLeader`
[leader @mention delegation via `enqueueCommentAgentTriggers`],
`EnqueueChatTask`, `EnqueueQuickCreateTask`) **and** `env_dispatch`'s separate
`EnqueueAgentRun` path (`handler/env_dispatch.go`). There is **no
`/api/agent/start` route** in this Go server (the db_bridge `agent_start`
channel targets a different le-agent SaaS API) — do not hook it.

When a new task's project has a `training_dispatch` row AND
`task.agent_id == train_agent_id` AND the task has no session yet:
1. `start_session(task_id=agent_task.id, group_size=1)` via the RL bridge
   client → `{session_id, proxy_key}`.
2. Store `session_id` **and `proxy_key`** on the task row (the close hook needs
   the proxy_key for session-key-authed set_reward/end_session).
3. Inject into `task.context`:
   `{"areal_proxy": {"provider":"areal","model":"areal-default",
   "api_key":<proxy_key>,"base_url":<proxy_url>}}`.
Idempotent: a task that already has a session is skipped (retry-safe).

### 4.4 Runtime provider wiring (multica daemon `execenv`)
Per T1/1c this is **NEEDS-NEW-FIELD**: there is no existing per-task,
context-sourced provider/base_url/api_key override (provider is per-runtime;
base_url/api_key only via agent-scoped `CustomEnv`). Add a new field on the
daemon `Task`/`TaskAgentData`, populated in `ClaimTaskByRuntime`
(`handler/daemon.go`) from `task.context.areal_proxy`, consumed when building
`ExecOptions` (`daemon.go`) → `pi -p --provider areal --model areal-default
--api-key <proxy_key>` with `base_url=<base_url>`. (Confirm the pi runtime's
env-var names for key/base_url in `pkg/agent/pi.go` during implementation.)

**T6 refinement (2026-07-06, folded into D close-out):** pi has no `--base-url`
flag. T6 injects the proxy base URL as env `AREAL_PROXY_BASE_URL`; T8 must wire
the `areal` provider entry in pi's `models.json` (or the daemon's provider
config) so its `baseURL` reads `$AREAL_PROXY_BASE_URL`, so the trained pi
actually routes to the bridge stub. This closes the `base_url=<base_url>` loose
end above.

### 4.5 Session-close hook (multica, task completion)
Per T1/1b the terminal transitions are `TaskService.CompleteTask` / `FailTask` /
`CancelTask` (`internal/service/task.go`), driven by daemon completion reports.
When a training task (has a stored `session_id`+`proxy_key`) reaches a terminal
state: `set_reward(reward=default_reward)` **then** `end_session`, both
authenticated with the stored **`proxy_key`** (session-key auth; no session_id
arg — experimental contract §4.6). Errors logged; best-effort.
**Known gap (deferred):** timeout/stale expiry runs through
`cmd/server/runtime_sweeper.go` (`FailStaleTasks`, raw SQL) and **bypasses**
`FailTask`, so timed-out trainings won't auto-close. Consistent with deferring a
session reaper (§4.5 was already best-effort); a reaper is future hardening, not D.

### 4.6 RL bridge client (multica Go) — new
A small Go client that POSTs to the local db_bridge **stub** for the
`gateway`-group channels (`/rl/start_session`, `/rl/set_reward`,
`/rl/end_session`). The channels already exist in `db_bridge/channels.py`
(stub on the multica/le-agent side, executor forwards to AReaL).

**Contract = the EXPERIMENTAL openai-proxy stack** (`areal/experimental/openai/
proxy/proxy_gateway.py` + `proxy_rollout_server.py`) — the ONLY stack that
serves `end_session`, and it matches the `tiggered_training` online-flow docs
(confirmed T1/1d, user-approved 2026-07-06):
- `start_session` — **admin-key** auth; body `{task_id, group_size:1}`; returns
  **flat** `{session_id, api_key}` (api_key = the per-session `proxy_key`).
  multica passes `agent_task_queue.id` as `task_id`.
- `set_reward` — **session-key** auth (Authorization: Bearer `<proxy_key>`);
  body `{reward}`; NO session_id argument.
- `end_session` — **session-key** auth (Bearer `<proxy_key>`); NO session_id body.
So the client holds the admin key (for start) and, per session, the returned
`proxy_key` (for set_reward + end_session). The close hook therefore needs the
stored `proxy_key`, not just `session_id`.

### 4.7 Config
`AREAL_PROXY_URL` (default `http://db_bridge_stub:9100/v1`),
`AREAL_BRIDGE_STUB_URL` (for the RL client), `AREAL_ADMIN_API_KEY`,
`TRAINING_DEFAULT_REWARD` (default `1.0`).

### 4.8 AReaL side
The full lifecycle (`start_session`/`set_reward`/`end_session`) is served ONLY
by the **experimental openai-proxy stack** (`areal/experimental/openai/proxy/
proxy_gateway.py` + `proxy_rollout_server.py`); the v2 `inference_service`
gateway lacks `end_session`. D targets the experimental stack (user-approved).
Confirm-only: the bridge `rl_*` executor points at the running experimental
gateway. Expected **no AReaL code change**.

## 5. Out of scope (⇒ sub-project E)
Real reward computation, entropy recording, critic-agent squad membership,
environment save/`env_id` emission back to AReaL. D emits only a default reward.

## 6. Open items — RESOLVED by Task 1 (`docs/superpowers/notes/2026-07-06-D-seams.md`)
1. Chokepoints = `Enqueue*`→`CreateAgentTask`/`CreateChatTask` + `EnqueueAgentRun`
   (there is NO `/api/agent/start` route here). See §4.3.
2. Close hook = `CompleteTask`/`FailTask`/`CancelTask`; sweeper-timeout gap
   deferred. See §4.5.
3. execenv = NEEDS-NEW-FIELD (new claim-time field from `context`). See §4.4.
4. RL contract = experimental (flat `{session_id, api_key}`, session-key auth
   for set_reward/end_session). task_id = `agent_task.id`. See §4.6.
5. project join confirmed (Issue.ProjectID / ChatSession.ProjectID). See §4.2.

## 7. Test strategy
- Go service unit tests: train_agent_id validation; training_dispatch persist;
  session-open hook (fake RL client asserts start_session called once with
  group_size=1, session_id stored, context injected); session-close hook
  (asserts set_reward(default)+end_session order). Scope Go build/test to
  touched packages (pre-existing webpush `./...` failure; 16 pre-existing
  ON CONFLICT handler failures).
- db_bridge: no new channels needed (reuse existing) — a smoke test that the
  RL client targets the right paths.
- execenv: provider-wiring test if §6.3 needs a new field.

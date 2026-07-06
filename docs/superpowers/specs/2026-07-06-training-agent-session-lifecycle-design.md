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
At the server-side chokepoint where an agent task is created for the trained
member (see §6 open item — mention-delegation and/or `/api/agent/start` /
leader-spawn), when the task's project has a `training_dispatch` row AND
`task.agent_id == train_agent_id` AND the task has no session yet:
1. `start_session(task_id, group_size=1)` via the RL bridge client →
   `{session_id, proxy_key}`.
2. Store `session_id` on the task row.
3. Inject into `task.context`:
   `{"areal_proxy": {"provider":"areal","model":"areal-default",
   "api_key":<proxy_key>,"base_url":<proxy_url>}}`.
Idempotent: a task that already has a session is skipped (retry-safe).

### 4.4 Runtime provider wiring (multica daemon `execenv`)
When a claimed task's `context` carries `areal_proxy`, `execenv` configures the
agent runtime to `provider=areal, model=areal-default, api_key=<proxy_key>,
base_url=<base_url>` — i.e. the `pi -p --provider areal --model areal-default
--api-key <proxy_key>` invocation. (§6 open item: confirm whether execenv
already supports a per-task provider/base_url override via context or needs a
new field.)

### 4.5 Session-close hook (multica, task completion)
When a training task transitions to `completed`/`failed` (task-completion path
/ daemon completion report), if the task has a training session:
`set_reward(session_id, default_reward)` then `end_session(session_id)` via the
RL bridge client. Errors logged; end is best-effort (a reaper for stale
sessions is a later hardening, not D).

### 4.6 RL bridge client (multica Go) — new
A small Go client that POSTs to the local db_bridge **stub** for the
`gateway`-group channels (`/rl/start_session`, `/rl/set_reward`,
`/rl/end_session`). Config: stub base URL + admin api key (env). This is the
multica→AReaL direction; the channels already exist in `db_bridge/channels.py`.

### 4.7 Config
`AREAL_PROXY_URL` (default `http://db_bridge_stub:9100/v1`),
`AREAL_BRIDGE_STUB_URL` (for the RL client), `AREAL_ADMIN_API_KEY`,
`TRAINING_DEFAULT_REWARD` (default `1.0`).

### 4.8 AReaL side
`start_session`/`set_reward`/`end_session` already exist
(`areal/v2/inference_service/data_proxy/session.py`;
`StartSessionResponse{group_id, sessions:[{session_id, session_api_key}]}`).
Confirm-only: the bridge routes `/rl/*` to the running gateway. Expected **no
AReaL code change**.

## 5. Out of scope (⇒ sub-project E)
Real reward computation, entropy recording, critic-agent squad membership,
environment save/`env_id` emission back to AReaL. D emits only a default reward.

## 6. Open items to pin in implementation Task 1 (read-and-document)
1. The exact server-side task-creation chokepoint(s) for a spawned squad
   teammate (mention-delegation comment→task vs `/api/agent/start` w/
   `parent_task_id` vs a leader-spawn service method). Confirm one hook or N.
2. The exact task-completion path (daemon completion report → server handler)
   to attach the close hook, incl. failed/cancelled/timeout transitions.
3. Whether `execenv` already supports per-task provider/base_url/api_key
   override via `context` (so §4.4 is minimal) or needs a new field.
4. Codegen: any new query (training_dispatch, store session_id, mark session
   config) must follow the hand-written-generated rule (no repo-wide `sqlc
   generate`).
5. Confirm `start_session`'s `task_id` semantics vs multica's `task_id` (the
   AReaL session is keyed by a task_id string — decide what multica passes).

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

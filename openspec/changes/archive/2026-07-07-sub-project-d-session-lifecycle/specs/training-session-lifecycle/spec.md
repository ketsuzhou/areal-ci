## ADDED Requirements

### Requirement: Session-open on trained-member task creation

The system SHALL open an AReaL RL proxy session when a task is created server-side for
an agent marked as a training target (via `env_dispatch` `train_agent_id`), before the
task is claimable by the daemon.

The session-open hook SHALL:

- fire at every server-side task-creation chokepoint that produces a trained teammate
  task (the `Enqueue*` family in `internal/service/task.go` and `env_dispatch`
  `EnqueueAgentRun`);
- call `start_session(task_id=agent_task_queue.id, group_size=1)` with admin-key auth
  against the experimental openai-proxy stack;
- store `session_id` and `proxy_key` (the returned `api_key`) into
  `task.context.areal_proxy`;
- inject `provider=areal`, `model=areal-default`, `api_key=proxy_key`,
  `base_url=proxy_url` into `task.context.areal_proxy`;
- be idempotent — a task that already has `context.areal_proxy` is skipped;
- leave non-trained tasks (no `training_dispatch` row, or `agent_id` !=
  `train_agent_id`) untouched, with no RL call and no `context.areal_proxy`.

#### Scenario: Trained teammate task created via leader @mention delegation

- **WHEN** a squad leader delegates a task to the trained member via @mention (the
  trained member's `agent_id` matches a `training_dispatch.train_agent_id` for the
  task's project)
- **THEN** the system calls `start_session` with the new task's id and `group_size=1`,
  stores `session_id` + `proxy_key` into `task.context.areal_proxy`, and injects the
  areal proxy provider config

#### Scenario: Idempotent retry on already-sessioned task

- **WHEN** the session-open hook fires on a task whose `context.areal_proxy` is already
  populated
- **THEN** the system skips `start_session` (no duplicate session) and leaves
  `context.areal_proxy` unchanged

#### Scenario: Non-trained task is untouched

- **WHEN** a task is created for an agent that is NOT a training target (no
  `training_dispatch` row for the project, or `agent_id` != `train_agent_id`)
- **THEN** the system makes no RL call and does not set `context.areal_proxy`

### Requirement: Trained-agent LLM routing through AReaL proxy

When a trained-member task is claimed by the daemon, the runtime SHALL route the trained
agent's LLM traffic through the AReaL proxy configured in `task.context.areal_proxy`.

The runtime SHALL:

- read `provider`, `model`, `api_key`, `base_url` from `task.context.areal_proxy` at
  `ClaimTaskByRuntime`;
- configure `pi` with `--provider areal --model areal-default --api-key <proxy_key>`;
- inject `AREAL_PROXY_BASE_URL` as an env var on the pi process;
- wire the `areal` provider entry in pi's `models.json` (or daemon provider config) so
  its base URL reads `$AREAL_PROXY_BASE_URL`.

#### Scenario: Trained task claimed by daemon routes through proxy

- **WHEN** a task with `context.areal_proxy` set is claimed by the daemon
- **THEN** the daemon's ExecOptions produce
  `pi --provider areal --model areal-default --api-key <proxy_key>` with
  `AREAL_PROXY_BASE_URL` set, and the pi `models.json` `areal` provider entry resolves
  its base URL from that env var

#### Scenario: Non-trained task claimed by daemon uses normal provider

- **WHEN** a task without `context.areal_proxy` is claimed by the daemon
- **THEN** the daemon uses the agent's normal provider config (no areal proxy override)

### Requirement: Session-close on task completion

The system SHALL close the RL session when a trained-member task reaches a terminal
state via the standard service paths.

The session-close hook SHALL:

- attach to `CompleteTask`, `FailTask`, and `CancelTask` / `CancelTaskWithResult` in
  `internal/service/task.go`;
- call `SetReward(proxy_key, default_reward)` THEN `EndSession(proxy_key)`, both with
  session-key Bearer auth;
- read `proxy_key` and `session_id` from `task.context.areal_proxy`;
- read `default_reward` from `training_dispatch.default_reward` (fallback to
  `TRAINING_DEFAULT_REWARD` config, default 1.0);
- be idempotent — tasks without `context.areal_proxy` are skipped;
- log RL errors without failing the task transition (best-effort close).

#### Scenario: Trained task completes — reward then end_session in order

- **WHEN** a task whose `context.areal_proxy` carries `session_id` + `api_key`
  (proxy_key) transitions to `completed` via `CompleteTask`
- **THEN** the system calls `SetReward(proxy_key, default_reward)` followed by
  `EndSession(proxy_key)`, in that order, both authenticated with the session-key Bearer
  header

#### Scenario: Trained task fails or is cancelled — close hook still fires

- **WHEN** a trained task transitions to `failed` (via `FailTask`) or `cancelled` (via
  `CancelTask` / `CancelTaskWithResult`)
- **THEN** the system calls `SetReward` then `EndSession` (same as completion)

#### Scenario: RL error during close does not fail the transition

- **WHEN** `SetReward` or `EndSession` returns a non-2xx response during close
- **THEN** the system logs the error and the task transition still completes
  (best-effort close)

#### Scenario: Non-trained task completion — no RL calls

- **WHEN** a task without `context.areal_proxy` reaches a terminal state
- **THEN** the system makes no RL call

### Requirement: Config guard against un-proxied training runs

The session-open hook SHALL fail loudly when training is requested (`train_agent_id` set
on `env_dispatch`) but the bridge configuration is incomplete (`AREAL_BRIDGE_STUB_URL`
or `AREAL_ADMIN_API_KEY` unset), rather than silently running the trained agent
un-proxied.

#### Scenario: Training requested but bridge config missing

- **WHEN** `env_dispatch` carries a `train_agent_id` AND `AREAL_BRIDGE_STUB_URL` or
  `AREAL_ADMIN_API_KEY` is unset
- **THEN** the session-open hook returns a loud error and the trained task is not
  created in an un-proxied state

### Requirement: Default placeholder reward on close

D SHALL write a default placeholder reward (`TRAINING_DEFAULT_REWARD`, default 1.0) on
session close so every trajectory is valid. Real reward computation, entropy recording,
and critic env-save are out of scope (sub-project E).

#### Scenario: Default reward written from training_dispatch

- **WHEN** a trained task closes and `training_dispatch.default_reward` is set
- **THEN** the close hook calls `SetReward` with that value

#### Scenario: Default reward falls back to config

- **WHEN** a trained task closes and `training_dispatch.default_reward` is unset (or no
  `training_dispatch` row)
- **THEN** the close hook calls `SetReward` with `TRAINING_DEFAULT_REWARD` (default 1.0)

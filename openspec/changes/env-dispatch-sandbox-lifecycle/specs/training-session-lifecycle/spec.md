## MODIFIED Requirements

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
- preserve any env-dispatch sandbox lifecycle handle associated with the trained task,
  including env id and sandbox-instance refs needed by checkpoint save/resume;
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

#### Scenario: Sandbox lifecycle handle preserved for checkpointing

- **WHEN** a trained task is created from env-dispatch data containing sandbox-instance
  refs
- **THEN** the task/session context preserves the env id and sandbox-instance refs so
  later checkpoint creation can save and resume the same environment

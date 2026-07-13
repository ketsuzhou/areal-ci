## ADDED Requirements

### Requirement: Resume-trigger captured at checkpoint create

The system SHALL capture a resume-trigger descriptor at checkpoint-create time that names the
in-flight agent runtime and task to re-engage on resume. The descriptor SHALL be persisted on
the `env_checkpoint` row as a `resume_trigger` JSONB field and SHALL identify the in-flight
`agent_task_queue` task (`task_id`), the bound `agent_runtime` (`runtime_id`), the `agent_id`,
the owning `issue_id` or `chat_session_id`, the `project_id`, and a `kind` distinguishing
issue-task resume from chat-task resume.

The descriptor SHALL be resolved server-side at checkpoint-create time from the project's
in-flight `agent_task_queue` task (status `running` or `dispatched`), so the caller is not
required to supply multica-internal task or runtime identifiers. If no in-flight task exists for
the project at create time, the descriptor SHALL be empty and resume degrades to sandbox-only.

#### Scenario: Checkpoint records resume-trigger from in-flight task

- **WHEN** a trained rollout creates a checkpoint at a structural event with an in-flight task
- **THEN** the checkpoint row stores a `resume_trigger` descriptor carrying the in-flight
  `task_id`, `runtime_id`, `agent_id`, `issue_id` or `chat_session_id`, `project_id`, and `kind`

#### Scenario: Trigger descriptor resolved server-side

- **WHEN** a checkpoint is created for a project that has an in-flight (running or dispatched)
  task
- **THEN** the system resolves the in-flight task server-side and stores a `resume_trigger`
  descriptor carrying its `task_id`, `runtime_id`, `agent_id`, `issue_id` or `chat_session_id`,
  `project_id`, and `kind`, without requiring the caller to supply those identifiers

#### Scenario: No in-flight task yields empty trigger

- **WHEN** a checkpoint is created for a project that has no in-flight (running or dispatched)
  task at create time
- **THEN** the checkpoint stores an empty `resume_trigger` and a later resume re-engages no task
  (sandbox-only resume)

#### Scenario: Resume-trigger round-trips through storage

- **WHEN** a checkpoint with a `resume_trigger` is created and later retrieved via get or list
- **THEN** the returned checkpoint carries the same `resume_trigger` descriptor

### Requirement: Resume-agent-run primitive re-engages in-flight task

The system SHALL provide a resume-agent-run primitive that re-activates the **existing** in-flight
`agent_task_queue` task against the resumed `agent_runtime`, so the agent continues the same
task from the checkpointed intermediate state. The primitive MUST NOT create a new task row
(unlike `EnqueueTaskForIssue`) and MUST NOT be implemented as a literal chat message.

The primitive SHALL validate that the referenced task is still resumable (not terminal) and that
the referenced runtime corresponds to a resumed sandbox; a task that has transitioned terminal
between checkpoint create and resume SHALL be rejected with a typed error rather than
double-running.

#### Scenario: Primitive re-activates existing in-flight task

- **WHEN** the resume-agent-run primitive is invoked with a valid resume-trigger after the
  sandbox resumed
- **THEN** the existing in-flight task is re-activated against the resumed runtime and no new
  task row is created

#### Scenario: Terminal task is not double-run

- **WHEN** the primitive is invoked for a task that has already transitioned terminal between
  checkpoint create and resume
- **THEN** the primitive rejects the trigger with a typed error and does not re-run the task

#### Scenario: Unknown runtime or task rejected

- **WHEN** the primitive is invoked with a trigger referencing an unknown runtime or task
- **THEN** the primitive returns a typed error and performs no re-activation

### Requirement: Resume-from-checkpoint executes trigger after sandbox resume

`ResumeFromCheckpoint` SHALL execute the stored resume-trigger via the resume-agent-run primitive
after the checkpoint's sandbox instances have been resumed, so the agent runtime actually
continues the rollout from the checkpointed state. The result returned to the caller SHALL
report whether trigger execution succeeded.

A trigger-execution failure (sandbox resumed but agent not re-engaged) SHALL be surfaced as a
typed partial-resume result, not a silent no-op.

#### Scenario: Resume executes trigger and continues the rollout

- **WHEN** an authorized caller resumes a completed checkpoint that carries a resume-trigger
- **THEN** the system resumes the sandbox instances, executes the resume-trigger to re-engage the
  agent runtime, and returns a rollout handle reporting trigger-execution success

#### Scenario: Trigger-execution failure is surfaced as partial resume

- **WHEN** sandbox resume succeeds but the resume-trigger execution fails
- **THEN** the system returns a typed partial-resume result indicating the sandbox is up but the
  agent was not re-engaged, rather than silently returning a success handle

### Requirement: Legacy checkpoint without trigger degrades gracefully

The system SHALL resume a checkpoint whose `resume_trigger` is empty using the prior behavior -
resume the sandbox instances and return a rollout handle - without invoking the resume-agent-run
primitive. This preserves backward compatibility with checkpoints created before this change and
with non-trained rollouts.

#### Scenario: Checkpoint without trigger resumes as before

- **WHEN** an authorized caller resumes a completed checkpoint whose `resume_trigger` is empty
- **THEN** the system resumes the sandbox instances and returns a rollout handle without invoking
  the resume-agent-run primitive

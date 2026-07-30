## ADDED Requirements

### Requirement: Agent continuation goes through a single seam

Re-engaging an agent after its environment returns SHALL be performed through one
continuation seam with exactly one strategy selected by the checkpoint's save mode. No
code path outside that seam SHALL enqueue or re-activate an agent task as part of
environment restoration.

#### Scenario: Save mode selects the strategy

- **WHEN** a checkpoint is resumed
- **THEN** the continuation seam is invoked with the checkpoint's continuation
  descriptor
- **AND** the same-runtime strategy is used for `pause_in_place`
- **AND** the forked-runtime strategy is used for `snapshot`

#### Scenario: Branch continuation uses the seam

- **WHEN** a branch dispatch materializes a lane
- **THEN** that lane's agent is re-engaged through the continuation seam
- **AND** not through provisioning-local enqueue logic

### Requirement: Same-runtime continuation reuses the existing task

The same-runtime strategy SHALL re-activate the existing in-flight task row bound to the
resumed runtime, without creating a new task row, and SHALL wake that runtime so it
claims the task promptly. It SHALL reject a task that has reached a terminal state and
SHALL reject a task whose bound runtime does not match the continuation descriptor.

#### Scenario: Interrupted task is re-activated in place

- **WHEN** a `pause_in_place` checkpoint carrying a continuation descriptor is resumed
- **THEN** the descriptor's existing task row becomes claimable again by its bound
  runtime
- **AND** no new task row is created
- **AND** the task's preserved context, issue or chat association, and runtime binding
  are unchanged

#### Scenario: Terminal task is rejected

- **WHEN** the descriptor's task reached a terminal state between checkpoint creation
  and resume
- **THEN** the continuation is rejected
- **AND** the task is not run a second time

#### Scenario: Runtime mismatch is rejected

- **WHEN** the descriptor's task is bound to a runtime other than the one named in the
  descriptor
- **THEN** the continuation is rejected

### Requirement: Forked-runtime continuation enqueues per lane

The forked-runtime strategy SHALL enqueue a task for each materialized lane against that
lane's copied project subtree and its own agent runtime. Lanes SHALL NOT share a task
row, an agent runtime, or a runtime identity with each other or with the source
environment.

#### Scenario: Each lane gets its own task and runtime

- **WHEN** a `snapshot` checkpoint is resumed into three lanes
- **THEN** three distinct task rows are enqueued, one per lane
- **AND** each is bound to that lane's own agent runtime
- **AND** the source environment's task and runtime are unaffected

#### Scenario: Lane runtime identity is reset

- **WHEN** a lane is materialized from a savepoint that captured a running agent daemon
- **THEN** the daemon inherited from the savepoint is stopped before the lane's runtime
  starts
- **AND** the lane registers under its own runtime identity rather than the source's

### Requirement: Continuation outcome is reported, not silently dropped

A resume SHALL report the outcome of continuation as a distinct status covering
executed, skipped because no continuation descriptor was recorded, and failed. A failed
continuation after a successful environment restore SHALL be reported as a partial
resume rather than as success.

#### Scenario: Missing descriptor is reported as skipped

- **WHEN** a checkpoint was created while no task was in flight, and is then resumed
- **THEN** the environment is restored
- **AND** the continuation outcome is reported as skipped
- **AND** no continuation strategy is invoked

#### Scenario: Failed continuation is reported as partial

- **WHEN** the environment restores successfully but the continuation strategy fails
- **THEN** the resume reports a failed continuation outcome
- **AND** the caller can distinguish it from a fully successful resume

#### Scenario: Continuation failure is reported per lane

- **WHEN** a resume materializes several lanes and the task enqueue fails for one of
  them
- **THEN** that lane's continuation outcome is reported as failed
- **AND** the other lanes' continuation outcomes are reported as executed
- **AND** the resume is not reported as fully successful

### Requirement: Session continuation policy depends on the save mode

A `pause_in_place` resume SHALL continue the interrupted agent session, resolved from
the resumed task's own recorded session. A lane materialized from a savepoint SHALL
start a fresh session and SHALL NOT continue the session recorded in the savepoint.

#### Scenario: Pause-in-place continues the interrupted session

- **WHEN** an agent task that had recorded a session mid-flight is checkpointed with
  `pause_in_place` and later resumed
- **THEN** the re-activated task continues that recorded session
- **AND** it does not start a fresh session

#### Scenario: Forked lanes start fresh sessions

- **WHEN** a `snapshot` checkpoint is resumed into multiple lanes
- **THEN** each lane starts a fresh session
- **AND** no two lanes continue the same recorded session

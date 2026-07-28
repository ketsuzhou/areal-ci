## ADDED Requirements

### Requirement: Env checkpoints declare a save mode

An env checkpoint SHALL declare a save mode of either `pause_in_place` or `snapshot`.
`pause_in_place` SHALL suspend the source sandbox instances and record no savepoint.
`snapshot` SHALL record an immutable savepoint per source instance and SHALL leave the
source instances running. Checkpoints created without an explicit save mode SHALL
default to `pause_in_place`.

#### Scenario: Snapshot mode leaves the source running

- **WHEN** a checkpoint is created with save mode `snapshot` against a live env whose
  agent is mid-task
- **THEN** the checkpoint records one savepoint reference per source sandbox instance
- **AND** every source instance remains in a running state
- **AND** the source agent's in-flight task remains in its pre-checkpoint state

#### Scenario: Pause-in-place mode suspends the source and records no savepoint

- **WHEN** a checkpoint is created with save mode `pause_in_place`
- **THEN** every source sandbox instance is suspended
- **AND** the checkpoint's savepoint references are empty

#### Scenario: Existing checkpoints keep their behavior

- **WHEN** a checkpoint row created before this capability is read
- **THEN** its save mode resolves to `pause_in_place`
- **AND** its savepoint references are empty
- **AND** resuming it behaves exactly as it did before this capability

### Requirement: Savepoints are immutable and independently owned

A savepoint created for a checkpoint SHALL be represented by a durable snapshot record
that reaches a ready state before the checkpoint is reported complete. The savepoint
SHALL NOT be reclaimed when a lane is materialized from it.

#### Scenario: Savepoint is ready before the checkpoint completes

- **WHEN** a `snapshot` mode checkpoint is created
- **THEN** each savepoint's snapshot record is in a ready state
- **AND** only then is the checkpoint's save status reported complete

#### Scenario: Savepoint survives lane materialization

- **WHEN** a lane has been materialized from a checkpoint's savepoint
- **THEN** the savepoint's snapshot record is still ready
- **AND** it can materialize a further lane

#### Scenario: Savepoint creation failure fails the save

- **WHEN** a savepoint's snapshot record reaches a failed state during checkpoint
  creation
- **THEN** the checkpoint's save status is reported failed
- **AND** the checkpoint is not resumable

### Requirement: Resume materializes a requested number of lanes

Resuming a `snapshot` mode checkpoint SHALL accept a requested lane count and SHALL
materialize that many sandbox instances from the checkpoint's savepoints, taking one
snapshot per source instance rather than one per lane. Each lane SHALL receive its own
copy of the captured project subtree and its own agent runtime.

#### Scenario: Three lanes from one savepoint

- **WHEN** a `snapshot` mode checkpoint is resumed with a requested lane count of three
- **THEN** three sandbox instances are created from the checkpoint's savepoint
- **AND** each lane has its own copied project subtree and its own agent runtime
- **AND** no additional snapshot of the source is taken

#### Scenario: Pause-in-place rejects fan-out

- **WHEN** a `pause_in_place` checkpoint is resumed with a requested lane count greater
  than one
- **THEN** the resume is rejected with a typed error
- **AND** no sandbox instance is created

#### Scenario: Pause-in-place resumes the same instance

- **WHEN** a `pause_in_place` checkpoint is resumed with a requested lane count of one
- **THEN** the previously suspended sandbox instances are resumed
- **AND** no new sandbox instance is created

### Requirement: Resume is idempotent per lane key

Every resume request SHALL carry a lane key. A request whose lane key already has a
materialized lane for that checkpoint SHALL return the existing lane without creating
another. A request with an unused lane key SHALL materialize a new lane.

#### Scenario: Repeating a lane key returns the existing lane

- **WHEN** a checkpoint is resumed twice with the same lane key
- **THEN** both responses identify the same lane
- **AND** exactly one sandbox instance was created for that lane key

#### Scenario: A new lane key expands the frontier again

- **WHEN** a checkpoint that already has lanes is resumed with a previously unused lane
  key
- **THEN** a new lane is materialized from the same savepoint
- **AND** the previously materialized lanes are unaffected
- **AND** no second checkpoint is created for that frontier

#### Scenario: A retried request does not double the lanes

- **WHEN** a branch request that already produced lanes is retried under the same
  request identity
- **THEN** the lane keys derived for the retry match the original request's lane keys
- **AND** the existing lanes are returned rather than additional lanes being
  materialized

#### Scenario: A zero lane count is rejected

- **WHEN** a checkpoint is resumed with a requested lane count of zero
- **THEN** the request is rejected as invalid input
- **AND** no lane record is created

### Requirement: Lane materialization recovers from interruption

A lane SHALL record its materialization progress so an interrupted lane can be continued
rather than duplicated. Resuming with the lane key of an unfinished lane SHALL continue
that lane's materialization and SHALL NOT create a second sandbox instance for it.

#### Scenario: An interrupted lane is continued, not duplicated

- **WHEN** a lane's materialization was interrupted after its sandbox instance was
  created but before its task was enqueued
- **AND** the checkpoint is resumed again with that lane's key
- **THEN** materialization continues from the first incomplete step
- **AND** no second sandbox instance is created for that lane

#### Scenario: A lane whose savepoint is gone fails with a typed error

- **WHEN** a lane is materialized from a savepoint whose underlying snapshot no longer
  exists
- **THEN** the lane is recorded as failed with a typed error
- **AND** the savepoint is marked failed so subsequent resumes of that checkpoint fail
  fast

#### Scenario: A resume whose every lane fails is reported as failed

- **WHEN** every requested lane fails to materialize
- **THEN** the resume reports failure
- **AND** it does not report success with an empty lane set

### Requirement: Branch dispatch is served by checkpoint resume

An env dispatch requesting branch mode SHALL be served internally by creating or reusing
a `snapshot` mode checkpoint at the requested env and resuming it. The externally
visible dispatch request and response contract SHALL be unchanged.

#### Scenario: Branch dispatch produces lanes without its own provisioning path

- **WHEN** an env dispatch requests branch mode against a live env
- **THEN** the resulting child environments are materialized through checkpoint resume
- **AND** the dispatch response matches the pre-existing branch contract
- **AND** the source env continues running

### Requirement: Non-resumable checkpoints are rejected

A checkpoint whose save did not reach a complete status SHALL be rejected by resume with
a typed, distinguishable error rather than a generic failure.

#### Scenario: Timed-out save is not resumable

- **WHEN** a checkpoint's save exceeds its configured timeout and is recorded as timed
  out
- **AND** that checkpoint is resumed
- **THEN** the resume is rejected with a typed non-resumable error
- **AND** the caller can distinguish it from a transient error

### Requirement: A savepoint is owned by exactly one checkpoint

Each savepoint SHALL be owned by exactly one checkpoint and SHALL be released when that
checkpoint is deleted or expires. A checkpoint SHALL NOT be deleted while any of its
lanes is still being materialized, so that no sandbox is left without an owning lane
record.

#### Scenario: Deleting the owning checkpoint releases the savepoint

- **WHEN** a checkpoint that owns a savepoint is deleted and none of its lanes is being
  materialized
- **THEN** the savepoint's snapshot record is scheduled for deletion
- **AND** the checkpoint's lane records are removed

#### Scenario: Deletion is refused while a lane is being materialized

- **WHEN** a checkpoint is deleted while one of its lanes is still being materialized
- **THEN** the deletion is rejected with a typed error
- **AND** the savepoint, the lane record, and the lane's sandbox are all retained

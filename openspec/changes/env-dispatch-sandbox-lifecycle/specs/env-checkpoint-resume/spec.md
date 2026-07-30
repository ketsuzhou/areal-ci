## ADDED Requirements

### Requirement: Pause-in-place checkpoint creation

The system SHALL allow trained env-dispatch rollouts to create environment checkpoints backed by sandbox-instance refs and pause-in-place sandbox saves.

Checkpoint creation SHALL record project/workspace context, event reference, checkpoint kind, sandbox-instance refs, env id map, inline JSONB DB subtree snapshot, save status, timestamp, configured save timeout, and optional entropy score.

Checkpoint creation SHALL synchronously wait for sandboxd save/stop completion up to the configured timeout before returning. If save completes within the timeout, the checkpoint is resumable. If save fails or times out, the checkpoint SHALL record a terminal non-resumable save status and the caller SHALL receive a typed failure response.

#### Scenario: Always-event checkpoint pauses sandbox synchronously

- **WHEN** a trained rollout requests an always-event checkpoint for an environment with sandbox-instance refs
- **THEN** the system records a checkpoint, enqueues sandbox save/stop for each affected sandbox instance, and waits up to the configured timeout for save completion before returning

#### Scenario: Entropy-gated checkpoint records entropy

- **WHEN** a trained rollout requests an entropy-gated checkpoint with an entropy score
- **THEN** the system records the entropy score and synchronously saves the affected sandbox instances before returning a resumable checkpoint

#### Scenario: Inline DB snapshot is recorded

- **WHEN** checkpoint creation captures the Multica project subtree
- **THEN** the checkpoint row stores issue, sub-issues, tasks, messages, comments, and event ordering data in an inline JSONB `db_snapshot` field

#### Scenario: Save timeout is recorded as non-resumable

- **WHEN** sandbox save does not complete before the configured checkpoint save timeout
- **THEN** the checkpoint records timed-out save status, the caller receives a typed timeout response, and the checkpoint cannot be resumed

#### Scenario: Save failure is recorded without crashing rollout control

- **WHEN** sandbox save fails during checkpoint creation
- **THEN** the checkpoint records failed save status and rollout control receives a typed failure or best-effort skip according to caller mode

### Requirement: Checkpoint listing and retrieval

The system SHALL expose checkpoint retrieval and project-scoped checkpoint listing for authorized callers.

Checkpoint listing SHALL return checkpoints for the requested project ordered newest first and MUST enforce workspace ownership.

#### Scenario: List project checkpoints

- **WHEN** an authorized caller lists checkpoints for a project
- **THEN** the system returns that project's checkpoints ordered by creation time descending

#### Scenario: Cross-workspace checkpoint access rejected

- **WHEN** a caller requests a checkpoint from another workspace
- **THEN** the request is rejected with a forbidden or not-found response and checkpoint details are not leaked

### Requirement: Resume from checkpoint

The system SHALL resume a saved environment checkpoint by using the existing sandboxd resume lifecycle for the checkpoint's sandbox-instance refs and returning a rollout handle suitable for AReaL continuation.

Resume MUST require a completed or otherwise resumable save status and MUST not expose immutable branch/fork semantics.

#### Scenario: Resume completed checkpoint

- **WHEN** an authorized caller resumes a checkpoint whose save status is complete
- **THEN** the system resumes the saved sandbox instances and returns a rollout handle containing the resumed environment identifiers

#### Scenario: Incomplete checkpoint cannot resume

- **WHEN** an authorized caller resumes a checkpoint whose save status is pending or failed
- **THEN** the system returns a typed error and does not enqueue resume jobs

#### Scenario: API naming uses resume not branch

- **WHEN** AReaL calls the checkpoint continuation API
- **THEN** the API and client method names use resume-from-checkpoint terminology rather than branch-from-checkpoint terminology

### Requirement: AReaL checkpoint client integration

The AReaL env-dispatch client SHALL provide helpers to create/list checkpoints and resume from checkpoint without requiring callers to know Multica sandboxd internals.

The client SHALL omit optional checkpoint or per-agent env fields when not provided and surface 403/404/409-style checkpoint errors as typed client errors.

#### Scenario: Resume-from-checkpoint client returns rollout handle

- **WHEN** AReaL calls `resume_from_checkpoint(checkpoint_id)` and Multica returns a successful response
- **THEN** the client returns the rollout handle to the tree-search caller

#### Scenario: Optional fields omitted when empty

- **WHEN** AReaL creates env-dispatch or checkpoint requests without optional per-agent env or entropy data
- **THEN** the client omits those fields or sends null only where the API contract permits it

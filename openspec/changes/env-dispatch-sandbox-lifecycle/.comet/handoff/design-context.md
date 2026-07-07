# Comet Design Handoff

- Change: env-dispatch-sandbox-lifecycle
- Phase: design
- Mode: compact
- Context hash: 5d7ab380e062213c7b0ae39ed7f2e223954bb0342874c782824f11225facd1c4

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-dispatch-sandbox-lifecycle/proposal.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/proposal.md
- Lines: 1-32
- SHA256: 9907c5f4220934b5656407aa4c30714e03399b162eb940e3dd41e6ff5002b76d

```md
## Why

Env-dispatch training needs a concrete, resumable environment lifecycle so rollouts can save state at useful decision points and later resume without inventing a parallel sandbox provider. `lijiannankai@126.com`'s Multica sandbox work already provides the needed foundation through `sandbox_instance` rows, sandboxd jobs, Cube pause/resume/delete, node ownership, and offline-node force-delete behavior.

## What Changes

- Integrate sandbox-instance-backed environment handles into env-dispatch so rollout environments reference Multica `sandbox_instance` records instead of opaque sandbox ids where save/resume is required.
- Add pause-in-place environment save semantics that reuse the existing sandboxd `stop` job and Cube `/pause` behavior.
- Add resume-from-checkpoint semantics that reuse existing sandboxd `resume` jobs and runtime restart metadata.
- Add checkpoint storage and APIs for create/list/resume so AReaL can save and resume rollout environments.
- Extend the AReaL env-dispatch client to send per-agent environment intent and call resume-from-checkpoint APIs.
- Preserve existing sandboxd delete and unreachable-node force-delete behavior; no immutable sandbox fork/snapshot capability is promised in this change.

## Capabilities

### New Capabilities

- `env-dispatch-sandbox-lifecycle`: Env-dispatch environments use sandbox-instance lifecycle handles for save, resume, delete, and per-agent sandbox assignment.
- `env-checkpoint-resume`: Training rollouts can create, list, and resume pause-in-place environment checkpoints backed by sandboxd jobs.

### Modified Capabilities

- `training-session-lifecycle`: Training sessions preserve the env-dispatch sandbox lifecycle handle needed to save and resume the rollout environment.
- `critic-driven-training-signal`: Entropy/logprob data captured for critic-driven training can optionally trigger checkpoint creation at uncertain tool-call decisions.

## Impact

- **multica/server**: env-dispatch handler/service contracts, sandbox lifecycle service wrapper, checkpoint service/handler, migrations, sqlc queries, and trigger seams around trained rollout events.
- **sandboxd/Cube lifecycle**: reuse existing create, stop, resume, delete, reconfigure, local_ref, runtime metadata, websocket wakeup, node ownership, and force-delete flows.
- **AReaL client**: `customized_areal/tree_search/agents/swe_lego_client.py` gains per-agent env serialization and checkpoint create/list/resume helpers.
- **Existing specs**: extends training-session lifecycle and critic-driven entropy capture without changing their core session-open/session-close contracts.
- **Out of scope**: immutable sandbox snapshots/forks, concurrent branches from a live rollout, copy-on-write sandbox optimization, and implementing code during the Open phase.
```

## openspec/changes/env-dispatch-sandbox-lifecycle/design.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/design.md
- Lines: 1-76
- SHA256: a7a15980caacc2b394b1304d00b0c8b53b3000fe7de9c4b066e2032f1f9322a2

```md
## Context

Env-dispatch currently orchestrates rollout reset and task dispatch, while sandbox lifecycle work lives in Multica's sandbox node gateway. The relevant `lijiannankai@126.com` commits added `sandbox_instance` persistence, sandboxd job execution, Cube create/pause/resume/delete calls, runtime reconfiguration, per-user node ownership, websocket job wakeups, and unreachable-node force-delete behavior.

Training work from earlier changes opens AReaL sessions for trained member tasks and captures critic/logprob signals. The missing piece is a stable environment lifecycle contract connecting env-dispatch, sandbox instances, checkpoint storage, and AReaL resume calls.

## Goals / Non-Goals

**Goals:**

- Treat `sandbox_instance` as the canonical environment lifecycle handle for env-dispatch save/resume paths.
- Reuse existing sandboxd `stop`, `resume`, `delete`, and `reconfigure` jobs instead of creating a parallel sandbox provider.
- Add checkpoint create/list/resume semantics for training rollouts.
- Preserve runtime metadata needed to resume a paused sandbox and continue the rollout.
- Extend AReaL client code with per-agent env intent and checkpoint resume helpers.

**Non-Goals:**

- Immutable sandbox snapshots, forks, copy-on-write, or concurrent branches from one live rollout.
- Replacing sandboxd/Cube lifecycle internals.
- Changing critic reward semantics or the core training session open/close contract.
- Implementing production garbage collection for old checkpoints beyond list/get visibility.

## Decisions

### D1: Use `sandbox_instance` refs as env-dispatch lifecycle handles

Env-dispatch will record structured sandbox refs that include `sandbox_instance.id`, `local_ref`, node id, template, status, endpoint/runtime metadata, and ownership/workspace context. Existing raw sandbox id fields remain compatibility data, but new save/resume code should consume the structured refs.

Alternative considered: add a new AReaL/Fleet/Daytona checkpoint provider. This is rejected for v1 because Multica already owns sandbox lifecycle and has production job/status semantics.

### D2: Save means synchronous pause-in-place via sandboxd `stop`

Checkpoint creation saves an environment by enqueueing the existing sandboxd `stop` job for each affected sandbox instance, then waiting up to the configured checkpoint save timeout for completion. Sandboxd calls Cube `/pause`, updates status through the existing stopped state, and records save status on the checkpoint. A completed save is resumable; a failed or timed-out save is recorded as a terminal non-resumable checkpoint status and returned as a typed failure.

Alternative considered: snapshot/fork the sandbox without interrupting the active rollout. This is deferred because the current sandbox node gateway exposes pause/resume/delete, not immutable fork semantics.

### D3: Resume means resume-from-checkpoint, not branch-from-checkpoint

The public API and client naming should use `resume_from_checkpoint` or `resume-from-checkpoint`. It resumes the saved sandbox instance and restores/uses the saved Multica DB subtree reference. It must not advertise branch/fork semantics.

### D4: Wrap existing sandbox handler/job logic behind a service seam

Add an internal lifecycle service used by checkpointing and env-dispatch. It should delegate to existing query/job code for create, save/stop, resume, delete, runtime restart metadata, websocket wakeup, and force-delete fallback. This avoids copying handler logic into checkpoint code.

### D5: Checkpoints capture sandbox refs and inline Multica state JSONB

A checkpoint stores sandbox instance refs plus an event reference and an inline JSONB DB subtree snapshot for the issue/project state. The snapshot should cover issue, sub-issues, tasks, messages, comments, and event ordering data so resume can re-enter the same Multica entity state. Inline JSONB is the v1 storage shape; object refs or normalized checkpoint tables are deferred until snapshot size or query requirements justify them.

### D6: Per-agent env intent is explicit and optional

Env-dispatch accepts optional per-agent environment specs. Specified agents resolve to their own sandbox instance/template; unspecified agents use the existing default/shared behavior. All agents still share the same Multica entity subtree for the rollout.

## Risks / Trade-offs

- Pause-in-place interrupts the active rollout -> restrict automatic saves to training flows that expect a pause and name APIs as resume, not branch.
- Synchronous checkpoint creation can block or time out -> keep the save timeout configurable and return typed timeout/failure responses.
- Sandbox pause and DB subtree serialization are not perfectly atomic -> store event refs, timestamps, and save status so resume can validate ordering and incomplete saves.
- Inline JSONB snapshots are simple but can grow large -> bound the captured subtree and leave room for a future object-ref migration.
- Legacy `environment.sandbox_ids` may diverge from structured refs -> keep a compatibility read path while new save/resume code prefers `sandbox_instance` refs.
- Offline nodes can make delete/resume unreliable -> preserve the existing force-delete behavior for deletes and return typed errors for resume when the node cannot be reached.
- Entropy logprobs may be unavailable -> skip entropy-gated checkpoints while preserving always-event checkpoint behavior.

## Migration Plan

1. Add schema for env checkpoint records and, if needed, per-agent env/sandbox refs.
2. Backfill or compatibility-read existing sandbox id fields only where needed; do not require existing rollouts to have checkpoint refs.
3. Add lifecycle service tests around existing sandbox job behavior before wiring checkpointing.
4. Add checkpoint APIs and AReaL client methods behind config gates.
5. Roll out save/resume triggers for trained env-dispatch sessions after scoped service tests pass.

Rollback is to disable env checkpointing by config and keep env-dispatch using existing raw sandbox behavior. The sandbox node gateway migrations remain independently useful and should not be rolled back as part of checkpoint rollback.

## Open Questions

- Which AReaL loop owns checkpoint selection policy: tree search frontier selection, critic post-processing, or a separate replay controller?
```

## openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/tasks.md
- Lines: 1-54
- SHA256: f9f2d1041e1678b52f35aa4882e9fd02f349c65170106e8a3132706ba4dca0e1

```md
## 1. Investigation and Seam Confirmation

- [ ] 1.1 Confirm the sandbox lifecycle commits from `lijiannankai@126.com` that introduced sandbox node gateway, sandboxd jobs, `sandbox_instance`, `local_ref`, resume/reconfigure metadata, node ownership, and force-delete behavior.
- [ ] 1.2 Trace current env-dispatch state flow in `multica/server/internal/service/env_dispatch.go`, handler request parsing, training dispatch persistence, and AReaL client serialization.
- [ ] 1.3 Document the transition from legacy raw sandbox ids to structured sandbox-instance refs, including compatibility reads and where new code must prefer structured refs.
- [ ] 1.4 Confirm DB subtree snapshot query scope for issue, sub-issues, tasks, messages, comments, and event refs.

## 2. Sandbox Lifecycle Service

- [ ] 2.1 Add failing Multica tests for save enqueuing existing sandbox `stop` jobs, resume enqueuing `resume` jobs with runtime metadata, delete preserving force-delete fallback, and missing sandbox typed errors.
- [ ] 2.2 Implement an internal env sandbox lifecycle service that wraps existing sandbox query/job machinery for create, save, resume, delete, and reconfigure.
- [ ] 2.3 Preserve sandboxd websocket wakeups, node ownership checks, Cube `local_ref`, runtime env/model metadata, and existing status transitions.
- [ ] 2.4 Run scoped Go tests for sandbox handler/lifecycle packages.

## 3. Env-dispatch Sandbox-instance Handles and Per-agent Envs

- [ ] 3.1 Add failing tests for per-agent env specs: valid assignment, unknown agent, unknown env spec, empty field preserving current behavior, and partial squad defaults.
- [ ] 3.2 Extend env-dispatch request/service input with optional per-agent env specs and validation against workspace/squad membership.
- [ ] 3.3 Persist or return structured sandbox-instance refs for rollout environments while keeping legacy raw sandbox ids readable.
- [ ] 3.4 Ensure trained task/session context preserves env id and sandbox-instance refs needed by checkpointing.

## 4. Checkpoint Storage and APIs

- [ ] 4.1 Add migration and generated queries for env checkpoint records with workspace/project ids, event ref, checkpoint kind, env id map, sandbox-instance refs, inline JSONB DB snapshot, entropy score, save timeout, save status, and timestamps.
- [ ] 4.2 Add failing service tests for create checkpoint, get checkpoint, list checkpoints, synchronous save completion, save timeout status, save failure status, workspace ownership, inline JSONB snapshot round trip, and newest-first ordering.
- [ ] 4.3 Implement checkpoint create/get/list service methods and DB subtree snapshot/reference capture.
- [ ] 4.4 Add HTTP handlers/routes for checkpoint create and list with config-gated behavior.

## 5. Resume-from-checkpoint

- [ ] 5.1 Add failing tests for successful resume, per-agent sandbox refs preserved, incomplete save rejected, checkpoint not found, and cross-workspace access rejected.
- [ ] 5.2 Implement resume-from-checkpoint using the env sandbox lifecycle service and existing sandboxd resume job flow.
- [ ] 5.3 Ensure API and client naming use resume-from-checkpoint terminology, not branch-from-checkpoint.
- [ ] 5.4 Return a rollout handle that AReaL can use to continue tree-search execution.

## 6. Entropy and Event-triggered Checkpoints

- [ ] 6.1 Add tests for always-event checkpoint triggers on trained rollout structural events and skips for non-trained, sweeper, autopilot, and sandbox lifecycle events.
- [ ] 6.2 Wire policy-relevant always-event triggers to checkpoint creation without triggering on sandbox lifecycle jobs.
- [ ] 6.3 Add AReaL tests for entropy computation, threshold behavior, and unavailable logprobs skip.
- [ ] 6.4 Implement AReaL entropy-gated checkpoint creation with optional entropy score in Multica requests.

## 7. AReaL Client Integration

- [ ] 7.1 Add tests that `create_env_dispatch` serializes per-agent env specs when provided and omits them when empty.
- [ ] 7.2 Add checkpoint create/list/resume client tests, including 403/404/409-style typed error handling.
- [ ] 7.3 Implement AReaL client helpers for checkpoint create/list/resume and per-agent env dispatch.

## 8. Verification and Documentation

- [ ] 8.1 Run scoped Multica Go tests for env-dispatch, sandbox lifecycle, checkpoint service/handler, and generated query users.
- [ ] 8.2 Run scoped AReaL Python tests for env-dispatch client, entropy helper, and checkpoint resume client.
- [ ] 8.3 Validate the OpenSpec change with `openspec validate env-dispatch-sandbox-lifecycle --strict`.
- [ ] 8.4 Document operational semantics: save pauses the active sandbox, resume resumes the same saved sandbox, immutable branching is deferred.
```

## openspec/changes/env-dispatch-sandbox-lifecycle/specs/critic-driven-training-signal/spec.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/specs/critic-driven-training-signal/spec.md
- Lines: 1-37
- SHA256: 2fd5047b87548a20c0dfd756d058ce9f29ccb33d9056d4dbf2541ef98483c2fe

```md
## MODIFIED Requirements

### Requirement: Entropy recording from proxied LLM traffic

AReaL's experimental openai-proxy SHALL capture logprobs from proxied LLM traffic by
requesting `logprobs=true` on all calls, so entropy is computable at session end and at
checkpoint-eligible tool-call decisions. This SHALL be transparent to multica — multica's
pi runtime does not need to request logprobs or know about entropy.

If the upstream LLM does not support logprobs, the proxy SHALL log the absence and
continue. Entropy-gated checkpoint creation SHALL be skipped when logprobs are
unavailable, while always-event checkpoint behavior remains available.

#### Scenario: Logprobs captured for entropy

- **WHEN** a trained session's LLM traffic flows through the proxy
- **THEN** the proxy requests `logprobs=true` on each call and stores the logprobs for
  later entropy computation

#### Scenario: Upstream LLM doesn't support logprobs

- **WHEN** the upstream LLM returns an error or empty result for `logprobs=true`
- **THEN** the proxy logs the absence and continues without entropy for that interaction
  (the trajectory remains valid)

#### Scenario: Missing logprobs skip entropy-gated checkpoint

- **WHEN** a tool-call decision has no usable logprobs
- **THEN** AReaL does not create an entropy-gated checkpoint for that decision and does
  not fail the trained rollout solely because entropy is unavailable

#### Scenario: High entropy can request checkpoint creation

- **WHEN** a checkpoint-eligible tool-call decision has entropy above the configured
  threshold
- **THEN** AReaL can request Multica checkpoint creation with checkpoint kind
  `entropy_gated` and the computed entropy score
```

## openspec/changes/env-dispatch-sandbox-lifecycle/specs/env-checkpoint-resume/spec.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/specs/env-checkpoint-resume/spec.md
- Lines: 1-87
- SHA256: 766c04c3264e5471087b01f4680c3af6ba2cec7dd6762a34b42e19304d828a1d

[TRUNCATED]

```md
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

```

Full source: openspec/changes/env-dispatch-sandbox-lifecycle/specs/env-checkpoint-resume/spec.md

## openspec/changes/env-dispatch-sandbox-lifecycle/specs/env-dispatch-sandbox-lifecycle/spec.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/specs/env-dispatch-sandbox-lifecycle/spec.md
- Lines: 1-59
- SHA256: eb069aa762186ee922ca0b3c4650599550c593a0b81eef68c2561e4c8e365b69

```md
## ADDED Requirements

### Requirement: Env-dispatch sandbox-instance lifecycle handles

The system SHALL represent save/resume-capable env-dispatch environments with structured sandbox-instance references rather than opaque sandbox id strings.

Each structured reference SHALL include the Multica `sandbox_instance.id`, workspace id, node id, `local_ref`, template or base env metadata, current status, and runtime metadata needed by sandboxd resume/reconfigure flows.

#### Scenario: Rollout environment records sandbox-instance refs

- **WHEN** env-dispatch creates or assigns a sandbox for a rollout environment
- **THEN** the environment lifecycle data includes structured sandbox-instance refs for the affected agent or group

#### Scenario: Legacy sandbox id data remains readable

- **WHEN** existing env-dispatch data only has legacy raw sandbox ids
- **THEN** compatibility reads can still load the environment, but new save/resume paths prefer structured sandbox-instance refs when present

### Requirement: Per-agent environment intent

The env-dispatch request SHALL accept optional per-agent environment specs that assign individual squad agents to sandbox templates or base environments while preserving a shared Multica entity subtree.

The system MUST validate that each referenced agent belongs to the workspace/squad and that each env spec resolves to an allowed sandbox template or base environment.

#### Scenario: Valid per-agent env specs assign distinct sandboxes

- **WHEN** env-dispatch receives valid per-agent env specs for multiple squad members
- **THEN** each specified agent is assigned a sandbox instance matching its env spec while the rollout shares one Multica entity subtree

#### Scenario: Unknown agent is rejected

- **WHEN** env-dispatch receives a per-agent env spec for an agent that is not in the workspace or squad
- **THEN** the request is rejected with a validation error and no partial rollout is created

#### Scenario: Omitted per-agent env specs preserve current behavior

- **WHEN** env-dispatch receives no per-agent env specs
- **THEN** existing default or shared sandbox assignment behavior is preserved

### Requirement: Sandbox lifecycle service reuse

The system SHALL expose an internal env sandbox lifecycle service for env-dispatch and checkpointing that delegates create, save, resume, delete, and reconfigure operations to the existing sandboxd job and query machinery.

The service MUST preserve sandboxd websocket wakeups, runtime env/model metadata handling, node ownership checks, and unreachable-node force-delete behavior.

#### Scenario: Save enqueues existing stop job

- **WHEN** the lifecycle service saves a sandbox-backed environment
- **THEN** it enqueues the existing sandbox `stop` job for the target sandbox instance

#### Scenario: Resume enqueues existing resume job

- **WHEN** the lifecycle service resumes a saved sandbox-backed environment
- **THEN** it enqueues the existing sandbox `resume` job with runtime metadata needed to restart the runtime when applicable

#### Scenario: Delete preserves force-delete fallback

- **WHEN** the lifecycle service deletes a sandbox instance whose node is unreachable
- **THEN** the existing direct DB force-delete fallback remains available
```

## openspec/changes/env-dispatch-sandbox-lifecycle/specs/training-session-lifecycle/spec.md

- Source: openspec/changes/env-dispatch-sandbox-lifecycle/specs/training-session-lifecycle/spec.md
- Lines: 1-53
- SHA256: 1665ea4290f77cecb6ea78cf02f83e46b7cc9d23edbb50e5ed3d691fd9d029fb

```md
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
```


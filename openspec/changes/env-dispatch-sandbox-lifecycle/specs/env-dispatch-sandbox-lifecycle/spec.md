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

The service SHALL expose a `Create` operation that inserts a `sandbox_instance` row, enqueues the existing sandbox `create` job, and notifies the owning node, mirroring the existing sandbox instance creation handler.

#### Scenario: Save enqueues existing stop job

- **WHEN** the lifecycle service saves a sandbox-backed environment
- **THEN** it enqueues the existing sandbox `stop` job for the target sandbox instance

#### Scenario: Resume enqueues existing resume job

- **WHEN** the lifecycle service resumes a saved sandbox-backed environment
- **THEN** it enqueues the existing sandbox `resume` job with runtime metadata needed to restart the runtime when applicable

#### Scenario: Create enqueues existing create job and wakes the node

- **WHEN** the lifecycle service creates a sandbox_instance-backed environment from a template
- **THEN** it inserts a `sandbox_instance` row, enqueues the existing sandbox `create` job, and notifies the owning node

#### Scenario: Delete preserves force-delete fallback

- **WHEN** the lifecycle service deletes a sandbox instance whose node is unreachable
- **THEN** the existing direct DB force-delete fallback remains available

### Requirement: Env-dispatch sandbox_instance backend bridge

Env-dispatch SHALL create sandbox_instance-backed environments through the env sandbox lifecycle service for save/resume-capable rollouts and populate structured `SandboxInstanceRef`s on the rollout, while preserving the existing Fleet fork/boot path for non-checkpointed rollouts.

Scratch (sandbox_instance-backed) SHALL create fresh sandbox_instances from the requested template or base env, one per agent or per per-agent env spec. Branch (sandbox_instance-backed) SHALL create fresh sandbox_instances from the source env's template rather than a live fork; the Multica DB subtree continues to be copied by the existing project-subtree copy so the child continues the copied conversation in a fresh sandbox.

Checkpoint save/resume SHALL operate only on sandbox_instance refs and MUST return a typed error when requested against a Fleet-only environment. True live-state fork of a sandbox_instance is out of scope for v1.

#### Scenario: Save/resume-capable rollout creates sandbox_instance refs

- **WHEN** env-dispatch creates a save/resume-capable rollout (trained rollout or a dispatch that requests checkpointing)
- **THEN** the rollout env is backed by sandbox_instance rows created through the lifecycle service and the rollout carries structured `SandboxInstanceRef`s

#### Scenario: Branch creates fresh sandbox_instances from source template

- **WHEN** env-dispatch branches a sandbox_instance-backed env
- **THEN** it creates fresh sandbox_instances from the source env's template and copies the Multica DB subtree, without live-forking the source sandbox filesystem state

#### Scenario: Non-checkpointed rollout preserves Fleet path

- **WHEN** env-dispatch creates a rollout that is not save/resume-capable
- **THEN** the existing Fleet fork/boot path and legacy raw sandbox ids are preserved unchanged

#### Scenario: Checkpoint against Fleet-only env is rejected

- **WHEN** a checkpoint create or resume is requested against an environment that has only Fleet sandbox ids and no sandbox_instance refs
- **THEN** the system returns a typed error and does not attempt sandboxd save or resume

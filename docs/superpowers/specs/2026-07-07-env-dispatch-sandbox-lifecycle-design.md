---
comet_change: env-dispatch-sandbox-lifecycle
role: technical-design
canonical_spec: openspec
---

# Env-dispatch Sandbox Lifecycle Technical Design

## Context

Env-dispatch already creates rollout projects, issues or chat sessions, agent runs, and training-dispatch records. Separately, Multica now has sandbox node gateway support from `lijiannankai@126.com`'s sandbox lifecycle work: `sandbox_instance` persistence, sandboxd jobs, Cube create/pause/resume/delete calls, runtime reconfiguration metadata, per-user node ownership, websocket wakeups, and offline-node force-delete behavior.

This design connects those existing pieces so trained env-dispatch rollouts can save a live environment, later resume it, and let AReaL request checkpoints without knowing sandboxd internals. It does not add immutable snapshots, sandbox forks, copy-on-write branches, or a new sandbox provider.

## Adopted Approach

Multica owns sandbox lifecycle and checkpointing. AReaL remains a thin policy and HTTP client layer that can request env-dispatch, checkpoint creation/listing, and resume-from-checkpoint.

The core Multica addition is an internal env sandbox lifecycle service used by env-dispatch and checkpointing. The service wraps existing sandboxd query/job machinery instead of copying handler logic. It exposes create, save/stop, resume, delete, and reconfigure operations over structured sandbox-instance refs, while preserving websocket job wakeups, node ownership checks, runtime env/model metadata, `local_ref`, and existing force-delete fallback behavior.

Env-dispatch should continue to support legacy raw sandbox ids for compatibility, but save/resume-capable flows must prefer structured refs that include `sandbox_instance.id`, workspace id, node id, status, template or base env metadata, `local_ref`, endpoint/runtime metadata, and any ownership context needed for authorization.

## Checkpoint Semantics

Checkpoint creation is synchronous with timeout. `POST /env-checkpoint` records the checkpoint candidate, asks the lifecycle service to save each affected sandbox instance through the existing sandboxd `stop` job and Cube pause behavior, then waits for terminal save status until the configured timeout expires.

If all sandbox saves complete before timeout, the checkpoint is returned with a resumable save status. If a sandbox save fails or the timeout expires, the checkpoint records the failed or timed-out status and the API returns a typed failure response. Resume-from-checkpoint must reject failed, timed-out, or otherwise incomplete checkpoints.

The timeout should be short and configurable, for example `ENV_CHECKPOINT_SAVE_TIMEOUT`. Handler and service code should avoid hiding timeout errors as generic internal errors because AReaL needs to distinguish retryable save delays from permanent checkpoint rejection.

Resume-from-checkpoint resumes the same saved sandbox instances through existing sandboxd `resume` jobs. Public route and client naming must use `resume-from-checkpoint` / `resume_from_checkpoint`; it must not advertise branch or fork semantics.

## Snapshot Storage

For v1, checkpoint DB subtree state is stored inline as JSONB on the checkpoint row, for example `env_checkpoint.db_snapshot`. The snapshot should include the project/issue subtree needed to re-enter the same Multica state: issue, sub-issues, tasks, messages, comments, and event ordering metadata.

Inline JSONB keeps the first implementation small, transactionally simple, and easy to test. The trade-off is lower queryability and possible row growth for large subtrees. The implementation should bound captured scope to the rollout project subtree, log or meter unusually large snapshots, and keep the schema open to a later migration to object refs if size becomes a problem.

## Per-agent Environment Intent

Env-dispatch accepts optional per-agent environment specs. Specified squad members can resolve to their own sandbox template or base environment; unspecified members use existing default or shared behavior. All members still share one Multica entity subtree for the rollout.

Validation must reject unknown agents, agents outside the workspace or squad, and unknown or unauthorized env specs before creating partial rollout state. Empty per-agent env fields preserve current env-dispatch behavior.

## Sandbox Backend Bridge

Env-dispatch today creates rollout sandboxes through the cloud-runtime/Fleet proxy, producing opaque sandbox ids. That path has no pause/resume, so it cannot back checkpoint save/resume. The sandbox node gateway (`sandbox_instance` + sandboxd jobs, Cube pause/resume) is a separate system with the lifecycle semantics checkpointing needs.

To connect them, env-dispatch gains a sandbox_instance creation path through the env sandbox lifecycle service. The lifecycle service exposes a `Create` operation that mirrors the existing `CreateSandboxInstance` handler: insert a `sandbox_instance` row, enqueue the `create` sandboxd job, and notify the owning node. Rollouts that are save/resume-capable (trained rollouts, or any dispatch that requests checkpointing) create sandbox_instances through this path and populate structured `SandboxInstanceRef`s on the rollout and env. The existing Fleet fork/boot path stays for non-checkpointed rollouts with unchanged behavior.

Scratch (sandbox_instance-backed) creates fresh sandbox_instances from the requested template or base env, one per agent or per per-agent env spec. Branch (sandbox_instance-backed) creates fresh sandboxInstances from the source env's template rather than a live fork; the Multica DB subtree (issues, chat sessions, messages) is still copied by the existing `CopyProjectSubtree`, so the child continues the copied conversation in a fresh sandbox. Live sandbox filesystem state is not carried, which is acceptable because trajectory state lives in Multica's DB.

Checkpoint save/resume only operates on sandbox_instance refs. A checkpoint request against a Fleet-only env returns a typed error. True live-state fork of a sandbox_instance is deferred.

## Data Flow

1. AReaL or another caller creates an env-dispatch rollout, optionally including per-agent env specs and training intent.
2. Env-dispatch resolves sandbox-instance refs for the rollout, persists or returns them with env lifecycle data, and preserves them in trained task/session context.
3. A trained rollout reaches a structural event or high-entropy tool-call decision and asks Multica to create a checkpoint.
4. Multica captures the inline JSONB DB subtree snapshot, enqueues sandbox save/stop jobs, waits for completion until timeout, and records terminal save status.
5. AReaL lists or selects a completed checkpoint and calls resume-from-checkpoint.
6. Multica validates workspace ownership and checkpoint save status, enqueues sandbox resume jobs through the lifecycle service, and returns a rollout handle suitable for continuation.

## Error Handling

Checkpoint create should return typed outcomes for validation errors, cross-workspace access, missing sandbox-instance refs, sandbox save failure, and save timeout. Timeout and sandbox failure should leave an auditable checkpoint row with terminal non-resumable status.

Resume should reject missing, unauthorized, failed, timed-out, or pending checkpoints without enqueueing resume jobs. Delete should preserve the existing offline-node force-delete path. Entropy-gated checkpoint requests should be skipped by AReaL when logprobs are unavailable, while always-event checkpoint behavior remains available.

## Testing Strategy

Use TDD around Multica service seams first, then handlers, then AReaL client integration.

Multica service tests should cover lifecycle save/stop job enqueueing, resume job enqueueing with runtime metadata, delete force-delete fallback, missing sandbox typed errors, per-agent env validation, structured refs, and trained task/session context preservation.

Checkpoint tests should cover successful synchronous save, save timeout, sandboxd failure, inline JSONB snapshot round trip, newest-first listing, workspace authorization, failed/incomplete resume rejection, and successful resume returning a continuation handle.

AReaL tests should cover per-agent env serialization, optional field omission, checkpoint create/list/resume helpers, 403/404/409-style typed errors, entropy threshold behavior, and missing logprobs skip.

Verification should include scoped Go tests for Multica env-dispatch, sandbox lifecycle, checkpoint service/handler, and generated query users; scoped Python tests for the AReaL env-dispatch client and entropy helpers; and `openspec validate env-dispatch-sandbox-lifecycle --strict`.

## Rollout And Rollback

Gate checkpoint APIs and automatic checkpoint triggers behind configuration. Roll out lifecycle service and structured refs first, then checkpoint create/list/resume APIs, then AReaL policy calls.

Rollback is to disable checkpoint creation and resume APIs by config while leaving existing sandbox node gateway behavior intact. Existing raw sandbox env-dispatch paths continue to work through compatibility reads.

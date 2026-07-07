## 1. Investigation and Seam Confirmation

- [x] 1.1 Confirm the sandbox lifecycle commits from `lijiannankai@126.com` that introduced sandbox node gateway, sandboxd jobs, `sandbox_instance`, `local_ref`, resume/reconfigure metadata, node ownership, and force-delete behavior.
- [x] 1.2 Trace current env-dispatch state flow in `multica/server/internal/service/env_dispatch.go`, handler request parsing, training dispatch persistence, and AReaL client serialization.
- [x] 1.3 Document the transition from legacy raw sandbox ids to structured sandbox-instance refs, including compatibility reads and where new code must prefer structured refs.
- [x] 1.4 Confirm DB subtree snapshot query scope for issue, sub-issues, tasks, messages, comments, and event refs.

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

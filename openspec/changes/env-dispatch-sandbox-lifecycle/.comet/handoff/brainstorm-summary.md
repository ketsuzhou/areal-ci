# Brainstorm Summary

- Change: env-dispatch-sandbox-lifecycle
- Date: 2026-07-07

## Confirmed Technical Approach

Multica owns sandbox lifecycle and checkpointing; AReaL remains a thin policy/client layer. The implementation should add an internal `EnvSandboxLifecycle` service that wraps existing sandboxd job/query machinery for create, save/stop, resume, delete, and reconfigure. Env-dispatch should carry structured `sandbox_instance` refs plus optional per-agent env specs.

Checkpoint APIs should support create/list/resume. Checkpoint creation uses synchronous save with timeout: `POST /env-checkpoint` waits for sandboxd `stop` / Cube pause completion up to a configured timeout before returning a completed checkpoint, and records/returns typed timeout or failure status if pause does not complete. Checkpoint snapshots use inline JSONB for v1: the checkpoint row stores issue, sub-issues, tasks, messages, comments, and event ordering data directly in `env_checkpoint.db_snapshot` rather than object refs or normalized child tables.

Resume uses existing sandboxd `resume`; public API/client naming is `resume-from-checkpoint`, not branch/fork. The system does not promise immutable snapshot, fork, copy-on-write, or concurrent branch semantics.

## Key Trade-offs and Risks

- Sync checkpoint create is simpler for AReaL because checkpoint creation returns only after pause has completed or timed out.
- Sync create can exceed HTTP timeout or block handler resources under slow/offline sandbox nodes; mitigate with `ENV_CHECKPOINT_SAVE_TIMEOUT`, clear `save_status`, and typed timeout/failure responses.
- Inline JSONB is fastest for v1 implementation and tests, but less queryable and may grow large for deep issue subtrees; mitigate with bounded subtree capture, metrics/logging, and future migration to object refs if needed.
- Pause-in-place interrupts the active rollout; only training flows that expect pause should trigger checkpoint creation.
- API naming should be resume-from-checkpoint, not branch/fork, to avoid implying immutable snapshot semantics.

## Testing Strategy

Use TDD on Multica service seams first, then handler/API tests, then AReaL serialization/client tests. Multica tests should cover lifecycle service save/resume/delete, env-dispatch per-agent refs, checkpoint create/list/resume, sync timeout/failure paths, workspace authorization, checkpoint status, and config disabled behavior. AReaL tests should cover per-agent env serialization, checkpoint create/list/resume helpers, entropy threshold behavior, and missing logprobs skip. Smoke/E2E should prove a trained rollout creates a checkpoint, pause completes, and resume returns a usable rollout handle.

## Spec Patches

- Update `env-checkpoint-resume` to require synchronous checkpoint create with configured timeout.
- Change checkpoint content wording from `db_snapshot_ref` to inline JSONB `db_snapshot` for v1, while preserving room for a future object-ref migration.

# Brainstorm Summary

- Change: env-dispatch-nontraining-dag
- Date: 2026-07-21

## Confirmed Technical Approach

The deep Design Doc was authored before the Comet change was opened and serves as
the confirmed design. Approach: require an explicit `training_mode` (`*bool` at the
HTTP boundary, dereferenced and service-validated); introduce a durable
`env_dispatch_run` row keyed by project to drive `/dag` readiness independent of
`training_dispatch`; add a dual-source segment model (`areal_tensor` trainable vs
`task_messages` non-trainable) via migrations 204 and 205; route every env-dispatch
agent through one terminal/delegation seam that uses the AReaL close/export path
when `context.areal_proxy` exists and otherwise records a deterministic
`multica:<task-id>` local segment; extend the server and Python `SegmentSpec` with
`trajectory_source`, `trainable`, and `trajectory` (nullable tensor fields) and make
the assembler resolve/clear only trainable tensors.

## Key Trade-offs and Risks

- BREAKING required `training_mode` with no inference from `train_agent_id`.
- Migration backfills existing rows as `areal_tensor`/trainable; nullable AReaL
  columns are additive.
- Local trajectory carries allowlisted message-derived content; the prior "no
  message text" rule becomes source-specific (trainable segments stay refs-only).
- Best-effort recording must not change task terminal results; only malformed
  trainable segments are hard DAG-consumer errors.

## Testing Strategy

TDD per task (RED -> implement -> GREEN). Go handler/service tests cover
validation, root/readiness via `env_dispatch_run`, local segment recording, mixed
trained/non-trained recording, and zero AReaL calls in non-training mode. Python
tests cover mixed-DAG parsing and trainable-only tensor resolution/cleanup.
Cross-layer regression adds `go vet`, `graphify update`, and a secret/AReaL-call
boundary review.

## Spec Patches

None. The OpenSpec delta specs created in the open phase already capture the
requirements and acceptance scenarios.

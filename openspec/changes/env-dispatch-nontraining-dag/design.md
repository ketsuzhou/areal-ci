## Context

The env-dispatch DAG path assumes a training dispatch: `GET /dag` readiness joins
`training_dispatch`, and segment recording only happens for agents that carry an
AReaL proxy. Non-training dispatches (no train agent) and mixed squads (a trained
leader plus non-trained peers) therefore cannot produce a complete interaction DAG,
and there is no explicit mode to distinguish a path that must make zero AReaL
lifecycle calls. The deep Design Doc
(`docs/superpowers/specs/2026-07-21-env-dispatch-nontraining-dag-design.md`) and
implementation plan
(`docs/superpowers/plans/2026-07-21-env-dispatch-nontraining-dag.md`) specify the
full contract; this document records the high-level architecture decisions.

## Goals / Non-Goals

**Goals:**

- Make training vs non-training an explicit, required dispatch choice.
- Make `/dag` ready and complete in both modes through a durable dispatch identity
  independent of `training_dispatch`.
- Represent every env-dispatch agent in the DAG - trained or not - with an explicit
  trajectory source and trainability.
- Keep AReaL tensor resolution and cleanup scoped to trainable segments only.

**Non-Goals:**

- Backward-compatible inference of training mode from `train_agent_id`.
- Changing AReaL training algorithms or the tensor format.
- Adding dependencies or installing Go.
- Changing one-segment-per-task or best-effort recording semantics.

## Decisions

### Required `training_mode` at the HTTP boundary

Use a `*bool` pointer in the handler to distinguish an omitted JSON field from
`false`, dereference it before constructing `EnvDispatchInput`, and validate at the
service layer: `false` forbids `train_agent_id`/`critic_agent_id`; `true` requires
`train_agent_id`. Inferring mode from `train_agent_id` was rejected because a
non-training path must provably make zero AReaL calls.

### Durable `env_dispatch_run` identity

Introduce a row keyed by project, carrying workspace ID, `training_mode`, and a
nullable `root_task_id`. The dispatch service creates it after the project exists
and binds `root_task_id` immediately after enqueuing the leader task. `/dag` reads
readiness and completeness exclusively through this row. Keeping the
`training_dispatch` join was rejected because non-training dispatches have no such
row, so readiness could never resolve.

### Dual-source segment model

Add `trajectory_source`, `trainable`, and `trajectory` to `interaction_dag_segment`
and make the AReaL-only `trajectory_id` and `tensor_ref` nullable; backfill existing
rows as `areal_tensor` with `trainable=true`. Source-specific DB checks enforce that
trainable segments carry AReaL fields and an empty trajectory, while task-message
segments carry null AReaL fields and a persisted message-range trajectory. A
separate table for local segments was rejected to preserve one-segment-per-task and
shared topology/edges.

### One terminal/delegation seam selects the source

At a close or delegation seam, use the existing AReaL close/export path when
`context.areal_proxy` exists; otherwise upsert a deterministic `multica:<task-id>`
session/run mapping and record a `task_messages` segment from the persisted message
range. Reuse the existing edge and one-segment-per-task guards so mixed
trained/non-trained dispatches share one DAG topology.

### Assembler preserves topology, resolves only trainable

`AssembledDag` keeps all segments and edges. `SegmentSpec` gains the three
dual-source fields with nullable tensor fields. The assembler resolves tensor
references and clears shards only for `trainable=true` segments; non-trainable
segments retain their identity, local trajectory, environment snapshot, and edges,
and contribute no policy-training tensors.

## Risks / Trade-offs

- [Backward-incompatible `training_mode`] -> Mitigation: it is a required field with
  no inference; callers must update. Documented as BREAKING in the proposal.
- [Existing rows during migration] -> Mitigation: backfill as `areal_tensor`/trainable
  with `NOT NULL DEFAULT`; nullable AReaL columns are additive.
- [Local trajectory carries message-derived content] -> Mitigation: allowlisted fields
  only, sourced from persisted `task_message` columns; provider API keys are never
  serialized. The prior "no message text" rule becomes source-specific: trainable
  segments stay refs-only.
- [Recording must not change task results] -> Mitigation: recording failures are
  observability warnings, not task failures; only malformed trainable segments remain
  hard DAG-consumer errors.

## Migration Plan

1. Add `env_dispatch_run` (migration 204) and dual-source columns (migration 205)
   with backfill.
2. Wire dispatch persistence and replace the `/dag` readiness lookup.
3. Implement local segment recording and the unified close seam.
4. Extend the server and Python `SegmentSpec`; update the assembler to resolve only
   trainable tensors.
5. Verify Go/Python tests, static checks, secret boundaries, and `graphify update`.

## Open Questions

None. The deep Design Doc and implementation plan fully specify the contract,
including security and failure behavior.

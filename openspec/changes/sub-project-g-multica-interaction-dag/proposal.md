> **Re-scoped 2026-07-13.** This container was originally the Multica-side
> interaction-DAG producer; that scope was superseded and delivered by the
> `multica-v2-segment-dag-*` arc (recording-assembly + training). The container
> is reused for a new, unrelated scope: hardening the AReaL-side v2 segment-DAG
> assembly/checkpoint path against lossy-serialization and topology-loss bug
> classes. The change name is retained by operator decision; it no longer
> describes the scope.

## Why

The v2 segment-DAG arc landed the AReaL consumer path
(`SuperNodeAssembler.assemble_from_refs`, `distribute_reward_over_dag`,
`segment_dag_trainer`) and, as a side effect, fixed two latent SuperNode bugs
that had been flagged for "Phase 1b/2": lossy checkpoint serialization and
multi-edge parent non-determinism. Auditing the landed v2 path surfaced two
**residual integrity gaps** in the same bug class. They are not yet live
failures, but they will silently corrupt training data or checkpoint resume
when the v2 path grows a consumer that trips them:

1. **`SuperNode.visit_count` is dropped by the "lossless" serializer.**
   `SuperNode.to_dict()` / `from_dict()` (execution_dag.py:119-178) serialize
   every field *except* `visit_count`. The field is mutated by `branch_backup`
   (dag_backup.py:88-110) as a running MCTS branch-value aggregate, so a
   checkpoint save/restore zeros it - a fork segment resumed from checkpoint
   loses its visit count. (The *canonical* visit count used by critic/advantage
   is the separate persisted `tree_store._visit_counts` dict, already saved -
   so this is a narrower, branch-backup-specific loss, but still a real
   round-trip hole in the envelope that is documented as lossless.)

2. **`assemble_from_refs` does not populate `SuperNode.incoming_edges` /
   `outgoing_edges`.** Only the tests-only `assemble()` path populates these
   tuples (supernode_assembler.py:336-345). The live v2 path
   (`assemble_from_refs`, :108-194) leaves them `()`, carrying topology only in
   `ExecutionDAG.edges`. Today this is harmless because the v2 path never
   checkpoint-serializes assembled SuperNodes and its backup
   (`distribute_reward_over_dag`) reads `dag.edges` directly. But `to_dict()`
   reads topology from those tuples - so the moment a v2-assembled DAG is
   checkpointed (or any consumer reads the tuples instead of the DAG), topology
   serializes as empty. It is a latent footgun sitting exactly where the
   original lossy-serialization bug sat.

This change closes both gaps proactively, before the v2 path grows consumers
that trip them, and pins the v2 path with regression tests against both bug
classes.

## What Changes

- **Lossless `visit_count` serialization**: add `visit_count` to
  `SuperNode.to_dict()` / `from_dict()` (execution_dag.py) so the envelope is
  truly lossless. `from_dict` tolerates a missing key (older checkpoints) by
  defaulting to 0.
- **Topology-complete `assemble_from_refs`**: after the edge-adding loop,
  populate each SuperNode's `incoming_edges` / `outgoing_edges` tuples in
  `assemble_from_refs()` (supernode_assembler.py), mirroring the pattern in
  `assemble()` (:336-345). No behavior change to the live backup path (which
  reads `dag.edges`); this only makes the SuperNodes serialization-correct.
- **v2-path regression tests**: add (a) a round-trip test
  (`assemble_from_refs` -> `to_dict` -> `from_dict` asserts edges,
  `incoming_edges`/`outgoing_edges`, `visit_count`, env fields preserved) and
  (b) a fan-in credit test (`assemble_from_refs` with multiple incoming
  DELEGATION/COMPLETION edges -> `distribute_reward_over_dag` credits every
  parent).

## Capabilities

### New Capabilities

- `v2-segment-dag-integrity`: the AReaL v2 segment-DAG assembly path produces
  SuperNodes whose serialization is lossless (all fields round-trip, including
  `visit_count`) and whose topology is complete (`incoming_edges` /
  `outgoing_edges` populated, matching `ExecutionDAG.edges`).

### Modified Capabilities

<!-- None. -->

## Impact

- **AReaL (Python, primary)**: `customized_areal/tree_search/agents/execution_dag.py`
  (`SuperNode.to_dict` / `from_dict`), `supernode_assembler.py`
  (`assemble_from_refs`), new tests under `customized_areal/tree_search/tests/`.
  No Multica-side changes. No change to `distribute_reward_over_dag`, the
  legacy `assemble()` path (left as-is, tests-only), or the training/backup
  control flow.
- **Depends on**: the landed v2 segment-DAG arc (`multica-v2-segment-dag-training`
  + `multica-v2-segment-dag-recording-assembly`).
- **Out of scope**: auditing for other latent bug classes, Multica-side changes,
  new features, removing the dead `assemble()` path, full sandbox snapshots (F),
  verifier/reward backup (E).

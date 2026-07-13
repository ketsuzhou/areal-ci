# Comet Design Handoff

- Change: sub-project-g-multica-interaction-dag
- Phase: design
- Mode: compact
- Context hash: 632d1d366796beff22519a19ab78c9e37ed4231d15bfe450b57d9e76857170f1

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/sub-project-g-multica-interaction-dag/proposal.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/proposal.md
- Lines: 1-89
- SHA256: ca370b2a9800c3981f25a5b891920f1e3c3b46f0e83c15fbd366841dd696a4ec

[TRUNCATED]

```md
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
```

Full source: openspec/changes/sub-project-g-multica-interaction-dag/proposal.md

## openspec/changes/sub-project-g-multica-interaction-dag/design.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/design.md
- Lines: 1-150
- SHA256: 30fbd8a71c522d3cb3302d5077ccb3949d99b695ddc77b0e960b66cf4bdf85e3

[TRUNCATED]

```md
## Context

The v2 segment-DAG arc landed two assembly paths in
`customized_areal/tree_search/agents/supernode_assembler.py`:

- `assemble_from_refs` (:108-194) - the **live v2 consumer path**. Resolves each
  segment's `tensor_ref` via a `TensorResolver`, builds one `SuperNode` per
  segment with `nodes=[]` (tensors attached to `metadata["tensors"]`), adds
  typed edges to an `ExecutionDAG`, and stamps `completion_index` from
  topological order. Used by `multi_agent_workflow.py` and `segment_dag_trainer.py`.
- `assemble` (:196-413) - the **legacy path** (tests-only). Slices an agent's
  `list[Node]` by `start_turn_idx` / `end_turn_idx` and runs a 6-step algorithm
  including Step 5 parent-node-id flattening with the fan-in fix (first source
  terminal -> `parent_node_id`, rest -> `extra_parent_node_ids`).

`SuperNode.to_dict()` / `from_dict()` (execution_dag.py:119-178) are the
"single source of truth" for SuperNode serialization, reused by
`TreeCheckpointManager._serialize_super_node` (checkpoint.py:258). They were
made lossless as part of Phase 1a (`6d2f28e7`) - but `visit_count` was missed.

The v2 backup, `distribute_reward_over_dag` (dag_backup.py:39-85), walks
`dag.edges` directly and already credits every parent on a fan-in join (with
optional `CreditAssignment`). It does **not** read `SuperNode.incoming_edges` /
`outgoing_edges`. So the v2 path is *backup-correct* but *serialization-incomplete*:
the SuperNodes it builds cannot currently survive a `to_dict()` round-trip with
their topology intact.

## Goals / Non-Goals

**Goals:**

- Make `SuperNode.to_dict()` / `from_dict()` truly lossless by including
  `visit_count`.
- Make `assemble_from_refs` produce topology-complete SuperNodes
  (`incoming_edges` / `outgoing_edges` populated, consistent with
  `ExecutionDAG.edges`).
- Pin both properties with v2-path regression tests (round-trip + fan-in).

**Non-Goals:**

- Auditing for other latent bug classes (separate effort).
- Any Multica-side change.
- Removing or altering the legacy `assemble()` path (left as-is, tests-only).
- Changing `distribute_reward_over_dag` or the training/backup control flow.
- Full sandbox snapshots (Sub-project F), verifier/reward backup (E).

## Decisions

### D1 - Add `visit_count` to `to_dict()` / `from_dict()`

Serialize `visit_count` alongside the other scalar fields, defaulting to 0 on
read for backward compatibility with checkpoints written before this change.

**Rationale:** `to_dict` / `from_dict` are documented as the lossless single
source of truth; `visit_count` is the one field omitted. `branch_backup`
(dag_backup.py:88-110) mutates `parent.visit_count` as a running MCTS
branch-value aggregate, so a checkpoint save/restore currently zeros it.

**Backward compatibility:** `from_dict` reads `d.get("visit_count", 0)`, so old
checkpoints (without the key) deserialize to 0 - the dataclass default and the
correct value for a never-branched SuperNode. No migration of existing
checkpoint files is needed.

**Alternative considered:** remove `SuperNode.visit_count` entirely and rely
only on the persisted `tree_store._visit_counts` dict - rejected because
`branch_backup` uses the field as its running aggregate and the dict is keyed
by `node_id` with different update semantics (per-visit, not per-branch-return);
conflating them changes MCTS value semantics. Keep both, serialize the field.

### D2 - Populate `incoming_edges` / `outgoing_edges` in `assemble_from_refs`

After the edge-adding loop (:181-188), populate each SuperNode's edge tuples
from `edag.edges`, mirroring the existing pattern in `assemble()` (:336-345):

```python
for super_node in edag.events:
    super_node.incoming_edges = tuple(
        (e.src, e.type) for e in edag.edges if e.dst == super_node.node_id
    )
    super_node.outgoing_edges = tuple(
```

Full source: openspec/changes/sub-project-g-multica-interaction-dag/design.md

## openspec/changes/sub-project-g-multica-interaction-dag/tasks.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/tasks.md
- Lines: 1-79
- SHA256: 142149b665ee0e52db2065473be4c92481d95917bfd63f255be9c39c46aac2c7

```md
## 1. Investigation - confirm impact and test foundation

- [ ] 1.1 Confirm whether `branch_backup` (dag_backup.py:88-110) is on a current
  v2-path execution, or only a near-future tree-search-branching phase. Decide
  whether a `branch_backup` + checkpoint-resume end-to-end test is in scope for
  Task 4 or deferred.
- [ ] 1.2 Grep-confirm no non-test consumer of `assemble_from_refs` SuperNodes
  reads `incoming_edges` / `outgoing_edges` tuples today (`event_codec.py`,
  `customized_grouped_workflow.py`); document that populating them is
  non-behavioral to the live backup.
- [ ] 1.3 Confirm the existing test foundation (`test_assembler_ref_resolve.py`,
  `test_checkpoint_super.py`, `test_segment_dag_training_path.py`) and choose
  which file each new test extends.
- [ ] 1.4 Document findings; commit: `docs(G): T1 v2-path hardening investigation`.

## 2. Lossless `visit_count` serialization (TDD)

**Files:** `customized_areal/tree_search/agents/execution_dag.py`,
`customized_areal/tree_search/tests/test_checkpoint_super.py`.

- [ ] 2.1 Failing test: a `SuperNode` with `visit_count=3` round-trips through
  `to_dict()` -> `from_dict()` with `visit_count` preserved (currently resets
  to 0).
- [ ] 2.2 Failing test: `from_dict()` on a dict without `visit_count` (old
  checkpoint shape) deserializes to 0 (backward compatibility).
- [ ] 2.3 Add `visit_count` to `to_dict()` (emit the int) and `from_dict()`
  (`d.get("visit_count", 0)`).
- [ ] 2.4 Commit: `fix(supernode): serialize visit_count in to_dict/from_dict`.

## 3. Topology-complete `assemble_from_refs` (TDD)

**Files:** `customized_areal/tree_search/agents/supernode_assembler.py`,
`customized_areal/tree_search/tests/test_assembler_ref_resolve.py`.

- [ ] 3.1 Failing test: `assemble_from_refs` on a 3-segment DAG with
  DELEGATION + COMPLETION edges populates each SuperNode's
  `incoming_edges` / `outgoing_edges` to match `edag.edges` (currently `()`).
- [ ] 3.2 Failing test: a leaf segment (no incoming/outgoing edges) has empty
  tuples (regression guard).
- [ ] 3.3 Implement: after the edge-adding loop, populate
  `incoming_edges` / `outgoing_edges` from `edag.edges` (mirror `assemble()`
  :336-345).
- [ ] 3.4 Assert no behavior change to `distribute_reward_over_dag` (it reads
  `dag.edges`): existing fan-in / backup tests still pass.
- [ ] 3.5 Commit: `fix(assembler): populate incoming/outgoing_edges in assemble_from_refs`.

## 4. v2-path round-trip + fan-in regression tests

**Files:** `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`
(or `test_segment_dag_training_path.py`).

- [ ] 4.1 Round-trip test: `assemble_from_refs` -> per-SuperNode
  `to_dict()` -> `from_dict()` asserts `incoming_edges` /
  `outgoing_edges` / `visit_count` / `closing_event` / `sandbox_ids` /
  `env_state` / `metadata["tensors"]` all survive. Seed `visit_count` on a
  fork segment to assert non-zero round-trip.
- [ ] 4.2 Fan-in credit test: `assemble_from_refs` with a segment having two
  incoming DELEGATION edges -> `distribute_reward_over_dag` credits both
  parent segments (not just one).
- [ ] 4.3 Commit: `test(supernode): v2-path round-trip + fan-in regression`.

## 5. Full regression + grep sweep

- [ ] 5.1 `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ -k
  'assembler or checkpoint or supernode or dag'` (per repo test-invocation
  note; do not trust `uv run pytest` - stale venv).
- [ ] 5.2 `ruff check` (from PATH, not `.venv-test/bin/ruff`) on touched files.
- [ ] 5.3 grep sweep: `visit_count` resolves in `to_dict` / `from_dict` +
  `branch_backup`; `incoming_edges` / `outgoing_edges` populated in both
  `assemble` and `assemble_from_refs`.
- [ ] 5.4 Final whole-branch review -> READY TO MERGE / NEEDS_CHANGES.
- [ ] 5.5 Commit: `docs(G): T5 full regression + grep sweep`.

## Test runners / constraints

- AReaL tests run from `backend/areal`. Prefer `.venv-test/bin/python -m pytest`
  (the `uv run pytest` venv path is stale per repo test-invocation note).
- Use `ruff` from PATH (not `.venv-test/bin/ruff`).
- No GPU / distributed tests required (pure-Python data-model + tests).
```

## openspec/changes/sub-project-g-multica-interaction-dag/specs/v2-segment-dag-integrity/spec.md

- Source: openspec/changes/sub-project-g-multica-interaction-dag/specs/v2-segment-dag-integrity/spec.md
- Lines: 1-34
- SHA256: 13f0202f80e70ee7a83f84d4a9095170ae9b538e746601ac93e33a5a7501f2c1

```md
## ADDED Requirements

### Requirement: Lossless SuperNode serialization
The system SHALL serialize every `SuperNode` field through `to_dict()` and restore it through `from_dict()` with no loss. The serialized envelope MUST include `visit_count` alongside the identity, topology, env, reward, and turn fields. `from_dict()` MUST tolerate a missing `visit_count` key (older checkpoints) by defaulting to 0.

#### Scenario: visit_count survives a round-trip
- **WHEN** a `SuperNode` with `visit_count` set to a non-zero value is serialized via `to_dict()` and restored via `from_dict()`
- **THEN** the restored `SuperNode`'s `visit_count` MUST equal the original value

#### Scenario: Old checkpoints without visit_count deserialize to zero
- **WHEN** `from_dict()` reads a dict that does not contain a `visit_count` key
- **THEN** the restored `SuperNode`'s `visit_count` MUST be 0

### Requirement: Topology-complete v2-assembled SuperNodes
The `SuperNodeAssembler.assemble_from_refs` path SHALL populate each assembled `SuperNode`'s `incoming_edges` and `outgoing_edges` tuples so they are consistent with the `ExecutionDAG.edges` of the containing DAG. A SuperNode serialized via `to_dict()` after v2 assembly MUST retain its full edge topology.

#### Scenario: v2 assembly populates edge tuples
- **WHEN** `assemble_from_refs` builds an `ExecutionDAG` with typed edges between segments
- **THEN** every assembled `SuperNode`'s `incoming_edges` and `outgoing_edges` MUST match the edges in the `ExecutionDAG` (same sources/destinations and edge types)

#### Scenario: v2-assembled topology survives serialization
- **WHEN** a SuperNode produced by `assemble_from_refs` is serialized via `to_dict()` and restored via `from_dict()`
- **THEN** the restored SuperNode's `incoming_edges` and `outgoing_edges` MUST equal the originals

#### Scenario: Leaf segment has empty edge tuples
- **WHEN** a segment with no incoming or outgoing edges is assembled
- **THEN** its `incoming_edges` and `outgoing_edges` MUST both be empty

### Requirement: v2-path fan-in credit is preserved
The v2 assembly path MUST NOT regress fan-in reward credit. When a segment has multiple incoming blocking edges (DELEGATION or COMPLETION), `distribute_reward_over_dag` MUST credit every parent segment.

#### Scenario: Fan-in join credits all parents on the v2 path
- **WHEN** `assemble_from_refs` produces a segment with two incoming DELEGATION edges and `distribute_reward_over_dag` distributes a terminal reward backward
- **THEN** both parent segments MUST receive non-zero credit
```


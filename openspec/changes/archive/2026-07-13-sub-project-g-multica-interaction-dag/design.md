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
        (e.dst, e.type) for e in edag.edges if e.src == super_node.node_id
    )
```

**Rationale:** `to_dict()` serializes topology from these tuples, not from
`edag.edges`. Leaving them empty means a serialized v2-assembled SuperNode
loses its topology - the exact shape of the original Phase 1b/2
lossy-checkpoint bug, relocated to the v2 path. Populating them is
O(edges x nodes) and runs once per assembly; cost is negligible.

**Alternative considered:** document the tuples as intentionally empty on the
v2 path and have `to_dict()` fall back to `edag.edges` - rejected because
`to_dict()` is a `SuperNode` method with no reference to the containing
`ExecutionDAG`, so it cannot fall back. Populating at assembly time is the
clean fix.

**No behavior change to the live backup:** `distribute_reward_over_dag` reads
`dag.edges`, not the tuples, so populating them does not alter reward backup.
The change only makes the SuperNodes serialization-correct.

### D3 - v2-path regression tests (round-trip + fan-in)

Add tests in `customized_areal/tree_search/tests/`:

- **Round-trip**: build an `AssembledDag` with >=3 segments and mixed edge
  types (DELEGATION + COMPLETION + a leaf), run `assemble_from_refs`, then
  `SuperNode.to_dict()` -> `from_dict()` on each SuperNode, asserting
  `incoming_edges` / `outgoing_edges` / `visit_count` / `closing_event` /
  `sandbox_ids` / `env_state` / `metadata["tensors"]` survive. Seed
  `visit_count` on a fork segment to assert it round-trips non-zero.
- **Fan-in credit**: build an `AssembledDag` with a segment that has two
  incoming DELEGATION edges, run `assemble_from_refs`, then
  `distribute_reward_over_dag` with a terminal reward, asserting both parent
  segments receive credit (not just one).

**Rationale:** the original Phase 1b/2 bugs were latent because Phase 1a tests
were single-agent and never exercised fan-in or checkpoint resume. These tests
pin the v2 path against both bug classes specifically.

## Risks / Trade-offs

- **`from_dict` backward compatibility** - old checkpoints lack `visit_count`:
  mitigated by `d.get("visit_count", 0)` (D1). Old checkpoints deserialize to
  the dataclass default; no migration required.
- **Populating edge tuples changes SuperNode state seen by any consumer that
  reads them** - the only non-test readers are `event_codec.py` (builds edges
  from the tuples) and `customized_grouped_workflow.py` (sets + reads them),
  neither of which is on the `assemble_from_refs` path. So populating them on
  the v2 path cannot regress those consumers. Verified by grep.
- **No live failure is being fixed** - this is proactive hardening. The risk
  of NOT doing it is silent topology/visit-count loss when the v2 path grows a
  checkpoint or a tuple-reading consumer.

## Migration Plan

No database migration (AReaL-side, in-memory data model only). No Multica
changes. The change is additive to serialization (one new optional key) and
non-behavioral to the live backup path. Rollback: revert the two code edits;
checkpoints written with the key remain readable by old code only if old code
is also patched to tolerate it - but since the field defaults to 0 on the
dataclass, old code reading a new checkpoint simply ignores the extra key
(`from_dict` pre-change does not read it), so the round-trip is safe in both
directions.

## Open Questions

- Whether `branch_backup` is on any current v2-path execution (versus a
  near-future tree-search-branching phase). Does not change the fix - the
  field should round-trip regardless - but determines whether a test should
  cover `branch_backup` + checkpoint-resume end-to-end. Resolve in Task 1.

---
comet_change: sub-project-g-multica-interaction-dag
role: technical-design
canonical_spec: openspec
archived-with: 2026-07-13-sub-project-g-multica-interaction-dag
status: final
---

# v2 Segment-DAG Path Hardening — Technical Design

## Context

The v2 segment-DAG arc (`multica-v2-segment-dag-training` +
`multica-v2-segment-dag-recording-assembly`) landed the AReaL consumer path and,
as a side effect, fixed two latent SuperNode bugs that had been flagged for
"Phase 1b/2": lossy checkpoint serialization and multi-edge parent
non-determinism. Auditing the landed v2 path surfaced two **residual integrity
gaps** in the same bug class. They are not live failures today, but they sit
exactly where the fixed bugs sat and will silently corrupt data when the v2
path grows a consumer that trips them.

This design closes both gaps proactively and pins the v2 path with regression
tests. Canonical capability spec: `openspec/changes/sub-project-g-multica-interaction-dag/specs/v2-segment-dag-integrity/spec.md`.

### The two assembly paths

`customized_areal/tree_search/agents/supernode_assembler.py` holds two paths:

- `assemble_from_refs` (:108-194) — the **live v2 consumer path**. Resolves each
  segment's `tensor_ref` via a `TensorResolver`, builds one `SuperNode` per
  segment with `nodes=[]` (tensors attached to `metadata["tensors"]`), adds
  typed edges to an `ExecutionDAG`, stamps `completion_index` from topological
  order. Used by `multi_agent_workflow.py` and `segment_dag_trainer.py`.
- `assemble` (:196-413) — the **legacy path** (tests-only). Slices an agent's
  `list[Node]` by `start_turn_idx`/`end_turn_idx`; runs a 6-step algorithm with
  Step 5 parent-node-id flattening (the fan-in fix lives here, :385-400).

### Serialization

`SuperNode.to_dict()` / `from_dict()` (execution_dag.py:119-178) are the
"single source of truth" for SuperNode serialization, reused by
`TreeCheckpointManager._serialize_super_node` (checkpoint.py:258). They were
made lossless in Phase 1a (`6d2f28e7`) — but `visit_count` was missed.

### Backup

`distribute_reward_over_dag` (dag_backup.py:39-85) walks `dag.edges` directly
and credits every parent on a fan-in join (optional `CreditAssignment`). It does
**not** read `SuperNode.incoming_edges` / `outgoing_edges`. The v2 path is
therefore *backup-correct* but *serialization-incomplete*.

## Resolved Investigation

Two questions from the open-phase design, answered by code investigation:

**Q1 — Is `branch_backup` on the v2 path?**
No. `branch_backup` (dag_backup.py:88-110) is called only from
`customized_grouped_workflow.py:2092` (the Phase 1a / grouped-workflow path).
It is never called from `segment_dag_trainer.py` or `multi_agent_workflow.py`.
**Consequence:** on the v2 path today, `SuperNode.visit_count` is always 0
(only `branch_backup` mutates it). Serializing it is forward-looking hardening
for when `branch_backup` is wired to a v2 tree-search-branching phase. No
synthetic `branch_backup` + checkpoint-resume end-to-end test is in scope
(it would test a path that does not exist).

**Q2 — Do v2 live callers read the edge tuples or checkpoint?**
No. `segment_dag_trainer` (the live consumer) uses only
`edag.topological_order()` + `assemble_node_advantages(...)` (lines 211-216). It
never touches `incoming_edges` / `outgoing_edges` and never calls `to_dict` or
checkpoint. `assemble_node_advantages` (dag_advantage.py) reads `edag.edges`,
not the tuples (verified by grep). **Consequence:** populating the tuples in
`assemble_from_refs` is provably non-behavioral to the live path.

## Approach

**Approach A — serialize `visit_count` + simple round-trip test.** Makes the
envelope truly lossless; the test pins the property by manually seeding
`visit_count`. Does not fabricate a `branch_backup` + resume E2E (branch_backup
isn't on the v2 path).

Rejected:
- **B. Skip `visit_count`, document as known gap** — it is always 0 on the v2
  path today, so there is no current loss; but the `to_dict` docstring claims
  "lossless," which is false without it. Proactive hardening was chosen.
- **C. Serialize + synthetic `branch_backup` + resume E2E** — over-scopes a
  path that does not exist on the v2 line today.

## Design

### Change 1 — Lossless `visit_count` serialization

**File:** `customized_areal/tree_search/agents/execution_dag.py`

`to_dict()` (:119-147): add `"visit_count": self.visit_count` to the emitted
dict (alongside `value`, `process_reward`, `outcome_reward`).

`from_dict()` (:149-178): add `visit_count=d.get("visit_count", 0)` to the
`cls(...)` constructor call.

**Backward compatibility:** `from_dict` reads `d.get("visit_count", 0)`, so old
checkpoints (without the key) deserialize to 0 — the dataclass default and the
correct value for a never-branched SuperNode. New checkpoints with the key are
read by old code as an extra ignored key (old `from_dict` does not read it), so
the round-trip is safe in both directions. **No migration of existing
checkpoint files is needed.**

**Rationale:** `to_dict` / `from_dict` are documented as the lossless single
source of truth; `visit_count` is the one field omitted. `branch_backup`
mutates `parent.visit_count` as a running MCTS branch-value aggregate; without
serialization a checkpoint save/restore zeros it.

### Change 2 — Topology-complete `assemble_from_refs`

**File:** `customized_areal/tree_search/agents/supernode_assembler.py`

After the edge-adding loop (:181-188) and the `completion_index` stamping
(:192-193) — order-independent, placed after to match `assemble()`'s
ordering — populate each SuperNode's edge tuples from `edag.edges`, mirroring
`assemble()` (:336-345):

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
`edag.edges` (it is a `SuperNode` method with no reference to the containing
`ExecutionDAG`). Leaving them empty means a serialized v2-assembled SuperNode
loses its topology — the exact shape of the original Phase 1b/2
lossy-checkpoint bug, relocated to the v2 path. Cost is O(edges × nodes), once
per assembly — negligible.

**Non-behavioral to the live backup:** `distribute_reward_over_dag` and
`assemble_node_advantages` read `dag.edges`, not the tuples (Q2). Populating
them cannot regress the live path; it only makes the SuperNodes
serialization-correct. The only non-test readers of the tuples
(`event_codec.py`, `customized_grouped_workflow.py`) are not on the
`assemble_from_refs` path.

**Rejected alternative:** have `to_dict()` fall back to `edg.edges` —
impossible, `to_dict` has no DAG reference. Populating at assembly time is the
clean fix.

### Change 3 — v2-path regression tests

**Files:** `customized_areal/tree_search/tests/test_checkpoint_super.py`,
`customized_areal/tree_search/tests/test_assembler_ref_resolve.py`.

- **`visit_count` round-trip** (test_checkpoint_super.py): a `SuperNode` with
  `visit_count=3` round-trips through `to_dict()` -> `from_dict()` preserved.
- **Old-checkpoint compat** (test_checkpoint_super.py): `from_dict()` on a dict
  without `visit_count` deserializes to 0.
- **Tuple population** (test_assembler_ref_resolve.py): `assemble_from_refs` on
  a 3-segment DAG with DELEGATION + COMPLETION edges populates each SuperNode's
  `incoming_edges` / `outgoing_edges` to match `edag.edges`.
- **Leaf-empty guard** (test_assembler_ref_resolve.py): a leaf segment has
  empty tuples.
- **Round-trip** (test_assembler_ref_resolve.py): `assemble_from_refs` ->
  per-SuperNode `to_dict()` -> `from_dict()` asserts `incoming_edges` /
  `outgoing_edges` / `visit_count` / `closing_event` / `sandbox_ids` /
  `env_state` / `metadata["tensors"]` survive. Seed `visit_count` on a fork
  segment to assert non-zero round-trip.
- **Fan-in credit** (test_assembler_ref_resolve.py): `assemble_from_refs` with
  a segment having two incoming DELEGATION edges -> `distribute_reward_over_dag`
  credits both parent segments.

## Testing Strategy

TDD: failing test first, then implement, for Changes 1 and 2. Change 3 is pure
regression (no implementation). Run from `backend/areal` with
`.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ -k
'assembler or checkpoint or supernode or dag'` (the `uv run pytest` venv path is
stale per repo test-invocation note). `ruff check` from PATH (not
`.venv-test/bin/ruff`). No GPU / distributed tests — pure-Python data model.

Existing fan-in / backup tests must continue to pass (they read `dag.edges`);
they double as the non-regression guard for Change 2.

## Boundary Conditions & Risks

- **Old checkpoint compat** — `d.get("visit_count", 0)` (Change 1). No
  migration. Round-trip safe in both directions.
- **Tuple population order vs `completion_index`** — independent (tuples from
  `edag.edges`, index from `topological_order`). Placed after the topological
  stamping to match `assemble()`'s phase ordering.
- **`assemble_node_advantages` reading populated tuples** — verified it reads
  `edg.edges` / topological order, not tuples; no regression. If a future
  consumer reads tuples, populated > empty, so this is strictly safer.
- **No live failure is being fixed** — proactive hardening. Rollback = revert
  the two code edits; checkpoints remain readable both ways.
- **`branch_backup` not on v2 path** — `visit_count` is always 0 on the v2 path
  today, so Change 1 is observably a no-op until `branch_backup` is wired to v2.
  The test seeds the field manually to pin the lossless property regardless.

## Out of Scope

- Auditing for other latent bug classes (separate effort).
- Any Multica-side change.
- Removing or altering the legacy `assemble()` path (left as-is, tests-only).
- Changing `distribute_reward_over_dag` or the training/backup control flow.
- Full sandbox snapshots (Sub-project F), verifier / reward backup (E).
- A `branch_backup` + checkpoint-resume end-to-end test (branch_backup is not
  on the v2 path).

## Open Questions

Both resolved (Q1, Q2 above). No outstanding questions.

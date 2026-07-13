## 1. Investigation - confirm impact and test foundation

- [x] 1.1 Confirm whether `branch_backup` (dag_backup.py:88-110) is on a current
  v2-path execution, or only a near-future tree-search-branching phase. Decide
  whether a `branch_backup` + checkpoint-resume end-to-end test is in scope for
  Task 4 or deferred.
- [x] 1.2 Grep-confirm no non-test consumer of `assemble_from_refs` SuperNodes
  reads `incoming_edges` / `outgoing_edges` tuples today (`event_codec.py`,
  `customized_grouped_workflow.py`); document that populating them is
  non-behavioral to the live backup.
- [x] 1.3 Confirm the existing test foundation (`test_assembler_ref_resolve.py`,
  `test_checkpoint_super.py`, `test_segment_dag_training_path.py`) and choose
  which file each new test extends.
- [x] 1.4 Document findings; commit: `docs(G): T1 v2-path hardening investigation`.

## 2. Lossless `visit_count` serialization (TDD)

**Files:** `customized_areal/tree_search/agents/execution_dag.py`,
`customized_areal/tree_search/tests/test_checkpoint_super.py`.

- [x] 2.1 Failing test: a `SuperNode` with `visit_count=3` round-trips through
  `to_dict()` -> `from_dict()` with `visit_count` preserved (currently resets
  to 0).
- [x] 2.2 Failing test: `from_dict()` on a dict without `visit_count` (old
  checkpoint shape) deserializes to 0 (backward compatibility).
- [x] 2.3 Add `visit_count` to `to_dict()` (emit the int) and `from_dict()`
  (`d.get("visit_count", 0)`).
- [x] 2.4 Commit: `fix(supernode): serialize visit_count in to_dict/from_dict`.

## 3. Topology-complete `assemble_from_refs` (TDD)

**Files:** `customized_areal/tree_search/agents/supernode_assembler.py`,
`customized_areal/tree_search/tests/test_assembler_ref_resolve.py`.

- [x] 3.1 Failing test: `assemble_from_refs` on a 3-segment DAG with
  DELEGATION + COMPLETION edges populates each SuperNode's
  `incoming_edges` / `outgoing_edges` to match `edag.edges` (currently `()`).
- [x] 3.2 Failing test: a leaf segment (no incoming/outgoing edges) has empty
  tuples (regression guard).
- [x] 3.3 Implement: after the edge-adding loop, populate
  `incoming_edges` / `outgoing_edges` from `edag.edges` (mirror `assemble()`
  :336-345).
- [x] 3.4 Assert no behavior change to `distribute_reward_over_dag` (it reads
  `dag.edges`): existing fan-in / backup tests still pass.
- [x] 3.5 Commit: `fix(assembler): populate incoming/outgoing_edges in assemble_from_refs`.

## 4. v2-path round-trip + fan-in regression tests

**Files:** `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`
(or `test_segment_dag_training_path.py`).

- [x] 4.1 Round-trip test: `assemble_from_refs` -> per-SuperNode
  `to_dict()` -> `from_dict()` asserts `incoming_edges` /
  `outgoing_edges` / `visit_count` / `closing_event` / `sandbox_ids` /
  `env_state` / `metadata["tensors"]` all survive. Seed `visit_count` on a
  fork segment to assert non-zero round-trip.
- [x] 4.2 Fan-in credit test: `assemble_from_refs` with a segment having two
  incoming DELEGATION edges -> `distribute_reward_over_dag` credits both
  parent segments (not just one).
- [x] 4.3 Commit: `test(supernode): v2-path round-trip + fan-in regression`.

## 5. Full regression + grep sweep

- [x] 5.1 `.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ -k
  'assembler or checkpoint or supernode or dag'` (per repo test-invocation
  note; do not trust `uv run pytest` - stale venv).
- [x] 5.2 `ruff check` (from PATH, not `.venv-test/bin/ruff`) on touched files.
- [x] 5.3 grep sweep: `visit_count` resolves in `to_dict` / `from_dict` +
  `branch_backup`; `incoming_edges` / `outgoing_edges` populated in both
  `assemble` and `assemble_from_refs`.
- [x] 5.4 Final whole-branch review -> READY TO MERGE / NEEDS_CHANGES.
- [x] 5.5 Commit: `docs(G): T5 full regression + grep sweep`.

## Test runners / constraints

- AReaL tests run from `backend/areal`. Prefer `.venv-test/bin/python -m pytest`
  (the `uv run pytest` venv path is stale per repo test-invocation note).
- Use `ruff` from PATH (not `.venv-test/bin/ruff`).
- No GPU / distributed tests required (pure-Python data-model + tests).

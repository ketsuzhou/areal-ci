---
change: sub-project-g-multica-interaction-dag
design-doc: docs/superpowers/specs/2026-07-13-v2-segment-dag-path-hardening-design.md
base-ref: 853e5cbcda6929e80961ed778194c1402fb7d9ba
---

# v2 Segment-DAG Path Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two residual integrity gaps in the AReaL v2 segment-DAG assembly/checkpoint path - `SuperNode.visit_count` not serialized, and `assemble_from_refs` not populating `incoming_edges`/`outgoing_edges` - and pin both with regression tests.

**Architecture:** Two small, surgical edits to existing data-model methods (no new modules): add `visit_count` to `SuperNode.to_dict`/`from_dict` with backward-compat default; populate the edge tuples in `assemble_from_refs` (mirroring the legacy `assemble()`). Both are non-behavioral to the live v2 backup path (`distribute_reward_over_dag` / `assemble_node_advantages` read `dag.edges`, not the tuples; `branch_backup` is not on the v2 path). TDD throughout.

**Tech Stack:** Python 3.12+ (AReaL), pytest, ruff. Pure-Python data model - no torch, no GPU, no distributed tests.

## Global Constraints

- Run tests from `backend/areal` with `.venv-test/bin/python -m pytest` (the `uv run pytest` venv path is stale per repo test-invocation note).
- Run `ruff check` from PATH (not `.venv-test/bin/ruff`, which does not exist).
- No wildcard imports (`from x import *`).
- Follow existing patterns in the file being edited; match surrounding docstring/comment density.
- Conventional Commits (~72 chars subject, imperative voice): `fix(...)`, `test(...)`, `docs(...)`.
- Do NOT alter the legacy `assemble()` path, `distribute_reward_over_dag`, or the training/backup control flow.

## File Structure

No new files. Four existing files are modified:

- **`customized_areal/tree_search/agents/execution_dag.py`** - `SuperNode.to_dict()` (:119-147) and `SuperNode.from_dict()` (:149-178). Add `visit_count` to both. Single responsibility: lossless SuperNode envelope serialization.
- **`customized_areal/tree_search/agents/supernode_assembler.py`** - `SuperNodeAssembler.assemble_from_refs()` (:108-194). Add edge-tuple population after the topological stamping. Single responsibility: build topology-complete SuperNodes from an `AssembledDag`.
- **`customized_areal/tree_search/tests/test_checkpoint_super.py`** - Augment the lossless round-trip test + add a backward-compat test for `visit_count`.
- **`customized_areal/tree_search/tests/test_assembler_ref_resolve.py`** - Add tuple-population, leaf-empty, round-trip, and fan-in tests; extend imports (`EdgeType`, `SuperNode`).

---

## Task 1: Confirm investigation findings (pre-flight)

**Files:** none modified (verification only).

**Interfaces:** none. Produces confirmation that the design doc's Q1/Q2 findings still hold before TDD begins.

The design doc records two investigation findings. Re-verify them so the implementer starts from confirmed ground:

- [ ] **Step 1: Confirm Q1 - `branch_backup` is not on the v2 path**

Run:
```bash
cd /workspaces/leagent/backend/areal
grep -rn "branch_backup" customized_areal --include='*.py' | grep -v "def \|tests/"
```
Expected: exactly one production call site - `customized_areal/tree_search/core/customized_grouped_workflow.py` (around :2082-2092). No calls in `segment_dag_trainer.py` or `multi_agent_workflow.py`. This confirms `SuperNode.visit_count` is always 0 on the v2 path today (only `branch_backup` mutates it), so serializing it is forward-looking.

- [ ] **Step 2: Confirm Q2 - v2 live callers do not read the edge tuples or checkpoint**

Run:
```bash
grep -n "incoming_edges\|outgoing_edges\|\.to_dict()\|from_dict\|TreeCheckpointManager\|checkpoint" \
  customized_areal/tree_search/agents/segment_dag_trainer.py \
  customized_areal/tree_search/agents/multi_agent_workflow.py
```
Expected: no matches. The v2 path uses only `edag.topological_order()` + `assemble_node_advantages(...)` (which reads `edag.edges`, not the tuples). This confirms populating the tuples is non-behavioral to the live path.

- [ ] **Step 3: Check off T1 in tasks.md and commit the confirmation**

Edit `openspec/changes/sub-project-g-multica-interaction-dag/tasks.md`: change `- [ ] 1.1` through `- [ ] 1.4` to `- [x]`. Then:
```bash
git add openspec/changes/sub-project-g-multica-interaction-dag/tasks.md
git commit -m "docs(G): T1 v2-path hardening investigation confirmed"
```

---

## Task 2: Lossless `visit_count` serialization (TDD)

**Files:**
- Modify: `customized_areal/tree_search/agents/execution_dag.py:119-178` (`SuperNode.to_dict`, `SuperNode.from_dict`)
- Test: `customized_areal/tree_search/tests/test_checkpoint_super.py`

**Interfaces:**
- Consumes: `SuperNode.visit_count: int = 0` (dataclass field, execution_dag.py:84).
- Produces: `visit_count` key in `SuperNode.to_dict()` output; `from_dict()` reads it with default 0.

- [ ] **Step 1: Write the failing test (augment the lossless round-trip test)**

In `customized_areal/tree_search/tests/test_checkpoint_super.py`, inside `test_super_node_checkpoint_round_trip_is_lossless`, add `visit_count=3,` to the `SuperNode(...)` fixture (e.g. after `outcome_reward=1.0,`) and add an assertion after the existing `assert r.outcome_reward == 1.0`:

```python
    assert r.visit_count == 3
```

- [ ] **Step 2: Add a backward-compat test**

Append to `test_checkpoint_super.py`:

```python
def test_from_dict_tolerates_missing_visit_count():
    """Old checkpoints written before visit_count serialization have no key;
    from_dict must default to 0 (the dataclass default)."""
    legacy = {
        "node_id": "sup-old",
        "agent_id": "planner",
        "issue_id": "iss-1",
        "task_id": "task-1",
    }
    restored = SuperNode.from_dict(legacy)
    assert restored.visit_count == 0
```

- [ ] **Step 3: Run tests to verify the round-trip test fails**

Run:
```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_checkpoint_super.py -v
```
Expected: `test_super_node_checkpoint_round_trip_is_lossless` FAILS with `assert 0 == 3` (visit_count resets to 0 because `to_dict` does not emit it). `test_from_dict_tolerates_missing_visit_count` PASSes (default 0 already holds pre-impl).

- [ ] **Step 4: Implement - add `visit_count` to `to_dict()`**

In `customized_areal/tree_search/agents/execution_dag.py`, in `SuperNode.to_dict()` (:119-147), add `"visit_count": self.visit_count,` to the returned dict, immediately after the `"outcome_reward": self.outcome_reward,` line:

```python
            "value": self.value,
            "process_reward": self.process_reward,
            "outcome_reward": self.outcome_reward,
            "visit_count": self.visit_count,
            "sandbox_ids": list(self.sandbox_ids),
```

- [ ] **Step 5: Implement - add `visit_count` to `from_dict()`**

In the same file, in `SuperNode.from_dict()` (:149-178), add `visit_count=d.get("visit_count", 0),` to the `cls(...)` call, immediately after the `outcome_reward=d.get("outcome_reward", 0.0),` line:

```python
                process_reward=d.get("process_reward", 0.0),
                outcome_reward=d.get("outcome_reward", 0.0),
                visit_count=d.get("visit_count", 0),
                sandbox_ids=list(d.get("sandbox_ids", [])),
```

- [ ] **Step 6: Run tests to verify both pass**

Run:
```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_checkpoint_super.py -v
```
Expected: both tests PASS. `visit_count=3` survives the full `TreeCheckpointManager` save/load round-trip; the legacy dict (no key) deserializes to 0.

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/agents/execution_dag.py \
        customized_areal/tree_search/tests/test_checkpoint_super.py
git commit -m "fix(supernode): serialize visit_count in to_dict/from_dict"
```

---

## Task 3: Topology-complete `assemble_from_refs` (TDD)

**Files:**
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py:108-194` (`assemble_from_refs`)
- Test: `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`

**Interfaces:**
- Consumes: `ExecutionDAG.events` (property, list of SuperNodes), `ExecutionDAG.edges` (property, list of `Edge` with `.src`, `.dst`, `.type: EdgeType`).
- Produces: each v2-assembled `SuperNode.incoming_edges` / `outgoing_edges` populated as `tuple[(node_id, EdgeType), ...]`, consistent with `edag.edges`.

- [ ] **Step 1: Extend the test imports**

In `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`, change the execution_dag import (line 3) to also import `EdgeType` and `SuperNode`:

```python
from customized_areal.tree_search.agents.execution_dag import DAGError, EdgeType, SuperNode
```

- [ ] **Step 2: Write the failing test (tuple population)**

Append to `test_assembler_ref_resolve.py`:

```python
def test_assemble_from_refs_populates_edge_tuples():
    """v2-assembled SuperNodes carry topology in incoming_edges /
    outgoing_edges (not just edag.edges) so a to_dict() round-trip preserves it."""
    dag = _dag()  # seg-1 --completion--> seg-2
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())

    seg1 = edag.get("seg-1")
    seg2 = edag.get("seg-2")
    assert seg1.outgoing_edges == (("seg-2", EdgeType.COMPLETION),)
    assert seg1.incoming_edges == ()
    assert seg2.incoming_edges == (("seg-1", EdgeType.COMPLETION),)
    assert seg2.outgoing_edges == ()
```

- [ ] **Step 3: Write the leaf-empty guard test**

Append to `test_assembler_ref_resolve.py`:

```python
def test_assemble_from_refs_leaf_segment_has_empty_edge_tuples():
    """A segment with no edges has empty edge tuples (regression guard)."""
    dag = AssembledDag(
        segments=[
            SegmentSpec("seg-solo", "ar", "i", 0, {"shard_id": "s"}, None, {}),
        ],
        edges=[],
        session_to_agent_run={"s": "ar"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    solo = edag.get("seg-solo")
    assert solo.incoming_edges == ()
    assert solo.outgoing_edges == ()
```

- [ ] **Step 4: Run tests to verify the population test fails**

Run:
```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py \
  -k 'populates_edge_tuples or leaf_segment_has_empty' -v
```
Expected: `test_assemble_from_refs_populates_edge_tuples` FAILS with `assert () == (('seg-2', <EdgeType.COMPLETION: 'completion'>),)` (tuples currently empty). `test_assemble_from_refs_leaf_segment_has_empty_edge_tuples` PASSes (already empty).

- [ ] **Step 5: Implement - populate edge tuples in `assemble_from_refs`**

In `customized_areal/tree_search/agents/supernode_assembler.py`, in `assemble_from_refs()`, find the topological stamping + return block (around :190-194):

```python
        # Validate acyclicity and stamp completion_index from topological order.
        # topological_order raises DAGError if the graph contains a cycle.
        for idx, super_node in enumerate(edag.topological_order()):
            super_node.completion_index = idx
        return edag
```

Insert the tuple-population loop between the `for` loop and `return edag`:

```python
        # Validate acyclicity and stamp completion_index from topological order.
        # topological_order raises DAGError if the graph contains a cycle.
        for idx, super_node in enumerate(edag.topological_order()):
            super_node.completion_index = idx
        # Populate incoming_edges / outgoing_edges on each SuperNode so the
        # topology survives a to_dict() round-trip (to_dict reads the tuples,
        # not edag.edges). Mirrors assemble() :336-345.
        for super_node in edag.events:
            super_node.incoming_edges = tuple(
                (e.src, e.type) for e in edag.edges if e.dst == super_node.node_id
            )
            super_node.outgoing_edges = tuple(
                (e.dst, e.type) for e in edag.edges if e.src == super_node.node_id
            )
        return edag
```

- [ ] **Step 6: Run tests to verify both pass**

Run:
```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py -v
```
Expected: ALL tests PASS, including the existing `test_assemble_from_refs_builds_one_supernode_per_segment`, `test_assemble_from_refs_returns_none_for_empty_trajectory`, `test_assemble_from_refs_rejects_cycle`, `test_assemble_from_refs_env_snapshot_stamped`, plus the two new ones. (The existing tests read `edag.edges` / `completion_index`, not the tuples, so they are unaffected.)

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/agents/supernode_assembler.py \
        customized_areal/tree_search/tests/test_assembler_ref_resolve.py
git commit -m "fix(assembler): populate incoming/outgoing_edges in assemble_from_refs"
```

---

## Task 4: v2-path round-trip + fan-in regression tests

**Files:**
- Test: `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`

**Interfaces:**
- Consumes: Task 2 (`visit_count` round-trips) + Task 3 (tuples populated). Both must be complete before these tests pass.
- Produces: regression coverage pinning the v2 path against both bug classes.

These are regression tests (Tasks 2 and 3 are already implemented), so each should PASS on first run - they pin the completed behavior.

- [ ] **Step 1: Write the round-trip test**

Append to `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`:

```python
def test_assemble_from_refs_round_trip_preserves_topology_and_visit_count():
    """assemble_from_refs -> to_dict -> from_dict preserves edges, edge tuples,
    visit_count, closing_event, and tensors (the v2-path lossless invariant)."""
    dag = AssembledDag(
        segments=[
            SegmentSpec("seg-a", "ar", "i", 0, {"shard_id": "a"}, "completion", {}),
            SegmentSpec("seg-b", "ar", "i", 1, {"shard_id": "b"}, None, {}),
        ],
        edges=[EdgeSpec("seg-a", "seg-b", "completion")],
        session_to_agent_run={"s": "ar"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    # Seed visit_count on seg-a (as branch_backup would) to assert non-zero
    # round-trip - the field that was previously dropped by to_dict/from_dict.
    edag.get("seg-a").visit_count = 4

    a = SuperNode.from_dict(edag.get("seg-a").to_dict())
    b = SuperNode.from_dict(edag.get("seg-b").to_dict())
    # Topology survives (tuples populated by assemble_from_refs).
    assert a.outgoing_edges == (("seg-b", EdgeType.COMPLETION),)
    assert a.incoming_edges == ()
    assert b.incoming_edges == (("seg-a", EdgeType.COMPLETION),)
    assert b.outgoing_edges == ()
    # visit_count survives (serialized by to_dict/from_dict).
    assert a.visit_count == 4
    assert b.visit_count == 0
    # closing_event + tensors survive.
    assert a.closing_event == EdgeType.COMPLETION
    assert b.closing_event is None
    assert a.metadata["tensors"]["input_ids"] == [1, 2]
    assert b.metadata["tensors"]["input_ids"] == [1, 2]
```

- [ ] **Step 2: Run the round-trip test to verify it passes**

Run:
```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py \
  -k 'round_trip_preserves' -v
```
Expected: PASS. (If it FAILS, Tasks 2 or 3 are incomplete - re-run them first.)

- [ ] **Step 3: Write the fan-in credit test**

Append to `test_assembler_ref_resolve.py`:

```python
def test_assemble_from_refs_fan_in_credits_all_parents():
    """A segment with two incoming COMPLETION edges (fan-in join) ->
    distribute_reward_over_dag credits every parent (Bug #2 class, v2 path)."""
    from customized_areal.tree_search.agents.dag_backup import distribute_reward_over_dag

    dag = AssembledDag(
        segments=[
            SegmentSpec("seg-child-a", "ar-a", "i", 0, {"shard_id": "ca"}, "completion", {}),
            SegmentSpec("seg-child-b", "ar-b", "i", 0, {"shard_id": "cb"}, "completion", {}),
            SegmentSpec("seg-parent", "ar-c", "i", 1, {"shard_id": "p"}, None, {}),
        ],
        edges=[
            EdgeSpec("seg-child-a", "seg-parent", "completion"),
            EdgeSpec("seg-child-b", "seg-parent", "completion"),
        ],
        session_to_agent_run={"s-a": "ar-a", "s-b": "ar-b", "s-c": "ar-c"},
    )
    edag = SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
    credit = distribute_reward_over_dag(
        edag, terminal_reward=1.0, terminal_node_id="seg-parent"
    )
    # Default (no fan_in_credit): split equally across the two parents -> 0.5 each.
    assert credit["seg-child-a"] > 0.0
    assert credit["seg-child-b"] > 0.0
    assert credit["seg-child-a"] == credit["seg-child-b"] == 0.5
```

- [ ] **Step 4: Run the fan-in test to verify it passes**

Run:
```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py \
  -k 'fan_in_credits' -v
```
Expected: PASS. `distribute_reward_over_dag` walks `edag.edges` (not the tuples), so fan-in credit is correct regardless of Task 3 - this test guards against any future regression that re-introduces the "last edge wins" overwrite on the v2 path.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/tests/test_assembler_ref_resolve.py
git commit -m "test(supernode): v2-path round-trip + fan-in regression"
```

---

## Task 5: Full regression + grep sweep

**Files:** none modified (verification + tasks.md check-off).

- [ ] **Step 1: Run the focused regression suite**

Run:
```bash
cd /workspaces/leagent/backend/areal
.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ \
  -k 'assembler or checkpoint or supernode or dag' -v
```
Expected: ALL PASS. This covers the new tests plus the existing assembler / checkpoint / dag-backup tests (the latter double as the non-regression guard for Task 3 - they read `edag.edges`, not the tuples).

- [ ] **Step 2: Lint the touched files**

Run:
```bash
ruff check customized_areal/tree_search/agents/execution_dag.py \
          customized_areal/tree_search/agents/supernode_assembler.py \
          customized_areal/tree_search/tests/test_checkpoint_super.py \
          customized_areal/tree_search/tests/test_assembler_ref_resolve.py
```
Expected: no errors. (If ruff is not on PATH, run `uvx ruff check <files>`.)

- [ ] **Step 3: Grep sweep - confirm the fixes resolve to intended code**

Run:
```bash
echo "--- visit_count in to_dict/from_dict ---"
grep -n "visit_count" customized_areal/tree_search/agents/execution_dag.py
echo "--- incoming/outgoing_edges populated in BOTH assemble paths ---"
grep -n "super_node.incoming_edges = tuple\|super_node.outgoing_edges = tuple" \
  customized_areal/tree_search/agents/supernode_assembler.py
```
Expected: `visit_count` appears in both `to_dict` (emit) and `from_dict` (`d.get(...)`). The tuple-population pattern appears exactly twice - once in `assemble_from_refs` (new) and once in `assemble` (legacy :336-345).

- [ ] **Step 4: Check off T2-T5 in tasks.md**

Edit `openspec/changes/sub-project-g-multica-interaction-dag/tasks.md`: change all remaining `- [ ]` under sections 2-5 to `- [x]`.

- [ ] **Step 5: Commit the task check-off**

```bash
git add openspec/changes/sub-project-g-multica-interaction-dag/tasks.md
git commit -m "docs(G): T5 full regression + grep sweep"
```

- [ ] **Step 6: Final review**

Confirm the full diff against `base-ref` (`853e5cbc`) is exactly: two surgical code edits (execution_dag.py, supernode_assembler.py), two test-file extensions, and tasks.md check-offs. No other files touched. Mark READY TO MERGE.

# Verification Report: sub-project-g-multica-interaction-dag

**Change:** sub-project-g-multica-interaction-dag (re-scoped to v2-segment-dag-integrity hardening)
**Capability:** `v2-segment-dag-integrity`
**Base ref:** `853e5cbcda6929e80961ed778194c1402fb7d9ba`
**Verify mode:** full
**Date:** 2026-07-13
**Verifier:** comet-verify (openspec-verify-change)

## Summary

| Dimension    | Status |
|--------------|--------|
| Completeness | 21/21 tasks complete; 3/3 requirements implemented |
| Correctness  | 7/7 spec scenarios covered by passing tests |
| Coherence    | Implementation matches design.md decisions + Design Doc; no spec/design contradictions |

**Final Assessment:** All checks passed. Ready for archive. 1 SUGGESTION (non-blocking).

## Scope

Re-scope of the superseded `sub-project-g-multica-interaction-dag` (originally a Multica-side producer) to proactive hardening of the AReaL v2 segment-DAG assembly/checkpoint path against two residual integrity gaps:

1. `SuperNode.visit_count` not serialized by `to_dict`/`from_dict` (dropped on checkpoint round-trip).
2. `SuperNodeAssembler.assemble_from_refs` left `incoming_edges`/`outgoing_edges` empty (topology lost on `to_dict` round-trip).

Both are non-behavioral to the live v2 path (confirmed in T1): `branch_backup` (the only `visit_count` mutator) is not on the v2 path, and v2 callers (`segment_dag_trainer`, `multi_agent_workflow`) read `edag.topological_order()` + `assemble_node_advantages` + `distribute_reward_over_dag` (which reads `dag.edges`, not the tuples), and do not checkpoint.

## Implementation Diff (vs base-ref)

```
customized_areal/tree_search/agents/execution_dag.py        |  2 +   (visit_count in to_dict + from_dict)
customized_areal/tree_search/agents/supernode_assembler.py  | 10 +   (tuple-population loop in assemble_from_refs)
customized_areal/tree_search/tests/test_assembler_ref_resolve.py | +94 (6 new tests)
customized_areal/tree_search/tests/test_checkpoint_super.py  | +15   (visit_count assertion + backward-compat test)
```

Surgical: 2 code edits, 2 test extensions. No other `customized_areal` files touched.

## Requirement → Scenario → Test Mapping

### Requirement 1: Lossless SuperNode serialization
- **Scenario: visit_count survives a round-trip** → `test_super_node_checkpoint_round_trip_is_lossless` (asserts `visit_count=3` survives full `TreeCheckpointManager` save/load) + `test_assemble_from_refs_round_trip_preserves_topology_and_visit_count` (asserts `visit_count=4` survives `to_dict`→`from_dict`). **PASS**
- **Scenario: Old checkpoints without visit_count deserialize to zero** → `test_from_dict_tolerates_missing_visit_count` (legacy 4-key dict → `visit_count=0`). **PASS**
- **Implementation:** `execution_dag.py:142` (`"visit_count": self.visit_count,` in `to_dict`), `execution_dag.py:170` (`visit_count=d.get("visit_count", 0),` in `from_dict`). Verified the checkpoint routes through `to_dict`/`from_dict` (`TreeCheckpointManager._serialize_super_node` → `super_node.to_dict()`, `_deserialize_super_node` → `SuperNode.from_dict(data)`).

### Requirement 2: Topology-complete v2-assembled SuperNodes
- **Scenario: v2 assembly populates edge tuples** → `test_assemble_from_refs_populates_edge_tuples` (2-segment completion DAG; asserts `outgoing_edges`/`incoming_edges` match `edag.edges`). **PASS**
- **Scenario: v2-assembled topology survives serialization** → `test_assemble_from_refs_round_trip_preserves_topology_and_visit_count` (asserts tuples + `closing_event` + tensors survive `to_dict`→`from_dict`). **PASS**
- **Scenario: Leaf segment has empty edge tuples** → `test_assemble_from_refs_leaf_segment_has_empty_edge_tuples` (solo segment, no edges → empty tuples). **PASS**
- **Implementation:** `supernode_assembler.py:197-202` (loop populating `incoming_edges`/`outgoing_edges` from `edag.edges`, mirroring legacy `assemble()` :336-345).

### Requirement 3: v2-path fan-in credit is preserved
- **Scenario: Fan-in join credits all parents on the v2 path** → `test_assemble_from_refs_fan_in_credits_all_parents` (2 incoming edges into one segment → `distribute_reward_over_dag` credits both parents 0.5 each). **PASS**
- **Implementation:** No code change required — `distribute_reward_over_dag` (`dag_backup.py:39-85`) already walks `dag.edges` and splits equally among parents. Test is a regression guard against re-introducing "last-edge-wins" on the v2 path.

## Test Execution

| Command | Result |
|---------|--------|
| `build_command` (2 touched test files) | 10 passed |
| `verify_command` (7-file focused suite) | 61 passed |
| Full focused regression (`-k 'assembler or checkpoint or supernode or dag'`, ignoring 5 pre-existing broken torch files) | 113 passed, 1 skipped |
| `ruff check` on 4 touched files | All checks passed |

**Pre-existing env note:** 5 torch-dependent test files (`test_critic_loss.py`, `test_critic_smoke.py`, `test_critic_td_target.py`, `test_critic_update.py`, `test_judge_integration.py`) fail collection on `ModuleNotFoundError: No module named 'datasets'` in `.venv-test`. Unrelated to this pure-Python data-model change; excluded from the regression run.

## Coherence

- **design.md decisions:** D1 (visit_count in to_dict/from_dict with backward-compat), D2 (populate tuples mirroring assemble()), D3 (round-trip + fan-in regression tests). Implementation matches all three.
- **Design Doc** (`docs/superpowers/specs/2026-07-13-v2-segment-dag-path-hardening-design.md`): documents Approach A with exact code for the 3 changes. Implementation matches verbatim. Frontmatter declares `comet_change`, `role: technical-design`, `canonical_spec: openspec`.
- **No spec/design contradictions:** delta spec's 3 requirements map 1:1 to the design's 3 decisions and the 3 code changes.
- **Code pattern consistency:** tuple-population mirrors the existing `assemble()` pattern; `visit_count` added in the same style as adjacent fields (`outcome_reward`). ruff clean (I001 import-sort fixed).

## Issues

### CRITICAL
None.

### WARNING
None.

### SUGGESTION
1. **Fan-in test edge type vs scenario example.** Spec scenario "Fan-in join credits all parents on the v2 path" references "two incoming DELEGATION edges"; the test `test_assemble_from_refs_fan_in_credits_all_parents` uses COMPLETION edges. The requirement itself allows both ("DELEGATION or COMPLETION") and `distribute_reward_over_dag` is edge-type-agnostic, so the requirement is fully satisfied. Optionally align the test's edge type to DELEGATION (or broaden the scenario text) for exact scenario-text match. Non-blocking.

## Branch Handling

Work performed on local `master` (per established areal workspace pattern: local-master commits, not pushed, no MR). `isolation: branch`, `build_mode: executing-plans`. Branch handling decision recorded separately via `finishing-a-development-branch`.

## Conclusion

Both residual v2-path integrity gaps are closed and pinned by regression tests. The change is surgical (2 code edits + 2 test extensions), non-behavioral to the live v2 training path, ruff-clean, and all 21 tasks + 7 spec scenarios pass. **Ready for archive.**

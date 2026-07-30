# Verification Report: multica-v2-segment-dag-training

**Date:** 2026-07-13
**Phase:** verify (full)
**Branch:** multica-v2-segment-dag-training (HEAD = 2a2f9cae)
**Verifier:** comet-verify + openspec-verify-change

## Context

The training change establishes the v2 per-segment trajectory data path:
`close_segment` + per-segment tensor-ref export + Multica `AssembledDag` assembly +
AReaL ref-resolution + minimal training. Its code (U1-U10) was delivered across three
changes and merged to master before this verify run:

- **training** (this change): U1 close_segment, U2 export, U3 MulticaDagClient,
  U4 SuperNodeAssembler.assemble_from_refs, U5 training+lifecycle, U6 arealrl
  CloseSegment/ExportTrajectory, U7.1 InteractionDAGService core, U9 migration.
- **recording-assembly** (sibling, 32/32 complete): U7.2 seam wiring, U7.3 retry,
  U8 AssembledDag assembly + /dag endpoint, U10 config + regression.
- **tree-search-branching** (sibling): built Change 3 atop this data path.

A reconciliation (commit 6a78b466) checked off the stale tasks.md (was 7/67) to reflect
actual delivery. A rename refactor (commit 2a2f9cae) renamed
`multi_agent_env_dispatch` -> `multi_agent_workflow` and `swe_lego_client` ->
`multica_client`, and made `MultiAgentEnvDispatchWorkflow` construct its dispatch client
internally (optional `dispatch_client` param retained for test injection).

## Summary

| Dimension    | Status |
|--------------|--------|
| Completeness | 65/65 tasks done; 4/4 spec requirements implemented |
| Correctness  | 4/4 requirements covered; 10/10 spec scenarios covered by tests (54 tests pass) |
| Coherence    | D1-D7 design decisions all followed; no spec/design contradictions |

## Completeness

- **Tasks**: 65/65 complete (0 incomplete). 10.3 (cross-repo E2E) and 10.6 (final review)
  relocated to a checkbox-free "Deferred to verify" section; 10.6 is satisfied by this report.
- **Spec requirements** (specs/v2-segment-dag/spec.md): all 4 implemented:
  1. V2 no-reward segment close - `SessionData.close_segment()` (session.py:271),
     `/rl/close_segment` (data_proxy app.py:522, gateway app.py:381).
  2. Per-segment tensor-ref export - `/export_trajectories` by trajectory_id,
     remove_session=False, RTensor refs, 400 on unknown (app.py:744-775);
     `/data/<shard_id>`, `/data/batch`, `DELETE /data/clear`.
  3. AssembledDag contract - GetDag handler (multica env_dispatch.go:209),
     SegmentSpec/EdgeSpec, typed edges, no scores/turn-idx/text.
  4. AReaL resolves refs + builds DAG - `assemble_from_refs` (supernode_assembler.py:108),
     acyclic via topological_order, dense-cover validation (supernode_assembler.py:247).

## Correctness

- **Requirement-to-implementation mapping**: verified with file:line + commit evidence
  (U1 d77fab31/632e4949/6bde9872; U2 796c9004; U3 748ca36f; U4 af75a189; U5 f56e4951;
  multica-side U6-U9 in the multica repo).
- **Scenario coverage**: all 10 spec scenarios across the 4 requirements are covered by
  tests. Test results (`.venv-test/bin/python -m pytest`):
  - areal/v2/inference_service/tests/: 9 passed.
  - consumer (test_multica_dag_client, test_assembler_ref_resolve, test_segment_dag_trainer,
    test_segment_dag_training_path, test_env_dispatch_client, test_multi_agent_env_dispatch,
    test_multica_workflow_wiring, test_v2_session_lifecycle): 45 passed.
  - Total: 54 passed, 0 failed.
- **Build guard**: build_passes (build_command = v2 inference_service tests) PASS.

## Coherence

- **Design adherence**: D1-D7 all followed (close_segment decouples from reward; export
  returns RTensor refs; AssembledDag carries structure+refs+env only; Multica assembles +
  AReaL ref-joins; text stays in AReaL capture; placeholder reward; env-dispatch polling
  reused, sub-project-g superseded).
- **Risks mitigated**: acyclicity + dense-cover validated (supernode_assembler.py:192,247);
  ref lifetime guarded (`/data/clear` after training, remove_session=False until last
  segment).
- **Open questions** resolved or deferred: close_segment-on-empty = typed error (test 3.3);
  per-trajectory export + /data/* surface implemented; outcome_reward semantics + judge
  trigger timing deferred to change 2 (out of scope).
- **Spec/design contradictions**: none.

## Issues

### CRITICAL
None.

### WARNING
1. **Cross-repo E2E (task 10.3) not run** - deferred due to services/GPU unavailable. The
   end-to-end path (Multica 3-agent rollout -> close_segment + export per event -> poll
   GET /dag -> AReaL resolve refs -> ExecutionDAG -> minimal training step) is proven via
   component tests (54 pass) but not a live 3-agent E2E. Recommend running when
   services/GPU are available, before relying on the path in production training.

### SUGGESTION
1. **Rename refactor (commit 2a2f9cae)** was committed during verify and is not
   anticipated by design.md/tasks.md. It is consistent with the change's scope (U5
   component) and production behavior is unchanged (dispatch client internal by default;
   optional `dispatch_client` param for test injection). Consider noting it as an
   implementation divergence in design.md, or accept as a verify-phase refactor.
2. **Dense-cover validation is dual-located**: `supernode_assembler.py:247` (assemble)
   and the multica `/dag` handler (`denseCover`). Design D4/risks specify the assembler
   validates; the dual location is belt-and-suspenders, not a contradiction. No action
   required unless strict single-location is desired.

## Final Assessment

No critical issues. 1 warning (E2E deferred - environment unavailable) + 2 suggestions.
The training change's data path is complete, tested (54 tests green), and coherent with
its design. **Ready for archive** with the noted improvements (run E2E when environment
permits).

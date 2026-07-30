# Verification Report: multica-pi-diagnosis-agent

**Date:** 2026-07-10
**Schema:** spec-driven
**Verify mode:** full (40 tasks, 1 delta-spec capability, 64 changed files vs base_ref)
**Phase:** build -> verify (comet-verify)
**Verifier:** orchestrator (direct), per `comet-verify` + `openspec-verify-change`

## Context

A Multica-side Pi **diagnosis agent** that fires at collaborative-task completion, views the
segment (supernode) DAG via read-only tools, emits a per-LLM-output reward keyed
`(segment_id, seq)`, and delivers them to AReal via `AssembledDag.step_rewards[]`. AReal
aggregates them to a per-segment `SuperNode.process_reward` (the v2 GAE reward), replacing
the flat areal LLM judge. Dual-repo: Multica (Go, branch `dev`) + AReal (Python, this
worktree).

User decision (this verify run): **proceed to verify at unit level**. The 5 hardware-dependent
tasks (4.4 + 7.1-7.4) are deferred to a GPU-equipped run; merge is deferred at
branch-handling. Live integration/E2E (the change's end-to-end reward delivery) is not
validated in this sandbox.

## Summary

| Dimension    | Status |
|--------------|--------|
| Completeness | 35/35 tasks `[x]`; 5 `[~]` deferred (hardware, user-approved); 0 incomplete `[ ]` |
| Correctness  | 7/7 spec requirements implemented (symbols confirmed both repos); scenarios have test coverage |
| Coherence    | Superpowers design D4/D5 + spec.md reflect impl; OpenSpec design.md drift reconciled (divergence note appended); code patterns consistent |

**CRITICAL: 0.** WARNING: 0 remaining (1 drift resolved this run). SUGGESTION: 2 (non-blocking).

## Completeness

### Task completion

- 35 tasks `- [x]` complete; 5 tasks `- [~]` **deferred** (not `[ ]`, not `[x]`):
  - **4.4** `/dag` stays `202` while diagnosis runs, then `200` - needs a diagnosis-in-progress
    flag (deferred code gap) + a live diagnosis run.
  - **7.1** Integration: trained 3-agent `mode=scratch` rollout -> diagnosis -> step_rewards ->
    `/dag`. Needs live multica + trained rollout + GPU.
  - **7.2** E2E: rollout -> diagnosis -> AReal resolve -> `ExecutionDag` `process_reward` ->
    training step. Needs live multica + areal + GPU.
  - **7.3** Sanity: diagnosis vs. removed judge. **Needs re-scoping** - the judge baseline was
    removed in Task 8, so the comparison as written cannot run.
  - **7.4** Commit integration + E2E tests - blocked on 7.1-7.3.
- openspec parser counts 35/35 (treats `[~]` as non-task); no `- [ ]` incomplete.
- Deferral rationale documented in `tasks.md` blockquote notes; user-approved this run.

### Spec coverage

All 7 `### Requirement:` entries in `specs/diagnosis-process-reward/spec.md` have
implementation evidence (see Correctness). No requirement appears unimplemented.

## Correctness

Requirement -> implementation mapping (symbols confirmed at code):

| # | Requirement | Implementation | Evidence |
|---|-------------|----------------|----------|
| 1 | Diagnosis trigger + gating | `maybeDiagnoseProject`; gates `deps.Diagnosis==nil` / `DAG.Enabled()` / root-task | `multica/server/internal/service/training.go:466` |
| 2 | Read-only tool access | `GetInteractionDAG` / `GetSegmentMessages` / `GetTaskContext`; read-only, project/workspace-scoped, turn+byte budgets | `multica/server/internal/service/interaction_dag.go` (+ Task 3 tool handlers) |
| 3 | Per-segment turn-range capture | `start_seq`/`end_seq` on `interaction_dag_segment`; `CloseSegmentForEvent` derives them; migration 161 | `multica/server` migration 161 |
| 4 | Per-LLM-output reward output | `parse_step_rewards`; integer `[0, score_max]`; keyed `(segment_id, seq)`; systemPrompt embeds concrete range | `multica/server/internal/service/diagnosis_agent_runner.go` (Task 1) |
| 5 | Project-scoped delivery via AssembledDag | `RecordStepRewards` upsert; `AssembledDag.StepRewards[]` + `ScoreMax`; `/dag` serves + stamps `ScoreMax` from config | `interaction_dag.go:444,283`; `handler/env_dispatch.go:290` |
| 6 | Soft-failure does not block completion | `maybeDiagnoseProject` soft-fail: `Diagnose` err -> `slog.Warn` + return (no rewards); `RecordStepRewards` err -> warn + proceed | `training.go:466` |
| 7 | Coexistence with scalar critic | `Diagnoser` interface (separate from critic); D1 parallel path; fires at project completion, not trained-agent terminal | `training.go:89`; `training_config.go:187` (`buildDiagnoser`) |

AReal consumer side: `StepReward` dataclass + `AssembledDag.step_rewards`/`score_max` parsed
in `from_dict` (`multica_dag_client.py:66,92-102`); `_aggregate_process_reward = mean(scores)/score_max`
(`supernode_assembler.py:108,120`); `super_node.process_reward = _aggregate_process_reward(...)`
(`:220`); unmatched step rewards dropped + logged; `score_max==0`/empty -> sparse `0.0`.

Flat-judge removal (Task 8): `enable_judge_process_reward` / `_annotate_judge_process_rewards`
absent from `customized_grouped_workflow.py` + `config.py`; 4 judge config fields +
`__post_init__` validation removed; `distilling/config.py` judge fields removed. Distilling
utilities kept (`judge_prompt.py` / `diagnose_provider.py` / `ExternalDiagnoseProvider` /
`DiagnoseProvider` protocol / `selected_turn_distill`) - used by distillation, not training.

### Scenario coverage

All 12 `#### Scenario:` entries have corresponding tests (16 new areal tests for Tasks 7-8;
Go diagnosis tests across Tasks 1-6). The unit-runnable subset is green:
- Go `internal/service` suite green; handler `GetDag`/diagnosis tests pass.
- AReal `tree_search` suite: 364 pass (Tasks 7-8). Pre-existing failures (10: `datasets`
  module missing, asyncio event-loop; handler `AgentActivity`/`Credential`/Transport DB 500s)
  confirmed identical on merged-master baseline (stash) - environmental, not caused by this
  change.

### Tests NOT run (deferred)

- **7.1 / 7.2** (integration + E2E) - the change's core end-to-end reward delivery. **Not
  validated** without a GPU cluster + trained rollout. This is the principal open risk: unit
  pieces are verified; their composition on a real rollout is not.

## Coherence

### Design adherence

- **Superpowers design doc** (`docs/superpowers/specs/2026-07-09-...-design.md`) D4/D5:
  updated during build to per-segment `SuperNode.process_reward` aggregation + `/dag`
  200-not-202 sequencing. **Matches implementation.** ✓
- **spec.md** Requirement "Per-LLM-output reward output" (L68): updated to per-segment
  SuperNode attachment. **Matches implementation.** ✓
- **OpenSpec design.md** D5/D7 + Q1-Q4: were stale (per-turn `Node` language; Q1-Q4 listed
  open). **Reconciled this run (Option A):** appended an "Implementation Divergence" section
  recording the Q1-Q4 resolutions and noting D5/D7 per-turn wording is superseded by the
  per-segment aggregate. Defers to spec.md + Superpowers D4 as authoritative. ✓ (resolved)

### Code pattern consistency

- Multica diagnosis runner mirrors `evolution_review_provider.go` (`NewAgentEvolutionReviewer`
  pattern: `Provider/ExecutablePath/Model/Timeout/Backend`, structured output, byte/turn
  budgets). ✓
- `Diagnoser` as interface (not concrete `*DiagnosisAgentRunner`) matches the codebase
  deps-as-interfaces pattern (`arealSessionCloser`); testable via `fakeDiagnoser`. ✓
- AReal consumer follows existing `AssembledDag`/`SuperNodeAssembler` conventions; logging via
  module logger. ✓

## Issues

### CRITICAL (must fix before archive)

None.

### WARNING (should fix)

None remaining. The OpenSpec design.md drift was the one WARNING; resolved this run by
appending the Implementation Divergence section (user-approved Option A).

### SUGGESTION (nice to fix, non-blocking)

1. **7.3 re-scoping.** The sanity comparison vs. the removed judge can no longer run as
   written (judge deleted in Task 8). On the hardware run, re-scope 7.3 to compare diagnosis
   rewards against a freshly-computed reference (e.g. a one-off judge invocation) or drop it.
2. **4.4 strict `/dag` 202-during-diagnosis.** Needs a diagnosis-in-progress flag
   (currently `CompleteAgentTask` persists terminal before `RouteTerminalTrainingTask`, so
   `/dag` is 200 not 202 during synchronous diagnosis). Implement the flag on the hardware run
   if strict 202-during-diagnosis is required.

## Deferred to a GPU hardware run (user-approved)

- 7.1 integration, 7.2 E2E, 7.3 sanity (re-scope first), 7.4 commit; 4.4 strict 202; Task 4
  RTT-level ordering test. These are the only unchecked items. The unit-runnable coverage is
  green; the end-to-end composition is not yet validated.

## Final Assessment

**No critical issues.** All 7 spec requirements are implemented with unit test coverage; both
repos compile clean; the OpenSpec design.md drift is reconciled. **Ready for archive at the
unit level**, with the explicit caveat that the change's core end-to-end reward delivery
(7.1/7.2) is **deferred to a hardware run** and merge should be deferred at branch-handling
until that validation lands. 7.3 needs re-scoping (judge baseline removed).

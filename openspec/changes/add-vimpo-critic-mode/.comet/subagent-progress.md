# Comet SDD Progress - add-vimpo-critic-mode

- Worktree: `/workspaces/leagent/backend/areal/.claude/worktrees/add-vimpo-critic-mode`
  | branch `worktree-add-vimpo-critic-mode`
- build_mode: subagent-driven-development | tdd_mode: direct | review_mode: standard
- Env: `/workspaces/leagent/backend/areal/.venv-test` (pytest/ruff/pre-commit binaries;
  NO `uv run` - wheels/ absent). customized_areal imports from worktree cwd.
- Plan -> OpenSpec: T1->1.1-1.4 | T2->4.1-4.3 | T3->4.4,4.5 | T4->3.1-3.4 | T5->2.1-2.4
  | T6->5.1,5.2,5.5 | T7->5.3,5.4,6.1 | T8->6.2-6.4 | T9->6.5,7.1

## Minor findings deferred to final review

- T1: dead `vimpo_logger` at `core/advantage.py:713` (unused). Remove or wire in.
- T2: incidental ruff-format of pre-existing `fresh_vids` in `_finalize_episode`
  (`customized_grouped_workflow.py:1884-1888`). Behavior-neutral.
- T3: `vimpo_turn_index` validation reads `[row, 0]` only (convention, not a bug).
- T4: 5 Minor from per-task review (deferred - recorded in checkoff commit 89a609b1 +
  task-4-report.md).
- T5: 3 Minor from re-review (deferred): (a) GPU test flat-vs-packed parity only
  directly asserts sampled_logp + candidate_ids (candidate_logp/retained_mass
  transitively covered via brute-force ref) - add 2 assert_close lines; (b)
  `_tree_seq_predict_mask` doesn't validate seq_out_len vs slice length (defensive
  assert); (c) GPU test depends on real Qwen3-0.6B model path
  `/storage/openpsi/models/Qwen__Qwen3-0.6B/` (infra; confirm availability on GPU host).
- T6: 5 Minor from review (deferred): (1) `dist.get_rank()` guard inconsistent with
  existing `_prepare_mb_list` (fsdp_engine.py:188 vs :1865); (2) `episode_count == 0`
  check dead code (vimpo.py:1179-1183); (3) `_validate_vimpo_batch` doesn't shape-check
  vimpo_turn_index/vimpo_expected_turn_count/input_ids; (4) `_validate_vimpo_batch`
  doesn't assert attention_mask is 2D; (5) `test_train_vimpo_batch_scales_per_formula`
  doesn't replicate full \_prepare_vimpo_mb_list pipeline (single-mb no-padding only).
  Re-review added 1 Minor: (6) `test_train_vimpo_batch_rejects_vl_moe_models` only
  asserts `zero_grad_calls==0`, missing `forward_backward_calls==0`/`step_calls==0` its
  siblings assert.

## Pre-existing test failures (NOT VIMPO regressions; verify-phase will document)

- `test_critic_smoke::test_end_to_end_pipeline` + ~9 test_critic/test_assembler: Python
  3.12/uvloop `asyncio.get_event_loop()` deprecation. Present at base 19c4aaf7.

## Current task

- Stage: FINAL-FIX dispatched (bg, round 1/1) 2026-07-22. Final lightweight review:
  NOT READY TO MERGE - 3 CRITICAL (C1 zero-padded vimpo_* metadata trips row-constancy
  checks on unequal-length batches; C2 per-query vimpo_episode_index collides across
  queries -> silent cross-query advantage chaining + guaranteed multi-query crash; C3
  actor silently overwrites spec'd query-local centered terminal target with
  batch-global row-weighted mean) + 2 IMPORTANT (I1 missing self.train() in
  _vimpo_update; I2 stock-SGLang /get_model_info GET-vs-POST + omitted-field identity
  gap). Fix agent must also extend the CPU smoke test to multi-query + unequal-length
  batches (regression net for C1-C3). Review handoff: .superpowers/sdd/final-review-1.md
  | Fix report: .superpowers/sdd/task-10-report.md. Deferred Minor list triaged by
  final review as follow-up-acceptable (T1-T7 items); concern (b) unexecuted 2-GPU test
  upgraded: smoke-test extension now mandatory, done in this fix round. If re-review
  fails -> BLOCKED, pause, hand to user.

## Completed tasks

- T1 (config + advantage math): complete. Impl 19c4aaf7..244e2e8a + checkoff 97c1d962.
  Review approved, 1 Minor. OpenSpec 1.1-1.4.
- T2 (workflow metadata + centered targets): complete. Impl 97c1d962..8c9dcc19 +
  checkoff c5d19d32. Review approved, 1 Minor. OpenSpec 4.1-4.3.
- T3 (episode-atomic batching): complete. Impl c5d19d32..b7af2d70 + checkoff 4810d861.
  Review approved, 1 Minor. OpenSpec 4.4-4.5 (section 4 done).
- T4 (frozen SGLang scorer): complete. Impl 4810d861..b1b4136e + checkoff 89a609b1.
  Review approved, 5 Minor (deferred). OpenSpec 3.1-3.4.
- T5 (FSDP actor candidate stats): complete. Impl 89a609b1..0542ac30 + fix
  0542ac30..8d353c1a + checkoff 575faf51. Re-review Approved (0 Critical/Important, 3
  Minor deferred). OpenSpec 2.1-2.4.
- T6 (combined VIMPO loss + two-denominator backward): complete. Impl 575faf51..0c7cad7b
  \+ fix 0c7cad7b..c968f0b2 + checkoff 85f2adb6. Re-review Approved (0
  Critical/Important, 5+1 Minor deferred). 20 passed/1 pre-existing warning, ruff green.
  OpenSpec 5.1, 5.2, 5.5.
- T7 (dedicated VIMPO actor + trainer wiring): complete. Impl 85f2adb6..d72f39d1 + fix
  d72f39d1..20587220 + checkoff 5e96c384. Re-review Approved (1 IMPORTANT fixed: Muon
  attrs; revision-wildcard scorer fix; Minors #2/#3/#6 fixed, #4/#5 deferred). 42
  passed/1 GPU skip, ruff green. OpenSpec 5.3, 5.4, 6.1.
- T8 (metrics, CPU smoke, docs): complete. Impl 5e96c384..3e8ef242 + fix
  3e8ef242..f02c0483 + checkoff fbb0f522. Re-review Approved (1 IMPORTANT fixed: loss
  scalars dp_size scaling; F2-F5 Minors fixed incl. scorer retry_count, quantile guard).
  100 passed/2 hardware skips, ruff green. OpenSpec 6.2, 6.3, 6.4.

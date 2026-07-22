# Comet SDD Progress - add-vimpo-critic-mode

- Worktree: `/workspaces/leagent/backend/areal/.claude/worktrees/add-vimpo-critic-mode` | branch `worktree-add-vimpo-critic-mode`
- build_mode: subagent-driven-development | tdd_mode: direct | review_mode: standard
- Env: `/workspaces/leagent/backend/areal/.venv-test` (pytest/ruff/pre-commit binaries; NO `uv run` - wheels/ absent). customized_areal imports from worktree cwd.
- Plan -> OpenSpec: T1->1.1-1.4 | T2->4.1-4.3 | T3->4.4,4.5 | T4->3.1-3.4 | T5->2.1-2.4 | T6->5.1,5.2,5.5 | T7->5.3,5.4,6.1 | T8->6.2-6.4 | T9->6.5,7.1

## Minor findings deferred to final review
- T1: dead `vimpo_logger` at `core/advantage.py:713` (unused). Remove or wire in.
- T2: incidental ruff-format of pre-existing `fresh_vids` in `_finalize_episode` (`customized_grouped_workflow.py:1884-1888`). Behavior-neutral.
- T3: `vimpo_turn_index` validation reads `[row, 0]` only (convention, not a bug).
- T4: 5 Minor from per-task review (deferred - recorded in checkoff commit 89a609b1 + task-4-report.md).
- T5: 3 Minor from re-review (deferred): (a) GPU test flat-vs-packed parity only directly asserts sampled_logp + candidate_ids (candidate_logp/retained_mass transitively covered via brute-force ref) - add 2 assert_close lines; (b) `_tree_seq_predict_mask` doesn't validate seq_out_len vs slice length (defensive assert); (c) GPU test depends on real Qwen3-0.6B model path `/storage/openpsi/models/Qwen__Qwen3-0.6B/` (infra; confirm availability on GPU host).
- T6: 5 Minor from review (deferred): (1) `dist.get_rank()` guard inconsistent with existing `_prepare_mb_list` (fsdp_engine.py:188 vs :1865); (2) `episode_count == 0` check dead code (vimpo.py:1179-1183); (3) `_validate_vimpo_batch` doesn't shape-check vimpo_turn_index/vimpo_expected_turn_count/input_ids; (4) `_validate_vimpo_batch` doesn't assert attention_mask is 2D; (5) `test_train_vimpo_batch_scales_per_formula` doesn't replicate full _prepare_vimpo_mb_list pipeline (single-mb no-padding only). Re-review added 1 Minor: (6) `test_train_vimpo_batch_rejects_vl_moe_models` only asserts `zero_grad_calls==0`, missing `forward_backward_calls==0`/`step_calls==0` its siblings assert.

## Pre-existing test failures (NOT VIMPO regressions; verify-phase will document)
- `test_critic_smoke::test_end_to_end_pipeline` + ~9 test_critic/test_assembler: Python 3.12/uvloop `asyncio.get_event_loop()` deprecation. Present at base 19c4aaf7.

## Current task
- Plan task: Task 7 - Dedicated VIMPO FSDP Actor and Trainer Wiring
- OpenSpec items: 5.3, 5.4, 6.1
- Stage: RE-REVIEW dispatched (bg, round 1/1) 2026-07-22. Fix agent DONE_WITH_CONCERNS, fix commit 20587220 (5 files incl. scorer-side revision-wildcard fix in vimpo_reference.py; tests 42 passed/1 GPU skip/0 failed, ruff clean; muon regression test added). Fix concern recorded for final review: stock SGLang /get_model_info also omits vocab_size/tokenizer_vocab_size/special-token IDs -> same identity-mismatch class; broader design decision (shim vs relaxed validation), deferred. Review package: .superpowers/sdd/review-d72f39d1..20587220.diff.
- Impl BASE: 85f2adb6 (T6 checkoff HEAD; code base = c968f0b2 fix HEAD)
- Brief: `.superpowers/sdd/task-7-brief.md` | Report: `.superpowers/sdd/task-7-report.md`
- Modifies `training/actor.py` (`VIMPOFSDPPPOActor`: compute_advantages, ppo_update, destroy), `training/trainer.py` (`_create_train_engine` VIMPO branch: backend==fsdp gate, copy vimpo_* config, select dedicated actor, no distill/critic patches), `engine/__init__.py` + `__init__.py` (lazy exports); creates `tests/test_vimpo_actor.py` + `tests/test_vimpo_trainer.py`.
- Consumes T1-T6: `MultiCandidateFSDPEngine`, `PPOActor`/`PPOActorConfig`, `VIMPOReferenceScorer`/`SGLangVIMPOReferenceScorer`/`ReferenceScoreRequest` (T4), `VIMPOAdvantageComputer` (T1), `compute_vimpo_candidate_stats` (T5), `split_episode_atomic_batches` (T3), `train_vimpo_batch` (T6), `AdvantageMode`, `CustomizedPPOTrainer`.
- Risk: integration of all prior contracts; must preserve existing distillation/Muon/clip-cov/critic patch behavior; VIMPO uses kl_ctl=0 + ref=None. Integration task -> sonnet implementer + standard review.
- NOTE: brief uses `uv run pytest` in cmd text - use `.venv-test/bin/pytest` (NO uv run).

## Completed tasks
- T1 (config + advantage math): complete. Impl 19c4aaf7..244e2e8a + checkoff 97c1d962. Review approved, 1 Minor. OpenSpec 1.1-1.4.
- T2 (workflow metadata + centered targets): complete. Impl 97c1d962..8c9dcc19 + checkoff c5d19d32. Review approved, 1 Minor. OpenSpec 4.1-4.3.
- T3 (episode-atomic batching): complete. Impl c5d19d32..b7af2d70 + checkoff 4810d861. Review approved, 1 Minor. OpenSpec 4.4-4.5 (section 4 done).
- T4 (frozen SGLang scorer): complete. Impl 4810d861..b1b4136e + checkoff 89a609b1. Review approved, 5 Minor (deferred). OpenSpec 3.1-3.4.
- T5 (FSDP actor candidate stats): complete. Impl 89a609b1..0542ac30 + fix 0542ac30..8d353c1a + checkoff 575faf51. Re-review Approved (0 Critical/Important, 3 Minor deferred). OpenSpec 2.1-2.4.
- T6 (combined VIMPO loss + two-denominator backward): complete. Impl 575faf51..0c7cad7b + fix 0c7cad7b..c968f0b2 + checkoff 85f2adb6. Re-review Approved (0 Critical/Important, 5+1 Minor deferred). 20 passed/1 pre-existing warning, ruff green. OpenSpec 5.1, 5.2, 5.5.

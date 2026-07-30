# Hybrid MC/GAE Advantage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-turn advantage estimator that blends a leave-one-out Monte-Carlo
value (from MCTS tree backups) with the learned critic via an MSE-optimal per-node
weight, feeding the blended value into the existing GAE recursion; and add an optional
entropy x TD-error branch-selection score. All new behaviour is config-gated and
defaults to current behaviour.

**Architecture:** Value-level blend `v_hat = w*v_mc + (1-w)*v_theta` then standard GAE.
`w` derives from node visit count (MC variance) and a running critic-bias estimate. Tree
store gains second-moment tracking and accessors. Branch selection gains an optional
multiplicative entropy/TD score.

**Tech Stack:** Python 3.12+, PyTorch, no new dependencies.

______________________________________________________________________

### Task 1: Tree store value statistics

**Files:**

- Modify: `customized_areal/tree_search/core/tree_store.py`

- Create: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] Add failing tests: after several `_backup` calls, `get_visit_count` returns the
  count, `get_return_variance` matches population variance, `get_mc_value_loo(node, r)`
  equals `(total - r)/(K-1)` and falls back to `get_q_value` when `K < 2`.

- [ ] Extend `_backup` to accumulate `_total_squared_values[node_id]`.

- [ ] Add `get_visit_count`, `get_return_variance`, `get_mc_value_loo`.

- [ ] Ensure new dict is cleared in `clear()` and survives `insert_batch`.

- [ ] Run: `uv run pytest tests/customized_areal/test_hybrid_advantage.py -v -k "store"`

______________________________________________________________________

### Task 2: Running critic-bias helper

**Files:**

- Modify: `customized_areal/tree_search/core/advantage.py`

- Modify: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] Add failing tests for an EMA bias tracker: `update(td_errors)` produces an EMA of
  mean squared TD-error; respects `bias_init` and `ema` decay.

- [ ] Implement a small `CriticBiasTracker(bias_init, ema)` with `b2` property and
  `update(values, returns)`.

- [ ] Run: `uv run pytest tests/customized_areal/test_hybrid_advantage.py -v -k "bias"`

______________________________________________________________________

### Task 3: HybridAdvantageComputer — weight + blend

**Files:**

- Modify: `customized_areal/tree_search/core/advantage.py`

- Modify: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] Add failing tests:

  - `w_t == 0` when `K < min_visits` -> output equals `GAEAdvantageComputer`
    byte-for-byte on the same nodes.
  - `w_t` matches `(sigma_theta2 + b2)/(var_mc + sigma_theta2 + b2)`.
  - large `b2` pushes `v_hat` toward `v_mc`.
  - leave-one-out excludes the current episode's return.
  - episodic boundary `v_hat(s_{T+1}) = 0`; sparse reward path unchanged.

- [ ] Implement `HybridAdvantageComputer` mirroring `GAEAdvantageComputer`'s grouping,
  ordering, reward construction (including `judge_beta` path), and write contract
  (`advantages`, `returns`, `set_normalized_advantage/return`), but running GAE on
  `v_hat`.

- [ ] Run:
  `uv run pytest tests/customized_areal/test_hybrid_advantage.py -v -k "hybrid"`

______________________________________________________________________

### Task 4: Config fields + validation

**Files:**

- Modify: `customized_areal/tree_search/config.py`

- Modify: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] Add failing tests for new defaults and validation errors.

- [ ] Add `AdvantageMode.HYBRID` and the `hybrid_*` / `branch_score_*` fields from the
  spec with validation.

- [ ] Run:
  `uv run pytest tests/customized_areal/test_hybrid_advantage.py -v -k "config"`

______________________________________________________________________

### Task 5: Wire HYBRID mode into the trainer/workflow

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py` (computer
  construction site)

- Modify: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] Add failing test: with `advantage_mode == HYBRID`, the workflow/trainer constructs
  `HybridAdvantageComputer` and `_compute_advantages_for_batch` passes through
  precomputed advantages (same skip path as TREE mode).

- [ ] Select the computer by `advantage_mode` (GAE / TREE / HYBRID).

- [ ] Run:
  `uv run pytest tests/customized_areal/test_hybrid_advantage.py -v -k "wiring"`

______________________________________________________________________

### Task 6: Entropy x TD-error branch scoring

**Files:**

- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`

- Modify: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] Add failing tests:

  - `branch_score_mode="entropy"` reproduces current `max(_max_entropy)` ranking.
  - `"entropy_td"` ranks high-entropy/high-TD above high-entropy/low-TD.
  - empty `entropy_stats` -> entropy term 0; never-branched node -> neutral TD term.

- [ ] Add a `_branch_score(node, mode, alpha, beta, tree_store)` and use it as the key
  in `select_branch_candidate`; thread the config through.

- [ ] Run:
  `uv run pytest tests/customized_areal/test_hybrid_advantage.py -v -k "branch"`

______________________________________________________________________

### Task 7: End-to-end + regression validation

**Files:**

- Modify: `tests/customized_areal/test_hybrid_advantage.py`

- [ ] E2E: a small multi-turn tree with branched/unbranched nodes produces finite
  advantages/returns; HYBRID == GAE when no node meets `min_visits`.

- [ ] Run existing suites most likely to regress:

```bash
uv run pytest tests/test_tree_search/test_advantage.py -v
uv run pytest tests/test_tree_search/test_tree_search_grouped_workflow.py -v
uv run pytest tests/test_tree_search/test_batch_consistency.py -v

- [ ] Run hooks: pre-commit run --all-files

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Notes

- New behaviour is fully gated; default config is unchanged from today.
- Blending is at the value level (not advantage tensors) for a single well-defined

  estimator.

- Leave-one-out MC keeps the baseline gradient-unbiased.
- branch_score_mode="entropy_td" is the two-stage progressive-widening signal:

  entropy proposes, TD refines.
```

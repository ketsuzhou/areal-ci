# Dynamic Group Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dynamic group size mode to `TreeSearchGroupedRolloutWorkflow` that
allocates more episodes to uncertain queries, fewer to converged ones, and cleanly
integrates with TREE-mode precomputed advantages.

**Architecture:** Dynamic sampling stays fully per-query. The workflow computes a
step-adjusted Bayesian uncertainty score and keeps sampling until the score falls below
an absolute threshold or reaches `max_group_size`. In `AdvantageMode.TREE`, tree
advantages computed inside the workflow are passed through to training without
trainer-side recomputation.

**Tech Stack:** Python 3.12+, PyTorch, no new dependencies.

---

### Task 1: Add config fields to `TreeBackupConfig`

**Files:**
- Modify: `customized_areal/tree_search/config.py`
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add failing tests asserting these defaults:
  - `dynamic_group_size is False`
  - `initial_group_size == 4`
  - `max_group_size == 64`
  - `uncertainty_threshold == 0.05`
  - `reward_type == "binary"`

- [ ] Add config fields:

```python
dynamic_group_size: bool = False
initial_group_size: int = 4
max_group_size: int = 64
uncertainty_threshold: float = 0.05
reward_type: str = "binary"
```

- [ ] Add validation:
  - `initial_group_size >= 1`
  - `max_group_size >= initial_group_size`
  - `uncertainty_threshold >= 0`
  - `reward_type in {"binary", "continuous"}`

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v -k "config"
```

---

### Task 2: Implement uncertainty helpers

**Files:**
- Create: `customized_areal/tree_search/core/uncertainty.py`
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add failing tests for:
  - binary uncertainty
  - continuous uncertainty with `n=1`
  - continuous uncertainty with `n>=2`
  - more steps increasing uncertainty
  - zero-episode behavior
  - discard behavior for 0, 1, 2+ identical, and mixed rewards

- [ ] Implement:

```python
def compute_query_uncertainty(
    episode_rewards: list[float],
    episode_steps: list[int],
    reward_type: str,
) -> float:
    ...

def should_discard_query(episode_rewards: list[float]) -> bool:
    ...
```

- [ ] Use:
  - `Beta(1, 1)` posterior variance for binary rewards
  - Normal-Inverse-Gamma posterior mean variance for continuous rewards with unknown
    variance

- [ ] `should_discard_query(...)` semantics:
  - `False` for `len(rewards) < 2`
  - `True` only if `len(rewards) >= 2` and all rewards are identical

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v -k "uncertainty or discard"
```

---

### Task 3: Add constructor fields and workflow wiring

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`
- Modify: `areal/experimental/inference_service/controller/controller.py`
- Modify: `areal/infra/remote_inf_engine.py`
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add failing constructor tests for:
  - explicit dynamic config
  - fallback `initial_group_size <- group_size`
  - non-dynamic backward compatibility

- [ ] Extend `TreeSearchGroupedRolloutWorkflow.__init__` with:

```python
dynamic_group_size: bool = False
initial_group_size: int = 0
max_group_size: int = 64
uncertainty_threshold: float = 0.05
reward_type: str = "binary"
```

- [ ] Store:
  - `self.dynamic_group_size`
  - `self.initial_group_size`
  - `self.max_group_size`
  - `self.uncertainty_threshold`
  - `self.reward_type`

- [ ] Wire new args through both call sites:
  - `areal/experimental/inference_service/controller/controller.py`
  - `areal/infra/remote_inf_engine.py`

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v -k "workflow_constructor"
```

---

### Task 4: Implement fixed zero-variance discard

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add failing tests for:
  - 2+ identical-reward episodes -> `None`
  - mixed-reward episodes -> kept
  - single-episode result -> kept

- [ ] Import `should_discard_query` and apply it after all cached/fresh episodes are
  assembled and before distillation / tensor conversion returns the final trajectory.

- [ ] Ensure discard operates on episode-level rewards, not per-node duplicates.

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v -k "zero_variance"
```

---

### Task 5: Implement per-query dynamic sampling loop

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add failing tests for:
  - stop once `U(q) <= uncertainty_threshold`
  - cap at `max_group_size`
  - reuse cached episodes toward `initial_group_size`
  - tolerate failed extra samples

- [ ] Split `_arun_episode_impl(...)` into two internal paths:
  - fixed-size legacy path
  - dynamic path

- [ ] Dynamic path algorithm:

```python
need_gen = max(0, self.initial_group_size - cached_count)
sample initial_group_size episodes
compute uncertainty
while episode_count < self.max_group_size:
    if uncertainty <= self.uncertainty_threshold:
        break
    sample one more episode
    recompute uncertainty
```

- [ ] Do not introduce workflow-level batch state, percentile tracking, or reset hooks.

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v -k "dynamic_group_size"
```

---

### Task 6: Preserve TREE-mode precomputed advantages

**Files:**
- Modify: `areal/trainer/rl_trainer.py`
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add a failing test covering:
  - rollout trajectories already contain `advantages` and `returns`
  - trainer must not call `self.actor.compute_advantages(...)`

- [ ] Update trainer rollout handling around:

```python
adv_batch = self.actor.compute_advantages(rollout_batch)
```

to:

```python
if rollout_batch and all(
    "advantages" in traj and "returns" in traj for traj in rollout_batch
):
    adv_batch = rollout_batch
else:
    adv_batch = self.actor.compute_advantages(rollout_batch)
```

- [ ] Keep logging around the branch so TREE-mode behavior is observable.

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v -k "precomputed_advantages or trainer"
```

---

### Task 7: End-to-end workflow coverage

**Files:**
- Modify: `tests/customized_areal/test_dynamic_group_size.py`

- [ ] Add end-to-end tests for:
  - fixed mode unchanged
  - dynamic mode with binary rewards
  - dynamic mode with continuous rewards
  - discard after 2+ identical rewards
  - TREE-mode trajectory contains precomputed `advantages` and `returns`

- [ ] Verify that workflow output remains valid for existing batching utilities.

- [ ] Run:

```bash
uv run pytest tests/customized_areal/test_dynamic_group_size.py -v
```

---

### Task 8: Repo validation

- [ ] Run targeted existing tests that are most likely to regress:

```bash
uv run pytest tests/test_tree_search/test_tree_search_grouped_workflow.py -v
uv run pytest tests/test_tree_search/test_advantage.py -v
uv run pytest tests/test_tree_search/test_batch_consistency.py -v
```

- [ ] If the tree-search workflow is exercised through remote rollout paths, also run:

```bash
uv run pytest tests/test_prepare_batch.py -v
uv run pytest tests/test_rollout_controller.py -v
```

- [ ] Run formatting/lint hooks required by repo policy:

```bash
pre-commit run --all-files
```

---

### Notes

- This plan intentionally removes the earlier batch-percentile threshold design.
- This plan intentionally removes the earlier batch reset task.
- TREE-mode precomputed advantages now require a trainer integration change; this is no
  longer a workflow-only change.

# Tree Advantage Skip Design

## Problem

When `advantage_mode == TREE`, the `TreeSearchGroupedRolloutWorkflow` computes
per-query GRPO-normalized advantages and sets them on each Node (via
`TreeAdvantageComputer.compute()`). These advantages are carried into the batched
tensor dict via `_node_to_tensor_dict()`.

However, the base training loop in `rl_trainer.py:654` unconditionally calls
`self.actor.compute_advantages(rollout_batch)`, which recomputes advantages from
scratch using GAE and **overwrites** the pre-computed tree advantages. The
workflow's advantages are discarded.

Additionally, reward preprocessing (overlong penalty, reward scaling/clipping)
currently happens inside `_compute_advantages` in `actor.py`. When we skip
`compute_advantages` for tree mode, this preprocessing must still occur — it
needs to move into `TreeAdvantageComputer.compute()`.

Note: `reward_norm` (the `Normalization` class) is not moved because it
operates on GPU tensors with distributed all-reduce and batch/group-level
statistics. The tree GRPO normalization already provides per-query group
normalization, which is the intended behavior for tree mode.

## Design

### 1. Extract `_compute_advantages_for_batch` in `rl_trainer.py`

Replace lines 646-655:

```python
with (stats_tracker.record_timing("compute_advantage"), ...):
    adv_batch = self.actor.compute_advantages(rollout_batch)
    self.actor.get_device_stats().log("compute advantages")
```

With:

```python
adv_batch = self._compute_advantages_for_batch(rollout_batch, global_step)
```

New method in `RLTrainer`:

```python
def _compute_advantages_for_batch(self, rollout_batch, global_step):
    with (
        stats_tracker.record_timing("compute_advantage"),
        perf_tracer.trace_scope(
            "train.compute_advantage",
            category=Category.COMPUTE,
            args={"global_step": global_step},
        ),
    ):
        adv_batch = self.actor.compute_advantages(rollout_batch)
        self.actor.get_device_stats().log("compute advantages")
    return adv_batch
```

This preserves the existing behavior and is a minimal extraction — no logic
changes.

### 2. Override in `CacheAwarePPOTrainer`

```python
def _compute_advantages_for_batch(self, rollout_batch, global_step):
    if self.tree_backup_config.advantage_mode == AdvantageMode.TREE:
        # Advantages already set by TreeAdvantageComputer in the workflow.
        return rollout_batch
    return super()._compute_advantages_for_batch(rollout_batch, global_step)
```

When `advantage_mode == GAE`, the base class behavior runs unchanged.

### 3. Move reward preprocessing to `TreeAdvantageComputer`

The reward preprocessing currently in `_compute_advantages` (`actor.py:152-173`)
needs to also be applied in tree mode. Move this into `TreeAdvantageComputer.compute()`:

- Overlong reward penalty (`reward_overlong_penalty`)
- Reward scaling: `(reward + reward_bias) * reward_scaling`
- Reward clipping: `clip(reward, -reward_clip, reward_clip)`

`reward_norm` is NOT moved — the `Normalization` class operates on GPU
tensors with distributed all-reduce. The tree GRPO normalization already
provides per-query group normalization, which is the intended behavior.

`TreeBackupConfig` gains fields for these preprocessing parameters:

```python
@dataclass
class TreeBackupConfig:
    # ... existing fields ...
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    overlong_reward_penalty: bool = False
    overlong_tokens: int | None = None
    overlong_penalty_factor: float | None = None
```

`TreeAdvantageComputer.__init__` accepts these parameters and applies them to
`outcome_reward` before GRPO normalization.

## Files Changed

| File | Change |
|------|--------|
| `areal/trainer/rl_trainer.py` | Extract `_compute_advantages_for_batch` method |
| `customized_areal/tree_search/trainer.py` | Override `_compute_advantages_for_batch` |
| `customized_areal/tree_search/advantage.py` | Add reward preprocessing before GRPO norm |
| `customized_areal/tree_search/config.py` | Add reward preprocessing fields to `TreeBackupConfig` |

## Testing

- Unit test: `TreeAdvantageComputer` with reward preprocessing (bias, scaling,
  clipping, overlong penalty) produces correct normalized advantages
- Unit test: `CacheAwarePPOTrainer._compute_advantages_for_batch` returns
  `rollout_batch` directly when `advantage_mode == TREE`
- Unit test: `CacheAwarePPOTrainer._compute_advantages_for_batch` delegates to
  `super()` when `advantage_mode == GAE`
- Integration: verify that with `advantage_mode == TREE`, the advantages in the
  training batch match what `TreeAdvantageComputer` computed (not GAE)

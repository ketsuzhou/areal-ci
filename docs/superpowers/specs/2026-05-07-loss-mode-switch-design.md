# Loss Mode Switch Design

## Summary

Add a `LossMode` switch to `CacheAwarePPOTrainer` that enables selecting between
standard GRPO loss, position-level GRPO distillation loss, or a weighted combination of
both. This merges the distillation functionality from `TreeDistillPPOTrainer` directly
into `CacheAwarePPOTrainer`, controlled by a config field.

## Motivation

Currently, adding distillation to tree search training requires using a separate
`TreeDistillPPOTrainer` class. Users should be able to toggle between GRPO-only,
distill-only, or combined training via a simple config switch on the base trainer,
without switching trainer classes.

## Config Changes

**File:** `customized_areal/tree_search/config.py`

Add `LossMode` enum and extend `TreeBackupConfig`:

```python
class LossMode(str, Enum):
    GRPO = "grpo"           # Standard GRPO/PPO loss only
    DISTILL = "distill"     # Position-level GRPO distillation loss only
    BOTH = "both"           # Weighted combination

@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.TREE
    loss_mode: LossMode = LossMode.GRPO
    rl_loss_weight: float = 1.0
    distill_loss_weight: float = 0.005
```

Default `loss_mode=GRPO` preserves existing behavior.

## Trainer Changes

**File:** `customized_areal/tree_search/trainer.py`

### `__init__`

Store `self.tree_backup_config.loss_mode` for later dispatch.

### `_create_actor` override

- `loss_mode != GRPO` → return `MultiCandidateFSDPPPOActor` (enables multi-candidate
  logprob gathering for position-level rewards)
- `loss_mode == GRPO` → delegate to base `PPOTrainer._create_actor`

### `_init_patches`

When `loss_mode != GRPO`:

1. Patch `PPOActor._ppo_update` with `grpo_distill_loss_fn` via
   `patch_ppo_actor_class_to_use_distill_loss()`
1. Store reference for unpatching in `close()`

### `_cache_aware_prepare_batch`

After tree operations and before returning trajectories, inject loss weights into each
trajectory dict:

- `DISTILL` mode: `traj["rl_loss_weight"] = 0.0`
- `BOTH` mode: `traj["rl_loss_weight"] = self.tree_backup_config.rl_loss_weight`
- Both non-GRPO modes:
  `traj["distill_loss_weight"] = self.tree_backup_config.distill_loss_weight`

### `close`

When `loss_mode != GRPO`, unpatch the distill loss (restore original
`PPOActor._ppo_update`).

## Loss Behavior by Mode

The existing `grpo_distill_loss_fn` in
`customized_areal/on_policy_distill/training/loss.py` handles all three modes via weight
values:

| Mode      | `rl_loss_weight` | `distill_loss_weight` | Result                                       |
| --------- | ---------------- | --------------------- | -------------------------------------------- |
| `GRPO`    | N/A              | N/A                   | Standard PPO/GRPO loss (uses `grpo_loss_fn`) |
| `DISTILL` | 0.0              | 0.005                 | Position-level GRPO distillation only        |
| `BOTH`    | 1.0              | 0.005                 | Combined: `1.0 * GRPO + 0.005 * distill`     |

When `position_rewards` is `None` (no teacher configured), `grpo_distill_loss_fn` falls
back to pure GRPO loss regardless of mode.

## Files Changed

| File                                      | Change                                               |
| ----------------------------------------- | ---------------------------------------------------- |
| `customized_areal/tree_search/config.py`  | Add `LossMode` enum; extend `TreeBackupConfig`       |
| `customized_areal/tree_search/trainer.py` | Add `_create_actor`, loss patching, weight injection |

No changes to loss functions or actor patching infrastructure — existing code handles
all modes.

## Backward Compatibility

- Default `loss_mode=GRPO` means existing `CacheAwarePPOTrainer` usage is unchanged
- `TreeDistillPPOTrainer` remains as-is for backward compatibility; new functionality is
  available through `CacheAwarePPOTrainer` with `loss_mode=DISTILL` or `loss_mode=BOTH`

# TreeDistillPPOTrainer Design

## Summary

A combined trainer that merges MCTS tree backup advantages with on-policy distillation
loss and rollout caching in a single training step. Inherits from
`CacheAwarePPOTrainer`, adds distillation components from `OnPolicyDistillationTrainer`.

## Architecture

### Class: `TreeDistillPPOTrainer(CacheAwarePPOTrainer)`

Located at: `customized_areal/tree_search_distilling/trainer.py`

**Inheritance chain:**

```
PPOTrainer → CacheAwarePPOTrainer → TreeDistillPPOTrainer
```

### Initialization Order

1. Patch `PPOActor._ppo_update` with `grpo_distill_loss_fn` (before super init)
1. Initialize distillation workflow/agent (`OpenAIProxyWorkflow` + agent)
1. Call `CacheAwarePPOTrainer.__init__()` which:
   - Calls `PPOTrainer.__init__()` (creates actor via `_create_actor` override)
   - Sets up MCTS tree store, advantage computer, checkpoint manager
   - Patches `PPOActor.compute_advantages` for tree backup

### Key Overrides

- **`_create_actor(actor_config)`** → returns `MultiCandidateFSDPPPOActor` instead of
  default `FSDPPPOActor`. Enables multi-candidate logprob gathering for position-level
  distillation loss.
- **`close()`** → No override needed. `CacheAwarePPOTrainer.close()` unpatches
  `compute_advantages`; distill loss patch has no unpatch (one-time global).

### Config

Keep configs separate (follow `CacheAwarePPOTrainer` pattern):

- `OnPolicyDistillConfig` for distillation settings (proxy, teacher, reward scaling)
- `RolloutCacheConfig` parameter for cache settings
- `TreeBackupConfig` parameter for tree backup settings

## Data Flow

```
Step 1: Rollout
  OpenAIProxyWorkflow.arun_episode()
  → rollout_batch with position_rewards (PositionRewardInfo)

Step 2-5: Standard PPO pipeline
  critic.compute_values, ref.compute_logp, teacher.compute_logp, prox_logp

Step 6: Compute advantages [TREE BACKUP PATCHED]
  Original GAE runs first (KL rewards, scaling, normalization)
  → tree_store.insert_batch(result) — inserts with raw rewards
  → tree_advantage_computer.compute() — overwrites advantages/returns with MCTS Q-values
  → kl_rewards, tot_rewards, loss_mask, logprobs preserved from GAE

Step 7: PPO update [DISTILL LOSS PATCHED]
  grpo_distill_loss_fn computes:
    total_loss = rl_loss_weight * GRPO_loss(tree_advantages, chosen_logprobs)
               + distill_loss_weight * position_GRPO_loss(position_rewards, multi_candidate_logprobs)

Step 8: Mark trajectories as trained (CacheAwarePPOTrainer logic)
Step 9: Save checkpoints (model weights + MCTS tree state)
```

### Compatibility

Tree backup uses rewards in step 6 (`insert_batch`), distill loss removes them in step
7\. Since step 6 runs first in the training loop, there is no data flow conflict.

## File Structure

```
customized_areal/tree_search_distilling/
├── __init__.py                          # Export TreeDistillPPOTrainer
├── trainer.py                           # TreeDistillPPOTrainer class
├── scripts/
│   └── train_tree_search_distilling.py  # Entry point script
└── configs/
    └── config_tree_search_distilling.yaml # Training config
```

## Entry Point Script

```python
# customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py
from customized_areal.on_policy_distill.core.config import OnPolicyDistillConfig
from customized_areal.tree_search.config import RolloutCacheConfig, TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer
from areal.utils.config import load_expr_config

def main():
    config, args = load_expr_config(args, OnPolicyDistillConfig)
    cache_config = RolloutCacheConfig(
        cache_dir=getattr(config, "cache_dir", ""),
        enabled=True,
        n_samples=getattr(config, "n_samples", 1),
    )
    tree_backup_config = TreeBackupConfig(
        mode=TreeBackupMode.CROSS_TRAINING,
        assistant_marker=getattr(config, "assistant_marker", ""),
        checkpoint_dir=getattr(config, "cache_dir", ""),
    )
    trainer = TreeDistillPPOTrainer(
        config, cache_config, tree_backup_config
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

## YAML Config

Based on existing `config_on_policy_distill.yaml` with added cache/tree fields:

- `cache_dir`: directory for rollout cache and tree checkpoints
- `n_samples`: number of samples per prompt to cache
- `tree_backup_mode`: "off" | "in_training" | "cross_training"
- `assistant_marker`: auto-detect if empty

## No-Teacher Behavior

When no teacher model is configured, the trainer falls back to tree-search-only GRPO
training — but **student position-level logprobs must still be saved for
logging/analysis**.

### What changes without a teacher

1. **`config.teacher is None`** — `PPOTrainer` will not create a teacher engine, skip
   `teacher.compute_logp()`, and never inject `teacher_logp`, `rl_loss_weight`, or
   `distill_loss_weight` into batch dicts.
1. **`teacher_model_name == ""`** — the agent will not build a `TeacherClient`.
1. **`grpo_distill_loss_fn`** — when `position_rewards` is `None` in `input_data`, the
   distillation branch is entirely skipped and loss equals pure GRPO loss.

### Saving student position-level logprobs without a teacher

Currently, `OnPolicyDistillAgent` only builds `PositionRewardInfo` objects when
`teacher_client is not None` — because the reward computation requires teacher logprobs.
We need a **student-only path** that builds `PositionRewardInfo` from the student's own
top-k logprobs when no teacher is available:

- **`candidate_token_ids`**: from `student_top_k_logprobs` (top-k token IDs at each
  position)
- **`logprobs`**: student's own logprobs for those tokens
- **`rewards`**: all zeros (no teacher signal)
- **`chosen_index`**: position of the actually-generated token in the candidate list

This ensures `MultiCandidateFSDPEngine` still gathers multi-candidate logprobs during
training (the 2D label tensor is built from `candidate_token_ids`), and the student's
per-position logprob distribution is available for logging. The position-level GRPO loss
will contribute nothing because rewards are zero.

Implementation: add a `_build_student_only_position_rewards()` method to the agent (or a
helper function) that constructs `PositionRewardInfo` objects from
`student_top_k_logprobs` without calling the teacher.

## Implementation Notes

- Both monkey-patches target different PPOActor methods (`compute_advantages` vs
  `_ppo_update`), so they compose without conflict.
- The distill loss patch has a global `_patch_applied` guard preventing double-patching.
- The tree backup patch preserves original via `_original_compute_advantages` for safe
  restore.
- `MultiCandidateFSDPPPOActor` is required for multi-candidate logprob gathering;
  standard `FSDPPPOActor` cannot produce the 2D logprobs `[seq_len, num_candidates]`
  that `grpo_distill_loss_fn` expects when `position_rewards` are present.
- Only FSDP backend is supported (same constraint as `OnPolicyDistillationTrainer`).

# Offline On-Policy Distillation Training Script Design

## Summary

A standalone training script (`offline_train.py`) that exercises the on-policy
distillation training pipeline **without inference**. It directly reuses AReaL modules
(`MultiCandidateFSDPEngine`, `grpo_distill_loss_fn`,
`gather_logprobs_entropy_multi_candidates`) to validate the production code path.
Supports real model weights via HuggingFace + FSDP2, with data from saved rollout files
or synthetic mock generation.

## Identified Bugs

### Bug 1: `distill_stat` not detached (loss.py:146)

`distill_stat = position_grpo_loss` captures a live tensor with grad. Passing it to
`stats_tracker.stat()` retains the computational graph, causing memory leaks and
incorrect stat aggregation. **Fix**: `distill_stat = position_grpo_loss.detach()`

### Bug 2: Missing reward logging (actor.py:49-50)

The patched `_ppo_update_with_distill_loss` pops `rewards`, `tot_rewards`, `kl_rewards`
before logging them. The original `_ppo_update` logs these stats before consuming them.
The patched version loses all reward-related logging (correct/incorrect sequence counts,
task reward), making it impossible to monitor training quality. **Fix**: Log reward
stats before popping.

### Bug 3: prompt_len only for first sample (loss.py:120-125)

When `loss_mask` has batch dim `[batch, seq_len]`, `loss_mask.squeeze(0)` reduces to
`[seq_len]` only when batch=1. With batch>1, `squeeze(0)` is a no-op and the loop
iterates over the wrong dimension. Even when batch=1, it only finds prompt_len for the
first sample, which is wrong when samples have different prompt lengths. Since
PositionRewardInfo has a `sample_index` field, we need per-sample prompt lengths to
correctly offset positions. **Fix**: Compute prompt_len per sample using `loss_mask`
rows: `prompt_len[i] = (loss_mask[i] == 0).sum()` or the first-true-index approach per
row. Then use `prompt_len[pr.sample_index]` when computing the absolute position.

### Bug 4: GPU-CPU sync in hot path (loss.py:342)

`loss_mask.sum().item()` forces a GPU-CPU sync. This is called in the loss function
which runs per minibatch per training step. Note: `int(tensor)` is equivalent to
`.item()` — both sync. **Fix**: Keep the computation on GPU: use
`loss_mask.sum().clamp(min=1)` directly as a tensor in division, and avoid `.item()` for
`output_len` by using `loss_mask.sum().long()` for tensor ops. Lower priority (perf, not
correctness).

### Bug 5: Position indexing with padded sequences (loss.py:309)

`new_logprobs = logprobs[positions_t, :]` uses absolute positions. If `logprobs`
includes padding tokens at the end and `position` values are based on the unpadded
sequence length, indexing could exceed bounds or select wrong positions. **Fix**: Add
bounds checks: `position = min(position, logprobs.shape[0] - 1)`.

### Bug 6: Duplicate denominator (actor.py:88-98)

After `grpo_distill_loss_fn` already registers `n_valid_tokens` denominator
(loss.py:138-142), the patched `_ppo_update` registers it again (actor.py:98). This
creates duplicate entries. **Fix**: Remove the duplicate denominator registration from
`_ppo_update_with_distill_loss`.

## Architecture

### File Location

`customized_areal/on_policy_distill/training/offline_train.py`

### Components

1. **Model Setup**: Load HF model, wrap in FSDP2 using `MultiCandidateFSDPEngine`
1. **Data Loader**: Load `.pt` files from disk or generate mock data
1. **Training Loop**: Call `engine.train_batch` with `grpo_distill_loss_fn`
1. **Stats Logging**: Log loss, clip_ratio, KL, distill_loss per step
1. **Checkpointing**: Save model/optimizer state periodically

### Data Flow

```
[data on disk / mock generator]
  → load batch dict (input_ids, attention_mask, loss_mask, advantages, logprobs, position_rewards)
  → engine.train_batch(data, loss_fn=grpo_distill_loss_fn, ...)
  → engine internally: forward pass → _compute_logprobs_entropy → loss_fn → backward
  → optimizer.step()
```

## Data Format

### Saved Rollout Data (`.pt` files)

| Key                | Shape                      | Description                        |
| ------------------ | -------------------------- | ---------------------------------- |
| `input_ids`        | `[batch, seq_len]`         | Token IDs                          |
| `attention_mask`   | `[batch, seq_len]`         | 1 for real tokens                  |
| `loss_mask`        | `[batch, seq_len]`         | 1 for output tokens (0 for prompt) |
| `logprobs`         | `[batch, seq_len]`         | Old logprobs from rollout          |
| `advantages`       | `[batch, seq_len]`         | Precomputed advantages             |
| `position_rewards` | `list[PositionRewardInfo]` | Per-position candidate info        |

### Mock Data

When `--mock_data` is set (or `--data_path` is omitted), the script generates:

- Random `input_ids` in valid vocab range
- Proper `attention_mask` / `loss_mask` patterns
- Random `advantages` and `logprobs`
- Synthetic `PositionRewardInfo` with 2-4 candidates per position

## CLI Interface

```
python offline_train.py \
  --model_path /path/to/Qwen2.5-3B \
  --data_path /path/to/rollout_data/ \   # optional
  --output_dir ./checkpoints/ \
  --num_epochs 3 \
  --batch_size 4 \
  --seq_len 512 \
  --lr 1e-6 \
  --eps_clip 0.2 \
  --distill_loss_weight 0.005 \
  --rl_loss_weight 1.0 \
  --save_every 100 \
  --tp_size 1 \
  --mock_data
```

## Training Loop

```
1. Parse CLI args
2. Load model from HF path, wrap in FSDP2
3. Create optimizer (AdamW) and LR scheduler (cosine)
4. For each epoch:
   a. Load batch data (from disk or mock)
   b. Prepare model inputs
   c. Call engine.train_batch() with grpo_distill_loss_fn
   d. Log stats (loss, clip_ratio, KL, distill_loss)
   e. Step optimizer
   f. Every N steps: save checkpoint
5. Final checkpoint save
```

## Bug Fixes in Standalone Script

The script will apply workarounds for the 6 identified bugs with clear comments marking
each fix. These workarounds validate that the fixes work correctly before they are
applied to the production code.

## Constraints

- No inference/rollout — all data comes from disk or mock generation
- No scheduler/RPC/proxy — single-process or local multi-GPU only
- Uses `patch_ppo_actor_class_to_use_distill_loss()` to ensure the same code path
- Requires GPU (FSDP2)

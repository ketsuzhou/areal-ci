# train_id Checkpoint Persistence

## Problem

When training with `CacheAwarePPOTrainer`, a `train_id` (UUID) is generated at
startup and stored in `os.environ["TRAIN_ID"]`. `MCTSTreeStore` uses this to
determine which nodes have already been trained (`is_trained()` checks
`node.train_id == current_train_id`).

On resume from a model checkpoint, a **new** `train_id` is generated, breaking
the link: all previously trained nodes appear untrained, causing redundant
re-training of cached trajectories.

## Solution

Persist `train_id` as a sidecar `train_id.json` file alongside both recover
and HF model checkpoints. On resume, restore the `train_id` from the recover
checkpoint's sidecar file, overriding any env value.

## Constraints

- No modifications to `areal/` — all changes in `customized_areal/`
- Override `CacheAwarePPOTrainer._save_hf` and
  `CacheAwarePPOTrainer._save_recover_checkpoint` to write sidecar files
- On startup, check for existing recover checkpoint before generating a new
  `train_id`

## Design

### 1. Sidecar file format

File: `train_id.json` in the checkpoint directory

```json
{"train_id": "a1b2c3d4e5f6..."}
```

Atomic write pattern: write to `.tmp`, then `os.replace()`.

### 2. Save: write `train_id.json` on checkpoint

**Recover checkpoint** — override `_save_recover_checkpoint` in
`CacheAwarePPOTrainer`:

1. Call `super()._save_recover_checkpoint(epoch, epoch_step, global_step)`
2. Compute recover checkpoint path via `Saver.get_recover_checkpoint_path()`
   using `self.config.recover.experiment_name`, `trial_name`, `fileroot`
3. Write `train_id.json` into the recover checkpoint directory for each engine
   name (`"default"`, `"critic"`)

**HF model checkpoint** — override `_save_hf` in `CacheAwarePPOTrainer`:

1. Call `super()._save_hf(epoch, epoch_step, global_step)`
2. Compute model save path via `Saver.get_model_save_path()`
3. Write `train_id.json` into the HF checkpoint directory for each engine name

### 3. Load: restore `train_id` on startup

In `train_tpfc_tree_search.py`, **before** the existing `TRAIN_ID` generation:

1. Check if a recover checkpoint exists by computing
   `Saver.get_recover_checkpoint_path()` from the config's
   `experiment_name`, `trial_name`, `fileroot`
2. If `train_id.json` exists in that path, read and set
   `os.environ["TRAIN_ID"]` — this overrides any existing value
3. If no `train_id.json` found, generate a new UUID (current behavior)

This ensures `MCTSTreeStore.__init__` picks up the restored `train_id` from
the environment.

### 4. Files to modify

| File | Change |
|------|--------|
| `customized_areal/tpfc/scripts/train_tpfc_tree_search.py` | Load train_id from recover checkpoint before generating UUID |
| `customized_areal/tree_search/trainer.py` | Override `_save_hf` and `_save_recover_checkpoint` to write `train_id.json` sidecar |

### 5. Error handling

- If `train_id.json` exists but is corrupt (invalid JSON, missing key), log a
  warning and fall back to generating a new UUID
- Only rank 0 writes the sidecar file (consistent with RecoverHandler pattern)

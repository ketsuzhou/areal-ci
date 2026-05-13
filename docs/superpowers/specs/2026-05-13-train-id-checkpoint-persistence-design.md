# Train ID Checkpoint Persistence

## Problem

When resuming training from a model checkpoint, a new `train_id` UUID is
generated. This breaks the link between cached MCTS nodes and the resumed
training run: `is_trained()` returns `False` for all previously trained nodes,
causing unnecessary re-training.

## Design

### Sidecar `train_id.json` alongside checkpoints

Save a `train_id.json` file in every checkpoint directory (both recover/DCP and
HF model checkpoints). The file contains `{"train_id": "<hex>"}`.

**On save** — write `train_id.json` atomically (tmp + rename) after the
checkpoint is written:
- `RecoverHandler._save_checkpoint()` in `areal/utils/recover.py`
- `Saver.save()` in `areal/utils/saver.py` (both sync and async paths)

**On resume** — the training script checks for a recover checkpoint's
`train_id.json` before deciding the train_id:

1. Compute the recover checkpoint path from config: `Saver.get_recover_checkpoint_path(experiment_name, trial_name, fileroot)` → `{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/default/recover_checkpoint/`
2. If `train_id.json` exists in that directory → read the `train_id` and set `os.environ["TRAIN_ID"]`
3. If not → generate a new UUID and set `os.environ["TRAIN_ID"]`

This replaces the current logic in `train_tpfc_tree_search.py` that only
generates a UUID when `TRAIN_ID` is not already in the environment.

### Files to modify

1. **`areal/utils/recover.py`** — `RecoverHandler._save_checkpoint()`: write `train_id.json` after saving the DCP checkpoint
2. **`areal/utils/saver.py`** — `Saver.save()`: write `train_id.json` after saving the HF checkpoint (sync and async paths)
3. **`customized_areal/tpfc/scripts/train_tpfc_tree_search.py`** — Replace the current `TRAIN_ID` generation logic with: check for recover checkpoint `train_id.json` first, then fall back to generating a new UUID

### `train_id.json` format

```json
{"train_id": "a1b2c3d4e5f6..."}
```

### Resume flow

```
Script starts
  → Determine recover checkpoint path from config
  → If train_id.json exists at recover checkpoint path
      → Read train_id, set os.environ["TRAIN_ID"]
  → Else
      → Generate new UUID, set os.environ["TRAIN_ID"]
  → Load config, create trainer
  → MCTSTreeStore.__init__ reads os.environ["TRAIN_ID"]
  → Cached nodes with matching train_id are correctly marked as trained
```

### What this does NOT change

- The MCTS tree checkpoint (`TreeCheckpointManager`) already saves/loads
  `current_train_id` in `metadata.json` — no changes needed there
- The `MCTSTreeStore.is_trained()` logic — it already works correctly once
  `current_train_id` is set properly
- The recover checkpoint format (`RecoverInfo`) — train_id is kept in a
  separate sidecar file, not embedded in `step_info.json`

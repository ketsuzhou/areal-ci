# Rollout Cache with Tree Backup Training

**Date**: 2026-04-17 **Status**: Approved

## Problem

On-policy distillation training requires generating rollouts via inference each training
step. When restarting training from scratch, all rollouts must be regenerated even if
the same prompts were seen before. This wastes inference compute. We want to cache
rollouts so that a second training run can reuse them without re-generating, only
rolling out the remaining samples needed to form a complete GRPO group.

## Solution

A `CacheAwarePPOTrainer` that:

1. Saves generated rollout trajectories into a self-contained MCTS tree structure
1. On subsequent training runs, loads cached trajectories from the tree and skips
   inference for prompts with enough cached samples
1. For partially-cached prompts, generates only the remaining `n_samples - cached`
   trajectories, merges with cached ones, and trains on the combined batch
1. Tracks a `trained` flag per trajectory to avoid re-training already-consumed data
1. Integrates with `TreeBackupPPOTrainer`'s MCTS backup for advantage computation

## Components

### 1. Extended TrieNode

Extend `TrieNode` to store all fields needed for training, not just tokens:

```python
@dataclass
class TrieNode:
    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)
    # NEW: per-token training metadata
    logprobs: list[float] = field(default_factory=list)      # logprob per token
    versions: list[int] = field(default_factory=list)         # policy version per token
```

### 2. Extended MCTSTreeStore

Add `trained` flag tracking and `reward` storage:

```python
class MCTSTreeStore:
    # Existing fields...
    _trained: dict[tuple[str, int], bool]  # NEW: (query_id, seq_id) -> trained flag
    _rewards: dict[tuple[str, int], float]  # NEW: (query_id, seq_id) -> episode reward

    def set_trained(self, query_id: str, seq_id: int, trained: bool = True) -> None:
        ...

    def is_trained(self, query_id: str, seq_id: int) -> bool:
        ...

    def get_untrained_count(self, query_id: str) -> int:
        """Count trajectories for a prompt that haven't been trained on."""
        ...

    def load_trajectories(self, query_id: str, n_samples: int) -> list[dict]:
        """Extract up to n_samples untrained trajectories from tree as training dicts."""
        ...

    def reset_trained_flags(self) -> None:
        """Mark all trajectories as untrained (for new training run from scratch)."""
        ...
```

`load_trajectories` walks tree paths for untrained seq_ids and reconstructs the standard
trajectory dict format:

```python
{
    "input_ids":     torch.tensor([1, seq_len], dtype=torch.int32),
    "loss_mask":     torch.tensor([1, seq_len], dtype=torch.int32),
    "logprobs":      torch.tensor([1, seq_len], dtype=torch.float32),
    "versions":      torch.tensor([1, seq_len], dtype=torch.int32),
    "attention_mask": torch.tensor([1, seq_len], dtype=torch.bool),
    "rewards":       torch.tensor([1], dtype=torch.float32),  # scalar
}
```

### 3. Extended TreeCheckpointManager

Update serialization/deserialization to handle the new fields (`logprobs`, `versions`,
`trained`, `rewards`):

- `_serialize_node`: also serialize `logprobs`, `versions`
- `_deserialize_node`: also deserialize `logprobs`, `versions`
- `save`: also persist `_trained` and `_rewards` dicts in metadata
- `load`: also restore `_trained` and `_rewards` dicts

### 4. RolloutCacheConfig

```python
@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1  # must match gconfig.n_samples
```

### 5. CacheAwarePPOTrainer

Subclass of `PPOTrainer` that intercepts the training loop for cache-aware rollout
generation.

**Initialization**:

1. If `cache_config.enabled` and `cache_dir` exists: load tree checkpoint
1. Reset all `trained` flags to `False` (new training run from scratch)
1. Set up tree backup (patch `PPOActor.compute_advantages`)

**Per-step logic** (overrides the training step):

For each prompt in the batch:

1. Compute `prompt_hash` from prompt tokens
1. `untrained_count = tree_store.get_untrained_count(prompt_hash)`
1. Three cases:
   - **Enough cached** (`untrained_count >= n_samples`): Load `n_samples` trajectories
     from tree, skip inference entirely
   - **Partially cached** (`0 < untrained_count < n_samples`): Load all untrained
     trajectories, generate `n_samples - untrained_count` new ones via `prepare_batch`
     with `group_size = n_samples - untrained_count`, insert new trajectories into tree,
     merge with loaded ones via `concat_padded_tensors()`
   - **Not cached** (`untrained_count == 0`): Generate all `n_samples` trajectories
     normally, insert into tree
1. After `compute_advantages` (which runs tree backup): mark all used trajectories as
   `trained=True`
1. Run `ppo_update` on the merged batch

**Tree checkpoint saving**:

- Save tree checkpoint at the same cadence as regular checkpoints
- Uses `TreeBackupMode.CROSS_TRAINING` to persist across runs

### 6. Training Script

File: `customized_areal/on_policy_distill/scripts/train_with_cache.py`

```python
def main(args):
    config, _ = load_expr_config(args, OnPolicyDistillConfig)
    cache_config = RolloutCacheConfig(
        cache_dir=config.cache_dir,
        n_samples=config.gconfig.n_samples,
    )
    tree_backup_config = TreeBackupConfig(
        mode=TreeBackupMode.CROSS_TRAINING,
        checkpoint_dir=cache_config.cache_dir,
        assistant_marker=getattr(config, "assistant_marker", ""),
    )
    trainer = CacheAwarePPOTrainer(
        config=config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
    )
    trainer.train()
```

## Data Flow

### First Training Run (no cache)

1. Prompt -> `prepare_batch(group_size=n_samples)` generates `n_samples` trajectories
1. `compute_advantages` -> GAE + tree backup (insert trajectories, compute Q-values)
1. `ppo_update` -> train on batch
1. Mark used trajectories `trained=True`
1. Save tree checkpoint with extended fields + trained flags

### Second Training Run (from scratch, cache exists)

1. Load tree checkpoint -> reset all `trained=False`
1. For each prompt:
   - Count untrained cached trajectories
   - Enough cached -> load from tree, skip inference
   - Partial -> load cached + generate remainder -> merge -> insert new into tree
1. `compute_advantages` -> tree backup on merged batch
1. `ppo_update` -> train, mark used trajectories `trained=True`
1. Save updated tree checkpoint

## File Changes

| File                                                             | Change                                                                                                     |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/trie_node.py`                      | Add `logprobs`, `versions` fields                                                                          |
| `customized_areal/tree_search/mcts_tree_store.py`                | Add `trained` tracking, `load_trajectories`, `get_untrained_count`, `reset_trained_flags`, `_rewards` dict |
| `customized_areal/tree_search/checkpoint.py`                     | Serialize/deserialize new fields, `_trained`, `_rewards`                                                   |
| `customized_areal/tree_search/config.py`                         | Add `RolloutCacheConfig`                                                                                   |
| `customized_areal/tree_search/trainer.py`                        | New `CacheAwarePPOTrainer` class                                                                           |
| `customized_areal/on_policy_distill/scripts/train_with_cache.py` | New training script                                                                                        |

## Key Decisions

1. **Tree is the primary cache** — no separate trajectory files; the MCTS tree stores
   all data needed for training
1. **`trained` flag per trajectory** — prevents re-training already-consumed data in
   subsequent runs; reset on new training run from scratch
1. **Partial group reuse** — if a prompt has some cached but not enough for a full GRPO
   group, generate only the remainder
1. **Patch PPOTrainer loop** — subclass and override training step rather than wrapping
   the workflow (matches existing pattern in TreeBackupPPOTrainer)

# Tree Search: MCTS Tree Backup for PPO Training

This module replaces GAE advantage computation with MCTS tree backup Q-values, enabling
rollout caching across training steps. It also supports on-policy distillation with a
teacher model. It is a customization layer on top of AReaL's `PPOTrainer`.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    CacheAwarePPOTrainer                          │
│  (extends PPOTrainer with MultiCandidateFSDPPPOActor support)   │
│                                                                  │
│  __init__()                                                      │
│   ├─ Accepts TreeBackupConfig, RolloutCacheConfig               │
│   └─ Stores tree_backup_config for later use                    │
│                                                                  │
│  _create_train_engine()                                          │
│   ├─ If loss_mode != GRPO: returns MultiCandidateFSDPPPOActor   │
│   └─ Otherwise: delegates to standard PPOTrainer engine          │
│                                                                  │
│  train()                                                         │
│   ├─ If loss_mode != GRPO: applies distill loss patch           │
│   ├─ Calls super().train() (standard training loop)             │
│   └─ Restores patch in finally block                             │
│                                                                  │
│  close()                                                         │
│   └─ Delegates to parent                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              TreeSearchGroupedRolloutWorkflow                    │
│  (extends RolloutWorkflow with cache reuse + tree ops)           │
│                                                                  │
│  arun_episode()                                                  │
│   ├─ Checks cache: how many untrained episodes exist?            │
│   ├─ Generates only needed fresh episodes (partial reuse)        │
│   ├─ Converts fresh results to Nodes via interactions_dict_to_nodes
│   ├─ Loads cached Nodes from MCTSTreeStore                       │
│   ├─ Combines fresh + cached Nodes                               │
│   ├─ Inserts fresh Nodes into tree_store                         │
│   ├─ Computes tree advantages (TREE mode)                        │
│   ├─ Marks all nodes as trained                                  │
│   ├─ Saves tree checkpoint (CROSS_TRAINING mode)                 │
│   └─ Converts to batched tensor dict via _nodes_to_batched_tensor_dict
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         MCTSTreeStore                            │
│  (flat trajectory store with MCTS statistics)                    │
│                                                                  │
│  insert_batch() → store trajectories                             │
│  load_trajectories() → retrieve untrained Nodes                  │
│  get_untrained_count() → check cache availability                │
│  set_trained() / is_trained() → track usage                      │
│  _backup() → update MCTS Q-values                                │
└──────────────────────────────────────────────────────────────────┘
```

## Component Reference

### 1. Config (`config.py`)

Dataclasses controlling tree backup, caching, and advantage computation.

| Class                | Field                 | Type            | Default | Description                             |
| -------------------- | --------------------- | --------------- | ------- | --------------------------------------- |
| `TreeBackupConfig`   | `mode`                | `CacheMode`     | `OFF`   | Controls when/how tree backup activates |
|                      | `checkpoint_dir`      | `str`           | `""`    | Directory for MCTS tree checkpoints     |
|                      | `advantage_mode`      | `AdvantageMode` | `TREE`  | TREE (Q-values) or GAE advantages       |
|                      | `loss_mode`           | `LossMode`      | `GRPO`  | GRPO, DISTILL, or BOTH                  |
|                      | `rl_loss_weight`      | `float`         | `1.0`   | Weight for RL loss in BOTH mode         |
|                      | `distill_loss_weight` | `float`         | `0.005` | Weight for distillation loss            |
| `RolloutCacheConfig` | `cache_dir`           | `str`           | `""`    | Directory for rollout cache             |
|                      | `enabled`             | `bool`          | `True`  | Enable/disable caching                  |
|                      | `n_samples`           | `int`           | `1`     | Number of rollout samples per prompt    |

**`CacheMode`** values:

- `OFF` — standard PPOTrainer, no tree backup
- `IN_TRAINING` — tree backup within a single training run (no checkpoint save/load)
- `CROSS_TRAINING` — tree persists across runs; checkpoint is saved/loaded

**`AdvantageMode`** values:

- `GAE` — standard GAE advantages (tree store is still populated for caching)
- `TREE` — MCTS Q-value advantages override GAE

**`LossMode`** values:

- `GRPO` — standard GRPO loss
- `DISTILL` — distillation loss only (rl_loss_weight=0)
- `BOTH` — combined GRPO + distillation loss

### 2. MCTS Tree Store (`mcts_tree_store.py`)

The central data structure. Manages a flat per-query list of `Node` objects, tracks MCTS
statistics per trajectory, and provides cached trajectory loading.

#### Node Dataclass

A `Node` represents one assistant response turn with its full conversation context (all
tokens from the beginning through this turn's response). Nodes are linked via `node_id`
/ `parent_node_id` and grouped into episodes via `episode_id`.

| Field            | Type                        | Description                                     |
| ---------------- | --------------------------- | ----------------------------------------------- |
| `input_ids`      | `list[int]`                 | Full token sequence (prompt + response)         |
| `loss_mask`      | `list[int]`                 | 0=prompt tokens, 1=response tokens              |
| `logprobs`       | `list[float]`               | Per-token log probabilities                     |
| `versions`       | `list[int]`                 | Policy version per token (-1 on prompt)         |
| `node_id`        | `str`                       | Globally unique interaction ID (UUID)           |
| `parent_node_id` | `str \| None`               | Parent interaction ID (None for root)           |
| `episode_id`     | `str`                       | Groups turns into a trajectory path             |
| `turn_idx`       | `int`                       | 1-based turn position within episode            |
| `query_id`       | `str`                       | Dataset query identifier                        |
| `outcome_reward` | `float`                     | Trajectory-level reward                         |
| `advantages`     | `torch.Tensor \| None`      | Tree-computed per-token advantages              |
| `returns`        | `torch.Tensor \| None`      | Tree-computed per-token returns                 |
| `topk_ids`       | `list[list[int]] \| None`   | Top-k candidate token IDs per response position |
| `topk_logp`      | `list[list[float]] \| None` | Top-k candidate log probabilities               |
| `distill_reward` | `list[list[float]] \| None` | Per-position distillation rewards               |
| `teacher_logp`   | `list[list[float]] \| None` | Teacher log probabilities per position          |

**Turn boundaries** are derived from `loss_mask` transitions (0→1 = response start, 1→0
= response end) via `_find_turn_boundaries()`, rather than using tokenizer-specific
assistant markers.

#### Store Methods

| Method                                  | Description                                                                |
| --------------------------------------- | -------------------------------------------------------------------------- |
| `insert_batch(trajectories)`            | Insert trajectories (Node objects) from rollout; skip already-cached nodes |
| `get_q_value(node_id)`                  | Raw Q-value (mean reward) for a trajectory                                 |
| `set_trained(node_id)` / `is_trained()` | Mark/check whether a trajectory has been used                              |
| `get_untrained_count(query_id)`         | Count untrained trajectories for a query                                   |
| `get_untrained_node_ids(query_id, n)`   | Get up to N untrained node IDs                                             |
| `load_trajectories(query_id, n)`        | Load untrained Node objects                                                |
| `reset_trained_flags()`                 | Reset all trained flags (for fresh training run)                           |
| `mark_episodes_trained(episode_ids)`    | Mark trained by episode ID set (for recover checkpoint restore)            |
| `clear()`                               | Reset all state                                                            |
| `set/get_normalized_advantage(node_id)` | Store/retrieve GRPO-normalized advantage                                   |
| `set/get_normalized_return(node_id)`    | Store/retrieve GRPO-normalized return                                      |

**MCTS backup** (`_backup`): Each trajectory gets a single Q-value = mean reward (visit
count = 1 currently). Stored in `_visit_counts`, `_total_values`, `_q_values`.

**Node ID assignment** (`_insert_single`): Each Node receives its `node_id` from the
inference engine (a UUID string). The Node's `query_id` is set during insertion.

### 3. Advantage Computer (`advantage.py`)

`TreeAdvantageComputer` replaces GAE advantages with normalized MCTS Q-values.

```
tree_advantage_computer.compute(trajectories)
```

For each trajectory:

1. Collect all `(query_id, node_id)` pairs across the batch
1. **Per-query GRPO normalization of outcome_rewards** for returns: normalize rewards to
   zero-mean unit-variance within each query group (so episodes for the same prompt are
   compared against each other)
1. For each trajectory, compute per-token advantages: normalized Q-value × prompt_mask
   (value on response tokens, 0 on prompt tokens)
1. Set `node.advantages` and `node.returns` in-place

Handles Node objects directly, setting attributes on the Node dataclass.

### 4. Checkpoint Manager (`checkpoint.py`)

Serializes/deserializes the full MCTS tree state to disk.

| Method                              | Description                                                                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `save(tree_store)`                  | Save per-query trajectory records as `query_{sanitized_id}.json` + `metadata.json` (node_id indices, MCTS stats, trained flags, rewards) |
| `load()`                            | Restore `MCTSTreeStore` from disk. No rebuild needed — stats keyed by string node_id.                                         |
| `exists()`                          | Check if a checkpoint directory exists                                                                                        |
| `save_trained_episodes(dir, store)` | Save trained episode IDs to recover checkpoint directory                                                                      |
| `load_trained_episodes(dir)`        | Load trained episode IDs from recover checkpoint directory                                                                    |

### 5. Tree Search Grouped Rollout Workflow (`tree_search_grouped_workflow.py`)

`TreeSearchGroupedRolloutWorkflow` is the core component that extends `RolloutWorkflow`
to provide tree-search-aware rollout with cache reuse.

**Initialization (`__init__`):**

- Creates `TreeCheckpointManager` and `MCTSTreeStore`
- On `CROSS_TRAINING` mode, loads existing tree checkpoint if available
- Creates `TreeAdvantageComputer`
- Resets trained flags for a fresh training run

**Per-episode flow (`arun_episode`):**

1. **Check cache**: Count untrained episodes for the query via `tree_store.get_untrained_count()`
2. **Generate fresh episodes** if needed: Run `group_size - cached_count` parallel rollouts via `asyncio.gather`
3. **Convert results to Nodes**: `interactions_dict_to_nodes()` converts `InteractionWithTokenLogpReward` objects to `list[Node]`
4. **Load cached nodes**: `tree_store.load_trajectories(query_id, cached_count)`
5. **Combine**: Merge fresh and cached nodes (total = group_size)
6. **Insert fresh nodes**: `tree_store.insert_batch(fresh_nodes)`
7. **Compute tree advantages**: `tree_advantage_computer.compute(all_nodes)` (TREE mode)
8. **Mark trained**: Set trained flags for all nodes
9. **Save checkpoint**: `tree_checkpoint_manager.save()` (CROSS_TRAINING mode)
10. **Convert to tensor dict**: `_nodes_to_batched_tensor_dict()` converts `list[Node]` to batched tensor dict
11. **Inject distill weights**: Set `rl_loss_weight` and `distill_loss_weight` if loss_mode != GRPO

**Utility functions:**

| Function                          | Description                                                                 |
| --------------------------------- | --------------------------------------------------------------------------- |
| `interactions_dict_to_nodes()`    | Convert `dict[str, InteractionWithTokenLogpReward]` to `list[Node]`         |
| `_nodes_to_batched_tensor_dict()` | Convert `list[Node]` to batched tensor dict via `concat_padded_tensors`     |

### 6. Trainer (`trainer.py`)

#### `CacheAwarePPOTrainer`

PPO trainer with tree-search-aware rollout support. Extends `PPOTrainer` directly.

**Key design**: All cache logic, tree ops, and checkpoint saving happen inside
`TreeSearchGroupedRolloutWorkflow` (activated by the `.env` flag in
`customized_areal/.env`). The trainer itself is minimal:

**Initialization (`__init__`):**

- Accepts `tree_backup_config` and stores it
- Delegates to `PPOTrainer.__init__()`

**`_create_train_engine`:**

- When `loss_mode != GRPO`: Returns `MultiCandidateFSDPPPOActor` (requires FSDP backend)
- Otherwise: Delegates to standard `PPOTrainer._create_train_engine()`

**`train()`:**

- When `loss_mode != GRPO`: Applies distill loss patch, calls `super().train()`, restores patch in `finally`
- Otherwise: Delegates to `super().train()`

### 7. Distillation Support

#### `distill_types.py`

| Class                             | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `PositionRewardInfo`              | Per-position candidate tokens, logprobs, and rewards              |
| `InteractionWithTokenLevelReward` | Extended interaction with `token_rewards` and `token_reward_mask` |

#### `core/` — On-Policy Distillation Core

| File                     | Purpose                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| `core/config.py`         | `OnPolicyDistillConfig` (extends PPOConfig) and `AgentConfig`      |
| `core/agent.py`          | `OnPolicyDistillAgent` — agent class for distillation training     |
| `core/reward_compute.py` | `_compute_token_rewards()` — student vs teacher logprob comparison |
| `core/teacher_client.py` | `TeacherClient` — async client for remote teacher model inference  |

#### `engine/` — Multi-Candidate Engine

| File                    | Purpose                                                                         |
| ----------------------- | ------------------------------------------------------------------------------- |
| `engine/fsdp_engine.py` | `MultiCandidateFSDPEngine` — FSDP engine with multi-candidate logprob gathering |
| `engine/actor.py`       | `MultiCandidateFSDPPPOActor` — PPO actor wrapping `MultiCandidateFSDPEngine`    |

**Key capabilities of `MultiCandidateFSDPEngine`:**

- `_compute_logprobs_entropy()`: Computes logprobs for multiple candidate tokens per
  position using `gather_logprobs_entropy_multi_candidates`
- `_prepare_multi_candidate_labels()`: Creates 2D labels tensor
  `[seq_len, max_candidates]` from `position_rewards`
- `_compute_logprobs_and_loss()`: Prepares multi-candidate labels and passes them to
  loss function
- Supports vocab-parallel logprob gathering with TP (tensor parallelism)
- Handles Ulysses sequence parallelism for multi-candidate tensors

#### `training/` — Distillation Training

| File                   | Purpose                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `training/loss.py`     | `grpo_distill_loss_fn()` — combined GRPO + position-level distillation loss          |
| `training/actor.py`    | `patch_ppo_actor_class_to_use_distill_loss()` — patches PPOActor to use distill loss |
| `training/logprobs.py` | `gather_logprobs_entropy_multi_candidates()` — multi-candidate logprob gathering     |

**`grpo_distill_loss_fn` computes:**

1. Standard GRPO loss using chosen token logprobs
1. Position-level GRPO loss using multi-candidate logprobs:
   - Per-position reward normalization (zero-mean, unit-variance)
   - Importance-weighted log probability: `-E[iw * reward * logp]`
   - Uses old logprobs from rollout for off-policy importance weighting
1. Combined loss:
   `rl_loss_weight * grpo_loss + distill_loss_weight * position_grpo_loss`

## Data Flow

### Cache-Aware Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CacheAwarePPOTrainer.train()                          │
│                                                                          │
│  If loss_mode != GRPO:                                                   │
│   ├─ patch_ppo_actor_class_to_use_distill_loss()                        │
│   ├─ super().train()  (standard training loop)                          │
│   └─ unpatch_ppo_actor_distill_loss()  (in finally)                     │
│  Otherwise:                                                              │
│   └─ super().train()  (standard training loop)                          │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  │  per training step
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TreeSearchGroupedRolloutWorkflow.arun_episode()                         │
│                                                                          │
│  1. CHECK CACHE                                                          │
│     ├─ query_id = data.get("query_id", "")                               │
│     ├─ cached_count = tree_store.get_untrained_count(query_id)           │
│     └─ need_gen = max(0, group_size - cached_count)                      │
│                                                                          │
│  2. GENERATE FRESH EPISODES (if need_gen > 0)                            │
│     ├─ Run need_gen parallel rollouts via asyncio.gather                 │
│     ├─ Retry failed episodes up to max_retries                           │
│     └─ Convert results to Nodes via interactions_dict_to_nodes()        │
│                                                                          │
│  3. LOAD CACHED NODES (if cached_count > 0)                              │
│     └─ tree_store.load_trajectories(query_id, cached_count)              │
│                                                                          │
│  4. COMBINE fresh_nodes + cached_nodes                                   │
│                                                                          │
│  5. TREE OPERATIONS                                                      │
│     ├─ tree_store.insert_batch(fresh_nodes)                              │
│     ├─ tree_advantage_computer.compute(all_nodes)  (TREE mode)          │
│     ├─ Mark all nodes as trained                                         │
│     └─ Save checkpoint (CROSS_TRAINING mode)                             │
│                                                                          │
│  6. CONVERT TO TENSOR DICT                                               │
│     ├─ _nodes_to_batched_tensor_dict(all_nodes)                          │
│     └─ Inject distill weights if loss_mode != GRPO                       │
│                                                                          │
│  Return: dict[str, torch.Tensor]  (batched tensor dict)                  │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Standard PPO Training Pipeline (AReaL)                                  │
│                                                                          │
│  ├─ compute_advantages()  (GAE or TREE)                                  │
│  ├─ ppo_update()                                                         │
│  │   └─ If loss_mode != GRPO: uses grpo_distill_loss_fn()               │
│  └─ Standard logging and checkpointing                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Metadata Propagation

Key metadata fields attached to trajectory dicts throughout the pipeline:

| Field      | Attached by                               | Type  | Used by                                         |
| ---------- | ----------------------------------------- | ----- | ----------------------------------------------- |
| `query_id` | `TreeSearchGroupedRolloutWorkflow`        | `str` | Tree lookup, cache splitting, advantage compute |
| `node_id`  | `insert_batch()` / inference engine       | `str` | Advantage lookup, mark trained                  |

## Public API

### Direct Imports

```python
from customized_areal.tree_search import (
    CacheAwarePPOTrainer,
    MCTSTreeStore,
    Node,
    TreeBackupConfig,
    RolloutCacheConfig,
    CacheMode,
    AdvantageMode,
    LossMode,
    TreeAdvantageComputer,
    TreeCheckpointManager,
    TreeSearchGroupedRolloutWorkflow,
    PositionRewardInfo,
    InteractionWithTokenLevelReward,
)
```

### Lazy Imports

The following are available via `__getattr__` for reduced import overhead:

```python
from customized_areal.tree_search import (
    OnPolicyDistillConfig,      # from core.config
    OnPolicyDistillAgent,       # from core.agent
    TeacherConfig,              # from core.teacher_client
    TeacherClient,              # from core.teacher_client
    MultiCandidateFSDPEngine,   # from engine
    MultiCandidateFSDPPPOActor, # from engine
    grpo_distill_loss_fn,       # from training.loss
    gather_logprobs_entropy_multi_candidates,  # from training.logprobs
    _compute_token_rewards,     # from core.reward_compute
)
```

## Usage Example

```python
from customized_areal.tree_search.config import (
    AdvantageMode,
    LossMode,
    RolloutCacheConfig,
    TreeBackupConfig,
    CacheMode,
)
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

cache_config = RolloutCacheConfig(
    cache_dir="/path/to/tree_cache",
    enabled=True,
    n_samples=8,
)

tree_backup_config = TreeBackupConfig(
    mode=CacheMode.CROSS_TRAINING,
    checkpoint_dir="/path/to/tree_cache",
    advantage_mode=AdvantageMode.TREE,
    loss_mode=LossMode.GRPO,
)

with CacheAwarePPOTrainer(
    config,
    cache_config=cache_config,
    tree_backup_config=tree_backup_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
) as trainer:
    trainer.train(
        workflow=config.workflow,
        eval_workflow=config.eval_workflow,
        workflow_kwargs=workflow_kwargs,
        eval_workflow_kwargs=eval_workflow_kwargs,
    )
```

## Known Issues

See [BUGS.md](BUGS.md) for a list of known bugs and their recommended fixes.

## File Index

| File                          | Purpose                                                                            |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `__init__.py`                 | Public API exports and lazy imports for distillation components                    |
| `config.py`                   | `TreeBackupConfig`, `RolloutCacheConfig`, `CacheMode`, `AdvantageMode`, `LossMode` |
| `mcts_tree_store.py`          | `MCTSTreeStore`, `Node` — flat trajectory store with MCTS statistics               |
| `advantage.py`                | `TreeAdvantageComputer` — GRPO-normalized tree Q-value advantages                  |
| `checkpoint.py`               | `TreeCheckpointManager` — serialize/deserialize tree state to JSON                 |
| `trainer.py`                  | `CacheAwarePPOTrainer` — PPO trainer with distillation engine support              |
| `tree_search_grouped_workflow.py` | `TreeSearchGroupedRolloutWorkflow` — core workflow with cache reuse + tree ops |
| `distill_types.py`            | `PositionRewardInfo`, `InteractionWithTokenLevelReward`                            |
| `core/config.py`              | `OnPolicyDistillConfig`, `AgentConfig`                                             |
| `core/agent.py`               | `OnPolicyDistillAgent` — agent for distillation training                           |
| `core/reward_compute.py`      | Student vs teacher logprob reward computation                                      |
| `core/teacher_client.py`      | `TeacherClient` — async teacher model inference client                             |
| `engine/fsdp_engine.py`       | `MultiCandidateFSDPEngine` — multi-candidate logprob gathering                     |
| `engine/actor.py`             | `MultiCandidateFSDPPPOActor` — PPO actor for distillation                          |
| `training/loss.py`            | `grpo_distill_loss_fn` — combined GRPO + distillation loss                         |
| `training/actor.py`           | Patch to use distillation loss in PPOActor                                         |
| `training/logprobs.py`        | Multi-candidate logprob/entropy gathering utilities                                |

# Tree Search: MCTS Tree Backup for PPO Training

This module replaces GAE advantage computation with MCTS tree backup Q-values, enabling
rollout caching across training steps. It also supports on-policy distillation with a
teacher model. It is a customization layer on top of AReaL's `PPOTrainer`.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    CacheAwarePPOTrainer                          │
│  (extends PPOTrainer with rollout caching + tree backup)         │
│                                                                  │
│  __init__()                                                      │
│   ├─ Creates MCTSTreeStore, TreeAdvantageComputer                │
│   ├─ Creates TreeCheckpointManager                               │
│   ├─ On CROSS_TRAINING: loads existing checkpoint                │
│   ├─ Creates TreeSearchPatches (not applied yet)                 │
│   └─ Creates _CacheAwareBatchBuilder                             │
│                                                                  │
│  train()                                                         │
│   ├─ patches.apply()  (monkey-patch all components)              │
│   ├─ monkey-patches self.actor.prepare_batch                     │
│   ├─ super().train()  (runs training loop)                       │
│   └─ patches.restore()  (in finally block)                       │
│                                                                  │
│  _cache_aware_prepare_batch()  ← replaces prepare_batch          │
│       ├─ split_prompts()          → cached / need-generation     │
│       ├─ load_cached_trajectories() from tree store              │
│       ├─ rollout_batch()           → generate missing only       │
│       ├─ tree_store.insert_batch() → store trajectories          │
│       ├─ tree_advantage_computer.compute() → stash tree adv      │
│       ├─ _mark_batch_trained()     → mark as used                │
│       └─ tree_checkpoint_manager.save() (CROSS_TRAINING)         │
│                                                                  │
│   └─ [patched] PPOActor.compute_advantages()                     │
│       ├─ original GAE pipeline (KL, scaling, normalization)      │
│       └─ restore tree advantages from _tree_advantages/_returns  │
│                                                                  │
│  _save_recover_checkpoint()                                      │
│   ├─ super()._save_recover_checkpoint()                          │
│   └─ TreeCheckpointManager.save() + save_trained_episodes()      │
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
| `node_id`        | `int`                       | Unique sequence ID (assigned by store)          |
| `parent_node_id` | `int \| None`               | Parent sequence ID (None for root)              |
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

**Node ID assignment** (`_insert_single`): Each Node receives a globally unique
monotonic `node_id`. The Node's `query_id` is set during insertion.

### 3. Advantage Computer (`advantage.py`)

`TreeAdvantageComputer` replaces GAE advantages with normalized MCTS Q-values.

```
tree_advantage_computer.compute(trajectories)
```

For each trajectory:

1. Collect all `(query_id, node_id)` pairs across the batch
1. **Per-query GRPO normalization of Q-values** for advantages: normalize Q-values to
   zero-mean unit-variance within each query group (so episodes for the same prompt are
   compared against each other)
1. **Per-query GRPO normalization of outcome_rewards** for returns: normalize rewards
   similarly
1. For each trajectory, compute per-token advantages: normalized Q-value × prompt_mask
   (value on response tokens, 0 on prompt tokens)
1. Set `node.advantages` and `node.returns` in-place

Handles Node objects directly, setting attributes on the Node dataclass.

### 4. Checkpoint Manager (`checkpoint.py`)

Serializes/deserializes the full MCTS tree state to disk.

| Method                              | Description                                                                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `save(tree_store)`                  | Save per-query trajectory records as `query_{id}.json` + `metadata.json` (seq_id indices, MCTS stats, trained flags, rewards) |
| `load()`                            | Restore `MCTSTreeStore` from disk. No rebuild needed — stats keyed by int seq_id.                                             |
| `exists()`                          | Check if a checkpoint directory exists                                                                                        |
| `save_trained_episodes(dir, store)` | Save trained episode IDs to recover checkpoint directory                                                                      |
| `load_trained_episodes(dir)`        | Load trained episode IDs from recover checkpoint directory                                                                    |

### 5. Patch Manager (`patches.py`)

`TreeSearchPatches` is a consolidated monkey-patch manager that handles all patches
needed for tree search training. It provides atomic apply/restore and a context manager
protocol.

**Patches applied:**

1. **PPOActor.compute_advantages** — restores tree advantages after GAE runs
1. **engine.\_wrap_openai_agent** — returns `TreeSearchGroupedRolloutWorkflow`
1. **engine.\_resolve_workflow** — prevents double-wrapping with
   `GroupedRolloutWorkflow`
1. **engine.workflow_executor** — replaces with `TreeSearchWorkflowExecutor`
1. **PPOActor.\_ppo_update** (conditional) — uses distillation loss when loss_mode is
   DISTILL or BOTH

**Usage:**

```python
patches = TreeSearchPatches(
    rollout_engine=engine,
    advantage_mode=AdvantageMode.TREE,
    loss_mode=LossMode.GRPO,
    group_size=4,
)
patches.apply()
try:
    ...  # training loop
finally:
    patches.restore()
```

### 6. Trainer (`trainer.py`)

#### `CacheAwarePPOTrainer`

PPO trainer with rollout caching **and** tree backup. Extends `PPOTrainer` directly.

**Initialization (`__init__`):**

When `cache_config.enabled` and `tree_backup_config.mode != OFF`:

1. Creates `MCTSTreeStore`, `TreeAdvantageComputer`, `TreeCheckpointManager`
1. On `CROSS_TRAINING` mode, loads existing tree checkpoint if available
1. Restores trained flags from recover checkpoint, or resets for fresh run
1. Creates `_CacheAwareBatchBuilder` for prompt splitting
1. Creates `TreeSearchPatches` (not applied yet)
1. If `loss_mode != GRPO`, overrides `_create_train_engine` to use
   `MultiCandidateFSDPPPOActor`

**Training flow — per step:**

1. **Split prompts** into cached / needs-generation via
   `_CacheAwareBatchBuilder.split_prompts()`
   - Query ID derived from `prompt["query_id"]` (dataset-provided)
   - If all prompts have enough cached (untrained) trajectories → load from cache
   - Otherwise → generate all prompts fresh via `rollout_batch()`
1. **Tree insert** via `tree_store.insert_batch(trajs)` — while `query_id`/`node_id` are
   available
1. **Tree advantage compute** (TREE mode) via `tree_advantage_computer.compute(trajs)`
   - Stashes results as `node.advantages` / `node.returns`
1. **Mark trained** via `_mark_batch_trained()` — so rollouts aren't reused
1. **Save checkpoint** (CROSS_TRAINING mode)
1. **Convert to tensor dicts** for the downstream PPO pipeline
1. **Distillation weights** injected if loss_mode is DISTILL or BOTH
1. **GAE runs** via patched `compute_advantages()` — GAE computes normally, then tree
   advantages are restored
1. **PPO update** uses tree advantages (TREE mode) or GAE advantages (GAE mode)

**Key methods:**

| Method                         | Description                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------- |
| `train()`                      | Applies patches, monkey-patches prepare_batch, runs training, restores in finally |
| `_cache_aware_prepare_batch()` | Cache-aware batch preparation with tree operations                                |
| `_save_recover_checkpoint()`   | Saves MCTS tree checkpoint + trained episodes on CROSS_TRAINING mode              |
| `close()`                      | Safety net: restores patches if train() crashed before finally                    |

**Custom train engine:** When `loss_mode != GRPO`, `_create_train_engine` returns
`MultiCandidateFSDPPPOActor` instead of the standard actor, enabling multi-candidate
logprob gathering for distillation.

### 7. Grouped Rollout Workflow (`grouped_workflow.py`)

`TreeSearchGroupedRolloutWorkflow` extends `GroupedRolloutWorkflow` to run multiple
rollouts per query and collect all per-turn `Node` objects into a flat `list[Node]`.

- `arun_episode()`: runs `group_size` parallel rollouts via `asyncio.gather`, tags each
  Node with `episode_id` and `query_id`, returns flat `list[Node]`

### 8. Proxy Workflow (`proxy_workflow.py`)

`QueryIDProxyWorkflow` extends `OpenAIProxyWorkflow` to inject dataset `query_id` into
trajectories and convert `InteractionWithTokenLogpReward` objects to `list[Node]`.

- `_interactions_to_nodes()`: converts interaction dict to Node objects, reconstructing
  full-sequence `logprobs`/`loss_mask`/`versions` by concatenating parent context with
  new response tokens
- `arun_episode()`: calls parent, converts result to `list[Node]` with `query_id`
  injected
- Supports `agent_path` kwarg for dotted import path resolution

### 9. Workflow Executor (`workflow_executor.py`)

`TreeSearchWorkflowExecutor` extends `WorkflowExecutor` to handle `list[dict]` returns
from `arun_episode` (the standard executor expects a single dict).

- `_create_workflow_task()`: wraps workflow execution, handles three return types
  (list\[dict\], dict, InteractionWithTokenLogpReward dict), applies acceptance
  filtering
- `wait()`: extracts `_TreeSearchRolloutResult` trajectories, flattening into a single
  list
- `rollout_batch()`: submits all items and returns flattened results

### 10. Distillation Support

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
│                        CacheAwarePPOTrainer.train()                      │
│                                                                          │
│  patches.apply()  ← TreeSearchPatches manages all monkey-patches        │
│  monkey-patches self.actor.prepare_batch → _cache_aware_prepare_batch() │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  │  per training step
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  1. BATCH PREPARATION  (_cache_aware_prepare_batch)                      │
│                                                                          │
│  ┌──────────────┐                                                       │
│  │  Dataloader   │  raw prompts (list[dict])                             │
│  └──────┬───────┘                                                       │
│         │                                                                │
│  ┌──────▼─────────────────────────────────┐                              │
│  │  _CacheAwareBatchBuilder.split_prompts │                              │
│  │                                        │                              │
│  │  For each prompt:                      │
│  │   • prompt["query_id"]                 │
│  │   • tree_store.get_untrained_count()   │── check cached rollouts      │
│  │                                        │                              │
│  │   fully cached (≥ n_samples)  ──────► │  cached_items[]               │
│  │   not fully cached            ──────► │  need_gen_items[]             │
│  └──┬──────────────────────────────────┬─┘                              │
│     │                                   │                                │
│     ▼                                   ▼                                │
│  ┌──────────────────────┐  ┌──────────────────────┐                     │
│  │ load_cached_trajs()  │  │  rollout_batch()     │                     │
│  │                      │  │  (inference engine)   │                     │
│  │ tree_store.load_     │  │                      │                     │
│  │  trajectories(qid,n) │  │ TreeSearchGrouped    │                     │
│  │                      │  │ RolloutWorkflow runs │                     │
│  │ Returns list[Node]   │  │ group_size episodes  │                     │
│  └──────┬───────────────┘  │                      │                     │
│         │                  │ Returns flat list    │                     │
│         │                  │ of per-episode dicts │                     │
│         │                  └──────────┬───────────┘                     │
│         │                             │                                 │
│         └──────────┬──────────────────┘                                 │
│                    ▼                                                    │
│  ┌──────────────────────────────────────┐                               │
│  │  Tree Operations                     │                               │
│  │                                      │                               │
│  │  1. tree_store.insert_batch(trajs)   │                               │
│  │     For each trajectory:             │                               │
│  │      • Skip if node_id already set   │                               │
│  │      • Assign global node_id         │                               │
│  │      • Store Node record            │                               │
│  │      • MCTS backup (Q = reward)     │                               │
│  │                                      │                               │
│  │  2. tree_advantage_computer.compute  │                               │
│  │     (TREE mode)                      │                               │
│  │     • Per-query GRPO normalize adv   │                               │
│  │     • Per-query GRPO normalize ret   │                               │
│  │     • Set node.advantages/returns    │                               │
│  │                                      │                               │
│  │  3. _mark_batch_trained()            │                               │
│  │                                      │                               │
│  │  4. Save checkpoint (CROSS_TRAINING) │                               │
│  └──────────────────┬───────────────────┘                               │
│                     │                                                    │
│  ┌──────────────────▼───────────────────┐                               │
│  │  Convert to tensor dicts             │                               │
│  │  • Node → _node_to_tensor_dict()     │                               │
│  │                                      │                               │
│  │  Inject distillation loss weights:   │                               │
│  │  • DISTILL: rl_loss_weight = 0.0     │                               │
│  │  • BOTH:    rl_loss_weight = config  │                               │
│  └──────────────────┬───────────────────┘                               │
│                     │                                                    │
└─────────────────────┼────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. ADVANTAGE COMPUTATION  (patched compute_advantages)                  │
│                                                                          │
│  ┌─────────────────────────────────────┐                                 │
│  │  Step A: Original GAE pipeline      │                                 │
│  │  • Compute KL rewards              │                                 │
│  │  • Scale rewards                   │                                 │
│  │  • GAE λ-returns → advantages      │                                 │
│  │  • Compute loss_mask, logprobs     │                                 │
│  │                                     │                                 │
│  │  Preserved for logging: kl_rewards, │                                 │
│  │  tot_rewards, loss_mask, logprobs   │                                 │
│  └──────────────────────┬──────────────┘                                 │
│                         │                                                │
│  ┌──────────────────────▼──────────────┐                                 │
│  │  Step B: Restore tree advantages    │                                 │
│  │  (TREE mode only)                   │                                 │
│  │                                     │                                 │
│  │  For each trajectory:               │                                 │
│  │   Pop node.advantages → advantages  │                                 │
│  │   Pop node.returns → returns        │                                 │
│  └─────────────────────────────────────┘                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. PPO UPDATE                                                           │
│                                                                          │
│  • GRPO loss: clipped loss on advantages (tree Q-values in TREE mode)   │
│  • Value loss on returns                                                 │
│  • Distillation loss (DISTILL/BOTH modes):                               │
│    position-level GRPO using teacher logprobs                            │
│  • KL metadata available for logging                                     │
└──────────────────────────────────────────────────────────────────────────┘

After PPO update (CROSS_TRAINING mode):

┌──────────────────────────────────────────────┐
│  _save_recover_checkpoint()                  │
│  ├─ super()._save_recover_checkpoint()       │
│  ├─ TreeCheckpointManager.save(tree_store)   │
│  │   • Each query → query_{id}.json           │
│  │   • metadata.json (seq_id indices,         │
│  │     MCTS stats, trained flags)             │
│  └─ TreeCheckpointManager.save_trained_ep... │
│     • trained_episodes.json (episode IDs)    │
└──────────────────────────────────────────────┘
```

### Metadata Propagation

Key metadata fields attached to trajectory dicts throughout the pipeline:

| Field      | Attached by                               | Type  | Used by                                         |
| ---------- | ----------------------------------------- | ----- | ----------------------------------------------- |
| `query_id` | `insert_batch()` / `QueryIDProxyWorkflow` | `str` | Tree lookup, cache splitting, advantage compute |
| `node_id`  | `insert_batch()`                          | `int` | Advantage lookup, mark trained                  |

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
    QueryIDProxyWorkflow,
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

## File Index

| File                     | Purpose                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `__init__.py`            | Public API exports and lazy imports for distillation components                    |
| `config.py`              | `TreeBackupConfig`, `RolloutCacheConfig`, `CacheMode`, `AdvantageMode`, `LossMode` |
| `mcts_tree_store.py`     | `MCTSTreeStore`, `Node` — flat trajectory store with MCTS statistics               |
| `advantage.py`           | `TreeAdvantageComputer` — GRPO-normalized tree Q-value advantages                  |
| `checkpoint.py`          | `TreeCheckpointManager` — serialize/deserialize tree state to JSON                 |
| `trainer.py`             | `CacheAwarePPOTrainer` — PPO trainer with rollout caching + tree backup            |
| `patches.py`             | `TreeSearchPatches` — consolidated monkey-patch manager                            |
| `grouped_workflow.py`    | `TreeSearchGroupedRolloutWorkflow` — runs group_size episodes per query            |
| `proxy_workflow.py`      | `QueryIDProxyWorkflow` — injects query_id, converts interactions to Nodes          |
| `workflow_executor.py`   | `TreeSearchWorkflowExecutor` — handles list\[dict\] returns from arun_episode      |
| `distill_types.py`       | `PositionRewardInfo`, `InteractionWithTokenLevelReward`                            |
| `core/config.py`         | `OnPolicyDistillConfig`, `AgentConfig`                                             |
| `core/agent.py`          | `OnPolicyDistillAgent` — agent for distillation training                           |
| `core/reward_compute.py` | Student vs teacher logprob reward computation                                      |
| `core/teacher_client.py` | `TeacherClient` — async teacher model inference client                             |
| `engine/fsdp_engine.py`  | `MultiCandidateFSDPEngine` — multi-candidate logprob gathering                     |
| `engine/actor.py`        | `MultiCandidateFSDPPPOActor` — PPO actor for distillation                          |
| `training/loss.py`       | `grpo_distill_loss_fn` — combined GRPO + distillation loss                         |
| `training/actor.py`      | Patch to use distillation loss in PPOActor                                         |
| `training/logprobs.py`   | Multi-candidate logprob/entropy gathering utilities                                |

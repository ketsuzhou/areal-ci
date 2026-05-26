# Tree Search: MCTS Tree Backup for PPO Training

This module replaces GAE advantage computation with MCTS tree backup Q-values, enabling
rollout caching across training steps. It also supports on-policy distillation with a
teacher model. It is a customization layer on top of AReaL's `PPOTrainer`.

## Two "Tree" Concepts

This system uses **two distinct tree concepts** that work together:

1. **MCTS Tree Search** (`tree_search/`): Organizes rollouts into a tree structure for
   caching, advantage computation, and episode management. Each `Node` represents one
   turn with parent-child relationships.

1. **Tree Attention (Trie Packing)** (`areal/models/tree_attn/`): Packs sequences with
   shared prefixes into a compressed trie (`TrieNode`) for efficient attention
   computation during training. Shared prefix tokens are computed only once,
   dramatically reducing training cost.

These two trees are **unrelated data structures** that operate at different layers:

- **MCTS Tree** is a logical structure for RL (episodes, turns, rewards)
- **Trie** is a physical packing structure for efficient transformer attention

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
│   ├─ Generates only needed fresh episodes (with retry support)   │
│   ├─ Converts fresh results to Nodes via _result_to_nodes()      │
│   ├─ Loads cached episode Nodes from MCTSTreeStore               │
│   ├─ Resets versions to 0 on cached Nodes (decoupled PPO)        │
│   ├─ Distillation (if loss_mode != GRPO):                        │
│   │   ├─ Diagnoses episodes to find turns needing improvement    │
│   │   ├─ Gets teacher logprobs for selected turns                │
│   │   └─ Applies to fresh and cached node groups separately      │
│   ├─ Combines fresh + cached Nodes                               │
│   ├─ Inserts fresh Nodes into tree_store                         │
│   ├─ Computes tree advantages (TREE mode)                        │
│   ├─ Marks all nodes as trained                                  │
│   ├─ Saves tree checkpoint (CROSS_TRAINING mode)                 │
│   └─ Converts to batched tensor dict + injects distill weights   │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                         MCTSTreeStore                            │
│  (flat trajectory store with MCTS statistics)                    │
│                                                                  │
│  insert_batch() → store trajectories                             │
│  load_untrained_episodes() → retrieve untrained Nodes             │
│  get_untrained_episode_count() → check cache availability         │
│  set_trained() / is_trained() → track usage                      │
│  _backup() → update MCTS Q-values                                │
└──────────────────────────────────────────────────────────────────┘
```

## Component Reference

### 1. Config (`config.py`)

Dataclasses controlling tree backup, caching, and advantage computation.

| Class                | Field                     | Type            | Default                   | Description                                        |
| -------------------- | ------------------------- | --------------- | ------------------------- | -------------------------------------------------- |
| `TreeBackupConfig`   | `mode`                    | `CacheMode`     | `OFF`                     | Controls when/how tree backup activates            |
|                      | `enabled`                 | `bool`          | `True`                    | Enable/disable tree backup                         |
|                      | `checkpoint_dir`          | `str`           | `""`                      | Directory for MCTS tree checkpoints                |
|                      | `advantage_mode`          | `AdvantageMode` | `TREE`                    | TREE (Q-values) or GAE advantages                  |
|                      | `loss_mode`               | `LossMode`      | `GRPO`                    | GRPO, DISTILL, or BOTH                             |
|                      | `max_reasoning_tokens`    | `int`           | `1000`                    | Max tokens for reasoning                           |
|                      | `rl_loss_weight`          | `float`         | `1.0`                     | Weight for RL loss in BOTH mode                    |
|                      | `distill_loss_weight`     | `float`         | `0.005`                   | Weight for distillation loss                       |
|                      | `reward_bias`             | `float`         | `0.0`                     | Bias added to outcome rewards                      |
|                      | `reward_scaling`          | `float`         | `1.0`                     | Scaling factor for outcome rewards                 |
|                      | `reward_clip`             | `float`         | `20.0`                    | Reward clipping threshold                          |
|                      | `overlong_reward_penalty` | `bool`          | `False`                   | Apply penalty for overlong episodes                |
|                      | `overlong_tokens`         | `int \| None`   | `None`                    | Token threshold for overlong penalty               |
|                      | `overlong_penalty_factor` | `float \| None` | `None`                    | Penalty factor for overlong episodes               |
|                      | `topk_distill`            | `bool`          | `False`                   | Use top-k distillation                             |
|                      | `teacher_provider`        | `str`           | `"external"`              | Teacher provider type (`"external"` or `"engine"`) |
|                      | `teacher_base_url`        | `str`           | `"http://localhost:8001"` | Teacher API endpoint                               |
|                      | `teacher_model_name`      | `str`           | `""`                      | Teacher model identifier                           |
|                      | `teacher_top_k`           | `int`           | `10`                      | Top-k tokens from teacher                          |
|                      | `teacher_max_retries`     | `int`           | `3`                       | Max retries for teacher requests                   |
|                      | `teacher_timeout`         | `float`         | `60.0`                    | Timeout for teacher requests                       |
|                      | `teacher_missing_logprob` | `float`         | `-23.0`                   | Default logprob for missing teacher tokens         |
|                      | `diagnose_model_name`     | `str`           | `""`                      | Model name for episode diagnosis                   |
|                      | `diagnose_max_tokens`     | `int`           | `1024`                    | Max tokens for diagnosis responses                 |
|                      | `diagnose_temperature`    | `float`         | `0.0`                     | Temperature for diagnosis sampling                 |
|                      | `diagnose_base_url`       | `str`           | `""`                      | Base URL for diagnosis API                         |
|                      | `diagnose_api_key`        | `str`           | `""`                      | API key for diagnosis endpoint                     |
|                      | `strict_distill_json`     | `bool`          | `True`                    | Enforce strict JSON parsing in distillation        |
| `RolloutCacheConfig` | `cache_dir`               | `str`           | `""`                      | Directory for rollout cache                        |
|                      | `enabled`                 | `bool`          | `True`                    | Enable/disable caching                             |
|                      | `n_samples`               | `int`           | `1`                       | Number of rollout samples per prompt               |

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

| Field            | Type                        | Description                                          |
| ---------------- | --------------------------- | ---------------------------------------------------- |
| `input_ids`      | `list[int]`                 | Full token sequence (prompt + response)              |
| `loss_mask`      | `list[int]`                 | 0=prompt tokens, 1=response tokens                   |
| `logprobs`       | `list[float]`               | Per-token log probabilities                          |
| `versions`       | `list[int]`                 | Policy version per token (-1 on prompt)              |
| `node_id`        | `str`                       | Globally unique interaction ID (UUID)                |
| `parent_node_id` | `str \| None`               | Parent interaction ID (None for root)                |
| `episode_id`     | `str`                       | Groups turns into a trajectory path                  |
| `turn_idx`       | `int`                       | 1-based turn position within episode                 |
| `query_id`       | `str`                       | Dataset query identifier                             |
| `train_id`       | `str`                       | Training run that trained this node ("" = untrained) |
| `outcome_reward` | `float`                     | Trajectory-level reward                              |
| `advantages`     | `torch.Tensor \| None`      | Tree-computed per-token advantages                   |
| `returns`        | `torch.Tensor \| None`      | Tree-computed per-token returns                      |
| `topk_ids`       | `list[list[int]] \| None`   | Top-k candidate token IDs per response position      |
| `topk_logp`      | `list[list[float]] \| None` | Top-k candidate log probabilities                    |
| `distill_reward` | `list[list[float]] \| None` | Per-position distillation rewards                    |
| `teacher_logp`   | `list[list[float]] \| None` | Teacher log probabilities per position               |
| `guidance`       | `dict[int, str] \| None`    | Turn index → guidance text map (on leaf nodes)       |

**Turn boundaries** are derived from `loss_mask` transitions (0→1 = response start, 1→0
= response end) via `_find_turn_boundaries()`, rather than using tokenizer-specific
assistant markers.

#### Store Methods

| Method                                          | Description                                                                |
| ----------------------------------------------- | -------------------------------------------------------------------------- |
| `insert_batch(trajectories)`                    | Insert trajectories (Node objects) from rollout; skip already-cached nodes |
| `get_q_value(node_id)`                          | Raw Q-value (mean reward) for a trajectory                                 |
| `set_trained(node_id)` / `is_trained(node_id)`  | Mark/check whether a single node has been trained                          |
| `get_untrained_count(query_id)`                 | Count untrained nodes for a query                                          |
| `get_untrained_episode_count(query_id)`         | Count untrained episodes for a query (used by workflow)                    |
| `get_untrained_node_ids(query_id, n)`           | Get up to N untrained node IDs                                             |
| `load_untrained_episodes(query_id, n_episodes)` | Load untrained Node objects grouped by episode (used by workflow)          |
| `load_trajectories(query_id, n_samples)`        | Load untrained Node objects by sample count                                |
| `reset_trained_flags()`                         | Reset all trained flags (for fresh training run)                           |
| `mark_episodes_trained(episode_ids)`            | Mark trained by episode ID set (for recover checkpoint restore)            |
| `clear()`                                       | Reset all state                                                            |
| `set/get_normalized_advantage(node_id)`         | Store/retrieve GRPO-normalized advantage                                   |
| `set/get_normalized_return(node_id)`            | Store/retrieve GRPO-normalized return                                      |

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

| Method                              | Description                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `save(tree_store)`                  | Save self-contained per-query trajectory records as `query_{sanitized_id}.json` files with per-query metadata |
| `load()`                            | Restore `MCTSTreeStore` from disk. No rebuild needed — stats keyed by string node_id.                         |
| `exists()`                          | Check if a checkpoint directory exists                                                                        |
| `save_trained_episodes(dir, store)` | Save trained episode IDs to recover checkpoint directory                                                      |
| `load_trained_episodes(dir)`        | Load trained episode IDs from recover checkpoint directory                                                    |

### 5. Tree Search Grouped Rollout Workflow (`tree_search_grouped_workflow.py`)

`TreeSearchGroupedRolloutWorkflow` is the core component that extends `RolloutWorkflow`
to provide tree-search-aware rollout with cache reuse.

**Initialization (`__init__`):**

Accepts the full set of configuration parameters (see `TreeBackupConfig` above), plus:

| Parameter        | Type              | Description                                      |
| ---------------- | ----------------- | ------------------------------------------------ |
| `workflow`       | `RolloutWorkflow` | Base workflow for episode generation             |
| `group_size`     | `int`             | Number of episodes per query (must be >= 1)      |
| `checkpoint_dir` | `str`             | Directory for tree checkpoint persistence        |
| `advantage_mode` | `AdvantageMode`   | TREE or GAE advantage computation                |
| `loss_mode`      | `LossMode`        | GRPO, DISTILL, or BOTH                           |
| `cache_mode`     | `CacheMode`       | OFF, IN_TRAINING, or CROSS_TRAINING              |
| `tokenizer_path` | `str`             | Path to HF tokenizer (required for distillation) |
| ...              | ...               | All `TreeBackupConfig` fields (see config table) |

- Creates `TreeCheckpointManager` and `MCTSTreeStore`
- On `CROSS_TRAINING` mode, loads existing tree checkpoint if available
- Creates `TreeAdvantageComputer`

**Per-episode flow (`arun_episode`):**

1. **Check cache**: Count untrained episodes for the query via
   `tree_store.get_untrained_episode_count()`
1. **Generate fresh episodes** if needed: Run `group_size - cached_count` parallel
   rollouts via `asyncio.gather`, with retry support (`_retry_episode`)
1. **Convert results to Nodes**: `_result_to_nodes()` converts each arun_episode result
   (dict or list of `InteractionWithTokenLogpReward`) to `list[Node]`, assigning
   `episode_id`, `query_id`, and `turn_idx`
1. **Load cached nodes**: `tree_store.load_untrained_episodes(query_id, cached_count)`.
   Reset `versions` to `0` on response tokens so decoupled PPO treats them as current
   behavior policy rollouts
1. **Teacher model reward computation** (if `loss_mode != GRPO`):
   - Load tokenizer from `tokenizer_path`
   - Build teacher provider (external API or engine-based) via
     `_build_teacher_provider()`
   - For each episode group, diagnose to find turns needing improvement
     (`_prepare_distill_for_episode()` → `provider.diagnose_episode()`)
   - For selected turns, get teacher logprobs for candidate tokens
     (`selected_turn_to_position_rewards()`)
   - Build `PositionRewardInfo` with `candidate_token_ids`, `teacher_logprobs`, and
     `rewards`
   - Store distillation data in `node.distill_reward` and `node.teacher_logp`
   - Also store diagnosis guidance in `node.guidance` on leaf nodes
   - Applied separately to fresh and cached node groups via
     `_prepare_distill_for_node_groups()`
   - In `DISTILL` mode, episodes with no diagnosis or no selected turns are filtered out
1. **Combine**: Merge fresh and cached nodes (total = group_size)
1. **Insert fresh nodes**: `tree_store.insert_batch(fresh_nodes)`
1. **Compute tree advantages**: `tree_advantage_computer.compute(all_nodes)` (TREE mode)
1. **Mark trained**: `tree_store.set_trained(node.node_id, True)` for all nodes
1. **Save checkpoint**: `tree_checkpoint_manager.save()` (CROSS_TRAINING mode)
1. **Convert to tensor dict**: `_nodes_to_batched_tensor_dict()` converts `list[Node]`
   to batched tensor dict
1. **Inject distill weights**: Set `rl_loss_weight`, `distill_loss_weight`, and
   `position_rewards` (via `_set_position_reward_sample_indices()`) if loss_mode != GRPO

**Utility functions:**

| Function                                | Description                                                                                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `interactions_dict_to_nodes()`          | Convert `dict[str, InteractionWithTokenLogpReward]` to `list[Node]` (also handles proxy-deserialized data where `model_response` is None) |
| `_result_to_nodes()`                    | Convert a single arun_episode result (dict or list) to `list[Node]` with episode metadata                                                 |
| `_nodes_to_batched_tensor_dict()`       | Convert `list[Node]` to batched tensor dict via `concat_padded_tensors`                                                                   |
| `_retry_episode()`                      | Retry a failed episode with exponential backoff (up to 1 retry)                                                                           |
| `_prepare_distill_for_episode()`        | Diagnose one episode and compute position-level teacher rewards                                                                           |
| `_prepare_distill_for_node_groups()`    | Apply distillation to multiple episode groups with error handling                                                                         |
| `_group_nodes_by_episode()`             | Group a flat list of Nodes by `episode_id`                                                                                                |
| `_filter_distill_episode_failure()`     | In DISTILL mode, return empty list on failure (drop episode); otherwise return nodes unchanged                                            |
| `_set_position_reward_sample_indices()` | Assign `sample_index` to each `PositionRewardInfo` based on node position in batch                                                        |

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

- When `loss_mode != GRPO`: Applies distill loss patch, calls `super().train()`,
  restores patch in `finally`
- Otherwise: Delegates to `super().train()`

### 7. Tree Attention (Trie Packing)

The `areal/models/tree_attn/` module provides efficient attention for sequences with
shared prefixes. This is **independent** of the MCTS tree but is used during training to
compute attention efficiently.

#### How It Works

1. **Build Trie**: `build_packed_tree_batch()` in `tree.py` takes multiple sequences and
   builds a compressed trie (`TrieNode`) where sequences with shared prefixes share
   nodes.

1. **Pack Inputs**: Sequences are packed into a single tensor where shared prefix tokens
   appear only once. Each token's position IDs are computed from the tree structure.

1. **Tree Attention Mask**: A custom attention mask is built where each token can only
   attend to its ancestors in the trie (causal + tree structure). This is represented
   as:

   - `tree_triton_data` for Triton kernel (fast path)
   - `tree_block_mask` (BlockMask) for PyTorch flex attention

1. **Forward Pass**: In `FSDPEngine.forward_backward_batch()`, tree attention kwargs are
   injected into model inputs:

   ```python
   tree_kwargs = build_tree_attn_kwargs(ctx.trie_node, padded_size, self.device)
   inputs.update(tree_kwargs)
   ```

1. **Tree Attention Function**: `_tree_attn_fwd_func()` in `module_fsdp.py` handles the
   actual attention computation:

   - Triton path: Custom kernel with O(1) memory for tree attention
   - Flex Attention path: Uses BlockMask with PyTorch's compiled flex_attention

#### Key Data Structures

| Class/Function                          | Purpose                                                             |
| --------------------------------------- | ------------------------------------------------------------------- |
| `TrieNode`                              | Compressed trie node with token sequences, sequence IDs, children   |
| `build_packed_tree_batch()`             | Main entry point: packs batch into trie structure                   |
| `build_tree_attn_kwargs()`              | Builds kwargs for model forward (selects Triton or Flex)            |
| `build_block_mask_from_trie()`          | Creates BlockMask for flex attention                                |
| `build_triton_attn_data_from_trie()`    | Precomputes Triton kernel data structures                           |
| `gather_packed_tree_logprobs()`         | Computes logprobs respecting tree structure (shared prefix caching) |
| `gather_packed_tree_logprobs_entropy()` | Computes logprobs + entropy for tree-packed sequences               |
| `patch_fsdp_for_tree_training()`        | Monkey-patches FSDP attention to use tree attention                 |

#### Tree Logprob Gathering

For tree-packed sequences, standard rolling of `input_ids` doesn't work because
sequences share prefixes. The `functional.py` module provides tree-aware logprob
computation:

- `_gather_packed_tree_logprobs()`: Computes logprobs for all sequences with
  **node-level caching** (shared prefix logprobs are computed once and reused)
- `_compute_internal_node_logprobs()`: Logprobs within a single trie node
- `_compute_transition_logprob()`: Logprobs for transitions between nodes
- `gather_packed_tree_vocab_stats()`: Vocab min/max logits for tree-packed sequences

### 8. Distillation Support

#### `distill_types.py`

| Class                             | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `PositionRewardInfo`              | Per-position candidate tokens, logprobs, and rewards              |
| `InteractionWithTokenLevelReward` | Extended interaction with `token_rewards` and `token_reward_mask` |

#### `core/` — On-Policy Distillation Core

| File                            | Purpose                                                            |
| ------------------------------- | ------------------------------------------------------------------ |
| `core/config.py`                | `OnPolicyDistillConfig` (extends PPOConfig) and `AgentConfig`      |
| `core/agent.py`                 | `OnPolicyDistillAgent` — agent class for distillation training     |
| `core/reward_compute.py`        | `_compute_token_rewards()` — student vs teacher logprob comparison |
| `core/teacher_client.py`        | `TeacherClient` — async client for remote teacher model inference  |
| `core/teacher_provider.py`      | `TeacherProvider` — abstraction for teacher logprob sources        |
| `core/selected_turn_distill.py` | Diagnoses episodes and builds position-level teacher rewards       |

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
- **Tree training support**: When `enable_tree_training=True`, works with trie-packed
  inputs and tree attention

#### `training/` — Distillation Training

| File                   | Purpose                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `training/loss.py`     | `grpo_distill_loss_fn()` — combined GRPO + position-level distillation loss          |
| `training/actor.py`    | `patch_ppo_actor_class_to_use_distill_loss()` — patches PPOActor to use distill loss |
| `training/logprobs.py` | `gather_logprobs_entropy_multi_candidates()` — multi-candidate logprob gathering     |

**`grpo_distill_loss_fn` computes:**

1. Standard GRPO loss using chosen token logprobs
1. Teacher KL distillation loss from `position_rewards`:
   - For each position with teacher logprobs: `student_logp - teacher_logp`
   - Mean over all positions and candidates
   - Added to the actor loss with weight `distill_loss_weight`
1. Combined loss: `rl_loss_weight * grpo_loss + distill_loss_weight * teacher_kl_loss`

## Data Flow

### Cache-Aware Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CacheAwarePPOTrainer.train()                          │
│                                                                          │
│  If loss_mode != GRPO:                                                   │
│   ├─ patch_ppo_actor_class_to_use_distill_loss()                         │
│   ├─ super().train()  (standard training loop)                           │
│   └─ unpatch_ppo_actor_distill_loss()  (in finally)                      │
│  Otherwise:                                                              │
│   └─ super().train()  (standard training loop)                           │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  │  per training step
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TreeSearchGroupedRolloutWorkflow.arun_episode()                         │
│                                                                          │
│  1. CHECK CACHE                                                          │
│     ├─ query_id = data.get("query_id", "")                               │
│     ├─ cached_count = tree_store.get_untrained_episode_count(query_id)   │
│     └─ need_gen = max(0, group_size - cached_count)                      │
│                                                                          │
│  2. GENERATE FRESH EPISODES (if need_gen > 0)                            │
│     ├─ Run need_gen parallel rollouts via asyncio.gather                 │
│     ├─ Retry failed episodes via _retry_episode()                        │
│     └─ Convert results to Nodes via _result_to_nodes()                   │
│                                                                          │
│  3. LOAD CACHED NODES (if cached_count > 0)                              │
│     ├─ tree_store.load_untrained_episodes(query_id, cached_count)        │
│     └─ Reset versions to 0 on response tokens (decoupled PPO)            │
│                                                                          │
│  4. DISTILLATION (if loss_mode != GRPO)                                  │
│     ├─ Get teacher provider (external API or engine)                     │
│     ├─ Diagnose episodes to find turns needing improvement               │
│     ├─ Get teacher logprobs for selected turns                           │
│     ├─ Build PositionRewardInfo with candidate tokens + teacher logprobs │
│     └─ Applied separately to fresh and cached node groups                │
│                                                                          │
│  5. COMBINE fresh_nodes + cached_nodes                                   │
│                                                                          │
│  6. TREE OPERATIONS                                                      │
│     ├─ tree_store.insert_batch(fresh_nodes)                              │
│     ├─ tree_advantage_computer.compute(all_nodes)  (TREE mode)           │
│     ├─ Mark all nodes as trained via tree_store.set_trained()            │
│     └─ Save checkpoint (CROSS_TRAINING mode)                             │
│                                                                          │
│  7. CONVERT TO TENSOR DICT                                               │
│     ├─ _nodes_to_batched_tensor_dict(all_nodes)                          │
│     └─ Inject distill weights and position_rewards if loss_mode != GRPO  │
│                                                                          │
│  Return: dict[str, torch.Tensor]  (batched tensor dict)                  │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Training Engine (MultiCandidateFSDPEngine)                              │
│                                                                          │
│  ├─ build_packed_tree_batch() → packs sequences into trie                │
│  ├─ forward() with tree attention (TrieNode → tree_block_mask)           │
│  ├─ _compute_logprobs_entropy() → multi-candidate logprobs               │
│  ├─ ppo_update() with grpo_distill_loss_fn()                             │
│  │   ├─ Standard GRPO loss (chosen token)                                │
│  │   └─ Teacher KL loss (all candidates)                                 │
│  └─ Standard logging and checkpointing                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Metadata Propagation

Key metadata fields attached to trajectory dicts throughout the pipeline:

| Field                 | Attached by                         | Type                       | Used by                                         |
| --------------------- | ----------------------------------- | -------------------------- | ----------------------------------------------- |
| `query_id`            | `TreeSearchGroupedRolloutWorkflow`  | `str`                      | Tree lookup, cache splitting, advantage compute |
| `node_id`             | `insert_batch()` / inference engine | `str`                      | Advantage lookup, mark trained                  |
| `position_rewards`    | `TreeSearchGroupedRolloutWorkflow`  | `list[PositionRewardInfo]` | Multi-candidate logprob computation             |
| `distill_loss_weight` | `TreeSearchGroupedRolloutWorkflow`  | `float`                    | Weight for teacher KL loss                      |
| `rl_loss_weight`      | `TreeSearchGroupedRolloutWorkflow`  | `float`                    | Weight for GRPO loss                            |

## How Distillation Works with Tree Attention

When distillation is enabled (`loss_mode` = `DISTILL` or `BOTH`), the system combines
tree attention efficiency with teacher supervision:

1. **Episode Generation**: The workflow generates episodes and stores them as `Node`
   objects in the MCTS tree.

1. **Teacher Diagnosis**: For each episode, a teacher model (or external API) diagnoses
   which turns need improvement and provides guidance.

1. **Selected-Turn Distillation**: For selected turns:

   - Builds teacher prompt with guidance
   - Gets teacher logprobs for candidate tokens at each position
   - Creates `PositionRewardInfo` with `candidate_token_ids`, `teacher_logprobs`, and
     `rewards`

1. **Tree Packing**: During training, sequences are packed into a trie for efficient
   attention. The `TrieNode` structure ensures shared prefixes are computed only once.

1. **Multi-Candidate Logprobs**: The engine computes logprobs for **all candidate
   tokens** (not just the chosen one) at each position using
   `gather_logprobs_entropy_multi_candidates()`. This is necessary for the distillation
   loss which needs logprobs for all candidates.

1. **Tree-Aware Logprob Gathering**: For tree-packed sequences, logprobs are gathered
   respecting the trie structure via `gather_packed_tree_logprobs()` in `functional.py`.
   Shared prefix logprobs are cached and reused across sequences.

1. **Combined Loss**: The loss function combines:

   - **GRPO loss**: Standard policy gradient on chosen tokens
   - **Teacher KL loss**: `mean(student_logp - teacher_logp)` for all candidates,
     weighted by `distill_loss_weight`

1. **Tree Attention in Forward**: During the forward pass, tree attention metadata
   (`tree_triton_data` or `tree_block_mask`) is injected into the model inputs, allowing
   the transformer to attend according to the trie structure.

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
    loss_mode=LossMode.BOTH,  # or DISTILL for distillation only
    distill_loss_weight=0.005,
    teacher_provider="external",
    teacher_base_url="http://localhost:8001",
    teacher_model_name="teacher-model",
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

| File                              | Purpose                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `__init__.py`                     | Public API exports and lazy imports for distillation components                    |
| `config.py`                       | `TreeBackupConfig`, `RolloutCacheConfig`, `CacheMode`, `AdvantageMode`, `LossMode` |
| `mcts_tree_store.py`              | `MCTSTreeStore`, `Node` — flat trajectory store with MCTS statistics               |
| `advantage.py`                    | `TreeAdvantageComputer` — GRPO-normalized tree Q-value advantages                  |
| `checkpoint.py`                   | `TreeCheckpointManager` — serialize/deserialize tree state to JSON                 |
| `trainer.py`                      | `CacheAwarePPOTrainer` — PPO trainer with distillation engine support              |
| `tree_search_grouped_workflow.py` | `TreeSearchGroupedRolloutWorkflow` — core workflow with cache reuse + tree ops     |
| `distill_types.py`                | `PositionRewardInfo`, `InteractionWithTokenLevelReward`                            |
| `core/config.py`                  | `OnPolicyDistillConfig`, `AgentConfig`                                             |
| `core/agent.py`                   | `OnPolicyDistillAgent` — agent for distillation training                           |
| `core/reward_compute.py`          | Student vs teacher logprob reward computation                                      |
| `core/teacher_client.py`          | `TeacherClient` — async teacher model inference client                             |
| `core/teacher_provider.py`        | `TeacherProvider` — abstraction for teacher logprob sources                        |
| `core/selected_turn_distill.py`   | Diagnoses episodes and builds position-level teacher rewards                       |
| `engine/fsdp_engine.py`           | `MultiCandidateFSDPEngine` — multi-candidate logprob gathering                     |
| `engine/actor.py`                 | `MultiCandidateFSDPPPOActor` — PPO actor for distillation                          |
| `training/loss.py`                | `grpo_distill_loss_fn` — combined GRPO + distillation loss                         |
| `training/actor.py`               | Patch to use distillation loss in PPOActor                                         |
| `training/logprobs.py`            | Multi-candidate logprob/entropy gathering utilities                                |

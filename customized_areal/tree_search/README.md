# Tree Search: MCTS Tree Cache, Branching, Backup for PPO Training, and Experience-Guided Distilling

This module replaces GAE advantage computation with MCTS tree backup Q-values, enabling
rollout caching across training steps. It also supports on-policy distillation with a
teacher model, and branch sampling from cached trajectories. It is a customization layer
on top of AReaL's `PPOTrainer`.

## Two "Tree" Concepts

This system uses **two distinct tree concepts** that work together:

1. **MCTS Tree Search** (`tree_search/core/`): Organizes rollouts into a tree structure
   for caching, advantage computation, and episode management. Each `Node` represents
   one turn with parent-child relationships.

1. **Tree Attention (Trie Packing)** (`areal/models/tree_attn/`): Packs sequences with
   shared prefixes into a compressed trie (`TrieNode`) for efficient attention
   computation during training. Shared prefix tokens are computed only once,
   dramatically reducing training cost.

These two trees are **unrelated data structures** that operate at different layers:

- **MCTS Tree** is a logical structure for RL (episodes, turns, rewards)
- **Trie** is a physical packing structure for efficient transformer attention

## Architecture Overview

```mermaid
flowchart TD
    trainer["CustomizedPPOTrainer<br/>extends PPOTrainer with MultiCandidateFSDPPPOActor support"]
    trainer_init["__init__<br/>Accept Config and RolloutCacheConfig<br/>Store tree_search_config"]
    trainer_engine["_create_train_engine<br/>loss_mode != GRPO: MultiCandidateFSDPPPOActor<br/>otherwise: standard PPOTrainer engine"]
    trainer_train["train<br/>patch distill loss when enabled<br/>call super().train()<br/>restore patch in finally"]
    trainer_save["_save_hf / _save_recover_checkpoint<br/>write train_id.json beside checkpoints"]

    workflow["TreeSearchGroupedRolloutWorkflow<br/>extends RolloutWorkflow with cache reuse and tree ops"]
    cache["Check cache<br/>count untrained episodes"]
    fresh["Generate only missing fresh episodes<br/>retry support"]
    source{"SampleSource"}
    scratch["SCRATCH<br/>run fresh episode"]
    branch["BRANCH<br/>select candidate<br/>build branch task<br/>run from branch point"]
    annotate["Annotate fresh Nodes<br/>task_id, entropy_stats, need_branch, sandbox"]
    convert["Convert results to Nodes<br/>_result_to_nodes()"]
    load_cached["Load cached episode Nodes"]
    insert["Insert fresh Nodes"]
    combine["Combine fresh + cached node groups"]
    distill{"loss_mode != GRPO?"}
    distill_steps["Distillation<br/>diagnose episodes<br/>reuse cached guidance<br/>attach teacher logprobs and top-k ids"]
    advantage["Compute tree advantages<br/>TREE mode"]
    mark["Mark nodes as trained"]
    save_query["Save per-query checkpoint<br/>CROSS_TRAINING mode"]
    tensor["Convert to batched tensor dict"]

    store["MCTSTreeStore<br/>flat trajectory store with MCTS statistics"]
    store_insert["insert_batch<br/>store trajectories"]
    store_load["load_untrained_episodes<br/>retrieve untrained Nodes"]
    store_count["get_untrained_episode_count<br/>check cache availability"]
    store_trained["set_trained / is_trained<br/>track usage"]
    store_backup["_backup<br/>update MCTS Q-values"]

    trainer --> trainer_init
    trainer --> trainer_engine
    trainer --> trainer_train
    trainer --> trainer_save
    trainer_train --> workflow

    workflow --> cache --> fresh --> source
    source --> scratch
    source --> branch
    scratch --> annotate
    branch --> annotate
    annotate --> convert --> load_cached --> insert --> combine --> distill
    distill -- yes --> distill_steps --> advantage
    distill -- no --> advantage
    advantage --> mark --> save_query --> tensor

    cache -.-> store_count
    load_cached -.-> store_load
    insert -.-> store_insert
    mark -.-> store_trained
    advantage -.-> store_backup
    store_count --> store
    store_load --> store
    store_insert --> store
    store_trained --> store
    store_backup --> store
```

## Component Reference

### 1. Config (`config.py`)

Dataclasses controlling tree backup, caching, and advantage computation.

| Class                | Field                     | Type            | Default                   | Description                                        |
| -------------------- | ------------------------- | --------------- | ------------------------- | -------------------------------------------------- |
| `Config`             | `mode`                    | `CacheMode`     | `OFF`                     | Controls when/how tree backup activates            |
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
|                      | `teacher_backend`         | `str`           | `"openai"`                | Teacher backend type (`"openai"` or `"sglang"`)    |
|                      | `teacher_model_name`      | `str`           | `""`                      | Teacher model identifier                           |
|                      | `teacher_api_key`         | `str`           | `""`                      | API key for teacher endpoint                       |
|                      | `teacher_top_k`           | `int`           | `10`                      | Top-k tokens from teacher                          |
|                      | `teacher_max_retries`     | `int`           | `3`                       | Max retries for teacher requests                   |
|                      | `teacher_timeout`         | `float`         | `300.0`                   | Timeout for teacher requests                       |
|                      | `teacher_missing_logprob` | `float`         | `-23.0`                   | Default logprob for missing teacher tokens         |
|                      | `diagnose_model_name`     | `str`           | `""`                      | Model name for episode diagnosis                   |
|                      | `diagnose_max_tokens`     | `int`           | `1024`                    | Max tokens for diagnosis responses                 |
|                      | `diagnose_temperature`    | `float`         | `0.0`                     | Temperature for diagnosis sampling                 |
|                      | `diagnose_base_url`       | `str`           | `""`                      | Base URL for diagnosis API                         |
|                      | `diagnose_api_key`        | `str`           | `""`                      | API key for diagnosis endpoint                     |
|                      | `strict_distill_json`     | `bool`          | `True`                    | Enforce strict JSON parsing in distillation        |
|                      | `sample_source`           | `SampleSource`  | `SCRATCH`                 | Episode sampling strategy                          |
|                      | `branch_probability`      | `float`         | `0.5`                     | Probability of branch when MIXED                   |
|                      | `use_fresh_query`         | `bool`          | `False`                   | Enable database-backed query loading               |
|                      | `fresh_query_table`       | `str`           | `""`                      | DB table name (or `FRESH_QUERY_TABLE` env var)     |
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

**`SampleSource`** values:

- `SCRATCH` — always generate fresh episodes from scratch
- `BRANCH` — branch from cached trajectories when candidates exist
- `MIXED` — probabilistically choose between scratch and branch (controlled by
  `branch_probability`)

### 2. MCTS Tree Store (`core/mcts_tree_store.py`)

The central data structure. Manages a flat per-query list of `Node` objects, tracks MCTS
statistics per trajectory, and provides cached trajectory loading.

#### Node Dataclass

A `Node` represents one assistant response turn with its full conversation context (all
tokens from the beginning through this turn's response). Nodes are linked via `node_id`
/ `parent_node_id` and grouped into episodes via `episode_id`.

| Field               | Type                        | Description                                          |
| ------------------- | --------------------------- | ---------------------------------------------------- |
| `input_ids`         | `list[int]`                 | Full token sequence (prompt + response)              |
| `loss_mask`         | `list[int]`                 | 0=prompt tokens, 1=response tokens                   |
| `logprobs`          | `list[float]`               | Per-token log probabilities                          |
| `versions`          | `list[int]`                 | Policy version per token (-1 on prompt)              |
| `node_id`           | `str`                       | Globally unique interaction ID (UUID)                |
| `parent_node_id`    | `str \| None`               | Parent interaction ID (None for root)                |
| `episode_id`        | `str`                       | Groups turns into a trajectory path                  |
| `turn_idx`          | `int`                       | 1-based turn position within episode                 |
| `query_id`          | `str`                       | Dataset query identifier                             |
| `train_id`          | `str`                       | Training run that trained this node ("" = untrained) |
| `task_id`           | `str`                       | TPFC backend task that produced this node            |
| `entropy_stats`     | `dict \| None`              | Entropy statistics from TPFC assistant metadata      |
| `need_branch`       | `bool`                      | Whether this node is a candidate for branch sampling |
| `branch_sandbox_id` | `str \| None`               | Sandbox ID for branch task creation                  |
| `outcome_reward`    | `float`                     | Trajectory-level reward                              |
| `advantages`        | `torch.Tensor \| None`      | Tree-computed per-token advantages                   |
| `returns`           | `torch.Tensor \| None`      | Tree-computed per-token returns                      |
| `topk_ids`          | `list[list[int]] \| None`   | Top-k candidate token IDs per response position      |
| `topk_logp`         | `list[list[float]] \| None` | Top-k candidate log probabilities                    |
| `distill_reward`    | `list[list[float]] \| None` | Per-position distillation rewards                    |
| `teacher_logp`      | `list[list[float]] \| None` | Teacher log probabilities per position               |
| `guidance`          | `dict[int, str] \| None`    | Turn index → guidance text map (on leaf nodes)       |

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

### 3. Advantage Computer (`core/advantage.py`)

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

### 4. Checkpoint Manager (`core/checkpoint.py`)

Serializes/deserializes the full MCTS tree state to disk.

| Method                              | Description                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `save(tree_store)`                  | Save self-contained per-query trajectory records as `query_{sanitized_id}.json` files with per-query metadata |
| `save_query(tree_store, query_id)`  | Save checkpoint for a single query (used per-episode in the workflow)                                         |
| `load()`                            | Restore `MCTSTreeStore` from disk. No rebuild needed — stats keyed by string node_id.                         |
| `exists()`                          | Check if a checkpoint directory exists                                                                        |
| `save_trained_episodes(dir, store)` | Save trained episode IDs to recover checkpoint directory                                                      |
| `load_trained_episodes(dir)`        | Load trained episode IDs from recover checkpoint directory                                                    |

### 5. Tree Search Grouped Rollout Workflow (`core/customized_grouped_workflow.py`)

`TreeSearchGroupedRolloutWorkflow` is the core component that extends `RolloutWorkflow`
to provide tree-search-aware rollout with cache reuse and branch sampling.

**Initialization (`__init__`):**

Accepts the full set of configuration parameters (see `Config` above), plus:

| Parameter            | Type              | Description                                      |
| -------------------- | ----------------- | ------------------------------------------------ |
| `workflow`           | `RolloutWorkflow` | Base workflow for episode generation             |
| `group_size`         | `int`             | Number of episodes per query (must be >= 1)      |
| `checkpoint_dir`     | `str`             | Directory for tree checkpoint persistence        |
| `advantage_mode`     | `AdvantageMode`   | TREE or GAE advantage computation                |
| `loss_mode`          | `LossMode`        | GRPO, DISTILL, or BOTH                           |
| `cache_mode`         | `CacheMode`       | OFF, IN_TRAINING, or CROSS_TRAINING              |
| `tokenizer_path`     | `str`             | Path to HF tokenizer (required for distillation) |
| `max_tokens`         | `int`             | Max tokens per node sequence (0 = no truncation) |
| `sample_source`      | `SampleSource`    | SCRATCH, BRANCH, or MIXED                        |
| `branch_probability` | `float`           | Probability of branch in MIXED mode              |
| ...                  | ...               | All `Config` fields (see config table)           |

- Creates `TreeCheckpointManager` and `MCTSTreeStore`
- On `CROSS_TRAINING` mode, loads existing tree checkpoint if available
- Creates `TreeAdvantageComputer`

**Per-episode flow (`arun_episode`):**

1. **Check cache**: Count untrained episodes for the query via
   `tree_store.get_untrained_episode_count()`
1. **Generate fresh episodes** if needed: For each of `group_size - cached_count`
   episodes, decide the sampling strategy:
   - **SCRATCH**: Run a fresh episode from scratch via `_retry_episode()`
   - **BRANCH**: Select a branch candidate node (highest max-entropy), build a branch
     task from its sandbox, run the episode from the branch point, then clean up the
     branch sandbox via `_cleanup_branch()`
   - **MIXED**: Probabilistically choose between SCRATCH and BRANCH based on
     `branch_probability`
   - Each fresh episode result is wrapped in `EpisodeRunResult` (carrying `task_id` and
     `raw_messages` from the TPFC backend)
1. **Annotate Nodes**: `annotate_nodes_from_run()` copies TPFC assistant-message
   metadata (task_id, entropy_stats, need_branch, branch_sandbox_id) onto fresh Nodes
1. **Convert results to Nodes**: `_result_to_nodes()` converts each arun_episode result
   (dict or list of `InteractionWithTokenLogpReward`) to `list[Node]`, assigning
   `episode_id`, `query_id`, and `turn_idx`
1. **Load cached nodes**: `tree_store.load_untrained_episodes(query_id, cached_count)`
1. **Insert fresh nodes**: `tree_store.insert_batch(fresh_nodes)`
1. **Combine**: Merge fresh and cached nodes (total = group_size)
1. **Teacher model reward computation** (if `loss_mode != GRPO`):
   - Load tokenizer from `tokenizer_path`
   - Build teacher provider (external API or engine-based) via
     `_setup_distill_provider()`
   - Apply distillation on combined node groups via `_prepare_distill_for_node_groups()`
   - For each episode group, diagnose to find turns needing improvement
     (`_prepare_distill_for_episode()` → `provider.diagnose_episode()`)
   - Reuse cached guidance from previous diagnoses when available
   - For selected turns, get teacher logprobs for candidate tokens
     (`selected_turn_to_position_rewards()`)
   - Store distillation data in `node.teacher_logp` and `node.topk_ids`
   - Also store diagnosis guidance in `node.guidance` on leaf nodes
   - In `DISTILL` mode, episodes with no diagnosis or no selected turns are filtered out
1. **Compute tree advantages**: `tree_advantage_computer.compute(all_nodes)` (TREE mode)
1. **Mark trained**: `tree_store.set_trained(node.node_id, True)` for all nodes
1. **Save checkpoint**: `tree_checkpoint_manager.save_query(tree_store, query_id)`
   (CROSS_TRAINING mode)
1. **Convert to tensor dict**: `_nodes_to_batched_tensor_dict()` converts `list[Node]`
   to batched tensor dict

**Utility functions and dataclasses:**

| Name                                    | Description                                                                                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `EpisodeRunResult`                      | Dataclass wrapping an episode result with `task_id` and `raw_messages` from the TPFC backend                                              |
| `choose_sample_source()`                | Decide SCRATCH/BRANCH/MIXED based on mode, candidate availability, and random value                                                       |
| `select_branch_candidate()`             | Select the best node for branching (highest max-entropy among `need_branch` nodes with a sandbox)                                         |
| `build_branch_task()`                   | Create a TPFC branch task from a candidate node's sandbox and truncated message prefix                                                    |
| `annotate_nodes_from_run()`             | Copy TPFC assistant-message metadata (task_id, entropy_stats, need_branch, branch_sandbox_id) onto Nodes by turn_idx                      |
| `_with_episode_metadata()`              | Wrap an episode result in `EpisodeRunResult` if backend metadata is available                                                             |
| `_max_entropy()`                        | Extract max_entropy value from a Node's entropy_stats                                                                                     |
| `interactions_dict_to_nodes()`          | Convert `dict[str, InteractionWithTokenLogpReward]` to `list[Node]` (also handles proxy-deserialized data where `model_response` is None) |
| `_result_to_nodes()`                    | Convert a single arun_episode result (dict or list) to `list[Node]` with episode metadata                                                 |
| `_nodes_to_batched_tensor_dict()`       | Convert `list[Node]` to batched tensor dict via `concat_padded_tensors`                                                                   |
| `_input_ids_to_messages()`              | Convert full-context token IDs to a list of role/content message dicts using chat template markers                                        |
| `_retry_episode()`                      | Retry a failed episode with exponential backoff (up to 1 retry)                                                                           |
| `_prepare_distill_for_episode()`        | Diagnose one episode and compute position-level teacher rewards (with diagnosis retry and cached guidance reuse)                          |
| `_prepare_distill_for_node_groups()`    | Apply distillation to multiple episode groups with error handling                                                                         |
| `_group_nodes_by_episode()`             | Group a flat list of Nodes by `episode_id`                                                                                                |
| `_filter_distill_episode_failure()`     | In DISTILL mode, return empty list on failure (drop episode); otherwise return nodes unchanged                                            |
| `_set_position_reward_sample_indices()` | Assign `sample_index` to each `PositionRewardInfo` based on node position in batch                                                        |

**Methods:**

| Method                      | Description                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `_run_fresh_episode()`      | Run a single fresh episode, deciding between scratch and branch sampling             |
| `_prepare_branch_task()`    | Create a TPFC branch task from a branch candidate node                               |
| `_cleanup_branch()`         | Delete branch sandbox and mark node as branched to prevent re-use                    |
| `_get_tokenizer()`          | Lazy-load and cache HF tokenizer (shared across episodes via class-level cache)      |
| `_setup_distill_provider()` | Build `ExternalTeacherProvider` with auto-detected engine addresses and backend type |

### 6. Trainer (`training/trainer.py`)

#### `CustomizedPPOTrainer`

PPO trainer with tree-search-aware rollout support. Extends `PPOTrainer` directly.

**Key design**: All cache logic, tree ops, and checkpoint saving happen inside
`TreeSearchGroupedRolloutWorkflow` (activated by the `.env` flag in
`customized_areal/.env`). The trainer itself is minimal:

**Initialization (`__init__`):**

- Accepts `tree_search_config` and stores it
- Delegates to `PPOTrainer.__init__()`

**`_create_train_engine`:**

- When `loss_mode != GRPO`: Returns `MultiCandidateFSDPPPOActor` (requires FSDP backend)
- Otherwise: Delegates to standard `PPOTrainer._create_train_engine()`

**`train()`:**

- When `loss_mode != GRPO`: Applies distill loss patch, calls `super().train()`,
  restores patch in `finally`
- Otherwise: Delegates to `super().train()`

**`_save_hf` / `_save_recover_checkpoint`:**

- Override to write `train_id.json` sidecar alongside each model checkpoint
- Enables tracking which training run produced each checkpoint

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

#### `distilling/distill_types.py`

| Class                             | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `PositionRewardInfo`              | Per-position candidate tokens, logprobs, and rewards              |
| `DiagnosisTurn`                   | Single turn diagnosis with `should_improve` flag and guidance     |
| `EpisodeDiagnosis`                | Collection of `DiagnosisTurn`s with `selected_turns` property     |
| `InteractionWithTokenLevelReward` | Extended interaction with `token_rewards` and `token_reward_mask` |

#### `distilling/` — On-Policy Distillation

| File                                  | Purpose                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `distilling/config.py`                | `OnPolicyDistillConfig` (extends PPOConfig) and `AgentConfig`                  |
| `distilling/agent.py`                 | `OnPolicyDistillAgent` — agent class for distillation training                 |
| `distilling/reward_compute.py`        | `_compute_token_rewards()` — student vs teacher logprob comparison             |
| `distilling/teacher_client.py`        | `TeacherConfig`, `TeacherClient` — async teacher model inference               |
| `distilling/teacher_provider.py`      | `TeacherProvider` protocol, `ExternalTeacherProvider`, `EngineTeacherProvider` |
| `distilling/selected_turn_distill.py` | Diagnoses episodes and builds position-level teacher rewards                   |

#### `engine/` — Multi-Candidate Engine

| File                    | Purpose                                                                         |
| ----------------------- | ------------------------------------------------------------------------------- |
| `engine/fsdp_engine.py` | `MultiCandidateFSDPEngine` — FSDP engine with multi-candidate logprob gathering |

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

| File                   | Purpose                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `training/loss.py`     | `grpo_distill_loss_fn()` — combined GRPO + position-level distillation loss                                      |
| `training/actor.py`    | `MultiCandidateFSDPPPOActor`, `patch_ppo_actor_class_to_use_distill_loss()` / `unpatch_ppo_actor_distill_loss()` |
| `training/logprobs.py` | `gather_logprobs_entropy_multi_candidates()` — multi-candidate logprob gathering                                 |
| `training/trainer.py`  | `CustomizedPPOTrainer` — PPO trainer with distillation engine support                                            |

**`grpo_distill_loss_fn` computes:**

1. Standard GRPO loss using chosen token logprobs
1. Teacher KL distillation loss from `position_rewards`:
   - For each position with teacher logprobs: `student_logp - teacher_logp`
   - Mean over all positions and candidates
   - Added to the actor loss with weight `distill_loss_weight`
1. Combined loss: `rl_loss_weight * grpo_loss + distill_loss_weight * teacher_kl_loss`

## Branch Sampling

When `sample_source` is `BRANCH` or `MIXED`, the workflow can reuse cached trajectories
as starting points for new episodes instead of always starting from scratch. This
leverages TPFC backend infrastructure to create branch tasks from existing sandboxes.

### How Branch Sampling Works

1. **Candidate Selection**: `select_branch_candidate()` picks the best node for
   branching among cached nodes for the query. Candidates must have `need_branch=True`,
   a `task_id`, and a `branch_sandbox_id`. The candidate with the highest `max_entropy`
   is chosen (entropy signals uncertainty where branching is most valuable).

1. **Branch Task Creation**: `build_branch_task()` creates a new TPFC task, binds the
   candidate's sandbox to it, and copies the conversation prefix (messages up to the
   branch point) into the new task. This allows the episode to resume from the branch
   point.

1. **Episode Execution**: The episode runs from the branch point, generating new
   responses from the selected turn onward.

1. **Cleanup**: `_cleanup_branch()` deletes the branch sandbox and clears the candidate
   node's `need_branch` and `branch_sandbox_id` to prevent re-use.

### Episode Generation Flow

```mermaid
flowchart TD
    START["_run_fresh_episode()"]
    CAND["select_branch_candidate()<br/>(highest entropy among need_branch nodes)"]
    SRC["choose_sample_source()"]

    START --> CAND --> SRC

    SRC --> M{"mode?"}
    M -- SCRATCH --> SCRATCH["_retry_episode()<br/>(fresh from start)"]
    M -- BRANCH --> BC{"has_candidate?"}
    M -- MIXED --> MC{"random < branch_probability<br/>AND has_candidate?"}

    BC -- Yes --> BRANCH
    BC -- No --> SCRATCH
    MC -- Yes --> BRANCH
    MC -- No --> SCRATCH

    subgraph branch_path["BRANCH Path"]
        BRANCH["build_branch_task()"]
        BT["Create TPFC task<br/>Bind candidate sandbox<br/>Copy message prefix"]
        RUN["_retry_episode()<br/>(from branch point)"]
        CLEAN["_cleanup_branch()<br/>Delete sandbox, clear candidate flags"]
        BRANCH --> BT --> RUN --> CLEAN
    end

    SCRATCH --> META["_with_episode_metadata()"]
    RUN --> META
    META --> CONV["_result_to_nodes()"]
    CONV --> ANN["annotate_nodes_from_run()"]
```

### SampleSource Decision Logic

```mermaid
flowchart TD
    START["choose_sample_source(mode, branch_probability, has_candidate, random_value)"]
    MODE{"mode?"}

    SCRATCH_RES["Return SCRATCH"]
    BRANCH_RES["Return BRANCH"]

    HAS_B{"has_candidate?"}
    HAS_M{"has_candidate AND<br/>random_value < branch_probability?"}

    START --> MODE
    MODE -- SCRATCH --> SCRATCH_RES
    MODE -- BRANCH --> HAS_B
    MODE -- MIXED --> HAS_M
    HAS_B -- Yes --> BRANCH_RES
    HAS_B -- No --> SCRATCH_RES
    HAS_M -- Yes --> BRANCH_RES
    HAS_M -- No --> SCRATCH_RES
```

### Dynamic Group Size

When `dynamic_group_size=True`, the workflow iteratively samples episodes until
uncertainty drops below a threshold, rather than using a fixed `group_size`:

```mermaid
flowchart TD
    START["_arun_episode_dynamic()"]
    CACHE["1. Check cache<br/>get_untrained_episode_count()"]
    INIT_GEN["2. Generate initial_group_size<br/>− cached_count episodes"]
    INIT_LOAD["3. Load cached + fresh nodes"]

    START --> CACHE --> INIT_GEN --> INIT_LOAD

    INIT_LOAD --> COMPUTE["4. Compute initial uncertainty U(q)<br/>via Bayesian posterior variance"]

    COMPUTE --> LOOP{"5. Iterative sampling loop"}

    subgraph loop["Iterative Sampling"]
        CHECK{"U(q) ≤ threshold<br/>OR episodes ≥ max_group_size?"}
        SAMPLE["Sample one more episode"]
        UPDATE["Recompute U(q)"]
        FAIL_CHECK{"Episode failed?"}
        CIRCUIT{"Consecutive failures<br/>≥ max_failed?"}
        CHECK -- Yes --> DONE["Stop sampling"]
        CHECK -- No --> SAMPLE --> FAIL_CHECK
        FAIL_CHECK -- Yes --> CIRCUIT
        CIRCUIT -- Yes --> DONE
        CIRCUIT -- No --> CHECK
        FAIL_CHECK -- No --> UPDATE --> CHECK
    end

    LOOP --> loop
    DONE --> FINAL["_finalize_episode()"]
```

The uncertainty metric U(q) uses Bayesian posterior variance adjusted by mean episode
steps. For binary rewards it uses a Beta(1,1) posterior; for continuous rewards it uses
a Normal-Inverse-Gamma posterior with weak priors.

## Fresh Query Mode

When `use_fresh_query=True`, the training pipeline dynamically loads training queries
from a shared database table instead of iterating over a static local dataset. This is
designed for distributed training where multiple runs need non-overlapping samples, or
when training data is populated in real time by an external pipeline.

### Configuration

| Field               | Type   | Default | Description                                                          |
| ------------------- | ------ | ------- | -------------------------------------------------------------------- |
| `use_fresh_query`   | `bool` | `False` | Enable database-backed query loading                                 |
| `fresh_query_table` | `str`  | `""`    | Database table name (or set `FRESH_QUERY_TABLE` env var)             |
| `TRAIN_ID` env var  | `str`  | —       | **Required when enabled**. Unique training-run ID for claim tracking |

Validation in `Config.__post_init__`:

- `fresh_query_table` must be non-empty when `use_fresh_query=True` (falls back to
  `FRESH_QUERY_TABLE` env var, then raises `ValueError` if still empty).
- `TRAIN_ID` env var must be set at runtime (checked in `_load_fresh_query_data`).
- `total_train_steps` must be set in the training config (checked in
  `CustomizedPPOTrainer.__init__`).
- `total_train_steps // total_train_epochs >= 1` (at least one step per epoch).

### Database Table Schema

The expected table schema (see `core/fresh_query.sql`):

```sql
CREATE TABLE IF NOT EXISTS public.query_bank (
    query_id         TEXT PRIMARY KEY,
    query            TEXT NOT NULL,
    gold_answer      TEXT NOT NULL,
    evaluation_rubric TEXT[] NOT NULL DEFAULT '{}',
    used4train       TEXT[] NOT NULL DEFAULT '{}',   -- array of TRAIN_IDs that claimed this row
    synthetic        BOOLEAN DEFAULT NULL,
    level            TEXT,
    task_id          TEXT,
    label            TEXT[] NOT NULL DEFAULT '{}',
    file_paths       TEXT[] NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

The `used4train` column is the claim mechanism: each training run appends its `TRAIN_ID`
to this array when it claims a row, preventing other runs from picking the same query.

### How It Works

1. **Trainer side** — `CustomizedPPOTrainer` detects `use_fresh_query=True` in
   `__init__`:

   - Replaces the training dataset with `_FreshQueryDatasetPlaceholder` (a 1-item dummy
     so `PPOTrainer` enters dataset-backed setup).
   - Overrides `_create_dataloader` to return `_EmptyDataLoader`, which yields empty
     dicts for exactly `total_train_steps // total_train_epochs` steps per epoch. The
     actual query content comes from the database, not the dataloader.

1. **Workflow side** — At the start of each `arun_episode` call, if
   `use_fresh_query=True`:

   - Calls `_load_fresh_query_data(data)` which fetches an eligible row from
     `fresh_query_table` and atomically claims it.
   - If no eligible row is found, returns `None` (episode skipped).
   - The claimed row's fields overwrite the placeholder data dict.

1. **Claim flow** (`_load_fresh_query_data`):

   ```mermaid
   flowchart TD
       START["_load_fresh_query_data(data)"]
       ENV["Get TRAIN_ID from env<br/>(raise ValueError if missing)"]
       CLIENT["Get Supabase client<br/>via DBConnection"]
       SELECT["SELECT query_id, query, gold_answer,<br/>evaluation_rubric, used4train<br/>FROM fresh_query_table<br/>LIMIT 100"]
       FILTER["Filter: used4train NOT CONTAINS TRAIN_ID"]
       ROWS["Iterate returned rows"]
       SKIP{"train_id in<br/>used4train?"}
       CLAIM["UPDATE used4train = [..., train_id]<br/>WHERE query_id = row.query_id"]
       RACE{"Affected rows >= 1?"}
       WIN["Claimed! Return merged data"]
       LOSE["Lost race → try next row"]
       EXHAUSTED["No eligible rows found → return None"]

       START --> ENV --> CLIENT --> SELECT --> FILTER --> ROWS
       ROWS --> SKIP
       SKIP -- Yes --> ROWS
       SKIP -- No --> CLAIM --> RACE
       RACE -- Yes --> WIN
       RACE -- No --> LOSE --> ROWS
       ROWS -- exhausted --> EXHAUSTED
   ```

   The claim is **optimistic**: the `not_.contains("used4train", [train_id])` filter
   excludes already-claimed rows at select time, and the update-result row-count check
   handles concurrent races. If two workers select the same row simultaneously, only one
   update succeeds (the other sees `< 1` affected rows) and the loser retries with the
   next row.

1. **Data merge** (`_apply_fresh_query_row`): The claimed row's fields overwrite the
   placeholder data:

   | Claimed DB field    | Merged data key     |
   | ------------------- | ------------------- |
   | `query_id`          | `query_id`          |
   | `query`             | `query`             |
   | `gold_answer`       | `answer`            |
   | `evaluation_rubric` | `evaluation_rubric` |
   | `used4train`        | `used4train`        |
   | `file_paths`        | `file_paths`        |

### Comparison: Static vs Fresh Query

| Aspect                  | `use_fresh_query=False`                       | `use_fresh_query=True`                                   |
| ----------------------- | --------------------------------------------- | -------------------------------------------------------- |
| **Data source**         | Local parquet files via `get_tpfc_rl_dataset` | Database table (`fresh_query_table`)                     |
| **Dataloader**          | Standard dataset-backed                       | `_EmptyDataLoader` (empty dicts, step-count driven)      |
| **Query selection**     | Sequential iteration over dataset             | Dynamic fetch + atomic claim per step                    |
| **Deduplication**       | N/A (single-run)                              | `used4train` array + `TRAIN_ID` prevents cross-run reuse |
| **Env requirements**    | Standard training variables                   | `TRAIN_ID` + `FRESH_QUERY_TABLE` + DB access             |
| **On query exhaustion** | N/A (wraps around)                            | Episode skipped (returns `None`), training continues     |

### Example YAML Configuration

```yaml
tree_search:
  use_fresh_query: true
  fresh_query_table: query_bank
  # ... other tree_search fields ...
```

And the required environment variables:

```bash
export TRAIN_ID="run-2026-06-16-abc123"
export FRESH_QUERY_TABLE=query_bank   # optional if set in config
```

## Data Flow

### Cache-Aware Training

```mermaid
flowchart TD
    subgraph Trainer["CustomizedPPOTrainer"]
        T1["train()"]
        T2{"loss_mode != GRPO?"}
        T3["patch distill loss"]
        T4["super().train()<br/>(per-step loop)"]
        T5["unpatch distill loss"]
        T6["super().train()"]
        T1 --> T2
        T2 -- Yes --> T3 --> T4 --> T5
        T2 -- No --> T6
    end

    T4 --> STEP["per training step"]
    T6 --> STEP

    subgraph WF["TreeSearchGroupedRolloutWorkflow"]
        W0["arun_episode(engine, data)"]
        W1{"dynamic_group_size?"}
        W2["_arun_episode_fixed()"]
        W3["_arun_episode_dynamic()"]
        W0 --> W1
        W1 -- No --> W2
        W1 -- Yes --> W3
    end

    STEP --> W0

    W2 --> CC["1. Check cache<br/>cached_count = get_untrained_episode_count()"]
    W3 --> CC
    CC --> NG["need_gen = group_size − cached_count"]

    NG --> GEN["2. Generate fresh episodes (parallel)<br/>see Branch Sampling diagram"]

    GEN --> LOAD["3. Load cached + insert fresh + combine"]

    LOAD --> FINAL["_finalize_episode()"]

    FINAL --> ZVD{"Zero-variance<br/>discard?"}
    ZVD -- Yes --> NONE["Return None"]
    ZVD -- No --> INS["insert_batch(fresh_nodes)"]

    INS --> DIST{"loss_mode != GRPO?"}
    DIST -- Yes --> DIST_RUN["Run distillation<br/>(see Distillation Pipeline diagram)"]
    DIST -- No --> ADV
    DIST_RUN --> ADV{"advantage_mode == TREE?"}

    ADV -- Yes --> ADV_RUN["tree_advantage_computer.compute()"]
    ADV -- No --> CONV
    ADV_RUN --> CONV["_nodes_to_batched_tensor_dict()"]

    CONV --> MARK["set_trained()"]
    MARK --> SAVE["save_query()"]

    subgraph Engine["Training Engine (MultiCandidateFSDPEngine)"]
        E1["build_packed_tree_batch()"]
        E2["forward() with tree attention"]
        E3["_compute_logprobs_entropy()"]
        E4["ppo_update() with grpo_distill_loss_fn()"]
        E1 --> E2 --> E3 --> E4
    end

    SAVE --> E1
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
   which turns need improvement and provides guidance. Cached guidance from previous
   diagnoses is reused to avoid redundant teacher calls.

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

### Distillation Pipeline

```mermaid
flowchart TD
    START["_prepare_distill_for_node_groups()"]
    SETUP["Setup: get tokenizer + build provider<br/>(TeacherClient + DiagnoseProvider)"]
    START --> SETUP

    subgraph per_episode["Per Episode: _prepare_distill_for_episode()"]
        CACHED{"node.guidance<br/>cached?"}
        DIAGNOSE["provider.diagnose_episode()"]
        PARSE["parse_episode_diagnosis()"]
        RETRY["Retry on parse failure<br/>(up to 3x, increasing temperature)"]
        USE_CACHED["Reuse cached guidance"]
        CACHED -- Yes --> USE_CACHED
        CACHED -- No --> DIAGNOSE --> PARSE
        PARSE -- Parse error --> RETRY --> DIAGNOSE
        PARSE -- Success --> HAS_TURNS

        USE_CACHED --> HAS_TURNS{"Selected turns<br/>exist?"}
        HAS_TURNS -- No --> SKIP["Skip distillation<br/>(filter if DISTILL mode)"]

        HAS_TURNS -- Yes --> PER_TURN

        subgraph per_turn["Per Selected Turn"]
            PT_BUILD["build_teacher_prompt_ids()<br/>prefix + guidance + generation"]
            PT_CAND{"topk_distill?"}
            PT_TOPK["Use node.topk_ids or<br/>_recompute_student_topk()"]
            PT_GEN["Candidates = [generated_token_id]"]
            PT_ALIGN["_ensure_generated_token_first()"]
            PT_TEACHER["provider.get_logprobs_for_prompt()"]
            PT_REWARD["Build PositionRewardInfo<br/>(candidate_token_ids, teacher_logprobs, chosen_index)"]
            PT_BUILD --> PT_CAND
            PT_CAND -- Yes --> PT_TOPK --> PT_ALIGN
            PT_CAND -- No --> PT_GEN --> PT_ALIGN
            PT_ALIGN --> PT_TEACHER --> PT_REWARD
        end

        PER_TURN --> STORE["Store teacher_logp, topk_ids on Node<br/>Cache guidance on leaf node"]
    end

    SETUP --> per_episode
```

### Loss Computation

The `grpo_distill_loss_fn` operates in three modes depending on `distill_loss_mode` and
`rl_loss_weight`:

```mermaid
flowchart TD
    START["grpo_distill_loss_fn()"]
    INPUTS["Extract: old_logp, advantages, loss_mask,<br/>teacher_logp, rl_loss_weight, distill_loss_weight"]
    START --> INPUTS --> MODE{"distill_loss_mode +<br/>rl_loss_weight"}

    subgraph evidence["Evidence/RLSD Mode<br/>(teacher_logp present + rl_loss_weight != 0)"]
        E1["_compute_distill_reweighted_advantages()<br/>δ = student_logp − teacher_logp<br/>evidence_weight = sigmoid(λ · δ · advantage)"]
        E2["_compute_grpo_loss()<br/>with reweighted advantages"]
        E3["loss = rl_loss_weight × GRPO_loss"]
        E1 --> E2 --> E3
    end

    subgraph distill["DISTILL Mode (rl_loss_weight == 0)"]
        D1["_compute_teacher_kl_loss()"]
        D2["loss = distill_loss_weight × KL_loss"]
        D1 --> D2
    end

    subgraph both["BOTH Mode (default)"]
        B1["_compute_grpo_loss()<br/>with original advantages"]
        B2["_compute_teacher_kl_loss()"]
        B3["loss = rl_loss_weight × GRPO_loss<br/>+ distill_loss_weight × KL_loss"]
        B1 --> B2 --> B3
    end

    MODE -- "evidence / rlsd" --> evidence
    MODE -- "DISTILL<br/>(rl = 0)" --> distill
    MODE -- "BOTH (default)<br/>(teacher_logp + default mode)" --> both

    evidence --> RETURN["Return loss"]
    distill --> RETURN
    both --> RETURN
```

## Public API

### Direct Imports

```python
from customized_areal.tree_search import (
    CustomizedPPOTrainer,
    MCTSTreeStore,
    Node,
    Config,
    RolloutCacheConfig,
    CacheMode,
    AdvantageMode,
    LossMode,
    SampleSource,
    TreeAdvantageComputer,
    TreeCheckpointManager,
    TreeSearchGroupedRolloutWorkflow,
    PositionRewardInfo,
    DiagnosisTurn,
    EpisodeDiagnosis,
    InteractionWithTokenLevelReward,
)
```

### Lazy Imports

The following are available via `__getattr__` for reduced import overhead:

```python
from customized_areal.tree_search import (
    OnPolicyDistillConfig,              # from distilling.config
    OnPolicyDistillAgent,               # from distilling.agent
    TeacherConfig,                       # from distilling.teacher_client
    TeacherClient,                       # from distilling.teacher_client
    TeacherProvider,                     # from distilling.teacher_provider
    EngineTeacherProvider,              # from distilling.teacher_provider
    ExternalTeacherProvider,            # from distilling.teacher_provider
    MultiCandidateFSDPEngine,           # from engine
    MultiCandidateFSDPPPOActor,         # from training (via engine)
    grpo_distill_loss_fn,               # from training.loss
    gather_logprobs_entropy_multi_candidates,  # from training.logprobs
    _compute_token_rewards,             # from distilling.reward_compute
)
```

## Usage Example

```python
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
    RolloutCacheConfig,
    SampleSource,
    Config,
)
from customized_areal.tree_search.training.trainer import CustomizedPPOTrainer

cache_config = RolloutCacheConfig(
    cache_dir="/path/to/tree_cache",
    enabled=True,
    n_samples=8,
)

tree_search_config = Config(
    mode=CacheMode.CROSS_TRAINING,
    checkpoint_dir="/path/to/tree_cache",
    advantage_mode=AdvantageMode.TREE,
    loss_mode=LossMode.BOTH,  # or DISTILL for distillation only
    distill_loss_weight=0.005,
    teacher_provider="external",
    teacher_base_url="http://localhost:8001",
    teacher_backend="openai",
    teacher_model_name="teacher-model",
    sample_source=SampleSource.MIXED,
    branch_probability=0.5,
)

with CustomizedPPOTrainer(
    config,
    cache_config=cache_config,
    tree_search_config=tree_search_config,
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

| File                                  | Purpose                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------- |
| `__init__.py`                         | Public API exports and lazy imports for distillation components                              |
| `config.py`                           | `Config`, `RolloutCacheConfig`, `CacheMode`, `AdvantageMode`, `LossMode`, `SampleSource`     |
| `core/advantage.py`                   | `TreeAdvantageComputer` — GRPO-normalized tree Q-value advantages                            |
| `core/checkpoint.py`                  | `TreeCheckpointManager` — serialize/deserialize tree state to JSON                           |
| `core/tree_store.py`                  | `MCTSTreeStore`, `Node` — flat trajectory store with MCTS statistics                         |
| `core/customized_grouped_workflow.py` | `TreeSearchGroupedRolloutWorkflow` — core workflow with cache reuse + tree ops               |
| `distilling/__init__.py`              | Distilling subpackage exports                                                                |
| `distilling/config.py`                | `OnPolicyDistillConfig`, `AgentConfig`                                                       |
| `distilling/agent.py`                 | `OnPolicyDistillAgent` — agent for distillation training                                     |
| `distilling/distill_types.py`         | `PositionRewardInfo`, `DiagnosisTurn`, `EpisodeDiagnosis`, `InteractionWithTokenLevelReward` |
| `distilling/reward_compute.py`        | Student vs teacher logprob reward computation                                                |
| `distilling/teacher_client.py`        | `TeacherConfig`, `TeacherClient` — async teacher model inference client                      |
| `distilling/teacher_provider.py`      | `TeacherProvider` protocol, `ExternalTeacherProvider`, `EngineTeacherProvider`               |
| `distilling/selected_turn_distill.py` | Diagnoses episodes and builds position-level teacher rewards                                 |
| `engine/__init__.py`                  | Engine subpackage exports                                                                    |
| `engine/fsdp_engine.py`               | `MultiCandidateFSDPEngine` — multi-candidate logprob gathering                               |
| `training/__init__.py`                | Training subpackage exports                                                                  |
| `training/actor.py`                   | `MultiCandidateFSDPPPOActor`, distill-loss patching functions                                |
| `training/loss.py`                    | `grpo_distill_loss_fn` — combined GRPO + distillation loss                                   |
| `training/logprobs.py`                | Multi-candidate logprob/entropy gathering utilities                                          |
| `training/trainer.py`                 | `CustomizedPPOTrainer` — PPO trainer with distillation engine support                        |

## Generative Critic (Shared Model) with GAE

When `tree_search.enable_generative_critic=true`, the **actor's own model** (same
weights, same SGLang server) doubles as a **generative critic** that scores
partial-solution states, and those scores drive **GAE** advantages for the actor.

### How it works

1. **Value computation (rollout).** For each turn `t` of an episode, the critic prompt
   is built from the actual conversation messages through that turn plus an appended
   instruction:

   ```
   [system, turn 1, turn 2, ... turn t]  +  [critic_instruction]
   ```

   The `critic_instruction` asks the model to estimate its probability of eventual
   success as an integer in `[0, critic_score_max]` (the average dataset success rate
   `critic_avg_success_rate`, default `0.29`, is embedded in the prompt). The model is
   queried for the answer-position distribution over the digit tokens and the **soft
   expected value** `v_phi(s_t) = Σ_i p_i · (i / score_max)` is stored on `Node.value`.

1. **GAE advantages.** With `enable_generative_critic=true`, `advantage_mode` is forced
   to `gae`. `GAEAdvantageComputer` treats each turn as a step `s_t`, with sparse
   rewards (`r_t = 0` intermediate, `r_T = outcome_reward` at the leaf) and a zero
   terminal bootstrap:

   ```
   delta_t = r_t + gamma · v(s_{t+1}) - v(s_t)
   A_t     = delta_t + gamma · lambda · A_{t+1}
   ret_t   = A_t + v(s_t)
   ```

   Defaults `gamma=1.0`, `lambda=0.95`. `A_t`/`ret_t` are broadcast over each node's
   response positions and consumed by the standard PPO actor update.

1. **Critic regression (training).** The same shared model is additionally trained with
   an **expected-value soft regression**: at the answer position it produces a
   distribution over the digit tokens, and the expected value is regressed (MSE) toward
   the tree-stored MCTS `q_value` target,
   `target = clamp(q_value / critic_target_scale, 0, 1)`. The combined objective is

   ```
   loss = actor_loss + critic_loss_weight · critic_loss
   ```

   This step reuses the multi-candidate logprob gathering path (the digit tokens are the
   candidates at the answer position), so it is active when the
   `MultiCandidateFSDPEngine` is selected (`loss_mode=BOTH`/`DISTILL`). With
   `loss_mode=GRPO` the critic still drives GAE advantages, while the extra regression
   step is skipped gracefully.

### Config fields (`tree_search`)

| Field                      | Default | Meaning                                                     |
| -------------------------- | ------- | ----------------------------------------------------------- |
| `enable_generative_critic` | `false` | Enable the shared-model generative critic (forces GAE).     |
| `critic_avg_success_rate`  | `0.29`  | Average dataset success rate embedded in the critic prompt. |
| `critic_gamma`             | `1.0`   | GAE discount.                                               |
| `critic_lambda`            | `0.95`  | GAE lambda.                                                 |
| `critic_score_max`         | `10`    | Max integer score label (`0..score_max`).                   |
| `critic_target_scale`      | `1.0`   | Divisor applied to `q_value` before clamping to `[0, 1]`.   |
| `critic_max_new_tokens`    | `1024`  | Max tokens for the critic's generation.                     |
| `critic_temperature`       | `0.0`   | Critic generation temperature.                              |
| `critic_loss_weight`       | `1.0`   | Weight of the critic regression term in the combined loss.  |

### Tokenization caveat

A single answer position has one next-token distribution, so multi-token labels (e.g.
`"10"`) are represented by their **leading** token. For `critic_score_max=10` the labels
`"1"` and `"10"` share a leading token, a documented approximation; set
`critic_score_max=9` for a collision-free 0–9 scale.

### Example

See `customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B_generative_critic.yaml`. Run
it the same way as the other tree-search configs:

```bash
uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py \
  --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B_generative_critic.yaml
```

Logged stats: `critic_loss`, `critic_value_mean`, `critic_target_mean`.

# Fix Individual Export for Tree Search: Episode Reconstruction and GRPO Normalization

**Date:** 2026-04-29 **Status:** Draft

## Problem

When using `export_interactions(style="individual")` with `CacheAwarePPOTrainer`,
multi-turn episodes are flattened into independent batch rows. Each turn becomes a
separate row with `loss_mask = [0...0, 1...1]` and its own reward. This breaks:

1. **Tree insertion**: `insert_batch` assigns one `seq_id` per turn, not per episode. An
   episode with 3 turns gets 3 unrelated records.
1. **Advantage computation**: No episode-level Q-value. Per-turn Q-values are just
   per-turn rewards with no GRPO normalization.
1. **GRPO normalization**: `Normalization(mean_level="group")` slices consecutive rows,
   mixing turns from different episodes/queries.
1. **Variable turn counts**: Episodes with different turn counts break fixed-size group
   slicing.

## Design

### Component 1: `TreeSearchGroupedRolloutWorkflow`

**File:** `customized_areal/tree_search/grouped_workflow.py`

Replaces both `QueryIDProxyWorkflow` and `GroupedRolloutWorkflow`. Subclasses
`GroupedRolloutWorkflow` and overrides `arun_episode`.

**Responsibilities:**

- Reconstruct episode metadata from `InteractionWithTokenLogpReward` parent chains
- Convert each turn to an individual-style tensor dict `[1, seq_len]`
- Stack all turns from all `group_size` episodes into `[total_turns, max_seq_len]`
- Preserve per-episode metadata (turn IDs, parent IDs, rewards) as list-valued keys

**`arun_episode` logic:**

1. Run inner workflow `group_size` times via `asyncio.gather` (same as base)
1. For each result (a `dict[InteractionWithTokenLogpReward]`): a. Sort interactions by
   creation order (cache insertion order) b. Call `to_tensor_dict()` on each →
   individual-style `[1, seq_len]` per turn c. Collect turn metadata: `interaction_id`,
   `parent.interaction_id`, `reward` d. Determine outcome reward: last turn's reward (or
   sum, configurable)
1. `concat_padded_tensors` to stack all turns into `[total_turns, max_seq_len]`
1. Add episode-level metadata as list-valued keys

**Output tensor shape:** `[group_size * max_turns_per_episode, max_seq_len]`

**Metadata keys (non-tensor, survive concat_padded_tensors as `values[0]`):**

| Key                        | Type                    | Description                                    |
| -------------------------- | ----------------------- | ---------------------------------------------- |
| `_mcts_query_id`           | `str`                   | Query identifier                               |
| `_episode_num_turns`       | `list[int]`             | Turn count per episode, length = group_size    |
| `_episode_turn_offsets`    | `list[int]`             | Cumulative turn offsets: `[0, n0, n0+n1, ...]` |
| `_episode_turn_ids`        | `list[list[str]]`       | Interaction IDs per episode                    |
| `_episode_parent_turn_ids` | `list[list[str\|None]]` | Parent interaction IDs per episode             |
| `_episode_turn_rewards`    | `list[list[float]]`     | Per-turn rewards per episode                   |
| `_episode_outcome_reward`  | `list[float]`           | Outcome reward per episode                     |

**Note on `concat_padded_tensors` behavior:** Non-tensor list values are
flat-concatenated by `concat_padded_tensors`. To preserve the nested structure,
episode-level metadata must NOT be stored as list keys in the per-episode dicts before
concat. Instead, `TreeSearchGroupedRolloutWorkflow` builds metadata *after* stacking by
iterating over the valid_results list and collecting per-episode metadata into the final
dict as a post-processing step (not relying on `concat_padded_tensors` to pass them
through).

**Variable turn counts:** Episodes with fewer turns are padded with zero rows.
`_episode_num_turns` and `_episode_turn_offsets` allow slicing back into per-episode
turn groups.

**Fallback for non-strict-prefix:** If a child turn's `input_len <= parent_len`,
`to_tensor_dict()` already handles this by masking out the parent turn and logging a
warning. The episode still has all turns, but the non-prefix turn gets
`loss_mask = [0...0, 1...1]` with no parent contribution. This is the existing behavior
and is acceptable.

### Component 2: `_cache_aware_prepare_batch` — Split into per-turn dicts

After `rollout_batch` returns a list of stacked dicts (one per query), split each into a
flat list of per-turn dicts before passing to `insert_batch`.

**Splitting logic:**

```python
def _split_to_turn_dicts(trajs):
    """Split stacked trajectory dicts into flat list of per-turn dicts."""
    flat = []
    for traj in trajs:
        total_turns = traj["input_ids"].shape[0]
        offsets = traj["_episode_turn_offsets"]
        num_turns_list = traj["_episode_num_turns"]

        for ep_idx, num_turns in enumerate(num_turns_list):
            start = offsets[ep_idx]
            end = start + num_turns
            for t in range(start, end):
                local_turn_idx = t - start
                turn_dict = {
                    k: v[t : t + 1] if isinstance(v, torch.Tensor) else v
                    for k, v in traj.items()
                    if k not in EPISODE_LEVEL_METADATA_KEYS
                }
                turn_dict["_mcts_query_id"] = traj["_mcts_query_id"]
                turn_dict["_episode_idx"] = ep_idx
                turn_dict["_turn_idx_in_episode"] = local_turn_idx
                turn_dict["_turn_id"] = traj["_episode_turn_ids"][ep_idx][local_turn_idx]
                turn_dict["_parent_turn_id"] = traj["_episode_parent_turn_ids"][ep_idx][local_turn_idx]
                turn_dict["_turn_reward"] = traj["_episode_turn_rewards"][ep_idx][local_turn_idx]
                turn_dict["_outcome_reward"] = traj["_episode_outcome_reward"][ep_idx]
                turn_dict["_num_turns_in_episode"] = num_turns
                flat.append(turn_dict)
    return flat
```

### Component 3: `MCTSTreeStore` changes

**`TrajectoryRecord` extension:**

```python
@dataclass
class TrajectoryRecord:
    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    reward: float                       # outcome reward (for Q-value)
    turn_response_starts: list[int]     # unchanged
    turn_response_ends: list[int]       # unchanged
    # New fields:
    turn_ids: list[str]                 # interaction ID per turn
    parent_turn_ids: list[str | None]   # parent interaction ID per turn
    turn_rewards: list[float]           # per-turn reward
    outcome_reward: float               # outcome reward (separate from reward)
```

**`insert_batch` changes:**

Group consecutive turn dicts with the same `_mcts_query_id` and `_episode_idx` into a
single episode record. For each episode:

1. Concatenate turn `input_ids`, `loss_mask`, `logprobs`, `versions` into a single
   sequence (concat-style within the episode)
1. Build `turn_response_starts`/`turn_response_ends` from `loss_mask` transitions
1. Set `reward = outcome_reward`, `outcome_reward = _outcome_reward`
1. Store `turn_ids`, `parent_turn_ids`, `turn_rewards` from metadata
1. Register `turn_id → seq_id` mappings in `_turn_nodes` for shared-node MCTS

**Shared-node MCTS support:**

```python
self._turn_nodes: dict[str, int]  # turn_id → seq_id

def _insert_single(self, query_id, record):
    seq_id = ...  # existing logic
    for turn_id in record.turn_ids:
        if turn_id not in self._turn_nodes:
            self._turn_nodes[turn_id] = seq_id
    return seq_id
```

When multiple episodes share a `turn_id` (same assistant response, different
continuations), the tree can trace shared ancestry through `_turn_nodes`.

**`get_advantages` changes:**

No change to signature. Returns per-token advantages: normalized Q-value on response
tokens, 0 on prompt tokens. The Q-value now comes from episode-level computation (see
Component 4).

**`load_trajectories` changes:**

Must reconstruct per-turn dicts from stored episode records. Each episode is split back
into per-turn dicts using `turn_response_starts`/`turn_response_ends` to slice the
stored sequence. Metadata keys (`_turn_id`, `_parent_turn_id`, `_turn_reward`,
`_outcome_reward`, etc.) are populated from `TrajectoryRecord`.

### Component 4: `TreeAdvantageComputer` with GRPO normalization

**`compute` changes:**

1. **Group episodes by `query_id`**: Collect all `seq_ids` for each query
1. **Compute episode Q-value**: `Q = outcome_reward` (from `_rewards[seq_id]`)
1. **Per-query GRPO normalization:**

```python
for query_id, seq_ids in query_groups.items():
    q_values = [self.tree_store._rewards[sid] for sid in seq_ids]
    mean_q = sum(q_values) / len(q_values)
    std_q = (sum((q - mean_q) ** 2 for q in q_values) / len(q_values)) ** 0.5
    for sid, q in zip(seq_ids, q_values):
        normalized_q = (q - mean_q) / (std_q + eps)
        self.tree_store._normalized_advantages[sid] = normalized_q
```

4. **Assign per-token advantages:** `normalized_q` on response tokens, 0 on prompt
   tokens (same structure as current `get_advantages`)

**`_compute_single` changes:**

Replace raw Q-value with normalized Q-value:

```python
def _compute_single(self, traj, query_id, seq_id, seq_len):
    normalized_q = self.tree_store._normalized_advantages.get(seq_id, 0.0)
    advantages = torch.zeros(seq_len, dtype=torch.float32)
    # Fill response token positions with normalized_q
    prompt_mask = self.tree_store.get_prompt_mask(query_id, seq_id)
    for start, end in zip(record.turn_response_starts, record.turn_response_ends):
        advantages[start:end] = normalized_q
    advantages = advantages * prompt_mask.float()
    return advantages
```

**`advantage_mode=GAE` passthrough:** When `advantage_mode=GAE`, the
`TreeAdvantageComputer.compute` is skipped entirely (existing behavior). GAE advantages
are used as-is. No normalization is applied by the tree component.

### Component 5: Trainer integration

**`CacheAwarePPOTrainer._init_patches` changes:**

Replace `_patch_wrap_openai_agent_for_query_id` with `TreeSearchGroupedRolloutWorkflow`
wrapping:

```python
def _init_patches(self):
    # Patch PPOActor.compute_advantages (unchanged)
    patch_ppo_actor_for_tree_backup(advantage_mode=self.tree_backup_config.advantage_mode)

    # Patch _wrap_openai_agent to use TreeSearchGroupedRolloutWorkflow
    _patch_wrap_openai_agent_for_tree_search(
        self.rollout,
        group_size=self.cache_config.n_samples,
    )
```

**New patch function:**

```python
def _patch_wrap_openai_agent_for_tree_search(rollout_engine, group_size):
    original_wrap = rollout_engine._wrap_openai_agent

    def _tree_search_wrap(agent, proxy_addr):
        from areal.api.cli_args import OpenAIProxyConfig
        openai_cfg = rollout_engine.config.openai or OpenAIProxyConfig()
        inner = original_wrap(agent, proxy_addr)
        return TreeSearchGroupedRolloutWorkflow(
            workflow=inner,
            group_size=group_size,
            logger=...,
        )

    rollout_engine._wrap_openai_agent = _tree_search_wrap
```

**`_cache_aware_prepare_batch` changes:**

After `rollout_batch`, split stacked dicts into flat per-turn dicts before tree
operations:

```python
new_trajs = self.actor.rollout_batch(...)
trajs = _split_to_turn_dicts(new_trajs)  # NEW
# Then proceed with existing tree operations:
self.tree_store.insert_batch(trajs)
...
```

## Data Flow Summary

```
OpenAIProxyWorkflow.arun_episode
  → dict[InteractionWithTokenLogpReward]  (per-turn interactions with parent chain)

TreeSearchGroupedRolloutWorkflow.arun_episode
  → to_tensor_dict() per turn → [1, seq_len] each
  → concat_padded_tensors → [total_turns, max_seq_len]
  → add episode metadata (_episode_num_turns, _episode_turn_ids, etc.)
  → returns single stacked dict

rollout_batch
  → list[stacked_dict]  (one per query)

_split_to_turn_dicts
  → list[per_turn_dict]  (flat, each [1, seq_len] with episode metadata)

insert_batch
  → groups turns by (_mcts_query_id, _episode_idx) → one TrajectoryRecord per episode
  → registers turn_id → seq_id mappings

TreeAdvantageComputer.compute
  → groups episodes by query_id
  → per-query GRPO normalization of Q-values
  → assigns normalized Q on response tokens as advantages

compute_advantages (patched)
  → GAE runs (overwrites advantages)
  → restores tree advantages from _tree_advantages (if TREE mode)
```

## Variable Turn Counts

The design handles variable turn counts naturally:

- `TreeSearchGroupedRolloutWorkflow` pads shorter episodes with zero rows and tracks
  per-episode turn counts in `_episode_num_turns`
- `_split_to_turn_dicts` uses offsets to skip padding rows
- `insert_batch` groups by episode, concatenating only real turns
- Advantage computation operates on episodes, not turns, so variable counts don't affect
  normalization

## Testing Plan

1. **Unit test: `TreeSearchGroupedRolloutWorkflow`** — mock
   `InteractionWithTokenLogpReward` with 2-turn and 3-turn episodes, verify output shape
   and metadata
1. **Unit test: `_split_to_turn_dicts`** — verify correct slicing with variable turn
   counts and padding
1. **Unit test: `insert_batch` with per-turn dicts** — verify episode grouping, one
   `seq_id` per episode, turn_id registration
1. **Unit test: GRPO normalization** — verify Q-values normalized within query groups,
   not across queries
1. **Unit test: variable turn counts** — 2 episodes with 2 turns + 1 episode with 3
   turns for same query, verify normalization is correct
1. **Unit test: shared turn_id** — two episodes sharing turn 0 but diverging at turn 1,
   verify `_turn_nodes` mapping

# Episode-Aware cached_count and Advantage Normalization

**Date:** 2026-05-13
**Status:** Draft

## Problem

`self.group_size` in `TreeSearchGroupedRolloutWorkflow` represents the number of
**episodes** needed for advantage computation and training. However,
`cached_count` is computed via `get_untrained_count(query_id)`, which counts
untrained **nodes** (individual turns). Since one episode contains multiple
nodes, the arithmetic `need_gen = group_size - cached_count` is semantically
wrong — it subtracts node counts from episode counts.

Similarly, `TreeAdvantageComputer.compute()` normalizes per-node: each node's
`outcome_reward` is normalized independently across all nodes in the query
group. The correct behavior is episode-level normalization: all nodes in the
same episode share one outcome_reward, and GRPO normalization operates across
episodes within the query group. The normalized return is then broadcast to
every node in the episode.

## Design

### 1. MCTSTreeStore — Episode-aware methods

Add two new methods that derive episode information from existing node data:

**`get_untrained_episode_count(query_id) -> int`**

- Scan all nodes for the query
- Group by `episode_id`
- Count episodes where at least one node has `train_id != current_train_id`
- An episode is "untrained" if any of its nodes is untrained
- Return 0 if query_id is not in the store

**`load_untrained_episodes(query_id, n_episodes) -> list[Node]`**

- Build mapping `episode_id → [node_ids]` from the query's nodes
- Filter to episodes where any node's `train_id != current_train_id`
- Take first `n_episodes` such episodes (insertion order)
- Return all nodes belonging to the selected episodes as a flat list

No new data structures required — these methods derive episode grouping from
existing `episode_id` attributes on Nodes.

### 2. TreeAdvantageComputer — Episode-level GRPO normalization

Change `compute()` to group by `(query_id, episode_id)` instead of
`(query_id, node_id)`:

1. Build `query_id → {episode_id → [node_ids]}` with one reward per episode
   (taken from the first node's `outcome_reward`; all nodes in an episode
   share the same reward)
2. For each `query_id`, collect per-episode rewards and compute GRPO
   mean/std across episodes
3. Each episode gets one `norm_return = (reward - mean) / (std + eps)`
4. Broadcast to all nodes: `node.advantages = mask.float() * norm_return`,
   `node.returns = mask.float() * norm_return`

Edge case: if a query has fewer than 2 episodes, normalized return is 0.0
(same as current behavior).

`set_normalized_return` / `get_normalized_return` on the store remain keyed
by `node_id` — all nodes in the same episode are set to the same value.

### 3. Workflow — Use episode-aware caching

Changes to `TreeSearchGroupedRolloutWorkflow.arun_episode()`:

- **Step 1**: Replace `get_untrained_count(query_id)` with
  `get_untrained_episode_count(query_id)`. Now `cached_count` = number of
  untrained episodes, making `need_gen = group_size - cached_count` correct.
- **Step 3**: Replace `load_trajectories(query_id, cached_count)` with
  `load_untrained_episodes(query_id, cached_count)`. Returns all nodes from
  `cached_count` untrained episodes.
- **Step 7**: Mark nodes as trained — unchanged. After training,
  `set_trained(node.node_id, True)` stamps each node with `current_train_id`,
  making the episode "trained" on next lookup.

## Files Changed

| File | Change |
|------|--------|
| `customized_areal/tree_search/mcts_tree_store.py` | Add `get_untrained_episode_count`, `load_untrained_episodes` |
| `customized_areal/tree_search/advantage.py` | Episode-level GRPO normalization in `compute()` |
| `customized_areal/tree_search/tree_search_grouped_workflow.py` | Use episode-aware methods in `arun_episode()` |

## Testing

- Unit tests for `get_untrained_episode_count`: verify correct counting with
  multiple episodes, partially-trained episodes, empty queries
- Unit tests for `load_untrained_episodes`: verify returns all nodes from
  selected episodes, respects `n_episodes` limit
- Unit tests for `TreeAdvantageComputer.compute()`: verify episode-level
  normalization matches expected GRPO values, all nodes in an episode get
  the same advantage

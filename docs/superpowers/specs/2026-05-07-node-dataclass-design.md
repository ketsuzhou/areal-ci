# Node Dataclass Design

**Date**: 2026-05-07 **Status**: approved

## Summary

Replace the ad-hoc `dict[str, Any]` trajectory representation and `TrajectoryRecord`
dataclass with a unified `Node` dataclass in `mcts_tree_store.py`. A `Node` represents a
single turn in a multi-turn conversation tree.

## Motivation

- Two representations (`dict` and `TrajectoryRecord`) carry the same data but with
  slightly different field names and structures, creating conversion overhead
- Dicts have no type safety; typos in string keys cause runtime errors
- `TrajectoryRecord` field names (`turn_ids`, `parent_turn_ids`) don't match the
  tree-search mental model (`node_id`, `parent_node_id`)
- `reward` and `outcome_reward` are always identical but both stored
- `response_ids` and `logp` are derivable from other fields

## Design

### Node Dataclass

```python
from dataclasses import dataclass

@dataclass
class Node:
    """A single turn in a multi-turn conversation tree."""

    # Core sequence (full turn: prompt + response)
    input_ids: list[int]
    loss_mask: list[int]              # 0=prompt, 1=response
    logprobs: list[float]             # full sequence (0.0 on prompt positions)
    versions: list[int]               # policy version (-1 on prompt)

    # Tree structure
    node_id: str                      # interaction ID for this turn
    parent_node_id: str | None        # parent interaction ID (None for root)
    episode_id: str                   # groups turns into a trajectory path

    # Reward
    outcome_reward: float = 0.0       # terminal/averaged reward

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
```

### What Was Removed

| Removed field     | Reason                                     |
| ----------------- | ------------------------------------------ |
| `reward`          | Same as `outcome_reward`                   |
| `turn_rewards`    | Derivative for single-turn node            |
| `response_ids`    | Derivable from `input_ids[-response_len:]` |
| `logp`            | Derivable from `logprobs[loss_mask==1]`    |
| `turn_ids`        | Renamed to `node_id`                       |
| `parent_turn_ids` | Renamed to `parent_node_id`                |
| `attention_mask`  | Always `[1]*len`, derived                  |

### What Was Renamed

| Old (`TrajectoryRecord`)           | New (`Node`)                | Why                           |
| ---------------------------------- | --------------------------- | ----------------------------- |
| `turn_ids: list[str]`              | `node_id: str`              | Single turn, tree terminology |
| `parent_turn_ids: list[str\|None]` | `parent_node_id: str\|None` | Single turn, tree terminology |

### What Is New

- `episode_id: str` — groups turns belonging to the same trajectory path

## Data Flow

```
InteractionWithTokenLogpReward
        │
        ▼
_interactions_to_turn_dicts()      → list[Node]  (per-turn)
        │
        ▼
_merge_turn_dicts_to_episode()     → list[Node]  (per-episode, merged turns)
        │
        ▼
MCTSTreeStore.insert_batch()       → stores Node objects
MCTSTreeStore.load_trajectories()  → returns list[Node]
        │
        ▼
TreeAdvantageComputer.compute()    → reads Node attrs, sets advantages
        │
        ▼
_list_dict_to_tensor()             → converts Node → tensor dict for PPO
```

### Storage Model

`MCTSTreeStore.trajectories` changes from `dict[str, list[TrajectoryRecord]]` to
`dict[str, list[Node]]`. Each query_id maps to a list of Node objects representing
completed trajectory paths. The tree structure is encoded via `node_id`/`parent_node_id`
links within each episode (grouped by `episode_id`).

## Implementation Plan

### File Changes

1. **`mcts_tree_store.py`**

   - Replace `TrajectoryRecord` dataclass with `Node`
   - Update `_make_record`, `_insert_list_dict`, `_insert_per_turn_dicts` to build
     `Node`
   - Update `load_trajectories` to return `Node` objects
   - Update `get_advantages`, `get_prompt_mask` to read `Node` attrs

1. **`proxy_workflow.py`**

   - `_interactions_to_turn_dicts` builds `Node` objects instead of dicts

1. **`grouped_workflow.py`**

   - `_merge_turn_dicts_to_episode` merges `Node` attributes into episode `Node`

1. **`advantage.py`**

   - Read `Node` attributes instead of dict keys

1. **`trainer.py`**

   - `_list_dict_to_tensor` converts `Node` → tensor dict
   - `_is_list_traj` checks for `Node` type

1. **`checkpoint.py`**

   - Serialize/deserialize `Node` fields instead of `TrajectoryRecord`

### Episode Reconstruction

Store individual per-turn `Node`s in the tree store. Reconstruct episodes by traversing
`node_id`/`parent_node_id` links grouped by `episode_id`. When loading trajectories for
training, concatenate linked `Node` sequences to produce the full episode tensor dict.

### Constraints

- `TrajectoryRecord` removed entirely — no backwards compat shim
- Node fields use Python lists, never tensors
- `outcome_reward` defaults to `0.0`
- All optional fields default to `None`

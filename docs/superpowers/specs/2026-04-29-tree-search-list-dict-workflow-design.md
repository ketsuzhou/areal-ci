# Tree Search List\[Dict\] Workflow Executor

## Problem

The tree search pipeline currently passes data as stacked tensor dicts (shape
`[num_turns, seq_len]`) through `arun_episode → concat_padded_tensors`. This requires
later splitting back into per-turn dicts for MCTS tree insertion, and the tensor format
doesn't align with `TrajectoryRecord` which stores unpadded Python lists.

The base `WorkflowExecutor` also asserts `traj is None or isinstance(traj, dict)`,
blocking `list[dict]` returns from `arun_episode`.

## Design

Change the tree search pipeline so that `arun_episode` returns `list[dict]` directly,
where each dict uses Python lists (matching `TrajectoryRecord`) instead of stacked
tensors. A custom `TreeSearchWorkflowExecutor` handles the new return type.

### Per-dict structure

Every dict at every level (per-turn, per-episode, per-query) has the same schema:

```python
{
    "input_ids": list[int],              # unpadded token IDs
    "loss_mask": list[int],              # 0=prompt, 1=response
    "logprobs": list[float],             # chosen token log prob per position
    "versions": list[int],               # policy version per token
    "reward": float,                     # outcome reward (for Q-value)
    "turn_response_starts": list[int],   # response start indices per turn
    "turn_response_ends": list[int],     # response end indices per turn
    "turn_ids": list[str],               # interaction ID per turn
    "parent_turn_ids": list[str | None], # parent interaction ID per turn
    "turn_rewards": list[float],         # per-turn reward
    "outcome_reward": float,             # outcome reward (separate from reward)
    "response_ids": list[int],           # chosen response token IDs
    "logp": list[float],                 # chosen token log probs
    "topk_ids": list[list[int]],         # top-k candidate token IDs per response position
    "topk_logp": list[list[float]],      # top-k candidate log probs per response position
    "distill_reward": list[list[float]], # per-response-position distillation reward
    "teacher_logp": list[list[float]],   # teacher log probs per response position (aligned with topk_ids)
}
```

- `response_ids` is the flat list of chosen response token IDs (input_ids where
  `loss_mask=1`). `logp` is the corresponding chosen token log probs.
- `topk_ids[i]` and `topk_logp[i]` are the top-k candidate tokens and their log probs at
  the i-th response position. `logprobs[j]` is the chosen token's log prob at position j
  (equivalent to `topk_logp[i][0]` when the chosen token is the top-1 candidate).
- `distill_reward[i]` is the distillation reward at the i-th response position.
- `teacher_logp[i]` is the teacher model's log probs over the top-k candidates at the
  i-th response position (aligned with `topk_ids[i]`).

Only `individual` export style is supported.

### Data flow

| Level       | Source                          | Returns      | Semantics                                                                                                           |
| ----------- | ------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------- |
| Per-turn    | `proxy_workflow.arun_episode`   | `list[dict]` | Each dict = one turn of one episode. Single-turn episodes have `len=1`. Multi-turn episodes have `len=N` (N turns). |
| Per-episode | `grouped_workflow.arun_episode` | `list[dict]` | Each dict = one complete episode (turns merged into episode-level lists). One dict per group_size episode.          |
| Per-query   | `rollout_batch`                 | `list[dict]` | Flat list of per-episode dicts across all queries.                                                                  |

### Components

#### 1. `TreeSearchWorkflowExecutor` (new file: `customized_areal/tree_search/workflow_executor.py`)

Subclasses `WorkflowExecutor`. Overrides:

- **`_create_workflow_task`**: Accept `list[dict]` from `arun_episode`. Skip
  `InteractionWithTokenLogpReward` conversion (already handled in workflow). Skip
  `assert isinstance(traj, dict)`. Store result in
  `_TreeSearchRolloutResult.trajectories: list[dict]`.

- **`rollout_batch`**: Override to flatten `list[list[dict]]` → `list[dict]` from the
  wait results.

- **`_TreeSearchRolloutResult`** (new dataclass):

  ```python
  @dataclass
  class _TreeSearchRolloutResult:
      task_id: int
      trajectories: list[dict[str, Any]]
  ```

- **`wait`**: Override to extract `trajectories` from `_TreeSearchRolloutResult` instead
  of `trajectory` from `_RolloutResult`.

#### 2. `proxy_workflow.py` changes

`QueryIDProxyWorkflow.arun_episode` returns `list[dict] | None` instead of
`dict | None`.

For each `InteractionWithTokenLogpReward` in the result:

- Convert to a per-turn dict with Python lists
- Compute `turn_response_starts`, `turn_response_ends` from `loss_mask`
- Extract `turn_ids`, `parent_turn_ids`, `turn_rewards` from the interaction
- Extract `response_ids` (chosen response token IDs) and `logp` (chosen token log probs)
  from the interaction's data
- Extract `topk_ids` and `topk_logp` (top-k candidate tokens and log probs per response
  position) from the interaction's token logp data
- Extract `distill_reward` (per-response-position distillation reward)
- Set `reward` and `outcome_reward`
- Inject `_mcts_query_id` into each dict

#### 3. `grouped_workflow.py` changes

`TreeSearchGroupedRolloutWorkflow.arun_episode` returns `list[dict] | None` instead of
`dict | None`.

- Calls inner workflow's `arun_episode` `group_size` times
- Each inner call returns `list[dict]` (per-turn dicts from proxy_workflow)
- Merges per-turn dicts into per-episode dicts: combine `input_ids`, `loss_mask`,
  `logprobs`, `versions` lists; compute episode-level `turn_response_starts/ends`,
  `turn_ids`, `parent_turn_ids`, `turn_rewards`, `outcome_reward`
- Returns flat `list[dict]` (one dict per episode)

#### 4. `trainer.py` changes

- Patch `_wrap_openai_agent` to also swap the engine's `workflow_executor` with a
  `TreeSearchWorkflowExecutor`
- Remove `_split_to_turn_dicts` call (no longer needed — data is already per-episode
  dicts)
- Remove `_split_to_turn_dicts` function if no longer used
- `MCTSTreeStore.insert_batch` and `TreeAdvantageComputer.compute` receive per-episode
  dicts directly (may need minor adjustments for Python lists vs tensors)

#### 5. `mcts_tree_store.py` changes

**`TrajectoryRecord`** gains new optional fields:

```python
@dataclass
class TrajectoryRecord:
    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    reward: float
    turn_response_starts: list[int]
    turn_response_ends: list[int]
    turn_ids: list[str] | None = None
    parent_turn_ids: list[str | None] | None = None
    turn_rewards: list[float] | None = None
    outcome_reward: float = 0.0
    # New fields:
    logp: list[float] | None = None                    # chosen token log probs (response only)
    topk_ids: list[list[int]] | None = None            # top-k candidate token IDs per response position
    topk_logp: list[list[float]] | None = None         # top-k candidate log probs per response position
    distill_reward: list[list[float]] | None = None     # per-response-position distillation reward
    teacher_logp: list[list[float]] | None = None       # teacher log probs per response position (aligned with topk_ids)
```

**`insert_batch`**: When receiving per-episode dicts with Python lists, extract `logp`,
`topk_ids`, `topk_logp`, `distill_reward`, `teacher_logp` directly from the dict (no
tensor conversion needed). When receiving legacy tensor dicts, these fields default to
`None`.

**`_insert_per_turn_dicts`**: Merge per-turn `logp`/`topk_ids`/`topk_logp`/
`distill_reward`/`teacher_logp` lists when reconstructing episodes from per-turn dicts
(concatenate response-only lists across turns, adjusting
`topk_ids`/`topk_logp`/`distill_reward`/`teacher_logp` indices to align with the full
episode sequence).

**`load_trajectories`**: When splitting an episode back into per-turn dicts, slice
`logp`/`topk_ids`/`topk_logp`/`distill_reward`/`teacher_logp` by turn boundaries
(`turn_response_starts`/`turn_response_ends`).

### Out of scope

- Changes to the base `WorkflowExecutor` or `RolloutWorkflow` contract
- `concat` export style support (individual only)
- Changes to `workflow_executor._dump_trajectory` (not used in tree search)

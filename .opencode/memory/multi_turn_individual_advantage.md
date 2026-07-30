# AReaL Multi-Turn Agent with `individual` Export Style — Memory

## Context

When using multi-turn agent workflows (e.g., `MultiturnRLVRWorkflow`) with
`export_interactions(style="individual")`, the relationship between rollout episodes,
turns, and GRPO groups becomes non-trivial. This document captures the exact data flow
and identifies pitfalls in advantage computation.

______________________________________________________________________

## 1. Data Structure of `rollout_batch`

> **前提条件**：本 Section 描述的是 **多轮交互 agent**、**每轮前缀不同**、且使用
> **`export_interactions(style="individual")`** 的情况。
>
> 如果各轮共享前缀（strict prefix），应使用 `style="concat"`，此时一个 episode 会被拼接为单条序列，数据结构完全不同（见 Section
> 4）。

### Source: `GroupedRolloutWorkflow.arun_episode`

因为每轮前缀不同，无法使用 `concat` 模式将多轮拼接成一条序列。`individual` 模式下，每轮交互被当作独立的 completion/response 返回。

```python
# areal/infra/remote_inf_engine.py:86
results = await asyncio.gather(
    *[self.workflow.arun_episode(engine, data) for _ in range(group_size)]
)
# Merges all InteractionWithTokenLogpReward dicts
merged.update(result)  # key = completion/response ID
```

### Source: `WorkflowExecutor._execute_workflow`

```python
# areal/infra/workflow_executor.py:1069
if isinstance(traj, dict) and all(isinstance(v, InteractionWithTokenLogpReward) for v in traj.values()):
    traj = concat_padded_tensors(
        [v.to_tensor_dict() for v in traj.values()]
    )
```

`InteractionWithTokenLogpReward.to_tensor_dict()` returns tensors with shape
**`[1, seqlen]`** (via `unsqueeze(0)`). `concat_padded_tensors` concatenates along **dim
0 (batch dimension)**.

### Result

`rollout_batch` is a `list[dict[str, tensor]]` where:

- **List length** = query batch size (e.g., 2 queries)
- **Dict `batch_size`** = `group_size × num_turns` for that query

Example with `group_size=4`:

- Query 1 (2 turns): `batch_size = 4 × 2 = 8`
- Query 2 (3 turns): `batch_size = 4 × 3 = 12`

______________________________________________________________________

## 2. `compute_advantages` Behavior

### Source: `PPOActor._compute_advantages`

```python
# areal/trainer/ppo/actor.py:136
bs = data["input_ids"].shape[0]  # query1: 8, query2: 12
reward_score = data["rewards"]   # shape [8] or [12]
```

- `batched_call` invokes `_compute_advantages` **independently** for each dict in
  `rollout_batch`
- Each sequence gets its own reward injected at the EOS position
- GAE is computed **per-sequence**; there is **no temporal bootstrap across turns**
- Turns are treated as independent samples in the same batch

______________________________________________________________________

## 3. GRPO Group Normalization Pitfall

### Source: `Normalization.__call__` with `mean_level="group"`

```python
# areal/utils/data.py:1404
for i in range(0, bs // self.group_size):
    s = slice(i * self.group_size, (i + 1) * self.group_size)
    # Computes mean/std within this slice
```

### The Problem

The slice assumes that every `group_size` consecutive samples belong to the same GRPO
group. However, with `individual` export style, the batch ordering is:

```
[sample0-turn0, sample0-turn1, sample1-turn0, sample1-turn1, ...]
```

**Consequences:**

1. **Wrong grouping**: A single "group" may contain different samples AND different
   turns
   - Example: indices `[0:4]` = sample0-turn0, sample0-turn1, sample1-turn0,
     sample1-turn1
   - These are not the same GRPO group!
1. **Broken semantics**: GRPO normalization should aggregate over all responses to the
   **same query**, but the fixed-size slice breaks this

### Variable Turn Counts: Even Worse

If different samples terminate after different numbers of turns (e.g., some agents get
reward=1 early):

- Sample 0: 2 turns
- Sample 1: 1 turn
- Sample 2: 3 turns

Total batch size = 6. With `group_size=4`:

- Group 0: indices `[0:4]` (4 samples)
- Group 1: indices `[4:6]` (2 samples) — **only 2 remain, but loop runs
  `bs // group_size = 1` time**
- The remaining 2 samples **never get group-normalized** (fall back to default mean=0)

______________________________________________________________________

## 4. The Strict Prefix Requirement in `concat` Mode

When using `export_interactions(style="concat")`, AReaL builds a conversation tree by
linking parent-child interactions whose input messages form a **strict prefix**
relationship (`areal/experimental/openai/cache.py:100`). The child turn must contain all
tokens from the parent turn as its prefix.

### What Happens When Prefixes Are Not Shared

**Source: `InteractionWithTokenLogpReward.to_tensor_dict()`**

```python
# areal/experimental/openai/types.py:143
if self.chat_template_type == "concat" and self.parent is not None:
    parent_len = len(parent_logprobs)
    if resp.input_len > parent_len:
        # Normal case: child input contains parent as prefix
        logprobs = parent_logprobs + [0.0] * (resp.input_len - parent_len) + resp.output_logprobs
        loss_mask = parent_loss_mask + [0] * (resp.input_len - parent_len) + [1] * resp.output_len
    else:
        # CHILD INPUT DOES NOT CONTAIN PARENT AS PREFIX
        logger.warning(
            "Ignoring the parent ... by masking them out. \n"
            "Parent input token ids: ...\n"
            "Child input token ids: ..."
        )
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
```

**Behavior:**

- If `resp.input_len <= parent_len` (child does not extend parent), AReaL **discards the
  parent turn entirely**
- Parent tokens are replaced with `logprobs = [0.0] * resp.input_len` and
  `loss_mask = [0] * resp.input_len`
- Only the current (child) turn's output tokens get `loss_mask = 1`

### Why Strict Prefix Is Required

LLM causal attention assumes a **linear token sequence**. If turn N does not contain
turn N-1 as a strict prefix:

1. **Cannot represent in a single `input_ids` tensor**: There is no valid causal mask
   where turn N attends to a subset of turn N-1's tokens while both coexist in the same
   sequence
1. **Position encoding breaks**: Positional embeddings assume contiguous token positions
1. **Gradient flow is undefined**: Backpropagating through attention across non-prefix
   boundaries is mathematically invalid

### Example Scenario

**Turn 1:** User asks "What is 2+2?" **Turn 2:** System adds "Let me think..." then
model generates "The answer is 4"

If Turn 2's input does NOT start with Turn 1's full token sequence (e.g., because the
system injected a new prefix, or the chat template added special tokens differently),
the `concat` mode will:

- Log a warning
- Mask out Turn 1 completely (`loss_mask = 0`)
- Only train on Turn 2's output tokens

**Result:** The "multi-turn" episode degenerates into training only the last turn,
losing all credit assignment for earlier reasoning steps.

______________________________________________________________________

## 5. Summary of Issues

| Issue                                       | Impact                                                                  | Location                                 |
| ------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------- |
| Group normalization slices wrong samples    | Advantage scale is computed over unrelated turns/samples                | `areal/utils/data.py:1406`               |
| Variable turn counts per sample             | Some samples escape group normalization entirely                        | `areal/utils/data.py:1406`               |
| No cross-turn bootstrap in GAE              | Multi-turn credit assignment is lost; each turn optimized independently | `areal/trainer/ppo/actor.py:215`         |
| `individual` mode flattens turns into batch | Cannot distinguish "group" vs "turn" in batch dimension                 | `areal/infra/workflow_executor.py:1072`  |
| Non-prefix `concat` falls back to masking   | Parent turns are silently masked out, losing multi-turn signal          | `areal/experimental/openai/types.py:166` |

______________________________________________________________________

## 6. Recommendations

### If you need strict GRPO semantics with multi-turn agents:

1. **Use `concat` export style** and ensure strict prefix relationships between turns

   - This preserves the episode as a single sequence
   - GAE bootstraps across turns naturally
   - Group normalization applies to the full batch correctly

1. **If `individual` is required**, do NOT use `mean_level="group"` for advantage
   normalization

   - Use `mean_level="batch"` or disable normalization (`mean_level=None`)
   - Group-level GRPO advantage computation is fundamentally incompatible with flattened
     individual turns

1. **For proper per-group advantage with `individual`**

   - AReaL currently has no mechanism to tag which batch indices belong to which query
     group
   - Would require adding metadata (e.g., `group_id` tensor) to each trajectory and
     modifying `Normalization` to respect it

### If turns do NOT share strict prefixes:

4. **Do NOT use `concat` export style**

   - AReaL will silently mask out parent turns (see Section 4)
   - The episode degenerates into single-turn training with a warning log
   - Use `individual` style instead, accepting that cross-turn GAE is lost

1. **Manually ensure strict prefix in your workflow**

   - If you control the conversation construction, append new messages to the previous
     message list without reformatting
   - Example: `MultiTurnWorkflow` manually appends tokens:
     `input_ids = input_ids + resp.output_tokens + multi_turn_prompt_ids`
   - Avoid changing chat templates or inserting system messages between turns

1. **Consider custom advantage computation**

   - For prefix-free multi-turn episodes, standard GAE cannot bootstrap across turns
   - You may need custom reward shaping (e.g., `apply_reward_discount`) so each turn
     gets a meaningful reward
   - Or implement a tree-based advantage estimator that respects the conversation graph
     structure

______________________________________________________________________

## Related Code Paths

- `areal/infra/remote_inf_engine.py:68` — `GroupedRolloutWorkflow`
- `areal/infra/workflow_executor.py:1069` — `InteractionWithTokenLogpReward` → tensor
  conversion
- `areal/experimental/openai/types.py:137` — `to_tensor_dict()` implementation (prefix
  handling at line 143)
- `areal/experimental/openai/cache.py:100` — Parent-child relationship building via
  strict prefix check
- `areal/trainer/ppo/actor.py:133` — `compute_advantages`
- `areal/utils/data.py:1360` — `Normalization` class

______________________________________________________________________

*Created: 2026-04-28* *Applies to: AReaL multi-turn workflows with both
`export_interactions(style="individual")` and `export_interactions(style="concat")`
where strict prefix relationships may not hold*

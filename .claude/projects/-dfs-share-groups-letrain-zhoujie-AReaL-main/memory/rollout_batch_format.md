---
name: Rollout Batch Data Format
description: rollout_batch return format: single-turn, multi-turn (shared prefix), and agent (no shared prefix) trajectories in AReaL
type: project
---

## rollout_batch Return Format

`rollout_batch` returns `list[dict[str, Any]]` — one dict per accepted rollout.

### Single-turn (RLVRWorkflow)

Each dict, shape `[1, seq_len]`:

- `input_ids`: int32 — prompt + generated tokens
- `attention_mask`: bool — all 1s
- `loss_mask`: int32 — 0 for prompt, 1 for generated
- `logprobs`: float32 — 0.0 for prompt, model logprob for generated
- `versions`: int32 — -1 for prompt, policy version for generated
- `rewards`: float32 — scalar reward broadcast to sequence

### Multi-turn with shared prefix (MultiTurnWorkflow)

Same keys as single-turn, but `input_ids` = prompt + turn1_output + turn2_prompt +
turn2_output + ...

- `loss_mask`: 0 for all prompt segments (including multi-turn prompts), 1 for all
  outputs
- `logprobs`: 0.0 for prompt segments, real logprobs for outputs
- `rewards`: scalar with `turn_discount` applied

All turns concatenated into one flat sequence because each turn's input_ids is a strict
superset of the previous.

### Agent multi-turn via OpenAI Proxy (InteractionWithTokenLogpReward)

**Two export styles:**

**style="concat"** (default, requires prefix compatibility):

- Only leaf nodes (last turn) returned
- `to_tensor_dict()` concatenates parent's logprobs/loss_mask/versions
- Final shape: `[1, total_len]` same as MultiTurnWorkflow
- If child's `input_len <= parent_len` (prefix broken), parent data is **silently
  dropped** with a warning — degrades to individual behavior

**style="individual"** (no prefix requirement):

- Each turn returned as independent `InteractionWithTokenLogpReward`
- `to_tensor_dict()` produces `[1, turn_seq_len]` per turn, no parent concatenation
- `concat_padded_tensors` stacks turns along dim=0 → `[n_turns, max_turn_seq_len]`
- Each turn has its own reward

### After \_compute_advantages, additional keys are added:

- `advantages`, `returns` — GAE values
- `kl_rewards`, `tot_rewards` — KL penalty and total reward
- `ref_logp`, `prox_logp` — reference/proximal policy logprobs (if computed)

### Key source files:

- `areal/workflow/rlvr.py` — single-turn
- `areal/workflow/multi_turn.py` — multi-turn shared prefix
- `areal/experimental/openai/types.py` — InteractionWithTokenLogpReward.to_tensor_dict()
- `areal/experimental/openai/cache.py` — export_interactions (concat vs individual)
- `areal/infra/workflow_executor.py:1068-1074` — InteractionWithTokenLogpReward →
  concat_padded_tensors conversion
- `areal/infra/remote_inf_engine.py:68-120` — GroupedRolloutWorkflow (handles both
  tensor dict and InteractionWithTokenLogpReward paths)
- `areal/utils/data.py:238-295` — concat_padded_tensors (pad + cat dim 0 for tensors,
  flat-concat for lists, first-dict-wins for scalars)

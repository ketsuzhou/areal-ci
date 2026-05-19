# Selected-Turn Tree Distillation Design

## Goal

Add selected-turn teacher distillation to
`TreeSearchGroupedRolloutWorkflow`. When `LossMode` is `distill` or `both`,
each episode is diagnosed once using the last node context and the gold answer.
The diagnosis selects turns whose assistant generation can be improved and
provides turn-wise guidance. Only selected turns receive teacher logprob
evaluation and distillation loss.

The default teacher path is an external OpenAI-compatible API for Qwen-397B.
The implementation should also define an engine-backed provider interface so
future runtimes can supply diagnosis and logprob calls through the inference
engine passed to `arun_episode`.

## Architecture

`TreeSearchGroupedRolloutWorkflow` owns the distill orchestration. The new
stage runs after rollout results are converted to `Node` objects and before
nodes are inserted into the tree store or converted to the training
`result_dict`.

Add a class-level tokenizer cache on `TreeSearchGroupedRolloutWorkflow`, keyed
by `tokenizer_path`. The tokenizer is loaded lazily with `load_hf_tokenizer()`
and shared by workflow instances in the same process. A lock protects the first
load for each path. A missing `tokenizer_path` is an error only when
`loss_mode != LossMode.GRPO`.

Add a small teacher provider interface with:

- `diagnose_episode(...)`: returns turn-wise improvement guidance.
- `get_logprobs_for_prompt(...)`: returns teacher logprobs for candidate token
  ids over the selected generation span.

The external provider wraps `customized_areal.tree_search.core.teacher_client`.
The engine provider is interface-first: initialize it only if the current engine
exposes compatible logprob methods; otherwise fail early with a clear
`NotImplementedError`.

## Diagnosis

For each episode, decode the last node as the full episode context and combine
it with `data["answer"]` as the gold answer. The diagnosis request asks the
teacher to identify which turns need better assistance generation and to return
strict JSON:

```json
{
  "turns": [
    {"turn_idx": 1, "should_improve": true, "guidance": "..."},
    {"turn_idx": 2, "should_improve": false, "guidance": ""}
  ]
}
```

Only entries with `should_improve=true` and non-empty `guidance` are selected
for distillation. Guidance is turn-wise. There is no shared episode-level
guidance in the training prompt.

If diagnosis JSON parsing fails:

- `LossMode.BOTH`: keep the episode for GRPO/tree training and omit distill
  metadata for that episode.
- `LossMode.DISTILL`: discard that episode from the returned batch.

## Selected-Turn Prompt And Span

For each selected turn, split the node by `loss_mask`:

- `student_each_turn_prefix`: tokens before the selected assistant generation.
- `student_each_turn_generation`: tokens in the current turn's contiguous
  response span. In concat-mode nodes, parent assistant spans can also have
  `loss_mask == 1`, so use the latest contiguous response span for the selected
  node.

The teacher evaluation context is:

```text
{student_each_turn_prefix}
{turn_guidance}
{student_each_turn_generation}
```

Only `student_each_turn_generation` contributes distill positions. Prefix and
guidance tokens are context only and must not contribute to `position_rewards`
or KL loss.

## `topk_distill`

Add `topk_distill: bool`, default `false`.

When `topk_distill=false`, each selected generation position has one candidate:
the generated token. The teacher scores that generated token sequence under the
teacher context.

When `topk_distill=true`, each selected generation position uses the generated
token at candidate index 0 plus student top-k alternatives. If a node already
has `topk_ids/topk_logp`, reuse them. If they are missing, recompute student
top-k for the selected generation span through the inference engine, then save
`topk_ids/topk_logp` on the node. The teacher then scores those candidate token
ids and the workflow saves `teacher_logp` and `distill_reward` on the node.

Cached nodes participate in distillation. If cached nodes are missing teacher
metadata, evaluate the selected turns and persist `teacher_logp` and
`distill_reward`. If `topk_distill=true` and cached nodes are missing student
top-k metadata, recompute and persist `topk_ids/topk_logp` before teacher
evaluation.

## Result Dict Contract

The workflow returns `result_dict["position_rewards"]` as a Python list of
`PositionRewardInfo`-compatible objects for selected turn positions only. Extend
the type or add a sibling type so each position carries explicit
`teacher_logprobs` aligned with `candidate_token_ids`. The existing `rewards`
field may continue to store `student_logp - teacher_logp` for compatibility and
inspection, but training must read teacher logprobs directly for the KL loss.
`sample_index` must match the row index of the corresponding node in the final
batched tensor dict after any filtering and fresh/cached combination.

For `LossMode.DISTILL`, episodes without valid distill metadata are filtered
out. If all episodes are filtered, `arun_episode` returns `None`.

For `LossMode.BOTH`, episodes without valid distill metadata remain in the
batch and contribute only GRPO/tree loss.

The workflow still injects:

- `rl_loss_weight = 0.0` for `LossMode.DISTILL`.
- configured `rl_loss_weight` for `LossMode.BOTH`.
- configured `distill_loss_weight` for both distill modes.

## Loss Behavior

Update `grpo_distill_loss_fn` from the current position-level GRPO reward
objective to a direct teacher-logprob KL-style objective.

For `LossMode.BOTH`, effective loss is:

```python
loss = grpo_loss + distill_loss_weight * kl_loss
```

For `LossMode.DISTILL`, effective loss is:

```python
loss = distill_loss_weight * kl_loss
```

`kl_loss` applies only to selected generation positions that have valid teacher
logprobs:

```python
kl_loss = mean(student_logp - teacher_logp)
```

`student_logp` is the current train-time model logprob with gradients.
`teacher_logp` is a detached constant from the teacher provider. Turns without
`position_rewards` contribute no KL term. In `topk_distill=true`, the same
formula applies to the candidate rows gathered by the training engine, with
teacher logprobs aligned by row. In `topk_distill=false`, each position has a
single candidate row for the generated token.

## Config

Read these env/config values near the existing tree-search env parsing in
`areal/infra/remote_inf_engine.py` and pass them into
`TreeSearchGroupedRolloutWorkflow`:

- `TREE_SEARCH_TOPK_DISTILL`, default `false`.
- `TREE_SEARCH_TEACHER_PROVIDER`, default `external`, optional `engine`.
- `TREE_SEARCH_TEACHER_BASE_URL`.
- `TREE_SEARCH_TEACHER_MODEL_NAME`.
- `TREE_SEARCH_TEACHER_TOP_K`.
- `TREE_SEARCH_TEACHER_MAX_RETRIES`.
- `TREE_SEARCH_TEACHER_TIMEOUT`.
- `TREE_SEARCH_TEACHER_MISSING_LOGPROB`.
- `TREE_SEARCH_DIAGNOSE_MODEL_NAME`, default teacher model if unset.
- `TREE_SEARCH_DIAGNOSE_MAX_TOKENS`.
- `TREE_SEARCH_DIAGNOSE_TEMPERATURE`, default `0.0`.
- `TREE_SEARCH_STRICT_DISTILL_JSON`, default `true`.

## Error Handling

Teacher and diagnosis failures are contained at episode granularity. Logs should
include query id, episode id, turn index when available, provider type, and
failure reason.

Failure policy:

- `BOTH`: keep failed episodes without distill entries.
- `DISTILL`: drop failed episodes.

The rollout should not raise a teacher failure that kills the entire batch
unless all episodes are filtered and the workflow returns `None`.

## Tests

Add focused unit tests for:

- class-level tokenizer cache reuses the tokenizer for a path;
- diagnosis JSON parsing selects only `should_improve=true` turns and preserves
  turn-wise guidance;
- teacher prompt construction masks loss to only the selected generation span;
- `topk_distill=false` creates single-candidate position rewards;
- `topk_distill=true` reuses cached `topk_ids/topk_logp`;
- `topk_distill=true` recomputes missing student top-k and persists it on
  `Node`;
- direct KL loss uses explicit teacher logprobs rather than normalized
  position-level rewards;
- cached nodes missing teacher metadata are evaluated and updated;
- `BOTH` keeps failed episodes without distill entries;
- `DISTILL` drops failed episodes;
- `position_rewards.sample_index` matches final batch rows after filtering and
  combining fresh/cached nodes.

## Out Of Scope

This design does not change tree advantage computation, cache episode
selection, public dataset formats, or non-tree-search PPO behavior. It also does
not require a complete engine-provider implementation if the current engine API
does not expose the necessary top-k/logprob methods; the provider boundary is
added so that support can be implemented cleanly when available.

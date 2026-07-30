# On-Policy Distillation Training Pipeline: Bug and Instability Analysis

## Summary

Analysis of `customized_areal/on_policy_distill/training/trainer.py` and its full
pipeline identified 12 bugs. 2 are critical (crash every run), 3 are high-severity
(intermittent crashes or cascading failures), 3 are medium-severity instability risks,
and 4 are correctness/performance issues.

## Scope

- **Focus**: Training crashes and exceptions
- **Pipeline coverage**: `OnPolicyDistillationTrainer` → `TokenRewardRolloutController`
  → `OpenAIProxyWorkflow` → `OnPolicyDistillAgent` → proxy server →
  `MultiCandidateFSDPPPOActor` → `grpo_distill_loss_fn`
- **Files analyzed**:
  - `customized_areal/on_policy_distill/training/trainer.py`
  - `customized_areal/on_policy_distill/training/actor.py`
  - `customized_areal/on_policy_distill/training/loss.py`
  - `customized_areal/on_policy_distill/training/logprobs.py`
  - `customized_areal/on_policy_distill/engine/fsdp_engine.py`
  - `customized_areal/on_policy_distill/proxy/server.py`
  - `customized_areal/on_policy_distill/proxy/client.py`
  - `customized_areal/on_policy_distill/proxy/workflow.py`
  - `customized_areal/on_policy_distill/proxy/cache.py`
  - `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`
  - `customized_areal/on_policy_distill/core/agent.py`
  - `customized_areal/on_policy_distill/core/reward_compute.py`

______________________________________________________________________

## Critical Bugs (Crash Every Run)

### Bug 1: Agent generates wrong `completion_id` — teacher distillation always fails

**File**: `core/agent.py:216-218` **Severity**: CRITICAL

`OnPolicyDistillAgent.run()` generates `completion_id` as an MD5 hash of
`completion_messages`:

```python
completion_id = hashlib.md5(str(completion_messages).encode()).hexdigest()[:16]
```

The proxy server's interactions use IDs like `chatcmpl-xxx` assigned by the inference
engine. When `_process_rewards` (workflow.py:211-232) calls
`client.set_position_rewards(completion_id, ...)` and
`client.set_reward(completion_id, ...)`:

1. **`set_position_rewards`** (proxy_rollout_server.py:1043-1049): Does NOT validate
   `interaction_id` against the completions cache. Stores rewards under a key that never
   matches any interaction. Rewards silently lost.

1. **`set_reward`** (proxy_rollout_server.py:759-763): DOES validate `interaction_id`
   and raises `HTTPException(400)`. This crashes the agent.

**Result**: Every training step with teacher distillation fails with HTTP 400. Position
rewards are also silently discarded.

**Fix**: Use the actual interaction ID from the proxy server:

```python
interaction = await proxy_client.get_last_interaction()
if interaction is not None:
    completion_id = interaction.interaction_id
```

______________________________________________________________________

### Bug 2: `export_interactions` called after session end — may fail

**File**: `proxy/workflow.py:275-297` **Severity**: HIGH (intermittent)

`arun_episode` calls `client.export_interactions()` OUTSIDE the `async with client:`
block. The context manager exit calls `end_session()`, marking the session as completed.
The server's `export_trajectories` (proxy_rollout_server.py:676-712) waits for
`wait_for_finish()`, exports, then **removes the session from cache** (line 708). If the
stale session cleanup task runs between `end_session` and `export_trajectories`, the
export fails with `HTTPException(404, "Session not found")`.

**Result**: Intermittent crashes during training, especially under load.

**Fix**: Move `export_interactions` inside the `async with client:` block before session
end, or make the server defer session removal to the cleanup task instead of removing
immediately on export.

______________________________________________________________________

## High-Severity Instability Risks

### Bug 3: `threading.Lock` in async server — event loop blocking

**File**: `proxy/server.py:112`, `proxy/proxy_rollout_server.py:219` **Severity**: HIGH
(cascading timeouts)

`TokenRewardSessionData` uses `threading.Lock()` inside an async FastAPI server. When
the lock is contended, the async event loop blocks, stalling all concurrent requests.
Under high concurrency (multiple rollout workers setting rewards simultaneously), the
proxy server becomes unresponsive.

**Result**: Rollout timeouts in the trainer that look like `Rollout timed out after Xs`
(trainer.py:178).

**Fix**: Replace `threading.Lock()` with `asyncio.Lock()` for session-internal
operations. Note: the global `_lock` must remain `threading.Lock` since it's used by
both sync and async code paths.

______________________________________________________________________

### Bug 4: Stale session cleanup uses wrong timeout constant

**File**: `proxy/proxy_rollout_server.py:1118` **Severity**: HIGH (OOM risk)

`_cleanup_stale_sessions()` uses hardcoded `SESSION_TIMEOUT_SECONDS` (3600s) instead of
the configured `_session_timeout_seconds`:

```python
# Line 1118: uses wrong constant
if session.is_stale(SESSION_TIMEOUT_SECONDS):
# Should be:
if session.is_stale(_session_timeout_seconds):
```

**Result**: Custom timeout configurations are silently ignored. Sessions accumulate in
memory, causing OOM on long training runs.

**Fix**: One-line change: replace `SESSION_TIMEOUT_SECONDS` with
`_session_timeout_seconds`.

______________________________________________________________________

### Bug 5: `_total_reward` drift from fragile save/restore pattern

**File**: `proxy/server.py:144-148` **Severity**: MEDIUM-HIGH

`TokenRewardSessionData.set_token_rewards()` saves the scalar reward, calls
`set_rewards()` (which overwrites `interaction.reward` and updates `_total_reward`),
then restores the scalar reward. The restore
(`self.completions[interaction_id].reward = saved_reward`) does NOT update
`_total_reward`, so it drifts from reality permanently.

If `set_rewards()` raises between the overwrite and restore, the scalar reward is
corrupted with no recovery path.

**Result**: Reward statistics diverge. Downstream code depending on `_total_reward`
(e.g., `total_reward` property used by reward logging) produces incorrect values.

**Fix**: Modify `InteractionCache.set_rewards()` to accept `preserve_scalar_reward=True`
option that skips scalar reward update entirely, eliminating the save/restore pattern.

______________________________________________________________________

## Medium-Severity Instability Risks

### Bug 6: `_distribute_position_rewards` — `None` entries silently drop data

**File**: `training/actor.py:108-143`

When `position_rewards` contain a `sample_index` that doesn't map to any minibatch
(because `forward_indices` leaves entries as `None`), those position_rewards are
silently dropped (line 135: `if mb_i is not None`). If all position_rewards for a
minibatch are dropped, the distillation loss returns `torch.tensor(0.0, ...)` — a
non-leaf tensor that may cause gradient computation shape issues.

**Fix**: Add a warning when `mb_assignment[orig_idx]` is `None`. Validate that every
`position_rewards` entry has a valid `sample_index`.

______________________________________________________________________

### Bug 7: `PositionRewardInfo` dual definition — dataclass vs Pydantic

**Files**: `proxy/cache.py:44` (dataclass) vs `proxy/server.py:44` (Pydantic)

Same type defined twice with different base classes. The server endpoint converts
Pydantic → dataclass at proxy_rollout_server.py:1052-1066, but any behavioral
differences (default values, validation, missing fields) could cause type mismatches
downstream.

**Fix**: Consolidate into a single canonical definition (dataclass). Create a Pydantic
adapter (`PositionRewardInfoAPI`) only for HTTP serialization.

______________________________________________________________________

### Bug 8: `model_inputs` in-place mutation — potential race condition

**File**: `engine/fsdp_engine.py:302-313`

`_compute_logprobs_and_loss` temporarily mutates `ctx.model_inputs` to set
multi-candidate labels, then restores in `finally`. If minibatch processing overlaps
(gradient accumulation with async prefetching), this shared-state mutation could cause
one step to see wrong labels.

**Fix**: Pass `multi_candidate_labels` as a separate parameter instead of mutating
`ctx.model_inputs`.

______________________________________________________________________

## Correctness and Performance Issues

### Bug 9: Position clamping silently corrupts gradient signal

**File**: `training/loss.py:293-296`

When `position >= logprobs.shape[0]`, it's clamped to the last token. Multiple positions
then map to the same gradient source, corrupting the distillation signal silently.

**Fix**: Log a warning when clamping occurs. Root cause is likely the prompt length
offset being incorrect.

______________________________________________________________________

### Bug 10: `prompt_lens` computation — O(batch * seq_len) Python loop

**File**: `training/loss.py:120-135`

Nested Python loop over batch and sequence dimensions. For large batches or long
sequences, this causes severe slowdowns that can make training appear to hang.

**Fix**: Replace with vectorized operations:

```python
# For 2D loss_mask [batch, seq_len]
prompt_lens = (loss_mask.bool().cumsum(dim=1) == 1).int().argmax(dim=1).tolist()
# For 1D loss_mask [seq_len]
prompt_len = (loss_mask.bool().cumsum(dim=0) == 1).int().argmax(dim=0).item()
```

______________________________________________________________________

### Bug 11: `distill_stat.item()` — GPU-CPU sync on every training step

**File**: `training/loss.py:159`

`distill_stat.item()` forces a GPU-CPU sync. The comment at line 369 says "Bug 4 fix:
avoid .item()" but `.item()` is still called here.

**Fix**: Use `distill_stat` directly — `torch.full(..., distill_stat, ...)` works
without `.item()`.

______________________________________________________________________

### Bug 12: `_chunked_apply` assumes first dim is seq_len

**File**: `training/logprobs.py:82-100`

Chunking splits along `logits.shape[0]`, but when logits has a batch dim
(`[1, seq_len, vocab]`), it splits along batch. Works by accident because batch=1, but
would break with batch>1.

**Fix**: No immediate change needed, but add a shape assertion and comment.

______________________________________________________________________

## Priority Fix Order

| Priority | Bug                               | Impact                         | Effort           |
| -------- | --------------------------------- | ------------------------------ | ---------------- |
| P0       | Bug 1: Wrong completion_id        | Crashes every run with teacher | Small            |
| P0       | Bug 2: export after session close | Intermittent crashes           | Small            |
| P1       | Bug 4: Wrong timeout constant     | OOM risk                       | Trivial (1 line) |
| P1       | Bug 3: threading.Lock in async    | Cascading timeouts             | Medium           |
| P1       | Bug 5: \_total_reward drift       | Stat corruption                | Small            |
| P2       | Bug 6: mb_assignment None         | Silent data loss               | Medium           |
| P2       | Bug 7: Dual PositionRewardInfo    | Type confusion                 | Medium           |
| P2       | Bug 8: model_inputs mutation      | Race condition                 | Medium           |
| P3       | Bug 9: Position clamping          | Silent corruption              | Small            |
| P3       | Bug 10: Python loop prompt_lens   | Performance                    | Small            |
| P3       | Bug 11: .item() GPU-CPU sync      | Performance                    | Trivial          |
| P3       | Bug 12: chunked_apply shape       | Latent breakage                | Trivial          |

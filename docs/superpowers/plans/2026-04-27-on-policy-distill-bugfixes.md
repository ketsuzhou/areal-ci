# On-Policy Distillation Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 12 bugs in the on-policy distillation training pipeline, prioritized by
crash risk.

**Architecture:** Fix bugs in dependency order — server/client bugs first (they affect
the data pipeline), then training-side bugs. Each fix is self-contained with its own
test.

**Tech Stack:** Python 3.12+ | PyTorch | pytest | aiohttp (for async tests)

______________________________________________________________________

## File Structure

| File                                                               | Action | Responsibility                                                           |
| ------------------------------------------------------------------ | ------ | ------------------------------------------------------------------------ |
| `customized_areal/on_policy_distill/core/agent.py`                 | Modify | Bug 1: Use real interaction ID                                           |
| `customized_areal/on_policy_distill/proxy/workflow.py`             | Modify | Bug 2: Move export inside session                                        |
| `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py` | Modify | Bug 4: Fix timeout constant                                              |
| `customized_areal/on_policy_distill/proxy/server.py`               | Modify | Bug 5: Eliminate save/restore pattern                                    |
| `customized_areal/on_policy_distill/proxy/cache.py`                | Modify | Bug 5: Add preserve_scalar_reward option                                 |
| `customized_areal/on_policy_distill/training/actor.py`             | Modify | Bug 6: Warn on unmapped sample_index                                     |
| `customized_areal/on_policy_distill/training/loss.py`              | Modify | Bugs 9, 10, 11: Clamping warning, vectorized prompt_lens, remove .item() |
| `customized_areal/on_policy_distill/engine/fsdp_engine.py`         | Modify | Bug 8: Avoid model_inputs mutation                                       |
| `customized_areal/on_policy_distill/training/logprobs.py`          | Modify | Bug 12: Add shape assertion                                              |
| `tests/test_distill_bugfixes.py`                                   | Create | All unit tests                                                           |

______________________________________________________________________

### Task 1: Fix Bug 1 — Wrong completion_id in OnPolicyDistillAgent

**Files:**

- Modify: `customized_areal/on_policy_distill/core/agent.py:186-244`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distill_bugfixes.py
"""Tests for on-policy distillation pipeline bug fixes."""

import inspect

import pytest


def test_bug1_completion_id_uses_interaction_id():
    """Bug 1: OnPolicyDistillAgent should use the interaction's actual ID
    from the proxy server, not an MD5 hash of completion_messages."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.core.agent",
            fromlist=["OnPolicyDistillAgent"],
        ).OnPolicyDistillAgent.run
    )
    # The MD5 hash approach produces wrong IDs that don't match any
    # interaction on the server. After the fix, the agent should use
    # interaction.interaction_id instead.
    assert "hashlib.md5" not in source, (
        "OnPolicyDistillAgent.run() should not use hashlib.md5 for "
        "completion_id. Use interaction.interaction_id from the proxy server."
    )
    # Should use the real interaction_id
    assert "interaction_id" in source, (
        "OnPolicyDistillAgent.run() should use interaction.interaction_id "
        "from the proxy server as the completion_id."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug1_completion_id_uses_interaction_id -v`
Expected: FAIL — `hashlib.md5` found in source

- [ ] **Step 3: Fix agent.py to use real interaction_id**

In `customized_areal/on_policy_distill/core/agent.py`, replace lines 216-218:

```python
# OLD:
completion_id = hashlib.md5(
    str(completion_messages).encode()
).hexdigest()[:16]

# NEW:
completion_id = interaction.interaction_id
```

Also remove the unused `import hashlib` at the top of the file (line 8).

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug1_completion_id_uses_interaction_id -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/core/agent.py tests/test_distill_bugfixes.py
git commit -m "fix(distill): use real interaction_id instead of MD5 hash for completion_id

The agent was generating completion_id via hashlib.md5 which never
matches any interaction on the proxy server. This caused set_reward()
to fail with HTTP 400 every time teacher distillation was enabled.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 2: Fix Bug 4 — Wrong timeout constant in stale session cleanup

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py:1118`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug4_stale_session_uses_configured_timeout():
    """Bug 4: _cleanup_stale_sessions should use _session_timeout_seconds
    (from config) instead of hardcoded SESSION_TIMEOUT_SECONDS."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.proxy_rollout_server",
            fromlist=["_cleanup_stale_sessions"],
        )._cleanup_stale_sessions
    )
    # Find is_stale calls
    for line in source.split("\n"):
        stripped = line.strip()
        if "is_stale" in stripped and "SESSION_TIMEOUT_SECONDS" in stripped:
            pytest.fail(
                "_cleanup_stale_sessions uses hardcoded SESSION_TIMEOUT_SECONDS "
                "instead of _session_timeout_seconds. Custom timeout configs "
                "are silently ignored, causing OOM from uncleaned sessions."
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug4_stale_session_uses_configured_timeout -v`
Expected: FAIL — `SESSION_TIMEOUT_SECONDS` found in `is_stale` call

- [ ] **Step 3: Fix proxy_rollout_server.py**

In `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`, change line 1118:

```python
# OLD:
if session.is_stale(SESSION_TIMEOUT_SECONDS):
# NEW:
if session.is_stale(_session_timeout_seconds):
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug4_stale_session_uses_configured_timeout -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py tests/test_distill_bugfixes.py
git commit -m "fix(proxy): use configured session timeout in stale session cleanup

_cleanup_stale_sessions used hardcoded SESSION_TIMEOUT_SECONDS (3600s)
instead of the configured _session_timeout_seconds, silently ignoring
custom timeout configs and risking OOM from uncleaned sessions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 3: Fix Bug 2 — export_interactions called after session end

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/workflow.py:275-297`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug2_export_inside_session_context():
    """Bug 2: export_interactions should be called inside the async with
    client block, before end_session marks the session as completed.
    Otherwise a race with the cleanup task can cause HTTP 404."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.workflow",
            fromlist=["OpenAIProxyWorkflow"],
        ).OpenAIProxyWorkflow.arun_episode
    )
    # Find the async with client: block
    lines = source.split("\n")
    in_async_with = False
    async_with_end = None
    export_line = None

    for i, line in enumerate(lines):
        if "async with client" in line:
            in_async_with = True
        if in_async_with and line.strip() and not line.strip().startswith("#"):
            # Track indentation to find end of async with block
            pass
        if "export_interactions" in line:
            export_line = i

    # After fix: export_interactions should appear INSIDE the async with block
    # Find the indentation of 'async with client:' and check export_interactions
    # is at a deeper indent level
    async_with_indent = None
    for i, line in enumerate(lines):
        if "async with client" in line:
            async_with_indent = len(line) - len(line.lstrip())
            break

    export_indent = None
    for i, line in enumerate(lines):
        if "export_interactions" in line and "await" in line:
            export_indent = len(line) - len(line.lstrip())
            break

    if async_with_indent is not None and export_indent is not None:
        assert export_indent > async_with_indent, (
            "export_interactions should be called inside the 'async with client:' "
            "block (before end_session) to avoid race with cleanup task."
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug2_export_inside_session_context -v`
Expected: FAIL — `export_indent` is not greater than `async_with_indent`

- [ ] **Step 3: Fix workflow.py — move export_interactions inside async with block**

In `customized_areal/on_policy_distill/proxy/workflow.py`, replace lines 275-297:

```python
# OLD:
        async with client:
            try:
                rewards = await self._run_agent(
                    client.session_api_key, data, proxy_client=client
                )
            except Exception as e:
                logger.warning(
                    f"Agent task failed: {e}. This trajectory will be rejected."
                )
                raise

            # Apply rewards from the agent
            if rewards is not None:
                await self._process_rewards(client, rewards)

        # Export interactions from the server after session ends
        # The server applies token-level rewards during export
        interactions = await client.export_interactions(
            discount=self.discount,
            style=self.export_style,
        )

# NEW:
        async with client:
            try:
                rewards = await self._run_agent(
                    client.session_api_key, data, proxy_client=client
                )
            except Exception as e:
                logger.warning(
                    f"Agent task failed: {e}. This trajectory will be rejected."
                )
                raise

            # Apply rewards from the agent
            if rewards is not None:
                await self._process_rewards(client, rewards)

            # Export interactions BEFORE session ends to avoid race with
            # the cleanup task removing the session between end_session
            # and export_trajectories.
            interactions = await client.export_interactions(
                discount=self.discount,
                style=self.export_style,
            )
```

- [ ] **Step 3: Fix proxy_rollout_server.py — don't remove session on export**

Moving `export_interactions` inside the `async with` block would cause a deadlock: the
server's `export_trajectories` calls `wait_for_finish()`, but `end_session` hasn't been
called yet. Instead, defer session removal to the cleanup task.

In `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`, modify the
`export_trajectories` endpoint (around lines 706-709). Remove the session removal from
`export_trajectories`:

```python
# OLD (lines 706-709):
    # Remove session from cache and clean up API key mapping
    with _lock:
        _session_cache.pop(session_id, None)
        _remove_api_keys_for_session(session_id)

# NEW:
    # Session will be removed by _cleanup_stale_sessions after it becomes stale.
    # Don't remove here — the client may call export_trajectories after
    # end_session, and removing eagerly causes a race with the cleanup task
    # that can result in HTTP 404 if cleanup runs between end_session and
    # export_trajectories.
    with _lock:
        _remove_api_keys_for_session(session_id)
```

Also update the test to verify the session is NOT removed on export:

```python
def test_bug2_export_does_not_remove_session():
    """Bug 2: export_trajectories should not remove the session from cache.
    Session removal should be deferred to _cleanup_stale_sessions to avoid
    race conditions when export is called after end_session."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.proxy_rollout_server",
            fromlist=["export_trajectories"],
        )
    )
    # Should not pop from _session_cache inside export_trajectories
    for line in source.split("\n"):
        stripped = line.strip()
        if "_session_cache.pop" in stripped:
            pytest.fail(
                "export_trajectories should not remove session from "
                "_session_cache. Defer removal to _cleanup_stale_sessions."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug2_export_does_not_remove_session -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py tests/test_distill_bugfixes.py
git commit -m "fix(proxy): defer session removal to cleanup task instead of export

export_trajectories removed the session from cache immediately after
export, but the client calls it after end_session. If the cleanup task
ran between these calls, export would fail with HTTP 404. Now only
API key mappings are removed on export; session data is cleaned up by
the stale session cleanup task.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 4: Fix Bug 5 — \_total_reward drift from save/restore pattern

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/cache.py:251-283,337-395`

- Modify: `customized_areal/on_policy_distill/proxy/server.py:140-148`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug5_set_rewards_preserve_scalar():
    """Bug 5: InteractionCache.set_rewards should support preserving
    the scalar reward to avoid _total_reward drift from save/restore."""
    from customized_areal.on_policy_distill.proxy.cache import (
        InteractionCache,
        PositionRewardInfo,
    )
    from customized_areal.on_policy_distill.proxy.types import (
        InteractionWithTokenLevelReward,
    )

    # Create a minimal interaction
    cache = InteractionCache()

    class MockModelResponse:
        output_tokens = [1, 2, 3]
        input_tokens = [0]
        input_len = 1
        output_len = 3
        output_logprobs = [-1.0, -0.5, -0.3]

    interaction = InteractionWithTokenLevelReward(
        interaction_id="test-1",
        messages=[{"role": "user", "content": "hi"}],
        reward=5.0,
        model_response=MockModelResponse(),
    )
    cache["test-1"] = interaction

    # Set token rewards while preserving scalar reward
    cache.set_rewards("test-1", [0.1, 0.2, 0.3], preserve_scalar_reward=True)

    # Scalar reward should be preserved (was 5.0)
    assert cache["test-1"].reward == 5.0, (
        f"Scalar reward should be preserved, got {cache['test-1'].reward}"
    )
    # _total_reward should still reflect the original scalar reward
    assert cache.total_reward == 5.0, (
        f"total_reward should be 5.0, got {cache.total_reward}"
    )


def test_bug5_server_no_save_restore():
    """Bug 5: TokenRewardSessionData.set_token_rewards should not use
    save/restore pattern for scalar reward."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.server",
            fromlist=["TokenRewardSessionData"],
        ).TokenRewardSessionData.set_token_rewards
    )
    assert "saved_reward" not in source, (
        "TokenRewardSessionData.set_token_rewards should not use a "
        "save/restore pattern for scalar reward. Use preserve_scalar_reward=True "
        "in InteractionCache.set_rewards() instead."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug5_set_rewards_preserve_scalar tests/test_distill_bugfixes.py::test_bug5_server_no_save_restore -v`
Expected: FAIL — `preserve_scalar_reward` parameter doesn't exist yet, and
`saved_reward` found in source

- [ ] **Step 3: Add preserve_scalar_reward to InteractionCache.set_rewards**

In `customized_areal/on_policy_distill/proxy/cache.py`, modify `set_rewards` (line 251)
and `_set_rewards_internal` (line 337):

```python
# Replace set_rewards (line 251-283):
    def set_rewards(
        self,
        completion_id: str,
        token_rewards: list[float],
        preserve_scalar_reward: bool = False,
    ) -> None:
        with self._lock:
            self._set_rewards_internal(
                completion_id, token_rewards, preserve_scalar_reward
            )

# Replace _set_rewards_internal (line 337-395):
    def _set_rewards_internal(
        self,
        completion_id: str,
        token_rewards: list[float],
        preserve_scalar_reward: bool = False,
    ) -> None:
        """Internal version of set_rewards without lock acquisition.

        Assumes caller already holds self._lock.
        """
        if completion_id not in self:
            raise KeyError(f"Completion {completion_id} not found in cache")

        if len(token_rewards) == 0:
            raise ValueError("Token rewards list cannot be empty")

        interaction = self[completion_id]

        # Validate length matches output tokens if available
        if (
            hasattr(interaction, "model_response")
            and interaction.model_response is not None
        ):
            expected_len = len(interaction.model_response.output_tokens)
            if len(token_rewards) != expected_len:
                raise ValueError(
                    f"Token rewards length ({len(token_rewards)}) must match "
                    f"output tokens length ({expected_len})"
                )

        # Store the token-wise rewards
        interaction.token_rewards_list = token_rewards  # type: ignore

        # Update token_rewards field if using InteractionWithTokenLevelReward
        if hasattr(interaction, "token_rewards"):
            try:
                interaction.set_token_rewards(token_rewards)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not set token_rewards: {e}")

        # Update total list reward (element-wise sum)
        if self._total_list_reward is None:
            self._total_list_reward = token_rewards.copy()
        else:
            max_len = max(len(self._total_list_reward), len(token_rewards))
            self._total_list_reward += [0.0] * (max_len - len(self._total_list_reward))
            rewards_padded = token_rewards + [0.0] * (max_len - len(token_rewards))
            self._total_list_reward = [
                a + b for a, b in zip(self._total_list_reward, rewards_padded)
            ]

        if not preserve_scalar_reward:
            # Update scalar reward tracking (subtract old, add new)
            scalar_reward = sum(token_rewards)
            old_reward = interaction.reward or 0.0
            self._total_reward -= old_reward
            interaction.reward = float(scalar_reward)
            self._total_reward += float(scalar_reward)
```

- [ ] **Step 4: Fix TokenRewardSessionData.set_token_rewards**

In `customized_areal/on_policy_distill/proxy/server.py`, replace lines 140-148:

```python
# OLD:
        with self._lock:
            self._token_rewards[interaction_id] = token_rewards
            # Delegate to extended cache if interaction is present
            if interaction_id in self.completions:
                saved_reward = self.completions[interaction_id].reward
                self.completions.set_rewards(interaction_id, token_rewards)
                # Restore scalar reward if it was explicitly set via set_reward()
                if saved_reward is not None:
                    self.completions[interaction_id].reward = saved_reward

# NEW:
        with self._lock:
            self._token_rewards[interaction_id] = token_rewards
            # Delegate to extended cache, preserving the scalar reward
            # to avoid _total_reward drift from save/restore pattern.
            if interaction_id in self.completions:
                self.completions.set_rewards(
                    interaction_id, token_rewards, preserve_scalar_reward=True
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug5_set_rewards_preserve_scalar tests/test_distill_bugfixes.py::test_bug5_server_no_save_restore -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/cache.py customized_areal/on_policy_distill/proxy/server.py tests/test_distill_bugfixes.py
git commit -m "fix(proxy): eliminate save/restore pattern for scalar reward in set_token_rewards

TokenRewardSessionData.set_token_rewards used a fragile save/restore
pattern that caused _total_reward drift. Add preserve_scalar_reward
option to InteractionCache.set_rewards() and use it instead.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 5: Fix Bug 6 — Warn on unmapped sample_index in \_distribute_position_rewards

**Files:**

- Modify: `customized_areal/on_policy_distill/training/actor.py:108-143`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug6_distribute_position_rewards_warns_on_unmapped():
    """Bug 6: _distribute_position_rewards should warn when a
    position_reward's sample_index doesn't map to any minibatch."""
    from unittest.mock import patch

    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.actor import (
        _distribute_position_rewards,
    )

    # Create a MicroBatchList-like object with 1 minibatch of batch_size=2
    mb = {
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    mb_inputs = type("MB", (), {"mbs": [mb], "forward_indices": [0, 1]})()

    # Position reward with sample_index=99 (out of range)
    bad_pr = PositionRewardInfo(
        position=0,
        candidates=["a", "b"],
        candidate_token_ids=[1, 2],
        rewards=[0.5, -0.3],
        chosen_index=0,
        sample_index=99,  # Doesn't map to any minibatch
    )
    good_pr = PositionRewardInfo(
        position=1,
        candidates=["c", "d"],
        candidate_token_ids=[3, 4],
        rewards=[0.2, -0.1],
        chosen_index=0,
        sample_index=0,
    )

    # Should warn about the unmapped sample_index
    import logging
    with patch("customized_areal.on_policy_distill.training.actor.logger") as mock_logger:
        _distribute_position_rewards(mb_inputs, [bad_pr, good_pr])
        # Check that a warning was logged for the unmapped sample_index
        warning_calls = [
            c for c in mock_logger.method_calls if "warning" in str(c)
        ]
        assert len(warning_calls) > 0, (
            "_distribute_position_rewards should log a warning when "
            "a position_reward's sample_index doesn't map to any minibatch"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug6_distribute_position_rewards_warns_on_unmapped -v`
Expected: FAIL — no warning logged

- [ ] **Step 3: Fix actor.py**

In `customized_areal/on_policy_distill/training/actor.py`, replace lines 131-136:

```python
# OLD:
    per_mb_prs: dict[int, list] = {}
    for pr in position_rewards:
        mb_i = mb_assignment[pr.sample_index]
        if mb_i is not None:
            per_mb_prs.setdefault(mb_i, []).append(pr)

# NEW:
    per_mb_prs: dict[int, list] = {}
    for pr in position_rewards:
        if pr.sample_index >= len(mb_assignment):
            logger.warning(
                "position_reward sample_index=%d exceeds batch_size=%d, "
                "dropping position=%d",
                pr.sample_index,
                len(mb_assignment),
                pr.position,
            )
            continue
        mb_i = mb_assignment[pr.sample_index]
        if mb_i is None:
            logger.warning(
                "position_reward sample_index=%d not mapped to any minibatch, "
                "dropping position=%d",
                pr.sample_index,
                pr.position,
            )
            continue
        per_mb_prs.setdefault(mb_i, []).append(pr)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug6_distribute_position_rewards_warns_on_unmapped -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/training/actor.py tests/test_distill_bugfixes.py
git commit -m "fix(distill): warn when position_reward sample_index is unmapped

_distribute_position_rewards silently dropped position_rewards whose
sample_index didn't map to any minibatch. Now logs a warning to help
diagnose data flow issues.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 6: Fix Bugs 9, 10, 11 — Loss function correctness and performance

**Files:**

- Modify: `customized_areal/on_policy_distill/training/loss.py:116-165,292-296`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug10_prompt_lens_vectorized():
    """Bug 10: prompt_lens computation in grpo_distill_loss_fn should use
    vectorized PyTorch ops instead of O(batch*seq_len) Python loops."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.training.loss",
            fromlist=["grpo_distill_loss_fn"],
        ).grpo_distill_loss_fn
    )
    # Should not contain nested for loops over loss_mask dimensions
    lines = source.split("\n")
    in_prompt_loop = False
    for line in lines:
        if "prompt_lens = []" in line or "prompt_lens.append" in line:
            in_prompt_loop = True
        if in_prompt_loop and "for b in range(loss_mask.shape" in line:
            pytest.fail(
                "prompt_lens computation uses O(batch*seq_len) Python loop. "
                "Use vectorized: prompt_lens = (loss_mask.bool().cumsum(dim=1)==1)"
                ".int().argmax(dim=1).tolist()"
            )


def test_bug11_no_item_in_distill_stat():
    """Bug 11: distill_stat.item() in grpo_distill_loss_fn forces GPU-CPU
    sync on every training step. Use the tensor directly."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.training.loss",
            fromlist=["grpo_distill_loss_fn"],
        ).grpo_distill_loss_fn
    )
    for line in source.split("\n"):
        if "distill_stat" in line and ".item()" in line:
            pytest.fail(
                "distill_stat.item() forces GPU-CPU sync. Use tensor directly: "
                "torch.full(..., distill_stat, ...)"
            )


def test_bug9_position_clamping_warns():
    """Bug 9: Position clamping in _compute_position_level_grpo_loss should
    log a warning instead of silently corrupting gradient signal."""
    from unittest.mock import patch

    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import (
        _compute_position_level_grpo_loss,
    )

    seq_len = 5
    num_candidates = 2
    logprobs = torch.randn(seq_len, num_candidates, requires_grad=True)
    loss_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)

    # Position 10 is out of bounds for seq_len=5
    position_rewards = [
        PositionRewardInfo(
            position=10,
            candidates=["a", "b"],
            candidate_token_ids=[1, 2],
            logprobs=[-1.0, -2.0],
            rewards=[0.5, -0.3],
            chosen_index=0,
            sample_index=0,
        ),
    ]

    with patch("customized_areal.on_policy_distill.training.loss.logger") as mock_logger:
        _compute_position_level_grpo_loss(
            position_rewards=position_rewards,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[0],
        )
        warning_calls = [
            c for c in mock_logger.method_calls if "warning" in str(c)
        ]
        assert len(warning_calls) > 0, (
            "_compute_position_level_grpo_loss should log a warning when "
            "position is clamped to valid range"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug10_prompt_lens_vectorized tests/test_distill_bugfixes.py::test_bug11_no_item_in_distill_stat tests/test_distill_bugfixes.py::test_bug9_position_clamping_warns -v`
Expected: FAIL on all three

- [ ] **Step 3: Fix loss.py — vectorize prompt_lens**

In `customized_areal/on_policy_distill/training/loss.py`, replace lines 116-135:

```python
# OLD:
        # Determine prompt length per sample from loss_mask (0 = prompt, 1 = output)
        # Bug 3 fix: compute prompt_len per sample for correct position offsetting
        if loss_mask.dim() > 1:
            # [batch, seq_len] -> compute per-sample prompt_len
            prompt_lens = []
            for b in range(loss_mask.shape[0]):
                pl = 0
                for i in range(loss_mask.shape[1]):
                    if loss_mask[b, i]:
                        pl = i
                        break
                prompt_lens.append(pl)
        else:
            # [seq_len] -> single sample
            prompt_len = 0
            for i in range(loss_mask.shape[0]):
                if loss_mask[i]:
                    prompt_len = i
                    break
            prompt_lens = [prompt_len]

# NEW:
        # Determine prompt length per sample from loss_mask (0 = prompt, 1 = output)
        # Vectorized: find first True position per sample
        if loss_mask.dim() > 1:
            # [batch, seq_len] -> per-sample prompt_len
            first_true = (loss_mask.bool().cumsum(dim=1) == 1)
            prompt_lens = first_true.int().argmax(dim=1).tolist()
        else:
            # [seq_len] -> single sample
            prompt_len = (
                (loss_mask.bool().cumsum(dim=0) == 1).int().argmax(dim=0).item()
            )
            prompt_lens = [prompt_len]
```

- [ ] **Step 4: Fix loss.py — remove .item() from distill_stat**

Replace lines 154-165 in `customized_areal/on_policy_distill/training/loss.py`:

```python
# OLD:
    if distill_stat is not None:
        # Expand distill_stat to match the shape of loss_mask for stats_tracker
        distill_loss_expanded = torch.full(
            loss_mask.shape,
            distill_stat.item(),
            dtype=torch.float32,
            device=loss_mask.device,
        )
        stats_tracker.stat(
            distill_loss=distill_loss_expanded,
            denominator="n_valid_tokens",
        )

# NEW:
    if distill_stat is not None:
        # Expand distill_stat to match the shape of loss_mask for stats_tracker.
        # Use tensor directly to avoid GPU-CPU sync from .item().
        distill_loss_expanded = torch.full(
            loss_mask.shape,
            distill_stat,
            dtype=torch.float32,
            device=loss_mask.device,
        )
        stats_tracker.stat(
            distill_loss=distill_loss_expanded,
            denominator="n_valid_tokens",
        )
```

- [ ] **Step 5: Fix loss.py — add warning on position clamping**

Replace lines 292-296 in `customized_areal/on_policy_distill/training/loss.py`:

```python
# OLD:
        # Bug 5 fix: clamp position to valid range instead of silently skipping
        if position >= logprobs.shape[0]:
            position = logprobs.shape[0] - 1
        if position < 0:
            continue

# NEW:
        if position >= logprobs.shape[0]:
            logger.warning(
                "Position %d + prompt_len=%d = %d exceeds logprobs length %d, "
                "clamping to last position. This may indicate incorrect "
                "prompt_len computation.",
                pr.position,
                pl,
                position,
                logprobs.shape[0],
            )
            position = logprobs.shape[0] - 1
        if position < 0:
            continue
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug10_prompt_lens_vectorized tests/test_distill_bugfixes.py::test_bug11_no_item_in_distill_stat tests/test_distill_bugfixes.py::test_bug9_position_clamping_warns -v`
Expected: PASS

- [ ] **Step 7: Run existing tests to check for regressions**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/training/test_offline_train.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add customized_areal/on_policy_distill/training/loss.py tests/test_distill_bugfixes.py
git commit -m "fix(distill): vectorize prompt_lens, remove .item() sync, warn on position clamp

- Replace O(batch*seq_len) Python loop for prompt_lens with vectorized
  cumsum+argmax approach
- Remove distill_stat.item() GPU-CPU sync from stats tracking
- Log warning when position is clamped to valid range instead of silent
  gradient corruption

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 7: Fix Bug 8 — model_inputs in-place mutation in fsdp_engine

**Files:**

- Modify: `customized_areal/on_policy_distill/engine/fsdp_engine.py:41-111,299-313`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug8_no_model_inputs_mutation():
    """Bug 8: _compute_logprobs_and_loss should not mutate ctx.model_inputs
    by temporarily overriding rolled_input_ids. Pass labels separately."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.engine.fsdp_engine",
            fromlist=["MultiCandidateFSDPEngine"],
        ).MultiCandidateFSDPEngine._compute_logprobs_and_loss
    )
    # Should not contain rolled_input_ids override
    assert 'ctx.model_inputs["rolled_input_ids"]' not in source, (
        "_compute_logprobs_and_loss should not mutate ctx.model_inputs by "
        "overriding rolled_input_ids. Pass multi_candidate_labels as a "
        "separate parameter to _compute_logprobs_entropy instead."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug8_no_model_inputs_mutation -v`
Expected: FAIL — `rolled_input_ids` mutation found

- [ ] **Step 3: Fix fsdp_engine.py — add labels parameter to
  \_compute_logprobs_entropy**

In `customized_areal/on_policy_distill/engine/fsdp_engine.py`, modify
`_compute_logprobs_entropy` (line 41) to accept optional `labels_override`:

```python
# Replace the method signature at line 41-42:
    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
        labels_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
```

Then, at the beginning of the method (after the docstring), add labels_override logic
before the existing labels computation:

```python
        # Use labels_override if provided (avoids mutating inputs)
        if labels_override is not None:
            labels = labels_override
        else:
            # Try to get rolled_input_ids (if Ulysses SP is enabled)
            labels = inputs.get(
                "rolled_input_ids",
                torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
            )
```

Remove the old lines that computed labels (original lines 72-76):

```python
        # OLD: Remove these lines (they're replaced by the if/else above)
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
```

Then in `_compute_logprobs_and_loss`, replace the mutation block (lines 299-313):

```python
# OLD:
                    if multi_candidate_labels is not None:
                        # Use multi-candidate labels for gathering
                        # Temporarily override rolled_input_ids
                        original_rolled = ctx.model_inputs.get("rolled_input_ids")
                        ctx.model_inputs["rolled_input_ids"] = multi_candidate_labels

                        try:
                            logprobs, entropy = self._compute_logprobs_entropy(
                                logits, ctx.model_inputs, ctx.ulysses_pad_size
                            )
                        finally:
                            if original_rolled is not None:
                                ctx.model_inputs["rolled_input_ids"] = original_rolled
                            else:
                                ctx.model_inputs.pop("rolled_input_ids", None)

# NEW:
                    if multi_candidate_labels is not None:
                        # Pass multi-candidate labels directly without mutating
                        # ctx.model_inputs, avoiding potential race conditions.
                        logprobs, entropy = self._compute_logprobs_entropy(
                            logits, ctx.model_inputs, ctx.ulysses_pad_size,
                            labels_override=multi_candidate_labels,
                        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug8_no_model_inputs_mutation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/engine/fsdp_engine.py tests/test_distill_bugfixes.py
git commit -m "fix(engine): pass multi-candidate labels directly instead of mutating model_inputs

_compute_logprobs_and_loss temporarily overwrote ctx.model_inputs to
pass multi-candidate labels, creating a potential race condition with
overlapping minibatch processing. Add labels_override parameter to
_compute_logprobs_entropy and pass labels directly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 8: Fix Bug 12 — Add shape assertion to \_chunked_apply

**Files:**

- Modify: `customized_areal/on_policy_distill/training/logprobs.py:81-100`

- Test: `tests/test_distill_bugfixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distill_bugfixes.py`:

```python
def test_bug12_chunked_apply_has_shape_assertion():
    """Bug 12: _chunked_apply should assert that logits is 2D (seq_len first)
    since it splits along dim=0."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.training.logprobs",
            fromlist=["_chunked_apply"],
        )._chunked_apply
    )
    # Should contain an assertion about logits dimensions
    assert "ndim" in source or "dim" in source, (
        "_chunked_apply should assert logits.ndim == 2 since it splits "
        "along dim=0 assuming seq_len is the first dimension."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug12_chunked_apply_has_shape_assertion -v`
Expected: FAIL — no ndim/dim assertion

- [ ] **Step 3: Fix logprobs.py**

In `customized_areal/on_policy_distill/training/logprobs.py`, modify `_chunked_apply`
(line 81):

```python
# Replace the function body:
def _chunked_apply(
    fn: Callable[[torch.Tensor, torch.Tensor], T],
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> T:
    """Apply a function in chunks along the first dimension to reduce peak memory.

    Assumes logits is 2D [seq_len, vocab_size] with batch dim already squeezed.
    The caller must handle batch dimensions before calling this function.
    """
    assert logits.ndim == 2, (
        f"_chunked_apply expects 2D logits [seq_len, vocab_size], "
        f"got {logits.ndim}D with shape {logits.shape}. "
        f"Squeeze batch dimension before calling."
    )
    total_seqlen = logits.shape[0]
    assert total_seqlen > 0, "Input logits must have at least one element"
    results: list = []

    for i in range(0, total_seqlen, chunk_size):
        end_idx = min(i + chunk_size, total_seqlen)
        chunk_result = fn(logits[i:end_idx], labels[i:end_idx])
        results.append(chunk_result)

    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        return tuple(torch.cat([r[i] for r in results]) for i in range(num_outputs))
    return torch.cat(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py::test_bug12_chunked_apply_has_shape_assertion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/training/logprobs.py tests/test_distill_bugfixes.py
git commit -m "fix(logprobs): add shape assertion to _chunked_apply

_chunked_apply splits along dim=0 assuming seq_len is first, but
silently produced wrong results if logits had a batch dimension.
Add assertion to catch this misuse early.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 9: Run full test suite and pre-commit

- [ ] **Step 1: Run all bugfix tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_distill_bugfixes.py -v`
Expected: All PASS

- [ ] **Step 2: Run existing offline training tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/training/test_offline_train.py -v`
Expected: All PASS

- [ ] **Step 3: Run existing proxy tests (no GPU needed)**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/ -v -k "not integration" 2>&1 | head -80`
Expected: Most PASS (some may fail for unrelated reasons)

- [ ] **Step 4: Run pre-commit**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && pre-commit run --all-files 2>&1 | tail -30`
Expected: PASS (fix any formatting issues first)

- [ ] **Step 5: Final commit if pre-commit made changes**

```bash
git add -A
git commit -m "style: pre-commit fixes for bugfix changes

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

## Bugs Not Implemented (Deferred)

| Bug                              | Reason                                                                                                                                        |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Bug 3 (threading.Lock → asyncio) | Requires careful analysis of which code paths are sync vs async; high risk of introducing new deadlocks. Should be a separate task.           |
| Bug 7 (dual PositionRewardInfo)  | Refactoring to consolidate types affects serialization throughout proxy server/client. Needs careful coordination. Should be a separate task. |

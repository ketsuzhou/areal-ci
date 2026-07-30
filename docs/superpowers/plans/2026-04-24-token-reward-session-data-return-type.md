# TokenRewardSessionData Return Type Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `TokenRewardSessionData.export_interactions` return
`dict[str, InteractionWithTokenLevelReward]` with correct type annotations and proper
data flow through serialization/deserialization.

**Architecture:** Override `TokenRewardSessionData.__init__` to use the extended
`InteractionCache`, widen return types across the server/client/workflow chain, delegate
reward application to the cache at set-time, and fix deserialization to create
`InteractionWithTokenLevelReward` objects instead of base-type objects.

**Tech Stack:** Python 3.12+ | Pydantic | pytest

______________________________________________________________________

## File Structure

| File                                                                     | Change | Responsibility                                                    |
| ------------------------------------------------------------------------ | ------ | ----------------------------------------------------------------- |
| `customized_areal/on_policy_distill/proxy/server.py`                     | Modify | Override `__init__`, widen return type, delegate rewards to cache |
| `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`       | Modify | Fix deserialization to create `InteractionWithTokenLevelReward`   |
| `customized_areal/on_policy_distill/proxy/client.py`                     | Modify | Type `export_interactions` return as `TokenRewardInteractions`    |
| `customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py` | Create | New tests for return type fix and data flow                       |

______________________________________________________________________

### Task 1: Write failing tests for the return type fix

**Files:**

- Create: `customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py`

- [ ] **Step 1: Write tests that verify the current behavior is broken**

```python
"""Tests for TokenRewardSessionData.export_interactions return type fix.

Verifies that:
1. TokenRewardSessionData uses the extended InteractionCache
2. export_interactions returns InteractionWithTokenLevelReward objects
3. Deserialization creates InteractionWithTokenLevelReward objects
4. token_rewards survive to_tensor_dict() after cache invalidation
5. Position rewards survive the full round-trip
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from customized_areal.on_policy_distill.proxy.cache import (
    InteractionCache as ExtendedInteractionCache,
)
from customized_areal.on_policy_distill.proxy.proxy_rollout_server import (
    deserialize_interactions_with_position_rewards,
    serialize_interactions_with_position_rewards,
)
from customized_areal.on_policy_distill.proxy.server import (
    PositionRewardInfo,
    TokenRewardSessionData,
)
from customized_areal.on_policy_distill.proxy.types import (
    InteractionWithTokenLevelReward,
)


class TestSessionDataUsesExtendedCache:
    """Verify TokenRewardSessionData uses the extended InteractionCache."""

    def test_completions_is_extended_cache(self):
        session = TokenRewardSessionData("test-session")
        assert isinstance(session.completions, ExtendedInteractionCache), (
            f"Expected ExtendedInteractionCache, got {type(session.completions).__name__}"
        )


class TestExportInteractionsReturnType:
    """Verify export_interactions returns InteractionWithTokenLevelReward objects."""

    def test_returned_interaction_is_correct_type(self):
        session = TokenRewardSessionData("test-session")
        # Add an interaction with minimal setup
        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-1"),
            reward=1.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        session.completions["comp-1"] = interaction

        session.set_token_rewards("comp-1", [0.1, 0.2, 0.3])
        session.finish()

        result = session.export_interactions(discount=1.0, style="individual")

        assert "comp-1" in result
        assert isinstance(result["comp-1"], InteractionWithTokenLevelReward), (
            f"Expected InteractionWithTokenLevelReward, got {type(result['comp-1']).__name__}"
        )


class TestDeserializationCreatesCorrectType:
    """Verify deserialize_interactions_with_position_rewards creates
    InteractionWithTokenLevelReward objects."""

    def test_deserialized_is_extended_type(self):
        mock_interaction = Mock()
        mock_interaction.reward = 1.0
        mock_interaction.interaction_id = "comp-deser-test"
        mock_interaction.position_rewards = None
        mock_interaction.token_rewards = [0.1, 0.2, 0.3]

        mock_interaction.to_tensor_dict.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
            "rewards": torch.tensor([1.0]),
            "logprobs": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.3, -0.8]]
            ),
            "versions": torch.tensor([[-1, -1, -1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        }

        interactions = {"comp-deser-test": mock_interaction}
        serialized = serialize_interactions_with_position_rewards(interactions)
        deserialized = deserialize_interactions_with_position_rewards(serialized)

        result = deserialized["comp-deser-test"]
        assert isinstance(result, InteractionWithTokenLevelReward), (
            f"Expected InteractionWithTokenLevelReward, got {type(result).__name__}"
        )

    def test_deserialized_token_rewards_is_proper_field(self):
        """token_rewards should be a proper dataclass field, not a dynamic attribute."""
        mock_interaction = Mock()
        mock_interaction.reward = 0.0
        mock_interaction.interaction_id = "comp-field-test"
        mock_interaction.position_rewards = None
        mock_interaction.token_rewards = [0.1, 0.2, 0.3]

        mock_interaction.to_tensor_dict.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
            "rewards": torch.tensor([0.0]),
            "logprobs": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.3, -0.8]]
            ),
            "versions": torch.tensor([[-1, -1, -1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        }

        interactions = {"comp-field-test": mock_interaction}
        serialized = serialize_interactions_with_position_rewards(interactions)
        deserialized = deserialize_interactions_with_position_rewards(serialized)

        result = deserialized["comp-field-test"]
        # InteractionWithTokenLevelReward has token_rewards as a dataclass field
        # with a default of None. InteractionWithTokenLogpReward does NOT have it.
        assert "token_rewards" in [
            f.name for f in result.__dataclass_fields__.values()
        ], "token_rewards should be a declared dataclass field, not a dynamic attribute"


class TestTokenRewardsCacheRecomputation:
    """Verify to_tensor_dict() includes token_rewards even after cache invalidation.

    This is the key safety improvement: if _cache is invalidated (set to None),
    to_tensor_dict() should recompute correctly including token_rewards. This only
    works when the object is InteractionWithTokenLevelReward, which overrides
    to_tensor_dict() to include token_rewards.
    """

    def test_tensor_dict_after_cache_invalidation(self):
        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-recomp"),
            reward=1.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        interaction.token_rewards = [0.1, 0.2, 0.3]

        # Compute once to populate cache
        td1 = interaction.to_tensor_dict()
        assert "token_rewards" in td1

        # Invalidate cache
        interaction._cache = None

        # Recompute — should still include token_rewards
        td2 = interaction.to_tensor_dict()
        assert "token_rewards" in td2, (
            "token_rewards lost after cache invalidation. "
            "to_tensor_dict() recomputation must include token_rewards."
        )


class TestScalarRewardPreservedAfterDelegation:
    """Verify scalar reward is NOT overwritten when delegating to extended cache."""

    def test_set_token_rewards_preserves_scalar(self):
        session = TokenRewardSessionData("test-session")

        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-scalar"),
            reward=5.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        session.completions["comp-scalar"] = interaction

        # Set scalar reward
        session.completions.set_reward("comp-scalar", 5.0)

        # Set token rewards (sum = 0.6, different from 5.0)
        session.set_token_rewards("comp-scalar", [0.1, 0.2, 0.3])

        # Scalar reward MUST be preserved
        assert session.completions["comp-scalar"].reward == 5.0, (
            f"Scalar reward should be 5.0, got {session.completions['comp-scalar'].reward}. "
            "set_token_rewards must NOT overwrite trajectory-level scalar reward."
        )

    def test_set_position_rewards_preserves_scalar(self):
        session = TokenRewardSessionData("test-session")

        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-pos-scalar"),
            reward=5.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        session.completions["comp-pos-scalar"] = interaction

        session.completions.set_reward("comp-pos-scalar", 5.0)

        position_rewards = [
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                rewards=[0.1, 0.5],
                chosen_index=1,
            ),
            PositionRewardInfo(
                position=1,
                candidates=["c", "d"],
                rewards=[0.2, 0.6],
                chosen_index=0,
            ),
            PositionRewardInfo(
                position=2,
                candidates=["e", "f"],
                rewards=[0.3, 0.9],
                chosen_index=1,
            ),
        ]
        session.set_position_rewards("comp-pos-scalar", position_rewards)

        assert session.completions["comp-pos-scalar"].reward == 5.0, (
            f"Scalar reward should be 5.0, got {session.completions['comp-pos-scalar'].reward}. "
            "set_position_rewards must NOT overwrite trajectory-level scalar reward."
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py -v 2>&1 | head -80`

Expected: Several tests FAIL — `test_completions_is_extended_cache` fails because
`session.completions` is the base `InteractionCache`;
`test_returned_interaction_is_correct_type` may pass or fail depending on runtime types;
`test_deserialized_is_extended_type` fails because deserialization creates
`InteractionWithTokenLogpReward`; `test_scalar_reward_preserved_after_delegation` may
fail because `set_token_rewards` doesn't delegate to cache yet.

- [ ] **Step 3: Commit the test file**

```bash
git add customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py
git commit -m "test: add failing tests for TokenRewardSessionData return type fix"
```

______________________________________________________________________

### Task 2: Server — Override `__init__`, widen return type, delegate rewards

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/server.py`

This task implements Changes 1, 2, and 3 from the spec: override `__init__` to use the
extended `InteractionCache`, widen the return type of `export_interactions`, and
delegate reward application to the cache at set-time.

- [ ] **Step 1: Add the extended InteractionCache import and override `__init__`**

In `customized_areal/on_policy_distill/proxy/server.py`, add the import at the top
(after the existing imports) and modify `TokenRewardSessionData.__init__`:

Replace:

```python
from areal.experimental.openai.proxy.server import (
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    SESSION_TIMEOUT_SECONDS,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SessionData,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    deserialize_interactions,
    serialize_interactions,
)

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
```

With:

```python
from areal.experimental.openai.proxy.server import (
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    SESSION_TIMEOUT_SECONDS,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SessionData,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    deserialize_interactions,
    serialize_interactions,
)

from .cache import InteractionCache as ExtendedInteractionCache
from .types import InteractionWithTokenLevelReward
```

Then replace `TokenRewardSessionData.__init__`:

```python
    def __init__(self, session_id: str):
        super().__init__(session_id)
        # Use extended InteractionCache that stores
        # InteractionWithTokenLevelReward objects
        self._completions = ExtendedInteractionCache()
        # Store token-level rewards separately for fallback during export
        # (when interactions aren't in cache yet at set-time)
        self._token_rewards: dict[str, list[float]] = {}
        self._position_rewards: dict[str, list[PositionRewardInfo]] = {}
        self._lock = threading.Lock()
```

- [ ] **Step 2: Widen the return type of `export_interactions`**

Replace the method signature:

```python
    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLevelReward]:
```

- [ ] **Step 3: Delegate reward application in `set_token_rewards`**

Replace the `set_token_rewards` method body. The key invariant: scalar reward set via
`set_reward()` must NOT be overwritten by token rewards. The extended cache's
`set_rewards()` method overwrites `interaction.reward` with `sum(token_rewards)`, so we
save and restore the scalar reward.

```python
    def set_token_rewards(
        self, interaction_id: str, token_rewards: list[float]
    ) -> None:
        """
        Set token-wise rewards for an interaction.

        Delegates to the extended cache if the interaction is present.
        Preserves the scalar (trajectory-level) reward — it is NOT
        overwritten by sum(token_rewards).

        Raises
        ------
        RuntimeError
            If the session has already been finished.

        Parameters
        ----------
        interaction_id : str
            The interaction/completion ID
        token_rewards : list[float]
            Token-wise rewards, one per output token
        """
        if self.is_completed:
            raise RuntimeError(
                f"Cannot set token rewards on finished session {self.session_id}"
            )
        with self._lock:
            self._token_rewards[interaction_id] = token_rewards
            # Delegate to extended cache if interaction is present
            if interaction_id in self.completions:
                saved_reward = self.completions[interaction_id].reward
                self.completions.set_rewards(interaction_id, token_rewards)
                # Restore scalar reward if it was explicitly set via set_reward()
                if saved_reward is not None:
                    self.completions[interaction_id].reward = saved_reward
```

- [ ] **Step 4: Delegate reward application in `set_position_rewards`**

```python
    def set_position_rewards(
        self, interaction_id: str, position_rewards: list[PositionRewardInfo]
    ) -> None:
        """
        Set position-wise rewards for an interaction.

        Delegates to the extended cache if the interaction is present.
        Preserves the scalar (trajectory-level) reward — it is NOT
        overwritten by position-level rewards.

        Raises
        ------
        RuntimeError
            If the session has already been finished.

        Parameters
        ----------
        interaction_id : str
            The interaction/completion ID
        position_rewards : list[PositionRewardInfo]
            Position-wise candidate rewards
        """
        if self.is_completed:
            raise RuntimeError(
                f"Cannot set position rewards on finished session {self.session_id}"
            )
        with self._lock:
            self._position_rewards[interaction_id] = position_rewards
            # Extract chosen token rewards for token-wise storage
            chosen_rewards = [
                pr.rewards[pr.chosen_index] if pr.rewards else 0.0
                for pr in position_rewards
            ]
            self._token_rewards[interaction_id] = chosen_rewards
            # Delegate to extended cache if interaction is present
            if interaction_id in self.completions:
                self.completions.set_position_rewards(interaction_id, position_rewards)
```

- [ ] **Step 5: Update `export_interactions` body**

The export method still needs a fallback loop for interactions that weren't in the cache
at set-time (timing race). But now it can use the typed setter as the primary path:

```python
    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLevelReward]:
        """
        Export interactions with token-level rewards applied.

        Overrides base method to apply token-level rewards before export.
        The scalar (trajectory-level) reward is preserved for use by
        tree backup advantage computation; position-level rewards are
        stored separately for distillation loss only.
        """
        # Apply token-level rewards to interactions not yet handled
        # by set-time delegation (e.g., interaction wasn't in cache yet)
        with self._lock:
            for interaction_id, token_rewards in self._token_rewards.items():
                if interaction_id in self.completions:
                    interaction = self.completions[interaction_id]
                    # Use typed setter if available (InteractionWithTokenLevelReward)
                    if hasattr(interaction, "set_token_rewards"):
                        try:
                            interaction.set_token_rewards(token_rewards)
                        except (ValueError, AttributeError):
                            interaction.token_rewards = token_rewards
                    else:
                        interaction.token_rewards = token_rewards
                    # Fallback: set scalar reward from token rewards only if None
                    if interaction.reward is None:
                        interaction.reward = sum(token_rewards)

            # Attach position-level rewards to interaction objects for
            # distillation loss. These are stored as a Python attribute
            # (not in to_tensor_dict) and flow through to mb_input.
            for interaction_id, pos_rewards in self._position_rewards.items():
                if interaction_id in self.completions:
                    interaction = self.completions[interaction_id]
                    interaction.position_rewards = pos_rewards  # type: ignore[attr-defined]

        # Call base export
        return super().export_interactions(discount, style)
```

- [ ] **Step 6: Run the server-focused tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py::TestSessionDataUsesExtendedCache customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py::TestExportInteractionsReturnType customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py::TestScalarRewardPreservedAfterDelegation -v 2>&1 | tail -30`

Expected: PASS

- [ ] **Step 7: Run the existing server tests to verify no regressions**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_server.py customized_areal/on_policy_distill/proxy/tests/test_bug_fixes.py -v 2>&1 | tail -40`

Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/server.py
git commit -m "feat(proxy): use extended InteractionCache and type export_interactions as InteractionWithTokenLevelReward"
```

______________________________________________________________________

### Task 3: Deserialization — Create `InteractionWithTokenLevelReward` objects

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py:154-207`

Change `deserialize_interactions_with_position_rewards` to create
`InteractionWithTokenLevelReward` instead of `InteractionWithTokenLogpReward`.

- [ ] **Step 1: Update the deserialization function**

In `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`, replace
`deserialize_interactions_with_position_rewards` (lines 154-207):

```python
def deserialize_interactions_with_position_rewards(
    data: dict,
) -> dict:
    """Deserialize interactions including position_rewards and token_rewards for distillation.

    Extends the base deserialize_interactions to reconstruct position_rewards
    and token_rewards from the serialized data. Creates InteractionWithTokenLevelReward
    objects so that token_rewards is a proper dataclass field and to_tensor_dict()
    correctly recomputes token_rewards after cache invalidation.
    """
    from areal.infra.rpc.serialization import deserialize_value

    from .types import InteractionWithTokenLevelReward

    data = deserialize_value(data)
    result = {}
    for key, item in data.items():
        interaction = InteractionWithTokenLevelReward()
        interaction._cache = item["tensor_dict"]
        interaction.reward = item["reward"]
        interaction.interaction_id = item["interaction_id"]

        # Set token_rewards via the typed dataclass field (validates length
        # if model_response is present). Unlike the base InteractionWithTokenLogpReward,
        # InteractionWithTokenLevelReward has token_rewards as a declared field,
        # so to_tensor_dict() correctly includes it after cache invalidation.
        token_rewards_data = item.get("token_rewards")
        if token_rewards_data is not None:
            interaction.token_rewards = token_rewards_data

        # Reconstruct position_rewards if available and inject into the
        # interaction object so they flow through to the distillation loss.
        pos_rewards_data = item.get("position_rewards")
        if pos_rewards_data is not None:
            from .server import PositionRewardInfo as PRI

            pos_rewards = [
                PRI(
                    position=pr["position"],
                    candidates=pr["candidates"],
                    candidate_token_ids=pr["candidate_token_ids"],
                    logprobs=pr["logprobs"],
                    rewards=pr["rewards"],
                    chosen_index=pr["chosen_index"],
                    sample_index=pr.get("sample_index", 0),
                )
                for pr in pos_rewards_data
            ]
            # Store as a Python attribute — the workflow extracts it after
            # to_tensor_dict() conversion and attaches it to the tensor dict.
            # We do NOT inject it into _cache to avoid concat_padded_tensors
            # key consistency issues when some interactions have position_rewards
            # and others don't (e.g., multi-turn conversations).
            interaction.position_rewards = pos_rewards  # type: ignore[attr-defined]

        result[key] = interaction
    return result
```

- [ ] **Step 2: Run deserialization tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py::TestDeserializationCreatesCorrectType -v 2>&1 | tail -20`

Expected: PASS

- [ ] **Step 3: Run the Bug 4 serialization tests (they directly test the round-trip)**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_bug_fixes.py::TestBug4_TokenRewardsLostInSerialization -v 2>&1 | tail -20`

Expected: PASS (the mock-based tests should work because `Mock` objects still have
`token_rewards` and `to_tensor_dict` as set up)

- [ ] **Step 4: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/proxy_rollout_server.py
git commit -m "feat(proxy): deserialize as InteractionWithTokenLevelReward for proper token_rewards support"
```

______________________________________________________________________

### Task 4: Client — Type `export_interactions` return as `TokenRewardInteractions`

**Files:**

- Modify: `customized_areal/on_policy_distill/proxy/client.py:312-316`

- [ ] **Step 1: Add the type import and update the return type**

In `customized_areal/on_policy_distill/proxy/client.py`, add the import at the top
(after existing imports from `.proxy_rollout_server` and `.server`):

```python
from .types import TokenRewardInteractions
```

Then update the `export_interactions` method signature:

```python
    async def export_interactions(
        self,
        discount: float = 1.0,
        style: str = "individual",
    ) -> TokenRewardInteractions:
```

And update the docstring:

```python
        """Export interactions with position_rewards support.

        Overrides the base class method to use custom deserialization
        that reconstructs position_rewards for the distillation loss.
        Returns InteractionWithTokenLevelReward objects with proper
        token_rewards dataclass fields.
        Uses post_json_with_retry for resilience against transient failures.
        """
```

- [ ] **Step 2: Run the full new test suite**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py -v 2>&1 | tail -30`

Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/on_policy_distill/proxy/client.py
git commit -m "feat(proxy): type export_interactions return as TokenRewardInteractions"
```

______________________________________________________________________

### Task 5: Run full test suite and verify end-to-end

**Files:**

- No new files

- [ ] **Step 1: Run all proxy tests**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/ -v --timeout=60 -k "not GPU" 2>&1 | tail -60`

Expected: All PASS

- [ ] **Step 2: Run the integration tests (excluding GPU tests)**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest customized_areal/on_policy_distill/proxy/tests/test_server_integration.py -v --timeout=120 -k "not GPU" 2>&1 | tail -40`

Expected: All PASS — these tests verify the full HTTP round-trip including
serialization/deserialization with token_rewards and position_rewards

- [ ] **Step 3: Run linter**

Run:
`cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pre-commit run --files customized_areal/on_policy_distill/proxy/server.py customized_areal/on_policy_distill/proxy/proxy_rollout_server.py customized_areal/on_policy_distill/proxy/client.py customized_areal/on_policy_distill/proxy/tests/test_return_type_fix.py 2>&1 | tail -30`

Expected: No errors

- [ ] **Step 4: Verify the end-to-end data flow manually**

Quick sanity check that the types chain correctly. Run:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "
from customized_areal.on_policy_distill.proxy.server import TokenRewardSessionData
from customized_areal.on_policy_distill.proxy.cache import InteractionCache as ExtendedCache
from customized_areal.on_policy_distill.proxy.types import InteractionWithTokenLevelReward, TokenRewardInteractions
from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient

# Verify TokenRewardSessionData uses extended cache
s = TokenRewardSessionData('test')
assert isinstance(s.completions, ExtendedCache), 'Wrong cache type'

# Verify return type annotation
import inspect
sig = inspect.signature(s.export_interactions)
ret = sig.return_annotation
print(f'export_interactions return type: {ret}')
assert ret == dict[str, InteractionWithTokenLevelReward], f'Wrong return type: {ret}'

# Verify client return type
sig2 = inspect.signature(OpenAIProxyClient.export_interactions)
ret2 = sig2.return_annotation
print(f'client.export_interactions return type: {ret2}')
assert ret2 == TokenRewardInteractions, f'Wrong client return type: {ret2}'

print('All type checks passed!')
"
```

Expected: `All type checks passed!`

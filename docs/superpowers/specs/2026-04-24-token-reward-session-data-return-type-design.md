# TokenRewardSessionData.export_interactions Return Type Fix

## Problem

`TokenRewardSessionData.export_interactions` declares return type
`dict[str, InteractionWithTokenLogpReward]` (base type), but the method enriches
interactions with `token_rewards` and `position_rewards` before returning them.
Downstream consumers need `InteractionWithTokenLevelReward` — the declared type hides
the extended fields.

## End-to-End Data Flow

The full round-trip from server to workflow must preserve position-level tokens and
rewards:

```
Server                              Client / Workflow
──────                              ─────────────────
TokenRewardSessionData
  .set_token_rewards()
  .set_position_rewards()
  .export_interactions() ──►  serialize_interactions_with_position_rewards()
    applies token_rewards        includes token_rewards + position_rewards
    applies position_rewards     includes to_tensor_dict() (has token_rewards key)
           │
           ▼
    HTTP response
           │
           ▼
                              deserialize_interactions_with_position_rewards()
                                creates InteractionWithTokenLogpReward()  ← WRONG TYPE
                                sets .token_rewards as dynamic attr
                                sets .position_rewards as dynamic attr
                                pre-populates ._cache from serialized tensor dict
           │
           ▼
                              workflow.arun_episode()
                                interactions = client.export_interactions()
                                v.to_tensor_dict()  ← uses _cache (token_rewards included)
                                getattr(v, "position_rewards")  ← dynamic attr
                                → concat_padded_tensors()
                                → tensor_dict["position_rewards"] = all_position_rewards
```

### What works today

- `token_rewards` flows through `to_tensor_dict()` because it's stored in `_cache` at
  deserialization time, and the serialized tensor dict includes the `token_rewards` key
  (set server-side by `InteractionWithTokenLevelReward.to_tensor_dict()`).
- `position_rewards` flows as a Python dynamic attribute: serialized separately,
  deserialized and set on the interaction object, extracted via `getattr` in
  `workflow.arun_episode()`.

### What's broken

1. **Wrong return type**: `export_interactions` returns
   `dict[str, InteractionWithTokenLogpReward]` but the objects have `token_rewards` and
   `position_rewards` — consumers can't see these.

1. **Wrong deserialization type**: `deserialize_interactions_with_position_rewards()`
   creates `InteractionWithTokenLogpReward()` objects instead of
   `InteractionWithTokenLevelReward()`. This means:

   - `token_rewards` is set as a dynamic attribute, not through the typed
     `set_token_rewards()` method — no length validation, no `_cache` invalidation.
   - If `to_tensor_dict()` is ever called without `_cache` (recomputed from scratch),
     the base class implementation won't include `token_rewards` or `token_reward_mask`
     keys — they'd be silently lost.
   - `position_rewards` is always a dynamic attribute (not a dataclass field on either
     type), but `InteractionWithTokenLevelReward` is the semantically correct container.

1. **Wrong cache type on server**: `TokenRewardSessionData` inherits
   `self._completions = InteractionCache()` from `SessionData.__init__`, which creates
   the **base** `InteractionCache` from `areal.experimental.openai.cache`. The
   `export_interactions` override sets `token_rewards` via `hasattr` guard — fragile and
   untyped.

## Changes

### Change 1: Server — Override `__init__` to use extended InteractionCache

Replace `self._completions` with an instance of the extended `InteractionCache` from
`customized_areal/on_policy_distill/proxy/cache.py`.

**Why**: The extended cache stores `InteractionWithTokenLevelReward` objects and has
built-in `set_rewards()`, `set_position_rewards()`, and `export_interactions()` that
work with the extended type.

**File**: `customized_areal/on_policy_distill/proxy/server.py`

```python
from .cache import InteractionCache as ExtendedInteractionCache

class TokenRewardSessionData(SessionData):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self._completions = ExtendedInteractionCache()
        self._token_rewards: dict[str, list[float]] = {}
        self._position_rewards: dict[str, list[PositionRewardInfo]] = {}
        self._lock = threading.Lock()
```

### Change 2: Server — Widen return type of `export_interactions`

**File**: `customized_areal/on_policy_distill/proxy/server.py`

```python
from .types import InteractionWithTokenLevelReward

def export_interactions(
    self, discount: float, style: str
) -> dict[str, InteractionWithTokenLevelReward]:
```

### Change 3: Server — Delegate reward application to the extended cache

Delegate `set_token_rewards` and `set_position_rewards` to the extended cache at
set-time, instead of staging in separate dicts and replaying during export.

**Preserve scalar reward invariant**: The extended cache's `set_rewards()` overwrites
`interaction.reward` with `sum(token_rewards)` (see `cache.py:394`). The current
`TokenRewardSessionData` explicitly preserves the scalar reward set via `set_reward()`
and only falls back to `sum(token_rewards)` when `reward is None`. Save and restore the
scalar reward after delegation.

**File**: `customized_areal/on_policy_distill/proxy/server.py`

```python
def set_token_rewards(self, interaction_id: str, token_rewards: list[float]) -> None:
    if self.is_completed:
        raise RuntimeError(...)
    with self._lock:
        self._token_rewards[interaction_id] = token_rewards
        if interaction_id in self.completions:
            saved_reward = self.completions[interaction_id].reward
            self.completions.set_rewards(interaction_id, token_rewards)
            if saved_reward is not None:
                self.completions[interaction_id].reward = saved_reward

def set_position_rewards(self, interaction_id: str, position_rewards: list[PositionRewardInfo]) -> None:
    if self.is_completed:
        raise RuntimeError(...)
    with self._lock:
        self._position_rewards[interaction_id] = position_rewards
        if interaction_id in self.completions:
            self.completions.set_position_rewards(interaction_id, position_rewards)

def export_interactions(self, discount: float, style: str) -> dict[str, InteractionWithTokenLevelReward]:
    # Apply rewards for interactions not yet in the cache at set-time.
    with self._lock:
        for interaction_id, token_rewards in self._token_rewards.items():
            if interaction_id in self.completions:
                interaction = self.completions[interaction_id]
                # Primary: typed setter
                if hasattr(interaction, "set_token_rewards"):
                    try:
                        interaction.set_token_rewards(token_rewards)
                    except (ValueError, AttributeError):
                        interaction.token_rewards = token_rewards
                else:
                    interaction.token_rewards = token_rewards
                if interaction.reward is None:
                    interaction.reward = sum(token_rewards)
        for interaction_id, pos_rewards in self._position_rewards.items():
            if interaction_id in self.completions:
                interaction = self.completions[interaction_id]
                interaction.position_rewards = pos_rewards
    return super().export_interactions(discount, style)
```

### Change 4: Client deserialization — Create `InteractionWithTokenLevelReward` objects

`deserialize_interactions_with_position_rewards()` currently creates
`InteractionWithTokenLogpReward()` objects. Change it to create
`InteractionWithTokenLevelReward()` objects so that `token_rewards` is a proper
dataclass field with validation.

**File**: `customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

```python
def deserialize_interactions_with_position_rewards(data: dict) -> dict:
    from .types import InteractionWithTokenLevelReward
    from areal.infra.rpc.serialization import deserialize_value

    data = deserialize_value(data)
    result = {}
    for key, item in data.items():
        interaction = InteractionWithTokenLevelReward()
        interaction._cache = item["tensor_dict"]
        interaction.reward = item["reward"]
        interaction.interaction_id = item["interaction_id"]

        # Set token_rewards via the typed field (validates length if model_response present)
        token_rewards_data = item.get("token_rewards")
        if token_rewards_data is not None:
            interaction.token_rewards = token_rewards_data

        # Reconstruct position_rewards
        pos_rewards_data = item.get("position_rewards")
        if pos_rewards_data is not None:
            from .server import PositionRewardInfo as PRI
            pos_rewards = [PRI(...) for pr in pos_rewards_data]
            interaction.position_rewards = pos_rewards  # dynamic attribute

        result[key] = interaction
    return result
```

**Why this matters for `to_tensor_dict()`**: Currently, if `_cache` is populated,
`to_tensor_dict()` returns the cached dict directly. This works because the serialized
tensor dict already contains `token_rewards` and `token_reward_mask` keys. But if
`_cache` is ever invalidated (e.g., by `set_token_rewards()` via
`interaction._cache = None`), the next `to_tensor_dict()` call would recompute from
scratch. With a base-type object, this recomputation would NOT include `token_rewards`
keys. With `InteractionWithTokenLevelReward`, the override correctly includes them.

### Change 5: Client `export_interactions` — Type the return value

**File**: `customized_areal/on_policy_distill/proxy/client.py`

```python
from .types import InteractionWithTokenLevelReward, TokenRewardInteractions

async def export_interactions(
    self,
    discount: float = 1.0,
    style: str = "individual",
) -> TokenRewardInteractions:
```

This gives the workflow proper type information when accessing `.token_rewards`,
`.position_rewards`, and `.set_token_rewards()` on returned interactions.

### Change 6: Workflow — Type-narrow the interactions dict

**File**: `customized_areal/on_policy_distill/proxy/workflow.py`

The `arun_episode` method currently does
`getattr(interaction, "position_rewards", None)` because the declared type is
`InteractionWithTokenLogpReward`. After Changes 4–5, the type is
`InteractionWithTokenLevelReward`, and `position_rewards` is a documented dynamic
attribute. No logic change needed — the `getattr` call remains correct since
`position_rewards` is a dynamic attribute (not a dataclass field on any type).

## Summary of what flows through after all changes

| Data                             | Server → Serialize                                                              | Client Deserialize                                              | Workflow Access                                            |
| -------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- |
| `token_rewards` (in tensor_dict) | `InteractionWithTokenLevelReward.to_tensor_dict()` includes `token_rewards` key | `_cache` has `token_rewards` key; `to_tensor_dict()` returns it | `concat_padded_tensors()` includes it                      |
| `token_rewards` (field)          | Set via `set_token_rewards()` on `InteractionWithTokenLevelReward`              | Set via `InteractionWithTokenLevelReward.token_rewards` field   | Available as `.token_rewards` with proper type             |
| `position_rewards`               | Set as dynamic attr during export; serialized separately                        | Set as dynamic attr on `InteractionWithTokenLevelReward`        | `getattr(interaction, "position_rewards")` — same as today |
| `token_reward_mask`              | `InteractionWithTokenLevelReward.to_tensor_dict()` includes it                  | `_cache` has it                                                 | `concat_padded_tensors()` includes it                      |

## Affected callers

| File                          | Line                                       | Impact                                           |
| ----------------------------- | ------------------------------------------ | ------------------------------------------------ |
| `proxy_rollout_server.py:695` | `session_data.export_interactions()`       | Return type widens                               |
| `proxy_rollout_server.py:513` | `completions.set_reward()`                 | Extended cache has same API                      |
| `proxy_rollout_server.py:507` | `completions.last_interaction_id`          | Extended cache has same property                 |
| `client.py:312-339`           | `export_interactions()`                    | Return type narrows to `TokenRewardInteractions` |
| `workflow.py:294-297`         | `client.export_interactions()`             | Gets `TokenRewardInteractions` — proper typing   |
| `workflow.py:332`             | `getattr(interaction, "position_rewards")` | Still dynamic attr; works the same               |

## Tests

Existing tests should pass. Key scenarios to verify:

1. `set_token_rewards` → `export_interactions` → serialize → deserialize →
   `to_tensor_dict()` includes `token_rewards` key
1. `set_position_rewards` → `export_interactions` → serialize → deserialize →
   `getattr(interaction, "position_rewards")` returns `PositionRewardInfo` list
1. Scalar reward set via `set_reward()` is NOT overwritten by token rewards
1. `set_token_rewards` after session `finish()` raises `RuntimeError`
1. `export_interactions` on empty session returns `{}`
1. `to_tensor_dict()` works correctly when `_cache` is invalidated (recomputed from
   `InteractionWithTokenLevelReward`)

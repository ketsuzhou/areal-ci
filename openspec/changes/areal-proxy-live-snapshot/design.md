## Context

`areal/experimental/openai/proxy/` is AReaL's experimental OpenAI-compatible RL proxy. Per
session, an `InteractionCache` (an `OrderedDict[str, InteractionWithTokenLogpReward]`)
records each `/chat/completions` turn in insertion order, keyed by `interaction_id`. The
session lifecycle is:

```
start_session → chat/completions ×N (append turns) → set_reward (terminal reward)
              → end_session (finish: _completed_event set) → export_trajectories (harvest)
```

`export_trajectories` (`proxy_rollout_server.py:988`) is the only endpoint that returns a
session's interactions. Its handler:

1. `await session_data.wait_for_finish()` — blocks until `end_session` / `finish()`;
2. `session_data.export_interactions(discount, style)` — which unconditionally calls
   `apply_reward_discount` (one-shot, backward RL reward propagation from the terminal
   turn, `cache.py:96`);
3. `_session_cache.pop(session_id)` + `_remove_api_keys_for_session(session_id)` —
   destructive cleanup.

So the endpoint is a terminal, one-shot, destructive harvest operation. There is no
non-blocking, mid-session way to read a session's interactions or turn count.

Sub-project G needs the canonical turn index at communication-event time
(mid-execution). Reading AReaL's own `InteractionCache` (rather than a Multca-maintained
counter) eliminates turn-index drift, because the count is the same object AReaL's
`list[Node]` assembler positions derive from.

## Goals / Non-Goals

**Goals:**

- Add a read-only, non-terminal, no-discount mode to `export_trajectories` that returns
  the current `InteractionCache` snapshot at any point during the rollout.
- Preserve the existing harvest path byte-for-byte (wait + discount + cleanup), as
  default.
- Make the live snapshot race-free against concurrent `chat/completions` appends.
- Do not burn the `apply_reward_discount` one-shot flag from a live snapshot.

**Non-Goals:**

- Changing the harvest path's semantics, its default request shape, or its cleanup.
- Reward computation, critic/GAE, or any RL training logic.
- Multca-side DAG recording (that is sub-project G; this change only provides the read
  endpoint G consumes).
- Any Pi / River2.0 change.

## Decisions

### D1 — Conditional branch on `export_trajectories` (not a separate endpoint)

Add a mode flag on `ExportTrajectoriesRequest` (e.g., `mode: "harvest" | "snapshot"`,
default `harvest`) that branches the handler. `harvest` is the existing path unchanged;
`snapshot` is the new live path.

**Rationale:** the user's directive — preserve original functionality, implement the
proposal as another conditional branch in `proxy_rollout_server.py`. Reuses the existing
endpoint, auth, and `serialize_interactions` machinery.

**Alternative considered:** a separate read-only endpoint (e.g.,
`GET /rl/session/{id}/snapshot`) — cleaner isolation from the harvest path, but rejected
per the directive to implement as a conditional branch on the existing handler. The mode
flag keeps the two paths in one function but branch-isolated; the spec enforces that
`harvest` is unchanged.

### D2 — `snapshot` branch skips wait, discount, and cleanup

The `snapshot` branch:

- does NOT call `wait_for_finish` (returns immediately);
- does NOT call `apply_reward_discount` (routes through
  `InteractionCache.export_interactions` with `reward_discount=None`, which already skips
  when `None` — `cache.py:323`);
- does NOT `_session_cache.pop` or revoke keys (session stays live).

**Rationale:** each of the three is what makes harvest terminal / destructive / one-shot.
Skipping all three is what makes a mid-session read safe and repeatable.

### D3 — Shared lock on `InteractionCache` (write + read)

Add a lock to `InteractionCache` (or the session) acquired by `__setitem__` (the
chat/completions append path, which builds parent-child links while iterating) and by the
`snapshot` read. The harvest path does not need the lock (it runs after `finish()`, no
concurrent writes).

**Rationale:** `InteractionCache` is an unlocked `OrderedDict`; a mid-session snapshot
iterating / dict-copying while `__setitem__` mutates can raise `dictionary changed size
during iteration` or yield a torn interaction. The lock must be on the write path too, or
a read-only lock is useless.

**Trade-off:** the lock sits on the hot chat/completions write path. Keep it thin: hold
only across the `__setitem__` mutation (parent-child link build), not across the LLM
call. Snapshot reads are rare (a handful per rollout, at communication events) and short
(a dict copy).

### D4 — `snapshot` returns the same serialized shape as `harvest`

The `snapshot` branch returns
`ExportTrajectoriesResponse(interactions=serialize_interactions(...))` — the same shape as
harvest, just without reward discounting (rewards are whatever `set_reward` has set so
far, or `None`). Callers (G) read `len(interactions)` (= current turn count) and
`last_interaction_id` (= latest turn) from it.

**Rationale:** reuses `serialize_interactions`; no new response model. G only needs the
count + last id, but returning the full interactions is cheap and lets G do semantic
verification (match a communication event to the `tool_calls` in a specific turn's
`output_message_list`).

### D5 — `snapshot` is best-effort consistent: only fully-appended turns are visible

Because the lock serializes append vs. snapshot, a snapshot returns a consistent prefix of
completed turns. If a communication event fires while turn N+1 is mid-append, the snapshot
sees up to turn N (turn N is the closing-event turn). This matches G's need: the event
fires after Pi received turn N's response and executed the tool call, so turn N is
complete.

## Risks / Trade-offs

- **Lock contention on the write path** → Mitigation: hold the lock only across
  `__setitem__`'s mutation, not the LLM call; snapshot reads are rare. Benchmark if
  rollouts are high-throughput.
- **Burning the one-shot discount flag** → Mitigation: `snapshot` routes through
  `export_interactions(reward_discount=None)`, which does NOT call
  `apply_reward_discount` and does NOT set `_apply_reward_discount_called`. Test: snapshot
  mid-session, then harvest, then assert discount was applied exactly once at harvest.
- **Harvest path regression** → Mitigation: a golden test asserting the harvest branch's
  behavior (wait, discount, cleanup) is unchanged before/after the change.
- **Snapshot of an unknown session** → 404, same as harvest.
- **G-side race (not this change's concern)**: the small window between turn-N completion
  and the communication event reaching Multca is G's to handle; this change only
  guarantees the snapshot is internally consistent.

## Migration Plan

None. Pure code change to `areal/experimental/openai/proxy/`; no DB, no schema, no
config. Rollback = revert the branch (the default `harvest` path is unchanged, so behavior
is identical to today when the flag is absent).

## Open Questions

- Exact request field name/shape (`live: bool` vs `mode: "harvest"|"snapshot"`) — resolve
  in Task 1; `mode` is more extensible if a third mode ever appears.
- Whether the lock lives on `InteractionCache` or on the session object — resolve in Task
  1 based on where `__setitem__` is most cheaply gated.
- Whether G reads this via db_bridge (forwarding) or directly — G's concern; this change
  only guarantees the endpoint exists and is correct.

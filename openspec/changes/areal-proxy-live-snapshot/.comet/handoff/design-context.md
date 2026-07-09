# Comet Design Handoff

- Change: areal-proxy-live-snapshot
- Phase: design
- Mode: compact
- Context hash: f08334ae257ed4894ee913b108a7d35fd91f1b75d8490ff88c9cae9b829c98aa

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/areal-proxy-live-snapshot/proposal.md

- Source: openspec/changes/areal-proxy-live-snapshot/proposal.md
- Lines: 1-73
- SHA256: 4c753e8ff8687041e08d0078112547307811af41e9bbacb68c1ea4624df926d2

```md
## Why

The AReaL experimental openai-proxy exposes a session's recorded LLM interactions only
through `export_trajectories`, and that endpoint is gated on session completion: it
`await`s `wait_for_finish()`, then runs the one-shot `apply_reward_discount` (RL credit
assignment), then destructively removes the session from the cache and revokes its API
key. That is correct for end-of-rollout training harvest, but it means **no caller can
read a session's interactions while the rollout is still running** — the only
interaction-returning endpoint blocks until session end and would destroy the live
session on return.

Sub-project G (multica interaction-DAG) needs the canonical, AReaL-sourced turn index at
the moment a communication event fires mid-execution, to stamp `end_turn_idx` without the
drift risk of Multca maintaining its own parallel turn counter. That read is impossible
today. This change adds a non-destructive, no-discount, mid-session read mode so the
canonical turn state is reachable while the rollout runs, **without altering the existing
training-harvest path**.

## What Changes

- **Live-snapshot branch in `export_trajectories`** (`proxy_rollout_server.py`): a
  request-selected conditional branch that returns the session's current
  `InteractionCache` snapshot **without** `wait_for_finish`, **without**
  `apply_reward_discount`, and **without** removing the session from `_session_cache` or
  revoking its key. The original branch (wait → discount → export → cleanup) is preserved
  unchanged and remains the default.
- **Shared lock on `InteractionCache`** (`cache.py`): a lock acquired by both the
  chat/completions write path (`__setitem__`) and the live-snapshot read, so a mid-session
  snapshot cannot race with an in-flight turn append. Today `InteractionCache` is an
  unlocked `OrderedDict` and `__setitem__` builds parent-child links while iterating
  existing entries.
- **Skip-discount path** (`server.py`): route the live branch through the cache's
  `export_interactions` with `reward_discount=None` (already supported) so the one-shot
  `_apply_reward_discount_called` flag is NOT burned by a live snapshot, leaving the later
  training harvest intact.
- **Request shape**: extend `ExportTrajectoriesRequest` with a mode flag (e.g., `live:
  bool` or `mode: "harvest" | "snapshot"`) selecting the branch. `snapshot` is read-only
  and non-terminal; default (`harvest`) is the existing behavior.

## Capabilities

### New Capabilities

- `live-interaction-snapshot`: the AReaL openai-proxy can return a non-destructive,
  no-discount snapshot of a session's current interactions at any point during the
  rollout, via a mode flag on `export_trajectories`, without affecting the existing
  end-of-session training harvest.

### Modified Capabilities

<!-- None. training-session-lifecycle covers Multca-side session open/close/routing
     hooks, not the proxy's export semantics; critic-driven-training-signal is
     unaffected. -->

## Impact

- **AReaL experimental openai-proxy** (primary): `proxy_rollout_server.py` (conditional
  branch + request shape), `cache.py` (shared lock on `InteractionCache`), `server.py`
  (skip-discount wiring). All within `areal/experimental/openai/proxy/`.
- **Training-harvest path unchanged**: the default `harvest` branch keeps
  `wait_for_finish` + `apply_reward_discount` + destructive cleanup; RL trajectory export
  and reward discounting are byte-for-byte unchanged.
- **No Pi / River2.0 change**: Pi remains a black box; this only adds a read mode on the
  AReaL proxy.
- **Enables sub-project-g**: G's DAG recorder reads the canonical turn index
  (`len(interactions)` + `last_interaction_id`) from AReaL's own cache at
  communication-event time, eliminating the turn-index drift risk of a Multca-maintained
  counter. G's `design.md` should record this as the turn-truth source (a D5 alternative).
- **Concurrency**: the new lock is the one cross-file change with blast radius — it sits
  on the hot chat/completions write path, so it must be thin and non-contended (snapshot
  reads are rare and short).
- **Experimental stack**: `areal/experimental/openai/` is experimental (low process risk),
  but the write-path lock warrants a focused concurrency test.
```

## openspec/changes/areal-proxy-live-snapshot/design.md

- Source: openspec/changes/areal-proxy-live-snapshot/design.md
- Lines: 1-148
- SHA256: 2097d0d345bca98152c4861b4d3e8ca7c602ea784753afee8ba2a8c6154e05bd

[TRUNCATED]

```md
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

```

Full source: openspec/changes/areal-proxy-live-snapshot/design.md

## openspec/changes/areal-proxy-live-snapshot/tasks.md

- Source: openspec/changes/areal-proxy-live-snapshot/tasks.md
- Lines: 1-88
- SHA256: bb26b500e3f9fc706c7943b95116f27237d943aac5993ef9ff16af2ea97be10d

[TRUNCATED]

```md
## 1. Investigation — lock placement, write path, request shape

- [ ] 1.1 Confirm where `InteractionCache.__setitem__` is called from the
  `chat/completions` path (`client.py` / `workflow.py`) and the minimal critical section
  for the shared lock (parent-child link build only, not the LLM call).
- [ ] 1.2 Decide lock ownership: on `InteractionCache` vs the session object — pick the
  one that gates `__setitem__` cheapest.
- [ ] 1.3 Confirm `InteractionCache.export_interactions` skips `apply_reward_discount`
  when `reward_discount=None` (`cache.py:323`), so the `snapshot` branch can reuse it
  without burning the one-shot flag.
- [ ] 1.4 Decide request field: `mode: "harvest"|"snapshot"` (default `harvest`) vs
  `live: bool`. Document choice.
- [ ] 1.5 Commit: `docs(areal-proxy-live-snapshot): T1 investigation`.

## 2. Shared lock on InteractionCache (TDD)

**Files:** `areal/experimental/openai/proxy/cache.py`, tests.

- [ ] 2.1 Failing test: a `snapshot`-style read (iterate + dict-copy) concurrent with
  `__setitem__` does not raise `dictionary changed size during iteration` and returns only
  fully-appended interactions.
- [ ] 2.2 Failing test: `__setitem__` parent-child link build is consistent under
  concurrent appends (no torn parent links).
- [ ] 2.3 Implement the lock: held by `__setitem__` across the mutation and by the
  snapshot read; harvest path (post-finish) unaffected.
- [ ] 2.4 Commit: `feat(areal-proxy-live-snapshot): shared lock on InteractionCache`.

## 3. Skip-discount path (TDD)

**Files:** `areal/experimental/openai/proxy/server.py`, `cache.py`, tests.

- [ ] 3.1 Failing test: calling the snapshot path mid-session does NOT set
  `_apply_reward_discount_called`.
- [ ] 3.2 Failing test: after a mid-session snapshot, a subsequent harvest still applies
  `apply_reward_discount` exactly once (no "should only be called once" error).
- [ ] 3.3 Wire the snapshot branch to `InteractionCache.export_interactions(reward_discount=None)`
  (or equivalent) so `server.export_interactions`'s unconditional `apply_reward_discount`
  is bypassed for snapshot.
- [ ] 3.4 Commit: `feat(areal-proxy-live-snapshot): skip-discount path for snapshot`.

## 4. Live-snapshot branch in export_trajectories (TDD)

**Files:** `areal/experimental/openai/proxy/proxy_rollout_server.py`, tests.

- [ ] 4.1 Failing test: `export_trajectories` with mode `snapshot` on a running session
  returns immediately (no `wait_for_finish`) with the current interactions; session
  remains in `_session_cache` with its API key still valid.
- [ ] 4.2 Failing test: after a `snapshot`, subsequent `chat/completions` for the same
  session succeed (session not destroyed).
- [ ] 4.3 Failing test: `snapshot` of an unknown session returns 404.
- [ ] 4.4 Failing test: `snapshot` response shape equals `harvest` response shape
  (`serialize_interactions`), with rewards undiscounted.
- [ ] 4.5 Implement the conditional branch: `mode == "snapshot"` → lock + skip-discount
  export + return (no wait, no cleanup); default `harvest` → existing path unchanged.
- [ ] 4.6 Extend `ExportTrajectoriesRequest` with the mode flag (default `harvest`).
- [ ] 4.7 Commit: `feat(areal-proxy-live-snapshot): live-snapshot branch on export_trajectories`.

## 5. Harvest-path regression guard (TDD)

**Files:** tests.

- [ ] 5.1 Failing test (golden): the `harvest` branch (default) waits for finish, applies
  `apply_reward_discount` once, exports, and removes the session + revokes the key —
  identical to pre-change behavior.
- [ ] 5.2 Failing test: `harvest` after one or more `snapshot` calls produces the same
  discounted trajectory as `harvest` with no prior snapshot.
- [ ] 5.3 Commit: `test(areal-proxy-live-snapshot): harvest-path regression guard`.

## 6. Full regression + grep sweep

- [ ] 6.1 Scoped pytest on `areal/experimental/openai/proxy/` (concurrency tests
  included); `uv run pytest areal/experimental/openai/proxy/`.
- [ ] 6.2 Grep sweep: `export_trajectories`, `apply_reward_discount`, `wait_for_finish`,
  `_session_cache.pop` resolve to intended code only; no stray discount calls in the
  snapshot path.
- [ ] 6.3 Note for sub-project-g: this endpoint is G's canonical turn-truth source; G's
  `design.md` D5 should be revised to read the turn index from `len(interactions)` +
  `last_interaction_id` via snapshot mode (not a Multca-maintained counter).
- [ ] 6.4 Final review → READY TO MERGE / NEEDS_CHANGES.
- [ ] 6.5 Commit: `docs(areal-proxy-live-snapshot): T6 regression + grep sweep`.
```

Full source: openspec/changes/areal-proxy-live-snapshot/tasks.md

## openspec/changes/areal-proxy-live-snapshot/specs/live-interaction-snapshot/spec.md

- Source: openspec/changes/areal-proxy-live-snapshot/specs/live-interaction-snapshot/spec.md
- Lines: 1-70
- SHA256: dbc88ad3a0a0ff62ecbbb230c320f56a08028293f6f450c4744fc2a83810d176

```md
## ADDED Requirements

### Requirement: Live snapshot mode on export_trajectories

The AReaL openai-proxy `export_trajectories` endpoint SHALL support a snapshot mode that
returns a session's current interactions at any point during the rollout, selected by a
mode flag on the request (default `harvest`).

The `snapshot` mode SHALL:

- return immediately without waiting for the session to finish;
- NOT apply reward discounting (`apply_reward_discount` SHALL NOT be called);
- NOT remove the session from the session cache or revoke its API key;
- return the same serialized interaction shape as `harvest` mode.

The `harvest` mode (default) SHALL remain unchanged: wait for finish, apply one-shot
reward discount, export, then remove the session and revoke its key.

#### Scenario: Snapshot mid-session returns current interactions without ending the session

- **WHEN** a caller sends `export_trajectories` with mode `snapshot` for a session that
  is still running (not finished)
- **THEN** the endpoint returns immediately with the session's current interactions (no
  `wait_for_finish`), the session remains in the cache with its API key valid, and
  subsequent `chat/completions` calls for that session continue to succeed

#### Scenario: Harvest mode is unchanged after snapshot

- **WHEN** a session is snapshotted mid-rollout (mode `snapshot`) and later harvested
  after `end_session` (mode `harvest`, the default)
- **THEN** the harvest applies `apply_reward_discount` exactly once, exports the full
  trajectory, and removes the session — identical to behavior when no snapshot was taken

#### Scenario: Default mode is harvest

- **WHEN** `export_trajectories` is called without a mode flag (or with mode `harvest`)
- **THEN** the endpoint behaves exactly as before this change: wait for finish, discount,
  export, cleanup

#### Scenario: Snapshot of unknown session

- **WHEN** `export_trajectories` with mode `snapshot` is called for a session id not in
  the cache
- **THEN** the endpoint returns 404

### Requirement: Race-free interaction snapshot

The `snapshot` read of a session's `InteractionCache` SHALL be guarded by a lock that is
also held by the `chat/completions` write path (`InteractionCache.__setitem__`), so that a
snapshot cannot observe a partially-appended interaction or raise a concurrent-mutation
error.

#### Scenario: Concurrent append during snapshot

- **WHEN** a `chat/completions` turn is appending to a session's `InteractionCache` at the
  same moment a `snapshot` read is taken
- **THEN** the snapshot returns a consistent set of fully-appended interactions (no torn
  interaction, no "dictionary changed size during iteration")

### Requirement: Snapshot does not consume the reward-discount one-shot

A `snapshot` read SHALL NOT set the `apply_reward_discount` one-shot flag, so that a later
`harvest` can still apply reward discounting.

#### Scenario: Harvest discount still applies after one or more snapshots

- **WHEN** one or more `snapshot` reads are taken during a rollout, and the session is
  later harvested
- **THEN** `apply_reward_discount` is called exactly once at harvest time (not zero, not
  more than once), and raises no "should only be called once" error
```


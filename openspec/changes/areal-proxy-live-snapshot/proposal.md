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

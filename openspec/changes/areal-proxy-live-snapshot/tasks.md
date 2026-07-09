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

## Test runners / constraints

- AReaL proxy tests run from `backend/areal` with
  `uv run pytest areal/experimental/openai/proxy/`.
- Concurrency tests must use real threads/tasks to exercise the lock; mock-based tests
  will not catch the race.
- Experimental stack: no GPU required for these unit/concurrency tests.

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

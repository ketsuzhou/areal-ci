## Why

Sub-project D (training-agent session lifecycle) is mid-implementation: T1-T6 are
committed on multica `main` (7969187a..816d1e86c), but the session-close hook
(T7), config + pi `models.json` wiring (T8), and full regression (T9) remain.
Without T7-T9, trained-agent runs won't auto-close RL sessions (no default
reward + `end_session` on completion) and the proxy routing won't actually work
end-to-end — T6 found that pi has no base-url flag, so the `areal` provider
entry in `models.json` must read `$AREAL_PROXY_BASE_URL` for the trained pi to
route to the bridge stub. This change closes out D so the trained-agent flow is
complete end-to-end with a default-placeholder reward, deferring real reward to
sub-project E.

## What Changes

- **T7 — Session-close hook on task completion.** Shared `maybeCloseTrainingSession`
  helper called from `CompleteTask` / `FailTask` / `CancelTask` in
  `internal/service/task.go`; reads `proxy_key` + `session_id` from
  `task.context.areal_proxy`; calls `SetReward(default_reward)` then
  `EndSession` (session-key Bearer auth, order asserted); RL errors logged, not
  fatal. Sweeper timeout gap (`FailStaleTasks` raw SQL bypasses `FailTask`)
  documented as deferred.
- **T8 — Config + pi `models.json` production wiring.** Env vars
  `AREAL_PROXY_URL`, `AREAL_BRIDGE_STUB_URL`, `AREAL_ADMIN_API_KEY`,
  `TRAINING_DEFAULT_REWARD` (default 1.0); pi `models.json` `areal` provider
  entry with `baseURL=$AREAL_PROXY_BASE_URL` (closes the T6 finding); guard
  fails loud if training requested but bridge config missing.
- **T9 — Full regression + AReaL confirm + grep sweep.** Scoped Go build/test
  (webpush pre-existing failure excluded), db_bridge smoke, AReaL confirm-only
  (no code change expected), grep sweep for `train_agent_id` /
  `training_dispatch` / `areal_proxy` / `arealrl`.

## Capabilities

### New Capabilities
- `training-session-lifecycle`: Server-side lifecycle for RL training sessions —
  opens an AReaL proxy session when a trained-member task is created, routes the
  trained agent's LLM traffic through the proxy, and closes the session with a
  default placeholder reward on task completion.

### Modified Capabilities
(none — `openspec/specs/` is empty; this change creates the first capability
spec in this repo.)

## Impact

- **Code**: multica Go server (`internal/service/task.go`, `internal/arealrl/`,
  config loader, pi `models.json`); no AReaL code change expected (confirm-only).
- **DB**: migration 152 (`training_dispatch`) already landed in T3; no new
  migration in T7-T9.
- **External**: AReaL experimental openai-proxy stack
  (`areal/experimental/openai/proxy/proxy_gateway.py` + `proxy_rollout_server.py`)
  must be running; db_bridge stub channels `/rl/*` already exist (T4 confirmed).
- **Prior work**: T1-T6 on multica main (7969187a..816d1e86c) — full technical
  design in areal `docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md`;
  implementation plan in `docs/superpowers/plans/2026-07-06-training-agent-session-lifecycle.md`;
  SDD ledger in `.superpowers/sdd/progress.md`.
- **Deferred (non-goals)**: real reward / entropy / critic env-save (sub-project
  E); D6 env-dispatch rewire of the wired loop (needs external le-agent
  materialize endpoint); sweeper timeout session reaper (future hardening).

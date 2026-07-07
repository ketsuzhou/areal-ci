# Design — sub-project-d-session-lifecycle

## Context

This OpenSpec change closes out sub-project D (T7-T9). The full technical design
lives in `docs/superpowers/specs/2026-07-06-training-agent-session-lifecycle-design.md`;
this document summarizes the high-level architecture decisions and points to the
Superpowers design doc for implementation details.

## Prior work (T1-T6, on multica main 7969187a..816d1e86c)

- **T1** investigation: confirmed Approach A (server-side lifecycle) is feasible;
  chokepoints mapped (see `docs/superpowers/notes/2026-07-06-D-seams.md`).
- **T2** contract: `train_agent_id` on `EnvDispatchRequest` +
  `service.EnvDispatchInput`; validation rule (requires `squad_id` or == `agent_id`).
- **T3** persistence: migration 152 `training_dispatch(project_id PK FK ON DELETE
  CASCADE, workspace_id, train_agent_id, default_reward dp default 1.0, created_at)`;
  `SaveTrainingDispatch` on Deps + adapter + stub + fake.
- **T4** RL client: `internal/arealrl.Client` — `StartSession` (admin-key, flat
  `{session_id, api_key}` response), `SetReward` / `EndSession` (session-key
  Bearer); experimental contract confirmed against `proxy_rollout_server.py`.
- **T5** session-open hook: `maybeOpenTrainingSession` at all `Enqueue*`
  chokepoints (`service/task.go`) + env_dispatch `EnqueueAgentRun`; merges
  `context.areal_proxy = {provider, model, api_key:proxy_key, base_url, session_id}`;
  idempotent; loud error if training target but bridge dep missing.
- **T6** execenv wiring: `ClaimTaskByRuntime` parses `context.areal_proxy` →
  daemon Task/AgentData omitempty fields → ExecOptions (`pi --provider areal
  --model areal-default --api-key <proxy_key>`, `AREAL_PROXY_BASE_URL` env).
  **Finding**: pi has NO base-url flag → T8 must wire `models.json` provider entry.

## Locked decisions (D1-D6, from Superpowers design doc §3)

- **D1 Ownership** — AReaL hosts the LLM proxy + mints session key (`proxy_key`);
  multica orchestrates.
- **D2 Trigger** — external driver (AReaL) tells multica to launch the team;
  `env_dispatch` payload marks the training target via `train_agent_id`.
- **D3 `proxy_url`** — multica deployment config (env var, default
  `http://db_bridge_stub:9100/v1`); `start_session` does NOT return it.
- **D4 Reward boundary** — D writes default placeholder reward on close; real
  reward / entropy / critic env-save = sub-project E.
- **D5 Lifecycle owner** — server-side (Approach A); multica Go owns start/close;
  daemon stays a thin consumer of injected config.
- **D6 Team-member creation** — leader task only at dispatch; teammates spawned
  later (leader @mention delegation). Session-open hook fires at trained-member
  task creation, not at `env_dispatch`.

## RL contract (experimental openai-proxy stack)

Served ONLY by `areal/experimental/openai/proxy/proxy_gateway.py` +
`proxy_rollout_server.py` (the v2 `inference_service` gateway lacks `end_session`).

| Endpoint | Auth | Body | Returns |
|---|---|---|---|
| `start_session` | admin-key (Bearer/`x-api-key`) | `{task_id, group_size:1}` | flat `{session_id, api_key}` (api_key = `proxy_key`) |
| `set_reward` | session-key (Bearer `proxy_key`) | `{reward}` (no `session_id`) | — |
| `end_session` | session-key (Bearer `proxy_key`) | (no body) | — |

`task_id` = multica `agent_task_queue.id`.

## T7-T9 architecture

### T7: Session-close hook

```
   task terminal transition (service/task.go)
   ┌────────────────────────────────────────────────┐
   │  CompleteTask  │  FailTask  │  CancelTask      │
   └────────────────┴───────────┴───────────────────┘
                       │
                       ▼
            maybeCloseTrainingSession(ctx, task)
                       │
       ┌───────────────┼────────────────────────┐
       ▼               ▼                        ▼
  has context.   default_reward         SetReward(proxy_key, default)
  areal_proxy?   from training_dispatch       │
       │           (fallback config)          ▼
       │               │              EndSession(proxy_key)
       ▼               │                      │
     skip ◀── no ──────┘                      ▼
       │                                  RL errors logged
       │ yes                              (not fatal)
       └─→ SetReward → EndSession
```

- Shared helper `maybeCloseTrainingSession(ctx, task)` invoked from the three
  terminal transitions in `internal/service/task.go` (`CompleteTask` :1285,
  `FailTask` :1468, `CancelTask`/`CancelTaskWithResult` :905/:915).
- Read `session_id` + `api_key` (proxy_key) from `task.context.areal_proxy`;
  read `default_reward` from `training_dispatch` (fallback `TRAINING_DEFAULT_REWARD`).
- Call `SetReward(proxy_key, default_reward)` then `EndSession(proxy_key)`.
- RL errors logged, not fatal (best-effort close).
- Idempotent: tasks without `context.areal_proxy` are skipped.
- **Deferred gap**: `runtime_sweeper.FailStaleTasks` (`cmd/server/runtime_sweeper.go`,
  raw SQL) bypasses `FailTask`, so timed-out trainings won't auto-close.
  Documented; reaper is future hardening, NOT D scope.

### T8: Config + pi `models.json`

- Env vars (config loader): `AREAL_PROXY_URL` (default
  `http://db_bridge_stub:9100/v1`), `AREAL_BRIDGE_STUB_URL`, `AREAL_ADMIN_API_KEY`,
  `TRAINING_DEFAULT_REWARD` (default 1.0).
- Construct `arealrl.Client` and inject into the service via Deps.
- pi `models.json` `areal` provider entry: `baseURL = $AREAL_PROXY_BASE_URL`
  (closes the T6 finding — pi has no base-url flag, so T6 injects the URL as
  env var; T8 wires the provider entry to read it). Confirm the exact
  `models.json`/provider-config seam during implementation.
- Guard: if training requested (`train_agent_id` set on dispatch) but
  `AREAL_BRIDGE_STUB_URL` / admin key unset → fail the open-hook loudly (no
  un-proxied training run).

### T9: Regression

- Scoped Go: `go build ./internal/handler/ ./internal/service/
  ./internal/daemon/execenv/ ./internal/arealrl/ ./pkg/db/generated/` + `go vet`
  same + `go test` same (confirm only pre-existing 16 ON CONFLICT handler
  fails; 0 new). `go build ./...` is pre-existing-broken on `webpush.go:180`.
- `gofmt -l` clean on touched files.
- db_bridge smoke (if touched): `cd multica/db_bridge && uv run pytest -q`.
- AReaL confirm-only: verify `start_session` / `set_reward` / `end_session`
  exist and the bridge routes `/rl/*` to the gateway (no code change expected).
- grep sweep: `train_agent_id`, `training_dispatch`, `areal_proxy`, `arealrl`
  resolve to intended code only.

## Test strategy

- **T7 Go service unit tests** (fake RL client): a task whose `context.areal_proxy`
  carries `session_id`+`api_key` reaching `completed` → `SetReward(proxy_key,
  default_reward)` THEN `EndSession(proxy_key)` (order asserted, session-key
  auth); also fires on `failed` / `cancelled`. Task without `areal_proxy` → no
  RL calls. RL errors logged, not fatal.
- **T8 tests**: pi `models.json` `areal` provider reads `$AREAL_PROXY_BASE_URL`;
  guard test (training requested but config missing → loud error).
- Scope Go build/test to touched packages (pre-existing webpush `./...` failure;
  16 pre-existing ON CONFLICT handler failures — both reproduced at base, not
  regressions).

## Implementation notes

- **Codegen rule**: do NOT run `sqlc generate` repo-wide (creates colliding
  `agent_skill_suggestion.sql.go` / `evolution.sql.go`, breaks build). For any
  new query: add to `queries/*.sql` AND hand-write the generated Go in
  `generated/<file>.sql.go` mirroring a sibling. T7-T9 are not expected to add
  queries.
- **Commit cadence**: one commit per task to multica `main`; record commit hash
  in `.superpowers/sdd/progress.md` ledger.

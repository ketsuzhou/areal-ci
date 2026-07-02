# env-dispatch feature params (default env, squad_id, resume) — design

**Date:** 2026-07-02
**Status:** Draft
**Scope:** multica server (`internal/handler/env_dispatch.go`, `internal/service/env_dispatch.go`, a migration, and the chat-task/daemon squad path), AReaL caller (`customized_areal/tree_search/agents/swe_lego_client.py` + runners).

## 1. Motivation

The unified env-dispatch API (spec 2026-07-01) covers scratch/branch × swe_lego/self_play with a single `agent_id` and a required `env_id`. Three capabilities are still missing for the AReaL DAG-RL workflow:

1. **Default self-play env** — self_play rollouts should not have to know a base `env_id`; multica should supply a per-workspace default.
2. **Team dispatch (`squad_id`)** — a rollout should be able to start an agent *team* (a squad, led by its leader), not only a single agent.
3. **`resume` verb** — the AReaL runner talks about "resume from a saved checkpoint," which is behaviorally a branch; the API should accept that vocabulary.

All three are additive request params on `POST /api/v1/env-dispatch`. Existing behavior is unchanged when the new params are absent.

## 2. Locked decisions (from brainstorming)

| # | Decision |
|---|---|
| D1 | `mode=resume` is an **alias** for `mode=branch`. Normalized to `branch` at the edge; one reset/dispatch code path. The original verb is retained only for logging/metrics. |
| D2 | `env_id` becomes **optional**, but empty is valid **only** for `mode=scratch` + `domain=self_play`, which resolves the workspace default base env. Empty `env_id` with swe_lego, or with branch/resume, is a 400. |
| D3 | The default self-play env is a **per-workspace configured base env**, stored in a new `workspace.default_self_play_env_id` column. It is **set out-of-band**; B only reads/resolves it. Not configured → 400. |
| D4 | Exactly **one** of `agent_id` / `squad_id` is required (neither or both → 400). When `squad_id` is set, `agent_id` is forbidden and the squad's leader is resolved server-side. |
| D5 | `squad_id` is supported for **both** domains (issue/swe_lego and message/self_play). |
| D6 | Squad dispatch reuses multica's existing **leader signals** so the leader runs as a team leader (issue: `assignee_type=squad` + `is_leader_task=true`; chat: a `squad_id` hint on the chat task that makes the daemon inject the squad-leader briefing). A plain leader-agent enqueue (no leader signal) is explicitly rejected as a design option because it would not produce a real team run. |

## 3. Non-goals

- Setting the workspace default env through the API (out-of-band per D3; a setter endpoint is a trivial follow-up).
- `resume` as a distinct reset behavior (it is exactly `branch`).
- `squad_id` on scratch **and** the multi-issue/multi-session branch source disambiguation — unchanged from the 2026-07-01 spec (branch source still requires exactly one issue/session).
- The entropy/critic env-save (sub-project E), BranchMaterializer removal (C), and the training_agent session lifecycle (D).
- Changing the `rollouts[]` response shape.

## 4. Request schema (deltas to §6.3 of the 2026-07-01 spec)

```jsonc
{
  "mode": "scratch" | "branch" | "resume",   // resume normalized -> branch
  "env_id": "uuid",                           // NOW OPTIONAL (see §5.2)
  "domain": "swe_lego" | "self_play",
  "dispatch_type": "issue" | "message",
  "group_size": 3,
  "agent_id": "uuid",                         // NOW OPTIONAL; exactly one of agent_id/squad_id
  "squad_id": "uuid",                         // NEW; team dispatch
  "idempotency_key": "uuid",
  "issue":   { /* unchanged */ },
  "message": { /* unchanged */ }
}
```

`rollouts[]` response shape is unchanged. For env_id-omitted self_play, `rollouts[].env_id` is the freshly forked env from the workspace default base. For squad dispatch, `rollouts[].agent_run_id` is the **leader's** run.

## 5. Behavior

### 5.1 `resume` normalization

The handler maps `mode` through the service unchanged; the service normalizes `resume → branch` as its first step (recording the original verb in a log field). Every downstream check (`EnvModeBranch`), reset, and dispatch path is the existing branch path. No new `EnvMode` constant beyond accepting the `resume` string at the boundary.

### 5.2 Optional `env_id` + default self-play env

- `env_id` is no longer UUID-validated in the handler when empty; a non-empty value is still parsed (400 on malformed).
- The service resolves `env_id` after `domain`/`mode` are known:
  - `env_id` present → existing behavior (scratch requires base env; branch requires state env).
  - `env_id` empty **and** `mode=scratch` **and** `domain=self_play` → look up the workspace default via a new dep `GetDefaultSelfPlayEnv(ctx, workspaceID) → (envID, error)`. If unset/empty → `validation_failed` (400: "default self-play env not configured"). The resolved env is treated as the scratch base (forked per rollout; new env_id returned).
  - `env_id` empty in any other combination (swe_lego, or branch/resume) → `validation_failed` (400).

### 5.3 Exactly-one of `agent_id` / `squad_id`

`validate()` requires exactly one to be non-empty:
- neither → 400 ("agent_id or squad_id is required")
- both → 400 ("agent_id and squad_id are mutually exclusive")
- `squad_id` present but not a UUID → 400; squad not found in workspace → 404.

### 5.4 Squad dispatch (Approach 1 — reuse leader signals)

When `squad_id` is set, the service passes it through the enqueue seam and the adapter resolves `squad.leader_id`, then:

- **issue / swe_lego:** the created issue is written with `assignee_type='squad'` / `assignee_id=squad_id`, and the leader is enqueued with `is_leader_task=true` (the same signal `EnqueueTaskForSquadLeader` sets). The daemon then layers the squad-leader briefing, so the leader delegates to members.
- **message / self_play:** the chat session is bound to the leader agent, and the chat task carries a `squad_id` hint (mirroring `QuickCreateContext.SquadID`). The daemon claim handler injects the squad-leader briefing for chat tasks — **new wiring** (see §7).

The `EnqueueAgentRun` dep is extended to accept `squadID` (empty = single-agent, unchanged path).

### 5.5 Validation table (delta)

| Condition | Status |
|---|---|
| `mode=resume` | accepted (→ branch) |
| `env_id` empty + scratch + self_play, default configured | proceed (fork default) |
| `env_id` empty + scratch + self_play, default **not** configured | 400 |
| `env_id` empty + swe_lego (any mode) | 400 |
| `env_id` empty + branch/resume (any domain) | 400 |
| neither `agent_id` nor `squad_id` | 400 |
| both `agent_id` and `squad_id` | 400 |
| `squad_id` not a UUID | 400 |
| `squad_id` not in workspace | 404 |

## 6. Data model

Migration `141_workspace_default_self_play_env` (next sequential after `140_environment_state`):

```sql
ALTER TABLE workspace
  ADD COLUMN default_self_play_env_id UUID NULL REFERENCES environment(id) ON DELETE SET NULL;
```

- Nullable; `ON DELETE SET NULL` so deleting the referenced base env clears the default rather than blocking.
- Set out-of-band (D3). Down migration drops the column.
- No change to the `environment` / `project` tables.

## 7. Notable new wiring / risk

self_play + squad is the only path without precedent. Today:
- chat tasks (`CreateChatTask`) carry no squad/leader hint, and
- the daemon injects the squad-leader briefing only for issue-bound tasks (`is_leader_task`) and quick-create tasks (`QuickCreateContext.SquadID`).

B extends this to the chat path: carry `squad_id` on the chat task (via the task context JSONB, reusing the `SquadID`-hint pattern rather than adding leader columns), and extend the daemon claim handler to inject the squad-leader briefing when a chat task carries the hint. The issue-path squad dispatch reuses proven mechanisms and is low risk. The plan will make the chat-squad wiring its highest-effort task and cover it with a dedicated test.

## 8. Layers touched

- `server/internal/handler/env_dispatch.go` — relax `env_id`/`agent_id` validation (allow empty), parse `squad_id`, accept `mode=resume`, thread new fields into `EnvDispatchInput`.
- `server/internal/service/env_dispatch.go` — normalize `resume→branch`; extend `validate()` (exactly-one agent/squad; conditional env_id); default-env resolution; extend `EnqueueAgentRun` dep with `squadID`; new dep `GetDefaultSelfPlayEnv`.
- `server/internal/handler` adapter — implement `GetDefaultSelfPlayEnv` (read `workspace.default_self_play_env_id`), resolve `squad.leader_id`, set issue assignee to the squad + `is_leader_task=true` (issue), stamp `squad_id` hint on the chat task (chat).
- daemon claim handler — inject the squad-leader briefing for chat tasks carrying the `squad_id` hint.
- `server/migrations/141_workspace_default_self_play_env.up.sql` / `.down.sql`.
- AReaL: `customized_areal/tree_search/agents/swe_lego_client.py` (`create_env_dispatch` gains optional `squad_id`; `env_id`/`agent_id` optional; `mode="resume"` accepted). Runners pass the new params where relevant.

## 9. Testing strategy

- **Service tests** (`env_dispatch_test.go` fake deps): `resume` normalizes to branch; default-env resolution (configured → fork; unconfigured → 400); exactly-one agent/squad (neither/both → 400); squad+issue sets `assignee_type=squad` and enqueues leader with `is_leader_task=true`; squad+chat stamps the `squad_id` hint and enqueues the leader; `env_id` empty rejected for swe_lego and for branch/resume; `env_id` empty accepted only for scratch+self_play.
- **Handler tests**: status mapping for the new 400/404 cases; `squad_id` UUID parse; `mode=resume` accepted.
- **Daemon test**: a chat task carrying the `squad_id` hint gets the squad-leader briefing injected.
- **Migration test**: `workspace.default_self_play_env_id` column + FK + `ON DELETE SET NULL`; round-trip down.
- **AReaL client tests**: payload includes `squad_id` when set; omits `env_id`/`agent_id` when unused; `mode="resume"` passes through.

## 10. Out of scope / follow-ups

- API setter for `workspace.default_self_play_env_id` (out-of-band for now).
- Squad on the branch path with multi-issue/multi-session sources (unchanged single-target v1 constraint).
- Sub-projects C (BranchMaterializer removal), D (training_agent lifecycle), E (entropy/critic env-save).

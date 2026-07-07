# Sub-project F — per-agent env customization + event-triggered env checkpointing (design)

Status: DRAFT (2026-07-06) Date: 2026-07-06 Repos: multica `main` (primary); AReaL
`master` (client + entropy helper)

## 1. Goal

Enable branching from ANY trainable decision point on the env-dispatch path (not just
terminal turns) by eagerly checkpointing (env_id + per-issue subtree DB snapshot + full
sandbox snapshot) at structural DAG events and entropy-gated tool calls. Additionally,
allow per-agent / per-group sandbox customization so squad members can run in isolated
sandboxes from different base envs. F dramatically increases trajectory diversity for RL
training and enables debugging/reproducibility via full state capture.

## 2. Actors / flow

```
env_dispatch (per-agent envs: {coder: python-env, tester: node-env, ...})
    │
    ▼
[multica] dispatch squad: each member gets its own sandbox from its base env
    │   shared multica entity subtree (issues / tasks / messages)
    ▼
trained member task created → D's open hook → start_session(env_id)
    │
    ▼
trained agent works → LLM traffic via AReaL proxy (E captures logprobs)
    │
    │  ┌── always-checkpoint events ──────────────────────────────┐
    │  │  delegation / mention / completion / failure / squad      │
    │  │  leader briefing  →  checkpoint(env_id, DB subtree,       │
    │  │                       sandbox snapshot, event_ref)        │
    │  └──────────────────────────────────────────────────────────┘
    │
    │  ┌── entropy-gated tool calls ──────────────────────────────┐
    │  │  each tool-call decision: compute entropy from E's        │
    │  │  logprobs; if entropy > threshold → checkpoint            │
    │  │  (includes file-change tool calls)                        │
    │  └──────────────────────────────────────────────────────────┘
    │
    ▼
AReaL selects a checkpoint as branch candidate
    │
    ▼  F's branch-from-checkpoint
[multica] restore sandbox snapshot + DB subtree → new env_dispatch
          (mode=branch-from-checkpoint, checkpoint_id=...)
    │
    ▼
branched rollout begins (new RL session via D's open hook)
```

## 3. Decisions (locked)

### F1 — Two pillars, one change

Per-agent env customization and event-triggered checkpointing are distinct features but
ship together as F. They share the env-dispatch contract surface and the multica
handler/service layer. A rollout may use either or both.

### F2 — Checkpoint trigger taxonomy

Three tiers, mapped to whether the triggering event reflects an LLM policy decision
(trainable) or a code rule (not trainable):

| Tier          | Events                                                                                                       | Why                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Always        | delegation, mention, completion (LLM path), failure (LLM path), squad leader briefing                        | Structural DAG-growth + terminal-decision points. Rare, high-impact. Alternative policy choices here are the most consequential. |
| Entropy-gated | all tool calls (incl. file changes)                                                                          | Frequent, low-stakes individually. Entropy from E's logprobs gates: high entropy = LLM was uncertain = good branch point.        |
| Never         | claiming, cascade cancellations, sweeper timeouts, autopilot retry, sandbox lifecycle, RL session management | Code-driven, no policy signal. Checkpointing wastes storage and complicates the trajectory graph.                                |

### F3 — Per-agent env customization granularity

Both per-agent (within a squad) AND per-group (across squads). The env-dispatch request
accepts an optional `per_agent_envs: {<agent_id>: <base_env_spec>}` map. When omitted,
today's behavior (all lanes fork from one base env) is preserved.

### F4 — Checkpoint content

Each checkpoint captures:

- `env_id` — current env per agent (a map if per-agent envs are in use)
- `db_snapshot` — per-issue subtree (the issue + sub-issues + their tasks + messages +
  comments) at checkpoint time. Scoped to one rollout's entity tree.
- `sandbox_snapshot` — full filesystem snapshot per env_id (eager, accept cost)
- `event_ref` — activity_log row id that triggered the checkpoint
- `timestamp`
- `entropy_score` — only for entropy-gated checkpoints (null for always)

### F5 — Checkpoint storage

New `env_checkpoint` table in multica (migration NNN). Columns: id, workspace_id,
project_id (env_dispatch project), issue_id, event_ref (activity_log id), env_id_map
(JSONB — agent_id → env_id), db_snapshot_ref (JSONB or structured),
sandbox_snapshot_refs (JSONB — agent_id → snapshot_id), entropy_score (nullable),
checkpoint_kind (always | entropy_gated), created_at.

Sandbox snapshots are stored via the existing Fleet sandbox snapshot mechanism (C's
lazy-snapshot path used the same provider; F calls it eagerly).

### F6 — Branch-from-checkpoint

New env-dispatch mode: `mode=branch-from-checkpoint`. Takes a `checkpoint_id`. multica:

1. Reads the checkpoint row.
1. Restores each sandbox snapshot → new sandbox per agent.
1. Restores the DB subtree (issue + sub-issues + tasks + messages + comments) by copying
   the snapshot into new rows under a new project_id.
1. Creates a new env_dispatch project with the restored envs and DB state.
1. Returns the new rollout handle (same shape as `mode=branch`).

AReaL client gains `branch_from_checkpoint(checkpoint_id)` → rollout handle.

### F7 — Entropy computation

AReaL-side helper computes entropy per tool-call decision from E's captured logprobs.
The entropy is over the tool-choice tokens (the tokens that determine WHICH tool is
called and its key arguments). Implementation:

- Aggregate logprob entropy over the tool-call decision span.
- Compare against configurable threshold `ENV_CHECKPOINT_ENTROPY_THRESHOLD` (default:
  TBD — calibrate from sample rollouts).
- If logprobs unavailable (E's graceful degradation), skip entropy-gated checkpoints for
  that interaction (always-checkpoint events still fire).

### F8 — Failure modes

- Checkpoint creation fails (DB error, snapshot error) → log, continue. The rollout is
  not blocked; the checkpoint is simply missing from the branch candidate set.
  (Best-effort, like D's RL error handling.)
- Branch-from-checkpoint fails (snapshot corrupt, DB state missing) → return error to
  AReaL; AReaL falls back to C's `mode=branch` from a live env_id, or skips the branch.
- Entropy computation fails (logprobs malformed) → skip entropy-gated checkpoints for
  that interaction; log.
- Per-agent env spec is invalid (unknown agent_id, missing base env) → env-dispatch
  request fails loudly (400), like B's validation rules.

## 4. Layers / changes

### 4.1 Per-agent env contract (multica handler + service)

- `EnvDispatchRequest` + `service.EnvDispatchInput` gain optional
  `per_agent_envs: map[string]string` (agent_id → base_env_id). Handler validates: each
  agent_id must resolve to a real agent in the workspace; each base_env_id must resolve
  to a real env; cannot mix per_agent_envs with a top-level env_id for the same agent.
  Empty ⇒ today's behavior.
- Service: when per_agent_envs is set, dispatch each agent to its own sandbox from its
  specified base env. The squad shares one multica entity subtree.

### 4.2 Persist per-agent env intent (multica, DB)

- Migration NNN: `ALTER TABLE training_dispatch ADD COLUMN per_agent_envs JSONB NULL`
  (or a new `env_dispatch_per_agent_envs` table if a join is cleaner). Hand-write
  generated Go (mirror sibling) — do NOT run `sqlc generate` repo-wide (D's codegen
  rule).
- Queries: extend `CreateTrainingDispatch` / `GetTrainingDispatchByProject` to handle
  `per_agent_envs`.

### 4.3 Checkpoint trigger hooks (multica service)

New `maybeCheckpoint(ctx, task, eventKind)` invoked from:

- `EnqueueTaskForIssue` / `EnqueueTaskForMention` / `EnqueueTaskForSquadLeader`
  (delegation + mention — always)
- `CompleteTask` (completion — always, but only when called via the daemon LLM path, not
  sweeper/autopilot)
- `FailTask` (failure — always, but only when called via the daemon LLM path)
- Squad leader briefing generation site (always — T1/1d confirms the exact seam)
- Each tool-call boundary (entropy-gated — see §4.4)

`maybeCheckpoint` checks: is this a trained rollout (training_dispatch exists for the
project)? If not, skip. If yes, create the checkpoint (§4.5).

### 4.4 Entropy-gated tool-call checkpoints (multica + AReaL)

The tool-call boundary is in the agent runtime (pi / daemon), not multica. Two options
(T1/1e decides):

- **Option A**: AReaL-side entropy computation. The AReaL proxy (which already captures
  logprobs per E) computes entropy per tool-call decision and calls a new multica
  endpoint `/api/v1/env-checkpoint` to create the checkpoint when entropy > threshold.
- **Option B**: multica-side entropy computation. The daemon reports tool-call
  boundaries to multica; multica fetches logprobs from AReaL and computes entropy. More
  coupling.

**Decision: Option A** — AReaL owns logprobs (per E1 ownership split), so AReaL owns
entropy computation. multica owns checkpoint storage and branch-from-checkpoint.

### 4.5 Checkpoint storage (multica, DB + sandbox)

New `env_checkpoint` table (migration NNN). New `internal/service/env_checkpoint.go`
with `CreateCheckpoint`, `GetCheckpoint`, `ListCheckpointsForProject`,
`BranchFromCheckpoint`.

Sandbox snapshots: multica calls the existing Fleet snapshot endpoint (used by C's
lazy-snapshot path) eagerly per checkpoint. The snapshot id is stored in
`env_checkpoint.sandbox_snapshot_refs`.

DB subtree snapshot: multica serializes the issue subtree (issue + sub-issues

- tasks + messages + comments) at checkpoint time. Stored as JSONB in
  `env_checkpoint.db_snapshot_ref` (or a side table if large).

### 4.6 Branch-from-checkpoint (multica service + handler)

New `POST /api/v1/env-dispatch` with `mode=branch-from-checkpoint, checkpoint_id=<id>`.
Handler validates the checkpoint belongs to the workspace. Service:

1. Reads the checkpoint row.
1. Forks each sandbox snapshot → new sandbox per agent (Fleet fork endpoint).
1. Restores the DB subtree under a new project_id (copy rows).
1. Returns the new rollout handle.

### 4.7 AReaL client (Python)

`customized_areal/tree_search/agents/swe_lego_client.py`:

- `create_env_dispatch` accepts `per_agent_envs` (optional dict).
- New `branch_from_checkpoint(checkpoint_id)` method.

New `customized_areal/tree_search/agents/env_checkpoint.py` (or extension of existing):

- `compute_tool_call_entropy(logprobs)` → float
- `maybe_create_checkpoint(...)` — calls multica `/api/v1/env-checkpoint` when entropy >
  threshold (T1/1e confirms the exact hook point in the proxy).

### 4.8 Config (multica)

- `ENV_CHECKPOINT_ENABLED` (default true) — master toggle.
- `ENV_CHECKPOINT_ENTROPY_THRESHOLD` (default TBD — calibrate) — only entropy-gated
  checkpoints above this threshold are created.
- `ENV_CHECKPOINT_ALWAYS_EVENTS` (default: delegation,mention,completion,
  failure,squad_briefing) — configurable list of always-checkpoint events.
- No new config for per-agent envs (the spec comes from env_dispatch).

## 5. Out of scope (⇒ future hardening or separate sub-project)

- D6 env-dispatch rewire of the wired loop (stays deferred — F is env-dispatch only).
- tree_search wired loop integration (the wired loop keeps its own branching).
- Cost optimization: COW / incremental snapshots, lazy sandbox snapshot at branch time,
  per-rollout configurability. F ships the simple eager path.
- Per-interaction reward (F is about branching, not reward signal richness).
- Critic agent training; multi-critic ensembles; reward shaping.
- Checkpoint garbage collection (old checkpoints → reaped by future hardening).
- Cross-workspace checkpoint sharing.

## 6. Test strategy

### multica Go unit tests (TDD)

- `env_dispatch` per-agent envs validation (shape, agent resolution, base env
  resolution, mixing rule, empty ⇒ today's behavior).
- `training_dispatch` persists `per_agent_envs` (migration NNN).
- `maybeCheckpoint`:
  - always events (delegation / mention / completion LLM path / failure LLM path / squad
    briefing) → checkpoint created with correct kind.
  - non-trained rollout → no checkpoint.
  - sweeper/autopilot FailTask → no checkpoint (not LLM path).
  - checkpoint creation failure → logged, rollout continues.
- `BranchFromCheckpoint`:
  - valid checkpoint → new sandbox per agent + DB subtree restored + new project_id.
  - corrupt snapshot → error returned.
  - cross-workspace → 403.
- `env_checkpoint` table: migration up/down, indexes.

### AReaL Python tests (TDD)

- `compute_tool_call_entropy(logprobs)` returns expected float for sample logprobs.
- `maybe_create_checkpoint` calls multica only when entropy > threshold.
- `branch_from_checkpoint(checkpoint_id)` returns rollout handle.
- `create_env_dispatch` passes `per_agent_envs` when set; omits when empty.
- Logprobs unavailable → entropy-gated checkpoints skipped (graceful).

### db_bridge smoke

- `/api/v1/env-dispatch` carries `per_agent_envs` in JSON body.
- New `/api/v1/env-checkpoint` channel: create / list / branch-from.

### Cross-repo E2E (if feasible)

- A trained rollout with per-agent envs produces checkpoints at structural events; AReaL
  can branch from a checkpoint and the branched rollout inherits the sandbox + DB state.

### Test runners / constraints (from D/E, still apply)

- multica Go: scope build/test to touched packages (pre-existing webpush `./...`
  failure; 16 pre-existing ON CONFLICT handler failures).
- Codegen rule: do NOT run `sqlc generate` repo-wide. Hand-write generated Go.
- AReaL Python: `uv run pytest` on touched tests; `pre-commit run --files` before
  commit.
- Commit each task to multica `main` (multica-side) or areal `master` (areal-side);
  record commit hashes in `.superpowers/sdd/progress.md`.

## 7. Dependencies

- **Sub-project E** (critic-driven training signal) — F's entropy-gated checkpoints
  consume E's logprobs. E is at build phase, not yet implemented. F's always-checkpoint
  events and per-agent env customization are independent of E and could ship first if
  needed.
- **Sub-project D** (session lifecycle) — F's checkpoints coexist with D's RL sessions.
  D is paused at plan-ready. F layers on top of D's session lifecycle; no contract
  change to D.
- **Sub-project C** (branch via env-dispatch) — F's branch-from-checkpoint extends C's
  `mode=branch`. C is complete on multica main.

## 8. Risks

- **Checkpoint storage cost**: full eager snapshots per checkpoint is expensive
  (especially with entropy-gated tool calls adding more). Mitigation: F ships the simple
  eager path; COW / lazy sandbox / per-rollout config are deferred. If cost becomes a
  real bottleneck, a follow-up hardening change can add COW without changing F's
  contract.
- **Entropy threshold calibration**: the default `ENV_CHECKPOINT_ENTROPY_THRESHOLD` is
  TBD. Too low → snapshot explosion; too high → miss meaningful branch points.
  Mitigation: calibrate from sample rollouts in T1; make configurable; document the
  calibration method.
- **DB subtree snapshot size**: large issue subtrees (many sub-issues / messages /
  comments) produce large JSONB snapshots. Mitigation: scope to the rollout's entity
  tree (not the full workspace); future hardening could add compression or incremental
  snapshots.
- **Branch-from-checkpoint fidelity**: sandbox snapshot may be slightly ahead of the
  event that triggered the checkpoint (the same skew C's lazy-snapshot design notes).
  Mitigation: the DB subtree snapshot is the source of truth for entity state; sandbox
  skew affects only filesystem side effects.
- **Per-agent env complexity**: heterogeneous base envs within a squad may break
  assumptions (e.g., a tester agent expecting a Python env but assigned a Node env).
  Mitigation: validation at env-dispatch time; the workspace owner is responsible for
  sensible per-agent env specs.

# Unified env-dispatch API — design

**Date:** 2026-07-01
**Status:** Draft
**Scope:** multica server (`server/`), areal caller (`customized_areal/tree_search/agents/`)

## 1. Motivation

AReaL's DAG RL training needs two operations against multica:

1. **Branch** — fork a running training state (project + sandbox) into N continuations, each on its own isolated environment.
2. **Scratch** — start a new training state from a pre-built docker image, then dispatch an issue or a self-play query into it.

Today multica exposes `POST /api/v1/swe-lego/issues` (scratch, SWE-Lego only, builds the docker image inline) and `POST /api/issues/{id}/fork` (DB-only issue subtree fork, no sandbox). Neither unifies branch+scratch, neither handles self-play chat dispatch, and the build-inline coupling makes the SWE-Lego path unusable for general RL workflows that build images out-of-band.

This spec introduces one unified endpoint, `POST /api/v1/env-dispatch`, plus a base-env creation endpoint `POST /api/v1/env`, that cover all four combinations (scratch/branch × swe_lego/self_play) with N-way parallel rollouts and clean sandbox ownership on the multica side.

## 2. Goals

- One endpoint that handles branch and scratch, swe_lego and self_play, issue and message dispatch, with `group_size=N` parallel rollouts.
- Areal never sees `sandbox_id` directly — multica owns the sandbox lifecycle and exposes a stable `env_id` handle.
- Each rollout in a `group_size=N` call gets its own fully isolated project (full subtree copy in branch mode).
- Reset phase is atomic per rollout (rollback on failure); dispatch phase is best-effort (keep environment, report per-rollout errors).
- Docker image build stays out of env-dispatch — the caller obtains a base `env_id` separately.

## 3. Non-goals

- Designing the docker image build pipeline. The image is built out-of-band; env-dispatch only boots/forks sandboxes from a caller-supplied `env_id`.
- Self-play + `dispatch_type=issue` (501 for now).
- Branch + `domain=swe_lego` + `dispatch_type=message` (400, swe_lego is issue-only).
- A concurrent-reset optimization beyond a fixed semaphore cap. Rollouts run concurrently up to the cap (default 8); within a single rollout, reset and dispatch steps are sequential. Dynamic cap tuning, per-step parallelism within a rollout, and backpressure are out of scope for v1.
- Backwards compatibility for the old `POST /api/v1/swe-lego/issues` URL or response shape. AReaL is updated in lockstep.

## 4. Architecture

### 4.1 env_id model

`env_id` is the multica-side handle for a sandbox state. Areal stores `env_id` against each DAG state and sends it back to multica to branch or resume. Multica owns the `sandbox_id` ↔ `env_id` mapping; areal never touches `sandbox_id`.

- **Base env** (`mode='base'`): created via `POST /api/v1/env` from a pre-built docker image. No project. The root of a training tree.
- **Scratch env** (`mode='scratch'`): created by env-dispatch scratch, forked from a base env. Has a new empty project.
- **Branch env** (`mode='branch'`): created by env-dispatch branch, always forked from a state env (including `group_size=1`). Has a copied project.

Relationship: `project.env_id` (FK → `environment.id`, **1:1 for state envs** — every scratch/branch env is referenced by exactly one project, enforced by a partial unique index; base envs have no project). A branch always forks, so an env_id is never shared across projects. This 1:1 invariant is what lets branch resolve a source `env_id` to a single source project without any extra request field.

### 4.2 Combination matrix

| mode | domain | dispatch_type | group_size | sandbox | project |
|---|---|---|---|---|---|
| scratch | swe_lego | issue | N | fork base × N | ×N: new empty project + new swe_lego issue |
| scratch | self_play | message | N | fork base × N | ×N: new empty project + new chat session + new message |
| branch | swe_lego | issue | N | fork source × N | ×N: copy source project (issue copied, run against it, no new issue) |
| branch | self_play | message | N | fork source × N | ×N: copy source project + append a new message to the copied chat session |

`domain` is **required** (a domain fixes the dispatch semantics), and the two domains each pin a single `dispatch_type`: `swe_lego` ⇒ `issue`, `self_play` ⇒ `message`. `dispatch_type` is still an explicit field (it guards the request and leaves room for the future `self_play`+`issue` path), but every currently-valid request is one of the four rows above.

**Rejected combinations:**
- `domain` omitted → 400 (a dispatch has no meaning without a domain; there is no domain-less mode)
- `scratch` + `swe_lego` + `message` → 400 (swe_lego is issue-only)
- `branch` + `swe_lego` + `message` → 400
- `scratch` + `self_play` + `issue` → 501 (not implemented yet)
- `branch` + `self_play` + `issue` → 501

### 4.3 Sandbox rule

- `scratch` → always fork the base env's sandbox (even for N=1). The base is shared/immutable.
- `branch` → always fork `source_env_id`'s sandbox N times (including N=1). Each copied project references its own new env_id (`mode='branch'`, `parent_env_id=<source>`).

The source env's sandbox is never mutated in place — a branch always forks. This keeps the source state re-branchable, which is required for tree search (MCTS), where the same node is expanded multiple times. (An earlier draft reused the source sandbox for N=1; that was dropped because it both (a) made `env_id → source project` ambiguous and (b) let a child run corrupt the source node's sandbox.)

### 4.4 Self-play shape (N fully isolated rollouts)

Self-play creates N independent projects (not N chat sessions in one project). Each rollout has its own project, sandbox fork, and agent run. This is RL-clean: N independent samples of the same query, no cross-contamination.

- Scratch + self_play: N new empty projects, N new chat sessions (one per project, bound to it), N user messages (the caller-supplied query text), N forked sandboxes (from one base env), N agent runs.
- Branch + self_play: N project copies of the source project (each copy carries the source's chat session + prior messages), and for each copy the new user query is **appended to the copied chat session** (no new session) so the branch continues the existing conversation. N forked sandboxes (from the source env, including N=1), N agent runs. v1 requires the source project to contain exactly one chat session (see §7.4); zero or multiple → 400.

### 4.5 SWE-Lego shape (1 problem, N rollouts)

SWE-Lego creates N projects (one per rollout, same as self_play), each with its own copy of the swe_lego issue and its own sandbox fork. The image is built out-of-band; the base env is obtained via `POST /api/v1/env`. For branch+swe_lego, the copied issue is the dispatch target (no new issue created).

### 4.6 Layers

- `server/internal/handler/env.go` — `CreateEnv`, `DeleteEnv` handlers.
- `server/internal/handler/env_dispatch.go` — `EnvDispatch`, `DeleteEnvDispatchProject` handlers. Validates the discriminated body, calls `EnvDispatchService`, maps errors.
- `server/internal/service/env_dispatch.go` — new `EnvDispatchService` with `Deps` seam. Reset → dispatch, concurrent per rollout.
- `server/internal/service/swe_lego_issue.go` — `SweLegoIssueService` and its handler `CreateSweLegoIssue` are removed. SWE-Lego-specific issue fields (fail_to_pass, pass_to_pass, acceptance_criteria) fold into the env-dispatch `issue` sub-object. Build utilities (`swe_lego_image.go`, `swe_lego_build_node.go`) stay in the repo for the out-of-band build pipeline, not used by env-dispatch.
- Migration `127_environment_state.up.sql` — adds `environment` table + `project.env_id` column.

## 5. Data model

### 5.1 Migration `127_environment_state.up.sql`

```sql
-- environment: a sandbox state handle. Base envs (mode='base') have no
-- project; scratch/branch envs are forked from a parent env and associated
-- with projects via project.env_id (1:1 — a branch always forks, so a state
-- env is never shared across projects).
--
-- sandbox_ids holds ONE OR MORE sandbox handles: an environment can host many
-- agents, and each agent runs in its own sandbox. Base envs are booted with a
-- single sandbox; branching an env forks every sandbox in the set so each
-- agent's state is preserved independently.
CREATE TABLE environment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    sandbox_ids TEXT[] NOT NULL DEFAULT '{}',
    parent_env_id UUID REFERENCES environment(id) ON DELETE SET NULL,
    mode TEXT NOT NULL CHECK (mode IN ('base', 'scratch', 'branch')),
    domain TEXT CHECK (domain IN ('swe_lego', 'self_play')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_environment_workspace ON environment(workspace_id);
CREATE INDEX idx_environment_parent ON environment(parent_env_id) WHERE parent_env_id IS NOT NULL;

-- A project references an env (its sandbox state). A state env (scratch/branch)
-- is referenced by exactly one project — a branch always forks, so env_id is
-- never shared. The partial UNIQUE index enforces this 1:1 invariant and lets
-- the service resolve a source env_id to its single project (GetProjectByEnvID).
-- Base envs have no project. ON DELETE RESTRICT: an env cannot be deleted while
-- a project still references it — the caller must delete the project first via
-- DELETE /api/v1/env-dispatch/{projectID}. This prevents silently orphaning a
-- live project (which would leave its agent runs pointing at a dead sandbox).
ALTER TABLE project ADD COLUMN env_id UUID REFERENCES environment(id) ON DELETE RESTRICT;
CREATE UNIQUE INDEX idx_project_env_unique ON project(env_id) WHERE env_id IS NOT NULL;

-- env_dispatch_request: idempotency ledger (spec §7.7). Stores the response
-- for a given (workspace_id, idempotency_key) so a retried POST /env-dispatch
-- replays the original rollouts[] instead of forking/creating again. The
-- response row is written in the reset transaction of the first rollout.
CREATE TABLE env_dispatch_request (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    idempotency_key UUID NOT NULL,
    response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, idempotency_key)
);
```

Down migration drops `env_dispatch_request`, then the `project.env_id` column + unique index, then the `environment` table.

### 5.2 Provenance

- `environment.parent_env_id` — points to the env this one was forked from. NULL for base envs.
- `project.forked_from_project_id` (NOT added in this spec — branch copy creates new project IDs with no back-pointer at the project level; the env_id chain captures the branching structure, which is what areal's DAG needs).
- `issue.forked_from_issue_id` (migration 126) — NOT touched during branch copy. That field tracks RL transcript-branch points (per-issue fork at a task_message.seq), a different concept from env-dispatch branch (full project subtree copy).

## 6. Endpoints

### 6.1 `POST /api/v1/env` — create base env

```json
// request
{ "image_ref": "swe-lego:abc123" }

// 201 response
{ "env_id": "uuid", "sandbox_id": "uuid" }
```

Multica boots a sandbox from `image_ref` via the existing cloud-runtime proxy (`cloud_runtime.go`), creates an `environment` row (`mode='base'`, `parent_env_id=NULL`, `domain=NULL`, `sandbox_id=<booted>`), returns `env_id`.

**Validation:**
- `image_ref` non-empty string, max 256 chars → 400 otherwise.
- Auth: inside `RequireWorkspaceMember`. `workspace_id` from context.

### 6.2 `DELETE /api/v1/env/{envID}` — delete env

```json
// 204 No Content
```

Deletes the `environment` row + its sandbox (via cloud-runtime `DELETE /api/v1/sandboxes/{id}` proxy, idempotent on 404). If a project still references this env_id, the delete is **refused with 409 Conflict** (`ON DELETE RESTRICT`) — the caller must delete the project first via `DELETE /api/v1/env-dispatch/{projectID}`. This makes orphaning impossible: an env is only deletable once no project depends on its sandbox.

**Validation:** `envID` path param validated as UUID via `parseUUIDOrBadRequest`. 404 if not found or not in caller's workspace. 409 if a project still references it.

### 6.3 `POST /api/v1/env-dispatch` — unified dispatch

#### Request

```json
{
  "mode": "scratch",                 // required: "scratch" | "branch"
  "env_id": "uuid",                  // required: base env (scratch) or state env (branch)
  "domain": "swe_lego",              // required: "swe_lego" | "self_play"
  "dispatch_type": "issue",          // required: "issue" | "message" (fixed by domain: swe_lego⇒issue, self_play⇒message)
  "group_size": 3,                   // optional, default 1, range [1, 64]; at most ENV_DISPATCH_CONCURRENCY (default 8) run at once
  "agent_id": "uuid",                // required: the agent to run (agents are first-class rows; there is no separate agent_config)
  "idempotency_key": "uuid",         // optional; dedupes retries — see §7.6

  // dispatch_type=issue (required except in branch+swe_lego, where the copied issue is reused)
  "issue": {
    "title": "fix auth bug",         // required when issue dispatch creates a new issue
    "description": "...",            // optional
    "acceptance_criteria": ["..."],  // optional, default []
    "fail_to_pass": ["test_x"],      // optional, default [] (swe_lego convention)
    "pass_to_pass": ["test_y"]       // optional, default []
  },

  // dispatch_type=message
  "message": {
    "content": "the query text"      // required, non-empty
  }
}
```

`creator_id` is **not** a request field — the creator of the issue/chat session is the authenticated user, taken from context (`requireUserID`), consistent with every other create path.

#### Response — one shape, always `rollouts[]`

```json
{
  "rollouts": [
    {
      "env_id": "uuid",              // always a new env_id (branch always forks, incl. N=1)
      "project_id": "uuid",
      "issue_id": "uuid",            // present iff dispatch_type=issue
      "chat_session_id": "uuid",     // present iff dispatch_type=message
      "agent_run_id": "uuid",        // UUID if dispatch succeeded, null if this rollout's dispatch failed
      "error": "..."                 // omitted/null on success; set to the failure detail on a failed rollout
    }
  ]
}
```

Array length = `group_size`; index `i` corresponds to rollout `i`. There is a **single response shape** for success, partial success, and total dispatch failure — the caller always parses `rollouts[]` and checks each element's `agent_run_id`/`error`.

**Status rule (per-rollout dispatch is best-effort):**
- **`201 Created`** — the reset phase succeeded for all rollouts and **at least one** rollout dispatched. Failed rollouts (if any) appear in the same `rollouts[]` array with `agent_run_id: null` and an `error`. The caller retries or cleans up those individually.
- **`500 Internal Server Error`** — reset succeeded but **every** rollout's dispatch failed. Body is the same `rollouts[]` shape (all elements carry `error`), so the caller can still see the created envs/projects to clean up.
- **`503 Service Unavailable`** — the reset phase itself failed (see §8); no rollouts are returned (all rolled back).
- Validation errors are `400`/`501` before any work starts (§6.3 table).

**env_id semantics per mode:**
- `scratch`: `env_id` = base env. Fork its sandbox N times → N new env_ids (`mode='scratch'`, `parent_env_id=<base>`). Each new project references its new env_id.
- `branch`: `env_id` = source state env. Fork its sandbox N times (including N=1) → N new env_ids (`mode='branch'`, `parent_env_id=<source>`). Each copied project references its new env_id. `rollouts[].env_id` are always freshly created; the source `env_id` is never returned and its project stays intact and re-branchable.

#### Validation rules

| Rule | Status |
|---|---|
| `mode` missing or not in {scratch, branch} | 400 |
| `env_id` missing or not a valid UUID | 400 |
| `env_id` does not exist or not in caller's workspace | 404 |
| `domain` missing, or not in {swe_lego, self_play} | 400 |
| `dispatch_type` missing or not in {issue, message} | 400 |
| `group_size` < 1 or > 64 | 400 |
| `domain=swe_lego` and `dispatch_type=message` | 400 |
| `domain=self_play` and `dispatch_type=issue` | 501 |
| `domain=swe_lego` and `mode=branch` and `issue` present | 400 (issue comes from the copy; caller must not supply one) |
| `domain=swe_lego` and `mode=scratch` and `issue` missing | 400 |
| `dispatch_type=message` and `message.content` empty | 400 |
| `agent_id` missing, not a UUID, or not in workspace | 400 / 404 |
| `idempotency_key` present but not a valid UUID | 400 |
| `mode=scratch` and `env_id` is not a base env (`mode != 'base'`) | 400 |
| `mode=branch` and `env_id` is a base env (`mode == 'base'`) | 400 |

### 6.4 `DELETE /api/v1/env-dispatch/{projectID}` — delete project (renamed)

Renamed from `DELETE /api/v1/swe-lego/issues/{projectID}`. Same cascade: delete project → cascade to issues/chat_sessions/tasks/messages; the `environment` row persists (deleting the project leaves its env orphaned — no other project references it, per the 1:1 invariant). To reclaim the sandbox, areal calls `DELETE /api/v1/env/{envID}` separately.

Path param `projectID` validated as UUID via `parseUUIDOrBadRequest`. 404 if not found or not in caller's workspace.

## 7. Service layer

### 7.1 Deps seam

```go
// server/internal/service/env_dispatch.go

type EnvDispatchDeps interface {
    // Environment operations
    GetEnv(ctx context.Context, envID, workspaceID string) (Env, error)
    CreateEnv(ctx context.Context, workspaceID string, sandboxIDs []string, parentEnvID string, mode, domain string) (envID string, err error)
    DeleteEnv(ctx context.Context, envID string) error

    // Sandbox operations (proxy to cloud-runtime/Fleet).
    // idx is the rollout index; it is used to name/tag the forked sandbox
    // (e.g. "<source>-r<idx>") so concurrent forks are distinguishable in
    // Fleet and in logs. It carries no semantics beyond naming/observability.
    ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (sandboxID string, err error)
    DeleteSandbox(ctx context.Context, sandboxID string) error
    BootSandbox(ctx context.Context, imageRef string) (sandboxID string, err error)  // for POST /api/v1/env

    // Project operations
    GetProjectByEnvID(ctx context.Context, envID, workspaceID string) (projectID string, err error)  // branch: resolve source env → its single project (1:1 invariant)
    CreateProject(ctx context.Context, workspaceID, name, envID string) (projectID string, err error)
    // CopyProjectSubtree deep-copies the source project (issues + chat sessions
    // + messages) under a new project bound to envID. It returns maps from
    // source → copied IDs so the dispatch phase can target the copied issue
    // (branch+swe_lego) or the copied chat session (branch+self_play).
    CopyProjectSubtree(ctx context.Context, sourceProjectID, workspaceID, envID string) (newProjectID string, issueIDMap, chatSessionIDMap map[string]string, err error)
    DeleteProject(ctx context.Context, projectID string) error

    // Issue operations. creatorID is the authenticated user (from context).
    CreateIssue(ctx context.Context, projectID, workspaceID, creatorID, title, description string, acceptanceCriteria, failToPass, passToPass []string) (issueID string, err error)

    // Chat operations. creatorID is the authenticated user; agentID is the
    // first-class agent row to bind the session to (there is no agent_config).
    CreateChatSession(ctx context.Context, projectID, workspaceID, agentID, creatorID string) (sessionID string, err error)
    CreateChatMessage(ctx context.Context, sessionID, role, content string) (messageID string, err error)

    // Agent run
    EnqueueAgentRun(ctx context.Context, workspaceID, agentID, issueID, chatSessionID, sandboxID string, idx int) (runID string, err error)
}
```

The production adapter wires each method to the existing query/runtime layer (`db.Queries`, `cloudruntime.Client`). The test adapter is a fake. Mirrors the `SweLegoDeps` pattern (`swe_lego_issue.go:23-33`), minus `BuildImage`/`BootBaseSandbox`.

### 7.2 Reset phase (concurrent, bounded)

For each of N rollouts in parallel (semaphore cap 8, configurable via `ENV_DISPATCH_CONCURRENCY`):

0. **Resolve source (branch only)**: `source_project_id = GetProjectByEnvID(source_env.id, workspace_id)`. The 1:1 invariant guarantees exactly one project (validation already rejected base envs, which have none).

1. **Sandbox**: an env can hold several sandboxes (`sandbox_ids`, one per agent),
   so a fork forks *every* sandbox in the source set.
   - `scratch` → `ForkSandbox(sid, i)` for each sid in `base_env.sandbox_ids` → new sandbox set. Create new env row (`mode='scratch'`, `parent_env_id=base_env.id`, `domain=<domain>`, `sandbox_ids=<forked set>`).
   - `branch` → `ForkSandbox(sid, i)` for each sid in `source_env.sandbox_ids` (including N=1) → new sandbox set. Create new env row (`mode='branch'`, `parent_env_id=source_env.id`, `domain=<domain>`, `sandbox_ids=<forked set>`). On a partial fork failure the already-forked sandboxes are best-effort deleted before the rollout errors.

2. **Project**:
   - `scratch` → `CreateProject(workspace_id, name, new_env_id)`. `name` is server-generated, not caller-supplied: `env-dispatch-<new_env_id>` (a UUID-derived, guaranteed-unique name). The project title is not a meaningful field for RL rollouts, so no request field is exposed for it.
   - `branch` → `CopyProjectSubtree(source_project_id, workspace_id, new_env_id)` returns `(new_project_id, issue_id_map, chat_session_id_map)`.

Collect `{env_id, sandbox_id, project_id, issue_id_map, chat_session_id_map}` per rollout. If any rollout's reset fails, that rollout is rolled back (delete its project if created; delete its forked sandbox if forked; delete its env row if created). Other rollouts unaffected — partial reset is reported, not fatal.

### 7.3 Dispatch phase (concurrent, bounded, best-effort)

For each of N rollouts in parallel (same semaphore):

- **issue** (`domain=swe_lego`):
  - `scratch` → `CreateIssue(project_id, workspace_id, creator_id, ...)` → `EnqueueAgentRun(workspace_id, agent_id, issue_id, "", sandbox_id, i)`.
  - `branch` → look up the copied issue via `issue_id_map` (v1: exactly one, see §7.4) → `EnqueueAgentRun(workspace_id, agent_id, copied_issue_id, "", sandbox_id, i)`. No new issue.
- **message** (`domain=self_play`):
  - `scratch` → `CreateChatSession(project_id, workspace_id, agent_id, creator_id)` → `CreateChatMessage(session_id, "user", content)` → `EnqueueAgentRun(workspace_id, agent_id, "", session_id, sandbox_id, i)`.
  - `branch` → look up the copied chat session via `chat_session_id_map` (v1: exactly one, see §7.4) → **append** `CreateChatMessage(copied_session_id, "user", content)` (no new session, so the branch continues the copied conversation) → `EnqueueAgentRun(workspace_id, agent_id, "", copied_session_id, sandbox_id, i)`.

Per-rollout failure: no rollback. The rollout is returned in `rollouts[i]` with `agent_run_id: null` and `error` set (§6.3 response). Successful rollouts in the same call are unaffected.

### 7.4 Branch source lookup (issue / chat session)

`CopyProjectSubtree` returns `source_id → new_id` maps for both copied issues and copied chat sessions. The dispatch target in branch mode is the single copied entity of the relevant kind:

- **branch + swe_lego**: v1 requires the source project to contain exactly one swe_lego issue; the service 400s at validation time on zero or multiple. Dispatch runs against the single copied issue from `issue_id_map`.
- **branch + self_play**: v1 requires the source project to contain exactly one chat session; the service 400s on zero or multiple. Dispatch appends the new message to the single copied session from `chat_session_id_map`.

Disambiguating multi-issue / multi-session source projects (e.g. a `source_issue_id` / `source_session_id` field) is out of scope for v1.

### 7.5 Rollback rules

| Phase | Failure | Rollback |
|---|---|---|
| Reset — env fork / project create / project copy | fails | delete project (if created) + delete forked sandbox (if forked) + delete env row (if created) |
| Reset — one of N rollouts fails | other rollouts unaffected | only the failing rollout rolls back; response reports partial reset |
| Dispatch — issue/chat/run | fails | no rollback; keep project + sandbox + env; report the rollout in `rollouts[]` with `error` + `agent_run_id: null` |
| Dispatch — partial (some rollouts succeed, some fail) | n/a | all rollouts returned in one `rollouts[]`; failed ones carry `error` + `agent_run_id: null`, succeeded ones carry `agent_run_id` |

The "keep environment on dispatch failure" rule (decided in brainstorming) is honored: the caller gets back `env_id`/`project_id`/`sandbox_id` for failed rollouts and can retry dispatch or clean up via `DELETE /api/v1/env-dispatch/{projectID}`.

### 7.6 Consistency & transaction boundaries

Each rollout mixes DB writes (env row, project, issue/session/message, agent run) with external side effects (sandbox fork/delete via cloud-runtime). The two cannot share one transaction, so the boundary is:

- **DB writes within a step are transactional; the sandbox call sits outside.** Per rollout the order is: (1) `ForkSandbox` → obtain `sandbox_id`; (2) in a single DB transaction, insert the `environment` row (with that `sandbox_id`) + the project (+ copied subtree for branch). If the DB tx fails, roll it back and then best-effort `DeleteSandbox(sandbox_id)`. If `DeleteSandbox` also fails, the forked sandbox is leaked but no DB row references it — a background sweeper (the existing runtime GC, `internal/daemon/gc.go`) reclaims sandboxes with no owning env row.
- **Rollback is best-effort and idempotent.** Each rollback op (`DeleteProject`, `DeleteSandbox`, `DeleteEnv`) tolerates "already gone" (404 → success). A rollback-of-rollback failure never masks the original error; it is logged and the sweeper is the backstop. This is why the reset failure status is `503` (retryable) rather than `500`.
- **No cross-rollout transaction.** Rollouts are independent by design (§4.4/§4.5), so one rollout's failure never rolls back another's committed rows.

### 7.7 Idempotency & retries

`POST /api/v1/env-dispatch` accepts an optional `idempotency_key` (UUID). Since the caller (§7.5) is explicitly invited to retry after a dispatch failure, without dedup a naive retry would fork sandboxes and create projects a second time.

- The service persists `idempotency_key` (unique per workspace) alongside the created rollout set. A repeat request with the same key returns the **original** `rollouts[]` response (including any per-rollout `error`s) without doing any new work.
- Scope: the key dedupes the whole call (all N rollouts), not individual rollouts. To retry only the *failed* rollouts of a partial success, the caller issues a fresh dispatch (new key) for the remaining count, or re-drives the specific failed projects — full per-rollout retry is out of scope for v1.
- If `idempotency_key` is omitted, every request is treated as new (fork + create); the caller owns cleanup of any duplicates. Callers doing retries SHOULD send a key.
- Implementation: a `UNIQUE (workspace_id, idempotency_key)` row written in the same reset transaction as the first rollout; a duplicate insert (unique violation) signals "replay the stored response."

## 8. Error response shape

There is **no separate `partial_rollouts` schema**. Success, partial success, and total dispatch failure all use the single `rollouts[]` shape from §6.3 (each element carrying `agent_run_id` or `error`). Only phases that produce *no* rollouts use the flat error body:

```json
{ "error": "reset_failed", "message": "human-readable detail" }
```

| Situation | Status | Body |
|---|---|---|
| Validation failure | 400 / 501 (§6.3) | `{"error":"validation_failed","message":"..."}` |
| Reset phase failed (sandbox fork / env create / project copy) | 503 | `{"error":"reset_failed","message":"..."}` — all rollouts rolled back, no IDs returned |
| Reset OK, **all** rollouts' dispatch failed | 500 | `rollouts[]` (§6.3), every element with `error` + `agent_run_id: null` — envs/projects kept for cleanup |
| Reset OK, **≥1** rollout dispatched | 201 | `rollouts[]` (§6.3); failed rollouts (if any) carry `error` + `agent_run_id: null` |
| `DELETE /api/v1/env` while a project references the env | 409 | `{"error":"env_in_use","message":"..."}` |

The caller's parse path is uniform: on `201`/`500` read `rollouts[]` and inspect each element; on `503`/`400`/`501`/`409` read `{error, message}`.

## 9. AReaL caller changes

### 9.1 `MulticaSweLegoClient` → `MulticaEnvDispatchClient`

- New `create_base_env(image_ref) -> env_id` → `POST /api/v1/env`. Called once per image; the returned `env_id` is the root of a training tree.
- `create_swe_lego_issue()` → `create_env_dispatch()` calling `POST /api/v1/env-dispatch` with:
  ```json
  {
    "mode": "scratch",
    "env_id": "<base env_id>",
    "domain": "swe_lego",
    "dispatch_type": "issue",
    "group_size": N,
    "agent_id": "...",
    "issue": {
      "title": "...", "description": "...",
      "acceptance_criteria": [...], "fail_to_pass": [...], "pass_to_pass": [...]
    }
  }
  ```
- `cleanup_swe_lego_issue(project_id)` → calls `DELETE /api/v1/env-dispatch/{projectID}` (URL change). For `group_size=N`, loops over `rollouts[].project_id`.
- New `delete_env(env_id)` → `DELETE /api/v1/env/{envID}` (for retiring base envs or reclaimed state envs).

### 9.2 `SweLegoSetup` (`swe_lego_types.py`)

Old shape: `{project_id, issue_id, image_id, base_sandbox_id, base_sandbox_runtime_id, agent_run_ids[]}`.

New shape: `{rollouts: [{env_id, project_id, issue_id, agent_run_id}, ...]}`.

For `group_size=N`, N entries instead of 1 project + N agent_run_ids.

### 9.3 `swe_lego_issue_runner.py`

Iterates `rollouts` instead of zipping `agent_run_ids` with a single project/issue. Stores `env_id` per rollout so it can branch from any rollout later (sends that `env_id` with `mode=branch`).

### 9.4 New `self_play_runner.py`

Mirrors `swe_lego_issue_runner.py`'s structure. Reads a query from `query_bank` (via `query_bank_client.py`), calls `POST /api/v1/env-dispatch` with:
```json
{
  "mode": "scratch" | "branch",
  "env_id": "<base env_id or state env_id>",
  "domain": "self_play",
  "dispatch_type": "message",
  "group_size": N,
  "agent_id": "...",
  "idempotency_key": "<uuid, set on retries>",
  "message": {"content": "<query from query_bank>"}
}
```

For `mode=branch`, the message is appended to each copy's existing chat session (the branch continues the source conversation; see §4.4). Tracks N `agent_run_id`s in parallel (same pattern as `swe_lego_issue_runner`). After all runs complete, collects transcripts and computes reward against `query_bank_row["answer"]`.

### 9.5 Areal DAG state tracking

Each DAG state stores `env_id`. Branching from state X sends `env_id(X)` to env-dispatch with `mode=branch`; because every state env is 1:1 with its project, `env_id` alone unambiguously identifies the source project to copy — areal does not need to track or send `project_id` as a branch key. The base env_id (from `POST /api/v1/env`) is the root. Areal never sees `sandbox_id` directly.

## 10. Route registration (`server/cmd/server/router.go`)

- Add `r.Post("/api/v1/env", h.CreateEnv)`.
- Add `r.Delete("/api/v1/env/{envID}", h.DeleteEnv)`.
- Add `r.Post("/api/v1/env-dispatch", h.EnvDispatch)`.
- Rename `r.Delete("/api/v1/swe-lego/issues/{projectID}", h.DeleteSweLegoIssue)` → `r.Delete("/api/v1/env-dispatch/{projectID}", h.DeleteEnvDispatchProject)`.
- Remove the `/api/v1/swe-lego` route group entirely. `CreateSweLegoIssue` handler and `SweLegoIssueService` are deleted; their swe_lego-specific logic folds into `EnvDispatchService`.

All new routes sit inside the existing `RequireWorkspaceMember` group (same scope as the current SWE-Lego route at `router.go:957`).

## 11. Testing strategy

- **Service unit tests** (`server/internal/service/env_dispatch_test.go`): fake `EnvDispatchDeps`, exercise every combination in the matrix (§4.2) + every rejected combination. Verify rollback on reset failure, per-rollout dispatch-failure reporting in the unified `rollouts[]`, the status rule (201 with ≥1 dispatched / 500 when all fail / 503 on reset failure), `GetProjectByEnvID` source resolution for branch, branch+self_play **appends** to the single copied session (no new session), branch always forks a fresh env (including N=1) leaving the source env/project intact, and idempotency replay (same `idempotency_key` returns the stored `rollouts[]` without new forks).
- **Handler tests** (`server/internal/handler/env_dispatch_test.go`): validation rules (§6.3, incl. `domain` required and `agent_id`), status code mapping, auth/workspace scoping, unified response shape, and `409` on `DELETE /api/v1/env` while a project references the env.
- **Areal client tests**: update `tests/test_swe_lego_issue_runner.py` and `tests/test_integration_multica.py` to the new request/response shape (`agent_id`, `rollouts[]` with optional `error`). New `tests/test_self_play_runner.py` for the self-play caller.
- **Migration test**: apply `127_environment_state.up.sql` on a fresh DB, verify `environment` table, `project.env_id` column with the partial UNIQUE index and `ON DELETE RESTRICT`, and the `env_dispatch_request` table with its `(workspace_id, idempotency_key)` unique constraint. Round-trip down migration.

## 12. Open dependencies

- **Out-of-band docker image build:** env-dispatch does not build images. A separate pipeline (not in this spec) must produce a docker image and let areal call `POST /api/v1/env` with its `image_ref` to get a base `env_id`. The existing build utilities (`server/internal/service/swe_lego_image.go`, `swe_lego_build_node.go`) stay in the repo for whoever implements that pipeline.
- **Branch+swe_lego multi-issue disambiguation:** if the source project has multiple swe_lego issues, v1 400s. A future spec may add a `source_issue_id` field to disambiguate.
- **Self-play + issue dispatch (501):** not implemented. A future spec may add it if self-play ever needs issue-based rollouts.
- **Concurrent reset tuning:** semaphore cap 8 is a placeholder. Tune after measuring real `group_size` distributions and sandbox fork latency.

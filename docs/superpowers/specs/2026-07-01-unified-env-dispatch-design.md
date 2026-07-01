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
- **Branch env** (`mode='branch'`): created by env-dispatch branch, forked from a state env (or reused directly when `group_size=1`). Has a copied project.

Relationship: `project.env_id` (FK → `environment.id`, many projects → one env_id). Multiple projects share an env_id when the sandbox is reused (branch + N=1), not forked.

### 4.2 Combination matrix

| mode | domain | dispatch_type | group_size | sandbox | project |
|---|---|---|---|---|---|
| scratch | swe_lego | issue | N | fork base × N | ×N: new empty project + new swe_lego issue |
| scratch | self_play | message | N | fork base × N | ×N: new empty project + new chat session + new message |
| scratch | (none) | issue \| message | 1 only | fork base × 1 | 1 new empty project + 1 issue or 1 chat session (400 if N>1) |
| branch | swe_lego | issue | N | fork source × N (or reuse for N=1) | ×N: copy source project (issue copied, run against it, no new issue) |
| branch | self_play | message | N | fork source × N (or reuse for N=1) | ×N: copy source project + new chat session + new message |
| branch | (none) | issue \| message | 1 only | reuse source directly | 1 copied project + 1 issue or 1 chat session (400 if N>1) |

**Rejected combinations:**
- `scratch` + `swe_lego` + `message` → 400 (swe_lego is issue-only)
- `branch` + `swe_lego` + `message` → 400
- `scratch` + `self_play` + `issue` → 501 (not implemented yet)
- `branch` + `self_play` + `issue` → 501
- `scratch` + (none) + N>1 → 400 (no shared base to fork without a domain)
- `branch` + (none) + N>1 → 400 (use a domain for N rollouts)

### 4.3 Sandbox rule

- `scratch` → always fork the base env's sandbox (even for N=1). The base is shared/immutable.
- `branch` + N=1 → reuse `source_env_id`'s sandbox directly (no fork call). The new copied project references the same env_id.
- `branch` + N>1 → fork `source_env_id`'s sandbox N times. Each copied project references its own new env_id.

### 4.4 Self-play shape (N fully isolated rollouts)

Self-play creates N independent projects (not N chat sessions in one project). Each rollout has its own project, chat session, message, sandbox fork, and agent run. This is RL-clean: N independent samples of the same query, no cross-contamination.

- Scratch + self_play: N new empty projects, N new chat sessions (one per project, bound to it), N user messages (same query text from `query_bank`), N forked sandboxes (from one base env), N agent runs.
- Branch + self_play: N project copies of the source project, N new chat sessions (one per copied project), N user messages, N forked sandboxes (from the source env, or reused for N=1), N agent runs.

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
-- with projects via project.env_id (many projects → one env_id when the
-- sandbox is reused, not forked).
CREATE TABLE environment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    sandbox_id TEXT NOT NULL,
    parent_env_id UUID REFERENCES environment(id) ON DELETE SET NULL,
    mode TEXT NOT NULL CHECK (mode IN ('base', 'scratch', 'branch')),
    domain TEXT CHECK (domain IN ('swe_lego', 'self_play')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_environment_workspace ON environment(workspace_id);
CREATE INDEX idx_environment_parent ON environment(parent_env_id) WHERE parent_env_id IS NOT NULL;

-- A project references an env (its sandbox state). Many projects can share
-- one env_id (branch+N=1 reuses the source sandbox). ON DELETE SET NULL:
-- deleting an env orphans projects rather than cascading — caller's job to
-- clean up projects first via DELETE /api/v1/env-dispatch/{projectID}.
ALTER TABLE project ADD COLUMN env_id UUID REFERENCES environment(id) ON DELETE SET NULL;
CREATE INDEX idx_project_env ON project(env_id) WHERE env_id IS NOT NULL;
```

Down migration drops the `project.env_id` column + index, then drops the `environment` table.

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

Deletes the `environment` row + its sandbox (via cloud-runtime `DELETE /api/v1/sandboxes/{id}` proxy, idempotent on 404). Projects referencing this env_id have `env_id` set to NULL (`ON DELETE SET NULL`). Caller's responsibility to delete projects first via `DELETE /api/v1/env-dispatch/{projectID}` if clean cleanup is wanted; calling env-delete on an env with live projects orphans them (sandbox gone, agent runs will fail).

**Validation:** `envID` path param validated as UUID via `parseUUIDOrBadRequest`. 404 if not found or not in caller's workspace.

### 6.3 `POST /api/v1/env-dispatch` — unified dispatch

#### Request

```json
{
  "mode": "scratch",                 // required: "scratch" | "branch"
  "env_id": "uuid",                  // required: base env (scratch) or state env (branch)
  "domain": "swe_lego",              // optional: "swe_lego" | "self_play" (omitted = none)
  "dispatch_type": "issue",          // required: "issue" | "message"
  "group_size": 3,                   // optional, default 1, range [1, 64]
  "agent_config_id": "uuid",         // required

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

#### Response — `201 Created`

```json
{
  "rollouts": [
    {
      "env_id": "uuid",              // new env_id (fork) or source env_id (branch+N=1 reuse)
      "project_id": "uuid",
      "issue_id": "uuid",            // present iff dispatch_type=issue
      "chat_session_id": "uuid",     // present iff dispatch_type=message
      "agent_run_id": "uuid"
    }
  ]
}
```

Array length = `group_size`. For `group_size=1`, array has one element.

**env_id semantics per mode:**
- `scratch`: `env_id` = base env. Fork its sandbox N times → N new env_ids (`mode='scratch'`, `parent_env_id=<base>`). Each new project references its new env_id.
- `branch` + N=1: `env_id` = source state env. Reuse sandbox directly (no fork). Copy source project → new project references the same env_id. `rollouts[0].env_id` = source `env_id`.
- `branch` + N>1: `env_id` = source state env. Fork its sandbox N times → N new env_ids (`mode='branch'`, `parent_env_id=<source>`). Each copied project references its new env_id.

#### Validation rules

| Rule | Status |
|---|---|
| `mode` missing or not in {scratch, branch} | 400 |
| `env_id` missing or not a valid UUID | 400 |
| `env_id` does not exist or not in caller's workspace | 404 |
| `domain` present and not in {swe_lego, self_play} | 400 |
| `dispatch_type` missing or not in {issue, message} | 400 |
| `group_size` < 1 or > 64 | 400 |
| `domain=swe_lego` and `dispatch_type=message` | 400 |
| `domain=self_play` and `dispatch_type=issue` | 501 |
| `domain=swe_lego` and `mode=branch` and `issue` present | 400 (issue comes from the copy; caller must not supply one) |
| `domain=swe_lego` and `mode=scratch` and `issue` missing | 400 |
| `dispatch_type=issue` (non-branch-swe_lego) and `issue` missing | 400 |
| `dispatch_type=message` and `message.content` empty | 400 |
| `domain` omitted and `group_size` > 1 | 400 |
| `agent_config_id` missing, not a UUID, or not in workspace | 400 / 404 |
| `mode=scratch` and `env_id` is not a base env (`mode != 'base'`) | 400 |
| `mode=branch` and `env_id` is a base env (`mode == 'base'`) | 400 |

### 6.4 `DELETE /api/v1/env-dispatch/{projectID}` — delete project (renamed)

Renamed from `DELETE /api/v1/swe-lego/issues/{projectID}`. Same cascade: delete project → cascade to issues/chat_sessions/tasks/messages; the `environment` row persists (other projects may reference it, or areal may reuse it). If areal wants to reclaim the sandbox, it calls `DELETE /api/v1/env/{envID}` separately.

Path param `projectID` validated as UUID via `parseUUIDOrBadRequest`. 404 if not found or not in caller's workspace.

## 7. Service layer

### 7.1 Deps seam

```go
// server/internal/service/env_dispatch.go

type EnvDispatchDeps interface {
    // Environment operations
    GetEnv(ctx context.Context, envID, workspaceID string) (Env, error)
    CreateEnv(ctx context.Context, workspaceID, sandboxID, parentEnvID string, mode, domain string) (envID string, err error)
    DeleteEnv(ctx context.Context, envID string) error

    // Sandbox operations (proxy to cloud-runtime/Fleet)
    ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (sandboxID string, err error)
    DeleteSandbox(ctx context.Context, sandboxID string) error
    BootSandbox(ctx context.Context, imageRef string) (sandboxID string, err error)  // for POST /api/v1/env

    // Project operations
    CreateProject(ctx context.Context, workspaceID, name, envID string) (projectID string, err error)
    CopyProjectSubtree(ctx context.Context, sourceProjectID, workspaceID, envID string) (newProjectID string, issueIDMap map[string]string, err error)
    DeleteProject(ctx context.Context, projectID string) error

    // Issue operations
    CreateIssue(ctx context.Context, projectID, workspaceID, creatorID, title, description string, acceptanceCriteria, failToPass, passToPass []string) (issueID string, err error)

    // Chat operations
    CreateChatSession(ctx context.Context, projectID, workspaceID, agentID, creatorID string) (sessionID string, err error)
    CreateChatMessage(ctx context.Context, sessionID, role, content string) (messageID string, err error)

    // Agent run
    EnqueueAgentRun(ctx context.Context, workspaceID, agentID, issueID, chatSessionID, sandboxID string, idx int) (runID string, err error)
}
```

The production adapter wires each method to the existing query/runtime layer (`db.Queries`, `cloudruntime.Client`). The test adapter is a fake. Mirrors the `SweLegoDeps` pattern (`swe_lego_issue.go:23-33`), minus `BuildImage`/`BootBaseSandbox`.

### 7.2 Reset phase (concurrent, bounded)

For each of N rollouts in parallel (semaphore cap 8, configurable via `ENV_DISPATCH_CONCURRENCY`):

1. **Sandbox**:
   - `scratch` → `ForkSandbox(base_env.sandbox_id, i)` → new sandbox. Create new env row (`mode='scratch'`, `parent_env_id=base_env.id`, `domain=<domain>`).
   - `branch` + N=1 → reuse `source_env.sandbox_id` directly. New project references `source_env.id`.
   - `branch` + N>1 → `ForkSandbox(source_env.sandbox_id, i)` → new sandbox. Create new env row (`mode='branch'`, `parent_env_id=source_env.id`, `domain=<domain>`).

2. **Project**:
   - `scratch` → `CreateProject(workspace_id, name, new_env_id)`.
   - `branch` → `CopyProjectSubtree(source_project_id, workspace_id, new_env_id_or_source_env_id)` returns `(new_project_id, issue_id_map)`.

Collect `{env_id, sandbox_id, project_id, issue_id_map}` per rollout. If any rollout's reset fails, that rollout is rolled back (delete its project if created; delete its forked sandbox if forked; delete its env row if created). Other rollouts unaffected — partial reset is reported, not fatal.

### 7.3 Dispatch phase (concurrent, bounded, best-effort)

For each of N rollouts in parallel (same semaphore):

- `dispatch_type=issue` (scratch+swe_lego, scratch+none, branch+none): `CreateIssue(project_id, ...)` → `EnqueueAgentRun(workspace_id, agent_id, issue_id, "", sandbox_id, i)`.
- `dispatch_type=issue` (branch+swe_lego): look up copied issue via `issue_id_map[source_swe_lego_issue_id]` → `EnqueueAgentRun(..., copied_issue_id, "", sandbox_id, i)`. No new issue.
- `dispatch_type=message`: `CreateChatSession(project_id, ...)` → `CreateChatMessage(session_id, "user", content)` → `EnqueueAgentRun(workspace_id, agent_id, "", chat_session_id, sandbox_id, i)`.

Per-rollout failure: no rollback. `partial_rollouts[i].agent_run_id = null`, `partial_rollouts[i].error = "..."`.

### 7.4 Branch + swe_lego issue lookup

`CopyProjectSubtree` returns a map of `source_issue_id → new_issue_id` for all copied issues. v1 requires the source project to contain exactly one swe_lego issue — the service 400s at validation time if there are zero or multiple issues in the source project. After copy, dispatch looks up the single copied issue via the map. Disambiguating multi-issue source projects (e.g. a `source_issue_id` field) is out of scope for v1.

### 7.5 Rollback rules

| Phase | Failure | Rollback |
|---|---|---|
| Reset — env fork / project create / project copy | fails | delete project (if created) + delete forked sandbox (if forked) + delete env row (if created) |
| Reset — one of N rollouts fails | other rollouts unaffected | only the failing rollout rolls back; response reports partial reset |
| Dispatch — issue/chat/run | fails | no rollback; keep project + sandbox + env; report in `partial_rollouts` |
| Dispatch — partial (some rollouts succeed, some fail) | n/a | successful rollouts returned normally; failed ones returned with `error` field and `agent_run_id: null` |

The "keep environment on dispatch failure" rule (decided in brainstorming) is honored: the caller gets back `env_id`/`project_id`/`sandbox_id` for failed rollouts and can retry dispatch or clean up via `DELETE /api/v1/env-dispatch/{projectID}`.

## 8. Error response shape

```json
{
  "error": "dispatch_failed",
  "message": "human-readable detail",
  "partial_rollouts": [
    {
      "env_id": "uuid",
      "project_id": "uuid",
      "issue_id": "uuid",
      "chat_session_id": "uuid",
      "agent_run_id": "uuid",
      "error": "..."
    }
  ]
}
```

- Reset failure (project copy / sandbox fork / env create): 503, no IDs returned (rolled back). Body: `{"error": "reset_failed", "message": "..."}`.
- Dispatch failure (issue/chat creation or agent run enqueue): 500, with `partial_rollouts` populated. Environment kept; caller decides cleanup or retry.
- Validation failure: 400/501 as tabled in §6.3, body `{"error": "validation_failed", "message": "..."}`.
- `agent_run_id` is `null` in `partial_rollouts` for rollouts where dispatch failed; UUID where it succeeded.

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
    "agent_config_id": "...",
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
  "agent_config_id": "...",
  "message": {"content": "<query from query_bank>"}
}
```

Tracks N `agent_run_id`s in parallel (same pattern as `swe_lego_issue_runner`). After all runs complete, collects transcripts and computes reward against `query_bank_row["answer"]`.

### 9.5 Areal DAG state tracking

Each DAG state stores `env_id`. Branching from state X sends `env_id(X)` to env-dispatch with `mode=branch`. The base env_id (from `POST /api/v1/env`) is the root. Areal never sees `sandbox_id` directly.

## 10. Route registration (`server/cmd/server/router.go`)

- Add `r.Post("/api/v1/env", h.CreateEnv)`.
- Add `r.Delete("/api/v1/env/{envID}", h.DeleteEnv)`.
- Add `r.Post("/api/v1/env-dispatch", h.EnvDispatch)`.
- Rename `r.Delete("/api/v1/swe-lego/issues/{projectID}", h.DeleteSweLegoIssue)` → `r.Delete("/api/v1/env-dispatch/{projectID}", h.DeleteEnvDispatchProject)`.
- Remove the `/api/v1/swe-lego` route group entirely. `CreateSweLegoIssue` handler and `SweLegoIssueService` are deleted; their swe_lego-specific logic folds into `EnvDispatchService`.

All new routes sit inside the existing `RequireWorkspaceMember` group (same scope as the current SWE-Lego route at `router.go:957`).

## 11. Testing strategy

- **Service unit tests** (`server/internal/service/env_dispatch_test.go`): fake `EnvDispatchDeps`, exercise every combination in the matrix (§4.2) + every rejected combination. Verify rollback on reset failure, partial-rollout reporting on dispatch failure, env_id reuse for branch+N=1, env_id fork for all other cases.
- **Handler tests** (`server/internal/handler/env_dispatch_test.go`): validation rules (§6.3), status code mapping, auth/workspace scoping, response shape.
- **Areal client tests**: update `tests/test_swe_lego_issue_runner.py` and `tests/test_integration_multica.py` to the new request/response shape. New `tests/test_self_play_runner.py` for the self-play caller.
- **Migration test**: apply `127_environment_state.up.sql` on a fresh DB, verify `environment` table + `project.env_id` column + indexes. Round-trip down migration.

## 12. Open dependencies

- **Out-of-band docker image build:** env-dispatch does not build images. A separate pipeline (not in this spec) must produce a docker image and let areal call `POST /api/v1/env` with its `image_ref` to get a base `env_id`. The existing build utilities (`server/internal/service/swe_lego_image.go`, `swe_lego_build_node.go`) stay in the repo for whoever implements that pipeline.
- **Branch+swe_lego multi-issue disambiguation:** if the source project has multiple swe_lego issues, v1 400s. A future spec may add a `source_issue_id` field to disambiguate.
- **Self-play + issue dispatch (501):** not implemented. A future spec may add it if self-play ever needs issue-based rollouts.
- **Concurrent reset tuning:** semaphore cap 8 is a placeholder. Tune after measuring real `group_size` distributions and sandbox fork latency.

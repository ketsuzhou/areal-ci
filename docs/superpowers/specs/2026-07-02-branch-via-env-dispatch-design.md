# Branch via env-dispatch — Design Spec (sub-project C)

## Status

Approved (design). Successor to sub-project A (env-dispatch over db_bridge) and B
(env-dispatch feature params). Depends on B's `mode=branch` server behavior.

## 1. Motivation

Branching in tree-search RL is currently implemented by **two** mechanisms:

- **Unwired cloud path** — `BranchMaterializer` + `MulticaIssueForker` +
  `agent_start_branch` (in `agents/integration.py`): client-orchestrated
  snapshot → fork sandbox → fork issue subtree → start-branch, across three
  endpoints. Never wired into the live episode loop (scaffolding).
- **Wired local path** — `build_branch_task` (in
  `core/customized_grouped_workflow.py`): binds a local `branch_sandbox_id`
  directly, truncates the transcript at `turn_idx`, copies the prefix into a new
  TPFC task, then drives it via the areal engine.

Sub-project B added `env_dispatch(mode="branch", …)`, which performs the
server-side equivalent (fork project subtree + fork sandboxes + enqueue agent
run) in a single db_bridge call. C collapses both branch mechanisms into that
one primitive.

## 2. Locked decisions

- **D1** — `env_dispatch(mode="branch", env_id=…)` becomes the single branch
  mechanism. Remove both `build_branch_task` and `BranchMaterializer`.
- **D2** — **Per-node `env_id`** is the branch frontier. Branching =
  `env_dispatch(mode="branch", env_id=node.env_id)`. No `seq`/`turn_idx`
  transcript truncation; env-dispatch is unchanged from B.
- **D3** — Each node's `env_id` comes from **per-turn assistant-message
  metadata** emitted by the backend's snapshot mechanism. The client reads
  `metadata["env_id"]` onto `node.env_id`.
- **D4** — **Full removal** of the now-dead transport: client pieces, the
  db_bridge `agent_start_branch` channel (+ schema + tests), and the multica
  server endpoints `/api/agent/start-branch` and `/api/issues/{id}/fork`.
- **D5** — **Clean break** for checkpoint/serialization: persist `env_id` only;
  drop `branch_sandbox_id` / `branch_issue_id` / `branch_env_snapshot_id`. Old
  checkpoints carrying the legacy keys are not supported.
- **D6** — ~~Rewire the wired `customized_grouped_workflow` branch site to the
  new env-dispatch `BranchDriver`.~~ **REVISED (post-implementation blocker):**
  the wired branch site is structurally coupled to the le-agent TPFC
  driven-generation contract (`task_id` + seeded messages → token-level Nodes),
  which the env-dispatch branch primitive (forked `env_id` + reward-only result)
  cannot supply without an external backend "materialize" endpoint. Therefore
  D6 reverts to the original option: **remove the wired loop's branch machinery
  entirely** — that loop stops branching (falls back to normal/scratch
  episodes); branching lives exclusively in the runner model
  (`run_swe_lego_issue` / `run_self_play` + `EnvDispatchBranchDriver`). Rewiring
  the wired loop via env-dispatch is deferred until an external le-agent
  materialize endpoint exists (out of C's scope).
- **Approach 2** — Implement the existing `_BranchDriver.drive_lane` Protocol
  seam (in `swe_lego_issue_runner.py` / `self_play_runner.py`) with a concrete
  env-dispatch-backed driver, rather than building a throwaway adapter inside
  the wired loop only.

## 3. Non-goals / external dependencies

- **Backend `env_id` emission is EXTERNAL to C.** C implements the client-side
  read (`metadata["env_id"]` → `node.env_id`), the branch dispatch, and the
  removals. The backend stamping a per-turn `env_id` into assistant-message
  metadata (via its existing snapshot mechanism) is assumed present or delivered
  out-of-band — analogous to B's out-of-band default-env configuration. C does
  not implement that emission.
- Not migrating the whole tree-search training entrypoint to the runner model.
- Not changing env-dispatch server branch semantics (B already built
  `mode=branch`).

## 4. Node model (`core/tree_store.py`)

- **Remove** fields: `branch_sandbox_id`, `branch_issue_id`,
  `branch_env_snapshot_id`.
- **Add** field: `env_id: str | None = None` (the state handle at this turn;
  the branch frontier).
- Keep `need_branch`, `task_id`, `turn_idx`.

## 5. Rollout annotation (`customized_grouped_workflow.py::annotate_nodes_from_run`)

Replace the `branch_sandbox_id` metadata read with:

```python
env_id = metadata.get("env_id")
node.env_id = env_id if isinstance(env_id, str) and env_id else None
```

Keep the `need_branch`, `entropy_stats`, and `topk_ids` reads unchanged.

## 6. Branch primitive + `BranchDriver`

- Add a concrete `BranchDriver` (new class, e.g. in
  `agents/branch_driver.py`) implementing the existing Protocol
  `async def drive_lane(*, agent_run_id: str, sandbox_id: str, session_id: str) -> str`.
- At a branch point, the driver calls:

  ```python
  setup = await self._multica.create_env_dispatch(
      mode="branch",
      env_id=node.env_id,          # per-node frontier (D2)
      dispatch_type=<"issue"|"message">,
      agent_id=<lane agent_id>,
      domain=<"swe_lego"|"self_play">,
  )
  ```

  and returns the new terminal `env_id` from the resulting rollout.
- **Wiring (Approach 2 + D6):**
  - `run_swe_lego_issue` / `run_self_play` receive the real `BranchDriver`
    (replacing `FakeBranchDriver` in tests / call sites).
  - `customized_grouped_workflow` branch site: `_prepare_branch_task` is
    replaced by a call through the same branch primitive using
    `candidate.env_id`; the returned `agent_run_id` / `session_id` / `env_id`
    feed `_retry_episode` (replacing `task_id` + `seed_messages_already_inserted`).

## 7. Removals (D4)

**AReaL client:**

- `agents/integration.py`: `BranchMaterializer`, `BranchMaterializationResult`,
  `MulticaIssueForker`, `BranchStarter`, `materialize_cloud_branch`,
  `cleanup_cloud_branch`. (Retain `finalize_with_verifier` / `VerifierResult`
  usage if still referenced.)
- `core/customized_grouped_workflow.py`: `build_branch_task`.
- `tpfc/backend_run.py`: `_start_branch_agent_run_for_task` and
  `_start_branch_agent_run_for_task_with_refresh`.
- `agents/__init__.py`: drop the removed exports.

**db_bridge:**

- `channels.py`: remove the `agent_start_branch` `Channel`.
- `schema.sql`: remove the `rpc_agent_start_branch` table row.
- tests: `test_leagent_channels.py` (START_BRANCH cases), schema tests, and the
  `agent_start_branch` reference in `test_integration_e2e.py`.

**multica server (Go, `multica/server`):**

- Remove handlers + route registrations for `POST /api/issues/{id}/fork` and
  `DELETE /api/issues/{id}/fork` (`internal/handler/issue_fork.go`,
  `internal/service/issue_fork.go`, `cmd/server/router.go` lines ~755-756),
  plus `internal/handler/issue_fork_test.go`,
  `internal/service/issue_fork_test.go`, and the fork rows in
  `cmd/server/router_fork_routes_test.go`.
- **NOTE:** `/api/agent/start-branch` is **not** served by `multica/server`; it
  lives in the external **le-agent** backend (the `leagent_api` executor host,
  not in this repo). Removing that server endpoint is therefore an **external**
  change tracked separately (like the backend `env_id` emission, §3). C removes
  its client (`backend_run.py`) and its db_bridge channel here.

## 8. Checkpoint & serialization (D5 — clean break)

- `core/checkpoint.py`, `agents/event_codec.py`, `agents/execution_dag.py`:
  serialize/deserialize `env_id` only. Remove all `branch_sandbox_id` /
  `branch_issue_id` / `branch_env_snapshot_id` reads and writes. No tolerant
  fallback for legacy checkpoints.

## 9. Layers touched

- **AReaL client:** `core/tree_store.py`, `core/customized_grouped_workflow.py`,
  `core/checkpoint.py`, `agents/integration.py`, `agents/__init__.py`,
  `agents/event_codec.py`, `agents/execution_dag.py`, `agents/branch_driver.py`
  (new), `agents/swe_lego_issue_runner.py`, `agents/self_play_runner.py`,
  `tpfc/backend_run.py`, plus their tests.
- **db_bridge:** `channels.py`, `schema.sql`, tests.
- **multica server (Go):** `internal/handler/issue_fork.go`,
  `internal/service/issue_fork.go`, `cmd/server/router.go`, and the fork tests.
  (`/api/agent/start-branch` server endpoint is external — le-agent backend —
  and out of this repo's scope; see §7.)

## 10. Risks

1. **Backend `env_id` emission (external).** If the backend does not yet stamp
   per-turn `env_id` into metadata, `node.env_id` will be `None` and branching
   will no-op until that emission lands. C treats this as an external dependency
   (§3); the client read is defensive (str-or-None).
2. **`_retry_episode` impedance (D6, highest-risk task).** The wired loop drives
   generation against a `task_id` + seeded messages; env-dispatch enqueues a
   server-side run returning `agent_run_id` / `session_id` / `env_id`. Adapting
   `_retry_episode` to consume env-dispatch results may require a thin adapter.
   The plan isolates this as its own task.

## 11. Testing strategy

- `BranchDriver.drive_lane` calls `create_env_dispatch` with
  `mode="branch"`, `env_id=node.env_id`, correct `domain`/`dispatch_type`;
  returns the new terminal `env_id`.
- `annotate_nodes_from_run` maps `metadata["env_id"]` → `node.env_id`
  (str-or-None), leaves `need_branch` intact.
- Clean-break round-trips: `checkpoint.py`, `event_codec.py`,
  `execution_dag.py` persist/restore `env_id` with no `branch_*` keys.
- Runner wiring: `run_swe_lego_issue` / `run_self_play` drive lanes through the
  real `BranchDriver` (fakes updated).
- Removal regressions: imports resolve after deletions; db_bridge channel +
  schema suites pass with `agent_start_branch` gone; multica route/handler
  removal tests.

## 12. Out of scope

Training-entrypoint migration to the runner model; env-dispatch server branch
semantics; backend per-turn `env_id` emission; removal of the
`/api/agent/start-branch` server endpoint in the external le-agent backend
(C removes only its client + db_bridge channel).

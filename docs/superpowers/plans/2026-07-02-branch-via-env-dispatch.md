# Branch via env-dispatch — Implementation Plan (sub-project C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse both branch mechanisms (`BranchMaterializer` cloud path, `build_branch_task` local path) into a single primitive — `env_dispatch(mode="branch", env_id=node.env_id)` — driven through a concrete `BranchDriver`; and fully remove the now-dead transport.

**Spec:** `docs/superpowers/specs/2026-07-02-branch-via-env-dispatch-design.md` (decisions D1-D6, Approach 2).

**Architecture:** Per-node `env_id` (populated from backend per-turn metadata) is the branch frontier. A new `BranchDriver` calls the existing `MulticaEnvDispatchClient.create_env_dispatch(mode="branch", env_id=…)` and is wired into the env-dispatch runners **and** the wired `customized_grouped_workflow` branch site. Legacy branch refs are removed from `Node` and `SuperNode` (clean break, no back-compat). Dead transport removed across areal client, db_bridge channel, and the multica issue-fork endpoints.

**Tech Stack:** Python 3.12 (areal client + db_bridge, `pytest`); Go 1.x (multica server, `go test`). No `sqlc generate` in this sub-project (issue-fork handler removal only).

## Global Constraints

- Three repos/areas: **areal client** (`customized_areal/**`, commit to `master`), **db_bridge** (`multica/db_bridge/**`, commit to multica `main`), **multica server** (`multica/server/**`, commit to multica `main`).
- **Backend `env_id` emission is EXTERNAL** (spec §3): C only reads `metadata["env_id"]`. Do NOT implement backend emission.
- **`/api/agent/start-branch` server endpoint is EXTERNAL** (le-agent backend, not in this repo): remove only its client + db_bridge channel here.
- **Clean break (D5):** no back-compat for legacy `branch_*` keys in checkpoints/serialization.
- Python: no wildcard imports. **Run areal tests with the working test venv: `/workspaces/leagent/backend/areal/.venv-test/bin/python -m pytest …`** (Python 3.12.13, pytest 9.1.1, torch 2.12.1+cpu — needed for `tree_search` paths that import torch). Do NOT use bare `python -m pytest` (no torch) or `uv run` (the areal-root `.venv` symlink is broken). db_bridge has its own working venv via `uv run`.
- Go: build/test scoped to touched packages; `gofmt` all Go. (The repo-wide `go build ./...` has a pre-existing unrelated `webpush.go:180` failure — do not fix, do not gate on it.)
- Follow existing patterns in each file. Read a file before editing it.

## Task ordering & dependencies

T1 → T2 → T3 → T4 → T5 (areal), then T6 (db_bridge), T7 (multica Go), T8 (regression). T3 provides the `BranchDriver` used by T4. T4 deletes `build_branch_task` (it rewires its only caller); T5 deletes `BranchMaterializer` + transport.

---

### Task 1: Node model + rollout annotation (areal)

**Files:**
- Modify: `customized_areal/tree_search/core/tree_store.py`
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py` (`annotate_nodes_from_run`)
- Test: `customized_areal/tree_search/tests/` (add/adjust a tree_store + annotate test)

**Interfaces:**
- Produces: `Node.env_id: str | None = None`; removes `Node.branch_sandbox_id`, `Node.branch_issue_id`, `Node.branch_env_snapshot_id`.
- `annotate_nodes_from_run` sets `node.env_id` from `metadata["env_id"]`.

- [ ] **Step 1: Write failing tests**

Add a test asserting `Node` has `env_id` and no longer has the branch fields, and that `annotate_nodes_from_run` maps metadata:

```python
def test_annotate_nodes_reads_env_id_from_metadata():
    from customized_areal.tree_search.core.tree_store import Node
    from customized_areal.tree_search.core.customized_grouped_workflow import annotate_nodes_from_run
    node = Node(input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0], turn_idx=1)
    raw = [{"role": "assistant", "metadata": {"env_id": "env-42", "need_branch": True}}]
    annotate_nodes_from_run([node], task_id="t1", raw_messages=raw)
    assert node.env_id == "env-42"
    assert node.need_branch is True
    assert not hasattr(node, "branch_sandbox_id")
```

- [ ] **Step 2: Run to confirm failure** — `cd /workspaces/leagent/backend/areal && /workspaces/leagent/backend/areal/.venv-test/bin/python -m pytest customized_areal/tree_search/tests/ -k env_id -v`

- [ ] **Step 3: Implement**
  - `tree_store.py`: delete the three `branch_*` fields (and the "Cloud-env branch refs" comment block); add `env_id: str | None = None` near `need_branch`.
  - `annotate_nodes_from_run`: replace the `branch_sandbox_id` block with:
    ```python
    env_id = metadata.get("env_id")
    node.env_id = env_id if isinstance(env_id, str) and env_id else None
    ```
  - Grep for any other `Node(` construction or `.branch_sandbox_id`/`.branch_issue_id`/`.branch_env_snapshot_id` reads in **non-test** code and update/remove; leave test-only references for Tasks that own those tests (note them for T5).

- [ ] **Step 4: Run tests + import check** — `python -m pytest customized_areal/tree_search/tests/ -k 'env_id or annotate' -v` and `python -c "import customized_areal.tree_search.core.tree_store"`

- [ ] **Step 5: Commit** (areal `master`): `feat(tree-search): per-node env_id replaces branch refs on Node`

---

### Task 2: SuperNode + serialization clean break (areal)

**Files:**
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py` (`SuperNode`)
- Modify: `customized_areal/tree_search/agents/execution_dag.py`
- Modify: `customized_areal/tree_search/agents/event_codec.py`
- Modify: `customized_areal/tree_search/core/checkpoint.py`
- Test: existing round-trip tests (`test_event_codec.py`, `test_checkpoint_super.py`, `test_checkpoint.py`)

**Interfaces:**
- `SuperNode` gains `env_id: str | None` and drops `branch_seq`, `branch_issue_id`, `branch_env_snapshot_id`.
- All (de)serialization persists `env_id` only.

- [ ] **Step 1: Update the round-trip tests to the new shape (TDD)**
  - In `test_checkpoint_super.py` / `test_event_codec.py`, replace `branch_env_snapshot_id="snap-9"` / `branch_issue_id=...` assertions with `env_id="env-9"` round-trips. Confirm no `branch_*` key survives.

- [ ] **Step 2: Run to confirm failure** — `python -m pytest customized_areal/tree_search/tests/test_event_codec.py customized_areal/tree_search/tests/test_checkpoint_super.py -v`

- [ ] **Step 3: Implement**
  - `supernode_assembler.py`: `SuperNode` — remove `branch_seq`, `branch_issue_id`, `branch_env_snapshot_id`; add `env_id: str | None = None`. Update where SuperNode is assembled from nodes to set `env_id=node.env_id` (drop the old branch-ref propagation, including `branch_seq`).
  - `execution_dag.py`: in `dag_to_supernodes` (SuperNode construction, ~L165-167) and `supernodes_to_dag`, replace `branch_seq`/`branch_issue_id`/`branch_env_snapshot_id` with `env_id=node.env_id`.
  - `event_codec.py`: the `to-dict` (~L93-94) and `from-dict` (~L161-162) blocks — persist/read `env_id` only; drop the branch keys. Update the `BranchReplay` docstring/type (~L188) or remove it if it exists solely to shape `BranchMaterializer.materialize` inputs (that class is removed in T5 — if `BranchReplay` is now unused, remove it and note it for T5).
  - `checkpoint.py`: node (de)serialization (~L249-252) — replace the three `branch_*` reads with `env_id=data.get("env_id")`.

- [ ] **Step 4: Run tests** — `python -m pytest customized_areal/tree_search/tests/test_event_codec.py customized_areal/tree_search/tests/test_checkpoint_super.py customized_areal/tree_search/tests/test_checkpoint.py -v`

- [ ] **Step 5: Commit** (areal `master`): `feat(tree-search): env_id-only serialization for SuperNode/checkpoint (clean break)`

---

### Task 3: Concrete BranchDriver via env-dispatch (areal, new)

**Files:**
- Create: `customized_areal/tree_search/agents/branch_driver.py`
- Test: `customized_areal/tree_search/tests/test_branch_driver.py` (new)
- Modify: `customized_areal/tree_search/tests/test_swe_lego_issue_runner.py` / `test_self_play_runner.py` (add a real-driver satisfaction test alongside the existing `FakeBranchDriver`)

**Interfaces:**
- Produces: `class EnvDispatchBranchDriver` implementing `async def drive_lane(*, agent_run_id: str, sandbox_id: str, session_id: str) -> str`, constructed with a `MulticaEnvDispatchClient`, a `domain`, a `dispatch_type`, and an `agent_id` (the lane's context).
- At a branch point it calls `create_env_dispatch(mode="branch", env_id=<node.env_id>, dispatch_type=…, agent_id=…, domain=…)` and returns the new terminal `env_id`.

- [ ] **Step 1: Write failing tests** (`test_branch_driver.py`)

```python
def test_branch_driver_calls_env_dispatch_branch():
    calls = {}
    class FakeClient:
        async def create_env_dispatch(self, **kw):
            calls.update(kw)
            from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup
            return _setup_with_env("env-child")   # helper building SweLegoSetup w/ one rollout
    drv = EnvDispatchBranchDriver(multica=FakeClient(), domain="swe_lego",
                                  dispatch_type="issue", agent_id="ag-1")
    terminal = asyncio.run(drv.drive_lane(agent_run_id="r1", sandbox_id="env-parent", session_id="s1"))
    assert calls["mode"] == "branch"
    assert calls["env_id"] == "env-parent"   # sandbox_id carries the source env_id (runner contract)
    assert calls["domain"] == "swe_lego"
    assert terminal == "env-child"
```

(Note the runner contract from `swe_lego_issue_runner.py`: `drive_lane` receives `sandbox_id=r.env_id`, i.e. the source env_id to branch from.)

- [ ] **Step 2: Run to confirm failure** — `python -m pytest customized_areal/tree_search/tests/test_branch_driver.py -v`

- [ ] **Step 3: Implement `branch_driver.py`** — the driver holds the client + lane context; `drive_lane` calls `create_env_dispatch(mode="branch", env_id=sandbox_id, dispatch_type=self._dispatch_type, agent_id=self._agent_id, domain=self._domain, group_size=1)` and returns `setup.rollouts[0].env_id`. Keep it importable without torch (stdlib logging only).

- [ ] **Step 4: Runner satisfaction test** — add a test that `run_swe_lego_issue` / `run_self_play` accept an `EnvDispatchBranchDriver` (structural `_BranchDriver` Protocol satisfied). Keep the existing `FakeBranchDriver` happy-path tests.

- [ ] **Step 5: Run** — `python -m pytest customized_areal/tree_search/tests/test_branch_driver.py customized_areal/tree_search/tests/test_swe_lego_issue_runner.py customized_areal/tree_search/tests/test_self_play_runner.py -v`

- [ ] **Step 6: Commit** (areal `master`): `feat(tree-search): EnvDispatchBranchDriver (branch via env-dispatch)`

---

### Task 4: Remove the wired workflow's branch machinery (areal — D6 revised to Option 3)

> **Execution note:** The original D6 (rewire the wired branch site to env-dispatch) was found BLOCKED during implementation — the wired site requires a le-agent TPFC `task_id` + seeded messages (token-level Node pipeline) that the env-dispatch branch primitive cannot supply without an external "materialize" endpoint. Per user decision, D6 reverts to **removing** the wired loop's branch machinery; branching lives only in the runner model (T3). A prior doc-only commit (`4ea01afa`) documents the blocker.

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py`
- Test: `tests/customized_areal/test_tree_search_branch_sampling.py`

**Goal:** The grouped workflow no longer branches (falls back to normal/scratch episodes). Remove `build_branch_task`, `_prepare_branch_task`, the `if branch_task_id:` branch block in the episode loop, and the branch-specific parts of `_cleanup_branch`; and fix the now-dead `candidate.branch_sandbox_id` references left after Task 1 (in `select_branch_candidate`, `build_branch_task`, `_cleanup_branch`) — which currently cause ~7 failing tests.

- [ ] **Step 1: Map the branch machinery.** Read `build_branch_task`, `_prepare_branch_task`, the `if branch_task_id:` block (~L1490), `_cleanup_branch`, and `select_branch_candidate`. Determine which are branch-only (remove) vs shared with the runner model / used elsewhere (keep). Grep every caller before deleting.

- [ ] **Step 2: Update tests to the no-branch contract.** In `test_tree_search_branch_sampling.py`, remove/rewrite the `build_branch_task` tests; assert the episode loop no longer attempts a branch (the candidate path is gone) and that no code references `branch_sandbox_id`. Run to confirm current failures/shape.

- [ ] **Step 3: Remove the branch machinery.**
  - Delete `build_branch_task` and `_prepare_branch_task`.
  - Remove the `if branch_task_id:` block from the episode loop (fall through to the normal `_retry_episode(episode_data)` path).
  - `_cleanup_branch`: remove entirely if now unused, or strip its branch-only body; remove all `candidate.branch_sandbox_id` references.
  - `select_branch_candidate`: if it exists solely to feed the removed wired branch path, remove it and its dead `branch_sandbox_id` reference; if it is shared (e.g., referenced by the runner model or DAG), keep it but remove the dead `branch_sandbox_id` access (candidates now carry `env_id`).
  - Prune now-unused imports (`truncate_messages_before_turn`, `bind_sandbox_to_task`, `copy_messages_to_task`) — grep first.

- [ ] **Step 4: Run tests + import** — `cd /workspaces/leagent/backend/areal && /workspaces/leagent/backend/areal/.venv-test/bin/python -m pytest tests/customized_areal/test_tree_search_branch_sampling.py -v` (the ~7 previously-failing branch tests are removed/green) and `/workspaces/leagent/backend/areal/.venv-test/bin/python -c "import customized_areal.tree_search.core.customized_grouped_workflow"`.

- [ ] **Step 5: Commit** (areal `master`): `refactor(tree-search): remove wired-loop branch machinery; branching now runner-only`

---

### Task 5: Client removals + import cleanup (areal, D4)

**Files:**
- Modify: `customized_areal/tree_search/agents/integration.py` (delete cloud-branch code)
- Modify: `customized_areal/tree_search/agents/__init__.py` (prune exports L82-91)
- Modify: `customized_areal/tpfc/backend_run.py` (delete `_start_branch_agent_run_for_task` + `_with_refresh`)
- Delete/trim tests: `tests/test_integration.py` (BranchMaterializer/MulticaIssueForker cases), `test_integration_multica.py`, `test_workflow_integration.py`, `test_e2e_branch_lifecycle.py`, `tests/customized_areal/test_tpfc_backend_auth.py` (start-branch cases)

**Interfaces:**
- Removes: `BranchMaterializer`, `BranchMaterializationResult`, `MulticaIssueForker`, `BranchStarter`, `BranchCandidate`, `materialize_cloud_branch`, `cleanup_cloud_branch` (integration.py). **Keep** `finalize_with_verifier` and `VerifierResult` usage.
- `agents/__init__.py`: drop the removed names from the `integration` import block (L82-91) and from `__all__` if listed.
- `backend_run.py`: remove the two `_start_branch_*` functions and the `LE_AGENT_BRANCH_RUN_ENDPOINT` handling; leave the non-branch agent-start path intact.

- [ ] **Step 1: Delete the code + prune exports + delete dead tests.** Grep each removed symbol to confirm no remaining **non-test** references (T1-T4 should have removed live users). Remove or gut the branch-only test files listed above (delete files whose sole purpose was the removed classes; trim mixed files).

- [ ] **Step 2: Import + collection check**
  - `python -c "import customized_areal.tree_search.agents"` (must succeed).
  - `python -m pytest customized_areal/tree_search/tests/ tests/customized_areal/ --collect-only -q` (no import errors from dangling references).

- [ ] **Step 3: Run the affected suites** — `python -m pytest customized_areal/tree_search/tests/ tests/customized_areal/test_tpfc_backend_auth.py -v`

- [ ] **Step 4: Commit** (areal `master`): `refactor(tree-search): remove BranchMaterializer + start-branch client (dead after env-dispatch)`

---

### Task 6: db_bridge channel removal (multica `main`, D4)

**Files:**
- Modify: `multica/db_bridge/channels.py` (remove `agent_start_branch` `Channel`)
- Modify: `multica/db_bridge/schema.sql` (remove the `rpc_agent_start_branch` row from the `tables[]` array)
- Modify: `multica/db_bridge/tests/test_leagent_channels.py` (remove `START_BRANCH` + its round-trip test), `multica/db_bridge/tests/test_integration_e2e.py` (remove the `BRIDGE_CONCURRENCY_AGENT_START_BRANCH` reference)
- Modify: `multica/db_bridge/README.md` (remove the channel's table row if present)

- [ ] **Step 1: Update tests first (TDD-ish).** Remove the `agent_start_branch` cases from `test_leagent_channels.py` and `test_integration_e2e.py`; add/keep an assertion that `agent_start_branch` is NOT in `CHANNELS_BY_NAME`.

- [ ] **Step 2: Confirm failure/scope** — `cd /workspaces/leagent/backend/areal/multica/db_bridge && uv run pytest -q` (expect failures referencing the removed channel until impl).

- [ ] **Step 3: Implement removal** — delete the `Channel(name="agent_start_branch", …)` from `channels.py`; grep `schema.sql` for `agent_start_branch` and remove that array entry; update README table.

- [ ] **Step 4: Run full db_bridge suite** — `cd multica/db_bridge && uv run pytest -q` (all green; note prior sub-project A's expected counts as baseline).

- [ ] **Step 5: Commit** (multica `main`): `refactor(db_bridge): remove agent_start_branch channel (branch now via env-dispatch)`

---

### Task 7: multica issue-fork endpoint removal (multica `main`, Go, D4)

**Files:**
- Delete: `multica/server/internal/handler/issue_fork.go`, `multica/server/internal/handler/issue_fork_test.go`, `multica/server/internal/service/issue_fork.go`, `multica/server/internal/service/issue_fork_test.go`
- Modify: `multica/server/cmd/server/router.go` (remove `r.Post("/fork", h.ForkIssue)` + `r.Delete("/fork", h.DeleteForkedIssue)`, L755-756, and prune the now-empty issue-`{id}` sub-route group if it only held these)
- Modify: `multica/server/cmd/server/router_fork_routes_test.go` (remove the two issue `/fork` route rows; KEEP the `/api/cloud-runtime/sandboxes/fork` row — that is a different, still-used endpoint)

**Interfaces:** removes `ForkIssue`, `DeleteForkedIssue` handlers and their service functions. Do NOT touch `ForkCloudRuntimeSandbox` / `/api/v1/sandboxes/fork` (used by env-dispatch's `ForkSandbox`).

- [ ] **Step 1: Grep for callers.** Confirm `ForkIssue`/`DeleteForkedIssue`/the service fork funcs are referenced only by their own files + router + tests. If a generated sqlc query exists solely for issue-fork, leave the query/generated code (harmless) — do NOT run `sqlc generate` (repo-wide regen is broken; see sub-project B ledger).

- [ ] **Step 2: Remove handlers/service/routes + trim tests.**

- [ ] **Step 3: Build + vet + test scoped** — `cd /workspaces/leagent/backend/areal/multica/server && go build ./internal/handler/ ./internal/service/ ./cmd/server/ && go vet ./internal/handler/ ./internal/service/ ./cmd/server/ && DATABASE_URL=postgres://multica:multica@localhost:5432/multica?sslmode=disable go test ./internal/handler/ ./internal/service/ ./cmd/server/` (note: the pre-existing 16 `ON CONFLICT` daemon/claim failures from sub-project B are unrelated — confirm the fork-removal tests are gone and nothing new breaks). `gofmt -l` clean.

- [ ] **Step 4: Commit** (multica `main`): `refactor(server): remove /api/issues/{id}/fork endpoints (branch now via env-dispatch)`

---

### Task 8: Full-suite regression (verification only)

**Files:** none

- [ ] **Step 1: areal client** — `cd /workspaces/leagent/backend/areal && /workspaces/leagent/backend/areal/.venv-test/bin/python -m pytest customized_areal/tree_search/ tests/customized_areal/ -q`. Expected: green; any failures must be pre-existing and reproduced at the pre-C base (check out the pre-C areal commit to confirm, as in sub-project B).
- [ ] **Step 2: import surface** — `python -c "import customized_areal.tree_search.agents; import customized_areal.tree_search.core.customized_grouped_workflow; import customized_areal.tree_search.core.tree_store"`.
- [ ] **Step 3: db_bridge** — `cd multica/db_bridge && uv run pytest -q`.
- [ ] **Step 4: multica Go** — `cd multica/server && go build ./internal/handler/ ./internal/service/ ./cmd/server/ && DATABASE_URL=… go test ./internal/handler/ ./internal/service/ ./cmd/server/` (only the pre-existing 16 `ON CONFLICT` failures allowed; prove pre-existing at base if in doubt).
- [ ] **Step 5: grep sweep** — no remaining non-test references to `BranchMaterializer`, `MulticaIssueForker`, `build_branch_task`, `agent_start_branch`, `branch_sandbox_id`, `branch_env_snapshot_id`, `branch_issue_id`, `ForkIssue` across `customized_areal/`, `multica/db_bridge/`, `multica/server/`.

---

## Self-Review

**Spec coverage:** D1 single mechanism → T3 (driver) + T4 (wired rewire) + T5/T6/T7 (removals). D2 per-node env_id frontier → T1 (Node.env_id) + T3 (branch uses env_id, no seq). D3 read metadata["env_id"] → T1 (`annotate_nodes_from_run`). D4 full removal → T5 (client) + T6 (channel) + T7 (issue-fork); start-branch server endpoint noted external. D5 clean break → T1 (Node) + T2 (SuperNode/serialization). D6 rewire wired loop → T4. Approach 2 (`_BranchDriver` seam) → T3.

**Placeholder scan:** No TBD/TODO. Test skeletons name the exact symbols and asserted keys; file targets cite the verified line ranges (execution_dag L165-167, event_codec L93-94/L161-162, checkpoint L249-252, router.go L755-756, __init__ L82-91). Where a helper (`_setup_with_env`) is introduced, the step says to add it.

**Type/contract consistency:** `Node.env_id` / `SuperNode.env_id` are `str | None`. `drive_lane(agent_run_id, sandbox_id, session_id) -> str` matches the existing Protocol in both runners; the driver treats `sandbox_id` as the source env_id per the runners' documented contract ("areal passes env_id to branch"). `create_env_dispatch(mode="branch", env_id=…)` matches the sub-project B client signature. Retained: `finalize_with_verifier`, `VerifierResult`, `ForkCloudRuntimeSandbox`/`/api/v1/sandboxes/fork`.

**Risk isolation:** T4 (`_retry_episode` impedance) is isolated with a mandatory read-and-document Step 1 and an explicit STOP-and-report if the seam doesn't fit — no silent leaky adapter.

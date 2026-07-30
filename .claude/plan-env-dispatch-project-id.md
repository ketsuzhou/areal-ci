# Plan: `create_env_dispatch` returns `project_id`; defer empty-trajectory check to assembler

## Confirmed decisions
- `create_env_dispatch` returns `project_id` (str) directly; the workflow only needs project_id.
- Remove the workflow's early `if not setup.rollouts: return None` guard; the empty-trajectory check moves **into the assembler** (returns `None` on empty dag).
- **Delete `branch_driver`** (dormant — not wired at `customized_grouped_workflow.py:943`; blocks a bare-project_id return because it reads `rollouts[0].env_id`).
- Add a top-level `project_id` to the multica dispatch API response (parsed by the areal client). Modify the untracked `multica/` checkout here for contract + test consistency.
- `multica/` is untracked in this repo (`?? multica/`) — Go changes won't deploy from here, but document the contract.

## Key findings driving the design
- API `EnvDispatchResponse` / `EnvDispatchResult` have **no** top-level `project_id`; each `EnvRollout` has its own `ProjectID` (created per-rollout in `resetOne`). Wired `group_size=1`, so one project per dispatch. Top-level `project_id = rollouts[0].ProjectID` (the single project the workflow already consumes via `get_dag`).
- `segment_dag_trainer.run_segment_dag_training_step` (L204) calls `assemble_from_refs` then `edag.topological_order()` — must handle `None`.
- `customized_grouped_workflow.py:943` constructs the workflow **without** `branch_driver` and doesn't set `branch_from_env_id` in data — removing the branch path is safe.

## Changes

### 1. multica server (untracked `multica/`)
- `server/internal/service/env_dispatch.go`: add `ProjectID string` to `EnvDispatchResult` (L112); set `ProjectID: rollouts[0].ProjectID` at `Dispatch()` L387 (after the all-or-nothing reset gate, all rollouts have ProjectID; GroupSize≥1).
- `server/internal/handler/env_dispatch.go`: add `ProjectID string \`json:"project_id"\`` to `EnvDispatchResponse` (L60); set it at L152 (201) and L181 (error path) from `res.ProjectID`.
- Tests: `mapRollouts` tests unaffected; `doEnvDispatch` tests check status, not exact body (backward-compatible). Verify no exact-JSON-body assertion; add a service test asserting `result.ProjectID == rollouts[0].ProjectID`.

### 2. areal client — `customized_areal/tree_search/agents/swe_lego_client.py`
- `create_env_dispatch(...)`: return type `SweLegoSetup` → `str`. Parse `body["project_id"]` and return it. Drop `body["rollouts"]` parsing + `SweLegoRollout`/`SweLegoSetup` construction.
- Remove now-unused imports `SweLegoRollout`, `SweLegoSetup` (keep `SweLegoIssue`).

### 3. areal workflow — `customized_areal/tree_search/agents/multi_agent_env_dispatch.py`
- `__init__`: drop `branch_driver` param + `self._branch_driver`.
- `arun_episode`: remove the `branch_from_env_id` branch path (always dispatch on `self.base_env_id`); `project_id = await self._dispatch.create_env_dispatch(mode="scratch", env_id=self.base_env_id, ...)`; remove `if not setup.rollouts` guard and `rollouts[0].project_id`; after `edag = assemble_from_refs(...)`, add `if edag is None: return None`; cleanup + return unchanged.
- Update module/class docstrings (drop branch-driver refs).

### 4. areal assembler — `customized_areal/tree_search/agents/supernode_assembler.py`
- `assemble_from_refs(...) -> ExecutionDAG | None`: `if not dag.segments: return None` at the top; update annotation + docstring.

### 5. areal trainer — `customized_areal/tree_search/agents/segment_dag_trainer.py`
- `run_segment_dag_training_step` (L204): `if edag is None: return assemble_node_advantages([], initial_value=0.0, gamma=gamma, lam=lam)` before `edag.topological_order()`.

### 6. Delete branch_driver
- Delete `agents/branch_driver.py` + `tests/test_branch_driver.py`.
- Update `agents/integration.py:7` docstring (drop `EnvDispatchBranchDriver` ref).

### 7. Dead-type cleanup (recommended)
- Remove `SweLegoSetup`/`SweLegoRollout` from `reward/swe_lego_types.py` (unused in non-test code after this) + their tests in `tests/test_swe_lego_types.py`. Keep `SweLegoIssue`; leave `SweLegoIssueResult` (pre-existing, out of scope).

### 8. areal test updates
- `tests/test_multi_agent_env_dispatch.py`: `_FakeDispatch.create_env_dispatch` returns project_id str; drop `branch_driver` from `_make_workflow`; remove the 3 branch tests; replace partial-squad test with empty-trajectory→None (`segments=[]`) and a non-empty success test (dag with a segment).
- `tests/test_env_dispatch_client.py`: assert `create_env_dispatch` returns project_id parsed from `body["project_id"]`; keep payload-construction assertions.
- `tests/test_v2_session_lifecycle.py`: fake `create_env_dispatch` returns project_id str.
- `tests/test_multica_workflow_wiring.py`: drop branch_driver/SweLegoSetup refs.
- `tests/test_supernode_assembler.py`: add `assemble_from_refs` returns `None` for `segments == []`.

### 9. Verification
- `uvx ruff check` on touched areal files.
- `.venv-test/bin/python -m pytest` the touched areal test files.
- `cd multica/server && go test ./internal/handler/ ./internal/service/` if Go toolchain available; else note as unverified (untracked anyway).

## Risks / notes
- `group_size>1`: top-level project_id = first rollout's project (workflow already consumes one project_id via single `get_dag`). N>1 not wired. Documented.
- Idempotent replay of pre-change saved `EnvDispatchResult` → empty `ProjectID` (rare, acceptable).
- `multica/` untracked → Go changes here are contract documentation, not deployment.

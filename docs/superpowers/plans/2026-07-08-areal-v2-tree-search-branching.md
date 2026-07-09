---
change: areal-v2-integration-and-tree-search-branching
design-doc: docs/superpowers/specs/2026-07-08-areal-v2-tree-search-branching-design.md
base-ref: d12c28813d226721950848b14ce78135c05f1722
---

# AReaL v2 Multi-Agent Integration + Tree-Search Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire multi-agent squad rollout (N sessions/task) and tree-search branching (`BRANCH` edge + MCTS backup) into AReaL's v2 online training path by **integrating the existing v2-segment-dag infrastructure** (not reimplementing it), so a pi-agent trains from Multica collaborative tasks.

**Architecture (revised - Approach B, reuse existing components):** A new thin `MultiAgentEnvDispatchWorkflow` (base `RolloutWorkflow`) orchestrates existing components per `arun_episode`: `MulticaEnvDispatchClient.create_env_dispatch` (async) -> `MulticaDagClient.get_dag` (sync, via `asyncio.to_thread`) -> `SuperNodeAssembler.assemble_from_refs` (sync, via `asyncio.to_thread`) -> `ExecutionDAG[SuperNode]`, with partial-squad validation + tensor-ref/session cleanup. `TreeSearchGroupedRolloutWorkflow` (Approach A, group-level) consumes the `ExecutionDAG[SuperNode]` via a new `_result_to_nodes`/`_finalize_episode` multica branch (SuperNode-preserving, not flattened to Nodes). Branching adds a `BRANCH` edge type + MCTS value backup; the branch fork uses the existing `EnvDispatchBranchDriver` (`create_env_dispatch(mode="branch")`) - **F-independent**.

**Tech Stack:** Python 3.12+ (AReaL), Go (Multica side), PyTorch, v2 inference_service (gateway/data_proxy/router/controller), OpenSpec.

## Global Constraints

- Python >= 3.12 (`backend/areal/pyproject.toml`); pre-commit: `pre-commit run --all-files` before commit.
- **AReaL NEVER calls `rl_session.start(...)` / `/rl/start_session` on the multi-agent path.** Multica owns `/rl/start_session(group_size=N)`, the `session_to_agent_run` binding, and per-agent credential distribution. AReaL only dispatches + polls.
- **Harvest = direct `AssembledDag` polling** (`GET .../dag` 202 in-progress / 200 done), NOT per-trajectory `/callback/online_ready` (that stays single-agent `InferenceServiceWorkflow` only).
- v2 mint-on-demand: no pre-grant, no capacity ratchet; `SessionStore.start_session` is unbounded.
- Staleness gates `rollout_batch` admission, not session minting.
- **Reuse existing v2-segment-dag components; do NOT reimplement:** `MulticaEnvDispatchClient.create_env_dispatch` (`swe_lego_client.py`), `MulticaDagClient.get_dag` (`multica_dag_client.py`), `SuperNodeAssembler.assemble_from_refs` (`supernode_assembler.py`), `DataProxyTensorResolver`/`DataProxySessionRemover` (`segment_dag_trainer.py`), `assemble_node_advantages` (`dag_advantage.py`), `EnvDispatchBranchDriver` (`branch_driver.py`), `MCTSTreeStore`/`TreeAdvantageComputer` (`tree_store.py`/`advantage.py`).
- **Sync->async bridge:** the existing segment-DAG pipeline is sync (httpx); the grouped workflow is async. Wrap sync calls in `asyncio.to_thread` so M parallel `arun_episode`s stay concurrent.
- **Branching is F-independent:** fork via `EnvDispatchBranchDriver.create_env_dispatch(mode="branch")` (Multica env-dispatch). Sub-project F (full sandbox snapshot/fork) is out of scope; `env_snapshot` is refs-only. Branch *execution* lives in the base workflow (`MultiAgentEnvDispatchWorkflow`); branch *selection/backup/bounds* live in the grouped workflow/`tree_store` (per `customized_grouped_workflow.py:1416` "branching now lives only in the env-dispatch runner model").
- `BRANCH` backup = MCTS value propagation to parent checkpoint node (R1=A); each branch keeps its own advantage.
- Partial squad failure -> drop the task (no partial `SuperNode`).
- N=1 degenerates to the single-agent leaf `SuperNode` (parity).
- `agent_versions` rows are immutable; RLS stays on; no wildcard imports; no hardcoded secrets.
- **Test runner: `python3 -m pytest`** (NOT `uv run pytest` - uv is broken in this env; `python3` = `.venv-test/bin/python3`, has deps + `pythonpath=["."]`). Lint: `uvx ruff check`. No GPU in dev env; unit tests CPU-only with fakes.

## Dependency Note (read before executing)

- Builds on `multica-v2-segment-dag-training` (current branch `multica-v2-segment-dag-training`, unmerged) which shipped: `close_segment`, per-segment export, `AssembledDag`/`SegmentSpec`/`EdgeSpec`, `MulticaDagClient.get_dag`, `SuperNodeAssembler.assemble_from_refs`, `DataProxyTensorResolver`/`DataProxySessionRemover`, `run_segment_dag_training_step`, `assemble_node_advantages`, `EnvDispatchBranchDriver`, `MCTSTreeStore`, `TreeAdvantageComputer`.
- This change = v2 roadmap **Change 3** (tree search/branching) + multi-agent integration: wire the segment-DAG path into `TreeSearchGroupedRolloutWorkflow` + add `BRANCH` edge + MCTS backup + branch execution. It does NOT touch the v2 segment-DAG data path (Change 1) or judge/V/GAE (Change 2).
- **All phases are F-independent** (branching uses `EnvDispatchBranchDriver`, not Sub-project F).

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `customized_areal/tree_search/agents/multi_agent_env_dispatch.py` | NEW thin `RolloutWorkflow`: `arun_episode` = create_env_dispatch + get_dag + assemble_from_refs (asyncio.to_thread) + partial-squad validation + cleanup + branch execution via `EnvDispatchBranchDriver` | create |
| `customized_areal/tree_search/core/customized_grouped_workflow.py` | `_result_to_nodes` multica branch (SuperNode-preserving); `_finalize_episode` multi-SuperNode DAG insertion + advantages; activate `multica_dag_client` hook + `self.workflow`; `max_group_size` branch bound | modify |
| `customized_areal/tree_search/agents/execution_dag.py` | `EdgeType.BRANCH` + `Edge` provenance fields + `to_records`/`from_records` | modify |
| `customized_areal/tree_search/dag/backup.py` | NEW MCTS value backup across `BRANCH` edges | create |
| `customized_areal/tree_search/core/tree_store.py` | `Node.visit_count` field (for MCTS backup) + ser/deser | modify |
| `customized_areal/tree_search/agents/__init__.py` | export `MultiAgentEnvDispatchWorkflow` | modify |
| tests under `customized_areal/tree_search/tests/` | unit tests for each component | create |

---

## Phase 1 - `MultiAgentEnvDispatchWorkflow` orchestrator (reuse existing)

### Task 1.1: `MultiAgentEnvDispatchWorkflow` skeleton + SCRATCH `arun_episode` (N=1, reuse)

**Files:**
- Create: `customized_areal/tree_search/agents/multi_agent_env_dispatch.py`
- Modify: `customized_areal/tree_search/agents/__init__.py` (export)
- Test: `customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py`

**Interfaces:**
- Consumes (all existing): `MulticaEnvDispatchClient.create_env_dispatch` (async), `MulticaDagClient.get_dag` (sync), `SuperNodeAssembler.assemble_from_refs` (sync), `DataProxyTensorResolver` (sync), `DataProxySessionRemover` (sync).
- Produces: `class MultiAgentEnvDispatchWorkflow(RolloutWorkflow)` with `async arun_episode(engine, data) -> dict | None` returning `{"assembled_dag": AssembledDag, "execution_dag": ExecutionDAG}` or `None`.

- [ ] **Step 1: Write the failing test** (N=1 SCRATCH with fakes)

```python
# customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py
import pytest
from customized_areal.tree_search.agents.multi_agent_env_dispatch import MultiAgentEnvDispatchWorkflow
from customized_areal.tree_search.agents.multica_dag_client import AssembledDag
from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup, SweLegoRollout

class _FakeDispatch:
    async def create_env_dispatch(self, **kw):
        return SweLegoSetup(rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")])

class _FakeDagClient:
    def get_dag(self, project_id, *, timeout, interval):  # sync
        return AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})

class _FakeAssembler:
    def assemble_from_refs(self, dag, resolver):  # sync
        edag = object()  # stand-in; real test uses/returns a real ExecutionDAG
        return edag

class _FakeResolver:
    def resolve(self, tensor_ref): return {}
    def clear(self, shard_ids): pass

class _FakeSessionRemover:
    def __init__(self): self.removed = []
    def remove(self, session_id): self.removed.append(session_id)

@pytest.mark.asyncio
async def test_arun_episode_scratch_n1_returns_execution_dag():
    sr = _FakeSessionRemover()
    wf = MultiAgentEnvDispatchWorkflow(
        dispatch_client=_FakeDispatch(), dag_client=_FakeDagClient(),
        assembler=_FakeAssembler(), resolver=_FakeResolver(), session_remover=sr,
        poll_timeout=5.0, poll_interval=0.0, group_size=1, base_env_id="e0")
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert out["assembled_dag"].session_to_agent_run == {"sess1": "r1"}
    assert sr.removed == ["sess1"]  # session cleaned up
```

- [ ] **Step 2: Run test to verify it fails** -> `python3 -m pytest customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -v` -> FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

```python
# customized_areal/tree_search/agents/multi_agent_env_dispatch.py
from __future__ import annotations
import asyncio
import logging
from typing import Any
from areal.api.workflow_api import RolloutWorkflow

logger = logging.getLogger("MultiAgentEnvDispatchWorkflow")

class MultiAgentEnvDispatchWorkflow(RolloutWorkflow):
    """One arun_episode = one Multica task = N agents = N sessions -> AssembledDag.

    Thin orchestrator over existing v2-segment-dag components: dispatches via
    MulticaEnvDispatchClient, polls via MulticaDagClient.get_dag, assembles via
    SuperNodeAssembler.assemble_from_refs. Multica owns /rl/start_session +
    session_to_agent_run + credentials; AReaL never calls start_session.
    Sync component calls run in asyncio.to_thread to keep M episodes concurrent.
    """

    def __init__(self, *, dispatch_client, dag_client, assembler, resolver,
                 session_remover, poll_timeout: float = 3600.0, poll_interval: float = 1.0,
                 group_size: int = 1, base_env_id: str = ""):
        self._dispatch = dispatch_client
        self._dag_client = dag_client
        self._assembler = assembler
        self._resolver = resolver
        self._session_remover = session_remover
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.group_size = group_size
        self.base_env_id = base_env_id

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        setup = await self._dispatch.create_env_dispatch(
            mode="scratch", env_id=self.base_env_id, dispatch_type="message",
            agent_id=data.get("agent_id", ""), group_size=self.group_size,
            domain="multica", message=data.get("message"))
        if not setup.rollouts:
            return None
        project_id = setup.rollouts[0].project_id
        from customized_areal.tree_search.agents.multica_dag_client import DagTimeout
        try:
            dag = await asyncio.to_thread(
                self._dag_client.get_dag, project_id,
                timeout=self.poll_timeout, interval=self.poll_interval)
        except DagTimeout:
            logger.warning("AssembledDag poll timed out for project %s; rejecting", project_id)
            return None
        expected = {r.agent_run_id for r in setup.rollouts if r.agent_run_id}
        covered = set(dag.session_to_agent_run.values())
        if not expected.issubset(covered):
            logger.warning("Partial squad for project %s: expected %s covered %s; dropping",
                           project_id, expected, covered)
            return None
        edag = await asyncio.to_thread(self._assembler.assemble_from_refs, dag, self._resolver)
        # Cleanup (success-path only): release shards, revoke sessions.
        shard_ids = [s.tensor_ref.get("shard_id") for s in dag.segments if s.tensor_ref.get("shard_id")]
        if shard_ids:
            await asyncio.to_thread(self._resolver.clear, shard_ids)
        for session_id in dag.session_to_agent_run:
            await asyncio.to_thread(self._session_remover.remove, session_id)
        return {"assembled_dag": dag, "execution_dag": edag}
```

Add to `customized_areal/tree_search/agents/__init__.py`:
```python
from customized_areal.tree_search.agents.multi_agent_env_dispatch import MultiAgentEnvDispatchWorkflow
```

- [ ] **Step 4: Run test to verify it passes** -> PASS (output pristine).
- [ ] **Step 5: Lint + commit** -> `uvx ruff check ...`; `git commit -m "feat(tree-search): MultiAgentEnvDispatchWorkflow orchestrator (reuse segment-DAG)"`.

### Task 1.2: N>1 multi-agent squad + partial-squad drop

**Files:** Modify `multi_agent_env_dispatch.py` (partial-squad validation already in 1.1; this task adds the N>1 test).

- [ ] **Step 1: Write failing tests** - N=2 squad, both `agent_run_id`s covered -> returns `execution_dag` with 2 segments covered; one `agent_run_id` missing from `session_to_agent_run` -> returns `None`, no cleanup of partial state (or cleanup of covered sessions - decide and assert).
- [ ] **Step 2: Run -> FAIL** (or pass if 1.1 validation covers it - assert the contract).
- [ ] **Step 3: Implement/verify** - confirm partial-squad validation drops; ensure no `SuperNode`/`execution_dag` is built on partial.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): N>1 squad + partial-squad drop"`.

### Task 1.3: Tensor-ref resolution + cleanup ordering (reuse `DataProxyTensorResolver`/`SessionRemover`)

**Files:** `multi_agent_env_dispatch.py` (cleanup sequence), test.

- [ ] **Step 1: Write failing test** - `resolver.resolve` is called per segment (via `assemble_from_refs`); `resolver.clear` called with the consumed `shard_id`s; `session_remover.remove` called per session; on `DAGError` from `assemble_from_refs`, cleanup is NOT called (shards/sessions preserved for retry - matches `run_segment_dag_training_step` success-path-only cleanup).
- [ ] **Step 2: Run -> FAIL** (or assert the contract).
- [ ] **Step 3: Implement/verify** - cleanup runs only after successful `assemble_from_refs`; `DAGError` propagates without cleanup.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): tensor-ref resolve + success-path cleanup ordering"`.

### Task 1.4: Polling timeout + typed DAG errors

**Files:** test.

- [ ] **Step 1: Write failing test** - `dag_client.get_dag` raises `DagTimeout` -> `arun_episode` returns `None`; raises `DagNotFound`/`DagForbidden` -> `arun_episode` returns `None` (or re-raises - decide; the grouped workflow's `_retry_episode` catches exceptions and retries, so re-raising may be appropriate; confirm and assert).
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement/verify** - decide timeout-vs-not-found handling (timeout -> None reject; not-found/forbidden -> ?). Document the choice.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): polling timeout + typed DAG error handling"`.

---

## Phase 2 - `TreeSearchGroupedRolloutWorkflow` wiring (Approach B: SuperNode-preserving)

### Task 2.1: `_result_to_nodes` multica branch (SuperNode-preserving)

**Files:** Modify `customized_areal/tree_search/core/customized_grouped_workflow.py:929` (`_result_to_nodes`); test `test_result_to_nodes_multica.py`.

**Interfaces:** Consumes `MultiAgentEnvDispatchWorkflow.arun_episode` output `{"assembled_dag", "execution_dag"}`. Preserves the `ExecutionDAG[SuperNode]` (does NOT flatten to `Node`s).

- [ ] **Step 1: Write failing test** - `_result_to_nodes` receives the multica result -> returns the `ExecutionDAG`'s `SuperNode`s (a new branch, distinct from the single-agent `InteractionWithTokenLogpReward` dict path). Single-agent dict path unchanged.
- [ ] **Step 2: Run -> FAIL** (current `_result_to_nodes` returns `None` for the dict-with-`execution_dag` shape).
- [ ] **Step 3: Implement** - add a branch in `_result_to_nodes`: when `result` is a dict carrying `"execution_dag"`, return its `SuperNode`s (preserve structure; stamp `query_id`/`group_idx` onto each SuperNode). **Do not flatten to `Node`s** - the multi-segment edge structure must survive to `_finalize_episode`. (The return type widens to `list[Node] | list[SuperNode] | None`; document this.)
- [ ] **Step 4: Run -> PASS** (incl. single-agent path unchanged).
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): _result_to_nodes multica branch (SuperNode-preserving)"`.

### Task 2.2: `_finalize_episode` multi-SuperNode DAG insertion + advantages

**Files:** Modify `customized_grouped_workflow.py:1754` (`_finalize_episode`); test.

- [ ] **Step 1: Write failing test** - a multica episode yields `list[SuperNode]` (multi-segment DAG) -> `_finalize_episode` inserts them via `tree_store.insert_super_batch` (NOT `_wrap_leaf_super`), computes advantages over the DAG, and produces a batched tensor dict. N=1 multica yields one leaf `SuperNode` identical to the single-agent path (parity).
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - add a multica branch in `_finalize_episode`: when fresh nodes are `SuperNode`s, `insert_super_batch(super_nodes)` + compute advantages. Choose the advantage computer: `TreeAdvantageComputer` if it handles `SuperNode`s, else `assemble_node_advantages` over `topological_order()`. **Confirm which computer handles the multi-SuperNode DAG** during implementation (the grouped workflow's `TreeAdvantageComputer.compute(all_nodes)` currently takes `list[Node]`; verify it works on `SuperNode`s or adapt). Skip zero-variance discard / distillation / judge for the multica path in this change (they're Change 2 concerns).
- [ ] **Step 4: Run -> PASS** (incl. N=1 parity).
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): _finalize_episode multi-SuperNode DAG insertion + advantages"`.

### Task 2.3: Activate `multica_dag_client` hook + wire `self.workflow`

**Files:** Modify `customized_grouped_workflow.py:682/852` (the hook + constructor); test.

- [ ] **Step 1: Write failing test** - when `multica_dag_enabled=True`, the grouped workflow's `self.workflow` is a `MultiAgentEnvDispatchWorkflow` (constructed with the `multica_dag_client` + a resolver/session_remover/assembler) and `self._multica_dag_client` is non-None. Non-multica paths (`OpenAIProxyWorkflow`/`InferenceServiceWorkflow`) unchanged.
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - in `__init__`, when `multica_dag_enabled`, construct `MultiAgentEnvDispatchWorkflow(dispatch_client=multica_dag_client, dag_client=..., assembler=..., resolver=..., session_remover=..., ...)` and assign to `self.workflow`. Document the construction contract (which client is the `dispatch_client` vs `dag_client` - note `MulticaEnvDispatchClient` does dispatch, `MulticaDagClient` does `get_dag`). Wire the `multica_dag_client` hook consistently (the existing hook stored at `:852`).
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): activate multica_dag_client hook + base workflow wiring"`.

### Task 2.4: `group_size=M` parallel multi-agent rollouts

**Files:** Modify `customized_grouped_workflow.py:1496` (gather shape); test.

- [ ] **Step 1: Write failing test** - `group_size=2` -> 2 parallel `arun_episode` calls -> 2 `AssembledDag`/`ExecutionDAG` results aggregated; cached-episode path (`load_untrained_episodes`) handles multi-SuperNode results.
- [ ] **Step 2: Run -> FAIL** (or assert the gather shape).
- [ ] **Step 3: Implement/verify** - confirm `_arun_episode_fixed` gather produces M multica results; fix any cached-episode path assuming single-agent leaf shape.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): group_size=M parallel multi-agent rollouts"`.

---

## Phase 3 - Tree-search branching (F-independent, via `EnvDispatchBranchDriver`)

### Task 3.1: `EdgeType.BRANCH` + provenance fields

**Files:** Modify `customized_areal/tree_search/agents/execution_dag.py` (`EdgeType:30`, `Edge:200`, `SuperNode.to_records/from_records`); test `test_branch_edge.py`.

- [ ] **Step 1: Write failing test** - `EdgeType.BRANCH == "branch"`; an `Edge` with `branch_from_segment_id`/`branch_from_checkpoint_id` round-trips through `SuperNode.to_records`/`from_records` and `ExecutionDAG.add_edge`.
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - add `BRANCH = "branch"` to `EdgeType`; add optional `branch_from_segment_id: str | None`/`branch_from_checkpoint_id: str | None` to `Edge`; update `to_records`/`from_records` + `_coerce_edge_type` to serialize/parse `branch`.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): BRANCH edge type + fork provenance fields"`.

### Task 3.2: `BRANCH` edge parsing in `AssembledDag`

**Files:** Modify `customized_areal/tree_search/agents/supernode_assembler.py` (edge mapping); test.

- [ ] **Step 1: Write failing test** - an `AssembledDag` with a `branch` edge (carrying `branch_from_segment_id`/`branch_from_checkpoint_id`) -> `assemble_from_refs` produces an `ExecutionDAG` with an `EdgeType.BRANCH` edge + provenance.
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - the assembler's edge mapping (`EdgeSpec.type` `"branch"` -> `EdgeType.BRANCH`) + provenance fields. (Multica emits `branch` edges; AReaL parses them.)
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): parse BRANCH edges in AssembledDag"`.

### Task 3.3: `Node.visit_count` + MCTS backup (`backup.py`)

**Files:** Modify `customized_areal/tree_search/core/tree_store.py` (`Node`); Create `customized_areal/tree_search/dag/backup.py`; test `test_branch_backup.py`.

- [ ] **Step 1: Write failing test**

```python
# test_branch_backup.py
from customized_areal.tree_search.dag.backup import branch_backup
from customized_areal.tree_search.core.tree_store import Node

def test_branch_return_updates_parent_value():
    parent = Node(node_id="ckpt1", value=0.0, visit_count=0)
    branch_backup(parent, branch_return=1.0, discount=0.9, visit_count=1)
    assert parent.value == 0.9
    assert parent.visit_count == 1

def test_multiple_branches_aggregate():
    parent = Node(node_id="ckpt1", value=0.0, visit_count=0)
    branch_backup(parent, branch_return=1.0, discount=1.0, visit_count=1)
    branch_backup(parent, branch_return=0.0, discount=1.0, visit_count=1)
    assert parent.value == 0.5
    assert parent.visit_count == 2
```

- [ ] **Step 2: Run -> FAIL** (`Node` has no `visit_count`; `backup.py` absent).
- [ ] **Step 3: Implement** - add `visit_count: int = 0` to `Node` (+ handle in `Node` ser/deser if any); create `backup.py`:

```python
# customized_areal/tree_search/dag/backup.py
from customized_areal.tree_search.core.tree_store import Node

def branch_backup(parent: Node, *, branch_return: float, discount: float, visit_count: int = 1) -> None:
    """MCTS-style: propagate discounted branch return to parent checkpoint node.

    parent.value = running mean of discounted branch returns over visits.
    """
    discounted = discount * branch_return
    new_count = parent.visit_count + visit_count
    parent.value = (parent.value * parent.visit_count + discounted * visit_count) / new_count
    parent.visit_count = new_count
```

- [ ] **Step 4: Run -> PASS** (2 tests).
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): Node.visit_count + MCTS value backup across BRANCH edges"`.

### Task 3.4: Branch execution in `MultiAgentEnvDispatchWorkflow` (via `EnvDispatchBranchDriver`)

**Files:** Modify `multi_agent_env_dispatch.py` (branch mode in `arun_episode`); test.

- [ ] **Step 1: Write failing test** - when `data` carries a branch source (`branch_from_env_id`/`branch_from_segment_id`), `arun_episode` forks via `EnvDispatchBranchDriver.drive_lane` (fake) instead of `mode="scratch"`, runs the branched squad, harvests the branched `AssembledDag` (with a `branch` edge), and returns it. SCRATCH path unchanged.
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - add a branch mode to `arun_episode`: if `data` indicates a branch, use `EnvDispatchBranchDriver.drive_lane(agent_run_id, sandbox_id=source_env_id, session_id=...)` to fork, then `create_env_dispatch(mode="scratch", env_id=child_env_id, ...)` for the branched squad, then poll + assemble as usual. (Branch *selection* - which node to branch from - stays in the grouped workflow's `select_branch_candidate`; this task is *execution* only.)
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): branch execution via EnvDispatchBranchDriver (F-independent)"`.

### Task 3.5: `max_group_size` bound + circuit breaker (grouped workflow)

**Files:** Modify `customized_grouped_workflow.py` (`choose_sample_source`/branch sampling); test.

- [ ] **Step 1: Write failing test** - `max_group_size=4`, `initial_group_size=2` -> at most 2 branch samples per query; consecutive branch failures trip the circuit breaker (stops branching).
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - enforce total branches <= `max_group_size - initial_group_size` in the branch-sampling path; the existing consecutive-failure circuit breaker applies to multi-agent branches.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): bound branches by max_group_size + circuit breaker"`.

### Task 3.6: MCTS backup wiring + branch cleanup

**Files:** Modify `customized_grouped_workflow.py` (`_finalize_episode`/backup hook); test.

- [ ] **Step 1: Write failing test** - a branched trajectory's terminal return propagates via `branch_backup` to the parent segment's checkpoint `Node` (value updated, `visit_count` incremented); branched sessions are removed (`session_remover.remove`) on cleanup.
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - wire `branch_backup` into the grouped workflow's backup path for `BRANCH` edges; branch cleanup releases branched sessions.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): wire MCTS branch backup + branch cleanup"`.

---

## Phase 4 - v2 online training integration (verification)

### Task 4.1: Staleness gates batch admission, not minting

**Files:** Test `customized_areal/tree_search/tests/test_staleness_batch_admission.py`.

- [ ] **Step 1: Write failing test** - `rollout_batch(M)` admits M `arun_episode`s; `start_session` is NOT capacity-gated; excess `AssembledDag`s hold at Multica's `202` until polled. (Assert the contract against the controller; if behavior already holds, the test documents it.)
- [ ] **Step 2: Run -> FAIL** (or pass if contract holds).
- [ ] **Step 3: Implement/verify** - confirm `controller.rollout_batch` gates via `staleness_manager`; no session-level gate added. Document in a code comment.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): staleness gates batch admission, not minting"`.

### Task 4.2: Cross-step staleness guard (R2)

**Files:** Test.

- [ ] **Step 1: Write failing test** - `rollout_batch` awaits all `arun_episode`s (incl. branch polling) before `set_version`; no trajectory leaks across steps.
- [ ] **Step 2: Run -> FAIL** (or pass if `rollout_batch` already awaits).
- [ ] **Step 3: Implement/verify** - assert `rollout_batch` does not call `set_version` until all submitted `arun_episode`s resolve; add a guard if absent.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): no cross-step staleness leak (R2)"`.

### Task 4.3: Session lifecycle (mint on demand, harvest + remove)

**Files:** Test.

- [ ] **Step 1: Write failing test** - N sessions minted by **Multica** (AReaL never calls `start_session`); each harvested once with `remove_session=True` via `DataProxySessionRemover`; `close_segment` per segment. `MultiAgentEnvDispatchWorkflow` does NOT mint.
- [ ] **Step 2: Run -> FAIL** (or assert the contract).
- [ ] **Step 3: Implement/verify** - confirm the harvest path (`DataProxySessionRemover.remove`) calls export with `remove_session=True`; AReaL mints nothing.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): session lifecycle (Multica mints, AReaL harvests+removes)"`.

---

## Phase 5 - Tests and end-to-end

### Task 5.1: Multi-agent orchestrator unit suite (consolidation)
- [ ] Consolidate Tasks 1.1-1.4 tests; run `python3 -m pytest customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -v`. Commit.

### Task 5.2: Integration - M x N parallel sessions on a fake v2 gateway
- [ ] **Step 1: Write failing test** - M=2, N=2 -> 4 sessions on a fake v2 gateway; router routes by `session_key`; no `429` from `SessionStore` (minting unbounded).
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - fake v2 gateway (`SessionStore` in-process) + fake router; drive `MultiAgentEnvDispatchWorkflow` x M.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): M×N parallel sessions, router routing, no 429"`.

### Task 5.3: Staleness backpressure test
- [ ] **Step 1: Write failing test** - Multica mints faster than trainer admits -> `AssembledDag`s hold at `202`; trainer drains in later batches.
- [ ] **Step 2: Run -> FAIL**.
- [ ] **Step 3: Implement** - fake controller with bounded `max_concurrent_rollouts`; assert 202-holding + drain.
- [ ] **Step 4: Run -> PASS**.
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): staleness backpressure (AssembledDag 202-holding)"`.

### Task 5.4: Multi-level branch-tree backup test
- [ ] Write a multi-level branch tree test (Task 3.3/3.6 extended): a 2-level branch tree -> `branch_backup` aggregates correctly at each checkpoint. Commit.

### Task 5.5: E2E (cloud-only, hardware-gated)
- [ ] **Step 1: Write E2E test** - N=2 squad, `group_size=2`, SCRATCH; then a `BRANCH` from a closed segment. Verify harvest + advantage + one weight update. Gate with `pytest.mark.skipif(not GPU, reason="multi-node hardware required")`.
- [ ] **Step 2-4: SKIP** (no hardware in CI; document in MR).
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): e2e squad+branch (cloud-only, hardware-gated)"`.

### Task 5.6: Pre-commit + final checkoff
- [ ] Run `pre-commit run --all-files`; fix issues.
- [ ] Run `python3 -m pytest customized_areal/tree_search/tests/ -v` for all touched modules.
- [ ] Check off all tasks.md items (`- [ ]` -> `- [x]`).
- [ ] Commit -> `git commit -m "chore(tree-search): pre-commit + tasks.md checkoff"`.

---

## Self-Review

**1. Spec coverage** (delta spec `v2-multi-agent-tree-search/spec.md` requirements):
- "Multi-agent squad rollout produces AssembledDag per task" -> Tasks 1.1-1.4 ✓ (reuse create_env_dispatch + get_dag; AReaL never mints)
- "AssembledDag resolves into multi-agent SuperNode" -> Tasks 2.1-2.2 ✓ (reuse assemble_from_refs; SuperNode-preserving)
- "BRANCH edge type with fork provenance" -> Tasks 3.1, 3.2 ✓
- "MCTS value backup across BRANCH edges" -> Tasks 3.3, 3.6 ✓
- "SCRATCH/BRANCH/MIXED bounded by max_group_size" -> Tasks 3.5, 2.4 ✓
- "Direct AssembledDag polling harvest" -> Task 1.1 (polling), 4.1 (staleness) ✓
- "Partial squad failure drops the task" -> Task 1.2 ✓
- N=1 parity -> Task 2.2 (parity assertion) ✓

**2. Reuse audit (no reimplemented components):** create_env_dispatch (existing), get_dag (existing), assemble_from_refs (existing), DataProxyTensorResolver/SessionRemover (existing), EnvDispatchBranchDriver (existing), TreeAdvantageComputer/MCTSTreeStore (existing). Genuinely new: MultiAgentEnvDispatchWorkflow (thin orchestrator), EdgeType.BRANCH+provenance, backup.py, Node.visit_count, grouped-workflow wiring.

**3. F-independence:** All phases F-independent (branching via EnvDispatchBranchDriver, not Sub-project F sandbox snapshot/fork). No F-gated tasks.

**4. Type consistency:** `arun_episode -> {"assembled_dag", "execution_dag"}` consistent in 1.1, 2.1. `branch_backup(parent, *, branch_return, discount, visit_count)` consistent in 3.3 tests + impl. `EdgeType.BRANCH` consistent in 3.1, 3.2.

**5. Constraint fixes vs. original plan:** original tasks.md had AReaL calling `start_session` (1.3) - FIXED (Multica owns it); per-trajectory callback harvest (1.5) - FIXED (polling); `InferenceServiceWorkflow` for multica path (4.1) - FIXED (`MultiAgentEnvDispatchWorkflow`); create-new `MulticaDagClient`/assembler (1.1/2.1/2.2) - FIXED (reuse existing).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-08-areal-v2-tree-search-branching.md`. Execution: subagent-driven-development (fresh subagent per task, two-stage review). ~22 executable tasks.

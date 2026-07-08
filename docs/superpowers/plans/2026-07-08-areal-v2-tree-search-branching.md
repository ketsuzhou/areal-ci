---
change: areal-v2-integration-and-tree-search-branching
design-doc: docs/superpowers/specs/2026-07-08-areal-v2-tree-search-branching-design.md
base-ref: c276e233cbd13424d4422cd0b48d6a7702c74785
---

# AReaL v2 Multi-Agent Integration + Tree-Search Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire multi-agent squad rollout (N sessions/task) and tree-search branching (`BRANCH` edge + MCTS backup) into AReaL's v2 online training path, so a pi-agent trains from Multica collaborative tasks.

**Architecture:** A new `MultiAgentEnvDispatchWorkflow` (base `RolloutWorkflow`) drives one task = N agents = N sessions and harvests an `AssembledDag` by direct polling. `TreeSearchGroupedRolloutWorkflow` (Approach A) stays group-level: `_result_to_nodes` + a multi-agent `SuperNode` builder consume the `AssembledDag`. Branching adds a `BRANCH` edge type with MCTS value backup; branch-from-checkpoint depends on Sub-project F.

**Tech Stack:** Python 3.12+ (AReaL), Go (Multica side), PyTorch, v2 inference_service (gateway/data_proxy/router/controller), OpenSpec.

## Global Constraints

- Python >= 3.12 (`backend/areal/pyproject.toml`); pre-commit: `pre-commit run --all-files` before commit.
- AReaL never calls `rl_session.start(agent_run_id=...)` on the multi-agent path; Multica owns `/rl/start_session` + `session_to_agent_run`.
- v2 mint-on-demand: no pre-grant, no capacity ratchet; `SessionStore.start_session` is unbounded.
- Staleness gates `rollout_batch` admission, not session minting.
- `BRANCH` backup = MCTS value propagation to parent checkpoint node (R1=A); each branch keeps its own advantage.
- Partial squad failure -> drop the task (no partial `SuperNode`).
- N=1 degenerates to the single-agent leaf `SuperNode` (parity).
- This change depends on `multica-v2-segment-dag-training` (unmerged, current branch) and Sub-project F (branching only). Build base is the current branch `multica-v2-segment-dag-training`, NOT `upstream/dev`.
- `agent_versions` rows are immutable; RLS stays on; no wildcard imports; no hardcoded secrets.

## Dependency Note (read before executing)

- **Phases 1, 2, 4, 5 (SCRATCH) are F-independent** and can execute now against the v2-segment-dag branch.
- **Phase 3 (branching) is F-gated**: `branch-from-checkpoint` requires Sub-project F's env-snapshot + issue-subtree fork. Tasks 3.2-3.6 block on F. Tasks 3.1 (EdgeType) and 3.4 (backup rule) can proceed; the fork call site (3.2) stubs F behind a `ForkableEnvironment` protocol until F lands.
- If F is far out, consider splitting Phase 3 into a separate change (see Step 4 of comet-build).

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `customized_areal/tree_search/agents/multi_agent_env_dispatch.py` | NEW base `RolloutWorkflow`: one task -> N sessions -> `AssembledDag` via polling | create |
| `customized_areal/tree_search/agents/multica_dag_client.py` | NEW: `create_env_dispatch` + `AssembledDag` polling client (activates the stored hook) | create |
| `customized_areal/tree_search/core/customized_grouped_workflow.py` | `_result_to_nodes` AssembledDag branch; multi-agent `SuperNode` builder; `multica_dag_client` wiring | modify |
| `customized_areal/tree_search/agents/execution_dag.py` | add `EdgeType.BRANCH`; `BRANCH` provenance fields | modify |
| `customized_areal/tree_search/dag/backup.py` | NEW: MCTS value backup across `BRANCH` edges (master-design Phase 3, unimplemented) | create |
| `customized_areal/tree_search/agents/__init__.py` | export `MultiAgentEnvDispatchWorkflow` | modify |
| tests under `customized_areal/tree_search/tests/` | unit tests for each component | create |

---

## Phase 1 - Multi-agent env-dispatch base workflow

### Task 1.1: `MulticaDagClient` polling + dispatch client

**Files:**
- Create: `customized_areal/tree_search/agents/multica_dag_client.py`
- Test: `customized_areal/tree_search/tests/test_multica_dag_client.py`

**Interfaces:**
- Produces: `class MulticaDagClient` with `async create_env_dispatch(*, mode, env_id, dispatch_type, agent_id, group_size, domain, issue, message) -> SweLegoSetup` and `async poll_assembled_dag(*, project_id, timeout, poll_interval) -> dict | None` (returns the `AssembledDag` dict or None on timeout).

- [ ] **Step 1: Write the failing test**

```python
# customized_areal/tree_search/tests/test_multica_dag_client.py
import asyncio
import pytest
from customized_areal.tree_search.agents.multica_dag_client import MulticaDagClient

class _FakeRouter:
    def __init__(self, dag_after=2):
        self.calls = 0
        self.dag_after = dag_after
        self.dag = {"segments": [], "edges": [], "session_to_agent_run": {}}
    async def get(self, url, headers=None, params=None):
        class R:
            def __init__(self, status, body): self.status_code=status; self._b=body
            def json(self): return self._b
        self.calls += 1
        if self.calls < self.dag_after:
            return R(202, {"status": "in_progress"})
        return R(200, self.dag)
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

@pytest.mark.asyncio
async def test_poll_returns_dag_after_in_progress():
    client = MulticaDagClient(base_url="http://multica", admin_api_key="k")
    client._session = _FakeRouter(dag_after=2)
    dag = await client.poll_assembled_dag(project_id="p1", timeout=5.0, poll_interval=0.0)
    assert dag is not None
    assert "segments" in dag

@pytest.mark.asyncio
async def test_poll_returns_none_on_timeout():
    client = MulticaDagClient(base_url="http://multica", admin_api_key="k")
    client._session = _FakeRouter(dag_after=100)  # never 200
    dag = await client.poll_assembled_dag(project_id="p1", timeout=0.05, poll_interval=0.01)
    assert dag is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# customized_areal/tree_search/agents/multica_dag_client.py
from __future__ import annotations
import asyncio, time
from typing import Any
import aiohttp
from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup

class MulticaDagClient:
    def __init__(self, *, base_url: str, admin_api_key: str):
        self.base_url = base_url.rstrip("/")
        self._admin_api_key = admin_api_key
        self._session: aiohttp.ClientSession | None = None

    async def create_env_dispatch(self, *, mode, env_id, dispatch_type, agent_id,
                                  group_size=1, domain=None, issue=None, message=None) -> SweLegoSetup:
        raise NotImplementedError  # wired in Task 1.3 against the real Multica endpoint

    async def poll_assembled_dag(self, *, project_id: str, timeout: float,
                                 poll_interval: float = 1.0) -> dict[str, Any] | None:
        url = f"{self.base_url}/api/v1/env-dispatch/{project_id}/dag"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            session = self._session or await self._get_session()
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status != 202:
                    return None
            await asyncio.sleep(poll_interval)
        return None

    async def _get_session(self):
        from areal.infra import workflow_context
        self._session = await workflow_context.get_aiohttp_session()
        return self._session
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/tests/test_multica_dag_client.py
git commit -m "feat(tree-search): add MulticaDagClient AssembledDag polling"
```

### Task 1.2: `MultiAgentEnvDispatchWorkflow` skeleton + N=1 parity

**Files:**
- Create: `customized_areal/tree_search/agents/multi_agent_env_dispatch.py`
- Modify: `customized_areal/tree_search/agents/__init__.py` (export)
- Test: `customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py`

**Interfaces:**
- Consumes: `MulticaDagClient.poll_assembled_dag` (Task 1.1).
- Produces: `class MultiAgentEnvDispatchWorkflow(RolloutWorkflow)` with `async arun_episode(engine, data) -> dict | None` returning `{"assembled_dag": ..., "tensors": ...}` or `None`.

- [ ] **Step 1: Write the failing test (N=1 polling path, fake client)**

```python
# customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py
import pytest
from customized_areal.tree_search.agents.multi_agent_env_dispatch import MultiAgentEnvDispatchWorkflow

class _FakeMultica:
    def __init__(self, dag): self._dag = dag; self.dispatched = False
    async def create_env_dispatch(self, **kw):
        self.dispatched = True
        from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup, SweLegoRollout
        return SweLegoSetup(rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")])
    async def poll_assembled_dag(self, *, project_id, timeout, poll_interval=1.0):
        return self._dag

@pytest.mark.asyncio
async def test_arun_episode_returns_assembled_dag():
    dag = {"segments": [{"segment_id": "s1", "agent_run_id": "r1", "issue_id": "i1",
                         "trajectory_id": "t1", "tensor_ref": {}, "closing_event": "completion",
                         "env_snapshot": {}}],
           "edges": [], "session_to_agent_run": {"sess1": "r1"}}
    wf = MultiAgentEnvDispatchWorkflow(multica=_FakeMultica(dag), gateway_addr="http://gw",
                                       admin_api_key="k", poll_timeout=5.0)
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert out["assembled_dag"]["session_to_agent_run"] == {"sess1": "r1"}

@pytest.mark.asyncio
async def test_arun_episode_returns_none_on_poll_timeout():
    class _Never:
        async def create_env_dispatch(self, **kw):
            from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup, SweLegoRollout
            return SweLegoSetup(rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")])
        async def poll_assembled_dag(self, *, project_id, timeout, poll_interval=1.0):
            return None
    wf = MultiAgentEnvDispatchWorkflow(multica=_Never(), gateway_addr="http://gw",
                                       admin_api_key="k", poll_timeout=0.01)
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# customized_areal/tree_search/agents/multi_agent_env_dispatch.py
from __future__ import annotations
import logging
from typing import Any, Protocol
from areal.api.workflow_api import RolloutWorkflow

logger = logging.getLogger("MultiAgentEnvDispatchWorkflow")

class _Multica(Protocol):
    async def create_env_dispatch(self, **kw) -> Any: ...
    async def poll_assembled_dag(self, *, project_id: str, timeout: float, poll_interval: float = 1.0) -> dict | None: ...

class MultiAgentEnvDispatchWorkflow(RolloutWorkflow):
    """One arun_episode = one Multica task = N agents = N sessions -> AssembledDag.

    Multica owns /rl/start_session(group_size=N) and session_to_agent_run.
    AReaL polls the env-dispatch endpoint for the AssembledDag (harvest=A).
    """
    def __init__(self, *, multica: _Multica, gateway_addr: str, admin_api_key: str,
                 poll_timeout: float = 3600.0, poll_interval: float = 1.0,
                 group_size: int = 1, base_env_id: str = ""):
        self._multica = multica
        self.gateway_addr = gateway_addr.rstrip("/")
        self._admin_api_key = admin_api_key
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.group_size = group_size
        self.base_env_id = base_env_id

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        setup = await self._multica.create_env_dispatch(
            mode="scratch", env_id=self.base_env_id, dispatch_type="message",
            agent_id=data.get("agent_id", ""), group_size=self.group_size,
            domain="multica", message=data.get("message"))
        if not setup.rollouts:
            return None
        # N agents run online on Multica's side (Multica owns start_session).
        project_id = setup.rollouts[0].project_id
        dag = await self._multica.poll_assembled_dag(
            project_id=project_id, timeout=self.poll_timeout, poll_interval=self.poll_interval)
        if dag is None:
            logger.warning("AssembledDag poll timed out for project %s; rejecting", project_id)
            return None
        return {"assembled_dag": dag, "tensors": {}}  # tensor resolution wired in Task 1.4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Export + commit**

```python
# add to customized_areal/tree_search/agents/__init__.py
from customized_areal.tree_search.agents.multi_agent_env_dispatch import MultiAgentEnvDispatchWorkflow
```

```bash
git add customized_areal/tree_search/agents/multi_agent_env_dispatch.py customized_areal/tree_search/agents/__init__.py customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py
git commit -m "feat(tree-search): add MultiAgentEnvDispatchWorkflow (AssembledDag polling harvest)"
```

### Task 1.3: `create_env_dispatch` wiring against Multica endpoint

**Files:**
- Modify: `customized_areal/tree_search/agents/multica_dag_client.py` (implement `create_env_dispatch`)
- Test: extend `test_multica_dag_client.py`

**Interfaces:**
- Produces: `create_env_dispatch` POSTs `/api/v1/env-dispatch`, returns `SweLegoSetup` parsed from JSON.

- [ ] **Step 1: Write the failing test**

```python
# append to test_multica_dag_client.py
@pytest.mark.asyncio
async def test_create_env_dispatch_posts_and_parses():
    class _FakePost:
        async def post(self, url, json=None, headers=None):
            class R:
                status_code = 200
                async def json(self): return {"rollouts": [{"agent_run_id": "r1", "env_id": "e1", "project_id": "p1"}]}
            return R()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
    client = MulticaDagClient(base_url="http://multica", admin_api_key="k")
    client._session = _FakePost()
    setup = await client.create_env_dispatch(mode="scratch", env_id="e0", dispatch_type="message",
                                             agent_id="a1", group_size=2, domain="multica", message="hi")
    assert setup.rollouts[0].agent_run_id == "r1"
```

- [ ] **Step 2: Run test to verify it fails** -> `uv run pytest ... -k create_env_dispatch -v` -> FAIL (NotImplementedError)

- [ ] **Step 3: Implement**

```python
# replace create_env_dispatch in multica_dag_client.py
    async def create_env_dispatch(self, *, mode, env_id, dispatch_type, agent_id,
                                  group_size=1, domain=None, issue=None, message=None) -> SweLegoSetup:
        url = f"{self.base_url}/api/v1/env-dispatch"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload = {"mode": mode, "env_id": env_id, "dispatch_type": dispatch_type,
                   "agent_id": agent_id, "group_size": group_size, "domain": domain,
                   "issue": issue, "message": message}
        session = self._session or await self._get_session()
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup, SweLegoRollout
        rollouts = [SweLegoRollout(**r) for r in data.get("rollouts", [])]
        return SweLegoSetup(rollouts=rollouts)
```

- [ ] **Step 4: Run test** -> PASS
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): wire MulticaDagClient.create_env_dispatch"`

### Task 1.4: Tensor-ref resolution via v2 `/data/*`

**Files:**
- Modify: `customized_areal/tree_search/agents/multi_agent_env_dispatch.py` (resolve `tensor_ref`s)
- Test: `test_multi_agent_env_dispatch.py` (add resolved-tensors assertion)

**Interfaces:**
- Consumes: `AssembledDag.segments[*].tensor_ref` (v2-segment-dag `RTensor` refs).
- Produces: `arun_episode` returns `{"assembled_dag": ..., "tensors": {segment_id: tokens/logprobs}}`.

- [ ] **Step 1: Write failing test** - assert `tensors` dict keyed by `segment_id` is populated from a fake `/data/batch` response.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - add `_resolve_tensor_refs(self, segments) -> dict` that POSTs `/data/batch` to the gateway with the `tensor_ref` shard ids and deserializes via `areal.infra.rpc.serialization.deserialize_value`; call it in `arun_episode` before returning.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): resolve AssembledDag tensor refs via v2 /data/batch"`

### Task 1.5: Partial squad failure -> drop

**Files:**
- Modify: `multi_agent_env_dispatch.py` (validate `AssembledDag` covers all N rollouts without gaps)
- Test: add a test where one `agent_run_id` is missing from `session_to_agent_run` -> returns `None`.

- [ ] **Step 1: Write failing test** - `AssembledDag` with 2 rollouts but only 1 in `session_to_agent_run` -> `arun_episode` returns `None`, no partial output.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - in `arun_episode`, after poll: `covered = set(dag["session_to_agent_run"].values()); expected = {r.agent_run_id for r in setup.rollouts}; if not expected.issubset(covered): return None`.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): drop task on partial squad failure"`

---

## Phase 2 - `TreeSearchGroupedRolloutWorkflow` multi-agent wiring (Approach A)

### Task 2.1: `_result_to_nodes` AssembledDag branch

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py:929` (`_result_to_nodes`)
- Test: `customized_areal/tree_search/tests/test_result_to_nodes_multica.py`

**Interfaces:**
- Consumes: `MultiAgentEnvDispatchWorkflow.arun_episode` output `{"assembled_dag": ..., "tensors": ...}`.
- Produces: per-segment `Node`s with one `episode_id` per `agent_run_id`.

- [ ] **Step 1: Write failing test**

```python
# test_result_to_nodes_multica.py
from customized_areal.tree_search.core.customized_grouped_workflow import TreeSearchGroupedRolloutWorkflow

def _wf_for_test():
    # construct with minimal required args (group_size=1, ...) - helper
    ...

def test_assembled_dag_yields_per_agent_run_nodes():
    result = {"assembled_dag": {
        "segments": [
            {"segment_id": "s1", "agent_run_id": "r1", "issue_id": "i1", "trajectory_id": "t1",
             "tensor_ref": {}, "closing_event": "completion", "env_snapshot": {}},
            {"segment_id": "s2", "agent_run_id": "r2", "issue_id": "i1", "trajectory_id": "t2",
             "tensor_ref": {}, "closing_event": "completion", "env_snapshot": {}},
        ],
        "edges": [], "session_to_agent_run": {"sess1": "r1", "sess2": "r2"}},
        "tensors": {}}
    nodes = TreeSearchGroupedRolloutWorkflow._result_to_nodes(None, result, "q1", 0)
    assert nodes is not None
    episode_ids = {n.episode_id for n in nodes}
    assert len(episode_ids) == 2  # one per agent_run_id
```

- [ ] **Step 2: Run -> FAIL** (current `_result_to_nodes` returns None for the dict shape)
- [ ] **Step 3: Implement** - add an `isinstance(result, dict) and "assembled_dag" in result` branch in `_result_to_nodes`: for each segment, build turn-`Node`s from `tensors[segment_id]` (via `interactions_dict_to_nodes`), stamp `episode_id = f"{query_id}_{group_idx}_{agent_run_id}"`, `query_id`, `turn_idx`. Return all `Node`s.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): _result_to_nodes consumes AssembledDag (per-agent-run episode_id)"`

### Task 2.2: Multi-agent `SuperNode` builder (generalize `_wrap_leaf_super`)

**Files:**
- Modify: `customized_grouped_workflow.py:91` (`_wrap_leaf_super` -> `_build_super_node`)
- Test: assert N=1 still produces a leaf `SuperNode` (parity); N=2 produces a `SuperNode` with 2 agent-runs + edges.

- [ ] **Step 1: Write failing test** - N=2 `AssembledDag` with a `delegation` edge -> `SuperNode` with 2 runs and the edge in `ExecutionDAG`.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - add `_build_super_node(nodes_by_agent_run, edges)` that creates one `SuperNode` per agent-run (or one combined `SuperNode` with all nodes + an `ExecutionDAG` of edges). Keep `_wrap_leaf_super` as the N=1 delegate. Wire `_result_to_nodes` to call `_build_super_node` for the AssembledDag branch.
- [ ] **Step 4: Run -> PASS** (incl. N=1 parity test from existing suite)
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): multi-agent SuperNode builder (leaf = N=1)"`

### Task 2.3: Activate `multica_dag_client` hook + wire `self.workflow`

**Files:**
- Modify: `customized_grouped_workflow.py:682/852` (use the hook), constructor wiring
- Test: construct `TreeSearchGroupedRolloutWorkflow` with a `MultiAgentEnvDispatchWorkflow` as `workflow` + `multica_dag_client`; assert it's used.

- [ ] **Step 1: Write failing test** - grouped workflow's `self.workflow` is the `MultiAgentEnvDispatchWorkflow`; `self._multica_dag_client` is non-None.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - in `__init__`, store `self._multica_dag_client`; when `multica_dag_enabled`, the caller passes a `MultiAgentEnvDispatchWorkflow` as `workflow` (constructed with the `multica_dag_client`). Document the construction contract in the docstring.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): activate multica_dag_client hook + base workflow wiring"`

### Task 2.4: `_finalize_episode` per-agent-run credit

**Files:**
- Modify: `customized_grouped_workflow.py:1754` (`_finalize_episode`)
- Test: a multi-agent `SuperNode` with a fan-in `completion` edge -> credit assigned at the join.

- [ ] **Step 1: Write failing test** - 2 agent-runs, 1 `completion` edge -> per-run credit at the fan-in join.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - extend `_finalize_episode` to call the existing `credit.py` (master-design Task 8) at fan-in joins over the multi-agent `SuperNode`; `TreeAdvantageComputer` consumes per-node credit.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): per-agent-run credit at fan-in joins"`

### Task 2.5: `group_size=M` parallel multi-agent rollouts

**Files:**
- Modify: `customized_grouped_workflow.py:1496` (verify gather shape unchanged)
- Test: `group_size=2` -> 2 parallel `arun_episode` calls -> 2 multi-agent `SuperNode`s.

- [ ] **Step 1: Write failing test** - `group_size=2`, fake base workflow returns 2 distinct `AssembledDag`s -> 2 `SuperNode`s aggregated.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - confirm `_arun_episode_fixed` gather produces M `SuperNode`s; fix any cached-episode path (`load_untrained_episodes`) that assumes single-agent leaf shape.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): group_size=M parallel multi-agent rollouts"`

---

## Phase 3 - Tree-search branching (F-gated)

### Task 3.1: `EdgeType.BRANCH` + provenance fields (F-independent)

**Files:**
- Modify: `customized_areal/tree_search/agents/execution_dag.py:30` (add `BRANCH = "branch"`)
- Modify: `Edge`/`SuperNode` to carry `branch_from_segment_id`, `branch_from_checkpoint_id`
- Test: `customized_areal/tree_search/tests/test_branch_edge.py`

- [ ] **Step 1: Write failing test** - `EdgeType.BRANCH == "branch"`; an edge with `branch_from_segment_id` round-trips through `to_records`/`from_records`.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - add `BRANCH` to `EdgeType`; add optional `branch_from_segment_id`/`branch_from_checkpoint_id` to `Edge`; update `to_records`/`from_records` serialization.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): add BRANCH edge type + provenance fields"`

### Task 3.2: Branch-from-segment-checkpoint via F (F-GATED)

**Files:**
- Modify: `customized_grouped_workflow.py:233` (`select_branch_candidate`), `multi_agent_env_dispatch.py`
- Depends on: Sub-project F `ForkableEnvironment` + issue-subtree fork.

- [ ] **Step 1: Write failing test** (with a fake `ForkableEnvironment`) - selecting a branch candidate forks the env+issue subtree and opens a new `start_session`.
- [ ] **Step 2: Run -> FAIL** (no fork call)
- [ ] **Step 3: Implement** - when `select_branch_candidate` returns a node, call F's fork primitive (behind a `ForkableEnvironment` protocol) to restore env snapshot + issue subtree, then `start_session(group_size=k)` for the branched agent(s). **If F is unavailable, this task blocks** - leave the call site behind the protocol and skip the test with `pytest.skip("Sub-project F fork primitive not available")`.
- [ ] **Step 4: Run -> PASS** (or skip if F absent)
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): branch-from-segment-checkpoint (F-gated)"`

### Task 3.3: `BRANCH` edge emission in `AssembledDag` (Multica side, F-GATED)

**Files:**
- Multica Go: `internal/service/interaction_dag.go` (or successor) - emit `branch` edges with `branch_from_segment_id`/`branch_from_checkpoint_id`.
- This is a Multica-side change; coordinate with the multica repo. AReaL-side: extend the `AssembledDag` parser to accept `branch` edges (covered by Task 3.1).

- [ ] **Step 1: Write failing test** - AReaL parses an `AssembledDag` with a `branch` edge -> `ExecutionDAG` has the edge with provenance.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - AReaL parser maps `branch` edge type to `EdgeType.BRANCH` (Task 3.1 already added it). Multica emission is a separate PR.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): parse BRANCH edges in AssembledDag"`

### Task 3.4: MCTS value backup across `BRANCH` edges (F-independent)

**Files:**
- Create: `customized_areal/tree_search/dag/backup.py` (master-design Phase 3, unimplemented)
- Test: `customized_areal/tree_search/tests/test_branch_backup.py`

**Interfaces:**
- Produces: `def branch_backup(parent_checkpoint_node, branch_return, discount, visit_count) -> None` updating the parent's value estimate.

- [ ] **Step 1: Write failing test**

```python
# test_branch_backup.py
from customized_areal.tree_search.dag.backup import branch_backup
from customized_areal.tree_search.core.tree_store import Node

def test_branch_return_updates_parent_value():
    parent = Node(node_id="ckpt1", value=0.0, visit_count=0)
    branch_backup(parent, branch_return=1.0, discount=0.9, visit_count=1)
    assert parent.value == 0.9  # discounted branch return backs up
    assert parent.visit_count == 1

def test_multiple_branches_aggregate():
    parent = Node(node_id="ckpt1", value=0.0, visit_count=0)
    branch_backup(parent, branch_return=1.0, discount=1.0, visit_count=1)
    branch_backup(parent, branch_return=0.0, discount=1.0, visit_count=1)
    assert parent.value == 0.5  # mean over 2 visits
    assert parent.visit_count == 2
```

- [ ] **Step 2: Run -> FAIL** (module not found)
- [ ] **Step 3: Implement**

```python
# customized_areal/tree_search/dag/backup.py
from __future__ import annotations
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

- [ ] **Step 4: Run -> PASS** (2 tests)
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): MCTS value backup across BRANCH edges"`

### Task 3.5: `max_group_size` bound + circuit breaker for branches

**Files:**
- Modify: `customized_grouped_workflow.py` (`choose_sample_source`, `_arun_episode_dynamic`)
- Test: total branches <= `max_group_size - initial_group_size`; consecutive-failure breaker stops branching.

- [ ] **Step 1: Write failing test** - `max_group_size=4`, `initial_group_size=2` -> at most 2 branches.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - enforce the bound in `choose_sample_source`; the existing consecutive-failure circuit breaker (`:28`) applies.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): bound branches by max_group_size + circuit breaker"`

### Task 3.6: Branch cleanup (F-GATED)

**Files:**
- Modify: `customized_grouped_workflow.py` (`_cleanup_branch`)
- Depends on: F fork primitive + v2 `remove_session`.

- [ ] **Step 1: Write failing test** - branch cleanup releases the branched session (`remove_session`) + forked env/issue subtree.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - extend `_cleanup_branch` to call `remove_session` for the branched session + F's fork cleanup. Skip if F absent.
- [ ] **Step 4: Run -> PASS** (or skip)
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): branch cleanup (F-gated)"`

---

## Phase 4 - v2 online training integration

### Task 4.1: Construct `MultiAgentEnvDispatchWorkflow` for the v2/multica config path

**Files:**
- Modify: the config/workflow construction site that builds `TreeSearchGroupedRolloutWorkflow` (find via `grep -rn "TreeSearchGroupedRolloutWorkflow(" customized_areal/`)
- Test: the v2/multica config constructs a `MultiAgentEnvDispatchWorkflow` as the wrapped `workflow`.

- [ ] **Step 1: Write failing test** - config flag `multica_dag_enabled=True` -> grouped workflow's `self.workflow` is `MultiAgentEnvDispatchWorkflow`.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - in the workflow factory, when `multica_dag_enabled`, build `MultiAgentEnvDispatchWorkflow(multica=MulticaDagClient(...), ...)` and pass as `workflow` + `multica_dag_client`.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "feat(tree-search): construct MultiAgentEnvDispatchWorkflow for v2/multica config"`

### Task 4.2: Staleness admission stays at batch level (verification)

**Files:**
- Test: `customized_areal/tree_search/tests/test_staleness_batch_admission.py`

- [ ] **Step 1: Write failing test** - `rollout_batch(M)` admits M `arun_episode`s; `start_session` is not capacity-gated; excess `AssembledDag`s hold at 202.
- [ ] **Step 2: Run -> FAIL** (or pass if behavior already holds - assert the contract)
- [ ] **Step 3: Implement/verify** - confirm `controller.rollout_batch` gates via `staleness_manager`; no session-level gate added. Document in a code comment.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): staleness gates batch admission, not minting"`

### Task 4.3: Cross-step staleness guard (R2)

**Files:**
- Test: a branch minted in batch N is harvested before `set_version(N+1)`.

- [ ] **Step 1: Write failing test** - `rollout_batch` awaits all `arun_episode`s (incl. branch polling) before `set_version`; no trajectory leaks across steps.
- [ ] **Step 2: Run -> FAIL** (or pass if `rollout_batch` already awaits)
- [ ] **Step 3: Implement/verify** - assert `rollout_batch` does not call `set_version` until all submitted `arun_episode`s resolve. Add an assertion/guard if absent.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): no cross-step staleness leak (R2)"`

### Task 4.4: Session lifecycle (mint on demand, harvest with remove_session)

**Files:**
- Test: M x N sessions minted via `start_session(group_size=N)`; harvested via `/export_trajectories(remove_session=True)`.

- [ ] **Step 1: Write failing test** - N sessions minted; each harvested once with `remove_session=True`.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement/verify** - `MultiAgentEnvDispatchWorkflow` does not mint (Multica does); AReaL harvests via export. Confirm the harvest path in `_finalize_episode`/training calls export with `remove_session=True`.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): session lifecycle (mint on demand, harvest+remove)"`

---

## Phase 5 - Tests and end-to-end

### Task 5.1: Multi-agent base workflow unit suite (consolidation)

- [ ] Consolidate Tasks 1.1-1.5 tests into a coherent suite; add a multi-level branch-tree backup test (Task 3.4 extended). Run `uv run pytest customized_areal/tree_search/tests/test_multi_agent_env_dispatch.py customized_areal/tree_search/tests/test_branch_backup.py -v`. Commit.

### Task 5.2: Integration test - M x N parallel sessions on v2 gateway

**Files:**
- Test: `customized_areal/tree_search/tests/test_integration_mn_sessions.py`

- [ ] **Step 1: Write failing test** - M=2, N=2 -> 4 sessions on a fake v2 gateway; router routes by `session_key`; no `429` from `SessionStore`.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - fake v2 gateway (`SessionStore` in-process) + fake router; drive `MultiAgentEnvDispatchWorkflow` x M.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): M×N parallel sessions, router routing, no 429"`

### Task 5.3: Staleness backpressure test

- [ ] **Step 1: Write failing test** - Multica mints faster than trainer admits -> `AssembledDag`s hold at 202; trainer drains in later batches.
- [ ] **Step 2: Run -> FAIL**
- [ ] **Step 3: Implement** - fake controller with bounded `max_concurrent_rollouts`; assert 202-holding + drain.
- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): staleness backpressure (AssembledDag 202-holding)"`

### Task 5.4: E2E (cloud-only, hardware-gated)

**Files:**
- Test: `customized_areal/tree_search/tests/test_e2e_multica_tree_search.py`

- [ ] **Step 1: Write E2E test** - N=2 squad, `group_size=2`, SCRATCH; then a `BRANCH` from a closed segment. Verify harvest + advantage + one weight update. Gate with `pytest.mark.skipif(not GPU, reason="multi-node hardware required")`.
- [ ] **Step 2: Run -> SKIP** (no hardware in CI)
- [ ] **Step 3: Implement** - the full path against a real Multica + cloud sandbox stack.
- [ ] **Step 4: Run -> SKIP** (document in MR)
- [ ] **Step 5: Commit** -> `git commit -m "test(tree-search): e2e squad+branch (cloud-only, hardware-gated)"`

### Task 5.5: Pre-commit + final checkoff

- [ ] Run `pre-commit run --all-files`; fix issues.
- [ ] Run `uv run pytest customized_areal/tree_search/tests/ -v` for all touched modules.
- [ ] Check off all tasks.md items (`- [ ]` -> `- [x]`).
- [ ] Commit -> `git commit -m "chore(tree-search): pre-commit + tasks.md checkoff"`

---

## Self-Review

**1. Spec coverage** (delta spec `v2-multi-agent-tree-search/spec.md` requirements):
- "Multi-agent squad rollout produces AssembledDag per task" -> Tasks 1.1-1.5 ✓
- "AssembledDag resolves into multi-agent SuperNode" -> Tasks 2.1-2.2 ✓
- "BRANCH edge type with fork provenance" -> Tasks 3.1, 3.3 ✓
- "MCTS value backup across BRANCH edges" -> Task 3.4 ✓
- "SCRATCH/BRANCH/MIXED bounded by max_group_size" -> Tasks 3.5, 2.5 ✓
- "Direct AssembledDag polling harvest" -> Task 1.1 (polling), 4.2 (staleness) ✓
- "Partial squad failure drops the task" -> Task 1.5 ✓
- N=1 parity -> Task 2.2 (parity assertion) ✓

**2. Placeholder scan:** No TBD/TODO. F-gated tasks (3.2, 3.3-multica, 3.6) explicitly mark F-dependence and skip-if-absent - not placeholders, but honest dependency gates. Tensor-resolution (1.4) and finalize-credit (2.4) steps describe the approach with the consuming primitive named (`interactions_dict_to_nodes`, `credit.py`) - acceptable since those are existing modules, not undefined refs.

**3. Type consistency:** `MultiAgentEnvDispatchWorkflow.arun_episode -> {"assembled_dag", "tensors"}` used consistently in Tasks 1.2, 2.1. `branch_backup(parent, branch_return, discount, visit_count)` signature consistent in 3.4 tests + impl. `EdgeType.BRANCH` consistent in 3.1, 3.3.

**4. Scope:** One cohesive change; branching is F-gated but not a separate subsystem. If F is far out, split Phase 3 into a new change (comet-build Step 4 / 50% threshold).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-08-areal-v2-tree-search-branching.md`. Execution options (chosen at comet-build Step 3):
1. **Subagent-Driven (recommended)** - fresh subagent per task, two-stage review (task count = ~24 executable tasks >= 3).
2. **Inline Execution** - executing-plans, batch with checkpoints.

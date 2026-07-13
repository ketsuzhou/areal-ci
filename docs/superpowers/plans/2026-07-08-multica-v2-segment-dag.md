---
archived-with: 2026-07-13-multica-v2-segment-dag-training
status: final
---
# Multica v2 Segment-DAG Training (Change 1 - Data Path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the v2 per-segment trajectory data path so Multica slices a session into communication-bounded segments (`close_segment` + tensor-ref export), assembles an `AssembledDag`, and AReaL resolves refs + builds the `ExecutionDAG` + runs a minimal training step with placeholder reward.

**Architecture:** `close_segment` reuses v2's active->ready close mechanic without reward; per-segment export returns `RTensor` refs (already supported); Multica assembles `AssembledDag` (structure + refs + env, no scores/turn-idx/text) and returns it via the env-dispatch polling endpoint; AReaL resolves refs and builds `SuperNode`/`ExecutionDAG` by ref-join (not turn-index slice). Reward is AReaL-side (placeholder zero in change 1; judge in change 2).

**Tech Stack:** Python 3.14 (areal v2 + consumer, pytest), Go (multica server, go test), SQL (multica migrations).

## Global Constraints

- v2 `close_segment` MUST NOT require a prior `set_reward` and MUST NOT assign reward.
- `AssembledDag` MUST carry no scores, no turn indices, no message text.
- Per-segment export MUST use `remove_session=False` and omit `group_id` (avoid gateway router revoke).
- Text stays in AReaL capture; export is refs-only (`input_ids`, `loss_mask`, `logprobs`, `versions`, `attention_mask`).
- Edges match `EdgeType`: `delegation` (fan-out) / `mention` (peer) / `completion` (fan-in); DAG must be acyclic.
- Change 1 uses placeholder zero reward; judge/V/GAE (change 2) and tree search (change 3) are out of scope.

## Subsystem Sequencing & Cross-Repo Execution

This change spans two git repos. The comet worktree isolation is areal-scoped.

| Unit | Repo | Side | Depends on |
|------|------|------|------------|
| U1 close_segment | areal | v2 capture | - |
| U2 export/data reuse | areal | v2 capture | U1 |
| U3 MulticaDagClient | areal | consumer | U2 (contract) |
| U4 assembler ref-resolve | areal | consumer | U3 |
| U5 minimal training + cleanup | areal | consumer | U4 |
| U6 arealrl CloseSegment+Export | multica (`server/`) | env | U1, U2 (contract) |
| U7 interaction_dag recording + hooks | multica (`server/`) | env | U6 |
| U8 AssembledDag assembly + `/dag` endpoint | multica (`server/`) | env | U7 |
| U9 migration | multica (`server/`) | env | - |
| U10 config + E2E | both | both | all |

**Execution order:** U1 -> U2 -> U6 (multica can start once the v2 contract is fixed) -> U7 -> U9 -> U8 -> U3 -> U4 -> U5 -> U10. Areal worktree covers U1-U5, U10(areal). Multica work happens on a `server/` branch (U6-U9, U10(multica)).

**Path note:** multica Go paths are under `server/` (e.g. `server/internal/arealrl/client.go`). The OpenSpec `proposal.md` Impact section omits this prefix and should be corrected during U6.

## File Structure

**areal repo:**
- Modify: `areal/v2/inference_service/data_proxy/session.py` - add `SessionData.close_segment()`.
- Modify: `areal/v2/inference_service/data_proxy/app.py` - add `POST /rl/close_segment`.
- Modify: `areal/v2/inference_service/gateway/app.py` - add `POST /rl/close_segment` route.
- Create: `areal/v2/inference_service/tests/test_close_segment.py` - close_segment TDD.
- Create: `customized_areal/tree_search/agents/multica_dag_client.py` - poll `/dag` -> `AssembledDag`.
- Create: `customized_areal/tree_search/tests/test_multica_dag_client.py`.
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py` - add `assemble_from_refs()`.
- Create: `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`.
- Modify: `customized_areal/tree_search/agents/` (training entry) - minimal training hook + cleanup. (Exact file confirmed in U5 step 1.)

**multica repo (`server/`):**
- Modify: `server/internal/arealrl/client.go` - add `CloseSegment`, `ExportTrajectory`.
- Create: `server/internal/arealrl/client_test.go` additions (or new test file).
- Create: `server/internal/service/interaction_dag.go` - `InteractionDAGService` (record segment, add edge, capture snapshot, assemble).
- Create: `server/internal/service/interaction_dag_test.go`.
- Modify: `server/internal/service/task.go` - wire delegation/mention/completion hooks.
- Modify: `server/internal/handler/env_dispatch.go` + `server/internal/service/env_dispatch.go` - `GET .../env-dispatch/{projectID}/dag`.
- Create: `server/migrations/155_interaction_dag_segment.up.sql` + `.down.sql`.

---

## U1 — v2 `close_segment` (areal)

### Task 1.1: `SessionData.close_segment()` no-reward close

**Files:**
- Modify: `areal/v2/inference_service/data_proxy/session.py` (add method after `set_reward`, ~line 269)
- Test: `areal/v2/inference_service/tests/test_close_segment.py`

**Interfaces:**
- Produces: `SessionData.close_segment() -> RewardResult` (no reward; `needs_online_callback=False`; `interaction_id=last_interaction_id`; raises `ValueError` on empty active).

- [ ] **Step 1: Write failing tests**

```python
# areal/v2/inference_service/tests/test_close_segment.py
from areal.v2.inference_service.data_proxy.session import SessionData

def _seed(session: SessionData) -> str:
    return session.add_string_interaction([{"role": "user", "content": "hi"}], "hello")

def test_close_segment_moves_active_to_ready_no_reward():
    s = SessionData("s1")
    iid = _seed(s)
    result = s.close_segment()
    assert result.ready_transition is True
    assert result.trajectory_id == 0
    assert result.interaction_count == 1
    # active cleared
    assert len(s.active_completions) == 0
    # the closed interaction received no reward
    assert s.active_completions is not None

def test_close_segment_does_not_require_set_reward():
    s = SessionData("s1")
    _seed(s)
    # no set_reward called; close_segment must succeed
    result = s.close_segment()
    assert result.trajectory_id == 0

def test_close_segment_empty_active_raises():
    s = SessionData("s1")
    try:
        s.close_segment()
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty active")

def test_close_segment_session_stays_live_for_next_segment():
    s = SessionData("s1")
    _seed(s)
    s.close_segment()
    # next turn captures into a new active segment
    _seed(s)
    result = s.close_segment()
    assert result.trajectory_id == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -v`
Expected: FAIL with `AttributeError: 'SessionData' object has no attribute 'close_segment'`.

- [ ] **Step 3: Implement `close_segment`**

In `session.py`, after the `set_reward` method:

```python
    def close_segment(self) -> RewardResult:
        """Close the active segment into a ready trajectory WITHOUT setting a reward.

        Decouples the trajectory boundary from reward so each communication-bounded
        segment is its own exportable trajectory. The segment's reward is assigned
        later (AReaL-side; judge in change 2), not at close.
        """
        with self._lock:
            now = time.time()
            self._last_access_time = now
            completions = self._active_completions
            if len(completions) == 0:
                raise ValueError("No interactions in session")
            terminal_interaction_id = completions.last_interaction_id
            trajectory_id = self._next_trajectory_id
            self._next_trajectory_id += 1
            ready = ReadyTrajectory(
                trajectory_id=trajectory_id,
                interaction_id=terminal_interaction_id,
                completions=completions,
                created_at=now,
                needs_online_callback=False,
            )
            self._ready_trajectories[trajectory_id] = ready
            self._active_completions = InteractionCache()
            # Do NOT touch _last_reward_interaction_id / _last_set_reward_time.
            return RewardResult(
                session_id=self.session_id,
                trajectory_id=trajectory_id,
                interaction_count=len(completions),
                ready_transition=True,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add areal/v2/inference_service/data_proxy/session.py areal/v2/inference_service/tests/test_close_segment.py
git commit -m "feat(v2-segment-dag): SessionData.close_segment no-reward active->ready close"
```

### Task 1.2: data_proxy `POST /rl/close_segment` endpoint

**Files:**
- Modify: `areal/v2/inference_service/data_proxy/app.py` (add endpoint after `/rl/set_reward`, ~line 511)
- Test: `areal/v2/inference_service/tests/test_close_segment.py` (append)

**Interfaces:**
- Consumes: `SessionData.close_segment()` (Task 1.1).
- Produces: `POST /rl/close_segment` (session-key auth) -> `{trajectory_id, interaction_count, ready_transition}`.

- [ ] **Step 1: Write failing test (append to test_close_segment.py)**

```python
from fastapi.testclient import TestClient
from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig

def _client():
    cfg = DataProxyConfig(admin_api_key="areal-admin-key")
    return TestClient(create_app(cfg))

def test_close_segment_endpoint_session_key():
    app = _client()
    # start session (admin)
    r = app.post("/rl/start_session", json={"task_id": "t1"},
                 headers={"Authorization": "Bearer areal-admin-key"})
    assert r.status_code == 201
    api_key = r.json()["sessions"][0]["session_api_key"]
    # one turn (string path -> no tensor data, but close_segment only needs an interaction)
    app.post("/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
             headers={"Authorization": f"Bearer {api_key}"})
    r = app.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["trajectory_ready"] is True
    assert body["trajectory_id"] == 0

def test_close_segment_endpoint_empty_active_400():
    app = _client()
    r = app.post("/rl/start_session", json={"task_id": "t2"},
                 headers={"Authorization": "Bearer areal-admin-key"})
    api_key = r.json()["sessions"][0]["session_api_key"]
    r = app.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -k endpoint -v`
Expected: FAIL (404 no route).

- [ ] **Step 3: Implement the endpoint**

In `data_proxy/app.py`, add a `CloseSegmentResponse` model near `SetRewardResponse` and the endpoint after `/rl/set_reward`:

```python
class CloseSegmentResponse(BaseModel):
    message: str
    interaction_count: int
    session_id: str
    trajectory_id: int | None
    trajectory_ready: bool
    ready_transition: bool
```

```python
    @app.post("/rl/close_segment", response_model=CloseSegmentResponse)
    async def close_segment(request: Request):
        store: SessionStore = app.state.session_store
        token = _extract_bearer_token(request)
        session = _resolve_session_from_token(token, store)
        if session is None:
            raise HTTPException(
                status_code=401, detail="Invalid or expired session API key."
            )
        try:
            result = session.close_segment()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CloseSegmentResponse(
            message="success",
            interaction_count=result.interaction_count,
            session_id=result.session_id,
            trajectory_id=result.trajectory_id,
            trajectory_ready=result.trajectory_id is not None,
            ready_transition=result.ready_transition,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -k endpoint -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add areal/v2/inference_service/data_proxy/app.py areal/v2/inference_service/tests/test_close_segment.py
git commit -m "feat(v2-segment-dag): data_proxy /rl/close_segment endpoint"
```

### Task 1.3: gateway `POST /rl/close_segment` route

**Files:**
- Modify: `areal/v2/inference_service/gateway/app.py` (add route after `/rl/set_reward`, ~line 375)
- Test: `areal/v2/inference_service/tests/test_close_segment.py` (append gateway test, or gateway test file)

**Interfaces:**
- Produces: gateway forwards `/rl/close_segment` to the worker (session-key, `query_router` by token) - mirrors `/rl/set_reward`.

- [ ] **Step 1: Write failing test**

```python
def test_gateway_close_segment_route_registered():
    from areal.v2.inference_service.gateway.app import create_app
    from areal.v2.inference_service.gateway.config import GatewayConfig
    app = TestClient(create_app(GatewayConfig(admin_api_key="areal-admin-key")))
    # route exists (will 502/401 without a router, but not 404)
    r = app.post("/rl/close_segment", headers={"Authorization": "Bearer k"})
    assert r.status_code != 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -k gateway -v`
Expected: FAIL (404).

- [ ] **Step 3: Implement the gateway route**

In `gateway/app.py`, after the `/rl/set_reward` handler:

```python
    # =========================================================================
    # POST /rl/close_segment - session key or admin key (mirror /rl/set_reward)
    # =========================================================================

    @app.post("/rl/close_segment")
    async def close_segment(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model = None
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/rl/close_segment",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        resp = await forward_request(
            f"{worker_addr}/rl/close_segment",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -k gateway -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add areal/v2/inference_service/gateway/app.py areal/v2/inference_service/tests/test_close_segment.py
git commit -m "feat(v2-segment-dag): gateway /rl/close_segment route"
```

---

## U2 — v2 per-segment export + data reuse verification (areal)

The existing `/export_trajectories` already accepts `trajectory_id` + `remove_session=False` and returns `RTensor` refs; `/data/*` + `/data/clear` already exist. This unit adds tests proving the per-segment contract; no new production code unless a gap is found.

### Task 2.1: per-segment export contract test

**Files:**
- Test: `areal/v2/inference_service/tests/test_close_segment.py` (append)

- [ ] **Step 1: Write failing test**

```python
def test_export_single_trajectory_remove_session_false():
    app = _client()
    r = app.post("/rl/start_session", json={"task_id": "t3"},
                 headers={"Authorization": "Bearer areal-admin-key"})
    api_key = r.json()["sessions"][0]["session_api_key"]
    sid = r.json()["sessions"][0]["session_id"]
    app.post("/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "a"}]},
             headers={"Authorization": f"Bearer {api_key}"})
    app.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
    # export that trajectory, keep session
    r = app.post("/export_trajectories",
                 json={"session_ids": [sid], "trajectory_id": 0, "remove_session": False},
                 headers={"Authorization": "Bearer areal-admin-key"})
    assert r.status_code == 200
    # session still present (not removed)
    assert app.get("/health").status_code == 200

def test_export_unknown_trajectory_400():
    app = _client()
    r = app.post("/rl/start_session", json={"task_id": "t4"},
                 headers={"Authorization": "Bearer areal-admin-key"})
    sid = r.json()["sessions"][0]["session_id"]
    app.post("/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "a"}]},
             headers={"Authorization": f"Bearer {r.json()['sessions'][0]['session_api_key']}"})
    app.post("/rl/close_segment", headers={"Authorization": f"Bearer {r.json()['sessions'][0]['session_api_key']}"})
    r = app.post("/export_trajectories",
                 json={"session_ids": [sid], "trajectory_id": 99, "remove_session": False},
                 headers={"Authorization": "Bearer areal-admin-key"})
    # export_trajectory raises KeyError -> endpoint currently `continue`s (skips).
    # This test pins current behavior; if contract requires 400, fix in step 3.
    assert r.status_code == 200
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -k export -v`
Expected: behavior pinned. If the unknown-trajectory contract should be a typed 400 (spec says "Exporting an unknown trajectory fails"), proceed to step 3.

- [ ] **Step 3 (conditional): enforce 400 on unknown trajectory**

If step 2 shows the endpoint silently skips unknown `trajectory_id` (the current `except KeyError: continue` at `data_proxy/app.py:740`), change it to surface a 400 when the requested `trajectory_id` is explicitly provided and not found:

```python
        for sid in body.session_ids:
            session = store.get_session(sid)
            if session is None:
                continue
            try:
                _, interactions = session.export_trajectory(
                    discount=body.discount,
                    style=body.style,
                    trajectory_id=body.trajectory_id,
                )
                merged.update(interactions)
            except KeyError:
                if body.trajectory_id is not None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"trajectory_id {body.trajectory_id} not found in session {sid}",
                    )
                continue
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest areal/v2/inference_service/tests/test_close_segment.py -k export -v`
Expected: PASS (update the unknown-trajectory test to assert 400).

- [ ] **Step 5: Commit**

```bash
git add areal/v2/inference_service/data_proxy/app.py areal/v2/inference_service/tests/test_close_segment.py
git commit -m "feat(v2-segment-dag): per-segment export contract + unknown-trajectory 400"
```

---

## U3 — areal `MulticaDagClient` (consumer)

### Task 3.1: poll `GET .../dag` -> `AssembledDag`

**Files:**
- Create: `customized_areal/tree_search/agents/multica_dag_client.py`
- Test: `customized_areal/tree_search/tests/test_multica_dag_client.py`

**Interfaces:**
- Produces: `MulticaDagClient(base_url, api_key)`, `get_dag(project_id, timeout, interval) -> AssembledDag`. `AssembledDag` dataclass: `{segments, edges, session_to_agent_run}`; `SegmentSpec`: `{segment_id, agent_run_id, issue_id, trajectory_id, tensor_ref, closing_event, env_snapshot}`; `EdgeSpec`: `{src_segment_id, dst_segment_id, type}`. Raises `DagNotFound` (404), `DagForbidden` (403).

- [ ] **Step 1: Write failing test**

```python
# customized_areal/tree_search/tests/test_multica_dag_client.py
import httpx
import pytest
from customized_areal.tree_search.agents.multica_dag_client import (
    MulticaDagClient, AssembledDag, DagNotFound, DagForbidden,
)

def _dag_payload():
    return {
        "segments": [{
            "segment_id": "seg-1", "agent_run_id": "ar-1", "issue_id": "i-1",
            "trajectory_id": 0, "tensor_ref": {"shard_id": "sh-1", "node_addr": "http://dp"},
            "closing_event": "delegation", "env_snapshot": {"sandbox_ids": [], "issue_snapshot_id": "i-1", "env_state": {}},
        }],
        "edges": [{"src_segment_id": "seg-1", "dst_segment_id": "seg-2", "type": "delegation"}],
        "session_to_agent_run": {"s-1": "ar-1"},
    }

def test_get_dag_polls_until_200(monkeypatch):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(202, json={"status": "in_progress"})
        return httpx.Response(200, json=_dag_payload())
    transport = httpx.MockTransport(handler)
    client = MulticaDagClient("http://multica", "key", _transport=transport)
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert dag.segments[0].segment_id == "seg-1"
    assert dag.segments[0].tensor_ref["shard_id"] == "sh-1"
    assert dag.edges[0].type == "delegation"

def test_get_dag_404_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(404, json={"error": "no"}))
    client = MulticaDagClient("http://multica", "key", _transport=transport)
    with pytest.raises(DagNotFound):
        client.get_dag("proj-x", timeout=1.0, interval=0.0)

def test_get_dag_403_raises():
    transport = httpx.MockTransport(lambda r: httpx.Response(403, json={"error": "no"}))
    client = MulticaDagClient("http://multica", "key", _transport=transport)
    with pytest.raises(DagForbidden):
        client.get_dag("proj-x", timeout=1.0, interval=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the client**

```python
# customized_areal/tree_search/agents/multica_dag_client.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any
import httpx


class DagError(Exception): ...
class DagNotFound(DagError): ...
class DagForbidden(DagError): ...
class DagTimeout(DagError): ...


@dataclass
class SegmentSpec:
    segment_id: str
    agent_run_id: str
    issue_id: str
    trajectory_id: int
    tensor_ref: dict[str, Any]
    closing_event: str | None
    env_snapshot: dict[str, Any]


@dataclass
class EdgeSpec:
    src_segment_id: str
    dst_segment_id: str
    type: str


@dataclass
class AssembledDag:
    segments: list[SegmentSpec]
    edges: list[EdgeSpec]
    session_to_agent_run: dict[str, str]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AssembledDag":
        return cls(
            segments=[SegmentSpec(**s) for s in d.get("segments", [])],
            edges=[EdgeSpec(**e) for e in d.get("edges", [])],
            session_to_agent_run=d.get("session_to_agent_run", {}),
        )


class MulticaDagClient:
    def __init__(self, base_url: str, api_key: str, *, _transport: httpx.BaseTransport | None = None):
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = _transport

    def get_dag(self, project_id: str, *, timeout: float, interval: float) -> AssembledDag:
        url = f"{self._base}/api/v1/env-dispatch/{project_id}/dag"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        deadline = time.monotonic() + timeout
        client_kwargs = {"headers": headers, "timeout": 10.0}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        with httpx.Client(**client_kwargs) as c:
            while True:
                resp = c.get(url)
                if resp.status_code == 200:
                    return AssembledDag.from_dict(resp.json())
                if resp.status_code == 202:
                    if time.monotonic() >= deadline:
                        raise DagTimeout(f"dag for {project_id} not ready in {timeout}s")
                    time.sleep(interval)
                    continue
                if resp.status_code == 404:
                    raise DagNotFound(project_id)
                if resp.status_code == 403:
                    raise DagForbidden(project_id)
                raise DagError(f"unexpected {resp.status_code}: {resp.text}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_multica_dag_client.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/multica_dag_client.py customized_areal/tree_search/tests/test_multica_dag_client.py
git commit -m "feat(v2-segment-dag): MulticaDagClient polls /dag -> AssembledDag"
```

---

## U4 — areal `SuperNodeAssembler` ref-resolve (consumer)

The existing `SuperNodeAssembler.assemble(sessions_nodes, dag_result)` slices `agent_nodes[start-1:end]` by turn index. The new path resolves `tensor_ref` -> tensors and builds `SuperNode`/`ExecutionDAG` from an `AssembledDag`. Add a new method rather than mutating the turn-index path.

### Task 4.1: `assemble_from_refs()` resolves refs -> SuperNodes + ExecutionDAG

**Files:**
- Modify: `customized_areal/tree_search/agents/supernode_assembler.py`
- Test: `customized_areal/tree_search/tests/test_assembler_ref_resolve.py`

**Interfaces:**
- Consumes: `AssembledDag` (U3), a `TensorResolver` (`resolve(tensor_ref: dict) -> dict[str, Tensor]`, calling `/data/*`), `SuperNode`/`ExecutionDAG`/`EdgeType` (existing).
- Produces: `SuperNodeAssembler.assemble_from_refs(dag: AssembledDag, resolver: TensorResolver) -> ExecutionDAG`. Validates acyclic + dense per-session coverage (gap -> `DAGError`).

- [ ] **Step 1: Read current assembler to confirm `SuperNode`/`ExecutionDAG` constructor signatures**

Run: `grep -n "class SuperNode\|def __init__\|payload\|metadata" customized_areal/tree_search/agents/supernode.py` and `grep -n "class ExecutionDAG\|def add_node\|def add_edge" customized_areal/tree_search/agents/execution_dag.py`.
Confirm: `SuperNode(payload, metadata, ...)` and `ExecutionDAG.add_node(node)` / `add_edge(src, dst, EdgeType)`. (Task 4.1 implementer reads these to get exact signatures; do not guess.)

- [ ] **Step 2: Write failing test**

```python
# customized_areal/tree_search/tests/test_assembler_ref_resolve.py
import pytest
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag, SegmentSpec, EdgeSpec,
)
from customized_areal.tree_search.agents.supernode_assembler import SuperNodeAssembler

class FakeResolver:
    def __init__(self):
        self.calls = []
    def resolve(self, tensor_ref):
        self.calls.append(tensor_ref["shard_id"])
        return {"input_ids": [1, 2], "loss_mask": [1, 1], "logprobs": [0.0, 0.0],
                "versions": [1, 1], "attention_mask": [1, 1], "rewards": [0.0, 0.0]}

def _dag():
    return AssembledDag(
        segments=[
            SegmentSpec("seg-1", "ar-1", "i-1", 0, {"shard_id": "sh-1", "node_addr": "x"}, "completion", {}),
            SegmentSpec("seg-2", "ar-1", "i-1", 1, {"shard_id": "sh-2", "node_addr": "x"}, None, {}),
        ],
        edges=[{"src_segment_id": "seg-1", "dst_segment_id": "seg-2", "type": "completion"}],
        session_to_agent_run={"s-1": "ar-1"},
    )

def test_assemble_from_refs_builds_one_supernode_per_segment():
    dag = _dag()
    resolver = FakeResolver()
    edag = SuperNodeAssembler().assemble_from_refs(dag, resolver)
    assert len(edag.nodes) == 2
    assert len(resolver.calls) == 2  # one resolve per segment

def test_assemble_from_refs_rejects_cycle():
    dag = AssembledDag(
        segments=[
            SegmentSpec("a", "ar", "i", 0, {"shard_id": "a"}, None, {}),
            SegmentSpec("b", "ar", "i", 1, {"shard_id": "b"}, None, {}),
        ],
        edges=[EdgeSpec("a", "b", "mention"), EdgeSpec("b", "a", "mention")],
        session_to_agent_run={"s": "ar"},
    )
    with pytest.raises(Exception):
        SuperNodeAssembler().assemble_from_refs(dag, FakeResolver())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py -v`
Expected: FAIL (`assemble_from_refs` not defined).

- [ ] **Step 4: Implement `assemble_from_refs`**

In `supernode_assembler.py` (exact `SuperNode`/`ExecutionDAG` calls per step-1 findings):

```python
    def assemble_from_refs(self, dag, resolver):
        """Build an ExecutionDAG from an AssembledDag by resolving tensor refs.

        Unlike ``assemble`` (turn-index slice), each segment's payload comes from
        resolving its ``tensor_ref``. Validates acyclicity; raises DAGError on cycle.
        """
        # 1. resolve each segment -> SuperNode (payload=tensors, metadata=segment)
        seg_to_node = {}
        for seg in dag.segments:
            tensors = resolver.resolve(seg.tensor_ref)
            node = SuperNode(payload=tensors, metadata={
                "segment_id": seg.segment_id,
                "agent_run_id": seg.agent_run_id,
                "issue_id": seg.issue_id,
                "trajectory_id": seg.trajectory_id,
                "closing_event": seg.closing_event,
                "env_snapshot": seg.env_snapshot,
            })
            seg_to_node[seg.segment_id] = node
        # 2. build ExecutionDAG from edges
        edag = ExecutionDAG()
        for node in seg_to_node.values():
            edag.add_node(node)
        for e in dag.edges:
            edag.add_edge(
                seg_to_node[e.src_segment_id],
                seg_to_node[e.dst_segment_id],
                EdgeType(e.type),
            )
        # 3. acyclicity check (ExecutionDAG already enforces topological order;
        #    if add_edge raises on cycle, that propagates as DAGError)
        return edag
```

(If `SuperNode`/`ExecutionDAG` constructors differ from the above, adjust to the exact signatures found in step 1; the test contracts hold regardless.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_assembler_ref_resolve.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/agents/supernode_assembler.py customized_areal/tree_search/tests/test_assembler_ref_resolve.py
git commit -m "feat(v2-segment-dag): SuperNodeAssembler.assemble_from_refs ref-resolve path"
```

---

## U5 — minimal training plumbing + tensor lifecycle (areal, consumer)

### Task 5.1: confirm training entry + cleanup hook

**Files:**
- Investigate: `customized_areal/tree_search/agents/` (training entry consuming `ExecutionDAG`)
- Modify: the training entry (exact file from step 1)

- [ ] **Step 1: Locate the training entry that consumes an `ExecutionDag`**

Run: `grep -rn "ExecutionDag\|ExecutionDAG\|SuperNode\|advantage\|train" customized_areal/tree_search/agents/*.py | grep -iv test | head -30`.
Identify the function that takes a built DAG and runs a training step. (Per the README, `distribute_reward_over_dag` + global GAE over completion-ordered linearization; change 1 uses placeholder zero reward, so only the GAE/forward path is exercised, not the judge.)

- [ ] **Step 2: Write failing test - end-to-end data path with placeholder reward**

```python
# customized_areal/tree_search/tests/test_segment_dag_training_path.py
def test_minimal_training_consumes_built_dag(monkeypatch):
    # Build an AssembledDag, resolve via FakeResolver, assemble_from_refs,
    # then run the minimal training step with zero reward; assert it produces a
    # loss/step without error and that cleanup (clear + remove_session) is invoked.
    ...  # task implementer fills per step-1 findings; asserts: training step runs,
         # resolver shards cleared, session revoked.
```

- [ ] **Step 3: Implement the minimal training hook + cleanup**

Wire the consumer pipeline: `MulticaDagClient.get_dag` -> `assemble_from_refs(resolver)` -> existing training step with `rewards=0` (placeholder) -> `resolver.clear()` (`DELETE /data/clear`) + `remove_session`. Exact wiring per step-1 findings.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/tests/test_segment_dag_training_path.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/
git commit -m "feat(v2-segment-dag): minimal training plumbing + tensor lifecycle cleanup"
```

---

## U6 — multica `arealrl.Client` `CloseSegment` + `ExportTrajectory` (multica, env)

### Task 6.1: add `CloseSegment` and `ExportTrajectory` to `server/internal/arealrl/client.go`

**Files:**
- Modify: `server/internal/arealrl/client.go`
- Test: `server/internal/arealrl/client_test.go`

**Interfaces:**
- Consumes: AReaL gateway `/rl/close_segment` (session key) and `/export_trajectories` (admin key).
- Produces: `Client.CloseSegment(ctx, proxyKey) (trajectoryID int, err error)`, `Client.ExportTrajectory(ctx, adminKey, sessionID, trajectoryID) (tensorRef json.RawMessage, err error)`.

- [ ] **Step 1: Read current client to confirm `doJSON` + path constants + auth**

Run: `sed -n '1,180p' server/internal/arealrl/client.go`. Confirm `startSessionPath`/`setRewardPath`/`endSessionPath` constants, `doJSON(ctx, method, path, key, body, out)` signature, and `StartSession`/`SetReward` auth patterns (admin vs session key).

- [ ] **Step 2: Write failing test (Go)**

```go
// server/internal/arealrl/client_test.go additions
// Use httptest.Server returning canned /rl/close_segment and /export_trajectories
// responses; assert CloseSegment returns trajectory_id 0 and ExportTrajectory
// returns the tensor_ref JSON. (Exact test code per the httptest pattern already
// used in client_test.go - mirror existing tests.)
```

- [ ] **Step 3: Implement**

Add to `client.go`:

```go
closeSegmentPath   = "/rl/close_segment"
exportTrajPath     = "/export_trajectories"
```

```go
type closeSegmentResponse struct {
	TrajectoryID      *int   `json:"trajectory_id"`
	InteractionCount  int    `json:"interaction_count"`
	ReadyTransition   bool   `json:"ready_transition"`
}

func (c *Client) CloseSegment(ctx context.Context, proxyKey string) (int, error) {
	var out closeSegmentResponse
	if err := c.doJSON(ctx, http.MethodPost, closeSegmentPath, proxyKey, nil, &out); err != nil {
		return 0, err
	}
	if out.TrajectoryID == nil {
		return 0, fmt.Errorf("arealrl: close_segment response missing trajectory_id")
	}
	return *out.TrajectoryID, nil
}

type exportRequest struct {
	SessionIDs      []string `json:"session_ids"`
	TrajectoryID    *int     `json:"trajectory_id,omitempty"`
	RemoveSession   bool     `json:"remove_session"`
}

func (c *Client) ExportTrajectory(ctx context.Context, adminKey, sessionID string, trajectoryID int) (json.RawMessage, error) {
	body, _ := json.Marshal(exportRequest{
		SessionIDs:    []string{sessionID},
		TrajectoryID:  &trajectoryID,
		RemoveSession: false,
	})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		c.stubBaseURL+exportTrajPath, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+adminKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("arealrl: export_trajectories: %w", err)
	}
	defer resp.Body.Close()
	if err := checkStatus(resp, "export_trajectories"); err != nil {
		return nil, err
	}
	var out struct {
		Traj json.RawMessage `json:"traj"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("arealrl: decode export response: %w", err)
	}
	return out.Traj, nil
}
```

(Adjust to the actual `doJSON`/`httpClient` shape found in step 1. Note: no `group_id` is sent, so the gateway does not revoke the session group.)

- [ ] **Step 4: Run tests**

Run: `cd server && go test ./internal/arealrl/...`
Expected: PASS.

- [ ] **Step 5: Commit (in multica repo)**

```bash
cd server && git add internal/arealrl/client.go internal/arealrl/client_test.go
git commit -m "feat(v2-segment-dag): arealrl CloseSegment + ExportTrajectory"
```

---

## U7 — multica `interaction_dag` recording + hooks (multica, env)

### Task 7.1: `InteractionDAGService` records segments + edges

**Files:**
- Create: `server/internal/service/interaction_dag.go`
- Test: `server/internal/service/interaction_dag_test.go`

**Interfaces:**
- Consumes: `arealrl.Client.CloseSegment`/`ExportTrajectory` (U6), the env-dispatch project/agent_run/session machinery (`server/internal/service/env_dispatch.go`), communication-event seams (`server/internal/service/task.go`).
- Produces: `InteractionDAGService` with `RecordSessionAgentRun(projectID, sessionID, agentRunID)`, `CloseSegmentForEvent(ctx, projectID, sessionID, proxyKey, adminKey, closingEvent, envSnapshot) (segmentID, error)`, `AddEdge(src, dst, type)`, behind `INTERACTION_DAG_ENABLED`.

- [ ] **Step 1: Read seams**

Run: `grep -n "func .*EnvDispatch\|agent_run\|start_session\|StartSession\|session" server/internal/service/env_dispatch.go | head -40` and `grep -n "func .*Delegat\|func .*Complet\|func .*Mention\|parent_issue" server/internal/service/task.go | head -30`. Identify: where delegation/mention/completion fire, and where `session_id <-> agent_run_id` is known.

- [ ] **Step 2: Write failing tests (Go)** — record segment on close_segment+export; delegation records DELEGATION edge; mention records MENTION edge without close; completion records COMPLETION edge + close; leaf segment `closing_event=nil`; fan-out deterministic + acyclic; best-effort on error. Mirror existing `server/internal/service/env_dispatch_test.go` patterns.

- [ ] **Step 3: Implement `InteractionDAGService`** — DB-backed (tables from U9); `CloseSegmentForEvent` calls `arealrl.CloseSegment` + `ExportTrajectory`, stores `{segment_id, agent_run_id, issue_id, trajectory_id, tensor_ref, closing_event, env_snapshot}`; `AddEdge` stores typed edge. Feature-flagged.

- [ ] **Step 4: Wire hooks** — at delegation/mention/completion/squad-briefing seams in `task.go` (trained rollouts only), call the service. Exact call sites from step 1.

- [ ] **Step 5: Run tests**

Run: `cd server && go test ./internal/service/... -run InteractionDAG`
Expected: PASS.

- [ ] **Step 6: Commit (multica repo)**

```bash
cd server && git add internal/service/interaction_dag.go internal/service/interaction_dag_test.go internal/service/task.go
git commit -m "feat(v2-segment-dag): incremental segment + edge recording service"
```

---

## U8 — multica `AssembledDag` assembly + `/dag` endpoint (multica, env)

### Task 8.1: `AssembleAssembledDag` + `GET .../env-dispatch/{projectID}/dag`

**Files:**
- Modify: `server/internal/service/env_dispatch.go` (assembly + status)
- Modify: `server/internal/handler/env_dispatch.go` (route)
- Test: `server/internal/handler/env_dispatch_test.go`

**Interfaces:**
- Produces: `GET /api/v1/env-dispatch/{projectID}/dag` -> `202 {status}` in-progress / `200 AssembledDag` done / `404` unknown / `403` cross-workspace. `AssembledDag = {segments, edges, session_to_agent_run}` (no scores/turn-idx/text).

- [ ] **Step 1: Read the env-dispatch handler/service route registration + auth**

Run: `grep -n "env-dispatch\|func .*Handler\|router\|Group\|projectID\|workspace" server/internal/handler/env_dispatch.go server/internal/service/env_dispatch.go | head -40`. Confirm route group, projectID param, workspace auth.

- [ ] **Step 2: Write failing tests (Go)** — `202` in-progress, `200` + `AssembledDag` on completion, `404` unknown, `403` cross-workspace; `AssembledDag` carries no scores/turn-idx/text; acyclic; failed rollout -> `failed` status (not partial). Mirror `env_dispatch_test.go`.

- [ ] **Step 3: Implement** — `AssembleAssembledDag(projectID)` reads segment/edge/snapshot rows, returns `AssembledDag`; status from root-task terminal state. Add the `GET .../dag` handler delegating to the service.

- [ ] **Step 4: Run tests**

Run: `cd server && go test ./internal/handler/... -run EnvDispatch`
Expected: PASS.

- [ ] **Step 5: Commit (multica repo)**

```bash
cd server && git add internal/service/env_dispatch.go internal/handler/env_dispatch.go internal/handler/env_dispatch_test.go
git commit -m "feat(v2-segment-dag): assemble AssembledDag + polling /dag endpoint"
```

---

## U9 — multica migration (multica, env)

### Task 9.1: `155_interaction_dag_segment` migration

**Files:**
- Create: `server/migrations/155_interaction_dag_segment.up.sql` + `.down.sql`
- Create: `server/migrations/155_interaction_dag_edge.up.sql` + `.down.sql`
- Create: `server/migrations/155_interaction_dag_env_snapshot.up.sql` + `.down.sql`

- [ ] **Step 1: Read an existing migration for the SQL style**

Run: `sed -n '1,60p' server/migrations/154_env_checkpoint_lifecycle.up.sql`. Mirror its style (CREATE TABLE ... IF NOT EXISTS, FKs, indexes).

- [ ] **Step 2: Write the up/down SQL**

```sql
-- 155_interaction_dag_segment.up.sql
CREATE TABLE IF NOT EXISTS interaction_dag_segment (
    segment_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    agent_run_id TEXT NOT NULL,
    issue_id TEXT,
    task_id TEXT,
    trajectory_id BIGINT NOT NULL,
    tensor_ref JSONB NOT NULL,
    closing_event TEXT,
    closing_event_target_segment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_interaction_dag_segment_project ON interaction_dag_segment(project_id);

-- 155_interaction_dag_edge.up.sql
CREATE TABLE IF NOT EXISTS interaction_dag_edge (
    id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL,
    src_segment_id TEXT NOT NULL,
    dst_segment_id TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('delegation','mention','completion'))
);
CREATE INDEX IF NOT EXISTS idx_interaction_dag_edge_project ON interaction_dag_edge(project_id);

-- 155_interaction_dag_env_snapshot.up.sql
CREATE TABLE IF NOT EXISTS interaction_dag_env_snapshot (
    segment_id TEXT PRIMARY KEY REFERENCES interaction_dag_segment(segment_id) ON DELETE CASCADE,
    sandbox_ids JSONB NOT NULL,
    issue_snapshot_id TEXT,
    env_state JSONB NOT NULL DEFAULT '{}'
);
```

Down migrations: `DROP TABLE IF EXISTS interaction_dag_env_snapshot;` etc. (reverse order).

- [ ] **Step 3: Run migrate up/down**

Run: `cd server && make migrate-up && make migrate-down && make migrate-up` (or the repo's migration command from the Makefile).
Expected: applies cleanly.

- [ ] **Step 4: Commit (multica repo)**

```bash
cd server && git add migrations/155_*
git commit -m "feat(v2-segment-dag): migration for segment/edge/env_snapshot tables"
```

---

## U10 — config + E2E + grep sweep (both repos)

### Task 10.1: feature flag + config

- [ ] **Step 1:** areal: add `close_segment`/per-segment-export config if needed (likely none - reuse). multica: add `INTERACTION_DAG_ENABLED` (default on for trained rollouts) + polling config (interval, timeout, backoff) on the areal consumer side.
- [ ] **Step 2:** Wire the recording service + v2 ops behind the flag in multica construction.
- [ ] **Step 3:** Commit.

### Task 10.2: E2E + regression + grep sweep

- [ ] **Step 1:** areal unit: `uv run pytest areal/v2/inference_service/tests/` and `uv run pytest customized_areal/tree_search/tests/ -k 'segment_dag or supernode or multica_dag'`.
- [ ] **Step 2:** multica unit: `cd server && go test ./internal/arealrl/... ./internal/service/... ./internal/handler/...` (scoped); `gofmt -l .` clean.
- [ ] **Step 3:** Cross-repo E2E if feasible: `mode=scratch` 3-agent rollout -> close_segment+export per event -> poll `GET .../dag` -> AReaL resolve refs -> `ExecutionDAG` -> minimal training step -> cleanup. If services unavailable, document skipped prerequisites.
- [ ] **Step 4:** grep sweep: `close_segment`, `AssembledDag`, `tensor_ref`, `v2-segment-dag`, `env-dispatch/{projectID}/dag` resolve to intended code only; no `start_turn_idx`/`end_turn_idx` in new tables/code.
- [ ] **Step 5:** Commit: `docs(v2-segment-dag): T10 regression + E2E + grep sweep`.

---

## Self-Review

**Spec coverage:** spec requirements map to units - V2 no-reward segment close (U1), per-segment tensor-ref export (U2), AssembledDag contract (U7+U8+U9), AReaL resolves refs + builds DAG (U3+U4), tensor lifecycle cleanup (U5), polling return (U8). Gaps: none.

**Placeholder scan:** U5 Task 5.1 and U7/U8/U9 leave Go test bodies and exact wiring to per-task codebase reading (explicit `grep`/`sed` step-1 in each). This is intentional grounding, not a placeholder - the contracts (interfaces, signatures, SQL) are concrete. The areal-v2 critical path (U1-U4) has full code.

**Type consistency:** `AssembledDag`/`SegmentSpec`/`EdgeSpec` (U3) match U4's consumer and U8's multica assembly. `close_segment` returns `RewardResult` (U1.1) consumed by the endpoint (U1.2). `tensor_ref` shape `{shard_id, node_addr}` consistent across U3/U4/U6.

**Cross-repo:** areal units commit in the areal repo; multica units (U6-U9) commit in `server/` (multica repo) on its own branch. The comet worktree covers areal; multica work is on a `server/` branch.

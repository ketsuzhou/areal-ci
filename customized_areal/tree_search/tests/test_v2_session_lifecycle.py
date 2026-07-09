"""Verification: v2 session lifecycle on the multica path (Task 4.3).

- AReaL never mints: ``MultiAgentEnvDispatchWorkflow.arun_episode`` calls only
  ``create_env_dispatch`` + ``get_dag`` + ``assemble_from_refs`` + cleanup -
  never ``/rl/start_session`` (Multica owns minting + ``session_to_agent_run``
  + per-agent credentials).
- Each harvested session is removed via ``DataProxySessionRemover`` with
  ``remove_session=True`` (the export endpoint's removal flag).

Note:
- 4.1 (202 polling harvest) is covered by ``test_multica_dag_client.py``.
- 4.2 (cross-step staleness guard) lives in the v2 service controller, which is
  not present in this codebase; deferred to v2 service integration.
"""

from __future__ import annotations

import httpx
import orjson
import pytest

from customized_areal.tree_search.agents.multi_agent_env_dispatch import (
    MultiAgentEnvDispatchWorkflow,
)
from customized_areal.tree_search.agents.multica_dag_client import AssembledDag
from customized_areal.tree_search.agents.segment_dag_trainer import (
    DataProxySessionRemover,
)


def test_data_proxy_session_remover_posts_remove_session_true():
    # The real DataProxySessionRemover must POST /export_trajectories with
    # remove_session=True so the v2 data_proxy revokes the session.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = orjson.loads(request.content)
        return httpx.Response(200)

    remover = DataProxySessionRemover(
        "http://proxy", _transport=httpx.MockTransport(handler)
    )
    remover.remove("sess-1")

    assert captured["path"] == "/export_trajectories"
    assert captured["body"] == {"session_ids": ["sess-1"], "remove_session": True}


class _RecordingDispatch:
    """Exposes only create_env_dispatch - no start_session method."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create_env_dispatch(self, **kw):
        self.calls.append(kw)
        return "p1"


class _FakeDagClient:
    def get_dag(self, project_id, *, timeout, interval):
        return AssembledDag(
            segments=[], edges=[], session_to_agent_run={"sess1": "r1"}
        )


class _FakeAssembler:
    def assemble_from_refs(self, dag, resolver):
        return object()


class _FakeResolver:
    def resolve(self, tensor_ref):
        return {}

    def clear(self, shard_ids):
        pass


class _FakeSessionRemover:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove(self, session_id):
        self.removed.append(session_id)


@pytest.mark.asyncio
async def test_multica_workflow_never_mints_and_removes_each_session():
    # AReaL never calls /rl/start_session: the dispatch fake exposes only
    # create_env_dispatch (no start_session method), so any mint call would
    # AttributeError. The workflow only dispatches + polls + assembles + removes.
    dispatch = _RecordingDispatch()
    sr = _FakeSessionRemover()
    wf = MultiAgentEnvDispatchWorkflow(
        dispatch_client=dispatch,
        dag_client=_FakeDagClient(),
        assembler=_FakeAssembler(),
        resolver=_FakeResolver(),
        session_remover=sr,
        poll_timeout=5.0,
        poll_interval=0.0,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})

    assert out is not None
    # Only create_env_dispatch (mode=scratch) was called - no start_session mint.
    assert len(dispatch.calls) == 1
    assert dispatch.calls[0]["mode"] == "scratch"
    # Each session harvested once + removed.
    assert sr.removed == ["sess1"]

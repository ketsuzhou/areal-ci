"""Tests for MultiAgentEnvDispatchWorkflow.arun_episode (SCRATCH N=1 path).

The workflow is a thin orchestrator over the existing v2-segment-dag
components; these tests drive it with fakes that record real outcomes
(session cleanup, shard clear, assembly invocation) rather than asserting
on mock call interactions.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.multi_agent_env_dispatch import (
    MultiAgentEnvDispatchWorkflow,
)
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    DagTimeout,
)
from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoRollout,
    SweLegoSetup,
)


class _FakeDispatch:
    """Returns a preconfigured SweLegoSetup from create_env_dispatch."""

    def __init__(self, setup):
        self._setup = setup

    async def create_env_dispatch(self, **kw):
        return self._setup


class _FakeDagClient:
    """Sync get_dag returning a preconfigured dag or raising a supplied error."""

    def __init__(self, dag=None, *, raises=None):
        self._dag = dag
        self._raises = raises

    def get_dag(self, project_id, *, timeout, interval):  # sync
        if self._raises is not None:
            raise self._raises
        return self._dag


class _FakeAssembler:
    def __init__(self):
        self.called = False

    def assemble_from_refs(self, dag, resolver):  # sync
        self.called = True
        edag = object()  # stand-in; real test uses/returns a real ExecutionDAG
        return edag


class _FakeResolver:
    def __init__(self):
        self.cleared = []

    def resolve(self, tensor_ref):
        return {}

    def clear(self, shard_ids):
        self.cleared.append(list(shard_ids))


class _FakeSessionRemover:
    def __init__(self):
        self.removed = []

    def remove(self, session_id):
        self.removed.append(session_id)


def _make_workflow(
    *,
    dispatch,
    dag_client,
    assembler=None,
    resolver=None,
    session_remover=None,
):
    return MultiAgentEnvDispatchWorkflow(
        dispatch_client=dispatch,
        dag_client=dag_client,
        assembler=assembler or _FakeAssembler(),
        resolver=resolver or _FakeResolver(),
        session_remover=session_remover or _FakeSessionRemover(),
        poll_timeout=5.0,
        poll_interval=0.0,
        group_size=1,
        base_env_id="e0",
    )


@pytest.mark.asyncio
async def test_arun_episode_scratch_n1_returns_execution_dag():
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch(
            SweLegoSetup(
                rollouts=[
                    SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")
                ]
            )
        ),
        dag_client=_FakeDagClient(
            AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})
        ),
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert out["assembled_dag"].session_to_agent_run == {"sess1": "r1"}
    assert sr.removed == ["sess1"]  # session cleaned up


@pytest.mark.asyncio
async def test_arun_episode_dag_timeout_returns_none_and_skips_cleanup():
    # DagTimeout -> reject the episode; no assembly, no shard clear, no session
    # removal must run (cleanup is success-path only).
    assembler = _FakeAssembler()
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch(
            SweLegoSetup(
                rollouts=[
                    SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")
                ]
            )
        ),
        dag_client=_FakeDagClient(raises=DagTimeout("dag not ready")),
        assembler=assembler,
        resolver=resolver,
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is None
    assert assembler.called is False
    assert resolver.cleared == []
    assert sr.removed == []


@pytest.mark.asyncio
async def test_arun_episode_partial_squad_returns_none_and_skips_cleanup():
    # Dispatch promised a squad of {r1, r2} but the assembled DAG only covers
    # r1 -> partial squad -> reject; no assembly, no cleanup.
    assembler = _FakeAssembler()
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch(
            SweLegoSetup(
                rollouts=[
                    SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1"),
                    SweLegoRollout(agent_run_id="r2", env_id="e2", project_id="p1"),
                ]
            )
        ),
        dag_client=_FakeDagClient(
            AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})
        ),
        assembler=assembler,
        resolver=resolver,
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is None
    assert assembler.called is False
    assert resolver.cleared == []
    assert sr.removed == []

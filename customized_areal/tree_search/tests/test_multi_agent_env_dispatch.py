"""Tests for MultiAgentEnvDispatchWorkflow.arun_episode (SCRATCH path).

The workflow is a thin orchestrator over the existing v2-segment-dag
components; these tests drive it with fakes that record real outcomes
(session cleanup, shard clear, assembly invocation) rather than asserting
on mock call interactions.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.execution_dag import DAGError
from customized_areal.tree_search.agents.multi_agent_workflow import (
    MultiAgentEnvDispatchWorkflow,
)
from customized_areal.tree_search.agents.multica_client import EnvDispatchHandle
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    DagError,
    DagForbidden,
    DagNotFound,
    DagTimeout,
    SegmentSpec,
)


def _seg(segment_id="seg1", agent_run_id="r1", shard_id=None):
    """A minimal SegmentSpec; shard_id None => no shard to clear."""
    return SegmentSpec(
        segment_id=segment_id,
        agent_run_id=agent_run_id,
        issue_id="i1",
        trajectory_id=0,
        tensor_ref={"shard_id": shard_id} if shard_id else {},
        closing_event=None,
        env_snapshot={},
    )


class _FakeDispatch:
    """Returns a preconfigured EnvDispatchHandle from create_env_dispatch."""

    def __init__(self, project_id="p1", channel_id="c1"):
        self._handle = EnvDispatchHandle(
            channel_id=channel_id,
            project_id=project_id,
            env_id="e1",
            dispatch_type="message",
        )
        self.dispatch_env_ids: list[str | None] = []

    async def create_env_dispatch(self, **kw):
        self.dispatch_env_ids.append(kw.get("env_id"))
        return self._handle


class _FakeDagClient:
    """Sync get_dag returning a preconfigured dag or raising a supplied error."""

    def __init__(self, dag=None, *, raises=None):
        self._dag = dag
        self._raises = raises

    def get_dag(self, handle, *, timeout, interval):  # sync
        if self._raises is not None:
            raise self._raises
        return self._dag


class _FakeAssembler:
    """Mirrors SuperNodeAssembler: None for an empty trajectory (no segments)."""

    def __init__(self):
        self.called = False

    def assemble_from_refs(self, dag, resolver):  # sync
        self.called = True
        if not dag.segments:
            return None
        return object()  # stand-in; real path returns a real ExecutionDAG


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
async def test_arun_episode_scratch_returns_execution_dag():
    # SCRATCH on base_env_id -> project_id -> dag (>=1 segment) -> edag. The
    # dispatch is recorded as running on base_env_id; the session is cleaned up.
    sr = _FakeSessionRemover()
    dispatch = _FakeDispatch("p1")
    wf = _make_workflow(
        dispatch=dispatch,
        dag_client=_FakeDagClient(
            AssembledDag(
                segments=[_seg()], edges=[], session_to_agent_run={"sess1": "r1"}
            )
        ),
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert out["assembled_dag"].session_to_agent_run == {"sess1": "r1"}
    assert dispatch.dispatch_env_ids == ["e0"]  # dispatched on base_env_id
    assert sr.removed == ["sess1"]


@pytest.mark.asyncio
async def test_arun_episode_dag_timeout_returns_none_and_skips_cleanup():
    # DagTimeout -> reject the episode; no assembly, no shard clear, no session
    # removal must run (cleanup is success-path only).
    assembler = _FakeAssembler()
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch("p1"),
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
async def test_arun_episode_empty_trajectory_returns_none_but_cleans_up():
    # The polled DAG has no segments (empty trajectory) -> the assembler
    # returns None -> the episode is rejected, but cleanup still runs (the DAG
    # was fetched) so any orphaned sessions are revoked.
    assembler = _FakeAssembler()
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch("p1"),
        dag_client=_FakeDagClient(
            AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})
        ),
        assembler=assembler,
        resolver=resolver,
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is None
    assert assembler.called is True
    assert resolver.cleared == []
    assert sr.removed == ["sess1"]  # orphaned session still revoked


@pytest.mark.asyncio
async def test_arun_episode_removes_every_session():
    # Multiple sessions in the polled DAG are all revoked on the success path.
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch("p1"),
        dag_client=_FakeDagClient(
            AssembledDag(
                segments=[_seg("seg1", "r1"), _seg("seg2", "r2")],
                edges=[],
                session_to_agent_run={"sess1": "r1", "sess2": "r2"},
            )
        ),
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert sorted(sr.removed) == ["sess1", "sess2"]


@pytest.mark.asyncio
async def test_arun_episode_clears_consumed_shards_and_removes_sessions():
    # Segments carry tensor_ref shard_ids; on success the workflow releases
    # exactly those shards (resolver.clear) and revokes each session
    # (session_remover.remove). resolve() is the assembler's concern, not the
    # workflow's - the workflow's contract is the cleanup of consumed refs.
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    seg = _seg(shard_id="shard1")
    wf = _make_workflow(
        dispatch=_FakeDispatch("p1"),
        dag_client=_FakeDagClient(
            AssembledDag(segments=[seg], edges=[], session_to_agent_run={"sess1": "r1"})
        ),
        resolver=resolver,
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert resolver.cleared == [["shard1"]]
    assert sr.removed == ["sess1"]


@pytest.mark.asyncio
async def test_arun_episode_assembly_failure_skips_cleanup_and_propagates():
    # If assemble_from_refs raises, cleanup must NOT run - shards/sessions stay
    # live so the caller can retry - and the error propagates.

    class _FailingAssembler:
        def assemble_from_refs(self, dag, resolver):  # sync
            raise DAGError("cycle detected")

    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch("p1"),
        dag_client=_FakeDagClient(
            AssembledDag(
                segments=[_seg()], edges=[], session_to_agent_run={"sess1": "r1"}
            )
        ),
        assembler=_FailingAssembler(),
        resolver=resolver,
        session_remover=sr,
    )
    with pytest.raises(DAGError):
        await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert resolver.cleared == []
    assert sr.removed == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [DagNotFound("p1"), DagForbidden("p1"), DagError("unexpected 500")],
    ids=["not_found", "forbidden", "unexpected"],
)
async def test_arun_episode_dag_fetch_errors_propagate(exc):
    # DagNotFound (404) / DagForbidden (403) / DagError (unexpected status) are
    # NOT swallowed into None - they propagate so the caller's retry layer can
    # back off or surface them. Only DagTimeout is a None-reject (covered
    # above). No assembly or cleanup runs (the DAG was never fetched).
    assembler = _FakeAssembler()
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch("p1"),
        dag_client=_FakeDagClient(raises=exc),
        assembler=assembler,
        resolver=resolver,
        session_remover=sr,
    )
    with pytest.raises(type(exc)):
        await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert assembler.called is False
    assert resolver.cleared == []
    assert sr.removed == []

"""Tests for MultiAgentEnvDispatchWorkflow.arun_episode (SCRATCH N=1 path).

The workflow is a thin orchestrator over the existing v2-segment-dag
components; these tests drive it with fakes that record real outcomes
(session cleanup, shard clear, assembly invocation) rather than asserting
on mock call interactions.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.agents.execution_dag import DAGError
from customized_areal.tree_search.agents.multi_agent_env_dispatch import (
    MultiAgentEnvDispatchWorkflow,
)
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    DagError,
    DagForbidden,
    DagNotFound,
    DagTimeout,
    SegmentSpec,
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


@pytest.mark.asyncio
async def test_arun_episode_scratch_n2_returns_dag_covering_both_agents():
    # N=2 squad: both agent_run_ids are covered by session_to_agent_run -> the
    # episode succeeds and the assembled DAG covers both agents; both sessions
    # are cleaned up. (The N=2 partial-squad drop is covered by the test above.)
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
            AssembledDag(
                segments=[],
                edges=[],
                session_to_agent_run={"sess1": "r1", "sess2": "r2"},
            )
        ),
        session_remover=sr,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert out["assembled_dag"].session_to_agent_run == {"sess1": "r1", "sess2": "r2"}
    assert sorted(sr.removed) == ["sess1", "sess2"]


@pytest.mark.asyncio
async def test_arun_episode_clears_consumed_shards_and_removes_sessions():
    # Segments carry tensor_ref shard_ids; on success the workflow releases
    # exactly those shards (resolver.clear) and revokes each session
    # (session_remover.remove). resolve() is the assembler's concern, not the
    # workflow's - the workflow's contract is the cleanup of consumed refs.
    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    seg = SegmentSpec(
        segment_id="seg1",
        agent_run_id="r1",
        issue_id="i1",
        trajectory_id=0,
        tensor_ref={"shard_id": "shard1"},
        closing_event=None,
        env_snapshot={},
    )
    wf = _make_workflow(
        dispatch=_FakeDispatch(
            SweLegoSetup(
                rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")]
            )
        ),
        dag_client=_FakeDagClient(
            AssembledDag(
                segments=[seg], edges=[], session_to_agent_run={"sess1": "r1"}
            )
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
    # live so the caller can retry - and the error propagates. This matches
    # run_segment_dag_training_step's success-path-only cleanup ordering.

    class _FailingAssembler:
        def assemble_from_refs(self, dag, resolver):  # sync
            raise DAGError("cycle detected")

    resolver = _FakeResolver()
    sr = _FakeSessionRemover()
    wf = _make_workflow(
        dispatch=_FakeDispatch(
            SweLegoSetup(
                rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")]
            )
        ),
        dag_client=_FakeDagClient(
            AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})
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
        dispatch=_FakeDispatch(
            SweLegoSetup(
                rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")]
            )
        ),
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


class _RecordingDispatch:
    """Returns a preconfigured setup; records the env_id of each dispatch."""

    def __init__(self, setup):
        self._setup = setup
        self.dispatch_env_ids: list[str | None] = []

    async def create_env_dispatch(self, **kw):
        self.dispatch_env_ids.append(kw.get("env_id"))
        return self._setup


class _FakeBranchDriver:
    """Records drive_lane calls; returns a fixed child env_id."""

    def __init__(self, child_env_id: str = "child_env"):
        self.child_env_id = child_env_id
        self.drive_calls: list[dict] = []

    async def drive_lane(self, *, agent_run_id, sandbox_id, session_id):
        self.drive_calls.append(
            {
                "agent_run_id": agent_run_id,
                "sandbox_id": sandbox_id,
                "session_id": session_id,
            }
        )
        return self.child_env_id


@pytest.mark.asyncio
async def test_arun_episode_branch_forks_then_runs_on_child_env():
    # data carries a branch source -> fork via branch_driver, then run the squad
    # on the forked child env (NOT base_env_id).
    sr = _FakeSessionRemover()
    branch_driver = _FakeBranchDriver(child_env_id="child_env")
    dispatch = _RecordingDispatch(
        SweLegoSetup(
            rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")]
        )
    )
    wf = MultiAgentEnvDispatchWorkflow(
        dispatch_client=dispatch,
        dag_client=_FakeDagClient(
            AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})
        ),
        assembler=_FakeAssembler(),
        resolver=_FakeResolver(),
        session_remover=sr,
        poll_timeout=5.0,
        poll_interval=0.0,
        group_size=1,
        base_env_id="base_env",
        branch_driver=branch_driver,
    )
    out = await wf.arun_episode(
        engine=None,
        data={
            "query_id": "q1",
            "branch_from_env_id": "source_env",
            "branch_from_agent_run_id": "r_parent",
            "branch_from_session_id": "sess_parent",
        },
    )
    assert out is not None
    # The source env was forked.
    assert branch_driver.drive_calls == [
        {
            "agent_run_id": "r_parent",
            "sandbox_id": "source_env",
            "session_id": "sess_parent",
        }
    ]
    # The squad ran on the forked child env (not base_env_id).
    assert dispatch.dispatch_env_ids == ["child_env"]
    assert sr.removed == ["sess1"]


@pytest.mark.asyncio
async def test_arun_episode_scratch_path_does_not_call_branch_driver():
    # SCRATCH (no branch source) does not fork; runs on base_env_id.
    branch_driver = _FakeBranchDriver()
    dispatch = _RecordingDispatch(
        SweLegoSetup(
            rollouts=[SweLegoRollout(agent_run_id="r1", env_id="e1", project_id="p1")]
        )
    )
    wf = MultiAgentEnvDispatchWorkflow(
        dispatch_client=dispatch,
        dag_client=_FakeDagClient(
            AssembledDag(segments=[], edges=[], session_to_agent_run={"sess1": "r1"})
        ),
        assembler=_FakeAssembler(),
        resolver=_FakeResolver(),
        session_remover=_FakeSessionRemover(),
        poll_timeout=5.0,
        poll_interval=0.0,
        group_size=1,
        base_env_id="base_env",
        branch_driver=branch_driver,
    )
    out = await wf.arun_episode(engine=None, data={"query_id": "q1"})
    assert out is not None
    assert branch_driver.drive_calls == []
    assert dispatch.dispatch_env_ids == ["base_env"]


@pytest.mark.asyncio
async def test_arun_episode_branch_without_driver_raises():
    wf = MultiAgentEnvDispatchWorkflow(
        dispatch_client=_RecordingDispatch(
            SweLegoSetup(rollouts=[])
        ),
        dag_client=_FakeDagClient(),
        assembler=_FakeAssembler(),
        resolver=_FakeResolver(),
        session_remover=_FakeSessionRemover(),
        base_env_id="base_env",  # no branch_driver configured
    )
    with pytest.raises(RuntimeError):
        await wf.arun_episode(
            engine=None, data={"branch_from_env_id": "source_env"}
        )

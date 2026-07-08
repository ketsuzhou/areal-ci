import httpx
import pytest

from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    DagError,
    DagForbidden,
    DagNotFound,
    DagTimeout,
    MulticaDagClient,
)


def _dag_payload() -> dict:
    return {
        "segments": [
            {
                "segment_id": "seg-1",
                "agent_run_id": "ar-1",
                "issue_id": "i-1",
                "trajectory_id": 0,
                "tensor_ref": {"shard_id": "sh-1", "node_addr": "http://dp"},
                "closing_event": "delegation",
                "env_snapshot": {
                    "sandbox_ids": [],
                    "issue_snapshot_id": "i-1",
                    "env_state": {},
                },
            }
        ],
        "edges": [
            {"src_segment_id": "seg-1", "dst_segment_id": "seg-2", "type": "delegation"}
        ],
        "session_to_agent_run": {"s-1": "ar-1"},
    }


def test_get_dag_polls_until_200():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        assert request.method == "GET"
        assert request.url.path == "/api/v1/env-dispatch/proj-1/dag"
        if calls["n"] < 2:
            return httpx.Response(202, json={"status": "in_progress"})
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient("http://multica", "key", _transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert dag.segments[0].segment_id == "seg-1"
    assert dag.segments[0].tensor_ref["shard_id"] == "sh-1"
    assert dag.segments[0].trajectory_id == 0
    assert dag.segments[0].env_snapshot["issue_snapshot_id"] == "i-1"
    assert dag.edges[0].type == "delegation"
    assert dag.session_to_agent_run == {"s-1": "ar-1"}
    assert calls["n"] == 2


def test_get_dag_404_raises():
    client = MulticaDagClient(
        "http://multica", "key", _transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with pytest.raises(DagNotFound):
        client.get_dag("proj-x", timeout=1.0, interval=0.0)


def test_get_dag_403_raises():
    client = MulticaDagClient(
        "http://multica", "key", _transport=httpx.MockTransport(lambda r: httpx.Response(403))
    )
    with pytest.raises(DagForbidden):
        client.get_dag("proj-x", timeout=1.0, interval=0.0)


def test_get_dag_timeout_raises():
    client = MulticaDagClient(
        "http://multica",
        "key",
        _transport=httpx.MockTransport(lambda r: httpx.Response(202)),
    )
    with pytest.raises(DagTimeout):
        client.get_dag("proj-1", timeout=0.0, interval=0.0)


def test_get_dag_other_status_raises_dag_error():
    client = MulticaDagClient(
        "http://multica",
        "key",
        _transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    with pytest.raises(DagError) as exc_info:
        client.get_dag("proj-1", timeout=1.0, interval=0.0)
    assert "500" in str(exc_info.value)

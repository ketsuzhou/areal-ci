import httpx
import pytest

from customized_areal.tree_search.agents import multica_auth
from customized_areal.tree_search.agents.multica_auth import save_credentials
from customized_areal.tree_search.agents.multica_client import EnvDispatchHandle
from customized_areal.tree_search.agents.multica_dag_client import (
    AssembledDag,
    DagError,
    DagForbidden,
    DagNotFound,
    DagTimeout,
    MulticaDagClient,
    StepReward,
)


@pytest.fixture(autouse=True)
def _default_multica_api_key(monkeypatch):
    monkeypatch.setenv("MULTICA_API_KEY", "mul_test")


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
            },
            {
                "segment_id": "seg-2",
                "agent_run_id": "ar-2",
                "issue_id": "i-2",
                "trajectory_id": 1,
                "tensor_ref": {"shard_id": "sh-2", "node_addr": "http://dp"},
                "closing_event": None,
                "env_snapshot": {
                    "sandbox_ids": [],
                    "issue_snapshot_id": "i-2",
                    "env_state": {},
                },
            },
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

    client = MulticaDagClient("http://multica", _transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert dag.segments[0].segment_id == "seg-1"
    assert dag.segments[0].tensor_ref["shard_id"] == "sh-1"
    assert dag.segments[0].trajectory_id == 0
    assert dag.segments[0].env_snapshot["issue_snapshot_id"] == "i-1"
    assert dag.edges[0].type == "delegation"
    assert dag.session_to_agent_run == {"s-1": "ar-1"}
    assert calls["n"] == 2


def test_assembled_dag_rejects_missing_diagnosis_turn_score():
    payload = _dag_payload()
    payload["segments"][0]["assistant_turn_seqs"] = [2, 7]
    payload["segments"][1]["assistant_turn_seqs"] = []
    payload["step_rewards"] = [
        {"segment_id": "seg-1", "seq": 2, "score": 8, "rationale": "good"},
    ]
    payload["score_max"] = 10

    dag = AssembledDag.from_dict(payload)
    with pytest.raises(DagError, match="missing diagnosis score"):
        dag.validate_diagnosis_coverage()


def test_assembled_dag_accepts_exact_diagnosis_turn_scores():
    payload = _dag_payload()
    payload["segments"][0]["assistant_turn_seqs"] = [2, 7]
    payload["segments"][1]["assistant_turn_seqs"] = []
    payload["step_rewards"] = [
        {"segment_id": "seg-1", "seq": 2, "score": 8, "rationale": "good"},
        {"segment_id": "seg-1", "seq": 7, "score": 5, "rationale": "partial"},
    ]
    payload["score_max"] = 10

    dag = AssembledDag.from_dict(payload)
    dag.validate_diagnosis_coverage()


def test_get_dag_repolls_200_with_no_segments_yet():
    # multica marks the root task terminal a couple of seconds before
    # CloseSegmentForEvent inserts the segment row. A 200 served in that window
    # carries an empty segments list; it must be re-polled, not treated as a
    # permanently segment-less DAG.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json={"segments": [], "edges": []})
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient("http://multica", _transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert calls["n"] == 3


def test_get_dag_repolls_200_status_failed():
    # A terminal root task whose session coverage is not yet dense is reported
    # as 200 {"status": "failed"}; coverage can still complete, so re-poll.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(200, json={"status": "failed"})
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient("http://multica", _transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert calls["n"] == 2


def test_get_dag_incomplete_until_deadline_raises_timeout():
    # When the DAG never completes, the failure must name the incompleteness
    # rather than surfacing as a structural "no segment" validation error.
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"status": "failed"})
        ),
    )
    with pytest.raises(DagTimeout) as excinfo:
        client.get_dag("proj-1", timeout=0.05, interval=0.0)
    assert "status=failed" in str(excinfo.value)


def test_get_dag_404_raises():
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(404)),
    )
    with pytest.raises(DagNotFound):
        client.get_dag("proj-x", timeout=1.0, interval=0.0)


def test_get_dag_403_raises():
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(403)),
    )
    with pytest.raises(DagForbidden):
        client.get_dag("proj-x", timeout=1.0, interval=0.0)


def test_get_dag_timeout_raises():
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(202)),
    )
    with pytest.raises(DagTimeout):
        client.get_dag("proj-1", timeout=0.0, interval=0.0)


def test_get_dag_other_status_raises_dag_error():
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    with pytest.raises(DagError) as exc_info:
        client.get_dag("proj-1", timeout=1.0, interval=0.0)
    assert "500" in str(exc_info.value)


def test_from_dict_still_rejects_empty_segments():
    # get_dag now re-polls an empty-segments 200 (it means "not assembled yet"),
    # so the structural guarantee is asserted on the validator directly.
    with pytest.raises(DagError, match="at least one segment"):
        AssembledDag.from_dict(
            {"segments": [], "edges": [], "session_to_agent_run": {}}
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "segments": _dag_payload()["segments"],
                "edges": [
                    {
                        "src_segment_id": "seg-1",
                        "dst_segment_id": "missing",
                        "type": "delegation",
                    }
                ],
                "session_to_agent_run": {},
            },
            "unknown destination",
        ),
        (
            {
                "segments": _dag_payload()["segments"],
                "edges": [
                    {
                        "src_segment_id": "seg-1",
                        "dst_segment_id": "seg-2",
                        "type": "delegation",
                    },
                    {
                        "src_segment_id": "seg-2",
                        "dst_segment_id": "seg-1",
                        "type": "delegation",
                    },
                ],
                "session_to_agent_run": {},
            },
            "cycle",
        ),
    ],
)
def test_get_dag_rejects_invalid_structure(payload, message):
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    )

    with pytest.raises(DagError, match=message):
        client.get_dag("proj-1", timeout=1.0, interval=0.0)


def test_get_dag_transport_error_drops_request_and_pat():
    secret = "mul_dag_secret"

    def handler(request):
        raise httpx.ConnectError(f"failed for {secret}", request=request)

    client = MulticaDagClient(
        "http://multica",
        api_key=secret,
        _transport=httpx.MockTransport(handler),
    )
    with pytest.raises(DagError, match="network request failed") as exc_info:
        client.get_dag("proj-1")

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert not hasattr(exc_info.value, "request")


def test_get_dag_backoff_grows_interval(monkeypatch):
    """The poll interval grows by the backoff factor up to the configured cap."""
    sleeps: list[float] = []
    monkeypatch.setattr(
        "customized_areal.tree_search.agents.multica_dag_client.time.sleep",
        lambda s: sleeps.append(s),
    )
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 4:
            return httpx.Response(202, json={"status": "in_progress"})
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient(
        "http://multica",
        poll_interval=1.0,
        poll_backoff=2.0,
        poll_max_interval=5.0,
        poll_timeout=100.0,
        _transport=httpx.MockTransport(handler),
    )
    # No per-call timeout/interval: client defaults drive polling.
    client.get_dag("proj-1")
    # 3 retries -> 3 sleeps. Interval grows 1.0 -> 2.0 -> 4.0 (capped at 5.0).
    assert len(sleeps) == 3
    assert sleeps[0] == pytest.approx(1.0)
    assert sleeps[1] == pytest.approx(2.0)
    assert sleeps[2] == pytest.approx(4.0)
    assert calls["n"] == 4


def test_get_dag_uses_client_defaults_without_overrides(monkeypatch):
    """get_dag without per-call kwargs uses the client's configured timeout."""
    monkeypatch.setattr(
        "customized_areal.tree_search.agents.multica_dag_client.time.sleep",
        lambda s: None,
    )

    def handler(request):
        return httpx.Response(202, json={"status": "in_progress"})

    client = MulticaDagClient(
        "http://multica",
        poll_timeout=0.0,  # immediate timeout
        _transport=httpx.MockTransport(handler),
    )
    with pytest.raises(DagTimeout):
        client.get_dag("proj-1")  # no timeout/interval kwargs


def test_get_dag_reads_direct_url_and_saved_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTICA_BASE_URL", "http://multica:8080")
    monkeypatch.delenv("AREAL_BRIDGE_STUB_URL", raising=False)
    monkeypatch.delenv("MULTICA_API_KEY")
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080", "mul_saved", credentials_path=path)
    monkeypatch.setattr(multica_auth, "DEFAULT_CREDENTIALS_PATH", path)
    seen: dict = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient(_transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert seen["url"].startswith("http://multica:8080")
    assert seen["auth"] == "Bearer mul_saved"


def test_get_dag_explicit_api_key_overrides_environment(monkeypatch):
    monkeypatch.setenv("MULTICA_API_KEY", "mul_env")

    def handler(request):
        assert request.headers["authorization"] == "Bearer mul_explicit"
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient(
        "http://multica",
        api_key="mul_explicit",
        _transport=httpx.MockTransport(handler),
    )
    client.get_dag("proj-1", timeout=5.0, interval=0.0)


def test_get_dag_401_points_to_login_without_leaking_pat():
    secret = "mul_dag_secret"
    client = MulticaDagClient(
        "http://multica",
        api_key=secret,
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text=f"rejected {secret}")
        ),
    )

    with pytest.raises(DagError, match="multica_auth login") as exc_info:
        client.get_dag("proj-1", timeout=1.0, interval=0.0)
    assert secret not in str(exc_info.value)


def test_get_dag_504_repolls_until_200():
    """A bridge 504 (timeout) is transient: re-poll rather than raise."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(504, json={"detail": "bridge timed out"})
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient("http://stub", _transport=httpx.MockTransport(handler))
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert calls["n"] == 2  # re-polled past the 504


def test_get_dag_requires_base_url_or_env(monkeypatch):
    monkeypatch.delenv("MULTICA_BASE_URL", raising=False)
    monkeypatch.delenv("AREAL_BRIDGE_STUB_URL", raising=False)
    with pytest.raises(ValueError):
        MulticaDagClient()


def test_assembled_dag_from_dict_parses_step_rewards_and_score_max():
    payload = _dag_payload()
    payload["step_rewards"] = [
        {"segment_id": "seg-1", "seq": 1, "score": 8, "rationale": "good"},
        {"segment_id": "seg-1", "seq": 2, "score": 6, "rationale": "ok"},
    ]
    payload["score_max"] = 10
    dag = AssembledDag.from_dict(payload)
    assert dag.score_max == 10
    assert len(dag.step_rewards) == 2
    assert isinstance(dag.step_rewards[0], StepReward)
    assert dag.step_rewards[0].segment_id == "seg-1"
    assert dag.step_rewards[0].seq == 1
    assert dag.step_rewards[0].score == 8
    assert dag.step_rewards[0].rationale == "good"


def test_assembled_dag_from_dict_defaults_step_rewards_absent():
    # When the diagnosis agent did not run, /dag omits step_rewards + score_max.
    # Absence stays distinguishable: empty list + 0 (no fabricated defaults).
    dag = AssembledDag.from_dict(_dag_payload())
    assert dag.step_rewards == []
    assert dag.score_max == 0


def test_get_dag_message_handle_routes_channel_first():
    """A message-dispatch handle polls the channel-scoped DAG route."""
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient("http://multica", _transport=httpx.MockTransport(handler))
    handle = EnvDispatchHandle(
        channel_id="c1",
        project_id="p1",
        env_id="e1",
        dispatch_type="message",
    )
    dag = client.get_dag(handle, timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)
    assert seen["path"] == "/api/v1/env-dispatch/channels/c1/dag"


def test_get_dag_message_handle_missing_channel_id_raises():
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_dag_payload())
        ),
    )
    bad = EnvDispatchHandle(
        channel_id=None, project_id="p1", env_id="", dispatch_type="message"
    )
    with pytest.raises(DagError, match="missing channel_id"):
        client.get_dag(bad, timeout=1.0, interval=0.0)


def test_get_dag_issue_handle_routes_project_first():
    """An issue-dispatch handle keeps the legacy project-scoped DAG route."""

    def handler(request):
        assert request.url.path == "/api/v1/env-dispatch/p1/dag"
        return httpx.Response(200, json=_dag_payload())

    client = MulticaDagClient("http://multica", _transport=httpx.MockTransport(handler))
    handle = EnvDispatchHandle(
        channel_id=None, project_id="p1", env_id="", dispatch_type="issue"
    )
    dag = client.get_dag(handle, timeout=5.0, interval=0.0)
    assert isinstance(dag, AssembledDag)


# ── Task 5: dual-source (mixed) DAG tests ────────────────────────────


def _mixed_dag_payload() -> dict:
    """Mixed DAG: one areal_tensor segment + one task_messages segment."""
    return {
        "segments": [
            {
                "segment_id": "seg-areal-1",
                "agent_run_id": "ar-1",
                "issue_id": "i-1",
                "trajectory_id": 42,
                "tensor_ref": {
                    "input_ids": {"shard_id": "sh-1", "node_addr": "http://dp"}
                },
                "closing_event": None,
                "env_snapshot": {
                    "sandbox_ids": [],
                    "issue_snapshot_id": "i-1",
                    "env_state": {},
                },
                "trajectory_source": "areal_tensor",
                "trainable": True,
                "trajectory": [],
            },
            {
                "segment_id": "seg-local-1",
                "agent_run_id": "ar-2",
                "issue_id": "i-1",
                "trajectory_id": None,
                "tensor_ref": None,
                "closing_event": "completion",
                "env_snapshot": {
                    "sandbox_ids": [],
                    "issue_snapshot_id": "i-1",
                    "env_state": {},
                },
                "trajectory_source": "task_messages",
                "trainable": False,
                "trajectory": [{"sequence": 1, "type": "user", "content": "hello"}],
            },
        ],
        "edges": [
            {
                "src_segment_id": "seg-areal-1",
                "dst_segment_id": "seg-local-1",
                "type": "completion",
            }
        ],
        "session_to_agent_run": {"s-1": "ar-1"},
    }


def test_mixed_dag_parses_dual_source_segments():
    """Both areal_tensor and task_messages segments parse correctly."""
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_mixed_dag_payload())
        ),
    )
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)

    assert len(dag.segments) == 2

    # areal_tensor segment
    areal_seg = dag.segments[0]
    assert areal_seg.segment_id == "seg-areal-1"
    assert areal_seg.trajectory_source == "areal_tensor"
    assert areal_seg.trainable is True
    assert areal_seg.trajectory_id == 42
    assert areal_seg.tensor_ref is not None
    assert areal_seg.trajectory == []

    # task_messages segment
    local_seg = dag.segments[1]
    assert local_seg.segment_id == "seg-local-1"
    assert local_seg.trajectory_source == "task_messages"
    assert local_seg.trainable is False
    assert local_seg.trajectory_id is None
    assert local_seg.tensor_ref is None
    assert local_seg.trajectory == [{"sequence": 1, "type": "user", "content": "hello"}]

    # edges preserved
    assert len(dag.edges) == 1
    assert dag.edges[0].src_segment_id == "seg-areal-1"
    assert dag.edges[0].dst_segment_id == "seg-local-1"


def test_areal_tensor_segment_missing_trajectory_id_raises():
    """areal_tensor segments require non-null trajectory_id."""
    payload = _mixed_dag_payload()
    payload["segments"][0]["trajectory_id"] = None

    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    with pytest.raises(DagError, match="missing trajectory_id"):
        client.get_dag("proj-1", timeout=5.0, interval=0.0)


def test_areal_tensor_segment_missing_tensor_ref_raises():
    """areal_tensor segments require non-null tensor_ref."""
    payload = _mixed_dag_payload()
    payload["segments"][0]["tensor_ref"] = None

    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    with pytest.raises(DagError, match="missing tensor_ref"):
        client.get_dag("proj-1", timeout=5.0, interval=0.0)


def test_task_messages_segment_with_unexpected_ids_raises():
    """task_messages segments must have null trajectory_id and tensor_ref."""
    payload = _mixed_dag_payload()
    payload["segments"][1]["trajectory_id"] = 99

    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    with pytest.raises(DagError, match="unexpected trajectory_id"):
        client.get_dag("proj-1", timeout=5.0, interval=0.0)


def test_task_messages_segment_with_unexpected_tensor_ref_raises():
    """task_messages segments must have null tensor_ref."""
    payload = _mixed_dag_payload()
    payload["segments"][1]["tensor_ref"] = {"shard_id": "bad"}

    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    with pytest.raises(DagError, match="unexpected tensor_ref"):
        client.get_dag("proj-1", timeout=5.0, interval=0.0)


def test_backward_compat_segments_default_to_areal_tensor_trainable():
    """Segments without trajectory_source/trainable default to areal_tensor/True."""
    client = MulticaDagClient(
        "http://multica",
        _transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_dag_payload())
        ),
    )
    dag = client.get_dag("proj-1", timeout=5.0, interval=0.0)
    for seg in dag.segments:
        assert seg.trajectory_source == "areal_tensor"
        assert seg.trainable is True
        assert seg.trajectory_id is not None
        assert seg.tensor_ref is not None

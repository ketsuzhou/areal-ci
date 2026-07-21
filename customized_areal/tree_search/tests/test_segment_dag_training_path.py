"""Tests for the minimal segment-DAG training plumbing + tensor lifecycle (U5).

The orchestration test is torch/infra-free: a FakeResolver returns plain
dicts, ``assemble_from_refs`` stays in pure Python, and
``assemble_node_advantages`` is float math - so the whole get_dag -> assemble
-> GAE -> cleanup path runs without torch/FSDP. The concrete data_proxy
clients are exercised via ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import orjson
import pytest

from customized_areal.tree_search.agents.execution_dag import DAGError
from customized_areal.tree_search.agents.multica_dag_client import (
    DagError,
    MulticaDagClient,
)
from customized_areal.tree_search.agents.segment_dag_trainer import (
    DataProxySessionRemover,
    DataProxyTensorResolver,
    run_segment_dag_training_step,
)


class _FakeResolver:
    """Records calls; returns a fixed tensor dict per shard_id."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cleared: list[list[str]] = []

    def resolve(self, tensor_ref: dict[str, Any]) -> dict[str, Any]:
        shard_id = tensor_ref["input_ids"]["shard_id"]
        self.calls.append(shard_id)
        return {
            "input_ids": [1, 2],
            "loss_mask": [1, 1],
            "logprobs": [0.0, 0.0],
            "versions": [1, 1],
            "attention_mask": [1, 1],
            "rewards": [0.0, 0.0],
        }

    def clear(self, shard_ids: list[str]) -> None:
        self.cleared.append(list(shard_ids))


class _FakeSessionRemover:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove(self, session_id: str) -> None:
        self.removed.append(session_id)


def _dag_json(segments: list[dict], edges: list[dict], s2ar: dict) -> dict:
    return {
        "segments": segments,
        "edges": edges,
        "session_to_agent_run": s2ar,
    }


def _seg(segment_id: str, agent_run_id: str, shard_id: str) -> dict:
    return {
        "segment_id": segment_id,
        "agent_run_id": agent_run_id,
        "issue_id": "issue-1",
        "trajectory_id": 0,
        "tensor_ref": {"input_ids": {"shard_id": shard_id, "node_addr": "node-a"}},
        "closing_event": None,
        "env_snapshot": {},
    }


def _dag_client(handler) -> MulticaDagClient:
    return MulticaDagClient(
        "http://multica",
        api_key="mul_test",
        _transport=httpx.MockTransport(handler),
    )


def test_run_segment_dag_training_step_zero_reward_path():
    dag = _dag_json(
        segments=[_seg("s1", "ar1", "shard-1"), _seg("s2", "ar2", "shard-2")],
        edges=[{"src_segment_id": "s1", "dst_segment_id": "s2", "type": "completion"}],
        s2ar={"sess-1": "ar1", "sess-2": "ar2"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dag)

    client = _dag_client(handler)
    resolver = _FakeResolver()
    remover = _FakeSessionRemover()

    advantages = run_segment_dag_training_step(
        client=client,
        resolver=resolver,
        session_remover=remover,
        project_id="proj-1",
    )

    # One SuperNode per segment, both with zero reward -> zero advantage.
    assert set(advantages.advantages) == {"s1", "s2"}
    assert all(v == 0.0 for v in advantages.advantages.values())
    assert all(v == 0.0 for v in advantages.returns.values())

    # Every segment's tensor_ref was resolved.
    assert resolver.calls == ["shard-1", "shard-2"]

    # Cleanup: shards cleared once with both ids; both sessions revoked.
    assert resolver.cleared == [["shard-1", "shard-2"]]
    assert remover.removed == ["sess-1", "sess-2"]


def test_run_segment_dag_training_step_cycle_propagates_without_cleanup():
    # s1 -> s2 -> s1 mention cycle: topological_order() raises DAGError.
    dag = _dag_json(
        segments=[_seg("s1", "ar1", "shard-1"), _seg("s2", "ar2", "shard-2")],
        edges=[
            {"src_segment_id": "s1", "dst_segment_id": "s2", "type": "mention"},
            {"src_segment_id": "s2", "dst_segment_id": "s1", "type": "mention"},
        ],
        s2ar={"sess-1": "ar1"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dag)

    client = _dag_client(handler)
    resolver = _FakeResolver()
    remover = _FakeSessionRemover()

    # A failed step propagates the error WITHOUT releasing shards/sessions
    # (cleanup is success-path only in change 1; the caller handles retry).
    # Cycle is detected by AssembledDag.from_dict (DagError) before the
    # assembler or tensor resolution is ever reached.
    with pytest.raises(DagError):
        run_segment_dag_training_step(
            client=client,
            resolver=resolver,
            session_remover=remover,
            project_id="proj-1",
        )

    assert resolver.cleared == []
    assert remover.removed == []


def test_data_proxy_tensor_resolver_resolves_and_clears():
    from areal.infra.rpc.serialization import serialize_value

    # Multi-shard contract: each field maps to its own shard; resolve fetches
    # every shard and reassembles a {field: tensor} dict.
    shards = {"shard-9": [1, 2, 3], "shard-10": [1, 1, 0]}
    bodies = {sid: orjson.dumps(serialize_value(val)) for sid, val in shards.items()}
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/data/" in str(request.url):
            sid = str(request.url).rsplit("/data/", 1)[1]
            seen.append(("GET", sid))
            if sid in bodies:
                return httpx.Response(
                    200,
                    content=bodies[sid],
                    headers={"content-type": "application/octet-stream"},
                )
            return httpx.Response(404)
        if request.method == "DELETE" and "/data/clear" in str(request.url):
            payload = json.loads(request.content.decode())
            seen.append(("DELETE", json.dumps(payload)))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    resolver = DataProxyTensorResolver(
        "http://proxy", _transport=httpx.MockTransport(handler)
    )

    resolved = resolver.resolve(
        {
            "input_ids": {"shard_id": "shard-9", "node_addr": "node-a"},
            "loss_mask": {"shard_id": "shard-10", "node_addr": "node-a"},
        }
    )
    assert resolved == {"input_ids": [1, 2, 3], "loss_mask": [1, 1, 0]}

    resolver.clear(["shard-9", "shard-10"])
    assert seen[0] == ("GET", "shard-9")
    assert seen[1] == ("GET", "shard-10")
    assert seen[2][0] == "DELETE"
    cleared = json.loads(seen[2][1])
    assert cleared == {"shard_ids": ["shard-9", "shard-10"]}


def test_data_proxy_tensor_resolver_404_raises_keyerror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    resolver = DataProxyTensorResolver(
        "http://proxy", _transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyError):
        resolver.resolve({"input_ids": {"shard_id": "missing", "node_addr": "node-a"}})


def test_data_proxy_tensor_resolver_clear_noop_on_empty():
    # No shard ids -> no HTTP call at all.
    called = {"hit": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hit"] = True
        return httpx.Response(200)

    resolver = DataProxyTensorResolver(
        "http://proxy", _transport=httpx.MockTransport(handler)
    )
    resolver.clear([])
    assert called["hit"] is False


def test_data_proxy_session_remover_calls_export_remove_session():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and "/export_trajectories" in str(request.url):
            seen.append(json.loads(request.content.decode()))
            return httpx.Response(200, json={"traj": {}})
        return httpx.Response(404)

    remover = DataProxySessionRemover(
        "http://proxy", _transport=httpx.MockTransport(handler)
    )
    remover.remove("sess-7")

    assert len(seen) == 1
    assert seen[0]["session_ids"] == ["sess-7"]
    assert seen[0]["remove_session"] is True


# ── Task 5: dual-source (mixed) DAG training path tests ──────────────


def _mixed_training_dag_json() -> dict:
    """Mixed DAG JSON: one areal_tensor + one task_messages segment."""
    return {
        "segments": [
            {
                "segment_id": "s-train",
                "agent_run_id": "ar-train",
                "issue_id": "issue-1",
                "trajectory_id": 42,
                "tensor_ref": {
                    "input_ids": {"shard_id": "shard-train", "node_addr": "node-a"}
                },
                "closing_event": None,
                "env_snapshot": {},
                "trajectory_source": "areal_tensor",
                "trainable": True,
                "trajectory": [],
            },
            {
                "segment_id": "s-local",
                "agent_run_id": "ar-local",
                "issue_id": "issue-1",
                "trajectory_id": None,
                "tensor_ref": None,
                "closing_event": "completion",
                "env_snapshot": {},
                "trajectory_source": "task_messages",
                "trainable": False,
                "trajectory": [
                    {"sequence": 1, "type": "user", "content": "hello"}
                ],
            },
        ],
        "edges": [
            {
                "src_segment_id": "s-train",
                "dst_segment_id": "s-local",
                "type": "completion",
            }
        ],
        "session_to_agent_run": {"sess-train": "ar-train"},
    }


def test_mixed_dag_cleanup_only_targets_trainable_segments():
    """Only trainable segments' shards are cleared; only trainable sessions removed."""
    dag_json = _mixed_training_dag_json()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dag_json)

    client = _dag_client(handler)
    resolver = _FakeResolver()
    remover = _FakeSessionRemover()

    advantages = run_segment_dag_training_step(
        client=client,
        resolver=resolver,
        session_remover=remover,
        project_id="proj-1",
    )

    # Both segments appear in advantages (non-trainable has zero reward -> zero advantage)
    assert set(advantages.advantages) == {"s-train", "s-local"}

    # Only trainable segment's shard was resolved
    assert resolver.calls == ["shard-train"]

    # Only trainable segment's shards are cleared
    assert resolver.cleared == [["shard-train"]]

    # Only the trainable agent_run's session is removed
    assert remover.removed == ["sess-train"]


def test_mixed_dag_non_trainable_never_reaches_cleanup():
    """Non-trainable segments are never passed to cleanup (clear/remove)."""
    dag_json = _mixed_training_dag_json()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dag_json)

    client = _dag_client(handler)
    resolver = _FakeResolver()
    remover = _FakeSessionRemover()

    run_segment_dag_training_step(
        client=client,
        resolver=resolver,
        session_remover=remover,
        project_id="proj-1",
    )

    # Verify non-trainable shard/session identifiers never appeared in cleanup
    all_cleared_shards = {sid for batch in resolver.cleared for sid in batch}
    assert "shard-train" in all_cleared_shards
    # No shard from the local segment was cleared (it has no tensor_ref at all)
    assert len(all_cleared_shards) == 1

    # The local agent_run's session was not removed
    assert "ar-local" not in remover.removed
    # Only the trainable session was cleaned up
    assert remover.removed == ["sess-train"]

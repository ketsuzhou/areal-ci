import asyncio
import json

import httpx
import pytest

from customized_areal.tree_search.agents.multica_client import (
    MulticaCheckpointError,
    MulticaEnvDispatchClient,
)
from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue


def _transport(handler):
    return httpx.MockTransport(handler)


def test_create_env_dispatch_scratch_swe_lego():
    def handler(req):
        body = json.loads(req.content)
        assert body["mode"] == "scratch"
        assert body["domain"] == "swe_lego"
        assert body["dispatch_type"] == "issue"
        assert body["group_size"] == 2
        return httpx.Response(201, json={"project_id": "p1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    issue = SweLegoIssue(
        repo_url="r",
        base_commit="c",
        issue_date="d",
        issue_text="x",
        issue_title="t",
        acceptance_criteria="a",
        fail_to_pass=["f"],
        pass_to_pass=["p"],
    )
    project_id = asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            env_id="base",
            dispatch_type="issue",
            agent_id="ag",
            group_size=2,
            domain="swe_lego",
            issue=issue,
        )
    )
    assert project_id == "p1"


def test_cleanup_env_dispatch_hits_renamed_url():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        return httpx.Response(204)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.cleanup_env_dispatch(project_id="p1"))
    assert seen["path"] == "/api/v1/env-dispatch/p1"


def test_create_env_dispatch_squad_omits_agent_and_env():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            env_id=None,
            dispatch_type="message",
            agent_id=None,
            squad_id="sq-1",
            group_size=1,
            domain="self_play",
            message="hi",
        )
    )
    assert "env_id" not in seen["body"]
    assert "agent_id" not in seen["body"]
    assert seen["body"]["squad_id"] == "sq-1"
    assert seen["body"]["mode"] == "scratch"


def test_create_env_dispatch_resume_passes_checkpoint_as_env_id():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    # Resume-from-checkpoint has no separate endpoint: the checkpoint id is carried
    # by env_id on a mode="resume" dispatch.
    asyncio.run(
        c.create_env_dispatch(
            mode="resume",
            env_id="cp-1",
            dispatch_type="issue",
            agent_id="ag",
            group_size=1,
            domain="swe_lego",
        )
    )
    assert seen["body"]["mode"] == "resume"
    assert seen["body"]["env_id"] == "cp-1"


def test_create_env_dispatch_serializes_per_agent_env_specs():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            dispatch_type="message",
            squad_id="sq",
            domain="self_play",
            message="hi",
            per_agent_env={"agent-1": {"template": "python"}},
        )
    )
    assert seen["body"]["per_agent_env"] == {"agent-1": {"template": "python"}}


def test_create_env_dispatch_omits_per_agent_env_when_empty():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            dispatch_type="message",
            squad_id="sq",
            domain="self_play",
            message="hi",
        )
    )
    assert "per_agent_env" not in seen["body"]


def test_create_checkpoint_posts_sync_timeout_fields():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["path"] = req.url.path
        return httpx.Response(
            201,
            json={
                "id": "cp-1",
                "project_id": "p1",
                "save_status": "complete",
                "save_timeout_ms": 5000,
            },
        )

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    result = asyncio.run(
        c.create_checkpoint(
            project_id="p1",
            event_ref="evt-1",
            checkpoint_kind="entropy_gated",
            env_id_map={"a1": "env-1"},
            sandbox_refs=[{"instance_id": "inst-1", "workspace_id": "ws"}],
            entropy_score=1.5,
            save_timeout_ms=5000,
        )
    )
    assert seen["path"] == "/api/v1/env-checkpoints"
    assert seen["body"]["project_id"] == "p1"
    assert seen["body"]["checkpoint_kind"] == "entropy_gated"
    assert seen["body"]["entropy_score"] == 1.5
    assert seen["body"]["save_timeout_ms"] == 5000
    assert seen["body"]["sandbox_refs"] == [
        {"instance_id": "inst-1", "workspace_id": "ws"}
    ]
    assert result["id"] == "cp-1"


def test_list_checkpoints_returns_items():
    def handler(req):
        assert req.url.path == "/api/v1/projects/p1/env-checkpoints"
        return httpx.Response(
            200,
            json={
                "checkpoints": [
                    {"id": "cp-1", "save_status": "complete"},
                    {"id": "cp-2", "save_status": "timed_out"},
                ]
            },
        )

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    items = asyncio.run(c.list_checkpoints(project_id="p1"))
    assert len(items) == 2
    assert items[0]["id"] == "cp-1"
    assert items[1]["save_status"] == "timed_out"


def test_checkpoint_conflict_raises_typed_error():
    def handler(req):
        return httpx.Response(409, json={"error": "checkpoint save timed out"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    with pytest.raises(MulticaCheckpointError) as exc_info:
        asyncio.run(
            c.create_checkpoint(
                project_id="p1",
                event_ref="evt-1",
                checkpoint_kind="structural",
                env_id_map={},
                sandbox_refs=[],
            )
        )
    assert exc_info.value.status_code == 409


def test_maybe_create_entropy_checkpoint_creates_when_above_threshold():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"id": "cp-1", "save_status": "complete"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    result = asyncio.run(
        c.maybe_create_entropy_checkpoint(
            project_id="p1",
            event_ref="evt-1",
            env_id_map={"a1": "env-1"},
            sandbox_refs=[{"instance_id": "inst-1"}],
            entropy=1.5,
            threshold=1.0,
        )
    )
    assert result is not None
    assert result["id"] == "cp-1"
    assert seen["body"]["checkpoint_kind"] == "entropy_gated"
    assert seen["body"]["entropy_score"] == 1.5


def test_maybe_create_entropy_checkpoint_skips_when_logprobs_missing():
    called = {"count": 0}

    def handler(req):
        called["count"] += 1
        return httpx.Response(201, json={"id": "cp-1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    result = asyncio.run(
        c.maybe_create_entropy_checkpoint(
            project_id="p1",
            event_ref="evt-1",
            env_id_map={},
            sandbox_refs=[],
            entropy=None,  # logprobs unavailable
            threshold=1.0,
        )
    )
    assert result is None
    assert called["count"] == 0  # no API call made


def test_maybe_create_entropy_checkpoint_skips_when_below_threshold():
    called = {"count": 0}

    def handler(req):
        called["count"] += 1
        return httpx.Response(201, json={"id": "cp-1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    result = asyncio.run(
        c.maybe_create_entropy_checkpoint(
            project_id="p1",
            event_ref="evt-1",
            env_id_map={},
            sandbox_refs=[],
            entropy=0.2,
            threshold=1.0,
        )
    )
    assert result is None
    assert called["count"] == 0

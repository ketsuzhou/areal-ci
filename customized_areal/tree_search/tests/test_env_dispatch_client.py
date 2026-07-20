import asyncio
import json

import httpx
import pytest

from customized_areal.tree_search.agents import multica_auth
from customized_areal.tree_search.agents.multica_auth import save_credentials
from customized_areal.tree_search.agents.multica_client import (
    EnvDispatchHandle,
    MulticaCheckpointError,
    MulticaEnvDispatchClient,
    build_debug_parser,
    main,
)
from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _default_multica_api_key(monkeypatch):
    monkeypatch.setenv("MULTICA_API_KEY", "mul_test")


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
    handle = asyncio.run(
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
    assert handle.project_id == "p1"
    assert handle.dispatch_type == "issue"
    assert handle.channel_id is None


def test_create_env_dispatch_sends_environment_api_key(monkeypatch):
    monkeypatch.setenv("MULTICA_API_KEY", "mul_dispatch")

    def handler(request):
        assert request.headers["authorization"] == "Bearer mul_dispatch"
        return httpx.Response(201, json={"project_id": "p1"})

    client = MulticaEnvDispatchClient(
        base_url="http://stub", transport=_transport(handler)
    )
    handle = asyncio.run(
        client.create_env_dispatch(
            mode="scratch", dispatch_type="issue", agent_id="agent-1"
        )
    )
    assert handle.project_id == "p1"


def test_create_env_dispatch_sends_saved_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("MULTICA_API_KEY")
    path = tmp_path / "credentials.json"
    save_credentials("http://multica", "mul_saved", credentials_path=path)
    monkeypatch.setattr(multica_auth, "DEFAULT_CREDENTIALS_PATH", path)

    def handler(request):
        assert request.headers["authorization"] == "Bearer mul_saved"
        return httpx.Response(201, json={"project_id": "p1"})

    client = MulticaEnvDispatchClient(
        base_url="http://multica", transport=_transport(handler)
    )
    handle = asyncio.run(
        client.create_env_dispatch(
            mode="scratch", dispatch_type="issue", agent_id="agent-1"
        )
    )
    assert handle.project_id == "p1"


def test_create_env_dispatch_message_returns_channel_first_handle():
    """A message dispatch must surface channel_id and route channel-first."""
    seen_paths: list[str] = []

    def handler(req):
        seen_paths.append(req.url.path)
        if req.method == "POST":
            return httpx.Response(
                201,
                json={
                    "channel_id": "c1",
                    "project_id": "p1",
                    "rollouts": [
                        {"channel_id": "c1", "project_id": "p1", "env_id": "e1"}
                    ],
                },
            )
        if req.method == "GET" and req.url.path.endswith("/dag"):
            return httpx.Response(200, json={"nodes": []})
        if req.method == "GET" and req.url.path.endswith("/env-checkpoints"):
            return httpx.Response(200, json={"checkpoints": []})
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    handle = asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            dispatch_type="message",
            agent_id="ag",
            domain="self_play",
            message="hi",
        )
    )
    assert handle.channel_id == "c1"
    assert handle.project_id == "p1"
    assert handle.env_id == "e1"
    assert handle.dispatch_type == "message"
    assert handle.primary_id == "c1"

    # Only the lifecycle routes are asserted below; the create POST was already
    # recorded above and is cleared so the assertion is exact.
    seen_paths.clear()
    asyncio.run(c.get_dag(handle=handle))
    asyncio.run(c.list_checkpoints(handle=handle))
    asyncio.run(c.cleanup_env_dispatch(handle=handle))

    assert seen_paths == [
        "/api/v1/env-dispatch/channels/c1/dag",
        "/api/v1/channels/c1/env-checkpoints",
        "/api/v1/env-dispatch/channels/c1",
    ]


def test_create_env_dispatch_message_missing_channel_id_raises():
    def handler(req):
        # Message dispatch response without channel_id is a contract violation.
        return httpx.Response(201, json={"project_id": "p1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    with pytest.raises(RuntimeError, match="missing channel_id"):
        asyncio.run(
            c.create_env_dispatch(
                mode="scratch",
                dispatch_type="message",
                agent_id="ag",
                domain="self_play",
                message="hi",
            )
        )


def test_create_env_dispatch_401_points_to_login_without_leaking_pat():
    secret = "mul_dispatch_secret"

    def handler(request):
        return httpx.Response(401, text=f"rejected {secret}")

    client = MulticaEnvDispatchClient(
        base_url="http://multica",
        api_key=secret,
        transport=_transport(handler),
    )
    with pytest.raises(RuntimeError, match="multica_auth login") as exc_info:
        asyncio.run(
            client.create_env_dispatch(
                mode="scratch", dispatch_type="issue", agent_id="agent-1"
            )
        )
    assert secret not in str(exc_info.value)


def test_create_env_dispatch_error_never_leaks_pat():
    secret = "mul_dispatch_secret"

    client = MulticaEnvDispatchClient(
        base_url="http://multica",
        api_key=secret,
        transport=_transport(
            lambda request: httpx.Response(500, text=f"failed for {secret}")
        ),
    )
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            client.create_env_dispatch(
                mode="scratch", dispatch_type="issue", agent_id="agent-1"
            )
        )
    assert secret not in str(exc_info.value)
    assert "failed for [REDACTED]" in str(exc_info.value)


def test_create_env_dispatch_redacts_before_truncating_error_body():
    secret = "mul_boundary_secret"
    prefix = "x" * 4090
    client = MulticaEnvDispatchClient(
        base_url="http://multica",
        api_key=secret,
        transport=_transport(
            lambda request: httpx.Response(500, text=f"{prefix}{secret}")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            client.create_env_dispatch(
                mode="scratch", dispatch_type="issue", agent_id="agent-1"
            )
        )

    assert secret not in str(exc_info.value)
    assert "mul_bo" not in str(exc_info.value)


def test_create_env_dispatch_transport_error_drops_request_and_pat():
    secret = "mul_dispatch_secret"

    def handler(request):
        raise httpx.ConnectError(f"failed for {secret}", request=request)

    client = MulticaEnvDispatchClient(
        base_url="http://multica",
        api_key=secret,
        transport=_transport(handler),
    )
    with pytest.raises(RuntimeError, match="network request failed") as exc_info:
        asyncio.run(
            client.create_env_dispatch(
                mode="scratch", dispatch_type="issue", agent_id="agent-1"
            )
        )

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert not hasattr(exc_info.value, "request")


def test_cleanup_env_dispatch_hits_renamed_url():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        return httpx.Response(204)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(
        c.cleanup_env_dispatch(
            handle=EnvDispatchHandle(
                channel_id=None,
                project_id="p1",
                env_id="",
                dispatch_type="issue",
            )
        )
    )
    assert seen["path"] == "/api/v1/env-dispatch/p1"


def test_create_env_dispatch_squad_omits_agent_and_env():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1", "channel_id": "c1"})

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
        return httpx.Response(201, json={"project_id": "p1", "channel_id": "c1"})

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


def test_create_env_dispatch_serializes_per_agent_env_runtime():
    """A nested external-model runtime policy is passed through verbatim.

    The client performs generic per_agent_env serialization, so the nested
    runtime object (base_url/api_key/model) reaches the server unchanged. The
    API key is intentionally present in the request body - the server needs it
    to start the sandbox - and non-disclosure is enforced server-side on
    responses, errors, and logs, not on the outbound request.
    """
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1", "channel_id": "c1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    runtime_policy = {
        "agent-1": {
            "runtime": {
                "base_url": "https://provider.invalid/v1",
                "api_key": "synthetic-secret-for-tests",
                "model": "model-a",
            }
        }
    }
    asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            dispatch_type="message",
            squad_id="sq",
            domain="self_play",
            message="hi",
            per_agent_env=runtime_policy,
        )
    )
    assert seen["body"]["per_agent_env"] == runtime_policy


def test_create_env_dispatch_omits_per_agent_env_when_empty():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1", "channel_id": "c1"})

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


def test_create_env_dispatch_serializes_train_agent_id():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1", "channel_id": "c1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(
        c.create_env_dispatch(
            mode="scratch",
            dispatch_type="message",
            squad_id="sq",
            domain="self_play",
            train_agent_id="agent-1",
            message="hi",
        )
    )
    assert seen["body"]["train_agent_id"] == "agent-1"


def test_create_env_dispatch_omits_train_agent_id_when_empty():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(201, json={"project_id": "p1", "channel_id": "c1"})

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
    assert "train_agent_id" not in seen["body"]


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
    items = asyncio.run(
        c.list_checkpoints(
            handle=EnvDispatchHandle(
                channel_id=None,
                project_id="p1",
                env_id="",
                dispatch_type="issue",
            )
        )
    )
    assert len(items) == 2
    assert items[0]["id"] == "cp-1"
    assert items[1]["save_status"] == "timed_out"


def test_list_checkpoints_message_handle_routes_channel_first():
    def handler(req):
        assert req.url.path == "/api/v1/channels/c1/env-checkpoints"
        return httpx.Response(200, json={"checkpoints": [{"id": "cp-1"}]})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    items = asyncio.run(
        c.list_checkpoints(
            handle=EnvDispatchHandle(
                channel_id="c1", project_id="p1", env_id="e1", dispatch_type="message"
            )
        )
    )
    assert items == [{"id": "cp-1"}]


def test_list_checkpoints_message_handle_missing_channel_id_raises():
    c = MulticaEnvDispatchClient(
        base_url="http://x",
        transport=_transport(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(RuntimeError, match="missing channel_id"):
        asyncio.run(
            c.list_checkpoints(
                handle=EnvDispatchHandle(
                    channel_id=None,
                    project_id="p1",
                    env_id="",
                    dispatch_type="message",
                )
            )
        )


def test_checkpoint_conflict_raises_typed_error():
    secret = "mul_checkpoint_secret"

    def handler(req):
        return httpx.Response(409, text=f"checkpoint failed for {secret}")

    c = MulticaEnvDispatchClient(
        base_url="http://x", api_key=secret, transport=_transport(handler)
    )
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
    assert secret not in str(exc_info.value)
    assert secret not in exc_info.value.body
    assert exc_info.value.body == "checkpoint failed for [REDACTED]"


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


def test_debug_parser_has_no_shared_deployment_defaults(monkeypatch):
    for key in (
        "MULTICA_BASE_URL",
        "MULTICA_WORKSPACE_SLUG",
        "MULTICA_WORKSPACE_ID",
        "MULTICA_AGENT_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    args = build_debug_parser().parse_args([])

    assert args.base_url is None
    assert args.workspace_slug is None
    assert args.workspace_id is None
    assert args.agent_id is None


def test_debug_main_requires_deployment_coordinates(monkeypatch, capsys):
    for key in (
        "MULTICA_BASE_URL",
        "MULTICA_WORKSPACE_SLUG",
        "MULTICA_WORKSPACE_ID",
        "MULTICA_AGENT_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(SystemExit) as exc:
        main(["--agent-id", "agent"])

    assert exc.value.code == 2
    assert "--base-url" in capsys.readouterr().err

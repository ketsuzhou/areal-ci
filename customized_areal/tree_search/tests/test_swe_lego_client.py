import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoSetup,
)
from customized_areal.tree_search.agents.swe_lego_client import MulticaSweLegoClient

Handler = Callable[[httpx.Request], "httpx.Response | Awaitable[httpx.Response]"]


def _router(routes: dict[tuple[str, str], Handler]) -> httpx.MockTransport:
    def dispatch(request: httpx.Request):
        path = request.url.path
        for (method, route), handler in routes.items():
            if request.method != method:
                continue
            if route == path or (route.endswith("*") and path.startswith(route[:-1])):
                return handler(request)
        return httpx.Response(
            404, json={"error": f"no route for {request.method} {path}"}
        )

    return httpx.MockTransport(dispatch)


def _issue() -> SweLegoIssue:
    return SweLegoIssue(
        repo_url="https://github.com/psf/requests.git",
        base_commit="abc123",
        issue_date="2025-03-14T09:30:00Z",
        issue_text="retry leaks",
        issue_title="Retry leaks",
        acceptance_criteria="must not leak",
        fail_to_pass=["tests/test_retry.py::test_leak"],
        pass_to_pass=["tests/test_retry.py::test_basic"],
    )


def test_create_swe_lego_issue_posts_to_atomic_endpoint():
    captured: dict = {}

    def handler(request: httpx.Request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            201,
            json={
                "project_id": "p1",
                "issue_id": "i1",
                "image_id": "img1",
                "build_node_id": "n1",
                "base_sandbox_id": "sbx-base",
                "base_sandbox_runtime_id": "rt-base",
                "agent_run_ids": ["r1", "r2"],
            },
        )

    transport = _router({("POST", "/api/v1/swe-lego/issues"): handler})
    client = MulticaSweLegoClient(
        base_url="https://multica.example", api_key="secret", transport=transport
    )
    setup = asyncio.run(
        client.create_swe_lego_issue(
            issue=_issue(), group_size=2, agent_config_id="agent-1"
        )
    )
    assert captured["path"] == "/api/v1/swe-lego/issues"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["group_size"] == 2
    assert captured["body"]["agent_config_id"] == "agent-1"
    assert captured["body"]["base_commit"] == "abc123"
    assert isinstance(setup, SweLegoSetup)
    assert setup.agent_run_ids == ["r1", "r2"]


def test_create_swe_lego_issue_raises_on_non_201():
    transport = _router(
        {
            ("POST", "/api/v1/swe-lego/issues*"): lambda r: httpx.Response(
                503, text="fork failed"
            )
        }
    )
    client = MulticaSweLegoClient(
        base_url="https://multica.example", transport=transport
    )
    with pytest.raises(RuntimeError, match="create swe-lego issue failed"):
        asyncio.run(
            client.create_swe_lego_issue(
                issue=_issue(), group_size=2, agent_config_id="a"
            )
        )


def test_cleanup_swe_lego_issue_posts_to_cleanup_endpoint():
    seen: list[str] = []

    def handler(request: httpx.Request):
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(204)

    transport = _router({("DELETE", "/api/v1/swe-lego/issues/p1*"): handler})
    client = MulticaSweLegoClient(
        base_url="https://multica.example", transport=transport
    )
    asyncio.run(client.cleanup_swe_lego_issue(project_id="p1"))
    assert seen == ["DELETE /api/v1/swe-lego/issues/p1"]

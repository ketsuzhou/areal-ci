"""Smoke test: BranchMaterializer works with MulticaSweLegoProvider.

Regression guard confirming spec invariant 3 (BranchMaterializer is
untouched — only the injected provider class differs). The materializer
accepts any ForkableEnvironment via Protocol, so swapping
FleetSandboxProvider for MulticaSweLegoProvider should "just work".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from customized_areal.tree_search.agents.environment import MulticaSweLegoProvider
from customized_areal.tree_search.agents.integration import (
    BranchMaterializer,
    MulticaIssueForker,
)


def _router(routes):
    def dispatch(request: httpx.Request):
        path = request.url.path
        for (method, route), handler in routes.items():
            if request.method != method:
                continue
            if route == path or (route.endswith("*") and path.startswith(route[:-1])):
                return handler(request)
        return httpx.Response(404, json={"error": f"no route for {request.method} {path}"})
    return httpx.MockTransport(dispatch)


@dataclass
class FakeStarter:
    started: list = field(default_factory=list)

    async def start_branch(self, *, forked_sandbox_id, forked_issue_id, replay_messages, drop_prior_session_id):
        self.started.append((forked_sandbox_id, forked_issue_id))
        return "branch-run-1"


def test_branch_materializer_works_with_multica_provider():
    # Snapshot + fork on the multica provider, fork-issue on the issue forker,
    # start_branch on the fake starter.
    routes = {
        ("POST", "/api/v1/sandboxes/sbx-1/snapshot*"): lambda r: httpx.Response(200, json={"snapshot_id": "snap-1"}),
        ("POST", "/api/v1/sandboxes/fork*"): lambda r: httpx.Response(200, json={"sandbox_id": "forked-sbx"}),
    }
    env = MulticaSweLegoProvider(base_url="https://multica.example", transport=_router(routes))

    issue_routes = {
        ("POST", "/api/issues/i1/fork*"): lambda r: httpx.Response(201, json={"forked_issue_id": "forked-i1"}),
    }
    forker = MulticaIssueForker(base_url="https://multica.example", transport=_router(issue_routes))

    starter = FakeStarter()
    mat = BranchMaterializer(env=env, forker=forker, starter=starter)
    result = asyncio.run(mat.materialize(
        source_sandbox_id="sbx-1", source_issue_id="i1", task_id="t1", seq=5,
        replay_messages=[{"role": "user", "content": "hi"}],
    ))
    assert result.branch_run_id == "branch-run-1"
    assert result.forked_sandbox_id == "forked-sbx"
    assert result.forked_issue_id == "forked-i1"
    assert starter.started == [("forked-sbx", "forked-i1")]

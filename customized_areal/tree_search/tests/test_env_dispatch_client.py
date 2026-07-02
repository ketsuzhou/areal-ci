import asyncio
import json

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue
from customized_areal.tree_search.agents.swe_lego_client import MulticaEnvDispatchClient


def _transport(handler):
    return httpx.MockTransport(handler)


def test_create_base_env_returns_env_id():
    def handler(req):
        assert req.method == "POST"
        assert req.url.path == "/api/v1/env"
        return httpx.Response(201, json={"env_id": "env-1", "sandbox_id": "sbx-1"})

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    env_id = asyncio.run(c.create_base_env(image_ref="img:tag"))
    assert env_id == "env-1"


def test_create_env_dispatch_scratch_swe_lego():
    def handler(req):
        body = json.loads(req.content)
        assert body["mode"] == "scratch"
        assert body["domain"] == "swe_lego"
        assert body["dispatch_type"] == "issue"
        assert body["group_size"] == 2
        return httpx.Response(
            201,
            json={"rollouts": [
                {"env_id": "e1", "project_id": "p1", "issue_id": "i1", "agent_run_id": "r1"},
                {"env_id": "e2", "project_id": "p2", "issue_id": "i2", "agent_run_id": "r2"},
            ]},
        )

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    issue = SweLegoIssue(
        repo_url="r", base_commit="c", issue_date="d", issue_text="x", issue_title="t",
        acceptance_criteria="a", fail_to_pass=["f"], pass_to_pass=["p"],
    )
    setup = asyncio.run(
        c.create_env_dispatch(
            mode="scratch", env_id="base", dispatch_type="issue",
            agent_id="ag", group_size=2, domain="swe_lego", issue=issue,
        )
    )
    assert len(setup.rollouts) == 2
    assert setup.rollouts[0].agent_run_id == "r1"
    assert setup.rollouts[1].env_id == "e2"


def test_cleanup_env_dispatch_hits_renamed_url():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        return httpx.Response(204)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.cleanup_env_dispatch(project_id="p1"))
    assert seen["path"] == "/api/v1/env-dispatch/p1"


def test_delete_env_idempotent_on_404():
    def handler(req):
        return httpx.Response(404)

    c = MulticaEnvDispatchClient(base_url="http://x", transport=_transport(handler))
    asyncio.run(c.delete_env(env_id="env-1"))  # no raise

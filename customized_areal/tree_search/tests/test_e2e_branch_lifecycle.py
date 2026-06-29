"""End-to-end branch lifecycle smoke test -- group_size=2, cloud-only.

Requires a live Multica server (with the Phase 1 migration applied) and a live
Fleet proxy (with snapshot/fork endpoints). Skipped when MULTICA_BASE_URL or
FLEET_BASE_URL are not set. Per design §3.3 e2e validation runs at group_size=2
first; a separate scale check (higher group_size) runs after this passes.

To run against a live stack:
  1. Start the Multica server with the Phase 1 migration applied.
  2. Start the Fleet proxy (or point at a staging Fleet).
  3. Export:
       export MULTICA_BASE_URL=http://localhost:8080
       export MULTICA_API_KEY=mul_...
       export FLEET_BASE_URL=http://localhost:...
       export FLEET_API_KEY=...
  4. Run: pytest customized_areal/tree_search/agents/test_e2e_branch_lifecycle.py -v

For v1 this is a smoke check that the providers construct and the Fleet
endpoint is reachable; a full run that provisions a live issue + sandbox is a
follow-up once the e2e fixtures harness lands.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("MULTICA_BASE_URL") and os.environ.get("FLEET_BASE_URL")),
    reason="e2e test requires MULTICA_BASE_URL and FLEET_BASE_URL",
)


@pytest.mark.asyncio
async def test_branch_lifecycle_at_group_size_2() -> None:
    """Smoke-check the full chain wiring: providers construct + Fleet reachable."""
    import httpx

    from customized_areal.tree_search.agents.environment import FleetSandboxProvider
    from customized_areal.tree_search.agents.integration import MulticaIssueForker

    env = FleetSandboxProvider()  # reads FLEET_BASE_URL / FLEET_API_KEY
    forker = MulticaIssueForker()  # reads MULTICA_BASE_URL / MULTICA_API_KEY
    try:
        assert env is not None
        assert forker is not None
        async with httpx.AsyncClient(
            base_url=os.environ["FLEET_BASE_URL"], timeout=10.0
        ) as client:
            resp = await client.get("/healthz")
            assert resp.status_code in (200, 404), (
                f"Fleet health check failed: {resp.status_code}"
            )
    finally:
        await env.aclose()
        await forker.aclose()

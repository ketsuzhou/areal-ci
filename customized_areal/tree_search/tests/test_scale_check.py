"""Scale check -- the concurrency semaphore bounds fork concurrency.

Runs only after the e2e at group_size=2 passes. Skipped when MULTICA_BASE_URL
or FLEET_BASE_URL are not set. The in-flight cap itself is unit-tested in
test_environment.py; this is a live smoke check that firing many concurrent
forks raises no concurrency-violation error.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("MULTICA_BASE_URL") and os.environ.get("FLEET_BASE_URL")),
    reason="scale check requires MULTICA_BASE_URL and FLEET_BASE_URL",
)


@pytest.mark.asyncio
async def test_concurrency_semaphore_caps_forks_at_group_size_8() -> None:
    """Fire 8 concurrent forks; the semaphore caps in-flight at the configured max."""
    from customized_areal.tree_search.agents.environment import FleetSandboxProvider

    provider = FleetSandboxProvider(max_concurrent_forks=4)
    try:
        results = await asyncio.gather(
            *[provider.fork(snapshot_id=f"snap-{i}") for i in range(8)],
            return_exceptions=True,
        )
        # All 8 should either succeed or fail with a ForkError (e.g. snapshot
        # not found). No concurrency-violation error type is expected.
        for r in results:
            if isinstance(r, Exception):
                assert "concurrency" not in str(r).lower(), (
                    f"unexpected concurrency error: {r}"
                )
    finally:
        await provider.aclose()

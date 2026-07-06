import asyncio
from dataclasses import dataclass, field

from customized_areal.tree_search.agents.branch_driver import EnvDispatchBranchDriver
from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoRollout,
    SweLegoSetup,
)


def _setup_with_env(env_id: str) -> SweLegoSetup:
    """Build a SweLegoSetup carrying a single rollout with ``env_id``."""
    return SweLegoSetup(
        rollouts=[
            SweLegoRollout(
                env_id=env_id,
                project_id="proj-child",
                issue_id="issue-1",
                chat_session_id="chat-1",
                agent_run_id="run-child",
            )
        ]
    )


@dataclass
class FakeClient:
    calls: list = field(default_factory=list)

    async def create_env_dispatch(self, **kw):
        self.calls.append(kw)
        return _setup_with_env("env-child")


def test_branch_driver_calls_env_dispatch_branch():
    client = FakeClient()
    drv = EnvDispatchBranchDriver(
        multica=client, domain="swe_lego", dispatch_type="issue", agent_id="ag-1"
    )
    terminal = asyncio.run(
        drv.drive_lane(agent_run_id="r1", sandbox_id="env-parent", session_id="s1")
    )
    assert len(client.calls) == 1
    kw = client.calls[0]
    assert kw["mode"] == "branch"
    # sandbox_id carries the source env_id (runner contract).
    assert kw["env_id"] == "env-parent"
    assert kw["dispatch_type"] == "issue"
    assert kw["agent_id"] == "ag-1"
    assert kw["domain"] == "swe_lego"
    assert kw["group_size"] == 1
    assert terminal == "env-child"


def test_branch_driver_returns_first_rollout_env_id():
    @dataclass
    class MultiRolloutClient:
        async def create_env_dispatch(self, **kw):
            return SweLegoSetup(
                rollouts=[
                    SweLegoRollout(env_id="env-first", project_id="p0"),
                    SweLegoRollout(env_id="env-second", project_id="p1"),
                ]
            )

    drv = EnvDispatchBranchDriver(
        multica=MultiRolloutClient(),
        domain="self_play",
        dispatch_type="message",
        agent_id="ag-2",
    )
    terminal = asyncio.run(
        drv.drive_lane(agent_run_id="r2", sandbox_id="env-src", session_id="s2")
    )
    assert terminal == "env-first"

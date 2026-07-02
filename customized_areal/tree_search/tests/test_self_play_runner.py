import asyncio
from dataclasses import dataclass, field

import pytest

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoRollout, SweLegoSetup,
)
from customized_areal.tree_search.agents.self_play_runner import (
    run_self_play, SelfPlayQuery, SelfPlayResult,
)
from customized_areal.tree_search.agents.verifier import VerifierResult


@dataclass
class FakeMulticaClient:
    rollouts: list = field(default_factory=list)
    cleanup_calls: list = field(default_factory=list)
    cleanup_raises: bool = False

    async def create_env_dispatch(self, *, mode, env_id, dispatch_type, agent_id,
                                  group_size, domain=None, issue=None, message=None):
        rollouts = [
            SweLegoRollout(env_id=f"env-{i}", project_id=f"proj-{i}",
                           chat_session_id=f"sess-{i}", agent_run_id=f"r{i+1}")
            for i in range(group_size)
        ]
        self.rollouts = rollouts
        return SweLegoSetup(rollouts=rollouts)

    async def cleanup_swe_lego_issue(self, *, project_id):
        self.cleanup_calls.append(project_id)
        if self.cleanup_raises:
            raise RuntimeError("cleanup crashed")


@dataclass
class FakeRlSession:
    sessions: list = field(default_factory=list)

    async def start(self, *, agent_run_id, issue_id):
        self.sessions.append(agent_run_id)
        return f"sess-{agent_run_id}"


@dataclass
class FakeVerifier:
    async def verify_and_reward(self, **kwargs):
        return VerifierResult(success=True, reward=1.0, source="objective")


@dataclass
class FakeBranchDriver:
    ran_lanes: list = field(default_factory=list)

    async def drive_lane(self, *, agent_run_id, sandbox_id, session_id):
        self.ran_lanes.append(agent_run_id)
        return sandbox_id


def _query() -> SelfPlayQuery:
    return SelfPlayQuery(query_id="q1", content="what is 2+2?", answer="4")


def test_run_self_play_happy_path():
    multica = FakeMulticaClient()
    rl = FakeRlSession()
    verifier = FakeVerifier()
    driver = FakeBranchDriver()
    result = asyncio.run(
        run_self_play(
            query=_query(), group_size=2, agent_id="ag", base_env_id="base",
            multica=multica, rl_session=rl, verifier=verifier, branch_driver=driver,
        )
    )
    assert isinstance(result, SelfPlayResult)
    assert len(multica.rollouts) == 2
    assert len(rl.sessions) == 2
    assert len(driver.ran_lanes) == 2
    assert result.per_agent_rewards == [1.0, 1.0]
    assert multica.cleanup_calls == ["proj-0", "proj-1"]


def test_run_self_play_cleans_up_on_verifier_failure():
    multica = FakeMulticaClient()
    rl = FakeRlSession()

    @dataclass
    class RaisingVerifier:
        async def verify_and_reward(self, **kwargs):
            raise RuntimeError("verifier crashed")

    with pytest.raises(RuntimeError, match="verifier crashed"):
        asyncio.run(
            run_self_play(
                query=_query(), group_size=2, agent_id="ag", base_env_id="base",
                multica=multica, rl_session=rl, verifier=RaisingVerifier(),
                branch_driver=FakeBranchDriver(),
            )
        )
    assert multica.cleanup_calls == ["proj-0", "proj-1"]

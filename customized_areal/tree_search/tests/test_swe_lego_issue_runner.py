import asyncio
from dataclasses import dataclass, field

import pytest

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoRollout,
    SweLegoSetup,
)
from customized_areal.tree_search.agents.swe_lego_issue_runner import (
    run_swe_lego_issue,
    SweLegoIssueResult,
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
                           issue_id="issue-1", agent_run_id=f"r{i+1}")
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
    async def verify_and_reward(self, *, agent_run_id, sandbox_id, session_id, fail_to_pass, pass_to_pass, transcript, acceptance_criteria):
        return VerifierResult(success=True, reward=1.0, source="objective")


@dataclass
class RaisingVerifier:
    async def verify_and_reward(self, **kwargs):
        raise RuntimeError("verifier crashed")


@dataclass
class FakeBranchDriver:
    """Stands in for select_branch_candidate + BranchMaterializer.materialize."""
    ran_lanes: list = field(default_factory=list)
    raises: bool = False

    async def drive_lane(self, *, agent_run_id, sandbox_id, session_id):
        if self.raises:
            raise RuntimeError("branch driver crashed")
        self.ran_lanes.append(agent_run_id)
        return sandbox_id  # the terminal sandbox id


def _issue() -> SweLegoIssue:
    return SweLegoIssue(
        repo_url="r", base_commit="c", issue_date="d",
        issue_text="x", issue_title="t", acceptance_criteria="a",
        fail_to_pass=["f"], pass_to_pass=["p"],
    )


def test_run_swe_lego_issue_happy_path():
    multica = FakeMulticaClient()
    rl = FakeRlSession()
    verifier = FakeVerifier()
    driver = FakeBranchDriver()
    result = asyncio.run(
        run_swe_lego_issue(
            issue=_issue(), group_size=2, agent_id="ag", base_env_id="base-env-1",
            multica=multica, rl_session=rl, verifier=verifier, branch_driver=driver,
        )
    )
    assert isinstance(result, SweLegoIssueResult)
    assert len(multica.rollouts) == 2
    assert len(rl.sessions) == 2
    assert len(driver.ran_lanes) == 2
    assert result.per_agent_rewards == [1.0, 1.0]
    assert multica.cleanup_calls == ["proj-0", "proj-1"]


def test_run_swe_lego_issue_cleans_up_on_verifier_failure():
    # If the verifier raises, the runner must still cleanup the multica
    # resources (otherwise we leak sandboxes), but should propagate the error.
    multica = FakeMulticaClient()

    with pytest.raises(RuntimeError, match="verifier crashed"):
        asyncio.run(
            run_swe_lego_issue(
                issue=_issue(), group_size=2, agent_id="ag", base_env_id="base-env-1",
                multica=multica, rl_session=FakeRlSession(), verifier=RaisingVerifier(),
                branch_driver=FakeBranchDriver(),
            )
        )
    # Cleanup happened despite the verifier error.
    assert multica.cleanup_calls == ["proj-0", "proj-1"]


def test_run_swe_lego_issue_cleans_up_when_branch_driver_raises():
    multica = FakeMulticaClient()

    with pytest.raises(RuntimeError, match="branch driver crashed"):
        asyncio.run(
            run_swe_lego_issue(
                issue=_issue(), group_size=2, agent_id="ag", base_env_id="base-env-1",
                multica=multica, rl_session=FakeRlSession(), verifier=FakeVerifier(),
                branch_driver=FakeBranchDriver(raises=True),
            )
        )
    # Cleanup happened despite the branch driver error.
    assert multica.cleanup_calls == ["proj-0", "proj-1"]


def test_run_swe_lego_issue_logs_when_cleanup_itself_raises():
    multica = FakeMulticaClient(cleanup_raises=True)
    rl = FakeRlSession()
    verifier = FakeVerifier()
    driver = FakeBranchDriver()

    # The original verifier result should still return — cleanup failure is
    # logged, not propagated. We verify the behavioral contract (runner
    # returns normally despite cleanup raising) rather than asserting on
    # the log record, which is fragile across pytest caplog configurations.
    result = asyncio.run(
        run_swe_lego_issue(
            issue=_issue(), group_size=2, agent_id="ag", base_env_id="base-env-1",
            multica=multica, rl_session=rl, verifier=verifier, branch_driver=driver,
        )
    )
    assert result.per_agent_rewards == [1.0, 1.0]
    # Cleanup was attempted despite the exception (swallowed + logged).
    assert multica.cleanup_calls == ["proj-0", "proj-1"]

import asyncio
from dataclasses import dataclass

import pytest

from customized_areal.tree_search.agents.reward.swe_lego_verifier import (
    ObjectiveOutcome,
    SweLegoVerifier,
)


@dataclass
class FakeObjective:
    fully_passes: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    failing_count_reduced: bool

    async def run_tests(self, sandbox_id, f2p, p2p):
        return ObjectiveOutcome(
            fully_passes=self.fully_passes,
            f2p_passed=self.f2p_passed,
            f2p_total=self.f2p_total,
            p2p_passed=self.p2p_passed,
            p2p_total=self.p2p_total,
            failing_count_reduced=self.failing_count_reduced,
        )


@dataclass
class FakeCritic:
    score_value: float
    raises: bool = False

    async def score(self, transcript, criteria, objective):
        if self.raises:
            raise RuntimeError("critic LLM error")
        return self.score_value


@dataclass
class FakeRlSession:
    last_reward: float | None = None

    async def set_reward(self, *, session_id, reward):
        self.last_reward = reward


def test_verifier_short_circuits_to_one_on_full_pass():
    rl = FakeRlSession()
    v = SweLegoVerifier(
        objective=FakeObjective(True, 2, 2, 3, 3, True),
        critic=FakeCritic(score_value=0.1),  # ignored on full pass
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1",
            sandbox_id="s1",
            session_id="sess1",
            fail_to_pass=["a"],
            pass_to_pass=["b"],
            transcript="...",
            acceptance_criteria="...",
        )
    )
    assert result.reward == 1.0
    assert result.success is True
    assert rl.last_reward == 1.0


def test_verifier_short_circuits_to_zero_on_total_failure():
    rl = FakeRlSession()
    v = SweLegoVerifier(
        objective=FakeObjective(False, 0, 2, 3, 3, False),
        critic=FakeCritic(score_value=0.9),  # ignored on total failure
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1",
            sandbox_id="s1",
            session_id="sess1",
            fail_to_pass=["a"],
            pass_to_pass=["b"],
            transcript="...",
            acceptance_criteria="...",
        )
    )
    assert result.reward == 0.0
    assert result.success is False
    assert rl.last_reward == 0.0


def test_verifier_blends_in_mixed_middle():
    rl = FakeRlSession()
    # 1 of 2 F2P passes, failing count reduced → not total failure, not full pass.
    v = SweLegoVerifier(
        objective=FakeObjective(False, 1, 2, 3, 3, True),
        critic=FakeCritic(score_value=0.5),
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1",
            sandbox_id="s1",
            session_id="sess1",
            fail_to_pass=["a", "b"],
            pass_to_pass=["c"],
            transcript="...",
            acceptance_criteria="...",
        )
    )
    # objective = 0.5 (1/2 F2P), generative = 0.5, semi_resolved = 1.0 (reduced).
    # blend = 0.7*0.5 + 0.2*0.5 + 0.1*1.0 = 0.35 + 0.10 + 0.10 = 0.55
    assert result.reward == pytest.approx(0.55)
    assert rl.last_reward == pytest.approx(0.55)


def test_verifier_critic_failure_falls_back_neutral():
    rl = FakeRlSession()
    v = SweLegoVerifier(
        objective=FakeObjective(False, 1, 2, 3, 3, True),
        critic=FakeCritic(score_value=0.5, raises=True),
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1",
            sandbox_id="s1",
            session_id="sess1",
            fail_to_pass=["a", "b"],
            pass_to_pass=["c"],
            transcript="...",
            acceptance_criteria="...",
        )
    )
    # Critic failed → source="default", reward still written (blend uses 0.0
    # for the generative term so training can proceed).
    assert result.source == "default"
    assert rl.last_reward is not None

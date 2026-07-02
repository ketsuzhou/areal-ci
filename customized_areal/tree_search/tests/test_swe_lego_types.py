"""Tests for the SWE-Lego issue dataclasses."""

from __future__ import annotations

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoIssueResult,
    SweLegoRollout,
    SweLegoSetup,
)


def test_swe_lego_issue_carries_test_lists():
    issue = SweLegoIssue(
        repo_url="https://github.com/psf/requests.git",
        base_commit="abc123",
        issue_date="2025-03-14T09:30:00Z",
        issue_text="retry leaks",
        issue_title="Retry leaks",
        acceptance_criteria="must not leak",
        fail_to_pass=["tests/test_retry.py::test_leak"],
        pass_to_pass=["tests/test_retry.py::test_basic"],
    )
    assert issue.fail_to_pass == ["tests/test_retry.py::test_leak"]
    assert issue.pass_to_pass == ["tests/test_retry.py::test_basic"]


def test_swe_lego_setup_holds_rollout_group():
    setup = SweLegoSetup(
        rollouts=[
            SweLegoRollout(env_id="e1", project_id="p1", issue_id="i1", agent_run_id="r1"),
            SweLegoRollout(env_id="e2", project_id="p2", issue_id="i2", agent_run_id="r2"),
            SweLegoRollout(env_id="e3", project_id="p3", issue_id="i3", agent_run_id="r3"),
        ]
    )
    assert len(setup.rollouts) == 3
    assert [r.agent_run_id for r in setup.rollouts] == ["r1", "r2", "r3"]


def test_swe_lego_issue_result_collects_per_agent_rewards():
    result = SweLegoIssueResult(per_agent_rewards=[1.0, 0.0, 0.5])
    assert result.per_agent_rewards == [1.0, 0.0, 0.5]

"""Tests for the SWE-Lego issue dataclasses."""

from __future__ import annotations

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoIssueResult,
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


def test_swe_lego_issue_result_collects_per_agent_rewards():
    result = SweLegoIssueResult(per_agent_rewards=[1.0, 0.0, 0.5])
    assert result.per_agent_rewards == [1.0, 0.0, 0.5]

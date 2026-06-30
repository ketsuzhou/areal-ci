"""Dataclasses shared by the SWE-Lego issue runner and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SweLegoIssue:
    """One SWE issue to train on.

    ``base_commit`` is the parent of the fixing PR's merge. ``issue_date`` is
    the cutoff for SWE-Lego anti-hacking: git history after this date is
    deleted at image-build time. ``fail_to_pass`` / ``pass_to_pass`` are the
    SWE-bench test lists the verifier runs.
    """

    repo_url: str
    base_commit: str
    issue_date: str
    issue_text: str
    issue_title: str
    acceptance_criteria: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]


@dataclass(frozen=True)
class SweLegoSetup:
    """The result of ``POST /api/v1/swe-lego/issues``.

    Mirrors the 201 response from the multica atomic endpoint (spec §4.1).
    """

    project_id: str
    issue_id: str
    image_id: str
    build_node_id: str
    base_sandbox_id: str
    base_sandbox_runtime_id: str
    agent_run_ids: list[str]


@dataclass(frozen=True)
class SweLegoIssueResult:
    """The outcome of one issue's training episode."""

    per_agent_rewards: list[float]
    per_agent_success: list[bool] = field(default_factory=list)

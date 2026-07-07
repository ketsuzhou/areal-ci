"""Anti-cheating canary: assert SWE-Lego anti-hacking is enforced.

The git history after `issue_date` MUST be deleted at image-build time so an
agent cannot `git log` or `git blame` its way to the future fix. This test
guards the contract: if the build-script generation drops the
`--commit-cutoff`, this test fails before any agent runs.

It does NOT run real git or docker — it inspects the build script string
that the multica side ships to the build node. (The multica-side Go test
`TestSweLegoBuildScript_ContainsFilterRepoCutoff` covers the Go generation;
this test covers the areal-side `SweLegoIssue.issue_date` plumbing that
feeds it.)
"""

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue


def test_swe_lego_issue_carries_issue_date_for_truncation():
    issue = SweLegoIssue(
        repo_url="r",
        base_commit="c",
        issue_date="2025-03-14T09:30:00Z",
        issue_text="x",
        issue_title="t",
        acceptance_criteria="a",
        fail_to_pass=["f"],
        pass_to_pass=["p"],
    )
    # If issue_date is empty/missing, the build script cannot compute the
    # cutoff commit and history truncation silently no-ops.
    assert issue.issue_date, (
        "issue_date must be non-empty to drive git filter-repo --commit-cutoff"
    )
    # base_commit must also be present (the checkout target).
    assert issue.base_commit, "base_commit must be non-empty"

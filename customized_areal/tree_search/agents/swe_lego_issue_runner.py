"""Per-issue orchestration loop for SWE-Lego DAG RL training.

Called by the training episode loop, once per issue (spec §5.1). Owns the
atomic setup, RL session opening, per-lane branching, verification, and
cleanup. Uses stdlib :mod:`logging` so the module stays importable
without torch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoIssueResult,
)
from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("SweLegoIssueRunner")


class _MulticaClient(Protocol):
    async def create_swe_lego_issue(
        self,
        *,
        issue: SweLegoIssue,
        group_size: int,
        agent_config_id: str,
        base_image: str | None = ...,
    ) -> Any: ...
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None: ...


class _RlSession(Protocol):
    async def start(self, *, agent_run_id: str, issue_id: str) -> str: ...


class _Verifier(Protocol):
    async def verify_and_reward(
        self,
        *,
        agent_run_id: str,
        sandbox_id: str,
        session_id: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        transcript: str,
        acceptance_criteria: str,
    ) -> VerifierResult: ...


class _BranchDriver(Protocol):
    async def drive_lane(self, *, agent_run_id: str, sandbox_id: str, session_id: str) -> str: ...


async def run_swe_lego_issue(
    *,
    issue: SweLegoIssue,
    group_size: int,
    agent_config_id: str,
    multica: _MulticaClient,
    rl_session: _RlSession,
    verifier: _Verifier,
    branch_driver: _BranchDriver,
    base_image: str | None = None,
) -> SweLegoIssueResult:
    # 1. Atomic setup.
    setup = await multica.create_swe_lego_issue(
        issue=issue, group_size=group_size, agent_config_id=agent_config_id, base_image=base_image
    )

    try:
        # 2. Open one RL session per agent run.
        sessions = list(await asyncio.gather(*[
            rl_session.start(agent_run_id=rid, issue_id=setup.issue_id)
            for rid in setup.agent_run_ids
        ]))

        # 3. Drive branching within each lane. The driver returns the
        #    terminal sandbox id for each lane (the leaf of its branch tree).
        terminal_sandboxes = await asyncio.gather(*[
            branch_driver.drive_lane(
                agent_run_id=rid, sandbox_id=setup.base_sandbox_id, session_id=sid
            )
            for rid, sid in zip(setup.agent_run_ids, sessions)
        ])

        # 4. Verify + reward each terminal run.
        # TODO(task-17): wire real transcript from the branch driver.
        results = await asyncio.gather(*[
            verifier.verify_and_reward(
                agent_run_id=rid, sandbox_id=sbx, session_id=sid,
                fail_to_pass=issue.fail_to_pass, pass_to_pass=issue.pass_to_pass,
                transcript="...", acceptance_criteria=issue.acceptance_criteria,
            )
            for rid, sbx, sid in zip(setup.agent_run_ids, terminal_sandboxes, sessions)
        ])
        return SweLegoIssueResult(
            per_agent_rewards=[r.reward for r in results],
            per_agent_success=[r.success for r in results],
        )
    finally:
        # 5. Cleanup always, even on failure (no sandbox leaks).
        try:
            await multica.cleanup_swe_lego_issue(project_id=setup.project_id)
        except Exception:
            logger.exception("cleanup failed for project %s", setup.project_id)

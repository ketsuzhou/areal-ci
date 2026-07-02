"""Per-issue orchestration loop for SWE-Lego DAG RL training.

Called by the training episode loop, once per issue (spec §5.1). Owns the
atomic setup, RL session opening, per-lane branching, verification, and
cleanup. Uses stdlib :mod:`logging` so the module stays importable
without torch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoIssueResult,
    SweLegoSetup,
)
from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("SweLegoIssueRunner")


class _MulticaClient(Protocol):
    async def create_env_dispatch(
        self, *, mode: str, env_id: str, dispatch_type: str,
        agent_id: str, group_size: int = ..., domain: str | None = ...,
        issue: SweLegoIssue | None = ..., message: str | None = ...,
    ) -> SweLegoSetup: ...
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
    agent_id: str,
    multica: _MulticaClient,
    rl_session: _RlSession,
    verifier: _Verifier,
    branch_driver: _BranchDriver,
    base_env_id: str,
) -> SweLegoIssueResult:
    # 1. Atomic dispatch.
    setup = await multica.create_env_dispatch(
        mode="scratch", env_id=base_env_id, dispatch_type="issue",
        agent_id=agent_id, group_size=group_size,
        domain="swe_lego", issue=issue,
    )

    try:
        # 2. Open one RL session per rollout.
        sessions = list(await asyncio.gather(*[
            rl_session.start(agent_run_id=r.agent_run_id, issue_id=r.issue_id)
            for r in setup.rollouts
        ]))

        # 3. Drive branching within each lane. The driver returns the
        #    terminal sandbox id for each lane.
        #    NOTE: multica owns sandbox_id; areal passes env_id to branch.
        #    The branch driver receives env_id (not sandbox_id) and asks
        #    multica to fork internally when it branches.
        terminal_env_ids = await asyncio.gather(*[
            branch_driver.drive_lane(
                agent_run_id=r.agent_run_id, sandbox_id=r.env_id, session_id=sid
            )
            for r, sid in zip(setup.rollouts, sessions)
        ])

        # 4. Verify + reward each terminal run.
        # TODO(task-17): wire real transcript from the branch driver.
        results = await asyncio.gather(*[
            verifier.verify_and_reward(
                agent_run_id=r.agent_run_id, sandbox_id=eid, session_id=sid,
                fail_to_pass=issue.fail_to_pass, pass_to_pass=issue.pass_to_pass,
                transcript="...", acceptance_criteria=issue.acceptance_criteria,
            )
            for r, eid, sid in zip(setup.rollouts, terminal_env_ids, sessions)
        ])
        return SweLegoIssueResult(
            per_agent_rewards=[r.reward for r in results],
            per_agent_success=[r.success for r in results],
        )
    finally:
        # 5. Cleanup always, even on failure (no sandbox leaks). One DELETE
        #    per rollout's project_id.
        for r in setup.rollouts:
            try:
                await multica.cleanup_swe_lego_issue(project_id=r.project_id)
            except Exception:
                logger.exception("cleanup failed for project %s", r.project_id)

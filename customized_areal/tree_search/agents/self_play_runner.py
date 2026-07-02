"""Self-play orchestration loop for DAG RL training.

Mirrors ``swe_lego_issue_runner.py`` but dispatches a query (from
``query_bank``) as a chat message via POST /api/v1/env-dispatch with
domain=self_play, dispatch_type=message. Uses stdlib :mod:`logging` so the
module stays importable without torch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoSetup
from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("SelfPlayRunner")


@dataclass(frozen=True)
class SelfPlayQuery:
    """One query from query_bank."""

    query_id: str
    content: str
    answer: str


@dataclass(frozen=True)
class SelfPlayResult:
    per_agent_rewards: list[float]
    per_agent_success: list[bool] = field(default_factory=list)


class _MulticaClient(Protocol):
    async def create_env_dispatch(
        self, *, mode: str, env_id: str, dispatch_type: str,
        agent_id: str, group_size: int = ..., domain: str | None = ...,
        issue=None, message: str | None = ...,
    ) -> SweLegoSetup: ...
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None: ...


class _RlSession(Protocol):
    async def start(self, *, agent_run_id: str, issue_id: str) -> str: ...


class _Verifier(Protocol):
    async def verify_and_reward(
        self, *, agent_run_id: str, sandbox_id: str, session_id: str,
        transcript: str, answer: str,
    ) -> VerifierResult: ...


class _BranchDriver(Protocol):
    async def drive_lane(self, *, agent_run_id: str, sandbox_id: str, session_id: str) -> str: ...


async def run_self_play(
    *,
    query: SelfPlayQuery,
    group_size: int,
    agent_id: str,
    base_env_id: str,
    multica: _MulticaClient,
    rl_session: _RlSession,
    verifier: _Verifier,
    branch_driver: _BranchDriver,
) -> SelfPlayResult:
    setup = await multica.create_env_dispatch(
        mode="scratch", env_id=base_env_id, dispatch_type="message",
        agent_id=agent_id, group_size=group_size,
        domain="self_play", message=query.content,
    )

    try:
        sessions = list(await asyncio.gather(*[
            rl_session.start(agent_run_id=r.agent_run_id, issue_id="")
            for r in setup.rollouts
        ]))

        terminal_env_ids = await asyncio.gather(*[
            branch_driver.drive_lane(
                agent_run_id=r.agent_run_id, sandbox_id=r.env_id, session_id=sid
            )
            for r, sid in zip(setup.rollouts, sessions)
        ])

        results = await asyncio.gather(*[
            verifier.verify_and_reward(
                agent_run_id=r.agent_run_id, sandbox_id=eid, session_id=sid,
                transcript="...", answer=query.answer,
            )
            for r, eid, sid in zip(setup.rollouts, terminal_env_ids, sessions)
        ])
        return SelfPlayResult(
            per_agent_rewards=[r.reward for r in results],
            per_agent_success=[r.success for r in results],
        )
    finally:
        for r in setup.rollouts:
            try:
                await multica.cleanup_swe_lego_issue(project_id=r.project_id)
            except Exception:
                logger.exception("cleanup failed for project %s", r.project_id)

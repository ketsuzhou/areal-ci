"""Concrete branch driver backed by the multica env-dispatch API.

At a branch point the driver forks the source environment via
``create_env_dispatch(mode="branch", env_id=<source env_id>)`` and returns the
new terminal ``env_id``. It satisfies the ``_BranchDriver`` Protocol used by
the SWE-Lego issue runner and the self-play runner (structural typing). Uses
stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
from typing import Protocol

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoSetup,
)

logger = logging.getLogger("EnvDispatchBranchDriver")


class _MulticaClient(Protocol):
    async def create_env_dispatch(
        self,
        *,
        mode: str,
        env_id: str,
        dispatch_type: str,
        agent_id: str,
        group_size: int = ...,
        domain: str | None = ...,
        issue: SweLegoIssue | None = ...,
        message: str | None = ...,
    ) -> SweLegoSetup: ...


class EnvDispatchBranchDriver:
    """Drive a lane's branch by forking the source env via env-dispatch.

    The runner contract (see ``swe_lego_issue_runner.py`` /
    ``self_play_runner.py``) passes ``sandbox_id=r.env_id``, i.e. the SOURCE
    ``env_id`` to branch from. This driver forwards that source ``env_id`` to
    ``create_env_dispatch(mode="branch", ...)`` and returns the child rollout's
    terminal ``env_id``.
    """

    def __init__(
        self,
        *,
        multica: _MulticaClient,
        domain: str,
        dispatch_type: str,
        agent_id: str,
    ) -> None:
        self._multica = multica
        self._domain = domain
        self._dispatch_type = dispatch_type
        self._agent_id = agent_id

    async def drive_lane(
        self, *, agent_run_id: str, sandbox_id: str, session_id: str
    ) -> str:
        """Fork ``sandbox_id`` (source env_id) and return the child env_id."""
        setup = await self._multica.create_env_dispatch(
            mode="branch",
            env_id=sandbox_id,
            dispatch_type=self._dispatch_type,
            agent_id=self._agent_id,
            domain=self._domain,
            group_size=1,
        )
        child_env_id = setup.rollouts[0].env_id
        logger.debug(
            "branched lane agent_run_id=%s source_env_id=%s -> child_env_id=%s",
            agent_run_id,
            sandbox_id,
            child_env_id,
        )
        return child_env_id

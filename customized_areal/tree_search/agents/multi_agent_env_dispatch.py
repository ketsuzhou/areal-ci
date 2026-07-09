"""MultiAgentEnvDispatchWorkflow: thin RolloutWorkflow orchestrator over the
existing v2-segment-dag components.

One ``arun_episode`` = one Multica task = N agents = N sessions, assembled
into a single :class:`AssembledDag`. This module orchestrates components that
already exist (dispatch client, DAG poller, supernode assembler, tensor
lifecycle); it does NOT reimplement dispatch/polling/assembly.

Boundary: Multica owns ``/rl/start_session(group_size=N)`` and the
``session_to_agent_run`` binding + per-agent credentials. AReaL never calls
``start_session`` on this path - it only dispatches, polls, assembles, and
cleans up. Sync component calls run in :func:`asyncio.to_thread` so M parallel
episodes stay concurrent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from areal.api.workflow_api import RolloutWorkflow

logger = logging.getLogger("MultiAgentEnvDispatchWorkflow")


class MultiAgentEnvDispatchWorkflow(RolloutWorkflow):
    """One arun_episode = one Multica task = N agents = N sessions -> AssembledDag.

    Thin orchestrator over existing v2-segment-dag components: dispatches via
    MulticaEnvDispatchClient, polls via MulticaDagClient.get_dag, assembles via
    SuperNodeAssembler.assemble_from_refs. Multica owns /rl/start_session +
    session_to_agent_run + credentials; AReaL never calls start_session.
    Sync component calls run in asyncio.to_thread to keep M episodes concurrent.
    """

    def __init__(
        self,
        *,
        dispatch_client,
        dag_client,
        assembler,
        resolver,
        session_remover,
        poll_timeout: float = 3600.0,
        poll_interval: float = 1.0,
        group_size: int = 1,
        base_env_id: str = "",
    ):
        self._dispatch = dispatch_client
        self._dag_client = dag_client
        self._assembler = assembler
        self._resolver = resolver
        self._session_remover = session_remover
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.group_size = group_size
        self.base_env_id = base_env_id

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        setup = await self._dispatch.create_env_dispatch(
            mode="scratch",
            env_id=self.base_env_id,
            dispatch_type="message",
            agent_id=data.get("agent_id", ""),
            group_size=self.group_size,
            domain="multica",
            message=data.get("message"),
        )
        if not setup.rollouts:
            return None
        project_id = setup.rollouts[0].project_id
        from customized_areal.tree_search.agents.multica_dag_client import DagTimeout

        try:
            dag = await asyncio.to_thread(
                self._dag_client.get_dag,
                project_id,
                timeout=self.poll_timeout,
                interval=self.poll_interval,
            )
        except DagTimeout:
            logger.warning(
                "AssembledDag poll timed out for project %s; rejecting", project_id
            )
            return None
        expected = {r.agent_run_id for r in setup.rollouts if r.agent_run_id}
        covered = set(dag.session_to_agent_run.values())
        if not expected.issubset(covered):
            logger.warning(
                "Partial squad for project %s: expected %s covered %s; dropping",
                project_id,
                expected,
                covered,
            )
            return None
        edag = await asyncio.to_thread(
            self._assembler.assemble_from_refs, dag, self._resolver
        )
        # Cleanup (success-path only): release shards, revoke sessions.
        shard_ids = [
            s.tensor_ref.get("shard_id")
            for s in dag.segments
            if s.tensor_ref.get("shard_id")
        ]
        if shard_ids:
            await asyncio.to_thread(self._resolver.clear, shard_ids)
        for session_id in dag.session_to_agent_run:
            await asyncio.to_thread(self._session_remover.remove, session_id)
        return {"assembled_dag": dag, "execution_dag": edag}

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

from customized_areal.tree_search.agents.multica_client import MulticaEnvDispatchClient

from areal.api.workflow_api import RolloutWorkflow

logger = logging.getLogger("MultiAgentEnvDispatchWorkflow")


def _dispatch_message(data: dict[str, Any]) -> str:
    """Return the user-facing task text for a Multica message dispatch.

    The TPFC dataset exposes it as ``query``; rows whose extraction produced
    nothing still carry the raw ``messages`` turns, so fall back to the last
    user turn.
    """
    query = (data.get("query") or "").strip()
    if query:
        return query
    for msg in reversed(data.get("messages") or []):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


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
        dag_client,
        assembler,
        resolver,
        session_remover,
        dispatch_client: MulticaEnvDispatchClient | None = None,
        poll_timeout: float = 3600.0,
        poll_interval: float = 1.0,
        group_size: int = 1,
        base_env_id: str = "",
    ):
        self._dispatch = dispatch_client or MulticaEnvDispatchClient()
        self._dag_client = dag_client
        self._assembler = assembler
        self._resolver = resolver
        self._session_remover = session_remover
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.group_size = group_size
        self.base_env_id = base_env_id

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        # SCRATCH dispatch on base_env_id -> the dispatch's project_id (one
        # Multica task = one project). create_env_dispatch returns the top-level
        # project_id directly; the empty-trajectory check is deferred to the
        # assembler (it returns None when the polled DAG has no segments).
        # RL datasets name the user turn "query" (see tpfc_dataset); "message" is
        # accepted for callers that already pass a ready dispatch message.
        message = data.get("message") or data.get("query")
        handle = await self._dispatch.create_env_dispatch(
            mode="scratch",
            env_id=self.base_env_id,
            dispatch_type="message",
            agent_id=data.get("agent_id", ""),
            group_size=self.group_size,
            domain="self_play",
            message=message,
            training_mode=True,
        )
        from customized_areal.tree_search.agents.multica_dag_client import DagTimeout

        # DAG fetch error policy: DagTimeout (the DAG never left 202 within the
        # poll window) is a rejected episode -> None (the squad stalled; an
        # immediate retry would re-dispatch, not re-poll the same DAG). The
        # remaining fetch errors - DagNotFound (404) / DagForbidden (403) /
        # DagError (unexpected status) - PROPAGATE so the caller's retry layer
        # can back off or surface them; they are not silently swallowed.
        try:
            dag = await asyncio.to_thread(
                self._dag_client.get_dag,
                handle,
                timeout=self.poll_timeout,
                interval=self.poll_interval,
            )
        except DagTimeout:
            logger.warning(
                "AssembledDag poll timed out for %s; rejecting", handle.primary_id
            )
            return None
        edag = await asyncio.to_thread(
            self._assembler.assemble_from_refs, dag, self._resolver
        )
        # Cleanup runs whenever the DAG was fetched (release shards, revoke
        # sessions), including an empty trajectory, so orphaned sessions do not
        # leak. DagTimeout skips it (no DAG to clean up).
        shard_ids = [
            s.tensor_ref.get("shard_id")
            for s in dag.segments
            if s.tensor_ref.get("shard_id")
        ]
        if shard_ids:
            await asyncio.to_thread(self._resolver.clear, shard_ids)
        for session_id in dag.session_to_agent_run:
            await asyncio.to_thread(self._session_remover.remove, session_id)
        if edag is None:
            # Empty trajectory (no segments recorded) -> reject the episode.
            return None
        return {"assembled_dag": dag, "execution_dag": edag}

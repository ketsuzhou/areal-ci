"""Minimal segment-DAG training plumbing + tensor lifecycle (change 1).

Wires the v2 consumer pipeline end-to-end with placeholder zero reward:

    MulticaDagClient.get_dag
      -> SuperNodeAssembler.assemble_from_refs(resolver)
      -> ExecutionDAG.topological_order()
      -> assemble_node_advantages (global GAE, zero reward)
      -> tensor lifecycle cleanup (clear shards + remove sessions)

Change 1 proves the data path; the judge / critic-V / real reward land in
change 2, so only the GAE forward path is exercised here. No torch/FSDP is
required for the orchestration: ``assemble_node_advantages`` is pure float
math, and ``events_from_nodes`` treats an unset SuperNode ``value`` as 0.0,
so the SuperNodes built by ``assemble_from_refs`` (zero reward, ``value`` left
``None``) flow straight through with zero advantages.

Tensor-ref contract (change 1): each segment's ``tensor_ref`` is
``{"shard_id": str}`` pointing at a v2 data_proxy shard that holds the
serialized tensor dict (``input_ids`` / ``loss_mask`` / ``logprobs`` / ...).
The exact ref shape is finalized when Multica (U6/U8) pins the export
contract; the resolver is the single place to update if it changes.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
import orjson

from customized_areal.tree_search.agents.dag_advantage import (
    AssembledAdvantages,
    assemble_node_advantages,
)
from customized_areal.tree_search.agents.multica_dag_client import MulticaDagClient
from customized_areal.tree_search.agents.supernode_assembler import (
    SuperNodeAssembler,
    TensorResolver,
)


class TrainingTensorResolver(TensorResolver, Protocol):
    """``TensorResolver`` (resolve) extended with tensor-lifecycle cleanup.

    ``resolve`` satisfies :class:`TensorResolver` so the same object is passed
    to :meth:`SuperNodeAssembler.assemble_from_refs`; ``clear`` releases the
    resolved shards once training has consumed them.
    """

    def resolve(self, tensor_ref: dict[str, Any]) -> dict[str, Any]: ...

    def clear(self, shard_ids: list[str]) -> None: ...


class SessionRemover(Protocol):
    """Revokes a v2 data_proxy session once its segments are consumed."""

    def remove(self, session_id: str) -> None: ...


class DataProxyTensorResolver:
    """Resolves tensor_refs via the v2 data_proxy ``/data/*`` endpoints.

    ``resolve`` GETs ``/data/<shard_id>`` and deserializes the shard (the
    data_proxy stores ``orjson.dumps(serialize_value(tensor))``); ``clear``
    DELETEs ``/data/clear`` with the consumed shard ids. ``httpx``-based with
    an injectable transport for tests.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = _transport

    def _client(self) -> httpx.Client:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        kwargs: dict[str, Any] = {"headers": headers, "timeout": 10.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def resolve(self, tensor_ref: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tensor_ref, dict) or not tensor_ref:
            raise KeyError("tensor_ref missing field->shard map")
        from areal.infra.rpc.serialization import deserialize_value

        # Multi-shard contract (Option B): tensor_ref maps each field name to a
        # shard ref ``{shard_id, node_addr}``; each shard is fetched separately
        # and reassembled into a ``{field: tensor}`` dict for the assembler.
        tensors: dict[str, Any] = {}
        with self._client() as client:
            for field, ref in tensor_ref.items():
                if not isinstance(ref, dict):
                    raise KeyError(f"tensor_ref field {field!r} is not a shard ref")
                shard_id = ref.get("shard_id")
                if not shard_id:
                    raise KeyError(f"tensor_ref field {field!r} missing 'shard_id'")
                resp = client.get(f"{self._base}/data/{shard_id}")
                if resp.status_code == 404:
                    raise KeyError(f"shard {shard_id} not found (field {field!r})")
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"unexpected {resp.status_code} resolving shard {shard_id}: {resp.text}"
                    )
                tensors[field] = deserialize_value(orjson.loads(resp.content))
        return tensors

    def clear(self, shard_ids: list[str]) -> None:
        if not shard_ids:
            return
        with self._client() as client:
            resp = client.request(
                "DELETE", f"{self._base}/data/clear", json={"shard_ids": shard_ids}
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"unexpected {resp.status_code} clearing shards: {resp.text}"
            )


class DataProxySessionRemover:
    """Removes a v2 session via ``POST /export_trajectories`` (remove_session=True).

    The data_proxy exposes session removal only through the export endpoint
    (``store.remove_session`` is called when ``remove_session`` is set); the
    exported traj is discarded - this is a cleanup-only call.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = _transport

    def _client(self) -> httpx.Client:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        kwargs: dict[str, Any] = {"headers": headers, "timeout": 10.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def remove(self, session_id: str) -> None:
        with self._client() as client:
            resp = client.post(
                f"{self._base}/export_trajectories",
                json={"session_ids": [session_id], "remove_session": True},
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"unexpected {resp.status_code} removing session {session_id}: {resp.text}"
            )


def run_segment_dag_training_step(
    *,
    client: MulticaDagClient,
    resolver: TrainingTensorResolver,
    session_remover: SessionRemover,
    project_id: str,
    gamma: float = 1.0,
    lam: float = 1.0,
    poll_timeout: float = 30.0,
    poll_interval: float = 1.0,
    assembler: SuperNodeAssembler | None = None,
) -> AssembledAdvantages:
    """Run one minimal segment-DAG training step with placeholder zero reward.

    Pipeline: fetch the assembled DAG from Multica, resolve tensor refs and
    build the ``ExecutionDAG``, run global GAE over the completion-ordered
    SuperNodes with zero reward (placeholder - change 2 adds judge + critic V),
    then release the resolved shards and revoke the sessions.

    Args:
        client: Polls Multica's ``GET .../dag`` for the ``AssembledDag``.
        resolver: Resolves each segment's ``tensor_ref`` and clears shards.
        session_remover: Revokes each consumed v2 session.
        project_id: The Multica project whose DAG to train on.
        gamma/lam: GAE discount / lambda (defaults are no-discount; change 2
            tunes them).
        poll_timeout/poll_interval: Forwarded to ``client.get_dag``.
        assembler: Inject for tests; defaults to a fresh ``SuperNodeAssembler``.

    Returns:
        Per-node GAE advantages/returns/baselines (zero with placeholder
        reward - the point is that the path runs end-to-end without error).

    Raises:
        DagError: If the DAG is not ready in time, or is missing/forbidden.
        DAGError: If the assembled graph contains a cycle or a dangling edge.
    """
    dag = client.get_dag(
        project_id, timeout=poll_timeout, interval=poll_interval
    )
    asm = assembler or SuperNodeAssembler()
    edag = asm.assemble_from_refs(dag, resolver)
    ordered = edag.topological_order()
    advantages = assemble_node_advantages(
        ordered, initial_value=0.0, gamma=gamma, lam=lam
    )

    # Tensor lifecycle cleanup: release resolved shards, then revoke sessions.
    shard_ids: list[str] = []
    for seg in dag.segments:
        for ref in (seg.tensor_ref or {}).values():
            if isinstance(ref, dict) and ref.get("shard_id"):
                shard_ids.append(ref["shard_id"])
    resolver.clear(shard_ids)
    for session_id in dag.session_to_agent_run:
        session_remover.remove(session_id)

    return advantages


__all__ = [
    "DataProxySessionRemover",
    "DataProxyTensorResolver",
    "SessionRemover",
    "TrainingTensorResolver",
    "run_segment_dag_training_step",
]

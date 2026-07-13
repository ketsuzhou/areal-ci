"""HTTP client for fetching Multica's assembled DAG via the db_bridge stub.

Polls ``GET /api/v1/env-dispatch/{project_id}/dag`` (through the areal-side
bridge stub at ``AREAL_BRIDGE_STUB_URL``) until it returns 200 with the assembled
DAG (structure only - no scores, no turn indices, no message text), or raises
``DagNotFound`` (404) / ``DagForbidden`` (403) / ``DagTimeout``. Bridge-level
transient responses (202 not-ready, 502/503/504) are re-polled, not raised.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


class DagError(Exception):
    """Base error for DAG fetch failures."""


class DagNotFound(DagError):
    """Raised when the endpoint returns 404."""


class DagForbidden(DagError):
    """Raised when the endpoint returns 403."""


class DagTimeout(DagError):
    """Raised when polling times out before the DAG is ready."""


@dataclass
class SegmentSpec:
    """One communication-bounded segment, as defined by Multica.

    Carries a ``tensor_ref`` (resolved later by the assembler) instead of turn
    indices or message text. No judge scores live here.

    tensor_ref is a field->shard map: {"input_ids": {"shard_id": str, "node_addr": str}, ...}.
    """

    segment_id: str
    agent_run_id: str
    issue_id: str
    trajectory_id: int
    tensor_ref: dict[str, Any]
    closing_event: str | None
    env_snapshot: dict[str, Any]


@dataclass
class EdgeSpec:
    """One typed DAG edge between segments, as defined by Multica."""

    src_segment_id: str
    dst_segment_id: str
    type: str
    # BRANCH provenance (None for non-branch edges). Multica emits these on
    # branch edges so AReaL can credit the forked checkpoint via MCTS backup.
    branch_from_segment_id: str | None = None
    branch_from_checkpoint_id: str | None = None


@dataclass
class AssembledDag:
    """The fully assembled DAG from Multica: segments, edges, and the
    session_id -> agent_run_id mapping areal uses to attribute trajectories."""

    segments: list[SegmentSpec]
    edges: list[EdgeSpec]
    session_to_agent_run: dict[str, str]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AssembledDag:
        return cls(
            segments=[SegmentSpec(**s) for s in d.get("segments", [])],
            edges=[EdgeSpec(**e) for e in d.get("edges", [])],
            session_to_agent_run=d.get("session_to_agent_run", {}),
        )


class MulticaDagClient:
    """Synchronous client that polls the db_bridge stub for an assembled DAG.

    The base URL defaults to ``base_url`` or the ``AREAL_BRIDGE_STUB_URL`` env
    var (the areal-side bridge stub that forwards to multica). No API key is
    sent: the stub is loopback and the multica executor injects the upstream
    key. Polling is config-driven with sane defaults: ``poll_interval`` (initial
    seconds between polls, default 2.0), ``poll_timeout`` (overall deadline,
    default 300.0), ``poll_backoff`` (interval growth factor, default 1.5), and
    ``poll_max_interval`` (backoff cap, default 10.0). ``http_timeout`` (default
    10.0) bounds each HTTP request. ``get_dag`` accepts per-call ``timeout`` /
    ``interval`` overrides that fall back to the configured defaults.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        poll_interval: float = 2.0,
        poll_timeout: float = 300.0,
        poll_backoff: float = 1.5,
        poll_max_interval: float = 10.0,
        http_timeout: float = 10.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = (
            base_url or os.environ.get("AREAL_BRIDGE_STUB_URL") or ""
        ).rstrip("/")
        if not self._base:
            raise ValueError(
                "MulticaDagClient requires base_url or AREAL_BRIDGE_STUB_URL"
            )
        self._transport = _transport
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._poll_backoff = poll_backoff
        self._poll_max_interval = poll_max_interval
        self._http_timeout = http_timeout

    def get_dag(
        self,
        project_id: str,
        *,
        timeout: float | None = None,
        interval: float | None = None,
    ) -> AssembledDag:
        url = f"{self._base}/api/v1/env-dispatch/{project_id}/dag"
        # Per-call overrides fall back to the client's configured defaults.
        poll_timeout = timeout if timeout is not None else self._poll_timeout
        poll_interval = interval if interval is not None else self._poll_interval
        deadline = time.monotonic() + poll_timeout
        client_kwargs: dict[str, Any] = {"timeout": self._http_timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        current_interval = poll_interval
        with httpx.Client(**client_kwargs) as client:
            while True:
                resp = client.get(url)
                if resp.status_code == 200:
                    return AssembledDag.from_dict(resp.json())
                # 202 = not ready yet; 502/503/504 = bridge/transient (timeout,
                # relay error, unavailable). All are re-polled up to the
                # wall-clock deadline, then DagTimeout.
                if resp.status_code in (202, 502, 503, 504):
                    if time.monotonic() >= deadline:
                        raise DagTimeout(
                            f"dag for {project_id} not ready in {poll_timeout}s"
                        )
                    time.sleep(current_interval)
                    # Backoff: grow the poll interval up to the configured cap so
                    # a slow-to-assemble DAG does not hammer the endpoint. The
                    # factor defaults to 1.5 (>1.0 grows; 1.0 is a steady poll).
                    if self._poll_backoff > 1.0:
                        current_interval = min(
                            current_interval * self._poll_backoff,
                            self._poll_max_interval,
                        )
                    continue
                if resp.status_code == 404:
                    raise DagNotFound(project_id)
                if resp.status_code == 403:
                    raise DagForbidden(project_id)
                raise DagError(f"unexpected {resp.status_code}: {resp.text}")


__all__ = [
    "AssembledDag",
    "DagError",
    "DagForbidden",
    "DagNotFound",
    "DagTimeout",
    "EdgeSpec",
    "MulticaDagClient",
    "SegmentSpec",
]

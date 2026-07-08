"""HTTP client for fetching Multica's assembled DAG.

Polls ``GET /api/v1/env-dispatch/{project_id}/dag`` until it returns 200 with the
assembled DAG (structure only - no scores, no turn indices, no message text), or
raises ``DagNotFound`` (404) / ``DagForbidden`` (403) / ``DagTimeout``.
"""

from __future__ import annotations

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
    """Synchronous client that polls Multica for an assembled DAG."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = _transport

    def get_dag(
        self, project_id: str, *, timeout: float, interval: float
    ) -> AssembledDag:
        url = f"{self._base}/api/v1/env-dispatch/{project_id}/dag"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        deadline = time.monotonic() + timeout
        client_kwargs: dict[str, Any] = {"headers": headers, "timeout": 10.0}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        with httpx.Client(**client_kwargs) as client:
            while True:
                resp = client.get(url)
                if resp.status_code == 200:
                    return AssembledDag.from_dict(resp.json())
                if resp.status_code == 202:
                    if time.monotonic() >= deadline:
                        raise DagTimeout(
                            f"dag for {project_id} not ready in {timeout}s"
                        )
                    time.sleep(interval)
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

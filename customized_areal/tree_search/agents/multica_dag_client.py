"""HTTP client for fetching MultiCA's assembled DAG directly.

Polls ``GET /api/v1/env-dispatch/{project_id}/dag`` on ``MULTICA_BASE_URL``
with the caller's MultiCA PAT until it returns an assembled DAG, or raises a
typed DAG error. Transient responses are re-polled up to the configured deadline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from customized_areal.tree_search.agents.multica_auth import (
    login_guidance,
    normalize_base_url,
    resolve_api_key,
)


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
class StepReward:
    """One per-LLM-output diagnosis reward, keyed by ``(segment_id, seq)``.

    Emitted by the Multica Pi diagnosis agent at collaborative-task terminal and
    served via ``/dag`` ``step_rewards[]``. ``score`` is an integer in
    ``[0, score_max]`` (clamped by the diagnosis runner). AReaL normalizes the
    per-turn scores into a per-segment ``SuperNode.process_reward``.
    """

    segment_id: str
    seq: int
    score: int
    rationale: str


@dataclass
class AssembledDag:
    """The fully assembled DAG from Multica: segments, edges, the
    session_id -> agent_run_id mapping areal uses to attribute trajectories, and
    the diagnosis agent's per-turn step rewards."""

    segments: list[SegmentSpec]
    edges: list[EdgeSpec]
    session_to_agent_run: dict[str, str]
    # Diagnosis per-LLM-output rewards + scoring scale (served by /dag). Absent
    # when the diagnosis agent did not run: empty list + 0 (never fabricated).
    step_rewards: list[StepReward] = field(default_factory=list)
    score_max: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AssembledDag:
        return cls(
            segments=[SegmentSpec(**s) for s in d.get("segments", [])],
            edges=[EdgeSpec(**e) for e in d.get("edges", [])],
            session_to_agent_run=d.get("session_to_agent_run", {}),
            step_rewards=[StepReward(**sr) for sr in d.get("step_rewards", [])],
            score_max=d.get("score_max", 0),
        )


class MulticaDagClient:
    """Synchronous client that polls MultiCA directly for an assembled DAG.

    The base URL defaults to ``base_url`` or ``MULTICA_BASE_URL``. Credentials
    resolve from ``api_key``, ``MULTICA_API_KEY``, then the saved PAT created by
    :mod:`multica_auth`. Polling uses ``poll_interval`` (initial
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
        api_key: str | None = None,
        poll_interval: float = 2.0,
        poll_timeout: float = 300.0,
        poll_backoff: float = 1.5,
        poll_max_interval: float = 10.0,
        http_timeout: float = 10.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_base_url = base_url or os.environ.get("MULTICA_BASE_URL") or ""
        if not resolved_base_url:
            raise ValueError("MulticaDagClient requires base_url or MULTICA_BASE_URL")
        self._base = normalize_base_url(resolved_base_url)
        self._api_key = resolve_api_key(self._base, api_key)
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
        headers = {"Authorization": f"Bearer {self._api_key}"}
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
                request_succeeded = False
                try:
                    resp = client.get(url, headers=headers)
                except httpx.RequestError:
                    pass
                else:
                    request_succeeded = True
                if not request_succeeded:
                    raise DagError("MultiCA DAG network request failed")
                if resp.status_code == 200:
                    return AssembledDag.from_dict(resp.json())
                # 202 = not ready yet; gateway-like 502/503/504 responses are
                # transient. All are re-polled to the wall-clock deadline.
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
                if resp.status_code == 401:
                    raise DagError(
                        "MultiCA authentication failed: status=401. "
                        + login_guidance(self._base)
                    )
                raise DagError(f"unexpected status={resp.status_code}")


__all__ = [
    "AssembledDag",
    "DagError",
    "DagForbidden",
    "DagNotFound",
    "DagTimeout",
    "EdgeSpec",
    "MulticaDagClient",
    "SegmentSpec",
    "StepReward",
]

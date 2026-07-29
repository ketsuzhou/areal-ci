"""HTTP client for fetching MultiCA's assembled DAG directly.

Polls the dispatch-scoped assembled-DAG endpoint on ``MULTICA_BASE_URL`` with
the caller's MultiCA PAT until it returns an assembled DAG, or raises a typed DAG
error. Message dispatches poll ``GET /api/v1/env-dispatch/channels/{channel_id}/dag``;
issue dispatches poll ``GET /api/v1/env-dispatch/{project_id}/dag``. Transient
responses are re-polled up to the configured deadline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from customized_areal.tree_search.agents.multica_auth import (
    login_guidance,
    normalize_base_url,
    resolve_api_key,
)

if TYPE_CHECKING:
    # EnvDispatchHandle is a lightweight dataclass; imported only for typing so
    # this module stays decoupled from multica_client at runtime.
    from customized_areal.tree_search.agents.multica_client import EnvDispatchHandle


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

    Dual-source segments: ``trajectory_source`` distinguishes between AReaL
    tensors (``areal_tensor``, trainable) and local Multica task messages
    (``task_messages``, non-trainable). Only ``trainable=true`` segments reach
    tensor resolution and shard cleanup.
    """

    segment_id: str
    agent_run_id: str
    issue_id: str
    trajectory_id: int | None = None
    tensor_ref: dict[str, Any] | None = None
    closing_event: str | None = None
    env_snapshot: dict[str, Any] = field(default_factory=dict)
    trajectory_source: str = "areal_tensor"
    trainable: bool = True
    trajectory: list = field(default_factory=list)


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
        if not isinstance(d, dict):
            raise DagError("assembled DAG payload must be an object")
        raw_segments = d.get("segments")
        raw_edges = d.get("edges")
        session_to_agent_run = d.get("session_to_agent_run")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise DagError("assembled DAG must contain at least one segment")
        if not isinstance(raw_edges, list):
            raise DagError("assembled DAG edges must be a list")
        if not isinstance(session_to_agent_run, dict):
            raise DagError("assembled DAG session_to_agent_run must be an object")

        try:
            segments = [SegmentSpec(**segment) for segment in raw_segments]
            edges = [EdgeSpec(**edge) for edge in raw_edges]
            step_rewards = [
                StepReward(**reward) for reward in d.get("step_rewards", [])
            ]
        except (TypeError, ValueError) as exc:
            raise DagError(f"invalid assembled DAG record: {exc}") from exc

        # ── Dual-source validation ──────────────────────────────────
        for segment in segments:
            if segment.trajectory_source == "areal_tensor":
                if segment.trajectory_id is None:
                    raise DagError(
                        f"areal_tensor segment {segment.segment_id!r} "
                        f"missing trajectory_id"
                    )
                if not segment.tensor_ref:
                    raise DagError(
                        f"areal_tensor segment {segment.segment_id!r} "
                        f"missing tensor_ref"
                    )
            elif segment.trajectory_source == "task_messages":
                if segment.trajectory_id is not None:
                    raise DagError(
                        f"task_messages segment {segment.segment_id!r} "
                        f"has unexpected trajectory_id"
                    )
                if segment.tensor_ref is not None:
                    raise DagError(
                        f"task_messages segment {segment.segment_id!r} "
                        f"has unexpected tensor_ref"
                    )

        segment_ids = [segment.segment_id for segment in segments]
        if any(not segment_id for segment_id in segment_ids):
            raise DagError("assembled DAG contains an empty segment_id")
        if len(set(segment_ids)) != len(segment_ids):
            raise DagError("assembled DAG contains duplicate segment_id values")

        known_segments = set(segment_ids)
        adjacency = {segment_id: [] for segment_id in segment_ids}
        indegree = {segment_id: 0 for segment_id in segment_ids}
        for edge in edges:
            if edge.src_segment_id not in known_segments:
                raise DagError(
                    f"assembled DAG edge has unknown source {edge.src_segment_id!r}"
                )
            if edge.dst_segment_id not in known_segments:
                raise DagError(
                    f"assembled DAG edge has unknown destination {edge.dst_segment_id!r}"
                )
            adjacency[edge.src_segment_id].append(edge.dst_segment_id)
            indegree[edge.dst_segment_id] += 1

        ready = [segment_id for segment_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            segment_id = ready.pop()
            visited += 1
            for child_id in adjacency[segment_id]:
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)
        if visited != len(segment_ids):
            raise DagError("assembled DAG contains a cycle")

        known_agent_runs = {segment.agent_run_id for segment in segments}
        unknown_agent_runs = {
            agent_run_id
            for agent_run_id in session_to_agent_run.values()
            if agent_run_id not in known_agent_runs
        }
        if unknown_agent_runs:
            raise DagError(
                "assembled DAG session mapping references an unknown agent run"
            )

        return cls(
            segments=segments,
            edges=edges,
            session_to_agent_run=session_to_agent_run,
            step_rewards=step_rewards,
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
        workspace_slug: str | None = None,
        workspace_id: str | None = None,
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
        # Mirrors MulticaClient: a user PAT carries no workspace, so the server
        # resolves it from ?workspace_slug / ?workspace_id. Without this the
        # dispatch-scoped /dag routes reject the request with 400 "workspace ID
        # required" before any lookup happens.
        workspace_slug = (
            workspace_slug or os.environ.get("MULTICA_WORKSPACE_SLUG") or None
        )
        workspace_id = workspace_id or os.environ.get("MULTICA_WORKSPACE_ID") or None
        if workspace_slug and workspace_id:
            raise ValueError("pass at most one of workspace_slug or workspace_id")
        self._workspace_slug = workspace_slug
        self._workspace_id = workspace_id

    def _params(self) -> dict[str, str]:
        """Workspace query params for user-PAT auth (empty for task-token auth)."""
        if self._workspace_slug:
            return {"workspace_slug": self._workspace_slug}
        if self._workspace_id:
            return {"workspace_id": self._workspace_id}
        return {}

    @staticmethod
    def _incomplete_reason(payload: Any) -> str | None:
        """Describe why a 200 payload is not an assembled DAG yet, else None.

        A 200 does not guarantee a complete DAG. multica marks the root task
        terminal before ``CloseSegmentForEvent`` inserts the segment row, and
        reports a terminal-but-not-yet-dense dispatch as ``{"status": ...}``.
        Both windows are transient, so they are re-polled to the deadline
        instead of failing the episode with a misleading structural error.
        Non-dict payloads fall through to ``from_dict`` for a precise message.
        """
        if not isinstance(payload, dict):
            return None
        segments = payload.get("segments")
        if isinstance(segments, list) and segments:
            return None
        status = payload.get("status")
        if isinstance(status, str) and status:
            return f"status={status}"
        return "no segments yet"

    def get_dag(
        self,
        handle: EnvDispatchHandle | str | None = None,
        *,
        project_id: str | None = None,
        channel_id: str | None = None,
        dispatch_type: str = "issue",
        timeout: float | None = None,
        interval: float | None = None,
    ) -> AssembledDag:
        """Poll the dispatch-scoped DAG endpoint until assembled.

        Accepts either an :class:`~.multica_client.EnvDispatchHandle` (preferred)
        or explicit ``channel_id`` / ``project_id`` + ``dispatch_type`` so
        existing issue callers stay source-compatible: a positional project id
        string is treated as an issue dispatch. Message dispatches route to
        ``/api/v1/env-dispatch/channels/{channel_id}/dag``; issue dispatches
        route to ``/api/v1/env-dispatch/{project_id}/dag``.

        ``get_dag`` accepts per-call ``timeout`` / ``interval`` overrides that
        fall back to the configured defaults.
        """
        if handle is not None:
            if isinstance(handle, str):
                # Legacy positional project_id (issue dispatch).
                project_id = handle
                dispatch_type = "issue"
            else:
                project_id = handle.project_id
                channel_id = handle.channel_id
                dispatch_type = handle.dispatch_type
        if dispatch_type == "message":
            if not channel_id:
                raise DagError("message dispatch handle missing channel_id")
            url = f"{self._base}/api/v1/env-dispatch/channels/{channel_id}/dag"
            label = channel_id
        else:
            if not project_id:
                raise DagError("issue dispatch handle missing project_id")
            url = f"{self._base}/api/v1/env-dispatch/{project_id}/dag"
            label = project_id
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # Per-call overrides fall back to the client's configured defaults.
        poll_timeout = timeout if timeout is not None else self._poll_timeout
        poll_interval = interval if interval is not None else self._poll_interval
        deadline = time.monotonic() + poll_timeout
        client_kwargs: dict[str, Any] = {"timeout": self._http_timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        current_interval = poll_interval
        not_ready_reason = "not ready"
        with httpx.Client(**client_kwargs) as client:
            while True:
                request_succeeded = False
                try:
                    resp = client.get(url, headers=headers, params=self._params())
                except httpx.RequestError:
                    pass
                else:
                    request_succeeded = True
                if not request_succeeded:
                    raise DagError("MultiCA DAG network request failed")
                retryable = False
                if resp.status_code == 200:
                    payload = resp.json()
                    incomplete = self._incomplete_reason(payload)
                    if incomplete is None:
                        return AssembledDag.from_dict(payload)
                    not_ready_reason = incomplete
                    retryable = True
                # 202 = not ready yet; gateway-like 502/503/504 responses are
                # transient. All are re-polled to the wall-clock deadline.
                elif resp.status_code in (202, 502, 503, 504):
                    not_ready_reason = f"http {resp.status_code}"
                    retryable = True
                if retryable:
                    if time.monotonic() >= deadline:
                        raise DagTimeout(
                            f"dag for {label} not ready in {poll_timeout}s "
                            f"({not_ready_reason})"
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
                    raise DagNotFound(label)
                if resp.status_code == 403:
                    raise DagForbidden(label)
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

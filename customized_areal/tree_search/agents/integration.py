"""Branch materialization -- orchestrates snapshot + issue fork + start branch.

Per design §3.1 (snapshot-at-frontier + transcript replay), when a node is
selected as a branch candidate:

  1. :meth:`ForkableEnvironment.snapshot` -- snapshot the source agent's sandbox.
  2. :meth:`ForkableEnvironment.fork` -- fork a fresh sandbox from the snapshot.
  3. :class:`MulticaIssueForker.fork` -- fork the Multica issue subtree at
     ``(task_id, seq)`` (the Phase 1 endpoint).
  4. ``starter.start_branch`` -- bind the forked sandbox + forked issue, replay
     ``messages <= seq``, and drop ``PriorSessionID``.

If any step fails, the prior steps are rolled back (paired operations stay
together -- ``.claude/rules/code-quality.md``).

Uses stdlib :mod:`logging` so the ``dag`` package stays importable without the
heavy training stack.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from customized_areal.tree_search.agents.environment import ForkableEnvironment
from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("BranchMaterializer")


class MulticaIssueForker:
    """HTTP client for the Multica issue fork endpoints (Phase 1).

    Wraps ``POST /api/issues/{id}/fork?task_id=...&seq=...`` and
    ``DELETE /api/issues/{id}/fork``. ``base_url`` falls back to
    ``MULTICA_BASE_URL`` and ``api_key`` to ``MULTICA_API_KEY``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError("MulticaIssueForker requires base_url or MULTICA_BASE_URL")
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def fork(self, *, issue_id: str, task_id: str, seq: int) -> str:
        """POST the fork endpoint; returns the new ``forked_issue_id``."""
        resp = await self._client.post(
            f"/api/issues/{issue_id}/fork",
            params={"task_id": task_id, "seq": str(seq)},
            headers=self._headers(),
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"Multica issue fork failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )
        body = resp.json()
        forked_id = body.get("forked_issue_id")
        if not isinstance(forked_id, str) or not forked_id:
            raise RuntimeError(f"fork response missing forked_issue_id: {body!r}")
        return forked_id

    async def delete_fork(self, *, issue_id: str) -> None:
        """DELETE the forked issue. Idempotent: 404 is treated as success."""
        resp = await self._client.delete(
            f"/api/issues/{issue_id}/fork", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"Multica issue fork delete failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )


class BranchStarter(Protocol):
    """The ``agent_start_branch`` channel client seam."""

    async def start_branch(
        self,
        *,
        forked_sandbox_id: str,
        forked_issue_id: str,
        replay_messages: list[dict[str, Any]],
        drop_prior_session_id: bool,
    ) -> str: ...


@dataclass(frozen=True)
class BranchMaterializationResult:
    """Result of a successful branch materialization."""

    branch_run_id: str
    forked_sandbox_id: str
    forked_issue_id: str
    snapshot_id: str


class BranchMaterializer:
    """Orchestrates snapshot + issue fork + ``agent_start_branch``.

    On any step failure, rolls back the prior steps (paired operations stay
    together). The snapshot itself needs no rollback (it is cheap and
    eventually GC'd by the Fleet side); the forked sandbox and forked issue are
    cleaned up on failure of a later step.
    """

    def __init__(
        self,
        *,
        env: ForkableEnvironment,
        forker: MulticaIssueForker,
        starter: BranchStarter,
    ) -> None:
        self._env = env
        self._forker = forker
        self._starter = starter

    async def materialize(
        self,
        *,
        source_sandbox_id: str,
        source_issue_id: str,
        task_id: str,
        seq: int,
        replay_messages: list[dict[str, Any]],
    ) -> BranchMaterializationResult:
        """Materialize a branch. Raises on failure after rolling back."""
        # Step 1: snapshot the live sandbox (no rollback needed on failure).
        snap = await self._env.snapshot(source_sandbox_id)

        # Step 2: fork a fresh sandbox from the snapshot.
        try:
            forked_sbx = await self._env.fork(snapshot_id=snap.snapshot_id)
        except Exception:
            logger.error("sandbox fork failed (snapshot_id=%s)", snap.snapshot_id)
            raise

        # Step 3: fork the Multica issue subtree.
        try:
            forked_issue_id = await self._forker.fork(
                issue_id=source_issue_id, task_id=task_id, seq=seq
            )
        except Exception:
            logger.error("issue fork failed; rolling back sandbox fork")
            await self._safe_cleanup(forked_sbx.sandbox_id)
            raise

        # Step 4: start the branch run (replay prefix, drop PriorSessionID).
        try:
            branch_run_id = await self._starter.start_branch(
                forked_sandbox_id=forked_sbx.sandbox_id,
                forked_issue_id=forked_issue_id,
                replay_messages=replay_messages,
                drop_prior_session_id=True,
            )
        except Exception:
            logger.error("agent_start_branch failed; rolling back fork + issue")
            await self._safe_cleanup(forked_sbx.sandbox_id)
            await self._safe_delete_fork(forked_issue_id)
            raise

        return BranchMaterializationResult(
            branch_run_id=branch_run_id,
            forked_sandbox_id=forked_sbx.sandbox_id,
            forked_issue_id=forked_issue_id,
            snapshot_id=snap.snapshot_id,
        )

    async def _safe_cleanup(self, sandbox_id: str) -> None:
        try:
            await self._env.cleanup(sandbox_id)
        except Exception:
            logger.warning(
                "cleanup failed during rollback (sandbox_id=%s)",
                sandbox_id,
                exc_info=True,
            )

    async def _safe_delete_fork(self, issue_id: str) -> None:
        try:
            await self._forker.delete_fork(issue_id=issue_id)
        except Exception:
            logger.warning(
                "delete_fork failed during rollback (issue_id=%s)",
                issue_id,
                exc_info=True,
            )


# --------------------------------------------------------------------------- #
# Workflow integration helpers (adapted Tasks 12/15)
#
# The Phase 4 plan placed these on a ``TreeSearchGroupedWorkflow`` class with
# ``_finalize_with_verifier`` / ``_materialize_cloud_branch`` / ``_cleanup_branch``
# methods. The real workflow (``core/customized_grouped_workflow.py``) has NO
# such class -- it uses module-level functions (``select_branch_candidate``,
# ``build_branch_task``). So these are module-level helpers matching the real
# structure. They are typed against the structural ``BranchCandidate`` Protocol
# (which ``core.tree_store.Node`` satisfies) so this module stays importable
# without torch.
#
# NOTE (deferred): wiring these into the live episode loop in
# ``customized_grouped_workflow.py`` is intentionally NOT done here. That step
# depends on the full Phase 2 (verifier agent) and Phase 3 (DAG reward backup +
# cloud-env candidate selection) being in place and on a runtime stack
# (torch + live Multica/Fleet) not available in unit tests. The integration
# point is ``build_branch_task`` (legacy local-sandbox path) -- a cloud-env
# branch should call ``materialize_cloud_branch`` when the candidate carries a
# ``branch_env_snapshot_id``.
# --------------------------------------------------------------------------- #
class BranchCandidate(Protocol):
    """Structural view of a tree-store ``Node`` for branch operations."""

    task_id: str
    turn_idx: int
    branch_sandbox_id: str | None
    branch_issue_id: str | None
    branch_env_snapshot_id: str | None


async def finalize_with_verifier(
    *,
    verifier: Any,
    writer: Any,
    session_id: str,
    run: dict[str, Any],
) -> VerifierResult:
    """Run the verifier and write the reward to the RL session.

    Replaces the constant ``set_reward(1.0)``. Returns the
    :class:`VerifierResult` so the caller can inspect it.
    """
    result = await verifier.verify(run)
    await writer.finalize(session_id=session_id, verifier_result=result)
    return result


async def materialize_cloud_branch(
    *,
    materializer: BranchMaterializer,
    candidate: BranchCandidate,
    source_sandbox_id: str,
    source_issue_id: str,
    replay_messages: list[dict[str, Any]],
) -> str:
    """Materialize a cloud-env branch for ``candidate``; returns branch_run_id.

    Dispatched for candidates carrying ``branch_env_snapshot_id`` (the cloud-env
    path). The legacy ``branch_sandbox_id`` path (``build_branch_task``) is
    unchanged.
    """
    result = await materializer.materialize(
        source_sandbox_id=source_sandbox_id,
        source_issue_id=source_issue_id,
        task_id=candidate.task_id,
        seq=candidate.turn_idx,
        replay_messages=replay_messages,
    )
    return result.branch_run_id


async def cleanup_cloud_branch(
    *,
    env: ForkableEnvironment,
    forker: MulticaIssueForker,
    candidate: BranchCandidate,
) -> None:
    """Delete the forked sandbox + forked issue for a cloud-env branch.

    Best-effort and idempotent: failures are logged, not raised. Mirrors the
    plan's extended ``_cleanup_branch`` adapted to the real
    :class:`ForkableEnvironment` seam.
    """
    if candidate.branch_sandbox_id:
        try:
            await env.cleanup(candidate.branch_sandbox_id)
        except Exception:
            logger.warning(
                "failed to delete branch sandbox (sandbox_id=%s)",
                candidate.branch_sandbox_id,
                exc_info=True,
            )
    if candidate.branch_issue_id:
        try:
            await forker.delete_fork(issue_id=candidate.branch_issue_id)
        except Exception:
            logger.warning(
                "failed to delete forked issue (issue_id=%s)",
                candidate.branch_issue_id,
                exc_info=True,
            )

"""HTTP client for the multica SWE-Lego atomic endpoint.

Wraps ``POST /api/v1/swe-lego/issues`` (spec §4.1) and the cleanup endpoint.
``base_url`` / ``api_key`` default to ``MULTICA_BASE_URL`` / ``MULTICA_API_KEY``.
Uses stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
import os

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoSetup,
)

logger = logging.getLogger("MulticaSweLegoClient")


class MulticaSweLegoClient:
    """Thin HTTP client for the multica SWE-Lego orchestration endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError(
                "MulticaSweLegoClient requires base_url or MULTICA_BASE_URL"
            )
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_swe_lego_issue(
        self,
        *,
        issue: SweLegoIssue,
        group_size: int,
        agent_config_id: str,
        base_image: str | None = None,
    ) -> SweLegoSetup:
        """POST the atomic endpoint; returns the :class:`SweLegoSetup`."""
        payload: dict = {
            "repo_url": issue.repo_url,
            "base_commit": issue.base_commit,
            "issue_date": issue.issue_date,
            "issue_text": issue.issue_text,
            "issue_title": issue.issue_title,
            "acceptance_criteria": issue.acceptance_criteria,
            "fail_to_pass": list(issue.fail_to_pass),
            "pass_to_pass": list(issue.pass_to_pass),
            "group_size": group_size,
            "agent_config_id": agent_config_id,
        }
        if base_image is not None:
            payload["base_image"] = base_image
        resp = await self._client.post(
            "/api/v1/swe-lego/issues", json=payload, headers=self._headers()
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"create swe-lego issue failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )
        body = resp.json()
        return SweLegoSetup(
            project_id=body["project_id"],
            issue_id=body["issue_id"],
            image_id=body["image_id"],
            build_node_id=body["build_node_id"],
            base_sandbox_id=body["base_sandbox_id"],
            base_sandbox_runtime_id=body["base_sandbox_runtime_id"],
            agent_run_ids=list(body["agent_run_ids"]),
        )

    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None:
        """DELETE the per-issue project (cascades to sandboxes + issue)."""
        resp = await self._client.delete(
            f"/api/v1/swe-lego/issues/{project_id}", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"cleanup swe-lego issue failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )

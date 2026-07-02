"""HTTP client for the multica unified env-dispatch API.

Wraps ``POST /api/v1/env``, ``DELETE /api/v1/env/{envID}``,
``POST /api/v1/env-dispatch``, and ``DELETE /api/v1/env-dispatch/{projectID}``
(spec §6). Uses stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
import os

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoRollout,
    SweLegoSetup,
)

logger = logging.getLogger("MulticaEnvDispatchClient")


class MulticaEnvDispatchClient:
    """Thin HTTP client for the multica env-dispatch API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            raise ValueError("MulticaEnvDispatchClient requires base_url or MULTICA_BASE_URL")
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

    async def create_base_env(self, *, image_ref: str) -> str:
        """POST /api/v1/env — boot a sandbox from image_ref, return env_id."""
        resp = await self._client.post(
            "/api/v1/env", json={"image_ref": image_ref}, headers=self._headers()
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"create_base_env failed: status={resp.status_code} body={resp.text[:200]}"
            )
        return resp.json()["env_id"]

    async def delete_env(self, *, env_id: str) -> None:
        """DELETE /api/v1/env/{envID} — idempotent on 404."""
        resp = await self._client.delete(
            f"/api/v1/env/{env_id}", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"delete_env failed: status={resp.status_code} body={resp.text[:200]}"
            )

    async def create_env_dispatch(
        self,
        *,
        mode: str,
        env_id: str | None = None,
        dispatch_type: str,
        agent_id: str | None = None,
        squad_id: str | None = None,
        group_size: int = 1,
        domain: str | None = None,
        issue: SweLegoIssue | None = None,
        message: str | None = None,
    ) -> SweLegoSetup:
        """POST /api/v1/env-dispatch — unified dispatch (spec §6.3)."""
        payload: dict = {
            "mode": mode,
            "dispatch_type": dispatch_type,
            "group_size": group_size,
        }
        if env_id:
            payload["env_id"] = env_id
        if agent_id:
            payload["agent_id"] = agent_id
        if squad_id:
            payload["squad_id"] = squad_id
        if domain is not None:
            payload["domain"] = domain
        if issue is not None:
            payload["issue"] = {
                "title": issue.issue_title,
                "description": issue.issue_text,
                "acceptance_criteria": [issue.acceptance_criteria] if issue.acceptance_criteria else [],
                "fail_to_pass": list(issue.fail_to_pass),
                "pass_to_pass": list(issue.pass_to_pass),
            }
        if message is not None:
            payload["message"] = {"content": message}

        resp = await self._client.post(
            "/api/v1/env-dispatch", json=payload, headers=self._headers()
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"create_env_dispatch failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        rollouts = [
            SweLegoRollout(
                env_id=r["env_id"],
                project_id=r["project_id"],
                issue_id=r.get("issue_id", ""),
                chat_session_id=r.get("chat_session_id", ""),
                agent_run_id=r.get("agent_run_id", ""),
            )
            for r in body["rollouts"]
        ]
        return SweLegoSetup(rollouts=rollouts)

    async def cleanup_env_dispatch(self, *, project_id: str) -> None:
        """DELETE /api/v1/env-dispatch/{projectID} — cascades to issues/chat/tasks."""
        resp = await self._client.delete(
            f"/api/v1/env-dispatch/{project_id}", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"cleanup_env_dispatch failed: status={resp.status_code} body={resp.text[:200]}"
            )

    # Back-compat alias for the old runner signature.
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None:
        await self.cleanup_env_dispatch(project_id=project_id)

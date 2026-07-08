"""HTTP client for the multica unified env-dispatch API.

Wraps ``POST /api/v1/env``, ``DELETE /api/v1/env/{envID}``,
``POST /api/v1/env-dispatch``, and ``DELETE /api/v1/env-dispatch/{projectID}``
(spec §6). Uses stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging

from customized_areal.tree_search.env_checkpoint import (
    should_create_entropy_checkpoint,
)
import os

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoRollout,
    SweLegoSetup,
)

logger = logging.getLogger("MulticaEnvDispatchClient")


class MulticaCheckpointError(RuntimeError):
    """Typed error for env-checkpoint API failures (403/404/409).

    Carries the HTTP status code so AReaL can distinguish a non-resumable
    checkpoint (409 conflict), a missing/forbidden checkpoint (404/403),
    and retry accordingly.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"checkpoint api failed: status={status_code} body={body}")
        self.status_code = status_code
        self.body = body


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
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError(
                "MulticaEnvDispatchClient requires base_url or MULTICA_BASE_URL"
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
        per_agent_env: dict[str, dict] | None = None,
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
                "acceptance_criteria": [issue.acceptance_criteria]
                if issue.acceptance_criteria
                else [],
                "fail_to_pass": list(issue.fail_to_pass),
                "pass_to_pass": list(issue.pass_to_pass),
            }
        if message is not None:
            payload["message"] = {"content": message}
        if per_agent_env:
            payload["per_agent_env"] = per_agent_env

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

    async def create_checkpoint(
        self,
        *,
        project_id: str,
        event_ref: str,
        checkpoint_kind: str,
        env_id_map: dict[str, str],
        sandbox_refs: list[dict],
        entropy_score: float | None = None,
        save_timeout_ms: int | None = None,
    ) -> dict:
        """POST /api/v1/env-checkpoints - synchronous save with timeout.

        Raises MulticaCheckpointError on 403/404/409 so AReaL can distinguish
        non-resumable checkpoints from transient errors.
        """
        payload: dict = {
            "project_id": project_id,
            "event_ref": event_ref,
            "checkpoint_kind": checkpoint_kind,
            "env_id_map": env_id_map,
            "sandbox_refs": sandbox_refs,
        }
        if entropy_score is not None:
            payload["entropy_score"] = entropy_score
        if save_timeout_ms is not None:
            payload["save_timeout_ms"] = save_timeout_ms
        resp = await self._client.post(
            "/api/v1/env-checkpoints", json=payload, headers=self._headers()
        )
        if resp.status_code == 201:
            return resp.json()
        if resp.status_code in (403, 404, 409):
            raise MulticaCheckpointError(resp.status_code, resp.text[:200])
        raise RuntimeError(
            f"create_checkpoint failed: status={resp.status_code} body={resp.text[:200]}"
        )

    async def list_checkpoints(self, *, project_id: str) -> list[dict]:
        """GET /api/v1/projects/{project_id}/env-checkpoints."""
        resp = await self._client.get(
            f"/api/v1/projects/{project_id}/env-checkpoints",
            headers=self._headers(),
        )
        if resp.status_code == 200:
            return resp.json().get("checkpoints", [])
        if resp.status_code in (403, 404, 409):
            raise MulticaCheckpointError(resp.status_code, resp.text[:200])
        raise RuntimeError(
            f"list_checkpoints failed: status={resp.status_code} body={resp.text[:200]}"
        )

    async def resume_from_checkpoint(self, *, checkpoint_id: str) -> dict:
        """POST /api/v1/env-checkpoints/{checkpoint_id}/resume."""
        resp = await self._client.post(
            f"/api/v1/env-checkpoints/{checkpoint_id}/resume",
            headers=self._headers(),
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (403, 404, 409):
            raise MulticaCheckpointError(resp.status_code, resp.text[:200])
        raise RuntimeError(
            f"resume_from_checkpoint failed: status={resp.status_code} body={resp.text[:200]}"
        )

    async def maybe_create_entropy_checkpoint(
        self,
        *,
        project_id: str,
        event_ref: str,
        env_id_map: dict[str, str],
        sandbox_refs: list[dict],
        entropy: float | None,
        threshold: float | None,
        save_timeout_ms: int | None = None,
    ) -> dict | None:
        """Create an entropy-gated checkpoint when entropy >= threshold.

        The AReaL rollout loop calls this at decision points with the model's
        entropy (from logprobs) and a configured threshold. Returns the
        checkpoint dict on creation, None when skipped (missing logprobs or
        below threshold). A skip never fails the rollout. Checkpoint API
        errors propagate as MulticaCheckpointError so the caller can decide
        whether to retry or continue.
        """
        if not should_create_entropy_checkpoint(entropy, threshold):
            return None
        return await self.create_checkpoint(
            project_id=project_id,
            event_ref=event_ref,
            checkpoint_kind="entropy_gated",
            env_id_map=env_id_map,
            sandbox_refs=sandbox_refs,
            entropy_score=entropy,
            save_timeout_ms=save_timeout_ms,
        )

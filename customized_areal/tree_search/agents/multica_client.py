"""HTTP client for the multica unified env-dispatch API.

Wraps ``POST /api/v1/env-dispatch`` (unified dispatch - fresh env, branch, or
resume-from-checkpoint), ``DELETE /api/v1/env-dispatch/{projectID}`` (cascade
cleanup), and the env-checkpoint channel (``POST /api/v1/env-checkpoints``,
``GET /api/v1/projects/{projectID}/env-checkpoints``) per spec §6. Env boot/teardown
and checkpoint-resume are not separate endpoints: a fresh env is created by the
dispatch itself, its teardown is the cascade cleanup, and resume-from-checkpoint is
expressed as ``create_env_dispatch(mode="resume", env_id=<checkpoint_id>)``. Uses
stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
import os

import httpx

from customized_areal.tree_search.agents.multica_auth import (
    login_guidance,
    resolve_api_key,
)
from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue
from customized_areal.tree_search.env_checkpoint import (
    should_create_entropy_checkpoint,
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
        self._api_key = resolve_api_key(self._base_url, api_key)
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _failure_message(self, operation: str, response: httpx.Response) -> str:
        if response.status_code == 401:
            return f"{operation} failed: status=401. {login_guidance(self._base_url)}"
        return (
            f"{operation} failed: status={response.status_code} "
            f"body={self._safe_response_body(response)}"
        )

    async def _request(
        self, operation: str, method: str, path: str, **kwargs
    ) -> httpx.Response:
        try:
            return await self._client.request(method, path, **kwargs)
        except httpx.RequestError:
            pass
        raise RuntimeError(f"{operation} failed: network request failed")

    def _safe_response_body(self, response: httpx.Response) -> str:
        return response.text.replace(self._api_key, "[REDACTED]")[:4096]

    async def aclose(self) -> None:
        await self._client.aclose()

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
    ) -> str:
        """POST /api/v1/env-dispatch — unified dispatch (spec §6.3).

        ``mode`` selects the dispatch kind: ``scratch`` (fresh env, booted by the
        dispatch itself), ``branch`` (fork ``env_id`` = source env), or ``resume``
        (resume from a checkpoint, with ``env_id`` = the checkpoint id). There is
        no separate env-boot or checkpoint-resume endpoint.
        """
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

        resp = await self._request(
            "create_env_dispatch",
            "POST",
            "/api/v1/env-dispatch",
            json=payload,
            headers=self._headers(),
        )
        if resp.status_code != 201:
            raise RuntimeError(self._failure_message("create_env_dispatch", resp))
        return resp.json()["project_id"]

    async def cleanup_env_dispatch(self, *, project_id: str) -> None:
        """DELETE /api/v1/env-dispatch/{projectID} — cascades to issues/chat/tasks."""
        resp = await self._request(
            "cleanup_env_dispatch",
            "DELETE",
            f"/api/v1/env-dispatch/{project_id}",
            headers=self._headers(),
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(self._failure_message("cleanup_env_dispatch", resp))

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
        resp = await self._request(
            "create_checkpoint",
            "POST",
            "/api/v1/env-checkpoints",
            json=payload,
            headers=self._headers(),
        )
        if resp.status_code == 201:
            return resp.json()
        if resp.status_code in (403, 404, 409):
            raise MulticaCheckpointError(
                resp.status_code, self._safe_response_body(resp)
            )
        raise RuntimeError(self._failure_message("create_checkpoint", resp))

    async def list_checkpoints(self, *, project_id: str) -> list[dict]:
        """GET /api/v1/projects/{project_id}/env-checkpoints."""
        resp = await self._request(
            "list_checkpoints",
            "GET",
            f"/api/v1/projects/{project_id}/env-checkpoints",
            headers=self._headers(),
        )
        if resp.status_code == 200:
            return resp.json().get("checkpoints", [])
        if resp.status_code in (403, 404, 409):
            raise MulticaCheckpointError(
                resp.status_code, self._safe_response_body(resp)
            )
        raise RuntimeError(self._failure_message("list_checkpoints", resp))

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

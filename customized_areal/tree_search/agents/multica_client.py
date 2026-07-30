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

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load customized_areal/.env once at import so MULTICA_BASE_URL /
    # MULTICA_WORKSPACE_ID / MULTICA_AGENT_ID serve as defaults for the client,
    # create_env_dispatch, and the debug CLI. override=False keeps real env
    # vars and test monkeypatches authoritative; the file only fills in values
    # that are otherwise unset. Mirrors the pattern in
    # customized_grouped_workflow.py so external callers inherit the same
    # defaults without each loading .env themselves.
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

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


@dataclass(frozen=True, slots=True)
class EnvDispatchHandle:
    """Immutable handle for one env-dispatch result.

    Message dispatch is channel-first: ``channel_id`` is the primary public
    identifier and the project is the internal owner of env/DAG/checkpoint
    state. Issue dispatch is project-first (``channel_id`` is None). Callers
    pass this handle to DAG polling, checkpoint listing, and cleanup so the
    client routes by ``dispatch_type`` rather than inferring message mode from
    a missing/present arbitrary string. Checkpoint *creation* stays
    project-internal (``project_id``).
    """

    channel_id: str | None
    project_id: str
    env_id: str
    dispatch_type: str

    @property
    def primary_id(self) -> str:
        """The public identifier AReaL polls/cleans: channel_id for message,
        project_id for issue."""
        return self.channel_id if self.dispatch_type == "message" else self.project_id


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
        workspace_slug: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError(
                "MulticaEnvDispatchClient requires base_url or MULTICA_BASE_URL"
            )
        self._api_key = resolve_api_key(self._base_url, api_key)
        # Workspace identification for user-PAT auth. A workspace-bound task
        # token carries its workspace in X-Workspace-ID (set server-side by the
        # auth middleware), so these stay None in production. A user PAT must
        # identify the workspace explicitly via ?workspace_slug / ?workspace_id
        # (resolveWorkspaceUUID priority 2/3 in middleware/workspace.go). For
        # local dev/debug the workspace may also come from MULTICA_WORKSPACE_SLUG
        # / MULTICA_WORKSPACE_ID in customized_areal/.env; explicit args win.
        workspace_slug = (
            workspace_slug or os.environ.get("MULTICA_WORKSPACE_SLUG") or None
        )
        workspace_id = workspace_id or os.environ.get("MULTICA_WORKSPACE_ID") or None
        if workspace_slug and workspace_id:
            raise ValueError("pass at most one of workspace_slug or workspace_id")
        self._workspace_slug = workspace_slug
        self._workspace_id = workspace_id
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _params(self) -> dict[str, str]:
        """Workspace query params for user-PAT auth (empty for task-token auth)."""
        if self._workspace_slug:
            return {"workspace_slug": self._workspace_slug}
        if self._workspace_id:
            return {"workspace_id": self._workspace_id}
        return {}

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
        workspace_params = self._params()
        if workspace_params:
            merged = dict(kwargs.get("params") or {})
            merged.update(workspace_params)
            kwargs["params"] = merged
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
        train_agent_id: str | None = None,
        issue: SweLegoIssue | None = None,
        message: str | None = None,
        per_agent_env: dict[str, dict] | None = None,
        training_mode: bool = False,
    ) -> EnvDispatchHandle:
        """POST /api/v1/env-dispatch — unified dispatch (spec §6.3).

        ``mode`` selects the dispatch kind: ``scratch`` (fresh env, booted by the
        dispatch itself), ``branch`` (fork ``env_id`` = source env), or ``resume``
        (resume from a checkpoint, with ``env_id`` = the checkpoint id). There is
        no separate env-boot or checkpoint-resume endpoint.

        ``train_agent_id`` selects the single training target (spec §4.1): the
        named agent is trainable, all other agents in the dispatch are not. Empty
        means no training session. For a single-agent dispatch it must equal
        ``agent_id``; for a squad dispatch (``squad_id`` set) it must be a squad
        member. The server enforces these rules in ``validate()``.

        ``agent_id`` defaults to ``MULTICA_AGENT_ID`` from customized_areal/.env
        when omitted on a single-agent dispatch. Squad dispatches (``squad_id``
        set) intentionally omit ``agent_id`` - the squad supplies its members -
        so the env default does not apply there (spec §4.1).
        """
        # Resolve the single-agent target from MULTICA_AGENT_ID (.env) when the
        # caller omits it. Squad dispatches leave agent_id unset: the env default
        # must not turn a squad dispatch into a single-agent one.
        if not agent_id and not squad_id:
            agent_id = os.environ.get("MULTICA_AGENT_ID") or None
        # A training dispatch needs a training target: an empty train_agent_id
        # means "no training session", which would leave the RL workflow with no
        # trajectory to collect. For the single-agent case the server requires
        # train_agent_id == agent_id, so it is the only valid value.
        if training_mode and not train_agent_id and not squad_id:
            train_agent_id = agent_id
        payload: dict = {
            "mode": mode,
            "dispatch_type": dispatch_type,
            "group_size": group_size,
            "training_mode": training_mode,
        }
        if env_id:
            payload["env_id"] = env_id
        if agent_id:
            payload["agent_id"] = agent_id
        if squad_id:
            payload["squad_id"] = squad_id
        if domain is not None:
            payload["domain"] = domain
        if train_agent_id:
            payload["train_agent_id"] = train_agent_id
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
        data = resp.json()
        rollouts = data.get("rollouts") or []
        rollout_errors = [
            str(rollout.get("error"))
            for rollout in rollouts
            if isinstance(rollout, dict) and rollout.get("error")
        ]
        if rollout_errors:
            # Partial-success dispatch (group_size > 1): the server returned 201
            # with a mix of succeeded and failed rollouts, so the successful
            # lanes already allocated a project/channel/sandbox/agent_run
            # server-side (best-effort dispatch, no rollback). Fail closed by
            # reclaiming the partial dispatch before raising so the caller never
            # observes a half-provisioned dispatch (AC-7 "while retaining
            # cleanup"). Cleanup is best-effort: a cleanup failure must not mask
            # the original rollout failure that has to propagate.
            partial_handle = self._partial_cleanup_handle(data, dispatch_type)
            if partial_handle is not None:
                try:
                    await self.cleanup_env_dispatch(handle=partial_handle)
                except Exception:
                    pass
            raise RuntimeError(
                f"env-dispatch rollout failed: {rollout_errors[0][:1024]}"
            )
        project_id = data.get("project_id") or ""
        if not project_id:
            raise RuntimeError(
                "create_env_dispatch failed: response missing project_id"
            )
        channel_id = data.get("channel_id") or None
        if dispatch_type == "message" and not channel_id:
            raise RuntimeError(
                "create_env_dispatch failed: "
                "message dispatch response missing channel_id"
            )
        env_id = ""
        if rollouts:
            env_id = rollouts[0].get("env_id") or ""
        return EnvDispatchHandle(
            channel_id=channel_id,
            project_id=project_id,
            env_id=env_id,
            dispatch_type=dispatch_type,
        )

    def _dispatch_lifecycle_prefix(self, handle: EnvDispatchHandle) -> str:
        """Resolve the dispatch-scoped REST prefix for a handle.

        Message dispatches are channel-first: their lifecycle routes through
        ``/api/v1/env-dispatch/channels/{channelID}``. Issue dispatches stay
        project-first: ``/api/v1/env-dispatch/{projectID}``. Routing is driven
        only by ``handle.dispatch_type`` - never by inferring message mode from
        the presence/absence of an arbitrary string.
        """
        if handle.dispatch_type == "message":
            if not handle.channel_id:
                raise RuntimeError(
                    "env-dispatch handle missing channel_id for message dispatch"
                )
            return f"/api/v1/env-dispatch/channels/{handle.channel_id}"
        if not handle.project_id:
            raise RuntimeError(
                "env-dispatch handle missing project_id for issue dispatch"
            )
        return f"/api/v1/env-dispatch/{handle.project_id}"

    def _partial_cleanup_handle(
        self, data: dict[str, object], dispatch_type: str
    ) -> EnvDispatchHandle | None:
        """Build a cleanup handle from a partial-success dispatch response.

        When ``create_env_dispatch`` gets a 201 with per-rollout errors, the
        successful lanes have already been provisioned server-side. This returns
        a handle addressing that partial dispatch so it can be reclaimed before
        the client raises. Returns ``None`` when the response lacks the
        identifiers needed to address the dispatch, so cleanup is skipped rather
        than raising a confusing secondary error.
        """
        project_id = data.get("project_id") or ""
        if not project_id:
            return None
        channel_id = data.get("channel_id") or None
        if dispatch_type == "message" and not channel_id:
            return None
        return EnvDispatchHandle(
            channel_id=channel_id,
            project_id=project_id,
            env_id="",
            dispatch_type=dispatch_type,
        )

    async def cleanup_env_dispatch(self, *, handle: EnvDispatchHandle) -> None:
        """DELETE the dispatch-scoped resource - cascades to issues/chat/tasks.

        Message dispatch deletes ``/api/v1/env-dispatch/channels/{channelID}``;
        issue dispatch deletes ``/api/v1/env-dispatch/{projectID}``. A 404 is
        treated as success so cleanup is idempotent across retries.
        """
        resp = await self._request(
            "cleanup_env_dispatch",
            "DELETE",
            self._dispatch_lifecycle_prefix(handle),
            headers=self._headers(),
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(self._failure_message("cleanup_env_dispatch", resp))

    # Back-compat alias for callers that still resolve an issue dispatch to its
    # project_id (the legacy swe_lego path). New code should pass the handle
    # returned by ``create_env_dispatch`` directly to ``cleanup_env_dispatch``.
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None:
        await self.cleanup_env_dispatch(
            handle=EnvDispatchHandle(
                channel_id=None,
                project_id=project_id,
                env_id="",
                dispatch_type="issue",
            )
        )

    async def get_dag(self, *, handle: EnvDispatchHandle) -> httpx.Response:
        """GET the dispatch-scoped assembled DAG (spec §6.3).

        Message dispatch queries ``/api/v1/env-dispatch/channels/{channelID}/dag``;
        issue dispatch queries ``/api/v1/env-dispatch/{projectID}/dag``. Returns
        the raw response so the debug main can print transient states (202
        not-ready) and the assembled payload alike. Production DAG polling uses
        :class:`MulticaDagClient`; this exposes the same endpoint on the async
        client for debugging.
        """
        return await self._request(
            "get_dag",
            "GET",
            f"{self._dispatch_lifecycle_prefix(handle)}/dag",
            headers=self._headers(),
        )

    async def diagnose_env_dispatch(self, *, handle: EnvDispatchHandle) -> dict:
        """Run the opt-in diagnosis agent for an already terminal dispatch."""
        resp = await self._request(
            "diagnose_env_dispatch",
            "POST",
            f"{self._dispatch_lifecycle_prefix(handle)}/diagnosis",
            headers=self._headers(),
        )
        if resp.status_code != 200:
            raise RuntimeError(self._failure_message("diagnose_env_dispatch", resp))
        try:
            report = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                "diagnose_env_dispatch failed: invalid JSON report"
            ) from exc
        if not isinstance(report, dict) or report.get("status") != "completed":
            raise RuntimeError(
                "diagnose_env_dispatch failed: diagnosis did not complete"
            )
        return report

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

    async def list_checkpoints(self, *, handle: EnvDispatchHandle) -> list[dict]:
        """GET the dispatch-scoped env-checkpoints.

        Message dispatch queries ``/api/v1/channels/{channelID}/env-checkpoints``;
        issue dispatch queries ``/api/v1/projects/{projectID}/env-checkpoints``.
        """
        if handle.dispatch_type == "message":
            if not handle.channel_id:
                raise RuntimeError(
                    "env-dispatch handle missing channel_id for message dispatch"
                )
            path = f"/api/v1/channels/{handle.channel_id}/env-checkpoints"
        else:
            if not handle.project_id:
                raise RuntimeError(
                    "env-dispatch handle missing project_id for issue dispatch"
                )
            path = f"/api/v1/projects/{handle.project_id}/env-checkpoints"
        resp = await self._request(
            "list_checkpoints",
            "GET",
            path,
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


async def _poll_dag(
    client: MulticaEnvDispatchClient,
    handle: EnvDispatchHandle,
    timeout: float,
    interval: float,
) -> dict:
    """Periodically GET the DAG endpoint and print each response.

    Stops on a 200 (assembled) or any terminal status (404/403/401/...); keeps
    polling on 202 / 502 / 503 / 504 (not-ready / gateway) up to ``timeout``.
    """
    print(f"polling dag for handle={handle} (timeout={timeout}s, interval={interval}s)")
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get_dag(handle=handle)
        print(f"dag -> status={resp.status_code} body={resp.text[:2048]}")
        if resp.status_code == 200:
            from customized_areal.tree_search.agents.multica_dag_client import (
                AssembledDag,
                DagError,
            )

            try:
                AssembledDag.from_dict(resp.json())
            except (DagError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid assembled DAG: {exc}") from exc
            print("dag assembled")
            return resp.json()
        if resp.status_code not in (202, 502, 503, 504):
            raise RuntimeError(
                f"DAG polling failed: status={resp.status_code} "
                f"body={client._safe_response_body(resp)}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"DAG readiness timeout after {timeout}s")
        await asyncio.sleep(interval)


def _debug_external_runtime_policy(
    agent_id: str, train_agent_id: str | None
) -> dict[str, dict] | None:
    """Build a secret-bearing debug runtime policy from local environment only."""
    names = (
        "MULTICA_EXTERNAL_PROVIDER",
        "MULTICA_EXTERNAL_BASE_URL",
        "MULTICA_EXTERNAL_API_KEY",
        "MULTICA_EXTERNAL_MODEL",
    )
    provider, base_url, api_key, model = tuple(
        (os.environ.get(name) or "").strip() for name in names
    )
    configured = (provider, base_url, api_key, model)
    if not any(configured):
        return None
    if not all(configured):
        raise RuntimeError(
            "MULTICA_EXTERNAL_PROVIDER, MULTICA_EXTERNAL_BASE_URL, "
            "MULTICA_EXTERNAL_API_KEY, and MULTICA_EXTERNAL_MODEL must be set together"
        )
    if train_agent_id:
        raise RuntimeError(
            "caller-provided external runtime is supported only for non-training "
            "debug dispatch"
        )
    return {
        agent_id: {
            "runtime": {
                "provider": provider,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }
        }
    }


async def _debug_run(args: argparse.Namespace) -> int:
    """Exercise create -> poll-dag -> list-checkpoints -> cleanup against a live MultiCA."""
    message = args.message
    dispatch_type = "message" if message is not None else args.dispatch_type
    if dispatch_type == "message" and not message:
        message = (
            f"Debug probe {os.urandom(4).hex()}: list the files in the "
            "current working directory and report the operating system."
        )
    domain = args.domain or "self_play"
    issue: SweLegoIssue | None = None
    if dispatch_type == "issue" and message is None:
        issue = SweLegoIssue(
            repo_url="https://github.com/example/debug-repo",
            base_commit="main",
            issue_date="2026-01-01",
            issue_text="Debug dispatch fired from multica_client __main__.",
            issue_title="Debug issue",
            acceptance_criteria="Client can reach the MultiCA env-dispatch API.",
            fail_to_pass=["test_debug"],
            pass_to_pass=[],
        )

    per_agent_env = _debug_external_runtime_policy(args.agent_id, args.train_agent_id)
    client = MulticaEnvDispatchClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        workspace_slug=args.workspace_slug,
        workspace_id=args.workspace_id,
    )
    handle: EnvDispatchHandle | None = None
    try:
        print(
            f"dispatch -> base_url={args.base_url} mode={args.mode} "
            f"dispatch_type={dispatch_type} agent_id={args.agent_id} "
            f"train_agent_id={args.train_agent_id or '(none)'} "
            f"workspace={args.workspace_slug or args.workspace_id or '(none)'} "
            f"external_runtime={'configured' if per_agent_env else '(none)'}"
        )
        handle = await client.create_env_dispatch(
            mode=args.mode,
            env_id=args.env_id,
            dispatch_type=dispatch_type,
            agent_id=args.agent_id,
            domain=domain,
            train_agent_id=args.train_agent_id,
            issue=issue,
            message=message,
            per_agent_env=per_agent_env,
            training_mode=args.training_mode,
        )
        print(f"created handle: {handle}")

        final_dag = await _poll_dag(
            client, handle, args.dag_timeout, args.dag_poll_interval
        )
        if args.diagnose:
            report = await client.diagnose_env_dispatch(handle=handle)
            print(f"diagnosis completed: {report}")
            final_dag = await _poll_dag(
                client, handle, args.dag_timeout, args.dag_poll_interval
            )
            from customized_areal.tree_search.agents.multica_dag_client import (
                AssembledDag,
            )

            AssembledDag.from_dict(final_dag).validate_diagnosis_coverage()
        if args.dag_out:
            Path(args.dag_out).write_text(
                json.dumps(final_dag, indent=2), encoding="utf-8"
            )
            print(f"wrote dag: {args.dag_out}")

        # list_checkpoints is diagnostic only. An env-checkpoints-disabled server
        # answers 404 here; that must not skip cleanup of the real dispatch.
        try:
            checkpoints = await client.list_checkpoints(handle=handle)
            print(f"checkpoints({len(checkpoints)}): {checkpoints}")
        except Exception as exc:  # noqa: BLE001 - keep cleanup on the rails
            print(f"list_checkpoints failed (non-fatal): {type(exc).__name__}: {exc}")

        return 0
    finally:
        # Always clean up the created env-dispatch, even if polling or
        # list_checkpoints raised, so a mid-run error never leaks a real cloud
        # resource. --keep intentionally leaves it for inspection.
        if handle is not None:
            if args.keep:
                print("--keep set; skipping cleanup")
            else:
                try:
                    await client.cleanup_env_dispatch(handle=handle)
                    print(f"cleaned up handle: {handle}")
                except Exception as exc:  # noqa: BLE001 - diagnostic
                    print(f"cleanup_env_dispatch failed: {type(exc).__name__}: {exc}")
        await client.aclose()


def build_debug_parser() -> argparse.ArgumentParser:
    """Build the debug CLI without deployment-specific repository defaults."""
    parser = argparse.ArgumentParser(
        description="Debug the MultiCA env-dispatch client against a live server"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MULTICA_BASE_URL"),
        help="MultiCA API base URL (defaults to MULTICA_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MULTICA_API_KEY"),
        help="MultiCA API key (defaults to MULTICA_API_KEY or saved credentials)",
    )
    parser.add_argument(
        "--workspace-slug",
        default=os.environ.get("MULTICA_WORKSPACE_SLUG"),
        help="workspace slug (defaults to MULTICA_WORKSPACE_SLUG); alternative to --workspace-id",
    )
    parser.add_argument(
        "--workspace-id",
        default=os.environ.get("MULTICA_WORKSPACE_ID"),
        help="workspace UUID (defaults to MULTICA_WORKSPACE_ID); alternative to --workspace-slug",
    )
    parser.add_argument(
        "--training-mode",
        action="store_true",
        default=False,
        help="training_mode payload field (default: false; set to enable training)",
    )
    parser.add_argument(
        "--mode",
        default="scratch",
        choices=["scratch", "branch", "resume"],
        help="env-dispatch mode (default: scratch)",
    )
    parser.add_argument(
        "--dispatch-type",
        default="message",
        help="dispatch_type payload field (default: message; use issue for swe_lego)",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("MULTICA_AGENT_ID"),
        help="agent_id payload field (defaults to MULTICA_AGENT_ID)",
    )
    parser.add_argument(
        "--train-agent-id",
        default=None,
        help="train_agent_id: the training target (default: none; must equal agent_id "
        "for single-agent training, or be a squad member when --squad-id is set)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="domain (swe_lego|self_play); default: self_play",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="source env_id (required for branch/resume modes)",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="message content; defaults to a random debug query for message dispatch",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="skip cleanup so the created project can be inspected",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--dag-timeout",
        type=float,
        default=60.0,
        help="max seconds to poll the DAG endpoint after create (default: 60)",
    )
    parser.add_argument(
        "--dag-poll-interval",
        type=float,
        default=3.0,
        help="seconds between DAG polls (default: 3)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="run opt-in MultiCA diagnosis after the dispatch becomes terminal",
    )
    parser.add_argument(
        "--dag-out",
        default=None,
        help="optional path for the final assembled DAG JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the debug flow after validating explicit deployment coordinates."""
    parser = build_debug_parser()
    args = parser.parse_args(argv)
    if not args.base_url:
        parser.error("--base-url or MULTICA_BASE_URL is required")
    if not args.agent_id:
        parser.error("--agent-id or MULTICA_AGENT_ID is required")
    workspace_selectors = bool(args.workspace_slug) + bool(args.workspace_id)
    if workspace_selectors != 1:
        parser.error(
            "exactly one of --workspace-slug/MULTICA_WORKSPACE_SLUG or "
            "--workspace-id/MULTICA_WORKSPACE_ID is required"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return asyncio.run(_debug_run(args))
    except (RuntimeError, MulticaCheckpointError) as exc:
        print(f"MultiCA debug run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

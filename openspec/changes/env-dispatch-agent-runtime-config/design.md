## Context

Frontend sandbox creation produces an online Pi runtime. The current env-dispatch
implementation pre-creates an offline runtime row, creates a sandbox through a separate
lifecycle adapter, and queues work before provider readiness. Live diagnostics showed
the sandbox reaching `running`, the expected daemon adopting the runtime row, Pi
remaining `offline`, the task staying queued, and the DAG staying `in_progress`.

The source agent must also remain reusable: concurrently addressing the same source
agent in multiple dispatches cannot repeatedly overwrite its global runtime binding.

## Goals / Non-Goals

**Goals:**

- Make frontend and env-dispatch use one sandbox creation implementation.
- Discover an online daemon-registered runtime without pre-creating its DB row.
- Create one derived global agent per source-agent sandbox and record lineage.
- Keep model credentials and training sessions one-to-one with the source binding.
- Preserve `session_id -> agent_run_id` by linking the real task after enqueue.
- Fail and compensate terminal provisioning errors instead of leaving an endless DAG.
- Clean up every resource owned by the derived dispatch.

**Non-Goals:**

- Rebinding the source agent's global runtime.
- Matching runtimes by mutable display name.
- Exposing provider, bootstrap, or session credentials in public APIs.
- Adding a general-purpose agent cloning API for unrelated product flows.

## Decisions

### Share the frontend sandbox service

The HTTP handler and env-dispatch call one service for node selection, instance insert,
metadata persistence, daemon PAT minting, runtime environment construction, job payload,
and notification. This removes the behavioral drift observed in production.

### Discover runtime after registration

Sandbox creation mints a unique daemon ID but does not insert `agent_runtime`.
Provisioning waits for a Pi runtime with matching workspace, daemon ID, and trusted
`sandbox_instance_id` metadata to report `online`. Runtime names are display-only.

### Clone a derived global agent

When runtime readiness succeeds, a transaction copies approved executable fields and
skills from the source agent, binds the new agent to the runtime, records
`source_agent_id`, replaces the source member for this dispatch channel, and records the
derived ID on the binding. The source agent remains unchanged.

### Resolve credentials before sandbox creation

Static agents consume the validated runtime object stored for their source binding.
Training agents call `start_session(session_ref, env_id)` with the persistent source
binding ID and build a server-owned runtime policy with model `areal-default`, the
configured bridge URL, and the returned session key. The bridge retains legacy `task_id`
compatibility. Once the real derived-agent task is inserted, env-dispatch links that
task to the session for DAG assembly. Retries reuse the recorded session.

Sandbox bootstrap PAT and model API key are distinct typed credentials. A binding's
`model_config_owner_agent_id` must equal its `source_agent_id`.

### Make readiness and cleanup transactional at the workflow level

The binding state machine single-flights credential resolution, sandbox creation,
runtime discovery, derived-agent creation, and task enqueue. Every failure step has
explicit compensation. Dispatch cleanup archives derived agents and deletes their
sandbox, runtime, credentials, and training session.

## Risks / Trade-offs

- Creating global derived agents increases agent-row volume. Cleanup ownership and
  archive indexing keep the lifecycle bounded.
- Waiting for runtime readiness increases dispatch latency. A bounded timeout gives
  truthful failure instead of indefinite DAG polling.
- Training credentials exist before task insertion. Binding persistence and a stable
  session reference provide idempotency; compensation closes orphaned sessions.
- Schema changes require coordinated deployment. New provisioning is feature-gated and
  old pending bindings are drained or rejected.

## Migration Plan

1. Add lineage and binding-state schema with backward-compatible nullable columns.
1. Deploy the shared sandbox service and runtime discovery seam behind a feature flag.
1. Deploy derived-agent cloning, credential resolution, orchestration, and cleanup.
1. Enable for test workspaces; verify static and training dispatches end-to-end.
1. Drain legacy pending bindings, then remove the pre-created-runtime path.

## Open Questions

None. Deployment supplies the AReaL bridge URL through existing training configuration;
production code does not hardcode a deployment endpoint.

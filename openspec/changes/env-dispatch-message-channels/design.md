## Context

`dispatch_type=issue` is already project-first and rollout-isolated. Message
dispatch is not: it reuses the agent default runtime, wakes the whole squad, and
has no isolated channel or resumable trigger. The authoritative technical design
is the superpowers Design Doc; this document records the key decisions for the
Comet change record.

**Authoritative design:**
`docs/superpowers/specs/2026-07-16-env-dispatch-message-channel-design.md`
**Implementation plan:**
`docs/superpowers/plans/2026-07-16-env-dispatch-message-channels.md`

## Goals / Non-Goals

**Goals:**
- Rollout-isolated project-backed group channels for message dispatch.
- Binding-routed sandbox execution (never the agent default runtime).
- Leader-only scratch wake; lazy first-mention peer provisioning.
- Branch resume from a persisted collaboration trigger with sandbox cloning.

**Non-Goals:**
- Changing ordinary channels to require sandbox execution.
- Removing project-based APIs or migrating issue dispatch away from projects.
- Automatically waking every squad member for the initial message.
- Provisioning an additional group-manager agent for EnvDispatch channels.

## Decisions

- **Project stays the internal owner; channel is the public handle.** `project.env_id`
  binds project to env; `channel.project_id` binds the visible channel to the
  project. The message API returns both `channel_id` (primary) and `project_id`.
  Rationale: reuses existing DAG/checkpoint/training ownership without a new
  association table.
- **`environment_agent_sandbox` is the routing authority.** `(env_id, agent_id)`
  primary key; channel delivery consults it before the ordinary
  default-runtime path. Rationale: single source of truth that prevents
  default-runtime fallback without scattering conditionals.
- **Single-flight claim via `UPDATE ... WHERE status IN ('pending','failed')`.**
  One winner; others observe the result. Rationale: DB-row lock is the
  serialization point, no external lock service.
- **Branch copies the channel graph in one transaction; trigger agent sandbox is
  cloned, not recreated from template.** A `clone` sandbox job snapshots the
  source filesystem into a new daemon/runtime. Rationale: preserves agent state
  (memory/work) across branches; recreating only the template loses it.
- **Trigger carries channel context; source task/runtime are never reused.**
  Branch remaps `channel_id`/`project_id`/`source_message_id`/`thread_root` to
  copied IDs and clears `task_id`/`runtime_id`. Rationale: provenance without
  re-executing the source task.

## Risks / Trade-offs

- [Clone job can fail after destination runtime precreate] -> CloneSandboxInstance
  compensates by force-deleting the destination instance + runtime; the binding
  returns to retryable `failed`.
- [Concurrent cleanup vs provisioning] -> cleanup marks bindings `deleting`;
  a provisioner observing `deleting` stops before creating external resources;
  cleanup observing `provisioning` waits for terminal state or compensates from
  recorded resource IDs.
- [Copied chat session vs provisioned channel session duplication] -> the branch
  trigger uses the freshly provisioned channel-agent session (new runtime), not
  the copied project chat session; the copied session is historical context only.

## Migration Plan

- Migration adds `environment.collaboration_trigger` and the
  `environment_agent_sandbox` table, and adds `clone` to the accepted
  `sandbox_job` job types. Down migration reverses both and restores the prior
  check verbatim. Apply order: migration, then deploy; existing issue/project
  routes are unaffected. Rollback: drop the table/column and restore the check.

## Open Questions

- None outstanding; remaining work is implementation completion (Task 6 Step 5,
  Tasks 7-9) per the plan.

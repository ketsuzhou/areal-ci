# EnvDispatch Message Channels Design

**Date:** 2026-07-16

## Goal

When `dispatch_type=message`, give every rollout its own group channel while
retaining a project as the internal owner of DAG, checkpoint, training, and env
state. Add the request's user and every requested agent to the channel. Agent
execution in this path is always sandboxed and rollout-isolated, but non-triggered
agents are provisioned lazily on their first mention.

`dispatch_type=issue` remains project-first and retains its existing behavior.

## Core invariants

- Every message rollout has a distinct env, project, and group channel.
- `project.env_id` binds the project to the rollout env.
- The existing `channel.project_id` binds the visible channel to the project.
- The message API uses `channel_id` as its primary public identifier and also
  returns `project_id` for internal and compatibility use.
- The EnvDispatch caller is a channel member. A single-agent request adds that
  agent; a squad request adds the canonical leader and every agent member of the
  squad, with duplicates removed.
- EnvDispatch-created channels do not auto-provision a Beckham group manager.
- An agent task created for an EnvDispatch channel must use that rollout's
  sandbox runtime. It must never fall back to the agent's shared default runtime.
- The same agent in two rollouts never shares a sandbox, daemon, or runtime.
- Scratch initially wakes the leader. Branch wakes the agent selected by the
  source env's persisted collaboration trigger.
- Branch requires a valid trigger. A missing, malformed, or unauthorized trigger
  is a validation error before new resources are created.

## Data model

### Existing relationships

No new project-channel association table is needed. Message rollouts populate:

```text
environment <- project.env_id
project     <- channel.project_id
```

An EnvDispatch project must have exactly one associated EnvDispatch group channel.
Branch source resolution follows `env -> project -> channel` and rejects zero or
multiple candidate channels rather than guessing.

### Per-agent rollout execution state

Add an env-scoped agent binding table with these logical fields:

```text
environment_agent_sandbox
- env_id
- channel_id
- agent_id
- status: pending | provisioning | ready | failed | deleting
- sandbox_instance_id, nullable until ready
- runtime_id, nullable until ready
- daemon_id, nullable until ready
- source_sandbox_instance_id, nullable
- sandbox_config, the resolved template/base-env policy for lazy creation
- last_error, nullable
- created_at
- updated_at
```

The primary key is `(env_id, agent_id)`. Non-null sandbox instance and runtime
IDs are unique. The source sandbox reference allows a branch to defer cloning a
non-triggered member without losing its source state.

The table is authoritative for EnvDispatch channel routing. A normal channel has
no such binding and keeps the current non-sandbox routing behavior.

### Collaboration trigger

Persist the current collaboration continuation trigger on the env. The trigger is
compatible with the existing checkpoint resume-trigger concept but includes the
channel context needed to copy and resume collaboration:

```json
{
  "agent_id": "agent UUID",
  "kind": "channel_message|mention|handoff|continuation",
  "channel_id": "channel UUID",
  "project_id": "project UUID",
  "chat_session_id": "chat session UUID",
  "source_message_id": "channel message UUID",
  "thread_root_message_id": "optional channel message UUID",
  "task_id": "source task UUID",
  "runtime_id": "source runtime UUID"
}
```

Agent collaboration control changes update this trigger atomically with the
associated task/message transition. It describes which agent and channel context
must continue after a branch. The source task and runtime IDs provide provenance;
they are never executed unchanged in a new branch.

## Scratch flow

For each rollout, the reset phase:

1. Resolve the agent roster. For a squad, include `squad.leader_id` plus all
   `squad_member` rows whose member type is agent; validate workspace membership.
2. Fork or create a distinct env.
3. Create a project bound to the env.
4. Create a group channel bound to the project without the normal automatic group
   manager provisioning.
5. Add the caller and resolved agent roster as channel members.
6. Insert a pending env-agent binding for every agent, preserving each agent's
   resolved `per_agent_env` or default sandbox policy.
7. Provision only the leader's sandbox, daemon, and preassigned runtime; create
   the leader's channel-agent session bound to that project and runtime.
8. Insert `message.content` once on the channel and enqueue only the leader task.
9. Persist the initial collaboration trigger once the leader task exists.

Reset remains all-or-nothing across the request's rollout group. Failure before
dispatch completes removes newly created channels, projects, envs, runtimes, and
sandboxes in dependency order.

## Lazy first-mention provisioning

The existing channel mention/handoff path gains an EnvDispatch-specific routing
step:

1. Resolve the channel's project, env, and target agent binding.
2. Lock the `(env_id, agent_id)` row and atomically claim `pending` or retryable
   `failed` state as `provisioning`.
3. Scratch creates from the saved per-agent policy. Branch clones the saved source
   sandbox when one exists; otherwise it performs the agent's first creation from
   the saved policy.
4. Precreate the agent runtime and daemon identity, create the sandbox, and create
   the channel-agent session using the rollout runtime.
5. Mark the binding ready, enqueue the mention/handoff task, and update the env
   trigger to the new collaboration continuation.

Concurrent mentions cannot provision twice. Other callers wait for or observe the
single provisioning result. Provisioning failure records a sanitized error and
leaves the row retryable. No task is enqueued and routing never falls back to the
shared agent runtime. If sandbox creation succeeds but task enqueue fails, the new
runtime and sandbox are reclaimed and the binding becomes retryable.

## Branch and resume flow

Before creating resources, branch validates that the source env has a decodable
trigger whose target agent belongs to its EnvDispatch channel. It also validates
that the requested agent roster matches the source channel's agent roster.

The branch then:

1. Forks a new env and copies the project subtree.
2. Creates a new channel bound to the new project.
3. Copies all channel members and the full message history, including reply,
   quote, thread-root, and conversation/read relationships.
4. Copies per-agent bindings as pending metadata. Existing source sandboxes are
   recorded as lazy clone sources; agents that never ran retain their original
   sandbox policy.
5. Builds complete old-to-new maps for project, channel, chat session, message,
   and thread identifiers.
6. Remaps the trigger's context to the copied entities. The source task and runtime
   are not reused.
7. Immediately clones or provisions only the trigger-selected agent, creates its
   new session and task, and executes the remapped continuation trigger.
8. Saves the remapped trigger on the new env.

Historical messages are copied as data and never re-trigger agents. A branch
request's new `message.content`, when present, is appended as context but does not
override the trigger-selected agent.

Sandbox-instance branch handling must perform a real state clone/fork. Recreating
only the source template is insufficient. Fleet-backed sources continue to use the
existing fork operation; the sandbox-instance lifecycle gains an equivalent clone
operation that preserves filesystem state while assigning a new daemon/runtime.

## API

The message response includes both identifiers:

```json
{
  "channel_id": "first rollout channel UUID",
  "project_id": "first rollout project UUID",
  "rollouts": [
    {
      "channel_id": "channel UUID",
      "project_id": "project UUID",
      "env_id": "env UUID",
      "leader_run_id": "triggered run UUID",
      "agent_sandboxes": {
        "triggered agent UUID": {
          "status": "ready",
          "sandbox_instance_id": "sandbox UUID",
          "runtime_id": "runtime UUID"
        },
        "other agent UUID": {"status": "pending"}
      }
    }
  ]
}
```

For `group_size > 1`, the top-level IDs retain the existing convention of naming
the first rollout while every rollout contains its own IDs. Idempotency persists
and replays the complete response.

Add channel-first facades that resolve the bound project internally:

- `GET /api/v1/env-dispatch/channels/{channelID}/dag`
- `DELETE /api/v1/env-dispatch/channels/{channelID}`
- `GET /api/v1/channels/{channelID}/env-checkpoints`

Existing project-based DAG, checkpoint, and cleanup routes remain available for
issue dispatch and compatibility. The AReaL message client holds both IDs but uses
the channel routes for polling, cleanup, and checkpoint listing.

## Cleanup and concurrency

Channel cleanup resolves the bound project and env, marks the rollout deleting,
and prevents new lazy provisioning. It then removes the channel, project, env,
ready sandboxes/runtimes, and pending bindings in foreign-key-safe order. Repeated
cleanup is idempotent.

Cleanup and first-mention provisioning serialize on the env-agent binding. A
provisioner that observes deleting state must stop before creating external
resources. Cleanup that observes provisioning waits for its terminal state or
performs compensation from the recorded resource IDs.

## Error behavior

- Invalid source trigger, source channel shape, or agent roster: HTTP 400 before
  resource creation.
- Reset failure in any rollout: compensate every rollout created by the request
  and return the existing reset-failed response class.
- Trigger execution failure after branch provisioning: compensate the new target
  runtime/sandbox and fail the reset; never wake the source rollout.
- Lazy provisioning failure after a successful rollout: preserve the channel,
  record a retryable failed binding, and do not enqueue the target task.
- All external errors exposed in API responses are sanitized; detailed stacks stay
  in existing server-side diagnostics.

## Test strategy

Implementation follows test-driven development. Required coverage includes:

- distinct env/project/channel and sandbox/runtime IDs across rollouts;
- project-channel binding and exact caller/squad membership;
- no automatic group manager;
- scratch provisions and wakes only the leader;
- first mention provisions exactly once under concurrency;
- EnvDispatch tasks never use a shared default runtime;
- ordinary channel routing remains unchanged;
- branch rejects missing, malformed, or unauthorized triggers before writes;
- branch remaps all copied channel relationships and wakes only the trigger agent;
- branch lazily clones non-triggered agents from their own source sandbox;
- idempotency replay creates no duplicate resources;
- channel-first DAG, checkpoint, and cleanup routes resolve the bound project;
- cleanup is idempotent and safe against concurrent provisioning;
- issue dispatch behavior and project-first routes do not regress.

Database-backed handler tests may skip when PostgreSQL is unavailable, following
the repository's existing convention. Service tests use fakes to exercise reset,
rollback, trigger mapping, and provisioning concurrency without external sandbox
infrastructure.

## Out of scope

- Changing ordinary channels to require sandbox execution.
- Removing project-based APIs or migrating issue dispatch away from projects.
- Automatically waking every squad member for the initial message.
- Provisioning an additional group-manager agent for EnvDispatch channels.

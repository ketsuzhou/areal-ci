## Why

`dispatch_type=issue` already owns a rollout-isolated project/env/sandbox, but
`dispatch_type=message` still routes through the shared project-chat path: it
reuses the agent's default runtime, wakes the whole squad, and has no
rollout-isolated channel or persisted collaboration trigger to resume from. This
blocks safe multi-turn self-play rollouts (and MCTS-style branching) because two
rollouts of the same agent share a sandbox/daemon and a branch cannot resume a
specific agent's collaboration without reusing its source runtime/task.

## What Changes

- Every `dispatch_type=message` rollout gets a distinct env, project, and group
  channel; the caller and the exact requested agent roster are the only members.
- EnvDispatch channels never auto-provision the Beckham group manager.
- An env-scoped `environment_agent_sandbox` binding table is the authority for
  sandbox routing; channel delivery consults it before ordinary runtime
  selection, so an EnvDispatch task never falls back to the agent default runtime.
- Scratch wakes only the canonical leader; peer members are provisioned lazily on
  their first directed mention (single-flight, no double provisioning).
- A collaboration trigger is persisted on the env and resumed by a branch: the
  branch copies the channel graph (messages, replies, quotes, threads, read
  state, members, bindings), clones only the trigger-selected agent's sandbox
  state, and appends request content as nondispatching context.
- Channel-first facades (`/env-dispatch/channels/{id}/dag`,
  `DELETE /env-dispatch/channels/{id}`, `/channels/{id}/env-checkpoints`)
  resolve the bound project internally; the AReaL message client uses
  `channel_id` as its primary handle while issue dispatch stays project-first.

## Capabilities

### New Capabilities
- `env-dispatch-message-channels`: Rollout-isolated, project-backed group
  channels for message dispatch, with per-agent sandbox bindings, leader-only
  scratch wake, lazy first-mention peer provisioning, and branch resume from a
  persisted collaboration trigger.

### Modified Capabilities
<!-- None: existing issue/project-first behavior is preserved unchanged. -->

## Impact

- `multica/server`: new migration (env trigger + binding table + clone job type),
  env-dispatch channel service/store/provision/copy/routes, sandbox-instance
  clone lifecycle, channel routing hook, channel-first DAG/checkpoint/cleanup
  routes.
- `customized_areal/tree_search`: channel-first `EnvDispatchHandle` in the
  MultiCA client + DAG client + multi-agent workflow.
- Public API: message response adds `channel_id`/`leader_run_id`/`agent_sandboxes`;
  three channel-first routes are added; existing project-first routes and issue
  dispatch behavior are unchanged.
- Design spec: `docs/superpowers/specs/2026-07-16-env-dispatch-message-channel-design.md`.
- Implementation plan: `docs/superpowers/plans/2026-07-16-env-dispatch-message-channels.md`.

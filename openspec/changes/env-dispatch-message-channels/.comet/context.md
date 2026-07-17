# Comet Design Handoff

- Change: env-dispatch-message-channels
- Phase: design
- Mode: compact
- Context hash: 0ad64d410678427570258dfe6937f4e4dc246ec8f8de43500a8434dc7d830a02

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/env-dispatch-message-channels/proposal.md

- Source: openspec/changes/env-dispatch-message-channels/proposal.md
- Lines: 1-53
- SHA256: 1d34a7d9172b37966b7ff6829034a92bb8c539d4dfa31593980c6ebaf1593d7f

```md
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

```

## openspec/changes/env-dispatch-message-channels/design.md

- Source: openspec/changes/env-dispatch-message-channels/design.md
- Lines: 1-75
- SHA256: af3d13636d16b637c2bd1f5b418f442c55ec748329d48d88cb25c1728787de4b

```md
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

```

## openspec/changes/env-dispatch-message-channels/tasks.md

- Source: openspec/changes/env-dispatch-message-channels/tasks.md
- Lines: 1-52
- SHA256: 876fe5dcaf3235443935fd695451d48b8a101946607e464afc860a765723dec7

```md
## 1. Persist EnvDispatch channel execution state

- [x] 1.1 Migration: env collaboration_trigger + environment_agent_sandbox table + clone job type
- [x] 1.2 Focused pgx store: bindings, claim/markReady/markFailed/markDeleting, trigger load/save

## 2. Clone sandbox-instance state through sandboxd

- [x] 2.1 CloneSandboxInstance lifecycle (offline runtime + clone job, compensation)
- [x] 2.2 Cube clone as snapshot/create/delete; completion writes destination external id

## 3. Create project-backed message channels and exact rosters

- [x] 3.1 ResolveMessageRoster + CreateEnvDispatchChannel (group channel, exact members, pending bindings, no Beckham)
- [x] 3.2 resetOne wires message channel creation after project; issue reset byte-for-byte preserved

## 4. Provision and wake only the scratch leader

- [x] 4.1 provisionEnvDispatchAgent (claim, precreate runtime, sandbox, session, markReady, compensate)
- [x] 4.2 dispatchScratchChannelMessage: leader-only provision + channel enqueue + initial trigger + channel-first response

## 5. Lazily provision peers on first mention

- [x] 5.1 routeEnvDispatchChannelAgent hook before session creation (ready/pending/failed/provisioning/deleting)
- [x] 5.2 Persist continuation trigger atomically with enqueue; compensate on enqueue failure

## 6. Copy channel history and resume from the source trigger

- [x] 6.1 ValidateBranchMessageSource before reset fan-out (missing/malformed/unauthorized/roster => 400)
- [x] 6.2 CopyEnvDispatchChannel: members, messages, replies/quotes/threads, read state, thread participants, pending bindings w/ clone sources
- [ ] 6.3 Remap trigger to copied entities; provision trigger agent via CloneSandboxInstance (clone when source binding ready, else create from policy); enqueue channel run; save remapped trigger
- [ ] 6.4 Append request message.content as nondispatching context without changing the trigger agent
- [ ] 6.5 Branch Step 5 tests pass (wakes only trigger w/ cloned sandbox; leaves peers pending w/ clone sources; appends content w/o changing trigger agent)

## 7. Channel-first facades and concurrency-safe cleanup

- [ ] 7.1 Channel-to-project resolver + GetEnvDispatchChannelDag, DeleteEnvDispatchChannel, ListChannelEnvCheckpoints handlers
- [ ] 7.2 Register channel-first routes in cmd/server/router.go
- [ ] 7.3 Serialized cleanup: lock env+bindings, mark deleting, compensate in-flight provisioning, delete in FK-safe order, idempotent

## 8. AReaL message client channel-first

- [ ] 8.1 EnvDispatchHandle (channel_id primary for message, project_id for issue)
- [ ] 8.2 create_env_dispatch returns handle; DAG/checkpoint/cleanup route by dispatch_type
- [ ] 8.3 multi_agent_workflow + multica_dag_client pass handle; issue callers stay source-compatible

## 9. Regression verification and protocol documentation

- [ ] 9.1 Document final request/response, channel-first routes, leader-only wake, lazy provisioning, branch errors in multica_environment_protocol.md
- [ ] 9.2 Go suites pass (service/handler/migrations/cmd/multica/cmd/server)
- [ ] 9.3 Python suites pass (test_env_dispatch_client, test_multica_dag_client)
- [ ] 9.4 gofmt, go test ./..., graphify update, pre-commit pass
- [ ] 9.5 Final invariant review (no default-runtime fallback; only intended files staged)

```

## openspec/changes/env-dispatch-message-channels/specs/env-dispatch-message-channels/spec.md

- Source: openspec/changes/env-dispatch-message-channels/specs/env-dispatch-message-channels/spec.md
- Lines: 1-86
- SHA256: b33c53900e5148a2b804b88429206c715da070b8673f5d350a48a39f76532787

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Message rollout isolation

Every `dispatch_type=message` rollout SHALL own a distinct env, project, and
group channel. The EnvDispatch caller and the exact requested agent roster
(squad leader plus agent members, deduplicated) SHALL be the only channel
members. EnvDispatch channels SHALL NOT auto-provision a group manager.

#### Scenario: Two rollouts are fully isolated
- **WHEN** a message dispatch requests group_size > 1
- **THEN** every rollout has a distinct env_id, project_id, channel_id, sandbox, daemon, and runtime namespace

#### Scenario: No automatic group manager
- **WHEN** an EnvDispatch group channel is created
- **THEN** no Beckham group-manager agent session is provisioned for that channel

### Requirement: Binding-routed sandbox execution

An agent task created for an EnvDispatch channel SHALL use that rollout's binding
runtime and SHALL NOT fall back to the agent's shared default runtime. The same
agent in two rollouts SHALL NOT share a sandbox, daemon, or runtime.

#### Scenario: Task uses the binding runtime, never the default
- **WHEN** the leader task is enqueued for a message rollout
- **THEN** the task's runtime_id equals the leader binding's runtime_id and differs from the agent's default runtime_id

### Requirement: Scratch wakes only the leader

A scratch message dispatch SHALL provision and wake only the canonical roster
leader. All other roster members SHALL remain pending bindings, provisioned on
their first directed mention.

#### Scenario: Peers are not woken on scratch
- **WHEN** a squad scratch message dispatch runs
- **THEN** exactly one leader binding is ready, every peer binding is pending, and exactly one leader task is enqueued

### Requirement: Lazy first-mention provisioning

A directed mention of a pending EnvDispatch agent SHALL provision that agent
exactly once under concurrency, using the binding runtime. Provisioning failure
SHALL record a retryable failed binding and SHALL NOT enqueue a task or fall
back to the shared runtime.

#### Scenario: Concurrent mentions provision once
- **WHEN** multiple concurrent mentions target the same pending binding
- **THEN** exactly one sandbox, one runtime, and one channel-agent session are created, and no task uses the default runtime

### Requirement: Branch validation before writes

A branch SHALL validate that the source env has a decodable trigger whose target
agent belongs to its EnvDispatch channel and that the requested roster matches
the source channel roster. A missing, malformed, unauthorized, or
roster-incompatible trigger SHALL return an error before any resource is
created.

#### Scenario: Invalid trigger creates no resources
- **WHEN** a branch is requested against an env with a missing or unauthorized trigger
- **THEN** no env fork, project, channel, runtime, or sandbox is created and the request fails with a validation error

### Requirement: Branch resume from copied collaboration

A branch SHALL copy the source channel's members, messages, replies, quotes,
threads, and read state, remap the persisted trigger to the copied entities, and
wake only the trigger-selected agent by cloning its source sandbox. Non-triggered
agents SHALL remain pending with their source sandbox recorded as a lazy clone
source. Request `message.content`, when non-empty, SHALL be appended as
nondispatching context without changing the trigger-selected agent.

#### Scenario: Only the trigger agent wakes with a cloned sandbox
- **WHEN** a branch resumes a collaboration whose trigger selects agent A
- **THEN** only agent A is provisioned (via sandbox clone from its source binding), its task is enqueued on the new runtime, peers remain pending, and the remapped trigger is saved on the new env

### Requirement: Channel-first lifecycle facades

Channel-first routes SHALL resolve the bound project internally:
`GET /api/v1/env-dispatch/channels/{channelID}/dag`,
`DELETE /api/v1/env-dispatch/channels/{channelID}`, and
`GET /api/v1/channels/{channelID}/env-checkpoints`. Cleanup SHALL be idempotent
and SHALL serialize with concurrent provisioning via the env-agent binding

```

Full source: openspec/changes/env-dispatch-message-channels/specs/env-dispatch-message-channels/spec.md

# AReaL, Multica, and Remote Sandbox Environment Protocol

This document explains how AReaL communicates with Multica for multi-agent DAG RL
rollouts, and how rollout environments are created, branched, and cleaned up through the
unified env-dispatch API. The Python training side intentionally sees HTTP seams and
small protocols only; it does not import a sandbox-vendor SDK.

## Actors

| Actor                                 | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AReaL trainer / tree-search agents    | Starts rollout groups, consumes RL `session_id`s produced by Multica, records trajectories, selects branch points, asks Multica to materialize branches via env-dispatch, verifies rewards, and cleans up.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Multica API                           | Owns the unified env-dispatch primitive: base env boot, scratch/branch dispatch, issue/chat/task orchestration, agent-run startup, and the mapping from rollout lanes to projects/issues/sandboxes. On a source agent's first address it provisions a sandbox (shared `EnvSandboxLifecycleService`), discovers the daemon-registered runtime, clones a **derived global agent** bound to it, and calls `db_bridge.start_session(session_ref=<binding-ID>)` to register the RL session and obtain a scoped `api_key`. Server-side owns sandbox snapshot/fork/issue-subtree-copy as part of `mode="branch"`; the fork is now an immutable savepoint shared by N lanes rather than a per-dispatch disposable snapshot. |
| db_bridge                             | AReaL-hosted control-plane + model-serving bridge. Exposes `start_session` (returns `session_id` + scoped `api_key`), `set_reward`, `end_session`, `export_trajectories`, and the AReaL-served model endpoint. Multica and the agent runtime call it over HTTP; AReaL never imports its internals.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Remote sandbox server / cloud runtime | Owns live sandbox lifecycle behind Multica's env-dispatch endpoints. Multica calls this service; AReaL talks to it only through `ForkableEnvironment` providers when an explicit injection path is wired.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Agent runtime                         | Runs inside the allocated sandbox, calls the AReaL-served model via db_bridge using the `api_key` Multica handed it (provider `areal`), emits messages/interactions, and carries RL session metadata used by AReaL.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## Design Boundaries

- AReaL treats Multica as the orchestration authority for issues, projects, chat
  sessions, agent runs, rollout groups, and branch materialization.
- AReaL treats sandbox operations as a vendor-neutral `ForkableEnvironment` protocol:
  `snapshot`, `fork`, `restore`, and `cleanup`. This seam is kept for explicit injection
  scenarios; the live branch path does **not** flow through it.
- Branching is **a single env-dispatch call**:
  `env_dispatch(mode="branch", env_id=<source>)`. Multica performs the sandbox fork and
  the issue-subtree copy server-side; AReaL never snapshots, forks, or replays messages
  itself.
- Multica hides the concrete sandbox vendor behind its cloud-runtime endpoints. Today
  the remote side may be Daytona/Fleet-backed, but that detail must not leak into
  trainer code.

### Pipeline Notes

- `env_id` is the canonical Multica-facing environment handle: base envs, lane envs, and
  branch envs are all identified by `env_id`. The `SuperNode.env_id` field is the branch
  frontier captured on each segment's terminal turn; the per-turn `Node.env_id` (in
  `core/tree_store.py`) is the same handle stamped from backend per-turn metadata.
- `sandbox_id` is the remote runtime handle used by `ForkableEnvironment`
  snapshot/fork/restore/delete. It is **not** used by the live branch path; the
  `_BranchDriver` Protocol parameter is named `sandbox_id` only for structural
  compatibility with the runner — the value passed in is the source `env_id`.
- Fresh rollout state is created by `env_dispatch(mode="scratch", ...)`; branch state is
  created by `env_dispatch(mode="branch", env_id=<source>)`, which the server serves by
  creating or reusing a `snapshot` checkpoint at the source env and resuming it into
  lanes; resume-from-checkpoint is created by
  `env_dispatch(mode="resume", env_id=<checkpoint_id>)`, and for a `pause_in_place`
  checkpoint that is still an in-place resume of the same instances (see "Env Checkpoint
  Semantics").
- The verifier writes rewards before trajectory harvest so AReaL never trains on an
  unrewarded terminal trajectory.
- Cleanup flows through Multica env-dispatch; `404` means the resource is already gone
  and is treated as success.

### Per-Agent Ephemeral Sandbox Pipeline (ARE-5 derived-agent provisioning)

This is the end-to-end flow for a `mode="scratch"`, `dispatch_type="message"` dispatch -
the path `MultiAgentEnvDispatchWorkflow` uses. ARE-5
(`env-dispatch-agent-runtime-config`) replaced the old "pre-create an offline
`agent_runtime` R′ and let the daemon adopt it" model with frontend-parity provisioning:
no runtime row is pre-created, the shared sandbox-creation service boots a Cube with a
daemon, the server **discovers** the runtime the daemon registers, **clones a derived
global agent** bound to that runtime, and only then enqueues the task. The `daemon_id`
is now a correlation nonce, not a pre-created-row reference.

> **Scope.** This no-pre-create flow is the scratch+message path only. The legacy
> `PrecreateAgentRuntime` flow is **still used** for `dispatch_type="issue"` (SWE-Lego,
> trained/instance-backed) and for `mode="branch"` message dispatch with a source
> sandbox (`provisionEnvDispatchAgentBranch`); ARE-5 rewired only scratch+message. The
> feature flag `envDispatchDerivedAgentEnabled` (default `true`) gates new derived
> provisioning at entry; `=false` rejects new provisioning without disrupting
> in-flight/ready bindings, and there is **no** legacy pre-create fallback for
> scratch+message.

```mermaid
sequenceDiagram
    autonumber

    participant A as 🟦 AReaL<br/>(MultiAgentEnvDispatchWorkflow)
    participant H as 🟩 Multica Handler<br/>(EnvDispatch)
    participant S as 🟩 EnvDispatchService<br/>(Dispatch)
    participant DB as 🟨 Postgres
    participant LC as 🟩 EnvSandboxLifecycle<br/>Service
    participant SD as 🟥 Sandboxd / Cube
    participant D as 🟪 In-Sandbox Daemon
    participant PR as 🟩 Channel Provisioner<br/>(provisionEnvDispatchAgent)
    participant G as 🟧 db_bridge<br/>(start_session)
    participant TS as 🟩 TaskService

    rect rgb(230, 245, 255)
        Note over A,S: Phase 1 - Request ingress & validation
        A->>H: POST /api/v1/env-dispatch<br/>Authorization: Bearer <PAT><br/>mode=scratch, dispatch_type=message,<br/>agent_id, group_size, message, train_agent_id
        H->>S: Dispatch(EnvDispatchInput)
        S->>S: validate - mode, dispatch_type, group_size,<br/>domain<->dispatch_type, train_agent_id
    end

    rect rgb(255, 245, 230)
        Note over S,DB: Phase 2 - Resolve base env, create project + channel + bindings
        S->>DB: GetEnv(base_env_id) -> sourceEnv (scratch expects a base env)
        S->>DB: CreateEnv(parentEnvID=sourceEnv, mode=scratch)
        S->>DB: CreateProject(envID)
        S->>DB: CreateEnvDispatchChannel -> channelID
        Note over DB: insert environment_agent_sandbox binding per roster agent:<br/>{env_id, channel_id, source_agent_id,<br/> model_config_owner_agent_id=source_agent_id, status=pending}
        Note over S: peers stay pending; the scratch leader's first dispatch<br/>counts as its first directed address (provisioned next)
    end

    rect rgb(230, 255, 230)
        Note over S,PR: Phase 3 - provisionEnvDispatchAgent (leader, single-flight)
        S->>PR: ProvisionEnvDispatchAgent(leader binding)
        PR->>PR: envDispatchDerivedAgentEnabled gate (default true)
        PR->>PR: claimProvisioning - single-flight winner<br/>(pending|failed_retryable -> credential_ready);<br/>concurrent mentions observe the winner and wait
        PR->>PR: validateEnvDispatchCredentialOwner<br/>(model_config_owner_agent_id == source_agent_id;<br/>mismatch -> fail closed -> failed_retryable)

        alt training target (train_agent_id set)
            PR->>G: ResolveEnvDispatchTrainingSession<br/>(session_ref = binding.ID, env_id)
            Note over PR,G: session_ref is the persistent env-agent binding ID,<br/>not a task id. Retry reuses the recorded session (Opened=false);<br/>first address opens it (Opened=true).
            G-->>PR: session_id + session_key (model=areal-default)
            PR->>DB: setTrainingSession(binding, session_id, key)
            PR->>PR: EnvDispatchTrainingRuntimePolicy<br/>(base_url=bridge_url, api_key=session_key, model=areal-default)
        else non-training source agent
            Note over PR: caller-supplied per-source runtime policy<br/>{base_url, api_key, model}; squad members never<br/>inherit another member's policy
        end

        PR->>LC: CreateSandboxInstance(template, DaemonEnabled=true)
        Note over LC: daemon_id = fresh correlation nonce (NO pre-created row);<br/>mint bootstrap PAT + runtime_env (SERVER_URL, PAT,<br/>WORKSPACE_ID, DAEMON_ENABLED=1, PROFILE)
        LC->>SD: EnqueueSandboxJob("create", payload with instance_id)
        SD-->>LC: sandbox_instance_id (running)

        PR->>PR: WaitForOnlineSandboxRuntime<br/>(workspace, provider=pi, daemon_id,<br/> sandbox_instance_id, status=online)
        Note over PR: identity mismatch -> fail closed; timeout -> readiness failure<br/>(no fallback to a shared / source runtime)

        PR->>PR: CloneEnvDispatchAgentTx (single tx): copy approved fields<br/>(instructions, skills, model, custom_env, mcp_config, ...);<br/>record source_agent_id; bind runtime_id; source agent immutable
        PR->>DB: replace source -> derived member in dispatch channel only
        PR->>PR: markReady (binding status=ready)
    end

    rect rgb(255, 230, 255)
        Note over SD,D: Phase 4 - Cube boot & daemon registration
        SD->>D: Inject runtime_env (MULTICA_DAEMON_ID, PAT, ...)
        D->>H: POST /api/daemon/register<br/>{daemon_id, runtimes:[{type,version,status}],<br/> sandbox_instance_id metadata}
        H->>DB: UpsertAgentRuntime(workspace, daemon_id, provider=pi,<br/>status=online, metadata.sandbox_instance_id)
        H-->>D: 200 {runtimes:[{id, ...}], daemon_token}
        Note over PR: WaitForOnlineSandboxRuntime resolves this row by the<br/>4-tuple (daemon_id + sandbox_instance_id + provider + online)
    end

    rect rgb(230, 255, 255)
        Note over S,TS: Phase 5 - dispatchScratchChannelMessage
        S->>DB: CreateChannelMessage(channelID, "user", content)
        S->>TS: EnqueueEnvDispatchChannelRun(derived_agent_id, runtime_id)
        Note over S: r.AgentRunID = real task id
        S->>S: LinkEnvDispatchTrainingSession (best-effort, AFTER enqueue)<br/>dagSvc.LinkSessionTask: session_id -> real task_id for DAG
        S-->>H: EnvDispatchResult {channelID, rollouts[]}
        H-->>A: 201 {project_id, channel_id,<br/> rollouts[{env_id, project_id, chat_session_id, agent_run_id,<br/>  agent_sandboxes:[{status, sandbox_instance_id, runtime_id}], error?}]}
    end

    rect rgb(240, 255, 240)
        Note over D,TS: Phase 6 - Daemon poller claim & execute (derived agent's runtime)
        loop runRuntimePoller per runtime
            D->>H: POST /api/daemon/runtimes/{runtimeID}/claim
            H->>TS: ClaimTaskForRuntime(ctx, runtimeID)
            H-->>D: {task:{id, agent_id=derived_agent_id, context,<br/> areal_proxy / ephemeral_sandbox, runtime policy}}
            alt task claimed
                D->>D: launch pi process, provider=areal,<br/>api_key = session_key (training) or caller key
                D->>H: stream messages / tool calls / usage
            else no task
                D->>D: sleep poll interval + jitter
            end
        end
    end

    rect rgb(255, 240, 230)
        Note over A,SD: Phase 7 - Terminal cleanup (AC-6 cascade)
        A->>H: DELETE /api/v1/env-dispatch/channels/{channelID}
        H->>S: deleteEnvDispatchChannelRollout
        S->>S: markDeleting; waitForEnvDispatchProvisioning; listBindings
        loop per ready binding
            S->>TS: CancelTasksForAgent(derived_agent_id)
            S->>DB: ArchiveAgent(derived_agent_id, archived_by=NULL)
            S->>LC: lifecycle.Delete (stop sandbox, revoke bootstrap PAT)
            S->>DB: DeleteAgentRuntime
            S->>G: EndSession(training_session_key)  (if present)
        end
        S->>DB: delete channel, project, environment_agent_sandbox, env
        Note over S: idempotent - already-absent resources are success;<br/>source agent + its other-channel memberships untouched
    end
```

**Key rows and their lifecycle:**

| Row                                   | Created by                                                                              | Status transitions                                                                                                                  | Reclaimed by                              |
| ------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `environment_agent_sandbox` binding   | `CreateEnvDispatchChannel` (`insertBinding`, status=pending)                            | `pending` → `credential_ready` (claimProvisioning) → `ready` (markReady); `→ failed_retryable` (markFailed); `→ deleting → deleted` | `deleteEnvDispatchChannelRollout` (AC-6)  |
| `sandbox_instance`                    | `EnvSandboxLifecycleService.Create`                                                     | `pending` → `creating` → `running`                                                                                                  | `lifecycle.Delete` (revoke bootstrap PAT) |
| `agent_runtime`                       | daemon `UpsertAgentRuntime` on register (online; **not** pre-created)                   | `online` (registered)                                                                                                               | `DeleteAgentRuntime` (AC-6)               |
| derived `agent`                       | `CloneEnvDispatchAgentTx` (lineage `source_agent_id`, bound `runtime_id`)               | static; source agent immutable                                                                                                      | `ArchiveAgent` (AC-6; `archived_by=NULL`) |
| `agent_task_queue`                    | `EnqueueEnvDispatchChannelRun`                                                          | `queued` → `dispatched` → `running` → terminal                                                                                      | `CancelTasksForAgent` (AC-6)              |
| training session                      | `ResolveEnvDispatchTrainingSession` → `db_bridge.start_session(session_ref=binding.ID)` | opened once; reused on retry                                                                                                        | `EndSession(training_session_key)` (AC-6) |
| `environment` / `project` / `channel` | `CreateEnv` / `CreateProject` / `CreateEnvDispatchChannel`                              | static                                                                                                                              | `deleteEnvDispatchChannelRollout` cascade |

**Provisioning state machine (actual):** the binding moves `pending` ->
`credential_ready` (single-flight `claimProvisioning`) -> `ready` (`markReady`), or
`-> failed_retryable` (`markFailed`). The finer states `sandbox_creating` /
`runtime_waiting` / `agent_creating` are defined in the schema and accepted as in-flight
by `routeEnvDispatchChannelAgent` (concurrent mentions wait on them) and by the
`markReady` / `markFailed` / `markDeleting` WHERE clauses, but the current provisioner
does not write them - it does all the work (sandbox create -> runtime discovery ->
derived-agent clone) between `credential_ready` and `ready`. A retry re-claims from
`failed_retryable` back to `credential_ready`.

**Single-flight + lazy peers:** only the first directed mention of a source agent claims
provisioning; concurrent mentions observe the winner and wait for the same terminal
result. The scratch leader's initial dispatch is its first address. Peer (non-trigger)
agents stay `pending` and are materialized lazily only when the collaboration actually
mentions them (same state machine).

**Credential isolation invariant:** before sandbox creation the server verifies
`binding.source_agent_id == model_config_owner_agent_id` (and, for training,
`== start_session source`); before enqueue it verifies
`binding.runtime_id == derived_agent.runtime_id` and
`runtime.sandbox_instance_id == binding.sandbox_instance_id`. Any mismatch fails closed
with compensation (delete partial sandbox, revoke PAT, close session) - no fallback to a
shared or source-agent runtime.

**Offline-runtime sweeper backstop:** the `runtime_sweeper`
(`cmd/server/runtime_sweeper.go`, `offlineRuntimeQueuedTTLSeconds=300`) still fails
queued tasks whose runtime has been `offline` >5 min (`failure_reason=runtime_offline`).
On the scratch+message path there is no pre-created offline runtime, so this primarily
backstops the legacy issue/branch pre-create flows; its in-code comment still references
the old pre-create model and is stale.

### Sandbox credential injection: training vs non-training agent

Each provisioned Cube receives **two layered credential bundles**, supplied at distinct
phases of the ARE-5 pipeline. They serve different purposes and never mix:

1. **Bootstrap `runtime_env` (sandbox boot, Phase 3 -> 4)** - identical for training and
   non-training agents. `EnvSandboxLifecycleService.Create` mints a *bootstrap* PAT and
   packs `runtime_env` into the sandbox-creation job. The Cube injects it into the
   daemon process as environment variables:

   | Env var                  | Source                                     | Purpose                                                         |
   | ------------------------ | ------------------------------------------ | --------------------------------------------------------------- |
   | `MULTICA_SERVER_URL`     | multica server base URL                    | Where the daemon registers itself (`POST /api/daemon/register`) |
   | `MULTICA_PAT`            | bootstrap PAT minted by `lifecycle.Create` | One-time daemon-registration credential; revoked on cleanup     |
   | `MULTICA_WORKSPACE_ID`   | resolved workspace                         | Scopes the daemon's runtime row to this workspace               |
   | `MULTICA_DAEMON_ID`      | fresh correlation nonce                    | Correlates the daemon's `register` to this sandbox instance     |
   | `MULTICA_DAEMON_ENABLED` | `1`                                        | Enables the daemon poller                                       |
   | `MULTICA_PROFILE`        | profile label                              | Groups runtimes for routing                                     |

   This bundle lets the daemon register a `runtime` row and poll for tasks. It is
   **not** the per-agent model-serving credential.

1. **Agent runtime policy (task claim, Phase 6)** - this is the layer that differs
   between training and non-training agents. The daemon, when it claims a task, receives
   the runtime policy in the task payload; it launches the agent process
   (`provider=areal`) with `api_key` read from that policy. The policy is computed in
   Phase 3 (`provisionEnvDispatchAgent`) before the sandbox is booted and recorded on
   the binding, then handed to the daemon at enqueue/claim time:

   | Agent role                                                                       | `base_url`                                                                                  | `api_key`                                                                                        | `model`                                              | How it is produced                                                                                                                                                                                                                                                                                                                                                                                                               |
   | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | **Training target** (`train_agent_id` set; equals `agent_id` or a squad member)  | db_bridge URL (`bridge_url`)                                                                | the scoped **session_key** returned by `db_bridge.start_session(session_ref=binding.ID, env_id)` | `areal-default`                                      | Phase 3 alt: `ResolveEnvDispatchTrainingSession` -> `start_session` -> `session_id + session_key`; `setTrainingSession(binding, session_id, key)` stores it; `EnvDispatchTrainingRuntimePolicy(base_url=bridge_url, api_key=session_key, model=areal-default)` builds the policy. A retry reuses the recorded session (no second `start_session`).                                                                               |
   | **Non-training source agent** (`train_agent_id` absent, or not the roster agent) | caller-supplied `per_agent_env[agent].runtime.base_url` (or the agent's configured runtime) | caller-supplied `per_agent_env[agent].runtime.api_key`                                           | caller-supplied `per_agent_env[agent].runtime.model` | Phase 3 else-branch: the caller-supplied per-source runtime policy overrides the agent's default. **Squad members never inherit another member's policy** - each non-training agent carries its own `{base_url, api_key, model}`. For local debug dispatch `multica_client._debug_external_runtime_policy` builds this from `MULTICA_EXTERNAL_{PROVIDER,BASE_URL,API_KEY,MODEL}` (and is rejected when `train_agent_id` is set). |

   The invariant is: the agent runtime cannot do model inference until it has claimed a
   task and read the runtime policy; the sandbox's bootstrap PAT only authorizes daemon
   \<-> multica control-plane traffic, never model serving. This is why the `api_key`
   for an **AReaL training** agent is db_bridge-scoped (provider `areal`, model
   `areal-default`) and cleanly revoked `EndSession` on cleanup, while a
   **non-training** agent may point at any external provider (e.g. an OpenAI-compatible
   endpoint supplied via `per_agent_env`) without ever touching db_bridge.

### Multi-agent team provisioning (each agent in its own sandbox)

A "team" rollout is a single `POST /api/v1/env-dispatch` with a roster of >1 agent -
supplied either as a `squad_id` (squad dispatch, `agent_id` intentionally omitted since
the squad supplies its members) or as repeated `agent_id`-keyed `per_agent_env` entries.
The roster is resolved into one `environment_agent_sandbox` binding per member at
dispatch time (Phase 2), all starting `pending`. Each agent is then materialized lazily
in its **own** Cube + daemon + derived-agent clone, on first address:

```mermaid
sequenceDiagram
    autonumber
    participant A as 🟦 AReaL<br/>(MultiAgentEnvDispatchWorkflow)
    participant S as 🟩 EnvDispatchService
    participant DB as 🟨 Postgres
    participant PR as 🟩 Channel Provisioner
    participant LC as 🟩 EnvSandboxLifecycle<br/>Service
    participant SD as 🟥 Sandboxd / Cube(s)
    participant D as 🟪 In-Sandbox Daemon(s)
    participant G as 🟧 db_bridge

    A->>S: POST /api/v1/env-dispatch<br/>mode=scratch, dispatch_type=message,<br/>squad_id (or agent roster), group_size, train_agent_id
    S->>DB: CreateEnvDispatchChannel -> channelID
    loop per roster agent
        S->>DB: insert binding {env_id, channel_id,<br/>source_agent_id, model_config_owner_agent_id=source_agent_id,<br/>status=pending}
    end
    Note over S: leader (first-directed) provisioning starts;<br/>peers stay pending

    rect rgb(230, 255, 230)
        Note over PR,SD: leader provisioning (single-flight, ARE-5)
        S->>PR: ProvisionEnvDispatchAgent(leader binding)
        PR->>PR: claimProvisioning (pending -> credential_ready)
        alt leader is the training target<br/>(train_agent_id == leader source_agent_id)
            PR->>G: ResolveEnvDispatchTrainingSession(session_ref=binding.ID)
            G-->>PR: session_id + session_key
            PR->>PR: training runtime policy<br/>{base_url=bridge_url, api_key=session_key, model=areal-default}
        else leader is non-training
            PR->>PR: caller-supplied per-source policy<br/>{base_url, api_key, model} (from per_agent_env)
        end
        PR->>LC: CreateSandboxInstance (mint bootstrap PAT + runtime_env)
        LC->>SD: EnqueueSandboxJob("create")
        SD-->>LC: sandbox_instance_id (running)
        SD->>D: inject runtime_env (SERVER_URL, PAT, DAEMON_ID, ...)
        D->>S: POST /api/daemon/register (daemon_id, runtimes)
        PR->>PR: WaitForOnlineSandboxRuntime (online)
        PR->>PR: CloneEnvDispatchAgentTx (bind runtime_id; record source_agent_id)
        PR->>DB: replace source -> derived member in channel;<br/>markReady (binding status=ready)
    end

    rect rgb(255, 245, 230)
        Note over S,SD: peer provisioning (lazy, on first @mention)
        loop each later-mentioned agent (not the leader)
            S->>PR: ProvisionEnvDispatchAgent(peer binding)
            Note over PR: same state machine as leader;<br/>single-flight so concurrent mentions win once
            alt peer == train_agent_id
                PR->>G: ResolveEnvDispatchTrainingSession(peer binding.ID)
                G-->>PR: session_id + session_key
                PR->>PR: training runtime policy
            else peer is non-training
                PR->>PR: caller-supplied per-source policy (peer's own)
            end
            PR->>LC: CreateSandboxInstance (own Cube + own daemon)
            SD->>D: inject runtime_env (own bootstrap PAT)
            D->>S: register (own agent_runtime row)
            PR->>PR: CloneEnvDispatchAgentTx (own derived agent, own runtime_id)
            PR->>DB: markReady
        end
    end

    S->>DB: CreateChannelMessage(leader's "user" content)
    S->>DB: EnqueueEnvDispatchChannelRun(leader derived_agent_id, runtime_id)
    S-->>A: 201 {project_id, channel_id, rollouts[]}
    Note over D: per-runtime poller claims its task, launches agent process<br/>with runtime-policy api_key (training=session_key, non-training=caller key)
```

Key properties of the team flow:

- **One binding per roster agent, one Cube per provisioned binding.** The dispatch
  records every roster member up front (Phase 2), but a Cube/daemon/derived-agent
  triplet is only built for an agent when it is actually addressed. Unaddressed peers
  never incur sandbox cost.
- **Single-flight per agent.** `claimProvisioning` makes the first mention win and
  concurrent mentions observe the winner and wait for the same terminal
  (`ready`/`failed_retryable`) result, so a team never provisions the same agent twice.
- **Exactly one training target.** `train_agent_id` names a single roster agent (it must
  equal `agent_id` for a single-agent dispatch, or be a squad member for a squad
  dispatch). Only that binding gets a `db_bridge` session + session key runtime policy;
  every other member uses its own non-training policy. The server enforces this in
  `validate()` (spec §4.1).
- **Credential isolation.** `validateEnvDispatchCredentialOwner` checks
  `binding.source_agent_id == model_config_owner_agent_id` before sandbox creation and
  `binding.runtime_id == derived_agent.runtime_id` before enqueue; any mismatch fails
  closed (delete partial sandbox, revoke bootstrap PAT, close session) - a non-training
  peer can never be bootstrapped with another member's session key, and the training
  agent can never fall back to a shared/runtime default.
- **Cleanup scales with the team.** `DELETE /api/v1/env-dispatch/channels/{channelID}`
  (AC-6) iterates every ready binding: cancels tasks, archives each derived agent,
  deletes each sandbox (revokes its bootstrap PAT), deletes each runtime row, and
  `EndSession`s the training session if present.

## AReaL → Multica API Surface

`agents/multica_client.py` wraps the unified env-dispatch API through
`MulticaEnvDispatchClient`, and `agents/multica_dag_client.py` polls the assembled DAG.
Both address `MULTICA_BASE_URL` directly and attach a MultiCA PAT as
`Authorization: Bearer <key>`. Obtain and save the PAT explicitly before starting
training:

```bash
uv run python -m customized_areal.tree_search.agents.multica_auth login \
  --base-url "$MULTICA_BASE_URL"
```

The clients prefer an explicit `api_key`, then `MULTICA_API_KEY`, then the PAT in
`agents/credentials.json`. The saved PAT is accepted only when its recorded base URL
matches the configured MultiCA server.

For Ray, Slurm, or other distributed launches, every worker must receive a
network-reachable (non-loopback) `MULTICA_BASE_URL`. Inject `MULTICA_API_KEY` through
the launcher's worker environment or secret mechanism, or mount the same checkout-local
`agents/credentials.json` path on every worker. Logging in only on the submission host
is insufficient when workers do not share that filesystem.

| AReaL call                                                                                                                                        | Direct Multica endpoint                                                                                                      | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `create_env_dispatch(...)`                                                                                                                        | `POST <MULTICA_BASE_URL>/api/v1/env-dispatch`                                                                                | Unified dispatch primitive. Returns an `EnvDispatchHandle` (`channel_id`, `project_id`, `env_id`, `dispatch_type`). For `dispatch_type="message"` the response carries a top-level `channel_id` (validated at the boundary); for `dispatch_type="issue"` `channel_id` is absent and `project_id` is the primary handle. A fresh env is booted **by the dispatch itself** (`mode="scratch"`): the client no longer calls `POST /api/v1/env` to pre-boot a base environment. Covers fresh rollouts (`mode="scratch"`), branch forks (`mode="branch"`), and resume-from-checkpoint (`mode="resume"`, sandbox instances resumed in place — not an alias for `branch`). Fail-closed on per-rollout failure: if any entry in `rollouts[]` carries an `error`, the client raises `RuntimeError("env-dispatch rollout failed: …")` instead of returning a half-formed handle, so a terminal provisioning failure on one lane can never train as success (ARE-5 AC-7). |
| `get_dag(handle=...)`                                                                                                                             | message: `GET .../api/v1/env-dispatch/channels/{channelID}/dag`; issue: `GET .../api/v1/env-dispatch/{projectID}/dag`        | Poll the assembled segment DAG, routed by `handle.dispatch_type`: `202` not-ready, `200` assembled DAG, `404` unknown, `403` cross-workspace. Transient `502`/`503`/`504` are re-polled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `list_checkpoints(handle=...)`                                                                                                                    | message: `GET .../api/v1/channels/{channelID}/env-checkpoints`; issue: `GET .../api/v1/projects/{projectID}/env-checkpoints` | List env checkpoints for the dispatch, routed by `handle.dispatch_type`, newest first. `403`/`404`/`409` raise `MulticaCheckpointError(status_code, body)` so AReaL can distinguish a non-resumable/missing checkpoint from a transient error.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `create_checkpoint(project_id=..., event_ref=..., checkpoint_kind=..., env_id_map=..., sandbox_refs=..., entropy_score=..., save_timeout_ms=...)` | `POST .../api/v1/env-checkpoints`                                                                                            | Synchronous in-place save. Returns the checkpoint dict on `201`. `403`/`404`/`409` raise `MulticaCheckpointError`; other non-2xx raise `RuntimeError`. Checkpoint creation stays project-internal (`project_id` in the body); `maybe_create_entropy_checkpoint(...)` wraps it with the `should_create_entropy_checkpoint(entropy, threshold)` gate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `cleanup_env_dispatch(handle=...)`                                                                                                                | message: `DELETE .../api/v1/env-dispatch/channels/{channelID}`; issue: `DELETE .../api/v1/env-dispatch/{projectID}`          | Serialized cascade cleanup for one dispatch: channel/project, issues, chat sessions, tasks, bindings, env, and runtime state. `404` is treated as success (idempotent).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

The DAG poller (`MulticaDagClient.get_dag`) is fail-closed. On `200` it parses the
payload through `AssembledDag.from_dict`, which structurally validates the DAG -
non-empty `segments`, `edges` is a list, `session_to_agent_run` is an object, no
empty/duplicate `segment_id`, every edge references a known segment, no cycles, and
every mapped agent run exists - raising `DagError` on any violation so an
incomplete/malformed DAG can never train as a valid one (ARE-5 AC-7/AC-8). Transient
`202`/`502`/`503`/`504` are re-polled up to the wall-clock deadline, then `DagTimeout`
is raised; `404` maps to `DagNotFound`, `403` to `DagForbidden`, `401` reports the
explicit login command without exposing the PAT, and any other status raises `DagError`.
The debug helper `_poll_dag` (in `multica_client.py`) mirrors this: it validates the
assembled DAG and raises on non-transient status or deadline instead of silently
returning.

### Segment close (no reward) via the gateway group

`POST /rl/close_segment` closes a session's current segment into a ready trajectory
**without** setting a reward. It decouples the trajectory boundary from reward so each
communication-bounded segment is its own exportable trajectory; the segment's reward is
assigned later (AReaL-side judge), not at close. This is the segment-DAG path that
slices one RL session into multiple per-segment trajectories.

Unlike the env-dispatch calls above (AReaL -> Multica), `close_segment` flows **Multica
-> db_bridge -> AReaL**: the multica `arealrl` client posts to the db_bridge stub
(MultiCA side) over the `gateway` group channel `rl_close_segment`, and the AReaL-side
executor forwards it to the real AReaL gateway. The session key passes through end to
end, mirroring `set_reward` / `end_session`; the channel is registered as
`rl_close_segment` so the stub no longer 404s.

| Aspect            | Value                                                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| db_bridge channel | `rl_close_segment` (group `gateway`, `POST /rl/close_segment`, `kind=json`, `default_timeout_s=30`, `default_concurrency=4`) |
| Auth              | Session key: `Authorization: Bearer <proxy_key>`. The gateway also accepts the admin key, mirroring `/rl/set_reward`.        |
| Body              | None / empty `{}`. No `session_id` is sent - the endpoint resolves the session from the bearer token.                        |
| Caller            | Multica `arealrl.Client.CloseSegment(ctx, proxyKey)`, invoked from `InteractionDAGService.CloseSegmentForEvent`.             |

**Gateway** (`areal/v2/inference_service/gateway/app.py`): extracts the bearer token and
body, best-effort reads `model` from the JSON body for router routing, queries the
router for the worker address, and forwards the request verbatim to
`{worker_addr}/rl/close_segment`. The worker's response status, content, and
content-type are returned unchanged.

**Data proxy** (`areal/v2/inference_service/data_proxy/app.py`): resolves the session
from the token (`401` on an invalid/expired key), calls `session.close_segment()` (`400`
on `ValueError`), and returns a `CloseSegmentResponse`.

**`Session.close_segment()`** (`data_proxy/session.py`): under the session lock, takes
the active segment's `last_interaction_id` as the terminal interaction, mints the next
`trajectory_id`, stores a `ReadyTrajectory` (`needs_online_callback=False`), and resets
`_active_completions` to a fresh `InteractionCache()` so the next turn starts a new
segment. It deliberately does **not** touch `_last_reward_interaction_id` /
`_last_set_reward_time`, so reward-timeout finalization is unaffected. Raises
`ValueError("No interactions in session")` when the active segment is empty.

Response (`CloseSegmentResponse`):

| Field               | Meaning                                               |
| ------------------- | ----------------------------------------------------- |
| `message`           | `"success"`.                                          |
| `interaction_count` | Number of interactions in the just-closed segment.    |
| `session_id`        | The session the segment belonged to.                  |
| `trajectory_id`     | The just-closed segment's trajectory id (nullable).   |
| `trajectory_ready`  | `true` when `trajectory_id` is present.               |
| `ready_transition`  | `true` - close always transitions a segment to ready. |

**Multica-side contract:** the `arealrl` client treats a missing/nil `trajectory_id` as
an error - every close must yield a trajectory to export. `CloseSegmentForEvent` then
runs the per-segment pipeline: (a) resolve `agent_run_id` + `issue_id` from the session
mapping, (b) `close_segment` -> `trajectory_id`, (c) `export_trajectories` (admin-key
auth, `remove_session=false`) for that trajectory, (d) decode the `tensor_ref`, and (e)
record the segment + env snapshot atomically. The session stays alive across per-segment
exports (`remove_session=false`).

The per-segment close pipeline (one iteration of `CloseSegmentForEvent`):

```mermaid
sequenceDiagram
    participant M as Multica<br/>(InteractionDAGService / arealrl)
    participant B as db_bridge<br/>(gateway group)
    participant A as AReaL<br/>(gateway -> data proxy)

    Note over M: segment boundary reached<br/>(communication-bounded segment)
    M->>M: (a) resolve agent_run_id + issue_id<br/>from session mapping
    M->>B: POST /rl/close_segment<br/>Authorization: Bearer <proxy_key><br/>body {} (no session_id)
    B->>A: forward (rl_close_segment channel)
    Note over A: gateway: router-resolve worker, forward<br/>data proxy: resolve session from token
    A->>A: Session.close_segment()
    Note over A: terminal = last_interaction_id;<br/>mint trajectory_id; store ReadyTrajectory;<br/>reset active completions;<br/>reward state untouched
    A-->>B: CloseSegmentResponse<br/>{trajectory_id, interaction_count,<br/>ready_transition=true}
    B-->>M: trajectory_id

    M->>B: POST /export_trajectories<br/>Authorization: Bearer <admin_key><br/>{session_ids, trajectory_id,<br/>remove_session=false}
    B->>A: forward (gateway group)
    A-->>B: {traj: merged interactions}
    B-->>M: raw traj JSON
    M->>M: (d) decode tensor_ref
    M->>M: (e) record segment + env snapshot<br/>atomically
    Note over M: reward assigned later (AReaL-side judge);<br/>session stays alive for next segment
```

`create_env_dispatch` accepts these key fields:

| Field               | Meaning                                                                                                                                                                                                                                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`              | `scratch` (fresh env, booted by the dispatch itself), `branch` (fork a source env; `env_id` = source env), or `resume` (resume from a checkpoint; `env_id` = checkpoint id). There is no separate env-boot or checkpoint-resume endpoint.                                                                                    |
| `env_id`            | Source env for `branch`; checkpoint id for `resume`; optional for `scratch` (only valid to omit when `domain=self_play` and the workspace has a configured default base env).                                                                                                                                                |
| `dispatch_type`     | `issue` for SWE-Lego or `message` for self-play.                                                                                                                                                                                                                                                                             |
| `agent_id`          | Agent/template to run in each lane. For a single-agent dispatch the client resolves `MULTICA_AGENT_ID` (from `customized_areal/.env`) when omitted; squad dispatches (`squad_id` set) intentionally omit `agent_id` — the squad supplies its members, so the env default does not apply (spec §4.1).                         |
| `squad_id`          | Optional squad identifier for team rollouts.                                                                                                                                                                                                                                                                                 |
| `group_size`        | Number of rollout lanes to create. Defaults to `1`. For a branch dispatch this is the requested lane count: the server captures one savepoint of the source and materializes `group_size` lanes from it.                                                                                                                     |
| `idempotency_key`   | Stable per-dispatch id. Lane keys are derived from it, so retrying a branch dispatch with the same key returns the existing lanes instead of expanding the frontier again. Not yet enforced server-side: a branch dispatch without one currently derives lane keys from the rollout env, which is not stable across retries. |
| `domain`            | Optional domain label such as `swe_lego` or `self_play`.                                                                                                                                                                                                                                                                     |
| `train_agent_id`    | The single trainable target (spec §4.1): the named agent is trainable, all other agents in the dispatch are not. Empty means no training session. For a single-agent dispatch it must equal `agent_id`; for a squad dispatch it must be a squad member. The server enforces these in `validate()`.                           |
| `per_agent_env`     | Optional per-agent runtime-policy override (e.g. an external `{provider, base_url, api_key, model}` runtime for non-training debug dispatch). Mutually exclusive with `train_agent_id`.                                                                                                                                      |
| `training_mode`     | Boolean training-mode flag (default `false`); set to enable training session semantics.                                                                                                                                                                                                                                      |
| `issue` / `message` | Domain payload. Exactly one is usually populated by the runner.                                                                                                                                                                                                                                                              |

The response is normalized into `SweLegoSetup(rollouts=[...])` — a list with one entry
per lane (so a `group_size=N` scratch dispatch returns N rollouts, and a branch dispatch
returns 1). Each `SweLegoRollout` contains:

| Field             | Meaning                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `env_id`          | Runtime environment ID assigned to this lane. For a branch dispatch this is the new child `env_id`. |
| `project_id`      | Multica project ID used for cleanup.                                                                |
| `issue_id`        | Issue ID for issue-based runs; empty for message/self-play runs.                                    |
| `chat_session_id` | Chat session created for the agent lane.                                                            |
| `agent_run_id`    | Agent run started by Multica.                                                                       |

## Fresh Rollout Lifecycle

```mermaid
sequenceDiagram
    participant A as AReaL runner
    participant G as db_bridge
    participant M as Multica API
    participant S as Remote sandbox server
    participant R as Agent runtime

    Note over A,M: mode=scratch: the dispatch itself boots the base env
    A->>M: POST /api/v1/env-dispatch (mode=scratch, group_size, issue/message, MultiCA PAT)
    M->>S: create/bootstrap + allocate per-lane sandboxes
    S-->>M: sandbox_ids
    loop each rollout lane
        M->>G: start_session(session_ref=binding.ID, env_id)
        G->>A: forward
        A-->>G: session_id + api_key (provider=areal)
        G-->>M: session_id + api_key (provider=areal)
        M->>R: start agent run with api_key (provider=areal)
    end
    M-->>A: 201 rollouts[env_id, project_id, issue_id, chat_session_id, agent_run_id]

    loop each rollout lane
        R->>G: model inference (api_key, provider=areal)
        G->>A: forward
    end

    M->>G: set_reward(session_id, reward)
    G->>A: forward
    M->>G: end_session(session_id) or export_trajectories(session_id)
    G->>A: forward
    A->>M: DELETE /api/v1/env-dispatch/{projectID} (per rollout)
    M->>S: delete lane sandboxes and runtime resources
```

The AReaL orchestrator is `MultiAgentEnvDispatchWorkflow` (a `RolloutWorkflow`, in
`agents/multi_agent_workflow.py`). One `arun_episode` = one Multica task = N agents = N
sessions, assembled into a single `AssembledDag`:

- dispatch via
  `MulticaEnvDispatchClient.create_env_dispatch(mode="scratch", dispatch_type="message", ...)`,
  then poll `MulticaDagClient.get_dag(handle)` until the DAG is assembled (`DagTimeout`
  rejects the episode; other fetch errors propagate);
- assemble via `SuperNodeAssembler.assemble_from_refs(dag, resolver)` into an
  `ExecutionDag`, then clean up every session in `dag.session_to_agent_run`
  (`DataProxySessionRemover.remove` -> `POST /export_trajectories` with
  `remove_session=true`).

AReaL **never calls `start_session`** on this path - Multica owns
`/rl/start_session(group_size=N)`, the `session_to_agent_run` binding, and the per-agent
credentials. The diagnosis agent's per-turn `step_rewards` (with `score_max`) arrive
inside the assembled DAG, not via a separate `set_reward` call; AReaL normalizes them
into per-segment `SuperNode.process_reward`. `SuperNodeAssembler` stamps the
`session_id` on every `SuperNode`, and `ExecutionDAG.session_map()` exposes
`{super_node_id: session_id}`.

> The older `run_swe_lego_issue` / `run_self_play` runners and the `_BranchDriver` /
> `EnvDispatchBranchDriver.drive_lane` branching seam referenced by earlier revisions of
> this doc have been removed. Branch execution via `mode="branch"` env-dispatch is not
> yet wired into the live workflow (the `BRANCH` edge type exists in `ExecutionDag`, but
> no driver calls `create_env_dispatch(mode="branch")`); see "Branch / Resume Lifecycle"
> for the intended protocol.

## RL Session Start and End

RL sessions are the control-plane link between Multica agent runs and AReaL training
trajectories. On the dispatch path **Multica owns** `start_session`: on a source agent's
first address it calls
`db_bridge.start_session(session_ref=<env-agent-binding-ID>, env_id)`, which the AReaL
bridge forwards to the AReaL rollout server. The server derives
`session_id = f"{session_ref}-{idx}"` from `canonical_session_ref` and returns
`session_id` + a scoped `api_key` (provider `areal`); a retry reuses the recorded
session (no second `start_session`). `session_ref` is the persistent
`environment_agent_sandbox` binding ID, not a task id - a session no longer requires a
synthetic or not-yet-inserted Multica task. Multica hands the `api_key` to the derived
agent's runtime, which uses it to call the AReaL-served model via db_bridge. AReaL
itself **never calls `start_session`** on this path; it reads each `session_id` from the
assembled DAG's `session_to_agent_run` map.

The verifier-side session seam (in `agents/reward/swe_lego_verifier.py`) is write-only:

```python
class _RlSession(Protocol):
    async def set_reward(self, *, session_id: str, reward: float) -> None: ...
```

System-level flow:

1. Multica creates the env-agent binding (status=pending) and, on first address, calls
   `db_bridge.start_session(session_ref=binding.ID, env_id)`.
1. The AReaL rollout server creates the session and returns `session_id` + `api_key`
   (provider `areal`); retry reuses the recorded session.
1. Multica passes the `api_key` to the derived agent's runtime; the agent calls the
   AReaL-served model via db_bridge.
1. `SuperNodeAssembler` stamps the `session_id` (from `dag.session_to_agent_run`) onto
   every `SuperNode` for that agent run; `ExecutionDAG.session_map()` exposes
   `{super_node_id: session_id}`.
1. After assembly, AReaL's `MultiAgentEnvDispatchWorkflow` cleans up each session via
   `DataProxySessionRemover.remove` (`POST /export_trajectories` with
   `remove_session=true`).

```mermaid
sequenceDiagram
    participant A as AReaL (MultiAgentEnvDispatchWorkflow)
    participant G as db_bridge
    participant M as Multica API
    participant R as Agent runtime (derived)
    participant D as ExecutionDAG/SuperNodes

    A->>M: POST /api/v1/env-dispatch (mode=scratch, dispatch_type=message, PAT)
    M->>G: start_session(session_ref=binding.ID, env_id)
    G->>A: forward (rollout server derives session_id)
    A-->>G: session_id + api_key (provider=areal)
    G-->>M: session_id + api_key (provider=areal)
    M->>R: start derived agent run with api_key (provider=areal)
    R->>G: model inference (api_key)
    G->>A: forward
    M-->>A: 200 rollouts[]; AReaL polls /dag -> AssembledDag
    A->>D: stamp session_id (session_to_agent_run) on SuperNode(s)
    D-->>A: session_map for verifier rewards
    A->>G: export_trajectories(session_id, remove_session=true) per session
```

Ending a session is driven by Multica's verifier agent, which calls db_bridge to write
the reward and then end or export the session. db_bridge forwards every call to the
AReaL runner; the AReaL-side Python finalizers enforce reward-before-export ordering on
the receiving end. Two paths are supported:

| Path             | Caller → mediator → callee                                           | Calls                                                                                              | When used                                                                        |
| ---------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Legacy finalizer | Multica verifier agent → db_bridge → AReaL (`RLSessionRewardWriter`) | `set_reward(session_id, reward)` then `end_session(session_id)`                                    | Backward-compatible single-verifier flow.                                        |
| Verifier harvest | Multica verifier agent → db_bridge → AReaL (`VerifierFinalizer`)     | `set_reward(session_id, reward)` for all sessions, then `export_trajectories(session_id)` for each | Preferred multi-agent verifier flow; export is terminal and revokes the session. |

The reward-before-end/export ordering is mandatory. If `set_reward` fails, the legacy
writer deliberately does **not** call `end_session`; the session stays open so the
caller can retry and avoid losing the trajectory. In the harvest flow, rewards for all
sessions are written first, then each session is exported exactly once.

The verifier agent extension can also call db_bridge directly:

| Verifier command                       | db_bridge call                         | Effect                                                         |
| -------------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `/rl/set_reward <session_id> <reward>` | `POST <db_bridge>/rl/set_reward`       | Writes or overwrites the reward for one session.               |
| `/export_trajectories <session_id>`    | `POST <db_bridge>/export_trajectories` | Exports the reward-stamped trajectory and revokes the session. |

The AReaL Python finalizer (`RLSessionRewardWriter` / `VerifierFinalizer`) remains
authoritative for ordering on the receiving side: it ensures rewards are persisted
before any export is acknowledged, even when the Multica verifier agent already issued
`/rl/set_reward` directly.

## ForkableEnvironment Seam (Explicit Injection Only)

`agents/environment.py` defines a vendor-neutral environment seam that remains for
scenarios where AReaL needs direct snapshot/fork/restore/cycle control — for example, an
explicitly injected provider used by the verifier or a future coordinator. The live
branch path does **not** flow through this seam; it uses env-dispatch exclusively.

```python
class ForkableEnvironment(Protocol):
    async def snapshot(self, sandbox_id: str) -> SnapshotResult: ...
    async def fork(self, *, source_sandbox_id: str | None = None, snapshot_id: str | None = None) -> ForkResult: ...
    async def restore(self, sandbox_id: str) -> None: ...
    async def cleanup(self, sandbox_id: str) -> None: ...
```

Two provider implementations currently exist:

| Provider                 | Endpoint prefix         | Intended use                                                                |
| ------------------------ | ----------------------- | --------------------------------------------------------------------------- |
| `FleetSandboxProvider`   | `/sandboxes/...`        | Generic Fleet cloud-runtime proxy. Uses `FLEET_BASE_URL` / `FLEET_API_KEY`. |
| `MulticaSweLegoProvider` | `/api/v1/sandboxes/...` | Multica cloud-runtime proxy. Uses `MULTICA_BASE_URL` / `MULTICA_API_KEY`.   |

The Multica-backed provider calls these endpoints:

| Operation | Endpoint                               | Request                                                        | Response                   |
| --------- | -------------------------------------- | -------------------------------------------------------------- | -------------------------- |
| Snapshot  | `POST /api/v1/sandboxes/{id}/snapshot` | Sandbox ID in path                                             | `{ "snapshot_id": "..." }` |
| Fork      | `POST /api/v1/sandboxes/fork`          | `{ "snapshot_id": "..." }` or `{ "source_sandbox_id": "..." }` | `{ "sandbox_id": "..." }`  |
| Restore   | `POST /api/v1/sandboxes/{id}/restore`  | Sandbox ID in path                                             | `200` or `204`             |
| Cleanup   | `DELETE /api/v1/sandboxes/{id}`        | Sandbox ID in path                                             | `200`, `204`, or `404`     |

`fork` requires exactly one source: either `snapshot_id` or `source_sandbox_id`. The
providers use a semaphore to cap concurrent forks; by default the cap comes from
`GROUP_SIZE`, falling back to `2`.

## Branch / Resume Lifecycle

Branching resumes work from a saved frontier. It is a single env-dispatch call
(`mode="branch"`); AReaL does not snapshot, fork, or replay messages itself. The
`BRANCH` edge type is defined in `ExecutionDag` and `mode="branch"` is accepted by the
client, but no coordinator is wired into the live `MultiAgentEnvDispatchWorkflow` to
issue branch dispatches yet - this section describes the intended protocol. (The earlier
`agents/branch_driver.py` / `EnvDispatchBranchDriver` / `_BranchDriver` / `drive_lane`
seam has been removed.)

```mermaid
sequenceDiagram
    participant A as AReaL branch selector
    participant M as Multica API
    participant S as Remote sandbox server

    A->>A: select branch point by entropy / TD gate (Node.env_id, Node.need_branch)

    A->>M: POST /api/v1/env-dispatch (mode=branch, env_id=source_env_id, group_size=1)
    M->>S: fork source sandbox server-side
    S-->>M: sandbox_id
    M->>M: copy issue/chat subtree server-side
    M->>S: start child agent run
    M-->>A: rollouts[0].env_id (child env_id, wraps sandbox_id)
```

Branch materialization is a single env-dispatch call:

1. **Select branch point**: AReaL picks a `Node` with `need_branch=True` (entropy / TD
   gate) — its `env_id` is the branch frontier. The same handle is mirrored on the
   `SuperNode.env_id` field by `SuperNodeAssembler`.
1. **Dispatch branch**: the coordinator calls
   `create_env_dispatch(mode="branch", env_id=<source_env_id>, group_size=1, ...)` with
   the same `domain` / `dispatch_type` / `agent_id` as the parent rollout.
1. **Server-side fork**: Multica forks the source sandbox, copies the issue/chat
   subtree, and starts a child agent run. AReaL does **not** snapshot, fork, replay
   messages, or drop `PriorSessionID` itself — those steps are server-side.
1. **Return child env_id**: AReaL reads `setup.rollouts[0].env_id` as the terminal
   `env_id` for the lane (the child env the branch produced).

`mode="resume"` resumes a saved checkpoint **in place**: `env_id` carries the checkpoint
id (not a source env). Multica resumes the same sandbox instances server-side and
returns a new `EnvDispatchHandle` for the continued dispatch. There is no separate
checkpoint-resume endpoint — AReaL expresses resume-from-checkpoint as a single
`create_env_dispatch(mode="resume", env_id=<checkpoint_id>)` call (the server's
`POST /api/v1/env-checkpoints/{checkpointID}/resume` may back this internally, but AReaL
does not call it directly). Resume is distinct from `branch`: `branch` forks a source
env into a child; `resume` continues the same sandbox instances from a saved point.

## Rollback and Cleanup Rules

Because branch materialization is a single env-dispatch call, partial-failure rollback
is owned by Multica server-side: the caller sees either a child `env_id` (the fork,
issue copy, and agent run all succeeded) or a non-2xx response (nothing was committed).
AReaL does not need to pair snapshot/ fork/ issue-fork/ start-branch rollback steps.

What AReaL owns is the outer rollout lifecycle, and it is structured for guaranteed
cleanup:

| Failure point                                                            | Rollback action                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `create_env_dispatch(mode="scratch", ...)` fails                         | Fail closed. A non-2xx response means nothing was allocated; raise. A `201` carrying a per-rollout `error` (`group_size > 1` partial success) means the successful lanes were already allocated server-side (best-effort `dispatchOne`, no rollback); the client reclaims the partial dispatch via the channel/project-scoped `DELETE` (best-effort - cleanup failures are swallowed so they cannot mask the rollout failure) before raising, so no half-provisioned dispatch leaks. |
| RL session start, branch drive, or verifier fails after scratch dispatch | The runner's `finally` block issues `DELETE /api/v1/env-dispatch/{projectID}` per rollout.                                                                                                                                                                                                                                                                                                                                                                                           |
| `set_reward` fails before export                                         | Legacy writer leaves the session open (no `end_session`); the caller retries. The cleanup still runs in `finally`.                                                                                                                                                                                                                                                                                                                                                                   |
| `export_trajectories` fails during harvest                               | The session is left un-exported; the failure is logged. Reward was already written. The cleanup still runs in `finally`.                                                                                                                                                                                                                                                                                                                                                             |

Cleanup is intentionally idempotent:

- Env-dispatch cleanup treats `404` as success.
- Sandbox cleanup (via `ForkableEnvironment`) treats `404` as success.
- Cleanup failures during the `finally` block are logged and not allowed to mask the
  original rollout failure.

## ID Flow Summary

| ID                     | Produced by                             | Consumed by                                                                                              | Purpose                                                                                                                                                                                       |
| ---------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env_id`               | Multica env/env-dispatch                | AReaL runners, branch driver, `SuperNode.env_id`, per-turn `Node.env_id`                                 | Canonical environment handle: base env, lane env, or branch child env. Branch dispatch takes `env_id=<source>` and returns a new child `env_id`.                                              |
| `sandbox_id`           | Remote sandbox server / Multica runtime | `ForkableEnvironment` only                                                                               | Live sandbox handle for snapshot/fork/restore/delete through explicit injection. Not used by the env-dispatch branch path.                                                                    |
| `snapshot_id`          | Remote sandbox server                   | `ForkableEnvironment.fork(snapshot_id=...)`                                                              | Durable save point for `ForkableEnvironment`-driven forks. Not produced by the env-dispatch branch path.                                                                                      |
| `project_id`           | Multica env-dispatch                    | Cleanup calls                                                                                            | Groups rollout resources for cascade deletion. One per rollout lane.                                                                                                                          |
| `issue_id`             | Multica env-dispatch                    | RL session start, verifier                                                                               | Identifies the issue subtree for the lane. Branch copies the subtree server-side; AReaL never forks issues itself.                                                                            |
| `agent_run_id`         | Multica agent runtime                   | DAG `session_to_agent_run`, verifier                                                                     | Identifies one agent execution lane.                                                                                                                                                          |
| `source_agent_id`      | Multica env-dispatch (roster)           | Derived-agent clone, credential invariant                                                                | The roster agent selected by the caller; template for the derived agent. Never rebound to an ephemeral runtime.                                                                               |
| `derived_agent_id`     | `CloneEnvDispatchAgentTx`               | Dispatch channel member, task enqueue, AC-6 cleanup                                                      | Global agent cloned from the source agent, permanently bound to the discovered runtime for the dispatch lifetime. Records `source_agent_id` lineage.                                          |
| `training_session_ref` | `environment_agent_sandbox` binding ID  | `ResolveEnvDispatchTrainingSession` → `db_bridge.start_session`                                          | Stable namespace for a training session (`session_ref`); the same session is reused on retry.                                                                                                 |
| `session_id`           | db_bridge (`start_session`)             | `MultiAgentEnvDispatchWorkflow` cleanup, `SuperNodeAssembler`, verifier/reward writer/trajectory harvest | Binds rewards and exported trajectories to one agent run. Multica creates it via db_bridge keyed by `session_ref=binding.ID`; AReaL reads it from the assembled DAG's `session_to_agent_run`. |
| `api_key`              | db_bridge (`start_session`)             | Multica → agent runtime                                                                                  | Scoped credential for one agent run to call the AReaL-served model via db_bridge (provider `areal`). Issued by db_bridge alongside `session_id`; not stored by AReaL.                         |

## Current Integration Status

The torch-free `agents/` modules define and test the Multica communication contracts,
the `MultiAgentEnvDispatchWorkflow` orchestrator (scratch message dispatch -> DAG poll
-> assemble -> cleanup), the env-dispatch branch protocol, and the verifier/harvest
finalizers. ARE-5 derived-agent provisioning (`envDispatchDerivedAgentEnabled`, default
`true`) is active on the scratch+message path: no pre-created runtime, shared sandbox
service, runtime discovery, derived-agent clone, and `session_ref`-keyed training
sessions. Branch execution via `mode="branch"` env-dispatch is defined (`BRANCH` edge
type, client `mode=branch`) but not yet wired into the live workflow. The shared
`SuperNode` store path is active: single-agent episodes are wrapped as leaf `SuperNode`s
before insertion into `MCTSTreeStore`, and per-turn `Node.env_id` is stamped from
backend metadata so the branch-frontier handle is available wherever a future
coordinator needs it.

## Env Checkpoint Semantics

Env checkpoint creation has two save modes.

`pause_in_place` (the default, and what every pre-existing checkpoint resolves to)
suspends the source sandbox instances and records no savepoint. Multica synchronously
waits for sandboxd stop / Cube pause up to the configured timeout and stores the project
subtree inline as JSONB. Resume returns the same sandbox instances and re-activates the
same task row, continuing the interrupted CLI session rather than starting cold.
`pause_in_place` rejects a lane count greater than one.

`snapshot` records an immutable savepoint per source instance (a `sandbox_snapshot` row
backed by a Cube snapshot template) and leaves every source instance **running**, with
its in-flight task undisturbed. Resume accepts a lane count and materializes that many
sandbox instances from the checkpoint's savepoint — **one snapshot per source instance,
not one per lane** — each with its own copied project subtree, its own agent runtime,
and a fresh CLI session. Every resume carries a lane key: repeating a key returns the
existing lane, while a new key expands the same frontier again without creating a second
checkpoint.

A savepoint is owned by exactly one checkpoint and is released when that checkpoint is
deleted, so it outlives its first use. Cube snapshots are memory-level
checkpoint/restore, not filesystem copies: a snapshot completes in ~1.2s, leaves the
source running, and a sandbox created from the resulting template comes up with the
source's processes still live.

### Endpoints

- `POST /api/v1/env-checkpoints` - create a checkpoint; body carries `project_id`,
  `event_ref`, `checkpoint_kind`, `env_id_map`, `sandbox_refs`, optional
  `entropy_score`, and optional `save_timeout_ms`. Returns 201 on synchronous save
  completion, 409 on save timeout/failure (non-resumable), 400 on validation error.
  Gated by `ENV_CHECKPOINTS_ENABLED` (disabled -> 404). AReaL wraps this in
  `create_checkpoint`; `403`/`404`/`409` raise
  `MulticaCheckpointError(status_code, body)` so the caller can distinguish a
  non-resumable checkpoint (409) from a missing/forbidden one (404/403).
- `GET /api/v1/env-checkpoints/{checkpointID}` - fetch a single checkpoint,
  workspace-scoped (cross-workspace -> 404).
- message: `GET /api/v1/channels/{channelID}/env-checkpoints`; issue:
  `GET /api/v1/projects/{projectID}/env-checkpoints` - list checkpoints for a dispatch,
  routed by `handle.dispatch_type`, newest first. The client (`list_checkpoints`) raises
  `MulticaCheckpointError` on `403`/`404`/`409`.
- `POST /api/v1/env-checkpoints/{checkpointID}/resume` - resume the saved sandbox
  instances; returns a `rollout_handle` AReaL uses to continue tree-search. Incomplete
  checkpoints (pending/timed_out/failed) -> 409. AReaL does **not** call this endpoint
  directly: resume-from-checkpoint is expressed as a single
  `create_env_dispatch(mode="resume", env_id=<checkpoint_id>)` call, which the server
  resolves into the resumed sandbox instances and returns a new `EnvDispatchHandle` for
  the continued dispatch.

### Save Statuses

`complete` (resumable), `timed_out` (non-resumable), `failed` (non-resumable), `pending`
(transient, set during synchronous save). Resume rejects anything other than `complete`.

### AReaL Entropy Gating

AReaL calls `should_create_entropy_checkpoint(entropy, threshold)` before hitting the
create API. When logprobs are unavailable (`entropy is None`) or no threshold is
configured, the checkpoint is skipped silently - the rollout continues without failing
or creating a spurious checkpoint. A skip never fails the rollout; checkpoint API errors
propagate as `MulticaCheckpointError` so the caller can decide whether to retry or
continue. `maybe_create_entropy_checkpoint` is the rollout-loop entry point; it creates
checkpoints with `checkpoint_kind="entropy_gated"` and forwards an optional
`save_timeout_ms`.

### Out of scope: forking inside a turn

A sandbox restored from a Cube snapshot carries live processes, including a mid-turn
agent — the prerequisite experiment observed the same PID and the same
append-in-progress log file continuing on the clone. Intra-turn forking is therefore
technically possible but deliberately out of scope, for two reasons:

- **Runtime identity.** N clones would share one `daemon.id` frozen into
  `~/.multica/daemon.id`, so each lane could re-register as the source sandbox's runtime
  and steal its row.
- **Duplicated in-flight requests.** Each clone would resume the source's in-flight
  model request, producing N duplicated calls for one logical turn.

Both are exactly what the `pkill` of the snapshot-restored daemon in
`buildStartRuntimeInCubeCode` avoids, which makes that `pkill` load-bearing correctness
rather than hygiene. Revisit only if intra-turn branch points prove valuable for tree
search.

## Channel-first message dispatch & branch collaboration

`dispatch_type="message"` (self_play, multi-agent chat) is **channel-first**: the
channel is the long-lived collaboration handle, and the dispatch lifecycle routes
through it. `dispatch_type="issue"` (swe_lego) stays **project-first**. AReaL never
infers which mode it is from the presence of an arbitrary string - it routes by the
`dispatch_type` carried on the handle returned at creation.

### EnvDispatchHandle

`create_env_dispatch` returns an immutable `EnvDispatchHandle`:

| Field           | Meaning                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `channel_id`    | Present for `dispatch_type="message"`; the primary handle. Absent for `dispatch_type="issue"`. |
| `project_id`    | Always present; the primary handle for `dispatch_type="issue"`.                                |
| `env_id`        | The lane env id from `rollouts[0].env_id` (empty if absent).                                   |
| `dispatch_type` | `"message"` or `"issue"`; drives all downstream routing.                                       |

`handle.primary_id` resolves to `channel_id` for message dispatches and `project_id` for
issue dispatches. Message responses are validated at the creation boundary: a
`dispatch_type="message"` response missing `channel_id` raises immediately rather than
producing a half-formed handle.

### Channel-first routes (Task 7)

Message dispatches expose their lifecycle on the channel; issue dispatches keep the
legacy project routes:

| Operation        | Message route                                       | Issue route                                        |
| ---------------- | --------------------------------------------------- | -------------------------------------------------- |
| DAG poll         | `GET /api/v1/env-dispatch/channels/{channelID}/dag` | `GET /api/v1/env-dispatch/{projectID}/dag`         |
| List checkpoints | `GET /api/v1/channels/{channelID}/env-checkpoints`  | `GET /api/v1/projects/{projectID}/env-checkpoints` |
| Cleanup          | `DELETE /api/v1/env-dispatch/channels/{channelID}`  | `DELETE /api/v1/env-dispatch/{projectID}`          |

`create_checkpoint` stays project-internal (it carries `project_id` in its body) and is
unchanged.

### Leader-only wake on branch

A `mode="branch"` message dispatch resumes a channel collaboration from a source trigger
without re-running the whole squad. Only the **trigger agent** is woken:

1. The branch source is validated pre-write: the source trigger, its source sandbox
   binding, and the channel message roster are resolved before any state is copied.
1. Channel history is copied transactionally into the branch (message ids remapped so
   the branch can append without mutating the source channel).
1. The trigger agent is provisioned from the **cloned source sandbox instance**
   (`CloneSandboxInstance` on the source binding's `sandbox_instance_id`), not a fresh
   sandbox and not a Fleet fork. The trigger is re-enqueued on the channel run.
1. Peer (non-trigger) agents are left **pending** - they are not provisioned at branch
   time. Their sandbox bindings are created in a pending state and materialized lazily
   only when the collaboration actually mentions them (see below).

This means a branch appends new message content without changing the trigger agent's
prior state beyond the cloned filesystem, and without paying the cost of booting peers
that may never participate.

### Lazy agent provisioning

Agents are not all provisioned at dispatch time. The dispatch records the
`environment_agent_sandbox` binding for each roster agent in a `pending` state; a
binding is provisioned only when the collaboration first addresses that agent (the
scratch leader's initial dispatch, a later mention, or an explicit turn handoff).
Provisioning uses a single-flight claim (`claimProvisioning`) so a
concurrently-addressed agent is provisioned exactly once; concurrent mentions observe
the winner and wait for the same terminal result.

On a **scratch** dispatch, first-address provisioning is the ARE-5 derived-agent flow:
`claimProvisioning` -> `validateEnvDispatchCredentialOwner` -> (training:
`ResolveEnvDispatchTrainingSession`) -> `EnvSandboxLifecycleService.Create` ->
`WaitForOnlineSandboxRuntime` -> `CloneEnvDispatchAgentTx` -> `markReady` (see the Phase
1-7 pipeline). On a **branch** dispatch the trigger is instead provisioned from the
cloned source sandbox instance (`CloneSandboxInstance` on the source binding's
`sandbox_instance_id`); see "Leader-only wake on branch".

### Branch errors

Branch validation is pre-write, so a malformed branch fails fast before any channel
history is copied or sandboxes are cloned:

| Condition                                                | Outcome                                                                                |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Source trigger / channel / message roster not resolvable | 4xx before any write; no branch project, channel copy, or clone is created.            |
| Source sandbox binding missing `sandbox_instance_id`     | Clone source cannot be resolved; the branch is rejected before provisioning.           |
| Clone of the source sandbox instance fails               | Branch provisioning fails; the trigger agent is not enqueued and the error propagates. |
| Branch message source fails shape validation             | `ValidateBranchMessageSource` rejects before `Dispatch` commits.                       |

Because the copy and clone are server-side and ordered after validation, a failed branch
leaves no orphaned channel copy or half-cloned trigger sandbox.

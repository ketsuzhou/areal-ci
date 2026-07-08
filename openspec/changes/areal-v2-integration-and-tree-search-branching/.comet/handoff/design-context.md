# Comet Design Handoff

- Change: areal-v2-integration-and-tree-search-branching
- Phase: design
- Mode: compact
- Context hash: 3f5e84fa04564799a675123b42df957e2b84b3c9212055fc4cff6af10ccb6729

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/areal-v2-integration-and-tree-search-branching/proposal.md

- Source: openspec/changes/areal-v2-integration-and-tree-search-branching/proposal.md
- Lines: 1-116
- SHA256: e42fc01f3376e1953414e999ac566ac7d5d87d2530d7f0cdec25112b63b58554

[TRUNCATED]

```md
## Why

`multica-v2-segment-dag-training` (in flight, current track) lands the v2 per-segment
trajectory data path: `close_segment` + per-segment tensor-ref export + `AssembledDag`
(segments / typed edges / `session_to_agent_run`) + AReaL ref-resolution into
`SuperNode` / `ExecutionDAG`. But two gaps remain before AReaL can actually *train* a
pi-agent from a Multica collaborative task under v2:

1. **Multi-agent coordination is not wired.** A Multica task is a *squad* of N
   collaborating agents, each needing its own session / trajectory. Today
   `TreeSearchGroupedRolloutWorkflow` produces single-agent *leaf* `SuperNode`s
   (`_wrap_leaf_super`: "multi-agent coordinator Phase 1b/2 not yet wired ... no DAG
   edges, no comm events"). One `self.workflow.arun_episode` call = one agent = one
   session. There is no path from "one task" to "N agent sessions assembled into a
   multi-agent DAG."
2. **Tree-search branching is absent on v2.** `choose_sample_source`
   (SCRATCH / BRANCH / MIXED) and `select_branch_candidate` exist for the legacy wired
   loop, but v2 has no branch primitive: branching from a segment's checkpoint requires
   opening a *new* session for the branched agent and recording branch provenance, which
   the `AssembledDag` contract (edges: `DELEGATION` / `MENTION` / `COMPLETION`) does not
   express.

This change wires both on the v2 online path: a multi-agent squad rollout (N
sessions/task) consumed as `AssembledDag` -> multi-agent `SuperNode`, plus tree-search
branching (`BRANCH` edge, branch-from-segment-checkpoint), so AReaL trains from Multica's
collaborative multi-agent tasks with trajectory diversity at trainable decision points.

## What Changes

- **Multi-agent env-dispatch base workflow (AReaL, primary).** A new multi-agent
  `RolloutWorkflow` wraps the squad: one `arun_episode` = one task = N agents = N
  sessions. Per call: `create_env_dispatch` (Multica creates the N-agent squad) ->
  `/rl/start_session(group_size=N)` mints N sessions on the v2 gateway (returns N
  `{session_id, api_key}`) -> credentials distributed to the N agents -> agents run the
  task (online mode) -> poll the env-dispatch endpoint -> `AssembledDag`. Returns the
  `AssembledDag` (not a single interaction dict).
- **Multica owns the session binding (unchanged contract).** `StartSessionRequest`
  carries no `agent_run_id`; the `session_id` <-> `agent_run_id` binding lives in
  Multica's `session_to_agent_run` and arrives in the `AssembledDag` (per
  `multica-v2-segment-dag-training` design §`session_to_agent_run -> run mapping`).
  AReaL never learns `agent_run_id` from `start_session`.
- **`TreeSearchGroupedRolloutWorkflow` multi-agent wiring (AReaL, primary).** Approach
  **A** (base workflow is multi-agent; grouped workflow stays group-level):
  - `_result_to_nodes` branches on result type: single-agent interaction dict (unchanged)
    vs `AssembledDag` (per-segment turn-Nodes, one `episode_id` per agent-run).
  - `_wrap_leaf_super` (leaf SuperNode, `agent_id=""`, no edges) generalizes to a
    multi-agent `SuperNode` builder: N agent-runs + typed edges from `AssembledDag`.
    Leaf is the degenerate N=1 case.
  - `multica_dag_client` hook (stored today, unused) activated.
  - `group_size=M` unchanged in shape: `asyncio.gather` over M `_run_fresh_episode`
    calls, each now returning a multi-agent `AssembledDag` instead of one trajectory.
- **Tree-search branching (AReaL + Multica).** A branch opens a *new*
  `/rl/start_session` for the branched agent(s) from a closed segment's checkpoint and
  records a **`BRANCH` edge** in the `AssembledDag` (provenance: which
  `segment_id`/checkpoint the branch forked from). `choose_sample_source`
  (SCRATCH / BRANCH / MIXED) and `select_branch_candidate` extend to the multi-agent
  setting; `max_group_size` bounds total branch count per query.
- **v2 online training integration (AReaL).** `InferenceServiceWorkflow` online mode
  (`agent=None` -> `_run_online` -> `wait_for_online_trajectory`) consumes ready
  trajectories via the controller callback server. The staleness manager gates *trainer
  batch admission*, not session minting; Multica mints sessions on demand
  (`start_session(group_size=N)`, no pre-grant / no capacity ratchet - v2
  `SessionStore.start_session` neither checks nor decrements `_capacity`). Excess ready
  trajectories backpressure via `_completed_online_results`.
- **Session lifecycle.** M groups x N agents = M x N sessions minted dynamically per
  batch; harvested via `/export_trajectories` (`remove_session=True`); `close_segment`
  per communication-bounded segment (v2-segment-dag).

## Capabilities

### New Capabilities

- `v2-multi-agent-tree-search`: Multi-agent squad rollout on the v2 online path - one
  task = N agents = N sessions, assembled into a multi-agent `SuperNode` / `ExecutionDAG`
  from the `AssembledDag` (consuming `session_to_agent_run` + typed edges) - plus
  tree-search branching where a branch opens a new session from a closed segment's
  checkpoint and records a `BRANCH` edge. Extends the v2-segment-dag data path with
  multi-agent coordination and branch-from-segment exploration.

### Modified Capabilities
```

Full source: openspec/changes/areal-v2-integration-and-tree-search-branching/proposal.md

## openspec/changes/areal-v2-integration-and-tree-search-branching/design.md

- Source: openspec/changes/areal-v2-integration-and-tree-search-branching/design.md
- Lines: 1-200
- SHA256: 6edbc9dea50989441ae9e9b3d8a24c95469cfba197b47ad6bd5c0782191f100d

[TRUNCATED]

```md
# Design - areal-v2-integration-and-tree-search-branching

## 1. Context

Builds on `multica-v2-segment-dag-training` (v2 per-segment data path: `close_segment`,
`AssembledDag`, `session_to_agent_run`, ref-resolution into `SuperNode`/`ExecutionDAG`).
This change adds (a) multi-agent squad coordination and (b) tree-search branching on the
v2 online training path, plus (c) v2 online integration. Reference design doc:
`docs/superpowers/specs/2026-07-08-multica-v2-segment-dag-design.md`.

## 2. v2 online training path (the substrate)

The v2 `inference_service` controller has two modes (`controller.py:786`):

```
external_mode = (config.api_url is not None)
  offline  : controller drives an agent (_run_offline)
  online   : external agents drive; controller waits for ready trajectories
```

Online mode (`InferenceServiceWorkflow(agent=None)`, `controller/workflow.py:230`):

```
arun_episode -> _run_online -> controller.wait_for_online_trajectory()  [blocks on future]
                                     ^ future resolved by callback server
                                     |
   data_proxy POSTs /callback/online_ready {session_id, trajectory_id}  (controller.py:904)
                                     |
   _export_interactions([session_id], trajectory_id) -> traj dict       (remove_session=True)
   return traj  -> PPO trainer
```

The controller is **trajectory-source-agnostic**: `wait_for_online_trajectory` consumes
any ready trajectory regardless of SCRATCH vs BRANCH origin. This is the key enabler -
branching is "Multica produces more ready trajectories," needing no special AReaL-side
branch harvest path.

Training loop: PPO trainer -> `controller.rollout_batch(N)` -> collect N trajectories ->
`/ppo/actor/compute_advantages` -> `/update_weights` (training_service) ->
`controller.set_version(v+1)`.

## 3. Key decisions

### D1 - Multica owns `/rl/start_session` (online / external-user path)

**Decision**: Multica's agents call `/rl/start_session` on the v2 gateway themselves;
AReaL does **not** call `rl_session.start(agent_run_id=...)` (that line in
`self_play_runner.py:100` is the legacy AReaL-orchestrated pattern, retired for this
path).

**Why**: aligns with the v2-segment-dag contract where Multica owns
`session_to_agent_run` and delivers it in the `AssembledDag`. `StartSessionRequest`
carries only `{task_id, api_key?, group_size}` - no `agent_run_id` - so the binding is
Multica-side by construction.

**Alternative rejected**: AReaL orchestrates (`self_play_runner` pattern). Would require
AReaL to drive the squad and own the binding, inverting the v2 contract and re-introducing
the AReaL<->Multica coupling the v2 track removed.

### D2 - Approach A: base workflow is multi-agent; grouped workflow stays group-level

**Decision**: a new multi-agent `RolloutWorkflow` is the `self.workflow` wrapped by
`TreeSearchGroupedRolloutWorkflow`. One `self.workflow.arun_episode` call = one task =
N agents = N sessions = one `AssembledDag`. The grouped workflow stays group-level:
`asyncio.gather` over M `_run_fresh_episode` calls (shape unchanged, `:1496`), each
returning a multi-agent `AssembledDag`.

**Why**: matches the code's stated intent (`customized_grouped_workflow.py:19` "env-dispatch
runner model"; `:95` "multi-agent coordinator Phase 1b/2 not yet wired"). Keeps multica
squad/edge/env-dispatch coordination out of the grouped workflow.

**Alternative rejected (B)**: grouped workflow calls `self.workflow.arun_episode` N times
per group and assembles the squad itself. Leaks squad/edge/session_to_agent_run
coordination into the grouped workflow; balloons `_wrap_leaf_super`.

### D3 - v2 mint-on-demand; no pre-grant / no capacity ratchet

**Decision**: sessions are minted dynamically via `start_session(group_size=N)`. No
pre-instantiation of M x N proxy workers, no ready-worker queue, no `grant_capacity`
pre-grant.
```

Full source: openspec/changes/areal-v2-integration-and-tree-search-branching/design.md

## openspec/changes/areal-v2-integration-and-tree-search-branching/tasks.md

- Source: openspec/changes/areal-v2-integration-and-tree-search-branching/tasks.md
- Lines: 1-114
- SHA256: 6ee3083c025b1225edf37b527eece914df7dc14966ffaff6bb156a95e1d0cc2a

[TRUNCATED]

```md
# Tasks - areal-v2-integration-and-tree-search-branching

Phased. Each task lists primary files. Check off on completion. Depends on
`multica-v2-segment-dag-training` (AssembledDag / close_segment / session_to_agent_run)
having landed; Sub-project F (checkpoint-fork) for branch primitives.

## Phase 1 - Multi-agent env-dispatch base workflow

- [ ] 1.1 Define `MultiAgentEnvDispatchWorkflow(RolloutWorkflow)` skeleton
      (`customized_areal/tree_search/agents/multi_agent_env_dispatch.py`): `__init__`
      (gateway_addr, admin_api_key, multica_dag_client, group_size, timeout, discount,
      export_style); `arun_episode(engine, data)` returning an `AssembledDag`.
- [ ] 1.2 `create_env_dispatch` call: drive Multica to create the N-agent squad for the
      task; obtain N `agent_run_id`s + `env_id`/`project_id` per rollout. Reuse the
      `_MulticaClient` protocol shape from `self_play_runner.py:37`.
- [ ] 1.3 `start_session(group_size=N)` against the v2 gateway
      (`/rl/start_session`, `controller/workflow.py:67` `_start_session` is the reference
      client): mint N sessions, receive N x `{session_id, session_api_key}`. Confirm
      Multica records `session_to_agent_run` (AReaL does not).
- [ ] 1.4 Credential distribution: return the N `api_key`s to Multica (or expose via the
      env-dispatch response) so each agent runs `chat/completions` with its own key. Define
      the AReaL->Multica handoff contract.
- [ ] 1.5 Ready-trajectory synchronization: await all N sessions' ready trajectories for
      one task (each session callbacks `/callback/online_ready` on the controller; the base
      workflow correlates by `group_id` / `session_to_agent_run`) before assembling.
- [ ] 1.6 `AssembledDag` polling: poll the env-dispatch endpoint (202 in-progress / 200 +
      `AssembledDag` done) via `multica_dag_client`; return the `AssembledDag`.
- [ ] 1.7 Export + cleanup: per-session `/export_trajectories` (`remove_session=True`)
      for tensor refs after the DAG is assembled; `DELETE /data/clear` + `remove_session`
      post-training (v2-segment-dag lifecycle).

## Phase 2 - `TreeSearchGroupedRolloutWorkflow` multi-agent wiring (Approach A)

- [ ] 2.1 Generalize `_result_to_nodes` (`customized_grouped_workflow.py:929`): branch on
      result type. `AssembledDag` path -> per-segment turn-Nodes via
      `interactions_dict_to_nodes`, one `episode_id` per `agent_run_id` (not per group).
      Single-agent dict path unchanged.
- [ ] 2.2 Replace `_wrap_leaf_super` (`:91`) with a multi-agent `SuperNode` builder: N
      agent-runs (`agent_id`, `issue_id` populated from `AssembledDag` segments) + typed
      edges (`DELEGATION`/`MENTION`/`COMPLETION`/`BRANCH`). Leaf SuperNode = degenerate
      N=1 case (preserve single-agent tests).
- [ ] 2.3 Activate `multica_dag_client` hook (`:682`/`:852`): wire it into the
      `MultiAgentEnvDispatchWorkflow` constructed as `self.workflow` for the multica path.
- [ ] 2.4 `_finalize_episode` (`:1754`): per-agent-run credit assignment at fan-in joins
      (decision 8 of the master design); aggregate multi-agent SuperNodes. Verify
      `TreeAdvantageComputer` consumes per-node credit across the DAG.
- [ ] 2.5 Wire `MultiAgentEnvDispatchWorkflow` as `self.workflow` in the grouped workflow
      constructor for the v2/multica config path; keep `OpenAIProxyWorkflow`/
      `InferenceServiceWorkflow` for non-multica paths.
- [ ] 2.6 `group_size=M` verification: confirm `asyncio.gather` over M
      `_run_fresh_episode` (`:1496`) still produces M multi-agent SuperNodes (one per
      squad rollout); cached-episode path (`load_untrained_episodes`) handles multi-agent
      SuperNodes.

## Phase 3 - Tree-search branching

- [ ] 3.1 `BRANCH` edge type: add to `EdgeType` (`execution_dag.py`) and the `AssembledDag`
      edge contract; carry provenance (`branch_from_segment_id`,
      `branch_from_checkpoint_id`).
- [ ] 3.2 Branch-from-segment-checkpoint: when `select_branch_candidate`
      (`customized_grouped_workflow.py:233`) returns a node, open a new
      `start_session(group_size=k)` for the branched agent(s) from the closed segment's
      checkpoint (Sub-project F fork primitive); run; close new segments.
- [ ] 3.3 `BRANCH` edge emission: Multica records `BRANCH` edges linking branched segments
      to the parent segment in the `AssembledDag`; `select_branch_candidate` /
      `choose_sample_source` (SCRATCH/BRANCH/MIXED, `:195`) extended to the multi-agent
      setting.
- [ ] 3.4 Branch advantage backup: define the backup rule across `BRANCH` edges (R1);
      extend `backup.py` structural backup to distribute reward along `BRANCH` provenance.
- [ ] 3.5 `max_group_size` bound: ensure total branch count per query <=
      `max_group_size - initial_group_size` (`:654`); consecutive-failure circuit breaker
      (`:28`) applies to multi-agent branches.
- [ ] 3.6 Branch cleanup: extend `_cleanup_branch` to release branched sessions
      (`remove_session`) + forked env/issue subtree (Sub-project F).

## Phase 4 - v2 online training integration

- [ ] 4.1 Online mode wiring: construct `InferenceServiceWorkflow(controller, agent=None,
      ...)` for the v2 online path so `arun_episode` -> `_run_online`
      (`controller/workflow.py:230`) -> `wait_for_online_trajectory`.
```

Full source: openspec/changes/areal-v2-integration-and-tree-search-branching/tasks.md

## openspec/changes/areal-v2-integration-and-tree-search-branching/specs/v2-multi-agent-tree-search/spec.md

- Source: openspec/changes/areal-v2-integration-and-tree-search-branching/specs/v2-multi-agent-tree-search/spec.md
- Lines: 1-136
- SHA256: f7c6084088ebde39def02707048677ec2a581401137a6a5c7dead5b3b2beb573

[TRUNCATED]

```md
# v2-multi-agent-tree-search

## ADDED Requirements

### Requirement: Multi-agent squad rollout produces an AssembledDag per task

AReaL SHALL provide a multi-agent `RolloutWorkflow` (`MultiAgentEnvDispatchWorkflow`) where
one `arun_episode` drives one Multica task over a squad of N collaborating agents. Per call
it SHALL `create_env_dispatch` (Multica creates N `agent_run_id`s), let the N agents run
online (Multica owns `/rl/start_session(group_size=N)` and the `session_to_agent_run`
binding), poll the env-dispatch endpoint, and return the `AssembledDag` (segments / edges /
`session_to_agent_run`). AReaL MUST NOT call `rl_session.start(agent_run_id=...)` on this
path - Multica owns session opening.

#### Scenario: One task drives N sessions
- **WHEN** `MultiAgentEnvDispatchWorkflow.arun_episode` runs for an N-agent task
- **THEN** Multica creates N agent runs, the N agents each open a session via
  `/rl/start_session`, and the workflow returns one `AssembledDag` covering all N runs

#### Scenario: AReaL does not open sessions on the multi-agent path
- **WHEN** the multi-agent path is active
- **THEN** AReaL never calls `rl_session.start(agent_run_id=...)`; the `session_id` to
  `agent_run_id` binding arrives only in `AssembledDag.session_to_agent_run`

#### Scenario: N=1 degenerates to single-agent
- **WHEN** a task has a single agent (N=1)
- **THEN** the workflow produces a leaf `SuperNode` with no DAG edges, identical to the
  existing single-agent path

### Requirement: AssembledDag resolves into a multi-agent SuperNode

`TreeSearchGroupedRolloutWorkflow._result_to_nodes` SHALL branch on result type: a
single-agent interaction dict (unchanged) vs an `AssembledDag`. For an `AssembledDag` it
SHALL produce per-segment turn-`Node`s with one `episode_id` per `agent_run_id`, and a
multi-agent `SuperNode` carrying N agent-runs plus typed edges from the `AssembledDag`. The
leaf `SuperNode` (single-agent, no edges) MUST remain the degenerate N=1 case.

#### Scenario: AssembledDag yields per-agent-run Nodes
- **WHEN** `_result_to_nodes` receives an `AssembledDag`
- **THEN** each segment's interactions become turn-`Node`s stamped with one `episode_id`
  per `agent_run_id`, and the assembled `SuperNode` carries all N agent-runs

#### Scenario: Edges populate the SuperNode
- **WHEN** the `AssembledDag` carries `delegation` / `mention` / `completion` / `branch` edges
- **THEN** the multi-agent `SuperNode` records those edges and the `ExecutionDAG` topology
  matches Multica's recorded structure

### Requirement: BRANCH edge type with fork provenance

The `AssembledDag` edge contract SHALL be extended with a `branch` edge type (in addition
to `delegation` / `mention` / `completion`). A `branch` edge MUST carry
`branch_from_segment_id` and `branch_from_checkpoint_id` provenance identifying the closed
segment and checkpoint the branch forked from. Branching SHALL fork the env + issue subtree
via Sub-project F's checkpoint-fork primitive and open a new `/rl/start_session` for the
branched agent(s).

#### Scenario: Branch records provenance
- **WHEN** a branch forks from a closed segment's checkpoint
- **THEN** the `AssembledDag` gains a `branch` edge carrying `branch_from_segment_id` and
  `branch_from_checkpoint_id`, and the branched agent runs under a new session

#### Scenario: Branch uses Sub-project F fork
- **WHEN** `select_branch_candidate` selects a branch point
- **THEN** the fork restores env snapshot + issue subtree via F's checkpoint-fork primitive
  before the branched agent runs

### Requirement: MCTS value backup across BRANCH edges

Advantage backup SHALL propagate a branch's terminal return along its `branch` edge to the
parent segment's checkpoint node (MCTS-style), updating the parent's value estimate from
branch outcomes. This extends the existing structural backup (`backup.py`) that distributes
terminal reward along `delegation` / `mention` / `completion` edges. Each branch trajectory
SHALL also retain its own advantage for policy-gradient training.

#### Scenario: Branch return backs up to parent checkpoint
- **WHEN** a branch trajectory completes with a terminal return
- **THEN** the return propagates along the `branch` edge to the parent segment's checkpoint
  node, updating that node's value estimate

#### Scenario: Branch keeps its own advantage
```

Full source: openspec/changes/areal-v2-integration-and-tree-search-branching/specs/v2-multi-agent-tree-search/spec.md


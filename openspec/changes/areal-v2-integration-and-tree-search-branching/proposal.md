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

<!-- None yet shipped. This change extends the `AssembledDag` contract from
     `multica-v2-segment-dag-training` (not yet archived -> not in openspec/specs/) with
     a `BRANCH` edge type and branch-from-segment semantics. Once v2-segment-dag archives,
     that extension should be captured as a delta to the `v2-segment-dag` spec; until then
     it lives in the new `v2-multi-agent-tree-search` capability. Existing shipped specs
     (`critic-driven-training-signal`, `training-session-lifecycle`) are consumed
     unchanged. -->

## Impact

- **AReaL `customized_areal/tree_search/core/customized_grouped_workflow.py` (primary)**:
  `_result_to_nodes` (AssembledDag branch), `_wrap_leaf_super` -> multi-agent SuperNode
  builder, `multica_dag_client` hook activation, `_finalize_episode` per-agent-run
  credit/advantage. `group_size` loop shape unchanged.
- **AReaL multi-agent base workflow (primary, new)**: new `RolloutWorkflow`
  (`customized_areal/tree_search/agents/`) driving `create_env_dispatch` +
  `start_session(group_size=N)` + `AssembledDag` polling. Replaces the single-agent
  `OpenAIProxyWorkflow`/`InferenceServiceWorkflow` as `self.workflow` for the multica
  path.
- **AReaL v2 inference_service (secondary)**: `InferenceServiceWorkflow` online mode
  (`controller/workflow.py`) is the harvest path; no change to `start_session` /
  `close_segment` / `export_trajectories` semantics (those are v2-segment-dag's). Staleness
  manager (`controller.py`) gates batch admission as today.
- **Multica (Go, secondary)**: branch-from-segment-checkpoint (fork env + issue subtree,
  overlaps Sub-project F), `BRANCH` edge emission in `AssembledDag`, per-branch
  `start_session` calls. `session_to_agent_run` already owned by Multica.
- **Depends on**: `multica-v2-segment-dag-training` (AssembledDag, close_segment,
  session_to_agent_run, ref-resolution) landing first; Sub-project F
  (per-agent-env-and-checkpointing) for checkpoint-fork primitives used by
  branch-from-segment.
- **Out of scope**: judge / process-reward scoring (separate v2 change 2a); actor-model
  V critic + GAE (v2 change 2b); full sandbox snapshot/fork internals (Sub-project F
  owns); legacy `areal/experimental/openai/proxy/` path (superseded by v2 - the
  `grant_capacity` ratchet / ready-worker model analyzed during exploration does not apply
  to v2's mint-on-demand `SessionStore`).

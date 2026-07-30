---
comet_change: areal-v2-integration-and-tree-search-branching
role: technical-design
canonical_spec: openspec
---

# AReaL v2 Multi-Agent Integration + Tree-Search Branching - Technical Design

This is the deep technical design for OpenSpec change
`areal-v2-integration-and-tree-search-branching`. The OpenSpec artifacts
(`proposal.md` / `design.md` / `tasks.md` / `specs/v2-multi-agent-tree-search/spec.md`) are
the upstream source of truth for requirements; this doc covers implementation approach,
technical risks, testing strategy, and boundary conditions. It builds on
`docs/superpowers/specs/2026-07-08-multica-v2-segment-dag-design.md` (the v2 per-segment
data path) and the master `2026-06-26-multica-dag-rl-design.md`.

## 1. Problem

`multica-v2-segment-dag-training` lands the v2 data path (`close_segment`, `AssembledDag`,
`session_to_agent_run`, ref-resolution) but two things are missing to train a pi-agent from
a Multica collaborative task: (a) multi-agent squad coordination - today
`TreeSearchGroupedRolloutWorkflow` produces single-agent leaf `SuperNode`s (the multi-agent
coordinator is "not yet wired"); (b) tree-search branching - no branch primitive on v2, and
the `AssembledDag` edge contract has no `BRANCH` type.

## 2. Resolved design decisions

| Decision | Resolution | Rationale |
|---|---|---|
| Session opener | Multica owns `/rl/start_session` | v2-segment-dag contract: Multica owns `session_to_agent_run`; `StartSessionRequest` carries no `agent_run_id` |
| Multi-agent placement | Approach A: base workflow multi-agent; grouped workflow stays group-level | Matches code's stated "env-dispatch runner model" intent; keeps squad logic out of the grouped workflow |
| Capacity | v2 mint-on-demand; no pre-grant / no ratchet | v2 `SessionStore.start_session` neither checks nor decrements `_capacity`; legacy proxy model does not apply |
| Staleness | Gates trainer batch admission, not minting | `staleness_manager.get_capacity()` is the bound; gateway `start_session` doesn't consult it |
| R1 - BRANCH backup | MCTS-style value propagation | Branch return backs up along `BRANCH` edge to parent checkpoint node; consistent with existing structural backup |
| Harvest | Direct `AssembledDag` polling (202->200) | Matches v2-segment-dag contract; controller awaits `arun_episode`; callback stays single-agent-only |
| R4 - branch checkpoint | F-independent via `EnvDispatchBranchDriver` | Branch fork uses existing `create_env_dispatch(mode="branch")` (Multica env-dispatch); Sub-project F sandbox snapshot/fork is out of scope; `env_snapshot` is refs-only |
| R2 - within-step staleness | Batch boundary guarantee | `rollout_batch` awaits all `arun_episode`s before `set_version`; no cross-step leak |
| R5 - backpressure | Staleness manager is the bound | Excess `AssembledDag`s hold at Multica's 202 until polled |
| Partial squad failure | Drop the task | Matches `_run_offline` group-abandon (`workflow.py:214-222`); no partial `SuperNode` |

## 3. Components

### 3.1 `MultiAgentEnvDispatchWorkflow` (new base `RolloutWorkflow`)

`customized_areal/tree_search/agents/multi_agent_env_dispatch.py`. One `arun_episode` =
one Multica task:

1. `create_env_dispatch` -> Multica creates the N-agent squad (N `agent_run_id`s,
   `env_id`/`project_id`). Reuse the `_MulticaClient` protocol shape from
   `self_play_runner.py:37`.
2. N agents run online - Multica owns `/rl/start_session(group_size=N)` (mints N sessions,
   returns N `{session_id, api_key}`) and records `session_to_agent_run`. AReaL distributes
   nothing; Multica hands credentials to its agents.
3. Poll `GET /api/v1/env-dispatch/{projectID}/dag` -> `202` in-progress / `200` +
   `AssembledDag` done. Configurable poll interval + timeout; on timeout, reject the
   trajectory (R-partial-failure).
4. `SuperNodeAssembler.assemble_from_refs(dag, DataProxyTensorResolver)` (existing, sync via
   `asyncio.to_thread`) -> resolves each segment's `tensor_ref` via `/data/*` and builds the
   `ExecutionDAG[SuperNode]` (one `SuperNode` per segment + typed edges). **Reuses the
   existing assembler - not reimplemented.**
5. Return `{"assembled_dag", "execution_dag"}` (consumed by `_result_to_nodes`'s multica
   branch). Success-path cleanup: `DataProxyTensorResolver.clear` + `DataProxySessionRemover.remove`.

**Boundary**: N=1 degenerates to the single-agent leaf path. The workflow is a *thin
orchestrator* over existing v2-segment-dag components (`MulticaEnvDispatchClient`,
`MulticaDagClient`, `SuperNodeAssembler`, `DataProxyTensorResolver`/`SessionRemover`) - it
does NOT reimplement dispatch/polling/assembly. AReaL NEVER calls `start_session` (Multica
owns it). Sync calls run in `asyncio.to_thread` so M parallel episodes stay concurrent.

**Interface contract**: `arun_episode(engine, data) -> {"assembled_dag", "execution_dag"} | None`
(`None` on drop/timeout/partial-squad). Internal polling/credential mechanics are private.

### 3.2 `TreeSearchGroupedRolloutWorkflow` multi-agent wiring (Approach B: SuperNode-preserving)

`customized_areal/tree_search/core/customized_grouped_workflow.py`:

- `_result_to_nodes` (`:929`): branch on result type. Single-agent interaction dict ->
  unchanged. Multica result (`{"assembled_dag", "execution_dag"}`) -> return the
  `ExecutionDAG`'s `SuperNode`s **without flattening to `Node`s** (the multi-segment edge
  structure must survive to `_finalize_episode`). Return type widens to
  `list[Node] | list[SuperNode] | None`.
- `_finalize_episode` (`:1754`): multica branch inserts the multi-`SuperNode` `ExecutionDAG`
  via `tree_store.insert_super_batch` (NOT `_wrap_leaf_super`) and computes advantages over
  the DAG (`TreeAdvantageComputer` if it handles `SuperNode`s, else `assemble_node_advantages`
  over `topological_order()`). N=1 multica = leaf `SuperNode` (parity). Judge/distillation
  stay Change 2.
- `multica_dag_client` hook (`:682`/`:852`): activated; `self.workflow` =
  `MultiAgentEnvDispatchWorkflow` (constructed with the dispatch/dag clients +
  resolver/session_remover/assembler) for the multica path.
- `_arun_episode_fixed` gather (`:1496`): shape unchanged - M parallel `_run_fresh_episode`
  calls, each returning a multica `{"assembled_dag", "execution_dag"}`.

**Non-multica paths** (`OpenAIProxyWorkflow` / `InferenceServiceWorkflow`) remain unchanged;
the multi-agent base workflow is selected only for the v2/multica config.

### 3.3 Branching (F-independent, via `EnvDispatchBranchDriver`)

- `EdgeType` (`execution_dag.py`): add `BRANCH` + `Edge` provenance
  (`branch_from_segment_id`, `branch_from_checkpoint_id`).
- `AssembledDag` edge contract: Multica emits `branch` edges;
  `SuperNodeAssembler.assemble_from_refs` maps them to `EdgeType.BRANCH` + provenance.
- Branch *execution* lives in `MultiAgentEnvDispatchWorkflow` (the env-dispatch runner model,
  per `customized_grouped_workflow.py:1416`): a branch `arun_episode` forks via the existing
  `EnvDispatchBranchDriver.drive_lane` (`create_env_dispatch(mode="branch", env_id=<source>)`),
  runs the branched squad, harvests the branched `AssembledDag` with a `branch` edge.
  **F-independent** - no Sub-project F sandbox snapshot/fork; `env_snapshot` is refs-only.
- Branch *selection/backup/bounds* live in the grouped workflow/`tree_store`:
  `select_branch_candidate` (`:233`) picks the branch point; `backup.py` (new) propagates
  branch return along `BRANCH` edges to the parent checkpoint `Node` (MCTS value update via
  `Node.visit_count`); each branch retains its own advantage; `max_group_size` bounds total
  branches per query (`max_group_size - initial_group_size`); consecutive-failure circuit
  breaker applies.

## 4. Data flow

```
PPO trainer -> controller.rollout_batch(M) -> submit M arun_episode (grouped workflow)
  per arun_episode:
    _arun_episode_fixed: asyncio.gather(M x _run_fresh_episode)        (:1496)
      each -> self.workflow.arun_episode = MultiAgentEnvDispatchWorkflow
        create_env_dispatch(scratch, N agents)  [Multica owns start_session + creds]
        asyncio.to_thread(get_dag) -> AssembledDag{segments, edges(+branch), session_to_agent_run}
        asyncio.to_thread(assemble_from_refs, dag, resolver) -> ExecutionDAG[SuperNode]
        cleanup: resolver.clear + session_remover.remove
        return {assembled_dag, execution_dag}
    _result_to_nodes -> list[SuperNode] (preserved, not flattened)                 (:929)
  _finalize_episode: insert_super_batch(SuperNodes) + DAG advantages + MCTS backup  (:1754)
-> compute_advantages -> update_weights -> set_version(v+1)
```

The controller is trajectory-source-agnostic: `rollout_batch` awaits the M `arun_episode`
returns; it does not distinguish SCRATCH vs BRANCH origin. Staleness gates admission at
this batch boundary.

## 5. Boundary conditions

- **N=1**: leaf `SuperNode`, no edges; parity with single-agent path (test asserts).
- **Branch failure**: branch `start_session`/fork/export failure -> branch dropped, logged;
  `max_group_size` circuit breaker; the parent task continues with other samples.
- **Partial squad failure**: any of N agents fails -> task dropped, no trajectory, no
  partial `SuperNode` (R-partial-failure).
- **Polling timeout**: no `200` within timeout -> trajectory rejected.
- **Cross-step staleness (R2)**: `rollout_batch` awaits all `arun_episode`s before
  `set_version`; branches minted within a batch are harvested within the same step. No
  branch trajectory leaks across a weight update.
- **Unbounded minting (R5)**: `SessionStore.start_session` is unbounded; the operative
  bound is `staleness_manager.max_concurrent_rollouts` (batch admission) + inference-engine
  throughput. Excess `AssembledDag`s hold at Multica's `202`.

## 6. Technical risks

- **R1 backup complexity**: MCTS value aggregation at parent checkpoint nodes (discount,
  visit counting) over deep branch trees. *Mitigation*: `max_group_size` bounds depth/width;
  unit-test backup over a multi-level branch tree.
- **F independence**: branching uses `EnvDispatchBranchDriver` (`create_env_dispatch(mode="branch")`),
  not Sub-project F's sandbox snapshot/fork; all phases ship together. Sub-project F remains
  out of scope for full sandbox snapshot/fork.
- **AssembledDag polling latency**: long tasks delay `arun_episode` return. *Mitigation*:
  configurable timeout; staleness gates batch not individual poll latency.
- **`BRANCH` edge delta to `v2-segment-dag`**: extends the shipped `AssembledDag` contract.
  *Mitigation*: until `multica-v2-segment-dag-training` archives, `BRANCH` lives in the
  `v2-multi-agent-tree-search` spec; on archive, capture as a delta to `v2-segment-dag`.

## 7. Testing strategy

- **Unit** (no real engine): `MultiAgentEnvDispatchWorkflow` with fake Multica + fake v2
  gateway; `_result_to_nodes` `AssembledDag` branch (per-agent-run Nodes + multi-agent
  `SuperNode` + `BRANCH` edges); MCTS backup over a multi-level branch tree;
  `max_group_size` bound + circuit breaker; N=1 parity; partial-squad-failure drop;
  polling-timeout reject.
- **Integration**: M x N parallel sessions on the v2 gateway; router routes by
  `session_key`; no `429` from `SessionStore` (minting unbounded); staleness backpressure
  (`_completed_online_results` queueing when Multica outpaces the trainer).
- **E2E (cloud-only)**: N=2 squad, `group_size=2`, SCRATCH; then a `BRANCH` from a closed
  segment. Verify harvest + advantage + one weight update. Skip with explanation when
  multi-node hardware unavailable (per `backend/areal/CLAUDE.md`).
- **Pre-commit**: `pre-commit run --all-files`; `python3 -m pytest` for touched modules
  (`uv run pytest` is broken in this env).

## 8. Out of scope

Judge / process-reward scoring (v2 change 2a); actor-model V critic + GAE (v2 change 2b);
full sandbox snapshot/fork internals (Sub-project F); legacy
`areal/experimental/openai/proxy/` path (superseded; its capacity model does not apply to
v2); `agent_versions` immutability, RLS, multi-tenancy (unchanged).

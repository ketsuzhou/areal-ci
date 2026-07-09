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

### Requirement: AssembledDag resolves into a multi-agent SuperNode (via SuperNodeAssembler)

`MultiAgentEnvDispatchWorkflow.arun_episode` SHALL build the `ExecutionDAG[SuperNode]` by
reusing the existing `SuperNodeAssembler.assemble_from_refs(dag, resolver)` (ref-resolution,
not turn-index slicing). `TreeSearchGroupedRolloutWorkflow._result_to_nodes` SHALL branch on
result type: a single-agent interaction dict (unchanged) vs the multica
`{"assembled_dag", "execution_dag"}` result. For the multica result it SHALL preserve the
`ExecutionDAG`'s `SuperNode`s (one per segment, carrying N agent-runs + typed edges) without
flattening to `Node`s. The leaf `SuperNode` (single-agent, no edges) MUST remain the
degenerate N=1 case.

#### Scenario: AssembledDag resolves to SuperNodes via the assembler
- **WHEN** `arun_episode` receives an `AssembledDag`
- **THEN** `SuperNodeAssembler.assemble_from_refs` resolves each segment's `tensor_ref` and
  builds one `SuperNode` per segment + an `ExecutionDAG` of typed edges, returned to
  `_result_to_nodes` without flattening

#### Scenario: Edges populate the ExecutionDAG
- **WHEN** the `AssembledDag` carries `delegation` / `mention` / `completion` / `branch` edges
- **THEN** the `ExecutionDAG` records those edges and its topology matches Multica's recorded
  structure

### Requirement: BRANCH edge type with fork provenance (F-independent)

The `AssembledDag` edge contract SHALL be extended with a `branch` edge type (in addition
to `delegation` / `mention` / `completion`). A `branch` edge MUST carry
`branch_from_segment_id` and `branch_from_checkpoint_id` provenance identifying the closed
segment the branch forked from. Branch execution SHALL fork the source env via the existing
`EnvDispatchBranchDriver` (`create_env_dispatch(mode="branch", env_id=<source>)`) - NOT
Sub-project F's sandbox snapshot/fork (out of scope; `env_snapshot` is refs-only). Multica
owns the new `/rl/start_session` for the branched agent(s).

#### Scenario: Branch records provenance
- **WHEN** a branch forks from a closed segment
- **THEN** the `AssembledDag` gains a `branch` edge carrying `branch_from_segment_id` and
  `branch_from_checkpoint_id`, and the branched agent runs under a new session minted by Multica

#### Scenario: Branch fork is F-independent
- **WHEN** `MultiAgentEnvDispatchWorkflow` executes a branch
- **THEN** it forks via `EnvDispatchBranchDriver.drive_lane` (`create_env_dispatch(mode="branch")`);
  no Sub-project F sandbox snapshot/fork is invoked

### Requirement: MCTS value backup across BRANCH edges

Advantage backup SHALL propagate a branch's terminal return along its `branch` edge to the
parent segment's checkpoint node (MCTS-style), updating the parent's value estimate (running
mean of discounted branch returns over `Node.visit_count`) from branch outcomes. A new
`backup.py` provides `branch_backup` for this; each branch trajectory SHALL also retain its
own advantage for policy-gradient training.

#### Scenario: Branch return backs up to parent checkpoint
- **WHEN** a branch trajectory completes with a terminal return
- **THEN** the return propagates along the `branch` edge to the parent segment's checkpoint
  node, updating that node's value estimate

#### Scenario: Branch keeps its own advantage
- **WHEN** advantage is computed for a branched trajectory
- **THEN** the branched trajectory retains its own per-turn advantage for policy gradient,
  independent of the backup to the parent

### Requirement: SCRATCH/BRANCH/MIXED sampling bounded by max_group_size

`choose_sample_source` (SCRATCH / BRANCH / MIXED) and `select_branch_candidate` SHALL
extend to the multi-agent setting. The total number of branch samples per query MUST NOT
exceed `max_group_size - initial_group_size`. A consecutive-failure circuit breaker SHALL
apply to multi-agent branches. `group_size=M` SHALL produce M parallel multi-agent
`AssembledDag` rollouts (one per squad rollout).

#### Scenario: Branch count is bounded
- **WHEN** branches are sampled for a query
- **THEN** the total branch count does not exceed `max_group_size - initial_group_size`

#### Scenario: group_size produces parallel squad rollouts
- **WHEN** `group_size=M` is configured
- **THEN** the grouped workflow runs M parallel `MultiAgentEnvDispatchWorkflow.arun_episode`
  calls, each returning one `AssembledDag`

### Requirement: Direct AssembledDag polling harvest

`MultiAgentEnvDispatchWorkflow.arun_episode` SHALL harvest its result by polling the
env-dispatch endpoint (`202` in-progress / `200` + `AssembledDag` done), not via the
controller's per-trajectory callback. The controller SHALL await `arun_episode` return;
the staleness manager SHALL gate admission at the `rollout_batch` (batch) level, not at
session-minting time. The per-trajectory `/callback/online_ready` path SHALL remain in use
only for the single-agent `InferenceServiceWorkflow` online path.

#### Scenario: Base workflow polls for the AssembledDag
- **WHEN** `MultiAgentEnvDispatchWorkflow.arun_episode` runs
- **THEN** it polls the env-dispatch endpoint until `200` + `AssembledDag` and returns it;
  it does not register a waiter on `/callback/online_ready`

#### Scenario: Staleness gates batch admission, not minting
- **WHEN** Multica mints sessions faster than the trainer admits batches
- **THEN** the staleness manager bounds `rollout_batch` admission and excess ready
  `AssembledDag`s hold at Multica's `202` until polled; Multica's `start_session` is not
  rejected by a session-level capacity gate

### Requirement: Partial squad failure drops the task

`MultiAgentEnvDispatchWorkflow.arun_episode` SHALL drop the task and return no trajectory
when any of the N agents in a squad fails (session error, export failure, or incomplete
`AssembledDag`), mirroring the offline group-abandon behavior. Partial `AssembledDag`
assembly MUST NOT produce a partial `SuperNode`.

#### Scenario: One agent failure drops the task
- **WHEN** one of N agents fails during a squad task
- **THEN** the workflow drops the task, returns no trajectory, and emits no partial
  `SuperNode`

#### Scenario: Polling timeout rejects the trajectory
- **WHEN** the env-dispatch endpoint does not return `200` within the configured timeout
- **THEN** the workflow rejects the trajectory and returns none

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

If any of the N agents in a squad fails (session error, export failure, or incomplete
`AssembledDag`), `MultiAgentEnvDispatchWorkflow.arun_episode` SHALL drop the task and
return no trajectory (the rollout is rejected), mirroring the offline group-abandon
behavior. Partial `AssembledDag` assembly MUST NOT produce a partial `SuperNode`.

#### Scenario: One agent failure drops the task
- **WHEN** one of N agents fails during a squad task
- **THEN** the workflow drops the task, returns no trajectory, and emits no partial
  `SuperNode`

#### Scenario: Polling timeout rejects the trajectory
- **WHEN** the env-dispatch endpoint does not return `200` within the configured timeout
- **THEN** the workflow rejects the trajectory and returns none

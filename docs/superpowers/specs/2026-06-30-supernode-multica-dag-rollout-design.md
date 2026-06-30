# SuperNode: Multi-Agent DAG Rollout with Unified Causal Backup

**Status:** Design (awaiting plan)
**Date:** 2026-06-30
**Scope:** Phase 1 (data model + DAG construction in workflow) + Phase 2 (verifier-driven rewards + Node-level backup across the unified causal chain)

## 1. Motivation

`TreeSearchGroupedRolloutWorkflow` (`customized_areal/tree_search/core/customized_grouped_workflow.py`) currently runs **one agent** per query. The `agents/` directory contains torch-free building blocks for a collaborative multi-agent DAG (planner delegates to workers, workers report back, synthesizer terminates), but these are tested in isolation and not wired into the workflow (`agents/integration.py` lines 228-235 explicitly defer the wiring).

Two structural changes are required to support multi-agent rollout:

1. **Rename + unify the linear-event and DAG-node types.** Today `Event` (`agents/event_model.py`) and `AgentRunNode` (`agents/execution_dag.py`) are duals — both describe "one completed agent run," one for the linear-log view and one for the DAG-node view. They must become a single type, `SuperNode`, that also carries the multi-turn payload `nodes: list[Node]`.
2. **Wire the multi-agent path.** `TreeSearchGroupedRolloutWorkflow.arun_episode` must dispatch to a coordinator that runs a team of agents on one query, builds the DAG, runs the verifier per `session_id`, and backs reward up.

A `SuperNode` is **not** "one agent run." It is a **contiguous segment of one agent's action sequence, bounded by communication events** (delegation, mention, completion). One agent run that performs multiple communications is split into multiple SuperNodes. This makes the DAG granularity match the credit-assignment granularity: credit is attributed to the specific segment that delegated/completed, not the whole run.

### Multica orchestrates; AReal consumes

The multi-agent topology is **not** decided by AReal. Multica orchestrates the entire multi-agent execution:

1. AReal submits a root task to Multica.
2. Multica configures the environment required for the task.
3. Multica submits the task via user or agent role (this starts the root agent).
4. For each agent session:
   - AReal obtains a `session_id` from the proxy (`/rl/start_session`).
   - AReal submits a task to Multica (with the `session_id` bound to this agent).
   - Multica configures the environment for this agent's task.
   - Multica submits via user or agent role; the agent runs.
   - When the agent session ends, Multica notifies AReal.
   - Multica saves the `session_id` for this agent.
5. When the root task completes:
   - Multica returns to AReal: **all `session_id`s + the DAG + all branching-point environment info**.
6. AReal builds SuperNodes and Nodes from the `session_id`s + DAG + branching env info.

Concretely, Multica hands AReal:
- **`session_id`s**: one per agent run, each already bound to the proxy's interaction cache.
- **DAG**: the segment-level structure (communication-bounded SuperNodes + typed edges `delegation`/`mention`/`completion`). Multica decides segment boundaries at communication events during execution — AReal does not infer them.
- **Branching-point environment info**: for each SuperNode, the team-wide environment snapshot (`sandbox_ids`, `issue_snapshot_id`, `env_state`) captured at the moment that segment's closing communication event fired.

AReal's role is therefore:
- Submit the root task to Multica and wait for completion.
- For each agent session (during execution): issue `session_id` from the proxy; submit the agent's task to Multica with `session_id` bound; receive session-end notification.
- At completion: receive (session_ids + DAG + branching env) from Multica.
- For each `session_id`: fetch the proxy's interactions (token-level turns) → `list[Node]` via the existing `interactions_dict_to_nodes` path.
- Assemble SuperNodes from Multica's segment specs, mapping each agent's turns into the segment that owns them.
- Stamp branching env info onto each SuperNode.
- Establish the unified `parent_node_id` chain (causal flattening, see below).
- Run the verifier per `session_id` → `outcome_reward`; finalize each RL session.
- Back reward up along the unified `parent_node_id` chain.
- Return the batched tensor dict.

**Dynamic delegation is Multica's concern, not AReal's.** Multica decides at runtime which agents to spawn, on which sub-issues, with which edges. AReal is delegation-agnostic — it consumes whatever DAG Multica returns. The earlier "static team config, design for dynamic later" decision is therefore moot.

### Reward backup: causal flattening

When AReal assembles SuperNodes from Multica's segment specs, it maintains **one unified `parent_node_id` chain across all agents and all segments**:

- Within one agent run: `n_{k+1}.parent_node_id = n_k.node_id` (sequential turns).
- Across a delegation boundary: when agent A's turn `n3` delegates to agent B, B's first turn `n1'` has `parent_node_id = n3.node_id` (cross-agent link).
- Across a completion boundary: when B completes and A continues, A's next turn `n4` has `parent_node_id = B's_terminal.node_id` (A's continuation depends on B's result).

The cross-agent linkage is concrete: Multica's segment spec identifies, for each delegation/completion edge, the **source segment's terminal turn** (by `node_id` once AReal has built the Nodes) and the **destination segment's first turn**. AReal's assembler reads these from the spec and sets `parent_node_id` accordingly.

This encodes the **full DAG causal order into the Node-level parent chain**. As a consequence, reward backup walks only this chain (`_backup_path`) — it transparently crosses SuperNode and agent boundaries. The DAG-edge backup function `distribute_reward_over_dag` is **not on the reward path**; it is retained in `agents/dag_backup.py` as a utility for future segment-level analysis, join-point explicit credit, and debugging.

### What SuperNode carries that Node does not

After flattening, SuperNode is no longer a unit of MCTS statistics (those stay per-Node). SuperNode carries:

- **DAG topology** (`incoming_edges`, `outgoing_edges`, `closing_event`, `closing_event_target`): for verifier queries and DAG introspection.
- **Communication-event provenance**: which turn performed which delegation/mention/completion.
- **RL session binding** (`session_id`): one `session_id` per agent run (per `/rl/start_session`), shared across all SuperNodes of that run; verifier assigns reward by `session_id`.
- **Team environment snapshot**: `sandbox_ids: list[str]` (one per agent in the team), `issue_snapshot_id`, `env_state: dict`. A SuperNode is a checkpoint of the **entire team's** environment state at the moment its closing event fired. Phase 3 SuperNode-level branching will fork this list (all agents' sandboxes + the Multica issue subtree).

## 2. Design Decisions (locked during brainstorming)

| # | Decision | Rationale |
|---|---|---|
| D1 | Unify `Event` + `AgentRunNode` → `SuperNode` (identity + edges + reward + `nodes: list[Node]` + env snapshot) | One type for both linear-log and DAG-node views |
| D2 | Refactor `core/tree_store.py` so `Node` is torch-lazy; `SuperNode.nodes: list[Node]` literal; DAG layer stays import-clean without torch | Preserve the torch-free test surface that `agents/` relies on |
| D3 | Collaborative agent team via DAG — full `agents/` wiring | Match the README's planner→workers→synthesizer scenario |
| D4 | Spec scope = Phase 1 (SuperNode + DAG construction) + Phase 2 (verifier + Node-level backup). Critic/GAE/cloud-env branching/dynamic delegation are Phase 3+ | One cohesive, ship-able increment; tests stay green |
| D5 | Multica orchestrates the multi-agent topology; AReal submits the root task, binds session_ids during execution, and consumes (session_ids + DAG + branching env) at completion | Multica already manages agent spawning/communication; AReal should not duplicate topology decisions. Dynamic delegation is Multica's concern. |
| D6 | `SuperNode.node_id` is a UUID4 | Globally unique without scheme coupling |
| D7 | A segment = all turns since the previous communication event (exclusive), up to **and including** the turn that performs the next communication event (Option A) | The event-causing turn is the segment's terminal; causality for backup; every segment has a DAG edge |
| D8 | `parent_node_id` links across SuperNode **and agent** boundaries (causal flattening) | Reward backup walks one unified chain; `distribute_reward_over_dag` becomes redundant on the reward path |
| D9 | `distribute_reward_over_dag` retained as a utility function; not called on the reward path | Future use (segment analysis, join-point credit, debugging) at zero runtime cost |
| D10 | `session_id` bound to **agent run**, shared across that run's SuperNodes | Matches `/rl/start_session` semantics (one session per agent run); verifier assigns one `outcome_reward` per run |
| D11 | `sandbox_ids: list[str]` (one per team agent), `issue_snapshot_id: str | None`, `env_state: dict` on SuperNode | Team-wide environment checkpoint; SuperNode-level branching forks the full list |
| D12 | MCTS statistics stay **per-Node** (visit counts, Q-values, judge scores, normalized advantages/returns) | The unified parent chain already covers cross-agent credit; segment-level stats have no current consumer |
| D13 | Land in two phases: Phase 1a (data model + codec + tests, single-agent path unchanged) → Phase 1b/2 (coordinator + verifier + backup) | Each PR independently verifiable; intermediate green state |
| D14 | Multica provides segment specs (boundaries + edges + branching env) at task completion; AReal's `SuperNodeAssembler` consumes them. No side-channel `_communication_events` key in `arun_episode` return | Multica is the orchestrator and already tracks communication events; AReal should not re-infer them |

## 3. Architecture (end state)

```
┌─────────────────────────────────────────────────────────────────────┐
│  TreeSearchGroupedRolloutWorkflow.arun_episode(engine, data)         │
│                                                                     │
│  if multica_dag_enabled: ──► TeamRolloutCoordinator.run(...) ◄── NEW │
│  else:                  ──► existing single-agent path (unchanged)  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/team_rollout.py  ◄── NEW                                    │
│                                                                     │
│  TeamRolloutCoordinator(multica_client, rl_bridge, verifier,         │
│                         rl_writer, assembler)                        │
│                                                                     │
│  run(query, data, engine, tree_store) -> batched result | None:      │
│                                                                     │
│  ── Phase A: submit + orchestrate ──────────────────────────────── │
│    1. multica_client.submit_root_task(task_spec) -> root_task_id    │
│       (AReal submits the root task; Multica configures env +        │
│        spawns agents via user/agent role)                           │
│    2. For each agent session (driven by Multica notifications):     │
│         a. rl_bridge.start_session(task_id) -> session_id           │
│            (AReal issues session_id from proxy /rl/start_session)   │
│         b. multica_client.bind_agent_session(agent_run_id,          │
│                                              session_id)            │
│            (AReal tells Multica the session_id for this agent)      │
│         c. [Multica runs the agent: configures env, submits via     │
│            user/agent role; agent produces interactions in proxy]  │
│         d. multica_client notifies session_end(agent_run_id)        │
│                                                                     │
│  ── Phase B: at root task completion ───────────────────────────── │
│    3. multica_client.collect_result(root_task_id) -> DagResult:     │
│         - session_ids: list[str]          (one per agent run)       │
│         - segments: list[SegmentSpec]    (Multica-defined          │
│                                            communication-bounded   │
│                                            segments)               │
│         - edges: list[EdgeSpec]          (typed DAG edges between  │
│                                            segments)               │
│         - env_snapshots: dict[segment_id, TeamEnvSnapshot]          │
│                                            (team-wide env at each  │
│                                            branching point)        │
│    4. For each session_id:                                          │
│         rl_bridge.export_trajectories(session_id)                   │
│           -> dict[str, InteractionWithTokenLogpReward]               │
│         interactions_dict_to_nodes(...) -> list[Node]               │
│           (existing path; one Node per turn)                       │
│    5. SuperNodeAssembler.assemble(                                  │
│         sessions_nodes: dict[session_id, list[Node]],              │
│         segments, edges, env_snapshots, session_to_agent            │
│       ) -> tuple[list[SuperNode], ExecutionDAG, root_node_id]       │
│         (maps each agent's turns into the segments Multica defined; │
│          sets the unified parent_node_id chain across agents and    │
│          segments per the causal-flattening rule; stamps env        │
│          snapshots on SuperNodes; binds session_id to each          │
│          SuperNode of that run)                                     │
│    6. tree_store.insert_super_batch(supers)                         │
│    7. For each leaf agent run:                                      │
│         verifier.verify(run) per session_id -> outcome_reward        │
│         rl_writer.finalize(session_id, result)                      │
│    8. backup_episode_terminal(root_terminal_node_id, reward)        │
│       (or backup_path_returns for dense per-node returns)            │
│       —— walks the unified parent_node_id chain across all agents.  │
│    9. Return batched tensor dict (same contract as single-agent).   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/multica_dag_client.py  ◄── NEW                              │
│                                                                     │
│  MulticaDagClient (Protocol + HTTP impl)                            │
│    - submit_root_task(task_spec) -> root_task_id                    │
│    - bind_agent_session(agent_run_id, session_id) -> None           │
│    - notify_session_end(agent_run_id) -> None                       │
│    - collect_result(root_task_id) -> DagResult                      │
│                                                                     │
│  DagResult:                                                         │
│    - session_ids: list[str]                                         │
│    - segments: list[SegmentSpec]                                    │
│    - edges: list[EdgeSpec]                                          │
│    - env_snapshots: dict[str, TeamEnvSnapshot]                      │
│    - session_to_agent: dict[str, str]                               │
│                                                                     │
│  Precedent: MulticaSweLegoClient (agents/swe_lego_client.py)         │
│  already wraps POST /api/v1/swe-lego/issues — confirmed available. │
│  The new client follows the same atomic-endpoint pattern but for    │
│  general multi-agent DAG tasks.                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/supernode_assembler.py  ◄── NEW (replaces SegmentSplitter)   │
│                                                                     │
│  SuperNodeAssembler.assemble(                                       │
│    sessions_nodes: dict[session_id, list[Node]],                    │
│    segments: list[SegmentSpec],                                     │
│    edges: list[EdgeSpec],                                           │
│    env_snapshots: dict[str, TeamEnvSnapshot],                       │
│    session_to_agent: dict[str, str],                                │
│  ) -> tuple[list[SuperNode], ExecutionDAG, str]                     │
│                                                                     │
│  Consumes Multica's segment specs (already communication-bounded);  │
│  does NOT infer segment boundaries. Maps each agent's list[Node]   │
│  into the segments Multica defined (by turn_idx range per spec).    │
│                                                                     │
│  Sets the unified parent_node_id chain:                              │
│    - Within run: n_{k+1}.parent = n_k                               │
│    - Cross-agent delegation (A's n3 -> B's n1'): n1'.parent = n3    │
│      (EdgeSpec identifies source segment's terminal turn + dest     │
│      segment's first turn; assembler reads these and sets parent)   │
│    - Cross-agent completion (B completes -> A continues@n4):         │
│      n4.parent = B's terminal                                       │
│                                                                     │
│  Stamps env_snapshots on each SuperNode.                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/execution_dag.py  (modified)                                │
│                                                                     │
│  SuperNode  ◄── replaces AgentRunNode + Event                       │
│    identity:        node_id (UUID4), agent_id, issue_id, task_id    │
│    DAG topology:    incoming_edges, outgoing_edges (EdgeType)        │
│    comm provenance: closing_event, closing_event_target              │
│    linear:          completion_index, completion_time               │
│    RL session:      session_id (one per agent run, shared across    │
│                     this run's SuperNodes)                          │
│    branch:          branch_seq, branch_issue_id,                    │
│                     branch_env_snapshot_id                          │
│    reward:          process_reward, outcome_reward, value           │
│    team env:        sandbox_ids: list[str], issue_snapshot_id,       │
│                     env_state: dict                                 │
│    turns:           nodes: list[Node]   ◄── from core.tree_store    │
│    metadata:        dict                                             │
│                                                                     │
│  ExecutionDAG  (API unchanged; holds SuperNodes instead of           │
│                AgentRunNodes)                                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/event_codec.py  (simplified)                                │
│                                                                     │
│  dag_to_supernodes(dag, ordering?) -> list[SuperNode]                │
│  supernodes_to_dag(supernodes) -> ExecutionDAG                       │
│  replay_prefix_for(supernodes, branch_point) -> ReplayPrefix        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/event_model.py  -> shrinks to EdgeRef + message_timeline     │
│  (Event class removed)                                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  core/tree_store.py  (Node torch-lazy; MCTSTreeStore unified)        │
│                                                                     │
│  Node: tensor-typed fields become Any; torch imported lazily in     │
│  _node_to_tensor_dict and any tensor-consuming method. Module       │
│  imports cleanly without torch.                                     │
│                                                                     │
│  MCTSTreeStore:                                                     │
│    trajectories: dict[str, list[SuperNode]]    ◄── was list[Node]    │
│    _super_id_to_key: dict[str, tuple[str, int]]                     │
│    _node_id_to_super: dict[str, tuple[str, int]]                     │
│    (per-Node MCTS stats: _visit_counts, _total_values, _q_values,    │
│     _sum_sq_values, _rewards, _values, _value_variances,             │
│     _judge_scores, _normalized_advantages, _normalized_returns)    │
│    (no per-SuperNode MCTS stats — D12)                              │
│                                                                     │
│  insert_super_batch(supers, backup=True) replaces insert_batch       │
│  backup_episode_terminal / backup_path_returns — unchanged logic,   │
│  walks parent_node_id across SuperNode/agent boundaries              │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Data Model

### 4.1 SuperNode (`agents/execution_dag.py`)

```python
@dataclass
class SuperNode:
    """One communication-bounded segment of one agent's action sequence.

    Bounded by communication events (delegation, mention, completion).
    ``nodes`` is the contiguous run of turns within this segment; the LAST
    node is the segment's terminal — the turn that performed the closing
    communication event (or the run's final turn for a leaf segment).

    ``sandbox_ids`` is a snapshot of the entire team's sandbox state at the
    moment the closing event fired (one sandbox_id per team agent). Phase 3
    SuperNode-level branching forks this list + the Multica issue subtree.
    """

    # Identity (UUID4)
    node_id: str

    # Communication-event provenance (which event closed this segment)
    closing_event: EdgeType | None = None        # None for leaf segments
    closing_event_target: str | None = None     # the other SuperNode's UUID

    # Agent context
    agent_id: str
    issue_id: str
    task_id: str

    # DAG edges (typed, both directions)
    incoming_edges: tuple[EdgeRef, ...] = ()
    outgoing_edges: tuple[EdgeRef, ...] = ()

    # Linear trajectory
    completion_index: int | None = None
    completion_time: float | None = None

    # RL session (None until /rl/start_session binds one; shared across
    # all SuperNodes of the same agent run)
    session_id: str | None = None

    # Branch provenance (filled when this segment becomes a branch source)
    branch_seq: int | None = None
    branch_issue_id: str | None = None
    branch_env_snapshot_id: str | None = None

    # Reward (torch-free floats; written by verifier)
    process_reward: float = 0.0
    outcome_reward: float = 0.0
    value: float | None = None     # critic V_{t+1} (Phase 3, out of scope here)

    # Team environment snapshot at close time
    sandbox_ids: list[str] = field(default_factory=list)
    issue_snapshot_id: str | None = None
    env_state: dict = field(default_factory=dict)

    # The turns within this segment (Node is torch-lazy so this dataclass
    # imports cleanly without torch).
    nodes: list[Node] = field(default_factory=list)

    # Free-form metadata
    metadata: dict = field(default_factory=dict)

    @property
    def terminal_node(self) -> Node | None:
        """The segment's last node — the closing-event turn or run-final."""
        return self.nodes[-1] if self.nodes else None

    @property
    def branch_node_id(self) -> str | None:
        """node_id of the terminal node (for branching keys)."""
        t = self.terminal_node
        return t.node_id if t is not None else None
```

**Removed** vs. today's `Event` + `AgentRunNode`:
- The `Event.messages: tuple[dict, ...]` transcript-slice field — replaced by `nodes: list[Node]` (the actual turns, not their serialized dicts).
- Per-node `credit` field (was proposed earlier, then dropped) — DAG backup is off the reward path; credit stays on `Node.credit` in `tree_store` and is set by the per-Node MCTS path.

**Added**:
- `closing_event` / `closing_event_target` (comm-event provenance).
- `sandbox_ids` / `issue_snapshot_id` / `env_state` (team env snapshot).

### 4.2 Serialization (`agents/event_codec.py`)

`SuperNode.to_dict()` / `SuperNode.from_dict(d)` mirror today's `Event` API:
- All fields above emitted; `nodes` serialized via a new `Node.to_dict()` helper (today `Node` has none — it is persisted via the tree_store checkpoint).
- `incoming_edges` / `outgoing_edges` use the `[[node_id, edge_type_str], ...]` shape.
- Edge symmetry + topological invariant validated on decode (same as today's `events_to_dag`).

`ExecutionDAG.to_records` / `from_records` round-trip SuperNodes (carrying `nodes`).

### 4.3 Node torch-lazy refactor (`core/tree_store.py`)

- Top-level `import torch` removed. Module imports cleanly without torch.
- Tensor-typed fields (`advantages: torch.Tensor | None`, `returns: torch.Tensor | None`) become `Any` with a comment pointing to `_ensure_torch()`.
- `_node_to_tensor_dict` and any function constructing tensors do `from customized_areal.utils.torch_utils import lazy_import_torch; torch = lazy_import_torch()` at first use. (If a `lazy_import_torch` helper does not exist, add one in `areal/utils/`.)
- Tests that do not exercise tensor paths run without torch installed.

### 4.4 Unified MCTSTreeStore (`core/tree_store.py`)

```python
class MCTSTreeStore:
    """Unified store: SuperNodes (segments) hold DAG topology, comm-event
    provenance, and team env snapshots; Nodes hold all MCTS stats and the
    unified causal parent_node_id chain.

    Reward backup: Node-level only (parent_node_id encodes the full DAG
    causal order across agents and segments).
    """

    # Primary storage: query_id -> SuperNodes in insertion order
    trajectories: dict[str, list[SuperNode]]

    # SuperNode-level index: SuperNode UUID -> (query_id, idx_in_trajectories)
    _super_id_to_key: dict[str, tuple[str, int]]

    # Node-level index: node_id -> (super_node_id, idx_in_super_node.nodes)
    _node_id_to_super: dict[str, tuple[str, int]]

    # Per-Node MCTS stats (unchanged): _visit_counts, _total_values,
    # _q_values, _sum_sq_values, _rewards, _values, _value_variances,
    # _judge_scores, _normalized_advantages, _normalized_returns
```

**Public API**:
- `insert_super_batch(supers: list[SuperNode], backup: bool = True)` replaces `insert_batch(list[Node])`.
- `get_super_node(uuid) -> SuperNode | None`.
- `get_node(node_id)` walks into a SuperNode's `nodes` via `_node_id_to_super`.
- `backup_episode_terminal(terminal_node_id, reward)` — unchanged; walks `parent_node_id` across boundaries.
- `backup_path_returns(terminal_node_id, returns_by_node_id)` — unchanged.
- All per-Node accessors unchanged: `get_q_value`, `get_visit_count`, `get_loo_value_and_variance`, `set_value`, `add_judge_score`, etc.

**Backward-compat**: the existing `insert_batch(list[Node])` API is replaced. Call sites in `customized_grouped_workflow.py` are updated in Phase 1a to wrap single-agent Nodes in a single leaf SuperNode (preserving today's behavior).

## 5. SuperNodeAssembler (`agents/supernode_assembler.py`, new module)

Replaces the originally-proposed `SegmentSplitter`. The splitter inferred segment boundaries from side-channel communication events; the assembler instead **consumes Multica's pre-defined segment specs** and maps each agent's `list[Node]` into them.

### 5.1 Multica-provided data structures

```python
@dataclass(frozen=True)
class SegmentSpec:
    """One communication-bounded segment, as defined by Multica.

    Multica decides segment boundaries at communication events during
    execution; AReal does not infer them.
    """
    segment_id: str               # Multica's UUID for this segment
    agent_run_id: str             # which agent run this segment belongs to
    issue_id: str
    task_id: str
    closing_event: EdgeType | None     # None for leaf segments
    closing_event_target_segment: str | None  # the other segment's UUID
    # The turn range within this agent run that belongs to this segment.
    # Multica tracks turn indices as the agent runs; AReal maps them to
    # Node positions in the list[Node] built from the proxy interactions.
    start_turn_idx: int           # 1-based, inclusive
    end_turn_idx: int             # 1-based, inclusive (the closing-event turn)


@dataclass(frozen=True)
class EdgeSpec:
    """One typed DAG edge between segments, as defined by Multica."""
    src_segment_id: str
    dst_segment_id: str
    type: EdgeType                # DELEGATION | MENTION | COMPLETION


@dataclass(frozen=True)
class TeamEnvSnapshot:
    """Team-wide environment state at a branching point, from Multica.

    One per segment, captured at the moment that segment's closing
    communication event fired. Multica owns the env state; AReal stamps
    it onto the corresponding SuperNode without interpretation.
    """
    sandbox_ids: list[str]        # one per team agent, ordered by role
    issue_snapshot_id: str | None
    env_state: dict


@dataclass(frozen=True)
class DagResult:
    """Multica's completion payload: the DAG of segments + env snapshots."""
    session_ids: list[str]                              # one per agent run
    session_to_agent_run: dict[str, str]                # session_id -> agent_run_id
    segments: list[SegmentSpec]
    edges: list[EdgeSpec]
    env_snapshots: dict[str, TeamEnvSnapshot]          # segment_id -> snapshot
```

### 5.2 SuperNodeAssembler

```python
class SuperNodeAssembler:
    """Assemble SuperNodes from Multica's segment specs + proxy interactions.

    Consumes Multica's pre-defined segments (does NOT infer boundaries).
    Maps each agent's list[Node] into the segments Multica defined, by
    the start_turn_idx / end_turn_idx range on each SegmentSpec.

    Maintains the unified parent_node_id chain (causal flattening):
      - Within one run: n_{k+1}.parent_node_id = n_k.node_id
      - Cross-agent delegation (A's n3 delegates -> B's n1'):
        n1'.parent_node_id = n3.node_id
        (EdgeSpec with type=DELEGATION identifies src segment's terminal
        turn + dst segment's first turn; assembler sets parent)
      - Cross-agent completion (B completes -> A continues@n4):
        n4.parent_node_id = B's terminal node_id
        (EdgeSpec with type=COMPLETION; same mechanism)

    Stamps TeamEnvSnapshot onto each SuperNode.
    Binds session_id to each SuperNode (from session_to_agent_run).
    """

    def assemble(
        self,
        *,
        sessions_nodes: dict[str, list[Node]],   # session_id -> list[Node]
        dag_result: DagResult,
    ) -> tuple[list[SuperNode], ExecutionDAG, str]:
        """Returns (super_nodes, dag, root_terminal_node_id).

        root_terminal_node_id is the terminal node of the DAG's root leaf
        — the starting point for backup_episode_terminal.
        """
        ...
```

### 5.3 Why the assembler does not infer segments

The earlier `SegmentSplitter` design had AReal infer segment boundaries by detecting communication events in the agent's turn stream. That approach:
- Required a side-channel `_communication_events` key in `arun_episode` return, which has no producer today (no code detects delegate/mention/completion).
- Or required post-hoc heuristic detection from `raw_messages` (brittle, error-prone).

Multica is the orchestrator and already tracks communication events as it spawns agents and routes messages. The assembler consumes Multica's pre-defined segments — no inference, no side-channel, no heuristics. This is the clean separation of concerns: Multica owns topology + segmentation; AReal owns token-level interactions + SuperNode/Node data model + reward + backup.

## 6. MulticaDagClient (`agents/multica_dag_client.py`, new module)

```python
class MulticaDagClient(Protocol):
    """AReal's interface to Multica's multi-agent DAG orchestration.

    Precedent: MulticaSweLegoClient (agents/swe_lego_client.py) already
    wraps POST /api/v1/swe-lego/issues (confirmed available). The new
    client follows the same atomic-endpoint pattern but for general
    multi-agent DAG tasks.
    """

    async def submit_root_task(
        self, *, task_spec: TaskSpec
    ) -> str:
        """Submit the root task; returns root_task_id.

        Multica configures the env and begins orchestrating (spawns agents
        via user/agent role as the DAG requires).
        """
        ...

    async def bind_agent_session(
        self, *, agent_run_id: str, session_id: str
    ) -> None:
        """Tell Multica the session_id AReal issued for this agent run."""
        ...

    async def notify_session_end(self, *, agent_run_id: str) -> None:
        """Notify Multica that an agent's session has ended."""
        ...

    async def collect_result(self, *, root_task_id: str) -> DagResult:
        """Block until the root task completes; return the DAG + env snapshots."""
        ...
```

The concrete HTTP implementation is a thin wrapper over Multica's endpoints (to be defined on the Multica side, following the `/api/v1/swe-lego/issues` precedent). AReal depends only on the Protocol; tests inject a fake.

## 7. TeamRolloutCoordinator (`agents/team_rollout.py`, new module)

```python
class TeamRolloutCoordinator:
    """Runs a collaborative agent team on one query, Multica-orchestrated.

    Phase 1+2 (this spec):
      - Multica orchestrates topology + segmentation + env snapshots.
      - AReal issues session_ids, builds SuperNode/Node, runs verifier,
        backs reward up along the unified parent_node_id chain.

    Designed-for-later (Phase 3+, not built here):
      - Critic + GAE advantages.
      - Cloud-env branching at segment boundaries.
      - SuperNode-level MCTS stats.
    """

    def __init__(
        self,
        *,
        multica_client: MulticaDagClient,
        rl_bridge: RLBridgeClient,
        verifier: Verifier,
        rl_writer: RLSessionRewardWriter,
        assembler: SuperNodeAssembler,
    ) -> None: ...

    async def run(
        self,
        *,
        engine,
        data: dict[str, Any],
        query_id: str,
        tree_store: MCTSTreeStore,
    ) -> dict[str, Any] | None:
        # ── Phase A: submit + orchestrate ───────────────────────────
        # 1. multica_client.submit_root_task(task_spec) -> root_task_id
        # 2. For each agent session (driven by Multica notifications):
        #      a. rl_bridge.start_session(task_id) -> session_id
        #      b. multica_client.bind_agent_session(agent_run_id, session_id)
        #      c. [Multica runs the agent; proxy caches interactions]
        #      d. multica_client.notify_session_end(agent_run_id)
        #
        # ── Phase B: at root task completion ──────────────────────────
        # 3. dag_result = multica_client.collect_result(root_task_id)
        #    -> DagResult(session_ids, segments, edges, env_snapshots,
        #                 session_to_agent_run)
        # 4. For each session_id in dag_result.session_ids:
        #      interactions = rl_bridge.export_trajectories(session_id)
        #      nodes = interactions_dict_to_nodes(interactions)
        #      sessions_nodes[session_id] = nodes
        # 5. super_nodes, dag, root_terminal_node_id = assembler.assemble(
        #      sessions_nodes=sessions_nodes, dag_result=dag_result)
        # 6. tree_store.insert_super_batch(super_nodes)
        # 7. For each leaf agent run:
        #      verifier.verify(run) per session_id -> VerifierResult
        #      rl_writer.finalize(session_id, result)
        # 8. backup_episode_terminal(root_terminal_node_id, reward)
        #    (or backup_path_returns for dense per-node returns)
        #    —— walks the unified parent_node_id chain across all agents.
        # 9. Return batched tensor dict (same contract as single-agent).
```

## 8. Workflow integration (`core/customized_grouped_workflow.py`)

```python
class TreeSearchGroupedRolloutWorkflow(RolloutWorkflow):
    def __init__(
        self,
        ...,
        multica_dag_enabled: bool = False,  # NEW; False = single-agent
        multica_dag_client: MulticaDagClient | None = None,
    ) -> None:
        ...
        self._multica_dag_enabled = multica_dag_enabled
        if multica_dag_enabled:
            self._coordinator = TeamRolloutCoordinator(
                multica_client=multica_dag_client,
                rl_bridge=...,             # RLBridgeClient adapter
                verifier=...,              # constructed from verifier deps
                rl_writer=...,             # RLSessionRewardWriter
                assembler=SuperNodeAssembler(),
            )
        else:
            self._coordinator = None

    async def arun_episode(self, engine, data):
        if self._coordinator is not None:
            return await self._coordinator.run(
                engine=engine, data=data,
                query_id=data.get("query_id", ""),
                tree_store=self.tree_store,
            )
        # Existing single-agent path (unchanged)
        ...
```
            )
        else:
            self._coordinator = None

    async def arun_episode(self, engine, data):
        if self._coordinator is not None:
            return await self._coordinator.run(
                engine=engine, data=data,
                query_id=data.get("query_id", ""),
                tree_store=self.tree_store,
            )
        # Existing single-agent path (unchanged)
        ...
```

## 8. `session_id` source and flow

`session_id` originates from the areal RL proxy layer (`areal/experimental/openai/proxy/`):

- `POST /rl/start_session` (`proxy_gateway.py:353`) takes `{task_id, api_key?}`, returns `{session_id, api_key}`. Each `session_id` corresponds to one `SessionData` (`server.py:66`) and one agent run.
- Agent turns (`/chat/completions` calls) carry the session API key; the proxy caches each as an `InteractionCache` entry with its own `interaction_id`.
- `POST /rl/set_reward` (`proxy_gateway.py:623`) takes `{interaction_id?, reward}`.
- `POST /rl/end_session` (`proxy_gateway.py:637`) ends the session and triggers trajectory export.

The `RLBridgeClient` protocol (`agents/rl_session.py:23`) abstracts this: `set_reward(*, session_id, reward)` and `end_session(*, session_id)`. A concrete adapter (to be implemented in Phase 1b/2) maps `session_id` to the proxy's `interaction_id` (e.g., the session's terminal interaction or a session-level aggregation) and forwards the HTTP calls.

In this design:
- One agent run = one `session_id` (one `/rl/start_session` call per run).
- All SuperNodes of that run share the `session_id`.
- The verifier assigns one `outcome_reward` per `session_id`.
- Reward backup walks the unified `parent_node_id` chain from the run's terminal node.

## 9. Out of scope (Phase 3+)

- Critic observations (`agents/critic_observation.py`), GAE (`agents/gae.py`), `assemble_node_advantages`.
- `BranchMaterializer` cloud-env branching at segment boundaries (`agents/integration.py`).
- SuperNode-level MCTS stats (segment visit counts, Q-values) — no current consumer.
- SuperNode-level advantage computer (segment-level GRPO analog).
- Per-segment judge scores (vs. today's per-Node judge scores).
- The Multica-side endpoints that `MulticaDagClient` calls — those are implemented in the Multica repo, following the `/api/v1/swe-lego/issues` precedent. This spec defines only AReal's client Protocol and consumption.

## 10. Two-phase landing

### Phase 1a (independent PR)
- `core/tree_store.py`: Node torch-lazy refactor.
- `agents/execution_dag.py`: `SuperNode` dataclass (replaces `AgentRunNode`); `ExecutionDAG` holds SuperNodes.
- `agents/event_model.py`: shrink to `EdgeRef` + `message_timeline` (Event class removed).
- `agents/event_codec.py`: simplify to `dag_to_supernodes` / `supernodes_to_dag` / `replay_prefix_for`.
- `agents/supernode_assembler.py`: new module (SuperNodeAssembler + SegmentSpec + EdgeSpec + TeamEnvSnapshot + DagResult).
- `core/tree_store.py`: unified `MCTSTreeStore` (`trajectories: dict[str, list[SuperNode]]`, `insert_super_batch`, dual indices).
- `agents/__init__.py`: update exports — remove `Event`, `AgentRunNode`, `dag_to_events`, `events_to_dag`; add `SuperNode`, `dag_to_supernodes`, `supernodes_to_dag`, `SuperNodeAssembler`, `SegmentSpec`, `EdgeSpec`, `TeamEnvSnapshot`, `DagResult`.
- `agents/gae.py` (Phase 3 module, out of scope but import-affected): update `events_from_nodes` to construct `SuperNode` instead of `Event`, and `GlobalEvent` to project from `SuperNode`. Logic unchanged; pure rename. (If this pulls in unwanted Phase 3 churn, alternative: leave `Event` as a thin deprecated alias for `SuperNode` until Phase 3 lands — but preferred path is the rename.)
- Update single-agent path in `customized_grouped_workflow.py` to wrap Nodes in a leaf SuperNode (behavior unchanged).
- Update existing tests (`tests/test_event_codec.py`, `tests/test_event_model.py`, `tests/test_execution_dag.py`, `tests/test_dag_backup.py`, `tests/test_session_map.py`, `tests/test_e2e_critic_gae.py`) to the new API.
- New unit tests for SuperNodeAssembler (synthetic sessions_nodes + DagResult → verify segment mapping, parent_node_id chain, env snapshot stamping).

### Phase 1b/2 (second PR)
- `agents/multica_dag_client.py`: new module (`MulticaDagClient` Protocol + HTTP impl; follows `MulticaSweLegoClient` precedent).
- `agents/team_rollout.py`: new module (`TeamRolloutCoordinator`).
- `RLBridgeClient` concrete adapter mapping `session_id` → `interaction_id` (or extending the proxy to accept session-level reward).
- Wire `multica_dag_enabled` + `multica_dag_client` params into `TreeSearchGroupedRolloutWorkflow.__init__` and `arun_episode`.
- Verifier + `RLSessionRewardWriter` integration in the coordinator.
- End-to-end multi-agent test with a fake `MulticaDagClient` returning a synthetic `DagResult` (3-segment DAG: planner→worker→synthesizer), verifying reward flows along the unified parent chain across agents.

## 11. Test strategy

- **Torch-free unit tests** (most of the new code):
  - SuperNodeAssembler with synthetic `sessions_nodes` + `DagResult`; verify segment mapping (turn ranges), parent_node_id chain (within-run, cross-agent delegation, cross-agent completion), env snapshot stamping, session_id binding.
  - `insert_super_batch` + dual-level lookup (`get_super_node`, `get_node`).
  - `backup_episode_terminal` across SuperNode and agent boundaries (causal flattening).
  - `dag_to_supernodes` / `supernodes_to_dag` round-trip + edge symmetry + topo invariant.
  - Coordinator with a fake `MulticaDagClient` + mock verifier + mock rl_bridge; verify DAG construction, session binding, verifier invocation, backup invocation, batched tensor dict output.
- **Integration tests** (`pytest.importorskip("torch")` for tensor paths):
  - End-to-end team episode with a fake `MulticaDagClient` returning a 3-segment DAG (planner→worker→synthesizer); verify credit flows along the unified parent chain and per-turn `parent_node_id` walks cross SuperNode/agent boundaries.
- **Existing tests preserved**: today's `test_tree_store.py`, `test_advantage.py`, etc. run unchanged on the single-agent path.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `parent_node_id` semantic shift ("episode-internal predecessor" → "causal predecessor") surprises callers | Document on `Node.parent_node_id`; audit all readers (today: `_backup_path`, `_node_parent_id`, `_backup_inserted_episodes`); the `visited` guard in `_backup_path` already prevents cycles from malformed chains. |
| Cross-agent `parent_node_id` link requires the assembler to know the source segment's terminal `node_id` at assembly time | Multica's `EdgeSpec` identifies source + destination segments; the assembler has already built those segments' Nodes (by turn range) before setting cross-agent parent links. The source segment's terminal `node_id` is `nodes[end_turn_idx - 1].node_id`. |
| `session_id` → `interaction_id` mapping is unspecified in the proxy today | Phase 1b/2 implements the `RLBridgeClient` adapter; if the proxy lacks a clean session→terminal-interaction lookup, the adapter uses the session's last interaction (or requires the verifier to return interaction_ids). |
| Node torch-lazy refactor breaks existing tensor-typed code paths | Field types become `Any`; `_node_to_tensor_dict` lazy-imports torch; existing tests with `pytest.importorskip("torch")` continue to cover the tensor paths. |
| Two-phase landing leaves Phase 1a in a state where SuperNode exists but is unused by multi-agent path | Phase 1a wraps single-agent Nodes in a leaf SuperNode so the data model is exercised; Phase 1b/2 wires the coordinator. Intermediate state is green and useful. |
| Multica-side endpoints for `MulticaDagClient` do not exist yet | AReal depends only on the `MulticaDagClient` Protocol; tests inject a fake. The Multica repo implements the endpoints (following `/api/v1/swe-lego/issues` precedent) on its own schedule. AReal's Phase 1b/2 can land against the Protocol and be activated once Multica's endpoints are live. |
| Multica's segment turn-indices may not align with AReal's `list[Node]` positions (off-by-one, missed turns) | The assembler validates `start_turn_idx`/`end_turn_idx` against the actual `list[Node]` length and raises on mismatch; tests cover the boundary cases. |

## 13. Glossary

- **SuperNode**: communication-bounded segment of one agent's action sequence; carries DAG topology, comm-event provenance, RL session binding, team env snapshot, and `list[Node]` turns.
- **Node**: one assistant turn (`input_ids`, `loss_mask`, `logprobs`, etc.); torch-lazy after refactor.
- **Causal flattening**: encoding the full DAG causal order into the Node-level `parent_node_id` chain, so reward backup walks one unified chain.
- **Multica**: the orchestration layer that spawns agents, tracks communication events, segments agent runs, and captures team env snapshots. AReal's `MulticaDagClient` talks to it.
- **DagResult**: Multica's completion payload — `session_ids` + `segments` (SegmentSpec list) + `edges` (EdgeSpec list) + `env_snapshots` (per-segment `TeamEnvSnapshot`).
- **SegmentSpec**: Multica's definition of one communication-bounded segment, including the turn-index range within the agent run.
- **TeamEnvSnapshot**: team-wide environment state at a branching point (sandbox_ids per team agent + issue_snapshot_id + env_state), captured by Multica.
- **SuperNodeAssembler**: AReal's module that consumes `DagResult` + proxy interactions and builds `list[SuperNode]` + `ExecutionDAG`, setting the unified `parent_node_id` chain.
- **session_id**: proxy-layer identifier for one agent run; shared across all SuperNodes of that run.
- **Precedent**: `MulticaSweLegoClient` (`agents/swe_lego_client.py`) already wraps `POST /api/v1/swe-lego/issues` — confirmed available. The new `MulticaDagClient` follows the same atomic-endpoint pattern.

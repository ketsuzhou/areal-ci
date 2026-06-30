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

### Reward backup: causal flattening

When the SegmentSplitter creates SuperNodes, it maintains **one unified `parent_node_id` chain across all agents and all segments**:

- Within one agent run: `n_{k+1}.parent_node_id = n_k.node_id` (sequential turns).
- Across a delegation boundary: when agent A's turn `n3` delegates to agent B, B's first turn `n1'` has `parent_node_id = n3.node_id` (cross-agent link).
- Across a completion boundary: when B completes and A continues, A's next turn `n4` has `parent_node_id = B's_terminal.node_id` (A's continuation depends on B's result).

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
| D5 | Static configured team (declarative roles + edges + role_to_issue); interfaces designed for dynamic delegation later | Ship v1 deterministically; do not paint into a corner |
| D6 | `SuperNode.node_id` is a UUID4 | Globally unique without scheme coupling |
| D7 | A segment = all turns since the previous communication event (exclusive), up to **and including** the turn that performs the next communication event (Option A) | The event-causing turn is the segment's terminal; causality for backup; every segment has a DAG edge |
| D8 | `parent_node_id` links across SuperNode **and agent** boundaries (causal flattening) | Reward backup walks one unified chain; `distribute_reward_over_dag` becomes redundant on the reward path |
| D9 | `distribute_reward_over_dag` retained as a utility function; not called on the reward path | Future use (segment analysis, join-point credit, debugging) at zero runtime cost |
| D10 | `session_id` bound to **agent run**, shared across that run's SuperNodes | Matches `/rl/start_session` semantics (one session per agent run); verifier assigns one `outcome_reward` per run |
| D11 | `sandbox_ids: list[str]` (one per team agent), `issue_snapshot_id: str | None`, `env_state: dict` on SuperNode | Team-wide environment checkpoint; SuperNode-level branching forks the full list |
| D12 | MCTS statistics stay **per-Node** (visit counts, Q-values, judge scores, normalized advantages/returns) | The unified parent chain already covers cross-agent credit; segment-level stats have no current consumer |
| D13 | Land in two phases: Phase 1a (data model + codec + tests, single-agent path unchanged) → Phase 1b/2 (coordinator + verifier + backup) | Each PR independently verifiable; intermediate green state |
| D14 | Side-channel `_communication_events: list[CommunicationEvent]` key in `arun_episode` return dict; absent → run treated as one leaf SuperNode | Preserves today's `arun_episode` return contract; graceful fallback for legacy workflows |

## 3. Architecture (end state)

```
┌─────────────────────────────────────────────────────────────────────┐
│  TreeSearchGroupedRolloutWorkflow.arun_episode(engine, data)         │
│                                                                     │
│  if team_config:   ──► TeamRolloutCoordinator.run(...)   ◄── NEW    │
│  else:             ──► existing single-agent path (unchanged)       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/team_rollout.py  ◄── NEW                                    │
│                                                                     │
│  TeamRolloutCoordinator(team_config, splitter, verifier, rl_writer)  │
│                                                                     │
│  run(query, data, engine, tree_store) -> batched result | None:      │
│    1. Build static DAG scaffold from team_config (roles + edges).    │
│    2. Topological order over roles. For each role:                    │
│         a. /rl/start_session -> session_id.                         │
│         b. role.workflow.arun_episode(engine, sub_data)              │
│            -> list[Node] + list[CommunicationEvent]                  │
│         c. SegmentSplitter.split(...) -> list[SuperNode]             │
│            (sets parent_node_id across agents per the causal rule;  │
│            stamps sandbox_ids/issue_snapshot_id/env_state on each    │
│            SuperNode from the team frontier at segment close time)   │
│         d. tree_store.insert_super_batch(supers)                     │
│         e. Bind session_id to each SuperNode of this run.           │
│    3. For each leaf agent run:                                      │
│         verifier.verify(run) per session_id -> outcome_reward        │
│         RLSessionRewardWriter.finalize(session_id, result)           │
│    4. backup_episode_terminal(run_terminal_node_id, reward)          │
│       (or backup_path_returns for dense per-node returns)            │
│       —— walks the unified parent_node_id chain across all agents.   │
│    5. Return batched tensor dict (same contract as single-agent).   │
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
│  agents/segment_splitter.py  ◄── NEW                                │
│                                                                     │
│  SegmentSplitter.split(agent_id, issue_id, task_id, nodes, events,  │
│                        start_parent_node_id, team_frontier_snapshot)│
│    -> SegmentSplitResult(super_nodes, edges)                        │
│                                                                     │
│  Rule: segment = turns since previous comm event, up to & including │
│  the turn performing the next comm event. Event-causing turn is the │
│  segment's terminal node.                                           │
│                                                                     │
│  Maintains the unified parent_node_id chain:                        │
│    - Within run: n_{k+1}.parent = n_k                               │
│    - Cross-agent (A delegates B@n3 -> B's n1'): n1'.parent = n3     │
│    - Cross-agent (B completes -> A continues@n4): n4.parent =        │
│      B's terminal                                                    │
│                                                                     │
│  Stamps team env snapshot (sandbox_ids, issue_snapshot_id, env_state)│
│  on each SuperNode from team_frontier_snapshot at close time.        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agents/event_codec.py  (simplified)                                │
│                                                                     │
│  dag_to_supernodes(dag, ordering?) -> list[SuperNode]                │
│    (SuperNode already carries nodes + edges; mostly topological     │
│     ordering + node_id assignment)                                  │
│  supernodes_to_dag(supernodes) -> ExecutionDAG                       │
│    (inverse; validates edge symmetry + topo invariant)              │
│  replay_prefix_for(supernodes, branch_point) -> ReplayPrefix        │
│    (unchanged shape; messages derived from nodes via                │
│     message_timeline)                                               │
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

## 5. SegmentSplitter (`agents/segment_splitter.py`, new module)

### 5.1 CommunicationEvent

```python
@dataclass(frozen=True)
class CommunicationEvent:
    """One communication event emitted during an agent run."""
    turn_idx: int                # 1-based turn position within the run
    event_type: EdgeType         # DELEGATION | MENTION | COMPLETION
    target_agent_id: str         # the other agent involved
    target_issue_id: str | None  # sub-issue (DELEGATION) or None
```

### 5.2 SegmentSplitter

```python
class SegmentSplitter:
    """Split an agent run's flat list[Node] + comm events into SuperNodes.

    Rule (Option A): a segment = all turns since the previous comm event
    (exclusive), up to AND INCLUDING the turn that performs the next comm
    event. The event-causing turn is the segment's terminal node. Leaf
    segments (no closing event) absorb the trailing turns.

    Maintains the unified parent_node_id chain:
      - Within one run: n_{k+1}.parent_node_id = n_k.node_id
      - Cross-agent delegation (A's n3 delegates -> B's n1'):
        n1'.parent_node_id = n3.node_id
      - Cross-agent completion (B completes -> A continues@n4):
        n4.parent_node_id = B's terminal node_id

    Stamps team env snapshot on each SuperNode.
    """

    def split(
        self,
        *,
        agent_id: str,
        issue_id: str,
        task_id: str,
        nodes: list[Node],
        events: list[CommunicationEvent],
        start_parent_node_id: str | None,
        team_frontier_snapshot: TeamFrontierSnapshot,
    ) -> SegmentSplitResult:
        ...
```

### 5.3 SegmentSplitResult

```python
@dataclass
class SegmentSplitResult:
    super_nodes: list[SuperNode]
    # Cross-run DAG edges (src_uuid, dst_uuid, type). Sequential within-run
    # edges (SN_{k+1} parent = SN_k) are encoded via _super_parent, not here.
    edges: list[tuple[str, str, EdgeType]]
```

### 5.4 TeamFrontierSnapshot

```python
@dataclass
class TeamFrontierSnapshot:
    """Snapshot of the entire team's environment state at a moment.

    Captured at each SuperNode close to populate sandbox_ids /
    issue_snapshot_id / env_state. Built by the coordinator from the
    per-agent sandbox trackers as each segment closes.
    """
    sandbox_ids: list[str]           # one per team agent, ordered by role
    issue_snapshot_id: str | None
    env_state: dict
```

## 6. TeamRolloutCoordinator (`agents/team_rollout.py`, new module)

### 6.1 Static team config

```python
@dataclass(frozen=True)
class AgentRoleSpec:
    role: str                       # "planner" | "worker" | "synthesizer"
    agent_id: str                   # the agent config to run
    workflow: RolloutWorkflow       # the base workflow for this role


@dataclass(frozen=True)
class TeamEdgeSpec:
    src_role: str
    dst_role: str
    type: EdgeType


@dataclass(frozen=True)
class TeamConfig:
    """Static team topology (Phase 1 — declarative roles + edges)."""
    roles: tuple[AgentRoleSpec, ...]
    edges: tuple[TeamEdgeSpec, ...]
    role_to_issue: dict[str, str]   # which role runs on which sub-issue
```

### 6.2 TeamRolloutCoordinator

```python
class TeamRolloutCoordinator:
    """Runs a collaborative agent team on one query.

    Phase 1+2 (this spec):
      - Static team config (no runtime delegation decisions).
      - DAG scaffold built from TeamConfig edges.
      - Verifier per session_id; reward backup via Node-level parent_node_id.

    Designed-for-later (Phase 3+, not built here):
      - Dynamic delegation (planner decides fan-out at runtime).
      - Critic + GAE advantages.
      - Cloud-env branching at segment boundaries.
      - SuperNode-level MCTS stats.
    """

    def __init__(
        self,
        *,
        team_config: TeamConfig,
        splitter: SegmentSplitter,
        verifier: Verifier,
        rl_writer: RLSessionRewardWriter,
        rl_bridge: RLBridgeClient,
    ) -> None: ...

    async def run(
        self,
        *,
        engine,
        data: dict[str, Any],
        query_id: str,
        tree_store: MCTSTreeStore,
    ) -> dict[str, Any] | None:
        # 1. Build static DAG scaffold from team_config (roles + edges).
        # 2. Topological order over roles.
        # 3. For each role in topo order:
        #      a. /rl/start_session via rl_bridge -> session_id.
        #      b. role.workflow.arun_episode(engine, sub_data)
        #         -> dict with _backend_run_raw_messages etc.
        #            AND _communication_events: list[CommunicationEvent]
        #      c. Convert raw messages to list[Node] (existing helper).
        #      d. SegmentSplitter.split(...) -> list[SuperNode] + edges.
        #      e. tree_store.insert_super_batch(supers).
        #      f. Bind session_id to each SuperNode of this run.
        #      g. Update team_frontier_snapshot for the next role.
        # 4. Resolve cross-run edges against the DAG scaffold; build
        #    ExecutionDAG of SuperNodes.
        # 5. For each leaf agent run:
        #      verifier.verify(run) per session_id -> VerifierResult
        #      rl_writer.finalize(session_id, result)  # set_reward + end_session
        # 6. backup_episode_terminal(run_terminal_node_id, reward)
        #    (or backup_path_returns for dense per-node returns)
        #    —— walks the unified parent_node_id chain across all agents.
        # 7. Return batched tensor dict (same contract as single-agent).
```

### 6.3 Side-channel contract

The base `RolloutWorkflow.arun_episode` returns a `dict[str, Any]` (today's contract). We extend the contract: the dict **may** carry a `_communication_events: list[CommunicationEvent]` key. The coordinator reads it; if absent (legacy workflow), the run is treated as **one leaf SuperNode containing all nodes** — graceful fallback.

## 7. Workflow integration (`core/customized_grouped_workflow.py`)

```python
class TreeSearchGroupedRolloutWorkflow(RolloutWorkflow):
    def __init__(
        self,
        ...,
        team_config: TeamConfig | None = None,  # NEW; None = single-agent
    ) -> None:
        ...
        self._team_config = team_config
        if team_config is not None:
            self._coordinator = TeamRolloutCoordinator(
                team_config=team_config,
                splitter=SegmentSplitter(),
                verifier=...,             # constructed from verifier deps
                rl_writer=...,            # RLSessionRewardWriter
                rl_bridge=...,            # RLBridgeClient adapter
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
- Dynamic delegation (planner decides fan-out at runtime) — `TeamConfig` is static in v1.
- SuperNode-level MCTS stats (segment visit counts, Q-values) — no current consumer.
- SuperNode-level advantage computer (segment-level GRPO analog).
- Per-segment judge scores (vs. today's per-Node judge scores).

## 10. Two-phase landing

### Phase 1a (independent PR)
- `core/tree_store.py`: Node torch-lazy refactor.
- `agents/execution_dag.py`: `SuperNode` dataclass (replaces `AgentRunNode`); `ExecutionDAG` holds SuperNodes.
- `agents/event_model.py`: shrink to `EdgeRef` + `message_timeline` (Event class removed).
- `agents/event_codec.py`: simplify to `dag_to_supernodes` / `supernodes_to_dag` / `replay_prefix_for`.
- `agents/segment_splitter.py`: new module (SegmentSplitter + CommunicationEvent + SegmentSplitResult + TeamFrontierSnapshot).
- `core/tree_store.py`: unified `MCTSTreeStore` (`trajectories: dict[str, list[SuperNode]]`, `insert_super_batch`, dual indices).
- `agents/__init__.py`: update exports — remove `Event`, `AgentRunNode`, `dag_to_events`, `events_to_dag`; add `SuperNode`, `dag_to_supernodes`, `supernodes_to_dag`, `SegmentSplitter`, `CommunicationEvent`, `TeamFrontierSnapshot`.
- `agents/gae.py` (Phase 3 module, out of scope but import-affected): update `events_from_nodes` to construct `SuperNode` instead of `Event`, and `GlobalEvent` to project from `SuperNode`. Logic unchanged; pure rename. (If this pulls in unwanted Phase 3 churn, alternative: leave `Event` as a thin deprecated alias for `SuperNode` until Phase 3 lands — but preferred path is the rename.)
- Update single-agent path in `customized_grouped_workflow.py` to wrap Nodes in a leaf SuperNode (behavior unchanged).
- Update existing tests (`tests/test_event_codec.py`, `tests/test_event_model.py`, `tests/test_execution_dag.py`, `tests/test_dag_backup.py`, `tests/test_session_map.py`, `tests/test_e2e_critic_gae.py`) to the new API.
- New unit tests for SegmentSplitter.

### Phase 1b/2 (second PR)
- `agents/team_rollout.py`: new module (TeamRolloutCoordinator + TeamConfig + AgentRoleSpec + TeamEdgeSpec).
- `RLBridgeClient` concrete adapter mapping `session_id` → `interaction_id`.
- Wire `team_config` param into `TreeSearchGroupedRolloutWorkflow.__init__` and `arun_episode`.
- Verifier + `RLSessionRewardWriter` integration in the coordinator.
- End-to-end multi-agent test with a 3-role config (planner→worker→synthesizer), verifying reward flows along the unified parent chain across agents.

## 11. Test strategy

- **Torch-free unit tests** (most of the new code):
  - SegmentSplitter with synthetic Node lists + events; verify segment boundaries, parent_node_id chain (within-run, cross-agent delegation, cross-agent completion), team env snapshot stamping.
  - `insert_super_batch` + dual-level lookup (`get_super_node`, `get_node`).
  - `backup_episode_terminal` across SuperNode and agent boundaries (causal flattening).
  - `dag_to_supernodes` / `supernodes_to_dag` round-trip + edge symmetry + topo invariant.
  - Coordinator with mock role workflows + mock verifier + mock rl_bridge; verify DAG construction, session binding, verifier invocation, backup invocation, batched tensor dict output.
- **Integration tests** (`pytest.importorskip("torch")` for tensor paths):
  - End-to-end team episode with a 3-role config (planner→worker→synthesizer); verify credit flows along the unified parent chain and per-turn `parent_node_id` walks cross SuperNode/agent boundaries.
  - Leaf-SuperNode fallback when `_communication_events` is absent.
- **Existing tests preserved**: today's `test_tree_store.py`, `test_advantage.py`, etc. run unchanged on the single-agent path.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `parent_node_id` semantic shift ("episode-internal predecessor" → "causal predecessor") surprises callers | Document on `Node.parent_node_id`; audit all readers (today: `_backup_path`, `_node_parent_id`, `_backup_inserted_episodes`); the `visited` guard in `_backup_path` already prevents cycles from malformed chains. |
| Cross-agent `parent_node_id` link requires the SegmentSplitter to know the source agent's terminal `node_id` at split time | Coordinator passes `start_parent_node_id` to the splitter (the source run's terminal `node_id` for delegation, or the completion source's terminal for completion edges). |
| `session_id` → `interaction_id` mapping is unspecified in the proxy today | Phase 1b/2 implements the `RLBridgeClient` adapter; if the proxy lacks a clean session→terminal-interaction lookup, the adapter uses the session's last interaction (or requires the verifier to return interaction_ids). |
| Node torch-lazy refactor breaks existing tensor-typed code paths | Field types become `Any`; `_node_to_tensor_dict` lazy-imports torch; existing tests with `pytest.importorskip("torch")` continue to cover the tensor paths. |
| Two-phase landing leaves Phase 1a in a state where SuperNode exists but is unused by multi-agent path | Phase 1a wraps single-agent Nodes in a leaf SuperNode so the data model is exercised; Phase 1b/2 wires the coordinator. Intermediate state is green and useful. |

## 13. Glossary

- **SuperNode**: communication-bounded segment of one agent's action sequence; carries DAG topology, comm-event provenance, RL session binding, team env snapshot, and `list[Node]` turns.
- **Node**: one assistant turn (`input_ids`, `loss_mask`, `logprobs`, etc.); torch-lazy after refactor.
- **CommunicationEvent**: one delegation/mention/completion emitted by an agent run, tagged with the turn that performed it.
- **Causal flattening**: encoding the full DAG causal order into the Node-level `parent_node_id` chain, so reward backup walks one unified chain.
- **TeamFrontierSnapshot**: snapshot of the entire team's sandbox/issue state at a SuperNode close time.
- **session_id**: proxy-layer identifier for one agent run; shared across all SuperNodes of that run.

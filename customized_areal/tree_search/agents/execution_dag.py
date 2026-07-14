"""Agent-execution DAG model for multi-agent RL training in Multica.

Multica runs many agents that collaborate: a planner files sub-issues, workers
pick them up, agents mention/trigger one another, and children report back to
parents. The execution graph is therefore a **DAG of agent runs** (not a tree),
with multiple agents potentially in flight at once.

This module is the in-memory contract for that DAG. It is intentionally
**torch-free** so it can be unit-tested and reused without the training stack.

Nodes are *agent runs* (one task executed by one agent on one issue). Edges are
the causal/data dependencies between runs:

- ``DELEGATION``  parent issue -> sub-issue (fan-out): a planner spawns workers.
- ``MENTION``     one run mentions/triggers another agent's run (peer edge).
- ``COMPLETION``  a child run completes and notifies its parent (fan-in).

Reward backs up over this DAG (see ``dag.backup``); the verifier assigns credit
per node, and structural backup distributes the terminal reward along edges.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum


class EdgeType(StrEnum):
    """Typed edges between agent-run nodes in the execution DAG."""

    DELEGATION = "delegation"  # parent issue -> sub-issue (fan-out)
    MENTION = "mention"  # one run mentions/triggers another (peer)
    COMPLETION = "completion"  # child run completes -> parent (fan-in)
    BRANCH = "branch"  # a fork from a closed segment's checkpoint (tree search)


@dataclass
class SuperNode:
    """One communication-bounded segment of one agent's action sequence.

    Bounded by communication events (delegation, mention, completion).
    ``nodes`` is the contiguous run of turns within this segment; the LAST
    node is the segment's terminal -- the turn that performed the closing
    communication event (or the run's final turn for a leaf segment).

    ``sandbox_ids`` is a snapshot of the entire team's sandbox state at the
    moment the closing event fired (one sandbox_id per team agent). Phase 3
    SuperNode-level branching forks this list + the Multica issue subtree.
    """

    # Identity (UUID4 once assembled; user-supplied in tests)
    node_id: str

    # Agent context
    agent_id: str
    issue_id: str
    task_id: str

    # Communication-event provenance (which event closed this segment)
    closing_event: EdgeType | None = None  # None for leaf segments
    closing_event_target: str | None = None  # the other SuperNode's node_id

    # RL session assigned at /rl/start_session time. One session_id per agent
    # run, shared across all SuperNodes of that run. None until bound.
    session_id: str | None = None

    # Branch frontier: the backend env_id captured on this segment's terminal
    # turn. A branch is created via ``env_dispatch(mode="branch", env_id=...)``.
    env_id: str | None = None

    # Reward bookkeeping (set by the verifier / backup). Plain floats so the
    # DAG stays torch-free; the training Node carries tensors.
    process_reward: float = 0.0
    outcome_reward: float = 0.0

    # Critic value V_{t+1} (Phase 3, out of scope here). None until scored.
    value: float | None = None

    # MCTS visit count for tree-search branching: how many branches have backed
    # up a return through this segment (the fork/checkpoint). Updated by
    # ``branch_backup``; 0 for non-fork segments.
    visit_count: int = 0

    # Team environment snapshot at close time.
    sandbox_ids: list[str] = field(default_factory=list)
    issue_snapshot_id: str | None = None
    env_state: dict = field(default_factory=dict)

    # The turns within this segment (Node is torch-lazy so this dataclass
    # imports cleanly without torch).
    nodes: list = field(default_factory=list)

    # Free-form metadata
    metadata: dict = field(default_factory=dict)

    # -- linear trajectory (filled by codec / assembler) ----------------

    completion_index: int | None = None
    completion_time: float | None = None

    # -- DAG edges (typed, both directions; filled by assembler) --------

    incoming_edges: tuple = ()
    outgoing_edges: tuple = ()

    @property
    def terminal_node(self):
        """The segment's last node -- the closing-event turn or run-final."""
        return self.nodes[-1] if self.nodes else None

    @property
    def branch_node_id(self) -> str | None:
        """node_id of the terminal node (for branching keys)."""
        t = self.terminal_node
        return t.node_id if t is not None else None

    def to_dict(self) -> dict:
        """Emit a plain dict with EdgeType values stringified and tuples as lists.

        ``nodes`` items are serialized via their own ``to_dict()`` when present;
        otherwise the raw object is passed through (caller's responsibility
        for JSON-safety). Edge tuples become ``[[node_id, edge_type_str], ...]``.
        """
        return {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "issue_id": self.issue_id,
            "task_id": self.task_id,
            "closing_event": self.closing_event.value if self.closing_event else None,
            "closing_event_target": self.closing_event_target,
            "session_id": self.session_id,
            "completion_index": self.completion_index,
            "completion_time": self.completion_time,
            "incoming_edges": [[s, t.value] for s, t in self.incoming_edges],
            "outgoing_edges": [[d, t.value] for d, t in self.outgoing_edges],
            "env_id": self.env_id,
            "value": self.value,
            "process_reward": self.process_reward,
            "outcome_reward": self.outcome_reward,
            "visit_count": self.visit_count,
            "sandbox_ids": list(self.sandbox_ids),
            "issue_snapshot_id": self.issue_snapshot_id,
            "env_state": dict(self.env_state),
            "nodes": [n.to_dict() if hasattr(n, "to_dict") else n for n in self.nodes],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SuperNode:
        """Exact inverse of :meth:`to_dict`. Raises ``DAGError`` on bad input."""
        try:
            return cls(
                node_id=d["node_id"],
                agent_id=d["agent_id"],
                issue_id=d["issue_id"],
                task_id=d["task_id"],
                closing_event=cls._coerce_edge_type(d.get("closing_event")),
                closing_event_target=d.get("closing_event_target"),
                session_id=d.get("session_id"),
                completion_index=d.get("completion_index"),
                completion_time=d.get("completion_time"),
                incoming_edges=cls._coerce_edges(d.get("incoming_edges", ())),
                outgoing_edges=cls._coerce_edges(d.get("outgoing_edges", ())),
                env_id=d.get("env_id"),
                value=d.get("value"),
                process_reward=d.get("process_reward", 0.0),
                outcome_reward=d.get("outcome_reward", 0.0),
                visit_count=d.get("visit_count", 0),
                sandbox_ids=list(d.get("sandbox_ids", [])),
                issue_snapshot_id=d.get("issue_snapshot_id"),
                env_state=dict(d.get("env_state", {})),
                nodes=list(d.get("nodes", [])),
                metadata=dict(d.get("metadata", {})),
            )
        except KeyError as exc:
            raise DAGError(
                f"SuperNode.from_dict missing required field: {exc}"
            ) from exc

    @staticmethod
    def _coerce_edge_type(raw) -> EdgeType | None:
        if raw is None:
            return None
        if isinstance(raw, EdgeType):
            return raw
        try:
            return EdgeType(raw)
        except ValueError as exc:
            raise DAGError(f"unknown EdgeType: {raw!r}") from exc

    @staticmethod
    def _coerce_edges(raw) -> tuple:
        out = []
        for item in raw:
            nid, etype = item[0], item[1]
            if not isinstance(etype, EdgeType):
                try:
                    etype = EdgeType(etype)
                except ValueError as exc:
                    raise DAGError(f"unknown EdgeType: {etype!r}") from exc
            out.append((nid, etype))
        return tuple(out)


@dataclass(frozen=True)
class Edge:
    """A directed dependency from ``src`` run to ``dst`` run."""

    src: str  # node_id
    dst: str  # node_id
    type: EdgeType
    # BRANCH provenance: the segment + checkpoint the branch forked from. None
    # for non-BRANCH edges and legacy edges without provenance.
    branch_from_segment_id: str | None = None
    branch_from_checkpoint_id: str | None = None


class DAGError(ValueError):
    """Raised when the DAG is malformed (unknown event, cycle, etc.)."""


class ExecutionDAG:
    """Directed acyclic graph of agent runs.

    Edges point from a *cause* run to an *effect* run (parent -> child for
    delegation, child -> parent for completion). The graph rejects edges that
    reference unknown events and detects cycles on demand.
    """

    def __init__(self) -> None:
        self._events: dict[str, SuperNode] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}

    # -- construction ----------------------------------------------------

    def add_event(self, event: SuperNode) -> SuperNode:
        if event.node_id in self._events:
            raise DAGError(f"duplicate event_id: {event.node_id!r}")
        self._events[event.node_id] = event
        self._out.setdefault(event.node_id, [])
        self._in.setdefault(event.node_id, [])
        return event

    def add_edge(
        self,
        src: str,
        dst: str,
        type: EdgeType,
        *,
        branch_from_segment_id: str | None = None,
        branch_from_checkpoint_id: str | None = None,
    ) -> Edge:
        if src not in self._events:
            raise DAGError(f"unknown src event: {src!r}")
        if dst not in self._events:
            raise DAGError(f"unknown dst event: {dst!r}")
        if src == dst:
            raise DAGError(f"self-loop not allowed: {src!r}")
        edge = Edge(
            src=src,
            dst=dst,
            type=type,
            branch_from_segment_id=branch_from_segment_id,
            branch_from_checkpoint_id=branch_from_checkpoint_id,
        )
        # Idempotent: don't double-add an identical edge.
        if edge in self._edges:
            return edge
        self._edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)
        return edge

    # -- accessors -------------------------------------------------------

    def __contains__(self, event_id: object) -> bool:
        return event_id in self._events

    def __len__(self) -> int:
        return len(self._events)

    def get(self, event_id: str) -> SuperNode:
        try:
            return self._events[event_id]
        except KeyError as exc:
            raise DAGError(f"unknown event: {event_id!r}") from exc

    @property
    def events(self) -> list[SuperNode]:
        return list(self._events.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def event_ids(self) -> list[str]:
        return list(self._events.keys())

    def parents(self, event_id: str) -> list[SuperNode]:
        """Runs this run causally depends on (incoming edges)."""
        if event_id not in self._events:
            raise DAGError(f"unknown event: {event_id!r}")
        return [self._events[e.src] for e in self._in[event_id]]

    def children(self, event_id: str) -> list[SuperNode]:
        """Runs that causally depend on this run (outgoing edges)."""
        if event_id not in self._events:
            raise DAGError(f"unknown event: {event_id!r}")
        return [self._events[e.dst] for e in self._out[event_id]]

    def in_degree(self, event_id: str) -> int:
        return len(self._in[event_id])

    def out_degree(self, event_id: str) -> int:
        return len(self._out[event_id])

    # -- RL session mapping ----------------------------------------------

    def set_session_id(self, event_id: str, session_id: str) -> None:
        """Bind an RL ``session_id`` to a run (called at ``rl_start_session``)."""
        self.get(event_id).session_id = session_id

    def session_map(self) -> dict[str, str | None]:
        """Return ``{node_id: session_id}`` for every run in the DAG.

        Session-less runs (no ``rl_start_session`` yet) map to ``None``. The
        verifier agent uses this to address per-agent rewards by ``session_id``.
        """
        return {eid: ev.session_id for eid, ev in self._events.items()}

    # -- serialization ---------------------------------------------------

    def to_records(self) -> tuple[list[dict], list[dict]]:
        """Serialize to ``(runs, edges)`` records for a checkpoint round-trip.

        The inverse of :meth:`from_records` with explicit edges. Every
        ``SuperNode`` field (including ``session_id`` and the ``env_id`` branch
        frontier) is emitted so the DAG can be reconstructed verbatim.
        """
        from dataclasses import asdict

        runs = [asdict(ev) for ev in self._events.values()]
        edges = []
        for e in self._edges:
            rec = {"src": e.src, "dst": e.dst, "type": e.type.value}
            if e.branch_from_segment_id is not None:
                rec["branch_from_segment_id"] = e.branch_from_segment_id
            if e.branch_from_checkpoint_id is not None:
                rec["branch_from_checkpoint_id"] = e.branch_from_checkpoint_id
            edges.append(rec)
        return runs, edges

    # -- structural queries ---------------------------------------------

    def roots(self) -> list[SuperNode]:
        """Events with no incoming edges (entry points of the DAG)."""
        return [ev for eid, ev in self._events.items() if not self._in[eid]]

    def leaves(self) -> list[SuperNode]:
        """Events with no outgoing edges (terminal runs)."""
        return [ev for eid, ev in self._events.items() if not self._out[eid]]

    def fork_events(self) -> list[SuperNode]:
        """Fan-out points: a run that spawns/triggers >= 2 downstream runs."""
        return [ev for eid, ev in self._events.items() if len(self._out[eid]) >= 2]

    def join_events(self) -> list[SuperNode]:
        """Fan-in points: a run fed by >= 2 upstream runs.

        These are exactly the events where the verifier must assign per-agent
        credit explicitly (decision 8), since there is no fixed aggregation
        rule across the contributing runs.
        """
        return [ev for eid, ev in self._events.items() if len(self._in[eid]) >= 2]

    def topological_order(self) -> list[SuperNode]:
        """Kahn's algorithm. Raises ``DAGError`` if the graph has a cycle."""
        indeg = {eid: len(self._in[eid]) for eid in self._events}
        queue: deque[str] = deque(sorted(eid for eid, d in indeg.items() if d == 0))
        order: list[SuperNode] = []
        while queue:
            eid = queue.popleft()
            order.append(self._events[eid])
            for edge in self._out[eid]:
                indeg[edge.dst] -= 1
                if indeg[edge.dst] == 0:
                    queue.append(edge.dst)
        if len(order) != len(self._events):
            raise DAGError("execution DAG contains a cycle")
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
            return True
        except DAGError:
            return False

    def ancestors(self, event_id: str) -> set[str]:
        """All event_ids that transitively precede ``event_id``."""
        if event_id not in self._events:
            raise DAGError(f"unknown event: {event_id!r}")
        seen: set[str] = set()
        stack = [e.src for e in self._in[event_id]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(e.src for e in self._in[cur])
        return seen

    def descendants(self, event_id: str) -> set[str]:
        """All event_ids that transitively follow ``event_id``."""
        if event_id not in self._events:
            raise DAGError(f"unknown event: {event_id!r}")
        seen: set[str] = set()
        stack = [e.dst for e in self._out[event_id]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(e.dst for e in self._out[cur])
        return seen

    def iter_topo(self) -> Iterator[SuperNode]:
        yield from self.topological_order()

    # -- builders --------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        runs: Iterable[dict],
        edges: Iterable[dict] | None = None,
    ) -> ExecutionDAG:
        """Build a DAG from plain run/edge records.

        ``runs`` items require ``node_id``/``agent_id``/``issue_id``/``task_id``;
        any other ``SuperNode`` field is optional. ``edges`` items require
        ``src``, ``dst``, and ``type`` (an ``EdgeType`` or its string value).

        If ``edges`` is omitted, delegation edges are inferred from a
        ``parent_issue_id`` field on the run records (a run on a sub-issue
        depends on the run that owns the parent issue).
        """
        dag = cls()
        known_fields = SuperNode.__dataclass_fields__.keys()
        issue_to_node: dict[str, str] = {}
        for rec in runs:
            kwargs = {k: rec[k] for k in known_fields if k in rec}
            event = SuperNode(**kwargs)
            dag.add_event(event)
            issue_to_node.setdefault(event.issue_id, event.node_id)

        if edges is not None:
            for e in edges:
                etype = e["type"]
                if not isinstance(etype, EdgeType):
                    etype = EdgeType(etype)
                dag.add_edge(
                    e["src"],
                    e["dst"],
                    etype,
                    branch_from_segment_id=e.get("branch_from_segment_id"),
                    branch_from_checkpoint_id=e.get("branch_from_checkpoint_id"),
                )
            return dag

        # Infer delegation edges from parent_issue_id on the records.
        for rec in runs if isinstance(runs, (list, tuple)) else []:
            parent_issue = rec.get("parent_issue_id")
            if parent_issue and parent_issue in issue_to_node:
                src = issue_to_node[parent_issue]
                dst = rec["node_id"]
                if src != dst:
                    dag.add_edge(src, dst, EdgeType.DELEGATION)
        return dag

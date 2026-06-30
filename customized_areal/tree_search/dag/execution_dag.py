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


@dataclass
class AgentRunNode:
    """A single agent run in the execution DAG.

    A run is one Multica task executed by one agent on one issue, bounded by
    the agent lifecycle (claim -> complete). ``branch_seq`` records the
    ``task_message.seq`` step this run is allowed to branch at when selected as
    a branch candidate; the ``branch_*`` provenance fields are populated once a
    branch is actually materialized (forked issue + sandbox snapshot).
    """

    node_id: str  # globally unique id for this agent run (typically task_id)
    agent_id: str
    issue_id: str
    task_id: str

    # Branch boundary: the (task_id, seq) step this run can be forked at.
    branch_seq: int | None = None

    # Fork provenance (filled when this node becomes a branch source).
    branch_issue_id: str | None = None
    branch_env_snapshot_id: str | None = None

    # Reward bookkeeping (set by the verifier / backup). Kept here as plain
    # floats so the DAG stays torch-free; the training Node carries tensors.
    process_reward: float = 0.0
    outcome_reward: float = 0.0

    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """A directed dependency from ``src`` run to ``dst`` run."""

    src: str  # node_id
    dst: str  # node_id
    type: EdgeType


class DAGError(ValueError):
    """Raised when the DAG is malformed (unknown node, cycle, etc.)."""


class ExecutionDAG:
    """Directed acyclic graph of agent runs.

    Edges point from a *cause* run to an *effect* run (parent -> child for
    delegation, child -> parent for completion). The graph rejects edges that
    reference unknown nodes and detects cycles on demand.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AgentRunNode] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}

    # -- construction ----------------------------------------------------

    def add_node(self, node: AgentRunNode) -> AgentRunNode:
        if node.node_id in self._nodes:
            raise DAGError(f"duplicate node_id: {node.node_id!r}")
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])
        return node

    def add_edge(self, src: str, dst: str, type: EdgeType) -> Edge:
        if src not in self._nodes:
            raise DAGError(f"unknown src node: {src!r}")
        if dst not in self._nodes:
            raise DAGError(f"unknown dst node: {dst!r}")
        if src == dst:
            raise DAGError(f"self-loop not allowed: {src!r}")
        edge = Edge(src=src, dst=dst, type=type)
        # Idempotent: don't double-add an identical edge.
        if edge in self._edges:
            return edge
        self._edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)
        return edge

    # -- accessors -------------------------------------------------------

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> AgentRunNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise DAGError(f"unknown node: {node_id!r}") from exc

    @property
    def nodes(self) -> list[AgentRunNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def parents(self, node_id: str) -> list[AgentRunNode]:
        """Runs this run causally depends on (incoming edges)."""
        if node_id not in self._nodes:
            raise DAGError(f"unknown node: {node_id!r}")
        return [self._nodes[e.src] for e in self._in[node_id]]

    def children(self, node_id: str) -> list[AgentRunNode]:
        """Runs that causally depend on this run (outgoing edges)."""
        if node_id not in self._nodes:
            raise DAGError(f"unknown node: {node_id!r}")
        return [self._nodes[e.dst] for e in self._out[node_id]]

    def in_degree(self, node_id: str) -> int:
        return len(self._in[node_id])

    def out_degree(self, node_id: str) -> int:
        return len(self._out[node_id])

    # -- structural queries ---------------------------------------------

    def roots(self) -> list[AgentRunNode]:
        """Nodes with no incoming edges (entry points of the DAG)."""
        return [n for nid, n in self._nodes.items() if not self._in[nid]]

    def leaves(self) -> list[AgentRunNode]:
        """Nodes with no outgoing edges (terminal runs)."""
        return [n for nid, n in self._nodes.items() if not self._out[nid]]

    def fork_nodes(self) -> list[AgentRunNode]:
        """Fan-out points: a run that spawns/triggers >= 2 downstream runs."""
        return [n for nid, n in self._nodes.items() if len(self._out[nid]) >= 2]

    def join_nodes(self) -> list[AgentRunNode]:
        """Fan-in points: a run fed by >= 2 upstream runs.

        These are exactly the nodes where the verifier must assign per-agent
        credit explicitly (decision 8), since there is no fixed aggregation
        rule across the contributing runs.
        """
        return [n for nid, n in self._nodes.items() if len(self._in[nid]) >= 2]

    def topological_order(self) -> list[AgentRunNode]:
        """Kahn's algorithm. Raises ``DAGError`` if the graph has a cycle."""
        indeg = {nid: len(self._in[nid]) for nid in self._nodes}
        queue: deque[str] = deque(sorted(nid for nid, d in indeg.items() if d == 0))
        order: list[AgentRunNode] = []
        while queue:
            nid = queue.popleft()
            order.append(self._nodes[nid])
            for edge in self._out[nid]:
                indeg[edge.dst] -= 1
                if indeg[edge.dst] == 0:
                    queue.append(edge.dst)
        if len(order) != len(self._nodes):
            raise DAGError("execution DAG contains a cycle")
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
            return True
        except DAGError:
            return False

    def ancestors(self, node_id: str) -> set[str]:
        """All node_ids that transitively precede ``node_id``."""
        if node_id not in self._nodes:
            raise DAGError(f"unknown node: {node_id!r}")
        seen: set[str] = set()
        stack = [e.src for e in self._in[node_id]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(e.src for e in self._in[cur])
        return seen

    def descendants(self, node_id: str) -> set[str]:
        """All node_ids that transitively follow ``node_id``."""
        if node_id not in self._nodes:
            raise DAGError(f"unknown node: {node_id!r}")
        seen: set[str] = set()
        stack = [e.dst for e in self._out[node_id]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(e.dst for e in self._out[cur])
        return seen

    def iter_topo(self) -> Iterator[AgentRunNode]:
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
        any other ``AgentRunNode`` field is optional. ``edges`` items require
        ``src``, ``dst``, and ``type`` (an ``EdgeType`` or its string value).

        If ``edges`` is omitted, delegation edges are inferred from a
        ``parent_issue_id`` field on the run records (a run on a sub-issue
        depends on the run that owns the parent issue).
        """
        dag = cls()
        known_fields = AgentRunNode.__dataclass_fields__.keys()
        issue_to_node: dict[str, str] = {}
        for rec in runs:
            kwargs = {k: rec[k] for k in known_fields if k in rec}
            node = AgentRunNode(**kwargs)
            dag.add_node(node)
            issue_to_node.setdefault(node.issue_id, node.node_id)

        if edges is not None:
            for e in edges:
                etype = e["type"]
                if not isinstance(etype, EdgeType):
                    etype = EdgeType(etype)
                dag.add_edge(e["src"], e["dst"], etype)
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

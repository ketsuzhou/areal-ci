# DAG ↔ Linear Event Codec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical, JSON-serializable linear `Event` type plus a pure
bidirectional codec (`dag_to_events` / `events_to_dag` / `replay_prefix_for`) between
the agent-execution DAG and its completion-ordered linear trajectory.

**Architecture:** Two new torch-free, I/O-free modules in
`customized_areal/tree_search/dag/`: `event_model.py` (the `Event` dataclass +
serialization + a derived `message_timeline` view) and `event_codec.py` (the
forward/reverse conversion). The existing `gae.events_from_nodes` is refactored into a
thin projection over `Event` with byte-for-byte identical output (parity-tested);
`critic_observation.build_critic_observations` is left unchanged and validated against
the canonical model via a parity test. `ExecutionDAG.to_records`/`from_records`,
`BranchMaterializer`, `environment.py`, and the `AgentRunNode` field set are untouched.

**Tech Stack:** Python 3.12 (`from __future__ import annotations`, `StrEnum`, frozen
dataclasses), pytest. No torch. Lint with the project ruff.

**Spec:** `docs/superpowers/specs/2026-06-29-dag-event-codec-design.md`

**Environment notes (this repo):**

- Run tests with: `.venv-test/bin/python -m pytest <path> -q`
- Lint/format with the project ruff binary:
  `RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff`
- All work happens under `/workspaces/leagent/backend/areal/`. Paths below are relative
  to that directory.
- Commit with the repo's author convention (no git config change):
  `git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "..."`

______________________________________________________________________

## File Structure

| File                                                         | Responsibility                                                        |
| ------------------------------------------------------------ | --------------------------------------------------------------------- |
| `customized_areal/tree_search/dag/event_model.py` (new)      | `Event` dataclass, `to_dict`/`from_dict`, `message_timeline`          |
| `customized_areal/tree_search/dag/event_codec.py` (new)      | `dag_to_events`, `events_to_dag`, `replay_prefix_for`, `ReplayPrefix` |
| `customized_areal/tree_search/dag/gae.py` (modify)           | `events_from_nodes` → thin projection over `Event`                    |
| `customized_areal/tree_search/dag/__init__.py` (modify)      | export new public names (stays torch-free)                            |
| `customized_areal/tree_search/dag/test_event_model.py` (new) | model + serialization + timeline tests                                |
| `customized_areal/tree_search/dag/test_event_codec.py` (new) | forward/reverse/round-trip/error/replay + parity tests                |

**Conventions to follow** (from the existing package):

- Every module starts with a docstring then `from __future__ import annotations`.
- Reuse `DAGError` (from `execution_dag`) for all conversion errors. Do not introduce a
  new exception type.
- Use `collections.abc.Sequence` for sequence type hints (ruff UP035).
- No quoted annotations (ruff UP037).

______________________________________________________________________

## Task 1: `Event` dataclass + serialization

**Files:**

- Create: `customized_areal/tree_search/dag/event_model.py`

- Test: `customized_areal/tree_search/dag/test_event_model.py`

- [ ] **Step 1: Write the failing test**

Create `customized_areal/tree_search/dag/test_event_model.py`:

```python
"""Tests for the canonical linear Event model (DAG trajectory log).

Torch-free.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.event_model import Event
from customized_areal.tree_search.dag.execution_dag import DAGError, EdgeType


def _sample_event() -> Event:
    return Event(
        node_id="C0",
        agent_id="coder",
        issue_id="iss-2",
        task_id="task-2",
        completion_index=2,
        incoming_edges=(("O0", EdgeType.DELEGATION), ("R0", EdgeType.MENTION)),
        outgoing_edges=(("T0", EdgeType.DELEGATION),),
        session_id="sess-2",
        completion_time=3.0,
        branch_seq=5,
        branch_issue_id="iss-2-fork",
        branch_env_snapshot_id="snap-2",
        value=0.6,
        process_reward=0.1,
        outcome_reward=0.0,
        messages=({"role": "assistant", "content": "draft"},),
        metadata={"k": "v"},
    )


def test_to_dict_is_json_safe() -> None:
    d = _sample_event().to_dict()
    # EdgeType serialized to its string value; tuples to lists.
    assert d["incoming_edges"] == [["O0", "delegation"], ["R0", "mention"]]
    assert d["outgoing_edges"] == [["T0", "delegation"]]
    assert isinstance(d["messages"], list)
    assert d["completion_index"] == 2


def test_from_dict_to_dict_round_trip_equal() -> None:
    e = _sample_event()
    assert Event.from_dict(e.to_dict()) == e


def test_from_dict_coerces_edge_type_strings() -> None:
    e = _sample_event()
    raw = e.to_dict()
    rebuilt = Event.from_dict(raw)
    assert rebuilt.incoming_edges[0] == ("O0", EdgeType.DELEGATION)
    assert isinstance(rebuilt.incoming_edges[0][1], EdgeType)


def test_from_dict_unknown_edge_type_raises() -> None:
    raw = _sample_event().to_dict()
    raw["incoming_edges"] = [["O0", "not-a-real-type"]]
    with pytest.raises(DAGError):
        Event.from_dict(raw)


def test_from_dict_missing_required_field_raises() -> None:
    raw = _sample_event().to_dict()
    del raw["node_id"]
    with pytest.raises(DAGError):
        Event.from_dict(raw)


def test_defaults_are_empty() -> None:
    e = Event(
        node_id="n", agent_id="a", issue_id="i", task_id="t", completion_index=0
    )
    assert e.incoming_edges == ()
    assert e.outgoing_edges == ()
    assert e.messages == ()
    assert e.value is None
    assert e.process_reward == 0.0
    assert e.metadata == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_model.py -q`
Expected: FAIL with `ModuleNotFoundError: ... event_model`.

- [ ] **Step 3: Write minimal implementation**

Create `customized_areal/tree_search/dag/event_model.py`:

```python
"""Canonical linear-event model for the agent-execution DAG (trajectory log).

This is the single source of truth for the *linear* (completion-ordered) view
of an :class:`ExecutionDAG`. One ``Event`` == one DAG node (a completed agent
turn). Forward/reverse conversion lives in ``event_codec``; this module holds
only the data definition + serialization + the derived message-timeline view.

Framework B: the linear order is the global completion order, which is always a
topological order of the causal DAG ("prefix = cut" -- see
``CRITIC_GAE_INTEGRATION.md``). The codec persists/reconstructs that order via
the explicit ``completion_index``.

Torch-free and I/O-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from customized_areal.tree_search.dag.execution_dag import DAGError, EdgeType

EdgeRef = tuple[str, EdgeType]


@dataclass(frozen=True)
class Event:
    """One completed turn in the global completion-ordered trajectory.

    ``completion_index`` is the authoritative linear position (0-based, dense).
    Edges are stored in BOTH directions for fidelity; ``event_codec`` validates
    their symmetry on decode. ``messages`` is the turn's transcript slice, with
    the turn's own output expected as the last element (see ``message_timeline``).
    """

    # identity
    node_id: str
    agent_id: str
    issue_id: str
    task_id: str
    # ordering (authoritative)
    completion_index: int
    # structure (bidirectional; symmetry-validated on decode)
    incoming_edges: tuple[EdgeRef, ...] = ()
    outgoing_edges: tuple[EdgeRef, ...] = ()
    # identity (optional)
    session_id: str | None = None
    # ordering provenance (never used for ordering decisions)
    completion_time: float | None = None
    # branch provenance
    branch_seq: int | None = None
    branch_issue_id: str | None = None
    branch_env_snapshot_id: str | None = None
    # RL signals + payload
    value: float | None = None
    process_reward: float = 0.0
    outcome_reward: float = 0.0
    messages: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Emit a plain JSON-safe dict (EdgeType -> str, tuples -> lists)."""
        return {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "issue_id": self.issue_id,
            "task_id": self.task_id,
            "completion_index": self.completion_index,
            "incoming_edges": [[s, t.value] for s, t in self.incoming_edges],
            "outgoing_edges": [[d, t.value] for d, t in self.outgoing_edges],
            "session_id": self.session_id,
            "completion_time": self.completion_time,
            "branch_seq": self.branch_seq,
            "branch_issue_id": self.branch_issue_id,
            "branch_env_snapshot_id": self.branch_env_snapshot_id,
            "value": self.value,
            "process_reward": self.process_reward,
            "outcome_reward": self.outcome_reward,
            "messages": [dict(m) for m in self.messages],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        """Exact inverse of :meth:`to_dict`. Raises ``DAGError`` on bad input."""
        try:
            return cls(
                node_id=d["node_id"],
                agent_id=d["agent_id"],
                issue_id=d["issue_id"],
                task_id=d["task_id"],
                completion_index=d["completion_index"],
                incoming_edges=cls._coerce_edges(d.get("incoming_edges", ())),
                outgoing_edges=cls._coerce_edges(d.get("outgoing_edges", ())),
                session_id=d.get("session_id"),
                completion_time=d.get("completion_time"),
                branch_seq=d.get("branch_seq"),
                branch_issue_id=d.get("branch_issue_id"),
                branch_env_snapshot_id=d.get("branch_env_snapshot_id"),
                value=d.get("value"),
                process_reward=d.get("process_reward", 0.0),
                outcome_reward=d.get("outcome_reward", 0.0),
                messages=tuple(dict(m) for m in d.get("messages", ())),
                metadata=dict(d.get("metadata", {})),
            )
        except KeyError as exc:
            raise DAGError(f"Event.from_dict missing required field: {exc}") from exc

    @staticmethod
    def _coerce_edges(raw: Iterable) -> tuple[EdgeRef, ...]:
        out: list[EdgeRef] = []
        for item in raw:
            nid, etype = item[0], item[1]
            if not isinstance(etype, EdgeType):
                try:
                    etype = EdgeType(etype)
                except ValueError as exc:
                    raise DAGError(f"unknown EdgeType: {etype!r}") from exc
            out.append((nid, etype))
        return tuple(out)


def message_timeline(events: Sequence[Event]) -> list[dict]:
    """Derived message-level view across events in completion order.

    Concatenates each event's ``messages`` payload in ``completion_index``
    order. The LAST message of each event's payload is the turn's own output and
    is tagged with the event's ``node_id`` (so the critic frontier builder can
    detect turn outputs); earlier payload messages (context / tool / user) pass
    through untagged. Tagged copies are emitted; the source payloads are not
    mutated.
    """
    ordered = sorted(events, key=lambda e: e.completion_index)
    timeline: list[dict] = []
    for ev in ordered:
        msgs = list(ev.messages)
        for i, m in enumerate(msgs):
            tagged = dict(m)
            if i == len(msgs) - 1:
                tagged["node_id"] = ev.node_id
            timeline.append(tagged)
    return timeline


__all__ = ["EdgeRef", "Event", "message_timeline"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_model.py -q`
Expected: PASS (6 tests; `message_timeline` is covered in Task 2).

- [ ] **Step 5: Lint and commit**

```bash
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/event_model.py customized_areal/tree_search/dag/test_event_model.py
$RUFF format customized_areal/tree_search/dag/event_model.py customized_areal/tree_search/dag/test_event_model.py
git add customized_areal/tree_search/dag/event_model.py customized_areal/tree_search/dag/test_event_model.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "feat(dag): add canonical linear Event model + serialization"
```

______________________________________________________________________

## Task 2: `message_timeline` derived view

**Files:**

- Modify: `customized_areal/tree_search/dag/event_model.py` (already contains
  `message_timeline` from Task 1 — this task adds its tests)

- Test: `customized_areal/tree_search/dag/test_event_model.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/dag/test_event_model.py`:

```python
from customized_areal.tree_search.dag.event_model import message_timeline


def test_message_timeline_orders_by_completion_index_and_tags_output() -> None:
    e1 = Event(
        node_id="O0", agent_id="orch", issue_id="i", task_id="t",
        completion_index=1,
        messages=({"role": "user", "content": "ctx"},
                  {"role": "assistant", "content": "plan"}),
    )
    e0 = Event(
        node_id="seed", agent_id="orch", issue_id="i", task_id="t",
        completion_index=0,
        messages=({"role": "assistant", "content": "boot"},),
    )
    timeline = message_timeline([e1, e0])  # deliberately out of order
    # Sorted by completion_index: e0 then e1.
    assert [m["content"] for m in timeline] == ["boot", "ctx", "plan"]
    # Last message of each event's payload is tagged with node_id.
    assert timeline[0]["node_id"] == "seed"
    assert "node_id" not in timeline[1]  # context message, untagged
    assert timeline[2]["node_id"] == "O0"


def test_message_timeline_does_not_mutate_source() -> None:
    src = {"role": "assistant", "content": "x"}
    e = Event(
        node_id="n", agent_id="a", issue_id="i", task_id="t",
        completion_index=0, messages=(src,),
    )
    message_timeline([e])
    assert "node_id" not in src  # original payload untouched
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_model.py -q`
Expected: These two tests PASS immediately (the implementation already exists from Task
1). If `message_timeline` were missing they would fail with `ImportError`. This task
documents/locks the behavior.

- [ ] **Step 3: (No implementation needed)**

`message_timeline` was implemented in Task 1. If the tests fail, fix `message_timeline`
until they pass.

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/dag/test_event_model.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "test(dag): lock message_timeline ordering + output tagging"
```

______________________________________________________________________

## Task 3: `dag_to_events` (forward path)

**Files:**

- Create: `customized_areal/tree_search/dag/event_codec.py`

- Test: `customized_areal/tree_search/dag/test_event_codec.py`

- [ ] **Step 1: Write the failing test**

Create `customized_areal/tree_search/dag/test_event_codec.py`:

```python
"""Tests for the bidirectional DAG <-> linear Event codec.

The multi-lane fixture mirrors the worked example in CRITIC_GAE_INTEGRATION.md:
    O0 --delegation--> R0 ; O0 --delegation--> C0 ; R0 --mention--> C0 ;
    C0 --delegation--> T0 ; T0 --completion--> C1 ; R0 --completion--> O1 ;
    C1 --completion--> O1
Global completion order: O0, R0, C0, T0, C1, O1.

Torch-free.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.event_codec import dag_to_events
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    DAGError,
    EdgeType,
    ExecutionDAG,
)

ORDER = ["O0", "R0", "C0", "T0", "C1", "O1"]
EDGES = [
    ("O0", "R0", EdgeType.DELEGATION),
    ("O0", "C0", EdgeType.DELEGATION),
    ("R0", "C0", EdgeType.MENTION),
    ("C0", "T0", EdgeType.DELEGATION),
    ("T0", "C1", EdgeType.COMPLETION),
    ("R0", "O1", EdgeType.COMPLETION),
    ("C1", "O1", EdgeType.COMPLETION),
]


def _build_dag() -> ExecutionDAG:
    dag = ExecutionDAG()
    for i, nid in enumerate(ORDER):
        node = AgentRunNode(
            node_id=nid, agent_id=nid[0], issue_id=f"iss-{nid}", task_id=f"task-{nid}"
        )
        node.value = 0.1 * i
        node.process_reward = 0.0
        node.metadata = {
            "messages": [{"role": "assistant", "content": f"{nid}-out"}],
            "completion_time": float(i),
        }
        dag.add_node(node)
    for src, dst, t in EDGES:
        dag.add_edge(src, dst, t)
    dag.get("O1").outcome_reward = 1.0  # terminal verifier reward
    return dag


def test_dag_to_events_uses_explicit_ordering_and_dense_index() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    assert [e.node_id for e in events] == ORDER
    assert [e.completion_index for e in events] == [0, 1, 2, 3, 4, 5]


def test_dag_to_events_fills_both_edge_directions() -> None:
    dag = _build_dag()
    events = {e.node_id: e for e in dag_to_events(dag, ordering=ORDER)}
    # C0 has incoming from O0 (delegation) and R0 (mention); outgoing to T0.
    assert set(events["C0"].incoming_edges) == {
        ("O0", EdgeType.DELEGATION),
        ("R0", EdgeType.MENTION),
    }
    assert events["C0"].outgoing_edges == (("T0", EdgeType.DELEGATION),)
    # O0 has no incoming; outgoing to R0 and C0.
    assert events["O0"].incoming_edges == ()
    assert set(events["O0"].outgoing_edges) == {
        ("R0", EdgeType.DELEGATION),
        ("C0", EdgeType.DELEGATION),
    }


def test_dag_to_events_copies_node_fields_and_messages() -> None:
    dag = _build_dag()
    events = {e.node_id: e for e in dag_to_events(dag, ordering=ORDER)}
    o1 = events["O1"]
    assert o1.outcome_reward == 1.0
    assert o1.value == pytest.approx(0.5)
    assert o1.task_id == "task-O1"
    assert o1.messages == ({"role": "assistant", "content": "O1-out"},)
    assert o1.completion_time == 5.0


def test_dag_to_events_falls_back_to_topological_order() -> None:
    dag = _build_dag()
    events = dag_to_events(dag)  # no explicit ordering
    order = [e.node_id for e in events]
    # Must be a valid topological order: every src precedes its dst.
    pos = {nid: i for i, nid in enumerate(order)}
    for src, dst, _ in EDGES:
        assert pos[src] < pos[dst]


def test_dag_to_events_rejects_non_permutation_ordering() -> None:
    dag = _build_dag()
    with pytest.raises(DAGError):
        dag_to_events(dag, ordering=["O0", "R0"])  # missing nodes


def test_dag_to_events_rejects_non_topological_ordering() -> None:
    dag = _build_dag()
    bad = ["R0", "O0", "C0", "T0", "C1", "O1"]  # R0 before its parent O0
    with pytest.raises(DAGError):
        dag_to_events(dag, ordering=bad)


def test_dag_to_events_messages_by_node_overrides_metadata() -> None:
    dag = _build_dag()
    override = {"O0": [{"role": "assistant", "content": "override"}]}
    events = {e.node_id: e for e in dag_to_events(dag, ordering=ORDER,
                                                  messages_by_node=override)}
    assert events["O0"].messages == ({"role": "assistant", "content": "override"},)
    # Non-overridden nodes still use metadata messages.
    assert events["R0"].messages == ({"role": "assistant", "content": "R0-out"},)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -q`
Expected: FAIL with `ModuleNotFoundError: ... event_codec`.

- [ ] **Step 3: Write minimal implementation**

Create `customized_areal/tree_search/dag/event_codec.py`:

```python
"""Bidirectional codec between an ExecutionDAG and its linear Event log.

Forward (``dag_to_events``): linearize the DAG into completion-ordered Events
for reward backup. Reverse (``events_to_dag``): losslessly rebuild the DAG from
a persisted Event log, then (``replay_prefix_for``) derive a branch replay
prefix shaped to ``BranchMaterializer.materialize``'s inputs.

Pure: no mutation of inputs, no I/O, torch-free. All failures raise ``DAGError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from customized_areal.tree_search.dag.event_model import Event, message_timeline
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    DAGError,
    EdgeType,
    ExecutionDAG,
)


def _adjacency(
    dag: ExecutionDAG,
) -> tuple[dict[str, list[tuple[str, EdgeType]]], dict[str, list[tuple[str, EdgeType]]]]:
    """Build per-node incoming/outgoing typed-edge lists from ``dag.edges``."""
    incoming: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.node_ids()}
    outgoing: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.node_ids()}
    for e in dag.edges:
        outgoing[e.src].append((e.dst, e.type))
        incoming[e.dst].append((e.src, e.type))
    return incoming, outgoing


def _validate_topological(dag: ExecutionDAG, order: Sequence[str]) -> None:
    index_of = {nid: i for i, nid in enumerate(order)}
    for e in dag.edges:
        if index_of[e.src] >= index_of[e.dst]:
            raise DAGError(
                f"ordering is not topological: edge {e.src!r}->{e.dst!r} has "
                f"index {index_of[e.src]} >= {index_of[e.dst]}"
            )


def dag_to_events(
    dag: ExecutionDAG,
    *,
    ordering: Sequence[str] | None = None,
    messages_by_node: dict | None = None,
) -> list[Event]:
    """Linearize ``dag`` into completion-ordered Events.

    ``ordering`` is an explicit completion order (list of node_id). If omitted,
    ``dag.topological_order()`` is used (deterministic; valid because completion
    order is always a topological order). ``messages_by_node`` overrides the
    per-node transcript payload (else ``node.metadata['messages']``).
    """
    node_ids = dag.node_ids()
    if ordering is None:
        order = [n.node_id for n in dag.topological_order()]
    else:
        order = list(ordering)
        if sorted(order) != sorted(node_ids):
            raise DAGError("ordering is not a permutation of the DAG node ids")
        _validate_topological(dag, order)

    incoming, outgoing = _adjacency(dag)
    msgs_map = messages_by_node or {}
    events: list[Event] = []
    for idx, nid in enumerate(order):
        node = dag.get(nid)
        payload = msgs_map.get(nid)
        if payload is None:
            payload = node.metadata.get("messages", ())
        events.append(
            Event(
                node_id=node.node_id,
                agent_id=node.agent_id,
                issue_id=node.issue_id,
                task_id=node.task_id,
                completion_index=idx,
                incoming_edges=tuple(incoming[nid]),
                outgoing_edges=tuple(outgoing[nid]),
                session_id=node.session_id,
                completion_time=node.metadata.get("completion_time"),
                branch_seq=node.branch_seq,
                branch_issue_id=node.branch_issue_id,
                branch_env_snapshot_id=node.branch_env_snapshot_id,
                value=node.value,
                process_reward=node.process_reward,
                outcome_reward=node.outcome_reward,
                messages=tuple(dict(m) for m in payload),
                metadata=dict(node.metadata),
            )
        )
    return events


__all__ = ["dag_to_events"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint and commit**

```bash
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
$RUFF format customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
git add customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "feat(dag): add dag_to_events forward codec"
```

______________________________________________________________________

## Task 4: `events_to_dag` (reverse path, lossless + validated)

**Files:**

- Modify: `customized_areal/tree_search/dag/event_codec.py`

- Test: `customized_areal/tree_search/dag/test_event_codec.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/dag/test_event_codec.py`:

```python
from customized_areal.tree_search.dag.event_codec import events_to_dag


def test_events_to_dag_round_trip_rebuilds_nodes_and_edges() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    rebuilt = events_to_dag(events)

    assert sorted(rebuilt.node_ids()) == sorted(ORDER)
    # Edges identical (as a typed set).
    orig = {(e.src, e.dst, e.type) for e in dag.edges}
    back = {(e.src, e.dst, e.type) for e in rebuilt.edges}
    assert back == orig
    # Node fields preserved.
    assert rebuilt.get("O1").outcome_reward == 1.0
    assert rebuilt.get("C0").value == pytest.approx(0.2)


def test_events_to_dag_rejects_non_dense_index() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    # Corrupt: duplicate index 0 (C0 also gets completion_index 0).
    events[2] = replace(events[2], completion_index=0)
    with pytest.raises(DAGError):
        events_to_dag(events)


def test_events_to_dag_rejects_asymmetric_edges() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    # Drop C0's incoming-from-O0 edge so O0.outgoing no longer matches C0.incoming.
    events = [
        replace(e, incoming_edges=(("R0", EdgeType.MENTION),))  # removed O0->C0
        if e.node_id == "C0"
        else e
        for e in events
    ]
    with pytest.raises(DAGError):
        events_to_dag(events)


def test_events_to_dag_rejects_non_topological_index() -> None:
    dag = _build_dag()
    events = list(dag_to_events(dag, ordering=ORDER))
    # Swap completion_index of O0 (0) and R0 (1) so R0 (child) precedes O0.
    swapped = []
    for e in events:
        if e.node_id == "O0":
            swapped.append(replace(e, completion_index=1))
        elif e.node_id == "R0":
            swapped.append(replace(e, completion_index=0))
        else:
            swapped.append(e)
    with pytest.raises(DAGError):
        events_to_dag(swapped)


def test_events_to_dag_empty_returns_empty_dag() -> None:
    rebuilt = events_to_dag([])
    assert rebuilt.node_ids() == []


def test_full_dict_round_trip_identity() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)
    # events -> dict -> events -> dag, identical to events -> dag.
    redecoded = [Event.from_dict(e.to_dict()) for e in events]
    assert redecoded == events
    dag_a = events_to_dag(events)
    dag_b = events_to_dag(redecoded)
    assert {(e.src, e.dst, e.type) for e in dag_a.edges} == {
        (e.src, e.dst, e.type) for e in dag_b.edges
    }
    assert sorted(dag_a.node_ids()) == sorted(dag_b.node_ids())
```

Add these imports at the top of the test file (`Event` for the dict round-trip test,
`replace` for the corruption tests):

```python
from dataclasses import replace

from customized_areal.tree_search.dag.event_model import Event
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -q`
Expected: FAIL with `ImportError: cannot import name 'events_to_dag'`.

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/dag/event_codec.py`, add the function and update
`__all__`:

```python
def events_to_dag(events: Sequence[Event]) -> ExecutionDAG:
    """Losslessly rebuild an ExecutionDAG from a linear Event log.

    Steps (each failure raises ``DAGError``):
      1. completion_index must be dense 0..n-1, unique, non-negative.
      2. edge lists must be symmetric (every A.outgoing (A->B) has a matching
         B.incoming (A->B) with the same EdgeType).
      3. add nodes (faithful AgentRunNode; messages/index stay in the log only).
      4. add edges (idempotent).
      5. enforce the topological-order invariant: for every edge src->dst,
         index(src) < index(dst).
    """
    events = list(events)
    if not events:
        return ExecutionDAG()

    # 1. dense/unique/non-negative completion_index.
    indices = sorted(e.completion_index for e in events)
    if indices != list(range(len(events))):
        raise DAGError(
            f"completion_index must be dense 0..{len(events) - 1}, got {indices}"
        )
    ordered = sorted(events, key=lambda e: e.completion_index)
    index_of = {e.node_id: e.completion_index for e in ordered}
    if len(index_of) != len(ordered):
        raise DAGError("duplicate node_id across events")

    # 2. edge symmetry: incoming-derived set must equal outgoing-derived set.
    incoming_set = {
        (src, e.node_id, t) for e in ordered for (src, t) in e.incoming_edges
    }
    outgoing_set = {
        (e.node_id, dst, t) for e in ordered for (dst, t) in e.outgoing_edges
    }
    if incoming_set != outgoing_set:
        diff = incoming_set ^ outgoing_set
        raise DAGError(f"edge symmetry mismatch (incoming XOR outgoing): {sorted(diff)}")

    # 3. add nodes (faithful AgentRunNode; messages/completion_index NOT stored).
    dag = ExecutionDAG()
    for e in ordered:
        dag.add_node(
            AgentRunNode(
                node_id=e.node_id,
                agent_id=e.agent_id,
                issue_id=e.issue_id,
                task_id=e.task_id,
                session_id=e.session_id,
                branch_seq=e.branch_seq,
                branch_issue_id=e.branch_issue_id,
                branch_env_snapshot_id=e.branch_env_snapshot_id,
                process_reward=e.process_reward,
                outcome_reward=e.outcome_reward,
                value=e.value,
                metadata=dict(e.metadata),
            )
        )

    # 4. add edges once (symmetry guarantees outgoing agrees). add_edge validates
    #    unknown endpoints and is idempotent.
    for src, dst, t in sorted(incoming_set):
        dag.add_edge(src, dst, t)

    # 5. topological-order invariant.
    for src, dst, _ in incoming_set:
        if index_of[src] >= index_of[dst]:
            raise DAGError(
                f"event order is not topological: edge {src!r}->{dst!r} has "
                f"index {index_of[src]} >= {index_of[dst]}"
            )
    return dag
```

Update the export line:

```python
__all__ = ["dag_to_events", "events_to_dag"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -q`
Expected: PASS (all forward + reverse tests).

- [ ] **Step 5: Lint and commit**

```bash
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
$RUFF format customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
git add customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "feat(dag): add events_to_dag reverse codec with symmetry + topo invariant"
```

______________________________________________________________________

## Task 5: `replay_prefix_for` + `ReplayPrefix`

**Files:**

- Modify: `customized_areal/tree_search/dag/event_codec.py`

- Test: `customized_areal/tree_search/dag/test_event_codec.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/dag/test_event_codec.py`:

```python
from customized_areal.tree_search.dag.event_codec import ReplayPrefix, replay_prefix_for


def _build_dag_with_branch() -> ExecutionDAG:
    dag = _build_dag()
    # Mark C1 as branchable at seq 7, with fork provenance.
    c1 = dag.get("C1")
    c1.branch_seq = 7
    c1.branch_env_snapshot_id = "snap-C1"
    return dag


def test_replay_prefix_for_returns_ancestor_slice() -> None:
    dag = _build_dag_with_branch()
    events = dag_to_events(dag, ordering=ORDER)
    prefix = replay_prefix_for(events, branch_point=("task-C1", 7))

    assert isinstance(prefix, ReplayPrefix)
    assert prefix.task_id == "task-C1"
    assert prefix.seq == 7
    assert prefix.source_issue_id == "iss-C1"
    assert prefix.branch_env_snapshot_id == "snap-C1"
    # C1's ancestors are O0, R0, C0, T0 (+ C1 itself); O1 is NOT included.
    contents = [m["content"] for m in prefix.replay_messages]
    assert contents == ["O0-out", "R0-out", "C0-out", "T0-out", "C1-out"]


def test_replay_prefix_for_unknown_branch_point_raises() -> None:
    dag = _build_dag_with_branch()
    events = dag_to_events(dag, ordering=ORDER)
    with pytest.raises(DAGError):
        replay_prefix_for(events, branch_point=("task-C1", 999))


def test_replay_prefix_for_ambiguous_branch_point_raises() -> None:
    dag = _build_dag_with_branch()
    # Make C0 also branchable at the same (task_id, seq) as a different node would be.
    # Force two matches by giving two nodes the same task_id + branch_seq.
    dag.get("C0").task_id = "task-dup"
    dag.get("C0").branch_seq = 42
    dag.get("T0").task_id = "task-dup"
    dag.get("T0").branch_seq = 42
    events = dag_to_events(dag, ordering=ORDER)
    with pytest.raises(DAGError):
        replay_prefix_for(events, branch_point=("task-dup", 42))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -q`
Expected: FAIL with `ImportError: cannot import name 'ReplayPrefix'`.

- [ ] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/dag/event_codec.py`, add the dataclass and function,
and update `__all__`:

```python
@dataclass(frozen=True)
class ReplayPrefix:
    """Branch replay data, shaped to ``BranchMaterializer.materialize`` inputs."""

    replay_messages: list[dict]
    task_id: str
    seq: int
    source_issue_id: str
    branch_env_snapshot_id: str | None


def replay_prefix_for(
    events: Sequence[Event],
    *,
    branch_point: tuple[str, int],
) -> ReplayPrefix:
    """Derive the replay prefix for a branch point from a linear Event log.

    ``branch_point = (task_id, seq)`` where ``seq`` is the ``task_message.seq``
    the run is allowed to branch at. Locates the unique Event with matching
    ``task_id`` and ``branch_seq == seq`` (zero or multiple matches -> DAGError),
    collects that node's ancestors (plus the node itself) in completion order,
    and flattens their message payloads via ``message_timeline``.
    """
    task_id, seq = branch_point
    dag = events_to_dag(events)
    matches = [e for e in events if e.task_id == task_id and e.branch_seq == seq]
    if len(matches) != 1:
        raise DAGError(
            f"branch point (task_id={task_id!r}, seq={seq}) matched {len(matches)} "
            f"nodes; expected exactly 1"
        )
    branch_ev = matches[0]
    ancestor_ids = dag.ancestors(branch_ev.node_id) | {branch_ev.node_id}
    prefix_events = sorted(
        (e for e in events if e.node_id in ancestor_ids),
        key=lambda e: e.completion_index,
    )
    return ReplayPrefix(
        replay_messages=message_timeline(prefix_events),
        task_id=task_id,
        seq=seq,
        source_issue_id=branch_ev.issue_id,
        branch_env_snapshot_id=branch_ev.branch_env_snapshot_id,
    )
```

Update the export line:

```python
__all__ = ["ReplayPrefix", "dag_to_events", "events_to_dag", "replay_prefix_for"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -q`
Expected: PASS (forward + reverse + replay).

- [ ] **Step 5: Lint and commit**

```bash
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
$RUFF format customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
git add customized_areal/tree_search/dag/event_codec.py customized_areal/tree_search/dag/test_event_codec.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "feat(dag): add replay_prefix_for branch-resume helper"
```

______________________________________________________________________

## Task 6: Refactor `events_from_nodes` into a thin projection over `Event`

**Files:**

- Modify: `customized_areal/tree_search/dag/gae.py` (the `events_from_nodes` function,
  near the bottom before `__all__`)
- Test: `customized_areal/tree_search/dag/test_event_codec.py` (parity test)

The current `events_from_nodes` (in `gae.py`) maps each node to
`GlobalEvent(node_id, value=value or 0.0, reward=process_reward+outcome_reward)`. We
rewrite it to build an `Event` per node (via `event_model`) and project, with a parity
test proving identical output.

- [ ] **Step 1: Write the failing parity test**

Append to `customized_areal/tree_search/dag/test_event_codec.py`:

```python
from customized_areal.tree_search.dag.gae import GlobalEvent, events_from_nodes


def _old_events_from_nodes(ordered_nodes):
    """Snapshot of the pre-refactor logic, for parity comparison."""
    out = []
    for node in ordered_nodes:
        value = getattr(node, "value", None)
        out.append(
            GlobalEvent(
                node_id=node.node_id,
                value=float(value) if value is not None else 0.0,
                reward=float(node.process_reward) + float(node.outcome_reward),
            )
        )
    return out


def test_events_from_nodes_parity_with_old_logic() -> None:
    dag = _build_dag()
    nodes = [dag.get(nid) for nid in ORDER]
    assert events_from_nodes(nodes) == _old_events_from_nodes(nodes)


def test_events_from_nodes_unscored_value_is_zero() -> None:
    node = AgentRunNode(node_id="n", agent_id="a", issue_id="i", task_id="t")
    node.process_reward = 0.25
    (ev,) = events_from_nodes([node])
    assert ev == GlobalEvent(node_id="n", value=0.0, reward=0.25)
```

- [ ] **Step 2: Run test to verify current behavior (baseline passes)**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -k events_from_nodes -q`
Expected: PASS against the current implementation (this establishes the parity baseline
before refactor).

- [ ] **Step 3: Refactor the implementation**

In `customized_areal/tree_search/dag/gae.py`, replace the body of `events_from_nodes` so
it routes through the canonical `Event`. Replace the existing function (keep the same
name, signature, and docstring intent):

```python
def events_from_nodes(ordered_nodes: list) -> list[GlobalEvent]:
    """Build the global event sequence from DAG nodes in completion order.

    Thin projection over the canonical :class:`Event`: each node becomes an
    ``Event`` (edges/messages irrelevant to GAE are left empty), then is
    projected to a :class:`GlobalEvent` with ``value`` (``V_{t+1}``; 0.0 if
    unscored) and a step reward of ``process_reward + outcome_reward`` -- so the
    verifier terminal reward (on ``outcome_reward``) flows in as ``r_t``.

    Duck-typed against ``AgentRunNode``; identity fields are read defensively so
    minimal node-likes still work (they do not affect the projection).
    """
    from customized_areal.tree_search.dag.event_model import Event

    events: list[GlobalEvent] = []
    for idx, node in enumerate(ordered_nodes):
        ev = Event(
            node_id=node.node_id,
            agent_id=getattr(node, "agent_id", ""),
            issue_id=getattr(node, "issue_id", ""),
            task_id=getattr(node, "task_id", ""),
            completion_index=idx,
            value=getattr(node, "value", None),
            process_reward=float(node.process_reward),
            outcome_reward=float(node.outcome_reward),
        )
        events.append(
            GlobalEvent(
                node_id=ev.node_id,
                value=float(ev.value) if ev.value is not None else 0.0,
                reward=float(ev.process_reward) + float(ev.outcome_reward),
            )
        )
    return events
```

Note: the import is function-local to avoid any import-order coupling at module load.
`gae.py` already declares `events_from_nodes` in its `__all__`; leave that unchanged.

- [ ] **Step 4: Run tests to verify parity holds**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py customized_areal/tree_search/dag/test_gae.py -q`
Expected: PASS — both the new parity test and the existing `test_gae.py`
(`test_events_from_nodes_reads_value_and_combines_rewards`) confirm identical output.

- [ ] **Step 5: Lint and commit**

```bash
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/gae.py customized_areal/tree_search/dag/test_event_codec.py
$RUFF format customized_areal/tree_search/dag/gae.py customized_areal/tree_search/dag/test_event_codec.py
git add customized_areal/tree_search/dag/gae.py customized_areal/tree_search/dag/test_event_codec.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "refactor(dag): express events_from_nodes as a projection over Event"
```

______________________________________________________________________

## Task 7: Critic-observation parity (no code change to the builder)

**Files:**

- Test: `customized_areal/tree_search/dag/test_event_codec.py`

`build_critic_observations` stays code-unchanged. This task proves it produces identical
frontier observations whether fed a hand-built timeline or the `message_timeline` of
`dag_to_events(dag)`.

- [ ] **Step 1: Write the failing/locking test**

Append to `customized_areal/tree_search/dag/test_event_codec.py`:

```python
from customized_areal.tree_search.dag.critic_observation import (
    build_critic_observations,
)
from customized_areal.tree_search.dag.event_model import message_timeline


def test_critic_observations_match_message_timeline_of_events() -> None:
    dag = _build_dag()
    events = dag_to_events(dag, ordering=ORDER)

    # Path A: the canonical message timeline derived from events.
    timeline_from_events = message_timeline(events)
    obs_from_events = build_critic_observations(timeline_from_events)

    # Path B: an equivalent hand-built timeline (one tagged output per node, in
    # completion order) -- the shape the critic builder consumes today.
    hand_built = [
        {"role": "assistant", "content": f"{nid}-out", "node_id": nid}
        for nid in ORDER
    ]
    obs_hand = build_critic_observations(hand_built)

    # Same number of observations and same node_id sequence (V_0..V_T).
    assert [o.node_id for o in obs_from_events] == [o.node_id for o in obs_hand]
    assert [o.value_index for o in obs_from_events] == [
        o.value_index for o in obs_hand
    ]
    # V_0 is the initial frontier with node_id None in both.
    assert obs_from_events[0].node_id is None
    assert len(obs_from_events) == len(ORDER) + 1
```

- [ ] **Step 2: Run test to verify it passes**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -k critic -q`
Expected: PASS. If it fails, the discrepancy is in `message_timeline`'s tagging or
ordering — fix `message_timeline` (Task 1/2), not `build_critic_observations`.

- [ ] **Step 3: (No implementation change)**

`build_critic_observations` is intentionally untouched.

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/dag/test_event_codec.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "test(dag): lock critic-observation parity via message_timeline"
```

______________________________________________________________________

## Task 8: Public exports + full-suite verification

**Files:**

- Modify: `customized_areal/tree_search/dag/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `customized_areal/tree_search/dag/test_event_codec.py`:

```python
def test_public_exports_available_from_package() -> None:
    import customized_areal.tree_search.dag as d

    for name in (
        "Event",
        "message_timeline",
        "dag_to_events",
        "events_to_dag",
        "replay_prefix_for",
        "ReplayPrefix",
    ):
        assert name in d.__all__, f"{name} missing from __all__"
        assert hasattr(d, name), f"{name} not importable from package"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`.venv-test/bin/python -m pytest customized_areal/tree_search/dag/test_event_codec.py -k public_exports -q`
Expected: FAIL — names not yet exported.

- [ ] **Step 3: Add exports**

In `customized_areal/tree_search/dag/__init__.py`, add these imports (alongside the
existing block, keeping alphabetical-ish grouping) — place after the
`critic_observation` import block:

```python
from customized_areal.tree_search.dag.event_codec import (
    ReplayPrefix,
    dag_to_events,
    events_to_dag,
    replay_prefix_for,
)
from customized_areal.tree_search.dag.event_model import (
    Event,
    message_timeline,
)
```

Then add to the `__all__` list (a new commented group):

```python
    # event codec (DAG <-> linear trajectory)
    "Event",
    "message_timeline",
    "dag_to_events",
    "events_to_dag",
    "replay_prefix_for",
    "ReplayPrefix",
```

**Constraint:** `__init__.py` must stay torch-free. `event_model` and `event_codec`
import only `execution_dag` (torch-free), so this holds. Do NOT import
`critic_score`/`critic_advantage` here (they remain excluded).

- [ ] **Step 4: Run the full dag suite + import check**

Run:

```bash
.venv-test/bin/python -c "import customized_areal.tree_search.dag as d; print('import ok')"
.venv-test/bin/python -m pytest customized_areal/tree_search/dag/ -q
```

Expected: import prints `import ok`; the full `dag/` suite passes (previously 81 passed
/ 2 skipped, plus the new `test_event_model.py` and `test_event_codec.py` tests).

- [ ] **Step 5: Lint and commit**

```bash
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/__init__.py
$RUFF format customized_areal/tree_search/dag/__init__.py
git add customized_areal/tree_search/dag/__init__.py customized_areal/tree_search/dag/test_event_codec.py
git -c user.name="Kiro Agent" -c user.email="kiro@local" commit -m "feat(dag): export event codec public API"
```

______________________________________________________________________

## Final verification

- [ ] Run the entire DAG package test suite and lint once more:

```bash
.venv-test/bin/python -m pytest customized_areal/tree_search/dag/ -q
RUFF=/home/vscode/.cache/uv/archive-v0/cj3x863YkgrEOZ4C/bin/ruff
$RUFF check customized_areal/tree_search/dag/
$RUFF format --check customized_areal/tree_search/dag/
```

Expected: all tests pass; ruff clean.

- [ ] Confirm no unintended edits to `execution_dag.py`, `integration.py`,
  `environment.py`, or `critic_observation.py` (only a test was added for the latter):

```bash
git status --porcelain
git diff --stat HEAD~8
```

Expected: changes limited to the new `event_model.py`/`event_codec.py`, their tests, the
`gae.py` `events_from_nodes` body, and `__init__.py` exports.

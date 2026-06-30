# SuperNode Multi-Agent DAG Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `Event` + `AgentRunNode` into a single `SuperNode` type that carries `list[Node]`, refactor `MCTSTreeStore` to hold SuperNodes while keeping MCTS stats per-Node, and lay the data-model foundation for Multica-orchestrated multi-agent DAG rollout.

**Architecture:** SuperNode = communication-bounded segment of one agent run (not the whole run). Multica decides segment boundaries and edges; AReal's `SuperNodeAssembler` consumes `DagResult` (segments + edges + env snapshots) and the per-session `list[Node]` from the N×M proxy workers, builds `list[SuperNode]` + `ExecutionDAG`, and sets a unified `parent_node_id` chain across SuperNode and agent boundaries (causal flattening). Reward backup walks Node-level only; `distribute_reward_over_dag` stays as a utility, off the reward path. Node becomes torch-lazy so the DAG layer imports cleanly without torch.

**Tech Stack:** Python 3.14, dataclasses, pytest, httpx (client only), stdlib logging. No torch in the agents/ layer (Node torch-lazy); no new runtime deps.

**Spec:** `docs/superpowers/specs/2026-06-30-supernode-multica-dag-rollout-design.md`

**Scope:** Phase 1a only (data model + codec + assembler + tests + single-agent path wrapping). Phase 1b/2 (coordinator, `MulticaDagClient`, verifier wiring) is a separate plan landed after this one merges. Every task below is Phase 1a unless tagged `[Phase 1b/2]`.

---

## File Structure

### Files created
- `customized_areal/tree_search/agents/supernode_assembler.py` — `SuperNodeAssembler`, `SegmentSpec`, `EdgeSpec`, `TeamEnvSnapshot`, `DagResult`. Pure, torch-free. The only new module in Phase 1a.
- `customized_areal/tree_search/tests/test_supernode_assembler.py` — unit tests for the assembler's 6-step algorithm.
- `customized_areal/tree_search/tests/test_tree_store_super.py` — unit tests for `insert_super_batch`, dual indices, cross-SuperNode backup.

### Files modified
- `customized_areal/tree_search/core/tree_store.py` — Node torch-lazy; `MCTSTreeStore.trajectories: dict[str, list[SuperNode]]`; `insert_super_batch` + dual indices; backup walks parent chain unchanged.
- `customized_areal/tree_search/agents/execution_dag.py` — `SuperNode` dataclass (replaces `AgentRunNode`); `ExecutionDAG` holds SuperNodes; `add_event`/`get`/`session_map`/`to_records`/`from_records` re-typed.
- `customized_areal/tree_search/agents/event_model.py` — shrink to `EdgeRef` + `message_timeline` (Event class removed).
- `customized_areal/tree_search/agents/event_codec.py` — `dag_to_supernodes` / `supernodes_to_dag` / `replay_prefix_for` (renamed, SuperNode-typed).
- `customized_areal/tree_search/agents/__init__.py` — update exports.
- `customized_areal/tree_search/agents/gae.py` — `events_from_nodes` constructs `SuperNode` (pure rename); `GlobalEvent` unchanged.
- `customized_areal/tree_search/core/customized_grouped_workflow.py` — single-agent path wraps Nodes in a leaf SuperNode before `insert_super_batch` (Phase 1a); coordinator params declared but `None` (Phase 1b/2 wire-up).
- `customized_areal/tree_search/tests/test_event_codec.py` — `Event` → `SuperNode`, `dag_to_events` → `dag_to_supernodes`, etc.
- `customized_areal/tree_search/tests/test_event_model.py` — shrinks to `message_timeline` tests only (Event class gone).
- `customized_areal/tree_search/tests/test_execution_dag.py` — `AgentRunNode` → `SuperNode`.
- `customized_areal/tree_search/tests/test_dag_backup.py` — `AgentRunNode` → `SuperNode`.
- `customized_areal/tree_search/tests/test_session_map.py` — `AgentRunNode` → `SuperNode`.
- `customized_areal/tree_search/tests/test_e2e_critic_gae.py` — `AgentRunNode` → `SuperNode` (rename only; logic unchanged).

---

## Task 0: Node torch-lazy refactor

**Why first:** Every subsequent task imports `Node` from `tree_store`. Making it torch-lazy now means the agents/ layer (which imports `Node` for the `SuperNode.nodes` field) stays import-clean without torch. This is the foundation — nothing else compiles cleanly without it.

**Files:**
- Modify: `customized_areal/tree_search/core/tree_store.py:14-20` (top-level `import torch`), `:81-82` (tensor field types), `:155` (`_optional_tensor_field`), `:199-205` (`_node_to_tensor_dict` tensor construction)

- [ ] **Step 1: Write the failing test — module imports without torch**

Create `customized_areal/tree_search/tests/test_node_torch_lazy.py`:

```python
"""Node + tree_store must import cleanly without torch installed.

The agents/ DAG layer imports Node for the SuperNode.nodes field; if tree_store
hard-imports torch at module top, every torch-free test in agents/ breaks when
torch is absent. This test guards the lazy import by hiding torch from the
import system before importing tree_store.
"""

from __future__ import annotations

import importlib
import sys


def test_tree_store_imports_without_torch(monkeypatch):
    # Block torch from being importable, then force a fresh import of tree_store.
    # If tree_store still has a top-level `import torch`, this raises ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    # Remove any cached import so the next import re-executes the module body.
    for mod in list(sys.modules):
        if mod.startswith("customized_areal.tree_search.core.tree_store"):
            del sys.modules[mod]
    tree_store = importlib.import_module(
        "customized_areal.tree_search.core.tree_store"
    )
    assert hasattr(tree_store, "Node")
    assert hasattr(tree_store, "MCTSTreeStore")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_node_torch_lazy.py -v`
Expected: FAIL — `ImportError: No module named 'torch'` (raised from `tree_store.py:20` `import torch`), caught by pytest as a collection error or a raised `ImportError` inside the test body.

- [ ] **Step 3: Make Node torch-lazy**

Edit `customized_areal/tree_search/core/tree_store.py`:

Replace the top of the file (lines 14-20):

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _lazy_torch():
    """Import and return torch on first use; None if unavailable.

    tree_store stays importable without torch. Tensor-consuming code paths
    (_node_to_tensor_dict, _optional_tensor_field) call this at first use.
    """
    import torch

    return torch
```

(Delete the top-level `import torch`.)

Change the tensor field type hints (lines 81-82) to `Any` with a comment:

```python
    # Tree-computed advantages/returns (set by TreeAdvantageComputer or
    # GAEAdvantageComputer). Typed Any (not torch.Tensor) so this module imports
    # cleanly without torch; tensor construction is deferred to lazy import in
    # _node_to_tensor_dict.
    advantages: Any = None
    returns: Any = None
```

In `_optional_tensor_field` (around line 155), replace the `torch.tensor(...)` calls with a lazy import:

```python
def _optional_tensor_field(
    traj: dict[str, Any],
    key: str,
    values: list | None,
    dtype: Any,
    start: int = 0,
    end: int | None = None,
    loss_mask: list[int] | None = None,
) -> None:
    """Add an unsqueezed tensor to traj if values is not None.

    Supports full-sequence rows, all-response rows, and current-response rows.
    If start/end are given and the field has full-sequence rows, slice by
    absolute token offsets. Response-only and current-response fields are
    already relative and are exported as-is.
    """
    if values is not None:
        sliced = values
        if end is not None:
            value_len = len(values)
            if loss_mask is not None:
                starts, ends = _find_turn_boundaries(loss_mask)
                total_response_len = sum(e - s for s, e in zip(starts, ends))
                current_response_len = (ends[-1] - starts[-1]) if starts else 0
                if value_len == seq_len := len(loss_mask):
                    sliced = values[start:end]
                elif value_len in (total_response_len, current_response_len):
                    sliced = values
                else:
                    sliced = values[start:end]
            else:
                sliced = values[start:end]
        torch = _lazy_torch()
        traj[key] = torch.tensor(sliced, dtype=dtype).unsqueeze(0)
```

In `_node_to_tensor_dict` (around line 199), replace each `torch.tensor(...)` / `torch.full(...)` / `torch.ones(...)` with a `torch = _lazy_torch()` call at the top of the function, then use `torch.tensor(...)`:

```python
def _node_to_tensor_dict(
    node: Node,
    query_id: str,
    node_id: str,
    max_tokens: int = 0,
    loss_mode: str | None = None,
) -> dict[str, Any]:
    """Convert a single Node to a tensor dict with shape [1, seq_len].

    If max_tokens > 0, the sequence is truncated to the last max_tokens
    tokens before conversion (full-sequence fields sliced, response-aligned
    fields trimmed to remaining output positions).
    """
    torch = _lazy_torch()
    input_ids = node.input_ids
    # ... rest unchanged, all torch.tensor() calls now resolve via the local torch
```

- [ ] **Step 4: Run the torch-lazy test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/tests/test_node_torch_lazy.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing tree_store tests to verify no regression**

Run: `uv run pytest customized_areal/tree_search/tests/test_tree_store_backup.py customized_areal/tree_search/tests/test_tree_store_loo.py -v`
Expected: PASS (these exercise the tensor paths with torch installed; lazy import resolves to the real torch).

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/tests/test_node_torch_lazy.py
git commit -m "refactor: make Node torch-lazy so agents/ layer imports clean

Top-level import torch removed from tree_store.py; tensor field types
become Any and _lazy_torch() imports torch on first use in the tensor-
consuming helpers. This lets the upcoming SuperNode.nodes: list[Node]
field live in a torch-free module without forcing every agents/ test
to install torch."
```

---

## Task 1: SuperNode dataclass in execution_dag.py

**Why:** `SuperNode` replaces `AgentRunNode` and absorbs `Event`'s role. It's the single type for both DAG-node and linear-log views. Everything downstream (codec, assembler, tree_store, tests) references this type, so it must land first.

**Files:**
- Modify: `customized_areal/tree_search/agents/execution_dag.py:38-77` (replace `AgentRunNode` with `SuperNode`)
- Test: `customized_areal/tree_search/tests/test_execution_dag.py`

- [ ] **Step 1: Write the failing test — SuperNode carries nodes + env snapshot**

Append to `customized_areal/tree_search/tests/test_execution_dag.py`:

```python
from customized_areal.tree_search.agents.execution_dag import SuperNode


def _super(node_id: str, *, nodes=None, sandbox_ids=None) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        issue_id=f"issue-{node_id}",
        task_id=f"task-{node_id}",
        nodes=list(nodes or []),
        sandbox_ids=list(sandbox_ids or []),
    )


def test_supernode_defaults_empty_nodes_and_env():
    s = SuperNode(
        node_id="s1", agent_id="a", issue_id="i", task_id="t"
    )
    assert s.nodes == []
    assert s.sandbox_ids == []
    assert s.issue_snapshot_id is None
    assert s.env_state == {}
    assert s.closing_event is None
    assert s.terminal_node is None
    assert s.branch_node_id is None


def test_supernode_terminal_node_is_last():
    # Lightweight stand-ins; Node is torch-lazy and heavy to build, so use
    # SimpleNamespace for the structural test (terminal_node only reads .node_id).
    from types import SimpleNamespace

    n0 = SimpleNamespace(node_id="n0")
    n1 = SimpleNamespace(node_id="n1")
    s = _super("s1", nodes=[n0, n1])
    assert s.terminal_node is n1
    assert s.branch_node_id == "n1"


def test_supernode_branch_node_id_none_when_empty():
    s = _super("s1")
    assert s.terminal_node is None
    assert s.branch_node_id is None
```

Update the existing `_node` helper at the top of the file to return `SuperNode` instead of `AgentRunNode`:

```python
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    Edge,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def _node(
    node_id: str, *, issue_id: str = "", parent_issue_id: str | None = None
) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        issue_id=issue_id or f"issue-{node_id}",
        task_id=f"task-{node_id}",
    )
```

(Run `replace_all` over the file to swap every remaining `AgentRunNode` token to `SuperNode` in the test bodies and helper functions.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest customized_areal/tree_search/tests/test_execution_dag.py -v`
Expected: FAIL — `ImportError: cannot import name 'SuperNode'` from `execution_dag`.

- [ ] **Step 3: Replace AgentRunNode with SuperNode in execution_dag.py**

Edit `customized_areal/tree_search/agents/execution_dag.py`:

Replace the `AgentRunNode` dataclass (lines 38-77) with:

```python
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
    closing_event: EdgeType | None = None        # None for leaf segments
    closing_event_target: str | None = None     # the other SuperNode's node_id

    # RL session assigned at /rl/start_session time. One session_id per agent
    # run, shared across all SuperNodes of that run. None until bound.
    session_id: str | None = None

    # Branch boundary + fork provenance (filled when this node becomes a branch
    # source; Phase 3).
    branch_seq: int | None = None
    branch_issue_id: str | None = None
    branch_env_snapshot_id: str | None = None

    # Reward bookkeeping (set by the verifier / backup). Plain floats so the
    # DAG stays torch-free; the training Node carries tensors.
    process_reward: float = 0.0
    outcome_reward: float = 0.0

    # Critic value V_{t+1} (Phase 3, out of scope here). None until scored.
    value: float | None = None

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
```

Update `ExecutionDAG` internal type references (lines 100-105):

```python
    def __init__(self) -> None:
        self._events: dict[str, SuperNode] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
```

Update `add_event` (line 108) to accept `SuperNode`:

```python
    def add_event(self, event: SuperNode) -> SuperNode:
        if event.node_id in self._events:
            raise DAGError(f"duplicate event_id: {event.node_id!r}")
        self._events[event.node_id] = event
        self._out.setdefault(event.node_id, [])
        self._in.setdefault(event.node_id, [])
        return event
```

Update `get` (line 140) return type:

```python
    def get(self, event_id: str) -> SuperNode:
        try:
            return self._events[event_id]
        except KeyError as exc:
            raise DAGError(f"unknown event: {event_id!r}") from exc
```

Update `events` property (line 147):

```python
    @property
    def events(self) -> list[SuperNode]:
        return list(self._events.values())
```

Update `parents` / `children` (lines 157, 163) return types:

```python
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
```

Update `roots` / `leaves` / `fork_events` / `join_events` (lines 208-227) return types:

```python
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
        """Fan-in points: a run fed by >= 2 upstream runs."""
        return [ev for eid, ev in self._events.items() if len(self._in[eid]) >= 2]
```

Update `topological_order` (line 229) return type:

```python
    def topological_order(self) -> list[SuperNode]:
        """Kahn's algorithm. Raises ``DAGError`` if the graph has a cycle."""
        # ... body unchanged
```

Update `iter_topo` (line 280):

```python
    def iter_topo(self) -> Iterator[SuperNode]:
        yield from self.topological_order()
```

Update `from_records` (line 286) to build `SuperNode` instead of `AgentRunNode`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_execution_dag.py -v`
Expected: PASS — all existing structural tests now use `SuperNode`, plus the 3 new SuperNode-specific tests.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/execution_dag.py customized_areal/tree_search/tests/test_execution_dag.py
git commit -m "feat: replace AgentRunNode with SuperNode in ExecutionDAG

SuperNode is one communication-bounded segment of one agent's action
sequence (not the whole run). Carries nodes: list[Node] (torch-lazy),
team env snapshot (sandbox_ids/issue_snapshot_id/env_state), and
comm-event provenance (closing_event/closing_event_target). ExecutionDAG
API is otherwise unchanged: add_event/get/edges/topological_order all
re-typed to SuperNode."
```

---

## Task 2: Shrink event_model.py — Event removed, message_timeline kept

**Why:** `Event` is now redundant with `SuperNode`. The only thing `event_model.py` still owns that nothing else does is `message_timeline` (used by the critic observation builder) and the `EdgeRef` type alias. Shrink the module to those two.

**Files:**
- Modify: `customized_areal/tree_search/agents/event_model.py` (whole file)
- Test: `customized_areal/tree_search/tests/test_event_model.py` (rewrite)

- [ ] **Step 1: Write the failing test — message_timeline still works, Event is gone**

Replace the entire contents of `customized_areal/tree_search/tests/test_event_model.py` with:

```python
"""Tests for the message_timeline helper (Event class removed).

After the SuperNode unification, event_model.py no longer defines Event;
the linear-log role is absorbed by SuperNode. message_timeline survives
because the critic observation builder consumes it. It now takes SuperNodes
(or any object with .completion_index, .node_id, and .nodes).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from customized_areal.tree_search.agents.event_model import (
    EdgeRef,
    message_timeline,
)


def test_edgeref_is_alias_for_tuple():
    # EdgeRef is a type alias: tuple[str, EdgeType].
    from customized_areal.tree_search.agents.execution_dag import EdgeType

    ref: EdgeRef = ("n0", EdgeType.DELEGATION)
    assert ref[0] == "n0"
    assert ref[1] is EdgeType.DELEGATION


def _super(node_id, completion_index, messages):
    """Build a SuperNode-like object with the fields message_timeline reads."""
    return SimpleNamespace(
        node_id=node_id,
        completion_index=completion_index,
        nodes=[SimpleNamespace(messages=[m]) for m in messages],
        # message_timeline reads .messages if present (legacy); we put the
        # actual message dicts directly on .messages to exercise the new path.
        messages=tuple(messages),
    )


def test_message_timeline_orders_by_completion_index_and_tags_output():
    s1 = _super("O0", completion_index=1, messages=[
        {"role": "user", "content": "ctx"},
        {"role": "assistant", "content": "plan"},
    ])
    s0 = _super("seed", completion_index=0, messages=[
        {"role": "assistant", "content": "boot"},
    ])
    timeline = message_timeline([s1, s0])
    assert [m["content"] for m in timeline] == ["boot", "ctx", "plan"]
    assert timeline[0]["node_id"] == "seed"
    assert "node_id" not in timeline[1]
    assert timeline[2]["node_id"] == "O0"


def test_message_timeline_does_not_mutate_source():
    src = {"role": "assistant", "content": "x"}
    s = _super("n", completion_index=0, messages=[src])
    message_timeline([s])
    assert "node_id" not in src


def test_event_class_removed():
    # Importing Event must now fail; it was absorbed by SuperNode.
    from customized_areal.tree_search.agents import event_model

    assert not hasattr(event_model, "Event")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_event_model.py -v`
Expected: FAIL — `Event` still exists; `test_event_class_removed` asserts `not hasattr(event_model, "Event")` which fails.

- [ ] **Step 3: Shrink event_model.py**

Replace the entire contents of `customized_areal/tree_search/agents/event_model.py` with:

```python
"""Edge ref type + message-timeline helper for the linear Event-log view.

After the SuperNode unification, this module no longer defines ``Event``;
the linear-log role is absorbed by :class:`SuperNode` (which carries
``completion_index`` and the ``nodes`` payload directly). What survives here:

- :data:`EdgeRef` -- the ``(node_id, EdgeType)`` alias used across the codec.
- :func:`message_timeline` -- the message-level view consumed by the critic
  observation builder.

Torch-free and I/O-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import Protocol

from customized_areal.tree_search.agents.execution_dag import EdgeType

EdgeRef = tuple[str, EdgeType]


class _HasCompletionIndex(Protocol):
    """Structural type for objects message_timeline can consume.

    SuperNode satisfies this: ``completion_index`` is its linear position and
    ``node_id`` identifies the segment. Messages come from the ``nodes`` field
    (each node's serialized messages) or a legacy ``messages`` tuple on the
    object itself.
    """

    node_id: str
    completion_index: int


def _extract_messages(obj: object) -> list[dict]:
    """Return the message payload for one SuperNode-like object.

    Prefers ``obj.messages`` (the legacy Event shape: a tuple of message dicts);
    falls back to flattening ``obj.nodes[i].messages`` (the SuperNode shape).
    Each node's own output is the LAST message of its payload.
    """
    messages = getattr(obj, "messages", None)
    if messages is not None:
        return [dict(m) for m in messages]
    # SuperNode path: each inner node carries its own message dict(s).
    out: list[dict] = []
    nodes = getattr(obj, "nodes", None) or ()
    for n in nodes:
        node_msgs = getattr(n, "messages", None) or ()
        for m in node_msgs:
            out.append(dict(m))
    return out


def message_timeline(events: Sequence) -> list[dict]:
    """Derived message-level view across SuperNodes in completion order.

    Concatenates each SuperNode's message payload in ``completion_index``
    order. The LAST message of each SuperNode's payload is the segment's
    own terminal output and is tagged with the SuperNode's ``node_id`` (so the
    critic frontier builder can detect turn outputs); earlier payload messages
    pass through untagged. Tagged copies are emitted; the source payloads are
    not mutated.
    """
    ordered = sorted(events, key=lambda e: e.completion_index)
    timeline: list[dict] = []
    for ev in ordered:
        msgs = _extract_messages(ev)
        for i, m in enumerate(msgs):
            tagged = dict(m)
            if i == len(msgs) - 1:
                tagged["node_id"] = ev.node_id
            timeline.append(tagged)
    return timeline


__all__ = ["EdgeRef", "message_timeline"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/tests/test_event_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/event_model.py customized_areal/tree_search/tests/test_event_model.py
git commit -m "refactor: shrink event_model.py — Event removed, message_timeline kept

Event is now redundant with SuperNode (both describe one completed segment
in the linear trajectory). event_model.py now exports only EdgeRef and
message_timeline; the latter accepts SuperNode-like objects (reads .nodes
or legacy .messages). Critic observation builder is unchanged."
```

---

## Task 3: Rename codec — dag_to_supernodes / supernodes_to_dag / replay_prefix_for

**Why:** The codec round-trips the DAG ↔ linear-log view. With `Event` gone and `SuperNode` the unified type, the codec functions take and return `list[SuperNode]`. Pure rename + type re-annotation; algorithm unchanged.

**Files:**
- Modify: `customized_areal/tree_search/agents/event_codec.py` (whole file)
- Test: `customized_areal/tree_search/tests/test_event_codec.py`

- [ ] **Step 1: Write the failing test — new function names, SuperNode-typed**

Update `customized_areal/tree_search/tests/test_event_codec.py`. Replace the imports and every call site. The test fixture `_build_dag` already builds SuperNodes (after Task 1's `AgentRunNode → SuperNode` rename in the test file). Replace the imports block:

```python
from customized_areal.tree_search.agents.event_codec import (
    ReplayPrefix,
    dag_to_supernodes,
    replay_prefix_for,
    supernodes_to_dag,
)
from customized_areal.tree_search.agents.event_model import message_timeline
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)
from customized_areal.tree_search.agents.gae import GlobalEvent, events_from_nodes
```

Replace every `dag_to_events` → `dag_to_supernodes` and `events_to_dag` → `supernodes_to_dag` in the test bodies (use `replace_all`).

Update the public-exports test at the bottom:

```python
def test_public_exports_available_from_package() -> None:
    import customized_areal.tree_search.agents as d

    for name in (
        "SuperNode",
        "message_timeline",
        "dag_to_supernodes",
        "supernodes_to_dag",
        "replay_prefix_for",
        "ReplayPrefix",
    ):
        assert name in d.__all__, f"{name} missing from __all__"
        assert hasattr(d, name), f"{name} not importable from package"
```

Also drop the `Event.from_dict` round-trip test (`test_full_dict_round_trip_identity`) — `Event` no longer exists. Replace it with a SuperNode-based dict round-trip:

```python
def test_supernode_dict_round_trip_identity() -> None:
    dag = _build_dag()
    supers = dag_to_supernodes(dag, ordering=ORDER)
    # SuperNode carries nodes; the codec emits them via to_dict. We rebuild
    # from the dict and check identity of the DAG-level fields.
    redecoded = [SuperNode.from_dict(s.to_dict()) for s in supers]
    assert redecoded == supers
    dag_a = supernodes_to_dag(supers)
    dag_b = supernodes_to_dag(redecoded)
    assert {(e.src, e.dst, e.type) for e in dag_a.edges} == {
        (e.src, e.dst, e.type) for e in dag_b.edges
    }
    assert sorted(dag_a.event_ids()) == sorted(dag_b.event_ids())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_event_codec.py -v`
Expected: FAIL — `ImportError: cannot import name 'dag_to_supernodes'` from `event_codec`.

- [ ] **Step 3: Rename the codec functions and re-type to SuperNode**

Edit `customized_areal/tree_search/agents/event_codec.py`. Replace the imports and the three function signatures. The full new file:

```python
"""Bidirectional codec between an ExecutionDAG and its linear SuperNode log.

Forward (``dag_to_supernodes``): linearize the DAG into completion-ordered
SuperNodes for reward backup. Reverse (``supernodes_to_dag``): losslessly
rebuild the DAG from a persisted SuperNode log, then
(``replay_prefix_for``) derive a branch replay prefix shaped to
``BranchMaterializer.materialize``'s inputs.

Pure: no mutation of inputs, no I/O, torch-free. All failures raise ``DAGError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from customized_areal.tree_search.agents.event_model import message_timeline
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def _adjacency(
    dag: ExecutionDAG,
) -> tuple[
    dict[str, list[tuple[str, EdgeType]]], dict[str, list[tuple[str, EdgeType]]]
]:
    """Build per-node incoming/outgoing typed-edge lists from ``dag.edges``."""
    incoming: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.event_ids()}
    outgoing: dict[str, list[tuple[str, EdgeType]]] = {n: [] for n in dag.event_ids()}
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
                f"index {index_of[e.src]} >= index_of[e.dst]}"
            )


def dag_to_supernodes(
    dag: ExecutionDAG,
    *,
    ordering: Sequence[str] | None = None,
    nodes_by_segment: dict | None = None,
) -> list[SuperNode]:
    """Linearize ``dag`` into completion-ordered SuperNodes.

    ``ordering`` is an explicit completion order (list of node_id). If omitted,
    ``dag.topological_order()`` is used (deterministic; valid because completion
    order is always a topological order). ``nodes_by_segment`` overrides the
    per-segment ``nodes`` payload (else ``segment.metadata['nodes']``).
    """
    event_ids = dag.event_ids()
    if ordering is None:
        order = [n.node_id for n in dag.topological_order()]
    else:
        order = list(ordering)
        if sorted(order) != sorted(event_ids):
            raise DAGError("ordering is not a permutation of the DAG event ids")
        _validate_topological(dag, order)

    incoming, outgoing = _adjacency(dag)
    nodes_map = nodes_by_segment or {}
    supers: list[SuperNode] = []
    for idx, nid in enumerate(order):
        node = dag.get(nid)
        payload = nodes_map.get(nid)
        if payload is None:
            payload = node.metadata.get("nodes", [])
        super_node = SuperNode(
            node_id=node.node_id,
            agent_id=node.agent_id,
            issue_id=node.issue_id,
            task_id=node.task_id,
            closing_event=node.closing_event,
            closing_event_target=node.closing_event_target,
            session_id=node.session_id,
            completion_index=idx,
            completion_time=node.metadata.get("completion_time"),
            incoming_edges=tuple(incoming[nid]),
            outgoing_edges=tuple(outgoing[nid]),
            branch_seq=node.branch_seq,
            branch_issue_id=node.branch_issue_id,
            branch_env_snapshot_id=node.branch_env_snapshot_id,
            value=node.value,
            process_reward=node.process_reward,
            outcome_reward=node.outcome_reward,
            sandbox_ids=list(node.sandbox_ids),
            issue_snapshot_id=node.issue_snapshot_id,
            env_state=dict(node.env_state),
            nodes=list(payload),
            metadata=dict(node.metadata),
        )
        supers.append(super_node)
    return supers


def supernodes_to_dag(supers: Sequence[SuperNode]) -> ExecutionDAG:
    """Losslessly rebuild an ExecutionDAG from a linear SuperNode log.

    Steps (each failure raises ``DAGError``):
      1. completion_index must be dense 0..n-1, unique, non-negative.
      2. node_ids must be unique (no duplicate node_id across SuperNodes).
      3. edge lists must be symmetric (every A.outgoing (A->B) has a matching
         B.incoming (A->B) with the same EdgeType).
      4. add SuperNodes (faithful; nodes/completion_index stay in the log only).
      5. add edges (idempotent).
      6. enforce the topological-order invariant: for every edge src->dst,
         index(src) < index(dst).
    """
    supers = list(supers)
    if not supers:
        return ExecutionDAG()

    indices = sorted(s.completion_index for s in supers)
    if indices != list(range(len(supers))):
        raise DAGError(
            f"completion_index must be dense 0..{len(supers) - 1}, got {indices}"
        )
    ordered = sorted(supers, key=lambda s: s.completion_index)
    index_of = {s.node_id: s.completion_index for s in ordered}
    if len(index_of) != len(ordered):
        raise DAGError("duplicate node_id across SuperNodes")

    incoming_set = {
        (src, s.node_id, t) for s in ordered for (src, t) in s.incoming_edges
    }
    outgoing_set = {
        (s.node_id, dst, t) for s in ordered for (dst, t) in s.outgoing_edges
    }
    if incoming_set != outgoing_set:
        diff = incoming_set ^ outgoing_set
        raise DAGError(
            f"edge symmetry mismatch (incoming XOR outgoing): {sorted(diff)}"
        )

    dag = ExecutionDAG()
    for s in ordered:
        dag.add_event(
            SuperNode(
                node_id=s.node_id,
                agent_id=s.agent_id,
                issue_id=s.issue_id,
                task_id=s.task_id,
                closing_event=s.closing_event,
                closing_event_target=s.closing_event_target,
                session_id=s.session_id,
                branch_seq=s.branch_seq,
                branch_issue_id=s.branch_issue_id,
                branch_env_snapshot_id=s.branch_env_snapshot_id,
                process_reward=s.process_reward,
                outcome_reward=s.outcome_reward,
                value=s.value,
                sandbox_ids=list(s.sandbox_ids),
                issue_snapshot_id=s.issue_snapshot_id,
                env_state=dict(s.env_state),
                nodes=list(s.nodes),
                metadata=dict(s.metadata),
            )
        )

    for src, dst, t in sorted(incoming_set):
        dag.add_edge(src, dst, t)

    for src, dst, _ in incoming_set:
        if index_of[src] >= index_of[dst]:
            raise DAGError(
                f"event order is not topological: edge {src!r}->{dst!r} has "
                f"index {index_of[src]} >= index_of[dst]}"
            )
    return dag


@dataclass(frozen=True)
class ReplayPrefix:
    """Branch replay data, shaped to ``BranchMaterializer.materialize`` inputs."""

    replay_messages: list[dict]
    task_id: str
    seq: int
    source_issue_id: str
    branch_env_snapshot_id: str | None


def replay_prefix_for(
    supers: Sequence[SuperNode],
    *,
    branch_point: tuple[str, int],
) -> ReplayPrefix:
    """Derive the replay prefix for a branch point from a linear SuperNode log.

    ``branch_point = (task_id, seq)`` where ``seq`` is the ``task_message.seq``
    the run is allowed to branch at. Locates the unique SuperNode with matching
    ``task_id`` and ``branch_seq == seq`` (zero or multiple matches -> DAGError),
    collects that node's ancestors (plus the node itself) in completion order,
    and flattens their message payloads via ``message_timeline``.
    """
    task_id, seq = branch_point
    dag = supernodes_to_dag(supers)
    matches = [s for s in supers if s.task_id == task_id and s.branch_seq == seq]
    if len(matches) != 1:
        raise DAGError(
            f"branch point (task_id={task_id!r}, seq={seq}) matched {len(matches)} "
            f"nodes; expected exactly 1"
        )
    branch_super = matches[0]
    ancestor_ids = dag.ancestors(branch_super.node_id) | {branch_super.node_id}
    prefix_supers = sorted(
        (s for s in supers if s.node_id in ancestor_ids),
        key=lambda s: s.completion_index,
    )
    return ReplayPrefix(
        replay_messages=message_timeline(prefix_supers),
        task_id=task_id,
        seq=seq,
        source_issue_id=branch_super.issue_id,
        branch_env_snapshot_id=branch_super.branch_env_snapshot_id,
    )


__all__ = ["ReplayPrefix", "dag_to_supernodes", "replay_prefix_for", "supernodes_to_dag"]
```

- [ ] **Step 4: Add to_dict / from_dict to SuperNode**

The codec test calls `SuperNode.to_dict()` / `SuperNode.from_dict()`. Add these to `customized_areal/tree_search/agents/execution_dag.py` inside the `SuperNode` class (after `branch_node_id` property):

```python
    def to_dict(self) -> dict:
        """Emit a plain JSON-safe dict (EdgeType -> str, tuples -> lists).

        ``nodes`` is serialized via each node's own ``to_dict()`` if present,
        else the raw object (caller's responsibility). Edge tuples become
        ``[[node_id, edge_type_str], ...]``.
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
            "branch_seq": self.branch_seq,
            "branch_issue_id": self.branch_issue_id,
            "branch_env_snapshot_id": self.branch_env_snapshot_id,
            "value": self.value,
            "process_reward": self.process_reward,
            "outcome_reward": self.outcome_reward,
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
                branch_seq=d.get("branch_seq"),
                branch_issue_id=d.get("branch_issue_id"),
                branch_env_snapshot_id=d.get("branch_env_snapshot_id"),
                value=d.get("value"),
                process_reward=d.get("process_reward", 0.0),
                outcome_reward=d.get("outcome_reward", 0.0),
                sandbox_ids=list(d.get("sandbox_ids", [])),
                issue_snapshot_id=d.get("issue_snapshot_id"),
                env_state=dict(d.get("env_state", {})),
                nodes=list(d.get("nodes", [])),
                metadata=dict(d.get("metadata", {})),
            )
        except KeyError as exc:
            raise DAGError(f"SuperNode.from_dict missing required field: {exc}") from exc

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_event_codec.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/agents/event_codec.py customized_areal/tree_search/agents/execution_dag.py customized_areal/tree_search/tests/test_event_codec.py
git commit -m "refactor: rename codec to dag_to_supernodes/supernodes_to_dag

Pure rename + type re-annotation; algorithm unchanged. dag_to_supernodes
linearizes an ExecutionDAG into list[SuperNode]; supernodes_to_dag rebuilds
it losslessly. Adds SuperNode.to_dict/from_dict for the round-trip test
(edges + env snapshot + nodes serialized)."
```

---

## Task 4: Update gae.py — events_from_nodes constructs SuperNode

**Why:** `gae.py` imports `Event` and constructs it in `events_from_nodes` (line 117-130). After the `Event` removal, this breaks at import time. Pure rename: construct `SuperNode` instead. `GlobalEvent` and the GAE algorithm are unchanged (Phase 3 logic, out of scope, but the import must not break).

**Files:**
- Modify: `customized_areal/tree_search/agents/gae.py:117-138`
- Test: `customized_areal/tree_search/tests/test_event_codec.py` (already has `events_from_nodes` parity test from Task 3 — verify it still passes)

- [ ] **Step 1: Write the failing test — events_from_nodes works with SuperNode**

Append to `customized_areal/tree_search/tests/test_event_codec.py`:

```python
def test_events_from_nodes_projects_supernode_to_global_event():
    """gae.events_from_nodes must accept SuperNode (not Event) after the rename."""
    from customized_areal.tree_search.agents.gae import events_from_nodes

    dag = _build_dag()
    nodes = [dag.get(nid) for nid in ORDER]
    events = events_from_nodes(nodes)
    assert [e.node_id for e in events] == ORDER
    # O1 has outcome_reward=1.0; its reward field = process + outcome = 1.0
    assert events[-1].reward == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_event_codec.py::test_events_from_nodes_projects_supernode_to_global_event -v`
Expected: FAIL — `ImportError: cannot import name 'Event' from 'event_model'` raised inside `gae.events_from_nodes`.

- [ ] **Step 3: Update gae.py to construct SuperNode**

Edit `customized_areal/tree_search/agents/gae.py`. Replace the `events_from_nodes` function (lines 105-138):

```python
def events_from_nodes(ordered_nodes: list) -> list[GlobalEvent]:
    """Build the global event sequence from DAG nodes in completion order.

    Thin projection over the canonical :class:`SuperNode`: each node becomes a
    ``SuperNode`` (edges/messages irrelevant to GAE are left empty), then is
    projected to a :class:`GlobalEvent` with ``value`` (``V_{t+1}``; 0.0 if
    unscored) and a step reward of ``process_reward + outcome_reward`` -- so the
    verifier terminal reward (on ``outcome_reward``) flows in as ``r_t``.

    Duck-typed against ``SuperNode``; identity fields are read defensively so
    minimal node-likes still work (they do not affect the projection).
    """
    from customized_areal.tree_search.agents.execution_dag import SuperNode

    events: list[GlobalEvent] = []
    for idx, node in enumerate(ordered_nodes):
        super_node = SuperNode(
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
                node_id=super_node.node_id,
                value=float(super_node.value) if super_node.value is not None else 0.0,
                reward=float(super_node.process_reward)
                + float(super_node.outcome_reward),
            )
        )
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_event_codec.py customized_areal/tree_search/tests/test_gae.py -v`
Expected: PASS — `events_from_nodes_parity_with_old_logic` and the new projection test both pass.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/gae.py customized_areal/tree_search/tests/test_event_codec.py
git commit -m "refactor: gae.events_from_nodes constructs SuperNode not Event

Pure rename; GAE algorithm unchanged. The function duck-types against
SuperNode's identity fields (node_id/agent_id/issue_id/task_id/value/
process_reward/outcome_reward), so the projection still works."
```

---

## Task 5: SuperNodeAssembler + Multica data structures (new module)

**Why:** This is the core new module of Phase 1a. `SuperNodeAssembler.assemble` implements the 6-step algorithm from spec §5.4: resolve session→agent_run→nodes, slice by turn range, construct SuperNodes, build DAG from edges, set the unified `parent_node_id` chain with explicit precedence, identify the unique sink. Pure, torch-free, no I/O.

**Files:**
- Create: `customized_areal/tree_search/agents/supernode_assembler.py`
- Test: `customized_areal/tree_search/tests/test_supernode_assembler.py`

- [ ] **Step 1: Write the failing test — the 3-segment planner→worker→synthesizer DAG**

Create `customized_areal/tree_search/tests/test_supernode_assembler.py`:

```python
"""Unit tests for SuperNodeAssembler (spec §5.4 assemble algorithm).

Pure, torch-free. Uses SimpleNamespace stand-ins for Node (the assembler only
reads .node_id and .parent_node_id; Node is torch-lazy and heavy to build).

Fixture: a 3-segment DAG --
  planner (1 segment, 3 turns) --delegation--> worker (1 segment, 2 turns)
  worker --completion--> synthesizer (1 segment, 1 turn)
The DAG's unique sink is the synthesizer segment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
)
from customized_areal.tree_search.agents.supernode_assembler import (
    DagResult,
    EdgeSpec,
    SegmentSpec,
    SuperNodeAssembler,
    TeamEnvSnapshot,
)


def _node(node_id: str) -> SimpleNamespace:
    """Build a Node-like stand-in (the assembler reads .node_id / .parent_node_id)."""
    return SimpleNamespace(node_id=node_id, parent_node_id=None)


def _nodes(*ids: str) -> list:
    return [_node(i) for i in ids]


def _planner_worker_synthesizer() -> tuple[dict, DagResult]:
    """3-segment DAG: planner delegates to worker; worker completes to synth."""
    # Agent run "planner" has 3 turns; worker has 2; synthesizer has 1.
    sessions_nodes = {
        "sess-planner": _nodes("p1", "p2", "p3"),
        "sess-worker": _nodes("w1", "w2"),
        "sess-synth": _nodes("s1"),
    }
    dag_result = DagResult(
        session_ids=["sess-planner", "sess-worker", "sess-synth"],
        session_to_agent_run={
            "sess-planner": "run-planner",
            "sess-worker": "run-worker",
            "sess-synth": "run-synth",
        },
        segments=[
            SegmentSpec(
                segment_id="seg-planner",
                agent_run_id="run-planner",
                issue_id="iss-planner",
                task_id="task-root",
                closing_event=EdgeType.DELEGATION,
                closing_event_target_segment="seg-worker",
                start_turn_idx=1,
                end_turn_idx=3,
            ),
            SegmentSpec(
                segment_id="seg-worker",
                agent_run_id="run-worker",
                issue_id="iss-worker",
                task_id="task-root",
                closing_event=EdgeType.COMPLETION,
                closing_event_target_segment="seg-synth",
                start_turn_idx=1,
                end_turn_idx=2,
            ),
            SegmentSpec(
                segment_id="seg-synth",
                agent_run_id="run-synth",
                issue_id="iss-synth",
                task_id="task-root",
                closing_event=None,
                closing_event_target_segment=None,
                start_turn_idx=1,
                end_turn_idx=1,
            ),
        ],
        edges=[
            EdgeSpec("seg-planner", "seg-worker", EdgeType.DELEGATION),
            EdgeSpec("seg-worker", "seg-synth", EdgeType.COMPLETION),
        ],
        env_snapshots={
            "seg-planner": TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id="iss-snap-planner",
                env_state={"phase": "plan"},
            ),
            "seg-worker": TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id="iss-snap-worker",
                env_state={"phase": "work"},
            ),
            "seg-synth": TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id=None,
                env_state={"phase": "done"},
            ),
        },
    )
    return sessions_nodes, dag_result


# -- Step 1+2: slicing ----------------------------------------------------


def test_assemble_slices_each_agent_run_by_turn_range():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert [n.node_id for n in by_seg["seg-planner"].nodes] == ["p1", "p2", "p3"]
    assert [n.node_id for n in by_seg["seg-worker"].nodes] == ["w1", "w2"]
    assert [n.node_id for n in by_seg["seg-synth"].nodes] == ["s1"]


def test_assemble_rejects_out_of_range_turn_indices():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Corrupt: planner segment claims turns 1..5 but only 3 exist.
    dag_result.segments[0] = SegmentSpec(
        segment_id="seg-planner",
        agent_run_id="run-planner",
        issue_id="iss-planner",
        task_id="task-root",
        closing_event=EdgeType.DELEGATION,
        closing_event_target_segment="seg-worker",
        start_turn_idx=1,
        end_turn_idx=5,
    )
    with pytest.raises(DAGError, match="out of range"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


def test_assemble_rejects_overlapping_segments_within_run():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Make two segments of the same run overlap on turn 2.
    dag_result.segments.append(
        SegmentSpec(
            segment_id="seg-planner-2",
            agent_run_id="run-planner",
            issue_id="iss-planner",
            task_id="task-root",
            closing_event=None,
            closing_event_target_segment=None,
            start_turn_idx=2,
            end_turn_idx=3,
        )
    )
    with pytest.raises(DAGError, match="overlap"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


# -- Step 3: SuperNode construction --------------------------------------


def test_assemble_stamps_env_snapshot_on_each_supernode():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert by_seg["seg-planner"].sandbox_ids == ["sb-p", "sb-w", "sb-s"]
    assert by_seg["seg-planner"].issue_snapshot_id == "iss-snap-planner"
    assert by_seg["seg-planner"].env_state == {"phase": "plan"}
    assert by_seg["seg-synth"].env_state == {"phase": "done"}


def test_assemble_binds_session_id_to_each_supernode():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert by_seg["seg-planner"].session_id == "sess-planner"
    assert by_seg["seg-worker"].session_id == "sess-worker"
    assert by_seg["seg-synth"].session_id == "sess-synth"


def test_assemble_sets_closing_event_and_target():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    assert by_seg["seg-planner"].closing_event is EdgeType.DELEGATION
    # closing_event_target is resolved to the other SuperNode's UUID.
    assert by_seg["seg-planner"].closing_event_target == by_seg["seg-worker"].node_id
    assert by_seg["seg-synth"].closing_event is None


def test_assemble_rejects_missing_env_snapshot():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    del dag_result.env_snapshots["seg-worker"]
    with pytest.raises(DAGError, match="env_snapshots"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


# -- Step 4: DAG construction --------------------------------------------


def test_assemble_builds_execution_dag_from_edges():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    _, dag, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in dag.events}
    # planner -> worker (delegation), worker -> synth (completion)
    assert {(e.src, e.dst, e.type) for e in dag.edges} == {
        (by_seg["seg-planner"].node_id, by_seg["seg-worker"].node_id, EdgeType.DELEGATION),
        (by_seg["seg-worker"].node_id, by_seg["seg-synth"].node_id, EdgeType.COMPLETION),
    }


def test_assemble_assigns_dense_completion_index():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    indices = sorted(s.completion_index for s in supers)
    assert indices == list(range(len(supers)))


def test_assemble_rejects_unknown_segment_in_edge():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    dag_result.edges.append(
        EdgeSpec("seg-planner", "seg-ghost", EdgeType.MENTION)
    )
    with pytest.raises(DAGError, match="unknown segment"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


# -- Step 5: parent_node_id chain (causal flattening) --------------------


def test_assemble_sets_within_segment_parent_chain():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # planner segment: p1 (root, parent None) -> p2 -> p3
    p_nodes = by_seg["seg-planner"].nodes
    assert p_nodes[0].parent_node_id is None
    assert p_nodes[1].parent_node_id == p_nodes[0].node_id
    assert p_nodes[2].parent_node_id == p_nodes[1].node_id


def test_assemble_sets_cross_agent_delegation_parent():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # planner's terminal (p3) delegates to worker; worker's first turn (w1)
    # has parent_node_id = p3.node_id.
    planner_terminal = by_seg["seg-planner"].terminal_node
    worker_first = by_seg["seg-worker"].nodes[0]
    assert worker_first.parent_node_id == planner_terminal.node_id


def test_assemble_sets_cross_agent_completion_parent():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # worker's terminal (w2) completes to synthesizer; synthesizer's first
    # turn (s1) has parent_node_id = w2.node_id.
    worker_terminal = by_seg["seg-worker"].terminal_node
    synth_first = by_seg["seg-synth"].nodes[0]
    assert synth_first.parent_node_id == worker_terminal.node_id


def test_assemble_mention_edge_does_not_set_parent():
    """A MENTION edge records topology only; it must NOT set parent_node_id."""
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Add a peer MENTION edge: planner mentions synth (non-blocking).
    dag_result.edges.append(
        EdgeSpec("seg-planner", "seg-synth", EdgeType.MENTION)
    )
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    by_seg = {s.metadata["_segment_id"]: s for s in supers}
    # synth's first turn's parent must still be worker's terminal (COMPLETION),
    # NOT planner's terminal (MENTION). MENTION is topology-only.
    worker_terminal = by_seg["seg-worker"].terminal_node
    synth_first = by_seg["seg-synth"].nodes[0]
    assert synth_first.parent_node_id == worker_terminal.node_id


# -- Step 6: root_terminal_node_id ---------------------------------------


def test_assemble_returns_unique_sink_terminal_node_id():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    _, _, root_terminal = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    # Synthesizer is the unique sink (no outgoing edges).
    by_seg = {s.metadata["_segment_id"]: s for s in _supers_from_assemble(sessions_nodes, dag_result)}
    assert root_terminal == by_seg["seg-synth"].terminal_node.node_id


def _supers_from_assemble(sessions_nodes, dag_result):
    supers, _, _ = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    return supers


def test_assemble_rejects_multiple_sinks():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Add a second leaf segment with no outgoing edges.
    dag_result.segments.append(
        SegmentSpec(
            segment_id="seg-orphan",
            agent_run_id="run-synth",  # reuse an existing run's session
            issue_id="iss-orphan",
            task_id="task-root",
            closing_event=None,
            closing_event_target_segment=None,
            start_turn_idx=1,
            end_turn_idx=1,
        )
    )
    # The orphan segment needs a session; reuse sess-synth's nodes (1 node).
    sessions_nodes["sess-synth"].append(_node("s1-orphan"))
    # Actually we need a 4th segment that's a sink. Easiest: make seg-orphan
    # its own run with its own session, no edges in or out.
    dag_result.segments[-1] = SegmentSpec(
        segment_id="seg-orphan",
        agent_run_id="run-orphan",
        issue_id="iss-orphan",
        task_id="task-root",
        closing_event=None,
        closing_event_target_segment=None,
        start_turn_idx=1,
        end_turn_idx=1,
    )
    sessions_nodes["sess-orphan"] = _nodes("o1")
    dag_result.session_ids.append("sess-orphan")
    dag_result.session_to_agent_run["sess-orphan"] = "run-orphan"
    dag_result.env_snapshots["seg-orphan"] = TeamEnvSnapshot(
        sandbox_ids=[], issue_snapshot_id=None, env_state={}
    )
    with pytest.raises(DAGError, match="multiple sinks"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )


def test_assemble_rejects_dangling_agent_run_id():
    sessions_nodes, dag_result = _planner_worker_synthesizer()
    # Reference an agent_run_id that has no session.
    dag_result.segments[0] = SegmentSpec(
        segment_id="seg-planner",
        agent_run_id="run-ghost",
        issue_id="iss-planner",
        task_id="task-root",
        closing_event=EdgeType.DELEGATION,
        closing_event_target_segment="seg-worker",
        start_turn_idx=1,
        end_turn_idx=3,
    )
    with pytest.raises(DAGError, match="dangling"):
        SuperNodeAssembler().assemble(
            sessions_nodes=sessions_nodes, dag_result=dag_result
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_supernode_assembler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.agents.supernode_assembler'`.

- [ ] **Step 3: Implement SuperNodeAssembler + Multica data structures**

Create `customized_areal/tree_search/agents/supernode_assembler.py`:

```python
"""Assemble SuperNodes from Multica's segment specs + proxy interactions.

Consumes Multica's pre-defined segments (does NOT infer boundaries). Maps each
agent's list[Node] into the segments Multica defined, by the start_turn_idx /
end_turn_idx range on each SegmentSpec.

Maintains the unified parent_node_id chain (causal flattening):
  - Within one run: n_{k+1}.parent_node_id = n_k.node_id
  - Cross-agent delegation (A's n3 delegates -> B's n1'):
    n1'.parent_node_id = n3.node_id
  - Cross-agent completion (B completes -> A continues@n4):
    n4.parent_node_id = B's terminal node_id
  - MENTION edges record topology only; they do NOT set parent_node_id.

Stamps TeamEnvSnapshot onto each SuperNode. Binds session_id to each SuperNode
(from session_to_agent_run).

Pure: no I/O, no mutation of inputs, torch-free. All failures raise DAGError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


@dataclass(frozen=True)
class SegmentSpec:
    """One communication-bounded segment, as defined by Multica.

    Multica decides segment boundaries at communication events during
    execution; AReal does not infer them.
    """

    segment_id: str
    agent_run_id: str
    issue_id: str
    task_id: str
    closing_event: EdgeType | None
    closing_event_target_segment: str | None
    # 1-based inclusive turn range within this agent run.
    start_turn_idx: int
    end_turn_idx: int


@dataclass(frozen=True)
class EdgeSpec:
    """One typed DAG edge between segments, as defined by Multica."""

    src_segment_id: str
    dst_segment_id: str
    type: EdgeType


@dataclass(frozen=True)
class TeamEnvSnapshot:
    """Team-wide environment state at a branching point, from Multica.

    One per segment, captured at the moment that segment's closing
    communication event fired. Multica owns the env state; AReal stamps
    it onto the corresponding SuperNode without interpretation.
    """

    sandbox_ids: list[str]
    issue_snapshot_id: str | None
    env_state: dict


@dataclass(frozen=True)
class DagResult:
    """Multica's completion payload: the DAG of segments + env snapshots."""

    session_ids: list[str]
    session_to_agent_run: dict[str, str]
    segments: list[SegmentSpec]
    edges: list[EdgeSpec]
    env_snapshots: dict[str, TeamEnvSnapshot]


class SuperNodeAssembler:
    """Assemble SuperNodes from Multica's segment specs + proxy interactions."""

    def assemble(
        self,
        *,
        sessions_nodes: dict[str, list[Any]],
        dag_result: DagResult,
    ) -> tuple[list[SuperNode], ExecutionDAG, str]:
        """Returns (super_nodes, dag, root_terminal_node_id).

        root_terminal_node_id is the terminal node of the DAG's unique sink
        -- the starting point for backup_episode_terminal.
        """
        # ── Step 1: resolve session -> agent_run -> list[Node] ─────────
        agent_run_to_session: dict[str, str] = {}
        for session_id, agent_run_id in dag_result.session_to_agent_run.items():
            agent_run_to_session[agent_run_id] = session_id

        # ── Step 2: slice each agent run's list[Node] into segments ────
        segment_id_to_nodes: dict[str, list[Any]] = {}
        # Track per-run turn coverage to detect gaps/overlaps.
        run_coverage: dict[str, list[tuple[int, int, str]]] = {}
        for spec in dag_result.segments:
            session_id = agent_run_to_session.get(spec.agent_run_id)
            if session_id is None:
                raise DAGError(
                    f"segment {spec.segment_id!r} references dangling "
                    f"agent_run_id {spec.agent_run_id!r}"
                )
            agent_nodes = sessions_nodes.get(session_id)
            if agent_nodes is None:
                raise DAGError(
                    f"segment {spec.segment_id!r}: session {session_id!r} "
                    f"not in sessions_nodes"
                )
            if spec.start_turn_idx < 1 or spec.end_turn_idx < spec.start_turn_idx:
                raise DAGError(
                    f"segment {spec.segment_id!r}: invalid turn range "
                    f"[{spec.start_turn_idx}, {spec.end_turn_idx}]"
                )
            if spec.end_turn_idx > len(agent_nodes):
                raise DAGError(
                    f"segment {spec.segment_id!r}: turn range out of range "
                    f"(end_turn_idx={spec.end_turn_idx} > "
                    f"len(nodes)={len(agent_nodes)})"
                )
            segment_nodes = agent_nodes[spec.start_turn_idx - 1 : spec.end_turn_idx]
            if not segment_nodes:
                raise DAGError(
                    f"segment {spec.segment_id!r}: slice is empty"
                )
            segment_id_to_nodes[spec.segment_id] = segment_nodes
            run_coverage.setdefault(spec.agent_run_id, []).append(
                (spec.start_turn_idx, spec.end_turn_idx, spec.segment_id)
            )
        # Validate dense, non-overlapping coverage per run.
        for agent_run_id, ranges in run_coverage.items():
            ranges_sorted = sorted(ranges, key=lambda r: r[0])
            for i in range(1, len(ranges_sorted)):
                prev_end = ranges_sorted[i - 1][1]
                cur_start = ranges_sorted[i][0]
                if cur_start <= prev_end:
                    raise DAGError(
                        f"agent_run {agent_run_id!r}: segments "
                        f"{ranges_sorted[i - 1][2]!r} and "
                        f"{ranges_sorted[i][2]!r} overlap or are not dense"
                    )

        # ── Step 3: construct each SuperNode ──────────────────────────
        segment_id_to_super: dict[str, SuperNode] = {}
        supers: list[SuperNode] = []
        for spec in dag_result.segments:
            env_snapshot = dag_result.env_snapshots.get(spec.segment_id)
            if env_snapshot is None:
                raise DAGError(
                    f"segment {spec.segment_id!r}: missing env_snapshots entry"
                )
            session_id = agent_run_to_session[spec.agent_run_id]
            super_node = SuperNode(
                node_id=self._fresh_uuid(),
                agent_id=spec.agent_run_id,  # use run id as agent_id stand-in
                issue_id=spec.issue_id,
                task_id=spec.task_id,
                closing_event=spec.closing_event,
                closing_event_target=None,  # resolved after all SuperNodes built
                session_id=session_id,
                sandbox_ids=list(env_snapshot.sandbox_ids),
                issue_snapshot_id=env_snapshot.issue_snapshot_id,
                env_state=dict(env_snapshot.env_state),
                nodes=list(segment_id_to_nodes[spec.segment_id]),
                metadata={"_segment_id": spec.segment_id},
            )
            segment_id_to_super[spec.segment_id] = super_node
            supers.append(super_node)
        # Backfill closing_event_target by resolving segment_id -> SuperNode UUID.
        for spec in dag_result.segments:
            if spec.closing_event_target_segment is None:
                continue
            target_super = segment_id_to_super.get(
                spec.closing_event_target_segment
            )
            if target_super is None:
                raise DAGError(
                    f"segment {spec.segment_id!r}: closing_event_target_segment "
                    f"{spec.closing_event_target_segment!r} is unknown"
                )
            segment_id_to_super[spec.segment_id].closing_event_target = (
                target_super.node_id
            )

        # ── Step 4: build ExecutionDAG from EdgeSpecs ──────────────────
        dag = ExecutionDAG()
        for super_node in supers:
            dag.add_event(super_node)
        for edge_spec in dag_result.edges:
            src_super = segment_id_to_super.get(edge_spec.src_segment_id)
            dst_super = segment_id_to_super.get(edge_spec.dst_segment_id)
            if src_super is None or dst_super is None:
                raise DAGError(
                    f"edge references unknown segment: "
                    f"{edge_spec.src_segment_id!r} -> {edge_spec.dst_segment_id!r}"
                )
            dag.add_edge(src_super.node_id, dst_super.node_id, edge_spec.type)
        # Assign completion_index from topological order.
        topo = dag.topological_order()
        for idx, super_node in enumerate(topo):
            super_node.completion_index = idx
        # Populate incoming_edges / outgoing_edges on each SuperNode.
        for super_node in supers:
            incoming = [
                (e.src, e.type) for e in dag.edges if e.dst == super_node.node_id
            ]
            outgoing = [
                (e.dst, e.type) for e in dag.edges if e.src == super_node.node_id
            ]
            super_node.incoming_edges = tuple(incoming)
            super_node.outgoing_edges = tuple(outgoing)

        # ── Step 5: set the unified parent_node_id chain ──────────────
        # (a) Within-segment sequential.
        for super_node in supers:
            for i in range(1, len(super_node.nodes)):
                super_node.nodes[i].parent_node_id = super_node.nodes[i - 1].node_id
        # Precompute, per segment, the set of incoming DELEGATION/COMPLETION
        # edges (for the (b) gate + (c)/(d) overrides).
        blocking_incoming: dict[str, list[EdgeSpec]] = {}
        for edge_spec in dag_result.edges:
            if edge_spec.type in (EdgeType.DELEGATION, EdgeType.COMPLETION):
                blocking_incoming.setdefault(
                    edge_spec.dst_segment_id, []
                ).append(edge_spec)
        # (b) Within-run cross-segment (default): for each run, sort segments
        # by start_turn_idx; for segment[k] (k>0) with no blocking incoming
        # edge, set its first node's parent to segment[k-1]'s terminal.
        segments_by_run: dict[str, list[SegmentSpec]] = {}
        for spec in dag_result.segments:
            segments_by_run.setdefault(spec.agent_run_id, []).append(spec)
        for agent_run_id, run_segments in segments_by_run.items():
            run_segments_sorted = sorted(run_segments, key=lambda s: s.start_turn_idx)
            for k in range(1, len(run_segments_sorted)):
                cur_spec = run_segments_sorted[k]
                if cur_spec.segment_id in blocking_incoming:
                    continue  # (c) or (d) will set the parent
                prev_spec = run_segments_sorted[k - 1]
                prev_super = segment_id_to_super[prev_spec.segment_id]
                cur_super = segment_id_to_super[cur_spec.segment_id]
                if cur_super.nodes:
                    prev_terminal = prev_super.terminal_node
                    if prev_terminal is not None:
                        cur_super.nodes[0].parent_node_id = prev_terminal.node_id
        # (c) + (d): for each blocking incoming edge, set dst's first node
        # parent to src's terminal node_id.
        for edge_spec in dag_result.edges:
            if edge_spec.type not in (EdgeType.DELEGATION, EdgeType.COMPLETION):
                continue
            src_super = segment_id_to_super[edge_spec.src_segment_id]
            dst_super = segment_id_to_super[edge_spec.dst_segment_id]
            src_terminal = src_super.terminal_node
            if src_terminal is None or not dst_super.nodes:
                continue
            dst_super.nodes[0].parent_node_id = src_terminal.node_id
        # MENTION edges: no parent_node_id set (topology-only), by omission.

        # ── Step 6: identify unique sink ──────────────────────────────
        sinks = [s for s in supers if not s.outgoing_edges]
        if len(sinks) != 1:
            raise DAGError(
                f"DAG must have exactly one sink (found {len(sinks)}); "
                f"sinks={[s.node_id for s in sinks]}"
            )
        root_terminal = sinks[0].terminal_node
        if root_terminal is None:
            raise DAGError(
                f"sink SuperNode {sinks[0].node_id!r} has no terminal node"
            )
        return supers, dag, root_terminal.node_id

    @staticmethod
    def _fresh_uuid() -> str:
        import uuid

        return str(uuid.uuid4())


__all__ = [
    "DagResult",
    "EdgeSpec",
    "SegmentSpec",
    "SuperNodeAssembler",
    "TeamEnvSnapshot",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_supernode_assembler.py -v`
Expected: PASS — all 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/agents/supernode_assembler.py customized_areal/tree_search/tests/test_supernode_assembler.py
git commit -m "feat: add SuperNodeAssembler + Multica data structures

SuperNodeAssembler.assemble implements the 6-step algorithm from spec
§5.4: resolve session->agent_run->nodes, slice by turn range, construct
SuperNodes (env snapshot + session_id + closing_event), build ExecutionDAG
from EdgeSpecs, set the unified parent_node_id chain (within-segment ->
within-run cross-segment -> cross-agent DELEGATION/COMPLETION override;
MENTION is topology-only), identify the unique sink for root_terminal.

Pure, torch-free, no I/O. All validation raises DAGError."
```

---

## Task 6: Unified MCTSTreeStore — trajectories holds SuperNodes

**Why:** The tree store currently holds `list[Node]` per query. After unification, it holds `list[SuperNode]` (each carrying its own `nodes`). MCTS stats stay per-Node (keyed by `node_id`). The dual indices let `get_node` walk into a SuperNode's `nodes` and `get_super_node` look up by UUID. `backup_episode_terminal` / `backup_path_returns` walk the unified `parent_node_id` chain unchanged — the cross-SuperNode and cross-agent links are already set by the assembler, so the walk transparently crosses boundaries.

**Files:**
- Modify: `customized_areal/tree_search/core/tree_store.py:251-843` (MCTSTreeStore class)
- Test: `customized_areal/tree_search/tests/test_tree_store_super.py`

- [ ] **Step 1: Write the failing test — insert_super_batch + cross-SuperNode backup**

Create `customized_areal/tree_search/tests/test_tree_store_super.py`:

```python
"""Tests for the unified MCTSTreeStore (trajectories holds SuperNodes).

SuperNodes hold DAG topology + team env snapshot; Nodes (inside SuperNode.nodes)
hold all MCTS stats + the unified parent_node_id chain. backup_episode_terminal
walks parent_node_id across SuperNode and agent boundaries (causal flattening).

Uses SimpleNamespace stand-ins for Node (the store reads .node_id /
.parent_node_id / .outcome_reward / .episode_id / .turn_idx / .train_id /
.discarded).
"""

from __future__ import annotations

from types import SimpleNamespace

from customized_areal.tree_search.core.tree_store import MCTSTreeStore
from customized_areal.tree_search.agents.execution_dag import SuperNode


def _node(node_id, *, parent_node_id=None, episode_id="ep", turn_idx=1,
          outcome_reward=0.0):
    return SimpleNamespace(
        node_id=node_id,
        parent_node_id=parent_node_id,
        episode_id=episode_id,
        turn_idx=turn_idx,
        outcome_reward=outcome_reward,
        train_id="",
        discarded=False,
    )


def _super(node_id, *, nodes, session_id=None):
    return SuperNode(
        node_id=node_id,
        agent_id="a",
        issue_id="i",
        task_id="t",
        session_id=session_id,
        nodes=nodes,
    )


def test_insert_super_batch_indexes_super_and_node_levels():
    store = MCTSTreeStore()
    s1 = _super("s1", nodes=[_node("n1"), _node("n2", parent_node_id="n1")])
    s2 = _super("s2", nodes=[_node("n3", parent_node_id="n2")])  # cross-super link
    store.insert_super_batch([s1, s2], query_id="q")
    assert store.get_super_node("s1") is s1
    assert store.get_super_node("s2") is s2
    assert store.get_node("n1") is s1.nodes[0]
    assert store.get_node("n2") is s1.nodes[1]
    assert store.get_node("n3") is s2.nodes[0]


def test_backup_episode_terminal_walks_across_super_boundary():
    # n1 -> n2 (in s1) -> n3 (in s2, parent=n2). Reward 1.0 backed up from n3.
    store = MCTSTreeStore()
    n1 = _node("n1", outcome_reward=0.0)
    n2 = _node("n2", parent_node_id="n1", outcome_reward=0.0)
    n3 = _node("n3", parent_node_id="n2", outcome_reward=0.0)
    s1 = _super("s1", nodes=[n1, n2])
    s2 = _super("s2", nodes=[n3])
    store.insert_super_batch([s1, s2], query_id="q", backup=False)
    store.backup_episode_terminal("n3", 1.0)
    # All three nodes get +1.0; n3 is terminal, n2 and n1 are ancestors via
    # the cross-super parent_node_id chain.
    assert store.get_visit_count("n3") == 1
    assert store.get_q_value("n3") == 1.0
    assert store.get_visit_count("n2") == 1
    assert store.get_q_value("n2") == 1.0
    assert store.get_visit_count("n1") == 1
    assert store.get_q_value("n1") == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_tree_store_super.py -v`
Expected: FAIL — `MCTSTreeStore` has no `insert_super_batch` method; `trajectories` is `dict[str, list[Node]]` not `dict[str, list[SuperNode]]`.

- [ ] **Step 3: Refactor MCTSTreeStore to hold SuperNodes**

Edit `customized_areal/tree_search/core/tree_store.py`. Replace the `MCTSTreeStore.__init__` (lines 260-287) and the indexing/insert/backup methods. The full new class body (replace lines 251-843):

```python
class MCTSTreeStore:
    """Unified store: SuperNodes (segments) hold DAG topology, comm-event
    provenance, and team env snapshots; Nodes hold all MCTS stats and the
    unified causal parent_node_id chain.

    Reward backup: Node-level only (parent_node_id encodes the full DAG
    causal order across agents and segments).
    """

    def __init__(self) -> None:
        # Primary storage: query_id -> SuperNodes in insertion order.
        self.trajectories: dict[str, list] = {}
        # SuperNode-level index: SuperNode UUID -> (query_id, idx_in_trajectories)
        self._super_id_to_key: dict[str, tuple[str, int]] = {}
        # Node-level index: node_id -> (super_node_id, idx_in_super_node.nodes)
        self._node_id_to_super: dict[str, tuple[str, int]] = {}
        self._query_node_ids: dict[str, list[str]] = {}

        # Per-Node MCTS stats (unchanged).
        self._visit_counts: dict[str, int] = {}
        self._total_values: dict[str, float] = {}
        self._q_values: dict[str, float] = {}
        self._sum_sq_values: dict[str, float] = {}

        self.current_train_id: str = os.environ.get("TRAIN_ID", "")
        self._rewards: dict[str, float] = {}

        self._turn_nodes: dict[str, str] = {}
        self._normalized_advantages: dict[str, float] = {}
        self._normalized_returns: dict[str, float] = {}
        self._values: dict[str, float] = {}
        self._value_variances: dict[str, float] = {}
        self._judge_scores: dict[str, list[float]] = {}

    # -- SuperNode / Node lookup -----------------------------------------

    def get_super_node(self, super_node_id: str):
        """Return the indexed SuperNode (or None if absent)."""
        key = self._super_id_to_key.get(super_node_id)
        if key is None:
            return None
        query_id, idx = key
        return self.trajectories[query_id][idx]

    def get_node(self, node_id: str):
        """Return the indexed Node (or None) by walking into its SuperNode."""
        key = self._node_id_to_super.get(node_id)
        if key is None:
            return None
        super_node_id, idx_in_nodes = key
        super_node = self.get_super_node(super_node_id)
        if super_node is None:
            return None
        return super_node.nodes[idx_in_nodes]

    def _node_parent_id(self, node_id: str) -> str | None:
        """Return the parent_node_id of an indexed node, or None."""
        node = self.get_node(node_id)
        if node is None:
            return None
        if isinstance(node, dict):
            return node.get("parent_node_id")
        return node.parent_node_id

    # -- MCTS backup (unchanged algorithm; walks parent_node_id) ---------

    def _backup_node(self, node_id: str, reward: float) -> None:
        """Add one Monte-Carlo sample (``reward``) to a single node's stats."""
        self._visit_counts[node_id] = self._visit_counts.get(node_id, 0) + 1
        self._total_values[node_id] = self._total_values.get(node_id, 0.0) + reward
        self._sum_sq_values[node_id] = (
            self._sum_sq_values.get(node_id, 0.0) + reward * reward
        )
        self._q_values[node_id] = (
            self._total_values[node_id] / self._visit_counts[node_id]
        )

    def _backup_path(self, terminal_node_id: str, reward: float) -> None:
        """Propagate one episode's return root-ward along the parent chain.

        Walks parent_node_id across SuperNode and agent boundaries (the
        unified causal chain set by SuperNodeAssembler). A ``visited`` guard
        makes the walk robust to malformed cycles.
        """
        visited: set[str] = set()
        current: str | None = terminal_node_id
        while current and current not in visited and current in self._node_id_to_super:
            visited.add(current)
            self._backup_node(current, reward)
            current = self._node_parent_id(current)

    def backup_episode_terminal(self, terminal_node_id: str, reward: float) -> None:
        """Public entry: root-ward backup of one episode's terminal return.

        Walks parent_node_id from terminal_node_id across SuperNode/agent
        boundaries (causal flattening).
        """
        self._backup_path(terminal_node_id, float(reward))

    def backup_path_returns(
        self, terminal_node_id: str, returns_by_node_id: dict[str, float]
    ) -> None:
        """Root-ward backup assigning each node on the path its own return-to-go."""
        visited: set[str] = set()
        current: str | None = terminal_node_id
        while current and current not in visited and current in self._node_id_to_super:
            visited.add(current)
            g = returns_by_node_id.get(current)
            if g is not None:
                self._backup_node(current, float(g))
            current = self._node_parent_id(current)

    # -- Insertion -------------------------------------------------------

    def insert_super_batch(
        self, supers: list, backup: bool = True, query_id: str = ""
    ) -> None:
        """Insert a batch of SuperNodes under ``query_id``.

        Indexes each SuperNode and each Node inside it. When ``backup`` is
        True, runs a root-ward backup per episode among freshly inserted nodes
        (same algorithm as the old insert_batch, but walking into SuperNodes).
        """
        inserted_node_ids: list[str] = []
        for super_node in supers:
            # Index the SuperNode.
            super_id = super_node.node_id
            if not super_id:
                raise ValueError(
                    "SuperNode must have a non-empty node_id before insert"
                )
            if super_id in self._super_id_to_key:
                continue  # idempotent
            qid = query_id or getattr(super_node, "query_id", "") or ""
            idx = len(self.trajectories.setdefault(qid, []))
            self.trajectories[qid].append(super_node)
            self._super_id_to_key[super_id] = (qid, idx)
            # Index each Node inside the SuperNode.
            for node_idx, node in enumerate(super_node.nodes):
                node_id = (
                    node.get("node_id", "") if isinstance(node, dict) else node.node_id
                )
                if not node_id:
                    continue
                if node_id in self._node_id_to_super:
                    continue  # idempotent across supers
                self._node_id_to_super[node_id] = (super_id, node_idx)
                self._query_node_ids.setdefault(qid, []).append(node_id)
                inserted_node_ids.append(node_id)
                # Record the node's own reward.
                if isinstance(node, dict):
                    outcome_reward = node.get("outcome_reward", node.get("reward", 0.0))
                else:
                    outcome_reward = node.outcome_reward
                self._rewards[node_id] = outcome_reward
        if backup:
            self._backup_inserted_episodes(inserted_node_ids)

    def _backup_inserted_episodes(self, node_ids: list[str]) -> None:
        """Run a root-ward backup once per episode among freshly inserted nodes."""
        episode_terminal: dict[str, tuple[int, str, float]] = {}
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node is None:
                continue
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "") or ""
                turn_idx = int(node.get("turn_idx", 0) or 0)
                reward = float(
                    node.get("outcome_reward", node.get("reward", 0.0)) or 0.0
                )
            else:
                ep_id = node.episode_id or ""
                turn_idx = int(node.turn_idx or 0)
                reward = float(node.outcome_reward or 0.0)
            if not ep_id:
                self._backup_path(node_id, reward)
                continue
            prev = episode_terminal.get(ep_id)
            if prev is None or turn_idx >= prev[0]:
                episode_terminal[ep_id] = (turn_idx, node_id, reward)
        for _turn_idx, terminal_id, reward in episode_terminal.values():
            self._backup_path(terminal_id, reward)

    # -- Per-Node accessors (unchanged signatures) -----------------------

    def set_trained(self, node_id: str, trained: bool = True) -> None:
        if not trained:
            return
        node = self.get_node(node_id)
        if node is None:
            return
        if isinstance(node, dict):
            node["train_id"] = self.current_train_id
        else:
            node.train_id = self.current_train_id

    def set_discarded(self, node_id: str, discarded: bool = True) -> None:
        node = self.get_node(node_id)
        if node is None:
            return
        if isinstance(node, dict):
            node["discarded"] = discarded
        else:
            node.discarded = discarded

    def is_discarded(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        if isinstance(node, dict):
            return bool(node.get("discarded", False))
        return node.discarded

    def is_trained(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        if isinstance(node, dict):
            train_id = node.get("train_id", "")
        else:
            train_id = node.train_id
        return bool(train_id) and train_id == self.current_train_id

    def get_reward(self, node_id: str) -> float:
        return self._rewards.get(node_id, 0.0)

    def get_q_value(self, node_id: str) -> float:
        return self._q_values.get(node_id, 0.0)

    def get_visit_count(self, node_id: str) -> int:
        return self._visit_counts.get(node_id, 0)

    def get_total_value(self, node_id: str) -> float:
        return self._total_values.get(node_id, 0.0)

    def get_sum_sq_value(self, node_id: str) -> float:
        return self._sum_sq_values.get(node_id, 0.0)

    def get_loo_value_and_variance(
        self, node_id: str, excluded_reward: float
    ) -> tuple[float, float, int]:
        """Leave-one-out MC value, variance-of-the-mean, and LOO sample size."""
        n = self._visit_counts.get(node_id, 0)
        n_loo = n - 1
        if n_loo < 2:
            return 0.0, -1.0, max(n_loo, 0)
        total = self._total_values.get(node_id, 0.0)
        sum_sq = self._sum_sq_values.get(node_id, 0.0)
        s_prime = total - excluded_reward
        q_prime = sum_sq - excluded_reward * excluded_reward
        loo_mean = s_prime / n_loo
        loo_var = (q_prime - s_prime * s_prime / n_loo) / (n_loo - 1)
        if loo_var < 0.0:
            loo_var = 0.0
        var_mc = loo_var / n_loo
        s_clamped = min(max(s_prime, 0.0), float(n_loo))
        a = s_clamped + 1.0
        b = (n_loo - s_clamped) + 1.0
        nn = a + b
        var_floor = (a * b) / (nn * nn * (nn + 1.0))
        if var_floor > var_mc:
            var_mc = var_floor
        return loo_mean, var_mc, n_loo

    def set_normalized_advantage(self, node_id: str, value: float) -> None:
        self._normalized_advantages[node_id] = value

    def get_normalized_advantage(self, node_id: str, default: float = 0.0) -> float:
        return self._normalized_advantages.get(node_id, default)

    def has_normalized_advantage(self, node_id: str) -> bool:
        return node_id in self._normalized_advantages

    def set_normalized_return(self, node_id: str, value: float) -> None:
        self._normalized_returns[node_id] = value

    def get_normalized_return(self, node_id: str, default: float = 0.0) -> float:
        return self._normalized_returns.get(node_id, default)

    def set_value(self, node_id: str, value: float) -> None:
        self._values[node_id] = value

    def get_value(self, node_id: str, default: float = 0.0) -> float:
        return self._values.get(node_id, default)

    def has_value(self, node_id: str) -> bool:
        return node_id in self._values

    def set_value_variance(self, node_id: str, variance: float) -> None:
        self._value_variances[node_id] = variance

    def get_value_variance(self, node_id: str, default: float = 0.0) -> float:
        return self._value_variances.get(node_id, default)

    def add_judge_score(self, node_id: str, score: float) -> None:
        self._judge_scores.setdefault(node_id, []).append(float(score))

    def get_judge_scores(self, node_id: str) -> list[float]:
        return list(self._judge_scores.get(node_id, []))

    def get_mean_judge_score(self, node_id: str) -> float | None:
        scores = self._judge_scores.get(node_id)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self._query_node_ids:
            return 0
        return sum(
            1
            for node_id in self._query_node_ids[query_id]
            if not self.is_trained(node_id) and not self.is_discarded(node_id)
        )

    def get_untrained_episode_count(self, query_id: str) -> int:
        if query_id not in self._query_node_ids:
            return 0
        episode_has_untrained: dict[str, bool] = {}
        for node_id in self._query_node_ids[query_id]:
            node = self.get_node(node_id)
            if node is None:
                continue
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_has_untrained:
                episode_has_untrained[ep_id] = False
            if not self.is_trained(node_id) and not self.is_discarded(node_id):
                episode_has_untrained[ep_id] = True
        return sum(1 for v in episode_has_untrained.values() if v)

    def load_untrained_episodes(self, query_id: str, n_episodes: int) -> list:
        if query_id not in self._query_node_ids:
            return []
        episode_nodes: dict[str, list[str]] = {}
        episode_order: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            node = self.get_node(node_id)
            if node is None:
                continue
            if isinstance(node, dict):
                ep_id = node.get("episode_id", "")
            else:
                ep_id = node.episode_id
            if not ep_id:
                continue
            if ep_id not in episode_nodes:
                episode_nodes[ep_id] = []
                episode_order.append(ep_id)
            episode_nodes[ep_id].append(node_id)
        selected: list = []
        count = 0
        for ep_id in episode_order:
            if count >= n_episodes:
                break
            is_untrained = False
            for node_id in episode_nodes[ep_id]:
                if not self.is_trained(node_id) and not self.is_discarded(node_id):
                    is_untrained = True
                    break
            if not is_untrained:
                continue
            count += 1
            for node_id in episode_nodes[ep_id]:
                node = self.get_node(node_id)
                if node is not None:
                    selected.append(node)
        return selected

    def get_untrained_node_ids(self, query_id: str, n_samples: int) -> list[str]:
        if query_id not in self._query_node_ids:
            return []
        result: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            if not self.is_trained(node_id) and not self.is_discarded(node_id):
                result.append(node_id)
                if len(result) >= n_samples:
                    break
        return result

    def load_trajectories(self, query_id: str, n_samples: int) -> list:
        if query_id not in self.trajectories:
            return []
        untrained_ids = self.get_untrained_node_ids(query_id, n_samples)
        result: list = []
        for node_id in untrained_ids:
            node = self.get_node(node_id)
            if node is not None:
                result.append(node)
        return result

    def mark_episodes_trained(self, episode_ids: set[str]) -> None:
        for query_id, supers in self.trajectories.items():
            for super_node in supers:
                for node in super_node.nodes:
                    if isinstance(node, dict):
                        ep_id = node.get("episode_id", "")
                    else:
                        ep_id = node.episode_id
                    if ep_id in episode_ids:
                        if isinstance(node, dict):
                            node["train_id"] = self.current_train_id
                        else:
                            node.train_id = self.current_train_id
                    else:
                        if isinstance(node, dict):
                            node["train_id"] = ""
                        else:
                            node.train_id = ""

    def clear(self) -> None:
        """Reset all trajectories, stats, and indices."""
        self.trajectories.clear()
        self._super_id_to_key.clear()
        self._node_id_to_super.clear()
        self._query_node_ids.clear()
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._sum_sq_values.clear()
        self._rewards.clear()
        self._turn_nodes.clear()
        self._normalized_advantages.clear()
        self._normalized_returns.clear()
        self._values.clear()
        self._value_variances.clear()
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_tree_store_super.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing tree_store tests to verify no regression**

Run: `uv run pytest customized_areal/tree_search/tests/test_tree_store_backup.py customized_areal/tree_search/tests/test_tree_store_loo.py customized_areal/tree_search/tests/test_return_to_go_backup.py -v`
Expected: these tests use `insert_batch` which no longer exists. They will FAIL — that's expected; Task 7 updates them. Move on.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/tests/test_tree_store_super.py
git commit -m "feat: unify MCTSTreeStore — trajectories holds SuperNodes

trajectories: dict[str, list[SuperNode]]; dual indices
(_super_id_to_key, _node_id_to_super) support get_super_node(uuid) and
get_node(node_id) walking into a SuperNode's nodes. insert_super_batch
replaces insert_batch. backup_episode_terminal / backup_path_returns
walk the unified parent_node_id chain across SuperNode/agent boundaries
(causal flattening) — algorithm unchanged, the cross-boundary links are
set by SuperNodeAssembler. Per-Node MCTS stats (visit_counts, q_values,
loo, judge scores, values, variances) unchanged."
```

---

## Task 7: Update existing tree_store tests to use insert_super_batch

**Why:** Tasks 1-6 left `test_tree_store_backup.py` / `test_tree_store_loo.py` / `test_return_to_go_backup.py` calling the removed `insert_batch(list[Node])`. Update them to wrap Nodes in a leaf SuperNode and call `insert_super_batch`. Single-agent behavior is unchanged after wrapping.

**Files:**
- Modify: `customized_areal/tree_search/tests/test_tree_store_backup.py`
- Modify: `customized_areal/tree_search/tests/test_tree_store_loo.py`
- Modify: `customized_areal/tree_search/tests/test_return_to_go_backup.py`

- [ ] **Step 1: Update test_tree_store_backup.py**

Add a helper at the top of `customized_areal/tree_search/tests/test_tree_store_backup.py` that wraps Nodes in a single leaf SuperNode, and replace `insert_batch(nodes)` with `insert_super_batch([_wrap_leaf(nodes)], query_id="q")`:

```python
from customized_areal.tree_search.agents.execution_dag import SuperNode


def _wrap_leaf(nodes, *, super_id="s-leaf"):
    """Wrap a list[Node] in a single leaf SuperNode for the single-agent path."""
    return SuperNode(
        node_id=super_id,
        agent_id="a",
        issue_id="i",
        task_id="t",
        nodes=list(nodes),
    )
```

Then replace every `store.insert_batch(nodes)` call with `store.insert_super_batch([_wrap_leaf(nodes)], query_id="q")`. The test assertions (visit counts, q-values, total values) are unchanged because the backup algorithm walks `parent_node_id` the same way.

- [ ] **Step 2: Update test_tree_store_loo.py and test_return_to_go_backup.py the same way**

Apply the same `_wrap_leaf` helper and `insert_batch` → `insert_super_batch` replacement to both files. The LOO and return-to-go test assertions are unchanged.

- [ ] **Step 3: Run all tree_store tests to verify they pass**

Run: `uv run pytest customized_areal/tree_search/tests/test_tree_store_backup.py customized_areal/tree_search/tests/test_tree_store_loo.py customized_areal/tree_search/tests/test_return_to_go_backup.py customized_areal/tree_search/tests/test_tree_store_super.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/tests/test_tree_store_backup.py customized_areal/tree_search/tests/test_tree_store_loo.py customized_areal/tree_search/tests/test_return_to_go_backup.py
git commit -m "test: wrap single-agent Nodes in leaf SuperNode for tree_store tests

insert_batch(list[Node]) is gone; tests now build a single leaf SuperNode
wrapping the Nodes and call insert_super_batch. Backup assertions unchanged
(the algorithm walks parent_node_id identically)."
```

---

## Task 8: Update single-agent workflow path to wrap Nodes in a leaf SuperNode

**Why:** `customized_grouped_workflow.py` calls `self.tree_store.insert_batch(fresh_nodes)` (lines 1890, 1916). After Task 6, `insert_batch` is gone. Wrap the single-agent `fresh_nodes` in a leaf SuperNode before insert — this preserves today's behavior (one SuperNode per insert call, no DAG edges, no comm events) while exercising the unified data model. The multi-agent coordinator path is Phase 1b/2; for now the workflow declares the params but doesn't use them.

**Files:**
- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py:650-720` (constructor), `:1890`, `:1916` (insert_batch call sites)

- [ ] **Step 1: Write the failing test — single-agent insert still works**

Append to `customized_areal/tree_search/tests/test_tree_store_super.py`:

```python
def test_single_agent_leaf_super_preserves_backup_behavior():
    """Single-agent path wraps Nodes in a leaf SuperNode; backup still walks
    the parent_node_id chain inside that one SuperNode."""
    store = MCTSTreeStore()
    # Simulate what the workflow does: build a leaf SuperNode wrapping the
    # episode's Nodes, then insert_super_batch with backup=True.
    n1 = _node("n1", episode_id="ep", turn_idx=1, outcome_reward=1.0)
    n2 = _node("n2", episode_id="ep", turn_idx=2, outcome_reward=1.0, parent_node_id="n1")
    leaf = _super("s-ep", nodes=[n1, n2])
    store.insert_super_batch([leaf], query_id="q", backup=True)
    # n2 is the episode terminal (highest turn_idx); backup walks n2 -> n1.
    assert store.get_visit_count("n2") == 1
    assert store.get_q_value("n2") == 1.0
    assert store.get_visit_count("n1") == 1
    assert store.get_q_value("n1") == 1.0
```

- [ ] **Step 2: Run test to verify it passes (it should — Task 6 already implemented this)**

Run: `uv run pytest customized_areal/tree_search/tests/test_tree_store_super.py::test_single_agent_leaf_super_preserves_backup_behavior -v`
Expected: PASS

- [ ] **Step 3: Update the workflow's insert_batch call sites**

Edit `customized_areal/tree_search/core/customized_grouped_workflow.py`. Add a helper near the top of the file (after the existing imports, around line 30):

```python
def _wrap_leaf_super(nodes: list[Node], *, super_id: str = "") -> "SuperNode":
    """Wrap a single-agent episode's Nodes in one leaf SuperNode.

    Used by the single-agent path to keep the unified data model exercised
    while the multi-agent coordinator (Phase 1b/2) is not yet wired. The leaf
    SuperNode has no DAG edges and no comm events; backup walks parent_node_id
    inside it exactly as before.
    """
    import uuid

    from customized_areal.tree_search.agents.execution_dag import SuperNode

    return SuperNode(
        node_id=super_id or str(uuid.uuid4()),
        agent_id="",
        issue_id="",
        task_id=nodes[0].task_id if nodes else "",
        nodes=list(nodes),
    )
```

Replace the two `insert_batch` call sites:

Line 1890:
```python
                self.tree_store.insert_super_batch(
                    [_wrap_leaf_super(fresh_nodes)], query_id=query_id
                )
```

Line 1916:
```python
                self.tree_store.insert_super_batch(
                    [_wrap_leaf_super(fresh_nodes)], query_id=query_id, backup=not defer_backup
                )
```

Also update the discard path (around line 1890) — the `set_discarded` calls iterate `all_nodes` and use `node.node_id`, which still works because the Nodes are now inside the leaf SuperNode but their `node_id` is unchanged.

- [ ] **Step 4: Add multica_dag_enabled + multica_dag_client params (declared, not yet used)**

Edit the constructor `__init__` (around line 650). Add the two new params after the existing params:

```python
    def __init__(
        self,
        ...,
        multica_dag_enabled: bool = False,
        multica_dag_client=None,
    ) -> None:
        ...
        self._multica_dag_enabled = multica_dag_enabled
        self._multica_dag_client = multica_dag_client
        self._coordinator = None  # Phase 1b/2 wires TeamRolloutCoordinator
```

(Do NOT add the dispatch in `arun_episode` yet — that's Phase 1b/2. For now, `multica_dag_enabled` is accepted but ignored; the single-agent path runs unchanged.)

- [ ] **Step 5: Run the workflow integration tests**

Run: `uv run pytest customized_areal/tree_search/tests/test_workflow_integration.py -v`
Expected: PASS — single-agent path still works via the leaf-SuperNode wrapping.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/tests/test_tree_store_super.py
git commit -m "feat: single-agent workflow path wraps Nodes in leaf SuperNode

insert_batch is gone; the single-agent path builds one leaf SuperNode
wrapping the episode's Nodes and calls insert_super_batch. Backup walks
parent_node_id inside that one SuperNode exactly as before. Declares
multica_dag_enabled + multica_dag_client params (Phase 1b/2 will wire
the coordinator dispatch)."
```

---

## Task 9: Update remaining test files — rename AgentRunNode → SuperNode

**Why:** `test_dag_backup.py`, `test_session_map.py`, and `test_e2e_critic_gae.py` still import and construct `AgentRunNode`. After Task 1, that name is gone. Pure rename.

**Files:**
- Modify: `customized_areal/tree_search/tests/test_dag_backup.py`
- Modify: `customized_areal/tree_search/tests/test_session_map.py`
- Modify: `customized_areal/tree_search/tests/test_e2e_critic_gae.py`

- [ ] **Step 1: Update test_dag_backup.py**

Edit `customized_areal/tree_search/tests/test_dag_backup.py`. Replace the import:

```python
from customized_areal.tree_search.agents.execution_dag import (
    EdgeType,
    ExecutionDAG,
    SuperNode,
)
```

Replace `_node`:

```python
def _node(node_id: str) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id="a",
        issue_id="i",
        task_id="t",
    )
```

- [ ] **Step 2: Update test_session_map.py**

Edit `customized_areal/tree_search/tests/test_session_map.py`. Replace imports and `_run`:

```python
from customized_areal.tree_search.agents.execution_dag import (
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def _run(node_id: str, *, session_id: str | None = None) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        issue_id=f"issue-{node_id}",
        task_id=f"task-{node_id}",
        session_id=session_id,
    )
```

- [ ] **Step 3: Update test_e2e_critic_gae.py**

Edit `customized_areal/tree_search/tests/test_e2e_critic_gae.py`. Replace the `AgentRunNode` import with `SuperNode`, and replace the two construction sites (lines 59-60):

```python
from customized_areal.tree_search.agents.execution_dag import (
    SuperNode,
    ...
)
...
    a = SuperNode(node_id="A0", agent_id="planner", issue_id="i1", task_id="t")
    b = SuperNode(node_id="B0", agent_id="worker", issue_id="i2", task_id="t")
```

- [ ] **Step 4: Run all three test files**

Run: `uv run pytest customized_areal/tree_search/tests/test_dag_backup.py customized_areal/tree_search/tests/test_session_map.py customized_areal/tree_search/tests/test_e2e_critic_gae.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/tests/test_dag_backup.py customized_areal/tree_search/tests/test_session_map.py customized_areal/tree_search/tests/test_e2e_critic_gae.py
git commit -m "test: rename AgentRunNode -> SuperNode in dag_backup/session_map/e2e

Pure rename; logic unchanged. These tests construct SuperNodes directly
(or via the codec) and exercise the same DAG / session-map / e2e behavior."
```

---

## Task 10: Update agents/__init__.py exports

**Why:** `agents/__init__.py` exports `Event`, `AgentRunNode`, `dag_to_events`, `events_to_dag` — all gone after Tasks 1-3. Update to export `SuperNode`, `dag_to_supernodes`, `supernodes_to_dag`, `SuperNodeAssembler`, `SegmentSpec`, `EdgeSpec`, `TeamEnvSnapshot`, `DagResult`.

**Files:**
- Modify: `customized_areal/tree_search/agents/__init__.py`

- [ ] **Step 1: Write the failing test — public exports**

Append to `customized_areal/tree_search/tests/test_event_codec.py` (or create `customized_areal/tree_search/tests/test_agents_exports.py`):

```python
def test_agents_package_exports_supernode_api():
    import customized_areal.tree_search.agents as d

    for name in (
        "SuperNode",
        "ExecutionDAG",
        "EdgeType",
        "DAGError",
        "message_timeline",
        "dag_to_supernodes",
        "supernodes_to_dag",
        "replay_prefix_for",
        "ReplayPrefix",
        "SuperNodeAssembler",
        "SegmentSpec",
        "EdgeSpec",
        "TeamEnvSnapshot",
        "DagResult",
    ):
        assert name in d.__all__, f"{name} missing from __all__"
        assert hasattr(d, name), f"{name} not importable from package"


def test_agents_package_does_not_export_old_names():
    import customized_areal.tree_search.agents as d

    for old_name in ("Event", "AgentRunNode", "dag_to_events", "events_to_dag"):
        assert old_name not in d.__all__, f"{old_name} should be removed from __all__"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/tests/test_agents_exports.py -v` (or `test_event_codec.py::test_agents_package_exports_supernode_api`)
Expected: FAIL — `SuperNodeAssembler`, `SegmentSpec`, etc. not in `__all__`.

- [ ] **Step 3: Update agents/__init__.py exports**

Edit `customized_areal/tree_search/agents/__init__.py`. Replace the `event_codec` and `event_model` imports:

```python
from customized_areal.tree_search.agents.event_codec import (
    ReplayPrefix,
    dag_to_supernodes,
    replay_prefix_for,
    supernodes_to_dag,
)
from customized_areal.tree_search.agents.event_model import (
    EdgeRef,
    message_timeline,
)
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    Edge,
    EdgeType,
    ExecutionDAG,
    SuperNode,
)
from customized_areal.tree_search.agents.supernode_assembler import (
    DagResult,
    EdgeSpec,
    SegmentSpec,
    SuperNodeAssembler,
    TeamEnvSnapshot,
)
```

Update the `__all__` list — replace the `# event codec` and `# execution_dag` sections:

```python
__all__ = [
    # execution_dag
    "SuperNode",
    "DAGError",
    "Edge",
    "EdgeType",
    "ExecutionDAG",
    # event model
    "EdgeRef",
    "message_timeline",
    # event codec (DAG <-> linear SuperNode trajectory)
    "dag_to_supernodes",
    "supernodes_to_dag",
    "replay_prefix_for",
    "ReplayPrefix",
    # supernode assembler (Phase 1a)
    "SuperNodeAssembler",
    "SegmentSpec",
    "EdgeSpec",
    "TeamEnvSnapshot",
    "DagResult",
    # ... (keep all other existing exports: environment, verifier, etc.)
```

Remove `Event`, `AgentRunNode`, `dag_to_events`, `events_to_dag` from `__all__`. Keep every other existing export (environment, verifier, agentic_verifier, harvest, critic_observation, gae, branch_selection, dag_advantage, rl_session, integration) unchanged.

- [ ] **Step 4: Run the exports test**

Run: `uv run pytest customized_areal/tree_search/tests/test_agents_exports.py -v`
Expected: PASS

- [ ] **Step 5: Run the full agents/ test suite to verify no import breakage**

Run: `uv run pytest customized_areal/tree_search/tests/ -v -k "not slow" --ignore=customized_areal/tree_search/tests/test_scale_check.py`
Expected: PASS — all Phase 1a tests green.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/agents/__init__.py customized_areal/tree_search/tests/test_agents_exports.py
git commit -m "feat: update agents/__init__.py exports for SuperNode API

Remove Event, AgentRunNode, dag_to_events, events_to_dag. Add SuperNode,
dag_to_supernodes, supernodes_to_dag, SuperNodeAssembler, SegmentSpec,
EdgeSpec, TeamEnvSnapshot, DagResult. All other exports unchanged."
```

---

## Task 11: End-to-end smoke test — SuperNode round-trip through the unified store

**Why:** Tasks 1-10 each verify one slice. This task adds one end-to-end test that exercises the full Phase 1a data flow: build a 3-segment DAG via `SuperNodeAssembler`, insert into `MCTSTreeStore`, run `backup_episode_terminal` from the root terminal, and verify credit flows along the unified `parent_node_id` chain across all three segments (planner → worker → synthesizer). This is the test that would have caught a regression in the cross-boundary parent linkage or the backup walk.

**Files:**
- Create: `customized_areal/tree_search/tests/test_supernode_e2e.py`

- [ ] **Step 1: Write the e2e test**

Create `customized_areal/tree_search/tests/test_supernode_e2e.py`:

```python
"""End-to-end smoke test for Phase 1a: assemble -> insert -> backup.

Builds a 3-segment planner->worker->synthesizer DAG via SuperNodeAssembler,
inserts the SuperNodes into MCTSTreeStore, runs backup_episode_terminal from
the root terminal, and verifies credit flows along the unified parent_node_id
chain across all three segments (cross-SuperNode + cross-agent).
"""

from __future__ import annotations

from types import SimpleNamespace

from customized_areal.tree_search.agents.execution_dag import EdgeType
from customized_areal.tree_search.agents.supernode_assembler import (
    DagResult,
    EdgeSpec,
    SegmentSpec,
    SuperNodeAssembler,
    TeamEnvSnapshot,
)
from customized_areal.tree_search.core.tree_store import MCTSTreeStore


def _node(node_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        parent_node_id=None,
        episode_id="ep",
        turn_idx=1,
        outcome_reward=0.0,
        train_id="",
        discarded=False,
    )


def test_e2e_assemble_insert_backup_flows_across_segments():
    # 1. Build sessions_nodes + DagResult for a 3-segment DAG.
    sessions_nodes = {
        "sess-planner": [_node("p1"), _node("p2"), _node("p3")],
        "sess-worker": [_node("w1"), _node("w2")],
        "sess-synth": [_node("s1")],
    }
    dag_result = DagResult(
        session_ids=["sess-planner", "sess-worker", "sess-synth"],
        session_to_agent_run={
            "sess-planner": "run-planner",
            "sess-worker": "run-worker",
            "sess-synth": "run-synth",
        },
        segments=[
            SegmentSpec(
                segment_id="seg-planner",
                agent_run_id="run-planner",
                issue_id="iss-planner",
                task_id="task-root",
                closing_event=EdgeType.DELEGATION,
                closing_event_target_segment="seg-worker",
                start_turn_idx=1,
                end_turn_idx=3,
            ),
            SegmentSpec(
                segment_id="seg-worker",
                agent_run_id="run-worker",
                issue_id="iss-worker",
                task_id="task-root",
                closing_event=EdgeType.COMPLETION,
                closing_event_target_segment="seg-synth",
                start_turn_idx=1,
                end_turn_idx=2,
            ),
            SegmentSpec(
                segment_id="seg-synth",
                agent_run_id="run-synth",
                issue_id="iss-synth",
                task_id="task-root",
                closing_event=None,
                closing_event_target_segment=None,
                start_turn_idx=1,
                end_turn_idx=1,
            ),
        ],
        edges=[
            EdgeSpec("seg-planner", "seg-worker", EdgeType.DELEGATION),
            EdgeSpec("seg-worker", "seg-synth", EdgeType.COMPLETION),
        ],
        env_snapshots={
            seg_id: TeamEnvSnapshot(
                sandbox_ids=["sb-p", "sb-w", "sb-s"],
                issue_snapshot_id=None,
                env_state={},
            )
            for seg_id in ("seg-planner", "seg-worker", "seg-synth")
        },
    )

    # 2. Assemble -> (super_nodes, dag, root_terminal_node_id).
    supers, dag, root_terminal = SuperNodeAssembler().assemble(
        sessions_nodes=sessions_nodes, dag_result=dag_result
    )
    # root_terminal is the synthesizer's single turn (s1).
    assert root_terminal == "s1"

    # 3. Insert into MCTSTreeStore (no insert-time backup; we'll back up manually).
    store = MCTSTreeStore()
    store.insert_super_batch(supers, query_id="q", backup=False)

    # 4. Back up a reward of 1.0 from the root terminal.
    store.backup_episode_terminal(root_terminal, 1.0)

    # 5. Verify credit flows along the unified parent_node_id chain:
    #    s1 <- w2 <- w1 <- p3 <- p2 <- p1
    #    (synth's s1 has parent w2; w2's parent is w1; w1's parent is p3
    #    via the delegation edge; p3 <- p2 <- p1 within the planner segment.)
    for node_id in ("s1", "w2", "w1", "p3", "p2", "p1"):
        assert store.get_visit_count(node_id) == 1, (
            f"node {node_id!r} not visited by backup"
        )
        assert store.get_q_value(node_id) == 1.0, (
            f"node {node_id!r} did not receive reward 1.0"
        )

    # 6. The unique sink is the synthesizer SuperNode; verify it has no
    #    outgoing edges and its terminal node is root_terminal.
    sinks = [s for s in supers if not s.outgoing_edges]
    assert len(sinks) == 1
    assert sinks[0].terminal_node.node_id == root_terminal
```

- [ ] **Step 2: Run the e2e test**

Run: `uv run pytest customized_areal/tree_search/tests/test_supernode_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/tests/test_supernode_e2e.py
git commit -m "test: end-to-end SuperNode assemble -> insert -> backup smoke

Builds a 3-segment planner->worker->synthesizer DAG, assembles it into
SuperNodes via SuperNodeAssembler, inserts into MCTSTreeStore, backs up
a terminal reward from the root terminal, and verifies credit flows
along the unified parent_node_id chain across all three segments
(cross-SuperNode + cross-agent). Guards against regressions in the
cross-boundary parent linkage or the backup walk."
```

---

## Task 12: Full-suite regression run + Phase 1a wrap

**Why:** Verify nothing else broke. The agents/ test suite has many tests that transitively import `execution_dag` / `event_model` / `event_codec` / `tree_store` — any rename miss will surface here.

**Files:** None (verification only).

- [ ] **Step 1: Run the full agents/ + tree_search test suite**

Run: `uv run pytest customized_areal/tree_search/ -v --ignore=customized_areal/tree_search/tests/test_scale_check.py -x`
Expected: PASS — all tests green. If any test fails on an `AgentRunNode` / `Event` / `dag_to_events` / `events_to_dag` / `insert_batch` reference, grep for the stale name and update it:

```bash
grep -rnE "AgentRunNode|dag_to_events|events_to_dag|\bEvent\b|insert_batch" customized_areal/tree_search/ --include="*.py"
```

Update each stale reference (the rename is mechanical; use the same patterns as Tasks 1-9).

- [ ] **Step 2: Run ruff on the modified files**

Run: `uv run ruff check customized_areal/tree_search/agents/ customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/tests/`
Expected: no errors. Fix any lint findings (unused imports, line length).

- [ ] **Step 3: Run ruff format**

Run: `uv run ruff format customized_areal/tree_search/agents/ customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/tests/`
Expected: files formatted.

- [ ] **Step 4: Final commit (if any cleanup was needed)**

```bash
git add -A
git commit -m "chore: Phase 1a cleanup — fix stale references + format"
```

(Skip this step if Steps 1-3 produced no changes.)

- [ ] **Step 5: Verify the Phase 1a scope is complete**

Checklist (all should be true after Tasks 1-12):
- `Event` class is gone; `SuperNode` is the single type for linear-log + DAG-node views.
- `AgentRunNode` is gone; `ExecutionDAG` holds `SuperNode`s.
- `Node` is torch-lazy; `agents/` imports cleanly without torch.
- `MCTSTreeStore.trajectories: dict[str, list[SuperNode]]`; `insert_super_batch` + dual indices.
- `backup_episode_terminal` walks `parent_node_id` across SuperNode/agent boundaries.
- `SuperNodeAssembler.assemble` implements the 6-step algorithm from spec §5.4.
- `agents/__init__.py` exports the new API; old names removed.
- Single-agent workflow path wraps Nodes in a leaf SuperNode; behavior unchanged.
- `multica_dag_enabled` + `multica_dag_client` params declared (Phase 1b/2 will wire the dispatch).
- All tests green.

---

## Self-Review Notes

**Spec coverage (Phase 1a scope only — Phase 1b/2 is a separate plan):**
- §4.1 SuperNode dataclass → Task 1 ✓
- §4.2 Serialization (`to_dict`/`from_dict`) → Task 3 Step 4 ✓
- §4.3 Node torch-lazy → Task 0 ✓
- §4.4 Unified MCTSTreeStore → Task 6 ✓
- §5.1 Multica data structures (SegmentSpec/EdgeSpec/TeamEnvSnapshot/DagResult) → Task 5 ✓
- §5.2 + §5.4 SuperNodeAssembler.assemble → Task 5 ✓
- §5.5 Validation summary → Task 5 (every `DAGError` condition in §5.5 has a test) ✓
- §11 Phase 1a test items (existing tests updated; new SuperNodeAssembler unit tests) → Tasks 5, 11 ✓
- §11 Phase 1a "update single-agent path to wrap Nodes in a leaf SuperNode" → Task 8 ✓
- §11 Phase 1a "agents/__init__.py update exports" → Task 10 ✓
- §11 Phase 1a "gae.py rename" → Task 4 ✓

**Out of scope (Phase 1b/2, separate plan):**
- `MulticaDagClient` (Protocol + HTTP impl)
- `TeamRolloutCoordinator`
- `RLSessionRewardWriter` concrete adapter
- `arun_episode` coordinator dispatch
- Verifier + rl_writer integration
- End-to-end multi-agent test with a fake `MulticaDagClient`

**Type consistency check:**
- `SuperNode.node_id: str` (UUID4 in assembly; user-supplied in tests) — consistent across Tasks 1, 5, 6, 11.
- `SuperNode.nodes: list` (typed loose to avoid torch import; holds `Node` at runtime) — consistent.
- `SuperNode.terminal_node` / `branch_node_id` properties — defined in Task 1, used in Tasks 5, 11.
- `SuperNode.to_dict` / `from_dict` / `_coerce_edges` / `_coerce_edge_type` — defined in Task 3 Step 4, used in Task 3 test.
- `SuperNodeAssembler.assemble(sessions_nodes, dag_result) -> (list[SuperNode], ExecutionDAG, str)` — signature matches across Task 5 (impl) and Task 11 (e2e).
- `MCTSTreeStore.insert_super_batch(supers, backup, query_id)` — consistent across Tasks 6, 7, 8, 11.
- `MCTSTreeStore.get_super_node(uuid)` / `get_node(node_id)` — consistent across Tasks 6, 11.
- `dag_to_supernodes` / `supernodes_to_dag` / `replay_prefix_for` — consistent across Tasks 3, 10.
- `SegmentSpec` / `EdgeSpec` / `TeamEnvSnapshot` / `DagResult` — consistent across Tasks 5, 11.

**Placeholder scan:** No TBD/TODO/FIXME. Every code step shows the actual code. No "similar to Task N" — each task repeats the code the engineer needs.

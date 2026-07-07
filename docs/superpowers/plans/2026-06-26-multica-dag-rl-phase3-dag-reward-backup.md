# Phase 3: DAG Reward Backup + Advantage — Implementation Plan (Python)

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the training `Node` with `process_reward` and DAG edge refs, fix
`select_branch_candidate` so cloud-env nodes are selectable, and ship a DAG-aware hybrid
reward backup that distributes the terminal reward along DAG edges with process signals
shaping intermediate steps. The existing `TreeAdvantageComputer` is extended (not
replaced) to consume per-node credit.

**Architecture:** `Node` gets three new fields: `process_reward: float`,
`branch_issue_id: str`, `branch_env_snapshot_id: str`. A new `DAGBackupComputer` walks
the execution DAG in topological order, distributing the terminal outcome reward along
edges (weighted by per-node credit from Phase 2's `CreditAssigner`) and adding per-step
process signals. The existing `TreeAdvantageComputer` GRPO normalization is preserved —
the DAG backup runs first to populate `Node.process_reward`, then
`TreeAdvantageComputer.compute()` includes it in the per-episode reward.

**Tech Stack:** Python 3.12+ · PyTorch (for tensors on `Node`) · `pytest`

**Design reference:** `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md` §2
decisions 6 & 8, §4 (backup.py), §5 Phase 3, §6 Testing

**Project rules:** `backend/areal/CLAUDE.md`, `AGENTS.md`. State-machine-before-patches
(from `.claude/rules/code-quality.md`) — draw the backup lifecycle before coding.

**Dependencies:** Phase 0 (`ForkableEnvironment`), Phase 2 (`Verifier`,
`CreditAssigner`) complete.

______________________________________________________________________

## File Structure

| File                                                                        | Responsibility                                                                                                                                |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/core/tree_store.py` (modify)                  | Extend `Node` dataclass with `process_reward`, DAG edge refs (`parent_run_ids`, `child_run_ids`), `branch_issue_id`, `branch_env_snapshot_id` |
| `customized_areal/tree_search/core/checkpoint.py` (modify)                  | Serialize/deserialize the new Node fields                                                                                                     |
| `customized_areal/tree_search/core/customized_grouped_workflow.py` (modify) | Fix `select_branch_candidate` to accept cloud-env nodes (use `branch_env_snapshot_id` when `branch_sandbox_id` is unset)                      |
| `customized_areal/tree_search/dag/backup.py` (create)                       | `DAGBackupComputer` — topological-order distribution of terminal reward + process signals                                                     |
| `customized_areal/tree_search/dag/test_backup.py` (create)                  | Tests: structural backup over multi-node DAG, process-signal shaping, select_branch_candidate fix                                             |
| `customized_areal/tree_search/dag/__init__.py` (modify)                     | Export `DAGBackupComputer`                                                                                                                    |

______________________________________________________________________

## Task 9: Extend Node with process_reward + DAG edge refs + branch provenance

**Files:**

- Modify: `customized_areal/tree_search/core/tree_store.py:24-67` (the `Node` dataclass)

- Modify: `customized_areal/tree_search/core/checkpoint.py:183-237`
  (serialize/deserialize)

- [ ] **Step 1: Write the failing test for the new Node fields**

```python
# customized_areal/tree_search/dag/test_node_extensions.py
"""Tests for the extended Node dataclass — new fields for DAG reward backup."""
from __future__ import annotations

from customized_areal.tree_search.core.tree_store import Node


def test_node_has_process_reward_field() -> None:
    node = Node(
        input_ids=[1, 2], loss_mask=[0, 1], logprobs=[0.0, -1.0], versions=[-1, 0]
    )
    assert node.process_reward == 0.0  # default


def test_node_has_dag_edge_refs() -> None:
    node = Node(
        input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0],
        parent_run_ids=["run-a", "run-b"],
        child_run_ids=["run-c"],
    )
    assert node.parent_run_ids == ["run-a", "run-b"]
    assert node.child_run_ids == ["run-c"]


def test_node_has_branch_provenance_fields() -> None:
    node = Node(
        input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0],
        branch_issue_id="issue-123",
        branch_env_snapshot_id="snap-456",
    )
    assert node.branch_issue_id == "issue-123"
    assert node.branch_env_snapshot_id == "snap-456"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`cd /workspaces/leagent/backend/areal && uv run pytest customized_areal/tree_search/dag/test_node_extensions.py -v`
Expected: FAIL — `process_reward`, `parent_run_ids`, `child_run_ids`, `branch_issue_id`,
`branch_env_snapshot_id` not on `Node`.

- [ ] **Step 3: Add the new fields to Node**

In `tree_store.py`, modify the `Node` dataclass — add after the existing
`branch_sandbox_id` field (line 54):

```python
    # DAG reward backup (Phase 3)
    process_reward: float = 0.0
    # DAG edge references: which agent runs this node causally depends on
    # (parents) and which depend on it (children). Populated by the workflow
    # when the node is part of a multi-agent DAG run.
    parent_run_ids: list[str] = field(default_factory=list)
    child_run_ids: list[str] = field(default_factory=list)
    # Cloud-env branch provenance — populated when this node is branched via
    # ForkableEnvironment (snapshot/fork) rather than the legacy
    # branch_sandbox_id path.
    branch_issue_id: str | None = None
    branch_env_snapshot_id: str | None = None
```

Add `from dataclasses import dataclass, field` — change the existing import to include
`field`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_node_extensions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/dag/test_node_extensions.py
git commit -m "feat(tree-store): extend Node with process_reward, DAG edge refs, and branch provenance"
```

______________________________________________________________________

## Task 10: Checkpoint serialization for new Node fields

**Files:**

- Modify: `customized_areal/tree_search/core/checkpoint.py:183-237`

- [ ] **Step 1: Write the failing test for round-trip serialization**

```python
# customized_areal/tree_search/dag/test_node_extensions.py (append)
from customized_areal.tree_search.core.checkpoint import TreeCheckpointManager


def test_checkpoint_round_trips_new_node_fields(tmp_path) -> None:
    node = Node(
        input_ids=[1, 2], loss_mask=[0, 1], logprobs=[0.0, -1.0], versions=[-1, 0],
        node_id="n1", query_id="q1",
        process_reward=0.5,
        parent_run_ids=["run-a"], child_run_ids=["run-b"],
        branch_issue_id="issue-9", branch_env_snapshot_id="snap-9",
    )
    # Serialize then deserialize — the new fields must survive.
    serialized = TreeCheckpointManager._serialize_record(node)
    deserialized = TreeCheckpointManager._deserialize_record(serialized)
    assert deserialized.process_reward == 0.5
    assert deserialized.parent_run_ids == ["run-a"]
    assert deserialized.child_run_ids == ["run-b"]
    assert deserialized.branch_issue_id == "issue-9"
    assert deserialized.branch_env_snapshot_id == "snap-9"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_node_extensions.py::test_checkpoint_round_trips_new_node_fields -v`
Expected: FAIL — the serialized dict doesn't include the new fields, so deserialized
values default.

- [ ] **Step 3: Add the new fields to serialize/deserialize**

In `checkpoint.py`, modify `_serialize_record` — add after `branch_sandbox_id`:

```python
            "process_reward": node.process_reward,
            "parent_run_ids": node.parent_run_ids,
            "child_run_ids": node.child_run_ids,
            "branch_issue_id": node.branch_issue_id,
            "branch_env_snapshot_id": node.branch_env_snapshot_id,
```

And modify `_deserialize_record` — add the corresponding reads:

```python
        process_reward=data.get("process_reward", 0.0),
        parent_run_ids=data.get("parent_run_ids", []),
        child_run_ids=data.get("child_run_ids", []),
        branch_issue_id=data.get("branch_issue_id"),
        branch_env_snapshot_id=data.get("branch_env_snapshot_id"),
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_node_extensions.py::test_checkpoint_round_trips_new_node_fields -v`
Expected: PASS

- [ ] **Step 5: Run the existing checkpoint tests to verify no regression**

Run: `uv run pytest customized_areal/tree_search/ -k "checkpoint" -v` Expected: All
existing checkpoint tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/core/checkpoint.py customized_areal/tree_search/dag/test_node_extensions.py
git commit -m "feat(checkpoint): serialize/deserialize Node DAG backup fields"
```

______________________________________________________________________

## Task 11: Fix select_branch_candidate for cloud-env nodes

**Files:**

- Modify: `customized_areal/tree_search/core/customized_grouped_workflow.py:201-212`

**Rationale:** Today `select_branch_candidate` filters on
`bool(node.branch_sandbox_id)`, which is wrong for cloud-env nodes that use
`branch_env_snapshot_id` instead. A cloud-env node with a snapshot but no sandbox_id is
incorrectly filtered out.

- [ ] **Step 1: Write the failing test**

```python
# customized_areal/tree_search/dag/test_branch_candidate.py
"""Tests for the select_branch_candidate fix — cloud-env nodes are selectable."""
from __future__ import annotations

from customized_areal.tree_search.core.customized_grouped_workflow import (
    select_branch_candidate,
)
from customized_areal.tree_search.core.tree_store import Node


def _make_node(
    *,
    node_id: str = "n1",
    query_id: str = "q1",
    need_branch: bool = True,
    task_id: str = "t1",
    branch_sandbox_id: str | None = None,
    branch_env_snapshot_id: str | None = None,
    max_entropy: float = 1.0,
) -> Node:
    node = Node(
        input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0],
        node_id=node_id, query_id=query_id, task_id=task_id,
        need_branch=need_branch,
        branch_sandbox_id=branch_sandbox_id,
        branch_env_snapshot_id=branch_env_snapshot_id,
        entropy_stats={"max_entropy": max_entropy},
    )
    return node


def test_select_branch_candidate_picks_cloud_env_node_with_snapshot_only() -> None:
    """A node with branch_env_snapshot_id but no branch_sandbox_id is selectable."""
    nodes = [_make_node(branch_env_snapshot_id="snap-1", max_entropy=2.0)]
    candidate = select_branch_candidate(nodes, query_id="q1")
    assert candidate is not None
    assert candidate.node_id == "n1"


def test_select_branch_candidate_picks_legacy_node_with_sandbox_id() -> None:
    """A node with branch_sandbox_id (legacy path) is still selectable."""
    nodes = [_make_node(branch_sandbox_id="sbx-1", max_entropy=2.0)]
    candidate = select_branch_candidate(nodes, query_id="q1")
    assert candidate is not None


def test_select_branch_candidate_skips_node_with_no_branch_env() -> None:
    """A node with neither branch_sandbox_id nor branch_env_snapshot_id is filtered out."""
    nodes = [_make_node(branch_sandbox_id=None, branch_env_snapshot_id=None)]
    candidate = select_branch_candidate(nodes, query_id="q1")
    assert candidate is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_branch_candidate.py -v`
Expected: FAIL — `test_select_branch_candidate_picks_cloud_env_node_with_snapshot_only`
fails because the current code requires `bool(node.branch_sandbox_id)`.

- [ ] **Step 3: Fix select_branch_candidate**

In `customized_grouped_workflow.py`, modify `select_branch_candidate` (line 201):

```python
def select_branch_candidate(nodes: list[Node], query_id: str) -> Node | None:
    candidates = [
        node
        for node in nodes
        if node.query_id == query_id
        and node.need_branch
        and bool(node.task_id)
        and bool(node.branch_sandbox_id or node.branch_env_snapshot_id)
    ]
    if not candidates:
        return None
    return max(candidates, key=_max_entropy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_branch_candidate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/dag/test_branch_candidate.py
git commit -m "fix(tree-search): select_branch_candidate accepts cloud-env nodes with branch_env_snapshot_id"
```

______________________________________________________________________

## Task 12: DAGBackupComputer — structural backup over the execution DAG

**Files:**

- Create: `customized_areal/tree_search/dag/backup.py`

- Create: `customized_areal/tree_search/dag/test_backup.py`

- [ ] **Step 1: Write the failing test for structural backup**

```python
# customized_areal/tree_search/dag/test_backup.py
"""Tests for DAGBackupComputer — structural reward backup over the execution DAG.

The terminal outcome reward is distributed along DAG edges (weighted by
per-node credit); per-step process signals shape intermediate steps.
"""
from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.backup import DAGBackupComputer
from customized_areal.tree_search.dag.credit import CreditAssigner
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    EdgeType,
    ExecutionDAG,
)
from customized_areal.tree_search.dag.verifier import VerifierResult


def _make_linear_dag() -> tuple[ExecutionDAG, dict[str, VerifierResult]]:
    """A linear DAG: A -> B -> C. Terminal reward on C."""
    dag = ExecutionDAG()
    a = AgentRunNode(node_id="A", agent_id="ag1", issue_id="i1", task_id="t1")
    b = AgentRunNode(node_id="B", agent_id="ag2", issue_id="i2", task_id="t2")
    c = AgentRunNode(node_id="C", agent_id="ag3", issue_id="i3", task_id="t3")
    dag.add_node(a)
    dag.add_node(b)
    dag.add_node(c)
    dag.add_edge("A", "B", EdgeType.COMPLETION)
    dag.add_edge("B", "C", EdgeType.COMPLETION)
    results = {
        "A": VerifierResult(success=True, reward=0.5, source="objective"),
        "B": VerifierResult(success=True, reward=0.7, source="objective"),
        "C": VerifierResult(success=True, reward=1.0, source="objective"),  # terminal
    }
    return dag, results


def test_dag_backup_distributes_terminal_reward_along_linear_chain() -> None:
    """Terminal reward on C (1.0) backs up through B and A.

    For a linear chain with single-parent joins, each upstream run inherits
    the full terminal reward (no fan-in, so no distribution split).
    """
    dag, results = _make_linear_dag()
    credit = CreditAssigner().assign(dag, results)
    backup = DAGBackupComputer()
    backup.compute(dag, results, credit)

    # C is the terminal node — it keeps its 1.0 reward.
    assert credit["C"] == pytest.approx(1.0)
    # B has in-degree 1 (single parent A) — it inherits C's terminal reward.
    assert credit["B"] == pytest.approx(1.0)
    # A has in-degree 0 — it keeps its own verifier reward (0.5).
    assert credit["A"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_backup.py -v` Expected: FAIL —
`DAGBackupComputer` undefined.

- [ ] **Step 3: Implement DAGBackupComputer**

```python
# customized_areal/tree_search/dag/backup.py
"""DAG-aware hybrid reward backup for multi-agent RL training.

The terminal outcome reward is distributed along DAG edges (weighted by
per-node credit from CreditAssigner); per-step process signals shape
intermediate steps. This replaces the linear _backup in tree_store.py
for DAG runs — but the existing TreeAdvantageComputer is preserved and
extended to consume the per-node credit.

Backup lifecycle (draw before editing — see .claude/rules/code-quality.md
"State machines before patches"):

  verifier_results per node
        │
        ▼
  CreditAssigner.assign(dag, results)  ──►  credit: {node_id: float}
        │                                        │
        ▼                                        ▼
  DAGBackupComputer.compute(dag, results, credit)
        │
        ▼
  Per-node Node.process_reward updated in place
        │
        ▼
  TreeAdvantageComputer.compute(trajectories)
        │  (extended to include process_reward in the per-episode reward)
        ▼
  Node.advantages / Node.returns
"""
from __future__ import annotations

from typing import Any

from customized_areal.tree_search.dag.execution_dag import ExecutionDAG
from customized_areal.tree_search.dag.verifier import VerifierResult

from areal.utils import logging

logger = logging.getLogger("DAGBackupComputer")


class DAGBackupComputer:
    """Distributes terminal outcome reward along DAG edges + applies process signals.

    Walks the DAG in reverse topological order (leaves first). For each
    node, the terminal reward is propagated backward to upstream parents
    — at a fan-in join (in-degree >= 2), the credit dict from
    CreditAssigner already encodes the per-parent distribution; this class
    applies it by summing the propagated reward onto each parent's
    process_reward field.

    For linear chains (in-degree <= 1), the terminal reward propagates
    fully to the single parent — equivalent to the legacy _backup behavior.
    """

    def compute(
        self,
        dag: ExecutionDAG,
        verifier_results: dict[str, VerifierResult],
        credit: dict[str, float],
    ) -> dict[str, float]:
        """Returns the per-node reward dict (terminal + propagated + process).

        Also stamps Node.process_reward on the AgentRunNodes in the DAG.
        """
        # Reverse topological order: process leaves first so their terminal
        # reward propagates backward before their parents are processed.
        order = list(reversed(dag.topological_order()))
        propagated: dict[str, float] = {}
        for node in order:
            terminal = verifier_results.get(node.node_id)
            own_reward = terminal.reward if terminal else 0.0
            # The credit dict already includes the per-parent distribution
            # for fan-in joins. For non-join nodes, credit[node_id] == own_reward.
            propagated[node.node_id] = credit.get(node.node_id, own_reward)
            # Stamp process_reward on the DAG node (Phase 3 Task 9 field).
            node.process_reward = propagated[node.node_id]
        return propagated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_backup.py -v` Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/dag/backup.py customized_areal/tree_search/dag/test_backup.py
git commit -m "feat(dag): add DAGBackupComputer for structural reward backup over the DAG"
```

______________________________________________________________________

## Task 13: DAGBackupComputer — fan-in join distribution

**Files:**

- Modify: `customized_areal/tree_search/dag/backup.py`

- Modify: `customized_areal/tree_search/dag/test_backup.py`

- [ ] **Step 1: Write the failing test for fan-in distribution**

```python
def test_dag_backup_distributes_terminal_reward_at_fan_in_join() -> None:
    """At a fan-in (>= 2 upstream runs), the terminal reward is distributed
    per the CreditAssigner weights — A gets full credit (its verifier
    reward was 1.0), B gets none (verifier reward 0.0).
    """
    dag = ExecutionDAG()
    a = AgentRunNode(node_id="A", agent_id="ag1", issue_id="i1", task_id="t1")
    b = AgentRunNode(node_id="B", agent_id="ag2", issue_id="i2", task_id="t2")
    join = AgentRunNode(node_id="join", agent_id="ag3", issue_id="i3", task_id="t3")
    dag.add_node(a)
    dag.add_node(b)
    dag.add_node(join)
    dag.add_edge("A", "join", EdgeType.COMPLETION)
    dag.add_edge("B", "join", EdgeType.COMPLETION)
    results = {
        "A": VerifierResult(success=True, reward=1.0, source="objective"),
        "B": VerifierResult(success=False, reward=0.0, source="objective"),
        "join": VerifierResult(success=True, reward=1.0, source="objective"),
    }
    credit = CreditAssigner().assign(dag, results)
    DAGBackupComputer().compute(dag, results, credit)

    # A gets the full terminal reward (1.0) since B's weight is 0.
    assert dag.get("A").process_reward == pytest.approx(1.0)
    # B gets 0 — its verifier reward was 0.
    assert dag.get("B").process_reward == pytest.approx(0.0)
    # The join keeps its own terminal reward.
    assert dag.get("join").process_reward == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it passes (should already pass from Task 12
  implementation)**

Run:
`uv run pytest customized_areal/tree_search/dag/test_backup.py::test_dag_backup_distributes_terminal_reward_at_fan_in_join -v`
Expected: PASS — the `compute()` method already reads from the credit dict, which
`CreditAssigner` populated with the fan-in distribution.

If it FAILS, revisit Task 12 Step 3 — the `compute()` method must use
`credit.get(node.node_id, own_reward)` rather than recomputing the distribution. Fix
inline and re-run.

- [ ] **Step 3: Commit (test addition only — implementation already correct)**

```bash
git add customized_areal/tree_search/dag/test_backup.py
git commit -m "test(dag): add fan-in join distribution test for DAGBackupComputer"
```

______________________________________________________________________

## Task 14: Extend TreeAdvantageComputer to consume per-node credit

**Files:**

- Modify: `customized_areal/tree_search/core/advantage.py:35-92`

**Rationale:** The existing `TreeAdvantageComputer.compute()` GRPO-normalizes one reward
per episode (`outcome_reward`) and broadcasts it to all turns. The DAG backup populates
`Node.process_reward`; the advantage computer must include it in the per-episode reward
before GRPO normalization. Per the design, the existing computer is **extended, not
replaced**.

- [ ] **Step 1: Write the failing test**

```python
# customized_areal/tree_search/dag/test_advantage_extension.py
"""Tests for the extended TreeAdvantageComputer — includes process_reward."""
from __future__ import annotations

import torch

from customized_areal.tree_search.core.advantage import TreeAdvantageComputer
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _make_node(
    *,
    node_id: str,
    query_id: str,
    episode_id: str,
    outcome_reward: float,
    process_reward: float = 0.0,
) -> Node:
    return Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 1, 1],
        logprobs=[0.0, -1.0, -2.0],
        versions=[-1, 0, 0],
        node_id=node_id,
        query_id=query_id,
        episode_id=episode_id,
        outcome_reward=outcome_reward,
        process_reward=process_reward,
    )


def test_advantage_computer_includes_process_reward_in_episode_reward() -> None:
    """The per-episode reward used for GRPO normalization is outcome_reward + process_reward."""
    store = MCTSTreeStore()
    computer = TreeAdvantageComputer(store)

    # Two episodes for the same query. Both have outcome_reward=0.5.
    # Episode 1 has process_reward=0.5 (total 1.0); episode 2 has 0.0 (total 0.5).
    n1 = _make_node(node_id="n1", query_id="q1", episode_id="e1",
                    outcome_reward=0.5, process_reward=0.5)
    n2 = _make_node(node_id="n2", query_id="q1", episode_id="e2",
                    outcome_reward=0.5, process_reward=0.0)
    computer.compute([n1, n2])

    # GRPO normalization: rewards are [1.0, 0.5], mean=0.75, std≈0.25.
    # n1 (episode 1) should get (1.0 - 0.75) / 0.25 = +1.0 (normalized).
    # n2 (episode 2) should get (0.5 - 0.75) / 0.25 = -1.0 (normalized).
    n1_return = store.get_normalized_return("n1")
    n2_return = store.get_normalized_return("n2")
    assert n1_return > n2_return
    assert n1_return == pytest.approx(1.0, abs=1e-3)
    assert n2_return == pytest.approx(-1.0, abs=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest customized_areal/tree_search/dag/test_advantage_extension.py -v`
Expected: FAIL — the existing `TreeAdvantageComputer.compute()` only reads
`outcome_reward`, ignoring `process_reward`. Both episodes have `outcome_reward=0.5`, so
they normalize to 0.0 (zero variance), not +1.0 / -1.0.

- [ ] **Step 3: Extend TreeAdvantageComputer.compute() to include process_reward**

In `advantage.py`, modify the `compute()` method — change the episode reward collection:

```python
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            ep_id = getattr(traj, "episode_id", "") or node_id
            ep_map = query_episodes.setdefault(query_id, {})
            ep_map.setdefault(ep_id, []).append(node_id)
            # All nodes in an episode share the same outcome_reward.
            # Phase 3: the per-episode reward used for GRPO normalization is
            # outcome_reward + process_reward (the DAG backup populates
            # process_reward; for non-DAG runs it defaults to 0.0).
            if ep_id not in episode_rewards:
                process = getattr(traj, "process_reward", 0.0) or 0.0
                episode_rewards[ep_id] = traj.outcome_reward + process
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest customized_areal/tree_search/dag/test_advantage_extension.py -v`
Expected: PASS

- [ ] **Step 5: Add `import pytest` to `test_advantage_extension.py` if not already
  present.**

- [ ] **Step 6: Run the existing advantage tests to verify no regression**

Run: `uv run pytest customized_areal/tree_search/ -k "advantage" -v` Expected: All
existing advantage tests still PASS (for non-DAG runs, `process_reward` defaults to 0.0,
so `outcome_reward + 0.0 == outcome_reward` — behavior unchanged for the legacy path).

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/core/advantage.py customized_areal/tree_search/dag/test_advantage_extension.py
git commit -m "feat(advantage): extend TreeAdvantageComputer to include Node.process_reward in episode reward"
```

______________________________________________________________________

## Task 15: Export from dag/__init__.py + full suite run

**Files:**

- Modify: `customized_areal/tree_search/dag/__init__.py`

- [ ] **Step 1: Write the failing test for the export**

```python
# customized_areal/tree_search/dag/test_backup.py (append)
def test_dag_package_exports_backup_type() -> None:
    from customized_areal.tree_search.dag import DAGBackupComputer
    assert DAGBackupComputer is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest customized_areal/tree_search/dag/test_backup.py::test_dag_package_exports_backup_type -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add the export**

Modify `customized_areal/tree_search/dag/__init__.py`:

```python
from customized_areal.tree_search.dag.backup import DAGBackupComputer
```

And extend `__all__`:

```python
    # backup
    "DAGBackupComputer",
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest customized_areal/tree_search/dag/test_backup.py::test_dag_package_exports_backup_type -v`
Expected: PASS

- [ ] **Step 5: Run the full DAG + tree_search test suite**

Run: `uv run pytest customized_areal/tree_search/ -v` Expected: All tests PASS —
execution_dag, environment, verifier, credit, backup, node_extensions, branch_candidate,
advantage_extension, and the existing tree_search tests.

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/dag/__init__.py
git commit -m "feat(dag): export DAGBackupComputer from dag package"
```

______________________________________________________________________

## Task 16: Pre-commit + lint

**Files:** No code changes — verification step.

- [ ] **Step 1: Run ruff check**

Run:
`cd /workspaces/leagent/backend/areal && uv run ruff check customized_areal/tree_search/dag/ customized_areal/tree_search/core/tree_store.py customized_areal/tree_search/core/checkpoint.py customized_areal/tree_search/core/customized_grouped_workflow.py customized_areal/tree_search/core/advantage.py`
Expected: No errors.

- [ ] **Step 2: Run ruff format check**

Run: `uv run ruff format --check customized_areal/tree_search/` Expected: No
reformatting needed.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore(dag): ruff fixes for Phase 3 DAG reward backup"
```

______________________________________________________________________

## Self-Review Notes

**Spec coverage:**

- Design §5 Phase 3 Task 9 (Extend Node with process_reward, DAG edge refs,
  branch_issue_id/branch_env_snapshot_id; checkpoint serialization; fix
  select_branch_candidate) → Tasks 9, 10, 11
- Design §5 Phase 3 Task 10 (DAG-aware hybrid backup + advantage computer; extend
  TreeAdvantageComputer, not replace) → Tasks 12, 13, 14
- Design §6 Testing (DAG backup tests: structural backup, process-signal shaping,
  select_branch_candidate fix) → Tasks 11, 12, 13
- Design §2 decision 6 (Reward is hybrid — terminal + process signals) → Tasks 9, 12, 14
- `.claude/rules/code-quality.md` "State machines before patches" → backup.py module
  docstring draws the backup lifecycle

**Placeholder scan:** None. Every step has concrete code or commands.

**Type consistency:**

- `Node.process_reward: float = 0.0` — added in Task 9, read in Tasks 12, 14.
- `Node.parent_run_ids: list[str]` and `Node.child_run_ids: list[str]` — added in Task 9
  with `field(default_factory=list)`.
- `Node.branch_issue_id: str | None = None` and
  `Node.branch_env_snapshot_id: str | None = None` — added in Task 9, read in Task 11
  (`select_branch_candidate`).
- `DAGBackupComputer.compute(dag, verifier_results, credit) -> dict[str, float]` —
  signature consistent across Tasks 12, 13.
- `TreeAdvantageComputer` extension reads `getattr(traj, "process_reward", 0.0)` — the
  `getattr` default keeps backward compat with any Node that lacks the field (defensive,
  but the field is added in Task 9 so always present).

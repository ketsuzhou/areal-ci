"""Tests for MCTS branch_backup (Task 3.3).

``branch_backup`` propagates a discounted branch return to the parent checkpoint
node as a running mean over visits. (Lives in ``agents/dag_backup.py`` alongside
``distribute_reward_over_dag`` rather than a new ``dag/`` package - the plan's
``dag/backup.py`` path predates the existing backup module.)
"""

from __future__ import annotations

from customized_areal.tree_search.agents.dag_backup import branch_backup
from customized_areal.tree_search.core.tree_store import Node


def _node(node_id: str = "ckpt1", value: float = 0.0, visit_count: int = 0) -> Node:
    # Node requires the four sequence fields; supply empty lists for the
    # backup-only tests.
    return Node(
        input_ids=[],
        loss_mask=[],
        logprobs=[],
        versions=[],
        node_id=node_id,
        value=value,
        visit_count=visit_count,
    )


def test_branch_return_updates_parent_value():
    parent = _node(value=0.0, visit_count=0)
    branch_backup(parent, branch_return=1.0, discount=0.9, visit_count=1)
    assert parent.value == 0.9
    assert parent.visit_count == 1


def test_multiple_branches_aggregate():
    parent = _node(value=0.0, visit_count=0)
    branch_backup(parent, branch_return=1.0, discount=1.0, visit_count=1)
    branch_backup(parent, branch_return=0.0, discount=1.0, visit_count=1)
    assert parent.value == 0.5
    assert parent.visit_count == 2


def test_branch_backup_preserves_existing_value_via_running_mean():
    # A node that already has value 0.8 over 2 visits; a new branch return 0.4
    # (discount 1.0) updates the running mean: (0.8*2 + 0.4*1)/3 = 2.0/3.
    parent = _node(value=0.8, visit_count=2)
    branch_backup(parent, branch_return=0.4, discount=1.0, visit_count=1)
    assert parent.value == 2.0 / 3.0
    assert parent.visit_count == 3

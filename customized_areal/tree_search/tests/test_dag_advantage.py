"""Tests for the DAG advantage assembler (Phase 3, Task 9).

This is the torch-free core of "GAE replaces GRPO for DAG runs": it takes the
DAG nodes in global completion order (each carrying the critic value V_{t+1} and
its rewards), runs the global GAE, and returns the per-node advantage/return the
trainer broadcasts onto actor tokens / regresses the critic against. Also the
``critic_loss_weight`` config field + the explained-variance metric.

Torch-free.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.config import Config
from customized_areal.tree_search.dag.dag_advantage import (
    AssembledAdvantages,
    assemble_node_advantages,
    explained_variance,
)
from customized_areal.tree_search.dag.execution_dag import AgentRunNode


def _node(nid: str, *, value: float, process: float = 0.0, outcome: float = 0.0):
    n = AgentRunNode(node_id=nid, agent_id="a", issue_id="i", task_id="t")
    n.value = value
    n.process_reward = process
    n.outcome_reward = outcome
    return n


def test_assemble_node_advantages_matches_global_gae() -> None:
    nodes = [
        _node("n0", value=0.5),
        _node("n1", value=0.7),
        _node("n2", value=0.0, outcome=1.0),  # terminal verifier reward
    ]
    out = assemble_node_advantages(nodes, initial_value=0.0, gamma=1.0, lam=1.0)
    assert isinstance(out, AssembledAdvantages)
    assert out.advantages == {
        "n0": pytest.approx(1.0),
        "n1": pytest.approx(0.5),
        "n2": pytest.approx(0.3),
    }
    assert out.returns == {
        "n0": pytest.approx(1.0),
        "n1": pytest.approx(1.0),
        "n2": pytest.approx(1.0),
    }


def test_explained_variance_perfect_and_zero() -> None:
    # Perfect value prediction -> explained variance 1.0.
    assert explained_variance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    # Constant prediction equal to mean of targets -> ev 0.0.
    assert explained_variance([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_config_has_critic_loss_weight_default_and_validation() -> None:
    cfg = Config()
    assert hasattr(cfg, "critic_loss_weight")
    assert cfg.critic_loss_weight >= 0.0
    with pytest.raises(ValueError):
        Config(critic_loss_weight=-1.0)

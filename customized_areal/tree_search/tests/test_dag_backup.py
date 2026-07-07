import pytest

from customized_areal.tree_search.agents.dag_backup import (
    CreditAssignment,
    distribute_reward_over_dag,
)
from customized_areal.tree_search.agents.execution_dag import (
    EdgeType,
    ExecutionDAG,
    SuperNode,
)


def _node(node_id: str) -> SuperNode:
    return SuperNode(
        node_id=node_id,
        agent_id="a",
        issue_id="i",
        task_id="t",
    )


def _dag_linear() -> ExecutionDAG:
    # root -> child1 -> child2 (sequential delegation chain)
    dag = ExecutionDAG()
    for nid in ("root", "c1", "c2"):
        dag.add_event(_node(nid))
    dag.add_edge("root", "c1", EdgeType.DELEGATION)
    dag.add_edge("c1", "c2", EdgeType.DELEGATION)
    return dag


def test_distribute_reward_over_linear_chain():
    dag = _dag_linear()
    credit = distribute_reward_over_dag(dag, terminal_reward=1.0, terminal_node_id="c2")
    # The terminal reward backs up along edges: c2 gets 1.0, c1 and root
    # each get the full pass-through (no attenuation).
    assert credit["c2"] == pytest.approx(1.0)
    assert credit["c1"] > 0.0
    assert credit["root"] > 0.0
    # Monotonic: closer to terminal => >= further from terminal.
    assert credit["c2"] >= credit["c1"] >= credit["root"]


def test_distribute_reward_unknown_terminal_raises():
    dag = _dag_linear()
    with pytest.raises(KeyError):
        distribute_reward_over_dag(dag, terminal_reward=1.0, terminal_node_id="nope")


def _dag_fan_in() -> ExecutionDAG:
    # root delegates to a and b; both report back to join (fan-in).
    dag = ExecutionDAG()
    for nid in ("root", "a", "b", "join"):
        dag.add_event(_node(nid))
    dag.add_edge("root", "a", EdgeType.DELEGATION)
    dag.add_edge("root", "b", EdgeType.DELEGATION)
    dag.add_edge("a", "join", EdgeType.COMPLETION)
    dag.add_edge("b", "join", EdgeType.COMPLETION)
    return dag


def test_fan_in_credit_is_explicit_not_aggregated():
    dag = _dag_fan_in()

    # The caller supplies explicit per-agent credit at the fan-in join
    # (spec §2 decision 8: no fixed sum/mean/max). a contributed more than b.
    assignment = CreditAssignment(per_node={"a": 0.7, "b": 0.3})
    credit = distribute_reward_over_dag(
        dag, terminal_reward=1.0, terminal_node_id="join", fan_in_credit=assignment
    )
    assert credit["a"] == pytest.approx(0.7)
    assert credit["b"] == pytest.approx(0.3)
    # root gets the sum (the join's reward backed up to root through a and b).
    assert credit["root"] == pytest.approx(1.0)

"""Tests for global joint-state GAE over the DAG event sequence (Phase 3, Task 7).

Framework B: GAE runs over the GLOBAL completion-ordered turn sequence (one
shared policy acting through every agent), not per-task_id lanes.

Conventions:
  - ``initial_value`` = V_0 (entering state of turn 0).
  - ``GlobalEvent.value`` = V_{t+1}, the next-state value the critic produced
    when that turn completed.
  - baseline for turn t is V_t (= the previous event's value, or V_0 for t=0)
    -> action-independent.
  - terminal bootstrap = 0; the verifier terminal reward enters as the last
    event's reward; per-node process signals are the intermediate rewards.

Torch-free.
"""

from __future__ import annotations

import pytest

from customized_areal.tree_search.dag.gae import (
    GlobalEvent,
    NodeGAEResult,
    compute_global_gae,
)


def test_single_event_reduces_to_reward_minus_baseline() -> None:
    # T=1: delta = r0 + gamma*0 - V0 ; adv = r0 - V0 ; return = r0.
    events = [GlobalEvent(node_id="n0", value=0.9, reward=1.0)]
    out = compute_global_gae(events, initial_value=0.2, gamma=1.0, lam=1.0)
    assert len(out) == 1
    assert isinstance(out[0], NodeGAEResult)
    assert out[0].node_id == "n0"
    assert out[0].advantage == pytest.approx(0.8)  # 1.0 - 0.2
    assert out[0].return_ == pytest.approx(1.0)
    assert out[0].baseline_value == pytest.approx(0.2)  # V_0


def test_three_event_chain_known_values() -> None:
    # gamma=lam=1, terminal reward 1.0 -> telescoping returns all 1.0.
    events = [
        GlobalEvent(node_id="n0", value=0.5, reward=0.0),  # V_1=0.5
        GlobalEvent(node_id="n1", value=0.7, reward=0.0),  # V_2=0.7
        GlobalEvent(node_id="n2", value=0.0, reward=1.0),  # terminal r=1.0
    ]
    out = compute_global_gae(events, initial_value=0.0, gamma=1.0, lam=1.0)
    advs = [round(o.advantage, 6) for o in out]
    rets = [round(o.return_, 6) for o in out]
    # Hand-computed: adv = [1.0, 0.5, 0.3]; returns = [1.0, 1.0, 1.0].
    assert advs == [1.0, 0.5, 0.3]
    assert rets == [1.0, 1.0, 1.0]
    # baselines are V_0, V_1, V_2.
    assert [round(o.baseline_value, 6) for o in out] == [0.0, 0.5, 0.7]


def test_discounting_with_gamma_and_lambda() -> None:
    events = [
        GlobalEvent(node_id="n0", value=1.0, reward=0.0),
        GlobalEvent(node_id="n1", value=0.0, reward=2.0),
    ]
    gamma, lam = 0.9, 0.5
    out = compute_global_gae(events, initial_value=0.0, gamma=gamma, lam=lam)
    # t=1: delta1 = 2.0 + gamma*0 - 1.0 = 1.0 ; adv1 = 1.0
    # t=0: delta0 = 0.0 + gamma*1.0 - 0.0 = 0.9 ; adv0 = 0.9 + gamma*lam*1.0 = 1.35
    assert out[1].advantage == pytest.approx(1.0)
    assert out[0].advantage == pytest.approx(0.9 + gamma * lam * 1.0)
    assert out[0].return_ == pytest.approx(out[0].advantage + 0.0)
    assert out[1].return_ == pytest.approx(out[1].advantage + 1.0)


def test_baseline_is_action_independent_of_its_own_value() -> None:
    events = [
        GlobalEvent(node_id="n0", value=0.5, reward=0.0),
        GlobalEvent(node_id="n1", value=0.7, reward=0.0),
        GlobalEvent(node_id="n2", value=0.0, reward=1.0),
    ]
    out_a = compute_global_gae(events, initial_value=0.0, gamma=1.0, lam=1.0)

    # Mutate turn n1's OWN next-state value (V_2). This is the action-dependent
    # quantity; it must NOT change n1's baseline (V_1), which comes from n0.
    mutated = list(events)
    mutated[1] = GlobalEvent(node_id="n1", value=0.999, reward=0.0)
    out_b = compute_global_gae(mutated, initial_value=0.0, gamma=1.0, lam=1.0)

    by_a = {o.node_id: o for o in out_a}
    by_b = {o.node_id: o for o in out_b}
    # n1's baseline (V_1) unchanged.
    assert by_a["n1"].baseline_value == pytest.approx(by_b["n1"].baseline_value)
    # n2's baseline (V_2) DID change (it equals n1's mutated value).
    assert by_a["n2"].baseline_value != pytest.approx(by_b["n2"].baseline_value)


def test_empty_event_sequence_returns_empty() -> None:
    assert compute_global_gae([], initial_value=0.0, gamma=1.0, lam=1.0) == []


def test_events_from_nodes_reads_value_and_combines_rewards() -> None:
    from customized_areal.tree_search.dag.execution_dag import AgentRunNode
    from customized_areal.tree_search.dag.gae import events_from_nodes

    n0 = AgentRunNode(node_id="n0", agent_id="a", issue_id="i", task_id="t")
    n0.value = 0.5
    n0.process_reward = 0.1
    n1 = AgentRunNode(node_id="n1", agent_id="a", issue_id="i", task_id="t")
    n1.value = 0.0
    n1.process_reward = 0.0
    n1.outcome_reward = 1.0  # verifier terminal reward lands on the node

    events = events_from_nodes([n0, n1])
    assert events[0] == GlobalEvent(node_id="n0", value=0.5, reward=0.1)
    # terminal node reward = process_reward + outcome_reward
    assert events[1] == GlobalEvent(node_id="n1", value=0.0, reward=1.0)


def test_node_value_defaults_to_none() -> None:
    from customized_areal.tree_search.dag.execution_dag import AgentRunNode

    n = AgentRunNode(node_id="n", agent_id="a", issue_id="i", task_id="t")
    assert n.value is None


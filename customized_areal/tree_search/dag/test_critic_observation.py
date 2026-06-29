"""Tests for the critic-agent observation builder (Phase 3, Task 5).

Framework B: the critic value V_t is the value of the **global joint state** of
all agents + environment at a moment, mapped to a frontier (cut) over the DAG.
The global timeline is the completion order of turns; each turn output that
carries a ``node_id`` is a critic trigger that produces a value.

Next-state indexing convention:
  - V_0 = the global state BEFORE any turn output (entering state of turn 0).
  - When turn j completes (its output at global position p_j), the observation
    includes the frontier up to and INCLUDING p_j -> it is V_{j+1}.
  - Turn j's baseline V_j was produced at turn (j-1)'s completion and does NOT
    include turn j's output -> action-independent.

Torch-free.
"""

from __future__ import annotations

from customized_areal.tree_search.dag.critic_observation import (
    DEFAULT_CRITIC_FIELDS,
    CriticObservation,
    build_critic_observations,
)


def _msg(
    role: str,
    content: str,
    *,
    node_id: str | None = None,
    agent_id: str = "ag",
    **extra,
):
    m = {"role": role, "content": content, "agent_id": agent_id, **extra}
    if node_id is not None:
        m["node_id"] = node_id
    return m


# A global, completion-ordered timeline of two agents (A, B) interleaved.
def _timeline() -> list[dict]:
    return [
        _msg("user", "task prompt"),  # seed / environment, no node_id
        _msg("assistant", "A turn 0 output", node_id="A0", agent_id="A"),
        _msg("assistant", "B turn 0 output", node_id="B0", agent_id="B"),
        _msg("assistant", "A turn 1 output", node_id="A1", agent_id="A"),
    ]


def test_sequence_length_is_num_turns_plus_one() -> None:
    obs = build_critic_observations(_timeline())
    # 3 turn outputs -> V_0, V_1, V_2, V_3
    assert [o.value_index for o in obs] == [0, 1, 2, 3]
    assert obs[0].node_id is None  # V_0 = initial state
    assert obs[1].node_id == "A0"
    assert obs[2].node_id == "B0"
    assert obs[3].node_id == "A1"


def test_initial_observation_excludes_all_turn_outputs() -> None:
    obs = build_critic_observations(_timeline())
    v0 = obs[0]
    contents = [m["content"] for m in v0.messages]
    assert contents == ["task prompt"]  # only the seed, no turn outputs


def test_frontier_is_global_across_lanes() -> None:
    # V after B's turn 0 must include A's earlier turn 0 (cross-lane frontier).
    obs = build_critic_observations(_timeline())
    v2 = obs[2]  # produced when B0 completed
    contents = [m["content"] for m in v2.messages]
    assert "A turn 0 output" in contents
    assert "B turn 0 output" in contents
    assert "A turn 1 output" not in contents  # not yet completed


def test_field_whitelist_drops_extraneous_fields() -> None:
    timeline = [
        _msg(
            "assistant",
            "out",
            node_id="A0",
            agent_id="A",
            logprobs=[0.1],
            token_ids=[1, 2],
        ),
    ]
    obs = build_critic_observations(timeline)
    v1 = obs[1]
    assert set(v1.messages[0].keys()) <= set(DEFAULT_CRITIC_FIELDS)
    assert "logprobs" not in v1.messages[0]
    assert "token_ids" not in v1.messages[0]


def test_value_is_next_state_and_baseline_is_action_independent() -> None:
    base = _timeline()
    obs_a = build_critic_observations(base)

    # Mutate turn 1 (A1)'s output content.
    mutated = _timeline()
    mutated[3] = _msg(
        "assistant", "A turn 1 COMPLETELY DIFFERENT", node_id="A1", agent_id="A"
    )
    obs_b = build_critic_observations(mutated)

    # V_3 = next-state value of turn A1 -> MUST change.
    assert obs_a[3].text != obs_b[3].text
    # V_2 = baseline for the turn that produced event index 3 region; the
    # baseline of A1 is V_2 (produced at B0's completion) and MUST be unchanged.
    assert obs_a[2].text == obs_b[2].text
    assert obs_a[1].text == obs_b[1].text
    assert obs_a[0].text == obs_b[0].text


def test_observation_is_frozen_dataclass() -> None:
    obs = build_critic_observations(_timeline())
    assert isinstance(obs[0], CriticObservation)
    assert obs[1].text  # rendered text is non-empty for a turn observation

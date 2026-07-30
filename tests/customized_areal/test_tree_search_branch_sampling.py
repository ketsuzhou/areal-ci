import copy
import pickle

import pytest

from customized_areal.db_service.messages import (
    copy_messages_to_task,
    truncate_messages_before_turn,
)
from customized_areal.tpfc.tpfc_agent import TPFCAgentResult
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
    annotate_nodes_from_run,
)
from customized_areal.tree_search.core.tree_store import Node


class FakeInsert:
    async def execute(self):
        return None


class FakeTable:
    def __init__(self):
        self.inserted_rows = None
        self.insert_calls = 0

    def insert(self, rows):
        self.insert_calls += 1
        self.inserted_rows = rows
        return FakeInsert()


class FakeClient:
    def __init__(self):
        self.tables = {}

    def table(self, name):
        table = FakeTable()
        self.tables[name] = table
        return table


def test_truncate_messages_before_high_entropy_assistant_turn():
    raw_messages = [
        {
            "message_id": "u1",
            "task_id": "task-1",
            "role": "user",
            "content": {"role": "user", "content": "q"},
            "created_at": "2026-05-27T00:00:00Z",
            "updated_at": "2026-05-27T00:00:00Z",
        },
        {
            "message_id": "a1",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step 1"},
            "metadata": {"entropy_stats": {"max_entropy": 0.1}},
        },
        {
            "message_id": "t1",
            "role": "tool",
            "content": {"role": "tool", "content": "obs"},
        },
        {
            "message_id": "a2",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step 2"},
            "metadata": {"need_branch": True},
        },
        {
            "message_id": "a3",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step 3"},
        },
    ]

    prefix = truncate_messages_before_turn(raw_messages, assistant_turn_idx=2)

    assert [row["role"] for row in prefix] == ["user", "assistant", "tool"]
    assert all("message_id" not in row for row in prefix)
    assert all("task_id" not in row for row in prefix)
    assert all("created_at" not in row for row in prefix)
    assert all("updated_at" not in row for row in prefix)


@pytest.mark.asyncio
async def test_copy_messages_to_task_sanitizes_rows_and_preserves_inputs():
    client = FakeClient()
    messages = [
        {
            "message_id": "m1",
            "task_id": "source-task",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step"},
            "metadata": {"entropy_stats": {"max_entropy": 0.7}},
            "created_at": "2026-05-27T00:00:00Z",
            "updated_at": "2026-05-27T00:00:00Z",
        }
    ]
    original_messages = copy.deepcopy(messages)

    await copy_messages_to_task(client, task_id="target-task", messages=[])

    assert client.tables == {}

    await copy_messages_to_task(client, task_id="target-task", messages=messages)

    inserted_rows = client.tables["messages"].inserted_rows
    assert client.tables["messages"].insert_calls == 1
    assert inserted_rows == [
        {
            "task_id": "target-task",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step"},
            "metadata": {"entropy_stats": {"max_entropy": 0.7}},
        }
    ]
    assert all("message_id" not in row for row in inserted_rows)
    assert all("created_at" not in row for row in inserted_rows)
    assert all("updated_at" not in row for row in inserted_rows)
    assert messages == original_messages


def _node(turn_idx: int) -> Node:
    return Node(
        input_ids=[turn_idx],
        loss_mask=[1],
        logprobs=[0.0],
        versions=[0],
        node_id=f"n{turn_idx}",
        turn_idx=turn_idx,
    )


def test_node_has_no_branch_sandbox_id():
    """Clean-break contract: the legacy branch_sandbox_id field is gone."""
    node = _node(1)
    assert not hasattr(node, "branch_sandbox_id")


def test_workflow_no_longer_exposes_branch_machinery():
    """The wired grouped workflow no longer branches; the helpers are removed."""
    import customized_areal.tree_search.core.customized_grouped_workflow as mod

    assert not hasattr(mod, "build_branch_task")
    assert not hasattr(mod.TreeSearchGroupedRolloutWorkflow, "_prepare_branch_task")
    assert not hasattr(mod.TreeSearchGroupedRolloutWorkflow, "_cleanup_branch")


def test_annotate_nodes_from_run_copies_entropy_metadata():
    nodes = [_node(1), _node(2)]
    raw_messages = [
        {
            "role": "assistant",
            "metadata": {
                "entropy_stats": {"max_entropy": 0.2},
                "need_branch": False,
            },
        },
        {
            "role": "assistant",
            "metadata": {
                "entropy_stats": {"max_entropy": 3.4},
                "need_branch": True,
                "env_id": "env-2",
            },
        },
    ]

    annotate_nodes_from_run(nodes, task_id="task-id", raw_messages=raw_messages)

    assert nodes[0].task_id == "task-id"
    assert nodes[0].entropy_stats == {"max_entropy": 0.2}
    assert nodes[0].need_branch is False
    assert nodes[0].env_id is None
    assert nodes[1].task_id == "task-id"
    assert nodes[1].entropy_stats == {"max_entropy": 3.4}
    assert nodes[1].need_branch is True
    assert nodes[1].env_id == "env-2"
    assert not hasattr(nodes[1], "branch_sandbox_id")


def test_annotate_nodes_from_run_populates_topk_from_raw_message_metadata():
    node = _node(1)
    raw_messages = [
        {
            "role": "assistant",
            "metadata": {
                "top_logprobs": [
                    [
                        {"token_id": 11, "token": "a", "logprob": -0.1},
                        {"token_id": 12, "token": "b", "logprob": -0.2},
                    ],
                    [
                        {"token_id": 21, "token": "c", "logprob": -0.3},
                    ],
                ],
            },
        }
    ]

    annotate_nodes_from_run([node], task_id="task-id", raw_messages=raw_messages)

    assert node.topk_ids == [[11, 12], [21]]
    assert node.topk_logp is None


def test_annotate_nodes_from_run_does_not_override_existing_topk():
    node = _node(1)
    node.topk_ids = [[101, 102]]
    raw_messages = [
        {
            "role": "assistant",
            "metadata": {
                "top_logprobs": [
                    [
                        {"token_id": 11, "token": "a", "logprob": -0.1},
                    ]
                ],
            },
        }
    ]

    annotate_nodes_from_run([node], task_id="task-id", raw_messages=raw_messages)

    assert node.topk_ids == [[101, 102]]
    assert node.topk_logp is None


def test_annotate_nodes_from_run_skips_invalid_turn_idx():
    node = _node(0)
    raw_messages = [
        {
            "role": "assistant",
            "metadata": {
                "entropy_stats": {"max_entropy": 9.9},
                "need_branch": True,
                "env_id": "wrong",
            },
        }
    ]

    annotate_nodes_from_run([node], task_id="task-id", raw_messages=raw_messages)

    assert node.task_id == "task-id"
    assert node.entropy_stats is None
    assert node.need_branch is False
    assert node.env_id is None


@pytest.mark.asyncio
async def test_run_fresh_episode_uses_isolated_data_for_scratch_metadata():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )

    seen_episode_data = []

    async def retry_episode(engine, episode_data, group_idx):
        seen_episode_data.append(episode_data)
        episode_data["_backend_run_task_id"] = f"task-{group_idx}"
        episode_data["_backend_run_raw_messages"] = [
            {"role": "assistant", "metadata": {"need_branch": True}}
        ]
        return {"group_idx": group_idx}

    workflow._retry_episode = retry_episode

    shared_data = {"query_id": "q"}
    first = await workflow._run_fresh_episode(None, shared_data, 1, "q")
    second = await workflow._run_fresh_episode(None, shared_data, 2, "q")

    assert shared_data == {"query_id": "q"}
    assert seen_episode_data[0] is not shared_data
    assert seen_episode_data[1] is not shared_data
    assert seen_episode_data[0] is not seen_episode_data[1]
    assert first.task_id == "task-1"
    assert second.task_id == "task-2"


@pytest.mark.asyncio
async def test_run_fresh_episode_never_branches_even_with_candidate():
    """The wired loop always runs a scratch episode, ignoring branch candidates."""
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    # A branch candidate present in the store must NOT trigger any branch path.
    candidate = _node(2)
    candidate.query_id = "q"
    candidate.task_id = "source-task"
    candidate.need_branch = True
    workflow.tree_store = type("Store", (), {"trajectories": {"q": [candidate]}})()

    calls = []

    async def retry_episode(engine, episode_data, group_idx):
        calls.append((episode_data, group_idx))
        return {"scratch": True}

    workflow._retry_episode = retry_episode

    data = {"query_id": "q"}
    result = await workflow._run_fresh_episode(None, data, 3, "q")

    assert result == {"scratch": True}
    assert calls == [(data, 3)]
    assert "seed_messages_already_inserted" not in data
    assert "_branch_point_node_id" not in data


def test_tpfca_agent_result_is_picklable():
    result = TPFCAgentResult(
        reward=0.75,
        task_id="task-123",
        raw_messages=[{"role": "assistant", "content": "hi"}],
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.reward == 0.75
    assert restored.task_id == "task-123"
    assert restored.raw_messages == [{"role": "assistant", "content": "hi"}]


def test_tpfca_agent_result_default_fields():
    result = TPFCAgentResult(reward=0.5)
    assert result.task_id == ""
    assert result.raw_messages == []


def test_with_episode_metadata_extracts_tpfca_result():
    """_with_episode_metadata should read _backend_run_task_id and _backend_run_raw_messages from data."""
    from customized_areal.tree_search.core.customized_grouped_workflow import (
        EpisodeRunResult,
        _with_episode_metadata,
    )

    data = {
        "_backend_run_task_id": "task-xyz",
        "_backend_run_raw_messages": [{"role": "assistant", "content": "hi"}],
    }
    result = _with_episode_metadata("some_result", data)
    assert isinstance(result, EpisodeRunResult)
    assert result.task_id == "task-xyz"
    assert result.raw_messages == [{"role": "assistant", "content": "hi"}]
    assert result.result == "some_result"


def test_tpfca_agent_result_propagates_to_data_dict():
    """Simulate what arun_episode does when it receives TPFCAgentResult."""
    data = {"query_id": "q1"}
    rewards = TPFCAgentResult(
        reward=0.9,
        task_id="task-abc",
        raw_messages=[{"role": "assistant", "content": "response"}],
    )

    # This is the duck-typing logic added to arun_episode:
    if hasattr(rewards, "task_id") and hasattr(rewards, "raw_messages"):
        data["_backend_run_task_id"] = rewards.task_id
        data["_backend_run_raw_messages"] = rewards.raw_messages
        rewards = rewards.reward

    assert rewards == 0.9
    assert data["_backend_run_task_id"] == "task-abc"
    assert data["_backend_run_raw_messages"] == [
        {"role": "assistant", "content": "response"}
    ]

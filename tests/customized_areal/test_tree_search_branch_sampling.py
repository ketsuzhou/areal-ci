import copy
import pickle

import pytest

from customized_areal.db_service.messages import (
    copy_messages_to_task,
    truncate_messages_before_turn,
)
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import SampleSource
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node
from customized_areal.tree_search.tree_search_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
    annotate_nodes_from_run,
    build_branch_task,
    choose_sample_source,
    select_branch_candidate,
)


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
                "branch_sandbox_id": "sb2",
            },
        },
    ]

    annotate_nodes_from_run(nodes, task_id="task-id", raw_messages=raw_messages)

    assert nodes[0].task_id == "task-id"
    assert nodes[0].entropy_stats == {"max_entropy": 0.2}
    assert nodes[0].need_branch is False
    assert nodes[1].task_id == "task-id"
    assert nodes[1].entropy_stats == {"max_entropy": 3.4}
    assert nodes[1].need_branch is True
    assert nodes[1].branch_sandbox_id == "sb2"


def test_annotate_nodes_from_run_skips_invalid_turn_idx():
    node = _node(0)
    raw_messages = [
        {
            "role": "assistant",
            "metadata": {
                "entropy_stats": {"max_entropy": 9.9},
                "need_branch": True,
                "branch_sandbox_id": "wrong",
            },
        }
    ]

    annotate_nodes_from_run([node], task_id="task-id", raw_messages=raw_messages)

    assert node.task_id == "task-id"
    assert node.entropy_stats is None
    assert node.need_branch is False
    assert node.branch_sandbox_id is None


def test_checkpoint_preserves_tpfc_node_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_ID", "train-now")
    store = MCTSTreeStore()
    node = _node(1)
    node.query_id = "query-1"
    node.task_id = "task-id"
    node.entropy_stats = {"max_entropy": 3.4}
    node.need_branch = True
    node.branch_sandbox_id = "sandbox-id"
    store.insert_batch([node])

    manager = TreeCheckpointManager(str(tmp_path))
    manager.save(store)
    loaded = manager.load()

    loaded_node = loaded.trajectories["query-1"][0]
    assert loaded_node.task_id == "task-id"
    assert loaded_node.entropy_stats == {"max_entropy": 3.4}
    assert loaded_node.need_branch is True
    assert loaded_node.branch_sandbox_id == "sandbox-id"


def test_choose_sample_source_modes():
    assert (
        choose_sample_source(
            SampleSource.SCRATCH,
            branch_probability=1.0,
            has_candidate=True,
            random_value=0.0,
        )
        == SampleSource.SCRATCH
    )
    assert (
        choose_sample_source(
            SampleSource.BRANCH,
            branch_probability=0.0,
            has_candidate=True,
            random_value=1.0,
        )
        == SampleSource.BRANCH
    )
    assert (
        choose_sample_source(
            SampleSource.BRANCH,
            branch_probability=1.0,
            has_candidate=False,
            random_value=0.0,
        )
        == SampleSource.SCRATCH
    )
    assert (
        choose_sample_source(
            SampleSource.MIXED,
            branch_probability=0.5,
            has_candidate=True,
            random_value=0.4,
        )
        == SampleSource.BRANCH
    )
    assert (
        choose_sample_source(
            SampleSource.MIXED,
            branch_probability=0.5,
            has_candidate=True,
            random_value=0.6,
        )
        == SampleSource.SCRATCH
    )


def test_select_branch_candidate_prefers_max_entropy():
    low = _node(1)
    low.query_id = "q"
    low.task_id = "task-low"
    low.need_branch = True
    low.branch_sandbox_id = "sb-low"
    low.entropy_stats = {"max_entropy": 2.5}

    high = _node(2)
    high.query_id = "q"
    high.task_id = "task-high"
    high.need_branch = True
    high.branch_sandbox_id = "sb-high"
    high.entropy_stats = {"max_entropy": 4.0}

    assert select_branch_candidate([low, high], "q") is high


def test_select_branch_candidate_filters_invalid_nodes_and_tolerates_bad_entropy():
    wrong_query = _node(1)
    wrong_query.query_id = "other"
    wrong_query.task_id = "task-wrong"
    wrong_query.need_branch = True
    wrong_query.branch_sandbox_id = "sb-wrong"
    wrong_query.entropy_stats = {"max_entropy": 99.0}

    missing_task = _node(2)
    missing_task.query_id = "q"
    missing_task.need_branch = True
    missing_task.branch_sandbox_id = "sb-missing-task"
    missing_task.entropy_stats = {"max_entropy": 98.0}

    missing_sandbox = _node(3)
    missing_sandbox.query_id = "q"
    missing_sandbox.task_id = "task-missing-sandbox"
    missing_sandbox.need_branch = True
    missing_sandbox.entropy_stats = {"max_entropy": 97.0}

    no_branch = _node(4)
    no_branch.query_id = "q"
    no_branch.task_id = "task-no-branch"
    no_branch.branch_sandbox_id = "sb-no-branch"
    no_branch.need_branch = False
    no_branch.entropy_stats = {"max_entropy": 96.0}

    candidate = _node(5)
    candidate.query_id = "q"
    candidate.task_id = "task-candidate"
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-candidate"
    candidate.entropy_stats = {"max_entropy": "not numeric"}

    assert (
        select_branch_candidate(
            [wrong_query, missing_task, missing_sandbox, no_branch, candidate],
            "q",
        )
        is candidate
    )
    assert select_branch_candidate([wrong_query, missing_task], "q") is None


@pytest.mark.asyncio
async def test_build_branch_task_uses_sandbox_directly_without_clone(monkeypatch):
    """build_branch_task should bind candidate.branch_sandbox_id directly, not clone it."""
    created = []
    copied = []

    class FakeClient:
        pass

    candidate = _node(2)
    candidate.task_id = "source-task"
    candidate.turn_idx = 2
    candidate.branch_sandbox_id = "direct-sandbox-id"

    raw_messages = [
        {
            "message_id": "u1",
            "role": "user",
            "content": {"role": "user", "content": "q"},
        },
        {
            "message_id": "a1",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step1"},
        },
        {
            "message_id": "a2",
            "role": "assistant",
            "content": {"role": "assistant", "content": "step2"},
        },
    ]

    async def fake_get_raw_messages(client, task_id):
        assert isinstance(client, FakeClient)
        assert task_id == "source-task"
        return raw_messages

    async def fake_create_task(**kwargs):
        created.append(kwargs)
        return "branch-task"

    async def fake_bind_sandbox_to_task(client, *, sandbox_id, task_id, account_id):
        assert isinstance(client, FakeClient)
        copied.append(("bind", sandbox_id, task_id, account_id))

    async def fake_copy_messages_to_task(client, *, task_id, messages):
        assert isinstance(client, FakeClient)
        copied.append(("messages", task_id, messages))

    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow._get_raw_messages_with_client",
        fake_get_raw_messages,
    )
    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.create_task",
        fake_create_task,
    )
    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.bind_sandbox_to_task",
        fake_bind_sandbox_to_task,
    )
    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.copy_messages_to_task",
        fake_copy_messages_to_task,
    )

    branch_task_id = await build_branch_task(
        client=FakeClient(),
        account_id="account",
        agent_id="agent",
        candidate=candidate,
        name="branch",
    )

    assert branch_task_id == "branch-task"
    assert created[0]["account_id"] == "account"
    assert created[0]["agent_id"] == "agent"
    assert created[0]["name"] == "branch"
    # Key assertion: sandbox bound is the DIRECT sandbox, not a clone
    assert copied[0] == ("bind", "direct-sandbox-id", "branch-task", "account")
    assert copied[1] == (
        "messages",
        "branch-task",
        [
            {"role": "user", "content": {"role": "user", "content": "q"}},
            {
                "role": "assistant",
                "content": {"role": "assistant", "content": "step1"},
            },
        ],
    )


@pytest.mark.asyncio
async def test_run_fresh_episode_falls_back_to_scratch_when_branch_prep_errors():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    workflow.sample_source = SampleSource.BRANCH
    workflow.branch_probability = 1.0

    candidate = _node(2)
    candidate.query_id = "q"
    candidate.task_id = "source-task"
    candidate.need_branch = True
    candidate.branch_sandbox_id = "source-sandbox"
    workflow.tree_store = type("Store", (), {"trajectories": {"q": [candidate]}})()

    calls = []

    async def fail_prepare_branch_task(data, candidate_arg):
        assert candidate_arg is candidate
        raise RuntimeError("copy failed")

    async def retry_episode(engine, episode_data, group_idx):
        calls.append((episode_data, group_idx))
        return {"scratch": True}

    workflow._prepare_branch_task = fail_prepare_branch_task
    workflow._retry_episode = retry_episode

    data = {"query_id": "q"}
    result = await workflow._run_fresh_episode(None, data, 3, "q")

    assert result == {"scratch": True}
    assert calls == [(data, 3)]
    assert "seed_messages_already_inserted" not in data


@pytest.mark.asyncio
async def test_run_fresh_episode_uses_isolated_data_for_scratch_metadata():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    workflow.sample_source = SampleSource.SCRATCH
    workflow.branch_probability = 0.0
    workflow.tree_store = type("Store", (), {"trajectories": {"q": []}})()

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


from customized_areal.tpfc.tpfc_agent import TPFCAgentResult


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
    from customized_areal.tree_search.tree_search_grouped_workflow import (
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


@pytest.mark.asyncio
async def test_cleanup_branch_deletes_sandbox_and_clears_node_state(monkeypatch):
    deleted_ids = []

    async def fake_delete_sandbox(sandbox_id):
        deleted_ids.append(sandbox_id)

    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.delete_sandbox",
        fake_delete_sandbox,
    )

    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    candidate = _node(2)
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-to-delete"

    await workflow._cleanup_branch(candidate)

    assert deleted_ids == ["sb-to-delete"]
    assert candidate.need_branch is False
    assert candidate.branch_sandbox_id is None


@pytest.mark.asyncio
async def test_cleanup_branch_tolerates_delete_failure(monkeypatch):
    async def failing_delete(sandbox_id):
        raise RuntimeError("sandbox API down")

    monkeypatch.setattr(
        "customized_areal.tree_search.tree_search_grouped_workflow.delete_sandbox",
        failing_delete,
    )

    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    candidate = _node(2)
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-failing"

    await workflow._cleanup_branch(candidate)

    # Node state should still be cleared even if delete fails
    assert candidate.need_branch is False
    assert candidate.branch_sandbox_id is None


@pytest.mark.asyncio
async def test_cleanup_branch_skips_delete_when_no_sandbox_id():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    candidate = _node(2)
    candidate.need_branch = True
    candidate.branch_sandbox_id = None

    await workflow._cleanup_branch(candidate)

    assert candidate.need_branch is False
    assert candidate.branch_sandbox_id is None


def test_select_branch_candidate_ignores_cleaned_up_node():
    """After _cleanup_branch, node should not be a branch candidate."""
    cleaned = _node(2)
    cleaned.query_id = "q"
    cleaned.task_id = "task-cleaned"
    cleaned.need_branch = False  # cleared by _cleanup_branch
    cleaned.branch_sandbox_id = None  # cleared by _cleanup_branch
    cleaned.entropy_stats = {"max_entropy": 5.0}

    assert select_branch_candidate([cleaned], "q") is None


@pytest.mark.asyncio
async def test_run_fresh_episode_calls_cleanup_after_branch():
    workflow = TreeSearchGroupedRolloutWorkflow.__new__(
        TreeSearchGroupedRolloutWorkflow
    )
    workflow.sample_source = SampleSource.BRANCH
    workflow.branch_probability = 1.0

    candidate = _node(2)
    candidate.query_id = "q"
    candidate.task_id = "source-task"
    candidate.need_branch = True
    candidate.branch_sandbox_id = "sb-branch"
    workflow.tree_store = type("Store", (), {"trajectories": {"q": [candidate]}})()

    cleanup_called = []

    async def fake_prepare_branch_task(data, candidate_arg):
        return "branch-task-id"

    async def fake_retry_episode(engine, episode_data, group_idx):
        return {"branch": True}

    async def fake_cleanup(candidate_arg):
        cleanup_called.append(candidate_arg)

    workflow._prepare_branch_task = fake_prepare_branch_task
    workflow._retry_episode = fake_retry_episode
    workflow._cleanup_branch = fake_cleanup

    data = {"query_id": "q"}
    result = await workflow._run_fresh_episode(None, data, 0, "q")

    assert len(cleanup_called) == 1
    assert cleanup_called[0] is candidate

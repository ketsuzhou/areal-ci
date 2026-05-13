#!/usr/bin/env python3

import torch

from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    Node,
    _find_turn_boundaries,
)


class TestFindTurnBoundaries:
    def test_single_turn(self):
        starts, ends = _find_turn_boundaries([0, 0, 0, 1, 1])
        assert starts == [3]
        assert ends == [5]

    def test_multi_turn(self):
        starts, ends = _find_turn_boundaries([0, 0, 1, 1, 0, 0, 1, 1])
        assert starts == [2, 6]
        assert ends == [4, 8]

    def test_all_zeros(self):
        starts, ends = _find_turn_boundaries([0, 0, 0, 0])
        assert starts == []
        assert ends == []

    def test_all_ones(self):
        starts, ends = _find_turn_boundaries([1, 1, 1, 1])
        assert starts == [0]
        assert ends == [4]

    def test_empty(self):
        starts, ends = _find_turn_boundaries([])
        assert starts == []
        assert ends == []

    def test_response_at_end(self):
        starts, ends = _find_turn_boundaries([0, 0, 1, 1, 1])
        assert starts == [2]
        assert ends == [5]

    def test_three_turns(self):
        loss_mask = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        starts, ends = _find_turn_boundaries(loss_mask)
        assert starts == [2, 6, 10]
        assert ends == [4, 8, 12]


class TestNode:
    def test_creation(self):
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="turn_0",
            parent_node_id=None,
            episode_id="ep_1",
            outcome_reward=1.0,
        )
        assert len(node.input_ids) == 5
        assert node.outcome_reward == 1.0
        assert node.node_id == "turn_0"
        assert node.parent_node_id is None

    def test_new_fields_default_to_none(self):
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="t1",
            parent_node_id=None,
            episode_id="ep_1",
        )
        assert node.topk_ids is None
        assert node.topk_logp is None
        assert node.distill_reward is None
        assert node.teacher_logp is None

    def test_new_fields_can_be_set(self):
        topk_ids = [[10, 20], [30, 40], [50, 60]]
        topk_logp = [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]
        distill_reward = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        teacher_logp = [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]]

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="t1",
            parent_node_id=None,
            episode_id="ep_1",
            outcome_reward=1.0,
            topk_ids=topk_ids,
            topk_logp=topk_logp,
            distill_reward=distill_reward,
            teacher_logp=teacher_logp,
        )
        assert node.topk_ids == topk_ids
        assert node.topk_logp == topk_logp
        assert node.distill_reward == distill_reward
        assert node.teacher_logp == teacher_logp


def _make_traj(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    logprobs: list[float] | None = None,
    versions: list[int] | None = None,
    query_id: str | None = None,
    node_id: str = "",
) -> Node:
    seq_len = len(input_ids)
    # Convert lists to tensors as expected by Node
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.int32)
    loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.int32)
    logprobs_tensor = (
        torch.tensor(logprobs, dtype=torch.float32) if logprobs is not None else None
    )
    versions_tensor = (
        torch.tensor(versions, dtype=torch.int32) if versions is not None else None
    )

    node = Node(
        input_ids=input_ids_tensor,
        loss_mask=loss_mask_tensor,
        logprobs=logprobs_tensor,
        versions=versions_tensor,
        outcome_reward=reward,
        node_id=node_id,
    )

    if query_id is not None:
        node.query_id = query_id

    return node


class TestMCTSTreeStoreInsertBatch:
    def test_insert_list_dict_basic(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "node_id": "test-dict-basic",
        }
        store.insert_batch([traj])
        assert "node_id" in traj
        assert "query_id" in traj
        assert len(store.trajectories) == 1
        query_id = traj["query_id"]
        assert len(store.trajectories[query_id]) == 1
        node = store.trajectories[query_id][0]
        assert node["input_ids"] == [1, 2, 3, 4, 5]
        assert node["loss_mask"] == [0, 0, 1, 1, 1]

    def test_insert_list_dict_with_new_fields(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 0, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1],
            "node_id": "test-dict-fields",
            "topk_ids": [[10, 20], [30, 40], [50, 60]],
            "topk_logp": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
            "distill_reward": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "teacher_logp": [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]],
        }
        store.insert_batch([traj])
        query_id = traj["query_id"]
        node = store.trajectories[query_id][0]
        assert node["topk_ids"] == [[10, 20], [30, 40], [50, 60]]
        assert node["topk_logp"] == [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]
        assert node["distill_reward"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert node["teacher_logp"] == [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]]

    def test_insert_list_dict_with_query_id(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "query_id": "custom_query_id",
            "node_id": "test-dict-qid",
        }
        store.insert_batch([traj])
        assert traj["query_id"] == "custom_query_id"
        assert "custom_query_id" in store.trajectories

    def test_insert_single_trajectory(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1", node_id="t1"
        )
        store.insert_batch([traj])
        assert hasattr(traj, "node_id")
        assert traj.query_id == "q1"
        assert len(store.trajectories["q1"]) == 1

    def test_insert_two_trajectories_same_query(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="t1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1", node_id="t2")
        store.insert_batch([t1, t2])
        assert len(store.trajectories["q1"]) == 2
        assert t1.node_id != t2.node_id

    def test_insert_grouped_trajectory(self):
        """Grouped (batched) trajectory dicts are stored as-is and receive a
        node_id during insertion."""
        store = MCTSTreeStore()
        traj = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.int32),
            "rewards": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool
            ),
            "query_id": "q1",
            "node_id": "test-grouped",
        }
        store.insert_batch([traj])
        assert "node_id" in traj
        assert traj["node_id"] == "test-grouped"
        node = store.trajectories["q1"][0]
        assert torch.equal(
            node["input_ids"],
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int32),
        )

    def test_insert_skips_already_inserted(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        seq_id_1 = traj.node_id
        store.insert_batch([traj])
        assert traj.node_id == seq_id_1
        assert len(store.trajectories["q1"]) == 1

    def test_insert_stores_logprobs_and_versions(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 1, 1, 1],
            query_id="q1",
            node_id="t1",
        )
        store.insert_batch([traj])
        node = store.trajectories["q1"][0]
        assert len(node.logprobs) == 5
        for expected, actual in zip([-0.1, -0.2, -0.3, -0.4, -0.5], node.logprobs):
            assert abs(expected - actual) < 1e-6
        assert torch.equal(
            node.versions, torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
        )

    def test_insert_node_objects(self):
        """Insert a list of Node objects directly (new workflow pipeline)."""
        store = MCTSTreeStore()
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="turn_0",
            parent_node_id=None,
            episode_id="q1_0",
            outcome_reward=1.0,
        )
        node.query_id = "q1"
        store.insert_batch([node])
        assert hasattr(node, "node_id")
        assert len(store.trajectories) == 1
        assert len(store.trajectories["q1"]) == 1


class TestMCTSTreeStoreTrainId:
    def test_train_id_default_empty(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        assert traj.train_id == ""

    def test_set_trained_stamps_current_train_id(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        store.set_trained(traj.node_id, True)
        assert store.is_trained(traj.node_id) is True
        assert traj.train_id == "run_001"

    def test_is_trained_false_when_different_train_id(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_002"
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        traj.train_id = "run_001"  # old run
        assert store.is_trained(traj.node_id) is False

    def test_is_trained_false_when_empty_train_id(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        assert traj.train_id == ""
        assert store.is_trained(traj.node_id) is False

    def test_get_untrained_count_with_different_train_ids(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_002"
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="t1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1", node_id="t2")
        t3 = _make_traj([7, 8, 9], [0, 0, 1], reward=0.3, query_id="q1", node_id="t3")
        store.insert_batch([t1, t2, t3])
        t1.train_id = "run_002"  # trained in current run
        t2.train_id = "run_001"  # trained in old run
        # t3.train_id = "" (untrained)
        assert store.get_untrained_count("q1") == 2  # t2 and t3
        store.set_trained(t2.node_id, True)
        assert store.get_untrained_count("q1") == 1  # only t3

    def test_mark_episodes_trained(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        n3 = Node(
            input_ids=[7, 8, 9], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3], versions=[0, 0, 0],
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=0.3, query_id="q1",
        )
        store.insert_batch([n1, n2, n3])
        store.mark_episodes_trained({"ep_a"})
        assert store.is_trained(n1.node_id) is True
        assert n1.train_id == "run_001"
        assert store.is_trained(n2.node_id) is False
        assert n2.train_id == ""
        assert store.is_trained(n3.node_id) is True
        assert n3.train_id == "run_001"

    def test_mark_episodes_trained_resets_others(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)
        store.set_trained(n2.node_id, True)
        store.mark_episodes_trained({"ep_b"})
        assert store.is_trained(n1.node_id) is False
        assert n1.train_id == ""
        assert store.is_trained(n2.node_id) is True
        assert n2.train_id == "run_001"

    def test_mark_episodes_trained_empty_set(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        store.set_trained(n1.node_id, True)
        store.mark_episodes_trained(set())
        assert store.is_trained(n1.node_id) is False
        assert n1.train_id == ""

    def test_mark_episodes_trained_unknown_episode(self):
        """Episode IDs not in the store are silently ignored."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        store.mark_episodes_trained({"nonexistent_episode"})
        assert store.is_trained(n1.node_id) is False
        assert n1.train_id == ""

    def test_get_reward(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="t1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1", node_id="t2")
        store.insert_batch([t1, t2])
        assert store.get_reward(t1.node_id) == 1.0
        assert store.get_reward(t2.node_id) == 0.5


class TestMCTSTreeStoreLoadTrajectories:
    def test_load_trajectories_returns_nodes(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "versions": [0, 0, 0, 0, 0],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "node_id": "test-load-dict",
            "topk_ids": [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            "topk_logp": [
                [-0.1, -0.2],
                [-0.3, -0.4],
                [-0.5, -0.6],
                [-0.7, -0.8],
                [-0.9, -1.0],
            ],
            "distill_reward": [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
            ],
            "teacher_logp": [
                [-1.1, -1.2],
                [-1.3, -1.4],
                [-1.5, -1.6],
                [-1.7, -1.8],
                [-1.9, -2.0],
            ],
        }
        store.insert_batch([traj])
        loaded = store.load_trajectories(traj["query_id"], n_samples=1)
        assert len(loaded) == 1
        node = loaded[0]
        assert isinstance(node, dict)
        assert node["input_ids"] == [1, 2, 3, 4, 5]
        assert node["reward"] == 1.0
        assert node["topk_ids"] == [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        assert node["teacher_logp"] == [
            [-1.1, -1.2],
            [-1.3, -1.4],
            [-1.5, -1.6],
            [-1.7, -1.8],
            [-1.9, -2.0],
        ]
        assert "query_id" in node
        assert "node_id" in node

    def test_load_trajectories_basic(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=1.0, query_id="q1", node_id="t1"
        )
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert len(loaded) == 1
        node = loaded[0]
        assert isinstance(node, Node)
        assert len(node.input_ids) == 5
        assert hasattr(node, "query_id")

    def test_load_trajectories_only_untrained(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="t1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1", node_id="t2")
        store.insert_batch([t1, t2])
        store.set_trained(t1.node_id, True)
        loaded = store.load_trajectories("q1", n_samples=2)
        assert len(loaded) == 1
        assert loaded[0].outcome_reward == 0.5

    def test_load_trajectories_unknown_query(self):
        store = MCTSTreeStore()
        assert store.load_trajectories("nonexistent", n_samples=1) == []


class TestMCTSTreeStoreClear:
    def test_clear(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1", node_id="t1")
        store.insert_batch([traj])
        store.clear()
        assert len(store.trajectories) == 0
        assert len(store._node_id_to_key) == 0
        assert len(store._query_node_ids) == 0
        assert len(store._visit_counts) == 0
        assert len(store._q_values) == 0


class TestMCTSTreeStoreMCTSStats:
    def test_backup_updates_stats(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], reward=2.0, query_id="q1", node_id="t1")
        store.insert_batch([traj])
        seq_id = traj.node_id
        assert store._visit_counts[seq_id] == 1
        assert store._q_values[seq_id] == 2.0

    def test_two_trajectories_separate_q_values(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="t1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1", node_id="t2")
        store.insert_batch([t1, t2])
        assert store._q_values[t1.node_id] == 1.0
        assert store._q_values[t2.node_id] == 0.0


class TestGetUntrainedEpisodeCount:
    def test_no_episodes(self):
        store = MCTSTreeStore()
        assert store.get_untrained_episode_count("q1") == 0

    def test_unknown_query(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        assert store.get_untrained_episode_count("q_other") == 0

    def test_all_untrained(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2])
        assert store.get_untrained_episode_count("q1") == 2

    def test_multi_node_episodes(self):
        """An episode with multiple nodes counts as one episode."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n3 = Node(
            input_ids=[7, 8, 9], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2, n3])
        assert store.get_untrained_episode_count("q1") == 2

    def test_trained_episode_excluded(self):
        """An episode where all nodes are trained is excluded from count."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)
        assert store.get_untrained_episode_count("q1") == 1

    def test_partially_trained_episode_still_counted(self):
        """An episode is untrained if any of its nodes is untrained."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)
        assert store.get_untrained_episode_count("q1") == 1


class TestLoadUntrainedEpisodes:
    def test_no_episodes(self):
        store = MCTSTreeStore()
        assert store.load_untrained_episodes("q1", n_episodes=1) == []

    def test_unknown_query(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1])
        assert store.load_untrained_episodes("q_other", n_episodes=1) == []

    def test_returns_all_nodes_from_episode(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_a_2", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        store.insert_batch([n1, n2])
        loaded = store.load_untrained_episodes("q1", n_episodes=1)
        assert len(loaded) == 2
        assert loaded[0].node_id == "ep_a_1"
        assert loaded[1].node_id == "ep_a_2"

    def test_respects_n_episodes_limit(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        n3 = Node(
            input_ids=[7, 8, 9], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.3], versions=[0, 0, 0],
            node_id="ep_c_1", episode_id="ep_c", outcome_reward=0.3, query_id="q1",
        )
        store.insert_batch([n1, n2, n3])
        loaded = store.load_untrained_episodes("q1", n_episodes=2)
        # 2 episodes, each with 1 node = 2 nodes total
        assert len(loaded) == 2
        episode_ids = {n.episode_id for n in loaded}
        assert len(episode_ids) == 2

    def test_skips_trained_episodes(self):
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)
        loaded = store.load_untrained_episodes("q1", n_episodes=2)
        assert len(loaded) == 1
        assert loaded[0].node_id == "ep_b_1"

    def test_preserves_episode_order(self):
        """Episodes are returned in insertion order."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1], versions=[0, 0, 0],
            node_id="ep_a_1", episode_id="ep_a", outcome_reward=1.0, query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6], loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2], versions=[0, 0, 0],
            node_id="ep_b_1", episode_id="ep_b", outcome_reward=0.5, query_id="q1",
        )
        store.insert_batch([n1, n2])
        loaded = store.load_untrained_episodes("q1", n_episodes=2)
        assert loaded[0].episode_id == "ep_a"
        assert loaded[1].episode_id == "ep_b"


class TestNodeToTensorDict:
    def test_response_only_fields_sliced(self):
        from customized_areal.tree_search.mcts_tree_store import (
            Node,
            _node_to_tensor_dict,
        )

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[0.0, 0.0, -0.3, -0.4, -0.5],
            versions=[-1, -1, 1, 1, 1],
            outcome_reward=1.0,
            query_id="q1",
            node_id="t1",
            topk_ids=[[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            topk_logp=[
                [-1.0, -2.0],
                [-3.0, -4.0],
                [-5.0, -6.0],
                [-7.0, -8.0],
                [-9.0, -10.0],
            ],
            distill_reward=[[0.1], [0.2], [0.3], [0.4], [0.5]],
            teacher_logp=[[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]],
        )
        result = _node_to_tensor_dict(node, "q1", "t1")
        # Response portion is positions 2:5 (loss_mask==1)
        assert result["topk_ids"].shape == (1, 3, 2)  # [1, resp_len, topk]
        assert result["topk_logp"].shape == (1, 3, 2)
        assert result["distill_reward"].shape == (1, 3, 1)
        assert result["teacher_logp"].shape == (1, 3, 1)
        # Metadata fields are single-element lists
        assert result["query_id"] == ["q1"]
        assert result["node_id"] == ["t1"]
        assert result["episode_id"] == [""]  # default: node.episode_id is empty
        assert result["turn_idx"] == [0]  # default: node.turn_idx is 0

    def test_logp_already_sliced(self):
        """logp is already correctly sliced — verify it stays that way."""
        from customized_areal.tree_search.mcts_tree_store import (
            Node,
            _node_to_tensor_dict,
        )

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[0.0, 0.0, -0.3, -0.4, -0.5],
            versions=[-1, -1, 1, 1, 1],
            outcome_reward=1.0,
            query_id="q1",
            node_id="t1",
        )
        result = _node_to_tensor_dict(node, "q1", "t1")
        assert result["logp"].shape == (1, 3)
        assert result["query_id"] == ["q1"]
        assert result["node_id"] == ["t1"]

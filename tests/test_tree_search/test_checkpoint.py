import itertools

from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    Node,
    _find_turn_boundaries,
)

_node_id_counter = itertools.count(1)


def _make_node(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    query_id: str = "q1",
    node_id: str | None = None,
) -> Node:
    if node_id is None:
        node_id = f"n{next(_node_id_counter)}"
    return Node(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=[0.0] * len(input_ids),
        versions=[-1] * len(input_ids),
        outcome_reward=reward,
        query_id=query_id,
        node_id=node_id,
    )


def _make_store_with_data() -> MCTSTreeStore:
    """Create a store with sample data for checkpoint tests."""
    store = MCTSTreeStore()
    t1 = _make_node([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
    t2 = _make_node([6, 7, 8], [0, 0, 1], reward=0.5, query_id="q2")
    store.insert_batch([t1, t2])
    return store


class TestTreeCheckpointManager:
    def test_save_and_load(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)
        assert manager.exists()

        loaded = manager.load()
        assert len(loaded.trajectories) == 2
        assert "q1" in loaded.trajectories
        assert "q2" in loaded.trajectories

    def test_exists_false_when_no_dir(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path / "nonexistent"))
        assert not manager.exists()

    def test_save_creates_directory(self, tmp_path):
        save_dir = str(tmp_path / "new_dir")
        manager = TreeCheckpointManager(save_dir)
        store = _make_store_with_data()
        manager.save(store)
        import os

        assert os.path.isdir(os.path.join(save_dir, "mcts_trees"))

    def test_load_preserves_node_id_counter(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        t3 = _make_node([9, 10], [0, 1], reward=1.0, query_id="q3")
        loaded.insert_batch([t3])
        # Verify the node was inserted with its provider-assigned node_id
        assert t3.node_id in loaded._node_id_to_key
        assert "q3" in loaded.trajectories

    def test_load_preserves_trajectory_data(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        record_q1 = loaded.trajectories["q1"][0]
        assert record_q1.input_ids == [1, 2, 3, 4, 5]
        assert record_q1.loss_mask == [0, 0, 1, 1, 1]
        assert record_q1.outcome_reward == 2.0
        record_q2 = loaded.trajectories["q2"][0]
        assert record_q2.input_ids == [6, 7, 8]
        assert record_q2.outcome_reward == 0.5

    def test_load_preserves_mcts_stats(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        node_ids = loaded._query_node_ids["q1"]
        assert loaded._q_values[node_ids[0]] == 2.0
        node_ids = loaded._query_node_ids["q2"]
        assert loaded._q_values[node_ids[0]] == 0.5

    def test_load_preserves_train_id(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        store.current_train_id = "run_001"
        node_ids = store._query_node_ids["q1"]
        store.set_trained(node_ids[0], True)
        manager.save(store)

        loaded = manager.load()
        assert loaded.current_train_id == "run_001"
        assert loaded.is_trained(node_ids[0]) is True

    def test_load_preserves_turn_boundaries(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore()
        node = _make_node(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
            query_id="q1",
        )
        store.insert_batch([node])
        manager.save(store)

        loaded = manager.load()
        record = loaded.trajectories["q1"][0]
        starts, ends = _find_turn_boundaries(record.loss_mask)
        assert starts == [2, 6]
        assert ends == [4, 8]

    def test_save_and_load_distill_fields(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore()
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[-1, -1, 1, 1, 1],
            node_id="distill_node",
            outcome_reward=1.0,
            query_id="q1",
            topk_ids=[[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            topk_logp=[
                [-0.1, -0.2],
                [-0.3, -0.4],
                [-0.5, -0.6],
                [-0.7, -0.8],
                [-0.9, -1.0],
            ],
            distill_reward=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
            teacher_logp=[
                [-1.1, -1.2],
                [-1.3, -1.4],
                [-1.5, -1.6],
                [-1.7, -1.8],
                [-1.9, -2.0],
            ],
        )
        store.insert_batch([node])
        manager.save(store)

        loaded = manager.load()
        record = loaded.trajectories["q1"][0]
        assert record.logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]
        assert record.topk_ids == [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        assert record.topk_logp == [
            [-0.1, -0.2],
            [-0.3, -0.4],
            [-0.5, -0.6],
            [-0.7, -0.8],
            [-0.9, -1.0],
        ]
        assert record.distill_reward == [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
        ]
        assert record.teacher_logp == [
            [-1.1, -1.2],
            [-1.3, -1.4],
            [-1.5, -1.6],
            [-1.7, -1.8],
            [-1.9, -2.0],
        ]

    def test_uuid_query_id_roundtrip(self, tmp_path):
        """UUID query_ids (the real-world format) survive save/load exactly."""
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore()
        uuid_qid = "a7143270374d4eeab8ad5f4716475e28"
        node = _make_node([1, 2, 3], [0, 0, 1], reward=1.0, query_id=uuid_qid)
        store.insert_batch([node])
        manager.save(store)

        loaded = manager.load()
        assert uuid_qid in loaded.trajectories
        assert loaded.trajectories[uuid_qid][0].outcome_reward == 1.0

    def test_atomic_save_no_partial_files(self, tmp_path):
        """Verify no .tmp files remain after successful save."""
        import os

        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        mcts_dir = os.path.join(str(tmp_path), "mcts_trees")
        tmp_files = [f for f in os.listdir(mcts_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_save_and_load_trained_episodes(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        store.current_train_id = "run_001"
        node_ids = store._query_node_ids["q1"]
        store.set_trained(node_ids[0], True)

        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded is not None
        # The node for q1 has episode_id="" (default), so that's what was saved
        assert "" in loaded

    def test_load_trained_episodes_missing_file(self, tmp_path):
        recover_dir = str(tmp_path / "nonexistent")
        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded is None

    def test_load_trained_episodes_corrupt_file(self, tmp_path):
        import os

        recover_dir = str(tmp_path / "recover_checkpoint")
        os.makedirs(recover_dir, exist_ok=True)
        filepath = os.path.join(recover_dir, "trained_episodes.json")
        with open(filepath, "w") as f:
            f.write("{invalid json")

        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded is None

    def test_save_trained_episodes_atomic(self, tmp_path):
        """Verify no .tmp files remain after successful save."""
        import os

        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        store.current_train_id = "run_001"

        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        tmp_files = [f for f in os.listdir(recover_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_save_trained_episodes_with_episode_ids(self, tmp_path):
        """Nodes with explicit episode_id should be tracked correctly."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[0, 0, 0],
            node_id="ep_alpha_node",
            episode_id="ep_alpha",
            outcome_reward=1.0,
            query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            node_id="ep_beta_node",
            episode_id="ep_beta",
            outcome_reward=0.5,
            query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)

        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded is not None
        assert "ep_alpha" in loaded
        assert "ep_beta" not in loaded


class TestTrainedEpisodesRestoreIntegration:
    def test_save_restore_cycle(self, tmp_path):
        """Full save -> load -> mark_episodes_trained cycle."""
        store = MCTSTreeStore()
        store.current_train_id = "run_001"
        n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[0, 0, 0],
            node_id="n1",
            episode_id="ep_1",
            outcome_reward=1.0,
            query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            node_id="n2",
            episode_id="ep_2",
            outcome_reward=0.5,
            query_id="q2",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)

        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        fresh_store = MCTSTreeStore()
        fresh_store.current_train_id = "run_001"
        fresh_n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[0, 0, 0],
            node_id="fn1",
            episode_id="ep_1",
            outcome_reward=1.0,
            query_id="q1",
        )
        fresh_n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            node_id="fn2",
            episode_id="ep_2",
            outcome_reward=0.5,
            query_id="q2",
        )
        fresh_store.insert_batch([fresh_n1, fresh_n2])

        trained_episodes = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert trained_episodes is not None
        fresh_store.mark_episodes_trained(trained_episodes)

        assert fresh_store.is_trained(fresh_n1.node_id) is True
        assert fresh_store.is_trained(fresh_n2.node_id) is False

    def test_no_sidecar_falls_back_to_reset(self, tmp_path):
        """When no trained_episodes.json exists, None is returned — caller should reset_trained_flags."""
        recover_dir = str(tmp_path / "nonexistent")
        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded is None

    def test_untrained_nodes_not_in_saved_episodes(self, tmp_path):
        """Only trained nodes' episode_ids appear in the saved file."""
        store = MCTSTreeStore()
        n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[0, 0, 0],
            node_id="n_trained",
            episode_id="ep_trained",
            outcome_reward=1.0,
            query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            node_id="n_untrained",
            episode_id="ep_untrained",
            outcome_reward=0.5,
            query_id="q1",
        )
        store.insert_batch([n1, n2])
        store.current_train_id = "run_001"
        # n1 is trained, n2 is not
        store.set_trained(n1.node_id, True)

        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded == {"ep_trained"}

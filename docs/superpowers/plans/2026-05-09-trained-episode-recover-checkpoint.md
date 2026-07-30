# Trained Episode Recover Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist trained episode IDs in the recover checkpoint so that resuming
training does not retrain already-trained nodes.

**Architecture:** Add `mark_episodes_trained()` to `MCTSTreeStore` for episode-level
trained flag restoration. Add `save_trained_episodes()` / `load_trained_episodes()`
static methods to `TreeCheckpointManager` for the `trained_episodes.json` sidecar file.
Modify `CacheAwarePPOTrainer._init_tree_components()` to restore from the sidecar
instead of calling `reset_trained_flags()`, and `_save_recover_checkpoint()` to write
it.

**Tech Stack:** Python 3.12+ | pytest | no new dependencies

______________________________________________________________________

### Task 1: Add `mark_episodes_trained` to `MCTSTreeStore`

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:319-321`

- Test: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_mcts_tree_store.py`, inside
`TestMCTSTreeStoreTrainedFlag`:

```python
def test_mark_episodes_trained(self):
    store = MCTSTreeStore()
    n1 = Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1],
        versions=[0, 0, 0],
        episode_id="ep_a",
        outcome_reward=1.0,
        query_id="q1",
    )
    n2 = Node(
        input_ids=[4, 5, 6],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.2],
        versions=[0, 0, 0],
        episode_id="ep_b",
        outcome_reward=0.5,
        query_id="q1",
    )
    n3 = Node(
        input_ids=[7, 8, 9],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.3],
        versions=[0, 0, 0],
        episode_id="ep_a",
        outcome_reward=0.3,
        query_id="q1",
    )
    store.insert_batch([n1, n2, n3])

    # Mark ep_a as trained, ep_b as untrained
    store.mark_episodes_trained({"ep_a"})

    assert store.is_trained(n1.node_id) is True
    assert store.is_trained(n2.node_id) is False
    assert store.is_trained(n3.node_id) is True

def test_mark_episodes_trained_resets_others(self):
    store = MCTSTreeStore()
    n1 = Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1],
        versions=[0, 0, 0],
        episode_id="ep_a",
        outcome_reward=1.0,
        query_id="q1",
    )
    n2 = Node(
        input_ids=[4, 5, 6],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.2],
        versions=[0, 0, 0],
        episode_id="ep_b",
        outcome_reward=0.5,
        query_id="q1",
    )
    store.insert_batch([n1, n2])
    store.set_trained(n1.node_id, True)
    store.set_trained(n2.node_id, True)

    # Only ep_b is in the set — ep_a should be reset to False
    store.mark_episodes_trained({"ep_b"})

    assert store.is_trained(n1.node_id) is False
    assert store.is_trained(n2.node_id) is True

def test_mark_episodes_trained_empty_set(self):
    store = MCTSTreeStore()
    n1 = Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1],
        versions=[0, 0, 0],
        episode_id="ep_a",
        outcome_reward=1.0,
        query_id="q1",
    )
    store.insert_batch([n1])
    store.set_trained(n1.node_id, True)

    # Empty set means nothing is trained
    store.mark_episodes_trained(set())

    assert store.is_trained(n1.node_id) is False

def test_mark_episodes_trained_unknown_episode(self):
    """Episode IDs not in the store are silently ignored."""
    store = MCTSTreeStore()
    n1 = Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1],
        versions=[0, 0, 0],
        episode_id="ep_a",
        outcome_reward=1.0,
        query_id="q1",
    )
    store.insert_batch([n1])

    store.mark_episodes_trained({"nonexistent_episode"})

    assert store.is_trained(n1.node_id) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreTrainedFlag::test_mark_episodes_trained -v`
Expected: FAIL with
`AttributeError: 'MCTSTreeStore' object has no attribute 'mark_episodes_trained'`

- [ ] **Step 3: Write the implementation**

Add to `customized_areal/tree_search/mcts_tree_store.py` after `reset_trained_flags`
(line 321):

```python
def mark_episodes_trained(self, episode_ids: set[str]) -> None:
    """Set trained flags based on episode IDs.

    Nodes whose episode_id is in the given set are marked trained.
    All other nodes are marked untrained. Episode IDs not present
    in the store are silently ignored.
    """
    for node_id in self._trained:
        self._trained[node_id] = False
    for query_id, records in self.trajectories.items():
        for node in records:
            if node.episode_id in episode_ids:
                self._trained[node.node_id] = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestMCTSTreeStoreTrainedFlag -v`
Expected: All 7 tests PASS (3 existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat: add mark_episodes_trained to MCTSTreeStore for episode-level trained flag restoration"
```

______________________________________________________________________

### Task 2: Add `save_trained_episodes` / `load_trained_episodes` to `TreeCheckpointManager`

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:22-171`

- Test: `tests/test_tree_search/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_checkpoint.py`, inside `TestTreeCheckpointManager`:

```python
def test_save_and_load_trained_episodes(self, tmp_path):
    manager = TreeCheckpointManager(str(tmp_path))
    store = _make_store_with_data()
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

    recover_dir = str(tmp_path / "recover_checkpoint")
    TreeCheckpointManager.save_trained_episodes(recover_dir, store)

    tmp_files = [f for f in os.listdir(recover_dir) if f.endswith(".tmp")]
    assert len(tmp_files) == 0

def test_save_trained_episodes_with_episode_ids(self, tmp_path):
    """Nodes with explicit episode_id should be tracked correctly."""
    store = MCTSTreeStore()
    n1 = Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.1],
        versions=[0, 0, 0],
        episode_id="ep_alpha",
        outcome_reward=1.0,
        query_id="q1",
    )
    n2 = Node(
        input_ids=[4, 5, 6],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.2],
        versions=[0, 0, 0],
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`uv run pytest tests/test_tree_search/test_checkpoint.py::TestTreeCheckpointManager::test_save_and_load_trained_episodes -v`
Expected: FAIL with
`AttributeError: type object 'TreeCheckpointManager' has no attribute 'save_trained_episodes'`

- [ ] **Step 3: Write the implementation**

Add two static methods to `TreeCheckpointManager` in
`customized_areal/tree_search/checkpoint.py`, after the `_deserialize_record` method
(after line 170):

```python
@staticmethod
def save_trained_episodes(
    recover_checkpoint_dir: str, tree_store: MCTSTreeStore
) -> None:
    """Save trained episode IDs to the recover checkpoint directory."""
    trained_ids: set[str] = set()
    for query_id, records in tree_store.trajectories.items():
        for node in records:
            if tree_store.is_trained(node.node_id):
                trained_ids.add(node.episode_id)
    data = {"trained_episode_ids": sorted(trained_ids)}
    os.makedirs(recover_checkpoint_dir, exist_ok=True)
    filepath = os.path.join(recover_checkpoint_dir, "trained_episodes.json")
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)

@staticmethod
def load_trained_episodes(
    recover_checkpoint_dir: str,
) -> set[str] | None:
    """Load trained episode IDs from the recover checkpoint directory.

    Returns the set of trained episode IDs, or None if the file does
    not exist or is corrupt.
    """
    filepath = os.path.join(recover_checkpoint_dir, "trained_episodes.json")
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return set(data["trained_episode_ids"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tree_search/test_checkpoint.py -v` Expected: All tests
PASS (existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py tests/test_tree_search/test_checkpoint.py
git commit -m "feat: add save/load_trained_episodes to TreeCheckpointManager for recover checkpoint sidecar"
```

______________________________________________________________________

### Task 3: Modify `_init_tree_components` to restore trained flags from recover checkpoint

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:202-221`

- Test: `tests/test_tree_search/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_checkpoint.py` (top-level, not inside the class), to
test the integration between `TreeCheckpointManager.load_trained_episodes` and
`MCTSTreeStore.mark_episodes_trained`:

```python
class TestTrainedEpisodesRestoreIntegration:
    def test_save_restore_cycle(self, tmp_path):
        """Full save → load → mark_episodes_trained cycle."""
        store = MCTSTreeStore()
        n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[0, 0, 0],
            episode_id="ep_1",
            outcome_reward=1.0,
            query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            episode_id="ep_2",
            outcome_reward=0.5,
            query_id="q2",
        )
        store.insert_batch([n1, n2])
        store.set_trained(n1.node_id, True)

        # Save trained episodes
        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        # Simulate a fresh store (as if loaded from tree checkpoint)
        fresh_store = MCTSTreeStore()
        fresh_n1 = Node(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.1],
            versions=[0, 0, 0],
            episode_id="ep_1",
            outcome_reward=1.0,
            query_id="q1",
        )
        fresh_n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            episode_id="ep_2",
            outcome_reward=0.5,
            query_id="q2",
        )
        fresh_store.insert_batch([fresh_n1, fresh_n2])

        # Restore trained flags from sidecar
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
            episode_id="ep_trained",
            outcome_reward=1.0,
            query_id="q1",
        )
        n2 = Node(
            input_ids=[4, 5, 6],
            loss_mask=[0, 0, 1],
            logprobs=[0.0, 0.0, -0.2],
            versions=[0, 0, 0],
            episode_id="ep_untrained",
            outcome_reward=0.5,
            query_id="q1",
        )
        store.insert_batch([n1, n2])
        # n1 is trained, n2 is not
        store.set_trained(n1.node_id, True)

        recover_dir = str(tmp_path / "recover_checkpoint")
        TreeCheckpointManager.save_trained_episodes(recover_dir, store)

        loaded = TreeCheckpointManager.load_trained_episodes(recover_dir)
        assert loaded == {"ep_trained"}
```

- [ ] **Step 2: Run tests to verify they pass**

Run:
`uv run pytest tests/test_tree_search/test_checkpoint.py::TestTrainedEpisodesRestoreIntegration -v`
Expected: PASS — these tests only use `TreeCheckpointManager` and `MCTSTreeStore`, which
are already implemented from Tasks 1 and 2.

- [ ] **Step 3: Modify `_init_tree_components` in `trainer.py`**

Replace lines 216-217 in `customized_areal/tree_search/trainer.py`:

```python
        # Reset trained flags for a fresh training run
        self.tree_store.reset_trained_flags()
```

with:

```python
        # Restore trained flags from recover checkpoint, or reset for fresh run
        from areal.utils.saver import Saver

        recover_dir = Saver.get_recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.cluster.fileroot,
        )
        trained_episodes = TreeCheckpointManager.load_trained_episodes(recover_dir)
        if trained_episodes is not None:
            self.tree_store.mark_episodes_trained(trained_episodes)
            logger.info(
                f"Restored trained flags for {len(trained_episodes)} episodes "
                f"from recover checkpoint"
            )
        else:
            self.tree_store.reset_trained_flags()
```

- [ ] **Step 4: Run existing trainer tests to verify no regressions**

Run: `uv run pytest tests/test_tree_search/ -v` Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/trainer.py tests/test_tree_search/test_checkpoint.py
git commit -m "feat: restore trained flags from recover checkpoint on resume instead of resetting all"
```

______________________________________________________________________

### Task 4: Modify `_save_recover_checkpoint` to persist trained episodes

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:223-234`

- [ ] **Step 1: Modify `_save_recover_checkpoint`**

Replace the method at `customized_areal/tree_search/trainer.py:223-234`:

```python
    def _save_recover_checkpoint(
        self, epoch: int, epoch_step: int, global_step: int
    ) -> None:
        """Save recover checkpoint including MCTS tree state."""
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)

        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode == CacheMode.CROSS_TRAINING
        ):
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.info("Saved MCTS tree checkpoint with rollout cache")

            # Save trained episode IDs to recover checkpoint directory
            from areal.utils.saver import Saver

            recover_dir = Saver.get_recover_checkpoint_path(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.cluster.fileroot,
            )
            TreeCheckpointManager.save_trained_episodes(recover_dir, self.tree_store)
```

- [ ] **Step 2: Move the `Saver` import to the top of the file**

The `from areal.utils.saver import Saver` import now appears in both
`_init_tree_components` and `_save_recover_checkpoint`. Move it to the file-level
imports at `customized_areal/tree_search/trainer.py` (after the existing `from areal`
imports around line 38-40):

```python
from areal.utils.saver import Saver
```

Then remove the inline `from areal.utils.saver import Saver` from both
`_init_tree_components` and `_save_recover_checkpoint`.

- [ ] **Step 3: Run all tree_search tests to verify no regressions**

Run: `uv run pytest tests/test_tree_search/ -v` Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat: save trained episode IDs alongside recover checkpoint"
```

______________________________________________________________________

## Self-Review

**1. Spec coverage:**

- "Track trained episode IDs in `trained_episodes.json`" → Task 2 (save/load) + Task 4
  (save in \_save_recover_checkpoint)
- "On resume, restore from sidecar instead of reset_trained_flags" → Task 3
  (\_init_tree_components)
- "`mark_episodes_trained` on MCTSTreeStore" → Task 1
- "Error handling: missing/corrupt file → fallback to reset_trained_flags" → Task 2
  (load_trained_episodes returns None) + Task 3 (falls back to reset_trained_flags)
- "Atomic writes" → Task 2 (uses .tmp + os.replace)
- "Recover checkpoint is single source of truth" → Task 3 (ignores tree checkpoint
  trained flags)
- No changes to core `areal/` → Verified

**2. Placeholder scan:** No TBD, TODO, or "implement later" patterns. All code blocks
contain complete implementations.

**3. Type consistency:**

- `mark_episodes_trained(episode_ids: set[str])` — defined in Task 1, called with
  `set[str]` from `load_trained_episodes()` in Task 3. Consistent.
- `save_trained_episodes(recover_checkpoint_dir: str, tree_store: MCTSTreeStore)` —
  defined in Task 2, called in Task 4. Consistent.
- `load_trained_episodes(recover_checkpoint_dir: str) -> set[str] | None` — defined in
  Task 2, called in Task 3. Consistent.
- `Saver.get_recover_checkpoint_path(experiment_name, trial_name, fileroot)` — matches
  the existing API from `areal/utils/saver.py:79-90`.

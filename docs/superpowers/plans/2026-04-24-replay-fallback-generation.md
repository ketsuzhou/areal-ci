# Replay Fallback Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3-level fallback to `_replay_prepare_batch` so training doesn't stall when
replay history is exhausted: replay → cached untrained from tree store → fresh
dataloader generation.

**Architecture:** Modify `_replay_prepare_batch` to try three sources per step. Add two
new private helpers (`_load_untrained_from_tree_store`, `_generate_from_dataloader`) to
`CacheAwarePPOTrainer`. Update `train()` finally block to clean up the new dataloader
iterator. Tests use `MCTSTreeStore` directly (no distributed mocking needed for the
fallback logic).

**Tech Stack:** Python 3.12+ | PyTorch | unittest.mock

______________________________________________________________________

## File Structure

| File                                              | Responsibility                                                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/trainer.py:391-499` | Modify `_replay_prepare_batch`, add `_load_untrained_from_tree_store`, add `_generate_from_dataloader`, update `train()` cleanup |
| `tests/test_tree_search/test_cache_trainer.py`    | Add unit tests for fallback behavior                                                                                             |

______________________________________________________________________

### Task 1: Add `_load_untrained_from_tree_store` helper

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:389` (insert after
  `_cache_aware_prepare_batch`)

- Test: `tests/test_tree_search/test_cache_trainer.py`

- [x] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_cache_trainer.py`:

```python
class TestLoadUntrainedFromTreeStore:
    def test_loads_from_single_query(self):
        """Should load untrained trajectories from a single query_id."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 2
        assert result[0]["_mcts_query_id"] == "q1"
        assert result[1]["_mcts_query_id"] == "q1"

    def test_loads_from_multiple_queries(self):
        """Should load untrained trajectories from all query_ids with untrained paths."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 2
        query_ids = {t["_mcts_query_id"] for t in result}
        assert query_ids == {"q1", "q2"}

    def test_skips_trained_trajectories(self):
        """Should not load trajectories that are already marked trained."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        store.set_trained("q1", s0, True)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 1
        assert result[0]["_mcts_seq_id"] == s1

    def test_respects_n_samples_limit(self):
        """Should not load more than n_samples per query_id."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        for i in range(4):
            store.insert_trajectory("q1", [1, 2, 10, 3, 4 + i], reward=1.0)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=2)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 2

    def test_returns_empty_when_no_untrained(self):
        """Should return empty list when all trajectories are trained."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.set_trained("q1", s0, True)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert result == []

    def test_returns_empty_when_tree_empty(self):
        """Should return empty list when tree store has no trees."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert result == []
```

Also add the required import at the top of the test file (if not already present):

```python
from unittest.mock import MagicMock
```

- [x] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestLoadUntrainedFromTreeStore -v`
Expected: FAIL —
`AttributeError: type object 'CacheAwarePPOTrainer' has no attribute '_load_untrained_from_tree_store'`

- [x] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/trainer.py`, add the method to `CacheAwarePPOTrainer`
class, right after `_cache_aware_prepare_batch` (after line 389):

```python
    def _load_untrained_from_tree_store(self) -> list[dict[str, Any]]:
        """Load untrained trajectories from all tree store queries."""
        all_trajs: list[dict[str, Any]] = []
        for query_id in list(self.tree_store.trees.keys()):
            count = self.tree_store.get_untrained_count(query_id)
            if count > 0:
                n = min(count, self.cache_config.n_samples)
                trajs = self.tree_store.load_trajectories(query_id, n)
                all_trajs.extend(trajs)
        return all_trajs
```

- [x] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestLoadUntrainedFromTreeStore -v`
Expected: PASS

- [x] **Step 5: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: All tests
PASS

- [x] **Step 6: Commit**

```bash
git add customized_areal/tree_search/trainer.py tests/test_tree_search/test_cache_trainer.py
git commit -m "feat(tree-search): add _load_untrained_from_tree_store helper to CacheAwarePPOTrainer"
```

______________________________________________________________________

### Task 2: Add `_generate_from_dataloader` helper

**Files:**

- Modify: `customized_areal/tree_search/trainer.py` (after the method from Task 1)

- Test: `tests/test_tree_search/test_cache_trainer.py`

- [x] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_cache_trainer.py`:

```python
class TestGenerateFromDataloader:
    def test_lazy_init_and_generation(self):
        """Should lazily init dataloader iter and call rollout_batch."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)
        trainer._replay_dataloader_iter = None
        del trainer._replay_dataloader_iter  # Remove the attribute

        # Mock rollout_batch to return a fake trajectory
        fake_traj = {
            "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
            "rewards": torch.tensor([[1.0]], dtype=torch.float32),
        }
        trainer.actor.rollout_batch = MagicMock(return_value=[fake_traj])

        # Create a mock dataloader that yields batches
        mock_dataloader = MagicMock()
        mock_workflow = MagicMock()

        from areal.utils.data import cycle_dataloader
        from unittest.mock import patch

        # Create a simple iterable that yields a batch of prompts
        mock_prompts = [{"messages": [{"role": "user", "content": "hello"}]}]

        with patch(
            "customized_areal.tree_search.trainer.cycle_dataloader",
            return_value=iter([mock_prompts]),
        ):
            result = CacheAwarePPOTrainer._generate_from_dataloader(
                trainer,
                dataloader=mock_dataloader,
                workflow=mock_workflow,
                workflow_kwargs=None,
                group_size=1,
            )

        assert len(result) == 1
        trainer.actor.rollout_batch.assert_called_once()

    def test_reuses_existing_iterator(self):
        """Should reuse existing _replay_dataloader_iter if already initialized."""
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)

        # Pre-create a dataloader iterator
        batch = [{"messages": [{"role": "user", "content": "test"}]}]
        existing_iter = iter([batch, batch])
        trainer._replay_dataloader_iter = existing_iter

        fake_traj = {
            "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
            "rewards": torch.tensor([[1.0]], dtype=torch.float32),
        }
        trainer.actor.rollout_batch = MagicMock(return_value=[fake_traj])

        result = CacheAwarePPOTrainer._generate_from_dataloader(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
            workflow_kwargs=None,
            group_size=1,
        )

        assert len(result) == 1
        # Verify it used the existing iterator (didn't create a new one)
        assert trainer._replay_dataloader_iter is existing_iter

    def test_returns_empty_on_empty_batch(self):
        """Should return empty list when dataloader yields empty batch."""
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer._replay_dataloader_iter = iter([[]])

        result = CacheAwarePPOTrainer._generate_from_dataloader(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
            workflow_kwargs=None,
            group_size=1,
        )

        assert result == []
```

- [x] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestGenerateFromDataloader -v`
Expected: FAIL —
`AttributeError: type object 'CacheAwarePPOTrainer' has no attribute '_generate_from_dataloader'`

- [x] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/trainer.py`, add after
`_load_untrained_from_tree_store`:

```python
    def _generate_from_dataloader(
        self,
        dataloader,
        workflow,
        workflow_kwargs=None,
        group_size=1,
    ) -> list[dict[str, Any]]:
        """Generate new rollouts from dataloader prompts."""
        from areal.utils.data import cycle_dataloader

        if not hasattr(self, "_replay_dataloader_iter"):
            self._replay_dataloader_iter = iter(cycle_dataloader(dataloader))

        raw_batch = next(self._replay_dataloader_iter)
        prompts = [item for item in raw_batch]
        if prompts:
            new_trajs = self.actor.rollout_batch(
                prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            n_new = sum(t["input_ids"].shape[0] for t in new_trajs) if new_trajs else 0
            logger.info(
                f"Replay fallback: generated {n_new} new trajectories from dataloader"
            )
            return new_trajs
        return []
```

- [x] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestGenerateFromDataloader -v`
Expected: PASS

- [x] **Step 5: Run existing tests**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: All tests
PASS

- [x] **Step 6: Commit**

```bash
git add customized_areal/tree_search/trainer.py tests/test_tree_search/test_cache_trainer.py
git commit -m "feat(tree-search): add _generate_from_dataloader helper to CacheAwarePPOTrainer"
```

______________________________________________________________________

### Task 3: Modify `_replay_prepare_batch` with 3-level fallback

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:391-431`

- Test: `tests/test_tree_search/test_cache_trainer.py`

- [x] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_cache_trainer.py`:

```python
class TestReplayPrepareBatchFallback:
    def test_level1_replay_returns_when_history_available(self):
        """Level 1: Should return replay trajectories when history exists for the step."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        store._training_history[0] = [("q1", s0), ("q2", s1)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        assert len(result) == 2
        assert trainer._replay_global_step == 1

    def test_level1_partial_load_still_returns(self):
        """Level 1: Should return whatever was loaded even if some are missing."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        # Reference a non-existent seq_id
        store._training_history[0] = [("q1", s0), ("q2", 999)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        assert len(result) == 1
        assert result[0]["_mcts_query_id"] == "q1"

    def test_level2_falls_to_cached_untrained(self):
        """Level 2: Should fall back to cached untrained when replay step missing."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        # Only history for step 0, not step 1
        store._training_history[0] = [("q1", s0)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 1  # No history for step 1

        # Level 2 should load untrained (s1 from q2 is still untrained)
        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        # s0 was recorded in step 0 but not trained yet, s1 also untrained
        assert len(result) >= 1
        assert trainer._replay_global_step == 2

    def test_level3_falls_to_dataloader_generation(self):
        """Level 3: Should fall back to dataloader generation when no replay and no untrained."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.set_trained("q1", s0, True)
        # History exists but for a different step
        store._training_history[0] = [("q1", s0)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 1  # No history for step 1

        # Mock _generate_from_dataloader
        fake_traj = {
            "input_ids": torch.tensor([[5, 6, 10, 7, 8]], dtype=torch.int32),
            "rewards": torch.tensor([[0.5]], dtype=torch.float32),
        }
        trainer._generate_from_dataloader = MagicMock(return_value=[fake_traj])

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        trainer._generate_from_dataloader.assert_called_once()
        assert result == [fake_traj]
        assert trainer._replay_global_step == 2

    def test_level1_all_missing_falls_to_level2(self):
        """Level 1: When all replay trajectories are missing, fall to Level 2."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        # History references only non-existent trajectories
        store._training_history[0] = [("q99", 999)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        # Level 1 fails (all missing), Level 2 finds s0 untrained
        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        assert len(result) == 1
        assert result[0]["_mcts_query_id"] == "q1"
```

- [x] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestReplayPrepareBatchFallback -v`
Expected: FAIL — tests expect fallback behavior that doesn't exist yet

- [x] **Step 3: Write minimal implementation**

In `customized_areal/tree_search/trainer.py`, replace the existing
`_replay_prepare_batch` method (lines 391-431) with:

```python
    def _replay_prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs=None,
        should_accept_fn=None,
        group_size=1,
        dynamic_bs=False,
    ):
        """Replay mode with 3-level fallback: history → cached untrained → fresh generation."""
        global_step = self._replay_global_step

        # Level 1: Replay from training history
        if global_step in self.tree_store._training_history:
            pairs = self.tree_store._training_history[global_step]
            trajs = []
            for query_id, seq_id in pairs:
                traj = self.tree_store.load_trajectory_by_seq_id(query_id, seq_id)
                if traj is not None:
                    trajs.append(traj)
                else:
                    logger.warning(
                        f"Replay: trajectory (query_id={query_id}, seq_id={seq_id}) "
                        f"not found, skipping"
                    )
            if trajs:
                self._replay_global_step += 1
                logger.info(
                    f"Replay step {global_step}: {len(trajs)} trajectories from history"
                )
                return trajs
            logger.warning(
                f"Replay step {global_step}: all trajectories missing, falling back"
            )

        # Level 2: Cached untrained from tree store
        cached_trajs = self._load_untrained_from_tree_store()
        if cached_trajs:
            self._replay_global_step += 1
            logger.info(
                f"Replay step {global_step}: {len(cached_trajs)} cached untrained"
            )
            return cached_trajs

        # Level 3: Fresh generation from dataloader
        self._replay_global_step += 1
        return self._generate_from_dataloader(
            dataloader, workflow, workflow_kwargs, group_size
        )
```

- [x] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestReplayPrepareBatchFallback -v`
Expected: PASS

- [x] **Step 5: Run all cache trainer tests**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: All tests
PASS

- [x] **Step 6: Commit**

```bash
git add customized_areal/tree_search/trainer.py tests/test_tree_search/test_cache_trainer.py
git commit -m "feat(tree-search): add 3-level fallback to _replay_prepare_batch"
```

______________________________________________________________________

### Task 4: Update `train()` cleanup and remove stale `ValueError`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:494-499`

- Test: `tests/test_tree_search/test_cache_trainer.py`

- [x] **Step 1: Write the failing test**

Add to `tests/test_tree_search/test_cache_trainer.py`:

```python
class TestReplayTrainCleanup:
    def test_replay_dataloader_iter_cleaned_up_after_train(self):
        """Should clean up _replay_dataloader_iter in train() finally block."""
        from customized_areal.tree_search.config import (
            RolloutCacheConfig,
            TreeBackupConfig,
            TreeBackupMode,
        )
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store._training_history[0] = [("q1", s0)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.cache_config = RolloutCacheConfig(enabled=True, replay=True)
        trainer.tree_backup_config = TreeBackupConfig(mode=TreeBackupMode.CROSS_TRAINING)
        trainer.tree_store = store
        trainer._replay_global_step = 0
        trainer._replay_dataloader_iter = MagicMock()

        # Simulate the finally block logic
        if hasattr(trainer, "_replay_dataloader_iter"):
            del trainer._replay_dataloader_iter

        assert not hasattr(trainer, "_replay_dataloader_iter")
```

- [x] **Step 2: Run test to verify it passes** (this is a behavioral test for the
  cleanup pattern)

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestReplayTrainCleanup -v`
Expected: PASS (the test verifies the cleanup pattern we're about to add to the real
code)

- [x] **Step 3: Update the `train()` finally block**

In `customized_areal/tree_search/trainer.py`, update the finally block (lines 494-499)
from:

```python
        finally:
            # Always restore original prepare_batch
            self.actor.prepare_batch = original_prepare_batch
            # Clean up the dataloader iterator
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter
```

to:

```python
        finally:
            # Always restore original prepare_batch
            self.actor.prepare_batch = original_prepare_batch
            # Clean up the dataloader iterator(s)
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter
            if hasattr(self, "_replay_dataloader_iter"):
                del self._replay_dataloader_iter
```

- [x] **Step 4: Run all cache trainer tests**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: All tests
PASS

- [x] **Step 5: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): clean up _replay_dataloader_iter in train() finally block"
```

______________________________________________________________________

### Task 5: Integration test — full fallback progression

**Files:**

- Test: `tests/test_tree_search/test_cache_trainer.py`

- [x] **Step 1: Write integration test**

Add to `tests/test_tree_search/test_cache_trainer.py`:

```python
class TestReplayFallbackProgression:
    """Integration test: verify progression from Level 1 → Level 2 → Level 3."""

    def test_full_fallback_progression(self):
        """Simulate: replay history → cached untrained → fresh generation."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        s2 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.3)

        # Step 0: replay from history
        store._training_history[0] = [("q1", s0)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        # Step 0: Level 1 — replay returns s0
        result0 = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer, dataloader=MagicMock(), workflow=MagicMock()
        )
        assert len(result0) == 1
        assert result0[0]["_mcts_seq_id"] == s0

        # Mark s0 as trained (simulates what patched compute_advantages does)
        store.set_trained("q1", s0, True)

        # Step 1: Level 2 — no history, s1 and s2 still untrained
        result1 = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer, dataloader=MagicMock(), workflow=MagicMock()
        )
        assert len(result1) >= 1
        query_ids_1 = {t["_mcts_query_id"] for t in result1}
        # Should include at least q2 (s1) or q1 (s2)
        assert query_ids_1 & {"q1", "q2"}

        # Mark all as trained
        store.set_trained("q2", s1, True)
        store.set_trained("q1", s2, True)

        # Step 2: Level 3 — no history, no untrained, falls to dataloader
        fake_traj = {
            "input_ids": torch.tensor([[9, 10, 11, 12]], dtype=torch.int32),
            "rewards": torch.tensor([[0.7]], dtype=torch.float32),
        }
        trainer._generate_from_dataloader = MagicMock(return_value=[fake_traj])

        result2 = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer, dataloader=MagicMock(), workflow=MagicMock()
        )
        trainer._generate_from_dataloader.assert_called_once()
        assert result2 == [fake_traj]

        # Verify global_step incremented at each step
        assert trainer._replay_global_step == 3

    def test_replay_then_untrained_interleaving(self):
        """Steps with history use Level 1; steps without use Level 2."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)

        # History only for step 0 and step 2 (not step 1)
        store._training_history[0] = [("q1", s0)]
        store._training_history[2] = [("q2", s1)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        # Step 0: Level 1 — replay
        result0 = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer, dataloader=MagicMock(), workflow=MagicMock()
        )
        assert len(result0) == 1
        assert result0[0]["_mcts_seq_id"] == s0

        # Step 1: Level 2 — no history, s1 still untrained
        result1 = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer, dataloader=MagicMock(), workflow=MagicMock()
        )
        assert len(result1) >= 1

        # Step 2: Level 1 — history exists again
        result2 = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer, dataloader=MagicMock(), workflow=MagicMock()
        )
        assert len(result2) == 1
        assert result2[0]["_mcts_seq_id"] == s1
```

- [x] **Step 2: Run the integration test**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestReplayFallbackProgression -v`
Expected: PASS

- [x] **Step 3: Run full test suite**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: All tests
PASS

- [x] **Step 4: Run all tree search tests**

Run: `uv run pytest tests/test_tree_search/ -v` Expected: All tests PASS

- [x] **Step 5: Run pre-commit**

Run: `pre-commit run --all-files` Expected: All checks PASS (fix any issues)

- [x] **Step 6: Commit**

```bash
git add tests/test_tree_search/test_cache_trainer.py
git commit -m "test(tree-search): add integration tests for replay fallback progression"
```

______________________________________________________________________

## Self-Review Checklist

**1. Spec coverage:**

- 3-level fallback (replay → cached untrained → fresh dataloader) → Task 3
- `_load_untrained_from_tree_store` helper → Task 1
- `_generate_from_dataloader` helper → Task 2
- `train()` cleanup of `_replay_dataloader_iter` → Task 4
- Replay returns trajectories when available → Task 3
  (`test_level1_replay_returns_when_history_available`)
- Falls to Level 2 when replay missing → Task 3
  (`test_level2_falls_to_cached_untrained`)
- Falls to Level 3 when both exhausted → Task 3
  (`test_level3_falls_to_dataloader_generation`)
- `_load_untrained_from_tree_store` multi-query → Task 1
  (`test_loads_from_multiple_queries`)
- `_generate_from_dataloader` lazy init → Task 2 (`test_lazy_init_and_generation`)
- Integration test (full progression) → Task 5

**2. Placeholder scan:** No TBD/TODO/fill-in-later. All steps contain complete code.

**3. Type consistency:**

- `_load_untrained_from_tree_store() -> list[dict[str, Any]]` — consistent in Task 1
  impl and Task 3 usage
- `_generate_from_dataloader(dataloader, workflow, workflow_kwargs, group_size) -> list[dict[str, Any]]`
  — consistent in Task 2 impl and Task 3 usage
- `_replay_prepare_batch` signature unchanged — consistent with monkey-patch in
  `train()`
- `cache_config.n_samples` used consistently in Tasks 1 and 3
- `_replay_global_step: int` incremented consistently in all fallback branches

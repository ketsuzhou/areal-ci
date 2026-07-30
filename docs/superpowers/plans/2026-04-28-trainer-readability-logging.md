# Trainer Readability & Logging Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove replay mode, improve readability, and add tiered logging to
`CacheAwarePPOTrainer`.

**Architecture:** Clean up `trainer.py` by removing dead replay code, decomposing
`__init__` into `_init_tree_components` / `_init_patches`, removing the duplicate
`_mark_trajectories_trained` method, and adding INFO/DEBUG logging throughout. Config,
tests, and README are updated in sync.

**Tech Stack:** Python 3.12+, pytest

______________________________________________________________________

## File Structure

| File                                           | Action | Responsibility                                                                    |
| ---------------------------------------------- | ------ | --------------------------------------------------------------------------------- |
| `customized_areal/tree_search/config.py`       | Modify | Remove `replay` field                                                             |
| `customized_areal/tree_search/trainer.py`      | Modify | All changes: replay removal, decomposition, naming, comments, docstrings, logging |
| `tests/test_tree_search/test_cache_trainer.py` | Modify | Remove replay test classes                                                        |
| `customized_areal/tree_search/README.md`       | Modify | Remove replay sections, update component table                                    |

______________________________________________________________________

### Task 1: Remove `replay` field from `RolloutCacheConfig`

**Files:**

- Modify: `customized_areal/tree_search/config.py:25-29`

- [ ] **Step 1: Remove the `replay` field**

In `customized_areal/tree_search/config.py`, remove line 29 (`replay: bool = False`):

```python
# BEFORE:
@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1
    replay: bool = False

# AFTER:
@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1
```

- [ ] **Step 2: Verify no other code references `cache_config.replay`**

Run: `grep -r "replay" customized_areal/tree_search/ --include="*.py"` Expected: Only
references in README (handled in Task 5) and possibly the trainer `_replay_mode` guard
(handled in Task 2). No `cache_config.replay` references should remain.

- [ ] **Step 3: Run existing tests to verify no breakage**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestCacheAwareBatchBuilder -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/config.py
git commit -m "refactor(tree-search): remove replay field from RolloutCacheConfig"
```

______________________________________________________________________

### Task 2: Remove replay remnants from `trainer.py`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:146-147,354-375,457`

- [ ] **Step 1: Remove the `_replay_mode` guard in `patch_ppo_actor_for_tree_backup`**

In `trainer.py` line 146-149, remove the replay guard so `record_training_step` always
runs:

```python
# BEFORE (lines 146-149):
        # 5. Record training step order for replay (skip during replay to avoid duplicates)
        if not getattr(tree_store, "_replay_mode", False):
            global_step = result[0].get("_global_step") if result else None
            tree_store.record_training_step(global_step, result)

# AFTER:
        # 5. Record training step order for replay/debugging
        global_step = result[0].get("_global_step") if result else None
        tree_store.record_training_step(global_step, result)
```

- [ ] **Step 2: Remove the `_mark_trajectories_trained` method (lines 354-375)**

This instance method duplicates the free function `_mark_batch_trained` (lines 42-56).
It has no external callers. Delete the entire method:

```python
# DELETE lines 354-375 entirely:
    def _mark_trajectories_trained(self, rollout_batch: list[dict[str, Any]]) -> None:
        """Mark all trajectories in the batch as trained.
        ...
        """
        if not self.cache_config.enabled:
            return
        for traj in rollout_batch:
            ...
```

- [ ] **Step 3: Fix the stale comment in `train()`**

In `trainer.py` line 457, fix the comment:

```python
# BEFORE:
        # Monkey-patch prepare_batch with cache-aware or replay version
# AFTER:
        # Monkey-patch prepare_batch with cache-aware version
```

- [ ] **Step 4: Run tests**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestCacheAwareBatchBuilder tests/test_tree_search/test_cache_trainer.py::TestSplitGroupedTrajectories -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor(tree-search): remove replay remnants and duplicate mark-trained method"
```

______________________________________________________________________

### Task 3: Decompose `__init__` and improve docstrings

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:267-339`

- [ ] **Step 1: Rewrite `CacheAwarePPOTrainer.__init__` with decomposed helpers**

Replace lines 267-339 of `trainer.py` with the following. This extracts
`_init_tree_components` and `_init_patches`, and improves the class docstring and
`train()` docstring:

```python
class CacheAwarePPOTrainer(PPOTrainer):
    """PPOTrainer with rollout caching and tree backup.

    On each training step:
    1. Check cache for available trajectories per prompt
    2. Load cached trajectories, generate only missing ones
    3. Merge cached + new trajectories
    4. Run tree backup advantages on merged batch
    5. Mark used trajectories as trained
    6. Save tree checkpoint (CROSS_TRAINING mode)

    Monkey-patches ``PPOActor.compute_advantages`` at the class level (not
    instance level) so that all PPOActor instances — including those created
    internally by the base PPOTrainer — use the tree backup version. Patches
    are cleaned up in ``close()``.
    """

    def __init__(
        self,
        config: Any,
        cache_config: RolloutCacheConfig | None = None,
        tree_backup_config: TreeBackupConfig | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
    ):
        self.cache_config = cache_config or RolloutCacheConfig()
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()

        super().__init__(config, train_dataset, valid_dataset)

        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode != TreeBackupMode.OFF
        ):
            self._init_tree_components()
            self._init_patches()
            logger.info(
                f"Cache-aware training enabled "
                f"(mode={self.tree_backup_config.mode.value}, "
                f"advantage={self.tree_backup_config.advantage_mode.value}, "
                f"n_samples={self.cache_config.n_samples})"
            )

    def _init_tree_components(self) -> None:
        """Create tree store, advantage computer, and checkpoint manager."""
        turn_splitter = make_turn_splitter(
            self.tokenizer, self.tree_backup_config.assistant_marker
        )
        self.tree_store = MCTSTreeStore(turn_splitter)
        self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
        self.tree_checkpoint_manager = TreeCheckpointManager(
            self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
        )

        # Load existing tree checkpoint if available (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")

        # Reset trained flags for a fresh training run
        self.tree_store.reset_trained_flags()

        self._batch_builder = _CacheAwareBatchBuilder(
            self.tree_store, self.cache_config.n_samples, self.tokenizer
        )

    def _init_patches(self) -> None:
        """Apply monkey-patches for tree backup and query_id injection."""
        patch_ppo_actor_for_tree_backup(
            self.tree_store,
            self.tree_advantage_computer,
            advantage_mode=self.tree_backup_config.advantage_mode,
        )
        logger.info(
            f"Patched compute_advantages for tree backup "
            f"(advantage_mode={self.tree_backup_config.advantage_mode.value})"
        )

        # Patch _wrap_openai_agent to use QueryIDProxyWorkflow so that
        # dataset query_id strings are injected into trajectories as
        # _mcts_query_id. Without this, the async rollout pipeline would
        # lose the query_id because concat_padded_tensors drops non-tensor keys.
        _patch_wrap_openai_agent_for_query_id(self.actor)
```

- [ ] **Step 2: Improve `train()` docstring**

Replace the `train()` method docstring (line 440-446) with:

```python
        """Train with cache-aware rollout generation.

        Monkey-patches ``self.actor.prepare_batch`` with a cache-aware version
        that loads cached trajectories and only generates missing ones. The
        original ``prepare_batch`` is always restored in the ``finally`` block,
        so the patch never leaks on error.
        """
```

- [ ] **Step 3: Improve `_cache_aware_prepare_batch` docstring**

Replace the docstring (lines 386-390) with:

```python
        """Cache-aware replacement for prepare_batch.

        Strategy: if *all* prompts in the batch have enough cached trajectories,
        use cache only. If *any* prompt lacks sufficient cache, regenerate all
        prompts via rollout_batch (all-or-nothing). This avoids mixing cached
        and freshly-generated trajectories in a single batch.

        Returns:
            Flat list of per-sample trajectory dicts, each with shape [1, seq_len],
            carrying ``_mcts_query_id`` and ``_mcts_seq_id`` metadata.
        """
```

- [ ] **Step 4: Run tests**

Run:
`uv run pytest tests/test_tree_search/test_cache_trainer.py::TestCacheAwareBatchBuilder tests/test_tree_search/test_cache_trainer.py::TestSplitGroupedTrajectories -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor(tree-search): decompose __init__ into _init_tree_components/_init_patches, improve docstrings"
```

______________________________________________________________________

### Task 4: Add tiered logging and inline comments

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Add logging to `_tree_backup_compute_advantages` closure**

Replace the closure body in `patch_ppo_actor_for_tree_backup` (lines 130-154) with:

```python
    def _tree_backup_compute_advantages(
        self, data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # 1. Run original GAE pipeline (KL rewards, scaling, normalization, etc.)
        result = original_compute_advantages(self, data)
        logger.debug(f"Step A: GAE completed for {len(result)} trajectories")

        # 2. Insert trajectories into tree with raw rewards
        tree_store.insert_batch(result)
        logger.debug(f"Step B: Inserted {len(result)} trajectories into tree")

        # 3. Overwrite advantages/returns with tree Q-values if TREE mode
        # In TREE mode, tree Q-values replace GAE advantages. In GAE mode,
        # trajectories are still inserted (for caching and MCTS statistics)
        # but the original GAE advantages are preserved.
        if advantage_mode == AdvantageMode.TREE:
            tree_advantage_computer.compute(result)
            logger.debug(
                f"Step C: Computed tree advantages for {len(result)} "
                f"trajectories (mode=TREE)"
            )

        # 4. Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(tree_store, result)
        logger.debug(f"Step D: Marked {len(result)} trajectories as trained")

        # 5. Record training step order for replay/debugging
        global_step = result[0].get("_global_step") if result else None
        tree_store.record_training_step(global_step, result)

        # advantages/returns already overwritten by compute() in TREE mode,
        # or preserved from GAE in GAE mode.
        # kl_rewards, tot_rewards, loss_mask, logprobs preserved from GAE
        return result
```

- [ ] **Step 2: Add DEBUG logging to `_mark_batch_trained`**

Replace `_mark_batch_trained` (lines 42-56) with:

```python
def _mark_batch_trained(
    tree_store: MCTSTreeStore, trajectories: list[dict[str, Any]]
) -> None:
    """Mark all trajectories in a batch as trained after tree backup."""
    count = 0
    for traj in trajectories:
        query_id = traj.get("_mcts_query_id")
        if query_id is None:
            continue
        seq_id = traj.get("_mcts_seq_id")
        if seq_id is not None:
            tree_store.set_trained(query_id, seq_id, True)
            count += 1
        seq_ids = traj.get("_mcts_seq_ids")
        if seq_ids is not None:
            for sid in seq_ids:
                tree_store.set_trained(query_id, sid, True)
                count += 1
    if count:
        logger.debug(f"Marked {count} trajectories as trained")
```

- [ ] **Step 3: Add DEBUG logging to `_split_grouped_trajectories` and inline comment**

Replace `_split_grouped_trajectories` (lines 168-195) with:

```python
def _split_grouped_trajectories(
    trajs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split grouped trajectory dicts into individual items.

    Grouped trajectories may have shape [group_size, seq_len]. We avoid
    concat_padded_tensors because it keeps only the first dict's value for
    non-tensor, non-list keys, which would lose per-trajectory ``_mcts_query_id``
    and ``_mcts_seq_id``. Keeping them as separate items preserves
    per-trajectory metadata.
    """
    result: list[dict[str, Any]] = []
    for traj in trajs:
        batch_size = traj["input_ids"].shape[0]
        # batch_size == 1 means the trajectory is already individual;
        # appending as-is avoids unnecessary tensor slicing.
        if batch_size == 1:
            result.append(traj)
            continue
        logger.debug(
            f"Split grouped trajectory (batch_size={batch_size}) "
            f"into {batch_size} individual items"
        )
        for i in range(batch_size):
            single: dict[str, Any] = {}
            for k, v in traj.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    single[k] = v[i : i + 1]
                elif isinstance(v, list) and k == "_mcts_seq_ids":
                    single["_mcts_seq_id"] = v[i]
                    single["_mcts_query_id"] = traj.get("_mcts_query_id")
                else:
                    single[k] = v
            result.append(single)
    return result
```

- [ ] **Step 4: Add DEBUG logging to `split_prompts` and improve its docstring**

Replace the `split_prompts` method docstring and body (lines 206-248) with:

```python
    def split_prompts(
        self, prompts: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split prompts into cached and needs-generation groups.

        Query ID derivation fallback chain:
        1. ``prompt["query_id"]`` — dataset-provided string (preferred)
        2. ``prompt["_mcts_query_id"]`` — from prior injection
        3. MD5 hash of tokenized messages via ``get_query_id_from_messages``
        4. Empty string (no tree lookup possible)

        Returns:
            cached: list of dicts with keys: prompt, query_id, cached_count,
                need_gen_count
            need_gen: list of dicts with keys: prompt, query_id
        """
        cached = []
        need_gen = []

        for prompt in prompts:
            query_id = prompt.get("query_id") or prompt.get("_mcts_query_id")
            if not query_id:
                messages = prompt.get("messages", [])
                if messages:
                    query_id = get_query_id_from_messages(messages, self.tokenizer)
                else:
                    query_id = ""

            untrained_count = (
                self.tree_store.get_untrained_count(query_id) if query_id else 0
            )

            logger.debug(
                f"Prompt query_id={query_id}: {untrained_count} untrained "
                f"(need {self.n_samples})"
            )

            if untrained_count >= self.n_samples:
                cached.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,
                        "cached_count": self.n_samples,
                        "need_gen_count": 0,
                    }
                )
            else:
                need_gen.append({"prompt": prompt, "query_id": query_id})

        return cached, need_gen
```

- [ ] **Step 5: Add INFO logging in `train()` finally block**

In the `finally` block of `train()`, add a log line after restoring `prepare_batch`:

```python
        finally:
            # Always restore original prepare_batch
            self.actor.prepare_batch = original_prepare_batch
            logger.info("Restored original prepare_batch")
            # Clean up the dataloader iterator
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter
```

- [ ] **Step 6: Add inline comment to `patch_ppo_actor_for_tree_backup` about
  class-level patching**

Add a comment at the top of `patch_ppo_actor_for_tree_backup` explaining the design
choice:

```python
def patch_ppo_actor_for_tree_backup(
    tree_store: MCTSTreeStore,
    tree_advantage_computer: TreeAdvantageComputer,
    advantage_mode: AdvantageMode = AdvantageMode.TREE,
) -> None:
    """Patch PPOActor.compute_advantages to add MCTS tree backup after GAE.

    Modifies ``PPOActor.compute_advantages`` at the class level so all
    instances (including those created internally by the base PPOTrainer)
    use the tree backup version. A subclass override would only apply if
    we also subclassed the actor.

    The patch is idempotent — if ``PPOActor._original_compute_advantages``
    already exists (from a prior patch), it reuses the true original instead
    of stacking patches. Must be cleaned up via ``unpatch_ppo_actor()``.

    The patched method:
    1. Calls the original compute_advantages (full GAE pipeline)
    2. Inserts trajectories into the tree with raw rewards
    3. If advantage_mode is TREE, overwrites advantages/returns with tree Q-values
    4. Marks trajectories as trained
    5. Records training step order

    When advantage_mode is GAE, trajectories are still inserted into the tree
    (for caching and MCTS statistics), but the original GAE advantages/returns
    are preserved unchanged.
    """
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: PASS
(only TestCacheAwareBatchBuilder and TestSplitGroupedTrajectories remain)

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): add tiered logging and inline comments to trainer"
```

______________________________________________________________________

### Task 5: Remove replay test classes

**Files:**

- Modify: `tests/test_tree_search/test_cache_trainer.py`

- [ ] **Step 1: Delete replay-related test classes**

Remove the following test classes entirely from `test_cache_trainer.py`:

- `TestLoadUntrainedFromTreeStore` (lines 145-244)
- `TestGenerateFromDataloader` (lines 247-394)
- `TestReplayPrepareBatchFallback` (lines 397-541)
- `TestReplayFallbackProgression` (lines 544-651)
- `TestReplayTrainCleanup` (lines 653-663)

Keep:

- `TestCacheAwareBatchBuilder` (lines 22-90)
- `TestSplitGroupedTrajectories` (lines 92-143)

Also remove the `from customized_areal.tree_search.config import RolloutCacheConfig`
import if it was only used by the deleted test classes — but since
`TestCacheAwareBatchBuilder` doesn't use it and `TestSplitGroupedTrajectories` doesn't
use it, check if `RolloutCacheConfig` is imported and remove it if unused. Actually,
looking at the file, `RolloutCacheConfig` is only imported inside individual test
methods (e.g., `from customized_areal.tree_search.config import RolloutCacheConfig`),
not at the top level. So no top-level import changes needed.

- [ ] **Step 2: Run remaining tests**

Run: `uv run pytest tests/test_tree_search/test_cache_trainer.py -v` Expected: PASS (2
test classes, ~8 test methods)

- [ ] **Step 3: Commit**

```bash
git add tests/test_tree_search/test_cache_trainer.py
git commit -m "refactor(tree-search): remove replay mode test classes"
```

______________________________________________________________________

### Task 6: Update README to remove replay sections

**Files:**

- Modify: `customized_areal/tree_search/README.md`

- [ ] **Step 1: Remove `replay` row from the config table**

In the `RolloutCacheConfig` table (around line 47), remove the row:

```
|                      | `replay`           | `bool`           | `False` | Replay recorded training order instead of generating      |
```

- [ ] **Step 2: Update `record_training_step` description**

Around line 138, change:

```
| `record_training_step(global_step, trajectories)` | Record training order for replay; appends step to leaf node's `training_steps` |
```

to:

```
| `record_training_step(global_step, trajectories)` | Record training order; appends step to leaf node's `training_steps` |
```

- [ ] **Step 3: Remove replay-related initialization line**

Around line 189, remove the line:

```
1. If `cache_config.replay=True`, enables replay mode (see below)
```

- [ ] **Step 4: Remove the entire "Training flow — replay mode" section**

Remove from the heading `**Training flow — replay mode**` (around line 204) through the
end of the 3-level fallback table (around line 214), including the text:

```
**Training flow — replay mode** (`RolloutCacheConfig.replay=True`):

Instead of generating new rollouts, `_replay_prepare_batch()` replays trajectories with
a 3-level fallback:

| Level | Source                   | Method                              | Description                                                               |
| ----- | ------------------------ | ----------------------------------- | ------------------------------------------------------------------------- |
| 1     | Training history         | `_training_history[global_step]`    | Exact replay of recorded step order (query_id, seq_id pairs)              |
| 2     | Cached untrained in tree | `_load_untrained_from_tree_store()` | Fallback: load any untrained trajectories still in the tree               |
| 3     | Fresh generation         | `_generate_from_dataloader()`       | Fallback: generate new rollouts, prioritizing novel queries (not in tree) |

`_generate_from_dataloader()` separates prompts into novel (query_id not in tree store)
vs existing, and generates for novel queries first to expand tree coverage before
re-generating for existing ones.

Useful for debugging, reproducibility, and re-running with different advantage
computations on the same rollout data.
```

- [ ] **Step 5: Update `train()` row in the Other methods table**

Around line 226, change:

```
| `train()`                      | Monkey-patches `self.actor.prepare_batch` with cache-aware or replay version; restores on exit |
```

to:

```
| `train()`                      | Monkey-patches `self.actor.prepare_batch` with cache-aware version; restores on exit |
```

- [ ] **Step 6: Update patching mechanism section**

Around line 239, remove the line about recording training step being skipped during
replay:

```
1. Records training step order (skipped during replay)
```

Change to:

```
5. Records training step order
```

Also update the surrounding numbered list to renumber properly (currently 1-5, make sure
they're sequential).

- [ ] **Step 7: Remove the "Mode 1:" / "Mode 2:" labels**

Around line 249, change:

```
### Mode 1: Cache-Aware Training (`replay=False`)
```

to:

```
### Cache-Aware Training
```

- [ ] **Step 8: Remove `_replay_mode` reference in data flow diagram**

Around line 384, remove the line:

```
│  │  Skipped if tree_store._replay_mode      │                              │
```

And change the surrounding Step E box to remove the replay reference. The Step E box
should just say "Record training step" without the "Skipped if replay_mode" note.

- [ ] **Step 9: Remove the entire "Mode 2: Replay Training" section**

Remove everything from `### Mode 2: Replay Training (\`replay=True\`)\` (around line
409\) through the end of that section's ASCII diagram (around line 463).

- [ ] **Step 10: Commit**

```bash
git add customized_areal/tree_search/README.md
git commit -m "docs(tree-search): remove replay mode from README"
```

______________________________________________________________________

### Task 7: Run full test suite and pre-commit

**Files:**

- All modified files

- [ ] **Step 1: Run pre-commit on all changed files**

Run:
`pre-commit run --files customized_areal/tree_search/config.py customized_areal/tree_search/trainer.py tests/test_tree_search/test_cache_trainer.py customized_areal/tree_search/README.md`
Expected: PASS (formatting/linting)

- [ ] **Step 2: Run the full tree_search test suite**

Run: `uv run pytest tests/test_tree_search/ -v` Expected: PASS

- [ ] **Step 3: Run the batch consistency test suite**

Run: `uv run pytest tests/test_tree_search/test_batch_consistency.py -v` Expected: PASS

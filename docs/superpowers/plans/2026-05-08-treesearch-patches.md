# TreeSearchPatches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate 6 top-level monkey-patch functions into a single
`TreeSearchPatches` class with atomic apply/restore, crash safety, and context manager
support.

**Architecture:** New `TreeSearchPatches` class in `patches.py` replaces the scattered
`_patch_*`/`_unpatch_*` functions. Patches are deferred from `__init__` to `train()` and
scoped via `try/finally`. A `_saved` undo list and separate `_distill_undo` handle
restore.

**Tech Stack:** Python 3.12+, PyTorch, pytest

______________________________________________________________________

## File Structure

| File                                       | Action    | Responsibility                                                                                                      |
| ------------------------------------------ | --------- | ------------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/patches.py`  | Create    | `TreeSearchPatches` class with `_build_*` methods, `apply()`, `restore()`, context manager                          |
| `customized_areal/tree_search/trainer.py`  | Modify    | Remove 6 top-level functions + `_get_underlying_engine`, refactor `CacheAwarePPOTrainer` to use `TreeSearchPatches` |
| `customized_areal/tree_search/__init__.py` | No change | `TreeSearchPatches` is an internal implementation detail, not re-exported                                           |
| `tests/test_treesearch_patches.py`         | Create    | Unit tests for `TreeSearchPatches`                                                                                  |

______________________________________________________________________

### Task 1: Create `TreeSearchPatches` class in `patches.py`

**Files:**

- Create: `customized_areal/tree_search/patches.py`

- Reference: `customized_areal/tree_search/trainer.py:70-288` (existing patch functions
  to port)

- Reference: `customized_areal/tree_search/config.py` (enums/types)

- [ ] **Step 1: Create `patches.py` with full `TreeSearchPatches` class**

```python
# customized_areal/tree_search/patches.py
"""Consolidated monkey-patch manager for tree search training.

All patches needed by CacheAwarePPOTrainer are managed here, providing
atomic apply/restore, crash safety (patches scoped to try/finally),
and a context manager protocol.
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.config import AdvantageMode, LossMode
from customized_areal.tree_search.grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow
from customized_areal.tree_search.workflow_executor import TreeSearchWorkflowExecutor

from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

logger = logging.getLogger("TreeSearchPatches")


class TreeSearchPatches:
    """Manages all monkey-patches needed for tree search training.

    Tracks every original value before overwriting, provides atomic
    apply/restore, and can be used as a context manager.

    Usage::

        patches = TreeSearchPatches(
            rollout_engine=engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        patches.apply()
        try:
            ...
        finally:
            patches.restore()

    Or as a context manager::

        with TreeSearchPatches(...) as patches:
            ...
    """

    def __init__(
        self,
        rollout_engine: Any,
        advantage_mode: AdvantageMode,
        loss_mode: LossMode,
        group_size: int,
    ):
        self._engine = self._unwrap_engine(rollout_engine)
        self._advantage_mode = advantage_mode
        self._loss_mode = loss_mode
        self._group_size = group_size

        # (target_obj, attr_name, original_value) for every setattr patch
        self._saved: list[tuple[Any, str, Any]] = []
        # Separate undo for distill loss (uses its own unpatch function)
        self._distill_undo: Any = None
        self._applied = False

    # ------------------------------------------------------------------
    # Engine unwrapping (single copy)
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_engine(engine: Any) -> Any:
        """Unwrap decorators (e.g. RemoteSGLangEngine) to RemoteInfEngine."""
        if not hasattr(engine, "_wrap_openai_agent") and hasattr(engine, "_engine"):
            return engine._engine
        return engine

    # ------------------------------------------------------------------
    # Low-level patch primitives
    # ------------------------------------------------------------------

    def _save_and_set(self, obj: Any, attr: str, new_value: Any) -> None:
        """Save current value of obj.attr, then replace it."""
        original = getattr(obj, attr)
        self._saved.append((obj, attr, original))
        setattr(obj, attr, new_value)

    def _save_and_set_method(self, obj: Any, attr: str, new_method: Any) -> None:
        """Save and replace a method (binds new_method to obj)."""
        original = getattr(obj, attr)
        self._saved.append((obj, attr, original))
        setattr(obj, attr, new_method.__get__(obj, type(obj)))

    # ------------------------------------------------------------------
    # Individual patch builders
    # ------------------------------------------------------------------

    def _build_tree_backup_compute_advantages(self):
        """Build patched compute_advantages that restores tree advantages."""
        if hasattr(PPOActor, "_original_compute_advantages"):
            original = PPOActor._original_compute_advantages
        else:
            original = PPOActor.compute_advantages
        advantage_mode = self._advantage_mode

        def _patched(self_actor, data):
            result = original(self_actor, data)
            if advantage_mode == AdvantageMode.TREE:
                restored = 0
                for traj in result:
                    tree_adv = traj.pop("_tree_advantages", None)
                    tree_ret = traj.pop("_tree_returns", None)
                    if tree_adv is not None:
                        traj["advantages"] = tree_adv
                        traj["returns"] = tree_ret
                        restored += 1
                if restored < len(result):
                    logger.warning(
                        f"Tree advantages missing for "
                        f"{len(result) - restored}/{len(result)} "
                        f"trajectories in TREE mode — fell back to GAE"
                    )
                elif restored:
                    logger.debug(
                        f"Restored tree advantages for {restored} "
                        f"trajectories (mode=TREE)"
                    )
            return result

        return _patched

    def _build_tree_search_wrap(self):
        """Build patched _wrap_openai_agent returning
        TreeSearchGroupedRolloutWorkflow."""
        engine = self._engine
        group_size = self._group_size

        def _tree_search_wrap(agent, proxy_addr):
            agent_cfg = engine.config.agent
            if agent_cfg is None:
                raise RuntimeError(
                    "config.agent is None; tree search workflow requires "
                    "agent configuration. Set agent.mode in the config."
                )
            inner = QueryIDProxyWorkflow(
                mode=agent_cfg.mode,
                agent=agent,
                proxy_addr=proxy_addr,
                admin_api_key=agent_cfg.admin_api_key,
                discount=agent_cfg.turn_discount,
                export_style=agent_cfg.export_style,
                subproc_max_workers=agent_cfg.subproc_max_workers,
                proxy_gateway_addr=getattr(engine, "_proxy_gateway_addr", None),
            )
            return TreeSearchGroupedRolloutWorkflow(
                workflow=inner,
                group_size=group_size,
                logger=logger,
            )

        return _tree_search_wrap

    def _build_patched_resolve(self):
        """Build patched _resolve_workflow that strips outer
        GroupedRolloutWorkflow when the inner workflow is already
        TreeSearchGroupedRolloutWorkflow.

        The upstream _resolve_workflow unconditionally wraps with
        GroupedRolloutWorkflow when group_size > 1
        (remote_inf_engine.py:560-562), but
        TreeSearchGroupedRolloutWorkflow already handles grouping
        internally.
        """
        engine = self._engine
        original_resolve = engine._resolve_workflow

        def _patched_resolve(self_engine, wf, wf_kwargs=None, gs=1):
            resolved = original_resolve(wf, wf_kwargs, gs)
            if isinstance(resolved, GroupedRolloutWorkflow) and isinstance(
                resolved.workflow, TreeSearchGroupedRolloutWorkflow
            ):
                logger.debug(
                    "Skipping outer GroupedRolloutWorkflow wrapper "
                    "(TreeSearchGroupedRolloutWorkflow already handles grouping)"
                )
                return resolved.workflow
            return resolved

        return _patched_resolve

    def _build_tree_search_executor(self):
        """Build a TreeSearchWorkflowExecutor replacing the original."""
        engine = self._engine
        original = engine.workflow_executor

        new_executor = TreeSearchWorkflowExecutor(
            config=engine.config,
            inference_engine=engine,
        )

        # Copy all internal state from the original executor.
        # Using vars() instead of listing attributes by name so that
        # upstream additions are automatically picked up.
        for attr, value in vars(original).items():
            if attr.startswith("__"):
                continue
            if attr in ("config", "inference_engine"):
                continue
            setattr(new_executor, attr, value)
        new_executor._initialized = True

        return new_executor

    # ------------------------------------------------------------------
    # Apply / Restore
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """Apply all patches. On failure, roll back any already-applied patches."""
        if self._applied:
            logger.warning("TreeSearchPatches.apply() called twice; skipping")
            return

        try:
            # Patch 1: PPOActor.compute_advantages (class-level)
            new_compute_adv = self._build_tree_backup_compute_advantages()
            if not hasattr(PPOActor, "_original_compute_advantages"):
                PPOActor._original_compute_advantages = PPOActor.compute_advantages
            self._saved.append(
                (PPOActor, "compute_advantages",
                 PPOActor._original_compute_advantages)
            )
            PPOActor.compute_advantages = new_compute_adv

            # Patch 2: engine._wrap_openai_agent
            if hasattr(self._engine, "_wrap_openai_agent"):
                self._save_and_set(
                    self._engine,
                    "_wrap_openai_agent",
                    self._build_tree_search_wrap(),
                )
            else:
                logger.warning(
                    "Engine has no _wrap_openai_agent method; "
                    "tree search workflow will not be available"
                )

            # Patch 2b: engine._resolve_workflow (double-wrapping prevention)
            if hasattr(self._engine, "_resolve_workflow"):
                self._save_and_set_method(
                    self._engine,
                    "_resolve_workflow",
                    self._build_patched_resolve(),
                )

            # Patch 3: engine.workflow_executor
            if hasattr(self._engine, "workflow_executor"):
                new_executor = self._build_tree_search_executor()
                self._save_and_set(self._engine, "workflow_executor", new_executor)
            else:
                logger.warning(
                    "Engine has no workflow_executor attribute; "
                    "tree search workflow executor will not be available"
                )

            # Patch 4 (conditional): PPOActor._ppo_update distill loss
            if self._loss_mode != LossMode.GRPO:
                from customized_areal.tree_search.training.actor import (
                    patch_ppo_actor_class_to_use_distill_loss,
                    unpatch_ppo_actor_distill_loss,
                )
                patch_ppo_actor_class_to_use_distill_loss()
                self._distill_undo = unpatch_ppo_actor_distill_loss

            self._applied = True
            logger.info(
                f"Applied tree search patches "
                f"(advantage={self._advantage_mode.value}, "
                f"loss={self._loss_mode.value}, "
                f"group_size={self._group_size})"
            )

        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        """Restore all original values in reverse order."""
        if not self._applied and not self._saved and self._distill_undo is None:
            return

        # Restore distill loss first (if applied)
        if self._distill_undo is not None:
            self._distill_undo()
            self._distill_undo = None

        # Restore in reverse order (LIFO)
        for obj, attr, original in reversed(self._saved):
            setattr(obj, attr, original)

        # Clean up idempotency marker
        if hasattr(PPOActor, "_original_compute_advantages"):
            del PPOActor._original_compute_advantages

        self._saved.clear()
        self._applied = False
        logger.info("Restored all tree search patches")

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
        return False
```

- [ ] **Step 2: Verify the file has no syntax errors**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/patches.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/patches.py
git commit -m "feat: add TreeSearchPatches class for consolidated monkey-patching"
```

______________________________________________________________________

### Task 2: Refactor `CacheAwarePPOTrainer` to use `TreeSearchPatches`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Remove the 7 top-level functions and `_get_underlying_engine`**

Delete lines 70-288 from `trainer.py` (the functions: `_get_underlying_engine`,
`_patch_wrap_openai_agent_for_tree_search`, `_unpatch_wrap_openai_agent`,
`_patch_workflow_executor`, `_unpatch_workflow_executor`,
`patch_ppo_actor_for_tree_backup`, `unpatch_ppo_actor`).

Also remove the now-unused imports from `trainer.py`:

- `GroupedRolloutWorkflow` (line 46) — moved to `patches.py`
- `QueryIDProxyWorkflow` (line 42) — moved to `patches.py`
- `TreeSearchWorkflowExecutor` (line 43) — moved to `patches.py`
- `PPOActor` (line 47) — only used in the deleted `patch_ppo_actor_for_tree_backup` /
  `unpatch_ppo_actor` functions; moved to `patches.py`

Keep imports that are still used in `trainer.py`:

- `TreeSearchGroupedRolloutWorkflow` (line 34-36) — still used in
  `_CacheAwareBatchBuilder`? No — check. Actually it's only used in the deleted
  `_patch_wrap_openai_agent_for_tree_search`. Remove it.

- `AdvantageMode`, `LossMode` (line 28-33) — still used in `trainer.py` body

- `TreeAdvantageComputer`, `TreeCheckpointManager`, `MCTSTreeStore`, `Node`,
  `_node_to_tensor_dict` — still used

- [ ] **Step 2: Add import for `TreeSearchPatches`**

At the top of `trainer.py`, add after the existing imports:

```python
from customized_areal.tree_search.patches import TreeSearchPatches
```

- [ ] **Step 3: Replace `_init_patches` with `TreeSearchPatches` instantiation in
  `__init__`**

Replace the `_init_patches` method (lines 460-491) and its call in `__init__` (line
405).

In `__init__`, replace:

```python
            self._init_patches()
```

with:

```python
            self._patches = TreeSearchPatches(
                rollout_engine=self.rollout,
                advantage_mode=self.tree_backup_config.advantage_mode,
                loss_mode=self.tree_backup_config.loss_mode,
                group_size=self.cache_config.n_samples,
            )
```

And add `self._patches: TreeSearchPatches | None = None` as an instance attribute
(initialized before the conditional block, or inside it — matching the pattern that
`_patches` is only set when tree components are initialized).

Delete the entire `_init_patches` method.

- [ ] **Step 4: Refactor `train()` to use `patches.apply()` / `patches.restore()`**

Replace the `train()` method. The `prepare_batch` patch stays as a manual
`setattr`/restore alongside `patches.apply()`/`restore()` in the `try/finally`:

```python
    def train(
        self,
        workflow=None,
        eval_workflow=None,
        workflow_kwargs=None,
        eval_workflow_kwargs=None,
        dynamic_filter_fn=None,
        total_epochs=None,
    ):
        """Train with cache-aware rollout generation.

        Monkey-patches ``self.actor.prepare_batch`` with a cache-aware version
        and applies tree search patches via TreeSearchPatches. Both are
        restored in the ``finally`` block, so patches never leak on error.
        """
        if not self.cache_config.enabled:
            return super().train(
                workflow=workflow,
                eval_workflow=eval_workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=dynamic_filter_fn,
                total_epochs=total_epochs,
            )

        original_prepare_batch = self.actor.prepare_batch

        def _prepare_batch_fn(
            dataloader,
            workflow,
            workflow_kwargs=None,
            should_accept_fn=None,
            group_size=1,
            dynamic_bs=False,
        ):
            return self._cache_aware_prepare_batch(
                dataloader=dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
                dynamic_bs=dynamic_bs,
            )

        assert self._patches is not None
        self._patches.apply()
        self.actor.prepare_batch = _prepare_batch_fn

        try:
            return super().train(
                workflow=workflow,
                eval_workflow=eval_workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=dynamic_filter_fn,
                total_epochs=total_epochs,
            )
        finally:
            self.actor.prepare_batch = original_prepare_batch
            logger.info("Restored original prepare_batch")
            self._patches.restore()
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter
```

- [ ] **Step 5: Simplify `close()`**

Replace the `close()` method. Since patches are restored in `train()`'s finally block,
`close()` only needs a safety-net restore:

```python
    def close(self) -> None:
        # Safety net: restore patches if train() was never called
        # or crashed before finally executed.
        if self._patches is not None:
            self._patches.restore()
        super().close()
```

- [ ] **Step 6: Update the class docstring**

Remove references to `patch_ppo_actor_for_tree_backup` / `unpatch_ppo_actor` from the
`CacheAwarePPOTrainer` class docstring. Update to mention `TreeSearchPatches`:

```python
    """PPOTrainer with rollout caching and tree backup.

    On each training step:
    1. _cache_aware_prepare_batch:
       a. Check cache / generate trajectories
       b. Insert into MCTS tree (while query_id is available)
       c. Compute tree advantages (TREE mode) and stash as
          _tree_advantages / _tree_returns
       d. Mark trajectories as trained
       e. Save tree checkpoint (CROSS_TRAINING mode)
    2. compute_advantages:
       a. GAE runs (overwrites advantages/returns)
       b. TreeSearchPatches restores tree advantages from
          _tree_advantages/_tree_returns
    3. ppo_update uses restored tree advantages

    Monkey-patches are managed by TreeSearchPatches, which applies them
    at the start of train() and restores them in the finally block.
    """
```

- [ ] **Step 7: Verify no syntax errors**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/trainer.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor: replace scattered patch functions with TreeSearchPatches"
```

______________________________________________________________________

### Task 3: Write unit tests for `TreeSearchPatches`

**Files:**

- Create: `tests/test_treesearch_patches.py`

- [ ] **Step 1: Write tests**

The tests need to work without real GPU/engine infrastructure, so we use mock objects to
stand in for the engine and PPOActor. The key behaviors to test:

1. `apply()` / `restore()` cycle restores all original values
1. Double-apply is idempotent
1. Context manager restores on exception
1. `_build_tree_search_wrap` raises `RuntimeError` on missing config
1. `restore()` is safe to call multiple times
1. Partial rollback on `apply()` failure

```python
"""Tests for TreeSearchPatches."""

from unittest.mock import MagicMock, patch

import pytest

from customized_areal.tree_search.config import AdvantageMode, LossMode
from customized_areal.tree_search.patches import TreeSearchPatches

from areal.trainer.ppo.actor import PPOActor


@pytest.fixture
def mock_engine():
    """Create a mock engine with all patched attributes."""
    engine = MagicMock()
    engine._wrap_openai_agent = MagicMock(return_value="original_wrap")
    engine._resolve_workflow = MagicMock(return_value="original_resolve")
    engine.workflow_executor = MagicMock()
    engine.config = MagicMock()
    engine.config.agent = MagicMock(
        mode="mode",
        admin_api_key="key",
        turn_discount=1.0,
        export_style="concat",
        subproc_max_workers=1,
    )
    engine._proxy_gateway_addr = None
    return engine


@pytest.fixture
def saved_ppo_actor_state():
    """Save and restore PPOActor class state around each test."""
    original_compute = PPOActor.compute_advantages
    had_sentinel = hasattr(PPOActor, "_original_compute_advantages")
    original_sentinel = getattr(
        PPOActor, "_original_compute_advantages", None
    )
    yield
    PPOActor.compute_advantages = original_compute
    if had_sentinel:
        PPOActor._original_compute_advantages = original_sentinel
    elif hasattr(PPOActor, "_original_compute_advantages"):
        del PPOActor._original_compute_advantages


class TestApplyRestore:
    """Test apply() / restore() cycle."""

    def test_apply_then_restore_restores_originals(
        self, mock_engine, saved_ppo_actor_state
    ):
        original_compute = PPOActor.compute_advantages
        original_wrap = mock_engine._wrap_openai_agent
        original_resolve = mock_engine._resolve_workflow
        original_executor = mock_engine.workflow_executor

        patches = TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        patches.apply()

        # Patches are active
        assert PPOActor.compute_advantages is not original_compute
        assert mock_engine._wrap_openai_agent is not original_wrap
        assert mock_engine._resolve_workflow is not original_resolve
        assert mock_engine.workflow_executor is not original_executor

        patches.restore()

        # Originals restored
        assert PPOActor.compute_advantages is original_compute
        assert mock_engine._wrap_openai_agent is original_wrap
        assert mock_engine._resolve_workflow is original_resolve
        assert mock_engine.workflow_executor is original_executor
        assert not hasattr(PPOActor, "_original_compute_advantages")


class TestIdempotency:
    """Test double-apply safety."""

    def test_apply_twice_is_noop(self, mock_engine, saved_ppo_actor_state):
        patches = TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        patches.apply()
        first_patched = PPOActor.compute_advantages
        patches.apply()  # should be a no-op
        assert PPOActor.compute_advantages is first_patched
        patches.restore()


class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager_restores_on_normal_exit(
        self, mock_engine, saved_ppo_actor_state
    ):
        original_compute = PPOActor.compute_advantages

        with TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        ):
            assert PPOActor.compute_advantages is not original_compute

        assert PPOActor.compute_advantages is original_compute

    def test_context_manager_restores_on_exception(
        self, mock_engine, saved_ppo_actor_state
    ):
        original_compute = PPOActor.compute_advantages

        with pytest.raises(ValueError):
            with TreeSearchPatches(
                rollout_engine=mock_engine,
                advantage_mode=AdvantageMode.TREE,
                loss_mode=LossMode.GRPO,
                group_size=4,
            ):
                assert PPOActor.compute_advantages is not original_compute
                raise ValueError("simulated crash")

        assert PPOActor.compute_advantages is original_compute


class TestTreeSearchWrap:
    """Test _build_tree_search_wrap behavior."""

    def test_raises_on_missing_agent_config(
        self, mock_engine, saved_ppo_actor_state
    ):
        mock_engine.config.agent = None

        patches = TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        wrapped = patches._build_tree_search_wrap()

        with pytest.raises(RuntimeError, match="config.agent is None"):
            wrapped(MagicMock(), "addr")

    def test_returns_treesearch_workflow(
        self, mock_engine, saved_ppo_actor_state
    ):
        from customized_areal.tree_search.grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        patches = TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        wrapped = patches._build_tree_search_wrap()
        result = wrapped(MagicMock(), "addr")
        assert isinstance(result, TreeSearchGroupedRolloutWorkflow)


class TestRestoreSafety:
    """Test that restore() is safe in edge cases."""

    def test_restore_without_apply_is_noop(
        self, mock_engine, saved_ppo_actor_state
    ):
        patches = TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        # Should not raise
        patches.restore()

    def test_restore_twice_is_safe(
        self, mock_engine, saved_ppo_actor_state
    ):
        original_compute = PPOActor.compute_advantages
        patches = TreeSearchPatches(
            rollout_engine=mock_engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        patches.apply()
        patches.restore()
        patches.restore()  # second call should be no-op
        assert PPOActor.compute_advantages is original_compute
```

- [ ] **Step 2: Run the tests**

Run:
`python -m pytest tests/test_treesearch_patches.py -v --no-header -x 2>&1 | head -50`
Expected: All tests PASS

Note: If imports fail due to missing dependencies (GPU, distributed), the tests may need
to be adjusted to mock those imports. The tests above use `MagicMock` for the engine, so
they should work without GPU. However, importing `PPOActor` may pull in
PyTorch/distributed dependencies. If so, add `@pytest.mark.skipif` guards or mock the
imports.

- [ ] **Step 3: Commit**

```bash
git add tests/test_treesearch_patches.py
git commit -m "test: add unit tests for TreeSearchPatches"
```

______________________________________________________________________

### Task 4: Verify no regressions in existing imports

**Files:**

- Verify: `customized_areal/tree_search/__init__.py`

- Verify: Any files that import from `trainer.py`

- [ ] **Step 1: Check for external consumers of removed functions**

Run:
`grep -r 'patch_ppo_actor_for_tree_backup\|unpatch_ppo_actor\|_patch_wrap_openai_agent\|_unpatch_wrap_openai_agent\|_patch_workflow_executor\|_unpatch_workflow_executor\|_get_underlying_engine' --include='*.py' customized_areal/ | grep -v 'patches.py' | grep -v 'trainer.py'`

Expected: No results (all consumers are within `trainer.py` itself).

If any external file imports these, update the import to use `TreeSearchPatches`
instead.

- [ ] **Step 2: Verify `__init__.py` does not re-export the removed functions**

The `__init__.py` imports `CacheAwarePPOTrainer` from `trainer.py`. The removed
top-level functions were never in `__init__.py`'s `__all__`, so no change needed.

- [ ] **Step 3: Verify syntax of all modified files**

Run:
`python -c "import ast; [ast.parse(open(f).read()) for f in ['customized_areal/tree_search/patches.py', 'customized_areal/tree_search/trainer.py']]; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit any import fixes (if needed)**

Only if Step 1 found external consumers.

______________________________________________________________________

### Task 5: Final verification and cleanup

- [ ] **Step 1: Run pre-commit on changed files**

Run:
`pre-commit run --files customized_areal/tree_search/patches.py customized_areal/tree_search/trainer.py tests/test_treesearch_patches.py`
Expected: All checks pass

- [ ] **Step 2: Run the unit tests one more time**

Run: `python -m pytest tests/test_treesearch_patches.py -v` Expected: All tests PASS

- [ ] **Step 3: Final commit if any formatting fixes were needed**

```bash
git add -u
git commit -m "style: apply pre-commit fixes to TreeSearchPatches"
```

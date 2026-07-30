# TreeSearchPatches: Consolidate Monkey-Patching

## Problem

`_init_patches` in `trainer.py:460-491` applies four interdependent monkey-patches
across three targets using 6 top-level functions (`_patch_*`, `_unpatch_*`,
`patch_ppo_actor_for_tree_backup`, `unpatch_ppo_actor`) plus `_get_underlying_engine`,
totaling ~140 lines.

Issues:

1. **No atomicity.** Patches applied sequentially with no rollback if a later patch
   fails after earlier ones succeed.
1. **No lifecycle guarantee.** Patches applied in `__init__`, restored only in
   `close()`. A crash between them leaves class-level patches corrupting subsequent
   trainer instances.
1. **Double-wrapping prevention hidden.** Patch 2b (strip outer
   `GroupedRolloutWorkflow`) is nested inside patch 2's function. The coupling to
   upstream `_resolve_workflow` is only in comments.
1. **Brittle attribute copy.** Patch 3 copies 7 private attributes by name — silently
   breaks on upstream renames/additions.
1. **Duplicated engine unwrapping.** `_get_underlying_engine` called 4 times (apply +
   restore for 2 functions).
1. **Idempotency ad-hoc.** Only `patch_ppo_actor` guards against double-apply; other
   patches have no such guard.
1. **Silent None on config error.** `_tree_search_wrap` returns `None` when
   `config.agent` is `None`, causing opaque errors downstream.

## Design

### `TreeSearchPatches` class

New file: `customized_areal/tree_search/patches.py`

A single class that:

- Receives config at construction (`rollout_engine`, `advantage_mode`, `loss_mode`,
  `group_size`)
- Unwraps the engine once in `__init__` (replaces 4 calls to `_get_underlying_engine`)
- Tracks every patch via `_saved: list[tuple[Any, str, Any]]`
- Builds replacement values via `_build_*` methods (one per patch)
- Applies atomically — `apply()` wraps all patches in `try/except` with full rollback on
  failure
- Restores in LIFO order — `restore()` reverses patches so inner dependencies are undone
  first
- Works as context manager — `with TreeSearchPatches(...) as p:`
- Stores distill loss unpatch callable in `self._distill_undo` (separate from the
  uniform `_saved` list, avoiding `None` sentinel tricks)

### Patch lifecycle

- `__init__` → stores `TreeSearchPatches` instance, does **not** call `apply()`
- `train()` → `patches.apply()` + `prepare_batch` override in a single `try/finally`;
  `finally` calls `patches.restore()` and restores `prepare_batch`
- `close()` → calls `patches.restore()` as safety net (idempotent, no-op if already
  restored)

Patches are only active during `train()`. Between `__init__` and the first `train()`
call, the system is unpatched. This is safe because nothing calls the patched methods in
that window.

The `prepare_batch` patch stays as a manual `setattr`/restore in `train()`'s
`try/finally` since it's an instance-level patch with naturally scoped lifecycle.

### Individual patches

| #   | Target                        | Builder method                            | Key changes from current                                                                                      |
| --- | ----------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1   | `PPOActor.compute_advantages` | `_build_tree_backup_compute_advantages()` | Closes over `advantage_mode`; adds warning when tree advantages are missing                                   |
| 2   | `engine._wrap_openai_agent`   | `_build_tree_search_wrap()`               | Raises `RuntimeError` instead of returning `None` on missing config                                           |
| 2b  | `engine._resolve_workflow`    | `_build_patched_resolve()`                | Separate builder with docstring explaining the double-wrapping coupling                                       |
| 3   | `engine.workflow_executor`    | `_build_tree_search_executor()`           | Uses `vars()` loop instead of listing attributes by name                                                      |
| 4   | `PPOActor._ppo_update`        | Conditional in `apply()`                  | Calls existing `patch_ppo_actor_class_to_use_distill_loss()`; stores unpatch callable in `self._distill_undo` |

### Deleted code

The following top-level functions are removed from `trainer.py`:

- `_get_underlying_engine`
- `_patch_wrap_openai_agent_for_tree_search`
- `_unpatch_wrap_openai_agent`
- `_patch_workflow_executor`
- `_unpatch_workflow_executor`
- `patch_ppo_actor_for_tree_backup`
- `unpatch_ppo_actor`

### `CacheAwarePPOTrainer` changes

- `__init__`: stores `self._patches = TreeSearchPatches(...)` instead of calling
  `_init_patches()`
- `_init_patches()`: removed entirely
- `train()`: calls `self._patches.apply()` before `try`, calls `self._patches.restore()`
  in `finally`
- `close()`: calls `self._patches.restore()` as safety net

### What this solves

| Problem                        | Before                              | After                                      |
| ------------------------------ | ----------------------------------- | ------------------------------------------ |
| No atomicity                   | No rollback on failure              | `apply()` has full rollback                |
| No lifecycle guarantee         | Patches live `__init__`→`close()`   | Patches scoped to `train()`                |
| Double-wrapping hidden         | Patch 2b nested inside patch 2      | Separate builder with docstring            |
| Brittle attr copy              | 7 attributes by name                | `vars()` loop                              |
| Duplicated unwrapping          | 4 calls to `_get_underlying_engine` | 1 call in `__init__`                       |
| No idempotency guard           | Only `patch_ppo_actor` checks       | `apply()` returns early if already applied |
| Silent None on config error    | Returns `None`                      | Raises `RuntimeError`                      |
| Missing tree advantage warning | Silent GAE fallback                 | Logs warning with count                    |

## Testing

Add a test that verifies:

1. All patches are fully restored after `restore()` (both normally and after a simulated
   exception in `apply()`)
1. Double-apply is idempotent (second `apply()` is a no-op)
1. The context manager restores on exception exit
1. `_build_tree_search_wrap` raises `RuntimeError` on missing config

## Out of scope

- Bug fixes from the code review (insert_batch skip, Bessel variance, query_id
  deserialization, etc.)
- Changes to the distill loss patching logic itself
- Changes to `prepare_batch` patching logic

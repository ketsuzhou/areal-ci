# Loss Mode Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `LossMode` switch to `CacheAwarePPOTrainer` that enables GRPO-only,
distill-only, or combined loss via a config field.

**Architecture:** Extend `TreeBackupConfig` with a `LossMode` enum and loss weight
fields. In `CacheAwarePPOTrainer`, override `_create_train_engine` to return
`MultiCandidateFSDPPPOActor` when distill is enabled, patch `PPOActor._ppo_update` with
`grpo_distill_loss_fn`, inject loss weights into trajectory dicts, and unpatch on close.

**Tech Stack:** Python 3.12+ | PyTorch | AReaL PPOTrainer infrastructure

______________________________________________________________________

## File Structure

| File                                      | Responsibility                                                                                        |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/config.py`  | `LossMode` enum; `TreeBackupConfig` with `loss_mode`, `rl_loss_weight`, `distill_loss_weight`         |
| `customized_areal/tree_search/trainer.py` | `CacheAwarePPOTrainer` with `_create_train_engine` override, loss patching, weight injection, cleanup |

No new files created. No changes to loss functions or actor patching infrastructure.

______________________________________________________________________

### Task 1: Add LossMode enum and extend TreeBackupConfig

**Files:**

- Modify: `customized_areal/tree_search/config.py`

- [ ] **Step 1: Add `LossMode` enum after `AdvantageMode`**

```python
class LossMode(str, Enum):
    GRPO = "grpo"
    DISTILL = "distill"
    BOTH = "both"
```

- [ ] **Step 2: Add `loss_mode`, `rl_loss_weight`, `distill_loss_weight` fields to
  `TreeBackupConfig`**

The full updated `TreeBackupConfig`:

```python
@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.TREE
    loss_mode: LossMode = LossMode.GRPO
    rl_loss_weight: float = 1.0
    distill_loss_weight: float = 0.005
```

- [ ] **Step 3: Verify the file parses correctly**

Run:
`python -c "from customized_areal.tree_search.config import TreeBackupConfig, LossMode; c = TreeBackupConfig(); print(c.loss_mode, c.rl_loss_weight, c.distill_loss_weight)"`
Expected: `LossMode.GRPO 1.0 0.005`

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/config.py
git commit -m "feat(tree-search): add LossMode enum and distill fields to TreeBackupConfig"
```

______________________________________________________________________

### Task 2: Override `_create_train_engine` in CacheAwarePPOTrainer

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

The base `PPOTrainer.__init__` calls
`self._create_train_engine(config.actor, self.actor_alloc)` at
`areal/trainer/rl_trainer.py:166`. We override this method to return
`MultiCandidateFSDPPPOActor` when `loss_mode != GRPO`.

- [ ] **Step 1: Add import for `LossMode` at the top of trainer.py**

Add `LossMode` to the import from `customized_areal.tree_search.config`:

```python
from customized_areal.tree_search.config import (
    AdvantageMode,
    LossMode,
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
```

- [ ] **Step 2: Add `_create_train_engine` override to `CacheAwarePPOTrainer`**

Place this method after `__init__`:

```python
def _create_train_engine(self, actor_config, alloc):
    """Override to use MultiCandidateFSDPPPOActor when distill loss is enabled."""
    if self.tree_backup_config.loss_mode != LossMode.GRPO:
        if alloc.backend != "fsdp":
            raise ValueError(
                f"Distillation loss mode requires FSDP backend, "
                f"got: {alloc.backend}"
            )
        from customized_areal.on_policy_distill.engine import (
            MultiCandidateFSDPPPOActor,
        )

        actor_cls = MultiCandidateFSDPPPOActor
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        logger.info(
            f"Created MultiCandidateFSDPPPOActor "
            f"(loss_mode={self.tree_backup_config.loss_mode.value})"
        )
        return actor
    return super()._create_train_engine(actor_config, alloc)
```

Also add the missing import at the top:

```python
from areal.utils.environ import is_single_controller
```

- [ ] **Step 3: Verify the import resolves**

Run:
`python -c "from customized_areal.tree_search.trainer import CacheAwarePPOTrainer; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): override _create_train_engine for MultiCandidate actor"
```

______________________________________________________________________

### Task 3: Patch `PPOActor._ppo_update` with distill loss in `_init_patches`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Add distill loss patching logic to `_init_patches`**

Append the following to `_init_patches`, after the existing `workflow_executor` patch
block:

```python
    # Patch PPOActor._ppo_update with grpo_distill_loss_fn when distill is enabled
    if self.tree_backup_config.loss_mode != LossMode.GRPO:
        from customized_areal.on_policy_distill.training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        patch_ppo_actor_class_to_use_distill_loss()
        logger.info(
            f"Patched PPOActor._ppo_update with grpo_distill_loss_fn "
            f"(loss_mode={self.tree_backup_config.loss_mode.value})"
        )
```

- [ ] **Step 2: Verify the patching import works**

Run:
`python -c "from customized_areal.on_policy_distill.training.actor import patch_ppo_actor_class_to_use_distill_loss; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): patch PPOActor with distill loss in _init_patches"
```

______________________________________________________________________

### Task 4: Inject loss weights into trajectory dicts in `_cache_aware_prepare_batch`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

The `grpo_distill_loss_fn` reads `rl_loss_weight` and `distill_loss_weight` from
`input_data`. We inject these into each trajectory dict in `_cache_aware_prepare_batch`,
after tree operations and before returning.

- [ ] **Step 1: Add weight injection after the "End tree operations" comment block**

Insert after the line `# --- End tree operations ---` (currently line 599) and before
the list-dict-to-tensor conversion (currently line 604):

```python
    # Inject distillation loss weights into trajectory dicts
    if self.tree_backup_config.loss_mode != LossMode.GRPO:
        for traj in trajs:
            if self.tree_backup_config.loss_mode == LossMode.DISTILL:
                traj["rl_loss_weight"] = 0.0
            else:
                traj["rl_loss_weight"] = self.tree_backup_config.rl_loss_weight
            traj["distill_loss_weight"] = self.tree_backup_config.distill_loss_weight
```

- [ ] **Step 2: Verify the file parses correctly**

Run:
`python -c "from customized_areal.tree_search.trainer import CacheAwarePPOTrainer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): inject distill loss weights into trajectory dicts"
```

______________________________________________________________________

### Task 5: Update `close` to handle distill loss cleanup and update `__init__` logging

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Update `__init__` info log to include `loss_mode`**

Change the existing log line in `__init__`:

```python
logger.info(
    f"Cache-aware training enabled "
    f"(mode={self.tree_backup_config.mode.value}, "
    f"advantage={self.tree_backup_config.advantage_mode.value}, "
    f"n_samples={self.cache_config.n_samples}, "
    f"loss_mode={self.tree_backup_config.loss_mode.value})"
)
```

- [ ] **Step 2: Add unpatch guard in `close`**

The existing `close` method unpatches tree backup and workflow patches. The
`patch_ppo_actor_class_to_use_distill_loss` function uses a global `_patch_applied`
guard and stores the original on `PPOActor._ppo_update`, but it has no unpatch function.
We need to add one.

Add an unpatch function call in `close`. We also need to save the original `_ppo_update`
for restoration. Since `patch_ppo_actor_class_to_use_distill_loss` stores the original
in a local `_original_ppo_update` (marked `F841` unused), we need a different approach.

The cleanest approach: add `unpatch_ppo_actor_distill_loss()` to
`customized_areal/on_policy_distill/training/actor.py`.

Add this function to `customized_areal/on_policy_distill/training/actor.py` after
`patch_ppo_actor_class_to_use_distill_loss`:

```python
def unpatch_ppo_actor_distill_loss() -> None:
    """Restore the original PPOActor._ppo_update method.

    Must be called after patch_ppo_actor_class_to_use_distill_loss().
    """
    global _patch_applied, _original_ppo_update
    if _patch_applied and _original_ppo_update is not None:
        PPOActor._ppo_update = _original_ppo_update
        _original_ppo_update = None
        _patch_applied = False
        logger.info("Restored original PPOActor._ppo_update")
```

Also fix the `_original_ppo_update` variable in the patch function — remove the
`# noqa: F841` and make it a module-level global:

```python
_patch_applied = False
_original_ppo_update = None


def patch_ppo_actor_class_to_use_distill_loss() -> None:
    """Patch PPOActor class to use grpo_distill_loss_fn globally."""
    global _patch_applied, _original_ppo_update
    if _patch_applied:
        return

    _original_ppo_update = PPOActor._ppo_update

    def _ppo_update_with_distill_loss(self, data: dict[str, Any]) -> None:
        # ... (existing implementation unchanged)
```

- [ ] **Step 3: Call the unpatch in `CacheAwarePPOTrainer.close`**

Update `close`:

```python
def close(self) -> None:
    if (
        self.cache_config.enabled
        and self.tree_backup_config.mode != TreeBackupMode.OFF
    ):
        unpatch_ppo_actor()
        _unpatch_wrap_openai_agent(self.rollout)
        _unpatch_workflow_executor(self.rollout)
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            from customized_areal.on_policy_distill.training.actor import (
                unpatch_ppo_actor_distill_loss,
            )

            unpatch_ppo_actor_distill_loss()
    super().close()
```

- [ ] **Step 4: Verify both files parse correctly**

Run:
`python -c "from customized_areal.on_policy_distill.training.actor import patch_ppo_actor_class_to_use_distill_loss, unpatch_ppo_actor_distill_loss; print('OK')"`
Expected: `OK`

Run:
`python -c "from customized_areal.tree_search.trainer import CacheAwarePPOTrainer; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add customized_areal/on_policy_distill/training/actor.py customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): add unpatch for distill loss and cleanup in close"
```

______________________________________________________________________

### Task 6: Update `__init__.py` exports

**Files:**

- Modify: `customized_areal/tree_search/__init__.py`

- [ ] **Step 1: Add `LossMode` to exports**

Read the current `__init__.py` and add `LossMode` to the imports/exports from `config`.

- [ ] **Step 2: Verify the import works**

Run:
`python -c "from customized_areal.tree_search import LossMode; print(LossMode.GRPO, LossMode.DISTILL, LossMode.BOTH)"`
Expected: `LossMode.GRPO LossMode.DISTILL LossMode.BOTH`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/__init__.py
git commit -m "feat(tree-search): export LossMode from package __init__"
```

______________________________________________________________________

### Task 7: Run pre-commit and final verification

**Files:**

- All modified files

- [ ] **Step 1: Run pre-commit on all modified files**

Run:
`pre-commit run --files customized_areal/tree_search/config.py customized_areal/tree_search/trainer.py customized_areal/on_policy_distill/training/actor.py customized_areal/tree_search/__init__.py`
Expected: All checks pass

- [ ] **Step 2: Verify the full import chain works**

Run:
`python -c "from customized_areal.tree_search.config import TreeBackupConfig, LossMode; c = TreeBackupConfig(loss_mode=LossMode.BOTH, rl_loss_weight=1.0, distill_loss_weight=0.01); print(c)"`
Expected:
`TreeBackupConfig(mode=TreeBackupMode.OFF, checkpoint_dir='', advantage_mode=AdvantageMode.TREE, loss_mode=LossMode.BOTH, rl_loss_weight=1.0, distill_loss_weight=0.01)`

- [ ] **Step 3: Final commit if pre-commit made formatting changes**

```bash
git add -u
git commit -m "style: pre-commit formatting fixes"
```

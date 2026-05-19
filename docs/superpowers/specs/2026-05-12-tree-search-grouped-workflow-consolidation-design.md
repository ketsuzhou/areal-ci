# Tree Search Grouped Workflow Consolidation

**Date**: 2026-05-12 **Status**: Draft **Supersedes**:
2026-05-11-eliminate-tree-search-patches-design.md

## Problem

Tree search training currently requires 4 auxiliary files and monkey-patching:

| File                   | Purpose                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `proxy_workflow.py`    | `QueryIDProxyWorkflow` — injects query_id, does tree ops, returns tensor dicts        |
| `grouped_workflow.py`  | `TreeSearchGroupedRolloutWorkflow` — groups episodes, collects Nodes                  |
| `workflow_executor.py` | `TreeSearchWorkflowExecutor` — handles `list[Node]` returns                           |
| `patches.py`           | `TreeSearchPatches` — monkey-patches `_wrap_openai_agent` to use QueryIDProxyWorkflow |

Additionally, `CacheAwarePPOTrainer` carries heavy cache logic
(`_cache_aware_prepare_batch`, `_CacheAwareBatchBuilder`, `_tensor_dicts_to_nodes`,
`_mark_batch_trained`) that uses an all-or-nothing strategy instead of partial cache
reuse.

**Goals**:

1. Consolidate all tree-search workflow logic into a single
   `TreeSearchGroupedRolloutWorkflow` class
1. Eliminate all monkey-patches by using a `.env` flag in `_resolve_workflow`
1. Move cache logic into the workflow with partial cache reuse (e.g. need 4, have 3
   cached, generate 1)
1. Make the workflow self-contained — it loads/saves tree_store from a checkpoint
   directory
1. Drastically simplify `CacheAwarePPOTrainer`

## Design

### Data flow

**Before** (4 files, patches, trainer-side cache):

```
Trainer._cache_aware_prepare_batch → cache split (all-or-nothing), tree ops
  → TreeSearchPatches.apply()
  → patches._wrap_openai_agent → QueryIDProxyWorkflow
  → TreeSearchGroupedRolloutWorkflow wraps QueryIDProxyWorkflow
  → TreeSearchWorkflowExecutor handles list[Node] returns
  → trainer does tree insert, advantage, mark-trained, checkpoint
```

**After** (1 class, no patches, workflow-side cache):

```
Trainer.train() → base prepare_batch (no cache logic)
  → _resolve_workflow reads use_TreeSearchGroupedRolloutWorkflow from .env
  → TreeSearchGroupedRolloutWorkflow(OpenAIProxyWorkflow)
    - loads tree_store from checkpoint_dir
    - cache lookup: how many untrained episodes exist for this query?
    - generates only the needed fresh episodes
    - converts fresh results → Nodes, loads cached Nodes
    - inserts fresh Nodes into tree_store
    - computes tree advantages (if TREE mode)
    - marks all episodes as trained
    - saves tree checkpoint
    - returns batched tensor dict
```

### 1. TreeSearchGroupedRolloutWorkflow

New class in `customized_areal/tree_search/tree_search_grouped_workflow.py`. Extends
`GroupedRolloutWorkflow`.

**Constructor**:

```python
class TreeSearchGroupedRolloutWorkflow(GroupedRolloutWorkflow):
    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        checkpoint_dir: str,
        advantage_mode: AdvantageMode,
        loss_mode: LossMode,
        cache_mode: CacheMode,
    ):
```

`group_size` serves as the number of episodes needed per query (previously called
`n_samples`).

On init:

1. Create `MCTSTreeStore` and `TreeAdvantageComputer`
1. Create `TreeCheckpointManager(checkpoint_dir)`
1. Load existing tree checkpoint if present (CROSS_TRAINING mode)
1. Reset trained flags for fresh training
1. Store config for advantage/loss/cache modes

**`arun_episode(engine, data)` flow**:

1. Get `query_id` from `data["query_id"]`
1. `cached_count = tree_store.get_untrained_count(query_id)` (0 if no query_id)
1. `need_gen = max(0, group_size - cached_count)`
1. If `need_gen > 0`:
   - Run `need_gen` fresh episodes via `self.workflow.arun_episode(engine, data)` using
     `asyncio.gather`
   - Convert results to `list[Node]` using `interactions_dict_to_nodes`
   - Set `query_id`, `episode_id`, `turn_idx` on fresh nodes
1. Load `cached_count` untrained nodes from
   `tree_store.load_trajectories(query_id, cached_count)`
1. Combine: `all_nodes = fresh_nodes + cached_nodes` (total = `group_size`)
1. Insert fresh nodes into tree_store: `tree_store.insert_batch(fresh_nodes)`
1. Compute tree advantages (if `advantage_mode == TREE`):
   `advantage_computer.compute(all_nodes)`
1. Mark all `group_size` nodes as trained: `tree_store.set_trained(node_id, True)` for
   each
1. Save tree checkpoint (if `cache_mode == CROSS_TRAINING`)
1. Convert `all_nodes` to batched tensor dict via `_nodes_to_batched_tensor_dict`
1. Inject distill loss weights if `loss_mode != GRPO`
1. Return batched tensor dict

**Key difference from current**: When `need_gen < group_size`, we generate only the
missing episodes and combine with cached ones. The current all-or-nothing strategy
regenerates all episodes even when some are cached.

### 2. Utilities moved from proxy_workflow.py

These functions move to `tree_search_grouped_workflow.py`:

- `interactions_dict_to_nodes(interactions: dict[str, Any]) -> list[Node]` — converts
  `InteractionWithTokenLogpReward` dicts to `list[Node]`
- `_nodes_to_batched_tensor_dict(nodes: list[Node]) -> dict[str, Any] | None` — converts
  `list[Node]` to batched tensor dict

No changes to their implementation — just moved to the new file.

### 3. .env configuration

New variables in `customized_areal/.env`:

```
use_TreeSearchGroupedRolloutWorkflow=True
TREE_SEARCH_CHECKPOINT_DIR=/path/to/checkpoints
TREE_SEARCH_ADVANTAGE_MODE=TREE
TREE_SEARCH_LOSS_MODE=GRPO
TREE_SEARCH_CACHE_MODE=CROSS_TRAINING
```

`group_size` is passed from the existing engine config (the same `group_size` parameter
that currently goes to `GroupedRolloutWorkflow`).

All tree search configuration is self-contained in `.env`. The trainer no longer needs
`RolloutCacheConfig` or `TreeBackupConfig` for workflow logic — only
`_create_train_engine` needs `loss_mode` to decide whether to use
`MultiCandidateFSDPPPOActor`.

### 4. \_resolve_workflow modification

In `areal/infra/remote_inf_engine.py`, `_resolve_workflow` reads the `.env` flag:

```python
# At the group_size > 1 wrapping point:
if group_size > 1:
    use_tree_search = os.getenv("use_TreeSearchGroupedRolloutWorkflow", "False").lower() == "true"
    if use_tree_search:
        from customized_areal.tree_search.tree_search_grouped_workflow import TreeSearchGroupedRolloutWorkflow
        resolved = TreeSearchGroupedRolloutWorkflow(
            resolved, group_size,
            checkpoint_dir=os.getenv("TREE_SEARCH_CHECKPOINT_DIR", ""),
            advantage_mode=AdvantageMode(os.getenv("TREE_SEARCH_ADVANTAGE_MODE", "GAE")),
            loss_mode=LossMode(os.getenv("TREE_SEARCH_LOSS_MODE", "GRPO")),
            cache_mode=CacheMode(os.getenv("TREE_SEARCH_CACHE_MODE", "OFF")),
        )
    else:
        resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)
```

The `.env` file is loaded once at module import or at the top of `_resolve_workflow`
using `dotenv.load_dotenv`.

### 5. Simplified CacheAwarePPOTrainer

**Before**: 300+ lines with `_cache_aware_prepare_batch`, `_CacheAwareBatchBuilder`,
`_tensor_dicts_to_nodes`, `_mark_batch_trained`, `_init_tree_components`,
`TreeSearchPatches`, monkey-patching `prepare_batch`, `_save_recover_checkpoint`.

**After**:

```python
class CacheAwarePPOTrainer(PPOTrainer):
    """PPOTrainer with tree-search-aware rollout via .env flag.

    All cache logic, tree ops, and checkpoint saving happen inside
    TreeSearchGroupedRolloutWorkflow (activated by .env flag).
    This class only overrides _create_train_engine to use
    MultiCandidateFSDPPPOActor when distill loss is enabled.
    """

    def __init__(self, config, cache_config=None, tree_backup_config=None, ...):
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()
        super().__init__(config, train_dataset, valid_dataset)

    def _create_train_engine(self, actor_config, alloc):
        """Override to use MultiCandidateFSDPPPOActor when distill loss is enabled."""
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            if alloc.backend != "fsdp":
                raise ValueError(
                    f"Distillation loss mode requires FSDP backend, got: {alloc.backend}"
                )
            from customized_areal.tree_search.engine import MultiCandidateFSDPPPOActor
            actor_cls = MultiCandidateFSDPPPOActor
            if is_single_controller():
                actor = actor_cls.as_controller(actor_config, self.scheduler)
            else:
                actor = actor_cls(config=actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            return actor
        return super()._create_train_engine(actor_config, alloc)

    def train(self, **kwargs):
        # Apply distill loss patch if needed, no workflow patches
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            from customized_areal.tree_search.training.actor import (
                patch_ppo_actor_class_to_use_distill_loss,
                unpatch_ppo_actor_distill_loss,
            )
            patch_ppo_actor_class_to_use_distill_loss()
            try:
                return super().train(**kwargs)
            finally:
                unpatch_ppo_actor_distill_loss()
        return super().train(**kwargs)
```

**Removed from trainer**:

- `_cache_aware_prepare_batch`
- `_CacheAwareBatchBuilder`
- `_tensor_dicts_to_nodes`
- `_mark_batch_trained`
- `_init_tree_components`
- `tree_store`, `tree_advantage_computer`, `tree_checkpoint_manager`, `_batch_builder`,
  `_patches` attributes
- `_save_recover_checkpoint` override
- `prepare_batch` monkey-patching in `train()`
- `TreeSearchPatches` import and usage (distill loss patch is called directly via
  `patch_ppo_actor_class_to_use_distill_loss` / `unpatch_ppo_actor_distill_loss` when
  `loss_mode != GRPO`)

### 6. Files deleted

| File                                                | Reason                                                                                                                                                                                                                                  |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/proxy_workflow.py`    | `QueryIDProxyWorkflow` absorbed into `TreeSearchGroupedRolloutWorkflow`                                                                                                                                                                 |
| `customized_areal/tree_search/grouped_workflow.py`  | Old `TreeSearchGroupedRolloutWorkflow` replaced                                                                                                                                                                                         |
| `customized_areal/tree_search/workflow_executor.py` | `TreeSearchWorkflowExecutor` no longer needed — base `WorkflowExecutor` handles tensor dicts                                                                                                                                            |
| `customized_areal/tree_search/patches.py`           | `TreeSearchPatches` deleted. The distill loss patch (`patch_ppo_actor_class_to_use_distill_loss` / `unpatch_ppo_actor_distill_loss`) stays in `customized_areal/tree_search/training/actor.py` and is called directly from the trainer. |

### 7. Files created

| File                                                           | Purpose                                                                 |
| -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `customized_areal/tree_search/tree_search_grouped_workflow.py` | New `TreeSearchGroupedRolloutWorkflow` with cache, tree ops, checkpoint |

### 8. Files modified

| File                                      | Change                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------ |
| `areal/infra/remote_inf_engine.py`        | `_resolve_workflow` reads `.env` flag, wraps with `TreeSearchGroupedRolloutWorkflow` |
| `customized_areal/tree_search/trainer.py` | Remove all cache/tree/patch logic, keep only `_create_train_engine`                  |
| `customized_areal/.env`                   | Add tree search config variables                                                     |

### 9. Group size handling

`TreeSearchGroupedRolloutWorkflow` extends `GroupedRolloutWorkflow` and overrides
`arun_episode`. The inner `self.workflow` is the base `OpenAIProxyWorkflow`. When
`group_size > 1`, the workflow handles grouping internally — it runs `need_gen` episodes
via `asyncio.gather` (where `need_gen = group_size - cached_count`). The caller
(`_resolve_workflow`) does not apply a second `GroupedRolloutWorkflow` wrapper.

### 10. Error handling

- If `tree_store.get_untrained_count` returns 0 or `query_id` is empty, the workflow
  generates all `group_size` episodes (same as current no-cache behavior)
- If all fresh episodes fail (all exceptions in `asyncio.gather`), the workflow returns
  `None`
- If some fresh episodes fail, the workflow uses the remaining valid results plus cached
  nodes

### 11. Mark-trained and checkpoint

Both happen inside `arun_episode`:

- After tree insertion and advantage computation, all `group_size` nodes (fresh +
  cached) are marked as trained
- After marking trained, tree checkpoint is saved if `cache_mode == CROSS_TRAINING`
- This ensures the checkpoint is always up-to-date after each batch

# Tree Search Grouped Workflow Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate 4 tree-search files (proxy_workflow.py, grouped_workflow.py,
workflow_executor.py, patches.py) into 1 new file (tree_search_grouped_workflow.py),
move cache logic into the workflow with partial reuse, eliminate all workflow/executor
monkey-patches via .env flag, and simplify the trainer to only handle distill loss
patching and actor creation.

**Architecture:** `TreeSearchGroupedRolloutWorkflow` extends `GroupedRolloutWorkflow`,
wraps the base `OpenAIProxyWorkflow`, and is self-contained: it loads/saves tree_store
from checkpoint_dir, does cache lookup, generates only needed fresh episodes, combines
cached+fresh, does tree ops, and returns batched tensor dicts. `_resolve_workflow` reads
`.env` flag to decide between `GroupedRolloutWorkflow` and
`TreeSearchGroupedRolloutWorkflow`. The distill loss patch stays in `training/actor.py`
and is called directly from the trainer.

**Tech Stack:** Python 3.12+, PyTorch, asyncio

______________________________________________________________________

### Task 1: Create `tree_search_grouped_workflow.py` with utilities

**Files:**

- Create: `customized_areal/tree_search/tree_search_grouped_workflow.py`

- [ ] **Step 1: Create the file with `interactions_dict_to_nodes` and
  `_nodes_to_batched_tensor_dict`**

Copy these two functions verbatim from `proxy_workflow.py` into the new file. Update
imports to use the new module path. Remove the `PATCH_VERIFICATION` debug logs.

```python
# customized_areal/tree_search/tree_search_grouped_workflow.py
"""Tree-search-aware grouped rollout workflow with cache reuse.

Consolidates the functionality of QueryIDProxyWorkflow,
TreeSearchGroupedRolloutWorkflow, and TreeSearchWorkflowExecutor into
a single class that:
- Loads/saves tree_store from a checkpoint directory
- Does per-query cache lookup to determine how many fresh episodes are needed
- Generates only the needed fresh episodes (partial cache reuse)
- Converts fresh results to Nodes, loads cached Nodes, combines them
- Inserts fresh Nodes into tree_store, computes advantages, marks trained
- Saves tree checkpoint
- Returns batched tensor dicts that the base WorkflowExecutor handles natively
"""

from __future__ import annotations

import uuid
from typing import Any

from customized_areal.tree_search.config import AdvantageMode, CacheMode, LossMode
from customized_areal.tree_search.mcts_tree_store import Node

from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")


def interactions_dict_to_nodes(interactions: dict[str, Any]) -> list[Node]:
    """Convert dict[str, InteractionWithTokenLogpReward] to list[Node].

    Each interaction becomes one Node representing a single turn.
    """
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    nodes: list[Node] = []

    for turn_idx, (interaction_id, interaction) in enumerate(
        interactions.items(), start=1
    ):
        if not isinstance(interaction, InteractionWithTokenLogpReward):
            logger.warning(
                "Skipping interaction %s (type=%s, expected InteractionWithTokenLogpReward)",
                interaction_id,
                type(interaction).__name__,
            )
            continue
        resp = interaction.model_response
        if resp is None:
            logger.warning(
                "Skipping interaction %s: model_response is None",
                interaction_id,
            )
            continue

        seq_tokens = resp.input_tokens + resp.output_tokens

        if (
            interaction.chat_template_type == "concat"
            and interaction.parent is not None
        ):
            parent_res = interaction.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)

            if resp.input_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
            else:
                logger.error(
                    "concat mode: resp.input_len (%d) <= parent_len (%d) — "
                    "expected monotonic growth. Zero-filling prompt context.",
                    resp.input_len,
                    parent_len,
                )
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

        outcome_reward = interaction.reward if interaction.reward is not None else 0.0

        topk_ids: list[list[int]] = []
        topk_logp: list[list[float]] = []
        if resp.output_top_logprobs is not None:
            for pos_logprobs in resp.output_top_logprobs:
                ids = []
                logps = []
                for token_id, lp in pos_logprobs:
                    ids.append(token_id)
                    logps.append(lp)
                topk_ids.append(ids)
                topk_logp.append(logps)

        pn_id: str | None = None
        if interaction.parent is not None:
            pn_id = interaction.parent.interaction_id

        node = Node(
            input_ids=seq_tokens,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            turn_idx=turn_idx,
            node_id=interaction_id,
            parent_node_id=pn_id,
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
        )

        nodes.append(node)

    return nodes


def _nodes_to_batched_tensor_dict(nodes: list[Node]) -> dict[str, Any] | None:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    Returns None if nodes is empty.
    """
    if not nodes:
        return None

    from areal.utils.data import concat_padded_tensors
    from customized_areal.tree_search.mcts_tree_store import _node_to_tensor_dict

    tensor_dicts = [
        _node_to_tensor_dict(
            node,
            query_id=node.query_id or "",
            node_id=node.node_id,
        )
        for node in nodes
    ]
    return concat_padded_tensors(tensor_dicts)
```

- [ ] **Step 2: Verify the file has no syntax errors**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/tree_search_grouped_workflow.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py
git commit -m "feat: add tree_search_grouped_workflow.py with moved utilities"
```

______________________________________________________________________

### Task 2: Add `TreeSearchGroupedRolloutWorkflow` class

**Files:**

- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py`

- [ ] **Step 1: Add the class after the `_nodes_to_batched_tensor_dict` function**

```python
class TreeSearchGroupedRolloutWorkflow(RolloutWorkflow):
    """GroupedRolloutWorkflow with tree-search cache reuse, tree ops, and checkpoint.

    Wraps the base OpenAIProxyWorkflow and overrides arun_episode to:
    1. Check cache: how many untrained episodes exist for this query?
    2. Generate only the needed fresh episodes (group_size - cached_count)
    3. Convert fresh results to Nodes, load cached Nodes
    4. Combine cached + fresh Nodes (total = group_size)
    5. Insert fresh Nodes into tree_store
    6. Compute tree advantages (if advantage_mode == TREE)
    7. Mark all nodes as trained
    8. Save tree checkpoint (if cache_mode == CROSS_TRAINING)
    9. Return batched tensor dict
    """

    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        checkpoint_dir: str,
        advantage_mode: AdvantageMode,
        loss_mode: LossMode,
        cache_mode: CacheMode,
        rl_loss_weight: float = 1.0,
        distill_loss_weight: float = 0.005,
    ) -> None:
        from customized_areal.tree_search.advantage import TreeAdvantageComputer
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager
        from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore

        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.advantage_mode = advantage_mode
        self.loss_mode = loss_mode
        self.cache_mode = cache_mode
        self.rl_loss_weight = rl_loss_weight
        self.distill_loss_weight = distill_loss_weight

        self.tree_store = MCTSTreeStore()
        self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
        self.tree_checkpoint_manager = TreeCheckpointManager(checkpoint_dir)

        # Load existing tree checkpoint if present (CROSS_TRAINING mode)
        if self.cache_mode == CacheMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load()
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")

        # Reset trained flags for a fresh training run
        self.tree_store.reset_trained_flags()

    def _result_to_nodes(self, result: Any, query_id: str, group_idx: int) -> list[Node] | None:
        """Convert a single arun_episode result to list[Node]."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        if isinstance(result, dict) and all(
            isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
        ):
            nodes = interactions_dict_to_nodes(result)
        elif (
            isinstance(result, list)
            and result
            and isinstance(result[0], InteractionWithTokenLogpReward)
        ):
            converted = {str(i): v for i, v in enumerate(result)}
            nodes = interactions_dict_to_nodes(converted)
        else:
            return None

        episode_id = (
            f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
            if query_id
            else f"{group_idx}_{uuid.uuid4().hex[:8]}"
        )
        for turn_idx, node in enumerate(nodes, start=1):
            node.episode_id = episode_id
            node.query_id = query_id
            if not node.turn_idx:
                node.turn_idx = turn_idx
        return nodes

    async def arun_episode(
        self, engine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        query_id = data.get("query_id") or ""

        # 1. Check cache
        cached_count = (
            self.tree_store.get_untrained_count(query_id) if query_id else 0
        )
        need_gen = max(0, self.group_size - cached_count)

        logger.info(
            "TreeSearchGroupedWorkflow: query_id=%s, group_size=%d, "
            "cached=%d, need_gen=%d",
            query_id,
            self.group_size,
            cached_count,
            need_gen,
        )

        # 2. Generate fresh episodes if needed
        fresh_nodes: list[Node] = []
        if need_gen > 0:
            import asyncio

            results = await asyncio.gather(
                *[
                    self.workflow.arun_episode(engine, data)
                    for _ in range(need_gen)
                ],
                return_exceptions=True,
            )

            for group_idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Episode %d failed: %s", group_idx, result
                    )
                    continue
                if result is None:
                    continue
                nodes = self._result_to_nodes(result, query_id, group_idx)
                if nodes:
                    fresh_nodes.extend(nodes)

        # 3. Load cached nodes
        cached_nodes: list[Node] = []
        if cached_count > 0 and query_id:
            cached_nodes = self.tree_store.load_trajectories(
                query_id, cached_count
            )

        # 4. Combine
        all_nodes = fresh_nodes + cached_nodes

        if not all_nodes:
            return None

        # 5. Insert fresh nodes into tree
        if fresh_nodes:
            self.tree_store.insert_batch(fresh_nodes)

        # 6. Compute tree advantages
        if self.advantage_mode == AdvantageMode.TREE:
            self.tree_advantage_computer.compute(all_nodes)

        # 7. Mark all nodes as trained
        for node in all_nodes:
            if node.node_id:
                self.tree_store.set_trained(node.node_id, True)

        # 8. Save tree checkpoint
        if self.cache_mode == CacheMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)

        # 9. Convert to batched tensor dict
        result_dict = _nodes_to_batched_tensor_dict(all_nodes)

        # 10. Inject distill loss weights
        if result_dict is not None and self.loss_mode != LossMode.GRPO:
            if self.loss_mode == LossMode.DISTILL:
                result_dict["rl_loss_weight"] = 0.0
            else:
                result_dict["rl_loss_weight"] = self.rl_loss_weight
            result_dict["distill_loss_weight"] = self.distill_loss_weight

        return result_dict
```

Add the missing import at the top of the file:

```python
from areal.api import InferenceEngine, RolloutWorkflow
```

- [ ] **Step 2: Verify the file has no syntax errors**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/tree_search_grouped_workflow.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py
git commit -m "feat: add TreeSearchGroupedRolloutWorkflow with cache reuse and tree ops"
```

______________________________________________________________________

### Task 3: Add `.env` variables for tree search config

**Files:**

- Modify: `customized_areal/.env`

- [ ] **Step 1: Add tree search config variables to .env**

Append to `customized_areal/.env`:

```
# Tree search workflow configuration
# When True, _resolve_workflow wraps with TreeSearchGroupedRolloutWorkflow instead of GroupedRolloutWorkflow
use_TreeSearchGroupedRolloutWorkflow=False
TREE_SEARCH_CHECKPOINT_DIR=
TREE_SEARCH_ADVANTAGE_MODE=TREE
TREE_SEARCH_LOSS_MODE=GRPO
TREE_SEARCH_CACHE_MODE=CROSS_TRAINING
TREE_SEARCH_RL_LOSS_WEIGHT=1.0
TREE_SEARCH_DISTILL_LOSS_WEIGHT=0.005
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/.env
git commit -m "feat: add tree search .env variables for workflow configuration"
```

______________________________________________________________________

### Task 4: Modify `_resolve_workflow` in `remote_inf_engine.py`

**Files:**

- Modify: `areal/infra/remote_inf_engine.py:698-701`

- [ ] **Step 1: Replace the group_size > 1 wrapping with .env-aware logic**

Replace lines 698-701:

```python
        # Wrap with GroupedRolloutWorkflow if group_size > 1
        if group_size > 1:
            resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)

        return resolved
```

With:

```python
        # Wrap with GroupedRolloutWorkflow if group_size > 1
        if group_size > 1:
            use_tree_search = os.getenv(
                "use_TreeSearchGroupedRolloutWorkflow", "False"
            ).lower() == "true"
            if use_tree_search:
                from dotenv import load_dotenv

                # Load .env from the project root (where customized_areal/ lives)
                _project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                _env_path = os.path.join(_project_root, "customized_areal", ".env")
                if os.path.isfile(_env_path):
                    load_dotenv(_env_path, override=False)

                from customized_areal.tree_search.config import (
                    AdvantageMode,
                    CacheMode,
                    LossMode,
                )
                from customized_areal.tree_search.tree_search_grouped_workflow import (
                    TreeSearchGroupedRolloutWorkflow,
                )

                checkpoint_dir = os.getenv("TREE_SEARCH_CHECKPOINT_DIR", "")
                advantage_mode = AdvantageMode(
                    os.getenv("TREE_SEARCH_ADVANTAGE_MODE", "GAE")
                )
                loss_mode = LossMode(os.getenv("TREE_SEARCH_LOSS_MODE", "GRPO"))
                cache_mode = CacheMode(
                    os.getenv("TREE_SEARCH_CACHE_MODE", "OFF")
                )
                rl_loss_weight = float(
                    os.getenv("TREE_SEARCH_RL_LOSS_WEIGHT", "1.0")
                )
                distill_loss_weight = float(
                    os.getenv("TREE_SEARCH_DISTILL_LOSS_WEIGHT", "0.005")
                )

                resolved = TreeSearchGroupedRolloutWorkflow(
                    resolved,
                    group_size,
                    checkpoint_dir=checkpoint_dir,
                    advantage_mode=advantage_mode,
                    loss_mode=loss_mode,
                    cache_mode=cache_mode,
                    rl_loss_weight=rl_loss_weight,
                    distill_loss_weight=distill_loss_weight,
                )
            else:
                resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)

        return resolved
```

Also add `from dotenv import load_dotenv` is done inline (lazy import). No top-level
import changes needed since `os` is already imported.

- [ ] **Step 2: Verify no syntax errors**

Run:
`python -c "import ast; ast.parse(open('areal/infra/remote_inf_engine.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add areal/infra/remote_inf_engine.py
git commit -m "feat: _resolve_workflow reads .env flag to use TreeSearchGroupedRolloutWorkflow"
```

______________________________________________________________________

### Task 5: Simplify `CacheAwarePPOTrainer`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Replace the entire file with the simplified trainer**

Write the complete new `trainer.py`:

```python
# customized_areal/tree_search/trainer.py
"""PPOTrainer with tree-search-aware rollout via .env flag.

All cache logic, tree ops, and checkpoint saving happen inside
TreeSearchGroupedRolloutWorkflow (activated by .env flag
use_TreeSearchGroupedRolloutWorkflow=True in customized_areal/.env).

This class only overrides:
- _create_train_engine: uses MultiCandidateFSDPPPOActor when distill loss
  is enabled
- train: applies/restores the distill loss PPOActor patch when loss_mode
  != GRPO
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.config import (
    LossMode,
    TreeBackupConfig,
)

from areal import PPOTrainer
from areal.utils import logging
from areal.utils.environ import is_single_controller

logger = logging.getLogger("TreeBackupPPOTrainer")


class CacheAwarePPOTrainer(PPOTrainer):
    """PPOTrainer with tree-search-aware rollout via .env flag.

    All cache logic, tree ops, and checkpoint saving happen inside
    TreeSearchGroupedRolloutWorkflow (activated by .env flag).
    This class only overrides _create_train_engine to use
    MultiCandidateFSDPPPOActor when distill loss is enabled, and
    applies the distill loss patch in train().
    """

    def __init__(
        self,
        config: Any,
        cache_config: Any | None = None,
        tree_backup_config: TreeBackupConfig | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
    ):
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()
        super().__init__(config, train_dataset, valid_dataset)

    def _create_train_engine(self, actor_config, alloc):
        """Override to use MultiCandidateFSDPPPOActor when distill loss is enabled."""
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            if alloc.backend != "fsdp":
                raise ValueError(
                    f"Distillation loss mode requires FSDP backend, "
                    f"got: {alloc.backend}"
                )
            from customized_areal.tree_search.engine import (
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

    def train(
        self,
        workflow=None,
        eval_workflow=None,
        workflow_kwargs=None,
        eval_workflow_kwargs=None,
        dynamic_filter_fn=None,
        total_epochs=None,
    ):
        """Train with distill loss patch applied if needed."""
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            from customized_areal.tree_search.training.actor import (
                patch_ppo_actor_class_to_use_distill_loss,
                unpatch_ppo_actor_distill_loss,
            )

            patch_ppo_actor_class_to_use_distill_loss()
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
                unpatch_ppo_actor_distill_loss()
        return super().train(
            workflow=workflow,
            eval_workflow=eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=dynamic_filter_fn,
            total_epochs=total_epochs,
        )

    def close(self) -> None:
        super().close()
```

- [ ] **Step 2: Verify no syntax errors**

Run:
`python -c "import ast; ast.parse(open('customized_areal/tree_search/trainer.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor: simplify CacheAwarePPOTrainer — remove cache logic, tree ops, patches"
```

______________________________________________________________________

### Task 6: Update `__init__.py` exports

**Files:**

- Modify: `customized_areal/tree_search/__init__.py`

- [ ] **Step 1: Replace `QueryIDProxyWorkflow` export with
  `TreeSearchGroupedRolloutWorkflow`**

Replace:

```python
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow
```

With:

```python
from customized_areal.tree_search.tree_search_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
```

And in `__all__`, replace `"QueryIDProxyWorkflow"` with
`"TreeSearchGroupedRolloutWorkflow"`.

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/__init__.py
git commit -m "refactor: update __init__.py exports for TreeSearchGroupedRolloutWorkflow"
```

______________________________________________________________________

### Task 7: Update test files for deleted modules

**Files:**

- Modify: `tests/test_treesearch_patches.py`

- Modify: `tests/test_treesearch_bugfixes.py`

- Modify: `tests/test_tree_search/test_trainer.py`

- [ ] **Step 1: Delete `tests/test_treesearch_patches.py`**

This file tests `TreeSearchPatches` which is being deleted. All its test cases
(apply/restore, idempotency, context manager, tree search wrap) are no longer relevant
since the `.env` flag replaces all patches.

Run: `git rm tests/test_treesearch_patches.py`

- [ ] **Step 2: Update `tests/test_treesearch_bugfixes.py`**

In `TestTurnIdxInInteractionsToNodes`, replace the import and test that uses
`QueryIDProxyWorkflow._interactions_to_nodes` with a direct call to the new module's
`interactions_dict_to_nodes`:

Replace the entire `TestTurnIdxInInteractionsToNodes` class:

```python
class TestTurnIdxInInteractionsToNodes:
    """tree_search_grouped_workflow.interactions_dict_to_nodes sets turn_idx 1-based."""

    def test_interactions_to_nodes_sets_turn_idx(self):
        from unittest.mock import MagicMock

        from customized_areal.tree_search.tree_search_grouped_workflow import (
            interactions_dict_to_nodes,
        )
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        def make_interaction():
            inter = MagicMock(spec=InteractionWithTokenLogpReward)
            inter.chat_template_type = "individual"
            inter.parent = None
            inter.reward = 1.0
            resp = MagicMock()
            resp.input_tokens = [1, 2]
            resp.output_tokens = [3, 4]
            resp.input_ids = [1, 2]
            resp.output_ids = [3, 4]
            resp.input_len = 2
            resp.output_len = 2
            resp.output_logprobs = [-0.5, -0.3]
            resp.output_versions = [0, 0]
            resp.output_top_logprobs = None
            inter.model_response = resp
            return inter

        interactions = {"turn_a": make_interaction(), "turn_b": make_interaction()}
        nodes = interactions_dict_to_nodes(interactions)

        assert len(nodes) == 2
        assert nodes[0].turn_idx == 1
        assert nodes[1].turn_idx == 2
```

- [ ] **Step 3: Delete `tests/test_tree_search/test_trainer.py`**

This file tests `TreeSearchPatches` apply/restore behavior (which is deleted). The new
trainer has no patches to test.

Run: `git rm tests/test_tree_search/test_trainer.py`

- [ ] **Step 4: Run remaining tests to verify no regressions**

Run:
`python -m pytest tests/test_treesearch_bugfixes.py tests/test_tree_search/test_advantage.py tests/test_tree_search/test_checkpoint.py tests/test_tree_search/test_mcts_tree_store.py -v`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_treesearch_bugfixes.py
git rm tests/test_treesearch_patches.py tests/test_tree_search/test_trainer.py
git commit -m "test: update tests for removed patches and deleted modules"
```

______________________________________________________________________

### Task 8: Delete old files

**Files:**

- Delete: `customized_areal/tree_search/proxy_workflow.py`

- Delete: `customized_areal/tree_search/grouped_workflow.py`

- Delete: `customized_areal/tree_search/workflow_executor.py`

- Delete: `customized_areal/tree_search/patches.py`

- [ ] **Step 1: Delete the 4 files**

```bash
git rm customized_areal/tree_search/proxy_workflow.py
git rm customized_areal/tree_search/grouped_workflow.py
git rm customized_areal/tree_search/workflow_executor.py
git rm customized_areal/tree_search/patches.py
```

- [ ] **Step 2: Verify no remaining imports reference deleted modules**

Run:
`grep -r "from customized_areal.tree_search.proxy_workflow\|from customized_areal.tree_search.grouped_workflow\|from customized_areal.tree_search.workflow_executor\|from customized_areal.tree_search.patches" --include="*.py" .`

Expected: No results (all references updated in previous tasks).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: delete proxy_workflow, grouped_workflow, workflow_executor, patches"
```

______________________________________________________________________

### Task 9: Run pre-commit and final verification

**Files:**

- All modified files

- [ ] **Step 1: Run pre-commit**

Run: `pre-commit run --all-files`

Expected: All hooks pass (or fix any issues and re-run).

- [ ] **Step 2: Verify the new workflow module imports cleanly**

Run:
`python -c "from customized_areal.tree_search.tree_search_grouped_workflow import TreeSearchGroupedRolloutWorkflow, interactions_dict_to_nodes, _nodes_to_batched_tensor_dict; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Verify the simplified trainer imports cleanly**

Run:
`python -c "from customized_areal.tree_search.trainer import CacheAwarePPOTrainer; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Run tree search test suite**

Run: `python -m pytest tests/test_tree_search/ tests/test_treesearch_bugfixes.py -v`

Expected: All pass.

- [ ] **Step 5: Commit any pre-commit fixes**

```bash
git add -A
git commit -m "style: apply pre-commit fixes"
```

# Eliminate Tree Search Patches — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move tree insertion, advantage computation, and mark-trained into
`QueryIDProxyWorkflow`, return batched tensor dicts so base `WorkflowExecutor` handles
them natively, and eliminate patches on `grouped_workflow.py` and
`workflow_executor.py`.

**Architecture:** `QueryIDProxyWorkflow` gains `tree_store`, `advantage_computer`,
`advantage_mode` constructor args. After building `list[Node]` (existing logic), it
inserts into tree, computes advantages, marks trained, then converts `list[Node]` →
batched tensor dict via a new `_nodes_to_batched_tensor_dict` helper. The base
`WorkflowExecutor` sees a normal `dict[str, Any]` return and handles it natively. The
trainer no longer does tree insert/advantage/mark-trained/conversion.

**Tech Stack:** Python 3.12+, PyTorch

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py` — `_node_to_tensor_dict`
  metadata as lists
- Modify: `customized_areal/tree_search/proxy_workflow.py` — new args, tree ops, dict
  return
- Modify: `customized_areal/tree_search/patches.py` — remove 3 patches, simplify 1
- Modify: `customized_areal/tree_search/trainer.py` — remove tree ops from prepare_batch
- Delete: `customized_areal/tree_search/grouped_workflow.py`
- Delete: `customized_areal/tree_search/workflow_executor.py`
- Modify: `tests/test_tree_search/test_mcts_tree_store.py` — expect list metadata
- Modify: `tests/test_treesearch_bugfixes.py` — remove grouped_workflow tests
- Modify: `tests/test_treesearch_patches.py` — remove executor/resolve_workflow/GAE
  tests

______________________________________________________________________

### Task 1: Update `_node_to_tensor_dict` — metadata as single-element lists

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:111-167`

**Why:** `concat_padded_tensors` concatenates list keys across dicts (keeping the first
value for non-tensor non-list keys). To preserve per-turn `query_id`, `node_id`,
`episode_id`, `turn_idx` through concat, these must be single-element lists, matching
the pattern `InteractionWithTokenLogpReward.to_tensor_dict` already uses for `node_id`.

- [ ] **Step 1: Read the current `_node_to_tensor_dict` function**

Already done — the function is at `mcts_tree_store.py:111-167`.

- [ ] **Step 2: Edit `_node_to_tensor_dict` — change metadata to single-element lists**

In `customized_areal/tree_search/mcts_tree_store.py`, replace lines 125-126:

```python
# Before (lines 125-126):
"query_id": query_id,
"node_id": node_id,

# After:
"query_id": [query_id],
"node_id": [node_id],
"episode_id": [node.episode_id or ""],
"turn_idx": [node.turn_idx or 0],
```

And update the `_turn_idx_in_episode` line (164) to read from node:

Actually, leave the underscore-prefixed convenience fields (`_turn_id`,
`_parent_turn_id`, `_turn_reward`, `_outcome_reward`, `_episode_idx`,
`_turn_idx_in_episode`, `_num_turns_in_episode`) as they are — they're not
list-concatenated through `concat_padded_tensors` because `_node_to_tensor_dict` is
called on individual nodes, one dict at a time. Those underscore fields are passed
straight through.

- [ ] **Step 3: Run existing `_node_to_tensor_dict` tests to verify breakage**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestResponseOnlyFieldsSliced::test_response_only_fields_sliced tests/test_tree_search/test_mcts_tree_store.py::TestResponseOnlyFieldsSliced::test_logp_already_sliced -xvs
```

Expected: FAIL on assertion because tests read `result["query_id"]` expecting a string,
but now it's a list.

- [ ] **Step 4: Update tests to expect list values**

In `tests/test_tree_search/test_mcts_tree_store.py`, update the two test methods:

`test_response_only_fields_sliced` (around line 605) — add assertions after
`_node_to_tensor_dict` call:

```python
result = _node_to_tensor_dict(node, "q1", "t1")
# ... existing shape assertions ...
# New: metadata fields are single-element lists
assert result["query_id"] == ["q1"]
assert result["node_id"] == ["t1"]
assert result["episode_id"] == [""]  # default: node.episode_id is empty
assert result["turn_idx"] == [0]  # default: node.turn_idx is 0
```

`test_logp_already_sliced` (around line 628) — add similar assertions:

```python
result = _node_to_tensor_dict(node, "q1", "t1")
assert result["logp"].shape == (1, 3)
assert result["query_id"] == ["q1"]
assert result["node_id"] == ["t1"]
```

If node.episode_id and node.turn_idx are not passed to the `Node()` constructor in these
tests, they default to `""` and `0` respectively.

- [ ] **Step 5: Run updated tests**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py::TestResponseOnlyFieldsSliced -xvs
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "refactor: store query_id/node_id/episode_id/turn_idx as single-element lists in _node_to_tensor_dict

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 2: Add `_nodes_to_batched_tensor_dict` helper to proxy_workflow.py

**Files:**

- Modify: `customized_areal/tree_search/proxy_workflow.py`

**Why:** `QueryIDProxyWorkflow.arun_episode` will convert `list[Node]` to a single
batched tensor dict. This helper function encapsulates that logic: call
`_node_to_tensor_dict` per Node, then `concat_padded_tensors` to produce a
`[N, seq_len]` dict.

- [ ] **Step 1: Add the helper function**

Insert after the `interactions_dict_to_nodes` function (after line 156) in
`proxy_workflow.py`:

```python
def _nodes_to_batched_tensor_dict(nodes: list[Node]) -> dict[str, Any] | None:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    Metadata (query_id, node_id, episode_id, turn_idx) is stored as
    single-element lists in each per-Node dict so that concat_padded_tensors
    flat-concatenates them across nodes.

    Returns None if nodes is empty.
    """
    if not nodes:
        return None

    from areal.utils.data import concat_padded_tensors

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

- [ ] **Step 2: Verify the import works**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.proxy_workflow import _nodes_to_batched_tensor_dict; print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/proxy_workflow.py
git commit -m "feat: add _nodes_to_batched_tensor_dict helper to proxy_workflow

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 3: Update `QueryIDProxyWorkflow` — tree ops + dict return

**Files:**

- Modify: `customized_areal/tree_search/proxy_workflow.py`

**Why:** Move tree insertion, advantage computation, and mark-trained into
`QueryIDProxyWorkflow.arun_episode`. Change the return type from `list[Node] | None` to
`dict[str, Any] | None` so the base `WorkflowExecutor` handles it natively.

- [ ] **Step 1: Add new constructor args**

Add imports at the top of `proxy_workflow.py`:

```python
from customized_areal.tree_search.config import AdvantageMode
from customized_areal.tree_search.advantage import TreeAdvantageComputer
```

Update `QueryIDProxyWorkflow.__init__` to accept the new args:

```python
def __init__(
    self,
    agent_path: str | None = None,
    group_size: int = 1,
    tree_store: Any | None = None,
    advantage_computer: Any | None = None,
    advantage_mode: Any | None = None,
    **kwargs: Any,
) -> None:
    if "agent" not in kwargs and agent_path is not None:
        agent_cls = import_from_string(agent_path)
        kwargs["agent"] = agent_cls()
    self.group_size = group_size
    self.tree_store = tree_store
    self.advantage_computer = advantage_computer
    self.advantage_mode = advantage_mode
    kwargs.pop("group_size", None)
    logger.warning(...)  # existing debug log
    super().__init__(**kwargs)
```

- [ ] **Step 2: Add `_post_rollout_tree_ops` helper method**

```python
def _post_rollout_tree_ops(self, nodes: list[Node]) -> None:
    """Insert nodes into tree, compute advantages, mark trained."""
    if self.tree_store is None:
        return
    self.tree_store.insert_batch(nodes)
    if (
        self.advantage_computer is not None
        and self.advantage_mode is not None
        and self.advantage_mode == AdvantageMode.TREE
    ):
        self.advantage_computer.compute(nodes)
    for node in nodes:
        if node.node_id:
            self.tree_store.set_trained(node.node_id, True)
```

- [ ] **Step 3: Update `_async_single_episode` — convert to dict, call tree ops**

Replace the existing `_async_single_episode` method. The key change: after converting
interactions to `list[Node]` and setting metadata, call `_post_rollout_tree_ops` then
`_nodes_to_batched_tensor_dict`.

```python
async def _async_single_episode(
    self, engine, data: dict, query_id: str
) -> dict[str, Any] | None:
    """Run a single episode, do tree ops, return batched tensor dict."""
    result = await super().arun_episode(engine, data)

    if result is None:
        return None

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    if isinstance(result, dict) and all(
        isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
    ):
        nodes = self._interactions_to_nodes(result)
        episode_id = uuid.uuid4().hex
        for node in nodes:
            node.episode_id = episode_id
            node.query_id = query_id
    elif isinstance(result, list):
        logger.warning(
            "QueryIDProxyWorkflow: super().arun_episode returned list "
            "instead of dict; attempting dict conversion"
        )
        if result and isinstance(result[0], InteractionWithTokenLogpReward):
            converted = {str(i): v for i, v in enumerate(result)}
            nodes = self._interactions_to_nodes(converted)
            episode_id = uuid.uuid4().hex
            for node in nodes:
                node.episode_id = episode_id
                node.query_id = query_id
        else:
            return None
    else:
        if result is not None:
            logger.warning(
                "QueryIDProxyWorkflow: unexpected result type %s",
                type(result).__name__,
            )
        return None

    self._post_rollout_tree_ops(nodes)
    return _nodes_to_batched_tensor_dict(nodes)
```

- [ ] **Step 4: Update `arun_episode` — call tree ops in grouped path too, return dict**

Replace the existing `arun_episode`:

```python
async def arun_episode(self, engine, data: dict) -> dict[str, Any] | None:
    query_id = data.get("query_id") or ""

    logger.warning(
        "PATCH_VERIFICATION: QueryIDProxyWorkflow.arun_episode CALLED — "
        "class=%s, query_id=%s, engine_type=%s, group_size=%d",
        type(self).__name__,
        query_id,
        type(engine).__name__,
        self.group_size,
    )

    if self.group_size <= 1:
        return await self._async_single_episode(engine, data, query_id)

    import asyncio

    # Run group_size episodes, each returns dict[str, InteractionWithTokenLogpReward]
    results = await asyncio.gather(
        *[
            super().arun_episode(engine, data)
            for _ in range(self.group_size)
        ],
        return_exceptions=True,
    )

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    all_nodes: list[Node] = []
    for group_idx, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            continue
        if isinstance(result, dict) and all(
            isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
        ):
            nodes = self._interactions_to_nodes(result)
        elif isinstance(result, list) and result and isinstance(result[0], InteractionWithTokenLogpReward):
            converted = {str(i): v for i, v in enumerate(result)}
            nodes = self._interactions_to_nodes(converted)
        else:
            continue

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
        all_nodes.extend(nodes)

    if not all_nodes:
        return None

    self._post_rollout_tree_ops(all_nodes)
    return _nodes_to_batched_tensor_dict(all_nodes)
```

Remove the `_single_episode` (synchronous) stub method — it's not used.

- [ ] **Step 5: Remove the now-unnecessary `interactions_dict_to_nodes` import from
  patched executor**

No action needed — the function stays where it is, it's used by
`_interactions_to_nodes`.

- [ ] **Step 6: Verify import**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow; print('OK')"
```

Expected: OK

- [ ] **Step 7: Commit**

```bash
git add customized_areal/tree_search/proxy_workflow.py
git commit -m "feat: QueryIDProxyWorkflow does tree ops and returns batched tensor dict

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 4: Simplify `TreeSearchPatches` — remove 3 patches

**Files:**

- Modify: `customized_areal/tree_search/patches.py`

**Why:** Remove GAE backup/restore patch (advantages pre-computed in proxy_workflow),
\_resolve_workflow patch (no double-wrapping), workflow_executor patch (base executor
handles dict returns). Simplify \_wrap_openai_agent patch to create
`QueryIDProxyWorkflow` directly with tree_store/advantage_computer args.

- [ ] **Step 1: Remove imports of deleted modules**

Remove the imports for `TreeSearchGroupedRolloutWorkflow` and
`TreeSearchWorkflowExecutor` (lines 14-18):

```python
# Remove these lines:
from customized_areal.tree_search.grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.workflow_executor import TreeSearchWorkflowExecutor
```

No new imports needed — `QueryIDProxyWorkflow` is already imported (line 17 of the
current patches.py), and `tree_store`/`advantage_computer` are passed through as
`Any`-typed values.

- [ ] **Step 2: Add tree_store/advantage_computer to constructor**

Update `__init__` to accept `tree_store` and `advantage_computer`:

```python
def __init__(
    self,
    rollout_engine: Any,
    advantage_mode: AdvantageMode,
    loss_mode: LossMode,
    group_size: int,
    tree_store: Any | None = None,
    advantage_computer: Any | None = None,
):
    self._engine = self._unwrap_engine(rollout_engine)
    self._advantage_mode = advantage_mode
    self._loss_mode = loss_mode
    self._group_size = group_size
    self._tree_store = tree_store
    self._advantage_computer = advantage_computer
    self._saved: list[tuple[Any, str, Any]] = []
    self._distill_undo: Any = None
    self._applied = False
```

- [ ] **Step 3: Remove `_build_tree_backup_compute_advantages` method**

Delete the entire method (lines 111-144).

- [ ] **Step 4: Simplify `_build_tree_search_wrap`**

Remove the `TreeSearchGroupedRolloutWorkflow` wrapping. This method should now create
`QueryIDProxyWorkflow` directly:

```python
def _build_tree_search_wrap(self):
    """Build patched _wrap_openai_agent returning QueryIDProxyWorkflow."""
    engine = self._engine
    tree_store = self._tree_store
    advantage_computer = self._advantage_computer
    advantage_mode = self._advantage_mode
    group_size = self._group_size

    def _tree_search_wrap(agent, proxy_addr):
        agent_cfg = engine.config.agent
        if agent_cfg is None:
            raise RuntimeError(
                "config.agent is None; tree search workflow requires "
                "agent configuration. Set agent.mode in the config."
            )
        return QueryIDProxyWorkflow(
            mode=agent_cfg.mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=agent_cfg.admin_api_key,
            discount=agent_cfg.turn_discount,
            export_style=agent_cfg.export_style,
            subproc_max_workers=agent_cfg.subproc_max_workers,
            proxy_gateway_addr=getattr(engine, "_proxy_gateway_addr", None),
            group_size=group_size,
            tree_store=tree_store,
            advantage_computer=advantage_computer,
            advantage_mode=advantage_mode,
        )

    return _tree_search_wrap
```

- [ ] **Step 5: Remove `_build_patched_resolve` method**

Delete the entire method (lines 177-203). No longer needed since `QueryIDProxyWorkflow`
handles grouping internally and no `TreeSearchGroupedRolloutWorkflow` wrapper exists.

- [ ] **Step 6: Remove `_build_tree_search_executor` method**

Delete the entire method (lines 205-224). Base `WorkflowExecutor` handles dict returns
natively.

- [ ] **Step 7: Simplify `apply()` — remove the deleted patches**

Update `apply()` to only apply the \_wrap_openai_agent patch (and distill loss if
applicable). The new apply() should:

```python
def apply(self) -> None:
    if self._applied:
        logger.warning("TreeSearchPatches.apply() called twice; skipping")
        return

    try:
        _is_controller = hasattr(self._engine, "inf_engine")

        logger.warning(
            "PATCH_VERIFICATION: TreeSearchPatches.apply — "
            "engine_type=%s, has_inf_engine=%s, is_controller=%s, "
            "has_wrap_openai_agent=%s",
            type(self._engine).__name__,
            hasattr(self._engine, "inf_engine"),
            _is_controller,
            hasattr(self._engine, "_wrap_openai_agent"),
        )

        if not _is_controller:
            # Patch: engine._wrap_openai_agent
            if hasattr(self._engine, "_wrap_openai_agent"):
                self._save_and_set(
                    self._engine,
                    "_wrap_openai_agent",
                    self._build_tree_search_wrap(),
                )
                logger.warning(
                    "PATCH_VERIFICATION: _wrap_openai_agent patched on %s",
                    type(self._engine).__name__,
                )
            else:
                logger.warning(
                    "Engine has no _wrap_openai_agent method; "
                    "tree search workflow will not be available"
                )
        else:
            logger.info(
                "Engine is a RolloutController; skipping worker-side "
                "patches (remote engine). Trainer-side "
                "_tensor_dicts_to_nodes will convert tensor dicts to Nodes."
            )

        # Patch: distill loss (conditional)
        if self._loss_mode != LossMode.GRPO:
            from customized_areal.tree_search.training.actor import (
                patch_ppo_actor_class_to_use_distill_loss,
                unpatch_ppo_actor_distill_loss,
            )
            self._distill_undo = unpatch_ppo_actor_distill_loss
            patch_ppo_actor_class_to_use_distill_loss()

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
```

- [ ] **Step 8: Update `restore()` — remove `_original_compute_advantages` cleanup**

The `restore()` method stays mostly the same, except remove the
`PPOActor._original_compute_advantages` cleanup block (lines 355-356):

```python
# Remove:
# Clean up idempotency marker
if hasattr(PPOActor, "_original_compute_advantages"):
    del PPOActor._original_compute_advantages
```

- [ ] **Step 9: Run existing patches tests (expect failures)**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_patches.py -xvs
```

Expected: FAIL — tests still reference old behavior. We'll update tests in Task 7.

- [ ] **Step 10: Commit**

```bash
git add customized_areal/tree_search/patches.py
git commit -m "refactor: simplify TreeSearchPatches — remove GAE/executor/resolve patches

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 5: Simplify `CacheAwarePPOTrainer._cache_aware_prepare_batch`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:299-421`

**Why:** Tree insertion, advantage computation, mark-trained, and Node→tensor conversion
happen in `QueryIDProxyWorkflow` now. The trainer only needs to handle cache loading,
checkpoint saving, and distill loss weight injection.

- [ ] **Step 1: Update `_cache_aware_prepare_batch` — remove tree ops**

Replace the tree operations block (lines 378-408) with a simplified version. The "After"
section:

The tree operations section (lines 378-420) becomes:

```python
        # --- Checkpoint save (after trajectories are generated) ---
        if self.tree_backup_config.mode == CacheMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.debug("Saved MCTS tree checkpoint after rollout batch")

        # --- End checkpoint save ---

        # Trajectories are already batched tensor dicts with metadata.
        # Convert to tensor dicts for the downstream PPO pipeline.
        # (Nodes are constructed only when needed for cache operations)
        converted: list[dict[str, Any]] = []

        for t in trajs:
            if isinstance(t, Node):
                # Cache-loaded trajectories are still Node objects
                query_id = t.query_id
                node_id = t.node_id
                converted.append(_node_to_tensor_dict(t, query_id, node_id))
            else:
                # Rollout-generated trajectories are already tensor dicts
                converted.append(t)

        trajs = converted

        # Inject distillation loss weights into trajectory dicts
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            for traj in trajs:
                if self.tree_backup_config.loss_mode == LossMode.DISTILL:
                    traj["rl_loss_weight"] = 0.0
                else:
                    traj["rl_loss_weight"] = self.tree_backup_config.rl_loss_weight
                traj["distill_loss_weight"] = (
                    self.tree_backup_config.distill_loss_weight
                )

        return trajs
```

- [ ] **Step 2: Update rollout path — keep `_tensor_dicts_to_nodes` for controller
  fallback**

In `customized_areal/tree_search/trainer.py`, update lines 364-368. In single-controller
mode, `QueryIDProxyWorkflow` patches don't apply to the remote engine, so tree ops must
happen in the trainer. Detect this by checking if tensor dicts already have list-typed
metadata (proxy_workflow was active) or not (controller fallback):

```python
            # Tree ops (insert, advantage, mark-trained) happen inside
            # QueryIDProxyWorkflow when the _wrap_openai_agent patch is
            # active (direct engine mode). In single-controller mode the
            # patch can't reach the remote engine, so we fall back to
            # trainer-side conversion and tree ops.
            if trajs and isinstance(trajs[0], dict):
                has_metadata = isinstance(trajs[0].get("query_id"), list)
                if not has_metadata:
                    trajs = self._tensor_dicts_to_nodes(trajs, all_prompts)
                    self.tree_store.insert_batch(trajs)
                    if (
                        self.tree_backup_config.advantage_mode
                        == AdvantageMode.TREE
                    ):
                        self.tree_advantage_computer.compute(trajs)
                    _mark_batch_trained(self.tree_store, trajs)
            elif trajs and not isinstance(trajs[0], Node):
                trajs = self._tensor_dicts_to_nodes(trajs, all_prompts)
```

- [ ] **Step 3: Replace tree ops block (lines 378-420)**

Replace the block from "--- Tree operations ---" through the return statement with:

```python
        # --- Checkpoint save (after trajectories are generated) ---

        if self.tree_backup_config.mode == CacheMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.debug("Saved MCTS tree checkpoint after rollout batch")

        # --- End checkpoint save ---

        # Convert any remaining Node objects to tensor dicts.
        # Cache-loaded trajectories are Nodes; rollout-generated are dicts.
        converted: list[dict[str, Any]] = []
        for t in trajs:
            if isinstance(t, Node):
                converted.append(
                    _node_to_tensor_dict(t, t.query_id, t.node_id)
                )
            else:
                converted.append(t)
        trajs = converted

        # Inject distillation loss weights into trajectory dicts
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            for traj in trajs:
                if self.tree_backup_config.loss_mode == LossMode.DISTILL:
                    traj["rl_loss_weight"] = 0.0
                else:
                    traj["rl_loss_weight"] = self.tree_backup_config.rl_loss_weight
                traj["distill_loss_weight"] = (
                    self.tree_backup_config.distill_loss_weight
                )

        return trajs
```

- [ ] **Step 4: Run existing trainer tests**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_cache_trainer.py -xvs
```

Expected: Tests that mock the rollout to return Node objects may still pass (they test
cache splitting, not tree ops). If they fail, update mocks.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor: remove tree ops from CacheAwarePPOTrainer._cache_aware_prepare_batch

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 6: Update `CacheAwarePPOTrainer.train()` — pass tree_store/advantage_computer to patches

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:130-175`

**Why:** `TreeSearchPatches` now needs `tree_store` and `advantage_computer` to pass
through to `QueryIDProxyWorkflow` via `_build_tree_search_wrap`.

- [ ] **Step 1: Update `__init__` — pass tree_store/advantage_computer to patches**

In `__init__`, update the `TreeSearchPatches` instantiation (lines 170-175):

```python
            self._patches = TreeSearchPatches(
                rollout_engine=self.rollout,
                advantage_mode=self.tree_backup_config.advantage_mode,
                loss_mode=self.tree_backup_config.loss_mode,
                group_size=self.cache_config.n_samples,
                tree_store=self.tree_store,
                advantage_computer=self.tree_advantage_computer,
            )
```

- [ ] **Step 2: Verify import**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search.trainer import CacheAwarePPOTrainer; print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat: pass tree_store and advantage_computer to TreeSearchPatches

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 7: Update test files for removed patches and classes

**Files:**

- Modify: `tests/test_treesearch_patches.py`

- Modify: `tests/test_treesearch_bugfixes.py`

- [ ] **Step 1: Update `tests/test_treesearch_patches.py`**

Changes needed:

1. Remove `test_apply_then_restore_restores_originals` assertions about
   `_resolve_workflow` and `workflow_executor` (lines 53-54, 59-60, 64-66)
1. Remove `test_returns_treesearch_workflow` test (lines 122-149) — references
   `TreeSearchGroupedRolloutWorkflow`

Updated `test_apply_then_restore_restores_originals`:

```python
class TestApplyRestore:
    def test_apply_then_restore_restores_originals(
        self, mock_engine, saved_ppo_actor_state
    ):
        patches = TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        original_wrap = mock_engine._wrap_openai_agent

        patches.apply()
        assert mock_engine._wrap_openai_agent != original_wrap

        patches.restore()
        assert mock_engine._wrap_openai_agent == original_wrap
```

Remove `test_returns_treesearch_workflow` entirely (it checks for
`TreeSearchGroupedRolloutWorkflow` which no longer exists).

Remove the import of `TreeSearchGroupedRolloutWorkflow` if it was imported (it's
imported in the test method, so just remove the method).

- [ ] **Step 2: Update `tests/test_treesearch_bugfixes.py`**

Remove:

- `TestEpisodeIdUniqueness` class (lines 131-180) — tests
  `TreeSearchGroupedRolloutWorkflow`
- `test_grouped_workflow_sets_turn_idx` method in `TestTurnIdx` — tests
  `TreeSearchGroupedRolloutWorkflow` (around line 271)

The `TestEpisodeIdUniqueness` and `test_grouped_workflow_sets_turn_idx` test
`TreeSearchGroupedRolloutWorkflow` which is being deleted. Remove those test
classes/methods:

Remove lines 131-180 (TestEpisodeIdUniqueness class). Remove
`test_grouped_workflow_sets_turn_idx` method (find its exact location and remove it).

- [ ] **Step 3: Run updated tests**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_patches.py tests/test_treesearch_bugfixes.py -xvs
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_treesearch_patches.py tests/test_treesearch_bugfixes.py
git commit -m "test: remove tests for deleted patches and TreeSearchGroupedRolloutWorkflow

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 8: Delete `grouped_workflow.py` and `workflow_executor.py`

**Files:**

- Delete: `customized_areal/tree_search/grouped_workflow.py`

- Delete: `customized_areal/tree_search/workflow_executor.py`

- [ ] **Step 1: Verify no remaining imports of these files**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && rg "from customized_areal.tree_search.grouped_workflow|from customized_areal.tree_search.workflow_executor" --no-ignore-vcs
```

Expected: Only in plan files, spec files, and test files (which we've already updated).
No references in production code.

- [ ] **Step 2: Delete grouped_workflow.py**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && git rm customized_areal/tree_search/grouped_workflow.py
```

- [ ] **Step 3: Delete workflow_executor.py**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && git rm customized_areal/tree_search/workflow_executor.py
```

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove TreeSearchGroupedRolloutWorkflow and TreeSearchWorkflowExecutor

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 9: Run all tree search tests

**Files:**

- None (verification only)

- [ ] **Step 1: Run the full tree search test suite**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/ -xvs
```

Expected: All tests PASS.

- [ ] **Step 2: Run treesearch-specific tests**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_treesearch_bugfixes.py tests/test_treesearch_patches.py -xvs
```

Expected: All tests PASS.

- [ ] **Step 3: Run pre-commit hooks**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pre-commit run --all-files
```

Expected: All hooks PASS (or pre-existing failures only).

- [ ] **Step 4: Commit any remaining changes**

```bash
git status
# If clean, no commit needed. Otherwise:
git add -A
git commit -m "chore: final cleanup after patch elimination

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

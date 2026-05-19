# Tree Advantage Skip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip `compute_advantages` in the training loop when tree advantages are
already pre-computed by `TreeAdvantageComputer`, and move reward preprocessing into the
tree advantage path.

**Architecture:** Extract `_compute_advantages_for_batch` from `rl_trainer.py`'s
training loop as an overridable method. Override it in `CacheAwarePPOTrainer` to bypass
GAE when `advantage_mode == TREE`. Move reward preprocessing (scaling, clipping,
overlong penalty) from `actor.py` into `TreeAdvantageComputer.compute()`.

**Tech Stack:** Python 3.12+ | PyTorch | pytest

______________________________________________________________________

### Task 1: Add reward preprocessing fields to `TreeBackupConfig`

**Files:**

- Modify: `customized_areal/tree_search/config.py:22-29`

- [ ] **Step 1: Add fields to `TreeBackupConfig`**

```python
@dataclass
class TreeBackupConfig:
    mode: CacheMode = CacheMode.OFF
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.TREE
    loss_mode: LossMode = LossMode.GRPO
    rl_loss_weight: float = 1.0
    distill_loss_weight: float = 0.005
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    overlong_reward_penalty: bool = False
    overlong_tokens: int | None = None
    overlong_penalty_factor: float | None = None
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `uv run pytest tests/test_tree_search/test_advantage.py -v` Expected: All existing
tests PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/config.py
git commit -m "feat: add reward preprocessing fields to TreeBackupConfig"
```

______________________________________________________________________

### Task 2: Add reward preprocessing to `TreeAdvantageComputer`

**Files:**

- Modify: `customized_areal/tree_search/advantage.py`

- Test: `tests/test_tree_search/test_advantage.py`

- [ ] **Step 1: Write failing test for reward scaling and clipping**

Add to `tests/test_tree_search/test_advantage.py`:

```python
class TestTreeAdvantageComputerRewardPreprocessing:
    def test_reward_scaling_and_bias(self):
        """Reward preprocessing: (reward + bias) * scaling is applied before GRPO norm."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(
            store, reward_bias=-0.5, reward_scaling=2.0, reward_clip=100.0
        )
        # Two nodes: raw rewards 1.0 and 0.0
        # After preprocessing: (1.0 - 0.5) * 2.0 = 1.0, (0.0 - 0.5) * 2.0 = -1.0
        # GRPO norm on [1.0, -1.0]: mean=0, std=1.0 → [1.0, -1.0]
        n1 = _make_node([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="n1")
        n2 = _make_node([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1", node_id="n2")
        store.insert_batch([n1, n2])
        computer.compute([n1, n2])
        # Response positions should have normalized values
        assert abs(n1.advantages[2].item() - 1.0) < 1e-5
        assert abs(n2.advantages[2].item() + 1.0) < 1e-5

    def test_reward_clipping(self):
        """Rewards are clipped after scaling."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(
            store, reward_bias=0.0, reward_scaling=1.0, reward_clip=5.0
        )
        # Two nodes: raw rewards 10.0 and -10.0
        # After clipping: 5.0 and -5.0
        n1 = _make_node([1, 2, 3], [0, 0, 1], reward=10.0, query_id="q1", node_id="n1")
        n2 = _make_node([4, 5, 6], [0, 0, 1], reward=-10.0, query_id="q1", node_id="n2")
        store.insert_batch([n1, n2])
        computer.compute([n1, n2])
        # GRPO norm on [5.0, -5.0]: mean=0, std=5.0 → [1.0, -1.0]
        assert abs(n1.advantages[2].item() - 1.0) < 1e-5
        assert abs(n2.advantages[2].item() + 1.0) < 1e-5

    def test_overlong_penalty(self):
        """Overlong penalty reduces reward when response exceeds limit."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(
            store,
            overlong_reward_penalty=True,
            overlong_tokens=2,
            overlong_penalty_factor=1.0,
            max_response_length=4,
        )
        # loss_mask has 4 response tokens → response_length=4
        # expected_len = max_response_length - overlong_tokens = 2
        # exceed_len = 4 - 2 = 2
        # overlong_reward = min(-2/2 * 1.0, 0) = -1.0
        # reward = 2.0 + (-1.0) = 1.0
        n1 = _make_node(
            [1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 1, 1],
            reward=2.0, query_id="q1", node_id="n1",
        )
        n2 = _make_node(
            [7, 8, 9, 10], [0, 0, 1, 1],
            reward=0.0, query_id="q1", node_id="n2",
        )
        store.insert_batch([n1, n2])
        computer.compute([n1, n2])
        # n2: no overlong (response_length=2 ≤ 4), reward stays 0.0
        # n1: reward = 2.0 - 1.0 = 1.0, n2: reward = 0.0
        # GRPO norm on [1.0, 0.0]: mean=0.5, std=0.5
        # n1 norm = (1.0 - 0.5) / 0.5 = 1.0
        # n2 norm = (0.0 - 0.5) / 0.5 = -1.0
        assert abs(n1.advantages[2].item() - 1.0) < 1e-5
        assert abs(n2.advantages[2].item() + 1.0) < 1e-5

    def test_no_preprocessing_defaults(self):
        """Default params (bias=0, scaling=1, clip=20, no overlong) preserve raw rewards."""
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        n1 = _make_node([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1", node_id="n1")
        n2 = _make_node([4, 5, 6], [0, 0, 1], reward=-1.0, query_id="q1", node_id="n2")
        store.insert_batch([n1, n2])
        computer.compute([n1, n2])
        # Same as without preprocessing: [1.0, -1.0] → GRPO → [1.0, -1.0]
        assert abs(n1.advantages[2].item() - 1.0) < 1e-5
        assert abs(n2.advantages[2].item() + 1.0) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
`uv run pytest tests/test_tree_search/test_advantage.py::TestTreeAdvantageComputerRewardPreprocessing -v`
Expected: FAIL — `TreeAdvantageComputer.__init__()` does not accept the new parameters

- [ ] **Step 3: Implement reward preprocessing in `TreeAdvantageComputer`**

Replace `customized_areal/tree_search/advantage.py` with:

```python
# customized_areal/tree_search/advantage.py
from __future__ import annotations

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

from areal.utils import logging

GRPO_NORM_EPS = 1e-8

logger = logging.getLogger("TreeAdvantageComputer")


class TreeAdvantageComputer:
    """Replace GAE advantages with per-query GRPO-normalized outcome_rewards.

    Reads query_id and node_id from Node objects. Sets advantages
    and returns on the Node in-place.

    outcome_rewards are normalized within each query group (all episodes
    for the same query), producing zero-mean unit-variance values for both
    advantages and returns.

    Reward preprocessing (bias, scaling, clipping, overlong penalty) is
    applied to outcome_reward before GRPO normalization.
    """

    def __init__(
        self,
        tree_store: MCTSTreeStore,
        grpo_eps: float = GRPO_NORM_EPS,
        reward_bias: float = 0.0,
        reward_scaling: float = 1.0,
        reward_clip: float = 20.0,
        overlong_reward_penalty: bool = False,
        overlong_tokens: int | None = None,
        overlong_penalty_factor: float | None = None,
        max_response_length: int | None = None,
    ):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps
        self.reward_bias = reward_bias
        self.reward_scaling = reward_scaling
        self.reward_clip = reward_clip
        self.overlong_reward_penalty = overlong_reward_penalty
        self.overlong_tokens = overlong_tokens
        self.overlong_penalty_factor = overlong_penalty_factor
        self.max_response_length = max_response_length

    @staticmethod
    def _get_query_id(traj: Node) -> str | None:
        """Extract query_id from Node."""
        return traj.query_id or None

    def _preprocess_reward(self, reward: float, response_length: int) -> float:
        """Apply bias, scaling, clipping, and overlong penalty to a reward."""
        reward = (reward + self.reward_bias) * self.reward_scaling
        reward = max(-self.reward_clip, min(self.reward_clip, reward))
        if self.overlong_reward_penalty:
            assert self.overlong_tokens is not None
            assert self.overlong_penalty_factor is not None
            assert self.max_response_length is not None
            expected_len = self.max_response_length - self.overlong_tokens
            exceed_len = response_length - expected_len
            overlong_reward = min(-exceed_len / self.overlong_tokens * self.overlong_penalty_factor, 0)
            reward += overlong_reward
        return reward

    def compute(self, trajectories: list[Node]) -> None:
        """Replace GAE advantages with per-episode GRPO-normalized outcome_rewards.

        Groups nodes by (query_id, episode_id). Each episode contributes one
        reward (all nodes in an episode share the same outcome_reward).
        GRPO normalization operates across episodes within each query group.
        The normalized return is broadcast to all response positions in
        every node of the episode.
        """
        # Build query_id → {episode_id → [node_ids]} and per-episode reward
        query_episodes: dict[str, dict[str, list[str]]] = {}
        episode_rewards: dict[str, float] = {}  # episode_id → reward

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            ep_id = getattr(traj, "episode_id", "") or node_id
            ep_map = query_episodes.setdefault(query_id, {})
            ep_map.setdefault(ep_id, []).append(node_id)
            # All nodes in an episode share the same outcome_reward
            if ep_id not in episode_rewards:
                response_length = sum(traj.loss_mask)
                preprocessed = self._preprocess_reward(
                    traj.outcome_reward, response_length
                )
                episode_rewards[ep_id] = preprocessed

        # Per-query GRPO normalization of per-episode rewards
        for query_id, ep_map in query_episodes.items():
            ep_ids = list(ep_map.keys())
            rewards = [episode_rewards[eid] for eid in ep_ids]
            if len(rewards) < 2:
                for eid in ep_ids:
                    for nid in ep_map[eid]:
                        self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = var_r**0.5
            for eid, r in zip(ep_ids, rewards):
                norm_val = (r - mean_r) / (std_r + self.grpo_eps)
                for nid in ep_map[eid]:
                    self.tree_store.set_normalized_return(nid, norm_val)

        # Compute per-trajectory advantages and returns
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            mask = traj.loss_mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            norm_return = self.tree_store.get_normalized_return(node_id)
            traj.advantages = mask.float() * norm_return
            traj.returns = mask.float() * norm_return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tree_search/test_advantage.py -v` Expected: All tests
PASS (both old and new)

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/advantage.py tests/test_tree_search/test_advantage.py
git commit -m "feat: add reward preprocessing to TreeAdvantageComputer"
```

______________________________________________________________________

### Task 3: Pass reward preprocessing params through the workflow

**Files:**

- Modify: `customized_areal/tree_search/tree_search_grouped_workflow.py:194-231`

- [ ] **Step 1: Add reward preprocessing params to
  `TreeSearchGroupedRolloutWorkflow.__init__`**

Change the `__init__` signature and the `TreeAdvantageComputer` construction. Replace
lines 194-231:

```python
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
        reward_bias: float = 0.0,
        reward_scaling: float = 1.0,
        reward_clip: float = 20.0,
        overlong_reward_penalty: bool = False,
        overlong_tokens: int | None = None,
        overlong_penalty_factor: float | None = None,
        max_response_length: int | None = None,
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

        self.tree_checkpoint_manager = TreeCheckpointManager(checkpoint_dir)

        # Load existing tree checkpoint if present (CROSS_TRAINING mode)
        if self.cache_mode == CacheMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load()
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")
            else:
                self.tree_store = MCTSTreeStore()
        else:
            self.tree_store = MCTSTreeStore()

        self.tree_advantage_computer = TreeAdvantageComputer(
            self.tree_store,
            reward_bias=reward_bias,
            reward_scaling=reward_scaling,
            reward_clip=reward_clip,
            overlong_reward_penalty=overlong_reward_penalty,
            overlong_tokens=overlong_tokens,
            overlong_penalty_factor=overlong_penalty_factor,
            max_response_length=max_response_length,
        )
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `uv run pytest tests/test_tree_search/ -v` Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/tree_search_grouped_workflow.py
git commit -m "feat: pass reward preprocessing params through TreeSearchGroupedRolloutWorkflow"
```

______________________________________________________________________

### Task 4: Extract `_compute_advantages_for_batch` in `rl_trainer.py`

**Files:**

- Modify: `areal/trainer/rl_trainer.py:646-655`

- [ ] **Step 1: Replace the `compute_advantages` block with a method call**

Replace lines 646-655 in `areal/trainer/rl_trainer.py`:

Old:

```python
            with (
                stats_tracker.record_timing("compute_advantage"),
                perf_tracer.trace_scope(
                    "train.compute_advantage",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                adv_batch = self.actor.compute_advantages(rollout_batch)
                self.actor.get_device_stats().log("compute advantages")
```

New:

```python
            adv_batch = self._compute_advantages_for_batch(rollout_batch, global_step)
```

- [ ] **Step 2: Add the `_compute_advantages_for_batch` method to `RLTrainer`**

Add this method to the `RLTrainer` class (before the `train` method, around line 502):

```python
    def _compute_advantages_for_batch(self, rollout_batch, global_step):
        with (
            stats_tracker.record_timing("compute_advantage"),
            perf_tracer.trace_scope(
                "train.compute_advantage",
                category=Category.COMPUTE,
                args={"global_step": global_step},
            ),
        ):
            adv_batch = self.actor.compute_advantages(rollout_batch)
            self.actor.get_device_stats().log("compute advantages")
        return adv_batch
```

- [ ] **Step 3: Run existing tests to verify no breakage**

Run: `uv run pytest tests/ -k "not gpu" -x --timeout=60 2>&1 | head -50` Expected: No
import errors or test failures related to the trainer

- [ ] **Step 4: Commit**

```bash
git add areal/trainer/rl_trainer.py
git commit -m "refactor: extract _compute_advantages_for_batch from training loop"
```

______________________________________________________________________

### Task 5: Override `_compute_advantages_for_batch` in `CacheAwarePPOTrainer`

**Files:**

- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Add the override method**

Add the following method to `CacheAwarePPOTrainer` (after `__init__`, before
`_create_train_engine`):

```python
    def _compute_advantages_for_batch(self, rollout_batch, global_step):
        if self.tree_backup_config.advantage_mode == AdvantageMode.TREE:
            # Advantages already set by TreeAdvantageComputer in the workflow.
            return rollout_batch
        return super()._compute_advantages_for_batch(rollout_batch, global_step)
```

Make sure `AdvantageMode` is imported (it already is via the `TreeBackupConfig` imports,
but verify the import includes it). Check the existing imports at the top of the file —
`TreeBackupConfig` is imported from `customized_areal.tree_search.config`, and
`AdvantageMode` is also defined there. Add it to the import if not present:

```python
from customized_areal.tree_search.config import (
    AdvantageMode,
    LossMode,
    TreeBackupConfig,
)
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `uv run pytest tests/test_tree_search/ tests/test_treesearch_bugfixes.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "feat: override _compute_advantages_for_batch to skip for tree advantages"
```

______________________________________________________________________

### Task 6: Wire reward preprocessing params to the workflow construction sites

There are two sites that construct `TreeSearchGroupedRolloutWorkflow`:

1. `areal/infra/remote_inf_engine.py:743` — uses env vars
1. The `TreeBackupConfig` params need to flow through to the workflow

**Files:**

- Modify: `areal/infra/remote_inf_engine.py:726-752`

- Modify: `customized_areal/tpfc/scripts/train_tpfc_tree_search.py:128-131`

- [ ] **Step 1: Add env var reads and pass to `TreeSearchGroupedRolloutWorkflow` in
  `remote_inf_engine.py`**

Add these lines after line 741 (after `distill_loss_weight`), and update the constructor
call:

```python
                reward_bias = float(
                    os.getenv("TREE_SEARCH_REWARD_BIAS", "0.0")
                )
                reward_scaling = float(
                    os.getenv("TREE_SEARCH_REWARD_SCALING", "1.0")
                )
                reward_clip = float(
                    os.getenv("TREE_SEARCH_REWARD_CLIP", "20.0")
                )
                overlong_reward_penalty = (
                    os.getenv("TREE_SEARCH_OVERLONG_REWARD_PENALTY", "False").lower()
                    == "true"
                )
                overlong_tokens = (
                    int(os.getenv("TREE_SEARCH_OVERLONG_TOKENS"))
                    if os.getenv("TREE_SEARCH_OVERLONG_TOKENS")
                    else None
                )
                overlong_penalty_factor = (
                    float(os.getenv("TREE_SEARCH_OVERLONG_PENALTY_FACTOR"))
                    if os.getenv("TREE_SEARCH_OVERLONG_PENALTY_FACTOR")
                    else None
                )
                max_response_length = (
                    int(os.getenv("TREE_SEARCH_MAX_RESPONSE_LENGTH"))
                    if os.getenv("TREE_SEARCH_MAX_RESPONSE_LENGTH")
                    else None
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
                    reward_bias=reward_bias,
                    reward_scaling=reward_scaling,
                    reward_clip=reward_clip,
                    overlong_reward_penalty=overlong_reward_penalty,
                    overlong_tokens=overlong_tokens,
                    overlong_penalty_factor=overlong_penalty_factor,
                    max_response_length=max_response_length,
                )
```

- [ ] **Step 2: Pass reward preprocessing params in `TreeBackupConfig` construction in
  training script**

In `train_tpfc_tree_search.py`, update the `TreeBackupConfig` construction to forward
the actor config's reward preprocessing params:

```python
    tree_backup_config = TreeBackupConfig(
        mode=tree_mode,
        checkpoint_dir=cache_dir,
        reward_bias=config.actor.reward_bias,
        reward_scaling=config.actor.reward_scaling,
        reward_clip=config.actor.reward_clip,
        overlong_reward_penalty=config.actor.overlong_reward_penalty,
        overlong_tokens=config.actor.overlong_tokens,
        overlong_penalty_factor=config.actor.overlong_penalty_factor,
    )
```

Also set the corresponding env vars so `remote_inf_engine.py` reads them. Add after the
`TreeBackupConfig` construction:

```python
    # Set env vars for remote_inf_engine's TreeSearchGroupedRolloutWorkflow construction
    os.environ["TREE_SEARCH_REWARD_BIAS"] = str(config.actor.reward_bias)
    os.environ["TREE_SEARCH_REWARD_SCALING"] = str(config.actor.reward_scaling)
    os.environ["TREE_SEARCH_REWARD_CLIP"] = str(config.actor.reward_clip)
    os.environ["TREE_SEARCH_OVERLONG_REWARD_PENALTY"] = str(
        config.actor.overlong_reward_penalty
    )
    if config.actor.overlong_tokens is not None:
        os.environ["TREE_SEARCH_OVERLONG_TOKENS"] = str(config.actor.overlong_tokens)
    if config.actor.overlong_penalty_factor is not None:
        os.environ["TREE_SEARCH_OVERLONG_PENALTY_FACTOR"] = str(
            config.actor.overlong_penalty_factor
        )
    if config.actor.max_new_tokens is not None:
        os.environ["TREE_SEARCH_MAX_RESPONSE_LENGTH"] = str(
            config.actor.max_new_tokens
        )
```

- [ ] **Step 3: Verify the config attributes exist**

Run:
`uv run python -c "from areal.api.cli_args import load_expr_config; c = load_expr_config('customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml'); print(c.actor.reward_bias, c.actor.reward_scaling, c.actor.reward_clip)"`
Expected: Prints `-0.5 10.0 20.0` (or the values from the yaml)

- [ ] **Step 4: Commit**

```bash
git add areal/infra/remote_inf_engine.py customized_areal/tpfc/scripts/train_tpfc_tree_search.py
git commit -m "feat: wire reward preprocessing params through env vars and TreeBackupConfig"
```

______________________________________________________________________

### Task 7: Integration verification

**Files:**

- No new files — verification only

- [ ] **Step 1: Run full test suite for tree search**

Run: `uv run pytest tests/test_tree_search/ tests/test_treesearch_bugfixes.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify the data flow with a quick smoke test**

Run: \`uv run python -c " from customized_areal.tree_search.config import
TreeBackupConfig, AdvantageMode from customized_areal.tree_search.advantage import
TreeAdvantageComputer from customized_areal.tree_search.mcts_tree_store import
MCTSTreeStore, Node import torch

# Simulate the tree advantage path with preprocessing

store = MCTSTreeStore() computer = TreeAdvantageComputer( store, reward_bias=-0.5,
reward_scaling=10.0, reward_clip=20.0, ) n1 = Node( input_ids=\[1,2,3\],
loss_mask=\[0,0,1\], logprobs=\[0,0,0\], versions=\[-1,-1,0\], outcome_reward=1.0,
query_id='q1', node_id='n1', ) n2 = Node( input_ids=\[4,5,6\], loss_mask=\[0,0,1\],
logprobs=\[0,0,0\], versions=\[-1,-1,0\], outcome_reward=0.0, query_id='q1',
node_id='n2', ) store.insert_batch(\[n1, n2\]) computer.compute(\[n1, n2\]) print(f'n1
advantages: {n1.advantages}') print(f'n2 advantages: {n2.advantages}')

# Verify advantages are set (non-zero for two different rewards)

assert n1.advantages is not None assert n2.advantages is not None assert not
torch.allclose(n1.advantages\[2:\], n2.advantages\[2:\]) print('Integration smoke test
PASSED') "\` Expected: Prints advantage values and "Integration smoke test PASSED"

- [ ] **Step 3: Run pre-commit**

Run: `pre-commit run --all-files` Expected: All checks PASS

- [ ] **Step 4: Final commit (if any formatting fixes needed)**

```bash
git add -A
git commit -m "style: fix pre-commit issues"
```

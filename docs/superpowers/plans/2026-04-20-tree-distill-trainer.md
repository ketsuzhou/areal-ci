# TreeDistillPPOTrainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a combined trainer that uses MCTS tree backup advantages with on-policy
distillation loss and rollout caching, plus a student-only logprobs path when no teacher
is configured.

**Architecture:** `TreeDistillPPOTrainer` inherits from `CacheAwarePPOTrainer` and
layers on distillation via the `OnPolicyDistillationTrainer` pattern (monkey-patching
`PPOActor._ppo_update` + using `MultiCandidateFSDPPPOActor`). A new helper
`_build_student_only_position_rewards` constructs `PositionRewardInfo` from student
top-k logprobs when no teacher is present.

**Tech Stack:** Python 3.12+, PyTorch, AReaL framework, existing `customized_areal`
modules

______________________________________________________________________

## File Structure

| Action | File                                                                                 | Responsibility                                      |
| ------ | ------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Create | `customized_areal/tree_search_distilling/__init__.py`                                | Module exports                                      |
| Create | `customized_areal/tree_search_distilling/trainer.py`                                 | TreeDistillPPOTrainer class                         |
| Create | `customized_areal/tree_search_distilling/agent.py`                                   | TreeDistillAgent with student-only position rewards |
| Create | `customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py`    | Entry point script                                  |
| Create | `customized_areal/tree_search_distilling/configs/config_tree_search_distilling.yaml` | Training config                                     |

______________________________________________________________________

### Task 1: Create module structure and `__init__.py`

**Files:**

- Create: `customized_areal/tree_search_distilling/__init__.py`

- [ ] **Step 1: Create directory and `__init__.py`**

```python
# customized_areal/tree_search_distilling/__init__.py
"""Tree Search Distilling module for AReaL.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.
"""

from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer

__all__ = ["TreeDistillPPOTrainer"]
```

- [ ] **Step 2: Create empty subdirectories**

```bash
mkdir -p customized_areal/tree_search_distilling/scripts
mkdir -p customized_areal/tree_search_distilling/configs
```

- [ ] **Step 3: Verify imports resolve**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "from customized_areal.tree_search_distilling import TreeDistillPPOTrainer; print('Import OK')"
```

Expected: `ImportError` (trainer.py doesn't exist yet). That's fine — this step just
verifies the directory structure and `__init__.py` are in place.

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search_distilling/__init__.py
git commit -m "feat(tree-search-distilling): add module structure and __init__.py"
```

______________________________________________________________________

### Task 2: Create `TreeDistillAgent` with student-only position rewards

**Files:**

- Create: `customized_areal/tree_search_distilling/agent.py`

This agent extends `OnPolicyDistillAgent` to add a student-only path that builds
`PositionRewardInfo` objects from the student's top-k logprobs when no teacher is
configured.

- [ ] **Step 1: Create `agent.py`**

```python
# customized_areal/tree_search_distilling/agent.py
"""Tree Search Distilling Agent for AReaL integration.

Extends OnPolicyDistillAgent with a student-only position rewards path
that builds PositionRewardInfo from student top-k logprobs when no
teacher model is configured. This ensures multi-candidate logprobs are
still gathered during training for logging and analysis.
"""

from __future__ import annotations

import hashlib
from typing import Any

from customized_areal.on_policy_distill.core.agent import OnPolicyDistillAgent
from customized_areal.on_policy_distill.core.reward_compute import (
    _compute_token_rewards,
)
from customized_areal.on_policy_distill.core.teacher_client import (
    TeacherConfig,
)
from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo

from areal.utils import logging

logger = logging.getLogger("TreeDistillAgent")


def _build_student_only_position_rewards(
    student_output_ids: list[int],
    student_top_k_logprobs: list[list[tuple[int, float]]],
    top_k: int = 10,
) -> list[PositionRewardInfo]:
    """Build PositionRewardInfo from student top-k logprobs without a teacher.

    Creates position-level reward data for logging and multi-candidate
    logprob gathering. Rewards are all zeros (no teacher signal), so the
    distillation loss contributes nothing while student logprobs are still
    captured.

    Parameters
    ----------
    student_output_ids : list[int]
        Token IDs actually generated by the student model.
    student_top_k_logprobs : list[list[tuple[int, float]]]
        For each output position, a list of (token_id, logprob) tuples.
    top_k : int
        Maximum number of candidate tokens per position. Defaults to 10.

    Returns
    -------
    list[PositionRewardInfo]
        One PositionRewardInfo per output position, with zero rewards
        and student logprobs for each candidate.
    """
    if not student_output_ids or not student_top_k_logprobs:
        return []

    results: list[PositionRewardInfo] = []
    for i, pos_logprobs in enumerate(student_top_k_logprobs):
        truncated = pos_logprobs[:top_k]
        candidate_token_ids = [tid for tid, _ in truncated]
        student_lps = [lp for _, lp in truncated]
        rewards = [0.0] * len(candidate_token_ids)

        chosen_tid = student_output_ids[i]
        try:
            chosen_index = candidate_token_ids.index(chosen_tid)
        except ValueError:
            chosen_index = 0

        results.append(
            PositionRewardInfo(
                position=i,
                candidates=[str(tid) for tid in candidate_token_ids],
                candidate_token_ids=candidate_token_ids,
                logprobs=student_lps,
                rewards=rewards,
                chosen_index=chosen_index,
            )
        )

    logger.debug(
        "Built student-only position rewards: %d positions", len(results)
    )
    return results


class TreeDistillAgent(OnPolicyDistillAgent):
    """On-policy distillation agent that also saves student logprobs without a teacher.

    When a teacher model is configured, behaves identically to
    OnPolicyDistillAgent (computing student - teacher rewards).

    When no teacher is configured, builds PositionRewardInfo from the
    student's own top-k logprobs with zero rewards, ensuring
    multi-candidate logprobs are still gathered during training for
    logging and analysis.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        teacher_config: TeacherConfig | None = None,
        student_top_k: int = 10,
        **kwargs,
    ):
        """Initialize TreeDistillAgent.

        Args:
            agent_id: Optional agent ID.
            user_id: Optional user ID for authentication.
            model_name: Optional model name for LLM calls.
            teacher_config: Optional TeacherConfig. If None, student-only
                position rewards are built with zero distillation rewards.
            student_top_k: Number of top candidate tokens per position
                for student-only logprob gathering. Defaults to 10.
            **kwargs: Additional keyword arguments passed to OnPolicyDistillAgent.
        """
        super().__init__(
            agent_id=agent_id,
            user_id=user_id,
            model_name=model_name,
            teacher_config=teacher_config,
            **kwargs,
        )
        self.student_top_k = student_top_k

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs,
    ) -> float | dict[str, dict[str, Any]]:
        """Execute a single agent run, returning reward with position-level data.

        When a teacher is configured, computes student - teacher logprob
        rewards as in OnPolicyDistillAgent. When no teacher is configured,
        builds PositionRewardInfo with student logprobs and zero rewards
        so that multi-candidate logprobs are still gathered during training.
        """
        # Run the parent agent first to get the base reward and teacher
        # position rewards (if teacher is configured)
        result = await super().run(data, **extra_kwargs)

        # If teacher was configured and produced position rewards, return as-is
        if isinstance(result, dict) and any(
            "position_rewards" in v for v in result.values() if isinstance(v, dict)
        ):
            return result

        # If no teacher was configured, try to build student-only position rewards
        if self.teacher_client is None:
            proxy_client = extra_kwargs.get("proxy_client")
            if proxy_client is not None:
                try:
                    interaction = await proxy_client.get_last_interaction()
                    if (
                        interaction
                        and hasattr(interaction, "model_response")
                        and interaction.model_response is not None
                    ):
                        student_output_ids = interaction.model_response.output_tokens
                        student_top_k_logprobs = getattr(
                            interaction.model_response, "output_top_logprobs", None
                        )
                        if student_top_k_logprobs is not None:
                            position_rewards = _build_student_only_position_rewards(
                                student_output_ids=student_output_ids,
                                student_top_k_logprobs=student_top_k_logprobs,
                                top_k=self.student_top_k,
                            )
                            if position_rewards:
                                # Compute completion_id the same way as parent
                                completion_messages = data.get("messages", [])
                                completion_id = hashlib.md5(
                                    str(completion_messages).encode()
                                ).hexdigest()[:16]
                                # result is a float (scalar reward) from parent
                                scalar_reward = result if isinstance(result, (int, float)) else 0.0
                                logger.info(
                                    "Built student-only position rewards: %d positions, "
                                    "scalar_reward=%.4f",
                                    len(position_rewards),
                                    scalar_reward,
                                )
                                return {
                                    completion_id: {
                                        "position_rewards": position_rewards,
                                        "scalar_reward": scalar_reward,
                                    }
                                }
                except Exception as e:
                    logger.warning(
                        "Failed to build student-only position rewards: %s", e
                    )

        return result
```

- [ ] **Step 2: Verify the module imports correctly**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "from customized_areal.tree_search_distilling.agent import TreeDistillAgent, _build_student_only_position_rewards; print('Agent import OK')"
```

Expected: `Agent import OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search_distilling/agent.py
git commit -m "feat(tree-search-distilling): add TreeDistillAgent with student-only position rewards"
```

______________________________________________________________________

### Task 3: Create `TreeDistillPPOTrainer`

**Files:**

- Create: `customized_areal/tree_search_distilling/trainer.py`

- Modify: `customized_areal/tree_search_distilling/__init__.py` (update imports after
  trainer is created)

- [ ] **Step 1: Create `trainer.py`**

```python
# customized_areal/tree_search_distilling/trainer.py
"""Tree Search Distilling Trainer for AReaL.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.

Inherits from CacheAwarePPOTrainer and layers on distillation components
from OnPolicyDistillationTrainer:
- Patches PPOActor._ppo_update with grpo_distill_loss_fn
- Uses MultiCandidateFSDPPPOActor for multi-candidate logprob gathering
- Creates OpenAIProxyWorkflow with TreeDistillAgent for rollout generation
"""

from __future__ import annotations

from typing import Any

from customized_areal.on_policy_distill.engine import MultiCandidateFSDPPPOActor
from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
)
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

from areal.api.cli_args import PPOActorConfig
from areal.utils import logging
from areal.utils.environ import is_single_controller

logger = logging.getLogger("TreeDistillPPOTrainer")


class TreeDistillPPOTrainer(CacheAwarePPOTrainer):
    """PPOTrainer combining MCTS tree backup, rollout caching, and on-policy distillation.

    This trainer inherits from CacheAwarePPOTrainer (which provides tree
    backup advantages and rollout caching) and adds on-policy distillation
    components:

    1. Patches PPOActor._ppo_update to use grpo_distill_loss_fn (from
       OnPolicyDistillationTrainer) so the training loss includes both
       GRPO with tree-backed advantages and position-level GRPO distillation.
    2. Overrides _create_actor to use MultiCandidateFSDPPPOActor, which
       gathers logprobs for multiple candidate tokens per position.
    3. Initializes OpenAIProxyWorkflow with TreeDistillAgent for rollout
       generation. TreeDistillAgent builds PositionRewardInfo from student
       top-k logprobs even when no teacher is configured.

    Args:
        config: OnPolicyDistillConfig instance.
        cache_config: RolloutCacheConfig for rollout caching.
        tree_backup_config: TreeBackupConfig for MCTS tree backup.
        train_dataset: Optional training dataset.
        valid_dataset: Optional validation dataset.
        workflow: Optional pre-configured workflow instance.
        agent: Optional pre-configured agent instance.
    """

    def __init__(
        self,
        config: Any,
        cache_config: RolloutCacheConfig | None = None,
        tree_backup_config: TreeBackupConfig | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
        workflow: OpenAIProxyWorkflow | None = None,
        agent: Any | None = None,
    ):
        from customized_areal.on_policy_distill.training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        # Patch PPOActor._ppo_update with grpo_distill_loss_fn BEFORE
        # super().__init__() so the patched loss is used when the actor
        # is created during PPOTrainer initialization.
        patch_ppo_actor_class_to_use_distill_loss()

        self.workflow = workflow
        self.agent = agent

        # Initialize components if workflow not provided
        if self.workflow is None:
            self._init_components()

        # Initialize base CacheAwarePPOTrainer, which:
        # 1. Calls PPOTrainer.__init__() (creates actor via _create_actor override)
        # 2. Sets up MCTS tree store, advantage computer, checkpoint manager
        # 3. Patches PPOActor.compute_advantages for tree backup
        super().__init__(
            config,
            cache_config=cache_config,
            tree_backup_config=tree_backup_config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )

    def _init_components(self) -> None:
        """Initialize workflow and agent for on-policy distillation."""
        logger.info("Initializing components for tree search distilling")

        # Create agent if not provided
        if self.agent is None:
            from customized_areal.tree_search_distilling.agent import TreeDistillAgent

            # Build TeacherConfig if teacher_model_name is set
            teacher_config = None
            teacher_model_name = getattr(self.config, "teacher_model_name", "")
            if teacher_model_name:
                from customized_areal.on_policy_distill.core.teacher_client import (
                    TeacherConfig,
                )

                teacher_config = TeacherConfig(
                    teacher_base_url=getattr(
                        self.config, "teacher_base_url", "http://localhost:8001"
                    ),
                    teacher_model_name=teacher_model_name,
                    teacher_top_k=getattr(self.config, "teacher_top_k", 10),
                    teacher_max_retries=getattr(self.config, "teacher_max_retries", 3),
                    teacher_timeout=getattr(self.config, "teacher_timeout", 60.0),
                    teacher_missing_logprob=getattr(
                        self.config, "teacher_missing_logprob", -23.0
                    ),
                )

            self.agent = TreeDistillAgent(
                teacher_config=teacher_config,
                student_top_k=getattr(self.config, "teacher_top_k", 10),
            )

        # Get workflow configuration from config
        proxy_base_url = getattr(self.config, "proxy_base_url", "http://localhost:8000")
        proxy_api_key = getattr(self.config, "proxy_api_key", "dummy-admin-key")
        turn_discount = getattr(self.config, "turn_discount", 1.0)
        export_style = getattr(self.config, "export_style", "individual")

        # Create workflow
        self.workflow = OpenAIProxyWorkflow(
            agent=self.agent,
            proxy_addr=proxy_base_url,
            admin_api_key=proxy_api_key,
            discount=turn_discount,
            export_style=export_style,
        )

        logger.info("Components initialized successfully")

    def _create_actor(self, actor_config: PPOActorConfig):
        """Create actor using MultiCandidateFSDPPPOActor for multi-candidate support.

        This overrides the base PPOTrainer._create_actor to use
        MultiCandidateFSDPPPOActor instead of standard FSDPPPOActor,
        enabling multi-candidate logprob gathering for position-level rewards.
        """
        if self.allocation_mode.train_backend != "fsdp":
            raise ValueError(
                f"TreeDistillPPOTrainer only supports FSDP backend, "
                f"got: {self.allocation_mode.train_backend}"
            )

        actor_cls = MultiCandidateFSDPPPOActor

        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)

        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        logger.info("Created MultiCandidateFSDPPPOActor for tree search distilling")
        return actor
```

- [ ] **Step 2: Update `__init__.py` to include agent export**

```python
# customized_areal/tree_search_distilling/__init__.py
"""Tree Search Distilling module for AReaL.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.
"""

from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer

__all__ = ["TreeDistillPPOTrainer"]
```

(No change needed — the `__init__.py` already exports `TreeDistillPPOTrainer`.)

- [ ] **Step 3: Verify the trainer module imports correctly**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer; print('Trainer import OK')"
```

Expected: `Trainer import OK`

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search_distilling/trainer.py
git commit -m "feat(tree-search-distilling): add TreeDistillPPOTrainer combining tree backup and distillation"
```

______________________________________________________________________

### Task 4: Create entry point script

**Files:**

- Create:
  `customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py`

- [ ] **Step 1: Create the training script**

```python
# customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py
"""Training script for tree search distilling.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.

When a teacher model is configured (teacher_model_name is set), position
rewards are computed as student_logp - teacher_logp for distillation.
When no teacher is configured, student position-level logprobs are still
saved for logging, with zero distillation rewards.

Usage:
    uv run customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py \\
        --config customized_areal/tree_search_distilling/configs/config_tree_search_distilling.yaml \\
        cache_dir=/path/to/cache
"""

import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.on_policy_distill.core.config import OnPolicyDistillConfig
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("TrainTreeSearchDistilling")


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting tree search distilling training")

    # Load configuration
    config, overrides = load_expr_config(args, OnPolicyDistillConfig)

    # Extract cache config from overrides or use defaults
    cache_dir = getattr(config, "cache_dir", "")
    n_samples = config.gconfig.n_samples
    assistant_marker = getattr(config, "assistant_marker", "")

    cache_config = RolloutCacheConfig(
        cache_dir=cache_dir,
        enabled=True,
        n_samples=n_samples,
    )

    tree_backup_config = TreeBackupConfig(
        mode=TreeBackupMode.CROSS_TRAINING,
        assistant_marker=assistant_marker,
        checkpoint_dir=cache_dir,
    )

    logger.info(
        "Cache config: dir=%s, n_samples=%d, "
        "tree_mode=%s, teacher=%s",
        cache_dir,
        n_samples,
        tree_backup_config.mode.value,
        getattr(config, "teacher_model_name", "") or "(none)",
    )

    # Create trainer and run
    trainer = TreeDistillPPOTrainer(
        config=config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
    )
    trainer.train()

    logger.info("Tree search distilling training completed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script can be parsed**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "import ast; ast.parse(open('customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py').read()); print('Script parse OK')"
```

Expected: `Script parse OK`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search_distilling/scripts/train_tree_search_distilling.py
git commit -m "feat(tree-search-distilling): add training entry point script"
```

______________________________________________________________________

### Task 5: Create YAML config

**Files:**

- Create:
  `customized_areal/tree_search_distilling/configs/config_tree_search_distilling.yaml`

- [ ] **Step 1: Create the config file**

Based on `config_on_policy_distill.yaml` with added cache/tree fields.

```yaml
experiment_name: tree-search-distilling
trial_name: trial0

seed: 1
enable_offload: false
total_train_epochs: 10
tokenizer_path: ${actor.path}

# Workflow configuration — use TreeDistillAgent for student-only logprobs
workflow: customized_areal.tree_search_distilling.TreeDistillAgent
eval_workflow: ${workflow}

cluster:
  n_nodes: 1
  n_gpus_per_node: 2
  fileroot: /dfs/share-groups/letrain/zhoujie/AReaL-main/tmp/areal/experiments
  name_resolve:
    type: nfs
    nfs_record_root: /dfs/share-groups/letrain/zhoujie/AReaL-main/tmp/areal/name_resolve

allocation_mode: sglang:d1+d1

scheduler:
  type: local

rollout:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  tokenizer_path: ${tokenizer_path}
  max_concurrent_rollouts: 8
  queue_size: null
  consumer_batch_size: ${train_dataset.batch_size}
  max_head_offpolicyness: 2
  enable_rollout_tracing: false
  scheduling_spec: ${actor.scheduling_spec}
  dump_to_file: true
  openai:
    mode: inline
    tool_call_parser: qwen25
    reasoning_parser: qwen3
    export_style: individual
    turn_discount: 1.0

gconfig:
  n_samples: 4
  min_new_tokens: 0
  max_new_tokens: 1024
  max_tokens: 2048
  greedy: false
  temperature: 1.0

actor:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  attn_impl: eager
  path: /dfs/share-groups/letrain/zhoujie/Qwen3-1.7B
  init_from_scratch: false
  disable_dropout: true
  gradient_checkpointing: true
  dtype: bfloat16
  mb_spec:
    max_tokens_per_mb: 10240
  optimizer:
    type: adam
    lr: 1.70e-5
    weight_decay: 0.017
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    lr_scheduler_type: constant
    gradient_clipping: 1.0
    warmup_steps_proportion: 0.001
  eps_clip: 0.4
  temperature: ${gconfig.temperature}
  reward_scaling: 10.0
  reward_bias: -0.5
  kl_ctl: 0.0
  ppo_n_minibatches: 1
  recompute_logprob: true
  use_decoupled_loss: true
  behave_imp_weight_cap: 5.0
  reward_norm: null
  adv_norm:
    mean_level: batch
    std_level: batch
  weight_update_mode: xccl
  max_new_tokens: ${gconfig.max_new_tokens}
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cmd: python3 -m areal.infra.rpc.rpc_server
      env_vars:
        NCCL_DEBUG: "WARN"
        NCCL_IB_DISABLE: "1"
        NCCL_SOCKET_IFNAME: "eth0"

ref:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: ${actor.path}
  init_from_scratch: false
  disable_dropout: true
  dtype: ${actor.dtype}
  mb_spec:
    max_tokens_per_mb: 10240
  optimizer: null
  scheduling_strategy:
    type: colocation
    target: actor
  scheduling_spec: ${actor.scheduling_spec}

# SGLang configuration
sglang:
  model_path: ${actor.path}
  random_seed: ${seed}
  skip_tokenizer_init: true
  dtype: ${actor.dtype}
  max_running_requests: null
  context_length: 40960
  mem_fraction_static: 0.8

# On-Policy Distillation settings
cache_size: 1000
proxy_base_url: http://localhost:8000
proxy_api_key: ""
proxy_model: qwen/qwen3-1.7b
proxy_temperature: 1.0
proxy_max_tokens: 1024
proxy_top_p: 1.0
turn_discount: 1.0
export_style: individual
use_reward_scaling: true
reward_scaling_factor: 10.0
reward_bias: -0.5

# Teacher model configuration
# Set teacher_model_name to enable teacher distillation.
# Leave empty (default) for student-only position logprobs (zero distillation rewards).
teacher_base_url: http://localhost:8001
teacher_model_name: ""
teacher_top_k: 10
teacher_max_retries: 3
teacher_timeout: 60.0
teacher_missing_logprob: -23.0

# Tree Search Distilling specific settings
# cache_dir: Directory for rollout cache and MCTS tree checkpoints
# Set via command line: cache_dir=/path/to/cache
cache_dir: ""
# n_samples: Number of rollout samples per prompt to cache (uses gconfig.n_samples)
# tree_backup_mode: off | in_training | cross_training
#   - off: No tree backup (standard GRPO + distill only)
#   - in_training: Tree backup within a single run, no persistence
#   - cross_training: Tree backup with checkpoint persistence across runs
# tree_backup_mode is set to cross_training by default in the training script.

# Datasets
train_dataset:
  batch_size: 2
  shuffle: true
  pin_memory: true
  num_workers: 4
  path: openai/gsm8k
  type: rl
  max_length: 1024

valid_dataset:
  batch_size: 2
  pin_memory: true
  num_workers: 4
  path: openai/gsm8k
  type: rl

# Utilities
saver:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: null

recover:
  mode: disabled
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600

evaluator:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: null

stats_logger:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  wandb:
    mode: disabled

perf_tracer:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  enabled: false
  session_tracer:
    enabled: false
```

- [ ] **Step 2: Verify the YAML is valid**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "import yaml; yaml.safe_load(open('customized_areal/tree_search_distilling/configs/config_tree_search_distilling.yaml')); print('YAML valid')"
```

Expected: `YAML valid`

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search_distilling/configs/config_tree_search_distilling.yaml
git commit -m "feat(tree-search-distilling): add training config YAML"
```

______________________________________________________________________

### Task 6: End-to-end import verification and lint

**Files:**

- All files created in Tasks 1-5

- [ ] **Step 1: Run full import verification**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "
from customized_areal.tree_search_distilling import TreeDistillPPOTrainer
from customized_areal.tree_search_distilling.agent import TreeDistillAgent, _build_student_only_position_rewards
from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer as T
print('All imports OK')
print(f'TreeDistillPPOTrainer MRO: {[c.__name__ for c in T.__mro__]}')
"
```

Expected: `All imports OK` and MRO showing
`TreeDistillPPOTrainer -> CacheAwarePPOTrainer -> PPOTrainer -> ...`

- [ ] **Step 2: Run pre-commit on new files**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && pre-commit run --files customized_areal/tree_search_distilling/**/*.py
```

Expected: All checks pass (or only minor formatting fixes needed).

- [ ] **Step 3: Verify `_build_student_only_position_rewards` works correctly**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "
from customized_areal.tree_search_distilling.agent import _build_student_only_position_rewards

# Test with sample data
output_ids = [200, 201, 202]
top_k_logprobs = [
    [(200, -0.5), (300, -1.2), (400, -2.0)],
    [(201, -0.3), (500, -0.8)],
    [(202, -1.0), (600, -1.5), (700, -2.5), (800, -3.0)],
]

result = _build_student_only_position_rewards(output_ids, top_k_logprobs, top_k=10)

assert len(result) == 3, f'Expected 3 positions, got {len(result)}'
assert result[0].candidate_token_ids == [200, 300, 400], f'Wrong token IDs: {result[0].candidate_token_ids}'
assert result[0].logprobs == [-0.5, -1.2, -2.0], f'Wrong logprobs: {result[0].logprobs}'
assert result[0].rewards == [0.0, 0.0, 0.0], f'Expected zero rewards, got: {result[0].rewards}'
assert result[0].chosen_index == 0, f'Expected chosen_index=0, got {result[0].chosen_index}'
assert result[1].chosen_index == 0, f'Expected chosen_index=0, got {result[1].chosen_index}'
assert result[2].chosen_index == 0, f'Expected chosen_index=0, got {result[2].chosen_index}'
print('_build_student_only_position_rewards: all tests passed')
"
```

Expected: `_build_student_only_position_rewards: all tests passed`

- [ ] **Step 4: Final commit if any formatting fixes were applied**

```bash
git add -u && git commit -m "style: apply pre-commit formatting fixes"
```

______________________________________________________________________

## Self-Review Checklist

### Spec Coverage

| Spec Requirement                                          | Task                                            |
| --------------------------------------------------------- | ----------------------------------------------- |
| Inherit from CacheAwarePPOTrainer                         | Task 3                                          |
| Patch PPOActor.\_ppo_update with grpo_distill_loss_fn     | Task 3 (in `__init__`)                          |
| Use MultiCandidateFSDPPPOActor                            | Task 3 (`_create_actor`)                        |
| Initialize OpenAIProxyWorkflow with agent                 | Task 3 (`_init_components`)                     |
| MCTS tree backup (inherited from CacheAwarePPOTrainer)    | Task 3 (via super())                            |
| Rollout caching (inherited from CacheAwarePPOTrainer)     | Task 3 (via super())                            |
| Student-only position rewards when no teacher             | Task 2 (`_build_student_only_position_rewards`) |
| Teacher position rewards when teacher configured          | Task 2 (inherits from OnPolicyDistillAgent)     |
| Entry point script                                        | Task 4                                          |
| YAML config                                               | Task 5                                          |
| FSDP-only constraint                                      | Task 3 (`_create_actor` raises ValueError)      |
| File location: `customized_areal/tree_search_distilling/` | Tasks 1-5                                       |

### Placeholder Scan

No TBD, TODO, or incomplete sections found. All steps contain complete code.

### Type Consistency

- `_build_student_only_position_rewards` returns `list[PositionRewardInfo]` — matches
  what `OnPolicyDistillAgent.run()` returns when teacher is present.
- `TreeDistillAgent.run()` return type is `float | dict[str, dict[str, Any]]` — matches
  parent class interface.
- `TreeDistillPPOTrainer.__init__` signature matches `CacheAwarePPOTrainer` plus
  `workflow` and `agent` params from `OnPolicyDistillationTrainer`.
- Config types: `cache_config: RolloutCacheConfig | None`,
  `tree_backup_config: TreeBackupConfig | None` — matches `CacheAwarePPOTrainer`
  signature.

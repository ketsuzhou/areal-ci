# customized_areal/tree_search/trainer.py
"""PPOTrainer with tree-search-aware rollout via .env flag.

All cache logic, tree ops, and checkpoint saving happen inside
TreeSearchGroupedRolloutWorkflow (activated by .env flag
use_TreeSearchGroupedRolloutWorkflow=True in customized_areal/.env).

This class overrides:
- _create_train_engine: uses MultiCandidateFSDPPPOActor when distill loss
  is enabled
- train: applies/restores the distill loss PPOActor patch when loss_mode
  != GRPO
- _save_hf / _save_recover_checkpoint: writes train_id.json sidecar
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch.distributed as dist

from customized_areal.tree_search.config import (
    LossMode,
    TreeBackupConfig,
)

from areal import PPOTrainer
from areal.utils import logging
from areal.utils.environ import is_single_controller
from areal.utils.saver import Saver

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
        logger.info(
            "CacheAwarePPOTrainer.train() called: workflow=%s, loss_mode=%s",
            workflow,
            self.tree_backup_config.loss_mode.value,
        )
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

    @staticmethod
    def _write_train_id_sidecar(checkpoint_dir: str) -> None:
        train_id = os.environ.get("TRAIN_ID", "")
        if not train_id:
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = os.path.join(checkpoint_dir, "train_id.json")
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"train_id": train_id}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int) -> None:
        super()._save_hf(epoch, epoch_step, global_step)
        if not dist.is_initialized() or dist.get_rank() == 0:
            saver_cfg = self.saver.config
            for name in ["default"] + (["critic"] if self.critic is not None else []):
                path = Saver.get_model_save_path(
                    saver_cfg.experiment_name,
                    saver_cfg.trial_name,
                    saver_cfg.fileroot,
                    epoch,
                    epoch_step,
                    global_step,
                    name,
                )
                self._write_train_id_sidecar(path)

    def _save_recover_checkpoint(
        self, epoch: int, epoch_step: int, global_step: int
    ) -> None:
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)
        if not dist.is_initialized() or dist.get_rank() == 0:
            recover_cfg = self.recover_handler.config
            for name in ["default"] + (["critic"] if self.critic is not None else []):
                path = Saver.get_recover_checkpoint_path(
                    recover_cfg.experiment_name,
                    recover_cfg.trial_name,
                    recover_cfg.fileroot,
                    name,
                )
                self._write_train_id_sidecar(path)

    def close(self) -> None:
        super().close()

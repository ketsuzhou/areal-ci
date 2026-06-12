# customized_areal/tree_search/training/trainer.py
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

from customized_areal.clip_cov import ClipCovConfig
from customized_areal.tree_search.config import (
    Config,
    LossMode,
)

from areal import PPOTrainer
from areal.utils import logging
from areal.utils.environ import is_single_controller
from areal.utils.saver import Saver

logger = logging.getLogger("TreeBackupPPOTrainer")


def _patch_clip_cov_from_actor_config(config: Any) -> None:
    from customized_areal.clip_cov import patch_ppo_actor_to_use_clip_cov_loss

    patch_ppo_actor_to_use_clip_cov_loss(
        ClipCovConfig(
            clip_ratio=config.clip_cov_clip_ratio,
            clip_cov_lb=config.clip_cov_lb,
            clip_cov_ub=config.clip_cov_ub,
        )
    )


class ClipCovFSDPPPOActor:
    """FSDP actor wrapper that patches PPOActor before worker actor creation."""

    def __new__(cls, config):
        from areal.engine import FSDPPPOActor

        _patch_clip_cov_from_actor_config(config)
        return FSDPPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


class ClipCovMegatronPPOActor:
    """Megatron actor wrapper that patches PPOActor before worker actor creation."""

    def __new__(cls, config):
        from areal.engine import MegatronPPOActor

        _patch_clip_cov_from_actor_config(config)
        return MegatronPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


def _patch_muon_from_actor_config(config: Any) -> None:
    from customized_areal.optimizers import patch_fsdp_engine_for_muon

    patch_fsdp_engine_for_muon(
        momentum=config.muon_momentum,
        muon_adam_lr=config.muon_adam_lr,
        ns_steps=config.muon_ns_steps,
        nesterov=config.muon_nesterov,
    )


class MuonFSDPPPOActor:
    """FSDP actor wrapper that patches optimizer creation before worker init."""

    def __new__(cls, config):
        from areal.engine import FSDPPPOActor

        _patch_muon_from_actor_config(config)
        return FSDPPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


class MuonMultiCandidateFSDPPPOActor:
    """Multi-candidate FSDP actor wrapper with Muon optimizer patching."""

    def __new__(cls, config):
        from customized_areal.tree_search.engine import MultiCandidateFSDPPPOActor

        _patch_muon_from_actor_config(config)
        return MultiCandidateFSDPPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


class CustomizedPPOTrainer(PPOTrainer):
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
        tree_search_config: Config | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
    ):
        self.tree_search_config = tree_search_config or Config()
        self._clip_cov_patch_applied = False
        self._muon_patch_applied = False
        if self.tree_search_config.use_clip_cov:
            if self.tree_search_config.loss_mode != LossMode.GRPO:
                raise ValueError(
                    "clip-cov is currently supported only when "
                    "tree_search_config.loss_mode == LossMode.GRPO"
                )
            self._patch_clip_cov_loss()
        if self.tree_search_config.use_muon_optimizer:
            self._patch_muon_optimizer(config)
        try:
            super().__init__(config, train_dataset, valid_dataset)
        except Exception:
            if self._clip_cov_patch_applied:
                self._unpatch_clip_cov_loss()
            if self._muon_patch_applied:
                self._unpatch_muon_optimizer()
            raise

    def _get_clip_cov_config(self) -> ClipCovConfig:
        return ClipCovConfig(
            clip_ratio=self.tree_search_config.clip_cov_clip_ratio,
            clip_cov_lb=self.tree_search_config.clip_cov_lb,
            clip_cov_ub=self.tree_search_config.clip_cov_ub,
        )

    def _patch_clip_cov_loss(self) -> None:
        from customized_areal.clip_cov import patch_ppo_actor_to_use_clip_cov_loss

        patch_ppo_actor_to_use_clip_cov_loss(self._get_clip_cov_config())
        self._clip_cov_patch_applied = True

    def _unpatch_clip_cov_loss(self) -> None:
        from customized_areal.clip_cov import unpatch_ppo_actor_clip_cov_loss

        unpatch_ppo_actor_clip_cov_loss()
        self._clip_cov_patch_applied = False

    def _set_muon_actor_attrs(self, actor_config: Any) -> None:
        setattr(actor_config, "muon_momentum", self.tree_search_config.muon_momentum)
        setattr(actor_config, "muon_adam_lr", self.tree_search_config.muon_adam_lr)
        setattr(actor_config, "muon_ns_steps", self.tree_search_config.muon_ns_steps)
        setattr(actor_config, "muon_nesterov", self.tree_search_config.muon_nesterov)

    def _patch_muon_optimizer(self, config: Any) -> None:
        optimizer = getattr(config.actor, "optimizer", None)
        if optimizer is None:
            raise ValueError("Muon optimizer requires actor.optimizer to be configured")
        actor_backend = getattr(config.actor, "backend", "")
        if not str(actor_backend).startswith("fsdp"):
            raise ValueError(
                f"Muon optimizer requires FSDP actor backend, got: {actor_backend}"
            )
        if getattr(config.actor.fsdp, "per_layer_optim_step", False):
            raise ValueError("Muon optimizer is incompatible with per_layer_optim_step")
        optimizer.type = "muon"
        _patch_muon_from_actor_config(self.tree_search_config)
        self._muon_patch_applied = True

    def _unpatch_muon_optimizer(self) -> None:
        from customized_areal.optimizers import unpatch_fsdp_engine_for_muon

        unpatch_fsdp_engine_for_muon()
        self._muon_patch_applied = False

    def _create_train_engine(self, actor_config, alloc):
        """Override to use MultiCandidateFSDPPPOActor when distill loss is enabled."""
        if self.tree_search_config.use_muon_optimizer:
            if alloc.backend != "fsdp":
                raise ValueError(
                    f"Muon optimizer requires FSDP backend, got: {alloc.backend}"
                )
            if actor_config.optimizer is not None:
                actor_config.optimizer.type = "muon"
            self._set_muon_actor_attrs(actor_config)
        setattr(actor_config, "use_clip_cov", self.tree_search_config.use_clip_cov)
        setattr(
            actor_config,
            "clip_cov_clip_ratio",
            self.tree_search_config.clip_cov_clip_ratio,
        )
        setattr(actor_config, "clip_cov_lb", self.tree_search_config.clip_cov_lb)
        setattr(actor_config, "clip_cov_ub", self.tree_search_config.clip_cov_ub)
        if self.tree_search_config.loss_mode != LossMode.GRPO:
            setattr(
                actor_config,
                "distill_kl_mode",
                self.tree_search_config.distill_kl_mode.value,
            )
            if alloc.backend != "fsdp":
                raise ValueError(
                    f"Distillation loss mode requires FSDP backend, "
                    f"got: {alloc.backend}"
                )
            from customized_areal.tree_search.engine import (
                MultiCandidateFSDPPPOActor,
            )

            actor_cls = MultiCandidateFSDPPPOActor
            if self.tree_search_config.use_muon_optimizer:
                actor_cls = MuonMultiCandidateFSDPPPOActor
            if is_single_controller():
                actor = actor_cls.as_controller(actor_config, self.scheduler)
            else:
                actor = actor_cls(config=actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            logger.info(
                f"Created MultiCandidateFSDPPPOActor "
                f"(loss_mode={self.tree_search_config.loss_mode.value})"
            )
            return actor
        if self.tree_search_config.use_muon_optimizer:
            actor_cls = MuonFSDPPPOActor
            if is_single_controller():
                actor = actor_cls.as_controller(actor_config, self.scheduler)
            else:
                actor = actor_cls(config=actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            logger.info(
                "Created MuonFSDPPPOActor (momentum=%.3f, aux_adam_lr=%g)",
                self.tree_search_config.muon_momentum,
                self.tree_search_config.muon_adam_lr,
            )
            return actor
        if self.tree_search_config.use_clip_cov:
            if alloc.backend == "fsdp":
                actor_cls = ClipCovFSDPPPOActor
            elif alloc.backend == "megatron":
                actor_cls = ClipCovMegatronPPOActor
            else:
                raise ValueError(
                    f"clip-cov requires FSDP or Megatron backend, got: {alloc.backend}"
                )
            if is_single_controller():
                actor = actor_cls.as_controller(actor_config, self.scheduler)
            else:
                actor = actor_cls(config=actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            logger.info(
                "Created %s (clip_ratio=%.4f, lb=%.1f, ub=%.1f)",
                actor_cls.__name__,
                self.tree_search_config.clip_cov_clip_ratio,
                self.tree_search_config.clip_cov_lb,
                self.tree_search_config.clip_cov_ub,
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
            "CustomizedPPOTrainer.train() called: workflow=%s, loss_mode=%s, "
            "use_clip_cov=%s",
            workflow,
            self.tree_search_config.loss_mode.value,
            self.tree_search_config.use_clip_cov,
        )
        if self.tree_search_config.loss_mode != LossMode.GRPO:
            from .actor import (
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
        try:
            super().close()
        finally:
            if self._clip_cov_patch_applied:
                self._unpatch_clip_cov_loss()
            if self._muon_patch_applied:
                self._unpatch_muon_optimizer()

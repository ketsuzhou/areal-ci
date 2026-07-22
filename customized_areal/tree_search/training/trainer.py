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
    AdvantageMode,
    Config,
    LossMode,
)

from areal import PPOTrainer
from areal.trainer.rl_trainer import _EmptyDataLoader
from areal.utils import logging
from areal.utils.environ import is_single_controller
from areal.utils.saver import Saver

from .actor import (
    VIMPO_ACTOR_CONFIG_FIELDS,
    ClipCovFSDPPPOActor,
    ClipCovMegatronPPOActor,
    MuonFSDPPPOActor,
    MuonMultiCandidateFSDPPPOActor,
    _patch_muon_from_actor_config,
)

logger = logging.getLogger("TreeBackupPPOTrainer")


class _FreshQueryDatasetPlaceholder:
    """Non-empty placeholder so PPOTrainer can enter dataset-backed setup."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:  # noqa: ARG002
        return {}


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
        self._combined_critic_patch_applied = False
        self._weight_update_timeout_patched = False
        self._v2_disk_fallback_patched = False
        self._use_fresh_query_dataloader = self.tree_search_config.use_fresh_query
        if self._use_fresh_query_dataloader:
            if config.total_train_steps is None:
                raise ValueError(
                    "total_train_steps must be set when "
                    "tree_search_config.use_fresh_query=True"
                )
            steps_per_epoch = config.total_train_steps // config.total_train_epochs
            if steps_per_epoch < 1:
                raise ValueError(
                    f"total_train_steps ({config.total_train_steps}) must be >= "
                    f"total_train_epochs ({config.total_train_epochs}) when "
                    "tree_search_config.use_fresh_query=True"
                )
            if train_dataset is None:
                train_dataset = _FreshQueryDatasetPlaceholder()
        if self.tree_search_config.use_clip_cov:
            if self.tree_search_config.loss_mode != LossMode.GRPO:
                raise ValueError(
                    "clip-cov is currently supported only when "
                    "tree_search_config.loss_mode == LossMode.GRPO"
                )
            self._patch_clip_cov_loss()
        if self.tree_search_config.use_muon_optimizer:
            self._patch_muon_optimizer(config)
        self._patch_weight_update_setup_timeout()
        self._patch_v2_disk_fallback(config)
        try:
            super().__init__(config, train_dataset, valid_dataset)
        except Exception:
            if self._clip_cov_patch_applied:
                self._unpatch_clip_cov_loss()
            if self._muon_patch_applied:
                self._unpatch_muon_optimizer()
            raise

    def _create_dataloader(self, dataset, dataset_config, rank, world_size):
        if self._use_fresh_query_dataloader:
            assert self.config.total_train_steps is not None
            steps_per_epoch = (
                self.config.total_train_steps // self.config.total_train_epochs
            )
            return _EmptyDataLoader(
                batch_size=self.config.train_dataset.batch_size,
                steps_per_epoch=steps_per_epoch,
            )
        return super()._create_dataloader(dataset, dataset_config, rank, world_size)

    def _create_vimpo_train_engine(self, actor_config, alloc):
        """Construct the dedicated VIMPO FSDP actor.

        VIMPO runs critic-free (no learned critic, no generic reference engine):
        ``actor.kl_ctl`` is forced to 0 so the base ``PPOTrainer`` never creates
        ``self.ref``, and no critic allocation is configured. The distillation
        and combined-critic monkey patches are NOT installed (VIMPO uses
        ``loss_mode=GRPO`` and ``enable_generative_critic=False``, both enforced
        by ``Config.__post_init__``).
        """
        if alloc.backend != "fsdp":
            raise ValueError(
                f"VIMPO requires FSDP actor backend, got: {alloc.backend!r}. "
                "Set the actor allocation backend to 'fsdp' for "
                "advantage_mode=vimpo."
            )
        # Copy every vimpo_* setting onto actor_config so the actor can build
        # its reference scorer and advantage computer from the actor config.
        for field in VIMPO_ACTOR_CONFIG_FIELDS:
            setattr(actor_config, field, getattr(self.tree_search_config, field))
        # VIMPO must NOT require a positive PPO KL-reward coefficient: force
        # kl_ctl=0 so the base trainer creates no generic reference engine.
        setattr(actor_config, "kl_ctl", 0.0)
        # Select the dedicated VIMPO actor (Muon-wrapped only if requested).
        from customized_areal.tree_search.engine import VIMPOFSDPPPOActor

        if self.tree_search_config.use_muon_optimizer:
            from customized_areal.tree_search.training.actor import (
                MuonVIMPOFSDPPPOActor,
            )

            actor_cls = MuonVIMPOFSDPPPOActor
        else:
            actor_cls = VIMPOFSDPPPOActor
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        logger.info(
            "Created VIMPOFSDPPPOActor (top_k=%d, beta=%g, ref=%s)",
            self.tree_search_config.vimpo_top_k,
            self.tree_search_config.vimpo_beta,
            self.tree_search_config.vimpo_ref_base_url,
        )
        return actor

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

    def _patch_weight_update_setup_timeout(self, timeout_s: float = 120.0) -> None:
        """Bump WeightUpdateController setup_timeout to survive slow areal imports.

        The weight-update gateway is a fresh `python -m areal.v2.weight_update.gateway`
        subprocess. `areal/__init__.py` eagerly imports infra/controller/launcher,
        pulling in torch/ray/megatron; cold-start takes ~35s. The default
        `setup_timeout=30.0` trips a spurious TimeoutError before uvicorn even
        binds. Wrap `WeightUpdateController.initialize` to raise the deadline to
        `timeout_s` (120s) so the gateway has time to finish importing.

        Dataclass field defaults are baked into the generated `__init__`
        signature, so setting `WeightUpdateControllerConfig.setup_timeout` as a
        class attribute does NOT propagate to new instances; we must mutate
        `self.config.setup_timeout` on the live instance instead.
        """
        from areal.v2.weight_update.controller.controller import (
            WeightUpdateController,
        )

        if getattr(WeightUpdateController.initialize, "_timeout_patched", False):
            self._weight_update_timeout_patched = True
            return

        original_initialize = WeightUpdateController.initialize

        def _initialize_with_extended_timeout(controller_self):
            if controller_self.config.setup_timeout < timeout_s:
                controller_self.config.setup_timeout = timeout_s
            return original_initialize(controller_self)

        _initialize_with_extended_timeout._timeout_patched = True
        WeightUpdateController.initialize = _initialize_with_extended_timeout
        self._weight_update_timeout_patched = True

    def _patch_v2_disk_fallback(self, config: Any) -> None:
        """Force v2 weight-update dispatch onto the disk path when configured.

        ``rl_trainer.py`` v2 branch (non-LoRA) hard-codes
        ``WeightUpdateMeta.from_awex()`` and ignores
        ``actor.weight_update_mode`` entirely. For models AWEX can't yet
        map (Qwen3.5 hybrid Mamba: train has 4 separate ``in_proj_{a,b,qkv,z}``
        but SGLang has 2 fused ``in_proj_{ba,qkvz}``), swap ``from_awex``
        for a ``from_disk`` factory so the gateway routes through
        ``_disk_transfer_weights`` and SGLang's native ``load_weights``
        handles the structural fusion.

        No-op when ``weight_update_mode != "disk"``, so switching back to
        ``xccl`` in config restores AWEX behavior without code changes.
        """
        if getattr(config, "actor", None) is None:
            return
        if config.actor.weight_update_mode != "disk":
            return

        from areal.api.io_struct import WeightUpdateMeta

        if getattr(WeightUpdateMeta.from_awex, "_disk_fallback_patched", False):
            self._v2_disk_fallback_patched = True
            return

        disk_kwargs = {
            "experiment_name": config.experiment_name,
            "trial_name": config.trial_name,
            "file_root": config.cluster.fileroot,
            "name": "default",
            "clear_checkpoint_after_load": True,
        }

        def _from_awex_disk_fallback(**kwargs):
            return WeightUpdateMeta.from_disk(**disk_kwargs)

        _from_awex_disk_fallback._disk_fallback_patched = True
        WeightUpdateMeta.from_awex = staticmethod(_from_awex_disk_fallback)
        self._v2_disk_fallback_patched = True

    def _create_train_engine(self, actor_config, alloc):
        """Override to use MultiCandidateFSDPPPOActor when distill loss is enabled."""
        # VIMPO: dedicated FSDP actor with frozen reference scoring. Must run
        # BEFORE the distillation/Muon/clip-cov branches and must NOT install
        # any combined-critic or distill monkey patch (VIMPO is critic-free and
        # runs with loss_mode=GRPO, so train() installs neither patch).
        if self.tree_search_config.advantage_mode is AdvantageMode.VIMPO:
            return self._create_vimpo_train_engine(actor_config, alloc)
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
            "use_clip_cov=%s, enable_generative_critic=%s",
            workflow,
            self.tree_search_config.loss_mode.value,
            self.tree_search_config.use_clip_cov,
            self.tree_search_config.enable_generative_critic,
        )
        critic_enabled = self.tree_search_config.enable_generative_critic
        if critic_enabled:
            logger.info(
                "Generative critic enabled (shared model): advantage_mode=%s, "
                "gamma=%.3g, lambda=%.3g, score_max=%d, loss_weight=%.4g. The actor "
                "trains on GAE advantages derived from critic state values; the "
                "shared model additionally runs a soft-regression critic step.",
                self.tree_search_config.advantage_mode.value,
                self.tree_search_config.critic_gamma,
                self.tree_search_config.critic_lambda,
                self.tree_search_config.critic_score_max,
                self.tree_search_config.critic_loss_weight,
            )
            self._patch_combined_critic_loss()
        try:
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
        finally:
            if critic_enabled:
                self._unpatch_combined_critic_loss()

    def _critic_pad_token_id(self) -> int:
        tok = getattr(self, "tokenizer", None)
        pad = getattr(tok, "pad_token_id", None) if tok is not None else None
        if pad is None:
            pad = getattr(tok, "eos_token_id", None) if tok is not None else None
        return int(pad) if pad is not None else 0

    def _patch_combined_critic_loss(self) -> None:
        from .actor import patch_ppo_actor_class_to_use_combined_critic_loss

        patch_ppo_actor_class_to_use_combined_critic_loss(
            self.tree_search_config.critic_loss_weight,
            pad_token_id=self._critic_pad_token_id(),
        )
        self._combined_critic_patch_applied = True

    def _unpatch_combined_critic_loss(self) -> None:
        from .actor import unpatch_combined_critic_loss

        unpatch_combined_critic_loss()
        self._combined_critic_patch_applied = False

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
            if self._combined_critic_patch_applied:
                self._unpatch_combined_critic_loss()

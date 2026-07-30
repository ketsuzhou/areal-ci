from __future__ import annotations

import time
from typing import Any

_original_create_optimizer = None
_muon_config: dict[str, Any] = {}


def patch_fsdp_engine_for_muon(
    momentum: float = 0.95,
    muon_adam_lr: float = 3e-4,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> None:
    """Replace FSDPEngine._create_optimizer with a Muon-aware version.

    Must be called before engine creation. Muon-specific hyperparameters
    are stored as module-level state and read by the patched method.

    Args:
        momentum: Muon momentum coefficient (default 0.95).
        muon_adam_lr: Adam learning rate for non-2D params (default 3e-4).
        ns_steps: Number of Newton-Schulz iterations (default 5).
        nesterov: Whether to use Nesterov momentum lookahead (default True).
    """
    global _original_create_optimizer, _muon_config
    from areal.engine.fsdp_engine import FSDPEngine

    # Check if already patched
    if _original_create_optimizer is None:
        _original_create_optimizer = FSDPEngine._create_optimizer

    # Update config regardless of patch status (in-place to keep references valid)
    _muon_config.clear()
    _muon_config.update(
        {
            "momentum": momentum,
            "muon_adam_lr": muon_adam_lr,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
        }
    )

    # Only patch if not already patched
    if FSDPEngine._create_optimizer is not _patched_create_optimizer:
        FSDPEngine._create_optimizer = _patched_create_optimizer


def unpatch_fsdp_engine_for_muon() -> None:
    """Restore the original FSDPEngine._create_optimizer."""
    global _original_create_optimizer, _muon_config
    from areal.engine.fsdp_engine import FSDPEngine

    if _original_create_optimizer is not None:
        FSDPEngine._create_optimizer = _original_create_optimizer
        # Reset the original tracker and config
        _original_create_optimizer = None
        _muon_config = {}


def _patched_create_optimizer(self, ft_spec) -> None:
    """Replacement for FSDPEngine._create_optimizer that supports optimizer_type='muon'."""
    if self.optimizer_config is None:
        return
    if self.optimizer_config.type != "muon":
        _original_create_optimizer(self, ft_spec)
        return

    from transformers import (
        get_constant_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    from customized_areal.optimizers.muon import MuonWithAuxAdam

    from areal.engine.fsdp_utils import get_cosine_schedule_with_warmup

    assert self.model is not None
    tik = time.perf_counter()

    muon_params = []
    adam_params = []
    for p in self.model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    lr = self.optimizer_config.lr
    weight_decay = self.optimizer_config.weight_decay

    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": lr,
            "momentum": _muon_config["momentum"],
            "weight_decay": weight_decay,
        },
        {
            "params": adam_params,
            "use_muon": False,
            "lr": _muon_config["muon_adam_lr"],
            "betas": (self.optimizer_config.beta1, self.optimizer_config.beta2),
            "eps": self.optimizer_config.eps,
            "weight_decay": weight_decay,
        },
    ]

    self.optimizer = MuonWithAuxAdam(
        param_groups,
        ns_steps=_muon_config["ns_steps"],
        nesterov=_muon_config["nesterov"],
    )

    total_train_steps = ft_spec.total_train_steps
    num_warmup_steps = int(
        self.optimizer_config.warmup_steps_proportion * total_train_steps
    )

    if self.optimizer_config.lr_scheduler_type == "cosine":
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps,
            total_train_steps,
            min_lr_ratio=self.optimizer_config.min_lr_ratio,
        )
    elif self.optimizer_config.lr_scheduler_type == "linear":
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps,
            total_train_steps,
        )
    elif self.optimizer_config.lr_scheduler_type == "constant":
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps,
        )
    else:
        raise ValueError(
            f"Unknown lr scheduler type {self.optimizer_config.lr_scheduler_type}"
        )
    self.logger.info(f"Create optimizer time: {time.perf_counter() - tik}")

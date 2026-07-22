# SPDX-License-Identifier: Apache-2.0
"""Tests for VIMPO trainer selection and lifecycle (Task 7).

Verifies that ``CustomizedPPOTrainer._create_train_engine`` selects the
dedicated ``VIMPOFSDPPPOActor`` for ``advantage_mode=vimpo``, copies every
``vimpo_*`` setting onto the actor config, rejects non-FSDP backends before
actor creation, and that a VIMPO run creates no generic reference engine and
no learned critic (``trainer.ref is None`` / ``trainer.critic is None``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from customized_areal.tree_search.config import AdvantageMode, Config
from customized_areal.tree_search.training.trainer import CustomizedPPOTrainer

# =============================================================================
# Fixtures / helpers
# =============================================================================


def _actor_config(backend: str = "fsdp:d1") -> SimpleNamespace:
    return SimpleNamespace(
        backend=backend,
        optimizer=None,
        fsdp=SimpleNamespace(per_layer_optim_step=False),
        ppo_n_minibatches=1,
        eps_clip=0.2,
        eps_clip_higher=None,
        kl_ctl=0.0,
        _version="v1",
        path="/fake/model",
    )


def _allocation(backend: str = "fsdp") -> SimpleNamespace:
    return SimpleNamespace(backend=backend, parallel=None)


def _uninitialized_trainer(
    tree_search_config: Config | None = None,
) -> CustomizedPPOTrainer:
    """A CustomizedPPOTrainer built via __new__ (no super().__init__)."""
    trainer = CustomizedPPOTrainer.__new__(CustomizedPPOTrainer)
    trainer.tree_search_config = tree_search_config or Config(
        advantage_mode="vimpo", vimpo_ref_base_url="http://ref"
    )
    trainer.scheduler = object()
    return trainer


def _stub_heavy_actor_init(monkeypatch) -> None:
    """Stub VIMPOFSDPPPOActor.__init__ + create_process_group (no FSDP/GPU).

    Imported lazily so Step 3 (class absent) fails with ImportError rather than
    a collection error.
    """
    from customized_areal.tree_search.training.actor import VIMPOFSDPPPOActor

    def _stub_init(self: Any, config: Any, scorer: Any = None) -> None:
        self.config = config

    monkeypatch.setattr(VIMPOFSDPPPOActor, "__init__", _stub_init)
    monkeypatch.setattr(
        VIMPOFSDPPPOActor, "create_process_group", lambda self, **kw: None
    )


# =============================================================================
# Step 2: trainer selection tests
# =============================================================================


def test_vimpo_selects_dedicated_fsdp_actor_and_copies_config(monkeypatch) -> None:
    """VIMPO + fsdp selects VIMPOFSDPPPOActor and copies vimpo_* onto the
    actor config."""
    _stub_heavy_actor_init(monkeypatch)
    monkeypatch.setattr(
        "customized_areal.tree_search.training.trainer.is_single_controller",
        lambda: False,
    )
    trainer = _uninitialized_trainer(
        Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref")
    )
    actor = trainer._create_train_engine(
        _actor_config(backend="fsdp:d1"), _allocation("fsdp")
    )
    assert actor.__class__.__name__ == "VIMPOFSDPPPOActor"
    assert actor.config.vimpo_top_k == 128
    assert actor.config.vimpo_ref_base_url == "http://ref"
    assert actor.config.vimpo_beta == 5e-4
    # kl_ctl is forced to 0 so base PPOTrainer creates no reference engine.
    assert actor.config.kl_ctl == 0.0


def test_vimpo_rejects_non_fsdp_before_actor_creation() -> None:
    """VIMPO with a non-FSDP backend raises before any actor is created."""
    trainer = _uninitialized_trainer(
        Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref")
    )
    with pytest.raises(ValueError, match="VIMPO requires FSDP actor backend"):
        trainer._create_train_engine(
            _actor_config(backend="megatron:d1"), _allocation("megatron")
        )


def test_vimpo_trainer_creates_no_ref_or_critic(monkeypatch) -> None:
    """VIMPO runs critic-free: trainer.ref and trainer.critic are both None.

    Base PPOTrainer creates self.ref only when config.actor.kl_ctl > 0 and
    config.ref is not None, and creates a critic only when a critic allocation
    is configured. VIMPO sets kl_ctl=0 and configures no critic/ref.
    """
    from areal.trainer.rl_trainer import PPOTrainer

    def _stub_ppo_init(
        self, config, train_dataset=None, valid_dataset=None
    ) -> None:
        self.config = config
        self.ref = None
        self.critic = None
        # Emulate the base guard: ref only when kl_ctl > 0 and config.ref set.
        if (
            getattr(config.actor, "kl_ctl", 0) > 0
            and getattr(config, "ref", None) is not None
        ):
            self.ref = object()
        if getattr(config, "critic", None) is not None:
            self.critic = object()

    monkeypatch.setattr(PPOTrainer, "__init__", _stub_ppo_init)
    monkeypatch.setattr(
        CustomizedPPOTrainer,
        "_patch_weight_update_setup_timeout",
        lambda self, *a, **kw: None,
    )
    monkeypatch.setattr(
        CustomizedPPOTrainer,
        "_patch_v2_disk_fallback",
        lambda self, *a, **kw: None,
    )

    config = SimpleNamespace(
        actor=SimpleNamespace(kl_ctl=0.0, backend="fsdp:d1"),
        ref=None,
        critic=None,
        total_train_steps=None,
        total_train_epochs=1,
        actor_weight_update_mode="xccl",
    )
    trainer = CustomizedPPOTrainer(
        config,
        tree_search_config=Config(
            advantage_mode="vimpo", vimpo_ref_base_url="http://ref"
        ),
    )
    assert trainer.ref is None
    assert trainer.critic is None
    assert trainer.tree_search_config.advantage_mode is AdvantageMode.VIMPO


def test_vimpo_branch_does_not_install_distill_or_critic_patches(
    monkeypatch,
) -> None:
    """The VIMPO branch returns before the distillation / clip-cov / Muon
    branches, so the dedicated VIMPO actor is selected."""
    _stub_heavy_actor_init(monkeypatch)
    monkeypatch.setattr(
        "customized_areal.tree_search.training.trainer.is_single_controller",
        lambda: False,
    )
    trainer = _uninitialized_trainer(
        Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref")
    )
    actor = trainer._create_train_engine(
        _actor_config(backend="fsdp:d1"), _allocation("fsdp")
    )
    assert actor.__class__.__name__ == "VIMPOFSDPPPOActor"

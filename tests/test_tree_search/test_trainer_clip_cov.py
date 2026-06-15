from types import SimpleNamespace

from customized_areal.tree_search.config import Config, LossMode
from customized_areal.tree_search.training import trainer as trainer_mod


def test_trainer_applies_clip_cov_patch_before_base_init(monkeypatch):
    calls = []

    def fake_patch(config):
        calls.append(
            ("patch", config.clip_ratio, config.clip_cov_lb, config.clip_cov_ub)
        )

    def fake_unpatch():
        calls.append(("unpatch",))

    def fake_base_init(self, config, train_dataset=None, valid_dataset=None):
        calls.append(("base_init",))

    monkeypatch.setattr(
        "customized_areal.clip_cov.patch_ppo_actor_to_use_clip_cov_loss",
        fake_patch,
    )
    monkeypatch.setattr(
        "customized_areal.clip_cov.unpatch_ppo_actor_clip_cov_loss",
        fake_unpatch,
    )
    monkeypatch.setattr(trainer_mod.PPOTrainer, "__init__", fake_base_init)
    monkeypatch.setattr(
        trainer_mod.PPOTrainer, "close", lambda self: calls.append(("close",))
    )

    config = Config(
        use_clip_cov=True,
        clip_cov_clip_ratio=0.01,
        clip_cov_lb=0.5,
        clip_cov_ub=2.0,
    )
    trainer = trainer_mod.CustomizedPPOTrainer(
        config=object(), tree_search_config=config
    )

    assert calls[:2] == [("patch", 0.01, 0.5, 2.0), ("base_init",)]

    trainer.close()

    assert calls[-2:] == [("close",), ("unpatch",)]


def test_trainer_rejects_clip_cov_with_distill_mode():
    config = Config(use_clip_cov=True, loss_mode=LossMode.DISTILL)

    try:
        trainer_mod.CustomizedPPOTrainer(config=object(), tree_search_config=config)
    except ValueError as exc:
        assert "loss_mode == LossMode.GRPO" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def _make_base_config(backend="fsdp", optimizer=None, per_layer_optim_step=False):
    return SimpleNamespace(
        total_train_steps=8,
        total_train_epochs=2,
        train_dataset=SimpleNamespace(batch_size=3),
        actor=SimpleNamespace(
            backend=backend,
            optimizer=optimizer or SimpleNamespace(type="adam"),
            fsdp=SimpleNamespace(per_layer_optim_step=per_layer_optim_step),
        ),
    )


def test_trainer_applies_muon_patch_before_base_init(monkeypatch):
    calls = []

    def fake_patch(momentum, muon_adam_lr, ns_steps, nesterov):
        calls.append(("patch", momentum, muon_adam_lr, ns_steps, nesterov))

    def fake_unpatch():
        calls.append(("unpatch",))

    def fake_base_init(self, config, train_dataset=None, valid_dataset=None):
        calls.append(("base_init", config.actor.optimizer.type))

    monkeypatch.setattr(
        "customized_areal.optimizers.patch_fsdp_engine_for_muon",
        fake_patch,
    )
    monkeypatch.setattr(
        "customized_areal.optimizers.unpatch_fsdp_engine_for_muon",
        fake_unpatch,
    )
    monkeypatch.setattr(trainer_mod.PPOTrainer, "__init__", fake_base_init)
    monkeypatch.setattr(
        trainer_mod.PPOTrainer, "close", lambda self: calls.append(("close",))
    )

    tree_config = Config(
        use_muon_optimizer=True,
        muon_momentum=0.9,
        muon_adam_lr=1e-4,
        muon_ns_steps=3,
        muon_nesterov=False,
    )
    trainer = trainer_mod.CustomizedPPOTrainer(
        config=_make_base_config(), tree_search_config=tree_config
    )

    assert calls[:2] == [("patch", 0.9, 1e-4, 3, False), ("base_init", "muon")]

    trainer.close()

    assert calls[-2:] == [("close",), ("unpatch",)]


def test_trainer_rejects_muon_without_optimizer():
    tree_config = Config(use_muon_optimizer=True)
    base_config = _make_base_config(optimizer=None)
    base_config.actor.optimizer = None

    try:
        trainer_mod.CustomizedPPOTrainer(
            config=base_config, tree_search_config=tree_config
        )
    except ValueError as exc:
        assert "actor.optimizer" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_trainer_rejects_muon_with_non_fsdp_backend():
    tree_config = Config(use_muon_optimizer=True)

    try:
        trainer_mod.CustomizedPPOTrainer(
            config=_make_base_config(backend="megatron"),
            tree_search_config=tree_config,
        )
    except ValueError as exc:
        assert "FSDP actor backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_create_train_engine_uses_muon_fsdp_actor(monkeypatch):
    created = []

    class FakeMuonActor:
        def __init__(self, config):
            created.append(("init", config.optimizer.type, config.muon_momentum))

        def create_process_group(self, parallel_strategy=None):
            created.append(("pg", parallel_strategy))

    monkeypatch.setattr(trainer_mod, "is_single_controller", lambda: False)
    monkeypatch.setattr(trainer_mod, "MuonFSDPPPOActor", FakeMuonActor)
    monkeypatch.setattr(
        trainer_mod.PPOTrainer, "__init__", lambda *args, **kwargs: None
    )

    trainer = trainer_mod.CustomizedPPOTrainer.__new__(trainer_mod.CustomizedPPOTrainer)
    trainer.tree_search_config = Config(use_muon_optimizer=True, muon_momentum=0.9)
    trainer.scheduler = None
    actor_config = SimpleNamespace(optimizer=SimpleNamespace(type="adam"))
    alloc = SimpleNamespace(backend="fsdp", parallel="dp")

    actor = trainer._create_train_engine(actor_config, alloc)

    assert isinstance(actor, FakeMuonActor)
    assert created == [("init", "muon", 0.9), ("pg", "dp")]


def test_fresh_query_requires_total_train_steps():
    tree_config = Config(use_fresh_query=True, fresh_query_table="query_bank")
    base_config = _make_base_config()
    base_config.total_train_steps = None

    try:
        trainer_mod.CustomizedPPOTrainer(
            config=base_config,
            tree_search_config=tree_config,
            train_dataset=None,
        )
    except ValueError as exc:
        assert "total_train_steps" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_fresh_query_passes_placeholder_dataset_to_base_init(monkeypatch):
    calls = []

    def fake_base_init(self, config, train_dataset=None, valid_dataset=None):
        calls.append((config, train_dataset, valid_dataset))

    monkeypatch.setattr(trainer_mod.PPOTrainer, "__init__", fake_base_init)
    tree_config = Config(use_fresh_query=True, fresh_query_table="query_bank")
    base_config = _make_base_config()

    trainer_mod.CustomizedPPOTrainer(
        config=base_config,
        tree_search_config=tree_config,
        train_dataset=None,
        valid_dataset="valid",
    )

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][1][0] == {}
    assert calls[0][2] == "valid"


def test_fresh_query_create_dataloader_returns_empty_loader():
    trainer = trainer_mod.CustomizedPPOTrainer.__new__(trainer_mod.CustomizedPPOTrainer)
    trainer._use_fresh_query_dataloader = True
    trainer.config = _make_base_config()

    dataloader = trainer._create_dataloader(
        dataset=object(),
        dataset_config=object(),
        rank=0,
        world_size=1,
    )

    assert len(dataloader) == 4
    assert dataloader.batch_size == 3
    assert next(iter(dataloader)) == [{}, {}, {}]

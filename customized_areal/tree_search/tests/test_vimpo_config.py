import re

import pytest

from customized_areal.tree_search.config import AdvantageMode, Config, LossMode


def test_vimpo_defaults_and_string_round_trip() -> None:
    cfg = Config(advantage_mode="vimpo", vimpo_ref_base_url="http://ref:30000")
    assert cfg.advantage_mode is AdvantageMode.VIMPO
    assert cfg.loss_mode is LossMode.GRPO
    assert cfg.vimpo_beta == pytest.approx(5e-4)
    assert cfg.vimpo_actor_coeff == pytest.approx(5e-3)
    assert cfg.vimpo_value_loss_weight == pytest.approx(1.0)
    assert cfg.vimpo_gamma == pytest.approx(1.0)
    assert cfg.vimpo_lambda == pytest.approx(1.0)
    assert cfg.vimpo_top_k == 128
    assert cfg.vimpo_whiten_advantages is True
    assert cfg.vimpo_detach_kl is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vimpo_beta": 0.0}, "vimpo_beta must be finite and > 0"),
        ({"vimpo_actor_coeff": -1.0}, "vimpo_actor_coeff must be finite and >= 0"),
        (
            {"vimpo_value_loss_weight": -1.0},
            "vimpo_value_loss_weight must be finite and >= 0",
        ),
        (
            {"vimpo_actor_coeff": 0.0, "vimpo_value_loss_weight": 0.0},
            "at least one VIMPO loss coefficient",
        ),
        ({"vimpo_gamma": 0.99}, "vimpo_gamma must equal 1.0"),
        ({"vimpo_lambda": 1.1}, "vimpo_lambda must be in [0, 1]"),
        ({"vimpo_top_k": 0}, "vimpo_top_k must be > 0"),
        ({"vimpo_detach_kl": False}, "vimpo_detach_kl=False is not supported"),
        ({"vimpo_ref_base_url": ""}, "vimpo_ref_base_url is required"),
        (
            {"enable_generative_critic": True},
            "incompatible with enable_generative_critic",
        ),
        ({"use_clip_cov": True}, "incompatible with use_clip_cov"),
        ({"loss_mode": LossMode.DISTILL}, "requires loss_mode='grpo'"),
    ],
)
def test_vimpo_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    config_kwargs = {
        "advantage_mode": AdvantageMode.VIMPO,
        "vimpo_ref_base_url": "http://ref:30000",
    }
    config_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=re.escape(message)):
        Config(**config_kwargs)


def test_non_vimpo_mode_does_not_require_reference_url() -> None:
    assert Config().advantage_mode is AdvantageMode.TREE


def test_vimpo_local_backend_does_not_require_reference_url() -> None:
    cfg = Config(advantage_mode="vimpo", vimpo_ref_backend="local")
    assert cfg.vimpo_ref_backend == "local"
    assert cfg.vimpo_ref_path == ""


def test_vimpo_local_backend_accepts_existing_ref_path(tmp_path) -> None:
    cfg = Config(
        advantage_mode="vimpo",
        vimpo_ref_backend="local",
        vimpo_ref_path=str(tmp_path),
    )
    assert cfg.vimpo_ref_path == str(tmp_path)


def test_vimpo_rejects_unknown_ref_backend() -> None:
    with pytest.raises(
        ValueError, match="vimpo_ref_backend must be 'sglang' or 'local'"
    ):
        Config(advantage_mode="vimpo", vimpo_ref_backend="vllm")


def test_vimpo_local_backend_rejects_missing_ref_path() -> None:
    with pytest.raises(
        ValueError, match="vimpo_ref_path must be an existing directory"
    ):
        Config(
            advantage_mode="vimpo",
            vimpo_ref_backend="local",
            vimpo_ref_path="/nonexistent/vimpo/ref",
        )


def test_vimpo_sglang_backend_still_validates_url_when_explicit() -> None:
    with pytest.raises(ValueError, match="vimpo_ref_base_url is required"):
        Config(advantage_mode="vimpo", vimpo_ref_backend="sglang")

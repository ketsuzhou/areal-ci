"""Train with PPOTrainer rollouts and a selectable actor loss.

Usage:
uv run customized_areal/tiggered_training/train_triggered_sft_loss.py \
  --config customized_areal/tiggered_training/config_Qwen3-5L-9B_trggered_training.yaml \
  --loss-mode sft

uv run customized_areal/tiggered_training/train_triggered_sft_loss.py \
  --config customized_areal/tiggered_training/config_Qwen3-5L-9B_trggered_training.yaml \
  --loss-mode grpo gconfig.n_samples=4 train_dataset.batch_size=1
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Literal

project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.tiggered_training.sft_loss import (  # noqa: E402
    TriggeredSFTFSDPPPOActor,
    patch_ppo_actor_to_use_triggered_sft_loss,
)
from customized_areal.tpfc.tpfc_config import TPFCConfig  # noqa: E402

import areal.engine as engine_pkg  # noqa: E402
import areal.engine.fsdp_engine as fsdp_engine  # noqa: E402
from areal import PPOTrainer  # noqa: E402
from areal.api.cli_args import NormConfig, load_expr_config  # noqa: E402

LossMode = Literal["sft", "grpo"]


def _parse_triggered_args(args: list[str]) -> tuple[LossMode, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--loss-mode",
        choices=("sft", "grpo"),
        default="sft",
        help=(
            "Actor loss for triggered online training. "
            "'sft' patches PPOActor to masked NLL; 'grpo' keeps the stock GRPO loss."
        ),
    )
    parsed, remaining = parser.parse_known_args(args)
    return parsed.loss_mode, remaining


def _prepare_grpo_config(config: TPFCConfig) -> None:
    n_samples = config.gconfig.n_samples
    if n_samples < 2:
        raise ValueError(
            "GRPO mode needs gconfig.n_samples >= 2 so each online query forms "
            f"a non-trivial reward-normalization group; got {n_samples}."
        )

    if config.train_dataset.batch_size != 1:
        raise ValueError(
            "GRPO mode for online triggered training expects train_dataset.batch_size=1. "
            "Each dataloader item is one query, and gconfig.n_samples sessions are "
            "grouped under that query."
        )

    if config.actor.adv_norm is None:
        config.actor.adv_norm = NormConfig(mean_level="group", std_level="group")

    config.actor.adv_norm.mean_level = "group"
    config.actor.adv_norm.std_level = "group"
    config.actor.adv_norm.group_size = n_samples


def main(args: list[str]) -> None:
    loss_mode, config_args = _parse_triggered_args(args)
    config, _ = load_expr_config(config_args, TPFCConfig)

    if loss_mode == "sft":
        patch_ppo_actor_to_use_triggered_sft_loss()
        engine_pkg.FSDPPPOActor = TriggeredSFTFSDPPPOActor
        fsdp_engine.FSDPPPOActor = TriggeredSFTFSDPPPOActor
    elif loss_mode == "grpo":
        _prepare_grpo_config(config)
    else:
        raise ValueError(f"Unsupported loss mode: {loss_mode}")

    with PPOTrainer(config, train_dataset=None, valid_dataset=None) as trainer:
        trainer.train(workflow=None)


if __name__ == "__main__":
    main(sys.argv[1:])

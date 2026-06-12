"""Train with PPOTrainer rollouts and a custom SFT-style actor loss.

Usage:
uv run customized_areal/tiggered_training/train_triggered_sft_loss.py \
  --config customized_areal/tiggered_training/config_Qwen3-5L-9B_trggered_training.yaml
"""

from __future__ import annotations

import pathlib
import sys

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
from areal.api.cli_args import load_expr_config  # noqa: E402


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, TPFCConfig)
    patch_ppo_actor_to_use_triggered_sft_loss()
    engine_pkg.FSDPPPOActor = TriggeredSFTFSDPPPOActor
    fsdp_engine.FSDPPPOActor = TriggeredSFTFSDPPPOActor

    with PPOTrainer(config, train_dataset=None, valid_dataset=None) as trainer:
        trainer.train(workflow=None)


if __name__ == "__main__":
    main(sys.argv[1:])

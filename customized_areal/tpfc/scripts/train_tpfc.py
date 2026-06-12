"""
Training script for TPFC Agent with AReaL.

This script demonstrates how to train the TPFC Agent using AReaL's
PPO/GRPO trainer with the OpenAI proxy workflow.

Usage:
uv run customized_areal/tpfc/scripts/train_tpfc.py  --config customized_areal/tpfc/configs/config_tpfc.yaml
"""

# ruff: noqa: E402

import pathlib
import sys

# Add project root to path so we can import areal
project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.tpfc.tpfc_config import TPFCConfig
from customized_areal.tpfc.tpfc_dataset import get_tpfc_rl_dataset

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, TPFCConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load TPFC dataset directly using the custom loader
    train_dataset = get_tpfc_rl_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )

    valid_dataset = get_tpfc_rl_dataset(
        path=config.valid_dataset.path,
        split="test",
        tokenizer=tokenizer,
        max_length=config.valid_dataset.max_length,
    )

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(valid_dataset)} validation samples")

    # Build workflow kwargs from config
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=getattr(config.gconfig, "top_p", 1.0),
        # For openai
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

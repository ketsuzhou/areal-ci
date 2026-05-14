"""Training script for TPFC Agent with MCTS tree backup and rollout caching.

Uses CacheAwarePPOTrainer from tree_search to combine the TPFC agent and
dataset with MCTS tree backup advantages and rollout caching.  Cached
trajectories are reused across training steps; only missing rollouts are
newly generated.

Usage:
    uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml  2>&1 | tee training.log
    uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml  2>&1 | tee training.log
"""

import json
import os
import pathlib
import sys
import uuid

project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.tpfc.tpfc_config import TPFCConfig
from customized_areal.tpfc.tpfc_dataset import get_tpfc_rl_dataset
from customized_areal.tree_search.config import (
    CacheMode,
    RolloutCacheConfig,
    TreeBackupConfig,
)
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.saver import Saver

logger = logging.getLogger("TrainTPFCTreeSearch")


def _try_load_train_id_from_checkpoint(config: TPFCConfig) -> str | None:
    recover_cfg = config.recover
    for name in ("default", "critic"):
        path = Saver.get_recover_checkpoint_path(
            recover_cfg.experiment_name,
            recover_cfg.trial_name,
            recover_cfg.fileroot,
            name,
        )
        sidecar = os.path.join(path, "train_id.json")
        if os.path.isfile(sidecar):
            try:
                with open(sidecar) as f:
                    data = json.load(f)
                train_id = data["train_id"]
                if train_id:
                    return train_id
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning(
                    "Corrupt train_id.json at %s, generating new train_id",
                    sidecar,
                )
    return None


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting TPFC tree search training")

    config, _ = load_expr_config(args, TPFCConfig)

    # Restore train_id from recover checkpoint if available, otherwise generate
    restored_id = _try_load_train_id_from_checkpoint(config)
    if restored_id is not None:
        os.environ["TRAIN_ID"] = restored_id
        logger.info("Restored Train ID from checkpoint: %s", restored_id)
    elif "TRAIN_ID" not in os.environ:
        os.environ["TRAIN_ID"] = uuid.uuid4().hex
        logger.info("Generated new Train ID: %s", os.environ["TRAIN_ID"])
    else:
        logger.info("Using Train ID from environment: %s", os.environ["TRAIN_ID"])
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load TPFC dataset
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

    logger.info("Loaded %d training samples", len(train_dataset))
    logger.info("Loaded %d validation samples", len(valid_dataset))

    # Build cache / tree backup configs from overrides
    cache_dir = config.cache_dir
    if not cache_dir:
        raise ValueError(
            "cache_dir must be set when using tree search training. "
            "Pass it as a CLI override, e.g. +cache_dir=/path/to/cache"
        )

    n_samples = config.gconfig.n_samples

    # Cache mode from config (default "cross_training" set in TPFCConfig)
    tree_mode_str = config.cache_mode
    try:
        tree_mode = CacheMode(tree_mode_str)
    except ValueError:
        raise ValueError(
            f"Invalid cache_mode={tree_mode_str}. "
            f"Must be one of: {[m.value for m in CacheMode]}"
        ) from None

    cache_config = RolloutCacheConfig(
        cache_dir=cache_dir,
        enabled=True,
        n_samples=n_samples,
    )

    tree_backup_config = TreeBackupConfig(
        mode=tree_mode,
        checkpoint_dir=cache_dir,
    )

    logger.info(
        "Cache config: dir=%s, n_samples=%d, tree_mode=%s",
        cache_dir,
        n_samples,
        tree_backup_config.mode.value,
    )

    # Build workflow kwargs
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=getattr(config.gconfig, "top_p", 1.0),
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with CacheAwarePPOTrainer(
        config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

    logger.info("TPFC tree search training completed")


if __name__ == "__main__":
    main()

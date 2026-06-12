"""Training script for TPFC Agent with MCTS tree backup and rollout caching.

Uses CustomizedPPOTrainer from tree_search to combine the TPFC agent and
dataset with MCTS tree backup advantages and rollout caching.  Cached
trajectories are reused across training steps; only missing rollouts are
newly generated.

Usage:
    uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml  2>&1 | tee training.log
    uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml  2>&1 | tee training.log
"""

# ruff: noqa: E402

import json
import os
import pathlib
import sys
import uuid

project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.tpfc.tpfc_config import TPFCConfig
from customized_areal.tpfc.tpfc_dataset import get_tpfc_rl_dataset
from customized_areal.tree_search.config import LossMode, RolloutCacheConfig
from customized_areal.tree_search.core.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.training.trainer import CustomizedPPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.saver import Saver

logger = logging.getLogger("TrainTPFCTreeSearch")


def _count_cached_tree_queries(checkpoint_dir: str) -> int:
    manager = TreeCheckpointManager(checkpoint_dir)
    save_dir = pathlib.Path(manager.save_dir)
    if not save_dir.is_dir():
        return 0
    return sum(
        1
        for path in save_dir.iterdir()
        if path.is_file()
        and path.name.startswith("query_")
        and path.name.endswith(".json")
    )


def _validate_tree_search_startup(config: TPFCConfig) -> None:
    tree_search = config.tree_search
    if not tree_search.checkpoint_dir:
        raise ValueError(
            "tree_search.checkpoint_dir must be set when using tree search training. "
            "Set it in the config YAML under tree_search.checkpoint_dir."
        )

    if tree_search.loss_mode == LossMode.DISTILL:
        cached_queries = _count_cached_tree_queries(tree_search.checkpoint_dir)
        if cached_queries == 0:
            raise ValueError(
                "tree_search.loss_mode=DISTILL is cache-only, but no cached MCTS "
                f"trees were found under {tree_search.checkpoint_dir!r}. Run GRPO "
                "or BOTH mode first to populate the cache, or change loss_mode to "
                "BOTH if fresh rollout generation should be allowed."
            )
        logger.warning(
            "tree_search.loss_mode=DISTILL is cache-only: fresh rollouts will not "
            "be generated. Found cached trees for %d queries under %s.",
            cached_queries,
            tree_search.checkpoint_dir,
        )

    if (
        tree_search.max_distill_tokens
        and tree_search.max_distill_tokens > config.gconfig.max_tokens
    ):
        logger.warning(
            "tree_search.max_distill_tokens=%d exceeds gconfig.max_tokens=%d; "
            "teacher requests may still be skipped by the runtime context limit.",
            tree_search.max_distill_tokens,
            config.gconfig.max_tokens,
        )


def _try_load_train_id_from_checkpoint(config: TPFCConfig) -> str | None:
    recover_cfg = config.recover
    for name in ("default", "critic"):
        path = Saver.get_recover_checkpoint_path(
            recover_cfg.experiment_name,
            recover_cfg.trial_name,
            recover_cfg.fileroot,
            name,
        )
        logger.info(f"path: {path}")
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

    # Load .env before config so env vars are available
    try:
        from dotenv import load_dotenv

        load_dotenv(pathlib.Path(__file__).resolve().parent.parent.parent / ".env")
    except Exception:
        pass

    config, _ = load_expr_config(args, TPFCConfig)

    # Inject diagnose API key from environment if configured
    if not config.tree_search.diagnose_api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            config.tree_search.diagnose_api_key = api_key
            logger.info("Using OPENROUTER_API_KEY from environment for diagnose")
    if not config.tree_search.diagnose_base_url:
        base_url = os.environ.get("OPENROUTER_BASE_URL", "")
        if base_url:
            config.tree_search.diagnose_base_url = base_url
            logger.info("Using OPENROUTER_BASE_URL from environment for diagnose")

    # os.environ["TRAIN_ID"] = uuid.uuid4().hex
    # logger.info("Generated new Train ID: %s", os.environ["TRAIN_ID"])

    # Restore train_id from recover checkpoint if available, otherwise generate
    restored_id = _try_load_train_id_from_checkpoint(config)
    # restored_id = None
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
    tree_search = config.tree_search
    _validate_tree_search_startup(config)

    n_samples = config.gconfig.n_samples

    cache_config = RolloutCacheConfig(
        cache_dir=tree_search.checkpoint_dir,
        enabled=True,
        n_samples=n_samples,
    )

    tree_search_config = tree_search

    logger.info(
        "Cache config: dir=%s, n_samples=%d, tree_mode=%s, loss_mode=%s",
        tree_search.checkpoint_dir,
        n_samples,
        tree_search_config.mode.value,
        tree_search_config.loss_mode.value,
    )

    # Build workflow kwargs
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=getattr(config.gconfig, "top_p", 1.0),
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_tokens=config.gconfig.max_tokens,
        tree_search_config=tree_search_config,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with CustomizedPPOTrainer(
        config,
        cache_config=cache_config,
        tree_search_config=tree_search_config,
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

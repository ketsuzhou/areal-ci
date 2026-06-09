from types import SimpleNamespace

import pytest

from customized_areal.tpfc.scripts.train_tpfc_tree_search import (
    _validate_tree_search_startup,
)
from customized_areal.tree_search.config import LossMode


def _config(tmp_path, *, loss_mode=LossMode.DISTILL):
    return SimpleNamespace(
        tree_search=SimpleNamespace(
            checkpoint_dir=str(tmp_path / "tree_cache"),
            loss_mode=loss_mode,
            max_distill_tokens=0,
        ),
        gconfig=SimpleNamespace(max_tokens=1024),
    )


def test_distill_startup_requires_existing_tree_cache(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="DISTILL is cache-only"):
        _validate_tree_search_startup(config)


def test_distill_startup_accepts_non_empty_tree_cache(tmp_path):
    config = _config(tmp_path)
    cache_dir = tmp_path / "tree_cache" / "mcts_trees"
    cache_dir.mkdir(parents=True)
    (cache_dir / "query_q1.json").write_text("{}", encoding="utf-8")

    _validate_tree_search_startup(config)


def test_grpo_startup_does_not_require_tree_cache(tmp_path):
    config = _config(tmp_path, loss_mode=LossMode.GRPO)

    _validate_tree_search_startup(config)

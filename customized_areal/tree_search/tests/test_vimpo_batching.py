import pytest
import torch

from customized_areal.tree_search.training.vimpo_batching import (
    split_episode_atomic_batches,
)

from areal.api.cli_args import MicroBatchSpec


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(24).view(4, 6),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        ),
        "vimpo_episode_index": torch.tensor([[0] * 6, [0] * 6, [1] * 6, [2] * 6]),
        "vimpo_predict_mask": torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        ),
    }


def test_allocator_never_splits_episode_rows() -> None:
    result = split_episode_atomic_batches(_batch(), MicroBatchSpec(n_mbs=2))
    memberships = [
        {int(v) for v in mb["vimpo_episode_index"][:, 0]} for mb in result.mbs
    ]
    assert sum(0 in members for members in memberships) == 1
    containing = next(
        mb
        for mb in result.mbs
        if 0 in {int(v) for v in mb["vimpo_episode_index"][:, 0]}
    )
    assert (containing["vimpo_episode_index"][:, 0] == 0).sum() == 2


def test_allocator_rejects_episode_larger_than_token_limit() -> None:
    with pytest.raises(ValueError, match="episode 0.*exceeds max_tokens_per_mb=6"):
        split_episode_atomic_batches(
            _batch(), MicroBatchSpec(n_mbs=2, max_tokens_per_mb=6)
        )


def test_allocator_rejects_empty_input() -> None:
    empty = {
        "input_ids": torch.empty((0, 6), dtype=torch.long),
        "attention_mask": torch.empty((0, 6), dtype=torch.bool),
        "vimpo_episode_index": torch.empty((0, 6), dtype=torch.long),
    }
    with pytest.raises(ValueError, match="empty"):
        split_episode_atomic_batches(empty, MicroBatchSpec(n_mbs=2))


def test_allocator_rejects_non_contiguous_episode_rows() -> None:
    batch = {
        "input_ids": torch.arange(24).view(4, 6),
        "attention_mask": torch.ones(4, 6, dtype=torch.bool),
        "vimpo_episode_index": torch.tensor([[0] * 6, [1] * 6, [0] * 6, [1] * 6]),
    }
    with pytest.raises(ValueError, match="contiguous"):
        split_episode_atomic_batches(batch, MicroBatchSpec(n_mbs=2))


def test_allocator_rejects_non_one_based_turn_index() -> None:
    batch = {
        "input_ids": torch.arange(24).view(4, 6),
        "attention_mask": torch.ones(4, 6, dtype=torch.bool),
        "vimpo_episode_index": torch.tensor([[0] * 6, [0] * 6, [1] * 6, [1] * 6]),
        # zero-based turn indices instead of one-based [1, 2] per episode
        "vimpo_turn_index": torch.tensor([[0] * 6, [1] * 6, [0] * 6, [1] * 6]),
    }
    with pytest.raises(ValueError, match="one-based"):
        split_episode_atomic_batches(batch, MicroBatchSpec(n_mbs=2))


def test_allocator_preserves_non_row_metadata() -> None:
    scalar_meta = {"global_step": torch.tensor(42, dtype=torch.long)}
    batch = {**_batch(), **scalar_meta}
    result = split_episode_atomic_batches(batch, MicroBatchSpec(n_mbs=2))
    for mb in result.mbs:
        assert mb["global_step"] is scalar_meta["global_step"]


def test_allocator_forward_backward_indices_are_inverse_permutation() -> None:
    result = split_episode_atomic_batches(_batch(), MicroBatchSpec(n_mbs=2))
    forward = result.forward_indices
    backward = result.backward_indices
    assert sorted(forward) == list(range(4))
    for new_pos, old_row in enumerate(forward):
        assert backward[old_row] == new_pos

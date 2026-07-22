"""Episode-atomic micro-batch allocation for VIMPO.

VIMPO episodes are multi-turn and indivisible at the minibatch boundary: the rows
belonging to a single episode must always land in the same micro-batch so the
advantage/loss computation sees the full turn sequence. ``split_episode_atomic_batches``
groups padded batch rows by ``vimpo_episode_index`` and greedily packs complete
episodes into ``mb_spec.n_mbs`` micro-batches while respecting an optional
``mb_spec.max_tokens_per_mb`` token budget.

The allocator preserves non-row tensor/list metadata exactly as the generic
``split_padded_tensor_dict_into_mb_list`` splitter does: only tensors whose first
dimension matches the batch (row) dimension are sliced by row index; everything
else is passed through to every micro-batch by reference.
"""

from __future__ import annotations

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import MicroBatchList

__all__ = ["split_episode_atomic_batches"]


def split_episode_atomic_batches(
    data: dict[str, torch.Tensor],
    mb_spec: MicroBatchSpec,
    *,
    episode_key: str = "vimpo_episode_index",
) -> MicroBatchList:
    """Split ``data`` into micro-batches that never split a VIMPO episode.

    Episodes are identified by the per-row constant values in ``data[episode_key]``.
    Row metadata is broadcast to every valid position but the batch is
    right-padded with zeros, so constancy is checked over the valid
    (``attention_mask``) positions only: the first valid element is the row's
    value. Each episode's rows must form a contiguous block in the padded
    batch and, when ``vimpo_turn_index`` is present, must carry one-based
    turn indices ``[1, 2, ..., n]`` in row order. Episodes are packed
    largest-first into the least-loaded eligible micro-batch; an episode
    that cannot fit any micro-batch's token budget raises ``ValueError``.
    """
    if "attention_mask" not in data:
        raise ValueError("Input data must be padded and contain 'attention_mask' key.")
    if episode_key not in data:
        raise ValueError(f"Input data must contain the episode key {episode_key!r}.")

    attention_mask = data["attention_mask"].bool()
    batch_size = attention_mask.shape[0]
    if batch_size == 0:
        raise ValueError("cannot split an empty VIMPO batch")

    # Group rows by episode. Each row must belong to exactly one episode.
    # Metadata is right-padded with zeros; read valid positions only.
    episode_rows: dict[int, list[int]] = {}
    for row in range(batch_size):
        values = torch.unique(data[episode_key][row][attention_mask[row]])
        if values.numel() != 1:
            raise ValueError(f"row {row} belongs to more than one VIMPO episode")
        episode_rows.setdefault(int(values[0]), []).append(row)

    # Enforce contiguous one-based turn rows per episode. Rows of an episode must
    # be a consecutive block of the padded batch; when turn metadata is present it
    # must read [1, 2, ..., n] in row order.
    has_turn_index = "vimpo_turn_index" in data
    for episode, rows in episode_rows.items():
        rows_sorted = sorted(rows)
        expected_rows = list(range(rows_sorted[0], rows_sorted[0] + len(rows_sorted)))
        if rows_sorted != expected_rows:
            raise ValueError(
                f"episode {episode} turn rows are not contiguous in the batch: {rows_sorted}"
            )
        if has_turn_index:
            turn_per_row = [
                int(data["vimpo_turn_index"][row][attention_mask[row]][0])
                for row in rows_sorted
            ]
            expected_turns = list(range(1, len(rows_sorted) + 1))
            if turn_per_row != expected_turns:
                raise ValueError(
                    f"episode {episode} turn indices {turn_per_row} are not "
                    f"one-based contiguous {expected_turns}"
                )

    costs = {
        episode: int(attention_mask[rows].sum())
        for episode, rows in episode_rows.items()
    }
    limit = mb_spec.max_tokens_per_mb
    if limit is not None:
        for episode, cost in costs.items():
            if cost > limit:
                raise ValueError(
                    f"episode {episode} token cost {cost} exceeds max_tokens_per_mb={limit}"
                )

    groups: list[list[int]] = [[] for _ in range(mb_spec.n_mbs)]
    loads = [0] * mb_spec.n_mbs
    for episode in sorted(episode_rows, key=lambda value: (-costs[value], value)):
        eligible = [
            index
            for index, load in enumerate(loads)
            if limit is None or load + costs[episode] <= limit
        ]
        if not eligible:
            raise ValueError(f"episode {episode} cannot fit any VIMPO microbatch")
        target = min(eligible, key=lambda index: (loads[index], index))
        groups[target].extend(episode_rows[episode])
        loads[target] += costs[episode]

    groups = [rows for rows in groups if rows]
    forward_indices = [row for rows in groups for row in rows]
    backward_indices = [0] * len(forward_indices)
    for new, old in enumerate(forward_indices):
        backward_indices[old] = new
    mbs = [
        {
            key: value[rows]
            if torch.is_tensor(value) and value.shape[:1] == attention_mask.shape[:1]
            else value
            for key, value in data.items()
        }
        for rows in groups
    ]
    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=mbs,
        group_lens=loads[: len(groups)],
        forward_indices=forward_indices,
        backward_indices=backward_indices,
    )

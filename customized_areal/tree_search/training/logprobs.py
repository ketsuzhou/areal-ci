"""Custom functional utilities for token reward computation."""

import functools
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import distributed as dist

T = TypeVar("T", torch.Tensor, tuple[torch.Tensor, torch.Tensor])


def _gather_logprobs_entropy_multi_candidates(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    """Compute logprobs and entropy for multiple candidates per position.

    Args:
        logits: [seq_len, vocab_size]
        labels: [seq_len] for single candidate per position or
                [seq_len, num_candidates] for multiple candidates per position.
        temperature: Softmax temperature

    Returns:
        logprobs_labels: Same shape as labels (logprobs at specified positions)
        entropy: [seq_len] (entropy of the distribution)
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must be 2D [seq_len, vocab_size], "
            f"got {logits.ndim}D with shape {tuple(logits.shape)}"
        )
    if labels.dim() not in (1, 2):
        raise ValueError(f"labels must be 1D or 2D, got {labels.dim()}D")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            "labels sequence length must match logits sequence length: "
            f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}"
        )

    labels = labels.long()
    if ((labels < 0) | (labels >= logits.shape[-1])).any():
        raise ValueError(
            "labels contain token ids outside the local vocab range; "
            "replace invalid candidates before gathering"
        )

    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)

    if labels.dim() == 1:
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
    elif labels.dim() == 2:
        log_probs_labels = log_probs.gather(dim=-1, index=labels)

    return log_probs_labels, entropy


def gather_logprobs_entropy_multi_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy for multiple candidates per position.

    Args:
        logits: Model logits with shape [seq_len, vocab_size] or
            [seq_len, vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices for which to compute log probabilities.
            Shape: [seq_len] for single candidate per position, or
            [seq_len, num_candidates] for multiple candidates per position.
        temperature: Softmax temperature scaling. Default is 1.0.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation.
        chunk_size: Chunk size for memory-efficient processing. Default is 1024.

    Returns:
        A tuple of (logprobs, entropy):
            - logprobs: Log probabilities at the label positions (same shape as labels).
            - entropy: Entropy of the probability distribution (shape without last dim).
    """
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(
            _vocab_parallel_logprobs_entropy_multi_candidates,
            tp_group=tp_group,
            temperature=temperature,
        )
        return _chunked_apply(fn, logits, labels, chunk_size)

    return _chunked_gather_logprobs_entropy_multi_candidates(
        logits, labels, temperature, chunk_size
    )


def _chunked_apply(
    fn: Callable[[torch.Tensor, torch.Tensor], T],
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> T:
    """Apply a function in chunks along the first dimension to reduce peak memory.

    Assumes logits is 2D [seq_len, vocab_size] with batch dim already squeezed.
    The caller must handle batch dimensions before calling this function.
    """
    assert logits.ndim == 2, (
        f"_chunked_apply expects 2D logits [seq_len, vocab_size], "
        f"got {logits.ndim}D with shape {logits.shape}. "
        f"Squeeze batch dimension before calling."
    )
    total_seqlen = logits.shape[0]
    assert total_seqlen > 0, "Input logits must have at least one element"
    results: list = []

    for i in range(0, total_seqlen, chunk_size):
        end_idx = min(i + chunk_size, total_seqlen)
        chunk_result = fn(logits[i:end_idx], labels[i:end_idx])
        results.append(chunk_result)

    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        return tuple(torch.cat([r[i] for r in results]) for i in range(num_outputs))
    return torch.cat(results)


def _chunked_gather_logprobs_entropy_multi_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = functools.partial(
        _gather_logprobs_entropy_multi_candidates, temperature=temperature
    )
    return _chunked_apply(fn, logits, labels, chunk)


def _vocab_parallel_logprobs_entropy_multi_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logprobs and entropy with vocab parallelism for multiple candidates.

    Args:
        logits: Sharded logits [..., vocab_size/tp] on each TP rank
        labels: Token indices (global vocab indices)
        tp_group: Process group for tensor parallelism
        temperature: Softmax temperature

    Returns:
        logprobs: Logprobs at label positions (same shape as labels)
        entropy: Entropy of the distribution
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must be 2D [seq_len, vocab_size/tp], "
            f"got {logits.ndim}D with shape {tuple(logits.shape)}"
        )
    if labels.dim() not in (1, 2):
        raise ValueError(f"labels must be 1D or 2D, got {labels.dim()}D")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            "labels sequence length must match logits sequence length: "
            f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}"
        )

    labels = labels.long()
    if (labels < 0).any():
        raise ValueError("labels contain negative token ids")
    tp_rank = dist.get_rank(tp_group)
    partition_vocab_size = logits.size(-1)
    vocab_start_index = tp_rank * partition_vocab_size
    vocab_end_index = vocab_start_index + partition_vocab_size

    logits_max = logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

    normalized_logits = logits - logits_max
    exp_logits = normalized_logits.exp()
    sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

    softmax = exp_logits.div_(sum_exp_logits)
    log_sum_exp = sum_exp_logits.log()
    log_probs = normalized_logits - log_sum_exp
    entropy = -torch.sum(softmax * log_probs, dim=-1)

    labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
    masked_labels = labels.clone() - vocab_start_index
    masked_labels[labels_mask] = 0

    if labels.dim() == 1:
        log_probs_labels = log_probs.gather(
            dim=-1, index=masked_labels.unsqueeze(-1)
        ).squeeze(-1)
    elif labels.dim() == 2:
        log_probs_labels = log_probs.gather(dim=-1, index=masked_labels)
    else:
        raise ValueError(f"labels must be 1D or 2D, got {labels.dim()}D")

    log_probs_labels[labels_mask] = 0.0
    dist.all_reduce(log_probs_labels, op=dist.ReduceOp.SUM, group=tp_group)

    return log_probs_labels, entropy

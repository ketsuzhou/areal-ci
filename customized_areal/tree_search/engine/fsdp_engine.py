"""Custom FSDP Engine with multi-candidate logprob gathering support.

This module provides MultiCandidateFSDPEngine which extends the standard FSDPEngine
to support gathering logprobs for multiple candidate tokens per position using
`_gather_logprobs_entropy_multi_candidates`.

It also exposes the VIMPO actor-side candidate statistics helpers
(``VIMPOCandidateStats``, ``vimpo_candidate_stats_from_logits``) and the
vocab-parallel collection path used by ``MultiCandidateFSDPEngine.compute_vimpo_candidate_stats``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F

from areal.engine.core import reorder_and_pad_outputs
from areal.engine.core.model import (
    is_qwen3_5_model,
    is_qwen3_moe_model,
    is_qwen3_vl_model,
    is_qwen_vl_model,
)
from areal.engine.fsdp_engine import FSDPEngine, FSDPTrainContext
from areal.models.tree_attn.functional import gather_packed_tree_vocab_stats
from areal.utils import logging
from areal.utils.data import (
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    pad_mb_list,
    unsqueeze_mb_list,
)

from ..training.logprobs import gather_logprobs_entropy_multi_candidates
from ..training.losses.vimpo import (
    validate_vimpo_episode_consistency,
    vimpo_loss_fn,
    vimpo_loss_terms,
)
from ..training.vimpo_batching import split_episode_atomic_batches

logger = logging.getLogger("MultiCandidateFSDPEngine")


# =============================================================================
# VIMPO actor candidate statistics
# =============================================================================


@dataclass(frozen=True)
class VIMPOCandidateStats:
    """Per-position actor candidate statistics for VIMPO.

    Carries the actor's top-k candidate token log-probabilities and the
    sampled token's log-prob, *without* exporting full-vocabulary logits.
    A later task (T7) converts ``candidate_ids`` into
    ``ReferenceScoreRequest.candidate_token_ids`` and feeds
    ``sampled_logp``/``candidate_logp`` to the frozen reference scorer.

    All tensors are dense and shaped ``[B, S, ...]``:

    - ``sampled_logp``: ``[B, S]`` float - log p(sampled_token | prefix).
    - ``candidate_ids``: ``[B, S, K]`` int - global token IDs of the actor's
      top-k candidates. Masked positions are filled with ``-1``.
    - ``candidate_logp``: ``[B, S, K]`` float - log p(candidate | prefix)
      under the actor. Masked positions are ``0``.
    - ``retained_mass``: ``[B, S]`` float - ``sum(exp(candidate_logp))``
      for diagnostics; the missing mass lives outside the top-k.
    - ``predict_mask``: ``[B, S]`` bool - True where the position predicts a
      next token; False on prompt/pad/terminal positions.
    """

    sampled_logp: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_logp: torch.Tensor
    retained_mass: torch.Tensor
    predict_mask: torch.Tensor


def vimpo_candidate_stats_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    predict_mask: torch.Tensor,
    *,
    top_k: int,
) -> VIMPOCandidateStats:
    """Single-rank VIMPO candidate statistics from full-vocabulary logits.

    Computes log-softmax over the full vocabulary, takes the actor top-k
    per position, and gathers the sampled token's log-prob. Masked rows
    are sentinelled (``candidate_ids = -1``, log-probs zeroed).

    Parameters
    ----------
    logits : torch.Tensor
        Full-vocabulary logits of shape ``[B, S, V]`` (or ``[S, V]``).
    labels : torch.Tensor
        Sampled next-token IDs of shape ``[B, S]`` (or ``[S]``).
    predict_mask : torch.Tensor
        Bool tensor of shape ``[B, S]`` (or ``[S]``); True where a next
        token is predicted.
    top_k : int
        Number of candidate tokens to retain per position.
    """
    effective_k = min(top_k, logits.shape[-1])
    logp = logits.float().log_softmax(-1)
    candidate_logp, candidate_ids = torch.topk(logp, effective_k, dim=-1)
    sampled_logp = logp.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    candidate_ids = candidate_ids.masked_fill(~predict_mask.unsqueeze(-1), -1)
    candidate_logp = candidate_logp.masked_fill(~predict_mask.unsqueeze(-1), 0)
    sampled_logp = sampled_logp.masked_fill(~predict_mask, 0)
    retained_mass = candidate_logp.exp().sum(-1).masked_fill(~predict_mask, 0)
    return VIMPOCandidateStats(
        sampled_logp, candidate_ids, candidate_logp, retained_mass, predict_mask.bool()
    )


def _vocab_parallel_vimpo_candidate_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    predict_mask: torch.Tensor,
    *,
    top_k: int,
    tp_group: dist.ProcessGroup | None,
) -> VIMPOCandidateStats:
    """Vocab-parallel VIMPO candidate stats from TP-sharded logits.

    Each TP rank holds ``logits[..., :V/tp]`` for its vocab shard. The global
    log-softmax normalizer is built with ``all_reduce(MAX, group=tp_group)``
    (numerical stability) followed by ``all_reduce(SUM, group=tp_group)`` of
    the shifted exponentials. Each rank then takes its local top-k, offsets
    the local IDs by ``tp_rank * (V/tp)`` to make them global, and
    ``all_gather``-s at most ``K`` candidates per rank. The global top-k is
    selected with token ID as the deterministic tie-breaker (smaller ID wins
    on equal log-prob). The sampled token's log-prob is gathered only from
    the owning shard (others contribute 0) and ``all_reduce(SUM)``-ed.

    When ``tp_group`` is ``None`` or of size 1, falls back to the single-rank
    :func:`vimpo_candidate_stats_from_logits`.
    """
    if tp_group is None or dist.get_world_size(tp_group) <= 1:
        return vimpo_candidate_stats_from_logits(
            logits, labels, predict_mask, top_k=top_k
        )

    # Squeeze a leading batch dim of 1 so the rest operates on 2D [S, V/tp].
    squeeze_batch = False
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
        squeeze_batch = True
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels.squeeze(0)
    if predict_mask.ndim == 2 and predict_mask.shape[0] == 1:
        predict_mask = predict_mask.squeeze(0)

    if logits.ndim != 2:
        raise ValueError(
            "vocab-parallel VIMPO stats expects 2D sharded logits "
            "[S, V/tp] (or [1, S, V/tp]); got "
            f"{tuple(logits.shape)}"
        )

    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    partition_vocab_size = logits.size(-1)
    vocab_start_index = tp_rank * partition_vocab_size
    vocab_end_index = vocab_start_index + partition_vocab_size
    global_vocab_size = partition_vocab_size * tp_size

    effective_k = min(top_k, global_vocab_size)
    local_k = min(top_k, partition_vocab_size)

    logits = logits.float()
    labels = labels.long()
    predict_mask = predict_mask.bool()

    # --- Global log-softmax via all_reduce(MAX) + all_reduce(SUM) ---
    logits_max = logits.max(dim=-1, keepdim=True).values  # [S, 1]
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

    shifted = logits - logits_max
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)  # [S, 1]
    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

    log_sum_exp = sum_exp.log()
    log_probs = shifted - log_sum_exp  # [S, V/tp] - globally correct log-softmax

    # --- Local top-k + vocab offset ---
    local_candidate_logp, local_candidate_ids = torch.topk(
        log_probs, local_k, dim=-1
    )  # [S, local_k]
    local_candidate_ids = local_candidate_ids + vocab_start_index  # global IDs

    # --- all_gather at most K candidates per rank ---
    gathered_logp = dist_F.all_gather(
        local_candidate_logp.contiguous(), group=tp_group
    )  # list of [S, local_k]
    gathered_ids = dist_F.all_gather(
        local_candidate_ids.contiguous(), group=tp_group
    )  # list of [S, local_k]
    global_candidate_logp = torch.cat(gathered_logp, dim=-1)  # [S, tp * local_k]
    global_candidate_ids = torch.cat(gathered_ids, dim=-1)  # [S, tp * local_k]

    # --- Global top-k with token ID as deterministic tie-breaker ---
    # Primary: log-prob descending. Tie-breaker: smaller token ID wins.
    # Two stable sorts achieve a lexicographic (logp desc, id asc) ordering.
    id_sort = torch.argsort(global_candidate_ids, dim=-1, stable=True)
    sorted_ids = global_candidate_ids.gather(-1, id_sort)
    sorted_logp = global_candidate_logp.gather(-1, id_sort)
    # Stable sort by -logp asc == logp desc; ties keep the prior id-asc order.
    neg_logp_sort = torch.argsort(-sorted_logp, dim=-1, stable=True)
    final_ids = sorted_ids.gather(-1, neg_logp_sort)[..., :effective_k]  # [S, K]
    final_logp = sorted_logp.gather(-1, neg_logp_sort)[..., :effective_k]  # [S, K]

    # --- Sampled token log-prob: only the owning shard contributes ---
    labels_out_of_shard = (labels < vocab_start_index) | (labels >= vocab_end_index)
    masked_labels = labels.clone() - vocab_start_index
    masked_labels[labels_out_of_shard] = 0
    sampled_logp = log_probs.gather(-1, masked_labels.unsqueeze(-1)).squeeze(-1)  # [S]
    sampled_logp = sampled_logp.masked_fill(labels_out_of_shard, 0.0)
    dist.all_reduce(sampled_logp, op=dist.ReduceOp.SUM, group=tp_group)

    # --- Apply predict_mask (sentinel -1 on candidate_ids, 0 on log-probs) ---
    pm = predict_mask
    final_ids = final_ids.masked_fill(~pm.unsqueeze(-1), -1)
    final_logp = final_logp.masked_fill(~pm.unsqueeze(-1), 0.0)
    sampled_logp = sampled_logp.masked_fill(~pm, 0.0)
    retained_mass = final_logp.exp().sum(-1).masked_fill(~pm, 0.0)

    if squeeze_batch:
        final_ids = final_ids.unsqueeze(0)
        final_logp = final_logp.unsqueeze(0)
        sampled_logp = sampled_logp.unsqueeze(0)
        retained_mass = retained_mass.unsqueeze(0)
        pm = pm.unsqueeze(0)

    return VIMPOCandidateStats(
        sampled_logp=sampled_logp,
        candidate_ids=final_ids,
        candidate_logp=final_logp,
        retained_mass=retained_mass,
        predict_mask=pm,
    )


class MultiCandidateFSDPEngine(FSDPEngine):
    """FSDP Engine with multi-candidate logprob gathering support.

    This engine extends the standard FSDPEngine to support multi-candidate
    logprob gathering during training. Instead of only gathering logprobs
    for the chosen token at each position, it can gather logprobs for
    multiple candidate tokens.

    Key differences from standard FSDPEngine:
    1. Uses `gather_logprobs_entropy_multi_candidates` instead of `gather_logprobs_entropy`
    2. Prepares 2D labels: [seq_len, num_candidates] for multi-candidate positions
    3. Passes multi-candidate logprobs directly to loss function

    Multi-candidate gathering is only supported in the non-tree training path.
    For tree training (enable_tree_training=True), the logprobs are computed
    per-sequence via gather_packed_tree_logprobs_entropy and the distill loss
    is handled by the loss function reading topk_ids/teacher_logp from mb_input.

    Usage:
        engine = MultiCandidateFSDPEngine(config)
        # Input data should contain topk_ids tensor [batch, resp_len, max_candidates]
    """

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
        labels_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logprobs and entropy with multi-candidate support.

        This method replaces the standard `_compute_logprobs_entropy` to use
        `gather_logprobs_entropy_multi_candidates` which supports both:
        - Single candidate: labels shape [seq_len] or [batch, seq_len]
        - Multi-candidate: labels shape [seq_len, num_candidates] or [batch, seq_len, num_candidates]

        Parameters
        ----------
        logits : torch.Tensor
            Model logits with shape [seq_len, vocab_size] or [batch, seq_len, vocab_size].
        inputs : dict[str, Any]
            Dictionary containing:
            - rolled_input_ids: Labels for which to compute logprobs.
              Can be 1D/2D for single/multi-candidate.
            - input_ids: Used to compute rolled_input_ids if not provided.
        ulysses_pad_size : int, optional
            Size of Ulysses padding to remove from outputs.
        labels_override : torch.Tensor | None, optional
            Override labels for logprob computation. If provided, this will be used
            instead of the labels from inputs (prevents mutation of inputs dict).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - logprobs: Logprobs at label positions, same shape as labels (without vocab dim)
            - entropy: Entropy of the distribution, shape [seq_len] or [batch, seq_len]
        """
        # Use labels_override if provided (avoids mutating inputs)
        if labels_override is not None:
            labels = labels_override
        else:
            # Try to get rolled_input_ids (if Ulysses SP is enabled)
            labels = inputs.get(
                "rolled_input_ids",
                torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
            )

        # The functional gatherer chunks along dim=0, so keep logits in the
        # upstream FSDP convention: [seq_len, vocab].
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        if labels.ndim in (2, 3) and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        if logits.ndim != 2 or labels.ndim not in (1, 2):
            raise ValueError(
                "Multi-candidate logprob gathering expects logits [seq_len, vocab] "
                "and labels [seq_len] or [seq_len, num_candidates], got "
                f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )
        if labels.shape[0] != logits.shape[0]:
            raise ValueError(
                "Logits and labels sequence lengths differ: "
                f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )

        # Use multi-candidate gathering function
        logprobs, entropy = gather_logprobs_entropy_multi_candidates(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
        )

        # Handle sequence parallelism (Ulysses)
        # NOTE: Must be done after removing batch dimension to ensure slicing
        # operates on the sequence dimension, not batch or candidate dimension
        if self.parallel_helper.sp_size > 1:
            # For multi-candidate logprobs [seq_len, num_candidates],
            # we need to transpose before all_gather to ensure concatenation
            # happens along the sequence dimension (dim=0), not candidate dimension
            #
            # Check if this is a multi-candidate case. Even K=1 with an explicit
            # candidate axis must keep that axis through SP gather for the loss.
            is_multi_candidate = logprobs.ndim == 2

            if is_multi_candidate:
                # Multi-candidate case: logprobs is [seq_len, num_candidates]
                # Transpose to [num_candidates, seq_len] so all_gather can concatenate
                # along the last dimension (which becomes seq_len after transpose)
                logprobs = logprobs.transpose(0, 1)
                # all_gather concatenates along dim=-1 (which is seq_len after transpose)
                logprobs = self._sp_all_gather(logprobs)
                entropy = self._sp_all_gather(entropy)

                if ulysses_pad_size > 0:
                    logprobs = logprobs[..., :-ulysses_pad_size]
                    entropy = entropy[..., :-ulysses_pad_size]

                # transpose back: [num_candidates, gathered_seq_len] -> [gathered_seq_len, num_candidates]
                logprobs = logprobs.transpose(0, 1)
            else:
                # Single-candidate case: logprobs is [seq_len], all_gather on dim=-1 is correct
                logprobs = self._sp_all_gather(logprobs)
                entropy = self._sp_all_gather(entropy)
                if ulysses_pad_size > 0:
                    logprobs = logprobs[:-ulysses_pad_size]
                    entropy = entropy[:-ulysses_pad_size]

        return logprobs, entropy

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        """Compute logprobs with multi-candidate support (entropy discarded).

        This is a convenience wrapper around `_compute_logprobs_entropy`
        that only returns logprobs.
        """
        logprobs, _ = self._compute_logprobs_entropy(logits, inputs, ulysses_pad_size)
        return logprobs

    def _compute_tree_multi_candidate_logprobs_entropy(
        self,
        logits: torch.Tensor,
        mb_input: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Gather multi-candidate logprobs from packed-tree logits.

        The packed tree stores shared-prefix logits once, while the loss expects
        per-sequence-concatenated logprobs in ``trie.all_sequence_ids`` order.
        This mirrors ``gather_packed_tree_logprobs_entropy`` but replaces
        response-position labels with ``topk_ids`` candidates.

        Uses node-level caching: for shared-prefix nodes (where all sequences
        have identical next-token labels), logprobs/entropy are computed once
        and reused. Only response nodes with per-sequence topk_ids overrides
        are computed individually.
        """
        trie = mb_input.get("trie_node")
        topk_ids = mb_input.get("topk_ids")
        tree_input_ids = mb_input.get("input_ids")
        loss_mask = mb_input.get("loss_mask")
        cu_seqlens = mb_input.get("cu_seqlens")

        if (
            trie is None
            or topk_ids is None
            or tree_input_ids is None
            or loss_mask is None
            or cu_seqlens is None
            or topk_ids.numel() == 0
            or topk_ids.dim() != 3
            or not trie.all_sequence_ids
        ):
            return None
        if (topk_ids < 0).all():
            return None

        tree_input_ids = tree_input_ids.squeeze(0)
        loss_mask = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask
        mb_bs, resp_len, max_candidates = topk_ids.shape
        if mb_bs != len(trie.all_sequence_ids):
            logger.warning(
                "Tree topk_ids batch size (%d) does not match trie sequence count (%d); "
                "falling back to chosen-token tree logprobs.",
                mb_bs,
                len(trie.all_sequence_ids),
            )
            return None

        tp_group = (
            self.parallel_helper.tp_group if self.parallel_helper.tp_size > 1 else None
        )

        # --- Node-level caching ---
        # For internal node predictions (pred_pos -> label_pos within a node),
        # the labels are always the next token from tree_input_ids regardless
        # of the sequence. We cache (logprobs, entropy) keyed by (start, end).
        # For transitions (end of node -> start of next node), cache by
        # (pred_pos, label_pos).
        # These caches use single-candidate labels (the actual next token).
        # Multi-candidate overrides from topk_ids only apply at response
        # positions and are sequence-specific, so those are not cached.
        internal_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        transition_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        def _get_internal_logprobs_entropy(
            start: int, end: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute logprobs/entropy for positions [start..end-1] predicting [start+1..end]."""
            key = (start, end)
            if key in internal_cache:
                return internal_cache[key]
            num = end - start
            if num <= 0:
                empty = torch.empty(0, device=logits.device, dtype=torch.float)
                result = (empty, empty)
            else:
                pred_logits = logits[start:end]  # [num, vocab]
                lbl = tree_input_ids[start + 1 : end + 1]  # [num]
                # Expand to multi-candidate with same token repeated
                lbl_mc = lbl.long().unsqueeze(-1).expand(-1, max_candidates)
                lp, ent = gather_logprobs_entropy_multi_candidates(
                    pred_logits,
                    lbl_mc,
                    temperature=self.config.temperature,
                    tp_group=tp_group,
                )
                result = (lp, ent)
            internal_cache[key] = result
            return result

        def _get_transition_logprobs_entropy(
            pred_pos: int, label_pos: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute logprobs/entropy for a single transition position."""
            key = (pred_pos, label_pos)
            if key in transition_cache:
                return transition_cache[key]
            pred_logit = logits[pred_pos : pred_pos + 1]  # [1, vocab]
            lbl = tree_input_ids[label_pos : label_pos + 1].long()
            lbl_mc = lbl.unsqueeze(-1).expand(-1, max_candidates)
            lp, ent = gather_logprobs_entropy_multi_candidates(
                pred_logit,
                lbl_mc,
                temperature=self.config.temperature,
                tp_group=tp_group,
            )
            transition_cache[key] = (lp, ent)
            return (lp, ent)

        # --- Per-sequence assembly ---
        logprob_parts: list[torch.Tensor] = []
        entropy_parts: list[torch.Tensor] = []

        for b, seq_id in enumerate(trie.all_sequence_ids):
            indices = trie.get_sequence_tree_indices(seq_id)
            if not indices:
                continue

            # Determine this sequence's prompt_len to know where topk overrides start
            seg_start = int(cu_seqlens[b].item())
            seg_end = int(cu_seqlens[b + 1].item())
            seg_mask = loss_mask[seg_start:seg_end].bool()
            prompt_len = int(seg_mask.int().argmax().item()) if seg_mask.any() else 0

            seq_lp_parts: list[torch.Tensor] = []
            seq_ent_parts: list[torch.Tensor] = []
            seq_offset = 0  # tracks position in the flattened per-sequence output

            for i, (start, end) in enumerate(indices):
                num_internal = end - start
                next_start = indices[i + 1][0] if i + 1 < len(indices) else 0

                # --- Internal node positions ---
                if num_internal > 0:
                    # Check if any internal positions fall in response range
                    # and have topk_ids overrides for this sequence
                    internal_resp_start = max(0, prompt_len - seq_offset)
                    internal_resp_end = min(
                        num_internal, prompt_len + resp_len - seq_offset
                    )

                    has_override = (
                        internal_resp_end > internal_resp_start
                        and internal_resp_start < num_internal
                    )

                    if not has_override:
                        # Pure prefix node — use cache
                        lp, ent = _get_internal_logprobs_entropy(start, end)
                    else:
                        # Node has response positions with potential topk overrides
                        pred_logits = logits[start:end]
                        lbl = (
                            tree_input_ids[start + 1 : end + 1]
                            .long()
                            .unsqueeze(-1)
                            .expand(-1, max_candidates)
                            .clone()
                        )
                        # Apply topk_ids overrides at response positions
                        for j in range(num_internal):
                            resp_idx = seq_offset + j - prompt_len
                            if 0 <= resp_idx < resp_len:
                                cand = topk_ids[b, resp_idx].long()
                                if (cand >= 0).any():
                                    lbl[j] = torch.where(cand >= 0, cand, lbl[j])
                        lp, ent = gather_logprobs_entropy_multi_candidates(
                            pred_logits,
                            lbl,
                            temperature=self.config.temperature,
                            tp_group=tp_group,
                        )
                        del pred_logits, lbl

                    if lp.numel() > 0:
                        seq_lp_parts.append(lp)
                        seq_ent_parts.append(ent)

                seq_offset += num_internal

                # --- Transition position (end -> next_start) ---
                if end >= logits.shape[0] or next_start >= tree_input_ids.shape[0]:
                    logger.warning(
                        "Skipping invalid tree transition: pred_pos=%d label_pos=%d "
                        "logits_len=%d input_ids_len=%d",
                        end,
                        next_start,
                        logits.shape[0],
                        tree_input_ids.shape[0],
                    )
                    continue
                trans_resp_idx = seq_offset - prompt_len
                has_trans_override = (
                    0 <= trans_resp_idx < resp_len
                    and (topk_ids[b, trans_resp_idx] >= 0).any()
                )

                if not has_trans_override:
                    # Use cache
                    lp, ent = _get_transition_logprobs_entropy(end, next_start)
                else:
                    # Compute with topk override
                    pred_logit = logits[end : end + 1]
                    fallback = (
                        tree_input_ids[next_start : next_start + 1]
                        .long()
                        .unsqueeze(-1)
                        .expand(-1, max_candidates)
                    )
                    cand = topk_ids[b, trans_resp_idx].long().unsqueeze(0)
                    cand = torch.where(cand >= 0, cand, fallback)
                    lp, ent = gather_logprobs_entropy_multi_candidates(
                        pred_logit,
                        cand,
                        temperature=self.config.temperature,
                        tp_group=tp_group,
                    )
                    del pred_logit

                seq_lp_parts.append(lp)
                seq_ent_parts.append(ent)
                seq_offset += 1

            if seq_lp_parts:
                logprob_parts.append(torch.cat(seq_lp_parts, dim=0))
                entropy_parts.append(torch.cat(seq_ent_parts, dim=0))

        if not logprob_parts:
            return None

        return torch.cat(logprob_parts, dim=0), torch.cat(entropy_parts, dim=0)

    def _prepare_multi_candidate_labels(
        self,
        model_inputs: dict[str, Any],
        mb_input: dict[str, Any],
        seq_len: int,
    ) -> torch.Tensor | None:
        """Prepare 2D labels for multi-candidate logprob gathering.

        Reads the response-aligned topk_ids tensor from mb_input and
        expands it to full-sequence labels [seq_len, max_candidates].
        Position i in topk_ids maps to absolute sequence position prompt_len + i.

        Positions where topk_ids has -1 sentinel (no distill data for
        that node) are filled with the actual next token from
        rolled_input_ids so the engine gathers single-candidate-equivalent
        logprobs at those positions.

        Handles both single-sequence (mb_bs=1) and multi-sequence packed
        (mb_bs > 1 with cu_seqlens) micro-batches.
        """
        topk_ids = mb_input.get("topk_ids")
        if topk_ids is None or topk_ids.numel() == 0:
            return None

        # topk_ids: [mb_bs, resp_len, max_cand]
        if topk_ids.dim() != 3:
            return None

        mb_bs, resp_len, max_candidates = topk_ids.shape

        # If every response position has -1 sentinel across all sequences, no distill data
        if (topk_ids < 0).all():
            return None

        loss_mask = model_inputs.get("loss_mask")
        input_ids = model_inputs.get("input_ids")
        cu_seqlens = model_inputs.get("cu_seqlens")

        device = topk_ids.device

        if mb_bs == 1 or cu_seqlens is None:
            # Single-sequence or no cu_seqlens: original path
            topk_2d = topk_ids.squeeze(0)  # [resp_len, max_cand]
            if topk_2d.dim() != 2:
                return None

            prompt_len = 0
            if loss_mask is not None:
                lm_flat = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask
                prompt_len = int(lm_flat.bool().int().argmax().item())

            # Get rolled_input_ids for prompt and non-distill positions
            rolled = None
            if input_ids is not None:
                ids_flat = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
                rolled = torch.roll(ids_flat, shifts=-1)[:seq_len]

            labels = torch.zeros(
                seq_len, max_candidates, dtype=torch.long, device=device
            )
            if rolled is not None:
                labels[:, 0] = rolled
                for c in range(1, max_candidates):
                    labels[:, c] = rolled

            end = min(prompt_len + resp_len, seq_len)
            if end > prompt_len:
                chunk = topk_2d[: end - prompt_len]
                rolled_chunk = labels[prompt_len:end]
                chunk = torch.where(chunk >= 0, chunk, rolled_chunk)
                valid = (chunk >= 0).any(dim=-1)
                if valid.any():
                    labels[prompt_len:end][valid] = chunk[valid]

            return labels

        # Multi-sequence packed: use cu_seqlens to build labels per sequence
        labels_parts = []
        ids_flat = (
            input_ids.squeeze(0)
            if input_ids is not None and input_ids.dim() > 1
            else input_ids
        )

        for b in range(mb_bs):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            seq_len_i = end - start

            # Compute prompt_len from loss_mask segment
            prompt_len_i = 0
            if loss_mask is not None:
                lm_flat = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask
                seg = lm_flat[start:end]
                if seg.bool().any():
                    prompt_len_i = int(seg.bool().int().argmax().item())

            # Create rolled input_ids for this sequence
            if ids_flat is not None:
                seg_ids = ids_flat[start:end]
                rolled_i = torch.roll(seg_ids, shifts=-1)
            else:
                rolled_i = torch.zeros(seq_len_i, dtype=torch.long, device=device)

            labels_i = torch.zeros(
                seq_len_i, max_candidates, dtype=torch.long, device=device
            )
            labels_i[:, 0] = rolled_i
            for c in range(1, max_candidates):
                labels_i[:, c] = rolled_i

            # Overwrite response positions with topk_ids for this sequence
            end_resp = min(prompt_len_i + resp_len, seq_len_i)
            if end_resp > prompt_len_i:
                chunk = topk_ids[b, : end_resp - prompt_len_i]
                rolled_chunk = labels_i[prompt_len_i:end_resp]
                chunk = torch.where(chunk >= 0, chunk, rolled_chunk)
                valid = (chunk >= 0).any(dim=-1)
                if valid.any():
                    labels_i[prompt_len_i:end_resp][valid] = chunk[valid]

            labels_parts.append(labels_i)

        return torch.cat(labels_parts, dim=0)

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: Any,  # FSDPTrainContext
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs/entropy and return scaled loss with multi-candidate support.

        This method overrides the parent to:
        1. Prepare multi-candidate labels from topk_ids tensor
        2. Compute multi-candidate logprobs
        3. Pass multi-candidate logprobs to loss function

        For tree training (enable_tree_training=True), delegates to the base
        FSDPEngine which uses gather_packed_tree_logprobs_entropy to correctly
        unpack per-sequence logprobs from the trie structure. Multi-candidate
        logprob gathering is only supported in the non-tree training path.
        """
        local_weight = loss_weight_fn(ctx.mb_input)
        if local_weight == 0:
            return logits.mean() * 0.0

        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )

        if not self.config.is_critic:
            if not self.enable_tree_training:
                # Standard path: prepare multi-candidate labels and compute logprobs
                # Robust seq_len extraction from logits tensor
                if logits.ndim == 2:
                    seq_len = logits.shape[0]
                elif logits.ndim == 3:
                    seq_len = logits.shape[1]
                else:
                    raise ValueError(
                        f"Unexpected logits ndim: {logits.ndim}. "
                        f"Expected 2D [seq_len, vocab] or 3D [batch, seq_len, vocab]"
                    )

                # Check for multi-candidate data from topk_ids tensor
                topk_ids = ctx.mb_input.get("topk_ids")
                if topk_ids is not None and topk_ids.numel() > 0:
                    multi_candidate_labels = self._prepare_multi_candidate_labels(
                        ctx.model_inputs, ctx.mb_input, seq_len
                    )

                    if multi_candidate_labels is not None:
                        logprobs, entropy = self._compute_logprobs_entropy(
                            logits,
                            ctx.model_inputs,
                            ctx.ulysses_pad_size,
                            labels_override=multi_candidate_labels,
                        )
                    else:
                        logprobs, entropy = self._compute_logprobs_entropy(
                            logits, ctx.model_inputs, ctx.ulysses_pad_size
                        )
                else:
                    logprobs, entropy = self._compute_logprobs_entropy(
                        logits, ctx.model_inputs, ctx.ulysses_pad_size
                    )

                vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                    logits, ctx.ulysses_pad_size
                )

                if ctx.pad_length > 0:
                    logprobs = logprobs[: -ctx.pad_length]
                    entropy = entropy[: -ctx.pad_length]
                    logits = logits[: -ctx.pad_length]
                    vocab_min_logits = vocab_min_logits[: -ctx.pad_length]
                    vocab_max_logits = vocab_max_logits[: -ctx.pad_length]

                loss = loss_fn(
                    logprobs,
                    entropy,
                    ctx.mb_input,
                    vocab_min_logits=vocab_min_logits,
                    vocab_max_logits=vocab_max_logits,
                )
            else:
                tree_multi = self._compute_tree_multi_candidate_logprobs_entropy(
                    logits,
                    ctx.mb_input,
                )
                if tree_multi is None:
                    return super()._compute_logprobs_and_loss(
                        logits,
                        ctx,
                        loss_fn,
                        loss_weight_fn,
                        total_loss_weight,
                        loss_multiplier,
                    )

                logprobs, entropy = tree_multi
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    logits,
                    ctx.trie_node,
                )
                loss = loss_fn(
                    logprobs,
                    entropy,
                    ctx.mb_input,
                    vocab_min_logits=vocab_min_logits,
                    vocab_max_logits=vocab_max_logits,
                )
        else:
            values = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = local_weight / total_loss_weight * loss_multiplier
        return loss * loss_scale

    # ------------------------------------------------------------------
    # VIMPO combined actor loss (one backward, two denominators)
    # ------------------------------------------------------------------

    # Keys consumed by ``vimpo_loss_terms``.  The process callback forwards
    # whichever of these are present in the micro-batch's ``orig_mb``.
    _VIMPO_LOSS_DATA_KEYS = (
        "vimpo_predict_mask",
        "vimpo_episode_index",
        "vimpo_centered_reward",
        "vimpo_ref_sample_logp",
        "vimpo_candidate_kl",
        "vimpo_turn_index",
        "vimpo_expected_turn_count",
        "logprobs",
        "prox_logp",
        "advantages",
    )

    _VIMPO_REQUIRED_FIELDS = (
        "input_ids",
        "attention_mask",
        "vimpo_predict_mask",
        "vimpo_episode_index",
        "vimpo_centered_reward",
        "vimpo_ref_sample_logp",
        "vimpo_candidate_kl",
        "advantages",
    )

    def _validate_vimpo_batch(self, data: dict[str, Any]) -> None:
        """Validate all VIMPO batch fields and reference tensors.

        Runs on the raw (pre-split) 2D batch BEFORE ``optimizer_zero_grad``.
        Raises ``ValueError`` on any missing field, shape mismatch,
        non-finite reference value, or empty predict mask so the optimizer
        state is never mutated on invalid input.
        """
        for key in self._VIMPO_REQUIRED_FIELDS:
            if key not in data:
                raise ValueError(f"VIMPO batch missing required field: {key!r}")
        if "prox_logp" not in data and "logprobs" not in data:
            raise ValueError(
                "VIMPO batch requires at least one of 'prox_logp' or 'logprobs'"
            )

        bs, seq_len = data["attention_mask"].shape[:2]
        for key in (
            "vimpo_predict_mask",
            "vimpo_episode_index",
            "vimpo_centered_reward",
            "vimpo_ref_sample_logp",
            "vimpo_candidate_kl",
            "advantages",
        ):
            tensor = data[key]
            if tensor.shape[:2] != (bs, seq_len):
                raise ValueError(
                    f"VIMPO batch field {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected first two dims ({bs}, {seq_len})"
                )
        for key in ("prox_logp", "logprobs"):
            if key in data:
                tensor = data[key]
                if tensor.shape[:2] != (bs, seq_len):
                    raise ValueError(
                        f"VIMPO batch field {key!r} has shape {tuple(tensor.shape)}, "
                        f"expected first two dims ({bs}, {seq_len})"
                    )

        for key in (
            "vimpo_ref_sample_logp",
            "vimpo_candidate_kl",
            "vimpo_centered_reward",
            "advantages",
        ):
            tensor = data[key]
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(
                    f"VIMPO batch field {key!r} contains non-finite values (inf/nan)"
                )
        for key in ("prox_logp", "logprobs"):
            if key in data:
                tensor = data[key]
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"VIMPO batch field {key!r} contains non-finite values (inf/nan)"
                    )

        if data["vimpo_predict_mask"].sum().item() == 0:
            raise ValueError(
                "VIMPO batch has no valid predict positions "
                "(vimpo_predict_mask is all False)"
            )

        # Per-episode consistency (partial-episode turn-count completeness and
        # centered-reward agreement). Shape-agnostic: runs on the 2D raw batch
        # here so a partial episode or inconsistent centered reward fails BEFORE
        # ``optimizer_zero_grad`` rather than inside the forward-backward callback.
        validate_vimpo_episode_consistency(data, data["vimpo_predict_mask"].bool())

    def _prepare_vimpo_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        """Episode-atomic split + pack/pad for VIMPO training.

        Uses :func:`split_episode_atomic_batches` (T3) to keep multi-turn
        episodes indivisible at the micro-batch boundary, then applies the
        same pack/pad/unsqueeze helpers as ``_prepare_mb_list``.
        """
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        if self.enable_tree_training:
            raise NotImplementedError(
                "VIMPO training with tree training is not yet supported; "
                "use the non-tree (padded) path."
            )

        # VL/MoE models need model-specific position_ids / attention_mask
        # handling (``compute_3d_position_ids`` for Qwen-VL, ``attention_mask
        # = None`` for Qwen3-MoE / Qwen3-VL / Qwen3-3.5) that this method does
        # not implement. Fail loud instead of silently producing wrong
        # position_ids / attention_mask.
        model_type = self.model_config.model_type
        if (
            is_qwen_vl_model(model_type)
            or is_qwen3_vl_model(model_type)
            or is_qwen3_moe_model(model_type)
            or is_qwen3_5_model(model_type)
        ):
            raise NotImplementedError(
                "VIMPO training does not yet support VL/MoE models "
                f"(model_type={model_type!r}); use a non-VL, non-MoE model."
            )

        # Add position_ids (same as _prepare_mb_list for non-VL models).
        input_ = amend_position_ids(input_)

        # Episode-atomic split: episodes are indivisible at the mb boundary.
        mb_list = split_episode_atomic_batches(input_, self.config.mb_spec)

        # Pack each mb: [B_mb, S, ...] -> [total_content_length, ...]
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]

        # Pad for memory fragmentation / SP alignment.
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.logger.info(
            f"VIMPO microbatch #tokens (rank {rank}): "
            f"{mb_list.group_lens}, padded to: {mb_list.padded_to_lengths}, "
            f"padding lengths: {mb_list.padding_lengths}"
        )

        # Unsqueeze packed tensors to [1, padded_length] for model forward.
        mb_list = unsqueeze_mb_list(mb_list)

        # Setup attention_mask / cu_seq_lens (same as _prepare_mb_list).
        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs):
            mb["max_length_q"] = mb["max_length_k"] = mb["max_seqlen"] = int(
                mb["max_seqlen"]
            )
            padded_mb["max_length_q"] = padded_mb["max_length_k"] = padded_mb[
                "max_seqlen"
            ] = int(padded_mb["max_seqlen"])
            mb["cu_seq_lens_q"] = mb["cu_seq_lens_k"] = mb["cu_seqlens"]
            padded_mb["cu_seq_lens_q"] = padded_mb["cu_seq_lens_k"] = padded_mb[
                "cu_seqlens"
            ]
            mb["use_cache"] = False
            padded_mb["use_cache"] = False
            mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
            padded_mb["attention_mask"] = dict(
                full_attention=None, sliding_attention=None
            )

        return mb_list

    def train_vimpo_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        *,
        actor_coeff: float,
        value_loss_weight: float,
        beta: float,
        eps_clip: float,
        eps_clip_higher: float | None = None,
    ) -> dict[str, float]:
        """One zero-grad / backward / optimizer-step with two distributed denominators.

        VIMPO combines a PPO token-mean and a terminal episode-mean in a
        single backward pass.  The two means use *different* denominators
        (``valid_token_count`` and ``episode_count``), so the generic
        ``train_batch`` (single ``loss_weight_fn``) cannot express this.

        Pipeline:

        1. Normalize input and validate ALL batches/reference tensors BEFORE
           ``optimizer_zero_grad`` (fail loud on missing/invalid data).
        2. Episode-atomic split + pack/pad (episodes indivisible at mb boundary).
        3. Compute and all-reduce ``[valid_token_count, episode_count]``
           (float64) over ``dp_group`` once.
        4. ``optimizer_zero_grad()``.
        5. ``forward_backward_batch`` with a process callback that gathers
           sampled-action logprobs, calls ``vimpo_loss_terms``, and scales
           per ``vimpo_loss_fn``.
        6. ``optimizer_step()`` once.

        Parameters
        ----------
        input_
            VIMPO batch dict (or list of dicts to concatenate).
        actor_coeff
            PPO actor loss coefficient (multiplies the token-mean term).
        value_loss_weight
            Terminal value loss coefficient (multiplies the episode-mean term).
        beta
            Terminal-value temperature scaling the policy-reference KL.
        eps_clip
            PPO clip ratio bound.
        eps_clip_higher
            Decoupled upper clip bound; ``None`` falls back to ``eps_clip``.
        """
        self._ensure_ready()

        # Step 1: Normalize input.
        input_batched, _ = self._normalize_batch_input(input_)

        # Step 2: Validate ALL batches/reference tensors BEFORE optimizer_zero_grad.
        self._validate_vimpo_batch(input_batched)

        # Step 3: Episode-atomic split + pack/pad.
        mb_list = self._prepare_vimpo_mb_list(input_batched).to(self.device)

        # Step 4: Compute local denominators (float64 for all-reduce stability).
        # ``mb_list.mbs[i]`` are pre-pad content-only 1D tensors (total_content_length).
        local_valid_tokens = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        local_episodes = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        for mb in mb_list.mbs:
            pm = mb["vimpo_predict_mask"].bool()
            local_valid_tokens += pm.sum().to(torch.float64)
            ep_idx = mb["vimpo_episode_index"]
            local_episodes += torch.unique(ep_idx[pm]).numel()

        global_valid_tokens = local_valid_tokens.clone()
        global_episodes = local_episodes.clone()
        # All-reduce over dp_group (identity when dp_size=1, but keeps the
        # contract so the math is correct under data parallelism).
        dist.all_reduce(global_valid_tokens, group=self.dp_group)
        dist.all_reduce(global_episodes, group=self.dp_group)

        # Step 5: zero_grad (AFTER validation and denominator computation).
        self.optimizer_zero_grad()

        # Step 6: Forward-backward with VIMPO loss scaling.
        metric_numerators: list[tuple[torch.Tensor, ...]] = []

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)

            # Gather sampled-action logprobs (single-candidate path).
            logprobs, _ = self._compute_logprobs_entropy(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
            # Trim batch-level padding to match ctx.mb_input (content-only, 1D).
            if ctx.pad_length > 0:
                logprobs = logprobs[: -ctx.pad_length]

            # Build vimpo_data from ctx.mb_input (already 1D content-only).
            vimpo_data: dict[str, Any] = {}
            for key in self._VIMPO_LOSS_DATA_KEYS:
                if key in ctx.mb_input:
                    vimpo_data[key] = ctx.mb_input[key]

            terms = vimpo_loss_terms(
                logprobs,
                vimpo_data,
                beta=beta,
                eps_clip=eps_clip,
                eps_clip_higher=eps_clip_higher,
            )

            # Accumulate detached metric numerators for logging.
            metric_numerators.append(
                (
                    terms.ppo_sum.detach(),
                    terms.value_sum.detach(),
                    terms.valid_token_count.detach(),
                    terms.episode_count.detach(),
                )
            )

            # Scale per the two-denominator formula.
            return vimpo_loss_fn(
                terms,
                actor_coeff=actor_coeff,
                value_loss_weight=value_loss_weight,
                global_valid_tokens=global_valid_tokens,
                global_episodes=global_episodes,
                dp_size=self.parallel_helper.dp_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 7: optimizer_step (once).
        stats = self.optimizer_step()
        stats["num_micro_batches"] = len(mb_list.mbs)

        # Aggregate detached metric numerators for logging.
        if metric_numerators:
            total_ppo = torch.stack([m[0] for m in metric_numerators]).sum().item()
            total_value = torch.stack([m[1] for m in metric_numerators]).sum().item()
            stats["vimpo_ppo_sum"] = total_ppo
            stats["vimpo_value_sum"] = total_value
            stats["vimpo_valid_tokens"] = global_valid_tokens.item()
            stats["vimpo_episode_count"] = global_episodes.item()

        return stats

    # ------------------------------------------------------------------
    # VIMPO actor candidate statistics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_vimpo_candidate_stats(
        self,
        data: list[dict[str, Any]] | dict[str, Any],
        *,
        top_k: int,
    ) -> VIMPOCandidateStats:
        """Snapshot VIMPO actor candidate statistics via a no-grad eval forward.

        Forces temperature ``1.0`` for the snapshot and restores the prior
        train/eval mode afterwards. Reuses the engine's existing
        ``forward_backward_batch`` machinery to produce per-micro-batch
        logits, then for each vocab shard:

        1. Builds the global log-softmax normalizer via
           ``all_reduce(MAX, group=tp_group)`` and
           ``all_reduce(SUM, group=tp_group)``.
        2. Takes the local top-k, offsets IDs by the shard's global vocab
           offset, and ``all_gather``-s at most ``K`` candidates per rank.
        3. Selects the global top-k with token ID as the deterministic
           tie-breaker (smaller ID wins on equal log-prob).
        4. Gathers the sampled token's log-prob only from the owning
           shard (others zero) and ``all_reduce(SUM)``-s it.

        ``_sp_all_gather`` is reused for Ulysses SP collection, and
        ``gather_packed_tree_vocab_stats``/trie mappings are reused for
        packed-tree position unpacking. Returns a ``VIMPOCandidateStats``
        with all fields shaped ``[B, S, ...]``.
        """
        self._ensure_ready()

        prev_temperature = self.config.temperature
        prev_training = self.model.training if self.model is not None else False
        self.config.temperature = 1.0
        if self.model is not None:
            self.model.eval()
        try:
            return self._compute_vimpo_candidate_stats_impl(data, top_k=top_k)
        finally:
            self.config.temperature = prev_temperature
            if self.model is not None:
                self.model.train(prev_training)

    def _compute_vimpo_candidate_stats_impl(
        self,
        data: list[dict[str, Any]] | dict[str, Any],
        *,
        top_k: int,
    ) -> VIMPOCandidateStats:
        input_batched, _ = self._normalize_batch_input(data)
        cu_seqlens = pack_tensor_dict(input_batched)["cu_seqlens"]
        output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        batch_size = len(output_seqlens)

        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        tp_group = (
            self.parallel_helper.tp_group if self.parallel_helper.tp_size > 1 else None
        )

        mb_stats: list[VIMPOCandidateStats | dict[int, VIMPOCandidateStats]] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx = FSDPTrainContext(**ctx_dict)
            stats = self._vimpo_stats_from_ctx(
                logits, ctx, top_k=top_k, tp_group=tp_group
            )
            mb_stats.append(stats)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        if self.enable_tree_training:
            return self._merge_tree_vimpo_stats(mb_stats, batch_size, top_k=top_k)
        return self._reorder_vimpo_stats(mb_stats, output_seqlens, mb_list)

    def _vimpo_stats_from_ctx(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        *,
        top_k: int,
        tp_group: dist.ProcessGroup | None,
    ) -> VIMPOCandidateStats | dict[int, VIMPOCandidateStats]:
        """Compute VIMPO stats for one micro-batch.

        Returns either a flat ``VIMPOCandidateStats`` (non-tree path) whose
        leading dim is the packed sequence length, or a ``dict[int, VIMPOCandidateStats]``
        keyed by sequence id (tree path) for later merging.
        """
        if self.enable_tree_training and ctx.trie_node is not None:
            return self._vimpo_stats_from_tree_logits(
                logits, ctx, top_k=top_k, tp_group=tp_group
            )
        return self._vimpo_stats_from_flat_logits(
            logits, ctx, top_k=top_k, tp_group=tp_group
        )

    def _vimpo_stats_from_flat_logits(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        *,
        top_k: int,
        tp_group: dist.ProcessGroup | None,
    ) -> VIMPOCandidateStats:
        """Non-tree path: vocab-parallel stats from [S, V/tp] logits."""
        # Squeeze batch dim of 1 so the helper operates on 2D [S, V/tp].
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)

        model_inputs = ctx.model_inputs
        labels = model_inputs.get(
            "rolled_input_ids",
            torch.roll(model_inputs["input_ids"], shifts=-1, dims=-1),
        )
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)

        predict_mask = self._extract_vimpo_predict_mask(ctx, labels.shape[-1])
        if predict_mask.ndim == 2 and predict_mask.shape[0] == 1:
            predict_mask = predict_mask.squeeze(0)

        stats = _vocab_parallel_vimpo_candidate_stats(
            logits, labels, predict_mask, top_k=top_k, tp_group=tp_group
        )

        # Ulysses SP: gather along the sequence dimension.
        if self.parallel_helper.sp_size > 1:
            stats = self._sp_gather_vimpo_stats(stats, ctx.ulysses_pad_size)

        if ctx.pad_length > 0:
            stats = self._trim_vimpo_stats(stats, ctx.pad_length)
        return stats

    @staticmethod
    def _tree_seq_predict_mask(
        seq_out_len: int,
        seq_batch_idx: int,
        device: torch.device,
        *,
        loss_mask_packed: torch.Tensor | None,
        cu_seqlens_packed: torch.Tensor | None,
        vimpo_pm_packed: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the per-sequence ``predict_mask`` for the packed-tree path.

        The per-sequence output is assembled in sequence order with length
        ``seq_len - 1``; output index ``i`` corresponds to sequence position
        ``i`` predicting position ``i+1``. So ``predict_mask[i]`` is True only
        where position ``i+1`` is a response (``loss_mask``) token - the same
        response-aligned convention as the non-tree path's
        ``_extract_vimpo_predict_mask`` (``roll(loss_mask, -1)`` + last-False).

        ``loss_mask`` is packed per-sequence contiguously, and ``cu_seqlens``
        gives each sequence's range ``[seg_start, seg_end)``. If
        ``vimpo_predict_mask`` is provided (already shifted:
        ``vimpo_predict_mask[p] = loss_mask[p+1]``), slice it directly;
        otherwise derive from ``loss_mask``; otherwise fall back to all-True
        (matching ``_extract_vimpo_predict_mask``'s missing-loss_mask fallback).
        """
        if seq_out_len == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        if vimpo_pm_packed is not None and cu_seqlens_packed is not None:
            seg_start = int(cu_seqlens_packed[seq_batch_idx].item())
            seg_end = int(cu_seqlens_packed[seq_batch_idx + 1].item())
            pm = (
                vimpo_pm_packed.squeeze()
                if vimpo_pm_packed.dim() > 1
                else vimpo_pm_packed
            )
            # vimpo_pm[p] = loss_mask[p+1]; output index i predicts seg_start+i+1
            # so predict_mask[i] = vimpo_pm[seg_start + i], slice [seg_start, seg_end-1].
            return pm[seg_start : seg_end - 1].bool().to(device)
        if loss_mask_packed is not None and cu_seqlens_packed is not None:
            seg_start = int(cu_seqlens_packed[seq_batch_idx].item())
            seg_end = int(cu_seqlens_packed[seq_batch_idx + 1].item())
            lm = (
                loss_mask_packed.squeeze()
                if loss_mask_packed.dim() > 1
                else loss_mask_packed
            )
            # predict_mask[i] = loss_mask[seg_start + i + 1], slice [seg_start+1, seg_end].
            return lm[seg_start + 1 : seg_end].bool().to(device)
        return torch.ones(seq_out_len, dtype=torch.bool, device=device)

    def _vimpo_stats_from_tree_logits(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        *,
        top_k: int,
        tp_group: dist.ProcessGroup | None,
    ) -> dict[int, VIMPOCandidateStats]:
        """Packed-tree path: unpack per-sequence stats via trie mappings.

        Computes the vocab-parallel global log-softmax once for all packed
        positions, then uses ``trie.get_sequence_tree_indices`` to slice
        per-sequence candidate IDs/log-probs. The sampled token log-prob
        is gathered per-sequence because the next-token label diverges at
        trie branch points. Reuses ``gather_packed_tree_vocab_stats``
        internally for consistency with the loss path's position mapping.
        """
        trie = ctx.trie_node
        if trie is None or not trie.all_sequence_ids:
            return {}

        tree_input_ids = ctx.mb_input.get("input_ids")
        if tree_input_ids is None:
            return {}
        tree_input_ids = (
            tree_input_ids.squeeze(0) if tree_input_ids.dim() > 1 else tree_input_ids
        )

        # Squeeze batch dim of 1.
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        if logits.ndim != 2:
            raise ValueError(
                "vimpo tree stats expects 2D logits [T, V/tp]; got "
                f"{tuple(logits.shape)}"
            )

        # Sanity-reuse gather_packed_tree_vocab_stats so the same position
        # mapping as the loss path is exercised; the returned min/max are
        # not consumed for top-k but validate the trie <-> logits alignment.
        _vocab_min, _vocab_max = gather_packed_tree_vocab_stats(logits, trie)

        # Global log-softmax across vocab shards (all_reduce MAX + SUM).
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            tp_rank = dist.get_rank(tp_group)
            partition_vocab_size = logits.size(-1)
            vocab_start_index = tp_rank * partition_vocab_size
            vocab_end_index = vocab_start_index + partition_vocab_size

            logits_max = logits.max(dim=-1, keepdim=True).values
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
            shifted = logits.float() - logits_max
            sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
            log_probs = shifted - sum_exp.log()  # [T, V/tp]

            # Global top-k across shards via all_gather + deterministic tie-break.
            effective_k = min(
                top_k, partition_vocab_size * dist.get_world_size(tp_group)
            )
            local_k = min(top_k, partition_vocab_size)
            local_lp, local_ids = torch.topk(log_probs, local_k, dim=-1)
            local_ids = local_ids + vocab_start_index
            gathered_lp = torch.cat(
                dist_F.all_gather(local_lp.contiguous(), group=tp_group), dim=-1
            )
            gathered_ids = torch.cat(
                dist_F.all_gather(local_ids.contiguous(), group=tp_group), dim=-1
            )
            id_sort = torch.argsort(gathered_ids, dim=-1, stable=True)
            s_ids = gathered_ids.gather(-1, id_sort)
            s_lp = gathered_lp.gather(-1, id_sort)
            final_sort = torch.argsort(-s_lp, dim=-1, stable=True)
            tree_candidate_ids = s_ids.gather(-1, final_sort)[..., :effective_k]
            tree_candidate_logp = s_lp.gather(-1, final_sort)[..., :effective_k]
        else:
            log_probs = logits.float().log_softmax(-1)
            effective_k = min(top_k, logits.shape[-1])
            tree_candidate_logp, tree_candidate_ids = torch.topk(
                log_probs, effective_k, dim=-1
            )
            vocab_start_index = 0
            partition_vocab_size = logits.size(-1)
            vocab_end_index = partition_vocab_size

        results: dict[int, VIMPOCandidateStats] = {}
        # Per-sequence predict_mask alignment with the non-tree path
        # (``_vimpo_stats_from_flat_logits`` -> ``_extract_vimpo_predict_mask``):
        # position ``i`` predicts position ``i+1``, and ``predict_mask[i]`` is
        # True only where position ``i+1`` is a response (``loss_mask``) token.
        # The tree-packed ``input_ids`` includes prompt+response (the loss path
        # at ``_compute_tree_multi_candidate_logprobs_entropy`` derives
        # ``prompt_len`` from ``loss_mask``, confirming prompt is packed), so
        # the tree path must respect ``loss_mask`` to avoid marking
        # prompt-internal positions as predict=True.
        # ``loss_mask`` is packed per-sequence contiguously (by
        # ``_pack_extra_data``), and ``cu_seqlens`` gives each sequence's range.
        loss_mask_packed = ctx.mb_input.get("loss_mask")
        cu_seqlens_packed = ctx.mb_input.get("cu_seqlens")
        vimpo_pm_packed = ctx.mb_input.get("vimpo_predict_mask")
        # ``build_packed_tree_batch`` does not add ``cu_seqlens`` to the tree mb
        # (only ``_pack_extra_data`` packs per-sequence tensors). Derive it from
        # the trie's per-sequence token counts so the ``loss_mask`` slice below
        # is correct even when the workflow did not supply ``cu_seqlens``.
        if (
            cu_seqlens_packed is None
            and (loss_mask_packed is not None or vimpo_pm_packed is not None)
            and trie.all_sequence_ids
        ):
            seq_token_counts = [
                sum(
                    end - start + 1
                    for start, end in trie.get_sequence_tree_indices(sid)
                )
                for sid in trie.all_sequence_ids
            ]
            cu_seqlens_packed = torch.zeros(
                len(seq_token_counts) + 1, dtype=torch.int32, device=logits.device
            )
            for i, cnt in enumerate(seq_token_counts):
                cu_seqlens_packed[i + 1] = cu_seqlens_packed[i] + cnt

        for b, seq_id in enumerate(trie.all_sequence_ids):
            indices = trie.get_sequence_tree_indices(seq_id)
            if not indices:
                results[seq_id] = VIMPOCandidateStats(
                    sampled_logp=torch.empty(
                        0, device=logits.device, dtype=torch.float
                    ),
                    candidate_ids=torch.empty(
                        0,
                        tree_candidate_ids.shape[-1],
                        dtype=tree_candidate_ids.dtype,
                        device=logits.device,
                    ),
                    candidate_logp=torch.empty(
                        0,
                        tree_candidate_ids.shape[-1],
                        dtype=torch.float,
                        device=logits.device,
                    ),
                    retained_mass=torch.empty(
                        0, device=logits.device, dtype=torch.float
                    ),
                    predict_mask=torch.empty(0, dtype=torch.bool, device=logits.device),
                )
                continue

            cand_id_parts: list[torch.Tensor] = []
            cand_lp_parts: list[torch.Tensor] = []
            sampled_parts: list[torch.Tensor] = []

            for i, (start, end) in enumerate(indices):
                num_internal = end - start
                # Internal positions [start, end-1] predict [start+1, end].
                if num_internal > 0:
                    cand_id_parts.append(tree_candidate_ids[start:end])
                    cand_lp_parts.append(tree_candidate_logp[start:end])
                    internal_labels = tree_input_ids[start + 1 : end + 1].long()
                    sampled_parts.append(
                        self._gather_sampled_logp(
                            log_probs,
                            internal_labels,
                            vocab_start_index,
                            vocab_end_index,
                            tp_group,
                        )
                    )
                # Transition at position `end` predicts next range's start.
                if i + 1 < len(indices):
                    next_start = indices[i + 1][0]
                    cand_id_parts.append(tree_candidate_ids[end : end + 1])
                    cand_lp_parts.append(tree_candidate_logp[end : end + 1])
                    trans_label = tree_input_ids[next_start : next_start + 1].long()
                    sampled_parts.append(
                        self._gather_sampled_logp(
                            log_probs,
                            trans_label,
                            vocab_start_index,
                            vocab_end_index,
                            tp_group,
                        )
                    )

            seq_candidate_ids = torch.cat(cand_id_parts, dim=0)
            seq_candidate_logp = torch.cat(cand_lp_parts, dim=0)
            seq_sampled_logp = torch.cat(sampled_parts, dim=0)

            # Build the per-sequence predict_mask. The per-sequence output is
            # assembled in sequence order (internal positions then transitions,
            # across trie ranges) with length ``seq_len - 1``; output index ``i``
            # corresponds to sequence position ``i`` predicting ``i+1``. So
            # ``predict_mask[i] = loss_mask[seg_start + i + 1]``. If
            # ``vimpo_predict_mask`` is provided (already shifted:
            # ``vimpo_predict_mask[p] = loss_mask[p+1]``), slice it directly;
            # else derive from ``loss_mask``; else fall back to all-True
            # (matching ``_extract_vimpo_predict_mask``'s missing-loss_mask
            # fallback).
            seq_predict_mask = self._tree_seq_predict_mask(
                seq_candidate_ids.shape[0],
                b,
                logits.device,
                loss_mask_packed=loss_mask_packed,
                cu_seqlens_packed=cu_seqlens_packed,
                vimpo_pm_packed=vimpo_pm_packed,
            )

            seq_candidate_ids = seq_candidate_ids.masked_fill(
                ~seq_predict_mask.unsqueeze(-1), -1
            )
            seq_candidate_logp = seq_candidate_logp.masked_fill(
                ~seq_predict_mask.unsqueeze(-1), 0.0
            )
            seq_sampled_logp = seq_sampled_logp.masked_fill(~seq_predict_mask, 0.0)
            seq_retained_mass = (
                seq_candidate_logp.exp().sum(-1).masked_fill(~seq_predict_mask, 0.0)
            )

            results[seq_id] = VIMPOCandidateStats(
                sampled_logp=seq_sampled_logp,
                candidate_ids=seq_candidate_ids,
                candidate_logp=seq_candidate_logp,
                retained_mass=seq_retained_mass,
                predict_mask=seq_predict_mask,
            )

        return results

    @staticmethod
    def _gather_sampled_logp(
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        tp_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        """Gather sampled-token log-prob from the owning shard only."""
        if tp_group is None or dist.get_world_size(tp_group) <= 1:
            return log_probs.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
        labels = labels.long()
        out_of_shard = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked = labels.clone() - vocab_start_index
        masked[out_of_shard] = 0
        lp = log_probs.gather(-1, masked.unsqueeze(-1)).squeeze(-1)
        lp = lp.masked_fill(out_of_shard, 0.0)
        dist.all_reduce(lp, op=dist.ReduceOp.SUM, group=tp_group)
        return lp

    def _extract_vimpo_predict_mask(
        self, ctx: FSDPTrainContext, seq_len: int
    ) -> torch.Tensor:
        """Read ``vimpo_predict_mask`` from the batch, or derive it from loss_mask."""
        pm = ctx.mb_input.get("vimpo_predict_mask")
        if pm is not None:
            return pm
        loss_mask = ctx.mb_input.get("loss_mask")
        if loss_mask is None:
            return torch.ones(
                seq_len, dtype=torch.bool, device=ctx.model_inputs["input_ids"].device
            )
        pm = torch.roll(loss_mask.bool(), shifts=-1, dims=-1)
        pm[..., -1] = False
        return pm

    def _sp_gather_vimpo_stats(
        self, stats: VIMPOCandidateStats, ulysses_pad_size: int
    ) -> VIMPOCandidateStats:
        """Gather VIMPO stats along the sequence dim (Ulysses SP)."""
        if self.parallel_helper.sp_size <= 1:
            return stats
        # 1-D fields: all_gather concatenates along dim=-1 (sequence).
        sampled_logp = self._sp_all_gather(stats.sampled_logp)
        retained_mass = self._sp_all_gather(stats.retained_mass)
        # NCCL does not support bool for collectives; cast to int8 for the
        # gather, then cast back to bool. See _sp_all_gather in the base class
        # (dist.nn.functional.all_gather + torch.cat, no dtype conversion).
        predict_mask = self._sp_all_gather(stats.predict_mask.to(torch.int8)).bool()
        # 2-D fields [S, K]: transpose to [K, S] so all_gather hits the S dim.
        candidate_ids = self._sp_all_gather(stats.candidate_ids.t()).t()
        candidate_logp = self._sp_all_gather(stats.candidate_logp.t()).t()
        if ulysses_pad_size > 0:
            sampled_logp = sampled_logp[:-ulysses_pad_size]
            retained_mass = retained_mass[:-ulysses_pad_size]
            predict_mask = predict_mask[:-ulysses_pad_size]
            candidate_ids = candidate_ids[:-ulysses_pad_size]
            candidate_logp = candidate_logp[:-ulysses_pad_size]
        return VIMPOCandidateStats(
            sampled_logp=sampled_logp,
            candidate_ids=candidate_ids,
            candidate_logp=candidate_logp,
            retained_mass=retained_mass,
            predict_mask=predict_mask,
        )

    @staticmethod
    def _trim_vimpo_stats(
        stats: VIMPOCandidateStats, pad_length: int
    ) -> VIMPOCandidateStats:
        """Drop ``pad_length`` trailing positions from every field."""
        if pad_length <= 0:
            return stats
        return VIMPOCandidateStats(
            sampled_logp=stats.sampled_logp[:-pad_length],
            candidate_ids=stats.candidate_ids[:-pad_length],
            candidate_logp=stats.candidate_logp[:-pad_length],
            retained_mass=stats.retained_mass[:-pad_length],
            predict_mask=stats.predict_mask[:-pad_length],
        )

    def _reorder_vimpo_stats(
        self,
        mb_stats: list[VIMPOCandidateStats],
        output_seqlens: list[int],
        mb_list: Any,
    ) -> VIMPOCandidateStats:
        """Aggregate non-tree per-micro-batch stats to [B, S, ...]."""
        # Apply the same aggregate/unpack/reorder/pad path as forward_batch
        # to each field independently.
        fields = [
            "sampled_logp",
            "candidate_ids",
            "candidate_logp",
            "retained_mass",
            "predict_mask",
        ]
        out: dict[str, torch.Tensor] = {}
        for f in fields:
            per_mb = [getattr(s, f) for s in mb_stats]
            out[f] = reorder_and_pad_outputs(per_mb, output_seqlens, mb_list)
        # ``reorder_and_pad_outputs`` pads every field with 0 via
        # ``pad_and_stack_tensors_along_first_dim`` (F.pad value=0.0). For
        # ``candidate_ids`` that would leave token id 0 at pad positions,
        # clashing with the -1 sentinel used everywhere else for masked
        # positions. Re-apply the sentinel wherever ``predict_mask`` is False
        # (covers both in-sequence masked rows and cross-sequence padding).
        # ``predict_mask`` is [B, S] but ``candidate_ids`` is [B, S, K], so
        # unsqueeze the mask to [B, S, 1] for broadcasting.
        predict_mask = out["predict_mask"].bool()
        candidate_ids = out["candidate_ids"].masked_fill(
            ~predict_mask.unsqueeze(-1), -1
        )
        return VIMPOCandidateStats(
            sampled_logp=out["sampled_logp"],
            candidate_ids=candidate_ids,
            candidate_logp=out["candidate_logp"],
            retained_mass=out["retained_mass"],
            predict_mask=predict_mask,
        )

    def _merge_tree_vimpo_stats(
        self,
        mb_stats: list[VIMPOCandidateStats | dict[int, VIMPOCandidateStats]],
        batch_size: int,
        *,
        top_k: int,
    ) -> VIMPOCandidateStats:
        """Aggregate tree per-sequence stats to [B, S, K]."""
        # Flatten all per-sequence dicts across micro-batches.
        combined: dict[int, VIMPOCandidateStats] = {}
        for entry in mb_stats:
            if isinstance(entry, VIMPOCandidateStats):
                # Non-tree fallback (trie was None); already [B, S, ...].
                return entry
            for seq_id, stats in entry.items():
                if seq_id in combined:
                    raise ValueError(
                        f"Duplicate sequence_id {seq_id} across microbatches"
                    )
                combined[seq_id] = stats

        if not combined:
            raise ValueError("No VIMPO stats produced from any packed-tree sequence")

        max_seq_len = max(s.sampled_logp.shape[0] for s in combined.values())
        k = next(iter(combined.values())).candidate_ids.shape[-1]
        device = next(iter(combined.values())).sampled_logp.device
        b = batch_size

        def _pad_1d(
            field: str, dtype: torch.dtype, fill: float | int | bool
        ) -> torch.Tensor:
            out = torch.full((b, max_seq_len), fill, dtype=dtype, device=device)
            for seq_id, stats in combined.items():
                t = getattr(stats, field)
                out[seq_id, : t.shape[0]] = t
            return out

        def _pad_2d(field: str, dtype: torch.dtype, fill: float | int) -> torch.Tensor:
            out = torch.full((b, max_seq_len, k), fill, dtype=dtype, device=device)
            for seq_id, stats in combined.items():
                t = getattr(stats, field)
                out[seq_id, : t.shape[0]] = t
            return out

        return VIMPOCandidateStats(
            sampled_logp=_pad_1d("sampled_logp", torch.float, 0.0),
            candidate_ids=_pad_2d("candidate_ids", torch.long, -1),
            candidate_logp=_pad_2d("candidate_logp", torch.float, 0.0),
            retained_mass=_pad_1d("retained_mass", torch.float, 0.0),
            predict_mask=_pad_1d("predict_mask", torch.bool, False).bool(),
        )

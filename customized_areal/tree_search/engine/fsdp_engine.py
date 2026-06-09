"""Custom FSDP Engine with multi-candidate logprob gathering support.

This module provides MultiCandidateFSDPEngine which extends the standard FSDPEngine
to support gathering logprobs for multiple candidate tokens per position using
`_gather_logprobs_entropy_multi_candidates`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from areal.engine.fsdp_engine import FSDPEngine
from areal.models.tree_attn.functional import gather_packed_tree_vocab_stats
from areal.utils import logging

from ..training.logprobs import gather_logprobs_entropy_multi_candidates

logger = logging.getLogger("MultiCandidateFSDPEngine")


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
                    pred_logits, lbl_mc,
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
                pred_logit, lbl_mc,
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
                    internal_resp_end = min(num_internal, prompt_len + resp_len - seq_offset)

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
                            pred_logits, lbl,
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
                        pred_logit, cand,
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

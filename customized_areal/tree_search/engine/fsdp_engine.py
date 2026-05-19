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

    Usage:
        engine = MultiCandidateFSDPEngine(config)
        # Input data should contain position_rewards with candidate_token_ids
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

        # Handle batch dimension: inputs (padded_mbs) has batch dim (1, seq_len, ...)
        # We need to match logits shape which may be [seq_len, vocab] or [1, seq_len, vocab]
        if labels.ndim == 2 and labels.shape[0] == 1 and logits.ndim == 2:
            # labels [1, seq_len], logits [seq_len, vocab] -> squeeze labels
            labels = labels.squeeze(0)
        elif labels.ndim == 3 and labels.shape[0] == 1 and logits.ndim == 2:
            # labels [1, seq_len, num_candidates], logits [seq_len, vocab] -> squeeze labels
            labels = labels.squeeze(0)

        # Ensure logits has batch dimension if labels does
        if labels.ndim == 2 and logits.ndim == 2:
            # labels [seq_len, num_candidates], logits [seq_len, vocab] -> add batch dim to logits
            logits = logits.unsqueeze(0)  # [1, seq_len, vocab]
            labels = labels.unsqueeze(0)  # [1, seq_len, num_candidates]
        elif labels.ndim == 1 and logits.ndim == 2:
            # Standard case: labels [seq_len], logits [seq_len, vocab]
            logits = logits.unsqueeze(0)  # [1, seq_len, vocab]
            labels = labels.unsqueeze(0)  # [1, seq_len]

        # Use multi-candidate gathering function
        logprobs, entropy = gather_logprobs_entropy_multi_candidates(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
        )

        # Remove batch dimension BEFORE Ulysses handling to ensure correct dim alignment
        if logprobs.ndim == 3 and logprobs.shape[0] == 1:
            logprobs = logprobs.squeeze(0)
        if entropy.ndim == 2 and entropy.shape[0] == 1:
            entropy = entropy.squeeze(0)

        # Handle sequence parallelism (Ulysses)
        # NOTE: Must be done after removing batch dimension to ensure slicing
        # operates on the sequence dimension, not batch or candidate dimension
        if self.parallel_helper.sp_size > 1:
            # For multi-candidate logprobs [seq_len, num_candidates],
            # we need to transpose before all_gather to ensure concatenation
            # happens along the sequence dimension (dim=0), not candidate dimension
            #
            # Check if this is a multi-candidate case:
            # - logprobs.ndim == 2 indicates [seq_len, num_candidates] shape
            # - logprobs.shape[1] > 1 indicates more than one candidate
            is_multi_candidate = logprobs.ndim == 2 and logprobs.shape[1] > 1

            if is_multi_candidate:
                # Multi-candidate case: logprobs is [seq_len, num_candidates]
                # Transpose to [num_candidates, seq_len] so all_gather can concatenate
                # along the last dimension (which becomes seq_len after transpose)
                logprobs = logprobs.transpose(0, 1)
                entropy = entropy.transpose(0, 1)

                # all_gather concatenates along dim=-1 (which is seq_len after transpose)
                logprobs = self._sp_all_gather(logprobs)
                entropy = self._sp_all_gather(entropy)

                if ulysses_pad_size > 0:
                    logprobs = logprobs[..., :-ulysses_pad_size]
                    entropy = entropy[..., :-ulysses_pad_size]

                # transpose back: [num_candidates, gathered_seq_len] -> [gathered_seq_len, num_candidates]
                logprobs = logprobs.transpose(0, 1)
                entropy = entropy.transpose(0, 1)
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

    def _prepare_multi_candidate_labels(
        self,
        model_inputs: dict[str, Any],
        position_rewards: list,
        seq_len: int,
    ) -> torch.Tensor | None:
        """Prepare 2D labels for multi-candidate logprob gathering.

        Creates a labels tensor of shape [seq_len, max_num_candidates] where each row
        contains the candidate token IDs for that position.

        Parameters
        ----------
        model_inputs : dict[str, Any]
            Standard model inputs (for fallback to single-candidate).
            Must contain "loss_mask" to determine prompt length.
        position_rewards : list
            List of PositionRewardInfo with candidate_token_ids.
        seq_len : int
            Sequence length.

        Returns
        -------
        torch.Tensor | None
            2D labels tensor [seq_len, max_num_candidates] or None if not available.
        """
        if not position_rewards:
            return None

        # Check if any position has candidate_token_ids
        has_candidates = any(
            hasattr(pr, "candidate_token_ids") and pr.candidate_token_ids
            for pr in position_rewards
        )
        if not has_candidates:
            return None

        # Determine prompt length from loss_mask (0=prompt, 1=output)
        # PositionRewardInfo.position is 0-indexed from the first output token,
        # but labels[seq_len, max_candidates] includes prompt token positions.
        loss_mask = model_inputs.get("loss_mask")
        prompt_len = 0
        if loss_mask is not None:
            lm_flat = loss_mask.squeeze(0) if loss_mask.dim() > 1 else loss_mask
            # Vectorized: avoid O(seq_len) GPU-CPU sync in a loop.
            prompt_len = int(lm_flat.bool().int().argmax().item())

        # Find max number of candidates across positions
        max_candidates = max(
            len(pr.candidate_token_ids)
            if hasattr(pr, "candidate_token_ids") and pr.candidate_token_ids
            else 1
            for pr in position_rewards
        )

        # Create 2D labels tensor filled with ignore_index (-100)
        device = model_inputs.get("input_ids", torch.tensor([])).device
        labels = torch.full(
            (seq_len, max_candidates), -100, dtype=torch.long, device=device
        )

        # Fill in candidate token IDs for each position
        for pr in position_rewards:
            # Offset position: PositionRewardInfo.position is 0-indexed from
            # the first output token, but labels includes prompt positions.
            position = pr.position + prompt_len
            if position >= seq_len:
                continue
            if hasattr(pr, "candidate_token_ids") and pr.candidate_token_ids:
                num_candidates = len(pr.candidate_token_ids)
                labels[position, :num_candidates] = torch.tensor(
                    pr.candidate_token_ids, dtype=torch.long, device=device
                )

        return labels

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
        1. Prepare multi-candidate labels from position_rewards
        2. Compute multi-candidate logprobs
        3. Pass multi-candidate logprobs to loss function
        """

        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )

        if not self.config.is_critic:
            if not self.enable_tree_training:
                return super()._compute_logprobs_and_loss(
                    logits,
                    ctx,
                    loss_fn,
                    loss_weight_fn,
                    total_loss_weight,
                    loss_multiplier,
                )
            else:
                # Handle empty tree (dummy trie for DP synchronization).
                # When trie has no sequences, return zero loss with grad connection.
                # This ensures backward() works correctly for FSDP synchronization.
                if ctx.trie_node is None or not ctx.trie_node.all_sequence_ids:
                    return logits.mean() * 0.0

                # Check if we have multi-candidate data
                position_rewards = ctx.mb_input.get("position_rewards")

                # Robust seq_len extraction from logits tensor
                # logits can be 2D [seq_len, vocab] or 3D [batch, seq_len, vocab]
                if logits.ndim == 2:
                    seq_len = logits.shape[0]
                elif logits.ndim == 3:
                    seq_len = logits.shape[1]
                else:
                    raise ValueError(
                        f"Unexpected logits ndim: {logits.ndim}. "
                        f"Expected 2D [seq_len, vocab] or 3D [batch, seq_len, vocab]"
                    )

                # Explicit None check and non-empty check for position_rewards
                if position_rewards is not None and len(position_rewards) > 0:
                    # Prepare multi-candidate labels
                    multi_candidate_labels = self._prepare_multi_candidate_labels(
                        ctx.model_inputs, position_rewards, seq_len
                    )

                    if multi_candidate_labels is not None:
                        # Pass multi-candidate labels directly without mutating
                        # ctx.model_inputs, avoiding potential race conditions.
                        logprobs, entropy = self._compute_logprobs_entropy(
                            logits,
                            ctx.model_inputs,
                            ctx.ulysses_pad_size,
                            labels_override=multi_candidate_labels,
                        )
                    else:
                        # Fallback to standard single-candidate
                        logprobs, entropy = self._compute_logprobs_entropy(
                            logits, ctx.model_inputs, ctx.ulysses_pad_size
                        )
                else:
                    # Standard single-candidate path
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

            # Pass multi-candidate logprobs to loss function
            # The loss function receives logprobs with gradients already computed
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

        loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
        return loss * loss_scale

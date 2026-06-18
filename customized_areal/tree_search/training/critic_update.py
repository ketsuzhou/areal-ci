# SPDX-License-Identifier: Apache-2.0
# customized_areal/tree_search/training/critic_update.py
"""Shared-model critic regression step (combined objective).

In shared mode the actor model *is* the critic. After the actor PPO update we run
an additional regression step on the same model: the critic prompt is forwarded,
the digit-label tokens ``0..score_max`` are gathered at the answer position (via
the existing multi-candidate logprob machinery), and the expected value is
regressed toward the tree ``q_value`` with ``critic_loss_weight``.

The combined objective is therefore::

    actor PPO update (GAE advantages from the critic's values)
    + critic_loss_weight * MSE(expected_value, q_value)

The critic minibatch sets ``loss_mask = 1`` only at the answer position and
``topk_ids`` there to the digit-label tokens, so the multi-candidate engine path
returns exactly the per-label logprobs the loss needs.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from areal.utils import logging, stats_tracker

from .losses.critic import critic_softreg_loss_from_logprobs

logger = logging.getLogger("GenerativeCritic")


def critic_multicandidate_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, Any],
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    critic_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Loss fn for the critic regression step on the multi-candidate engine path.

    ``logprobs`` are the per-candidate logprobs gathered at the answer positions,
    shape ``[B, score_max+1]`` (ordered by digit label). ``input_data`` carries
    ``critic_targets`` (``[B]``, pre-normalized to ``[0, 1]``) and
    ``critic_score_max``. The returned loss is scaled by ``critic_loss_weight``
    so it composes with the actor objective on the shared model.
    """
    targets = input_data["critic_targets"].to(logprobs.dtype)
    score_max = int(input_data["critic_score_max"])
    if logprobs.dim() == 1:
        logprobs = logprobs.unsqueeze(0)
    loss, stats = critic_softreg_loss_from_logprobs(logprobs, targets, score_max)

    stats_tracker.scalar(
        critic_loss=float(stats["critic_loss"].detach().cpu()),
        critic_value_mean=float(stats["critic_value"].mean().detach().cpu()),
        critic_target_mean=float(stats["critic_target"].mean().detach().cpu()),
    )
    return loss * critic_loss_weight


def build_critic_minibatch(
    critic_train_data: dict[str, Any],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Materialize a critic regression minibatch tensor dict.

    Parameters
    ----------
    critic_train_data : dict
        Output of ``build_critic_training_batch`` with keys ``prompt_ids``
        (list[list[int]]), ``answer_pos`` (LongTensor[B]), ``targets``
        (FloatTensor[B]), ``leading_token_ids`` (list[int]).
    pad_token_id : int
        Padding id for ragged prompts.

    Returns
    -------
    dict with ``input_ids``, ``attention_mask``, ``loss_mask``, ``topk_ids``
    (answer-position digit candidates), ``critic_targets``, ``critic_score_max``.
    """
    prompt_ids = critic_train_data["prompt_ids"]
    answer_pos = critic_train_data["answer_pos"]
    targets = critic_train_data["targets"]
    leading_token_ids = critic_train_data["leading_token_ids"]

    batch_size = len(prompt_ids)
    if batch_size == 0:
        raise ValueError("critic_train_data has no prompts")
    max_len = max(len(p) for p in prompt_ids)
    n_labels = len(leading_token_ids)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.int32)
    for i, p in enumerate(prompt_ids):
        input_ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, : len(p)] = True
        loss_mask[i, int(answer_pos[i])] = 1

    # One response position per sample (the answer position); candidates are the
    # digit-label leading tokens.
    topk_ids = (
        torch.tensor(leading_token_ids, dtype=torch.int32)
        .view(1, 1, n_labels)
        .expand(batch_size, 1, n_labels)
        .contiguous()
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "topk_ids": topk_ids,
        "critic_targets": targets.float(),
        "critic_score_max": torch.tensor(n_labels - 1, dtype=torch.long),
    }


def run_critic_regression_step(
    actor: Any,
    critic_train_data: dict[str, Any] | None,
    *,
    critic_loss_weight: float,
    pad_token_id: int = 0,
) -> bool:
    """Run one critic regression train step on the shared model.

    Returns True if a step ran, False if skipped (no critic data). Any engine
    error is caught and logged so it can never crash the actor update.
    """
    if not critic_train_data:
        return False
    try:
        minibatch = build_critic_minibatch(critic_train_data, pad_token_id)
        engine = actor.engine
        loss_fn = functools.partial(
            critic_multicandidate_loss_fn, critic_loss_weight=critic_loss_weight
        )
        engine.train_batch(
            minibatch,
            loss_fn=loss_fn,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Critic regression step skipped (engine error): %s", exc)
        return False

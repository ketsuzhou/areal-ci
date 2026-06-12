"""Teacher distillation and evidence-reweighting loss helpers.

This module implements the teacher-signal half of the combined GRPO + distill
objective (see ``combined.py::grpo_distill_loss_fn``). A *teacher* is a model
that scores the student's sampled tokens with access to privileged context
(e.g. a hint or the gold answer); the student is the policy being trained.
Two complementary ways of consuming the teacher signal are provided:

1. **Evidence reweighting** (``_compute_distill_reweighted_advantages``).
   Used when GRPO is still active (``distill_loss_mode in {"evidence", "rlsd"}``
   and ``rl_loss_weight != 0``). The per-token advantage is rescaled by a
   detached, direction-aware weight derived from the teacher/student
   log-probability gap ``delta = teacher_logp - student_logp``. The teacher
   only *redistributes credit*; it never contributes gradients directly.

2. **KL distillation** (``_compute_teacher_kl_loss``).
   Used as an auxiliary term (``distill_loss_weight``) or, when
   ``rl_loss_weight == 0`` ("DISTILL" mode), as the sole loss. The KL is taken
   over the candidate subset that the teacher actually scored.

Tensor-format conventions
--------------------------
Tensors arrive in one of two layouts, both supported throughout this module:

- **2D batched**: ``loss_mask`` is ``[batch, seq_len]`` and per-sequence
  tensors are indexed by row ``b``.
- **1D packed**: ``loss_mask`` is ``[total_tokens]`` (all sequences
  concatenated) and per-sequence boundaries come from
  ``input_data["cu_seqlens"]`` (cumulative sequence lengths, length
  ``num_seqs + 1``). This is the layout used by the FSDP packed-tree path.

``teacher_logp`` is built **response-aligned** upstream
(``tree_store._node_to_tensor_dict``): shape ``[batch, resp_len, num_candidates]``
where response index ``i`` maps to absolute position ``prompt_len + i``. The
trailing candidate axis is ``1`` for the evidence path (chosen token only) and
``>= 1`` for the multi-candidate KL path. Padding tokens are stored as exactly
``0.0`` and are filtered with an ``abs() > 1e-8`` validity test.

Glossary
--------
- ``target`` / ``student_context_logprobs``: rollout log-probs under the
  student prompt, in the same layout as ``loss_mask``.
- ``prompt_lens[b]``: number of leading non-response tokens for sequence ``b``.
- ``valid`` mask: positions where a real (non-padding) teacher value exists
  *and* ``loss_mask`` is set.
"""

from __future__ import annotations

import torch


def _select_chosen_logprobs(
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Select chosen-token logprobs from optional candidate dimensions.

    Assumes ``loss_mask`` is never expanded to match multi-candidate
    ``logprobs`` shape.  Standard shapes are ``[seq_len]`` or
    ``[batch, seq_len]``.
    """
    if logprobs.dim() == 1:
        return logprobs

    # Multi-candidate: logprobs has more dims than loss_mask.
    if logprobs.dim() > loss_mask.dim():
        if logprobs.dim() == 2:
            return logprobs[:, 0]
        if logprobs.dim() == 3:
            return logprobs[..., 0]

    # Same dims: single-candidate (return as-is).
    if logprobs.dim() == loss_mask.dim() and logprobs.dim() == 2:
        return logprobs

    raise ValueError(f"Unsupported logprobs shape for distill loss: {logprobs.shape}")


def _compute_distill_reweighted_advantages(
    advantages: torch.Tensor,
    student_context_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int],
    input_data: dict | None = None,
    eps_clip: float | None = None,
    mixing_coeff: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply direction-aware evidence reweighting to token advantages.

    For each response token the teacher/student log-probability gap
    ``delta = teacher_logp - student_logp`` is turned into a multiplicative
    credit weight::

        signed_delta   = sign(advantage) * delta
        evidence_weight = exp(signed_delta)                       # detached
        credit_weight   = clamp(evidence_weight, 1-eps, 1+eps)    # if eps_clip
        credit_weight   = (1 - mixing_coeff) + mixing_coeff * credit_weight
        reweighted_adv  = advantage * credit_weight

    Multiplying by ``sign(advantage)`` makes the reweighting direction-aware:
    when the teacher is *more* confident than the student (``delta > 0``) the
    magnitude of a positive advantage is amplified and a negative advantage is
    damped, and vice-versa. All weight terms are ``detach``-ed so the teacher
    signal only redistributes credit across tokens and never contributes
    gradients of its own.

    Parameters
    ----------
    advantages : torch.Tensor
        Per-token advantages, same layout as ``loss_mask``.
    student_context_logprobs : torch.Tensor
        Rollout log-probability of the sampled token under the *student*
        prompt (i.e. ``old_logp``). Doubles as the alignment ``target``.
    teacher_logprobs : torch.Tensor
        Response-aligned teacher log-probabilities for the same sampled token,
        shape ``[batch, resp_len, 1]`` (see module docstring).
    loss_mask : torch.Tensor
        ``1`` on response tokens, ``0`` elsewhere. 1D packed or 2D batched.
    prompt_lens : list[int]
        Per-sequence prompt length used to place response-aligned values.
    input_data : dict | None
        Carries ``cu_seqlens`` for the 1D packed layout.
    eps_clip : float | None
        If set (``>= 0``), clamp ``evidence_weight`` to ``[1-eps, 1+eps]``.
    mixing_coeff : float
        Interpolation in ``[0, 1]`` between no reweighting (``0`` -> weight 1)
        and full reweighting (``1``).

    Returns
    -------
    reweighted_advantages : torch.Tensor
        ``advantages`` scaled by the detached ``credit_weight``.
    stat : dict[str, torch.Tensor]
        Detached ``delta``, ``evidence_weight`` and ``credit_weight`` tensors
        (set to neutral values on invalid positions) for metric logging.

    Raises
    ------
    ValueError
        If ``eps_clip < 0`` or ``mixing_coeff`` is outside ``[0, 1]``.
    """
    teacher_chosen_logprobs, valid_mask = _align_teacher_chosen_logprobs(
        teacher_logprobs=teacher_logprobs,
        target=student_context_logprobs,
        loss_mask=loss_mask,
        prompt_lens=prompt_lens,
        input_data=input_data,
    )
    valid_mask = valid_mask & loss_mask.bool()

    delta = (teacher_chosen_logprobs - student_context_logprobs).detach()
    signed_delta = torch.sign(advantages.detach()) * delta
    evidence_weight = torch.exp(signed_delta)
    evidence_weight = torch.where(
        valid_mask, evidence_weight, torch.ones_like(evidence_weight)
    )

    if eps_clip is not None:
        if eps_clip < 0:
            raise ValueError(f"distill_eps_clip must be non-negative, got {eps_clip}")
        credit_weight = torch.clamp(evidence_weight, 1.0 - eps_clip, 1.0 + eps_clip)
    else:
        credit_weight = evidence_weight

    if not 0.0 <= mixing_coeff <= 1.0:
        raise ValueError(f"distill_mixing_coeff must be in [0, 1], got {mixing_coeff}")
    credit_weight = (1.0 - mixing_coeff) + mixing_coeff * credit_weight
    reweighted_advantages = advantages * credit_weight.detach()

    stat = {
        "delta": torch.where(valid_mask, delta, torch.zeros_like(delta)).detach(),
        "evidence_weight": torch.where(
            valid_mask, evidence_weight, torch.ones_like(evidence_weight)
        ).detach(),
        "credit_weight": torch.where(
            valid_mask, credit_weight, torch.ones_like(credit_weight)
        ).detach(),
    }
    return reweighted_advantages, stat


def _align_teacher_chosen_logprobs(
    teacher_logprobs: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int],
    input_data: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter response-aligned teacher log-probs onto the student layout.

    The teacher tensor is stored response-first (index ``0`` == first response
    token), whereas ``target``/``loss_mask`` span the full prompt+response
    sequence. This helper produces an ``aligned`` tensor shaped like ``target``
    with teacher values written at the matching response positions, plus a
    ``valid`` mask marking which of those positions hold a real (non-padding)
    teacher value.

    Supported shape combinations (checked in order):

    1. ``teacher.shape == target.shape`` -> used as-is (already aligned).
    2. 1D packed ``target`` + ``[B, resp_len(, 1)]`` teacher + ``cu_seqlens``:
       trailing singleton candidate dims are squeezed, then each sequence's
       response slice is filled via ``cu_seqlens`` boundaries.
    3. 1D packed ``target`` + 1D teacher: teacher values are dropped onto the
       first ``loss_mask`` non-zero positions (single-sequence fallback).
    4. 2D batched ``target`` + 2D teacher: per-row response slice fill.

    Parameters
    ----------
    teacher_logprobs : torch.Tensor
        Response-aligned teacher log-probs, ``[B, resp_len, num_candidates]``,
        ``[B, resp_len]``, or already-aligned matching ``target``.
    target : torch.Tensor
        Reference tensor defining the output layout (the student logprobs).
    loss_mask : torch.Tensor
        Response mask in the same layout as ``target``.
    prompt_lens : list[int]
        Per-sequence prompt length; the response is assumed to start here.
    input_data : dict | None
        Carries ``cu_seqlens`` for the 1D packed layout.

    Returns
    -------
    aligned : torch.Tensor
        ``zeros_like(target)`` with teacher values scattered into response
        positions.
    valid : torch.Tensor (bool)
        True where ``aligned`` holds a real teacher value.

    Raises
    ------
    ValueError
        If the ``teacher``/``target`` shape combination is not supported
        (e.g. a multi-candidate ``[B, resp_len, C>1]`` teacher against a 1D
        packed target, which the evidence path never produces).

    Notes
    -----
    Alignment assumes each sequence has a **single contiguous** response
    region: teacher values are written into the slice
    ``[start + prompt_len, start + prompt_len + resp_len)``. For sequences with
    multiple non-contiguous ``loss_mask`` segments (some multi-turn layouts)
    this contiguous placement can misalign later turns. This assumption is
    shared with the 2D batched branch. Padding-induced over-reads are bounded
    by ``teacher.shape[1]`` and filtered out by the ``abs() > 1e-8`` check.
    """
    if teacher_logprobs.numel() == 0:
        return torch.zeros_like(target), torch.zeros_like(target, dtype=torch.bool)

    # Case 1: teacher already matches the target layout.
    if teacher_logprobs.shape == target.shape:
        teacher = teacher_logprobs.to(device=target.device, dtype=target.dtype)
        valid = loss_mask.bool() & (teacher.abs() > 1e-8)
        return teacher, valid

    mask = loss_mask.bool()
    teacher = teacher_logprobs.to(device=target.device, dtype=target.dtype)
    aligned = torch.zeros_like(target)
    valid = torch.zeros_like(target, dtype=torch.bool)

    # Squeeze trailing singleton candidate dimensions first.
    # Handles teacher [B, resp_len, 1] -> [B, resp_len] when target is
    # 1D packed, where teacher has 2 extra dims (batch + candidates).
    while teacher.dim() > target.dim() + 1 and teacher.shape[-1] == 1:
        teacher = teacher[..., 0]

    cu_seqlens = (input_data or {}).get("cu_seqlens")
    # Case 2: 1D packed target with per-sequence cu_seqlens boundaries.
    if target.dim() == 1 and teacher.dim() == 2 and cu_seqlens is not None:
        for b in range(min(teacher.shape[0], len(cu_seqlens) - 1)):
            start = int(cu_seqlens[b].item())
            end = int(cu_seqlens[b + 1].item())
            pl = prompt_lens[b] if b < len(prompt_lens) else 0
            seg_mask = mask[start:end]
            resp_len = min(int(seg_mask[pl:].sum().item()), teacher.shape[1])
            if resp_len <= 0:
                continue
            dst = slice(start + pl, start + pl + resp_len)
            aligned[dst] = teacher[b, :resp_len]
            valid[dst] = seg_mask[pl : pl + resp_len] & (
                teacher[b, :resp_len].abs() > 1e-8
            )
        return aligned, valid

    # Non-packed fallback: collapse a single trailing candidate dim.
    if teacher.dim() == target.dim() + 1:
        teacher = teacher[..., 0]

    # Case 3: 1D packed target + 1D teacher (single-sequence fallback).
    if target.dim() == 1 and teacher.dim() == 1:
        response_positions = mask.nonzero(as_tuple=False).squeeze(-1)
        n_pos = min(response_positions.numel(), teacher.shape[0])
        if n_pos > 0:
            positions = response_positions[:n_pos]
            aligned[positions] = teacher[:n_pos]
            valid[positions] = teacher[:n_pos].abs() > 1e-8
        return aligned, valid

    # Case 4: 2D batched target + 2D teacher, per-row response slice.
    if target.dim() == 2 and teacher.dim() == 2:
        batch_size = min(target.shape[0], teacher.shape[0])
        for b in range(batch_size):
            pl = prompt_lens[b] if b < len(prompt_lens) else 0
            resp_len = min(int(mask[b, pl:].sum().item()), teacher.shape[1])
            if resp_len <= 0:
                continue
            aligned[b, pl : pl + resp_len] = teacher[b, :resp_len]
            valid[b, pl : pl + resp_len] = mask[b, pl : pl + resp_len] & (
                teacher[b, :resp_len].abs() > 1e-8
            )
        return aligned, valid

    raise ValueError(
        "Unsupported teacher_logp shape for distill evidence loss: "
        f"teacher={teacher_logprobs.shape}, target={target.shape}, "
        f"loss_mask={loss_mask.shape}"
    )


def _compute_teacher_kl_loss(
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int],
    teacher_logprobs: torch.Tensor | None = None,
    input_data: dict | None = None,
    distill_kl_mode: str = "reverse_kl",
    position_rewards: list | None = None,
) -> torch.Tensor:
    """Compute teacher KL distillation loss from batched teacher_logprobs tensor.

    teacher_logprobs is response-aligned with shape [batch, resp_len, max_candidates].
    Position i in the response maps to absolute sequence position prompt_len + i.

    Supports both batched (loss_mask 2D) and 1D packed (loss_mask 1D + cu_seqlens)
    formats for logprobs and loss_mask.
    """
    if teacher_logprobs is None and position_rewards is not None:
        return _compute_position_reward_teacher_kl_loss(
            position_rewards=position_rewards,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=prompt_lens,
        )
    if teacher_logprobs is None or teacher_logprobs.numel() == 0:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    terms: list[torch.Tensor] = []
    mask = loss_mask.bool()
    batch_size = teacher_logprobs.shape[0]
    max_resp = teacher_logprobs.shape[1]

    is_multi_candidate = logprobs.dim() > loss_mask.dim() or (
        logprobs.dim() == loss_mask.dim() and logprobs.shape != loss_mask.shape
    )

    # 1D packed format: use cu_seqlens for per-sequence boundaries
    cu_seqlens = (input_data or {}).get("cu_seqlens")
    if mask.dim() == 1 and cu_seqlens is not None:
        for b in range(len(cu_seqlens) - 1):
            if b >= batch_size:
                break
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            # Recompute prompt_len from loss_mask for this segment
            seg_mask = mask[start:end]
            pl = prompt_lens[b] if b < len(prompt_lens) else 0
            if seg_mask.any():
                pl = int(seg_mask.int().argmax().item())
            resp_len = (end - start) - pl
            n_pos = min(resp_len, max_resp)
            if n_pos == 0:
                continue

            if is_multi_candidate:
                num_cand = min(
                    teacher_logprobs.shape[2],
                    logprobs.shape[1] if logprobs.dim() == 2 else logprobs.shape[2],
                )
                student = logprobs[start + pl : start + pl + n_pos, :num_cand]
                teacher = teacher_logprobs[b, :n_pos, :num_cand]
                valid = teacher.abs().sum(dim=-1) > 1e-8
                if valid.any():
                    terms.append(
                        _compute_subset_kl(
                            student_logprobs=student[valid],
                            teacher_logprobs=teacher[valid],
                            distill_kl_mode=distill_kl_mode,
                        ).reshape(-1)
                    )
            else:
                student = logprobs[start + pl : start + pl + n_pos]
                teacher = teacher_logprobs[b, :n_pos, 0]
                valid = teacher.abs() > 1e-8
                if valid.any():
                    terms.append(
                        _compute_subset_kl(
                            student_logprobs=student[valid].unsqueeze(-1),
                            teacher_logprobs=teacher[valid].unsqueeze(-1),
                            distill_kl_mode=distill_kl_mode,
                        ).reshape(-1)
                    )

        if not terms:
            return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)
        return torch.cat(terms).mean()

    # Original batched format: loss_mask is 2D [batch, seq_len]
    is_batched = mask.dim() == 2 or logprobs.dim() >= 3

    for b in range(batch_size):
        pl = prompt_lens[b] if b < len(prompt_lens) else 0
        resp_len = mask[b, pl:].sum().item() if is_batched else mask[pl:].sum().item()
        n_pos = min(resp_len, max_resp)
        if n_pos == 0:
            continue

        if is_multi_candidate:
            num_cand = min(
                teacher_logprobs.shape[2],
                logprobs.shape[2] if logprobs.dim() == 3 else logprobs.shape[1],
            )
            if logprobs.dim() == 3:
                student = logprobs[b, pl : pl + n_pos, :num_cand]
            else:
                student = logprobs[pl : pl + n_pos, :num_cand]
            teacher = teacher_logprobs[b, :n_pos, :num_cand]
            valid = teacher.abs().sum(dim=-1) > 1e-8
            if valid.any():
                terms.append(
                    _compute_subset_kl(
                        student_logprobs=student[valid],
                        teacher_logprobs=teacher[valid],
                        distill_kl_mode=distill_kl_mode,
                    ).reshape(-1)
                )
        else:
            if is_batched:
                student = logprobs[b, pl : pl + n_pos]
            else:
                student = logprobs[pl : pl + n_pos]
            teacher = teacher_logprobs[b, :n_pos, 0]
            valid = teacher.abs() > 1e-8
            if valid.any():
                terms.append(
                    _compute_subset_kl(
                        student_logprobs=student[valid].unsqueeze(-1),
                        teacher_logprobs=teacher[valid].unsqueeze(-1),
                        distill_kl_mode=distill_kl_mode,
                    ).reshape(-1)
                )

    if not terms:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    return torch.cat(terms).mean()


def _compute_position_reward_teacher_kl_loss(
    position_rewards: list,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    prompt_lens: list[int],
) -> torch.Tensor:
    """Legacy PositionRewardInfo KL path used by older tests/callers."""
    if not position_rewards:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)

    terms: list[torch.Tensor] = []
    mask = loss_mask.bool()
    is_batched = mask.dim() == 2 or logprobs.dim() >= 3

    for pr in position_rewards:
        teacher_lps = getattr(pr, "teacher_logprobs", None)
        if not teacher_lps:
            continue

        sample_idx = getattr(pr, "sample_index", 0)
        if isinstance(prompt_lens, list):
            pl = prompt_lens[sample_idx] if sample_idx < len(prompt_lens) else 0
        else:
            pl = int(prompt_lens)
        abs_pos = int(getattr(pr, "position", 0)) + pl

        if is_batched:
            if sample_idx >= logprobs.shape[0] or abs_pos >= logprobs.shape[1]:
                continue
            pos_mask = mask[sample_idx, abs_pos]
            student_row = logprobs[sample_idx, abs_pos]
        else:
            if abs_pos >= logprobs.shape[0]:
                continue
            pos_mask = mask[abs_pos]
            student_row = logprobs[abs_pos]
        if not bool(pos_mask):
            continue

        teacher = torch.tensor(
            teacher_lps, dtype=logprobs.dtype, device=logprobs.device
        )
        if student_row.dim() == 0:
            chosen_index = min(int(getattr(pr, "chosen_index", 0)), teacher.numel() - 1)
            terms.append(student_row - teacher[chosen_index])
            continue

        n_cand = min(student_row.shape[-1], teacher.numel())
        if n_cand > 0:
            terms.append((student_row[:n_cand] - teacher[:n_cand]).reshape(-1))

    if not terms:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)
    return torch.cat([term.reshape(-1) for term in terms]).mean()


def _compute_subset_kl(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    distill_kl_mode: str,
) -> torch.Tensor:
    """Compute KL on the candidate subset for each position.

    The available support is the candidate subset, so both teacher and student
    are normalized over that subset before computing KL.
    """
    student_logprobs = torch.log_softmax(student_logprobs, dim=-1)
    teacher_logprobs = torch.log_softmax(teacher_logprobs, dim=-1)

    if distill_kl_mode == "forward_kl":
        teacher_probs = teacher_logprobs.exp()
        return (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    if distill_kl_mode == "reverse_kl":
        student_probs = student_logprobs.exp()
        return (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1)
    raise ValueError(
        f"distill_kl_mode must be 'forward_kl' or 'reverse_kl', got {distill_kl_mode!r}"
    )

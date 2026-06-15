"""Tests for on-policy distillation pipeline bug fixes."""

import inspect

import pytest


def test_bug1_completion_id_uses_interaction_id():
    """Bug 1: OnPolicyDistillAgent should use the interaction's actual ID
    from the proxy server, not an MD5 hash of completion_messages."""
    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.core.agent",
            fromlist=["OnPolicyDistillAgent"],
        ).OnPolicyDistillAgent.run
    )
    assert "hashlib.md5" not in source, (
        "OnPolicyDistillAgent.run() should not use hashlib.md5 for "
        "completion_id. Use interaction.interaction_id from the proxy server."
    )
    assert "interaction_id" in source, (
        "OnPolicyDistillAgent.run() should use interaction.interaction_id "
        "from the proxy server as the completion_id."
    )


def test_bug6_distribute_position_rewards_warns_on_unmapped():
    """Bug 6: _distribute_position_rewards should warn when a
    position_reward's sample_index doesn't map to any minibatch."""
    from unittest.mock import patch

    import torch

    from customized_areal.tree_search.distilling.distill_types import PositionRewardInfo
    from customized_areal.tree_search.training.actor import (
        _distribute_position_rewards,
    )

    mb = {
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    mb_inputs = type("MB", (), {"mbs": [mb], "forward_indices": [0, 1]})()

    bad_pr = PositionRewardInfo(
        position=0,
        candidates=["a", "b"],
        candidate_token_ids=[1, 2],
        rewards=[0.5, -0.3],
        chosen_index=0,
        sample_index=99,
    )
    good_pr = PositionRewardInfo(
        position=1,
        candidates=["c", "d"],
        candidate_token_ids=[3, 4],
        rewards=[0.2, -0.1],
        chosen_index=0,
        sample_index=0,
    )

    with patch("customized_areal.tree_search.training.actor.logger") as mock_logger:
        _distribute_position_rewards(mb_inputs, [bad_pr, good_pr])
        warning_calls = [c for c in mock_logger.method_calls if "warning" in str(c)]
        assert len(warning_calls) > 0, (
            "_distribute_position_rewards should log a warning when "
            "a position_reward's sample_index doesn't map to any minibatch"
        )


def test_bug10_prompt_lens_vectorized():
    """Bug 10: prompt_lens computation in grpo_distill_loss_fn should use
    vectorized PyTorch ops instead of O(batch*seq_len) Python loops."""
    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.training.loss",
            fromlist=["grpo_distill_loss_fn"],
        ).grpo_distill_loss_fn
    )
    lines = source.split("\n")
    in_prompt_loop = False
    for line in lines:
        if "prompt_lens = []" in line or "prompt_lens.append" in line:
            in_prompt_loop = True
        if in_prompt_loop and "for b in range(loss_mask.shape" in line:
            pytest.fail(
                "prompt_lens computation uses O(batch*seq_len) Python loop. "
                "Use vectorized: prompt_lens = (loss_mask.bool().cumsum(dim=1)==1)"
                ".int().argmax(dim=1).tolist()"
            )


def test_bug11_no_item_in_distill_stat():
    """Bug 11: distill_stat.item() in grpo_distill_loss_fn forces GPU-CPU
    sync on every training step. Use the tensor directly."""
    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.training.loss",
            fromlist=["grpo_distill_loss_fn"],
        ).grpo_distill_loss_fn
    )
    for line in source.split("\n"):
        if "distill_stat" in line and ".item()" in line:
            pytest.fail(
                "distill_stat.item() forces GPU-CPU sync. Use tensor directly: "
                "torch.full(..., distill_stat, ...)"
            )


def test_distill_loss_uses_prox_logp_method_config():
    """Distill loss should follow PPOActorConfig.prox_logp_method."""
    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.training.loss",
            fromlist=["grpo_distill_loss_fn"],
        ).grpo_distill_loss_fn
    )
    assert "prox_logp_method" in source
    assert "prox_clip" not in source


def test_distill_actor_registers_n_seqs_before_task_reward_stat():
    """Custom PPO update must register n_seqs before logging task_reward."""
    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.training.actor",
            fromlist=["patch_ppo_actor_class_to_use_distill_loss"],
        ).patch_ppo_actor_class_to_use_distill_loss
    )
    denominator_idx = source.index("n_seqs=torch.ones_like")
    stat_idx = source.index("task_reward=reward_score.float()")
    assert denominator_idx < stat_idx


def test_bug8_no_model_inputs_mutation():
    """Bug 8: _compute_logprobs_and_loss should not mutate ctx.model_inputs
    by temporarily overriding rolled_input_ids. Pass labels separately."""
    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.engine.fsdp_engine",
            fromlist=["MultiCandidateFSDPEngine"],
        ).MultiCandidateFSDPEngine._compute_logprobs_and_loss
    )
    assert 'ctx.model_inputs["rolled_input_ids"]' not in source, (
        "_compute_logprobs_and_loss should not mutate ctx.model_inputs by "
        "overriding rolled_input_ids. Pass multi_candidate_labels as a "
        "separate parameter to _compute_logprobs_entropy instead."
    )


def test_bug9_position_clamping_warns():
    """Bug 9: Position clamping in _compute_position_level_grpo_loss should
    log a warning instead of silently corrupting gradient signal."""
    from unittest.mock import patch

    import torch

    from customized_areal.tree_search.distilling.distill_types import PositionRewardInfo
    from customized_areal.tree_search.training.loss import (
        _compute_position_level_grpo_loss,
    )

    seq_len = 5
    num_candidates = 2
    logprobs = torch.randn(seq_len, num_candidates, requires_grad=True)
    loss_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)

    position_rewards = [
        PositionRewardInfo(
            position=10,
            candidates=["a", "b"],
            candidate_token_ids=[1, 2],
            logprobs=[-1.0, -2.0],
            rewards=[0.5, -0.3],
            chosen_index=0,
            sample_index=0,
        ),
    ]

    with patch("customized_areal.tree_search.training.loss.logger") as mock_logger:
        _compute_position_level_grpo_loss(
            position_rewards=position_rewards,
            logprobs=logprobs,
            loss_mask=loss_mask,
            prompt_lens=[0],
        )
        warning_calls = [c for c in mock_logger.method_calls if "warning" in str(c)]
        assert len(warning_calls) > 0, (
            "_compute_position_level_grpo_loss should log a warning when "
            "position is clamped to valid range"
        )


def test_bug12_chunked_apply_has_shape_assertion():
    """Bug 12: _chunked_apply should assert that logits is 2D (seq_len first)
    since it splits along dim=0."""
    import inspect

    source = inspect.getsource(
        __import__(
            "customized_areal.tree_search.training.logprobs",
            fromlist=["_chunked_apply"],
        )._chunked_apply
    )
    assert "ndim" in source, (
        "_chunked_apply should assert logits.ndim == 2 since it splits "
        "along dim=0 assuming seq_len is the first dimension."
    )


def test_bug13_align_teacher_squeezes_trailing_singleton_candidate_dim():
    """Bug 13: _align_teacher_chosen_logprobs must squeeze trailing singleton
    candidate dims before checking dimension differences.

    When teacher_logp is [B, resp_len, 1] and target is 1D packed with
    cu_seqlens, the function must squeeze the trailing candidate dim first
    to reach the cu_seqlens alignment path. Without this, dim diff is 2
    (batch + candidates) instead of 1, and the function raises ValueError.
    """
    import torch

    from customized_areal.tree_search.training.losses.distill import (
        _align_teacher_chosen_logprobs,
    )

    # 3 sequences packed into 1D: prompt_len=2 each, response_len=3 each
    # target: [0, 0, p1, p1, r1, r1, r1, p2, p2, r2, r2, r2, p3, p3, r3, r3, r3]
    target = torch.zeros(15)
    loss_mask = torch.tensor(
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=torch.int32
    )
    cu_seqlens = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    prompt_lens = [2, 2, 2]

    # teacher_logp: [3, 3, 1] — 3 sequences, 3 response tokens each, 1 candidate
    teacher_logp = torch.tensor(
        [[[-1.0], [-2.0], [-3.0]], [[-4.0], [-5.0], [-6.0]], [[-7.0], [-8.0], [-9.0]]]
    )

    aligned, valid = _align_teacher_chosen_logprobs(
        teacher_logprobs=teacher_logp,
        target=target,
        loss_mask=loss_mask,
        prompt_lens=prompt_lens,
        input_data={"cu_seqlens": cu_seqlens},
    )

    assert aligned.shape == target.shape, (
        f"aligned shape {aligned.shape} should match target {target.shape}"
    )
    # Response positions (indices 2-4, 7-9, 12-14) should have teacher values
    assert aligned[2].item() == -1.0
    assert aligned[3].item() == -2.0
    assert aligned[4].item() == -3.0
    assert aligned[7].item() == -4.0
    assert aligned[8].item() == -5.0
    assert aligned[9].item() == -6.0


def test_align_teacher_packed_selects_chosen_candidate():
    """1D-packed target with a multi-candidate teacher [B, resp_len, C>1]
    must select the chosen token (candidate index 0) and ignore the rest."""
    import torch

    from customized_areal.tree_search.training.losses.distill import (
        _align_teacher_chosen_logprobs,
    )

    target = torch.zeros(15)
    loss_mask = torch.tensor(
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=torch.int32
    )
    cu_seqlens = torch.tensor([0, 5, 10, 15], dtype=torch.int32)

    # [3, 3, 3]: chosen logprobs in column 0; non-chosen columns are bogus.
    teacher = torch.full((3, 3, 3), 99.0)
    teacher[..., 0] = torch.tensor(
        [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]]
    )

    aligned, valid = _align_teacher_chosen_logprobs(
        teacher_logprobs=teacher,
        target=target,
        loss_mask=loss_mask,
        prompt_lens=[2, 2, 2],
        input_data={"cu_seqlens": cu_seqlens},
    )

    assert aligned[2:5].tolist() == [-1.0, -2.0, -3.0]
    assert aligned[12:15].tolist() == [-7.0, -8.0, -9.0]
    # Non-chosen columns (99.0) must never leak into the aligned output.
    assert 99.0 not in aligned.tolist()
    assert valid[2:5].all() and not valid[:2].any()


def test_align_teacher_2d_batched_per_row_response():
    """2D-batched target [batch, seq_len] aligns each row's response slice."""
    import torch

    from customized_areal.tree_search.training.losses.distill import (
        _align_teacher_chosen_logprobs,
    )

    target = torch.zeros(2, 6)
    # row0: prompt_len=2, response at 2:5; row1: prompt_len=1, response at 1:3
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.int32
    )
    teacher = torch.tensor([[[-1.0], [-2.0], [-3.0]], [[-4.0], [-5.0], [0.0]]])

    aligned, valid = _align_teacher_chosen_logprobs(
        teacher_logprobs=teacher,
        target=target,
        loss_mask=loss_mask,
        prompt_lens=[2, 1],
        input_data=None,
    )

    assert aligned[0, 2:5].tolist() == [-1.0, -2.0, -3.0]
    assert aligned[1, 1:3].tolist() == [-4.0, -5.0]
    assert valid[0, 2:5].all() and valid[1, 1:3].all()
    # Prompt positions stay zero / invalid.
    assert not valid[0, :2].any() and not valid[1, 0].any()


def test_align_teacher_already_aligned_passthrough():
    """When teacher already matches target shape it is returned as-is, with
    validity gated by loss_mask and the non-zero check."""
    import torch

    from customized_areal.tree_search.training.losses.distill import (
        _align_teacher_chosen_logprobs,
    )

    target = torch.zeros(4)
    loss_mask = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
    teacher = torch.tensor([0.0, -1.0, -2.0, 0.0])

    aligned, valid = _align_teacher_chosen_logprobs(
        teacher_logprobs=teacher,
        target=target,
        loss_mask=loss_mask,
        prompt_lens=[1],
        input_data=None,
    )

    assert torch.equal(aligned, teacher)
    assert valid.tolist() == [False, True, True, False]


def test_align_teacher_zero_padding_marked_invalid():
    """Zero-valued (padding/missing) teacher entries inside the response
    region must be marked invalid even though they are written to aligned."""
    import torch

    from customized_areal.tree_search.training.losses.distill import (
        _align_teacher_chosen_logprobs,
    )

    target = torch.zeros(5)
    loss_mask = torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    # 4 response tokens, but the 3rd teacher value is a 0.0 "missing" entry.
    teacher = torch.tensor([[[-1.0], [-2.0], [0.0], [-4.0]]])

    aligned, valid = _align_teacher_chosen_logprobs(
        teacher_logprobs=teacher,
        target=target,
        loss_mask=loss_mask,
        prompt_lens=[1],
        input_data={"cu_seqlens": cu_seqlens},
    )

    assert aligned[1:5].tolist() == [-1.0, -2.0, 0.0, -4.0]
    # Position 3 (the 0.0 entry) is invalid; the others are valid.
    assert valid[1].item() and valid[2].item() and valid[4].item()
    assert not valid[3].item()


def test_align_teacher_unsupported_shape_raises():
    """An unsupported teacher/target layout (after candidate reduction) must
    raise ValueError rather than silently producing a wrong alignment."""
    import torch

    from customized_areal.tree_search.training.losses.distill import (
        _align_teacher_chosen_logprobs,
    )

    target = torch.zeros(15)
    loss_mask = torch.ones(15, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    # 4D teacher is not a supported layout.
    teacher = torch.zeros(3, 3, 3, 3)

    with pytest.raises(ValueError, match="Unsupported teacher_logp shape"):
        _align_teacher_chosen_logprobs(
            teacher_logprobs=teacher,
            target=target,
            loss_mask=loss_mask,
            prompt_lens=[2, 2, 2],
            input_data={"cu_seqlens": cu_seqlens},
        )

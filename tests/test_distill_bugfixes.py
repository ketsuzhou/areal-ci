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

    from customized_areal.tree_search.distill_types import PositionRewardInfo
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

    from customized_areal.tree_search.distill_types import PositionRewardInfo
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

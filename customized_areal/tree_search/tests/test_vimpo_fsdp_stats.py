"""Single-rank numerical tests for VIMPO actor candidate statistics.

These tests verify the module-level ``vimpo_candidate_stats_from_logits`` helper
against brute-force log-softmax / top-k computations on CPU tensors. No
distributed process groups are involved - the distributed (TP/SP) collection
path is covered by ``test_vimpo_fsdp_distributed.py``.
"""

import torch

from customized_areal.tree_search.engine.fsdp_engine import (
    VIMPOCandidateStats,
    vimpo_candidate_stats_from_logits,
)


def test_stats_use_full_vocab_normalizer_and_actor_topk() -> None:
    logits = torch.tensor([[[0.0, 3.0, 2.0, 1.0], [2.0, 0.0, 1.0, 3.0]]])
    labels = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True]])
    stats = vimpo_candidate_stats_from_logits(logits, labels, mask, top_k=2)
    expected = logits.log_softmax(-1)
    assert stats.candidate_ids.tolist() == [[[1, 2], [3, 0]]]
    torch.testing.assert_close(
        stats.candidate_logp,
        expected.gather(-1, stats.candidate_ids),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        stats.sampled_logp,
        expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        stats.retained_mass,
        stats.candidate_logp.exp().sum(-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_topk_is_capped_at_vocab_and_masked_rows_are_sentinel() -> None:
    stats = vimpo_candidate_stats_from_logits(
        torch.zeros(1, 2, 3),
        torch.tensor([[0, 1]]),
        torch.tensor([[True, False]]),
        top_k=8,
    )
    assert stats.candidate_ids.shape == (1, 2, 3)
    assert stats.candidate_ids[0, 1].tolist() == [-1, -1, -1]
    assert stats.predict_mask.tolist() == [[True, False]]


def test_stats_return_type_is_frozen_dataclass() -> None:
    """The brief pins VIMPOCandidateStats as a frozen dataclass; verify the
    public contract so downstream tasks (T7) can rely on field ordering."""
    logits = torch.zeros(1, 1, 2)
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    stats = vimpo_candidate_stats_from_logits(logits, labels, mask, top_k=1)
    assert isinstance(stats, VIMPOCandidateStats)
    # Frozen: attribute assignment must raise.
    try:
        stats.sampled_logp = stats.sampled_logp  # noqa: PLR6104
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("VIMPOCandidateStats must be frozen")
    # Field ordering matches the brief (T7 depends on this).
    fields = [f.name for f in stats.__dataclass_fields__.values()]
    assert fields == [
        "sampled_logp",
        "candidate_ids",
        "candidate_logp",
        "retained_mass",
        "predict_mask",
    ]

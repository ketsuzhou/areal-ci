"""Single-rank numerical tests for VIMPO actor candidate statistics.

These tests verify the module-level ``vimpo_candidate_stats_from_logits`` helper
against brute-force log-softmax / top-k computations on CPU tensors. No
distributed process groups are involved - the distributed (TP/SP) collection
path is covered by ``test_vimpo_fsdp_distributed.py``.

The engine-level aggregation methods (``_reorder_vimpo_stats``,
``_merge_tree_vimpo_stats``, ``_vimpo_stats_from_tree_logits``) are also
exercised here on CPU via lightweight ``MultiCandidateFSDPEngine`` instances
built with ``__new__`` (no model, no FSDP, no GPU).
"""

from __future__ import annotations

import pytest
import torch

from customized_areal.tree_search.engine.fsdp_engine import (
    MultiCandidateFSDPEngine,
    VIMPOCandidateStats,
    vimpo_candidate_stats_from_logits,
)

from areal.engine.fsdp_engine import FSDPTrainContext
from areal.models.tree_attn.tree import TrieNode


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


# =============================================================================
# Engine-level aggregation method tests (CPU, no model, no FSDP)
# =============================================================================


def _make_lightweight_engine() -> MultiCandidateFSDPEngine:
    """Build a ``MultiCandidateFSDPEngine`` via ``__new__`` with only the
    attributes the aggregation methods read (``parallel_helper``). No model,
    no FSDP, no GPU - suitable for CPU unit tests of pure-tensor aggregation.
    """
    engine = MultiCandidateFSDPEngine.__new__(MultiCandidateFSDPEngine)

    class _FakeParallelHelper:
        sp_size = 1
        tp_size = 1
        tp_group = None

    engine.parallel_helper = _FakeParallelHelper()
    return engine


def _make_stats(
    sampled_logp: list[float],
    candidate_ids: list[list[int]],
    candidate_logp: list[list[float]],
    predict_mask: list[bool],
) -> VIMPOCandidateStats:
    """Build a per-micro-batch ``VIMPOCandidateStats`` (1D / 2D, no batch dim)."""
    t_sampled = torch.tensor(sampled_logp, dtype=torch.float)
    t_ids = torch.tensor(candidate_ids, dtype=torch.long)
    t_lp = torch.tensor(candidate_logp, dtype=torch.float)
    t_mask = torch.tensor(predict_mask, dtype=torch.bool)
    retained = t_lp.exp().sum(-1).masked_fill(~t_mask, 0.0)
    return VIMPOCandidateStats(
        sampled_logp=t_sampled,
        candidate_ids=t_ids,
        candidate_logp=t_lp,
        retained_mass=retained,
        predict_mask=t_mask,
    )


class TestReorderVimpoStats:
    """``_reorder_vimpo_stats`` aggregates non-tree per-micro-batch stats to
    ``[B, S, ...]`` and must reapply the ``-1`` sentinel at cross-sequence pad
    positions (the implementer's fix)."""

    def test_reapplies_sentinel_at_cross_sequence_pad(self) -> None:
        """Two sequences of different lengths: the shorter one's pad positions
        (predict_mask=False) must have ``candidate_ids = -1``, not ``0``."""
        engine = _make_lightweight_engine()
        # Two micro-batches: seq 0 has length 3, seq 1 has length 2.
        # (For a single-micro-batch case, mb_list is only used for
        # forward_indices/backward_indices; we build a minimal stub.)
        mb_stats = [
            _make_stats(
                sampled_logp=[-1.0, -2.0, -3.0],
                candidate_ids=[[10, 11], [20, 21], [30, 31]],
                candidate_logp=[[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
                predict_mask=[True, True, False],
            ),
            _make_stats(
                sampled_logp=[-4.0, -5.0],
                candidate_ids=[[40, 41], [50, 51]],
                candidate_logp=[[-0.7, -0.8], [-0.9, -1.0]],
                predict_mask=[True, False],
            ),
        ]
        output_seqlens = [3, 2]

        # Minimal mb_list stub: reorder_and_pad_outputs reads
        # mb_list.forward_indices and mb_list.backward_indices.
        class _FakeMbList:
            forward_indices = [0, 1]
            backward_indices = [0, 1]

        result = engine._reorder_vimpo_stats(mb_stats, output_seqlens, _FakeMbList())

        assert result.candidate_ids.shape == (2, 3, 2)
        # Seq 0 (length 3): position 2 has predict_mask=False -> sentinel -1.
        assert result.candidate_ids[0, 2].tolist() == [-1, -1]
        assert result.predict_mask[0].tolist() == [True, True, False]
        # Seq 1 (length 2, padded to 3): position 1 has predict_mask=False,
        # position 2 is cross-sequence pad (predict_mask=False via 0-pad).
        # Both must have sentinel -1 (not token id 0).
        assert result.candidate_ids[1, 1].tolist() == [-1, -1]
        assert result.candidate_ids[1, 2].tolist() == [-1, -1], (
            "cross-sequence pad positions must be -1, not 0"
        )
        assert result.predict_mask[1].tolist() == [True, False, False]
        # sampled_logp / retained_mass at pad positions must be 0.
        assert result.sampled_logp[1, 2].item() == 0.0
        assert result.retained_mass[1, 2].item() == 0.0


class TestMergeTreeVimpoStats:
    """``_merge_tree_vimpo_stats`` aggregates per-sequence dict stats to
    ``[B, max_seq_len, K]`` with ``-1`` sentinel padding for ``candidate_ids``."""

    def test_merges_per_sequence_stats_with_sentinel_pad(self) -> None:
        engine = _make_lightweight_engine()
        # Two sequences: seq 0 has length 3, seq 1 has length 2.
        # Per-sequence stats already have the -1 sentinel applied at
        # predict_mask=False positions (as _vimpo_stats_from_tree_logits does);
        # _merge_tree_vimpo_stats only needs to pad cross-sequence positions.
        mb_stats: list[dict[int, VIMPOCandidateStats]] = [
            {
                0: _make_stats(
                    sampled_logp=[-1.0, -2.0, 0.0],
                    candidate_ids=[[10, 11], [20, 21], [-1, -1]],
                    candidate_logp=[[-0.1, -0.2], [-0.3, -0.4], [0.0, 0.0]],
                    predict_mask=[True, True, False],
                ),
                1: _make_stats(
                    sampled_logp=[-4.0, -5.0],
                    candidate_ids=[[40, 41], [50, 51]],
                    candidate_logp=[[-0.7, -0.8], [-0.9, -1.0]],
                    predict_mask=[True, True],
                ),
            }
        ]
        result = engine._merge_tree_vimpo_stats(mb_stats, batch_size=2, top_k=2)
        assert result.candidate_ids.shape == (2, 3, 2)
        # Seq 0: full length 3.
        assert result.candidate_ids[0, 0].tolist() == [10, 11]
        assert result.candidate_ids[0, 2].tolist() == [-1, -1]  # predict_mask=False
        # Seq 1: length 2, padded to 3 with -1 sentinel.
        assert result.candidate_ids[1, 0].tolist() == [40, 41]
        assert result.candidate_ids[1, 2].tolist() == [-1, -1], (
            "pad positions must be -1"
        )
        assert result.predict_mask[1].tolist() == [True, True, False]

    def test_raises_on_duplicate_seq_id_across_microbatches(self) -> None:
        engine = _make_lightweight_engine()
        stats0 = _make_stats(
            sampled_logp=[-1.0],
            candidate_ids=[[10, 11]],
            candidate_logp=[[-0.1, -0.2]],
            predict_mask=[True],
        )
        stats1 = _make_stats(
            sampled_logp=[-2.0],
            candidate_ids=[[20, 21]],
            candidate_logp=[[-0.3, -0.4]],
            predict_mask=[True],
        )
        mb_stats: list[dict[int, VIMPOCandidateStats]] = [{0: stats0}, {0: stats1}]
        with pytest.raises(ValueError, match="Duplicate sequence_id"):
            engine._merge_tree_vimpo_stats(mb_stats, batch_size=2, top_k=2)

    def test_raises_on_empty_input(self) -> None:
        engine = _make_lightweight_engine()
        with pytest.raises(ValueError, match="No VIMPO stats"):
            engine._merge_tree_vimpo_stats([], batch_size=2, top_k=2)

    def test_raises_on_all_empty_dicts(self) -> None:
        engine = _make_lightweight_engine()
        with pytest.raises(ValueError, match="No VIMPO stats"):
            engine._merge_tree_vimpo_stats([{}], batch_size=1, top_k=2)


class TestVimpoStatsFromTreeLogits:
    """``_vimpo_stats_from_tree_logits`` unpacks per-sequence stats via trie
    mappings. This test also verifies Finding 4's fix: the per-position
    ``predict_mask`` is response-aligned (prompt positions excluded, terminal
    position not predicted)."""

    def test_single_sequence_prompt_response_predict_mask(self) -> None:
        """Single-sequence trie with prompt+response packed. Assert
        predict_mask is False on prompt-internal positions and True on
        response positions (except the terminal which is not predicted)."""
        engine = _make_lightweight_engine()

        # Sequence: [1, 2, 3, 4, 5] (length 5). Tree input_ids packed in order.
        # loss_mask: [0, 0, 1, 1, 0] (prompt = [1,2], response = [3,4], last
        # position is terminal/padding).
        tree_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        loss_mask = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
        # Vocab size 8.
        torch.manual_seed(123)
        logits = torch.randn(5, 8, dtype=torch.float)

        # Single-node trie covering positions [0, 4] inclusive.
        trie = TrieNode(tree_id=0)
        trie.nodes = [
            TrieNode(
                tree_id=0,
                start_idx=0,
                end_idx=4,
                tokens=[1, 2, 3, 4, 5],
                sequence_ids=[0],
            )
        ]

        ctx = FSDPTrainContext(
            model_inputs={"input_ids": tree_input_ids},
            mb_input={
                "input_ids": tree_input_ids,
                "loss_mask": loss_mask,
                "cu_seqlens": cu_seqlens,
            },
            trie_node=trie,
        )

        results = engine._vimpo_stats_from_tree_logits(
            logits, ctx, top_k=3, tp_group=None
        )

        assert set(results.keys()) == {0}
        stats = results[0]
        # Per-sequence output length = seq_len - 1 = 4.
        assert stats.sampled_logp.shape == (4,)
        assert stats.candidate_ids.shape == (4, 3)
        assert stats.predict_mask.shape == (4,)

        # predict_mask = loss_mask[1:5] = [0, 1, 1, 0] (response-aligned).
        assert stats.predict_mask.tolist() == [False, True, True, False], (
            "predict_mask must be False on prompt-internal (pos 0 predicts pos 1 "
            "= prompt) and on the position predicting the terminal (pos 3 "
            "predicts pos 4 = loss_mask 0), True where predicting a response"
        )

        # Brute-force reference: full-vocab log_softmax at positions [0..3],
        # labels = input_ids[1..4].
        log_probs = logits.log_softmax(-1)
        ref_candidate_logp, ref_candidate_ids = torch.topk(log_probs[:4], 3, dim=-1)
        ref_sampled = (
            log_probs[:4]
            .gather(-1, tree_input_ids.squeeze(0)[1:5].unsqueeze(-1))
            .squeeze(-1)
        )
        # Apply predict_mask.
        ref_candidate_ids = ref_candidate_ids.masked_fill(
            ~stats.predict_mask.unsqueeze(-1), -1
        )
        ref_candidate_logp = ref_candidate_logp.masked_fill(
            ~stats.predict_mask.unsqueeze(-1), 0.0
        )
        ref_sampled = ref_sampled.masked_fill(~stats.predict_mask, 0.0)

        torch.testing.assert_close(
            stats.candidate_ids, ref_candidate_ids, rtol=0, atol=0
        )
        torch.testing.assert_close(
            stats.candidate_logp, ref_candidate_logp, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            stats.sampled_logp, ref_sampled, rtol=1e-6, atol=1e-6
        )
        # Masked positions have sampled_logp = 0.
        assert stats.sampled_logp[0].item() == 0.0
        assert stats.sampled_logp[3].item() == 0.0

    def test_multi_range_trie_transition_predict_mask(self) -> None:
        """Two-range trie (one transition). Verify predict_mask at the
        transition position follows loss_mask at the predicted position."""
        engine = _make_lightweight_engine()

        # Sequence: [1, 2, 3, 4, 5, 6] (length 6). loss_mask has prompt+response.
        tree_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        # prompt = [1,2], response = [3,4,5], last = terminal (loss_mask=0).
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
        torch.manual_seed(456)
        logits = torch.randn(6, 8, dtype=torch.float)

        # Two-node trie: node 0 covers [0,2] inclusive, node 1 covers [3,5]
        # inclusive. Both contain seq_id=0.
        trie = TrieNode(tree_id=0)
        trie.nodes = [
            TrieNode(
                tree_id=0,
                start_idx=0,
                end_idx=2,
                tokens=[1, 2, 3],
                sequence_ids=[0],
            ),
            TrieNode(
                tree_id=0,
                start_idx=3,
                end_idx=5,
                tokens=[4, 5, 6],
                sequence_ids=[0],
            ),
        ]

        ctx = FSDPTrainContext(
            model_inputs={"input_ids": tree_input_ids},
            mb_input={
                "input_ids": tree_input_ids,
                "loss_mask": loss_mask,
                "cu_seqlens": cu_seqlens,
            },
            trie_node=trie,
        )

        results = engine._vimpo_stats_from_tree_logits(
            logits, ctx, top_k=2, tp_group=None
        )

        stats = results[0]
        # Per-sequence output: range0 internal (2) + transition (1) +
        # range1 internal (2) = 5 = seq_len - 1.
        assert stats.predict_mask.shape == (5,)
        # predict_mask = loss_mask[1:6] = [0, 1, 1, 1, 0].
        assert stats.predict_mask.tolist() == [False, True, True, True, False], (
            "transition position (output idx 2, tree pos 2 predicting "
            "next_start=3 which is response) must be True; terminal-adjacent "
            "(output idx 4 predicting pos 5 = loss_mask 0) must be False"
        )

    def test_vimpo_predict_mask_overrides_loss_mask(self) -> None:
        """When ``vimpo_predict_mask`` is in mb_input, it is used directly
        instead of deriving from loss_mask."""
        engine = _make_lightweight_engine()

        tree_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        # vimpo_predict_mask is already shifted: pm[p] = loss_mask[p+1].
        # Here we set it to [F, T, T, F] meaning position 0 predicts a
        # non-response, positions 1,2 predict responses, position 3 is terminal.
        vimpo_pm = torch.tensor([[False, True, True, False]], dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
        torch.manual_seed(789)
        logits = torch.randn(4, 6, dtype=torch.float)

        trie = TrieNode(tree_id=0)
        trie.nodes = [
            TrieNode(
                tree_id=0,
                start_idx=0,
                end_idx=3,
                tokens=[1, 2, 3, 4],
                sequence_ids=[0],
            )
        ]

        ctx = FSDPTrainContext(
            model_inputs={"input_ids": tree_input_ids},
            mb_input={
                "input_ids": tree_input_ids,
                "vimpo_predict_mask": vimpo_pm,
                "cu_seqlens": cu_seqlens,
            },
            trie_node=trie,
        )

        results = engine._vimpo_stats_from_tree_logits(
            logits, ctx, top_k=2, tp_group=None
        )
        # Per-sequence predict_mask = vimpo_pm[0:3] = [F, T, T].
        assert results[0].predict_mask.tolist() == [False, True, True]

    def test_predict_mask_without_cu_seqlens_uses_trie_derived_offsets(self) -> None:
        """When ``cu_seqlens`` is not in mb_input (the common case -
        ``build_packed_tree_batch`` does not add it), the engine derives
        per-sequence offsets from the trie's token counts and still produces
        a response-aligned predict_mask from ``loss_mask``."""
        engine = _make_lightweight_engine()

        tree_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        # loss_mask packed per-sequence (same layout as _pack_extra_data).
        loss_mask = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.int32)
        # NO cu_seqlens in mb_input - the engine must derive it from the trie.
        torch.manual_seed(202)
        logits = torch.randn(5, 8, dtype=torch.float)

        trie = TrieNode(tree_id=0)
        trie.nodes = [
            TrieNode(
                tree_id=0,
                start_idx=0,
                end_idx=4,
                tokens=[1, 2, 3, 4, 5],
                sequence_ids=[0],
            )
        ]

        ctx = FSDPTrainContext(
            model_inputs={"input_ids": tree_input_ids},
            mb_input={
                "input_ids": tree_input_ids,
                "loss_mask": loss_mask,
                # Note: no "cu_seqlens" key.
            },
            trie_node=trie,
        )

        results = engine._vimpo_stats_from_tree_logits(
            logits, ctx, top_k=2, tp_group=None
        )
        # Same as test_single_sequence_prompt_response_predict_mask:
        # predict_mask = loss_mask[1:5] = [0, 1, 1, 0].
        assert results[0].predict_mask.tolist() == [False, True, True, False], (
            "predict_mask must be response-aligned even without cu_seqlens in mb_input"
        )

    def test_falls_back_to_all_true_when_no_masks(self) -> None:
        """When neither vimpo_predict_mask nor loss_mask is in mb_input,
        predict_mask falls back to all-True (matching _extract_vimpo_predict_mask)."""
        engine = _make_lightweight_engine()

        tree_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        torch.manual_seed(101)
        logits = torch.randn(3, 6, dtype=torch.float)

        trie = TrieNode(tree_id=0)
        trie.nodes = [
            TrieNode(
                tree_id=0,
                start_idx=0,
                end_idx=2,
                tokens=[1, 2, 3],
                sequence_ids=[0],
            )
        ]

        ctx = FSDPTrainContext(
            model_inputs={"input_ids": tree_input_ids},
            mb_input={"input_ids": tree_input_ids},
            trie_node=trie,
        )

        results = engine._vimpo_stats_from_tree_logits(
            logits, ctx, top_k=2, tp_group=None
        )
        # Per-sequence output length = 2, all-True fallback.
        assert results[0].predict_mask.tolist() == [True, True]

    def test_empty_trie_returns_empty_dict(self) -> None:
        engine = _make_lightweight_engine()
        ctx = FSDPTrainContext(
            model_inputs={"input_ids": torch.zeros(1, 1, dtype=torch.long)},
            mb_input={"input_ids": torch.zeros(1, 1, dtype=torch.long)},
            trie_node=None,
        )
        results = engine._vimpo_stats_from_tree_logits(
            torch.zeros(1, 4), ctx, top_k=2, tp_group=None
        )
        assert results == {}

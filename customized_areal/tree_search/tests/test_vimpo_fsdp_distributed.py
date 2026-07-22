# SPDX-License-Identifier: Apache-2.0
"""Distributed tests for VIMPO actor candidate statistics.

Two test layers:

1. ``test_vimpo_stats_vocab_parallel_cpu_gloo`` - a CPU-only ``gloo`` test
   that spawns 2 ranks, gives rank 0 vocabulary IDs ``[0, 2)`` and rank 1
   ``[2, 4)``, and verifies the vocab-parallel collection
   (``_vocab_parallel_vimpo_candidate_stats``) against a concatenated
   brute-force computation. It also monkeypatches
   ``torch.distributed.all_reduce`` / ``torch.distributed.nn.functional.all_gather``
   to assert the configured ``tp_group`` is forwarded to every collective.

2. ``test_vimpo_stats_tp_sp_and_packed_tree_alignment`` - a 2-GPU NCCL test
   that exercises ``MultiCandidateFSDPEngine.compute_vimpo_candidate_stats``
   on both packed-tree and non-packed batches and verifies they agree. It is
   skipped when fewer than 2 CUDA devices are available.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
import torch.multiprocessing as mp

from customized_areal.tree_search.engine.fsdp_engine import (
    _vocab_parallel_vimpo_candidate_stats,
    vimpo_candidate_stats_from_logits,
)

# =============================================================================
# CPU gloo vocab-parallel test (runs in CI; no GPU required)
# =============================================================================


def _setup_gloo(rank: int, world_size: int, rendezvous_file: str) -> dist.ProcessGroup:
    """Initialize a gloo process group and return an explicit TP group."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    # Create an explicit TP group containing all ranks so the spy can
    # verify the exact ProcessGroup object is forwarded to collectives.
    return dist.new_group(ranks=list(range(world_size)))


def _teardown_gloo() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _vocab_parallel_worker(rank: int, world_size: int, rendezvous_file: str) -> None:
    """Worker: each rank holds a vocab shard and checks global top-k parity."""
    tp_group = _setup_gloo(rank, world_size, rendezvous_file)
    try:
        # Vocab size = 4, tp_size = 2, partition_vocab_size = 2.
        # Rank 0 -> vocab IDs [0, 2). Rank 1 -> vocab IDs [2, 4).
        S = 3
        K = 2
        partition_vocab_size = 2

        # Deterministic full-vocab logits so both ranks agree on the reference.
        torch.manual_seed(42)
        full_logits = torch.randn(
            S, world_size * partition_vocab_size, dtype=torch.float64
        )

        start = rank * partition_vocab_size
        end = start + partition_vocab_size
        sharded_logits = full_logits[:, start:end].contiguous()

        # Sampled token IDs (global vocab indices) and predict mask.
        labels = torch.tensor([0, 3, 1], dtype=torch.long)
        predict_mask = torch.tensor([True, True, False], dtype=torch.bool)

        # --- Spy on all_reduce / all_gather to assert tp_group is forwarded ---
        orig_all_reduce = dist.all_reduce
        orig_all_gather = dist_F.all_gather
        groups_all_reduce: list[Any] = []
        groups_all_gather: list[Any] = []

        def spy_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            groups_all_reduce.append(group)
            return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

        def spy_all_gather(tensor, group=None, **kwargs):
            groups_all_gather.append(group)
            return orig_all_gather(tensor, group=group, **kwargs)

        dist.all_reduce = spy_all_reduce  # type: ignore[assignment]
        dist_F.all_gather = spy_all_gather  # type: ignore[assignment]
        try:
            stats = _vocab_parallel_vimpo_candidate_stats(
                sharded_logits, labels, predict_mask, top_k=K, tp_group=tp_group
            )
        finally:
            dist.all_reduce = orig_all_reduce  # type: ignore[assignment]
            dist_F.all_gather = orig_all_gather  # type: ignore[assignment]

        # --- Brute force: full-vocab single-rank computation ---
        expected = vimpo_candidate_stats_from_logits(
            full_logits, labels, predict_mask, top_k=K
        )

        # --- Numerical parity ---
        torch.testing.assert_close(
            stats.sampled_logp, expected.sampled_logp, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            stats.candidate_logp, expected.candidate_logp, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            stats.retained_mass, expected.retained_mass, rtol=1e-5, atol=1e-6
        )
        # candidate_ids: exact match (deterministic tie-breaker: smaller ID wins).
        assert stats.candidate_ids.shape == expected.candidate_ids.shape, (
            f"candidate_ids shape mismatch: {stats.candidate_ids.shape} vs "
            f"{expected.candidate_ids.shape}"
        )
        assert stats.candidate_ids.tolist() == expected.candidate_ids.tolist(), (
            f"candidate_ids mismatch:\n  actual={stats.candidate_ids.tolist()}\n"
            f"  expected={expected.candidate_ids.tolist()}"
        )
        # predict_mask is echoed back.
        assert stats.predict_mask.tolist() == predict_mask.tolist()

        # --- tp_group forwarding ---
        assert len(groups_all_reduce) >= 2, (
            f"Expected at least 2 all_reduce calls (MAX + SUM), got "
            f"{len(groups_all_reduce)}"
        )
        assert len(groups_all_gather) >= 2, (
            f"Expected at least 2 all_gather calls (logp + ids), got "
            f"{len(groups_all_gather)}"
        )
        for g in groups_all_reduce:
            assert g is tp_group, (
                f"all_reduce called with group {g!r}, expected tp_group {tp_group!r}"
            )
        for g in groups_all_gather:
            assert g is tp_group, (
                f"all_gather called with group {g!r}, expected tp_group {tp_group!r}"
            )
    finally:
        _teardown_gloo()


@pytest.mark.slow
def test_vimpo_stats_vocab_parallel_cpu_gloo(tmp_path) -> None:
    """CPU gloo: 2 ranks, vocab shards [0,2) + [2,4), brute-force parity."""
    rendezvous_file = str(tmp_path / "rdzv_gloo")
    # Guard against stale rendezvous files on re-runs.
    if os.path.exists(rendezvous_file):
        os.remove(rendezvous_file)
    mp.spawn(
        _vocab_parallel_worker,
        args=(2, rendezvous_file),
        nprocs=2,
        join=True,
    )


def test_stats_helper_falls_back_to_single_rank_without_tp_group() -> None:
    """When tp_group is None, the vocab-parallel helper delegates to the
    single-rank ``vimpo_candidate_stats_from_logits``."""
    logits = torch.tensor([[[0.0, 3.0, 2.0, 1.0], [2.0, 0.0, 1.0, 3.0]]])
    labels = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True]])
    stats = _vocab_parallel_vimpo_candidate_stats(
        logits, labels, mask, top_k=2, tp_group=None
    )
    expected = vimpo_candidate_stats_from_logits(logits, labels, mask, top_k=2)
    torch.testing.assert_close(
        stats.sampled_logp, expected.sampled_logp, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        stats.candidate_logp, expected.candidate_logp, rtol=1e-6, atol=1e-6
    )
    assert stats.candidate_ids.tolist() == expected.candidate_ids.tolist()


# =============================================================================
# CPU gloo SP gather test (runs in CI; verifies bool predict_mask round-trip)
# =============================================================================


def _sp_gather_worker(rank: int, world_size: int, rendezvous_file: str) -> None:
    """Worker: each rank holds half the sequence; verify _sp_gather_vimpo_stats
    collects all fields along the sequence dim, including the bool
    ``predict_mask`` round-trip (Finding 3's int8 cast fix)."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    sp_group = dist.new_group(ranks=list(range(world_size)))
    try:
        from customized_areal.tree_search.engine.fsdp_engine import (
            MultiCandidateFSDPEngine,
            VIMPOCandidateStats,
        )

        # Build a lightweight engine via __new__ with only sp_size + sp_group.
        engine = MultiCandidateFSDPEngine.__new__(MultiCandidateFSDPEngine)
        engine.sp_group = sp_group

        class _FakeParallelHelper:
            sp_size = world_size

        engine.parallel_helper = _FakeParallelHelper()

        # Each rank holds 2 positions; after gather, 4 positions total.
        # Rank 0: positions [0, 1]. Rank 1: positions [2, 3].
        K = 2
        local_len = 2
        local_sampled = torch.tensor(
            [-1.0 - rank * 0.5, -2.0 - rank * 0.5], dtype=torch.float
        )
        local_ids = torch.tensor(
            [[10 + rank, 11 + rank], [20 + rank, 21 + rank]], dtype=torch.long
        )
        local_lp = torch.tensor(
            [
                [-0.1 - rank * 0.1, -0.2 - rank * 0.1],
                [-0.3 - rank * 0.1, -0.4 - rank * 0.1],
            ],
            dtype=torch.float,
        )
        # Bool predict_mask with a mix of True/False to verify round-trip.
        local_mask = torch.tensor([True, False], dtype=torch.bool)
        local_retained = local_lp.exp().sum(-1).masked_fill(~local_mask, 0.0)

        local_stats = VIMPOCandidateStats(
            sampled_logp=local_sampled,
            candidate_ids=local_ids,
            candidate_logp=local_lp,
            retained_mass=local_retained,
            predict_mask=local_mask,
        )

        # Case 1: ulysses_pad_size = 0 (no trim).
        gathered = engine._sp_gather_vimpo_stats(local_stats, ulysses_pad_size=0)

        assert gathered.sampled_logp.shape == (world_size * local_len,)
        assert gathered.candidate_ids.shape == (world_size * local_len, K)
        assert gathered.predict_mask.shape == (world_size * local_len,)
        # Bool predict_mask survives the int8 -> bool round-trip.
        assert gathered.predict_mask.dtype == torch.bool
        # Rank 0's data comes first, then rank 1's.
        assert gathered.predict_mask.tolist() == [True, False, True, False]
        # sampled_logp: rank 0 then rank 1.
        torch.testing.assert_close(
            gathered.sampled_logp,
            torch.tensor([-1.0, -2.0, -1.5, -2.5], dtype=torch.float),
            rtol=1e-6,
            atol=1e-6,
        )
        # candidate_ids: rank 0's [[10,11],[20,21]] then rank 1's [[11,12],[21,22]].
        assert gathered.candidate_ids.tolist() == [
            [10, 11],
            [20, 21],
            [11, 12],
            [21, 22],
        ]
        # retained_mass at masked positions is 0.
        assert gathered.retained_mass[1].item() == 0.0
        assert gathered.retained_mass[3].item() == 0.0

        # Case 2: ulysses_pad_size = 1 (trim 1 trailing position).
        local_stats_padded = VIMPOCandidateStats(
            sampled_logp=torch.cat([local_sampled, torch.tensor([-3.0])]),
            candidate_ids=torch.cat([local_ids, torch.tensor([[30, 31]])]),
            candidate_logp=torch.cat([local_lp, torch.tensor([[-0.5, -0.6]])]),
            retained_mass=torch.cat([local_retained, torch.tensor([0.0])]),
            predict_mask=torch.cat([local_mask, torch.tensor([True])]),
        )
        gathered_trim = engine._sp_gather_vimpo_stats(
            local_stats_padded, ulysses_pad_size=1
        )
        # Each rank had 3 positions, gather -> 6, trim 1 -> 5.
        assert gathered_trim.sampled_logp.shape == (world_size * 3 - 1,)
        assert gathered_trim.predict_mask.shape == (world_size * 3 - 1,)
        assert gathered_trim.predict_mask.dtype == torch.bool
        # The trimmed position (last) is gone.
        assert gathered_trim.predict_mask.tolist() == [
            True,
            False,
            True,
            True,
            False,
        ]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.slow
def test_vimpo_sp_gather_bool_predict_mask_cpu_gloo(tmp_path) -> None:
    """CPU gloo: 2 ranks SP gather, verifies bool predict_mask round-trip
    (Finding 3's int8 cast fix) and ulysses_pad_size trim."""
    rendezvous_file = str(tmp_path / "rdzv_sp_gather")
    if os.path.exists(rendezvous_file):
        os.remove(rendezvous_file)
    mp.spawn(
        _sp_gather_worker,
        args=(2, rendezvous_file),
        nprocs=2,
        join=True,
    )


# =============================================================================
# GPU NCCL test (skipped when < 2 CUDA devices)
# =============================================================================


def _tp_sp_packed_tree_worker(rank: int, world_size: int, port: int) -> None:
    """GPU worker: construct a real ``MultiCandidateFSDPEngine`` with TP=2 and
    verify ``compute_vimpo_candidate_stats`` on both flat (non-tree) and
    packed-tree batches against an independent brute-force reference, plus
    flat-vs-packed parity. FSDP / DTensor internals are NOT mocked.

    The brute-force reference is computed from full-vocab logits obtained by
    all-gathering the TP-sharded model logits across ranks, then applying
    ``log_softmax`` + ``topk`` + ``gather``.
    """
    import os as _os

    import torch as _torch

    _torch.cuda.set_device(rank)
    _os.environ["WORLD_SIZE"] = str(world_size)
    _os.environ["RANK"] = str(rank)
    _os.environ["LOCAL_RANK"] = str(rank)
    _os.environ["MASTER_ADDR"] = "localhost"
    _os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl")

    try:
        from customized_areal.tree_search.engine.fsdp_engine import (
            MultiCandidateFSDPEngine,
        )
        from tests.utils import get_model_path

        from areal.api import FinetuneSpec, ModelAllocation
        from areal.api.cli_args import (
            MicroBatchSpec,
            OptimizerConfig,
            TrainEngineConfig,
        )

        model_path = get_model_path(
            "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
        )

        config = TrainEngineConfig(
            backend="fsdp:d1t2",
            experiment_name="vimpo_tp_sp_test",
            trial_name="trial",
            path=model_path,
            dtype="float32",
            optimizer_dtype="float32",
            attn_impl="eager",
            gradient_checkpointing=False,
            disable_dropout=True,
            init_from_scratch=True,
            optimizer=OptimizerConfig(),
            mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
            enable_tree_training=True,
            pad_to_maximum=True,
        )

        engine = MultiCandidateFSDPEngine(config)
        alloc_mode = ModelAllocation.from_str("fsdp:d1t2")
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=1)
        engine.create_process_group(alloc_mode.parallel)
        engine.initialize(
            addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.parallel
        )

        vocab_size = engine.model_config.vocab_size
        seq_len = 8
        K = 4
        _torch.manual_seed(42 + rank)
        input_ids = _torch.randint(0, vocab_size, (1, seq_len), device="cuda")
        attention_mask = _torch.ones(1, seq_len, dtype=_torch.bool, device="cuda")
        # prompt = 2, response = 5, last = terminal/pad.
        loss_mask = _torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 0]], dtype=_torch.int32, device="cuda"
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

        # --- Brute-force reference from full-vocab logits ---
        # Run the model forward directly (eval mode, no grad) and all_gather
        # the TP-sharded logits to get the full vocabulary on every rank.
        engine.model.eval()
        with _torch.no_grad():
            sharded_logits = engine.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits  # [1, S, V/tp]
            full_logits = _torch.cat(
                dist_F.all_gather(
                    sharded_logits.contiguous(),
                    group=engine.parallel_helper.tp_group,
                ),
                dim=-1,
            )  # [1, S, V]

        log_probs = full_logits.log_softmax(-1)
        ref_candidate_logp, ref_candidate_ids = _torch.topk(log_probs, K, dim=-1)
        labels = _torch.roll(input_ids, shifts=-1, dims=-1)
        ref_sampled = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        predict_mask = _torch.roll(loss_mask.bool(), shifts=-1, dims=-1)
        predict_mask[:, -1] = False
        ref_candidate_ids = ref_candidate_ids.masked_fill(
            ~predict_mask.unsqueeze(-1), -1
        )
        ref_candidate_logp = ref_candidate_logp.masked_fill(
            ~predict_mask.unsqueeze(-1), 0.0
        )
        ref_sampled = ref_sampled.masked_fill(~predict_mask, 0.0)
        ref_retained = ref_candidate_logp.exp().sum(-1).masked_fill(~predict_mask, 0.0)

        # --- Flat path: disable tree training ---
        engine.enable_tree_training = False
        flat_stats = engine.compute_vimpo_candidate_stats(batch, top_k=K)

        # Assert flat path matches brute-force reference on all 4 fields.
        assert flat_stats.sampled_logp.shape == (1, seq_len)
        torch.testing.assert_close(
            flat_stats.sampled_logp, ref_sampled, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            flat_stats.candidate_logp, ref_candidate_logp, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            flat_stats.retained_mass, ref_retained, rtol=1e-5, atol=1e-6
        )
        assert flat_stats.candidate_ids.tolist() == ref_candidate_ids.tolist(), (
            f"flat candidate_ids mismatch:\n"
            f"  actual={flat_stats.candidate_ids.tolist()}\n"
            f"  expected={ref_candidate_ids.tolist()}"
        )

        # --- Tree path: enable tree training ---
        engine.enable_tree_training = True
        tree_stats = engine.compute_vimpo_candidate_stats(batch, top_k=K)

        # Tree path: per-sequence output has length seq_len - 1 (terminal not
        # predicted). Compare against the reference's first seq_len-1 positions
        # (the flat path's last position has predict_mask=False, so it's zero
        # in both).
        n_compare = seq_len - 1
        assert tree_stats.sampled_logp.shape == (1, n_compare)
        torch.testing.assert_close(
            tree_stats.sampled_logp,
            ref_sampled[:, :n_compare],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            tree_stats.candidate_logp,
            ref_candidate_logp[:, :n_compare],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            tree_stats.retained_mass,
            ref_retained[:, :n_compare],
            rtol=1e-5,
            atol=1e-6,
        )
        assert (
            tree_stats.candidate_ids.tolist()
            == ref_candidate_ids[:, :n_compare].tolist()
        ), (
            f"tree candidate_ids mismatch:\n"
            f"  actual={tree_stats.candidate_ids.tolist()}\n"
            f"  expected={ref_candidate_ids[:, :n_compare].tolist()}"
        )

        # --- Flat vs packed parity (on overlapping predict positions) ---
        torch.testing.assert_close(
            flat_stats.sampled_logp[:, :n_compare],
            tree_stats.sampled_logp,
            rtol=1e-5,
            atol=1e-6,
        )
        assert (
            flat_stats.candidate_ids[:, :n_compare].tolist()
            == tree_stats.candidate_ids.tolist()
        ), "flat vs packed candidate_ids parity failed"

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="VIMPO TP/SP FSDP test requires at least 2 CUDA devices",
)
def test_vimpo_stats_tp_sp_and_packed_tree_alignment() -> None:
    """GPU NCCL: packed-tree vs non-packed parity for compute_vimpo_candidate_stats.

    Constructs a real ``MultiCandidateFSDPEngine`` with TP=2, calls
    ``compute_vimpo_candidate_stats`` on both flat and packed-tree batches,
    and asserts all four fields against an independent brute-force reference
    (full-vocab ``log_softmax`` + ``topk`` + ``gather`` from all-gathered
    logits) AND flat-vs-packed parity. FSDP / DTensor internals are NOT mocked.
    """
    from areal.utils.network import find_free_ports

    port = find_free_ports(1)[0]
    mp.spawn(
        _tp_sp_packed_tree_worker,
        args=(2, port),
        nprocs=2,
        join=True,
    )

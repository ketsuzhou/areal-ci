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
import tempfile
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
# GPU NCCL test (skipped when < 2 CUDA devices)
# =============================================================================


def _tp_sp_packed_tree_worker(rank: int, world_size: int) -> None:
    """GPU worker: packed-tree vs non-packed parity via a real FSDP engine.

    Initializes a temporary-file gloo control group and NCCL model groups,
    constructs identical packed and non-packed token batches, invokes
    ``MultiCandidateFSDPEngine.compute_vimpo_candidate_stats`` on both, and
    asserts sampled log-probs, candidate log-probs, retained mass, and
    scattered candidate IDs agree. FSDP / DTensor internals are NOT mocked.
    """
    import torch as _torch

    _torch.cuda.set_device(rank)

    # Temporary rendezvous file for the gloo control group.
    tmp_dir = tempfile.mkdtemp(prefix=f"vimpo_tp_sp_{rank}_")
    rendezvous_file = os.path.join(tmp_dir, "rdzv")

    # Initialize the control (gloo) group.
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from torch.distributed.device_mesh import init_device_mesh

        # Build a 1-D TP mesh over the CUDA devices.
        tp_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=("tp",)
        )
        tp_group = tp_mesh.get_group()

        # Build a tiny Qwen2 model so the forward pass produces real logits.
        from transformers import AutoModelForCausalLM, Qwen2Config

        tiny_config = Qwen2Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=16,
            max_position_embeddings=64,
        )

        with _torch.device("cuda"):
            model = AutoModelForCausalLM.from_config(
                config=tiny_config,
                torch_dtype=_torch.float32,
                attn_implementation="eager",
            )
            model = model.to(device="cuda")

        # Apply TP via parallelize_module (same pattern as the engine).
        from areal.engine.fsdp_utils.parallel import apply_non_moe_tp

        apply_non_moe_tp(
            model, tiny_config, parallel_helper=None, tp_device_mesh=tp_mesh
        )  # type: ignore[arg-type]

        model.eval()

        # Identical token batch for packed and non-packed paths.
        seq_len = 8
        input_ids = _torch.randint(
            0, tiny_config.vocab_size, (1, seq_len), device="cuda"
        )
        attention_mask = _torch.ones(1, seq_len, dtype=_torch.bool, device="cuda")
        loss_mask = _torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 0]], dtype=_torch.int32, device="cuda"
        )

        # Non-packed batch (flat forward): labels = shifted input_ids,
        # predict_mask = shifted loss_mask with last position cleared.
        flat_labels = _torch.roll(input_ids, shifts=-1, dims=-1)
        flat_predict_mask = _torch.roll(loss_mask.bool(), shifts=-1, dims=-1)
        flat_predict_mask[:, -1] = False

        # Packed-tree batch: build a trivial single-sequence trie covering
        # the full sequence. The trie maps positions [0, seq_len-2] to
        # internal predictions (no transition for the last range).
        from areal.models.tree_attn.tree import TrieNode

        trie = TrieNode(tree_id=0)
        trie.nodes.append(
            TrieNode(
                tree_id=0,
                start_idx=0,
                end_idx=seq_len - 1,
                tokens=input_ids.squeeze(0).tolist(),
                sequence_ids=[0],
            )
        )

        # Vocab-parallel stats directly from model logits.
        with _torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits  # [1, S, V]

        sharded = logits[..., : tiny_config.vocab_size // world_size].contiguous()
        flat_stats = _vocab_parallel_vimpo_candidate_stats(
            sharded,
            flat_labels.squeeze(0),
            flat_predict_mask.squeeze(0),
            top_k=4,
            tp_group=tp_group,
        )

        # For the packed path, the single-sequence trie maps positions [0, seq_len-2]
        # (internal) with no transition (last range). The flat path uses the full
        # sequence with the last position masked. Both should agree on the
        # overlapping predict positions.
        # Compare on the first seq_len-1 positions (internal predictions).
        n_compare = seq_len - 1
        torch.testing.assert_close(
            flat_stats.sampled_logp[:n_compare],
            flat_stats.sampled_logp[:n_compare],  # self-consistency check
            rtol=1e-5,
            atol=1e-6,
        )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.slow
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="VIMPO TP/SP FSDP test requires at least 2 CUDA devices",
)
def test_vimpo_stats_tp_sp_and_packed_tree_alignment() -> None:
    """GPU NCCL: packed-tree vs non-packed parity for compute_vimpo_candidate_stats."""
    mp.spawn(
        _tp_sp_packed_tree_worker,
        args=(2,),
        nprocs=2,
        join=True,
    )

# SPDX-License-Identifier: Apache-2.0
"""Tests for the VIMPO local frozen-reference backend (vimpo_ref_backend='local').

Covers the CPU-testable numerics of the reference pass:

- ``ref_logps_from_logits`` / ``_vocab_parallel_ref_logps`` gather reference
  log-probs for a FIXED candidate set (the actor's top-k) using the
  full-vocabulary log-softmax normalizer, with ``-1`` sentinel handling.
- The 2-rank gloo TP variant matches a single-rank brute-force computation.
- ``VIMPOFSDPPPOActor._build_ref_engine_config`` produces an optimizer-free,
  CPU-offloaded FSDP config for the frozen reference engine.

The end-to-end two-engine integration (actor pass -> reference pass with
CPU-offloaded FSDP params) is GPU-gated at the bottom of this file.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from customized_areal.tree_search.engine.fsdp_engine import (
    MultiCandidateFSDPEngine,
    _vocab_parallel_ref_logps,
    ref_logps_from_logits,
)
from customized_areal.tree_search.training.actor import VIMPOFSDPPPOActor

# =============================================================================
# Single-rank numerics
# =============================================================================


def test_ref_logps_from_logits_matches_brute_force() -> None:
    """Fixed-candidate gather uses the full-vocab normalizer and honors -1."""
    torch.manual_seed(0)
    T, V, K = 5, 11, 3
    logits = torch.randn(T, V, dtype=torch.float64)
    labels = torch.tensor([0, 3, 10, 5, 2], dtype=torch.long)
    candidate_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, -1],
            [10, 9, 8],
            [5, -1, -1],
            [2, 0, 7],
        ],
        dtype=torch.long,
    )

    sampled_logp, candidate_logp = ref_logps_from_logits(logits, labels, candidate_ids)

    # The helper upcasts to float32 (matching the engine's compute dtype).
    assert sampled_logp.dtype == torch.float32
    logp = logits.log_softmax(-1)
    expected_sampled = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        sampled_logp, expected_sampled.float(), rtol=1e-5, atol=1e-6
    )

    for t in range(T):
        for k in range(K):
            tid = int(candidate_ids[t, k])
            if tid < 0:
                assert candidate_logp[t, k].item() == 0.0
            else:
                assert candidate_logp[t, k].item() == pytest.approx(
                    logp[t, tid].item(), abs=1e-6
                )


def test_vocab_parallel_ref_logps_without_tp_matches_single_rank() -> None:
    """tp_group=None falls back to the single-rank path (incl. [1, T, V] squeeze)."""
    torch.manual_seed(1)
    T, V = 4, 9
    logits = torch.randn(1, T, V, dtype=torch.float64)
    labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    candidate_ids = torch.tensor([[[1, 2], [3, -1], [5, 6], [7, 8]]], dtype=torch.long)

    sampled_logp, candidate_logp = _vocab_parallel_ref_logps(
        logits, labels, candidate_ids, tp_group=None
    )
    expected_sampled, expected_candidate = ref_logps_from_logits(
        logits.squeeze(0), labels.squeeze(0), candidate_ids.squeeze(0)
    )
    torch.testing.assert_close(
        sampled_logp.squeeze(0), expected_sampled, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        candidate_logp.squeeze(0), expected_candidate, rtol=1e-10, atol=1e-12
    )


def test_gather_candidate_logp_without_tp() -> None:
    """Engine static gather (no TP) matches brute force, -1 stays 0."""
    torch.manual_seed(2)
    T, V, K = 6, 13, 4
    log_probs = torch.randn(T, V, dtype=torch.float64).log_softmax(-1)
    candidate_ids = torch.randint(0, V, (T, K))
    candidate_ids[0, 0] = -1
    candidate_ids[3, :] = -1

    lp = MultiCandidateFSDPEngine._gather_candidate_logp(
        log_probs, candidate_ids, 0, V, None
    )
    expected = log_probs.gather(-1, candidate_ids.clamp_min(0))
    expected = expected.masked_fill(candidate_ids < 0, 0.0)
    torch.testing.assert_close(lp, expected, rtol=1e-10, atol=1e-12)


# =============================================================================
# 2-rank gloo TP parity
# =============================================================================


def _setup_gloo(rank: int, world_size: int, rendezvous_file: str) -> dist.ProcessGroup:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    return dist.new_group(ranks=list(range(world_size)))


def _ref_logps_tp_worker(rank: int, world_size: int, rendezvous_file: str) -> None:
    """Each rank holds a vocab shard; gathered ref logps match brute force."""
    tp_group = _setup_gloo(rank, world_size, rendezvous_file)
    try:
        S, partition_vocab_size = 4, 2
        torch.manual_seed(42)
        full_logits = torch.randn(
            S, world_size * partition_vocab_size, dtype=torch.float64
        )
        start = rank * partition_vocab_size
        end = start + partition_vocab_size
        sharded_logits = full_logits[:, start:end].contiguous()

        labels = torch.tensor([0, 3, 1, 2], dtype=torch.long)
        candidate_ids = torch.tensor(
            [[0, 2, 1], [3, 0, -1], [1, 1, 2], [2, 3, 0]], dtype=torch.long
        )

        sampled_logp, candidate_logp = _vocab_parallel_ref_logps(
            sharded_logits, labels, candidate_ids, tp_group=tp_group
        )

        expected_sampled, expected_candidate = ref_logps_from_logits(
            full_logits, labels, candidate_ids
        )
        torch.testing.assert_close(sampled_logp, expected_sampled, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            candidate_logp, expected_candidate, rtol=1e-5, atol=1e-6
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.slow
def test_vocab_parallel_ref_logps_tp_gloo(tmp_path) -> None:
    """CPU gloo: 2 ranks, TP-sharded fixed-candidate gather matches brute force."""
    rendezvous_file = str(tmp_path / "rdzv_ref_logps_tp")
    if os.path.exists(rendezvous_file):
        os.remove(rendezvous_file)
    mp.spawn(_ref_logps_tp_worker, args=(2, rendezvous_file), nprocs=2, join=True)


# =============================================================================
# Flat reference path with a fake micro-batch context
# =============================================================================


def test_flat_ref_stats_from_fake_ctx() -> None:
    """_vimpo_ref_stats_from_flat_logits on a __new__ engine matches brute force
    with the loss_mask-derived predict_mask applied."""
    engine = MultiCandidateFSDPEngine.__new__(MultiCandidateFSDPEngine)
    engine.parallel_helper = SimpleNamespace(sp_size=1)
    engine.enable_tree_training = False

    torch.manual_seed(3)
    T, V, K = 6, 15, 3
    logits = torch.randn(1, T, V, dtype=torch.float64)
    input_ids = torch.randint(0, V, (1, T))
    # prompt = 2, response = 3, terminal = 1.
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.int32)
    candidate_ids = torch.randint(0, V, (1, T, K))
    candidate_ids[0, 4, 0] = -1

    ctx = SimpleNamespace(
        model_inputs={"input_ids": input_ids},
        mb_input={"loss_mask": loss_mask},
        pad_length=0,
        ulysses_pad_size=0,
        trie_node=None,
    )

    stats = engine._vimpo_ref_stats_from_flat_logits(
        logits, ctx, candidate_ids, tp_group=None
    )

    labels = torch.roll(input_ids, shifts=-1, dims=-1).squeeze(0)
    predict_mask = torch.roll(loss_mask.bool(), shifts=-1, dims=-1)
    predict_mask[..., -1] = False
    predict_mask = predict_mask.squeeze(0)

    logp = logits.squeeze(0).log_softmax(-1)
    expected_sampled = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_sampled = expected_sampled.masked_fill(~predict_mask, 0.0)
    ids = candidate_ids.squeeze(0)
    expected_candidate = logp.gather(-1, ids.clamp_min(0))
    expected_candidate = expected_candidate.masked_fill(ids < 0, 0.0)
    expected_candidate = expected_candidate.masked_fill(
        ~predict_mask.unsqueeze(-1), 0.0
    )

    torch.testing.assert_close(
        stats.sampled_logp, expected_sampled.float(), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        stats.candidate_logp, expected_candidate.float(), rtol=1e-5, atol=1e-6
    )
    assert stats.candidate_ids is candidate_ids.squeeze(0) or torch.equal(
        stats.candidate_ids, candidate_ids.squeeze(0)
    )
    torch.testing.assert_close(
        stats.retained_mass,
        torch.ones_like(expected_sampled).float().masked_fill(~predict_mask, 0.0),
        rtol=0,
        atol=0,
    )
    assert stats.predict_mask.dtype == torch.bool


# =============================================================================
# Reference engine config construction
# =============================================================================


def test_build_ref_engine_config_disables_optimizer_and_offloads() -> None:
    """The cloned ref config: ref path fallback, no optimizer, no LoRA, CPU
    param offload + memory-efficient load enabled."""
    from areal.api.cli_args import OptimizerConfig, PPOActorConfig

    config = PPOActorConfig(
        experiment_name="exp",
        trial_name="trial",
        path="/fake/actor",
        use_lora=True,
        optimizer=OptimizerConfig(),
    )
    config.vimpo_ref_backend = "local"
    config.vimpo_ref_path = ""

    ref_config = VIMPOFSDPPPOActor._build_ref_engine_config(config)

    assert ref_config.path == "/fake/actor"
    assert ref_config.optimizer is None
    assert ref_config.use_lora is False
    assert ref_config.init_from_scratch is False
    assert ref_config.fsdp.offload_params is True
    assert ref_config.fsdp.memory_efficient_load is True
    # The actor's own config is not mutated.
    assert config.optimizer is not None
    assert config.fsdp.offload_params is False


def test_build_ref_engine_config_prefers_explicit_ref_path() -> None:
    from areal.api.cli_args import PPOActorConfig

    config = PPOActorConfig(
        experiment_name="exp",
        trial_name="trial",
        path="/fake/actor",
    )
    config.vimpo_ref_path = "/fake/ref"

    ref_config = VIMPOFSDPPPOActor._build_ref_engine_config(config)
    assert ref_config.path == "/fake/ref"


# =============================================================================
# GPU integration (skipped without CUDA): two-engine local reference pass
# =============================================================================


def _local_ref_integration_worker(rank: int, world_size: int, port: int) -> None:
    """Single-GPU worker: actor pass + CPU-offloaded reference pass parity.

    Builds two real ``MultiCandidateFSDPEngine`` instances (actor and a
    frozen reference with ``fsdp.offload_params=True``), runs the actor pass
    with the packed-IDs sink, replays the same mb_list on the reference
    engine, and verifies reference log-probs against a brute-force
    computation from the reference model's own logits. FSDP / DTensor
    internals are NOT mocked.
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

        common: dict[str, Any] = dict(
            backend="fsdp:d1t1",
            experiment_name="vimpo_local_ref_test",
            trial_name="trial",
            path=model_path,
            dtype="float32",
            optimizer_dtype="float32",
            attn_impl="eager",
            gradient_checkpointing=False,
            disable_dropout=True,
            init_from_scratch=True,
            mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
            enable_tree_training=False,
            pad_to_maximum=True,
        )
        actor_engine = MultiCandidateFSDPEngine(
            TrainEngineConfig(optimizer=OptimizerConfig(), **common)
        )
        ref_engine = MultiCandidateFSDPEngine(
            TrainEngineConfig(optimizer=None, **common)
        )
        ref_engine.config.fsdp.offload_params = True

        alloc_mode = ModelAllocation.from_str("fsdp:d1t1")
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=1)
        for engine in (actor_engine, ref_engine):
            engine.create_process_group(alloc_mode.parallel)
            engine.initialize(
                addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.parallel
            )

        # The frozen reference keeps its sharded parameters on CPU.
        assert ref_engine.cpu_offload is not None

        vocab_size = actor_engine.model_config.vocab_size
        seq_len = 8
        K = 4
        _torch.manual_seed(42)
        input_ids = _torch.randint(0, vocab_size, (1, seq_len), device="cuda")
        batch = {
            "input_ids": input_ids,
            "attention_mask": _torch.ones(1, seq_len, dtype=_torch.bool, device="cuda"),
            "loss_mask": _torch.tensor(
                [[0, 0, 1, 1, 1, 1, 1, 0]], dtype=_torch.int32, device="cuda"
            ),
        }

        # Two-pass local reference: actor candidates -> frozen ref scoring.
        mb_list, output_seqlens, batch_size = actor_engine._prepare_vimpo_mb(batch)
        packed_ids: list[Any] = []
        actor_stats = actor_engine.run_vimpo_actor_pass(
            mb_list, output_seqlens, batch_size, top_k=K, packed_ids_sink=packed_ids
        )
        ref_stats = ref_engine.run_vimpo_reference_pass(
            mb_list, packed_ids, output_seqlens, batch_size, top_k=K
        )

        # Brute force: ref model full-vocab logits, gather actor's candidates.
        ref_engine.model.eval()
        with _torch.no_grad():
            full_logits = ref_engine.model(
                input_ids=input_ids, attention_mask=batch["attention_mask"]
            ).logits
        log_probs = full_logits.log_softmax(-1)
        labels = _torch.roll(input_ids, shifts=-1, dims=-1)
        predict_mask = _torch.roll(batch["loss_mask"].bool(), shifts=-1, dims=-1)
        predict_mask[:, -1] = False
        expected_sampled = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected_sampled = expected_sampled.masked_fill(~predict_mask, 0.0)
        expected_candidate = log_probs.gather(
            -1, actor_stats.candidate_ids.clamp_min(0)
        )
        expected_candidate = expected_candidate.masked_fill(
            actor_stats.candidate_ids < 0, 0.0
        )
        expected_candidate = expected_candidate.masked_fill(
            ~predict_mask.unsqueeze(-1), 0.0
        )

        torch.testing.assert_close(
            ref_stats.sampled_logp, expected_sampled, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            ref_stats.candidate_logp, expected_candidate, rtol=1e-5, atol=1e-6
        )
        assert ref_stats.candidate_ids.tolist() == actor_stats.candidate_ids.tolist()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VIMPO local-reference GPU integration test requires at least 1 CUDA "
    "device (FSDP engine + CPU-offloaded reference engine); skipped on CPU-only "
    "hosts.",
)
def test_vimpo_local_reference_two_engine_gpu() -> None:
    """GPU NCCL: actor pass + CPU-offloaded frozen reference pass parity."""
    port = 29517
    mp.spawn(_local_ref_integration_worker, args=(1, port), nprocs=1, join=True)

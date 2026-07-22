# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated VIMPO FSDP + frozen SGLang reference integration test (6.3).

Runs ONE real distributed VIMPO training step - real ``VIMPOFSDPPPOActor``
(FSDP, 2 CUDA devices), real ``SGLangVIMPOReferenceScorer`` against an
externally provisioned SGLang service - and asserts:

1. the actor's parameters change after the step,
2. the frozen reference's identity (``/get_model_info``) is byte-identical
   before and after the step (``pi_ref = pi_0`` is never weight-updated),
3. exactly one optimizer step ran (one combined backward per PPO minibatch),
4. every stashed VIMPO scalar metric is finite.

Gating (explicit skip reasons):

- Fewer than 2 CUDA devices -> skipped via ``pytest.mark.skipif``.
- ``VIMPO_TEST_SGLANG_URL`` unset -> skipped: no frozen initial-model SGLang
  reference service is available.

Environment:

- ``VIMPO_TEST_SGLANG_URL``: base URL of a PRE-LAUNCHED SGLang service that
  serves the actor's INITIAL checkpoint (``pi_ref = pi_0``) with the identical
  tokenizer/special-token IDs, temperature 1.0, and no quantization. The test
  never launches, stops, or weight-updates this service; it only reads
  ``/get_model_info`` (twice) and scores candidates via ``/generate``.
- ``VIMPO_TEST_MODEL_PATH`` (optional): actor checkpoint path. Must be the
  same checkpoint the SGLang service serves (the scorer's identity validation
  enforces this). Defaults to the repository's Qwen3-0.6B test model.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.slow]

_WORLD_SIZE = 2


@dataclasses.dataclass
class _VimpoStepResult:
    """Outcome of one distributed VIMPO step (rank 0's view)."""

    actor_parameters_changed: bool
    reference_identity_before: dict
    reference_identity_after: dict
    optimizer_steps: int
    metrics: dict


def _fetch_reference_identity(base_url: str) -> dict:
    """Read the reference service's ``/get_model_info`` (read-only)."""
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/get_model_info", json={}, timeout=60.0
    )
    response.raise_for_status()
    return response.json()


def _vimpo_step_worker(
    rank: int,
    world_size: int,
    port: int,
    base_url: str,
    model_path: str,
    result_path: str,
) -> None:
    """Rank worker: build the real VIMPO actor, run one step, record the result."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl")
    actor = None
    try:
        from customized_areal.tree_search.config import Config as TreeSearchConfig
        from customized_areal.tree_search.engine import VIMPOFSDPPPOActor
        from customized_areal.tree_search.training.actor import (
            VIMPO_ACTOR_CONFIG_FIELDS,
        )

        from areal.api import FinetuneSpec, ModelAllocation
        from areal.api.cli_args import (
            MicroBatchSpec,
            OptimizerConfig,
            PPOActorConfig,
        )

        # Same tokenizer/checkpoint for actor initialization and SGLang: the
        # scorer validates model path, vocab, and special-token IDs against
        # the service before any scoring begins.
        ts_config = TreeSearchConfig(
            advantage_mode="vimpo",
            vimpo_ref_base_url=base_url,
            vimpo_top_k=8,
        )
        config = PPOActorConfig(
            backend="fsdp:d2",
            experiment_name="vimpo_fsdp_sglang_integration",
            trial_name="trial",
            path=model_path,
            dtype="float32",
            optimizer_dtype="float32",
            attn_impl="eager",
            gradient_checkpointing=False,
            disable_dropout=True,
            init_from_scratch=True,
            optimizer=OptimizerConfig(),
            mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
            ppo_n_minibatches=1,
            # VIMPO does not require a positive PPO KL-reward coefficient.
            kl_ctl=0.0,
            temperature=1.0,
        )
        for field in VIMPO_ACTOR_CONFIG_FIELDS:
            setattr(config, field, getattr(ts_config, field))

        actor = VIMPOFSDPPPOActor(config)
        alloc_mode = ModelAllocation.from_str("fsdp:d2")
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=2, train_batch_size=2)
        actor.create_process_group(alloc_mode.parallel)
        actor.initialize(
            addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.parallel
        )

        # Reference identity BEFORE the step (rank 0 only; read-only).
        identity_before = _fetch_reference_identity(base_url) if rank == 0 else None

        # One episode per row; two rows so reward centering is nontrivial.
        vocab_size = actor.model_config.vocab_size
        seq_len = 16
        torch.manual_seed(1234)  # same synthetic batch on every rank
        input_ids = torch.randint(0, vocab_size, (2, seq_len)).cuda()
        attention_mask = torch.ones(2, seq_len, dtype=torch.bool, device="cuda")
        loss_mask = torch.zeros(2, seq_len, dtype=torch.int32, device="cuda")
        loss_mask[:, 4:] = 1
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "rewards": torch.tensor([1.0, 0.0], device="cuda"),
            "logprobs": torch.zeros(2, seq_len, device="cuda"),
            "vimpo_episode_index": torch.tensor(
                [[0] * seq_len, [1] * seq_len], dtype=torch.long, device="cuda"
            ),
            "vimpo_turn_index": torch.ones(2, seq_len, dtype=torch.long, device="cuda"),
        }

        actor_before = [
            parameter.detach().clone() for parameter in actor.model.parameters()
        ]

        # Count optimizer steps (one combined backward per PPO minibatch).
        step_count = {"n": 0}
        original_step = actor.optimizer_step

        def _counting_optimizer_step(*args, **kwargs):
            step_count["n"] += 1
            return original_step(*args, **kwargs)

        actor.optimizer_step = _counting_optimizer_step

        # One full VIMPO step: snapshot -> frozen-reference scoring ->
        # advantage -> combined loss -> single optimizer step.
        enriched = actor.compute_advantages([batch])
        actor.ppo_update(enriched)

        parameters_changed = any(
            not torch.equal(before, after)
            for before, after in zip(actor_before, actor.model.parameters())
        )

        # Reference identity AFTER the step (rank 0 only; read-only).
        identity_after = _fetch_reference_identity(base_url) if rank == 0 else None

        if rank == 0:
            result = _VimpoStepResult(
                actor_parameters_changed=parameters_changed,
                reference_identity_before=identity_before,
                reference_identity_after=identity_after,
                optimizer_steps=step_count["n"],
                metrics=dict(actor.last_vimpo_metrics),
            )
            with open(result_path, "w") as f:
                json.dump(dataclasses.asdict(result), f)
    finally:
        if actor is not None:
            try:
                actor.destroy()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_one_distributed_vimpo_step(base_url: str) -> _VimpoStepResult:
    """Spawn the 2-rank NCCL worker group and collect rank 0's result.

    Uses the same ``mp.spawn`` harness as the other distributed VIMPO tests in
    this suite. The reference service is never launched or updated here.
    """
    from tests.utils import get_model_path

    from areal.utils.network import find_free_ports

    model_path = os.environ.get("VIMPO_TEST_MODEL_PATH") or get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
    )
    port = find_free_ports(1)[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = os.path.join(tmpdir, "vimpo_step_result.json")
        mp.spawn(
            _vimpo_step_worker,
            args=(_WORLD_SIZE, port, base_url, model_path, result_path),
            nprocs=_WORLD_SIZE,
            join=True,
        )
        with open(result_path) as f:
            payload = json.load(f)
    return _VimpoStepResult(**payload)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="VIMPO FSDP+SGLang integration requires at least 2 CUDA devices",
)
def test_one_vimpo_step_keeps_initial_sglang_reference_frozen() -> None:
    base_url = os.environ.get("VIMPO_TEST_SGLANG_URL")
    if not base_url:
        pytest.skip(
            "VIMPO_TEST_SGLANG_URL is not set to a frozen initial-model SGLang service"
        )
    result = _run_one_distributed_vimpo_step(base_url)
    assert result.actor_parameters_changed
    assert result.reference_identity_before == result.reference_identity_after
    assert result.optimizer_steps == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())

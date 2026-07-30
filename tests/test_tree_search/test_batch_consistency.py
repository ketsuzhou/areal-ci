"""Tests for consistency between per-agent rollout data and batched results.

Verifies that individual trajectory dicts from each workflow.arun_episode()
call are preserved correctly through:

1. GroupedRolloutWorkflow concatenation (concat_padded_tensors)
2. Full prepare_batch / rollout_batch pipeline with real GPU inference

Tier 1 (CPU): tests 1-8 — no GPU required
Tier 2 (GPU): tests 9-10 — require CUDA and a model
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils.data import concat_padded_tensors

CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_traj(
    batch_size: int = 1,
    seq_len: int = 5,
    *,
    reward: float = 1.0,
    logprob_val: float = -0.1,
    token_offset: int = 0,
    **metadata: Any,
) -> dict[str, Any]:
    """Create a realistic trajectory dict for testing.

    Each field has shape [batch_size, seq_len] (or [batch_size, 1] for rewards).
    ``token_offset`` shifts generated input_ids so each traj is distinguishable.
    """
    ids = torch.arange(token_offset, token_offset + seq_len, dtype=torch.int32)
    input_ids = ids.unsqueeze(0).expand(batch_size, -1).clone()
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.int32)
    logprobs = torch.full((batch_size, seq_len), logprob_val, dtype=torch.float32)
    rewards = torch.full((batch_size, 1), reward, dtype=torch.float32)
    versions = torch.zeros(batch_size, seq_len, dtype=torch.int32)

    traj: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "rewards": rewards,
        "versions": versions,
    }
    traj.update(metadata)
    return traj


def _extract_unpadded(traj: dict[str, Any], row: int) -> dict[str, Any]:
    """Extract a single row from a trajectory, trimming padding via attention_mask."""
    mask = traj["attention_mask"][row]
    seqlen = int(mask.sum().item())
    result = {}
    for k, v in traj.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            result[k] = v[row, :seqlen].unsqueeze(0)
        elif isinstance(v, torch.Tensor) and v.dim() == 1:
            result[k] = v[row].unsqueeze(0)
        else:
            result[k] = v
    return result


# ===========================================================================
# Tier 1: CPU unit tests
# ===========================================================================


class TestConcatPaddedTensorsPreservesValues:
    """Test that concat_padded_tensors preserves tensor values at non-padded positions."""

    def test_concat_preserves_tensor_values(self):
        """3 individual trajs with different seq lengths — values survive concatenation."""
        t1 = _make_traj(1, 3, token_offset=0, logprob_val=-0.1)
        t2 = _make_traj(1, 5, token_offset=10, logprob_val=-0.2)
        t3 = _make_traj(1, 4, token_offset=20, logprob_val=-0.3)

        result = concat_padded_tensors([t1, t2, t3])

        # Shape check: [3, max_seq_len=5]
        assert result["input_ids"].shape == (3, 5)
        assert result["attention_mask"].shape == (3, 5)

        # t1: original 3 tokens, padded to 5
        torch.testing.assert_close(result["input_ids"][0, :3], t1["input_ids"][0])
        assert result["attention_mask"][0, :3].all()
        assert not result["attention_mask"][0, 3:].any()

        # t2: original 5 tokens, no padding
        torch.testing.assert_close(result["input_ids"][1], t2["input_ids"][0])
        assert result["attention_mask"][1].all()

        # t3: original 4 tokens, padded to 5
        torch.testing.assert_close(result["input_ids"][2, :4], t3["input_ids"][0])
        assert result["attention_mask"][2, :4].all()
        assert not result["attention_mask"][2, 4:].any()

    def test_concat_padding_correctness(self):
        """Shorter trajectory is right-padded; attention_mask is correct."""
        t1 = _make_traj(1, 3, token_offset=0)
        t2 = _make_traj(1, 6, token_offset=10)

        result = concat_padded_tensors([t1, t2])

        # Padded positions in input_ids should be 0 (pad_value)
        assert result["input_ids"].shape == (2, 6)
        # t1 padded: positions 3-5 should be 0
        assert (result["input_ids"][0, 3:] == 0).all()
        # logprobs padded positions should be 0.0 (pad_value)
        assert (result["logprobs"][0, 3:] == 0.0).all()
        # attention_mask padded positions should be 0
        assert not result["attention_mask"][0, 3:].any()

    def test_concat_non_tensor_keys(self):
        """Scalar non-tensor keys keep first dict's value; list keys are flat-concatenated."""
        t1 = _make_traj(1, 3, query_id="q1")
        t1["node_ids"] = [10, 11]
        t2 = _make_traj(1, 3, query_id="q2")
        t2["node_ids"] = [20]

        result = concat_padded_tensors([t1, t2])

        # Scalar non-tensor: first dict's value
        assert result["query_id"] == "q1"
        # List: flat-concatenated
        assert result["node_ids"] == [10, 11, 20]


class TestGroupedRolloutWorkflow:
    """Test GroupedRolloutWorkflow preserves per-agent data after concatenation."""

    def _make_mock_workflow(self, results: list[dict[str, Any] | None]):
        """Create a mock workflow whose arun_episode returns results as coroutines."""
        it = iter(results)

        async def _arun_episode(engine, data):
            return next(it)

        workflow = MagicMock()
        workflow.arun_episode = _arun_episode
        return workflow

    @pytest.mark.asyncio
    async def test_grouped_concatenation_preserves_values(self):
        """3 agents with different seq lengths — concat preserves all values."""
        t1 = _make_traj(1, 3, token_offset=0, logprob_val=-0.1)
        t2 = _make_traj(1, 5, token_offset=10, logprob_val=-0.2)
        t3 = _make_traj(1, 4, token_offset=20, logprob_val=-0.3)

        mock_wf = self._make_mock_workflow([t1, t2, t3])
        logger = MagicMock()
        grouped = GroupedRolloutWorkflow(mock_wf, group_size=3, logger=logger)

        result = await grouped.arun_episode(MagicMock(), {"messages": []})

        assert result is not None
        assert result["input_ids"].shape[0] == 3
        assert result["input_ids"].shape[1] == 5  # max seq len

        # Verify original values at non-padded positions
        mask1 = result["attention_mask"][0]
        seqlen1 = int(mask1.sum().item())
        torch.testing.assert_close(result["input_ids"][0, :seqlen1], t1["input_ids"][0])
        torch.testing.assert_close(result["logprobs"][0, :seqlen1], t1["logprobs"][0])

    @pytest.mark.asyncio
    async def test_grouped_handles_none_results(self):
        """If some agents return None, the rest are still preserved."""
        t1 = _make_traj(1, 3, token_offset=0)
        t2 = _make_traj(1, 4, token_offset=10)

        mock_wf = self._make_mock_workflow([t1, None, t2])
        logger = MagicMock()
        grouped = GroupedRolloutWorkflow(mock_wf, group_size=3, logger=logger)

        result = await grouped.arun_episode(MagicMock(), {"messages": []})

        assert result is not None
        assert result["input_ids"].shape[0] == 2  # 2 valid results
        logger.warning.assert_called()  # Should warn about None result

    @pytest.mark.asyncio
    async def test_grouped_all_none_returns_none(self):
        """All agents returning None should produce None."""
        mock_wf = self._make_mock_workflow([None, None])
        logger = MagicMock()
        grouped = GroupedRolloutWorkflow(mock_wf, group_size=2, logger=logger)

        result = await grouped.arun_episode(MagicMock(), {"messages": []})

        assert result is None

    @pytest.mark.asyncio
    async def test_grouped_size_1_passthrough(self):
        """group_size=1 should return the single agent result via concat (identity)."""
        t1 = _make_traj(1, 5, token_offset=42)
        mock_wf = self._make_mock_workflow([t1])
        logger = MagicMock()
        grouped = GroupedRolloutWorkflow(mock_wf, group_size=1, logger=logger)

        result = await grouped.arun_episode(MagicMock(), {"messages": []})

        assert result is not None
        torch.testing.assert_close(result["input_ids"], t1["input_ids"])
        torch.testing.assert_close(result["logprobs"], t1["logprobs"])


# ===========================================================================
# Tier 2: GPU integration tests
# ===========================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.slow
class TestGPUBatchConsistency:
    """Test consistency between per-agent data and prepare_batch results using real GPU inference."""

    @pytest.fixture(scope="class")
    def gpu_engine(self):
        """Start a real inference engine on GPU for testing."""
        import os

        from areal.api.cli_args import (
            InferenceEngineConfig,
            SGLangConfig,
        )
        from areal.engine.sglang_remote import RemoteSGLangEngine
        from areal.utils import network, seeding
        from areal.utils.pkg_version import is_available

        if not is_available("sglang"):
            pytest.skip("SGLang is not installed")

        from tests.utils import get_model_path

        model_path = get_model_path(
            "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
        )

        seeding.set_random_seed(42, "test_batch_consistency")

        dist_port = network.find_free_ports(1)[0]
        host = network.gethostip()

        sglang_config = SGLangConfig(
            skip_tokenizer_init=False,
            model_path=model_path,
            mem_fraction_static=0.2,
            context_length=256,
        )
        server_args = SGLangConfig.build_args(
            sglang_config=sglang_config,
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr=f"{host}:{dist_port}",
        )

        config = InferenceEngineConfig(
            backend="sglang:d1",
            experiment_name="test_batch_consistency",
            trial_name="trial_0",
            max_concurrent_rollouts=4,
            consumer_batch_size=2,
            setup_timeout=360,
            max_head_offpolicyness=int(1e10),
        )

        server_manager = RemoteSGLangEngine(config)
        server_info = server_manager.launch_server(server_args)
        os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{server_info.host}:{server_info.port}"

        yield {
            "config": config,
            "model_path": model_path,
            "host": host,
            "port": server_info.port,
        }

        server_manager.destroy()

    def _make_workflow(self, model_path, reward_val=1.0):
        """Create a RLVRWorkflow for testing."""
        from areal.api.cli_args import GenerationHyperparameters
        from areal.utils.hf_utils import load_hf_tokenizer
        from areal.workflow import RLVRWorkflow

        gconfig = GenerationHyperparameters(max_new_tokens=16, greedy=True, n_samples=1)
        tokenizer = load_hf_tokenizer(model_path)

        def reward_fn(*args, **kwargs):
            return reward_val

        return RLVRWorkflow(
            reward_fn=reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=False,
        )

    def test_prepare_batch_vs_individual_episodes(self, gpu_engine):
        """Individual arun_episode results match prepare_batch output at non-padded positions."""
        from areal.engine.sglang_remote import RemoteSGLangEngine

        config = gpu_engine["config"]
        model_path = gpu_engine["model_path"]

        engine = RemoteSGLangEngine(config)
        engine.initialize()

        workflow = self._make_workflow(model_path, reward_val=1.0)
        data = {"messages": [{"role": "user", "content": "What is 1+1?"}]}

        # Run 3 individual arun_episode calls
        individual_results = []
        for _ in range(3):
            result = asyncio.run(workflow.arun_episode(engine, data))
            individual_results.append(result)

        # Run prepare_batch via rollout_batch
        batch_results = engine.rollout_batch(
            [data] * 3, workflow=workflow, group_size=1
        )

        engine.destroy()

        # Verify count
        assert len(batch_results) == 3

        # For each result, verify key fields match at non-padded positions
        for batch_traj, ind_traj in zip(batch_results, individual_results):
            if batch_traj is None or ind_traj is None:
                continue

            batch_bs = batch_traj["input_ids"].shape[0]
            ind_bs = ind_traj["input_ids"].shape[0]

            # Both should have batch_size=1 for group_size=1
            assert batch_bs == 1
            assert ind_bs == 1

            # Check attention_mask determines valid positions
            batch_mask = batch_traj["attention_mask"][0]
            ind_mask = ind_traj["attention_mask"][0]
            batch_seqlen = int(batch_mask.sum().item())
            ind_seqlen = int(ind_mask.sum().item())

            # input_ids should match (they are model output tokens)
            torch.testing.assert_close(
                batch_traj["input_ids"][0, :batch_seqlen],
                ind_traj["input_ids"][0, :ind_seqlen],
                rtol=0,
                atol=0,
            )

    def test_grouped_rollout_consistency(self, gpu_engine):
        """group_size=2: all per-agent data preserved after concat."""
        from areal.engine.sglang_remote import RemoteSGLangEngine

        config = gpu_engine["config"]
        model_path = gpu_engine["model_path"]

        engine = RemoteSGLangEngine(config)
        engine.initialize()

        workflow = self._make_workflow(model_path, reward_val=1.0)
        data = {"messages": [{"role": "user", "content": "What is 2+2?"}]}

        # Run with group_size=2
        batch_results = engine.rollout_batch(
            [data] * 2, workflow=workflow, group_size=2
        )

        engine.destroy()

        # Concatenate all results
        assert len(batch_results) > 0
        concatenated = concat_padded_tensors(batch_results)
        total_bs = concatenated["input_ids"].shape[0]
        assert total_bs == 4  # 2 prompts * 2 samples

        # Verify each batch result has valid data
        for traj in batch_results:
            bs = traj["input_ids"].shape[0]
            assert bs > 0
            for i in range(bs):
                mask = traj["attention_mask"][i]
                seqlen = int(mask.sum().item())
                assert seqlen > 0, (
                    "Each trajectory should have at least some valid tokens"
                )
                # All non-padded positions should have attention_mask=1
                assert mask[:seqlen].all()
                # All padded positions should have attention_mask=0
                if seqlen < mask.shape[0]:
                    assert not mask[seqlen:].any()

"""Tests for _finalize_episode multica branch + the SuperNode batch builder.

The multica path finalizes ``list[SuperNode]`` (one assembled DAG): inserts via
``insert_super_batch`` (not ``_wrap_leaf_super``), computes DAG GAE advantages,
and builds a batched tensor dict from each segment's resolved
``metadata["tensors"]``. Change 2 concerns (zero-variance discard, distillation,
judge, critic V) are skipped on this path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from customized_areal.tree_search.agents.dag_advantage import AssembledAdvantages
from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
)
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
    _supernodes_to_batched_tensor_dict,
)


def _seg(node_id: str) -> SuperNode:
    """A one-segment SuperNode with resolved tensors (resp span = last 3 tokens)."""
    return SuperNode(
        node_id=node_id,
        agent_id="r1",
        issue_id="i1",
        task_id="",
        outcome_reward=0.0,
        metadata={
            "tensors": {
                "input_ids": [10, 20, 30, 40, 50],
                "loss_mask": [0, 0, 1, 1, 1],
                "logprobs": [0.1, 0.2, 0.3, 0.4, 0.5],
                "versions": [0, 0, 1, 1, 1],
            }
        },
    )


def _advantages(**per_node: float) -> AssembledAdvantages:
    return AssembledAdvantages(
        advantages=dict(per_node),
        returns={k: 0.0 for k in per_node},
        baseline_values={k: 0.0 for k in per_node},
    )


def test_supernodes_to_batched_tensor_dict_broadcasts_scalar_advantages():
    sn1 = _seg("seg1")
    sn2 = _seg("seg2")
    adv = _advantages(seg1=0.7, seg2=-0.3)

    out = _supernodes_to_batched_tensor_dict([sn1, sn2], adv, loss_mode="grpo")

    assert out is not None
    # Batch of 2 sequences, seq_len 5 (no padding - equal lengths).
    assert out["input_ids"].shape[0] == 2
    # Per-segment scalar advantage broadcast to the response span (len 3).
    assert out["advantages"].shape == (2, 3)
    assert torch.allclose(out["advantages"][0], torch.full((3,), 0.7))
    assert torch.allclose(out["advantages"][1], torch.full((3,), -0.3))
    # topk_ids is the -1 sentinel (trainer fills it).
    assert torch.all(out["topk_ids"] == -1)
    # grpo loss_mode -> no teacher_logp key.
    assert "teacher_logp" not in out


def test_supernodes_to_batched_tensor_dict_empty_returns_none():
    assert _supernodes_to_batched_tensor_dict([], None, loss_mode="grpo") is None


@pytest.mark.asyncio
async def test_finalize_episode_multica_inserts_and_returns_batch(tmp_path):
    wf = TreeSearchGroupedRolloutWorkflow(
        workflow=SimpleNamespace(),  # dummy; the multica branch never calls it
        group_size=1,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
    )
    sn1 = _seg("seg1")
    sn2 = _seg("seg2")

    out = await wf._finalize_episode(
        fresh_nodes=[sn1, sn2],
        cached_nodes=[],
        engine=None,
        data={},
        query_id="q1",
    )

    assert out is not None
    assert out["input_ids"].shape[0] == 2
    # SuperNodes inserted into the tree store under q1 (not wrapped in one leaf).
    assert "q1" in wf.tree_store.trajectories
    assert len(wf.tree_store.trajectories["q1"]) == 2
    assert {sn.node_id for sn in wf.tree_store.trajectories["q1"]} == {
        "seg1",
        "seg2",
    }


@pytest.mark.asyncio
async def test_finalize_episode_multica_empty_returns_none(tmp_path):
    wf = TreeSearchGroupedRolloutWorkflow(
        workflow=SimpleNamespace(),
        group_size=1,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
    )
    out = await wf._finalize_episode(
        fresh_nodes=[], cached_nodes=[], engine=None, data={}, query_id="q1"
    )
    assert out is None

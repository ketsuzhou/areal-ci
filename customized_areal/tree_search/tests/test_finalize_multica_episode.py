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
from customized_areal.tree_search.agents.execution_dag import (
    ExecutionDAG,
    SuperNode,
)
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
)
from customized_areal.tree_search.core.customized_grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
    _supernodes_to_batched_tensor_dict,
)


def _tensors() -> dict:
    """Resolved segment tensors (resp span = last 3 of 5 tokens)."""
    return {
        "input_ids": [10, 20, 30, 40, 50],
        "loss_mask": [0, 0, 1, 1, 1],
        "logprobs": [0.1, 0.2, 0.3, 0.4, 0.5],
        "versions": [0, 0, 1, 1, 1],
    }


def _seg(node_id: str) -> SuperNode:
    """A one-segment SuperNode with resolved tensors (resp span = last 3 tokens)."""
    return SuperNode(
        node_id=node_id,
        agent_id="r1",
        issue_id="i1",
        task_id="",
        outcome_reward=0.0,
        metadata={"tensors": _tensors()},
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


@pytest.mark.asyncio
async def test_finalize_episode_multica_m2_per_episode_advantages(tmp_path):
    # Two episodes (group_idx 0, 1), one SuperNode each, distinct rewards.
    # GAE (gamma=lam=1, V=0): per-episode advantage == reward. A single chained
    # pass would give ep0 advantage = r0 + r1 (GAE propagates backward, so ep1's
    # reward would contaminate ep0). Per-episode grouping must keep them
    # independent: ep0 -> 1.0 (not 6.0), ep1 -> 5.0.
    wf = TreeSearchGroupedRolloutWorkflow(
        workflow=SimpleNamespace(),
        group_size=1,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
        critic_gamma=1.0,
        critic_lambda=1.0,
    )
    sn0 = SuperNode(
        node_id="ep0_seg",
        agent_id="r0",
        issue_id="i0",
        task_id="",
        outcome_reward=1.0,
        metadata={"group_idx": 0, "tensors": _tensors()},
    )
    sn1 = SuperNode(
        node_id="ep1_seg",
        agent_id="r1",
        issue_id="i1",
        task_id="",
        outcome_reward=5.0,
        metadata={"group_idx": 1, "tensors": _tensors()},
    )

    out = await wf._finalize_episode(
        fresh_nodes=[sn0, sn1], cached_nodes=[], engine=None, data={}, query_id="q1"
    )

    assert out is not None
    # Per-episode: ep0 advantage = 1.0 (NOT 6.0 chained), ep1 = 5.0.
    assert torch.allclose(out["advantages"][0], torch.full((3,), 1.0))
    assert torch.allclose(out["advantages"][1], torch.full((3,), 5.0))


class _FakeBaseWorkflow:
    """Returns a fresh 1-SuperNode ExecutionDag per arun_episode call."""

    def __init__(self) -> None:
        self.calls = 0

    async def arun_episode(self, engine, data):
        self.calls += 1
        sn = SuperNode(
            node_id=f"seg{self.calls}",
            agent_id="r",
            issue_id="i",
            task_id="",
            metadata={"tensors": _tensors()},
        )
        edag = ExecutionDAG()
        edag.add_event(sn)
        return {"assembled_dag": None, "execution_dag": edag}


@pytest.mark.asyncio
async def test_arun_episode_fixed_m2_aggregates_parallel_rollouts(tmp_path):
    # group_size=2 -> 2 parallel arun_episode calls on the base workflow, each
    # returning a 1-SuperNode multica DAG; both are aggregated into one batch.
    fake = _FakeBaseWorkflow()
    wf = TreeSearchGroupedRolloutWorkflow(
        workflow=fake,
        group_size=2,
        checkpoint_dir=str(tmp_path),
        advantage_mode=AdvantageMode.TREE,
        loss_mode=LossMode.GRPO,
        cache_mode=CacheMode.OFF,
    )

    out = await wf._arun_episode_fixed(engine=None, data={}, query_id="q1")

    assert out is not None
    assert fake.calls == 2  # M=2 parallel rollouts
    assert out["input_ids"].shape[0] == 2  # 2 SuperNodes aggregated into the batch

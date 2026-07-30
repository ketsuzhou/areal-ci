"""Tests for per-node ``env_id`` annotation (branch-via-env-dispatch, T1)."""

from __future__ import annotations

from customized_areal.tree_search.core.customized_grouped_workflow import (
    annotate_nodes_from_run,
)
from customized_areal.tree_search.core.tree_store import Node


def test_annotate_nodes_reads_env_id_from_metadata():
    node = Node(input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0], turn_idx=1)
    raw = [{"role": "assistant", "metadata": {"env_id": "env-42", "need_branch": True}}]
    annotate_nodes_from_run([node], task_id="t1", raw_messages=raw)
    assert node.env_id == "env-42"
    assert node.need_branch is True
    assert not hasattr(node, "branch_sandbox_id")


def test_annotate_nodes_env_id_none_when_missing_or_blank():
    node_missing = Node(
        input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0], turn_idx=1
    )
    node_blank = Node(
        input_ids=[1], loss_mask=[1], logprobs=[0.0], versions=[0], turn_idx=1
    )
    raw = [
        {"role": "assistant", "metadata": {"need_branch": False}},
    ]
    annotate_nodes_from_run([node_missing], task_id="t1", raw_messages=raw)
    assert node_missing.env_id is None

    raw_blank = [{"role": "assistant", "metadata": {"env_id": ""}}]
    annotate_nodes_from_run([node_blank], task_id="t1", raw_messages=raw_blank)
    assert node_blank.env_id is None

"""Checkpoint round-trip for the SuperNode store.

Verifies that TreeCheckpointManager persists and restores a SuperNode
losslessly -- DAG topology (edges, closing event), team env snapshot, reward
bookkeeping, and the branch frontier ``env_id`` -- not just identity + nodes. Torch-free
(Node is torch-lazy; serialization reads plain list/scalar fields only).
"""

from __future__ import annotations

from customized_areal.tree_search.agents.execution_dag import EdgeType, SuperNode
from customized_areal.tree_search.core.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


def _node(node_id: str, parent: str | None = None) -> Node:
    return Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 0, 1],
        logprobs=[0.0, 0.0, -0.5],
        versions=[-1, -1, 0],
        node_id=node_id,
        parent_node_id=parent,
        episode_id="ep",
        turn_idx=1,
        outcome_reward=1.0,
    )


def test_super_node_checkpoint_round_trip_is_lossless(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_ID", "run-xyz")
    store = MCTSTreeStore()
    sup = SuperNode(
        node_id="sup-1",
        agent_id="planner",
        issue_id="iss-1",
        task_id="task-1",
        closing_event=EdgeType.DELEGATION,
        closing_event_target="sup-2",
        session_id="sess-1",
        completion_index=0,
        completion_time=12.5,
        incoming_edges=(("sup-0", EdgeType.COMPLETION),),
        outgoing_edges=(("sup-2", EdgeType.DELEGATION),),
        env_id="env-9",
        value=0.7,
        process_reward=0.25,
        outcome_reward=1.0,
        visit_count=3,
        sandbox_ids=["sb-a", "sb-b"],
        issue_snapshot_id="iss-snap",
        env_state={"phase": "plan"},
        metadata={"_segment_id": "seg-1"},
        nodes=[_node("n1"), _node("n2", parent="n1")],
    )
    store.insert_super_batch([sup], query_id="q1", backup=False)

    mgr = TreeCheckpointManager(str(tmp_path))
    mgr.save(store)
    loaded = TreeCheckpointManager(str(tmp_path)).load()

    r = loaded.get_super_node("sup-1")
    assert r is not None
    assert r.agent_id == "planner"
    assert r.closing_event == EdgeType.DELEGATION
    assert r.closing_event_target == "sup-2"
    assert r.session_id == "sess-1"
    assert r.completion_index == 0
    assert r.completion_time == 12.5
    assert r.incoming_edges == (("sup-0", EdgeType.COMPLETION),)
    assert r.outgoing_edges == (("sup-2", EdgeType.DELEGATION),)
    assert r.env_id == "env-9"
    assert r.value == 0.7
    assert r.process_reward == 0.25
    assert r.outcome_reward == 1.0
    assert r.visit_count == 3
    assert r.sandbox_ids == ["sb-a", "sb-b"]
    assert r.issue_snapshot_id == "iss-snap"
    assert r.env_state == {"phase": "plan"}
    assert r.metadata == {"_segment_id": "seg-1"}
    # Nodes and their indices round-trip too.
    assert [n.node_id for n in r.nodes] == ["n1", "n2"]
    assert loaded.get_node("n2").parent_node_id == "n1"


def test_from_dict_tolerates_missing_visit_count():
    """Old checkpoints written before visit_count serialization have no key;
    from_dict must default to 0 (the dataclass default)."""
    legacy = {
        "node_id": "sup-old",
        "agent_id": "planner",
        "issue_id": "iss-1",
        "task_id": "task-1",
    }
    restored = SuperNode.from_dict(legacy)
    assert restored.visit_count == 0

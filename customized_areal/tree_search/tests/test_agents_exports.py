# SPDX-License-Identifier: Apache-2.0
"""Tests for the public exports of the agents package."""

from __future__ import annotations


def test_agents_package_exports_supernode_api() -> None:
    import customized_areal.tree_search.agents as d

    for name in (
        "SuperNode",
        "ExecutionDAG",
        "EdgeType",
        "DAGError",
        "message_timeline",
        "dag_to_supernodes",
        "supernodes_to_dag",
        "SuperNodeAssembler",
        "SegmentSpec",
        "EdgeSpec",
        "TeamEnvSnapshot",
        "DagResult",
    ):
        assert name in d.__all__, f"{name} missing from __all__"
        assert hasattr(d, name), f"{name} not importable from package"


def test_agents_package_does_not_export_old_names() -> None:
    import customized_areal.tree_search.agents as d

    for old_name in ("Event", "AgentRunNode", "dag_to_events", "events_to_dag"):
        assert old_name not in d.__all__, f"{old_name} should be removed from __all__"

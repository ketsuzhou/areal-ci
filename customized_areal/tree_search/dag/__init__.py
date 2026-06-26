"""Cloud-only multi-agent DAG RL components for Multica.

Submodules:
- ``execution_dag``  : the agent-execution DAG model (torch-free).
- ``environment``    : the ``ForkableEnvironment`` abstraction + providers.
- ``verifier``       : verifier agent (objective checks + LLM-judge fallback).
- ``credit``         : per-agent / per-step credit assignment.
- ``backup``         : DAG-aware hybrid reward backup + advantages.
"""

from __future__ import annotations

from customized_areal.tree_search.dag.environment import (
    EnvironmentError,
    FleetSandboxProvider,
    ForkableEnvironment,
    ForkError,
    ForkResult,
    SnapshotError,
    SnapshotResult,
)
from customized_areal.tree_search.dag.execution_dag import (
    AgentRunNode,
    DAGError,
    Edge,
    EdgeType,
    ExecutionDAG,
)
from customized_areal.tree_search.dag.integration import (
    BranchCandidate,
    BranchMaterializationResult,
    BranchMaterializer,
    BranchStarter,
    MulticaIssueForker,
    cleanup_cloud_branch,
    finalize_with_verifier,
    materialize_cloud_branch,
)
from customized_areal.tree_search.dag.rl_session import (
    RLBridgeClient,
    RLSessionRewardWriter,
)
from customized_areal.tree_search.dag.verifier import (
    ObjectiveVerifier,
    Verifier,
    VerifierResult,
)

__all__ = [
    # execution_dag
    "AgentRunNode",
    "DAGError",
    "Edge",
    "EdgeType",
    "ExecutionDAG",
    # environment
    "EnvironmentError",
    "FleetSandboxProvider",
    "ForkableEnvironment",
    "ForkError",
    "ForkResult",
    "SnapshotError",
    "SnapshotResult",
    # verifier (Phase 2 slice)
    "ObjectiveVerifier",
    "Verifier",
    "VerifierResult",
    # rl_session
    "RLBridgeClient",
    "RLSessionRewardWriter",
    # integration
    "BranchCandidate",
    "BranchMaterializationResult",
    "BranchMaterializer",
    "BranchStarter",
    "MulticaIssueForker",
    "cleanup_cloud_branch",
    "finalize_with_verifier",
    "materialize_cloud_branch",
]

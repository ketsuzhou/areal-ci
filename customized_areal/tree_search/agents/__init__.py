"""Cloud-only multi-agent DAG RL components for Multica.

Submodules:
- ``execution_dag``     : the agent-execution DAG model (torch-free).
- ``environment``       : the ``ForkableEnvironment`` abstraction + providers.
- ``agentic_verifier`` : pi-agent verifier driver (Phase 2 -- replaces the
  in-process Python verifier; assigns per-session reward via the gateway).
- ``harvest``           : verifier-driven finalize + trajectory harvest.
- ``verifier``          : legacy objective verifier slice (superseded).
"""

from __future__ import annotations

from customized_areal.tree_search.agents.agentic_verifier import (
    AgenticVerifier,
    PiVerifierLauncher,
    VerifierReward,
    VerifierRun,
    build_verifier_prompt,
    parse_verifier_output,
)
from customized_areal.tree_search.agents.branch_selection import (
    BranchPoint,
    lane_successor_value,
    passes_gate,
    select_branch_points,
    td_error,
)
from customized_areal.tree_search.agents.critic_observation import (
    DEFAULT_CRITIC_FIELDS,
    CriticObservation,
    build_critic_observations,
    build_observation_after_turn,
)
from customized_areal.tree_search.agents.dag_advantage import (
    AssembledAdvantages,
    assemble_node_advantages,
    explained_variance,
)
from customized_areal.tree_search.agents.environment import (
    EnvironmentError,
    FleetSandboxProvider,
    ForkableEnvironment,
    ForkError,
    ForkResult,
    SnapshotError,
    SnapshotResult,
)
from customized_areal.tree_search.agents.event_codec import (
    ReplayPrefix,
    dag_to_events,
    events_to_dag,
    replay_prefix_for,
)
from customized_areal.tree_search.agents.event_model import (
    Event,
    message_timeline,
)
from customized_areal.tree_search.agents.execution_dag import (
    AgentRunNode,
    DAGError,
    Edge,
    EdgeType,
    ExecutionDAG,
)
from customized_areal.tree_search.agents.gae import (
    GlobalEvent,
    NodeGAEResult,
    compute_global_gae,
    events_from_nodes,
)
from customized_areal.tree_search.agents.harvest import (
    FinalizeResult,
    RewardWriter,
    TrajectoryHarvester,
    VerifierFinalizer,
)
from customized_areal.tree_search.agents.integration import (
    BranchCandidate,
    BranchMaterializationResult,
    BranchMaterializer,
    BranchStarter,
    MulticaIssueForker,
    cleanup_cloud_branch,
    finalize_with_verifier,
    materialize_cloud_branch,
)
from customized_areal.tree_search.agents.rl_session import (
    RLBridgeClient,
    RLSessionRewardWriter,
)
from customized_areal.tree_search.agents.verifier import (
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
    # agentic verifier (Phase 2 -- pi-agent verifier)
    "AgenticVerifier",
    "PiVerifierLauncher",
    "VerifierReward",
    "VerifierRun",
    "build_verifier_prompt",
    "parse_verifier_output",
    # harvest (verifier-driven finalize + trajectory harvest)
    "FinalizeResult",
    "RewardWriter",
    "TrajectoryHarvester",
    "VerifierFinalizer",
    # critic observation (Phase 3 -- global joint-state frontier)
    "DEFAULT_CRITIC_FIELDS",
    "CriticObservation",
    "build_critic_observations",
    "build_observation_after_turn",
    # event codec (DAG <-> linear trajectory)
    "Event",
    "message_timeline",
    "dag_to_events",
    "events_to_dag",
    "replay_prefix_for",
    "ReplayPrefix",
    # gae (Phase 3 -- global joint-state GAE)
    "GlobalEvent",
    "NodeGAEResult",
    "compute_global_gae",
    "events_from_nodes",
    # branch selection (branch-point selection policy over the Event sequence)
    "BranchPoint",
    "lane_successor_value",
    "passes_gate",
    "select_branch_points",
    "td_error",
    # dag advantage assembler (Phase 3 -- GAE replaces GRPO for DAG runs)
    "AssembledAdvantages",
    "assemble_node_advantages",
    "explained_variance",
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

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
    MulticaSweLegoProvider,
    SnapshotError,
    SnapshotResult,
)
from customized_areal.tree_search.agents.event_codec import (
    dag_to_supernodes,
    supernodes_to_dag,
)
from customized_areal.tree_search.agents.event_model import (
    EdgeRef,
    message_timeline,
)
from customized_areal.tree_search.agents.execution_dag import (
    DAGError,
    Edge,
    EdgeType,
    ExecutionDAG,
    SuperNode,
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
    finalize_with_verifier,
)
from customized_areal.tree_search.agents.rl_session import (
    RLBridgeClient,
    RLSessionRewardWriter,
)
from customized_areal.tree_search.agents.supernode_assembler import (
    DagResult,
    EdgeSpec,
    SegmentSpec,
    SuperNodeAssembler,
    TeamEnvSnapshot,
)
from customized_areal.tree_search.agents.verifier import (
    ObjectiveVerifier,
    Verifier,
    VerifierResult,
)


def __getattr__(name: str):
    """Load the training workflow only for callers that request it.

    The non-training MultiCA clients live in this package too, but do not need
    AReaL's full rollout runtime. Keeping this import lazy makes those client
    modules usable in lightweight tooling and contract tests.
    """
    if name == "MultiAgentEnvDispatchWorkflow":
        from customized_areal.tree_search.agents.multi_agent_workflow import (
            MultiAgentEnvDispatchWorkflow,
        )

        return MultiAgentEnvDispatchWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # execution_dag
    "SuperNode",
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
    "MulticaSweLegoProvider",
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
    "EdgeRef",
    "message_timeline",
    "dag_to_supernodes",
    "supernodes_to_dag",
    # supernode assembler (Phase 1a -- Multica segment specs -> SuperNodes)
    "SuperNodeAssembler",
    "SegmentSpec",
    "EdgeSpec",
    "TeamEnvSnapshot",
    "DagResult",
    # gae (Phase 3 -- global joint-state GAE)
    "GlobalEvent",
    "NodeGAEResult",
    "compute_global_gae",
    "events_from_nodes",
    # dag advantage assembler (Phase 3 -- GAE replaces GRPO for DAG runs)
    "AssembledAdvantages",
    "assemble_node_advantages",
    "explained_variance",
    # rl_session
    "RLBridgeClient",
    "RLSessionRewardWriter",
    # integration
    "finalize_with_verifier",
    # multi-agent env dispatch workflow (Phase 1 -- thin RolloutWorkflow
    # orchestrator over the v2-segment-dag components)
    "MultiAgentEnvDispatchWorkflow",
]

"""Configuration for TPFC Agent training experiments."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports when this file is imported directly
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from customized_areal.tree_search.config import TreeBackupConfig

from areal.api.cli_args import PPOConfig


@dataclass
class AgentConfig:
    """Configuration for the TPFC agent."""

    trial_name: str = field(
        default="",
        metadata={"help": "Trial name for the agent."},
    )
    train_id: str = field(
        default="",
        metadata={"help": "Training run identifier."},
    )
    user_id: str = field(
        default="",
        metadata={"help": "User identifier."},
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "Name of the model to use."},
    )
    judge_model_name: str | None = field(
        default=None,
        metadata={"help": "Name of the judge model."},
    )
    judge_base_url: str | None = field(
        default=None,
        metadata={"help": "Base URL for the judge API."},
    )
    judge_api_key: str | None = field(
        default=None,
        metadata={"help": "API key for the judge model."},
    )


@dataclass
class TPFCConfig(PPOConfig):
    """Configuration for TPFC Agent training experiments.

    Extends PPOConfig with agent-specific settings for the TPFC agent
    workflow using the OpenAI-compatible proxy approach.
    """

    workflow: str = field(
        default="customized_areal.tpfc_agent.TPFCAgent",
        metadata={"help": "Path to the TPFC workflow class for training."},
    )
    eval_workflow: str = field(
        default="${workflow}",
        metadata={"help": "Path to the TPFC workflow class for evaluation."},
    )
    agent: AgentConfig = field(default_factory=AgentConfig)
    tree_search: TreeBackupConfig = field(
        default_factory=TreeBackupConfig,
        metadata={
            "help": "Tree search configuration (MCTS, cache, loss, teacher, diagnose)."
        },
    )
    assistant_marker: str = field(
        default="",
        metadata={"help": "Marker string identifying assistant turns in tree backup."},
    )

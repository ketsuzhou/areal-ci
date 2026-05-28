"""Configuration for On-Policy Distillation training experiments."""

# ruff: noqa: E402

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports when this file is imported directly
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from areal.api.cli_args import PPOConfig


@dataclass
class AgentConfig:
    """Configuration for the agent used by OpenAIProxyWorkflow.

    Attributes:
        trial_name: AReaL trial name to tag agent runs with.
        train_id: Training run ID to tag agent runs with.
        user_id: User ID for authentication.
        model_name: Model name for LLM calls. None uses the agent default.
        judge_model_name: Model name for the judge LLM.
        judge_base_url: Base URL for the judge LLM API.
        judge_api_key: API key for the judge LLM.
    """

    trial_name: str = field(
        default="",
        metadata={"help": "AReaL trial name to tag agent runs with."},
    )
    train_id: str = field(
        default="",
        metadata={"help": "Training run ID to tag agent runs with."},
    )
    user_id: str = field(
        default="",
        metadata={"help": "User ID for authentication."},
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "Model name for LLM calls. None uses the agent default."},
    )
    judge_model_name: str | None = field(
        default=None,
        metadata={"help": "Model name for the judge LLM."},
    )
    judge_base_url: str | None = field(
        default=None,
        metadata={"help": "Base URL for the judge LLM API."},
    )
    judge_api_key: str | None = field(
        default=None,
        metadata={"help": "API key for the judge LLM."},
    )


@dataclass
class OnPolicyDistillConfig(PPOConfig):
    """Configuration for On-Policy Distillation training experiments.

    Extends PPOConfig with distillation-specific settings for training
    using OpenAI proxy workflow components.

    Attributes:
        workflow: Path to the workflow class (OpenAI proxy workflow).
        eval_workflow: Path to the eval workflow class.
        cache_size: Maximum size of the interaction cache.
        proxy_base_url: Base URL for the OpenAI proxy.
        proxy_api_key: API key for the OpenAI proxy.
        proxy_model: Model name to use for generation.
        proxy_temperature: Sampling temperature.
        proxy_max_tokens: Maximum tokens per completion.
        turn_discount: Discount factor for multi-turn rewards.
        export_style: Style for exporting interactions ('concat' or 'individual').
    """

    workflow: str = field(
        default="customized_areal.tree_search.OpenAIProxyWorkflow",
        metadata={"help": "Path to the OpenAI proxy workflow class."},
    )
    eval_workflow: str = field(
        default="${workflow}",
        metadata={"help": "Path to the eval workflow class."},
    )

    agent: AgentConfig = field(
        default_factory=AgentConfig,
        metadata={"help": "Agent configuration passed to the workflow."},
    )

    # Cache configuration
    cache_size: int = field(
        default=1000,
        metadata={"help": "Maximum number of interactions to cache."},
    )

    # OpenAI Proxy Client configuration
    proxy_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL for the OpenAI proxy server."},
    )
    proxy_api_key: str = field(
        default="",
        metadata={"help": "API key for the OpenAI proxy server."},
    )
    proxy_model: str = field(
        default="qwen/qwen3-1.7b",
        metadata={"help": "Model name to use for generation."},
    )
    proxy_temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for generation."},
    )
    proxy_max_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum tokens per completion."},
    )
    proxy_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling parameter."},
    )

    # Workflow configuration
    turn_discount: float = field(
        default=1.0,
        metadata={"help": "Discount factor for multi-turn reward propagation."},
    )
    export_style: str = field(
        default="individual",
        metadata={"help": "Export style: 'concat' or 'individual'."},
    )

    # Distillation-specific settings
    use_reward_scaling: bool = field(
        default=True,
        metadata={"help": "Whether to apply reward scaling."},
    )
    reward_scaling_factor: float = field(
        default=10.0,
        metadata={"help": "Factor for reward scaling."},
    )
    reward_bias: float = field(
        default=-0.5,
        metadata={"help": "Bias to add to rewards."},
    )

    # Tree search distilling configuration
    cache_dir: str = field(
        default="",
        metadata={"help": "Directory for rollout cache and MCTS tree checkpoints."},
    )
    assistant_marker: str = field(
        default="",
        metadata={"help": "Marker string for assistant turns. Auto-detected if empty."},
    )
    student_top_k: int = field(
        default=10,
        metadata={
            "help": "Number of top candidate tokens per position for student logprob gathering."
        },
    )

    # Teacher model configuration
    teacher_base_url: str = field(
        default="http://localhost:8001",
        metadata={"help": "Base URL for the teacher model inference API."},
    )
    teacher_model_name: str = field(
        default="",
        metadata={
            "help": "Teacher model name for the inference API. Required for teacher distillation."
        },
    )
    teacher_top_k: int = field(
        default=10,
        metadata={"help": "Number of top candidate tokens to evaluate per position."},
    )
    teacher_max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for teacher API calls."},
    )
    teacher_timeout: float = field(
        default=60.0,
        metadata={"help": "Request timeout in seconds for teacher API calls."},
    )
    teacher_missing_logprob: float = field(
        default=-23.0,
        metadata={
            "help": "Default logprob for tokens not in teacher's top-k (log(1e-10) ≈ -23.0)."
        },
    )

"""
On-Policy Distillation Agent for AReaL integration.

This module provides a class-based agent interface consistent with AReaL's
agentic RL training pattern, using the run_backend function from tpfc.
"""

from typing import Any

from customized_areal.tpfc.backend_run import run_backend
from customized_areal.tree_search.core.reward_compute import (
    _compute_token_rewards,
)
from customized_areal.tree_search.core.teacher_client import (
    TeacherClient,
    TeacherConfig,
)
from customized_areal.tree_search.distill_types import PositionRewardInfo

from areal.api import AsyncRewardWrapper
from areal.utils import logging

logger = logging.getLogger("OnPolicyDistillAgent")


def accuracy_reward(completion: str | list[dict], ground_truth: str) -> float:
    """Simple accuracy reward function.

    Args:
        completion: The model's completion (string or message list).
        ground_truth: The expected ground truth answer.

    Returns:
        Float reward between 0.0 and 1.0.
    """
    if isinstance(completion, list):
        texts = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                texts.append(msg["content"])
        completion_text = " ".join(texts)
    else:
        completion_text = str(completion)

    completion_norm = completion_text.strip().lower()
    ground_truth_norm = str(ground_truth).strip().lower()

    if completion_norm == ground_truth_norm:
        return 1.0

    if ground_truth_norm in completion_norm or completion_norm in ground_truth_norm:
        return 0.5

    return 0.0


def on_policy_distill_reward_fn(
    completions: list[dict[str, Any]], gt: str = ""
) -> float:
    """Reward function for on-policy distillation.

    Args:
        completions: List of completion dictionaries.
        gt: Ground truth answer.

    Returns:
        Float reward between 0.0 and 1.0.
    """
    if not completions:
        return 0.0

    last_completion = completions[-1]
    completion_text = last_completion.get("content", "")

    return accuracy_reward(completion_text, gt)


class OnPolicyDistillAgent:
    """On-Policy Distillation Agent for AReaL integration.

    This class wraps the run_backend function in a class-based structure
    compatible with AReaL's RolloutWorkflow and PPOTrainer.

    Attributes:
        default_agent_id: Default agent ID for agent runs.
    """

    default_agent_id: str = "8bba75cb-0d87-4efe-b566-87de77335b76"

    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        teacher_config: TeacherConfig | None = None,
        **kwargs,
    ):
        """Initialize OnPolicyDistillAgent.

        Args:
            agent_id: Optional agent ID to use. Falls back to default_agent_id.
            user_id: Optional user ID for authentication.
            model_name: Optional model name for LLM calls.
            teacher_config: Optional TeacherConfig for teacher distillation.
            **kwargs: Additional configuration options (ignored but accepted).
        """
        self.agent_id = agent_id or self.default_agent_id
        self.user_id = user_id
        self.model_name = model_name
        self.teacher_client = TeacherClient(teacher_config) if teacher_config else None

        logger.info(
            "OnPolicyDistillAgent initialized: agent_id=%s, model_name=%s, teacher=%s",
            self.agent_id,
            self.model_name,
            "enabled" if self.teacher_client else "disabled",
        )

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs,
    ) -> float | dict[str, float]:
        """Execute a single agent run and return the reward.

        This method is compatible with AReaL's OpenAIProxyWorkflow interface.
        The http_client from extra_kwargs is used to route LLM calls through
        AReaL's proxy server for token-level tracking.

        Args:
            data: Input data for the agent. Expected keys:
                - "messages": List of message dicts (last message contains the task)
                - "answer": Ground truth for reward calculation
            **extra_kwargs: Additional keyword arguments passed by OpenAIProxyWorkflow:
                - http_client: httpx.AsyncClient for proxy routing
                - base_url: Proxy server base URL
                - api_key: Session API key

        Returns:
            Float reward value (0.0 to 1.0).
        """
        # Extract task description from messages
        messages = data.get("messages", [])
        if messages:
            task_description = messages[-1].get("content", "")
        else:
            task_description = data.get("prompt", "")

        # Extract ground truth for for reward calculation
        gt = data.get("answer", "")

        # Get OpenAI proxy parameters (passed by OpenAIProxyWorkflow._run_agent)
        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")
        api_key = extra_kwargs.get("api_key")

        logger.info(
            "OnPolicyDistillAgent starting run: task=%s, agent_id=%s, has_ground_truth=%s, base_url=%s",
            task_description[:100] if task_description else None,
            self.agent_id,
            bool(gt),
            base_url,
        )

        try:
            # Execute the backend run
            run_result = await run_backend(
                task_description=task_description,
                task_file_path=[],
                log_path="./log.json",
                task_id="",
                gt=gt,
                tags=[],
                user_id=self.user_id,
                model_name=self.model_name,
                agent_id=self.agent_id,
                base_url=base_url,
                http_client=http_client,
                api_key=api_key,
                rebuild_llm_client=True,
            )
            completion_messages = run_result.messages

            # Compute position-level rewards using teacher model
            position_rewards: list[PositionRewardInfo] = []
            completion_id: str | None = None

            if completion_messages and self.teacher_client is not None:
                proxy_client = extra_kwargs.get("proxy_client")

                if proxy_client is not None:
                    try:
                        interaction = await proxy_client.get_last_interaction()
                        if (
                            interaction
                            and hasattr(interaction, "model_response")
                            and interaction.model_response is not None
                        ):
                            student_output_ids = (
                                interaction.model_response.output_tokens
                            )
                            student_input_ids = interaction.model_response.input_tokens
                            student_top_k_logprobs = getattr(
                                interaction.model_response, "output_top_logprobs", None
                            )

                            if student_top_k_logprobs is not None:
                                position_rewards = await _compute_token_rewards(
                                    student_output_ids=student_output_ids,
                                    student_input_ids=student_input_ids,
                                    student_top_k_logprobs=student_top_k_logprobs,
                                    teacher_client=self.teacher_client,
                                    top_k=self.teacher_client.config.teacher_top_k,
                                )
                                completion_id = interaction.interaction_id
                                logger.info(
                                    "Computed position rewards via teacher: %d positions",
                                    len(position_rewards),
                                )
                    except Exception as e:
                        logger.warning(
                            "Failed to compute position rewards via teacher: %s", e
                        )

            # Calculate reward using reward function
            reward_fn = AsyncRewardWrapper(on_policy_distill_reward_fn)
            reward = await reward_fn(completions=completion_messages, gt=gt)

            logger.info(
                "OnPolicyDistillAgent run completed: message_count=%d, reward=%.4f",
                len(completion_messages),
                reward,
            )

            if completion_id is not None:
                return {completion_id: reward}
            return reward

        except Exception as e:
            logger.error(f"OnPolicyDistillAgent run failed: {e}")
            raise

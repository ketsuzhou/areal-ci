"""
TPFC Agent wrapper for AReaL integration.

This module provides a class-based agent interface consistent with AReaL's
agentic RL training pattern, wrapping the existing run_backend functionality.
"""

import os
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from customized_areal.tpfc.backend_run import run_backend
from customized_areal.tpfc.gaia_final_reward import compute_reward

from areal.utils import logging

logger = logging.getLogger("TPFCAgent")


class TPFCAgent:
    """
    TPFC Agent for AReaL integration.

    This class wraps the run_backend function in a class-based structure
    compatible with AReal's RolloutWorkflow and PPOTrainer.

    Usage:
        agent = TPFCAgent()
        reward = await agent.run(data={"prompt": "task description", "answer": "ground_truth"}, **extra_kwargs)

    """

    def __init__(
        self,
        user_id: str | None = None,
        train_id: str | None = None,
        trial_name: str | None = None,
        model_name: str | None = None,
        judge_model_name: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize TPFC Agent.

        Args:
            user_id: Optional user ID for authentication.
            train_id: Optional training run ID to tag agent runs with.
            trial_name: Optional AReaL trial name to tag agent runs with.
            model_name: Optional model name for LLM calls.
            judge_model_name: Model name for the judge LLM (default: gpt-4o).
            judge_base_url: Base URL for the judge LLM API.
            judge_api_key: API key for the judge LLM.
            **kwargs: Additional configuration options (ignored but accepted for compatibility).
        """
        self.user_id = user_id
        self.train_id = train_id
        self.trial_name = trial_name
        self.model_name = model_name
        self.judge_model_name = judge_model_name
        self.judge_base_url = judge_base_url
        self.judge_api_key = judge_api_key

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs,
    ) -> float:
        """
        Execute a single agent run and return the reward.

        This method is compatible with AReaL's OpenAIProxyWorkflow interface.
        The http_client from extra_kwargs is used to route LLM calls through
        AReaL's proxy server for token-level tracking.

        The ``query_id`` from ``data`` is logged and preserved in the data
        dict so that ``QueryIDProxyWorkflow`` can inject it into the
        trajectory as ``query_id`` for tree search.

        Args:
            data: Input data for the agent. Expected keys:
                - "messages": List of message dicts (last message contains the task)
                - "answer": Ground truth for reward calculation
                - "query_id": String identifier from the dataset
            **extra_kwargs: Additional keyword arguments passed by OpenAIProxyWorkflow:
                - http_client: httpx.AsyncClient for proxy routing
                - base_url: Proxy server base URL
                - api_key: Session API key

        Returns:
            Float reward value (0.0 to 1.0).

        Raises:
            RuntimeError: If agent run fails to start.
            TimeoutError: If agent run doesn't complete within timeout.
        """
        try:
            # Extract task description from dataset query field (clean query text)
            task_description = data.get("query", "")
            query_id = data.get("query_id", "")

            # Extract ground truth for reward calculation
            gt = data.get("answer", "")

            # Extract image paths from dataset (new files_path column)
            task_file_path = data.get("files_path", [])
            if task_file_path is None:
                task_file_path = []

            # Get OpenAI proxy parameters (passed by OpenAIProxyWorkflow._run_agent)
            base_url = extra_kwargs.get("base_url")
            api_key = extra_kwargs.get("api_key")

            logger.info(
                "TPFCAgent starting run: task=%s, query_id=%s, has_ground_truth=%s, base_url=%s, n_images=%d",
                task_description[:100] if task_description else None,
                query_id,
                bool(gt),
                base_url,
                len(task_file_path),
            )

            # Build tags for traceability
            tags = []
            if self.trial_name:
                tags.extend(self.trial_name.split("&"))
            if self.train_id:
                tags.append(f"train_id={self.train_id}")
            if self.user_id:
                tags.append(f"user_id={self.user_id}")

            # Execute the backend run
            completion_messages, _final_answer, _log_path, _trace = await run_backend(
                task_description=task_description,
                task_file_path=task_file_path,
                log_path="./log.json",
                task_id="",
                gt=gt,
                tags=tags,
                user_id=self.user_id,
                model_name="openrouter/qwen/qwen3-vl-8b-thinking",
                base_url=base_url,
                api_key=api_key,
            )

            # Extract response text from the last assistant message
            response_text = ""
            for msg in reversed(completion_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        response_text = "".join(
                            p.get("text", p.get("content", ""))
                            if isinstance(p, dict)
                            else str(p)
                            for p in content
                        )
                    elif isinstance(content, dict):
                        response_text = content.get("text", content.get("content", ""))
                    else:
                        response_text = content
                    break

            # Calculate reward using gaia_final_reward.compute_reward
            judge_model_name = self.judge_model_name or os.environ.get(
                "TPFC_JUDGE_MODEL", "qwen/qwen3.5-397b-a17b"
            )
            judge_base_url = os.environ.get(
                "WORKSPACE_OPENAI_API_BASE", "https://openrouter.ai/api/v1"
            )
            judge_api_key = os.environ.get("WORKSPACE_OPENAI_API_KEY", "OPENROUTER_API_KEY")

            try:
                result = compute_reward(
                    response_text=response_text,
                    ground_truth=gt,
                    user_query=task_description,
                    model_name=judge_model_name,
                    base_url=judge_base_url,
                    api_key=judge_api_key,
                )
                reward = float(result.get("answer_reward", 0))
            except Exception as exc:
                logger.warning(
                    "GAIA reward computation failed: %s\n%s",
                    exc,
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
                reward = 0.0

            logger.info(
                "TPFCAgent run completed: message_count=%d, query_id=%s, reward=%.4f",
                len(completion_messages),
                query_id,
                reward,
            )

            return float(reward)
        except Exception as exc:
            logger.warning(
                "TPFCAgent run failed: %s\n%s",
                exc,
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            return 0.0

"""Teacher provider interfaces for selected-turn distillation."""

from __future__ import annotations

import asyncio
import inspect
from typing import Protocol

import httpx
from openai import OpenAI

from customized_areal.tree_search.distilling.teacher_client import TeacherClient

from areal.utils import logging

logger = logging.getLogger("TeacherProvider")


class TeacherProvider(Protocol):
    """Interface for teacher diagnosis and token logprob providers."""

    async def diagnose_episode(
        self, conversation: list[dict[str, str]], gold_answer: str
    ) -> str:
        """Return a teacher diagnosis for an episode."""
        ...

    async def get_logprobs_for_prompt(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[list[float]]:
        """Return teacher logprobs aligned to candidate token IDs."""
        ...


class ExternalTeacherProvider:
    """Teacher provider backed by the OpenAI-compatible teacher client."""

    def __init__(
        self,
        client: TeacherClient,
        diagnose_model_name: str = "",
        diagnose_max_tokens: int = 1024,
        diagnose_temperature: float = 0.0,
        diagnose_base_url: str = "",
        diagnose_api_key: str = "",
    ) -> None:
        self.client = client
        self.diagnose_model_name = diagnose_model_name
        self.diagnose_max_tokens = diagnose_max_tokens
        self.diagnose_temperature = diagnose_temperature
        self.diagnose_base_url = diagnose_base_url
        self.diagnose_api_key = diagnose_api_key
        self._openai_client: OpenAI | None = None

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            http_client = httpx.Client(verify=False)
            self._openai_client = OpenAI(
                api_key=self.diagnose_api_key or "unused",
                base_url=self.diagnose_base_url,
                http_client=http_client,
            )
        return self._openai_client

    async def diagnose_episode(
        self,
        conversation: list[dict[str, str]],
        gold_answer: str,
        temperature: float | None = None,
    ) -> str:
        instruction = (
            "You are diagnosing a multi-turn assistant trajectory above. "
            "Analyze each assistant turn and identify which ones can be "
            "improved toward the gold answer. "
            "You MUST output ONLY XML (no extra text) wrapped in "
            "```xml fences. The XML must have a top-level <diagnosis> element "
            "containing a <turns> element. Each <turn> must include "
            "<turn_idx> (int start from 1), <should_improve> (true/false), "
            "and <guidance> (string). "
            "Only include turns that should be improved.\n\n"
            f"Gold answer: {gold_answer}\n\n"
            "Example response format:\n"
            "```xml\n"
            "<diagnosis>\n"
            "  <turns>\n"
            "    <turn>\n"
            "      <turn_idx>1</turn_idx>\n"
            "      <should_improve>true</should_improve>\n"
            "      <guidance>In turn 1, the assistant should have verified "
            "the URL before finalizing the answer.</guidance>\n"
            "    </turn>\n"
            "  </turns>\n"
            "</diagnosis>\n"
            "```"
        )
        if isinstance(conversation, str):
            conversation_messages = [{"role": "user", "content": conversation}]
        else:
            conversation_messages = list(conversation)
        messages = conversation_messages + [{"role": "user", "content": instruction}]
        temp = temperature if temperature is not None else self.diagnose_temperature
        if self.diagnose_base_url:
            client = self._get_openai_client()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.diagnose_model_name,
                    messages=messages,
                    max_tokens=self.diagnose_max_tokens,
                    temperature=temp,
                    extra_body={"enable_thinking": False},
                ),
            )
            content = response.choices[0].message.content
            if not content:
                finish_reason = response.choices[0].finish_reason
                raise RuntimeError(
                    f"Diagnose API returned empty content "
                    f"(finish_reason={finish_reason}, "
                    f"model={self.diagnose_model_name})"
                )
            logger.debug(
                "Diagnose API response (len=%d, first 300): %s",
                len(content),
                content[:300],
            )
            if len(content) > 600:
                logger.debug(
                    "Diagnose API response (last 300): %s",
                    content[-300:],
                )
            return content

        client_config = getattr(self.client, "config", None)
        if client_config is not None and getattr(client_config, "teacher_base_url", ""):
            return await self.client.chat_complete(
                messages=messages,
                model=self.diagnose_model_name or None,
                max_tokens=self.diagnose_max_tokens,
                temperature=temp,
            )
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return await self.client.complete_text(
            prompt,
            model=self.diagnose_model_name or None,
            max_tokens=self.diagnose_max_tokens,
            temperature=temp,
        )

    async def get_logprobs_for_prompt(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[list[float]]:
        logprob_maps = await self.client.get_logprobs_for_candidates(
            input_ids=prompt_ids,
            output_ids=generation_ids,
            candidate_token_ids=candidate_token_ids,
        )
        return [
            [position_map[token_id] for token_id in position_candidates]
            for position_map, position_candidates in zip(
                logprob_maps, candidate_token_ids, strict=True
            )
        ]


class EngineTeacherProvider:
    """Teacher provider backed by an in-process inference engine."""

    def __init__(self, engine) -> None:
        get_logprobs_for_prompt = getattr(engine, "get_logprobs_for_prompt", None)
        if not inspect.iscoroutinefunction(get_logprobs_for_prompt):
            raise NotImplementedError(
                "engine-backed teacher provider requires async coroutine function "
                "engine.get_logprobs_for_prompt"
            )
        self.engine = engine

    async def diagnose_episode(
        self, conversation: list[dict[str, str]], gold_answer: str
    ) -> str:
        diagnose_episode = getattr(self.engine, "diagnose_episode", None)
        if not inspect.iscoroutinefunction(diagnose_episode):
            raise NotImplementedError(
                "engine-backed teacher provider requires async coroutine function "
                "engine.diagnose_episode"
            )
        context = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
        return await diagnose_episode(
            context=context,
            gold_answer=gold_answer,
        )

    async def get_logprobs_for_prompt(
        self,
        prompt_ids: list[int],
        generation_ids: list[int],
        candidate_token_ids: list[list[int]],
    ) -> list[list[float]]:
        return await self.engine.get_logprobs_for_prompt(
            prompt_ids=prompt_ids,
            generation_ids=generation_ids,
            candidate_token_ids=candidate_token_ids,
        )


__all__ = [
    "EngineTeacherProvider",
    "ExternalTeacherProvider",
    "TeacherProvider",
]

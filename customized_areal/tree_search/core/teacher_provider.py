"""Teacher provider interfaces for selected-turn distillation."""

from __future__ import annotations

from typing import Protocol

from customized_areal.tree_search.core.teacher_client import TeacherClient


class TeacherProvider(Protocol):
    """Interface for teacher diagnosis and token logprob providers."""

    async def diagnose_episode(self, context: str, gold_answer: str) -> str:
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
    ) -> None:
        self.client = client
        self.diagnose_model_name = diagnose_model_name
        self.diagnose_max_tokens = diagnose_max_tokens
        self.diagnose_temperature = diagnose_temperature

    async def diagnose_episode(self, context: str, gold_answer: str) -> str:
        prompt = (
            "You are diagnosing a multi-turn assistant trajectory. "
            "Return strict JSON with a top-level key 'turns'. Each turn must "
            "include 'turn_idx', 'should_improve', and 'guidance'. Select only "
            "turns where the assistant generation can be improved toward the "
            "gold answer.\n\n"
            f"Gold answer:\n{gold_answer}\n\n"
            f"Episode context:\n{context}\n"
        )
        return await self.client.complete_text(
            prompt,
            model=self.diagnose_model_name or None,
            max_tokens=self.diagnose_max_tokens,
            temperature=self.diagnose_temperature,
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
        if not callable(getattr(engine, "get_logprobs_for_prompt", None)):
            raise NotImplementedError(
                "engine-backed teacher provider requires engine.get_logprobs_for_prompt"
            )
        self.engine = engine

    async def diagnose_episode(self, context: str, gold_answer: str) -> str:
        diagnose_episode = getattr(self.engine, "diagnose_episode", None)
        if not callable(diagnose_episode):
            raise NotImplementedError(
                "engine-backed teacher provider requires engine.diagnose_episode"
            )
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

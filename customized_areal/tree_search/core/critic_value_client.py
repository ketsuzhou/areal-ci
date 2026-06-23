# SPDX-License-Identifier: Apache-2.0
# customized_areal/tree_search/core/critic_value_client.py
"""Generative-critic value computation during rollout (shared-model mode).

The critic reuses the actor's model served on the same inference engine
(``self.rollout``). For each turn (``Node``) of an episode it builds a critic
prompt -- the conversation through that turn plus a success-probability
instruction -- and asks the model for the answer-position distribution over the
integer labels ``0..score_max``. The soft expected value ``Sum_i p_i*(i/score_max)``
becomes the state value ``v_phi(s_t)`` written onto ``Node.value``.

Logprob extraction seam
-----------------------
Exact per-candidate logprobs at the answer position depend on the inference
stack exposing top-k logprobs (e.g. SGLang ``top_logprobs_num``). AReaL's default
``agenerate`` wrapper returns only chosen-token logprobs, so to keep this module
functional and engine-agnostic the candidate-logprob query is isolated behind
``logprob_query_fn(engine, prompt_ids) -> dict[int, float]``:

* When ``logprob_query_fn`` is provided, its returned ``{label: logprob}`` mapping
  is fed to :func:`expected_value_from_logprobs` to produce a genuinely *soft*
  rollout value. This is the intended production path once a top-k logprob query
  is wired to the serving stack.
* Otherwise the default greedily generates the critic's answer and parses the
  trailing integer, yielding a degenerate one-hot distribution (``value =
  label/score_max``). The critic *training* objective (see ``critic_loss``) always
  uses the true soft expected value computed from train-engine logits, so the
  soft-regression target is unaffected by this rollout-time fallback.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from customized_areal.tree_search.core.critic_prompt import (
    build_critic_instruction,
    build_critic_messages,
    digit_token_ids,
    expected_value_from_logprobs,
    reconstruct_messages_through_turn,
    variance_from_logprobs,
)

LogprobQueryFn = Callable[[Any, list[int]], Awaitable[dict[int, float]]]


class CriticValueClient:
    def __init__(
        self,
        tokenizer: Any,
        *,
        score_max: int = 10,
        avg_success_rate: float = 0.29,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        instruction_role: str = "user",
        system_prompt: str | None = None,
        logprob_query_fn: LogprobQueryFn | None = None,
    ) -> None:
        if score_max < 1:
            raise ValueError(f"score_max must be >= 1, got {score_max}")
        self.tokenizer = tokenizer
        self.score_max = score_max
        self.avg_success_rate = avg_success_rate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.instruction_role = instruction_role
        self.system_prompt = system_prompt
        self.logprob_query_fn = logprob_query_fn
        self.instruction = build_critic_instruction(avg_success_rate, score_max)
        # Used by tooling/diagnostics; not required for the default parse path.
        self._digit_ids = digit_token_ids(tokenizer, score_max)

    # -- prompt construction ------------------------------------------------
    def build_prompt_ids(self, node: Any) -> list[int]:
        messages = reconstruct_messages_through_turn(
            node, self.tokenizer, self.system_prompt
        )
        messages = build_critic_messages(
            messages, self.instruction, self.instruction_role
        )
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

    # -- engine interaction (isolated seam) ---------------------------------
    async def _query_label_logprobs(
        self, engine: Any, prompt_ids: list[int]
    ) -> dict[int, float]:
        if self.logprob_query_fn is not None:
            return await self.logprob_query_fn(engine, prompt_ids)
        return await self._default_query(engine, prompt_ids)

    async def _default_query(
        self, engine: Any, prompt_ids: list[int]
    ) -> dict[int, float]:
        # Imported lazily so this module stays importable without the heavy
        # areal runtime (e.g. in unit tests).
        from areal.api.cli_args import GenerationHyperparameters
        from areal.api.io_struct import ModelRequest

        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            greedy=(self.temperature == 0.0),
        )
        req = ModelRequest(input_ids=list(prompt_ids), gconfig=gconfig)
        resp = await engine.agenerate(req)
        label = self._parse_label_from_tokens(resp.output_tokens)
        if label is None:
            return {}
        return {label: 0.0}  # one-hot fallback -> value = label / score_max

    def _parse_label_from_tokens(self, output_tokens: list[int]) -> int | None:
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        matches = re.findall(r"\d+", text)
        for m in reversed(matches):
            v = int(m)
            if 0 <= v <= self.score_max:
                return v
        return None

    # -- value computation --------------------------------------------------
    async def compute_value_and_variance(
        self, engine: Any, node: Any
    ) -> tuple[float, float]:
        """Return ``(v_phi(s_t), var_theta(s_t))`` from one label-logprob query.

        ``var_theta`` is the categorical variance of the critic's score
        distribution; it is ``0.0`` on the one-hot fallback path (no soft
        logprobs available), in which case the consumer applies a variance
        floor.
        """
        prompt_ids = self.build_prompt_ids(node)
        label_logprobs = await self._query_label_logprobs(engine, prompt_ids)
        value = expected_value_from_logprobs(label_logprobs, self.score_max)
        variance = variance_from_logprobs(label_logprobs, self.score_max)
        return value, variance

    async def compute_value(self, engine: Any, node: Any) -> float:
        value, _variance = await self.compute_value_and_variance(engine, node)
        return value

    async def annotate_episode(
        self,
        engine: Any,
        nodes: list[Any],
        tree_store: Any | None = None,
    ) -> None:
        """Compute and store v_phi(s_t) for each node of an episode."""
        for node in nodes:
            value, variance = await self.compute_value_and_variance(engine, node)
            node.value = float(value)
            node.value_variance = float(variance)
            node_id = getattr(node, "node_id", None)
            if tree_store is not None and node_id:
                tree_store.set_value(node_id, float(value))
                tree_store.set_value_variance(node_id, float(variance))

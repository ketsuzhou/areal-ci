# SPDX-License-Identifier: Apache-2.0
"""CPU end-to-end smoke test for the VIMPO training pipeline (Task 8, 6.2).

Covers the full offline path with fake FSDP/SGLang components:

    Config -> workflow episode metadata (annotate_vimpo_episode_metadata)
    -> tensorization (_nodes_to_batched_tensor_dict) -> actor candidate
    snapshot (real vimpo_candidate_stats_from_logits on a tiny CPU model)
    -> deterministic frozen-reference scoring -> VIMPOAdvantageComputer
    -> episode-atomic minibatch split -> combined VIMPO loss
    (real vimpo_loss_terms + vimpo_loss_fn) -> one optimizer step.

The test asserts the pipeline updates ONLY the actor: the tiny model's
parameters change while the frozen reference's identity and version are
untouched, and every scalar in ``actor.last_vimpo_metrics`` is finite.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from customized_areal.tree_search.config import Config
from customized_areal.tree_search.core.advantage import VIMPOAdvantageComputer
from customized_areal.tree_search.core.customized_grouped_workflow import (
    _nodes_to_batched_tensor_dict,
    annotate_vimpo_episode_metadata,
)
from customized_areal.tree_search.core.tree_store import Node
from customized_areal.tree_search.engine.fsdp_engine import (
    vimpo_candidate_stats_from_logits,
)
from customized_areal.tree_search.training.actor import (
    VIMPO_ACTOR_CONFIG_FIELDS,
    VIMPOFSDPPPOActor,
)
from customized_areal.tree_search.training.losses.vimpo import (
    vimpo_loss_fn,
    vimpo_loss_terms,
)
from customized_areal.tree_search.training.vimpo_reference import ReferenceScore

_VOCAB_SIZE = 16
_HIDDEN = 8

# Scalar metric names the smoke test expects in actor.last_vimpo_metrics.
_EXPECTED_SCALAR_METRICS = {
    "vimpo/reference_latency_ms",
    "vimpo/reference_retries",
    "vimpo/effective_top_k",
    "vimpo/exact_kl",
    "vimpo/snapshot_policy_version",
    "vimpo/ppo_actor_loss",
    "vimpo/value_loss",
    "vimpo/combined_loss",
    "vimpo/terminal_rmse",
}


# =============================================================================
# Fixtures: nodes, tensorization, deterministic frozen reference, tiny actor
# =============================================================================


def _node(query: str, episode: str, turn: int, reward: float) -> Node:
    return Node(
        input_ids=[10, 11, 12],
        logprobs=[0.0, -0.2, -0.3],
        loss_mask=[0, 1, 1],
        versions=[0, 0, 0],
        outcome_reward=reward,
        node_id=f"{episode}-{turn}",
        query_id=query,
        episode_id=episode,
        turn_idx=turn,
    )


def _tensorize(nodes: list[Node]) -> dict[str, Any]:
    """Batch nodes exactly like the VIMPO branch of ``_finalize_episode``."""
    batch = _nodes_to_batched_tensor_dict(nodes, advantage_mode="vimpo")
    assert batch is not None
    return batch


class _DeterministicReference:
    """Frozen reference scorer backed by a fixed random transition matrix.

    Log-probabilities are full-softmax normalized over the vocabulary and are
    a pure function of the prefix's last token, so scoring is deterministic
    and the reference identity/version never change (``pi_ref = pi_0``).
    """

    def __init__(self, vocab_size: int = _VOCAB_SIZE) -> None:
        generator = torch.Generator().manual_seed(20260722)
        self._transition = torch.randn(vocab_size, vocab_size, generator=generator)
        self._version = 0

    def identity_and_version(self) -> tuple[str, int]:
        return ("deterministic-frozen-reference", self._version)

    def validate_identity(self, expected: Any) -> None:
        pass

    def score(self, requests: list[Any]) -> list[ReferenceScore]:
        scores: list[ReferenceScore] = []
        for request in requests:
            logp = self._transition[request.prefix_ids[-1]].log_softmax(-1)
            scores.append(
                ReferenceScore(
                    key=request.key,
                    sampled_logp=float(logp[request.sampled_token_id]),
                    candidate_logp=[
                        float(logp[tid]) for tid in request.candidate_token_ids
                    ],
                )
            )
        return scores

    def close(self) -> None:
        pass


def _deterministic_reference() -> _DeterministicReference:
    return _DeterministicReference()


class _TinyActorModel(torch.nn.Module):
    """Tiny CPU actor: embedding + linear head producing full-vocab logits."""

    def __init__(self, vocab_size: int = _VOCAB_SIZE, hidden: int = _HIDDEN) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def _tiny_cpu_vimpo_actor(
    config: Config, *, frozen_reference: _DeterministicReference
) -> VIMPOFSDPPPOActor:
    """VIMPO actor with real orchestration/loss math and fake FSDP/SGLang parts.

    Built via ``__new__`` (no FSDP/GPU/process groups). The candidate snapshot
    and the per-minibatch train call are instance-level CPU implementations
    that reuse the REAL ``vimpo_candidate_stats_from_logits``,
    ``vimpo_loss_terms``, and ``vimpo_loss_fn`` against a tiny model with a
    real SGD optimizer; the SGLang reference is replaced by the deterministic
    frozen scorer. Everything in between (``_compute_vimpo_advantages``,
    ``_assemble_vimpo_batch``, ``_vimpo_update``, ``_validate_vimpo_batch``,
    ``split_episode_atomic_batches``, ``VIMPOAdvantageComputer``) is the
    production code path.
    """
    actor = VIMPOFSDPPPOActor.__new__(VIMPOFSDPPPOActor)
    actor.config = SimpleNamespace(
        **{field: getattr(config, field) for field in VIMPO_ACTOR_CONFIG_FIELDS},
        eps_clip=0.2,
        eps_clip_higher=None,
        ppo_n_minibatches=1,
        path="tiny-cpu-actor",
    )
    actor.reference_scorer = frozen_reference
    actor.vimpo_advantage = VIMPOAdvantageComputer(
        beta=config.vimpo_beta,
        gamma=config.vimpo_gamma,
        lam=config.vimpo_lambda,
        whiten=config.vimpo_whiten_advantages,
        dp_group=None,
    )
    actor.mp_group = None
    actor.dp_group = None
    actor.model = _TinyActorModel()
    actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.1)
    actor.last_vimpo_metrics = {}

    def _snapshot(data: dict[str, Any], *, top_k: int) -> Any:
        input_ids = data["input_ids"].long()
        with torch.no_grad():
            logits = actor.model(input_ids)
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        return vimpo_candidate_stats_from_logits(
            logits, labels, data["vimpo_predict_mask"].bool(), top_k=top_k
        )

    actor.compute_vimpo_candidate_stats = _snapshot  # type: ignore[method-assign]

    def _train_vimpo_batch_cpu(
        mb: dict[str, Any],
        *,
        actor_coeff: float,
        value_loss_weight: float,
        beta: float,
        eps_clip: float,
        eps_clip_higher: float | None = None,
    ) -> dict[str, float]:
        input_ids = mb["input_ids"].long()
        logits = actor.model(input_ids)
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        logp = logits.float().log_softmax(-1)
        logprobs = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        terms = vimpo_loss_terms(
            logprobs,
            mb,
            beta=beta,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
        )
        valid = terms.valid_token_count.to(torch.float64)
        episodes = terms.episode_count.to(torch.float64)
        loss = vimpo_loss_fn(
            terms,
            actor_coeff=actor_coeff,
            value_loss_weight=value_loss_weight,
            global_valid_tokens=valid,
            global_episodes=episodes,
            dp_size=1,
        )
        actor.optimizer.zero_grad()
        loss.backward()
        actor.optimizer.step()
        ppo_actor_loss = actor_coeff * float(terms.ppo_sum.detach()) / float(valid)
        value_loss = (
            value_loss_weight * float(terms.value_sum.detach()) / float(episodes)
        )
        return {
            "vimpo/ppo_actor_loss": ppo_actor_loss,
            "vimpo/value_loss": value_loss,
            "vimpo/combined_loss": ppo_actor_loss + value_loss,
            "vimpo/terminal_rmse": float(
                terms.terminal_residual.detach().float().square().mean().sqrt()
            ),
        }

    actor.train_vimpo_batch = _train_vimpo_batch_cpu  # type: ignore[method-assign]
    return actor


# =============================================================================
# Smoke test
# =============================================================================


def test_vimpo_cpu_pipeline_changes_actor_only() -> None:
    config = Config(
        advantage_mode="vimpo", vimpo_ref_base_url="http://fake", vimpo_top_k=3
    )
    nodes = [_node("q", "a", 1, 1.0), _node("q", "a", 2, 1.0), _node("q", "b", 1, 0.0)]
    annotate_vimpo_episode_metadata(nodes)
    batch = _tensorize(nodes)
    actor = _tiny_cpu_vimpo_actor(config, frozen_reference=_deterministic_reference())
    actor_before = [
        parameter.detach().clone() for parameter in actor.model.parameters()
    ]
    reference_before = actor.reference_scorer.identity_and_version()
    enriched = actor.compute_advantages([batch])
    actor.ppo_update(enriched)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.model.parameters())
    )
    assert actor.reference_scorer.identity_and_version() == reference_before
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in actor.last_vimpo_metrics.values()
    )


def test_vimpo_cpu_pipeline_emits_expected_scalar_metrics() -> None:
    """The observability surface stashes every expected scalar metric name."""
    config = Config(
        advantage_mode="vimpo", vimpo_ref_base_url="http://fake", vimpo_top_k=3
    )
    nodes = [_node("q", "a", 1, 1.0), _node("q", "a", 2, 1.0), _node("q", "b", 1, 0.0)]
    annotate_vimpo_episode_metadata(nodes)
    actor = _tiny_cpu_vimpo_actor(config, frozen_reference=_deterministic_reference())
    actor.ppo_update(actor.compute_advantages([_tensorize(nodes)]))
    assert _EXPECTED_SCALAR_METRICS <= set(actor.last_vimpo_metrics)
    # top_k=3 < vocab_size=16: truncated candidate KL, never flagged exact.
    assert actor.last_vimpo_metrics["vimpo/effective_top_k"] == 3.0
    assert actor.last_vimpo_metrics["vimpo/exact_kl"] == 0.0


def test_vimpo_cpu_pipeline_exports_distribution_metrics() -> None:
    """stats_tracker receives the per-position/per-episode distributions."""
    from areal.utils import stats_tracker

    config = Config(
        advantage_mode="vimpo", vimpo_ref_base_url="http://fake", vimpo_top_k=3
    )
    nodes = [_node("q", "a", 1, 1.0), _node("q", "a", 2, 1.0), _node("q", "b", 1, 0.0)]
    annotate_vimpo_episode_metadata(nodes)
    actor = _tiny_cpu_vimpo_actor(config, frozen_reference=_deterministic_reference())
    actor.ppo_update(actor.compute_advantages([_tensorize(nodes)]))
    exported = stats_tracker.export()
    for name in (
        "vimpo/candidate_kl",
        "vimpo/retained_mass",
        "vimpo/raw_advantage",
        "vimpo/normalized_advantage",
    ):
        assert f"{name}/avg" in exported, f"missing exported metric {name}/avg"

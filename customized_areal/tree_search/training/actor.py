"""Custom PPO Actor with distill-loss patching and multi-candidate support.

This module provides:
- MultiCandidateFSDPPPOActor: PPO actor using MultiCandidateFSDPEngine for
  on-policy distillation with multi-candidate logprob gathering.
- ClipCovFSDPPPOActor / ClipCovMegatronPPOActor: Actor wrappers that patch
  PPOActor with clip-cov loss before worker creation.
- MuonFSDPPPOActor / MuonMultiCandidateFSDPPPOActor: Actor wrappers that patch
  optimizer creation for the Muon optimizer before worker init.
- patch/unpatch functions to swap PPOActor._ppo_update with grpo_distill_loss_fn.
"""

from __future__ import annotations

import copy
import functools
from typing import Any

import torch
import torch.distributed as dist

from areal.api import Scheduler
from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging, stats_tracker
from areal.utils.data import batched_call, split_padded_tensor_dict_into_mb_list

from ..core.advantage import VIMPOAdvantageComputer
from ..engine.fsdp_engine import MultiCandidateFSDPEngine, VIMPOCandidateStats
from .vimpo_batching import split_episode_atomic_batches
from .vimpo_reference import (
    ReferenceIdentity,
    ReferenceScore,
    ReferenceScoreRequest,
    SGLangVIMPOReferenceScorer,
    VIMPOReferenceScorer,
)

logger = logging.getLogger("OnPolicyDistill")


def _patch_clip_cov_from_actor_config(config: Any) -> None:
    from customized_areal.clip_cov import (
        ClipCovConfig,
        patch_ppo_actor_to_use_clip_cov_loss,
    )

    patch_ppo_actor_to_use_clip_cov_loss(
        ClipCovConfig(
            clip_ratio=config.clip_cov_clip_ratio,
            clip_cov_lb=config.clip_cov_lb,
            clip_cov_ub=config.clip_cov_ub,
        )
    )


class ClipCovFSDPPPOActor:
    """FSDP actor wrapper that patches PPOActor before worker actor creation."""

    def __new__(cls, config):
        from areal.engine import FSDPPPOActor

        _patch_clip_cov_from_actor_config(config)
        return FSDPPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


class ClipCovMegatronPPOActor:
    """Megatron actor wrapper that patches PPOActor before worker actor creation."""

    def __new__(cls, config):
        from areal.engine import MegatronPPOActor

        _patch_clip_cov_from_actor_config(config)
        return MegatronPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


def _patch_muon_from_actor_config(config: Any) -> None:
    from customized_areal.optimizers import patch_fsdp_engine_for_muon

    patch_fsdp_engine_for_muon(
        momentum=config.muon_momentum,
        muon_adam_lr=config.muon_adam_lr,
        ns_steps=config.muon_ns_steps,
        nesterov=config.muon_nesterov,
    )


class MuonFSDPPPOActor:
    """FSDP actor wrapper that patches optimizer creation before worker init."""

    def __new__(cls, config):
        from areal.engine import FSDPPPOActor

        _patch_muon_from_actor_config(config)
        return FSDPPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


class MuonMultiCandidateFSDPPPOActor:
    """Multi-candidate FSDP actor wrapper with Muon optimizer patching."""

    def __new__(cls, config):
        from customized_areal.tree_search.engine import MultiCandidateFSDPPPOActor

        _patch_muon_from_actor_config(config)
        return MultiCandidateFSDPPPOActor(config)

    @classmethod
    def as_controller(cls, config, scheduler):
        from areal.trainer.ppo.actor import PPOActorController, PPOActorControllerV2

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


_patch_applied = False
_original_ppo_update = None


class MultiCandidateFSDPPPOActor(MultiCandidateFSDPEngine):
    """PPO Actor implementation using MultiCandidateFSDPEngine backend.

    This actor extends MultiCandidateFSDPEngine instead of standard FSDPEngine
    to enable multi-candidate logprob gathering for on-policy distillation.

    The key difference from FSDPPPOActor:
    - Uses MultiCandidateFSDPEngine as base class
    - Supports position-level rewards via _prepare_multi_candidate_labels
    - Computes logprobs for all candidate tokens, not just chosen tokens

    Example usage:
        actor = MultiCandidateFSDPPPOActor(config)
        actor.initialize(role="actor", ...)
        actor.ppo_update(batch)  # Uses grpo_distill_loss_fn
    """

    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

        # Patch PPOActor._ppo_update on the worker process to use
        # grpo_distill_loss_fn.  The controller-side patch in
        # CustomizedPPOTrainer.train() only affects the controller process;
        # workers receive ppo_update via RPC and run the original
        # _ppo_update → grpo_loss_fn, which crashes on teacher_logp shape
        # mismatch (response-aligned vs full-sequence-length).
        patch_ppo_actor_class_to_use_distill_loss()

        logger.info("MultiCandidateFSDPPPOActor initialized")

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> list[torch.Tensor] | None:
        """Compute log probabilities for given trajectories."""
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Compute advantages for given trajectories."""
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        """Perform PPO update on the given batch."""
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        """Create a controller for this actor class.

        This method is used by the trainer to create a controller
        that manages distributed workers.
        """
        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(
            train_engine=cls,
            config=config,
            scheduler=scheduler,
        )


# VIMPO actor config fields that CustomizedPPOTrainer copies from the tree-search
# Config onto the PPOActorConfig before constructing the dedicated VIMPO actor.
VIMPO_ACTOR_CONFIG_FIELDS = (
    "vimpo_beta",
    "vimpo_actor_coeff",
    "vimpo_value_loss_weight",
    "vimpo_gamma",
    "vimpo_lambda",
    "vimpo_top_k",
    "vimpo_whiten_advantages",
    "vimpo_detach_kl",
    "vimpo_ref_base_url",
    "vimpo_ref_timeout",
    "vimpo_ref_max_concurrency",
    "vimpo_ref_max_retries",
)


class VIMPOFSDPPPOActor(MultiCandidateFSDPEngine):
    """Dedicated VIMPO FSDP actor with frozen-reference candidate scoring.

    Extends ``MultiCandidateFSDPEngine`` (like ``MultiCandidateFSDPPPOActor``)
    and owns two VIMPO-specific dependencies:

    - a **frozen SGLang reference scorer** (``pi_ref = pi_0``, the actor's
      initial checkpoint, never weight-updated), and
    - a :class:`VIMPOAdvantageComputer` (critic-free, policy-implied terminal
      value).

    ``compute_advantages`` orchestrates, in order: validate frozen-reference
    identity -> snapshot the actor's top-k candidates -> score them against the
    frozen reference -> compute detached advantages. Reference scoring happens
    BEFORE any optimizer mutation (OpenSpec 5.4): a reference failure in
    ``compute_advantages`` propagates before ``ppo_update`` is ever called.

    ``ppo_update`` validates every required key, episode-atomically splits the
    batch, and calls :meth:`train_vimpo_batch` once per complete PPO minibatch
    (one combined backward with two distributed denominators; never the generic
    ``train_batch``).

    VIMPO runs with ``actor.kl_ctl = 0`` and ``config.ref = None``: no learned
    critic and no generic reference engine are created by the base trainer.
    """

    def __init__(
        self,
        config: PPOActorConfig,
        scorer: VIMPOReferenceScorer | None = None,
    ) -> None:
        super().__init__(config)
        self.actor = PPOActor(config, self)
        self.reference_scorer: VIMPOReferenceScorer = (
            scorer
            or SGLangVIMPOReferenceScorer(
                config.vimpo_ref_base_url,
                timeout=config.vimpo_ref_timeout,
                max_concurrency=config.vimpo_ref_max_concurrency,
                max_retries=config.vimpo_ref_max_retries,
            )
        )
        self.vimpo_advantage = VIMPOAdvantageComputer(
            beta=config.vimpo_beta,
            gamma=config.vimpo_gamma,
            lam=config.vimpo_lambda,
            whiten=config.vimpo_whiten_advantages,
            dp_group=self.dp_group,
        )
        logger.info(
            "VIMPOFSDPPPOActor initialized (top_k=%d, beta=%g, ref=%s)",
            config.vimpo_top_k,
            config.vimpo_beta,
            config.vimpo_ref_base_url,
        )

    # -- PPO-facing surface (compute_logp delegates; the rest is VIMPO-specific) --

    @torch.no_grad()
    def compute_logp(self, *args: Any, **kwargs: Any) -> list[torch.Tensor] | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(
        self, data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return batched_call(self._compute_vimpo_advantages, data, pass_meta=True)

    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._vimpo_update, data, unpack=False)

    def destroy(self) -> None:
        try:
            self.reference_scorer.close()
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning("VIMPO reference scorer close failed (ignored): %s", e)
        super().destroy()

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        from areal.trainer.ppo.actor import (
            PPOActorController,
            PPOActorControllerV2,
        )

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(
            train_engine=cls, config=config, scheduler=scheduler
        )

    # -- VIMPO advantage orchestration --------------------------------

    def _expected_reference_identity(self) -> ReferenceIdentity:
        """Build the frozen-reference identity from the actor's own checkpoint.

        The reference policy is ``pi_ref = pi_0`` (the actor's initial
        checkpoint), so the expected identity is the actor's own model/tokenizer
        identity. ``validate_identity`` compares every field against the SGLang
        ``/get_model_info`` response before any scoring begins. ``revision``
        is left empty: stock SGLang does not report one, and an empty expected
        revision is treated as a wildcard by the scorer's comparison.
        """
        cached = getattr(self, "_vimpo_expected_identity", None)
        if cached is not None:
            return cached
        path = getattr(self.config, "path", "")
        model_config = getattr(self, "model_config", None)
        vocab_size = 0
        bos = eos = pad = None
        if model_config is not None:
            vocab_size = int(getattr(model_config, "vocab_size", 0) or 0)
            bos = getattr(model_config, "bos_token_id", None)
            eos = getattr(model_config, "eos_token_id", None)
            pad = getattr(model_config, "pad_token_id", None)
        tokenizer_vocab_size = vocab_size
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            tokenizer_vocab_size = int(
                getattr(tok, "vocab_size", vocab_size) or vocab_size
            )
            pad = pad if pad is not None else getattr(tok, "pad_token_id", None)
            bos = bos if bos is not None else getattr(tok, "bos_token_id", None)
            eos = eos if eos is not None else getattr(tok, "eos_token_id", None)
        identity = ReferenceIdentity(
            model_path=path,
            revision="",
            vocab_size=vocab_size,
            tokenizer_vocab_size=tokenizer_vocab_size,
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=pad,
        )
        self._vimpo_expected_identity = identity
        return identity

    def _is_reference_scoring_head(self) -> bool:
        """Only the model-parallel head builds requests and scores (one HTTP
        client); other ranks receive reference tensors via broadcast."""
        if not dist.is_initialized():
            return True
        return self.is_data_parallel_head()

    def _compute_vimpo_advantages(
        self, data: dict[str, Any], meta: Any = None
    ) -> dict[str, Any]:
        # Step 1: validate frozen-reference identity (MP head only).
        if self._is_reference_scoring_head():
            self.reference_scorer.validate_identity(
                self._expected_reference_identity()
            )
        # Step 2: snapshot the actor's top-k candidates (all ranks; eval forward).
        stats = self.compute_vimpo_candidate_stats(
            data, top_k=self.config.vimpo_top_k
        )
        # Step 3: build reference score requests and score (MP head only).
        scores: list[ReferenceScore] | None = None
        if self._is_reference_scoring_head():
            requests = self._build_reference_requests(data, stats)
            scores = self.reference_scorer.score(requests)
        # Step 4: assemble reference logps + centered reward + advantage.
        enriched = self._assemble_vimpo_batch(data, stats, scores)
        return enriched

    def _build_reference_requests(
        self,
        data: dict[str, Any],
        stats: VIMPOCandidateStats,
    ) -> list[ReferenceScoreRequest]:
        """Build one ``ReferenceScoreRequest`` per valid ``(row, position)``."""
        input_ids = data["input_ids"]
        predict_mask = stats.predict_mask
        candidate_ids = stats.candidate_ids
        batch_size, seq_len = predict_mask.shape
        # Convert to CPU lists once (avoid per-position GPU sync).
        input_ids_cpu = input_ids.tolist()
        mask_cpu = predict_mask.tolist()
        cand_ids_cpu = candidate_ids.tolist()
        requests: list[ReferenceScoreRequest] = []
        for row in range(batch_size):
            for pos in range(seq_len):
                if not mask_cpu[row][pos]:
                    continue
                if pos + 1 >= len(input_ids_cpu[row]):
                    continue
                prefix = input_ids_cpu[row][: pos + 1]
                sampled = input_ids_cpu[row][pos + 1]
                cands = [c for c in cand_ids_cpu[row][pos] if c >= 0]
                if not cands:
                    continue
                requests.append(
                    ReferenceScoreRequest(
                        key=(row, pos),
                        prefix_ids=prefix,
                        sampled_token_id=sampled,
                        candidate_token_ids=cands,
                    )
                )
        return requests

    def _assemble_vimpo_batch(
        self,
        data: dict[str, Any],
        stats: VIMPOCandidateStats,
        scores: list[ReferenceScore] | None,
    ) -> dict[str, Any]:
        """Fill reference logps from scores, broadcast, and compute advantages."""
        predict_mask = stats.predict_mask
        ref_sample_logp = torch.zeros_like(stats.sampled_logp)
        ref_candidate_logp = torch.zeros_like(stats.candidate_logp)
        if scores is not None:
            self._fill_reference_logps(
                ref_sample_logp, ref_candidate_logp, stats, scores
            )
        # Broadcast reference tensors from the MP head to all ranks.
        self._broadcast_vimpo_reference_tensors(ref_sample_logp, ref_candidate_logp)
        # Build the advantage batch.
        batch: dict[str, Any] = dict(data)
        batch["vimpo_predict_mask"] = predict_mask
        batch["vimpo_sample_logp"] = stats.sampled_logp
        batch["vimpo_candidate_logp"] = stats.candidate_logp
        batch["vimpo_candidate_ids"] = stats.candidate_ids
        batch["vimpo_ref_sample_logp"] = ref_sample_logp
        batch["vimpo_ref_candidate_logp"] = ref_candidate_logp
        for key in (
            "vimpo_episode_index",
            "vimpo_turn_index",
            "vimpo_expected_turn_count",
        ):
            if key in data:
                batch[key] = data[key]
        batch["vimpo_centered_reward"] = self._compute_centered_reward(
            data, predict_mask
        )
        # Invoke the advantage computer (writes vimpo_candidate_kl,
        # vimpo_retained_mass, advantages).
        batch = self.vimpo_advantage.compute(batch)
        batch["advantages"] = batch["advantages"].detach()
        # OpenSpec 6.1: surface candidate KL / retained-mass numerators.
        stats_tracker.denominator(vimpo_valid_tokens=predict_mask.bool())
        stats_tracker.stat(
            vimpo_candidate_kl=batch["vimpo_candidate_kl"].float(),
            vimpo_retained_mass=batch["vimpo_retained_mass"].float(),
            denominator="vimpo_valid_tokens",
        )
        return batch

    @staticmethod
    def _fill_reference_logps(
        ref_sample_logp: torch.Tensor,
        ref_candidate_logp: torch.Tensor,
        stats: VIMPOCandidateStats,
        scores: list[ReferenceScore],
    ) -> None:
        """Align ``ReferenceScore`` results by key into dense [B, S, ...] tensors."""
        candidate_ids = stats.candidate_ids
        k_dim = candidate_ids.shape[-1]
        cand_ids_cpu = candidate_ids.tolist()
        score_by_key = {score.key: score for score in scores}
        for key, score in score_by_key.items():
            row, pos = key
            ref_sample_logp[row, pos] = float(score.sampled_logp)
            for k in range(k_dim):
                tid = cand_ids_cpu[row][pos][k]
                if tid >= 0:
                    ref_candidate_logp[row, pos, k] = float(score.candidate_logp[k])

    def _broadcast_vimpo_reference_tensors(self, *tensors: torch.Tensor) -> None:
        """Broadcast reference tensors from the model-parallel head to all ranks."""
        if not dist.is_initialized() or self.mp_group is None:
            return
        mp_head = dist.get_process_group_ranks(self.mp_group)[0]
        for tensor in tensors:
            dist.broadcast(tensor, src=mp_head, group=self.mp_group)

    def _compute_centered_reward(
        self,
        data: dict[str, Any],
        predict_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Center the terminal task reward by the batch-mean baseline.

        VIMPO's terminal reward is shared across all positions of an episode.
        The centered reward (``reward - mean(reward over the group)``) is the
        task-reward component of the terminal value target, broadcast to every
        position's ``[B, S]`` shape.
        """
        rewards = data.get("rewards")
        if rewards is None:
            return torch.zeros(
                predict_mask.shape, dtype=torch.float32, device=predict_mask.device
            )
        rewards = rewards.float()
        centered = rewards - rewards.mean()
        if centered.ndim == 1:
            return centered.unsqueeze(-1).expand(*predict_mask.shape)
        return centered.expand(*predict_mask.shape)

    # -- VIMPO update orchestration -----------------------------------

    def _vimpo_update(self, data: dict[str, Any], meta: Any = None) -> None:
        # Step 1: validate every required key BEFORE any optimizer mutation.
        self._validate_vimpo_batch(data)
        # Step 2: log token/episode denominators (OpenSpec 6.1). Only n_seqs is
        # registered here: the [B, S] vimpo_valid_tokens denominator is already
        # registered by _assemble_vimpo_batch (double registration would
        # double-count the exported SUM).
        mask = data["vimpo_predict_mask"].bool()
        episode_index = data["vimpo_episode_index"]
        n_valid_tokens = int(mask.sum().item())
        n_episodes = int(torch.unique(episode_index[mask]).numel())
        batch_size = data["attention_mask"].shape[0]
        stats_tracker.denominator(
            n_seqs=torch.ones(batch_size, dtype=torch.bool, device=data["attention_mask"].device),
        )
        logger.info(
            "VIMPO ppo_update: %d valid tokens, %d episodes, %d PPO minibatches",
            n_valid_tokens,
            n_episodes,
            self.config.ppo_n_minibatches,
        )
        # Step 3: episode-atomic outer split into PPO minibatches.
        mb_spec = MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches)
        mb_list = split_episode_atomic_batches(data, mb_spec)
        # Step 4: call train_vimpo_batch once per complete PPO minibatch (one
        # combined backward with two distributed denominators; never train_batch).
        eps_clip_higher = getattr(self.config, "eps_clip_higher", None)
        for mb in mb_list.mbs:
            train_stat = self.train_vimpo_batch(
                mb,
                actor_coeff=self.config.vimpo_actor_coeff,
                value_loss_weight=self.config.vimpo_value_loss_weight,
                beta=self.config.vimpo_beta,
                eps_clip=self.config.eps_clip,
                eps_clip_higher=eps_clip_higher,
            )
            stats_tracker.scalar(**train_stat)


class MuonVIMPOFSDPPPOActor:
    """VIMPO FSDP actor wrapper with Muon optimizer patching.

    Patches optimizer construction before worker init (same pattern as
    ``MuonMultiCandidateFSDPPPOActor``), then delegates to
    :class:`VIMPOFSDPPPOActor`. VIMPO's loss/advantage path is untouched.
    """

    def __new__(cls, config: PPOActorConfig, scorer: VIMPOReferenceScorer | None = None):
        from customized_areal.tree_search.engine import VIMPOFSDPPPOActor

        _patch_muon_from_actor_config(config)
        return VIMPOFSDPPPOActor(config, scorer=scorer)

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        from areal.trainer.ppo.actor import (
            PPOActorController,
            PPOActorControllerV2,
        )

        controller_cls = (
            PPOActorControllerV2 if config._version == "v2" else PPOActorController
        )
        return controller_cls(
            train_engine=cls, config=config, scheduler=scheduler
        )


def patch_ppo_actor_class_to_use_distill_loss() -> None:
    """Patch PPOActor class to use grpo_distill_loss_fn globally.

    This replaces PPOActor._ppo_update with a version that uses
    grpo_distill_loss_fn instead of standard grpo_loss_fn.

    Only patches once, even if called multiple times.

    Note:
    -----
    For multi-candidate gathering during training, the AReaL engine must be
    modified to pass logits to the loss function. See ENGINE_MODIFICATION.md
    in the training directory.
    """
    global _patch_applied, _original_ppo_update
    if _patch_applied:
        return

    _original_ppo_update = PPOActor._ppo_update

    def _ppo_update_with_distill_loss(self, data: dict[str, Any]) -> None:
        """PPO update using grpo_distill_loss_fn."""
        from .loss import grpo_distill_loss_fn

        # Log reward stats before removing them (Bug 2 fix)
        reward_score = data.get("rewards")
        if reward_score is not None and isinstance(reward_score, torch.Tensor):
            correct_n = (reward_score > 0).bool()
            incorrect_n = (reward_score <= 0).bool()
            stats_tracker.denominator(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                correct_n_seqs=correct_n,
                incorrect_n_seqs=incorrect_n,
            )
            attn_mask = data.get("attention_mask")
            if attn_mask is not None:
                stats_tracker.stat(
                    task_reward=reward_score.float(),
                    denominator="n_seqs",
                )

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        # Extract position_rewards before splitting so we can distribute
        # the correct subset to each minibatch.  position_rewards is a
        # Python list and cannot be split by the generic tensor-based
        # minibatch splitter.
        position_rewards = data.pop("position_rewards", None)

        self.engine.train()

        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        # Distribute position_rewards to minibatches based on sample_index.
        # Each PositionRewardInfo.sample_index indicates which batch item
        # it belongs to.  We use the forward_indices from the minibatch
        # split to determine which samples are in which minibatch.
        if position_rewards is not None:
            _distribute_position_rewards(mb_inputs, position_rewards)

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_distill_loss_fn,
                        config=self.config,
                        current_version=current_version,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

    PPOActor._ppo_update = _ppo_update_with_distill_loss
    _patch_applied = True
    logger.info("PPOActor class patched to use grpo_distill_loss_fn")


def unpatch_ppo_actor_distill_loss() -> None:
    """Restore the original PPOActor._ppo_update method.

    Must be called after patch_ppo_actor_class_to_use_distill_loss().
    """
    global _patch_applied, _original_ppo_update
    if _patch_applied and _original_ppo_update is not None:
        PPOActor._ppo_update = _original_ppo_update
        _original_ppo_update = None
        _patch_applied = False
        logger.info("Restored original PPOActor._ppo_update")


_combined_critic_patch_applied = False
_combined_critic_original_ppo_update = None
_combined_critic_cfg: dict[str, Any] = {}


def patch_ppo_actor_class_to_use_combined_critic_loss(
    critic_loss_weight: float,
    pad_token_id: int = 0,
) -> None:
    """Patch PPOActor._ppo_update to add the shared-model critic regression step.

    Wraps whatever ``_ppo_update`` is currently installed (so it composes with
    the distill-loss patch). After the actor update runs, if the batch carries
    ``critic_train_data`` an additional critic soft-regression step is run on the
    same model, weighted by ``critic_loss_weight``.

    Idempotent. Restore with :func:`unpatch_combined_critic_loss`.
    """
    global _combined_critic_patch_applied, _combined_critic_original_ppo_update
    global _combined_critic_cfg
    if _combined_critic_patch_applied:
        return

    _combined_critic_original_ppo_update = PPOActor._ppo_update
    _combined_critic_cfg = {
        "weight": critic_loss_weight,
        "pad": pad_token_id,
    }
    inner = _combined_critic_original_ppo_update

    def _ppo_update_with_combined_critic(self, data: dict[str, Any]) -> None:
        from .critic_update import run_critic_regression_step

        critic_train_data = None
        if isinstance(data, dict):
            critic_train_data = data.pop("critic_train_data", None)

        inner(self, data)

        if critic_train_data is not None:
            run_critic_regression_step(
                self,
                critic_train_data,
                critic_loss_weight=_combined_critic_cfg["weight"],
                pad_token_id=_combined_critic_cfg["pad"],
            )

    PPOActor._ppo_update = _ppo_update_with_combined_critic
    _combined_critic_patch_applied = True
    logger.info(
        "PPOActor patched for combined generative-critic loss (weight=%.4g)",
        critic_loss_weight,
    )


def unpatch_combined_critic_loss() -> None:
    """Restore the pre-combined-critic PPOActor._ppo_update method."""
    global _combined_critic_patch_applied, _combined_critic_original_ppo_update
    if (
        _combined_critic_patch_applied
        and _combined_critic_original_ppo_update is not None
    ):
        PPOActor._ppo_update = _combined_critic_original_ppo_update
        _combined_critic_original_ppo_update = None
        _combined_critic_patch_applied = False
        logger.info("Restored PPOActor._ppo_update (combined critic loss)")


def _distribute_position_rewards(mb_inputs, position_rewards: list) -> None:
    """Distribute position_rewards to minibatches based on sample_index.

    Each PositionRewardInfo has a sample_index indicating which batch item
    it belongs to.  The MicroBatchList.forward_indices maps batch items to
    their position in the reordered minibatch sequence.  We use this to
    determine which position_rewards belong to which minibatch.
    """
    if not position_rewards:
        return

    forward_indices = mb_inputs.forward_indices
    # Build mapping: original batch index -> minibatch index
    batch_size = len(forward_indices)
    mb_assignment: list[int | None] = [None] * batch_size
    offset = 0
    for i, mb in enumerate(mb_inputs.mbs):
        mb_bs = mb["attention_mask"].shape[0]
        for j in range(mb_bs):
            orig_idx = int(forward_indices[offset + j])
            mb_assignment[orig_idx] = i
        offset += mb_bs

    # Group position_rewards by minibatch and rebase sample_index to the
    # sample's local index inside that minibatch.
    per_mb_prs: dict[int, list] = {}
    for pr in position_rewards:
        sample_index = int(pr.sample_index)
        if sample_index < 0 or sample_index >= len(mb_assignment):
            logger.warning(
                "position_reward sample_index=%d exceeds batch_size=%d, "
                "dropping position=%d",
                sample_index,
                len(mb_assignment),
                pr.position,
            )
            continue
        mb_i = mb_assignment[sample_index]
        if mb_i is None:
            logger.warning(
                "position_reward sample_index=%d not mapped to any minibatch, "
                "dropping position=%d",
                sample_index,
                pr.position,
            )
            continue
        local_sample_index = None
        offset = 0
        for i, mb in enumerate(mb_inputs.mbs):
            mb_bs = mb["attention_mask"].shape[0]
            if i == mb_i:
                for local_idx in range(mb_bs):
                    orig_idx = int(forward_indices[offset + local_idx])
                    if orig_idx == sample_index:
                        local_sample_index = local_idx
                        break
                break
            offset += mb_bs
        if local_sample_index is None:
            logger.warning(
                "position_reward sample_index=%d could not be rebased for minibatch %d, "
                "dropping position=%d",
                sample_index,
                mb_i,
                pr.position,
            )
            continue
        mb_pr = copy.copy(pr)
        mb_pr.sample_index = local_sample_index
        per_mb_prs.setdefault(mb_i, []).append(mb_pr)

    # Attach to minibatches
    for i, mb in enumerate(mb_inputs.mbs):
        if i in per_mb_prs:
            mb["position_rewards"] = per_mb_prs[i]
        else:
            mb["position_rewards"] = []

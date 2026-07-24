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
import time
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

# torch.quantile errors on inputs larger than 2**24 elements.
_QUANTILE_INPUT_LIMIT = 2**24


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
    "vimpo_ref_backend",
    "vimpo_ref_base_url",
    "vimpo_ref_timeout",
    "vimpo_ref_max_concurrency",
    "vimpo_ref_max_retries",
    "vimpo_ref_path",
)


class VIMPOFSDPPPOActor(MultiCandidateFSDPEngine):
    """Dedicated VIMPO FSDP actor with frozen-reference candidate scoring.

    Extends ``MultiCandidateFSDPEngine`` (like ``MultiCandidateFSDPPPOActor``)
    and owns two VIMPO-specific dependencies:

    - a **frozen reference** (``pi_ref = pi_0``, the actor's initial
      checkpoint, never weight-updated), served either by a remote SGLang
      scorer (``vimpo_ref_backend='sglang'``) or by a local second FSDP
      engine with CPU-resident sharded parameters
      (``vimpo_ref_backend='local'``), and
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
        self.vimpo_ref_backend = getattr(config, "vimpo_ref_backend", "sglang")
        self.reference_scorer: VIMPOReferenceScorer | None = None
        self._ref_engine: MultiCandidateFSDPEngine | None = None
        if scorer is not None:
            self.reference_scorer = scorer
        elif self.vimpo_ref_backend == "local":
            # Local frozen reference: a second FSDP engine sharded like the
            # actor whose parameters live on CPU (fsdp.offload_params=True)
            # and stream to GPU layer-by-layer during the no-grad scoring
            # forward. Weights are loaded once at initialize() and never
            # change, so no per-step load/offload bookkeeping is needed.
            self._ref_engine = MultiCandidateFSDPEngine(
                self._build_ref_engine_config(config)
            )
            self._validate_ref_identity()
        else:
            self.reference_scorer = SGLangVIMPOReferenceScorer(
                config.vimpo_ref_base_url,
                timeout=config.vimpo_ref_timeout,
                max_concurrency=config.vimpo_ref_max_concurrency,
                max_retries=config.vimpo_ref_max_retries,
            )
        self.vimpo_advantage = VIMPOAdvantageComputer(
            beta=config.vimpo_beta,
            gamma=config.vimpo_gamma,
            lam=config.vimpo_lambda,
            whiten=config.vimpo_whiten_advantages,
            dp_group=self.dp_group,
        )
        # Scalar VIMPO metrics from the most recent compute_advantages /
        # ppo_update cycle (component/combined losses, reference latency and
        # retries, effective top-k, exact-KL flag, snapshot policy version,
        # terminal RMSE). Keys match the stats_tracker metric names.
        self.last_vimpo_metrics: dict[str, float] = {}
        logger.info(
            "VIMPOFSDPPPOActor initialized (top_k=%d, beta=%g, ref_backend=%s, ref=%s)",
            config.vimpo_top_k,
            config.vimpo_beta,
            self.vimpo_ref_backend,
            getattr(config, "vimpo_ref_base_url", ""),
        )

    @staticmethod
    def _build_ref_engine_config(config: PPOActorConfig) -> PPOActorConfig:
        """Clone the actor config for the frozen local reference engine.

        The reference loads from ``vimpo_ref_path`` (default: the actor's own
        initial checkpoint ``config.path``), carries no optimizer, never uses
        LoRA, and keeps its FSDP-sharded parameters on CPU
        (``fsdp.offload_params=True``) with memory-efficient CPU loading so
        the full model never materializes on GPU - even at init.
        """
        ref_config = copy.deepcopy(config)
        ref_config.path = getattr(config, "vimpo_ref_path", "") or config.path
        ref_config.optimizer = None
        ref_config.use_lora = False
        ref_config.init_from_scratch = False
        ref_config.fsdp.offload_params = True
        ref_config.fsdp.memory_efficient_load = True
        return ref_config

    def _validate_ref_identity(self) -> None:
        """Fail fast if the local reference checkpoint is vocab-incompatible."""
        actor_vocab = int(getattr(self.model_config, "vocab_size", 0) or 0)
        ref_vocab = (
            int(getattr(self._ref_engine.model_config, "vocab_size", 0) or 0)
            if self._ref_engine is not None
            else 0
        )
        if actor_vocab and ref_vocab and actor_vocab != ref_vocab:
            raise ValueError(
                "VIMPO local reference vocab_size mismatch: "
                f"actor={actor_vocab} ref={ref_vocab} "
                f"(ref path={self._ref_engine.config.path!r})"
            )

    # -- PPO-facing surface (compute_logp delegates; the rest is VIMPO-specific) --

    @torch.no_grad()
    def compute_logp(self, *args: Any, **kwargs: Any) -> list[torch.Tensor] | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return batched_call(self._compute_vimpo_advantages, data, pass_meta=True)

    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._vimpo_update, data, unpack=False)

    def create_process_group(self, parallel_strategy: Any = None) -> None:
        super().create_process_group(parallel_strategy=parallel_strategy)
        if self._ref_engine is not None:
            # The reference engine shards over the same ranks/strategy; its
            # process groups are independent of the actor's (same pattern as
            # the upstream train/ref engine pair).
            self._ref_engine.create_process_group(parallel_strategy=parallel_strategy)

    def initialize(self, addr: str | None, ft_spec: Any, *args: Any, **kwargs: Any):
        super().initialize(addr, ft_spec, *args, **kwargs)
        if self._ref_engine is not None:
            self._ref_engine.initialize(addr, ft_spec, *args, **kwargs)
            # Frozen: eval mode (no dropout) and no gradient tracking.
            self._ref_engine.model.eval()
            for p in self._ref_engine.model.parameters():
                p.requires_grad_(False)

    def destroy(self) -> None:
        try:
            if self.reference_scorer is not None:
                self.reference_scorer.close()
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning("VIMPO reference scorer close failed (ignored): %s", e)
        try:
            ref_engine = getattr(self, "_ref_engine", None)
            if ref_engine is not None:
                ref_engine.destroy()
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning("VIMPO reference engine destroy failed (ignored): %s", e)
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
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)

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
        # Step 0: re-index per-query episode ids to batch-unique ids (the
        # workflow's counter resets per query, so identical indices from
        # different queries collide after the cross-query concat).
        data = self._reindex_vimpo_episodes(data, meta)
        # ``getattr`` keeps test doubles (built via ``__new__``) on the
        # default SGLang path.
        if getattr(self, "vimpo_ref_backend", "sglang") == "local":
            # Local frozen reference (all ranks): actor candidate pass + CPU
            # resident FSDP reference pass, then attach ref tensors directly
            # (no HTTP scorer, no broadcast). Reference identity was already
            # validated at construction (vocab compatibility) and any load
            # failure surfaced at initialize() - both before any optimizer
            # mutation, preserving the fail-before-update contract.
            start = time.perf_counter()
            stats, ref_stats = self.compute_vimpo_stats_with_local_reference(
                data, top_k=self.config.vimpo_top_k
            )
            latency_ms = (time.perf_counter() - start) * 1e3
            self._record_snapshot_scalars(stats, latency_ms)
            return self._assemble_vimpo_batch(
                data,
                stats,
                None,
                ref_tensors=(ref_stats.sampled_logp, ref_stats.candidate_logp),
            )
        # Step 1: validate frozen-reference identity (MP head only).
        if self._is_reference_scoring_head():
            self.reference_scorer.validate_identity(self._expected_reference_identity())
        # Step 2: snapshot the actor's top-k candidates (all ranks; eval forward).
        stats = self.compute_vimpo_candidate_stats(data, top_k=self.config.vimpo_top_k)
        # Step 3: build reference score requests and score (MP head only).
        scores: list[ReferenceScore] | None = None
        latency_ms: float | None = None
        if self._is_reference_scoring_head():
            requests = self._build_reference_requests(data, stats)
            start = time.perf_counter()
            scores = self.reference_scorer.score(requests)
            latency_ms = (time.perf_counter() - start) * 1e3
        # Observability (OpenSpec 6.2): snapshot/reference scalars.
        self._record_snapshot_scalars(stats, latency_ms)
        # Step 4: assemble reference logps + centered reward + advantage.
        enriched = self._assemble_vimpo_batch(data, stats, scores)
        return enriched

    @torch.no_grad()
    def compute_vimpo_stats_with_local_reference(
        self,
        data: list[dict[str, Any]] | dict[str, Any],
        *,
        top_k: int,
    ) -> tuple[VIMPOCandidateStats, VIMPOCandidateStats]:
        """Two-pass local-reference scoring: actor candidates + frozen ref logps.

        Pass 1 runs the actor's eval forward over one prepared mb_list and
        stashes each micro-batch's packed candidate IDs. Pass 2 replays the
        same mb_list on the frozen reference engine (CPU-resident FSDP params
        streamed per layer) and gathers reference log-probs for exactly those
        candidates. Returns ``(actor_stats, ref_stats)``, both shaped
        ``[B, S, ...]`` on every rank.
        """
        if self._ref_engine is None:
            raise RuntimeError(
                "compute_vimpo_stats_with_local_reference requires "
                "vimpo_ref_backend='local' with an initialized reference engine"
            )
        mb_list, output_seqlens, batch_size = self._prepare_vimpo_mb(data)
        packed_ids: list[Any] = []
        actor_stats = self.run_vimpo_actor_pass(
            mb_list, output_seqlens, batch_size, top_k=top_k, packed_ids_sink=packed_ids
        )
        ref_stats = self._ref_engine.run_vimpo_reference_pass(
            mb_list, packed_ids, output_seqlens, batch_size, top_k=top_k
        )
        return actor_stats, ref_stats

    @staticmethod
    def _reindex_vimpo_episodes(
        data: dict[str, Any], meta: Any = None
    ) -> dict[str, Any]:
        """Re-index per-query ``vimpo_episode_index`` to batch-unique ids.

        ``annotate_vimpo_episode_metadata`` runs once per query (its counter
        resets every call), so identical episode indices from different
        queries collide after the cross-query concat. ``vimpo_query_index``
        cannot disambiguate: it is also emitted per query and is therefore 0
        on every row of the batch. The concat metadata's per-dict row counts
        (``meta.traj_group_sizes``) mark the true query boundaries - each
        input dict is one query's ``_finalize_episode`` output. Within a
        group the emitted indices are already unique per episode, so
        offsetting each group by the running episode count yields compact
        batch-unique ids. Every downstream consumer (the reverse-lambda
        advantage recurrence, the episode-atomic batcher, the loss) then
        sees collision-free episode identity from this single canonical
        re-index. Rows are shifted uniformly, padded columns included; all
        consumers read valid positions only. ``vimpo_query_index`` (when
        present) is rewritten to the group index so the
        (query, episode) pair is batch-unique again.
        """
        if "vimpo_episode_index" not in data:
            return data
        episode_index = data["vimpo_episode_index"]
        attention_mask = data["attention_mask"].bool()
        group_sizes = getattr(meta, "traj_group_sizes", None)
        if not group_sizes:
            group_sizes = [episode_index.shape[0]]
        data = dict(data)
        new_index = episode_index.clone()
        new_query_index = (
            data["vimpo_query_index"].clone() if "vimpo_query_index" in data else None
        )
        offset = 0
        row = 0
        for group_index, size in enumerate(group_sizes):
            rows = slice(row, row + size)
            valid_values = episode_index[rows][attention_mask[rows]]
            if valid_values.numel() > 0:
                new_index[rows] = episode_index[rows] + offset
                offset += int(valid_values.max().item()) + 1
            if new_query_index is not None:
                new_query_index[rows] = group_index
            row += size
        data["vimpo_episode_index"] = new_index
        if new_query_index is not None:
            data["vimpo_query_index"] = new_query_index
        return data

    def _record_snapshot_scalars(
        self, stats: VIMPOCandidateStats, latency_ms: float | None
    ) -> None:
        """Emit snapshot/reference scalar metrics and stash them for inspection.

        ``vimpo/exact_kl`` is 1 only when the effective ``K`` equals the model
        vocabulary size (untruncated candidate KL); when the vocabulary size is
        unknown the flag stays 0. Reference latency/retry scalars are emitted
        only on the reference-scoring head (``latency_ms is None`` elsewhere).
        """
        effective_k = int(stats.candidate_ids.shape[-1])
        model_config = getattr(self, "model_config", None)
        vocab_size = (
            int(getattr(model_config, "vocab_size", 0) or 0)
            if model_config is not None
            else 0
        )
        # ``get_version`` reads ``self._version``, which only exists after
        # engine initialization - fall back to 0 for uninitialized doubles.
        version = self.get_version() if hasattr(self, "_version") else 0
        scalars: dict[str, float] = {
            "vimpo/effective_top_k": float(effective_k),
            "vimpo/exact_kl": (
                1.0 if vocab_size > 0 and effective_k == vocab_size else 0.0
            ),
            "vimpo/snapshot_policy_version": float(version),
        }
        if latency_ms is not None:
            scalars["vimpo/reference_latency_ms"] = float(latency_ms)
            if self.reference_scorer is not None:
                scalars["vimpo/reference_retries"] = float(
                    getattr(self.reference_scorer, "retry_count", 0) or 0
                )
        stats_tracker.scalar(**scalars)
        self.last_vimpo_metrics = {
            **getattr(self, "last_vimpo_metrics", {}),
            **scalars,
        }

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
        ref_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Fill reference logps from scores, broadcast, and compute advantages.

        When ``ref_tensors`` is provided (local reference backend), the
        per-rank ``(ref_sample_logp, ref_candidate_logp)`` tensors are used
        directly and the HTTP-scorer fill + MP-head broadcast are skipped.
        """
        predict_mask = stats.predict_mask
        if ref_tensors is not None:
            ref_sample_logp, ref_candidate_logp = ref_tensors
        else:
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
        # Forward the workflow's VIMPO metadata. ``vimpo_episode_index`` was
        # re-indexed to batch-unique ids by ``_reindex_vimpo_episodes``;
        # ``vimpo_centered_reward`` is the spec'd query-local centered
        # terminal target computed by ``annotate_vimpo_episode_metadata``
        # (mean over distinct episodes sharing the query) and MUST NOT be
        # recomputed here - a batch-global mean would violate the contract.
        for key in (
            "vimpo_episode_index",
            "vimpo_turn_index",
            "vimpo_expected_turn_count",
            "vimpo_centered_reward",
        ):
            if key in data:
                batch[key] = data[key]
        # Invoke the advantage computer (writes vimpo_candidate_kl,
        # vimpo_retained_mass, advantages).
        raw_advantages = self._raw_vimpo_advantages(batch)
        batch = self.vimpo_advantage.compute(batch)
        batch["advantages"] = batch["advantages"].detach()
        if raw_advantages is None:
            raw_advantages = batch["advantages"]
        # Observability (OpenSpec 6.2): per-position distributions over valid
        # tokens. Tensors are fed directly to the tracker (no .item()/.tolist()
        # on hot-path GPU tensors).
        mask = predict_mask.bool()
        stats_tracker.denominator(**{"vimpo/valid_tokens": mask})
        stats_tracker.stat(
            denominator="vimpo/valid_tokens",
            **{
                "vimpo/candidate_kl": batch["vimpo_candidate_kl"].float(),
                "vimpo/retained_mass": batch["vimpo_retained_mass"].float(),
                "vimpo/raw_advantage": raw_advantages.float(),
                "vimpo/normalized_advantage": batch["advantages"].float(),
            },
        )
        # Retained-mass quantiles diagnose top-k truncation quality (a low p10
        # means many positions omit substantial KL tail mass).
        valid_mass = batch["vimpo_retained_mass"].float()[mask]
        if valid_mass.numel() > _QUANTILE_INPUT_LIMIT:
            # torch.quantile errors on >2**24 elements; subsample
            # deterministically (uniform stride) to stay under the ceiling.
            stride = -(-valid_mass.numel() // _QUANTILE_INPUT_LIMIT)
            valid_mass = valid_mass[::stride]
        if valid_mass.numel() > 0:
            quantiles = torch.quantile(
                valid_mass,
                torch.tensor([0.1, 0.5, 0.9], device=valid_mass.device),
            )
            stats_tracker.scalar(
                **{
                    "vimpo/retained_mass/p10": quantiles[0],
                    "vimpo/retained_mass/p50": quantiles[1],
                    "vimpo/retained_mass/p90": quantiles[2],
                }
            )
        return batch

    def _raw_vimpo_advantages(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Pre-whitening advantages for observability.

        Returns ``None`` when whitening is disabled (the final advantages
        already ARE the raw ones). Otherwise recomputes advantages with an
        identical non-whitening computer - pure tensor ops, no model forward -
        so both raw and normalized distributions can be logged.
        """
        if not getattr(self.vimpo_advantage, "whiten", False):
            return None
        keys = (
            "vimpo_predict_mask",
            "vimpo_sample_logp",
            "vimpo_ref_sample_logp",
            "vimpo_candidate_logp",
            "vimpo_ref_candidate_logp",
            "vimpo_episode_index",
            "vimpo_turn_index",
            "vimpo_candidate_ids",
        )
        computer = VIMPOAdvantageComputer(
            beta=self.vimpo_advantage.beta,
            gamma=self.vimpo_advantage.gamma,
            lam=self.vimpo_advantage.lam,
            whiten=False,
            dp_group=None,
        )
        sub_batch = {key: batch[key] for key in keys if key in batch}
        return computer.compute(sub_batch)["advantages"]

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

    # -- VIMPO update orchestration -----------------------------------

    def _vimpo_update(self, data: dict[str, Any], meta: Any = None) -> None:
        # NOTE: calling train() is critical to enabling gradient checkpointing
        self.train()
        # Step 1: validate every required key BEFORE any optimizer mutation.
        self._validate_vimpo_batch(data)
        # Step 2: log batch shape info. Denominators are registered elsewhere:
        # ``vimpo/valid_tokens`` ([B, S]) by ``_assemble_vimpo_batch`` and
        # ``vimpo/complete_episodes`` by ``train_vimpo_batch`` - double
        # registration would double-count the exported SUM.
        mask = data["vimpo_predict_mask"].bool()
        episode_index = data["vimpo_episode_index"]
        n_valid_tokens = int(mask.sum().item())
        n_episodes = int(torch.unique(episode_index[mask]).numel())
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
        loss_keys = (
            "vimpo/ppo_actor_loss",
            "vimpo/value_loss",
            "vimpo/combined_loss",
            "vimpo/terminal_rmse",
        )
        collected: dict[str, list[float]] = {}
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
            for key in loss_keys:
                if key in train_stat:
                    collected.setdefault(key, []).append(float(train_stat[key]))
        averaged = {key: sum(values) / len(values) for key, values in collected.items()}
        self.last_vimpo_metrics = {
            **getattr(self, "last_vimpo_metrics", {}),
            **averaged,
        }


class MuonVIMPOFSDPPPOActor:
    """VIMPO FSDP actor wrapper with Muon optimizer patching.

    Patches optimizer construction before worker init (same pattern as
    ``MuonMultiCandidateFSDPPPOActor``), then delegates to
    :class:`VIMPOFSDPPPOActor`. VIMPO's loss/advantage path is untouched.
    """

    def __new__(
        cls, config: PPOActorConfig, scorer: VIMPOReferenceScorer | None = None
    ):
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
        return controller_cls(train_engine=cls, config=config, scheduler=scheduler)


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

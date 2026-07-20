## Context

VIMPO derives a value recurrence from the optimality conditions of fixed-reference,
KL-regularized reinforcement learning. For outcome-only rewards and `gamma = 1`, its
operational terminal objective is

```text
L_V = mean_i 0.5 * (
    sum_t beta * (log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t)
                  - stop_gradient(KL_t))
    - (R_i - mean_group(R))
)^2.
```

The same token term supplies a detached TD advantage, optionally accumulated with a
GAE-style `lambda` return and whitened over valid response tokens, for a PPO-clipped actor
loss. The combined update is `L_V + c_A * L_A`; there is no learned value network.

The current tree-search stack computes TREE/GAE variants before PPO training and can
optionally train a shared generative critic through a separate digit-regression step. That
path is not reusable for VIMPO: it has different inputs, applies a separate optimizer step,
and treats reward and actor credit differently. AReaL also keeps train and inference
engines separate. The selected deployment therefore uses an FSDP actor and an
inference-only SGLang server loaded once from the actor's initial checkpoint.

The requested configurable top-k computation is not exact full-vocabulary KL unless
`k == vocab_size`. It is the authors' candidate-set approximation: select the current
policy's top-k tokens, retain the full-vocabulary normalizer for both distributions, and
sum the unrenormalized policy-weighted log-ratio over only those candidates.

## Goals / Non-Goals

**Goals:**

- Add `AdvantageMode.VIMPO` and validated paper-oriented settings, including configurable
  `vimpo_top_k` (default 128), `beta=5e-4`, `actor_coeff=5e-3`, `gamma=1`, `lambda=1`,
  detached KL, advantage whitening, and squared terminal loss.
- Compute actor-selected top-k policy statistics in the customized FSDP path without
  moving full-vocabulary logits across RPC boundaries.
- Score those exact candidate IDs with a frozen initial-policy SGLang service using
  full-softmax-normalized reference log-probabilities.
- Compute token-level VIMPO advantages in `core/advantage.py` and the differentiable
  terminal value loss plus PPO actor loss in one FSDP optimizer update.
- Preserve a complete multi-turn episode as the unit of reward centering and terminal
  value aggregation.
- Fail loudly on missing or malformed reference statistics and expose diagnostics for KL,
  retained mass, terminal residual, advantage distribution, and both loss components.

**Non-Goals:**

- Training a PPO critic, the existing generative critic, or an external judge.
- Updating the reference policy during training.
- Exact full-vocabulary KL when `vimpo_top_k < vocab_size`, or top-k renormalization.
- Megatron, Archon, vLLM-reference, sampled k1/k2/k3 estimators, adaptive actor
  coefficients, reference schedules, or candidate-set unions in the first version.
- Changing tree backup semantics, rollout rewards, public AReaL config dataclasses, or
  launcher/scheduler allocation policy.

## Decisions

### D1. VIMPO is an advantage/training mode, not a learned critic

Add `VIMPO = "vimpo"` to `AdvantageMode`. Selecting it is mutually exclusive with
`enable_generative_critic=True`; it must not invoke `_annotate_critic_values`, attach
`critic_train_data`, create `self.critic`, or run `critic_update.py`. "Critic training"
means optimizing the policy-implied terminal value loss on actor parameters.

Alternative rejected: reuse the generative-critic combined patch. It learns a different
categorical value predictor and performs a separate regression update, violating the
VIMPO objective.

### D2. Use a frozen initial-model SGLang reference

The reference is loaded from the same initial checkpoint and tokenizer as the actor, uses
temperature 1, remains in evaluation mode, and never receives actor weight updates. It may
run on a separate GPU or node, so the FSDP training allocation does not hold a second model.

The SGLang adapter accepts response prefixes and ragged per-position actor candidate IDs,
then returns sampled-token and candidate log-probabilities normalized by the full reference
vocabulary. The first implementation may batch one prefix-state request per response
position and rely on SGLang radix-prefix caching; the adapter boundary permits a later
native ragged-candidate endpoint without changing VIMPO math.

Alternatives rejected:

- A second FSDP reference consumes training memory and lifecycle resources despite never
  requiring gradients.
- A hidden reference module inside the actor bypasses AReaL's one-model FSDP initialization,
  offload, checkpoint, and controller assumptions.
- SGLang's ordinary `top_logprobs_num=k` returns reference-selected candidates, not the
  required actor-selected candidates.

### D3. Use actor top-k with exact full-vocabulary normalizers

Before PPO update, the customized FSDP actor produces, for every valid response position:

- the sampled-token policy log-probability;
- global policy top-k token IDs and their full-softmax-normalized log-probabilities;
- retained policy probability mass.

For tensor-parallel vocabularies, each rank computes local top-k, candidates are gathered
with global token offsets, and a second top-k selects the global result. Full log-normalizers
use distributed max/sum reductions. Sequence-parallel and packed-tree layouts are mapped
back to response positions using existing multi-candidate/tree utilities.

After SGLang scores the same IDs, compute

```text
KL_k = sum_{a in TopK_pi} pi(a|s) * (log pi(a|s) - log pi_ref(a|s)).
```

Neither distribution is renormalized over the retained candidates. `k` is capped at the
vocabulary size; `k == vocab_size` recovers exact KL. Retained mass is logged so users can
judge truncation quality.

Alternative rejected: transferring full reference logits, whose communication and storage
scale as `O(tokens * vocab_size)`.

### D4. Compute detached token advantages after reference scoring

`VIMPOAdvantageComputer` is a tensor-oriented helper in `core/advantage.py`, separate from
the scalar-per-Node GAE class. Given snapshot policy log-probabilities, reference
log-probabilities, `KL_k`, and masks, it computes

```text
d_t = beta * (log pi_t - log pi_ref_t - KL_k,t)
A_t = sum_l (gamma * lambda)^l * d_{t+l}.
```

The result is detached and whitened over all valid response tokens in the optimization
batch. Whitened advantages feed the ordinary PPO clipped surrogate using rollout/proximal
log-probabilities. External reward does not enter this actor advantage directly.

### D5. Preserve episode-level terminal loss across multi-turn Nodes

Tree-search emits one Node per turn while VIMPO's terminal square covers the entire
rollout. Tensorization therefore carries stable query ID, episode ID, turn order, centered
outcome reward, and an episode grouping index. VIMPO batching is episode-atomic: all turns
of an episode remain in the same PPO minibatch so their differentiable token terms can be
summed before squaring. The loss is averaged by episode, not by token.

Reward baselines are computed across distinct episodes sharing a query. A one-episode group
has centered target zero but still regularizes cumulative policy/reference log-ratio toward
zero. Padding and prompt tokens never contribute.

Alternative rejected: squaring each Node or microbatch independently, which changes the
objective and duplicates a single terminal reward across turns.

### D6. Combine value and actor losses in one FSDP backward pass

Add a VIMPO-specific loss module and actor update path. The training forward recomputes
differentiable sampled-token actor log-probabilities. Reference log-probabilities and
candidate KL are detached. The terminal value loss and `c_A`-weighted PPO loss are combined
before a single backward/optimizer step. Missing reference data, non-finite statistics, or
shape/alignment errors raise rather than silently skipping the reward-bearing loss.

Existing TREE/GAE/HYBRID_GAE/VERSIONED_BACKUP and distillation/clip-cov patches remain
unchanged. VIMPO is initially incompatible with distillation and clip-cov to avoid ambiguous
patch composition.

## Data Flow

```text
tree-search rollout Nodes
  -> tensorize with query/episode/turn metadata and centered reward
  -> FSDP actor no-grad candidate-stat pass
       sampled logpi + actor top-k ids/logpi + retained mass
  -> frozen SGLang reference candidate scoring
       sampled ref logpi + ref logpi for actor ids
  -> KL_k + detached VIMPO lambda-advantages + whitening
  -> episode-atomic PPO minibatches
  -> FSDP actor train forward
       differentiable sampled logpi
       terminal policy-implied value loss + c_A * PPO loss
  -> one optimizer step
```

## Risks / Trade-offs

- **[Top-k bias]** Small `k` omits KL tail mass. -> Report retained mass, test convergence
  at `k=vocab_size`, and name metrics/configuration as candidate/truncated KL.
- **[Reference scoring overhead]** Per-position SGLang requests can be numerous. -> Batch
  requests, exploit prefix caching, bound concurrency, and keep a replaceable adapter for a
  native ragged endpoint.
- **[Snapshot staleness]** Candidate IDs and KL are computed before PPO minibatch updates.
  -> Treat them as detached step statistics, recompute every training step, and document
  this difference from per-microbatch co-located reference evaluation.
- **[Episode batching pressure]** Long multi-turn episodes can create uneven minibatches.
  -> Group by episode first, then greedily balance groups by valid token count without
  splitting an episode.
- **[Tokenizer/model mismatch]** Different vocabularies make candidate IDs invalid. ->
  Validate model/tokenizer identity and vocabulary size before the first step.
- **[Serving precision]** Quantized reference logits change KL geometry. -> Require the
  configured reference precision explicitly and warn/reject unsupported quantization in
  paper-faithful mode.
- **[Distributed alignment]** TP/SP and packed-tree layouts can misalign candidate rows. ->
  Reuse established multi-candidate mappings and add distributed/hardware-gated tests.

## Migration Plan

1. Add the mode, settings, pure math, and unit tests while leaving the mode disabled.
2. Add actor candidate-stat collection and the SGLang scoring adapter behind VIMPO mode.
3. Add episode metadata/batching and the combined FSDP loss path.
4. Run CPU numerical tests, FSDP distributed tests where hardware is available, and one
   SGLang smoke step against an initial-checkpoint reference.
5. Enable only in an explicit experiment configuration. Rollback is selecting the previous
   advantage mode and stopping the reference service; existing checkpoints remain readable.

## Open Questions

- Whether the first production deployment should use the generic per-position SGLang batch
  adapter or add a native ragged candidate-scoring endpoint depends on measured throughput;
  both implement the same contract and do not change acceptance semantics.

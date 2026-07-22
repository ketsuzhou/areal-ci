# Brainstorm Summary

- Change: add-vimpo-critic-mode
- Date: 2026-07-20

## Confirmed Technical Approach

Use a VIMPO-specific FSDP PPO actor and a frozen, separately deployed SGLang service
loaded from the actor's initial checkpoint. The FSDP actor selects its global top-k tokens
at each response position and retains full-vocabulary-normalized probabilities. A narrow,
transport-independent reference-scoring interface sends each prefix state and its exact
actor-selected token IDs to SGLang. Version 1 uses bounded, batched prefix-state requests
with radix-cache reuse; the adapter can later be replaced by a native ragged endpoint.

Compute unrenormalized candidate KL over the actor-selected set, detached reverse-lambda
advantages, and one terminal policy-implied value residual per complete multi-turn episode.
Keep episodes atomic through PPO and engine microbatch construction. Recompute sampled
actor log-probabilities with gradients and combine the terminal value loss with the PPO
surrogate before one backward/optimizer step. Existing learned/generative critic paths are
not invoked.

## Key Trade-offs and Risks

- Generic prefix-state scoring avoids a SGLang fork but may be throughput-bound. Stable
  request keys, bounded concurrency, radix caching, and latency metrics contain the risk.
- Top-k KL is biased whenever `k < vocab_size`; retained policy mass is mandatory and only
  `k == vocab_size` may be labeled exact KL.
- Candidate statistics are a detached pre-update snapshot. They are recomputed each PPO
  step but not after every optimizer minibatch.
- Episode-atomic batching can be imbalanced for long episodes. Allocation is greedy by
  valid token count, and an episode larger than the configured microbatch token limit
  fails with a diagnostic rather than being split.
- Tokenizer, vocabulary, checkpoint, precision, row-order, or finite-value mismatches must
  fail before optimizer mutation.

## Testing Strategy

- CPU unit tests for configuration, full-normalizer candidate KL, exactness at full
  vocabulary, masks, reverse-lambda accumulation, distributed whitening helpers, terminal
  episode aggregation, PPO clipping, and gradient isolation.
- Mocked SGLang contract tests for per-position actor-selected IDs, prefix alignment,
  bounded retries/timeouts, malformed responses, and rejection of reference-selected top-k.
- Fake-process-group and hardware-gated FSDP tests for global top-k, TP vocabulary offsets,
  sequence/packed-tree alignment, episode-atomic batching, and one combined optimizer step.
- A CPU end-to-end smoke test and a hardware-gated FSDP plus frozen-SGLang one-step test.
- Regression tests proving existing advantage modes and generative critic behavior are
  unchanged.

## Spec Patches

None. The current delta spec already contains the required boundary and failure scenarios.

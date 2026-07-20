## 1. Configuration and Core Mathematics

- [x] 1.1 Add `AdvantageMode.VIMPO` and paper-oriented VIMPO fields/defaults to
  `customized_areal/tree_search/config.py`, including configurable top-k and explicit
  validation/incompatibility errors.
- [x] 1.2 Implement masked reverse lambda accumulation, detached distributed whitening,
  and `VIMPOAdvantageComputer` tensor helpers in `core/advantage.py`.
- [x] 1.3 Implement pure candidate-KL helpers using policy-selected top-k values without
  candidate renormalization, including retained-mass metrics and the full-vocabulary case.
- [x] 1.4 Add CPU numerical and gradient-isolation tests for configuration, top-k KL,
  lambda advantages, masks, whitening, and `k == vocab_size` exactness.

## 2. FSDP Actor Candidate Statistics

- [ ] 2.1 Add a VIMPO FSDP actor/statistics path that returns sampled log-probabilities,
  global policy top-k token IDs/log-probabilities, and retained mass without exporting full
  logits.
- [ ] 2.2 Implement tensor-parallel global top-k and full-vocabulary log-normalization using
  explicit process groups and correct global vocabulary offsets.
- [ ] 2.3 Preserve sequence-parallel, padded, and packed-tree response-position alignment by
  reusing the existing multi-candidate mapping utilities.
- [ ] 2.4 Add CPU/fake-process-group tests and hardware-gated distributed tests for actor
  candidate statistics, masks, TP selection, SP gathering, and packed-tree alignment.

## 3. Frozen SGLang Reference Scoring

- [ ] 3.1 Add a reference candidate-scoring interface that accepts per-position actor token
  IDs and returns sampled-token plus candidate reference log-probabilities.
- [ ] 3.2 Implement the SGLang adapter using batched prefix-state scoring and radix-cache
  reuse, with bounded concurrency, timeouts, and stable response ordering.
- [ ] 3.3 Validate initial checkpoint/tokenizer/vocabulary identity, temperature 1,
  reference weight immutability, output shapes, and finite log-probabilities.
- [ ] 3.4 Add mocked SGLang contract tests for ragged per-position candidates, multi-turn
  prefixes, failures, and rejection of reference-selected top-k substitution.

## 4. Workflow Metadata and Episode Batching

- [x] 4.1 Add the VIMPO workflow dispatch path without invoking generative-critic value
  annotation or critic training-data construction.
- [x] 4.2 Preserve query ID, episode ID, turn order, final outcome reward, and response masks
  through Node tensorization and reference-stat attachment.
- [x] 4.3 Compute centered rewards across distinct episodes within each query, including
  variable group sizes and single-episode queries.
- [ ] 4.4 Implement episode-atomic PPO minibatch construction balanced by valid token count so
  no multi-turn terminal objective is split.
- [ ] 4.5 Add tests proving one terminal target per multi-turn episode, query-local baselines,
  correct padding behavior, and atomic minibatch membership.

## 5. Combined VIMPO Training Objective

- [ ] 5.1 Add `training/losses/vimpo.py` with episode-level terminal squared-error loss,
  detached VIMPO advantages, PPO clipping, combined coefficient weighting, and metrics.
- [ ] 5.2 Add a VIMPO actor update path that preserves rewards/episode metadata, recomputes
  differentiable sampled-token policy log-probabilities, and performs one combined backward
  and optimizer step.
- [ ] 5.3 Integrate actor candidate collection, frozen SGLang scoring, advantage construction,
  and VIMPO update ordering into `CustomizedPPOTrainer` while leaving existing modes intact.
- [ ] 5.4 Ensure VIMPO does not require a positive PPO KL-reward coefficient, does not create a
  learned critic, and fails before optimizer mutation on missing or invalid reference data.
- [ ] 5.5 Add loss/gradient tests for terminal residuals, episode averaging, KL/reference
  detachment, actor advantage detachment, PPO clipping, coefficient weighting, and fail-loud
  behavior.

## 6. Integration, Observability, and Verification

- [ ] 6.1 Emit distributed VIMPO metrics for candidate KL, retained mass, terminal prediction,
  target/residual/RMSE, raw/normalized advantages, component losses, and combined loss.
- [ ] 6.2 Add a CPU smoke test covering Config -> workflow metadata -> actor/reference stats ->
  advantage -> combined loss with fake FSDP/SGLang components.
- [ ] 6.3 Add a hardware-gated FSDP actor plus frozen SGLang reference one-step integration test
  and document the explicit skip reason when required GPUs/services are unavailable.
- [ ] 6.4 Document VIMPO configuration, the fixed-reference requirement, top-k truncation
  semantics, retained-mass guidance, and FSDP/SGLang deployment expectations.
- [ ] 6.5 Run targeted unit tests, relevant customized tree-search suites, Ruff/pre-commit, and
  `graphify update .`; record any hardware-gated or environment-dependent skips.

## 7. Single-Change Scope Rationale

- [ ] 7.1 Verify during implementation review that configuration, candidate scoring,
  advantage computation, and combined training remain one inseparable VIMPO capability;
  split a follow-up change only for optional native SGLang ragged-endpoint optimization or
  additional training backends.

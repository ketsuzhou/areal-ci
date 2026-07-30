# Comet Design Handoff

- Change: add-vimpo-critic-mode
- Phase: design
- Mode: compact
- Context hash: cee6023eb0a0c577dd28e7ca40b0e421bd663d28f128d188ffdc2890ce0549a9

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/add-vimpo-critic-mode/proposal.md

- Source: openspec/changes/add-vimpo-critic-mode/proposal.md
- Lines: 1-51
- SHA256: e27654e3000b45215fe09db170f99b582f13238a1a0c7dec274f0284986c9efd

```md
## Why

The tree-search trainer currently offers group-relative and learned generative-critic
advantages, but it cannot reproduce VIMPO's critic-free token-level credit assignment or
its policy-implied terminal value objective. Adding VIMPO provides dense credit assignment
without training a separate value network, while allowing the frozen initial reference
policy to run as an inference-only SGLang service instead of consuming FSDP training
memory.

## What Changes

- Add `VIMPO` as a tree-search advantage/training mode without enabling the existing
  generative critic or standalone PPO critic.
- Add validated VIMPO settings for the paper coefficients, GAE-style accumulation, and a
  configurable candidate vocabulary size `k`.
- Compute the actor-selected top-k candidate approximation to
  `KL(policy || reference)` using full-vocabulary-normalized probabilities. Report retained
  policy mass and identify the result as exact KL only when `k` equals the vocabulary size.
- Serve the frozen initial policy through SGLang and add candidate scoring for the actor's
  per-position token IDs.
- Compute detached VIMPO advantages in `core/advantage.py`, and train the FSDP actor with
  one combined terminal value-consistency and PPO actor objective.
- Preserve complete multi-turn episode aggregation through batching and add numerical,
  gradient, integration, and compatibility tests.

## Capabilities

### New Capabilities

- `vimpo-policy-implied-value`: Critic-free VIMPO configuration, SGLang reference scoring,
  top-k candidate KL estimation, token-level advantage computation, and combined FSDP
  actor/value training.

### Modified Capabilities

<!-- None. The existing critic-driven-training-signal capability describes an external
     judge and is intentionally unchanged; VIMPO does not train or invoke that critic. -->

## Impact

- Tree-search configuration and validation in `customized_areal/tree_search/config.py`.
- Advantage computation and workflow dispatch in
  `customized_areal/tree_search/core/advantage.py` and
  `customized_areal/tree_search/core/customized_grouped_workflow.py`.
- FSDP actor statistics, VIMPO loss, batching, and trainer orchestration under
  `customized_areal/tree_search/engine/` and `customized_areal/tree_search/training/`.
- SGLang reference requests through the existing remote inference integration, extended
  to score actor-selected per-position candidates.
- New CPU unit tests plus hardware-gated FSDP/SGLang integration coverage.
- No new dependency, learned critic model, reference weight update, or Megatron/Archon
  support is included in this change.

```

## openspec/changes/add-vimpo-critic-mode/design.md

- Source: openspec/changes/add-vimpo-critic-mode/design.md
- Lines: 1-213
- SHA256: 6ce093e2a8c071b2af765ae9736984fc039aef95d48add8213944fc88ae94e6b

[TRUNCATED]

```md
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

```

Full source: openspec/changes/add-vimpo-critic-mode/design.md

## openspec/changes/add-vimpo-critic-mode/tasks.md

- Source: openspec/changes/add-vimpo-critic-mode/tasks.md
- Lines: 1-82
- SHA256: fa8f49975f064df9039cbe9df44c69f150511aa016396eeb8a64d9346589b084

[TRUNCATED]

```md
## 1. Configuration and Core Mathematics

- [ ] 1.1 Add `AdvantageMode.VIMPO` and paper-oriented VIMPO fields/defaults to
  `customized_areal/tree_search/config.py`, including configurable top-k and explicit
  validation/incompatibility errors.
- [ ] 1.2 Implement masked reverse lambda accumulation, detached distributed whitening,
  and `VIMPOAdvantageComputer` tensor helpers in `core/advantage.py`.
- [ ] 1.3 Implement pure candidate-KL helpers using policy-selected top-k values without
  candidate renormalization, including retained-mass metrics and the full-vocabulary case.
- [ ] 1.4 Add CPU numerical and gradient-isolation tests for configuration, top-k KL,
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

- [ ] 4.1 Add the VIMPO workflow dispatch path without invoking generative-critic value
  annotation or critic training-data construction.
- [ ] 4.2 Preserve query ID, episode ID, turn order, final outcome reward, and response masks
  through Node tensorization and reference-stat attachment.
- [ ] 4.3 Compute centered rewards across distinct episodes within each query, including
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

```

Full source: openspec/changes/add-vimpo-critic-mode/tasks.md

## openspec/changes/add-vimpo-critic-mode/specs/vimpo-policy-implied-value/spec.md

- Source: openspec/changes/add-vimpo-critic-mode/specs/vimpo-policy-implied-value/spec.md
- Lines: 1-157
- SHA256: cc92041f4c846474e0c4bc42afc913c826689ec6488838faa0c1984b10e49290

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: VIMPO mode is critic-free and explicitly configured
The tree-search configuration SHALL expose `VIMPO` as an advantage/training mode with
validated settings for beta, actor coefficient, gamma, lambda, candidate top-k size,
detached KL, advantage whitening, and terminal value-loss weight. VIMPO mode MUST NOT
instantiate or train the existing generative critic or a standalone PPO critic.

#### Scenario: Valid paper-oriented configuration
- **WHEN** a user selects VIMPO with positive beta, non-negative actor and value
  coefficients, `gamma=1`, lambda in `[0, 1]`, and positive top-k
- **THEN** configuration succeeds and preserves those values for workflow and training

#### Scenario: VIMPO conflicts with generative critic
- **WHEN** VIMPO mode and `enable_generative_critic=True` are configured together
- **THEN** configuration fails with a message explaining that VIMPO is critic-free

#### Scenario: Existing modes remain unchanged
- **WHEN** TREE, GAE, HYBRID_GAE, or VERSIONED_BACKUP is selected
- **THEN** the system follows its existing advantage and training path without invoking
  VIMPO reference scoring or loss code

### Requirement: Reference policy is the frozen initial model served by SGLang
VIMPO SHALL use a reference policy loaded from the actor's initial checkpoint with the
same tokenizer and vocabulary. The reference SHALL run as an inference-only SGLang
service at temperature 1 and MUST NOT receive actor weight updates during training.

#### Scenario: Reference identity is validated
- **WHEN** VIMPO initializes against an SGLang reference
- **THEN** the system verifies checkpoint/tokenizer identity and vocabulary compatibility
  before the first optimization step

#### Scenario: Reference remains frozen
- **WHEN** one or more actor updates complete
- **THEN** subsequent reference requests use the same initial reference weight version

#### Scenario: Missing reference fails loudly
- **WHEN** VIMPO training begins without a reachable and compatible reference service
- **THEN** the training step fails before applying an optimizer update

### Requirement: SGLang scores actor-selected candidates
The reference adapter SHALL accept actor-selected token IDs for each valid response
position and SHALL return full-softmax-normalized reference log-probabilities for those
IDs plus the sampled token. It MUST NOT substitute the reference model's own top-k list or
renormalize probabilities within the candidate subset.

#### Scenario: Per-position candidates are scored
- **WHEN** different response positions provide different candidate token IDs
- **THEN** every returned reference row is aligned to the IDs supplied for that position

#### Scenario: Prefix scoring preserves alignment
- **WHEN** multi-turn response prefixes are submitted in a batched scoring request
- **THEN** returned rows align with the original episode, turn, token position, and mask

#### Scenario: Reference scoring is incomplete
- **WHEN** any valid candidate lacks a finite reference log-probability
- **THEN** the system rejects the VIMPO batch instead of inserting a sentinel probability

### Requirement: Candidate KL uses policy top-k and full-vocabulary normalization
For every valid response token, the FSDP actor SHALL select the current policy's global
top-k token IDs and compute their probabilities with the full-vocabulary normalizer. The
system SHALL compute the unrenormalized candidate approximation
`sum_a pi(a|s) * (log pi(a|s) - log pi_ref(a|s))` over those IDs. Configured `k` SHALL be
capped at vocabulary size, and `k == vocabulary_size` SHALL recover exact forward KL.

#### Scenario: Candidate KL matches a hand calculation
- **WHEN** policy and reference logits and a top-k value are supplied for a small vocabulary
- **THEN** computed KL equals the unrenormalized sum over policy-selected candidates using
  each distribution's full-vocabulary log-normalizer

#### Scenario: Full vocabulary recovers exact KL
- **WHEN** configured top-k equals or exceeds vocabulary size
- **THEN** candidate KL equals brute-force `KL(policy || reference)` within numeric tolerance

#### Scenario: Tensor-parallel candidates are global
- **WHEN** vocabulary logits are partitioned across FSDP tensor-parallel ranks
- **THEN** selected IDs and probabilities equal top-k selection over the combined vocabulary

#### Scenario: Retained mass is observable
- **WHEN** candidate KL is computed with `k < vocabulary_size`

```

Full source: openspec/changes/add-vimpo-critic-mode/specs/vimpo-policy-implied-value/spec.md

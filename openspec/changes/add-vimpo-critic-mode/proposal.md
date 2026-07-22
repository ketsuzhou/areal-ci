## Why

The tree-search trainer currently offers group-relative and learned generative-critic
advantages, but it cannot reproduce VIMPO's critic-free token-level credit assignment or
its policy-implied terminal value objective. Adding VIMPO provides dense credit
assignment without training a separate value network, while allowing the frozen initial
reference policy to run as an inference-only SGLang service instead of consuming FSDP
training memory.

## What Changes

- Add `VIMPO` as a tree-search advantage/training mode without enabling the existing
  generative critic or standalone PPO critic.
- Add validated VIMPO settings for the paper coefficients, GAE-style accumulation, and a
  configurable candidate vocabulary size `k`.
- Compute the actor-selected top-k candidate approximation to `KL(policy || reference)`
  using full-vocabulary-normalized probabilities. Report retained policy mass and
  identify the result as exact KL only when `k` equals the vocabulary size.
- Serve the frozen initial policy through SGLang and add candidate scoring for the
  actor's per-position token IDs.
- Compute detached VIMPO advantages in `core/advantage.py`, and train the FSDP actor
  with one combined terminal value-consistency and PPO actor objective.
- Preserve complete multi-turn episode aggregation through batching and add numerical,
  gradient, integration, and compatibility tests.

## Capabilities

### New Capabilities

- `vimpo-policy-implied-value`: Critic-free VIMPO configuration, SGLang reference
  scoring, top-k candidate KL estimation, token-level advantage computation, and
  combined FSDP actor/value training.

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

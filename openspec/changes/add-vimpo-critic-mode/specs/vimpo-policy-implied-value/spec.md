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
- **THEN** the system verifies checkpoint/tokenizer identity and vocabulary
  compatibility before the first optimization step

#### Scenario: Reference remains frozen

- **WHEN** one or more actor updates complete
- **THEN** subsequent reference requests use the same initial reference weight version

#### Scenario: Missing reference fails loudly

- **WHEN** VIMPO training begins without a reachable and compatible reference service
- **THEN** the training step fails before applying an optimizer update

### Requirement: SGLang scores actor-selected candidates

The reference adapter SHALL accept actor-selected token IDs for each valid response
position and SHALL return full-softmax-normalized reference log-probabilities for those
IDs plus the sampled token. It MUST NOT substitute the reference model's own top-k list
or renormalize probabilities within the candidate subset.

#### Scenario: Per-position candidates are scored

- **WHEN** different response positions provide different candidate token IDs
- **THEN** every returned reference row is aligned to the IDs supplied for that position

#### Scenario: Prefix scoring preserves alignment

- **WHEN** multi-turn response prefixes are submitted in a batched scoring request
- **THEN** returned rows align with the original episode, turn, token position, and mask

#### Scenario: Reference scoring is incomplete

- **WHEN** any valid candidate lacks a finite reference log-probability
- **THEN** the system rejects the VIMPO batch instead of inserting a sentinel
  probability

### Requirement: Candidate KL uses policy top-k and full-vocabulary normalization

For every valid response token, the FSDP actor SHALL select the current policy's global
top-k token IDs and compute their probabilities with the full-vocabulary normalizer. The
system SHALL compute the unrenormalized candidate approximation
`sum_a pi(a|s) * (log pi(a|s) - log pi_ref(a|s))` over those IDs. Configured `k` SHALL
be capped at vocabulary size, and `k == vocabulary_size` SHALL recover exact forward KL.

#### Scenario: Candidate KL matches a hand calculation

- **WHEN** policy and reference logits and a top-k value are supplied for a small
  vocabulary
- **THEN** computed KL equals the unrenormalized sum over policy-selected candidates
  using each distribution's full-vocabulary log-normalizer

#### Scenario: Full vocabulary recovers exact KL

- **WHEN** configured top-k equals or exceeds vocabulary size
- **THEN** candidate KL equals brute-force `KL(policy || reference)` within numeric
  tolerance

#### Scenario: Tensor-parallel candidates are global

- **WHEN** vocabulary logits are partitioned across FSDP tensor-parallel ranks
- **THEN** selected IDs and probabilities equal top-k selection over the combined
  vocabulary

#### Scenario: Retained mass is observable

- **WHEN** candidate KL is computed with `k < vocabulary_size`
- **THEN** the system reports retained policy probability mass and identifies the metric
  as truncated/candidate KL rather than exact KL

### Requirement: VIMPO advantages are token-level, detached, and normalized

The system SHALL compute token TD terms as
`beta * (log_pi - log_pi_ref - stop_gradient(candidate_KL))`, accumulate them backward
with `(gamma * lambda)` weighting, detach the result, and optionally whiten it over all
valid response tokens in the optimization batch. Prompt and padding positions MUST
remain zero and MUST NOT affect normalization statistics.

#### Scenario: Lambda accumulation matches expected values

- **WHEN** a finite sequence of token TD terms and a mask are provided
- **THEN** reverse accumulation produces the hand-calculated lambda advantages at valid
  positions and zeros elsewhere

#### Scenario: Actor advantages carry no gradient

- **WHEN** normalized VIMPO advantages feed the PPO surrogate
- **THEN** no gradient propagates through advantage construction, candidate KL, or the
  reference values

#### Scenario: Distributed normalization uses all valid tokens

- **WHEN** a batch spans multiple data-parallel ranks
- **THEN** whitening statistics are reduced over valid response tokens across the
  configured actor data-parallel group

### Requirement: Terminal value consistency is optimized per complete episode

For outcome-only rewards, the system SHALL center each episode's final reward by the
mean of distinct episodes sharing its query. It SHALL sum differentiable VIMPO token
terms over all valid tokens and turns in the episode, square the residual against the
centered reward, multiply by one half, and average losses by episode.

#### Scenario: Multi-turn episode has one terminal residual

- **WHEN** one episode contains multiple tree-search Nodes/turns
- **THEN** all valid token terms contribute to one episode sum and the final reward
  target is applied exactly once

#### Scenario: Reward baseline is query-local

- **WHEN** an optimization batch contains episodes from multiple queries
- **THEN** each centered reward uses only distinct episodes from the same query

#### Scenario: Episode survives minibatching

- **WHEN** PPO minibatches are constructed for variable-length multi-turn episodes
- **THEN** all turns from an episode remain together for differentiable terminal
  aggregation

#### Scenario: Single-episode query

- **WHEN** a query has one accepted episode
- **THEN** its centered reward target is zero and its cumulative policy/reference term
  is still regularized toward zero

### Requirement: Actor and value objectives share one optimizer update

The FSDP training step SHALL combine terminal value loss and the
actor-coefficient-weighted PPO clipped loss before a single backward pass and optimizer
step. The sampled actor log-probabilities used by the value loss SHALL be
differentiable, while reference log-probabilities, candidate KL, and actor advantages
SHALL be detached.

#### Scenario: Combined loss weighting

- **WHEN** value loss, actor loss, and configured coefficients are known
- **THEN** the backward scalar equals
  `value_loss_weight * value_loss + actor_coeff * actor_loss`

#### Scenario: Value loss trains actor parameters

- **WHEN** centered rewards differ from cumulative policy/reference terms
- **THEN** value loss produces finite non-zero gradients on actor parameters and no
  gradients on reference data

#### Scenario: Reward-bearing loss cannot be skipped

- **WHEN** VIMPO inputs have incompatible shapes, non-finite values, or missing episode
  data
- **THEN** the update raises an error and does not silently continue with actor-only
  training

### Requirement: VIMPO training exposes diagnostic metrics

The system SHALL report value loss, PPO actor loss, combined loss, candidate KL,
retained policy mass, terminal prediction and target, terminal residual/RMSE, and raw
and normalized advantage statistics using valid masks and distributed aggregation where
applicable.

#### Scenario: Successful update reports diagnostics

- **WHEN** a VIMPO optimizer step succeeds
- **THEN** all required diagnostic metrics are emitted with finite values and
  denominators matching their episode- or token-level definitions

## MODIFIED Requirements

### Requirement: Entropy recording from proxied LLM traffic

AReaL's experimental openai-proxy SHALL capture logprobs from proxied LLM traffic by
requesting `logprobs=true` on all calls, so entropy is computable at session end and at
checkpoint-eligible tool-call decisions. This SHALL be transparent to multica — multica's
pi runtime does not need to request logprobs or know about entropy.

If the upstream LLM does not support logprobs, the proxy SHALL log the absence and
continue. Entropy-gated checkpoint creation SHALL be skipped when logprobs are
unavailable, while always-event checkpoint behavior remains available.

#### Scenario: Logprobs captured for entropy

- **WHEN** a trained session's LLM traffic flows through the proxy
- **THEN** the proxy requests `logprobs=true` on each call and stores the logprobs for
  later entropy computation

#### Scenario: Upstream LLM doesn't support logprobs

- **WHEN** the upstream LLM returns an error or empty result for `logprobs=true`
- **THEN** the proxy logs the absence and continues without entropy for that interaction
  (the trajectory remains valid)

#### Scenario: Missing logprobs skip entropy-gated checkpoint

- **WHEN** a tool-call decision has no usable logprobs
- **THEN** AReaL does not create an entropy-gated checkpoint for that decision and does
  not fail the trained rollout solely because entropy is unavailable

#### Scenario: High entropy can request checkpoint creation

- **WHEN** a checkpoint-eligible tool-call decision has entropy above the configured
  threshold
- **THEN** AReaL can request Multica checkpoint creation with checkpoint kind
  `entropy_gated` and the computed entropy score

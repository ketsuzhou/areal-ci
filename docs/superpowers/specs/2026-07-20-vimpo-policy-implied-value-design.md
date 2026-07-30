---
comet_change: add-vimpo-critic-mode
role: technical-design
canonical_spec: openspec
---

# VIMPO Policy-Implied Value Technical Design

## 1. Design boundary

The canonical behavior and acceptance scenarios are defined by the OpenSpec change
`add-vimpo-critic-mode`. This document refines that specification into concrete AReaL
interfaces, tensor layouts, distributed operations, batching rules, and tests.

VIMPO is not a learned critic. It adds a policy-implied terminal value objective to the
FSDP actor and derives detached token advantages from the same fixed-reference
recurrence. The existing generative critic, PPO critic, critic prompt, critic regression
step, and critic checkpoint lifecycle are outside this path.

The first implementation supports:

- the customized tree-search Node workflow;
- an FSDP actor;
- an independently deployed, frozen SGLang reference loaded from the actor's initial
  checkpoint;
- actor-selected configurable top-k candidate KL;
- multi-turn episodes whose turns remain separate model sequences.

It does not add launcher allocation logic or update the SGLang reference. The reference
endpoint is supplied through the tree-search configuration and must already be
reachable.

## 2. Mathematical contract

For episode `e`, let `M_e` be its ordered valid response-token positions across all
turns. The reference policy is fixed at the initial actor checkpoint, `pi_ref = pi_0`.
At a snapshot policy position `t`, the actor selects `C_t = TopK(pi_snapshot(.|s_t), k)`
and computes

```text
KL_k(t) = sum_{a in C_t} pi_snapshot(a|s_t)
          * (log pi_snapshot(a|s_t) - log pi_ref(a|s_t)).
```

The probabilities on both sides use their full-vocabulary softmax normalizers. The
candidate probabilities are not renormalized. `KL_k` is a candidate/truncated estimate
when `k < vocab_size`; only `k == vocab_size` is exact forward KL.

The detached step statistic and reverse-lambda advantage are

```text
d_t = beta * (log pi_snapshot(a_t|s_t)
              - log pi_ref(a_t|s_t)
              - stop_gradient(KL_k(t)))

A_t = d_t + gamma * lambda * A_next,
```

where `A_next` follows the next valid response token, including the boundary between two
turns of the same episode. The recurrence resets at episode boundaries. `A_t` is
detached, optionally whitened over valid response tokens across the actor data-parallel
group, and then used by the PPO clipped surrogate.

For query `q`, the terminal target for each distinct episode is

```text
y_e = R_e - mean_{j in episodes(q)} R_j.
```

During the training forward, sampled-token actor log-probabilities are recomputed with
gradients. The policy-implied terminal prediction and loss are

```text
z_e(theta) = sum_{t in M_e} beta * (
    log pi_theta(a_t|s_t) - stop_gradient(log pi_ref(a_t|s_t) + KL_k(t))
)

L_V = mean_e 0.5 * (z_e(theta) - y_e)^2
L_total = value_loss_weight * L_V + actor_coeff * L_PPO.
```

The same differentiable sampled-token log-probability tensor feeds both terms. Reference
scores, candidate KL, old/proximal log-probabilities, and advantages never receive
gradients. One engine `train_batch` call performs one zero-grad, forward/backward
sequence, and optimizer step for the combined scalar.

## 3. Configuration

`customized_areal/tree_search/config.py` adds `AdvantageMode.VIMPO = "vimpo"` and the
following backward-compatible fields to `Config`:

| Field                       | Default | Validation and meaning                         |
| --------------------------- | ------: | ---------------------------------------------- |
| `vimpo_beta`                |  `5e-4` | finite and greater than zero                   |
| `vimpo_actor_coeff`         |  `5e-3` | finite and non-negative                        |
| `vimpo_value_loss_weight`   |   `1.0` | finite and non-negative                        |
| `vimpo_gamma`               |   `1.0` | version 1 requires exactly `1.0`               |
| `vimpo_lambda`              |   `1.0` | finite in `[0, 1]`                             |
| `vimpo_top_k`               |   `128` | positive; capped to runtime vocabulary size    |
| `vimpo_whiten_advantages`   |  `True` | enable global masked whitening                 |
| `vimpo_detach_kl`           |  `True` | version 1 rejects `False`                      |
| `vimpo_ref_base_url`        |    `""` | required non-empty HTTP endpoint in VIMPO mode |
| `vimpo_ref_timeout`         | `300.0` | finite and greater than zero, seconds          |
| `vimpo_ref_max_concurrency` |     `8` | positive request bound                         |
| `vimpo_ref_max_retries`     |     `3` | non-negative transient retry count             |

The actor's initial model path and tokenizer are read from the normal trainer/actor
config; they are not duplicated as a second source of truth in `Config`.

VIMPO validation additionally requires:

- `loss_mode == LossMode.GRPO`;
- `enable_generative_critic is False`;
- `use_clip_cov is False`;
- an FSDP actor backend;
- at least one of the two VIMPO loss coefficients is positive.

Muon remains permitted because it changes the actor optimizer rather than the loss
semantics. Existing modes do not validate or consume VIMPO fields.

## 4. Canonical token coordinates and batch tensors

All VIMPO tensors use next-token prediction coordinates. For a padded sequence of length
`S`, row `p` is the model logit at `input_ids[p]` predicting `input_ids[p + 1]`. Tensors
retain width `S` for compatibility with the existing PPO data path, and position `S - 1`
is always invalid.

```python
predict_mask = torch.roll(loss_mask.bool(), shifts=-1, dims=-1)
predict_mask[:, -1] = False
```

Prompt, padding, and terminal non-prediction rows are zeroed and excluded from
reductions. This convention removes the ambiguity between response-aligned fields and
full-sequence fields already present in `_node_to_tensor_dict`.

The VIMPO actor attaches these tensors to each batch:

| Key                        | Shape       | Dtype     | Meaning                                          |
| -------------------------- | ----------- | --------- | ------------------------------------------------ |
| `vimpo_predict_mask`       | `[B, S]`    | `bool`    | canonical valid-token mask                       |
| `vimpo_sample_logp`        | `[B, S]`    | `float32` | detached snapshot sampled-action log-probability |
| `vimpo_candidate_ids`      | `[B, S, K]` | `int64`   | actor global top-k IDs; `-1` outside mask        |
| `vimpo_candidate_logp`     | `[B, S, K]` | `float32` | actor full-softmax candidate log-probabilities   |
| `vimpo_ref_sample_logp`    | `[B, S]`    | `float32` | frozen reference sampled-action log-probability  |
| `vimpo_ref_candidate_logp` | `[B, S, K]` | `float32` | frozen reference scores for the same IDs         |
| `vimpo_candidate_kl`       | `[B, S]`    | `float32` | detached unrenormalized candidate KL             |
| `vimpo_retained_mass`      | `[B, S]`    | `float32` | sum of actor probabilities in `C_t`              |
| `advantages`               | `[B, S]`    | `float32` | detached, optionally whitened VIMPO advantage    |
| `vimpo_query_index`        | `[B, S]`    | `int64`   | repeated compact query index                     |
| `vimpo_episode_index`      | `[B, S]`    | `int64`   | repeated compact episode index                   |
| `vimpo_turn_index`         | `[B, S]`    | `int64`   | repeated one-based turn index                    |
| `vimpo_centered_reward`    | `[B, S]`    | `float32` | repeated episode terminal target                 |

Group metadata is repeated along `S` because AReaL's generic padded splitter only splits
sequence-shaped tensors. Loss code reads the first valid element for sequence-level
metadata and verifies that the value is constant across the sequence. Compact integer
indices are deterministic within one optimization batch; strings remain in Node storage
and diagnostic error messages but do not cross the tensor RPC boundary.

For every valid `(b, p)`, `input_ids[b, : p + 1]` is the reference prefix and
`input_ids[b, p + 1]` is the sampled action. A stable scoring key
`(batch_row, prediction_position)` preserves ordering across concurrent HTTP requests.

## 5. Components and interfaces

### 5.1 Workflow metadata and terminal targets

`customized_areal/tree_search/core/tree_store.py` extends `_node_to_tensor_dict` only
for VIMPO mode. It emits compact query/episode/turn tensors and the raw episode reward
while preserving the current output for all other modes.

`TreeSearchGroupedRolloutWorkflow` adds a VIMPO dispatch branch. It does not call
`_annotate_critic_values` and does not attach `critic_train_data`. Before tensorization
it:

1. groups Nodes by `(query_id, episode_id)`, falling back to `node_id` only for a
   missing episode identifier;
1. verifies all Nodes in one episode have the same final outcome reward;
1. orders turns by `(turn_idx, stable input order)` and rejects duplicate turn indices;
1. computes `R_e - mean(R)` over distinct episodes in the same query;
1. assigns deterministic compact query and episode indices;
1. materializes one centered target for every Node in the episode.

No standard tree/GAE advantage is written at this stage. VIMPO advantages require actor
and reference log-probabilities and are computed later on the training actor.

### 5.2 Actor candidate statistics

`customized_areal/tree_search/engine/fsdp_engine.py` adds a VIMPO no-grad forward helper
with a transport-neutral return contract:

```python
@dataclass(frozen=True)
class VIMPOCandidateStats:
    sampled_logp: torch.Tensor       # [B, S]
    candidate_ids: torch.Tensor      # [B, S, K]
    candidate_logp: torch.Tensor     # [B, S, K]
    retained_mass: torch.Tensor      # [B, S]
    predict_mask: torch.Tensor       # [B, S]

def compute_vimpo_candidate_stats(
    self,
    data: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[VIMPOCandidateStats]: ...
```

The forward runs in evaluation mode under `torch.no_grad()`. It uses temperature `1.0`
regardless of rollout sampling temperature.

For a vocabulary shard `[vocab_start, vocab_end)`, each model-parallel rank computes:

1. the row maximum, followed by an explicit max reduction over the tensor-parallel
   group;
1. the shifted exponential sum, followed by a sum reduction, to obtain the global log
   normalizer;
1. local top-k logits and IDs with the global vocabulary offset;
1. an all-gather of at most `K` values and IDs per rank, followed by a second top-k to
   obtain the global actor candidate set;
1. sampled-token logits from the owning shard, combined by a sum reduction.

When logits are not vocabulary-sharded, the same helper executes without collectives.
Every collective receives `parallel_helper.tp_group` explicitly. Sequence-parallel
output is gathered before restoring `[B, S, ...]` coordinates. Packed-tree output reuses
the existing trie sequence mapping so shared-prefix logits may be evaluated once but are
scattered back to every logical sequence position.

`K = min(vimpo_top_k, vocab_size)`. Top-k tie ordering is made deterministic by token ID
for test reproducibility. Candidate tensors stay on the actor worker until converted to
the normal batch transport; full vocabulary logits are never returned.

### 5.3 Frozen SGLang reference scorer

`customized_areal/tree_search/training/vimpo_reference.py` defines:

```python
@dataclass(frozen=True)
class ReferenceScoreRequest:
    key: tuple[int, int]
    prefix_ids: list[int]
    sampled_token_id: int
    candidate_token_ids: list[int]

@dataclass(frozen=True)
class ReferenceScore:
    key: tuple[int, int]
    sampled_logp: float
    candidate_logp: list[float]

class VIMPOReferenceScorer(Protocol):
    def validate_identity(self, expected: ReferenceIdentity) -> None: ...
    def score(self, requests: list[ReferenceScoreRequest]) -> list[ReferenceScore]: ...
    def close(self) -> None: ...
```

`SGLangVIMPOReferenceScorer` implements the confirmed generic prefix-state approach. One
logical request represents one valid response position. It sends the prefix plus the
union of the sampled token and actor-selected candidates through SGLang's token-ID
log-probability scoring facility. Returned values are full-softmax-normalized by SGLang.
The sampled token is deduplicated when it is already in the actor top-k, but candidate
output is restored to the original ordered `K` IDs.

Requests are sorted by prefix length, issued with a semaphore bounded by
`vimpo_ref_max_concurrency`, and restored by `key`, never by completion order. Prefixes
from the same episode naturally share initial token sequences, allowing the SGLang radix
cache to reuse prefix states. Transient connection, timeout, HTTP 429, and HTTP 5xx
failures use bounded retries. Schema, identity, alignment, and non-finite-value failures
are permanent and are not retried.

At actor initialization, the scorer queries SGLang model information and compares the
canonical model path/revision, vocabulary size, tokenizer vocabulary size, and special
token IDs with the actor's initial configuration. The service must report
temperature-one log-probabilities and must not use quantized scoring in paper-faithful
mode. The actor never calls a reference weight-update endpoint. A cached identity is
checked again after a transport reconnect, so accidentally repointing the URL fails
before training resumes.

If the SGLang version limits token-ID lists, the adapter chunks candidate IDs for the
same prefix and joins them by token ID. This preserves the mathematical contract. It
makes `k == vocab_size` expensive but exact rather than silently changing semantics.

Only the head rank of each actor model-parallel replica performs HTTP scoring. It
converts the ordered results into tensors and broadcasts them to the other TP/SP ranks
using the explicit model-parallel process group. Different data-parallel replicas score
only their own samples.

### 5.4 Candidate KL and advantages

`customized_areal/tree_search/core/advantage.py` adds pure tensor helpers and
`VIMPOAdvantageComputer`; it does not depend on Node or SGLang classes.

```python
def candidate_forward_kl(
    policy_candidate_logp: torch.Tensor,
    reference_candidate_logp: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...

def masked_episode_reverse_lambda(
    td: torch.Tensor,
    mask: torch.Tensor,
    episode_index: torch.Tensor,
    turn_index: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> torch.Tensor: ...

class VIMPOAdvantageComputer:
    def compute(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
```

`candidate_forward_kl` computes in float32 even under lower-precision model execution.
It returns `KL_k` and retained mass. It verifies identical `[B, S, K]` shapes, finite
valid values, unique valid candidate IDs, and zero contribution outside the mask.

The reverse-lambda helper constructs each episode's logical order from `turn_index` and
prediction position. It concatenates valid response positions across turns, scans
backward, and scatters results to `[B, S]`. It rejects repeated episode/turn rows and
episodes whose turn sequence is incomplete within the supplied batch.

Masked whitening computes global `count`, `sum`, and `sum_sq` in float64 accumulators
and all-reduces them over the actor data-parallel group. It uses population variance and
an epsilon of `1e-8`. With zero valid tokens it raises; with one valid token or zero
variance it returns zero on valid positions. The final advantage tensor is detached
explicitly.

### 5.5 VIMPO actor integration

`customized_areal/tree_search/training/actor.py` adds a dedicated
`VIMPOFSDPPPOActor(MultiCandidateFSDPEngine)`. It composes the existing `PPOActor` but
does not install a global monkey patch.

Its `compute_advantages(data)` performs, in order:

1. validate episode metadata and the frozen reference identity;
1. run `compute_vimpo_candidate_stats` on the FSDP actor;
1. create keyed prefix requests for every valid position;
1. score them with `SGLangVIMPOReferenceScorer` and attach aligned tensors;
1. compute candidate KL, retained mass, detached reverse-lambda advantages, and
   whitening;
1. return the enriched batch expected by `ppo_update`.

Candidate collection and reference scoring happen once per PPO step before any optimizer
mutation. The statistics are deliberately a detached snapshot. If PPO uses multiple
optimizer minibatches, the candidate set is not refreshed between them; the metric
`vimpo/snapshot_policy_version` records the actor version used.

Its `ppo_update(data)` validates the complete batch before calling the engine, builds
episode-atomic PPO minibatches, and invokes `train_batch` with the VIMPO loss. It never
calls the normal actor `compute_advantages`, so PPO KL reward shaping and the
`kl_ctl > 0` reference-engine path are bypassed. `kl_ctl` may be zero.

`CustomizedPPOTrainer._create_train_engine` selects this class only for VIMPO and copies
the validated VIMPO settings into the actor config. It rejects non-FSDP allocation
before actor construction. The trainer does not create `self.ref`, `self.critic`, or
generative-critic patches for VIMPO. The existing base loop remains intact:

```text
rollout batch
  -> optional chosen-token proximal log-probability recomputation
  -> VIMPOFSDPPPOActor.compute_advantages
  -> VIMPOFSDPPPOActor.ppo_update
  -> ordinary actor weight publication to rollout servers
```

Only rollout servers receive actor updates. The configured reference URL is excluded
from weight publication and version advancement.

## 6. Episode-atomic minibatching

VIMPO cannot use the generic sequence allocator unchanged because it may split turns
from one episode across engine microbatches and square partial sums.

`customized_areal/tree_search/training/vimpo_batching.py` adds an allocator operating on
episode groups:

```python
def split_episode_atomic_batches(
    data: dict[str, torch.Tensor],
    mb_spec: MicroBatchSpec,
    *,
    episode_key: str = "vimpo_episode_index",
) -> MicroBatchList: ...
```

The allocator:

1. verifies each sequence belongs to exactly one episode;
1. groups all rows sharing an episode index;
1. computes each group's attention-token cost;
1. sorts groups by descending cost with episode index as the stable tie-breaker;
1. greedily places each group into the currently lightest allowed minibatch;
1. preserves all tensor keys and records normal forward/backward indices;
1. rejects an episode whose token cost exceeds `max_tokens_per_mb` rather than splitting
   it.

The actor-level PPO minibatch allocation and the FSDP engine's internal microbatch
allocation both use this helper. For packed-tree training, allocation occurs before trie
packing: each selected set of complete episodes is passed independently to the existing
packed-tree builder. Thus prefix sharing is retained within a microbatch without
allowing the packer to move a turn across episode boundaries.

An outer PPO minibatch contains one or more complete episodes. One `engine.train_batch`
call may contain multiple internal microbatches, but every episode occurs wholly within
one of them. The engine computes a globally weighted episode mean: each microbatch
contributes the sum of complete episode losses divided by the all-reduced episode count.
Gradients accumulate over internal microbatches, followed by one optimizer step.

## 7. Combined loss implementation

`customized_areal/tree_search/training/losses/vimpo.py` contains only tensor math:

```python
def vimpo_loss_fn(
    logprobs: torch.Tensor,
    data: dict[str, torch.Tensor],
    *,
    beta: float,
    eps_clip: float,
    eps_clip_higher: float | None,
    actor_coeff: float,
    value_loss_weight: float,
) -> torch.Tensor: ...
```

The FSDP engine gathers only sampled-action log-probabilities during the train forward.
The loss function:

1. aligns them to `vimpo_predict_mask`;
1. computes the standard PPO ratio against `prox_logp` when present, otherwise rollout
   `logprobs`;
1. applies the existing asymmetric clipping bounds to the detached VIMPO advantages;
1. sums `beta * (current_logp - ref_sample_logp - candidate_kl)` by episode;
1. reads exactly one centered reward per episode and checks all repeated copies agree;
1. computes one half squared residual per complete episode;
1. combines the distributed episode mean and valid-token PPO mean using configured
   weights.

The loss uses float32 reductions. It does not backpropagate through candidate KL,
reference tensors, advantages, old log-probabilities, or targets. The FSDP engine's
normal loss multiplier remains responsible for data-parallel gradient averaging; loss
weights use valid-token count for PPO and complete-episode count for terminal loss.

Because the existing generic `loss_weight_fn` exposes one scalar denominator, the VIMPO
engine path supplies a specialized process-output callback with two all-reduced
denominators. It must not approximate episode averaging with token weighting.

## 8. Failure ordering and lifecycle

All semantic validation completes before `optimizer_zero_grad` and before model train
mode is entered. Failures include:

- missing or empty query/episode identity;
- non-contiguous or duplicate turn order;
- inconsistent rewards within an episode;
- an episode split across a supplied minibatch;
- unreachable or changed reference identity;
- vocabulary/tokenizer/model mismatch;
- reference-selected rather than actor-selected candidate IDs;
- missing, duplicate, reordered, non-finite, or out-of-range reference rows;
- incompatible tensor shapes or masks;
- an oversized episode that cannot fit the configured token limit.

Network retries finish before optimizer mutation. Once the FSDP forward/backward starts,
any exception aborts the update through the normal engine error path; no actor-only
fallback or partial VIMPO update is allowed.

The scorer owns its HTTP client and closes it from the VIMPO actor destruction path. It
stores no model weights. Checkpoints therefore contain only the actor and optimizer
state; on recovery, the same reference identity is revalidated before the first resumed
update.

## 9. Metrics

Metrics use the existing `stats_tracker` and explicit denominators:

- token-level: `vimpo/candidate_kl`, `vimpo/retained_mass`, `vimpo/raw_advantage`,
  `vimpo/normalized_advantage`;
- episode-level: `vimpo/terminal_prediction`, `vimpo/terminal_target`,
  `vimpo/terminal_residual`, `vimpo/terminal_rmse`;
- scalar losses: `vimpo/value_loss`, `vimpo/ppo_actor_loss`, `vimpo/combined_loss`;
- service/quality: `vimpo/reference_latency_ms`, `vimpo/reference_retries`,
  `vimpo/effective_top_k`, `vimpo/exact_kl`, `vimpo/snapshot_policy_version`.

`vimpo/exact_kl` is `1` only when effective `K == vocab_size`. Metric names and logs use
“candidate KL” otherwise. Retained mass includes mean, minimum, and lower quantiles so a
too-small `k` is visible without synchronizing individual tensors to CPU in the hot
path.

## 10. Verification strategy

### 10.1 CPU numerical tests

Add focused tests under `customized_areal/tree_search/tests/` for:

- valid defaults, string enum round-trip, every validation error, and unchanged existing
  modes;
- candidate KL against a hand-computed small vocabulary;
- no candidate renormalization and actor-selected rather than reference-selected IDs;
- convergence to brute-force forward KL at full vocabulary;
- reverse-lambda accumulation across token and turn boundaries, mask resets, and episode
  isolation;
- whitening with padding, one token, zero variance, and simulated distributed sufficient
  statistics;
- query-local reward centering with variable and single-episode groups;
- terminal loss averaging by episode rather than token;
- PPO clipping and combined coefficient weighting;
- non-zero actor gradients from terminal residuals and absence of gradients through
  reference, KL, target, and advantages.

### 10.2 Reference contract tests

Mock the SGLang HTTP transport to verify:

- different actor candidate IDs at every position;
- sampled-token deduplication without changing candidate order;
- out-of-order request completion restored by stable key;
- multi-turn prefix construction and radix-compatible shared prefixes;
- chunked candidate requests produce the same result as one request;
- bounded concurrency, retryable versus permanent failures, timeouts, and client
  closure;
- identity mismatch and reference endpoint replacement fail before scoring;
- missing/non-finite token scores fail rather than inserting a sentinel.

### 10.3 FSDP and integration tests

Fake-process-group tests cover full log normalization, global top-k with vocabulary
offsets, sampled-token ownership, and explicit process-group usage. Hardware-gated tests
cover tensor/sequence parallel gathering, packed-tree scatter alignment, episode-atomic
packing, and one combined optimizer step.

A CPU smoke test exercises:

```text
Config -> Node tensorization -> fake actor candidates -> fake reference scores
       -> candidate KL -> advantages -> episode batching -> combined loss
```

A separately marked integration test launches or connects to a frozen SGLang reference
and runs one FSDP actor step. It verifies that actor weights change, reference
identity/version does not change, and all required metrics are finite. The test skips
explicitly when the required GPU count or SGLang service is unavailable.

Regression tests run the current tree, GAE, hybrid, versioned-backup, distillation, and
generative-critic suites to prove their dispatch and optimizer behavior are unchanged.

## 11. Implementation sequence and rollback

Implementation proceeds in dependency order: configuration and pure math, metadata and
batching, actor candidate statistics, reference adapter, VIMPO actor/loss integration,
then integration tests and documentation. Each layer has a pure or mocked test boundary
before distributed wiring is added.

Rollback requires selecting any existing advantage mode and stopping the dedicated
reference service. No checkpoint schema migration is required because VIMPO adds no
learned critic parameters and existing actor checkpoints remain valid.

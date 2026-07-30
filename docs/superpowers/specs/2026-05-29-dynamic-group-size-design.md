# Dynamic Group Size for Tree Search Rollout

**Date**: 2026-05-29

## Problem

The current `TreeSearchGroupedRolloutWorkflow` uses a fixed `group_size`: every query
gets the same number of rollout episodes regardless of difficulty. Easy queries waste
compute on redundant episodes, while hard queries do not get enough samples for stable
tree-level reward normalization.

## Final Decisions

This design uses the following decisions:

1. `dynamic_group_size` uses a per-query absolute uncertainty threshold, not a
   batch-relative percentile threshold.
1. A query is discarded only when `n_episodes >= 2` and all sampled episodes have
   identical rewards.
1. Continuous rewards use a clean conjugate Bayesian model with unknown variance:
   Normal-Inverse-Gamma.
1. When `AdvantageMode.TREE` is enabled, the workflow-owned
   `self.tree_advantage_computer.compute(all_nodes)` is the source of truth for
   `advantages` and `returns`. Trainer-side `actor.compute_advantages(...)` must be
   skipped for batches that already contain precomputed tree advantages.

## Solution

Add a `dynamic_group_size` mode that:

1. Samples an initial batch of episodes per query
1. Measures per-query uncertainty from episode rewards, scaled by average step count
1. Iteratively adds episodes while uncertainty remains above a fixed absolute threshold
1. Discards reward-degenerate queries only when there are at least 2 episodes and all
   rewards are identical

## Uncertainty Measure

For each query `q` with `n` sampled episodes, compute:

```text
U(q) = posterior_var(q) * mean_steps(q)
```

`mean_steps(q)` is the average number of turns across all sampled episodes for the
query.

### Binary rewards (`reward_type="binary"`)

Model the Bernoulli success probability with a Beta posterior and uniform prior
`Beta(1, 1)`.

After observing `s` successes in `n` episodes:

```text
alpha = 1 + s
beta = 1 + n - s
posterior_var = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
```

### Continuous rewards (`reward_type="continuous"`)

Use a Normal-Inverse-Gamma prior over unknown mean and variance:

```text
mu | sigma^2 ~ Normal(mu0, sigma^2 / kappa0)
sigma^2 ~ InvGamma(alpha0, beta0)
```

With weak prior defaults:

```text
mu0 = 0
kappa0 = 1e-3
alpha0 = 2.0
beta0 = 1.0
```

Given rewards `r1, ..., rn`, let `r_bar` be the sample mean and
`ss = sum((ri - r_bar)^2)`.

Posterior parameters:

```text
kappa_n = kappa0 + n
mu_n = (kappa0 * mu0 + n * r_bar) / kappa_n
alpha_n = alpha0 + n / 2
beta_n = beta0 + 0.5 * ss + (kappa0 * n * (r_bar - mu0)^2) / (2 * kappa_n)
```

Use the posterior variance of the latent mean as the uncertainty term:

```text
posterior_var = beta_n / ((alpha_n - 1) * kappa_n)
```

This stays well-defined for `n = 1` under the weak prior and avoids the earlier
non-conjugate heuristic.

## Absolute Threshold

After the initial sampling round, stop sampling a query when:

```text
U(q) <= uncertainty_threshold
```

Otherwise, continue sampling one episode at a time until either:

1. `U(q) <= uncertainty_threshold`, or
1. `n_q >= max_group_size`

This rule is purely per-query. It does not depend on batch-relative state and does not
require any workflow-level batch reset mechanism.

## Zero-Variance Discard

Discard a query only when both conditions hold:

1. `n_episodes >= 2`
1. all episode rewards are identical

Single-episode queries are never discarded by this rule.

Rationale: identical rewards across 2 or more episodes imply zero GRPO-style
within-query learning signal, while a single episode is just insufficient evidence.

## Dynamic Sampling Flow

When `dynamic_group_size=True`, `_arun_episode_impl` follows this flow:

```text
1. Check cache -> determine how many fresh episodes are needed for initial_group_size
2. Sample initial_group_size episodes (cache + fresh)
3. Compute U(q)
4. Loop:
   a. If U(q) <= uncertainty_threshold or n_q >= max_group_size: break
   b. Generate 1 more episode
   c. Recompute U(q)
5. If n_episodes >= 2 and all rewards are identical: return None
6. Compute tree advantages if enabled
7. Convert nodes to tensor dict and return
```

## Advantage Ownership in TREE Mode

`TreeAdvantageComputer.compute(all_nodes)` already writes normalized tree advantages and
returns onto the `Node`s before `_nodes_to_batched_tensor_dict(...)` builds the rollout
tensor dict.

Therefore, when `AdvantageMode.TREE` is enabled:

1. the workflow computes `advantages` and `returns`
1. the rollout tensor dict returned from the workflow carries those precomputed values
1. the trainer must not recompute GAE/GRPO advantages via
   `self.actor.compute_advantages(rollout_batch)`

Recommended trainer behavior:

```text
if rollout_batch already contains advantages and returns:
    adv_batch = rollout_batch
else:
    adv_batch = self.actor.compute_advantages(rollout_batch)
```

This keeps trainer logic generic and avoids hard-coding tree-search config details into
the trainer loop.

## Configuration

New fields added to `TreeBackupConfig`:

| Field                   | Type  | Default    | Description                                                                      |
| ----------------------- | ----- | ---------- | -------------------------------------------------------------------------------- |
| `dynamic_group_size`    | bool  | False      | Enable dynamic group size mode                                                   |
| `initial_group_size`    | int   | 4          | Minimum episodes per query before uncertainty check                              |
| `max_group_size`        | int   | 64         | Hard cap on episodes per query                                                   |
| `uncertainty_threshold` | float | 0.05       | Stop sampling when `U(q)` is at or below this value                              |
| `reward_type`           | str   | `"binary"` | `"binary"` for Beta posterior, `"continuous"` for Normal-Inverse-Gamma posterior |

When `dynamic_group_size=True`:

- `initial_group_size` defaults to `group_size` if unset
- `group_size` remains the compatibility entrypoint for initial rollout count
- `max_group_size` is an independent cap

## Constructor Changes

`TreeSearchGroupedRolloutWorkflow.__init__` gains:

- `dynamic_group_size`
- `initial_group_size`
- `max_group_size`
- `uncertainty_threshold`
- `reward_type`

When dynamic mode is enabled:

- `self.initial_group_size = initial_group_size or group_size`
- `self.max_group_size = max_group_size`
- `self.uncertainty_threshold = uncertainty_threshold`
- `self.group_size` remains the initial target used for cache/fresh episode planning

## Files Changed

| File                                                               | Change                                                                                                     |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/config.py`                           | Add dynamic sampling config fields                                                                         |
| `customized_areal/tree_search/core/uncertainty.py`                 | Add Bayesian uncertainty helpers                                                                           |
| `customized_areal/tree_search/core/customized_grouped_workflow.py` | Add iterative sampling loop, discard rule, uncertainty computation                                         |
| `areal/experimental/inference_service/controller/controller.py`    | Wire new tree-search workflow args                                                                         |
| `areal/infra/remote_inf_engine.py`                                 | Wire new tree-search workflow args                                                                         |
| `areal/trainer/rl_trainer.py`                                      | Skip `actor.compute_advantages(...)` when rollout trajectories already contain precomputed tree advantages |
| `tests/customized_areal/test_dynamic_group_size.py`                | Add unit tests for config, uncertainty, discard, dynamic sampling, and trainer integration                 |

## Error Handling

- If a fresh episode fails during iterative sampling, skip it and continue if progress
  is still possible.
- If the initial `initial_group_size` episodes all fail, return `None`.
- If all surviving episodes are discarded by distillation-only filtering, return `None`.
- If `max_group_size < initial_group_size`, raise `ValueError`.
- If `uncertainty_threshold < 0`, raise `ValueError`.

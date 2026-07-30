# Hybrid Monte-Carlo / GAE Advantage with Entropy-TD Branch Selection

**Date**: 2026-06-22

## Problem

`GAEAdvantageComputer` estimates per-turn advantages from a learned critic value
`Node.value` (`V_theta(s_t)`). Early in training `V_theta` is noisy, which biases the
GAE targets whenever `lambda < 1` (the critic is bootstrapped into the target). The
critic-free alternative `TreeAdvantageComputer` (GRPO) is robust to this but only
produces a single outcome-level scalar per episode, with no per-turn credit assignment.

We want a single estimator that keeps GAE's per-turn credit assignment while replacing
or down-weighting the noisy critic where a cheaper, unbiased Monte-Carlo (MC) estimate
of `V(s_t)` is available. The MCTS tree already accumulates exactly such an estimate:
`MCTSTreeStore._backup` maintains per-node visit counts and mean backed-up returns
(`_q_values`), so `get_q_value(node_id)` is an MC estimate of `V(s_t)` and
`_visit_counts[node_id]` is its sample size.

Separately, branching (which produces those MC samples) is currently driven solely by
actor entropy (`select_branch_candidate` ranks by `_max_entropy`). High entropy marks
where the policy is uncertain, but not whether that uncertainty is consequential.
Combining entropy with critic disagreement (TD-error) concentrates branch budget where
MC sampling has the most value.

## Background (verified in code)

- `customized_areal/tree_search/core/advantage.py`
  - `GAEAdvantageComputer.compute` runs the backward recursion
    `delta_t = r_t + gamma*v(s_{t+1}) - v(s_t)`, `A_t = delta_t + gamma*lam*A_{t+1}`,
    `ret_t = A_t + v(s_t)`, with terminal bootstrap `v(s_{T+1}) = 0` and sparse reward
    (only the last turn carries `outcome_reward`, unless `judge_beta > 0`).
  - `_node_value(node)` reads `Node.value`.
- `customized_areal/tree_search/core/tree_store.py`
  - `_backup(node_id, reward)` updates `_visit_counts`, `_total_values`, `_q_values`.
  - Public accessors exist for value/normalized advantage/return and `get_q_value`, but
    there is **no** public visit-count accessor and **no** second moment tracked.
- `customized_areal/tree_search/core/customized_grouped_workflow.py`
  - `_max_entropy(node)` reads `node.entropy_stats["max_entropy"]`.
  - `select_branch_candidate(nodes, query_id)` returns
    `max(eligible, key=_max_entropy)`.
- `Node` carries `value`, `outcome_reward`, `turn_idx`, `episode_id`, `entropy_stats`,
  `need_branch`.

## Final Decisions

1. The hybrid estimator is a new `HybridAdvantageComputer` class in `advantage.py`,
   selected by a new `AdvantageMode.HYBRID`. `GAEAdvantageComputer` and
   `TreeAdvantageComputer` are left unchanged so existing modes are byte-for-byte
   identical.
1. Blending happens at the **value** level, not the advantage-tensor level:
   `v_hat(s_t) = w_t * v_mc(s_t) + (1 - w_t) * v_theta(s_t)`, then the standard GAE
   recursion runs on `v_hat`. This keeps a single, well-defined estimator.
1. The MC value `v_mc(s_t)` is the **leave-one-out** mean of returns backed up through
   the node, to keep the baseline independent of the current episode's action and thus
   gradient-unbiased.
1. The per-node weight `w_t` is the MSE-optimal convex combination of two estimators of
   `V(s_t)`: `w_t = (sigma_theta^2 + b^2) / (sigma_mc^2 + sigma_theta^2 + b^2)` where
   `sigma_mc^2 = Var_return(s_t) / K(s_t)`, `K` is the visit count, and `b^2` is a
   running estimate of critic bias (EMA of mean `|TD-error|`-squared). When
   `K(s_t) < mc_value_min_visits`, `w_t = 0` (pure critic).
1. Branch selection gains an optional `branch_score_mode = "entropy_td"` that ranks
   candidates by a multiplicative `entropy^alpha * |TD|^beta` score. Default remains
   `"entropy"` (no behavioural change).
1. All new behaviour is gated behind config defaults that reproduce current behaviour
   exactly (`advantage_mode` unchanged default, `branch_score_mode="entropy"`).

## Solution

### Value-level blend (HybridAdvantageComputer)

For each episode (grouped by `(query_id, episode_id)`, ordered by `turn_idx`):

```text
for each turn t:
    v_theta_t = Node.value
    K_t       = tree_store.get_visit_count(node_id)
    if K_t >= mc_value_min_visits:
        v_mc_t   = tree_store.get_mc_value_loo(node_id, this_episode_return)
        var_mc_t = tree_store.get_return_variance(node_id) / K_t
        w_t      = (sigma_theta2 + b2) / (var_mc_t + sigma_theta2 + b2)
    else:
        w_t = 0.0
        v_mc_t = v_theta_t
    v_hat_t = w_t * v_mc_t + (1 - w_t) * v_theta_t

# standard GAE on v_hat
delta_t = r_t + gamma * v_hat_{t+1} - v_hat_t
A_t     = delta_t + gamma * lam * A_{t+1}        (A_{T+1} = 0, v_hat_{T+1} = 0)
ret_t   = A_t + v_hat_t

b2 (critic bias proxy) and sigma_theta2 are read from a small running-stats helper
updated each call from observed |v_theta_t - G_t|.

Tree value statistics

MCTSTreeStore._backup is extended to also accumulate the sum of squared returns so a
running variance is available. New public accessors:

def get_visit_count(self, node_id: str) -> int: ...
def get_return_variance(self, node_id: str) -> float: ...   # population variance of backed-up returns
def get_mc_value_loo(self, node_id: str, exclude_return: float) -> float:
    # (total - exclude_return) / (K - 1), falls back to get_q_value when K < 2

Entropy x TD-error branch selection

select_branch_candidate gains a scoring mode. For "entropy_td":

score(node) = (max_entropy(node) + eps_h) ** alpha
            * (abs(v_theta(node) - mc_value(node)) + eps_td) ** beta

mc_value uses get_q_value (the backed-up return mean) as the cheap proxy for the
realized return when a node already has children; for never-branched nodes the TD term
falls back to a neutral constant so the score reduces to entropy ranking. This realizes
two-stage progressive widening: entropy proposes, TD refines once samples exist.

Configuration

New fields on TreeBackupConfig:

┌──────────────────────┬───────┬─────────────┬─────────────────────────────────────────────────────────────┐
│ Field                │ Type  │ Default     │ Description                                                 │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ advantage_mode       │ enum  │ (unchanged) │ Add HYBRID variant selecting HybridAdvantageComputer        │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ hybrid_mc_min_visits │ int   │ 2           │ Minimum node visit count before MC value is trusted (w_t>0) │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ hybrid_sigma_theta2  │ float │ 1.0         │ Critic variance term in the weight denominator              │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ hybrid_bias_ema      │ float │ 0.99        │ EMA decay for the running critic-bias b^2 estimate          │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ hybrid_bias_init     │ float │ 1.0         │ Initial b^2 (favours MC early when critic is untrained)     │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ hybrid_leave_one_out │ bool  │ True        │ Use leave-one-out MC value to keep the baseline unbiased    │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ branch_score_mode    │ str   │ "entropy"   │ "entropy" (current) or "entropy_td"                         │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ branch_entropy_alpha │ float │ 1.0         │ Exponent on the entropy term in entropy_td mode             │
├──────────────────────┼───────┼─────────────┼─────────────────────────────────────────────────────────────┤
│ branch_td_beta       │ float │ 1.0         │ Exponent on the TD term in entropy_td mode                  │
└──────────────────────┴───────┴─────────────┴─────────────────────────────────────────────────────────────┘

Validation: hybrid_mc_min_visits >= 1, 0 < hybrid_bias_ema < 1,
hybrid_sigma_theta2 >= 0, hybrid_bias_init >= 0,
branch_score_mode in {"entropy", "entropy_td"}, exponents >= 0.

Files Changed

┌──────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
│ File                                                             │ Change                                                            │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
│ customized_areal/tree_search/core/tree_store.py                  │ Track sum-of-squared returns; add visit/variance/LOO accessors    │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
│ customized_areal/tree_search/core/advantage.py                   │ Add HybridAdvantageComputer + running critic-bias helper          │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
│ customized_areal/tree_search/core/customized_grouped_workflow.py │ Add entropy_td branch scoring path                                │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
│ customized_areal/tree_search/config.py                           │ Add hybrid + branch-score config fields and validation            │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
│ customized_areal/tree_search/trainer.py                          │ Select HybridAdvantageComputer when advantage_mode == HYBRID      │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
│ tests/customized_areal/test_hybrid_advantage.py                  │ Unit tests for stats, blend math, GAE equivalence, branch scoring │
└──────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┘

Testing

- get_visit_count / get_return_variance / get_mc_value_loo correctness, incl.

  K<2 fallbacks.

- Blend reduces to plain GAE when w_t == 0 everywhere (all nodes below

  hybrid_mc_min_visits) — byte-for-byte equal to GAEAdvantageComputer.

- Blend reduces toward MC when b^2 is large / critic variance dominates.
- Leave-one-out value excludes the current episode's return.
- Weight formula matches (sigma_theta2 + b2)/(var_mc + sigma_theta2 + b2).
- entropy_td score ranks a high-entropy/high-TD node above a high-entropy/low-TD node;

  "entropy" mode unchanged from current selection.

- Episodic boundary: v_hat(s_{T+1}) = 0, sparse reward unchanged when judge_beta=0.

Error Handling / Backward Compatibility

- Default advantage_mode and branch_score_mode reproduce current behaviour exactly.
- Missing/empty entropy_stats -> entropy term is 0.0 (existing _max_entropy rule).
- Nodes with K < 2 -> MC disabled, pure critic, no division by zero.
- HybridAdvantageComputer shares Node/tree_store write contract with the existing

  computers (advantages, returns, set_normalized_advantage/return).

---
```

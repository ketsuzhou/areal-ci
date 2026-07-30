# Clip-Cov PPO Loss — Design Spec

## Summary

Implement `compute_policy_loss_clip_cov` from
[PRIME-RL/Entropy-Mechanism-of-RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py)
as a standalone monkey-patch module in `customized_areal/clip_cov/`. No changes to
`areal/`.

## Background

Clip-cov is a covariance-aware gradient masking strategy for PPO. It computes per-token
covariance between advantages and log-probabilities, then randomly zeroes gradients for
tokens whose covariance falls within a configurable range `[clip_cov_lb, clip_cov_ub]`.
Tokens already clipped by standard PPO are excluded. This suppresses gradient signal
from tokens with moderate covariance to advantage, which the PRIME-RL paper shows
improves training stability.

## Module Structure

```
customized_areal/clip_cov/
├── __init__.py          # Exports patch function and config
├── config.py            # ClipCovConfig dataclass
├── loss.py              # clip_cov_ppo_actor_loss_fn + clip_cov_grpo_loss_fn
└── patch.py             # patch_ppo_actor_to_use_clip_cov_loss()
```

## Components

### ClipCovConfig (`config.py`)

```python
@dataclass
class ClipCovConfig:
    clip_ratio: float = 0.0002   # Fraction of tokens to zero via cov clipping
    clip_cov_lb: float = 1.0     # Lower bound of covariance range
    clip_cov_ub: float = 5.0     # Upper bound of covariance range
```

Standard PPO params (eps_clip, eps_clip_higher, c_clip) come from PPOActorConfig — not
duplicated here.

### clip_cov_ppo_actor_loss_fn (`loss.py`)

Adapts the reference `compute_policy_loss_clip_cov` to AReaL conventions.

**Signature:**

```python
def clip_cov_ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    clip_ratio: float = 0.0002,
    clip_cov_lb: float = 1.0,
    clip_cov_ub: float = 5.0,
) -> tuple[torch.Tensor, dict]:
```

**Algorithm:**

1. Compute ratio: `ratio = exp(logprobs - proximal_logprobs)` where loss_mask is True,
   else 0
1. Standard PPO clipped loss:
   `pg_loss = max(-advantages * ratio, -advantages * clamp(ratio, 1-eps, 1+eps_higher))`
1. Compute per-token covariance:
   ```
   cov = (advantages - masked_mean(advantages)) * (logprobs - masked_mean(logprobs.detach()))
   ```
1. Mask out tokens from cov selection:
   - Tokens with loss_mask == 0 → set cov to -inf
   - Tokens clipped by standard PPO (pg_losses2 > pg_losses1) → set cov to -inf
1. Select tokens for zeroing:
   - `candidates = (cov > clip_cov_lb) & (cov < clip_cov_ub) & (loss_mask > 0)`
   - Randomly pick at most `max(int(clip_ratio * num_valid), 1)` tokens from candidates
1. Build `corr` tensor (initially all 1s), set selected positions to 0
1. Final loss: `(pg_loss * corr).sum() / loss_mask_count`
1. If c_clip is set, apply dual clipping before step 7 (same as AReaL's
   ppo_actor_loss_fn)

**Return:** `(loss_scalar, stat_dict)` where stat_dict contains:

- `loss`: per-token loss tensor (for logging)
- `importance_weight`: ratio tensor (AReaL-compatible)
- `approx_kl`: logprobs - proximal_logprobs (AReaL-compatible)
- `clip_mask`: PPO clip mask (AReaL-compatible)
- `clip_cov_mask`: boolean tensor showing which tokens were zeroed by clip-cov (new)

### clip_cov_grpo_loss_fn (`loss.py`)

A `grpo_loss_fn`-compatible wrapper that:

1. Resolves proximal log-probs using AReaL's `_resolve_proximal_logp`
1. Calls `clip_cov_ppo_actor_loss_fn` instead of `ppo_actor_loss_fn`
1. Does NOT support SAPO, decoupled loss, or teacher KL distillation branches (clip-cov
   is a standalone loss variant)

**Signature:**

```python
def clip_cov_grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    clip_ratio: float = 0.0002,
    clip_cov_lb: float = 1.0,
    clip_cov_ub: float = 5.0,
    prox_logp_method: str = "recompute",
    current_version: int | None = None,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
```

### patch_ppo_actor_to_use_clip_cov_loss (`patch.py`)

Replaces `PPOActor._ppo_update` with a version that calls `clip_cov_grpo_loss_fn`.

**Pattern:** Same as `customized_areal/on_policy_distill/training/actor.py` — global
`_patch_applied` guard, class-level method replacement.

**Signature:**

```python
def patch_ppo_actor_to_use_clip_cov_loss(config: ClipCovConfig) -> None:
```

**Behavior:**

1. If already patched, return early (idempotent)
1. Define `_ppo_update_with_clip_cov_loss(self, data)` that mirrors
   `PPOActor._ppo_update` but passes `clip_cov_grpo_loss_fn` with `ClipCovConfig` params
1. Replace `PPOActor._ppo_update` with the new method
1. Set `_patch_applied = True`

## Usage

```python
from customized_areal.clip_cov import ClipCovConfig, patch_ppo_actor_to_use_clip_cov_loss

config = ClipCovConfig(clip_ratio=0.0002, clip_cov_lb=1.0, clip_cov_ub=5.0)
patch_ppo_actor_to_use_clip_cov_loss(config)

# Then use PPOTrainer normally — it will use clip-cov loss
```

## Compatibility Notes

- `masked_mean` uses AReaL's convention: `(x * mask).sum() / mask.count_nonzero()`
- Loss is computed on CPU-safe `loss_mask_count = loss_mask.count_nonzero() or 1`
- The `corr` mask is applied as element-wise multiplication to `pg_loss`, not as a
  replacement for `loss_mask`
- Dual clipping (`c_clip`) is supported and applied before `corr` multiplication
- behave_imp_weight (decoupled loss) is NOT supported in clip-cov mode

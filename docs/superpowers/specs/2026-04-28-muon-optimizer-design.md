# Muon Optimizer Integration for AReaL

## Overview

Integrate the [Muon optimizer](https://github.com/KellerJordan/Muon) (momentum +
Newton-Schulz orthogonalization) into AReaL's FSDP Engine, implemented entirely within
`customized_areal/` without modifying the core `areal/` package.

## Background

Muon applies SGD momentum followed by a 5th-order Newton-Schulz iteration to
orthogonalize the update matrix, yielding a zero-power (spectral norm = 1) update
direction. It is designed for 2D weight matrices; non-2D parameters (biases, embeddings,
layer norms) use Adam instead.

AReaL's `FSDPEngine._create_optimizer` currently only supports `adam`, `adam_bf16`, and
`sgd`. We extend it via monkey-patching, following the existing convention in
`customized_areal/tpfc/scripts/gsm8k_rl.py`.

## Decisions

- **Variant**: `MuonWithAuxAdam` — Muon for 2D params, Adam for non-2D params
- **Engine**: FSDP Engine only (no Megatron)
- **Distributed**: Use `SingleDeviceMuonWithAuxAdam` — FSDP2 handles gradient
  synchronization; the original Muon's `dist.all_gather` is designed for DDP and is
  incorrect under FSDP
- **CPU offload**: Not supported (no `MuonKernel` for `PerLayerOptimWrapper`)
- **Integration approach**: Monkey-patch `FSDPEngine._create_optimizer`

## File Structure

```
customized_areal/optimizers/
    __init__.py          # Exports patch_fsdp_engine_for_muon, unpatch_fsdp_engine
    muon.py              # MuonWithAuxAdam optimizer
    patch.py             # Monkey-patch for FSDPEngine._create_optimizer
```

## Component Design

### muon.py — Optimizer Implementation

Adapted from `SingleDeviceMuonWithAuxAdam` in the original Muon repository.

**Core functions:**

- `zeropower_via_newtonschulz5(G, steps=5)`: Computes approximate zero-power
  (orthogonal) matrix via 5 iterations of a bfloat16-friendly cubic polynomial.
  Coefficients: `a=3.4445, b=-4.7750, c=2.0315`. Handles tall matrices by transposing
  before/after iteration.
- `muon_update(grad, momentum_buffer, beta=0.95, ns_steps=5, nesterov=True)`: Computes
  momentum update, then orthogonalizes via Newton-Schulz. Scales by
  `sqrt(max(1, rows/cols))` to account for aspect ratio.

**Optimizer class:**

```python
class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        # param_groups: list of dicts with "use_muon" key
        # Muon group: lr, momentum, weight_decay
        # Adam group: lr, betas, eps, weight_decay
```

**Parameter group splitting** (done at construction time, not inside the optimizer):

- 2D weight matrices (ndim >= 2) → Muon param group
- Biases, embeddings, layer norms (ndim == 1) → Adam param group

**Key differences from the original:**

1. No `dist.all_gather` — FSDP2 manages distributed state
1. No `p.grad = torch.zeros_like(p)` for missing grads — let the engine handle gradient
   accumulation
1. No `assert` on param group key sets — be lenient with extra keys from AReaL's config

### patch.py — FSDPEngine Monkey-Patch

Since `OptimizerConfig` in `areal/` cannot be modified, Muon-specific hyperparameters
(`momentum`, `muon_adam_lr`, `ns_steps`, `nesterov`) are passed as keyword arguments to
`patch_fsdp_engine_for_muon()`. These are stored as module-level variables that the
patched `_create_optimizer` reads at optimizer construction time.

```python
_original_create_optimizer = None
_muon_config = {}  # Module-level config set by patch_fsdp_engine_for_muon()

def patch_fsdp_engine_for_muon(
    momentum: float = 0.95,
    muon_adam_lr: float = 3e-4,
    ns_steps: int = 5,
    nesterov: bool = True,
):
    """Replace FSDPEngine._create_optimizer with Muon-aware version."""
    global _original_create_optimizer, _muon_config
    from areal.engine.fsdp_engine import FSDPEngine
    _original_create_optimizer = FSDPEngine._create_optimizer
    _muon_config = {
        "momentum": momentum,
        "muon_adam_lr": muon_adam_lr,
        "ns_steps": ns_steps,
        "nesterov": nesterov,
    }
    FSDPEngine._create_optimizer = _patched_create_optimizer

def unpatch_fsdp_engine_for_muon():
    """Restore original FSDPEngine._create_optimizer."""
    from areal.engine.fsdp_engine import FSDPEngine
    if _original_create_optimizer is not None:
        FSDPEngine._create_optimizer = _original_create_optimizer
```

**Patched `_create_optimizer` behavior:**

1. If `optimizer_config.type == "muon"`:
   - Split model parameters into 2D (Muon) and non-2D (Adam) groups
   - Create `MuonWithAuxAdam` with two param groups
   - Use `optimizer_config.lr` as Muon learning rate
   - Use `_muon_config["momentum"]` (default 0.95) as Muon momentum
   - Use `_muon_config["muon_adam_lr"]` (default 3e-4) as Adam learning rate for non-2D
     params
   - Use `_muon_config["ns_steps"]` (default 5) and `_muon_config["nesterov"]` (default
     True) for Newton-Schulz
   - Proceed with lr_scheduler creation as normal (scheduler wraps the optimizer)
1. Otherwise: delegate to original `_create_optimizer`

**Additional patch needed**: The original `_create_optimizer` asserts
`type in ["adam", "adam_bf16", "sgd"]`. The patched version must accept `"muon"` as
well. This is handled by the full replacement of the method.

### __init__.py — Public API

```python
from customized_areal.optimizers.patch import patch_fsdp_engine_for_muon, unpatch_fsdp_engine_for_muon
from customized_areal.optimizers.muon import MuonWithAuxAdam
```

## Usage

```python
from customized_areal.optimizers import patch_fsdp_engine_for_muon

# Must be called before engine creation
patch_fsdp_engine_for_muon(
    momentum=0.95,       # Muon momentum
    muon_adam_lr=3e-4,   # Adam lr for non-2D params
    ns_steps=5,          # Newton-Schulz iterations
    nesterov=True,       # Nesterov momentum
)

# In config YAML:
# optimizer:
#   type: muon
#   lr: 0.02
#   weight_decay: 0.01
```

## Hyperparameter Mapping

| Source                                         | Muon usage     | Adam usage   |
| ---------------------------------------------- | -------------- | ------------ |
| `OptimizerConfig.type`                         | "muon"         | "muon"       |
| `OptimizerConfig.lr`                           | Muon lr (0.02) | N/A          |
| `OptimizerConfig.weight_decay`                 | weight_decay   | weight_decay |
| `OptimizerConfig.beta1`                        | N/A            | 0.9          |
| `OptimizerConfig.beta2`                        | N/A            | 0.95         |
| `OptimizerConfig.eps`                          | N/A            | 1e-10        |
| `patch_fsdp_engine_for_muon(momentum=...)`     | 0.95           | N/A          |
| `patch_fsdp_engine_for_muon(muon_adam_lr=...)` | N/A            | 3e-4         |
| `patch_fsdp_engine_for_muon(ns_steps=...)`     | 5              | N/A          |
| `patch_fsdp_engine_for_muon(nesterov=...)`     | True           | N/A          |

Muon-specific hyperparameters are passed via `patch_fsdp_engine_for_muon()` kwargs
(stored as module-level `_muon_config`), not through `OptimizerConfig`. This avoids
modifying `areal/` code.

## Out of Scope

- `MuonKernel` for `PerLayerOptimWrapper` (CPU offload support)
- Megatron engine integration
- Checkpoint save/load modifications (standard `state_dict()` works)
- LR scheduler modifications
- `optimizer_type: "muon"` in the `OptimizerConfig` choices (that would require
  modifying `areal/`)

## Testing

- Unit test: verify `zeropower_via_newtonschulz5` produces near-orthogonal matrices
- Unit test: verify `MuonWithAuxAdam` step produces correct state updates for both param
  groups
- Integration test: verify patched `_create_optimizer` creates `MuonWithAuxAdam` when
  `type="muon"`
- Integration test: verify unpatch restores original behavior

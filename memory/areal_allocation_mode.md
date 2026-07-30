---
name: AReaL Allocation Mode Format
description: How to configure parallelism dimensions in AReaL backend strings
type: reference
---

## Allocation Mode Format

The backend string format is `<backend>:<parallelism_dims>`. Examples:

- `sglang:d1` — SGLang inference with 1 data-parallel replica
- `fsdp:d4` — FSDP training with 4-way data parallelism
- `fsdp:d1c3` — FSDP training with 1 DP + 3-way context/sequence parallelism
- `fsdp:d2c4t2` — FSDP training with 2 DP + 4 SP + 2 TP

### Parallelism Dimensions

| Dimension | Abbreviation | Description                            | FSDP-valid?           |
| --------- | ------------ | -------------------------------------- | --------------------- |
| Data      | `d`          | Number of model replicas (ZeRO)        | Yes                   |
| Tensor    | `t`          | Split ops across GPUs (TP)             | Yes                   |
| Pipeline  | `p`          | Split layers across GPUs               | No (rejected by FSDP) |
| Context   | `c`          | Split sequence length across GPUs (SP) | Yes                   |
| Expert    | `e`          | Split MoE experts across GPUs          | No (rejected by FSDP) |

### Key: Context Parallelism (`c`) = Sequence Parallelism

In AReaL, `c` (context parallelism) IS sequence parallelism. It maps to
`context_parallel_size` in `ParallelStrategy`, then to `sp_size` in `ParallelHelper`.
The engine applies Ulysses-style all-to-all communication to shard the sequence
dimension.

### When to Use SP vs TP

- **TP (`t`)**: Shard model weights across GPUs. Needed when the model is too large for
  one GPU.
- **SP (`c`)**: Shard sequence length across GPUs. Better when the model fits on one GPU
  but sequences are long. Each GPU holds the full model but only a fraction of the
  tokens.

### Config Fields to Update

When changing parallelism, update BOTH:

1. `allocation_mode` — top-level GPU allocation (e.g., `sglang:d1+fsdp:d1c3`)
1. `actor.backend` — per-engine backend string (e.g., `fsdp:d1c3`)

The `ref.backend` typically uses `${actor.backend}` and inherits automatically.

### For VLMs (e.g., Qwen3-VL-8B)

FSDPEngine has a `shard_vision_across_sp: bool` config field (default: false) that
shards the vision encoder across SP ranks. Set to true when using SP with vision models.

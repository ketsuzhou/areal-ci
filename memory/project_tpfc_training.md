---
name: TPFC GRPO Training Project
description: RL training project using TPFC agent workflow with Qwen3-VL-8B-Instruct
type: project
---

Training a Qwen3-VL-8B-Instruct model using GRPO with the TPFC (Tool-augmented Planning
and Function Calling) agent workflow.

**Current setup:**

- Model: `/dfs/share-groups/letrain/ckpt/Qwen3-VL-8B-Instruct`
- Config: `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct.yaml`
- Workflow: `customized_areal.tpfc.tpfc_agent.TPFCAgent`
- Dataset: `customized_areal/tpfc/data/generated_training_final_update.parquet`
- Inference: 1 GPU (SGLang), Training: 3 GPUs (FSDP with sequence parallelism)
- Parallelism: `sglang:d1+fsdp:d1c3`
- SGLang context_length: 40960, mem_fraction_static: 0.7

**Key config details:**

- Uses `qwen25` tool_call_parser and `qwen3` reasoning_parser in SGLang
- Gradient checkpointing enabled, bfloat16 dtype
- reward_scaling: 10.0, reward_bias: -0.5
- eps_clip: 0.4, kl_ctl: 0.0

**Why:** This project trains the model for tool-use and function calling scenarios using
GRPO. The Qwen3-VL model is a vision-language model being trained for multi-modal
reasoning with tool augmentation.

"""Convert FSDP2 DCP checkpoint to HuggingFace safetensors format.

Usage:
    python convert_dcp_to_hf.py \
        --ckpt_path tmp/areal/experiments/checkpoints/root/tpfc-grpo/trial_0525/default/recover_checkpoint \
        --output_path tmp/areal/experiments/checkpoints/root/tpfc-grpo/trial_0525/hf_checkpoint \
        --base_model_path /dfs/share-groups/letrain/ckpt/Qwen3.5-9B
"""

import argparse
import json
import os
import shutil

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader


def load_dcp_model_state(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load model weights from DCP checkpoint (no distributed env needed)."""
    reader = FileSystemReader(ckpt_path)
    metadata = reader.read_metadata()

    # Build placeholder state dict for model keys only
    state_dict = {}
    for key, info in metadata.state_dict_metadata.items():
        if key.startswith("dcp.model."):
            dtype_str = str(info.properties.dtype).replace("torch.", "")
            state_dict[key] = torch.empty(info.size, dtype=getattr(torch, dtype_str))

    print(f"Loading {len(state_dict)} model keys from DCP checkpoint...")

    # Load with no_dist=True to work without distributed env
    dcp.load(state_dict, checkpoint_id=ckpt_path, storage_reader=reader, no_dist=True)

    # Strip "dcp.model." prefix
    hf_state_dict = {}
    for key, tensor in state_dict.items():
        hf_key = key.removeprefix("dcp.model.")
        hf_state_dict[hf_key] = tensor

    return hf_state_dict


def _shard_state_dict(
    state_dict: dict[str, torch.Tensor],
    max_shard_size_bytes: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict | None]:
    """Split state dict into shards and generate index."""
    shards: dict[str, dict[str, torch.Tensor]] = {}
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 1

    for key in sorted(state_dict):
        tensor = state_dict[key]
        tensor_bytes = tensor.numel() * tensor.element_size()

        if current_size + tensor_bytes > max_shard_size_bytes and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            shards[shard_name] = current_shard
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_bytes

    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shards[shard_name] = current_shard

    # Fill in total shard count
    total = len(shards)
    final_shards = {}
    for name, shard in shards.items():
        final_name = name.replace("XXXXX", f"{total:05d}")
        final_shards[final_name] = shard

    # Build index
    weight_map = {}
    for shard_name, shard in final_shards.items():
        for key in shard:
            weight_map[key] = shard_name

    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }

    return final_shards, index


def save_hf_checkpoint(
    state_dict: dict[str, torch.Tensor],
    output_path: str,
    base_model_path: str,
    max_shard_size: str = "5GB",
):
    """Save state dict as sharded safetensors + copy config/tokenizer from base model."""
    os.makedirs(output_path, exist_ok=True)

    # Parse max_shard_size
    size_str = max_shard_size.upper()
    if size_str.endswith("GB"):
        max_bytes = int(float(size_str[:-2]) * 1024**3)
    elif size_str.endswith("MB"):
        max_bytes = int(float(size_str[:-2]) * 1024**2)
    else:
        max_bytes = int(size_str)

    shards, index = _shard_state_dict(state_dict, max_bytes)

    for shard_file, shard_state in shards.items():
        shard_path = os.path.join(output_path, shard_file)
        save_file(shard_state, shard_path)
        print(f"  Saved {shard_file} ({len(shard_state)} keys)")

    # Write index JSON
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print("  Saved model.safetensors.index.json")

    # Copy config, tokenizer, and other HF assets from base model
    # (but NOT model weights or index — those come from the checkpoint)
    skip_files = {
        "model.safetensors.index.json",
    }
    skip_extensions = {".safetensors", ".bin", ".pt", ".ckpt"}
    copy_extensions = {".json", ".py", ".txt", ".model", ".jinja"}
    copy_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "video_preprocessor_config.json",
    ]

    if os.path.isdir(base_model_path):
        for fname in os.listdir(base_model_path):
            src = os.path.join(base_model_path, fname)
            ext = os.path.splitext(fname)[1]
            if fname in skip_files or ext in skip_extensions:
                continue
            if fname in copy_files or (os.path.isfile(src) and ext in copy_extensions):
                dst = os.path.join(output_path, fname)
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")

    # Re-write index after copying base files (in case base model's index was copied)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone! HF checkpoint saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DCP checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--ckpt_path", required=True, help="Path to DCP recover_checkpoint directory"
    )
    parser.add_argument(
        "--output_path", required=True, help="Output directory for HF checkpoint"
    )
    parser.add_argument(
        "--base_model_path",
        required=True,
        help="Path to base model (for config/tokenizer)",
    )
    parser.add_argument(
        "--max_shard_size", default="5GB", help="Max shard size for safetensors"
    )
    args = parser.parse_args()

    state_dict = load_dcp_model_state(args.ckpt_path)
    save_hf_checkpoint(
        state_dict, args.output_path, args.base_model_path, args.max_shard_size
    )


if __name__ == "__main__":
    main()

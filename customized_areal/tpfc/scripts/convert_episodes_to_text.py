"""Load MCTS tree cache episodes and convert token IDs to text using Qwen3-VL tokenizer.

Usage:
    python convert_episodes_to_text.py \
        --data_dir /dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/data/tree_cache/mcts_trees \
        --tokenizer_path /dfs/share-groups/letrain/ckpt/Qwen3.5-9B \
        --output_dir ./converted_episodes

Output: one .json file per query, containing a list of episodes with decoded text and metadata.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def split_turns(text: str) -> list[dict]:
    """Split full_text into turns by <|im_end|><|im_start|>role markers.

    Returns list of {"role": str, "content": str} dicts.
    """
    parts = text.split("<|im_end|>")
    turns = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(re.escape("<|im_start|>") + r"(\w+)\n?", part)
        if m:
            role = m.group(1)
            content = part[m.end() :]
            turns.append({"role": role, "content": content})
        else:
            # Fallback: shouldn't happen in well-formed chat text
            turns.append({"role": "unknown", "content": part})
    return turns


def decode_record(tokenizer, rec: dict, include_loss_masked: bool = False) -> dict:
    """Decode input_ids to text for a single record."""
    input_ids = rec["input_ids"]
    loss_mask = rec["loss_mask"]

    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    turns = split_turns(full_text)

    loss_masked_text = ""
    if include_loss_masked:
        masked_ids = [tid for tid, m in zip(input_ids, loss_mask) if m == 1]
        if masked_ids:
            loss_masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)

    return {
        "node_id": rec["node_id"],
        "parent_node_id": rec["parent_node_id"],
        "turn_idx": rec["turn_idx"],
        "episode_id": rec["episode_id"],
        "query_id": rec["query_id"],
        "train_id": rec["train_id"],
        "outcome_reward": rec["outcome_reward"],
        "num_tokens": len(input_ids),
        "num_loss_tokens": sum(loss_mask),
        "turns": turns,
        "loss_masked_text": loss_masked_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MCTS tree cache episodes to text"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to mcts_trees directory"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./converted_episodes",
        help="Output directory",
    )
    parser.add_argument(
        "--loss_masked_only",
        action="store_true",
        help="Include loss-masked text in output",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit number of query files to process",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    data_dir = Path(args.data_dir)
    files = sorted(
        f
        for f in data_dir.iterdir()
        if f.name.startswith("query_")
        and f.suffix == ".json"
        and not f.name.endswith(".tmp")
    )
    print(f"Found {len(files)} query files in {data_dir}")

    if args.max_files:
        files = files[: args.max_files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_records = 0

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        # Group records by episode_id
        ep_groups: dict[str, list[dict]] = defaultdict(list)
        for rec in data["records"]:
            ep_groups[rec["episode_id"]].append(rec)

        # Build list of episodes
        episodes = []
        for ep_id, records in ep_groups.items():
            records_sorted = sorted(records, key=lambda r: r["turn_idx"])
            decoded = [
                decode_record(
                    tokenizer, r, include_loss_masked=not args.loss_masked_only
                )
                for r in records_sorted
            ]
            episodes.append(
                {
                    "episode_id": ep_id,
                    "num_records": len(decoded),
                    "records": decoded,
                }
            )
            total_records += len(decoded)

        query_data = {
            "query_id": fpath.stem.replace("query_", ""),
            "num_episodes": len(episodes),
            "episodes": episodes,
        }

        out_path = output_dir / f"{fpath.name}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(query_data, f, indent=2, ensure_ascii=False)

        total_episodes += len(episodes)
        print(
            f"  Wrote {out_path} ({len(episodes)} episodes, {sum(len(e['records']) for e in episodes)} records)"
        )

    print(
        f"Done! {total_episodes} episodes, {total_records} records across {len(files)} files."
    )


if __name__ == "__main__":
    main()

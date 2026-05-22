"""Find records where loss_masked_text is empty or missing assistant content.

After convert_episodes_to_text.py has run, this scans the converted JSON files
and reports records that:
  1. Have empty loss_masked_text (no tokens marked for loss)
  2. Have no assistant turn in the conversation
  3. Have non-empty loss_masked_text but the `assistant` role marker is absent

The Qwen3 chat template uses `<|im_start|>assistant\n...<|im_end|>` so we check
both the fully reconstructed text and the turns structure.
"""

import argparse
import json
from pathlib import Path


def load_all_records(data_dir: str):
    """Yield (file_path, query_data, episode, record) for every record."""
    for fpath in sorted(Path(data_dir).glob("query_*.json")):
        with open(fpath) as f:
            query_data = json.load(f)
        for ep in query_data["episodes"]:
            for rec in ep["records"]:
                yield fpath, query_data, ep, rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/dfs/share-groups/letrain/zhoujie/AReaL-main/customized_areal/tpfc/data/tree_cache/converted_episodes",
        help="Path to converted episodes directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output file for problem records summary",
    )
    args = parser.parse_args()

    empty_loss_masked = []   # loss_masked_text is ''
    no_assistant_turn = []   # turns contain no assistant role
    no_asst_marker_in_full = []  # <|im_start|>assistant not in full text

    total = 0

    for fpath, query_data, ep, rec in load_all_records(args.data_dir):
        total += 1
        lmt = rec.get("loss_masked_text", "")
        turns = rec.get("turns", [])

        has_asst_turn = any(t["role"] == "assistant" for t in turns)

        full_text = ""
        for t in turns:
            full_text += f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n"
        has_asst_marker = "<|im_start|>assistant" in full_text

        rec_id = {
            "file": fpath.name,
            "query_id": query_data["query_id"],
            "episode_id": ep["episode_id"],
            "node_id": rec["node_id"],
            "turn_idx": rec["turn_idx"],
            "num_tokens": rec["num_tokens"],
            "num_loss_tokens": rec["num_loss_tokens"],
        }

        if not lmt:
            empty_loss_masked.append(rec_id)
        if not has_asst_turn:
            no_assistant_turn.append({**rec_id, "roles": [t["role"] for t in turns]})
        if lmt and not has_asst_marker:
            no_asst_marker_in_full.append(rec_id)

    print(f"Total records: {total}")
    print(f"Records with empty loss_masked_text:          {len(empty_loss_masked)}")
    print(f"Records without assistant turn:               {len(no_assistant_turn)}")
    print(f"Records with loss_mask but no asst in full:   {len(no_asst_marker_in_full)}")

    if empty_loss_masked:
        print("\n--- Empty loss_masked_text (first 20) ---")
        for r in empty_loss_masked[:20]:
            print(f"  {r['file']} query={r['query_id']} ep={r['episode_id']} "
                  f"node={r['node_id']} turn={r['turn_idx']} "
                  f"tokens={r['num_tokens']} loss_tokens={r['num_loss_tokens']}")

    if no_assistant_turn:
        print("\n--- No assistant turn (first 20) ---")
        for r in no_assistant_turn[:20]:
            print(f"  {r['file']} query={r['query_id']} ep={r['episode_id']} "
                  f"node={r['node_id']} turn={r['turn_idx']} roles={r['roles']}")

    if args.output:
        summary = {
            "total_records": total,
            "total_empty_loss_masked": len(empty_loss_masked),
            "empty_loss_masked": empty_loss_masked,
            "total_no_assistant_turn": len(no_assistant_turn),
            "no_assistant_turn": no_assistant_turn,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()

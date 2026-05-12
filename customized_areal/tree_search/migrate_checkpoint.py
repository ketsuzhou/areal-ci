#!/usr/bin/env python3
"""Migrate old mcts_trees checkpoints to the new naming scheme.

Before this script, _sanitize_filename appended an md5 hash suffix to every
query_id when creating filenames. On load, the fallback logic treated the
sanitized filename (with hash suffix) as the query_id itself, causing
snowballing filenames and redundant duplicate files across save/load cycles.

This script:
1. Reads each per-query JSON file and extracts the true query_id from the
   records inside (node.query_id).
2. Merges records from redundant duplicate files into a single file per
   query_id, named `query_{true_query_id}.json`.
3. Rebuilds metadata.json with clean query_ids (removes the now-deleted
   `query_id_to_file` field and fixes any corrupted keys in
   `query_node_ids` / `node_id_to_key`).

Usage:
    python -m customized_areal.tree_search.migrate_checkpoint <save_dir>

    <save_dir> is the parent directory that contains the mcts_trees/ folder.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


def _is_clean_uuid(s: str) -> bool:
    return len(s) == 32 and all(c in "0123456789abcdef" for c in s)


def migrate(save_dir: str) -> None:
    mcts_dir = os.path.join(save_dir, "mcts_trees")
    if not os.path.isdir(mcts_dir):
        print(f"Directory not found: {mcts_dir}")
        return

    meta_path = os.path.join(mcts_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"metadata.json not found: {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    # 1. Collect records grouped by their TRUE query_id (from inside the file)
    true_qid_to_records: dict[str, list] = defaultdict(list)
    old_files: list[str] = []

    for filename in os.listdir(mcts_dir):
        if not filename.startswith("query_") or not filename.endswith(".json"):
            continue
        filepath = os.path.join(mcts_dir, filename)
        with open(filepath) as f:
            data = json.load(f)

        # Extract true query_id from the first record
        records = data.get("records", [])
        if records:
            true_qid = records[0].get("query_id", "")
            if not true_qid:
                # Fallback: strip known hash-suffix patterns from filename
                sanitized = filename[len("query_") : -len(".json")]
                true_qid = sanitized.split("_")[0]
        else:
            sanitized = filename[len("query_") : -len(".json")]
            true_qid = sanitized.split("_")[0]

        true_qid_to_records[true_qid].extend(records)
        old_files.append(filename)

    # 2. Deduplicate records by node_id (same node may appear in multiple files)
    for qid in true_qid_to_records:
        seen: set[str] = set()
        unique: list = []
        for r in true_qid_to_records[qid]:
            nid = r.get("node_id", "")
            if nid and nid in seen:
                continue
            if nid:
                seen.add(nid)
            unique.append(r)
        true_qid_to_records[qid] = unique

    # 3. Write new clean files
    new_files: set[str] = set()
    for qid, records in true_qid_to_records.items():
        new_filename = f"query_{qid}.json"
        new_files.add(new_filename)
        new_path = os.path.join(mcts_dir, new_filename)
        tmp_path = new_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"records": records}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, new_path)

    # 4. Delete old redundant files (keep only the new clean ones)
    for filename in old_files:
        if filename not in new_files:
            os.remove(os.path.join(mcts_dir, filename))

    # 5. Rebuild metadata with clean query_ids
    # Build mapping from corrupted query_ids found in metadata to true UUIDs
    # by cross-referencing node_id_to_key with the records
    corrupted_to_clean: dict[str, str] = {}

    # Collect all node_ids that belong to each true query_id
    true_qid_to_node_ids: dict[str, list[str]] = defaultdict(list)
    for qid, records in true_qid_to_records.items():
        for r in records:
            nid = r.get("node_id", "")
            if nid:
                true_qid_to_node_ids[qid].append(nid)

    # Build reverse: node_id -> true query_id
    node_id_to_true_qid: dict[str, str] = {}
    for qid, nids in true_qid_to_node_ids.items():
        for nid in nids:
            node_id_to_true_qid[nid] = qid

    # Fix node_id_to_key: values contain [query_id, idx]
    old_nid2k = meta.get("node_id_to_key", meta.get("seq_id_to_key", {}))
    new_nid2k: dict[str, list] = {}
    for nid, val in old_nid2k.items():
        true_qid = node_id_to_true_qid.get(nid)
        if true_qid:
            new_nid2k[nid] = [true_qid, val[1]]
            if val[0] != true_qid:
                corrupted_to_clean[val[0]] = true_qid
        else:
            new_nid2k[nid] = val  # keep as-is if no record match

    # Fix query_node_ids: keys may be corrupted
    old_qni = meta.get("query_node_ids", meta.get("query_seq_ids", {}))
    new_qni: dict[str, list] = {}
    for key, node_ids in old_qni.items():
        true_qid = corrupted_to_clean.get(key, key)
        if true_qid in new_qni:
            # Merge
            existing = set(new_qni[true_qid])
            new_qni[true_qid] = sorted(existing | set(node_ids))
        else:
            new_qni[true_qid] = node_ids

    # Rebuild metadata (drop query_id_to_file)
    new_meta = {
        "node_id_to_key": new_nid2k,
        "query_node_ids": new_qni,
        "visit_counts": meta.get("visit_counts", {}),
        "total_values": meta.get("total_values", {}),
        "q_values": meta.get("q_values", {}),
        "current_train_id": meta.get("current_train_id", ""),
        "rewards": meta.get("rewards", {}),
        "normalized_advantages": meta.get("normalized_advantages", {}),
        "normalized_returns": meta.get("normalized_returns", {}),
        "turn_nodes": meta.get("turn_nodes", {}),
    }

    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(new_meta, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_meta, meta_path)

    # Summary
    print(f"Migrated {len(old_files)} old files -> {len(new_files)} clean files")
    print(f"Unique query_ids: {len(true_qid_to_records)}")
    print(f"Corrupted query_ids fixed: {len(corrupted_to_clean)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <save_dir>")
        sys.exit(1)
    migrate(sys.argv[1])

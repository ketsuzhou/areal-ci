"""Checkpoint save/load for the flat Node store.

MCTS stats are keyed by node_id (string interaction IDs) and serialize
directly — no rebuild_mcts_stats() needed after loading.
Old TrieNode-based checkpoints are incompatible and must be discarded.
"""

from __future__ import annotations

import json
import os

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")

    def exists(self) -> bool:
        return os.path.isdir(self.save_dir) and os.path.isfile(
            os.path.join(self.save_dir, "metadata.json")
        )

    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save per-query trajectory records (atomic per file)
        query_id_to_file: dict[str, str] = {}
        for query_id, records in tree_store.trajectories.items():
            data = {"records": [self._serialize_record(r) for r in records]}
            query_id_to_file[query_id] = query_id
            filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)

        # Save metadata (atomic)
        metadata = {
            "node_id_to_key": {
                k: [v[0], v[1]] for k, v in tree_store._node_id_to_key.items()
            },
            "query_node_ids": {k: v for k, v in tree_store._query_node_ids.items()},
            "query_id_to_file": query_id_to_file,
            "visit_counts": {k: v for k, v in tree_store._visit_counts.items()},
            "total_values": {k: v for k, v in tree_store._total_values.items()},
            "q_values": {k: v for k, v in tree_store._q_values.items()},
            "current_train_id": tree_store.current_train_id,
            "rewards": {k: v for k, v in tree_store._rewards.items()},
            "normalized_advantages": {
                k: v for k, v in tree_store._normalized_advantages.items()
            },
            "normalized_returns": {
                k: v for k, v in tree_store._normalized_returns.items()
            },
            "turn_nodes": tree_store._turn_nodes,
        }
        meta_path = os.path.join(self.save_dir, "metadata.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(metadata, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_meta, meta_path)

    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)

        node_id_to_key_raw = metadata.get(
            "node_id_to_key", metadata.get("seq_id_to_key", {})
        )
        store._node_id_to_key = {k: (v[0], v[1]) for k, v in node_id_to_key_raw.items()}
        store._query_node_ids = metadata.get(
            "query_node_ids", metadata.get("query_seq_ids", {})
        )
        store._visit_counts = {
            k: v for k, v in metadata.get("visit_counts", {}).items()
        }
        store._total_values = {
            k: v for k, v in metadata.get("total_values", {}).items()
        }
        store._q_values = {k: v for k, v in metadata.get("q_values", {}).items()}
        store.current_train_id = metadata.get("current_train_id", "")
        store._rewards = {k: v for k, v in metadata.get("rewards", {}).items()}
        store._normalized_advantages = {
            k: v for k, v in metadata.get("normalized_advantages", {}).items()
        }
        store._normalized_returns = {
            k: v for k, v in metadata.get("normalized_returns", {}).items()
        }
        store._turn_nodes = metadata.get("turn_nodes", {})

        # Build reverse mapping from filenames back to query_ids
        # (needed for old checkpoints where filename may differ from query_id)
        query_id_to_file = metadata.get("query_id_to_file", {})
        file_to_query = {v: k for k, v in query_id_to_file.items()}

        # Load per-query trajectory records
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            file_key = filename[len("query_") : -len(".json")]
            # For new checkpoints, file_key == query_id; for old ones, use metadata mapping
            query_id = file_to_query.get(file_key, file_key)
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            store.trajectories[query_id] = [
                self._deserialize_record(r) for r in data["records"]
            ]

        return store

    @staticmethod
    def _serialize_record(node: Node) -> dict:
        data = {
            "input_ids": node.input_ids,
            "loss_mask": node.loss_mask,
            "logprobs": node.logprobs,
            "versions": node.versions,
            "outcome_reward": node.outcome_reward,
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "episode_id": node.episode_id,
            "turn_idx": node.turn_idx,
            "query_id": node.query_id,
            "train_id": node.train_id,
        }
        if node.topk_ids is not None:
            data["topk_ids"] = node.topk_ids
        if node.topk_logp is not None:
            data["topk_logp"] = node.topk_logp
        if node.distill_reward is not None:
            data["distill_reward"] = node.distill_reward
        if node.teacher_logp is not None:
            data["teacher_logp"] = node.teacher_logp
        return data

    @staticmethod
    def _deserialize_record(data: dict) -> Node:
        return Node(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            logprobs=data["logprobs"],
            versions=data["versions"],
            outcome_reward=data.get("outcome_reward", data.get("reward", 0.0)),
            node_id=data.get("node_id", ""),
            parent_node_id=data.get("parent_node_id"),
            episode_id=data.get("episode_id", ""),
            turn_idx=data.get("turn_idx", 0),
            query_id=data.get("query_id", ""),
            train_id=data.get("train_id", ""),
            topk_ids=data.get("topk_ids"),
            topk_logp=data.get("topk_logp"),
            distill_reward=data.get("distill_reward"),
            teacher_logp=data.get("teacher_logp"),
        )

    @staticmethod
    def save_trained_episodes(
        recover_checkpoint_dir: str, tree_store: MCTSTreeStore
    ) -> None:
        """Save trained episode IDs to the recover checkpoint directory."""
        if not tree_store.current_train_id:
            return
        trained_ids: set[str] = set()
        for query_id, records in tree_store.trajectories.items():
            for node in records:
                if isinstance(node, dict):
                    if node.get("train_id", "") == tree_store.current_train_id:
                        trained_ids.add(node.get("episode_id", ""))
                else:
                    if node.train_id == tree_store.current_train_id:
                        trained_ids.add(node.episode_id)
        data = {"trained_episode_ids": sorted(trained_ids)}
        os.makedirs(recover_checkpoint_dir, exist_ok=True)
        filepath = os.path.join(recover_checkpoint_dir, "trained_episodes.json")
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)

    @staticmethod
    def load_trained_episodes(
        recover_checkpoint_dir: str,
    ) -> set[str] | None:
        """Load trained episode IDs from the recover checkpoint directory.

        Returns the set of trained episode IDs, or None if the file does
        not exist or is corrupt.
        """
        filepath = os.path.join(recover_checkpoint_dir, "trained_episodes.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath) as f:
                data = json.load(f)
            return set(data["trained_episode_ids"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

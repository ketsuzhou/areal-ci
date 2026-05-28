"""Checkpoint save/load for the flat Node store.

MCTS stats are keyed by node_id (string interaction IDs) and serialize
directly — no rebuild_mcts_stats() needed after loading.
Old TrieNode-based checkpoints are incompatible and must be discarded.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")

    def exists(self) -> bool:
        if not os.path.isdir(self.save_dir):
            return False
        return any(
            filename.startswith("query_") and filename.endswith(".json")
            for filename in os.listdir(self.save_dir)
        )

    @contextmanager
    def _file_lock(self, lock_path: str, *, exclusive: bool) -> Iterator[None]:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "a") as lock_file:
            flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), flag)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _query_path(self, query_id: str) -> str:
        return os.path.join(self.save_dir, f"query_{query_id}.json")

    def _query_lock_path(self, query_id: str) -> str:
        return os.path.join(self.save_dir, f"query_{query_id}.lock")

    @staticmethod
    def _atomic_json_dump(path: str, data: dict) -> None:
        tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _query_metadata(tree_store: MCTSTreeStore, query_id: str) -> dict:
        node_ids = list(tree_store._query_node_ids.get(query_id, []))
        node_id_set = set(node_ids)
        return {
            "node_id_to_key": {
                k: [v[0], v[1]]
                for k, v in tree_store._node_id_to_key.items()
                if v[0] == query_id
            },
            "query_node_ids": node_ids,
            "visit_counts": {
                k: v for k, v in tree_store._visit_counts.items() if k in node_id_set
            },
            "total_values": {
                k: v for k, v in tree_store._total_values.items() if k in node_id_set
            },
            "q_values": {
                k: v for k, v in tree_store._q_values.items() if k in node_id_set
            },
            "rewards": {
                k: v for k, v in tree_store._rewards.items() if k in node_id_set
            },
            "normalized_advantages": {
                k: v
                for k, v in tree_store._normalized_advantages.items()
                if k in node_id_set
            },
            "normalized_returns": {
                k: v
                for k, v in tree_store._normalized_returns.items()
                if k in node_id_set
            },
            "turn_nodes": {
                k: v for k, v in tree_store._turn_nodes.items() if v in node_id_set
            },
        }

    def save_query(self, tree_store: MCTSTreeStore, query_id: str) -> None:
        """Save one query file under a per-query inter-process lock."""
        os.makedirs(self.save_dir, exist_ok=True)
        records = tree_store.trajectories.get(query_id)
        if records is None:
            return
        data = {
            "query_id": query_id,
            "records": [self._serialize_record(r) for r in records],
            "metadata": self._query_metadata(tree_store, query_id),
        }
        filepath = self._query_path(query_id)
        with self._file_lock(self._query_lock_path(query_id), exclusive=True):
            self._atomic_json_dump(filepath, data)

    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save per-query trajectory records. Each query has its own lock so
        # independent query writers do not block each other.
        for query_id in tree_store.trajectories:
            self.save_query(tree_store, query_id)

    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        # Runtime train_id from env identifies THIS run.
        # It MUST be set before loading a checkpoint.
        runtime_train_id = store.current_train_id
        if not runtime_train_id:
            raise RuntimeError(
                "TRAIN_ID environment variable is not set. "
                "It must be set before loading a tree checkpoint "
                "so that cached nodes can be correctly classified as "
                "trained or untrained for this run."
            )

        # Load per-query trajectory records
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            file_key = filename[len("query_") : -len(".json")]
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            query_id = data.get("query_id", file_key)
            store.trajectories[query_id] = [
                self._deserialize_record(r) for r in data["records"]
            ]
            query_metadata = data.get("metadata", {})
            node_id_to_key_raw = query_metadata.get("node_id_to_key", {})
            store._node_id_to_key.update(
                {k: (v[0], v[1]) for k, v in node_id_to_key_raw.items()}
            )
            query_node_ids = query_metadata.get("query_node_ids")
            if query_node_ids is not None:
                store._query_node_ids[query_id] = query_node_ids
            store._visit_counts.update(query_metadata.get("visit_counts", {}))
            store._total_values.update(query_metadata.get("total_values", {}))
            store._q_values.update(query_metadata.get("q_values", {}))
            store._rewards.update(query_metadata.get("rewards", {}))
            store._normalized_advantages.update(
                query_metadata.get("normalized_advantages", {})
            )
            store._normalized_returns.update(
                query_metadata.get("normalized_returns", {})
            )
            store._turn_nodes.update(query_metadata.get("turn_nodes", {}))

        # Rebuild indices from loaded trajectories if a query file is missing
        # per-query metadata.
        for query_id, records in store.trajectories.items():
            for idx, node in enumerate(records):
                node_id = node.node_id
                if not node_id:
                    continue
                if node_id not in store._node_id_to_key:
                    store._node_id_to_key[node_id] = (query_id, idx)
                    store._query_node_ids.setdefault(query_id, []).append(node_id)

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
            "task_id": node.task_id,
            "entropy_stats": node.entropy_stats,
            "need_branch": node.need_branch,
            "branch_sandbox_id": node.branch_sandbox_id,
        }
        if node.topk_ids is not None:
            data["topk_ids"] = node.topk_ids
        if node.topk_logp is not None:
            data["topk_logp"] = node.topk_logp
        if node.teacher_logp is not None:
            data["teacher_logp"] = node.teacher_logp
        if node.guidance is not None:
            data["guidance"] = {str(k): v for k, v in node.guidance.items()}
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
            task_id=data.get("task_id", ""),
            entropy_stats=data.get("entropy_stats"),
            need_branch=bool(data.get("need_branch", False)),
            branch_sandbox_id=data.get("branch_sandbox_id"),
            topk_ids=data.get("topk_ids"),
            topk_logp=data.get("topk_logp"),
            teacher_logp=data.get("teacher_logp"),
            guidance={int(k): v for k, v in data.get("guidance", {}).items()}
            if data.get("guidance")
            else None,
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

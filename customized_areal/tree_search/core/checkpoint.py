"""Checkpoint save/load for the SuperNode store.

MCTS stats are keyed by node_id (string interaction IDs) and serialize
directly — no rebuild_mcts_stats() needed after loading.
Trajectories hold ``SuperNode`` objects; each SuperNode's ``nodes`` list
holds the ``Node`` records that carry MCTS stats and tensors. Indices
(``_super_id_to_key``, ``_node_id_to_super``) are rebuilt on load from
the deserialized SuperNodes rather than persisted in the metadata.
Old TrieNode-based and flat-Node checkpoints are incompatible and must
be discarded.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from customized_areal.tree_search.agents.execution_dag import SuperNode
from customized_areal.tree_search.core.tree_store import MCTSTreeStore, Node


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
            "values": {k: v for k, v in tree_store._values.items() if k in node_id_set},
            "judge_scores": {
                k: v for k, v in tree_store._judge_scores.items() if k in node_id_set
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
            "records": [self._serialize_super_node(r) for r in records],
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
                self._deserialize_super_node(r) for r in data["records"]
            ]
            query_metadata = data.get("metadata", {})
            store._visit_counts.update(query_metadata.get("visit_counts", {}))
            store._total_values.update(query_metadata.get("total_values", {}))
            store._q_values.update(query_metadata.get("q_values", {}))
            store._values.update(query_metadata.get("values", {}))
            store._judge_scores.update(query_metadata.get("judge_scores", {}))
            store._rewards.update(query_metadata.get("rewards", {}))
            store._normalized_advantages.update(
                query_metadata.get("normalized_advantages", {})
            )
            store._normalized_returns.update(
                query_metadata.get("normalized_returns", {})
            )
            store._turn_nodes.update(query_metadata.get("turn_nodes", {}))

        # Rebuild SuperNode/Node indices from the loaded trajectories. The
        # per-query metadata no longer persists these indices (the index shape
        # changed when the store switched from list[Node] to list[SuperNode]);
        # rebuilding from trajectories is the single source of truth on load.
        # _query_node_ids is also rebuilt here (not restored from metadata) so
        # the index and the trajectory list stay consistent and duplicate-free.
        for query_id, supers in store.trajectories.items():
            for idx, super_node in enumerate(supers):
                super_id = super_node.node_id
                if not super_id:
                    continue
                if super_id not in store._super_id_to_key:
                    store._super_id_to_key[super_id] = (query_id, idx)
                for node_idx, node in enumerate(super_node.nodes):
                    node_id = (
                        node.get("node_id", "")
                        if isinstance(node, dict)
                        else node.node_id
                    )
                    if not node_id:
                        continue
                    if node_id not in store._node_id_to_super:
                        store._node_id_to_super[node_id] = (super_id, node_idx)
                        store._query_node_ids.setdefault(query_id, []).append(node_id)

        return store

    @staticmethod
    def _serialize_node(node) -> dict:
        data = {
            "input_ids": node.input_ids,
            "loss_mask": node.loss_mask,
            "logprobs": node.logprobs,
            "versions": node.versions,
            "outcome_reward": node.outcome_reward,
            "value": node.value,
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "extra_parent_node_ids": node.extra_parent_node_ids,
            "episode_id": node.episode_id,
            "turn_idx": node.turn_idx,
            "query_id": node.query_id,
            "train_id": node.train_id,
            "discarded": node.discarded,
            "task_id": node.task_id,
            "entropy_stats": node.entropy_stats,
            "need_branch": node.need_branch,
            "branch_sandbox_id": node.branch_sandbox_id,
            "branch_issue_id": node.branch_issue_id,
            "branch_env_snapshot_id": node.branch_env_snapshot_id,
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
    def _deserialize_node(data: dict) -> Node:
        return Node(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            logprobs=data["logprobs"],
            versions=data["versions"],
            outcome_reward=data.get("outcome_reward", data.get("reward", 0.0)),
            value=data.get("value", 0.0),
            node_id=data.get("node_id", ""),
            parent_node_id=data.get("parent_node_id"),
            extra_parent_node_ids=data.get("extra_parent_node_ids"),
            episode_id=data.get("episode_id", ""),
            turn_idx=data.get("turn_idx", 0),
            query_id=data.get("query_id", ""),
            train_id=data.get("train_id", ""),
            discarded=bool(data.get("discarded", False)),
            task_id=data.get("task_id", ""),
            entropy_stats=data.get("entropy_stats"),
            need_branch=bool(data.get("need_branch", False)),
            branch_sandbox_id=data.get("branch_sandbox_id"),
            branch_issue_id=data.get("branch_issue_id"),
            branch_env_snapshot_id=data.get("branch_env_snapshot_id"),
            topk_ids=data.get("topk_ids"),
            topk_logp=data.get("topk_logp"),
            teacher_logp=data.get("teacher_logp"),
            guidance={int(k): v for k, v in data.get("guidance", {}).items()}
            if data.get("guidance")
            else None,
        )

    @staticmethod
    def _serialize_super_node(super_node: SuperNode) -> dict:
        """Serialize a SuperNode losslessly (DAG topology + env + reward + turns).

        The SuperNode-level envelope is produced by ``SuperNode.to_dict()`` (the
        single source of truth for SuperNode field serialization, in
        ``execution_dag.py``); only ``nodes`` is overridden here. ``Node`` has no
        ``to_dict``, so ``to_dict()`` would pass raw Node objects through — the
        checkpoint replaces them with :meth:`_serialize_node`, which persists the
        MCTS-relevant Node fields (tensors, ``extra_parent_node_ids``, etc.).
        Reusing ``to_dict``/``from_dict`` avoids field drift between the two
        serializers when a SuperNode field is added.
        """
        data = super_node.to_dict()
        data["nodes"] = [
            TreeCheckpointManager._serialize_node(n) for n in super_node.nodes
        ]
        return data

    @staticmethod
    def _deserialize_super_node(data: dict) -> SuperNode:
        # SuperNode.from_dict rebuilds the envelope (edges/enum coercion,
        # required-field validation); we then swap in fully deserialized Nodes
        # (from_dict leaves the raw node dicts in place).
        super_node = SuperNode.from_dict(data)
        super_node.nodes = [
            TreeCheckpointManager._deserialize_node(n) for n in data.get("nodes", [])
        ]
        return super_node

    @staticmethod
    def save_trained_episodes(
        recover_checkpoint_dir: str, tree_store: MCTSTreeStore
    ) -> None:
        """Save trained episode IDs to the recover checkpoint directory."""
        if not tree_store.current_train_id:
            return
        trained_ids: set[str] = set()
        for supers in tree_store.trajectories.values():
            for super_node in supers:
                for node in super_node.nodes:
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

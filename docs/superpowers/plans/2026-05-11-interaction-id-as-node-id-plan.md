# Use interaction_id as node_id — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monotonically increasing `int` `node_id` with the globally unique
`str` `interaction_id` (UUID from inference engine) throughout Node, MCTSTreeStore, and
all consumers.

**Architecture:** `node_id` changes from `int = 0` to `str = ""` on the Node dataclass.
`MCTSTreeStore` drops its `_next_node_id` counter — the interaction_id is set at Node
construction time and read by `_insert_single`. All internal dicts change key type `int`
→ `str`. Checkpoint serialization drops the `int(k)` conversion since keys are already
strings.

**Tech Stack:** Python 3.12+, dataclasses, torch

______________________________________________________________________

### Task 1: Change Node dataclass field types

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:22-59`

- [ ] **Step 1: Update node_id and parent_node_id types**

```python
@dataclass
class Node:
    """A single turn in a multi-turn conversation tree.

    Each Node represents one assistant response turn, including its prompt
    context (all tokens from the beginning of the conversation up through
    this turn's response). Nodes are linked via node_id/parent_node_id and
    grouped into episodes via episode_id.

    node_id is the globally unique interaction ID (UUID string from the
    inference engine). query_id is set as metadata. advantages/returns are
    set by the advantage computer.
    """

    # Core sequence (full turn: prompt + response)
    input_ids: list[int]
    loss_mask: list[int]  # 0=prompt, 1=response
    logprobs: list[float]  # full sequence (0.0 on prompt positions)
    versions: list[int]  # policy version (-1 on prompt)

    # Tree structure
    node_id: str = ""  # globally unique interaction ID (UUID from inference engine)
    parent_node_id: str | None = None  # parent interaction ID (None for root)
    episode_id: str = ""  # groups turns into a trajectory path
    turn_idx: int = 0  # 1-based turn position within episode
    query_id: str = ""  # dataset query identifier

    # Reward
    outcome_reward: float = 0.0

    # Tree-computed advantages/returns (set by TreeAdvantageComputer)
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor: change Node.node_id type from int to str for interaction_id"
```

______________________________________________________________________

### Task 2: Set node_id and parent_node_id in interactions_dict_to_nodes

**Files:**

- Modify: `customized_areal/tree_search/proxy_workflow.py:127-136`

- [ ] **Step 1: Pass node_id and parent_node_id to Node constructor**

Change the Node construction at line 127 to include `node_id=interaction_id` and derive
`parent_node_id` from `interaction.parent`:

```python
        # Derive parent node_id from parent interaction when available
        pn_id: str | None = None
        if interaction.parent is not None:
            pn_id = interaction.parent.interaction_id

        node = Node(
            input_ids=seq_tokens,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            turn_idx=turn_idx,
            node_id=interaction_id,
            parent_node_id=pn_id,
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
        )
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/proxy_workflow.py
git commit -m "feat: set node_id=interaction_id and parent_node_id from parent interaction"
```

______________________________________________________________________

### Task 3: Update \_node_to_tensor_dict parameter type

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:110-112`

- [ ] **Step 1: Change node_id parameter from int to str**

Change the function signature:

```python
def _node_to_tensor_dict(
    node: Node, query_id: str, node_id: str, num_turns_in_episode: int = 1
) -> dict[str, Any]:
```

The body is unchanged — `traj["node_id"] = node_id` and
`traj["_turn_id"] = node.node_id` now store strings, and
`traj["_parent_turn_id"] = node.parent_node_id` stores `str | None`.

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor: change _node_to_tensor_dict node_id param from int to str"
```

______________________________________________________________________

### Task 4: Update MCTSTreeStore — remove counter, change all dict key types

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:169-367`

- [ ] **Step 1: Rewrite __init__ — remove \_next_node_id, change dict types**

```python
    def __init__(self) -> None:
        self.trajectories: dict[str, list[Node]] = {}
        self._node_id_to_key: dict[str, tuple[str, int]] = {}
        self._query_node_ids: dict[str, list[str]] = {}

        self._visit_counts: dict[str, int] = {}
        self._total_values: dict[str, float] = {}
        self._q_values: dict[str, float] = {}

        self._trained: dict[str, bool] = {}
        self._rewards: dict[str, float] = {}

        # Tree-search episode metadata
        self._turn_nodes: dict[str, str] = {}  # turn_id → node_id
        self._normalized_advantages: dict[str, float] = {}
        self._normalized_returns: dict[str, float] = {}
```

- [ ] **Step 2: Rewrite \_backup — change node_id param type**

```python
    def _backup(self, node_id: str, reward: float) -> None:
        """Update MCTS stats for a single trajectory."""
        self._visit_counts[node_id] = self._visit_counts.get(node_id, 0) + 1
        self._total_values[node_id] = self._total_values.get(node_id, 0.0) + reward
        self._q_values[node_id] = (
            self._total_values[node_id] / self._visit_counts[node_id]
        )
```

- [ ] **Step 3: Rewrite \_insert_single — read node_id, no counter**

```python
    def _insert_single(self, query_id: str, node: Node) -> str:
        """Insert a single Node, using its pre-assigned node_id.

        The node must have a non-empty node_id (interaction_id) set before
        insertion. No counter is used — the interaction_id is the authority.

        Supports both Node dataclass instances and plain dicts (the
        latter arriving when tree search patches aren't active on the
        remote engine and _convert_trajs_to_nodes hasn't converted yet).
        """
        node_id = node.node_id if isinstance(node, Node) else node.get("node_id", "")
        if not node_id:
            raise ValueError(
                "Node must have a non-empty node_id (interaction_id) before insert"
            )

        idx = len(self.trajectories.setdefault(query_id, []))
        self.trajectories[query_id].append(node)
        self._node_id_to_key[node_id] = (query_id, idx)
        self._query_node_ids.setdefault(query_id, []).append(node_id)

        if isinstance(node, dict):
            node["node_id"] = node_id
            node["query_id"] = query_id
            outcome_reward = node.get("outcome_reward", node.get("reward", 0.0))
        else:
            node.node_id = node_id
            node.query_id = query_id
            outcome_reward = node.outcome_reward

        self._backup(node_id, outcome_reward)
        self._trained[node_id] = False
        self._rewards[node_id] = outcome_reward

        return node_id
```

- [ ] **Step 4: Rewrite insert_batch — dedup sentinel from 0 to ""**

```python
    def insert_batch(self, trajectories: list[Node]) -> None:
        """Insert Node trajectories into the store.

        Each Node is inserted directly. Nodes that already have a
        node_id assigned (loaded from cache) are skipped.
        """
        for node in trajectories:
            existing_id = getattr(node, "node_id", "")
            if existing_id != "" and existing_id in self._node_id_to_key:
                continue
            query_id = (
                node.get("query_id", "")
                if isinstance(node, dict)
                else (node.query_id or "")
            )
            self._insert_single(query_id, node)
```

- [ ] **Step 5: Update all method signatures — int → str**

Every method taking or returning `node_id` changes type:

```python
    def get_advantages(self, query_id: str, node_id: str) -> torch.Tensor:
        # body unchanged
```

```python
    def get_prompt_mask(self, query_id: str, node_id: str) -> torch.Tensor:
        # body unchanged
```

```python
    def set_trained(self, node_id: str, trained: bool = True) -> None:
        self._trained[node_id] = trained

    def is_trained(self, node_id: str) -> bool:
        return self._trained.get(node_id, False)

    def get_reward(self, node_id: str) -> float:
        return self._rewards.get(node_id, 0.0)

    def get_q_value(self, node_id: str) -> float:
        return self._q_values.get(node_id, 0.0)

    def set_normalized_advantage(self, node_id: str, value: float) -> None:
        self._normalized_advantages[node_id] = value

    def get_normalized_advantage(self, node_id: str, default: float = 0.0) -> float:
        return self._normalized_advantages.get(node_id, default)

    def has_normalized_advantage(self, node_id: str) -> bool:
        return node_id in self._normalized_advantages

    def set_normalized_return(self, node_id: str, value: float) -> None:
        self._normalized_returns[node_id] = value

    def get_normalized_return(self, node_id: str, default: float = 0.0) -> float:
        return self._normalized_returns.get(node_id, default)
```

```python
    def get_untrained_node_ids(self, query_id: str, n_samples: int) -> list[str]:
        if query_id not in self._query_node_ids:
            return []
        result: list[str] = []
        for node_id in self._query_node_ids[query_id]:
            if not self._trained.get(node_id, False):
                result.append(node_id)
                if len(result) >= n_samples:
                    break
        return result
```

```python
    def load_trajectories(self, query_id: str, n_samples: int) -> list[Node]:
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_node_ids(query_id, n_samples)
        result: list[Node] = []
        for node_id in untrained_ids:
            qid, idx = self._node_id_to_key[node_id]
            node = self.trajectories[qid][idx]
            result.append(node)
        return result
```

The `get_untrained_count` method body is unchanged (returns `int`, not parameterized by
`node_id` type).

- [ ] **Step 6: Update reset_trained_flags and mark_episodes_trained**

```python
    def reset_trained_flags(self) -> None:
        for key in self._trained:
            self._trained[key] = False

    def mark_episodes_trained(self, episode_ids: set[str]) -> None:
        """Set trained flags based on episode IDs.

        Nodes whose episode_id is in the given set are marked trained.
        All other nodes are marked untrained. Episode IDs not present
        in the store are silently ignored.
        """
        for node_id in self._trained:
            self._trained[node_id] = False
        for query_id, records in self.trajectories.items():
            for node in records:
                if node.episode_id in episode_ids:
                    self._trained[node.node_id] = True
```

(Bodies unchanged — only dict key types changed. `node.node_id` is now `str`.)

- [ ] **Step 7: Update clear() — remove \_next_node_id**

```python
    def clear(self) -> None:
        """Reset all trajectories, stats, and indices."""
        self.trajectories.clear()
        self._node_id_to_key.clear()
        self._query_node_ids.clear()
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._trained.clear()
        self._rewards.clear()
        self._turn_nodes.clear()
        self._normalized_advantages.clear()
        self._normalized_returns.clear()
```

- [ ] **Step 8: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor: change MCTSTreeStore node_id dict keys from int to str, remove counter"
```

______________________________________________________________________

### Task 5: Update advantage.py type annotations

**Files:**

- Modify: `customized_areal/tree_search/advantage.py:40-41`

- [ ] **Step 1: Change dict key/set types from int to str**

```python
    def compute(self, trajectories: list[Node]) -> None:
        """Replace GAE advantages with per-query GRPO-normalized outcome_rewards.

        Both advantages and returns are set to the same per-query GRPO-normalized
        outcome_reward, broadcast across response positions via prompt mask.
        """
        # Collect unique (query_id → set of node_ids) and reward_per_node
        query_node_sets: dict[str, set[str]] = {}
        node_rewards: dict[str, float] = {}

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            nset = query_node_sets.setdefault(query_id, set())

            node_id = getattr(traj, "node_id", None)
            if node_id is not None:
                nset.add(node_id)
                node_rewards[node_id] = traj.outcome_reward

        # Per-query GRPO normalization of outcome_rewards for returns
        for query_id, node_id_set in query_node_sets.items():
            node_ids = list(node_id_set)
            rewards = [node_rewards[nid] for nid in node_ids]
            if len(rewards) < 2:
                for nid in node_ids:
                    self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = var_r**0.5
            for nid, r in zip(node_ids, rewards):
                self.tree_store.set_normalized_return(
                    nid, (r - mean_r) / (std_r + self.grpo_eps)
                )

        # Compute per-trajectory advantages and returns
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue

            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            mask = self.tree_store.get_prompt_mask(query_id, node_id)
            norm_return = self.tree_store.get_normalized_return(node_id)
            traj.advantages = mask.float() * norm_return
            traj.returns = mask.float() * norm_return
```

The only substantive changes are `set[int]` → `set[str]` and `dict[int, float]` →
`dict[str, float]`. The `getattr(traj, "node_id", None)` now returns `str | None`.

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/advantage.py
git commit -m "refactor: change advantage dict key/set types from int to str for string node_id"
```

______________________________________________________________________

### Task 6: Update checkpoint.py — drop \_next_node_id, stop int(k) conversion

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py:36-131,136-177`

- [ ] **Step 1: Update save() — remove next_node_id, stop str(k) conversion**

The metadata dict keys are already strings from `node_id` (now `str`), but checkpoint
currently converts them via `str(k)` — this is now the natural type, so the conversion
becomes a no-op. Remove `next_node_id`:

```python
    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save per-query trajectory records (atomic per file)
        query_id_to_file: dict[str, str] = {}
        for query_id, records in tree_store.trajectories.items():
            data = {"records": [self._serialize_record(r) for r in records]}
            sanitized = _sanitize_filename(query_id)
            query_id_to_file[query_id] = sanitized
            filepath = os.path.join(self.save_dir, f"query_{sanitized}.json")
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
            "visit_counts": {k: v for k, v in tree_store._visit_counts.items()},
            "total_values": {k: v for k, v in tree_store._total_values.items()},
            "q_values": {k: v for k, v in tree_store._q_values.items()},
            "trained": {k: v for k, v in tree_store._trained.items()},
            "rewards": {k: v for k, v in tree_store._rewards.items()},
            "normalized_advantages": {
                k: v for k, v in tree_store._normalized_advantages.items()
            },
            "normalized_returns": {
                k: v for k, v in tree_store._normalized_returns.items()
            },
            "turn_nodes": tree_store._turn_nodes,
            "query_id_to_file": query_id_to_file,
        }
        meta_path = os.path.join(self.save_dir, "metadata.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(metadata, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_meta, meta_path)
```

Key changes: removed `"next_node_id": tree_store._next_node_id`, removed all `str(k)`
conversions (keys are already `str`).

- [ ] **Step 2: Update load() — remove next_node_id, stop int(k) conversion**

```python
    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)

        node_id_to_key_raw = metadata.get(
            "node_id_to_key", metadata.get("seq_id_to_key", {})
        )
        store._node_id_to_key = {
            k: (v[0], v[1]) for k, v in node_id_to_key_raw.items()
        }
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
        store._trained = {k: v for k, v in metadata.get("trained", {}).items()}
        store._rewards = {k: v for k, v in metadata.get("rewards", {}).items()}
        store._normalized_advantages = {
            k: v for k, v in metadata.get("normalized_advantages", {}).items()
        }
        store._normalized_returns = {
            k: v for k, v in metadata.get("normalized_returns", {}).items()
        }
        store._turn_nodes = metadata.get("turn_nodes", {})

        # Build reverse mapping from sanitized filenames back to query_ids
        query_id_to_file = metadata.get("query_id_to_file", {})
        file_to_query = {v: k for k, v in query_id_to_file.items()}

        # Load per-query trajectory records
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            sanitized = filename[len("query_") : -len(".json")]
            query_id = file_to_query.get(sanitized, sanitized)
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            store.trajectories[query_id] = [
                self._deserialize_record(r) for r in data["records"]
            ]

        return store
```

Key changes: removed `store._next_node_id = metadata.get("next_node_id", ...)`, removed
all `int(k)` conversions, fallback key `"seq_id_to_key"` and `"query_seq_ids"` still
work (they map to the same types).

- [ ] **Step 3: Update docstring — remove stale int mention**

Change line 2-5:

```python
"""Checkpoint save/load for the flat Node store.

MCTS stats are keyed by node_id (string interaction IDs) and serialize
directly — no rebuild_mcts_stats() needed after loading.
Old TrieNode-based checkpoints are incompatible and must be discarded.
"""
```

- [ ] **Step 4: Update \_deserialize_record — node_id default from 0 to ""**

```python
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
            topk_ids=data.get("topk_ids"),
            topk_logp=data.get("topk_logp"),
            distill_reward=data.get("distill_reward"),
            teacher_logp=data.get("teacher_logp"),
        )
```

Only `node_id` default changes from `0` to `""`.

- [ ] **Step 5: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py
git commit -m "refactor: remove _next_node_id from checkpoint, stop int(k) conversion"
```

______________________________________________________________________

### Task 7: Verify trainer.py needs no changes

**Files:**

- Verify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Read and verify trainer.py type compatibility**

Run: `grep -n "node_id" customized_areal/tree_search/trainer.py`

All usages in trainer.py access `node.node_id` or `getattr(traj, "node_id", None)` and
pass it directly to `tree_store` methods. The type flows from Node → local variable →
store method — no explicit `int` annotations, so no changes needed:

- Line 51: `node_id = getattr(traj, "node_id", None)` → now `str | None`

- Line 53: `tree_store.set_trained(node_id, True)` → accepts `str`

- Line 419: `node_id = node.node_id` → now `str`

- Line 424: passed to `_node_to_tensor_dict(node, query_id, node_id, ...)` → accepts
  `str`

- [ ] **Step 2: Commit** (if needed — skip if no changes)

______________________________________________________________________

### Task 8: Run pre-commit and verify

**Files:**

- All modified files

- [ ] **Step 1: Run pre-commit hooks**

```bash
pre-commit run --all-files
```

Expected: all hooks pass (formatting, linting).

- [ ] **Step 2: Verify no remaining `int` references to node_id in the tree_search
  module**

```bash
grep -rn "node_id.*int\|int.*node_id" customized_areal/tree_search/
```

Expected: no matches (all converted to `str`).

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "chore: final cleanup after node_id type migration"
```

# Node Dataclass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `TrajectoryRecord` and ad-hoc `dict[str, Any]` trajectory
representations with a unified `Node` dataclass in `mcts_tree_store.py`.

**Architecture:** `Node` is a single-turn tree node with 12 fields (8 required, 4
optional). Per-turn Nodes are stored individually in the tree store. Episodes are
reconstructed by traversing `node_id`/`parent_node_id` links grouped by `episode_id`
when loading for training.

**Tech Stack:** Python 3.12+, dataclasses, torch

______________________________________________________________________

### Task 1: Define `Node` dataclass and remove `TrajectoryRecord`

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:1-53`

- [ ] **Step 1: Replace `TrajectoryRecord` with `Node` dataclass**

In `customized_areal/tree_search/mcts_tree_store.py`, replace the `TrajectoryRecord`
dataclass (lines 32-53) with:

```python
@dataclass
class Node:
    """A single turn in a multi-turn conversation tree.

    Each Node represents one assistant response turn, including its prompt
    context (all tokens from the beginning of the conversation up through
    this turn's response). Nodes are linked via node_id/parent_node_id and
    grouped into episodes via episode_id.

    Metadata fields (_mcts_query_id, _mcts_seq_id) are set by the tree
    store after insertion. advantages/returns are set by the advantage
    computer.
    """

    # Core sequence (full turn: prompt + response)
    input_ids: list[int]
    loss_mask: list[int]              # 0=prompt, 1=response
    logprobs: list[float]             # full sequence (0.0 on prompt positions)
    versions: list[int]               # policy version (-1 on prompt)

    # Tree structure
    node_id: str                      # interaction ID for this turn
    parent_node_id: str | None        # parent interaction ID (None for root)
    episode_id: str                   # groups turns into a trajectory path

    # Reward
    outcome_reward: float = 0.0

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None
```

Also update the module docstring (line 5): replace "TrajectoryRecord" with "Node".

- [ ] **Step 2: Update type annotations using `TrajectoryRecord`**

Replace all `TrajectoryRecord` references with `Node`:

- Line 94: `dict[str, list[TrajectoryRecord]]` → `dict[str, list[Node]]`

- Line 118: `) -> TrajectoryRecord:` → `) -> Node:`

- Line 198: `record: TrajectoryRecord` → `node: Node`

- Line 199: docstring "Insert a single TrajectoryRecord" → "Insert a single Node"

- Lines 228, 422: remaining docstring references to TrajectoryRecord → Node

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor(tree-search): replace TrajectoryRecord with Node dataclass"
```

______________________________________________________________________

### Task 2: Update `_make_record` to build `Node`

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:116-144`

- [ ] **Step 1: Rewrite `_make_record` to return `Node`**

```python
def _make_record(
    self, traj: dict[str, Any], idx: int, seq_len: int
) -> Node:
    """Extract an unpadded sample from traj[idx] and derive turn boundaries."""
    input_ids = traj["input_ids"][idx, :seq_len].tolist()
    loss_mask = traj["loss_mask"][idx, :seq_len].tolist()
    logprobs = (
        traj["logprobs"][idx, :seq_len].tolist()
        if "logprobs" in traj
        else [0.0] * seq_len
    )
    versions = (
        traj["versions"][idx, :seq_len].tolist()
        if "versions" in traj
        else [0] * seq_len
    )
    rewards = traj["rewards"]
    outcome_reward = rewards[idx].item() if rewards.dim() >= 1 else rewards.item()

    return Node(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=logprobs,
        versions=versions,
        outcome_reward=outcome_reward,
        node_id="",
        parent_node_id=None,
        episode_id="",
    )
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor(tree-search): update _make_record to return Node"
```

______________________________________________________________________

### Task 3: Update `_insert_list_dict` to build `Node`

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:146-195`

- [ ] **Step 1: Rewrite `_insert_list_dict`**

```python
def _insert_list_dict(self, traj: dict[str, Any]) -> None:
    """Insert a list-based trajectory dict into the tree store."""
    input_ids = traj["input_ids"]
    loss_mask = traj["loss_mask"]
    outcome_reward = traj.get("outcome_reward", traj.get("reward", 0.0))
    logprobs = traj.get("logprobs", [0.0] * len(input_ids))
    versions = traj.get("versions", [0] * len(input_ids))

    # New fields
    topk_ids = traj.get("topk_ids")
    topk_logp = traj.get("topk_logp")
    distill_reward = traj.get("distill_reward")
    teacher_logp = traj.get("teacher_logp")

    query_id = traj.get("_mcts_query_id", "")

    node = Node(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=logprobs,
        versions=versions,
        outcome_reward=outcome_reward,
        node_id=traj.get("node_id", traj.get("turn_ids", [""])[0] if traj.get("turn_ids") else ""),
        parent_node_id=traj.get("parent_node_id", traj.get("parent_turn_ids", [None])[0] if traj.get("parent_turn_ids") else None),
        episode_id=traj.get("episode_id", ""),
        topk_ids=topk_ids,
        topk_logp=topk_logp,
        distill_reward=distill_reward,
        teacher_logp=teacher_logp,
    )

    seq_id = self._insert_single(query_id, node)
    # Set metadata on the dict for callers that still read dict keys
    traj["_mcts_seq_id"] = seq_id
    traj["_mcts_query_id"] = query_id
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor(tree-search): update _insert_list_dict to build Node"
```

______________________________________________________________________

### Task 4: Update `_insert_single` and `_insert_per_turn_dicts` for per-turn Nodes

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:197-365`

- [ ] **Step 1: Update `_insert_single` for Node fields**

```python
def _insert_single(self, query_id: str, node: Node) -> int:
    """Insert a single Node and assign a seq_id."""
    seq_id = self._next_seq_id
    self._next_seq_id += 1

    idx = len(self.trajectories.setdefault(query_id, []))
    self.trajectories[query_id].append(node)
    self._seq_id_to_key[seq_id] = (query_id, idx)
    self._query_seq_ids.setdefault(query_id, []).append(seq_id)

    # Set metadata directly on the Node (dataclass permits non-field attrs when not frozen)
    object.__setattr__(node, '_mcts_seq_id', seq_id)
    object.__setattr__(node, '_mcts_query_id', query_id)

    self._backup(seq_id, node.outcome_reward)
    self._trained[seq_id] = False
    self._rewards[seq_id] = node.outcome_reward

    # Register node_id → seq_id mapping for shared-node MCTS
    if node.node_id and node.node_id not in self._turn_nodes:
        self._turn_nodes[node.node_id] = seq_id

    return seq_id
```

- [ ] **Step 2: Update `_insert_per_turn_dicts` to store individual per-turn Nodes**

```python
def _insert_per_turn_dicts(self, turn_dicts: list[dict[str, Any]]) -> None:
    """Insert per-turn dicts as individual Node objects.

    Each turn dict becomes a separate Node in the tree store.
    Turns from the same episode share the same episode_id (derived
    from _mcts_query_id + _episode_idx) and are linked via
    node_id/parent_node_id.
    """
    from itertools import groupby

    sorted_turns = sorted(
        turn_dicts,
        key=lambda d: (d.get("_mcts_query_id", ""), d.get("_episode_idx", 0)),
    )

    for (query_id, ep_idx), group_iter in groupby(
        sorted_turns,
        key=lambda d: (d.get("_mcts_query_id", ""), d.get("_episode_idx", 0)),
    ):
        turns = list(group_iter)
        turns.sort(key=lambda d: d.get("_turn_idx_in_episode", 0))

        episode_id = f"{query_id}_{ep_idx}"

        for turn in turns:
            seq_len = int(turn["attention_mask"].sum())
            input_ids = turn["input_ids"][0, :seq_len].tolist()
            loss_mask = turn["loss_mask"][0, :seq_len].tolist()
            logprobs = (
                turn["logprobs"][0, :seq_len].tolist()
                if "logprobs" in turn
                else [0.0] * seq_len
            )
            versions = (
                turn["versions"][0, :seq_len].tolist()
                if "versions" in turn
                else [0] * seq_len
            )

            node = Node(
                input_ids=input_ids,
                loss_mask=loss_mask,
                logprobs=logprobs,
                versions=versions,
                outcome_reward=turn.get("_outcome_reward", 0.0),
                node_id=turn.get("_turn_id", ""),
                parent_node_id=turn.get("_parent_turn_id"),
                episode_id=episode_id,
            )

            seq_id = self._insert_single(query_id, node)
            # Also set on dict for downstream dict-key readers
            turn["_mcts_seq_id"] = seq_id
            turn["_mcts_query_id"] = query_id
```

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor(tree-search): store per-turn Nodes in tree store"
```

______________________________________________________________________

### Task 5: Update `get_advantages` and `get_prompt_mask` for Node fields

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:367-412`

- [ ] **Step 1: Update both methods**

```python
def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
    """Return per-token advantages: Q-value on response tokens, 0 on prompt."""
    qid, idx = self._seq_id_to_key[seq_id]
    node = self.trajectories[qid][idx]
    q_val = self._q_values.get(seq_id, 0.0)
    seq_len = len(node.input_ids)
    advantages = torch.zeros(seq_len, dtype=torch.float32)
    starts, ends = _find_turn_boundaries(node.loss_mask)
    for start, end in zip(starts, ends):
        advantages[start:end] = q_val
    return advantages

def get_prompt_mask(self, query_id: str, seq_id: int) -> torch.Tensor:
    """Return boolean mask: True for response tokens, False for prompt."""
    qid, idx = self._seq_id_to_key[seq_id]
    node = self.trajectories[qid][idx]
    return torch.tensor(node.loss_mask, dtype=torch.bool)
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "refactor(tree-search): update get_advantages and get_prompt_mask for Node"
```

______________________________________________________________________

### Task 6: Update `load_trajectories` to return `list[Node]` + add episode reconstruction

**Files:**

- Modify: `customized_areal/tree_search/mcts_tree_store.py:413-618`

- [ ] **Step 1: Add `_reconstruct_episode` and `_response_span` helpers**

Add before `load_trajectories`:

```python
def _response_span(loss_mask: list[int]) -> tuple[int, int]:
    """Return (start, end) of the first response region in loss_mask."""
    starts, ends = _find_turn_boundaries(loss_mask)
    if starts:
        return starts[0], ends[0]
    return 0, 0


def _node_to_tensor_dict(node: Node, query_id: str, seq_id: int) -> dict[str, Any]:
    """Convert a single Node to a tensor dict with shape [1, seq_len]."""
    seq_len = len(node.input_ids)
    traj: dict[str, Any] = {
        "input_ids": torch.tensor(node.input_ids, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.tensor(node.loss_mask, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(node.logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(node.versions, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "rewards": torch.tensor([node.outcome_reward], dtype=torch.float32).unsqueeze(0),
        "_mcts_query_id": query_id,
        "_mcts_seq_id": seq_id,
    }
    # Response-only fields: extract response portion from full sequence
    resp_start, resp_end = _response_span(node.loss_mask)
    if node.topk_ids is not None:
        traj["topk_ids"] = torch.tensor(node.topk_ids, dtype=torch.int32).unsqueeze(0)
    if node.topk_logp is not None:
        traj["topk_logp"] = torch.tensor(node.topk_logp, dtype=torch.float32).unsqueeze(0)
    if node.distill_reward is not None:
        traj["distill_reward"] = torch.tensor(node.distill_reward, dtype=torch.float32).unsqueeze(0)
    if node.teacher_logp is not None:
        traj["teacher_logp"] = torch.tensor(node.teacher_logp, dtype=torch.float32).unsqueeze(0)
    # Derived from logprobs for response tokens
    if resp_end > resp_start:
        traj["logp"] = torch.tensor(
            node.logprobs[resp_start:resp_end], dtype=torch.float32
        ).unsqueeze(0)
    # Carry advantages if set by advantage computer
    if hasattr(node, 'advantages') and node.advantages is not None:
        traj["advantages"] = node.advantages.unsqueeze(0) if node.advantages.dim() == 1 else node.advantages
    if hasattr(node, 'returns') and node.returns is not None:
        traj["returns"] = node.returns.unsqueeze(0) if node.returns.dim() == 1 else node.returns
    # Turn metadata
    starts, ends = _find_turn_boundaries(node.loss_mask)
    if node.node_id:
        traj["_turn_id"] = node.node_id
    if node.parent_node_id is not None:
        traj["_parent_turn_id"] = node.parent_node_id
    traj["_turn_reward"] = node.outcome_reward
    traj["_outcome_reward"] = node.outcome_reward
    traj["_episode_idx"] = 0
    traj["_turn_idx_in_episode"] = 0
    traj["_num_turns_in_episode"] = 1
    return traj
```

- [ ] **Step 2: Rewrite `load_trajectories` to return `list[Node]`**

```python
def load_trajectories(
    self, query_id: str, n_samples: int
) -> list[Node]:
    """Load untrained trajectories as Node objects.

    Returns per-turn Node objects. Callers can:
    - Read Node attributes directly for advantage computation
    - Group Nodes by episode_id and reconstruct episodes for training
    - Convert to tensor dicts via _node_to_tensor_dict()

    Each Node carries _mcts_query_id and _mcts_seq_id set during
    insertion (accessible as regular attributes).
    """
    if query_id not in self.trajectories:
        return []

    untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
    result: list[Node] = []
    for seq_id in untrained_ids:
        qid, idx = self._seq_id_to_key[seq_id]
        node = self.trajectories[qid][idx]
        result.append(node)
    return result
```

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/mcts_tree_store.py
git commit -m "feat(tree-search): load_trajectories returns list[Node] with episode reconstruction helpers"
```

______________________________________________________________________

### Task 7: Update `proxy_workflow._interactions_to_turn_dicts` to return `list[Node]`

**Files:**

- Modify: `customized_areal/tree_search/proxy_workflow.py:64-191`

- [ ] **Step 1: Rewrite method to return `list[Node]`**

Rename method to `_interactions_to_nodes` and update:

```python
def _interactions_to_nodes(
    self, interactions: dict[str, Any]
) -> list[Node]:
    """Convert InteractionWithTokenLogpReward objects to list[Node].

    Each interaction becomes one Node representing a single turn.
    """
    from customized_areal.tree_search.mcts_tree_store import Node
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    nodes: list[Node] = []

    for interaction_id, interaction in interactions.items():
        assert isinstance(interaction, InteractionWithTokenLogpReward)
        resp = interaction.model_response
        assert resp is not None, "Model response is not set."

        seq_tokens = resp.input_tokens + resp.output_tokens

        if (
            interaction.chat_template_type == "concat"
            and interaction.parent is not None
        ):
            parent_res = interaction.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)

            if resp.input_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
            else:
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

        outcome_reward = interaction.reward if interaction.reward is not None else 0.0

        topk_ids: list[list[int]] = []
        topk_logp: list[list[float]] = []
        if resp.output_top_logprobs is not None:
            for pos_logprobs in resp.output_top_logprobs:
                ids = []
                logps = []
                for token_id, lp in pos_logprobs:
                    ids.append(token_id)
                    logps.append(lp)
                topk_ids.append(ids)
                topk_logp.append(logps)

        parent_id = (
            interaction.parent.interaction_id
            if interaction.parent and interaction.parent._interaction_id
            else None
        )

        node = Node(
            input_ids=seq_tokens,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            node_id=interaction_id,
            parent_node_id=parent_id,
            episode_id="",
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
            distill_reward=None,
            teacher_logp=None,
        )

        nodes.append(node)

    return nodes
```

- [ ] **Step 2: Update `arun_episode` to call renamed method**

Replace `_interactions_to_turn_dicts` call with `_interactions_to_nodes`:

```python
    nodes = self._interactions_to_nodes(result)
    if query_id:
        for node in nodes:
            node.episode_id = query_id
    return nodes
```

- [ ] **Step 3: Commit**

```bash
git add customized_areal/tree_search/proxy_workflow.py
git commit -m "refactor(tree-search): proxy_workflow returns list[Node]"
```

______________________________________________________________________

### Task 8: Update `grouped_workflow.py` for Node merging

**Files:**

- Modify: `customized_areal/tree_search/grouped_workflow.py:27-168`

- [ ] **Step 1: Rename and update `_merge_turn_dicts_to_episode` →
  `_merge_nodes_to_episode`**

Replace lines 27-117:

```python
def _merge_nodes_to_episode(nodes: list[Node]) -> Node:
    """Merge per-turn Nodes into a single per-episode Node.

    Concatenates all sequence fields from the nodes in order.
    The episode_id is taken from the first node.
    """
    if not nodes:
        return Node(
            input_ids=[], loss_mask=[], logprobs=[], versions=[],
            node_id="", parent_node_id=None, episode_id="",
        )

    all_input_ids: list[int] = []
    all_loss_mask: list[int] = []
    all_logprobs: list[float] = []
    all_versions: list[int] = []
    all_topk_ids: list[list[int]] = []
    all_topk_logp: list[list[float]] = []
    all_distill_reward: list[list[float]] = []
    all_teacher_logp: list[list[float]] = []

    for node in nodes:
        all_input_ids.extend(node.input_ids)
        all_loss_mask.extend(node.loss_mask)
        all_logprobs.extend(node.logprobs)
        all_versions.extend(node.versions)
        if node.topk_ids is not None:
            all_topk_ids.extend(node.topk_ids)
        if node.topk_logp is not None:
            all_topk_logp.extend(node.topk_logp)
        if node.distill_reward is not None:
            all_distill_reward.extend(node.distill_reward)
        if node.teacher_logp is not None:
            all_teacher_logp.extend(node.teacher_logp)

    return Node(
        input_ids=all_input_ids,
        loss_mask=all_loss_mask,
        logprobs=all_logprobs,
        versions=all_versions,
        node_id=nodes[-1].node_id,
        parent_node_id=nodes[-1].parent_node_id,
        episode_id=nodes[0].episode_id or nodes[0].node_id,
        outcome_reward=nodes[-1].outcome_reward,
        topk_ids=all_topk_ids if all_topk_ids else None,
        topk_logp=all_topk_logp if all_topk_logp else None,
        distill_reward=all_distill_reward if all_distill_reward else None,
        teacher_logp=all_teacher_logp if all_teacher_logp else None,
    )
```

Keep the old `_merge_turn_dicts_to_episode` function as-is for legacy dict fallback.

- [ ] **Step 2: Update `arun_episode` to detect Node format**

Replace the merge section (lines 148-168):

```python
    first = valid_results[0]
    if isinstance(first, list) and len(first) > 0:
        episode_nodes: list[Node] = []
        query_id = data.get("query_id") or ""

        if isinstance(first[0], Node):
            # New format: list[Node] per result
            for result in valid_results:
                merged = _merge_nodes_to_episode(result)
                if merged.input_ids:
                    if query_id:
                        merged.episode_id = f"{query_id}_{len(episode_nodes)}"
                    episode_nodes.append(merged)
            if not episode_nodes:
                return None
            return episode_nodes
        elif isinstance(first[0], dict):
            # Legacy format: list[dict] per result
            episode_trajs: list[dict[str, Any]] = []
            for result in valid_results:
                merged = _merge_turn_dicts_to_episode(result)
                if merged:
                    if query_id:
                        merged["_mcts_query_id"] = query_id
                    episode_trajs.append(merged)
            if not episode_trajs:
                return None
            return episode_trajs

    # Legacy tensor dicts
    concatenated = concat_padded_tensors(valid_results)
    return [concatenated] if concatenated else None
```

- [ ] **Step 3: Add imports at top of file**

```python
from customized_areal.tree_search.mcts_tree_store import Node
```

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/grouped_workflow.py
git commit -m "refactor(tree-search): grouped_workflow uses Node merging"
```

______________________________________________________________________

### Task 9: Update `advantage.py` to work with `Node` objects

**Files:**

- Modify: `customized_areal/tree_search/advantage.py:13-127`

- [ ] **Step 1: Rewrite `compute` and helpers for Node objects**

```python
class TreeAdvantageComputer:
    """Replace GAE advantages with tree Q-values from MCTS backup.

    Works with Node objects. Reads _mcts_query_id and _mcts_seq_id
    attributes. Sets node.advantages and node.returns (torch.Tensor).
    """

    def __init__(self, tree_store: MCTSTreeStore, grpo_eps: float = GRPO_NORM_EPS):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps

    def _compute_single(
        self, node: Node, seq_id: int
    ) -> torch.Tensor:
        """Compute tree Q-value advantages for a single Node."""
        normalized_q = self.tree_store._normalized_advantages.get(seq_id)
        if normalized_q is None:
            normalized_q = self.tree_store._q_values.get(seq_id, 0.0)

        seq_len = len(node.input_ids)
        mask = torch.tensor(node.loss_mask, dtype=torch.bool)
        advantages = mask.float() * normalized_q
        return advantages

    @staticmethod
    def _get_seq_len(node: Node) -> int:
        """Get sequence length from a Node."""
        return len(node.input_ids)

    def compute(self, trajectories: list[Node]) -> None:
        """Compute tree Q-value advantages. Mutates Node objects in-place.

        Sets node.advantages and node.returns as 1-D torch.Tensor.
        """
        # Collect unique (query_id, seq_id) pairs for GRPO normalization
        query_seq_sets: dict[str, dict[int, None]] = {}

        for node in trajectories:
            if not hasattr(node, '_mcts_query_id') or not hasattr(node, '_mcts_seq_id'):
                continue
            query_id = node._mcts_query_id
            seq_id = node._mcts_seq_id
            qset = query_seq_sets.setdefault(query_id, {})
            qset[seq_id] = None

        # Per-query GRPO normalization
        for query_id, seq_id_set in query_seq_sets.items():
            seq_ids = list(seq_id_set)
            q_values = [self.tree_store._rewards.get(sid, 0.0) for sid in seq_ids]
            if len(q_values) < 2:
                if seq_ids:
                    self.tree_store._normalized_advantages[seq_ids[0]] = q_values[0]
                continue
            mean_q = sum(q_values) / len(q_values)
            var_q = sum((q - mean_q) ** 2 for q in q_values) / len(q_values)
            std_q = var_q**0.5
            for sid, q in zip(seq_ids, q_values):
                self.tree_store._normalized_advantages[sid] = (q - mean_q) / (
                    std_q + self.grpo_eps
                )

        # Compute per-node advantages, set on Node objects
        for node in trajectories:
            if not hasattr(node, '_mcts_query_id') or not hasattr(node, '_mcts_seq_id'):
                continue
            seq_id = node._mcts_seq_id
            advantages = self._compute_single(node, seq_id)
            object.__setattr__(node, 'advantages', advantages)
            object.__setattr__(node, 'returns', advantages.clone())
```

Remove the `_get_seq_len` static method with `input_ids` parameter — replaced by the one
above.

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/advantage.py
git commit -m "refactor(tree-search): advantage.py works with Node objects"
```

______________________________________________________________________

### Task 10: Update `trainer.py` helpers to convert `Node` → tensor dict

**Files:**

- Modify: `customized_areal/tree_search/trainer.py:48-77, 319-606`

- [ ] **Step 1: Update `_is_list_traj` and `_list_dict_to_tensor`**

Replace helpers (lines 48-77):

```python
def _is_node(traj: Any) -> bool:
    """Check if a trajectory is a Node object."""
    from customized_areal.tree_search.mcts_tree_store import Node
    return isinstance(traj, Node)


def _is_list_traj(traj: dict[str, Any]) -> bool:
    """Check if a trajectory dict uses Python lists instead of tensors."""
    return isinstance(traj.get("input_ids"), list)


def _node_to_tensor_dict(traj: Any) -> dict[str, Any]:
    """Convert a Node or list-based trajectory dict to tensor format [1, seq_len].

    Node objects have advantages/returns set by TreeAdvantageComputer.
    List-based dicts are from legacy format.
    """
    from customized_areal.tree_search.mcts_tree_store import Node

    if isinstance(traj, Node):
        seq_len = len(traj.input_ids)
        result: dict[str, Any] = {
            "input_ids": torch.tensor(traj.input_ids, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(traj.loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(traj.logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(traj.versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "rewards": torch.tensor([traj.outcome_reward], dtype=torch.float32),
        }
        if traj.topk_ids is not None:
            result["topk_ids"] = torch.tensor(traj.topk_ids, dtype=torch.int32).unsqueeze(0)
        if traj.topk_logp is not None:
            result["topk_logp"] = torch.tensor(traj.topk_logp, dtype=torch.float32).unsqueeze(0)
        if traj.distill_reward is not None:
            result["distill_reward"] = torch.tensor(traj.distill_reward, dtype=torch.float32).unsqueeze(0)
        if traj.teacher_logp is not None:
            result["teacher_logp"] = torch.tensor(traj.teacher_logp, dtype=torch.float32).unsqueeze(0)
        if hasattr(traj, 'advantages') and traj.advantages is not None:
            result["advantages"] = traj.advantages.unsqueeze(0) if traj.advantages.dim() == 1 else traj.advantages
        if hasattr(traj, 'returns') and traj.returns is not None:
            result["returns"] = traj.returns.unsqueeze(0) if traj.returns.dim() == 1 else traj.returns
        if hasattr(traj, '_mcts_query_id'):
            result["_mcts_query_id"] = traj._mcts_query_id
        if hasattr(traj, '_mcts_seq_id'):
            result["_mcts_seq_id"] = traj._mcts_seq_id
        # Derive logp from logprobs for response tokens
        starts, ends = _find_turn_boundaries(traj.loss_mask)
        if starts and ends:
            resp_start, resp_end = starts[0], ends[0]
            if resp_end > resp_start:
                result["logp"] = torch.tensor(
                    traj.logprobs[resp_start:resp_end], dtype=torch.float32
                ).unsqueeze(0)
        return result

    # Legacy list-based dict
    seq_len = len(traj["input_ids"])
    result: dict[str, Any] = {
        "input_ids": torch.tensor(traj["input_ids"], dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.tensor(traj["loss_mask"], dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(traj["logprobs"], dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(traj["versions"], dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "rewards": torch.tensor([traj.get("outcome_reward", traj.get("reward", 0.0))], dtype=torch.float32),
    }
    for key in traj:
        if key not in result:
            result[key] = traj[key]
    return result
```

- [ ] **Step 2: Update `_mark_batch_trained` to work with Node objects**

Update `_mark_batch_trained` (lines 80-99):

```python
def _mark_batch_trained(
    tree_store: MCTSTreeStore, trajectories: list[Any]
) -> None:
    """Mark all trajectories in a batch as trained after tree backup.

    Handles both Node objects (with _mcts_query_id/_mcts_seq_id attrs)
    and legacy dicts (with _mcts_query_id/_mcts_seq_id keys).
    """
    count = 0
    for traj in trajectories:
        if hasattr(traj, '_mcts_query_id'):
            query_id = traj._mcts_query_id
            seq_id = getattr(traj, '_mcts_seq_id', None)
        else:
            query_id = traj.get("_mcts_query_id")
            seq_id = traj.get("_mcts_seq_id")
        if query_id is None:
            continue
        if seq_id is not None:
            tree_store.set_trained(query_id, seq_id, True)
            count += 1
        seq_ids = traj.get("_mcts_seq_ids") if isinstance(traj, dict) else None
        if seq_ids is not None:
            for sid in seq_ids:
                tree_store.set_trained(query_id, sid, True)
                count += 1
    if count:
        logger.debug(f"Marked {count} trajectories as trained")
```

- [ ] **Step 3: Update `_cache_aware_prepare_batch` for Node objects**

In the tree operations section (lines 570-604), update to work with Node objects:

```python
    # Insert trajectories into the MCTS tree
    self.tree_store.insert_batch(trajs)
    logger.debug(f"Inserted {len(trajs)} trajectories into tree")

    # Compute tree advantages and stash for post-GAE restoration
    if self.tree_backup_config.advantage_mode == AdvantageMode.TREE:
        # Filter to Node objects for advantage computation
        nodes = [t for t in trajs if _is_node(t)]
        if nodes:
            self.tree_advantage_computer.compute(nodes)
        for traj in trajs:
            if hasattr(traj, 'advantages') and traj.advantages is not None:
                traj._tree_advantages = (
                    traj.advantages.clone() if hasattr(traj.advantages, 'clone') else traj.advantages
                )
                traj._tree_returns = (
                    traj.returns.clone() if hasattr(traj.returns, 'clone') else traj.returns
                )
        logger.debug(
            f"Computed tree advantages for {len(trajs)} trajectories (mode=TREE)"
        )

    # Mark trajectories as trained
    _mark_batch_trained(self.tree_store, trajs)

    # Convert Node/list-dict to tensor dict for PPO pipeline
    trajs = [_node_to_tensor_dict(t) if (_is_node(t) or _is_list_traj(t)) else t for t in trajs]
```

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tree_search/trainer.py
git commit -m "refactor(tree-search): trainer.py handles Node objects and converts to tensor dict"
```

______________________________________________________________________

### Task 11: Update `checkpoint.py` for `Node` serialization

**Files:**

- Modify: `customized_areal/tree_search/checkpoint.py`

- [ ] **Step 1: Update imports and serialize/deserialize**

Replace `TrajectoryRecord` with `Node` in import (line 13):

```python
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node
```

Replace docstring (line 1): "TrajectoryRecord" → "Node".

Replace `_serialize_record` (lines 94-125):

```python
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
```

Replace `_deserialize_record` (lines 127-146):

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
        topk_ids=data.get("topk_ids"),
        topk_logp=data.get("topk_logp"),
        distill_reward=data.get("distill_reward"),
        teacher_logp=data.get("teacher_logp"),
    )
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/checkpoint.py
git commit -m "refactor(tree-search): checkpoint serialization for Node"
```

______________________________________________________________________

### Task 12: Update `__init__.py` exports

**Files:**

- Modify: `customized_areal/tree_search/__init__.py`

- [ ] **Step 1: Replace `TrajectoryRecord` with `Node`**

```python
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

__all__ = [
    "AdvantageMode",
    "CacheAwarePPOTrainer",
    "MCTSTreeStore",
    "Node",
    "QueryIDProxyWorkflow",
    "RolloutCacheConfig",
    "TreeAdvantageComputer",
    "TreeBackupConfig",
    "TreeBackupMode",
    "TreeCheckpointManager",
]
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/__init__.py
git commit -m "refactor(tree-search): export Node instead of TrajectoryRecord"
```

______________________________________________________________________

### Task 13: Update tests for `Node` dataclass

**Files:**

- Modify: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Update imports**

```python
from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    Node,
    _find_turn_boundaries,
)
```

- [ ] **Step 2: Replace `TestTrajectoryRecord` → `TestNode`**

```python
class TestNode:
    def test_creation(self):
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="turn_0",
            parent_node_id=None,
            episode_id="ep_1",
            outcome_reward=1.0,
        )
        assert len(node.input_ids) == 5
        assert node.outcome_reward == 1.0
        assert node.node_id == "turn_0"
        assert node.parent_node_id is None

    def test_new_fields_default_to_none(self):
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="t1",
            parent_node_id=None,
            episode_id="ep_1",
        )
        assert node.topk_ids is None
        assert node.topk_logp is None
        assert node.distill_reward is None
        assert node.teacher_logp is None

    def test_new_fields_can_be_set(self):
        topk_ids = [[10, 20], [30, 40], [50, 60]]
        topk_logp = [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]
        distill_reward = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        teacher_logp = [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]]

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="t1",
            parent_node_id=None,
            episode_id="ep_1",
            outcome_reward=1.0,
            topk_ids=topk_ids,
            topk_logp=topk_logp,
            distill_reward=distill_reward,
            teacher_logp=teacher_logp,
        )
        assert node.topk_ids == topk_ids
        assert node.topk_logp == topk_logp
        assert node.distill_reward == distill_reward
        assert node.teacher_logp == teacher_logp
```

- [ ] **Step 3: Update `TestMCTSTreeStoreInsertBatch` tests**

Update assertions in `test_insert_list_dict_basic`:

```python
def test_insert_list_dict_basic(self):
    store = MCTSTreeStore()
    traj = {
        "input_ids": [1, 2, 3, 4, 5],
        "loss_mask": [0, 0, 1, 1, 1],
        "outcome_reward": 1.0,
        "attention_mask": [1, 1, 1, 1, 1],
    }
    store.insert_batch([traj])
    assert "_mcts_seq_id" in traj
    assert "_mcts_query_id" in traj
    assert len(store.trajectories) == 1
    query_id = traj["_mcts_query_id"]
    assert len(store.trajectories[query_id]) == 1
    node = store.trajectories[query_id][0]
    assert node.input_ids == [1, 2, 3, 4, 5]
    assert node.loss_mask == [0, 0, 1, 1, 1]
```

Replace `record.reward` → `node.outcome_reward` and `record.logp` → `node.topk_ids` in
remaining assertions.

- [ ] **Step 4: Update `TestMCTSTreeStoreLoadTrajectories` tests**

`load_trajectories` now returns `list[Node]`. Update tests:

```python
class TestMCTSTreeStoreLoadTrajectories:
    def test_load_trajectories_returns_nodes(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "versions": [0, 0, 0, 0, 0],
            "outcome_reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "topk_ids": [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            "topk_logp": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
            "distill_reward": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
            "teacher_logp": [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6], [-1.7, -1.8], [-1.9, -2.0]],
        }
        store.insert_batch([traj])
        loaded = store.load_trajectories(traj["_mcts_query_id"], n_samples=1)
        assert len(loaded) == 1
        node = loaded[0]
        assert isinstance(node, Node)
        assert node.input_ids == [1, 2, 3, 4, 5]
        assert node.outcome_reward == 1.0
        assert node.topk_ids == [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        assert node.teacher_logp == [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6], [-1.7, -1.8], [-1.9, -2.0]]
        assert hasattr(node, '_mcts_query_id')
        assert hasattr(node, '_mcts_seq_id')

    def test_load_trajectories_basic(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=1.0, query_id="q1")
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert len(loaded) == 1
        node = loaded[0]
        assert isinstance(node, Node)
        assert len(node.input_ids) == 5
        assert hasattr(node, '_mcts_query_id')

    def test_load_trajectories_only_untrained(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        store.set_trained("q1", t1["_mcts_seq_id"], True)
        loaded = store.load_trajectories("q1", n_samples=2)
        assert len(loaded) == 1
        assert loaded[0].outcome_reward == 0.5

    def test_load_trajectories_unknown_query(self):
        store = MCTSTreeStore()
        assert store.load_trajectories("nonexistent", n_samples=1) == []
```

- [ ] **Step 5: Run tests**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run pytest tests/test_tree_search/test_mcts_tree_store.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_tree_search/test_mcts_tree_store.py
git commit -m "test(tree-search): update tests for Node dataclass"
```

______________________________________________________________________

### Task 14: Run formatting and linting

**Files:** All modified files

- [ ] **Step 1: Run pre-commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && pre-commit run --all-files
```

- [ ] **Step 2: Verify imports**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run python -c "from customized_areal.tree_search import MCTSTreeStore, Node, TreeCheckpointManager, CacheAwarePPOTrainer; print('imports OK')"
```

Expected: "imports OK"

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -A && git commit -m "style(tree-search): pre-commit fixes for Node dataclass"
```

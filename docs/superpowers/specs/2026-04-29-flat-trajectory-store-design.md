# Flat Trajectory Store Design

Replace the `TrieNode`-based MCTS tree with a flat `TrajectoryRecord` store. The trie
structure cannot correctly reconstruct multi-turn trajectories with variable prefixes.
The flat store keeps the complete `input_ids` and `loss_mask` from the rollout,
eliminating reconstruction bugs.

## Motivation

The current `TrieNode` trie has two fatal problems for multi-turn, variable-prefix
trajectories:

1. **Lost prompt context**: `turn_splitter` stores only the assistant marker tokens as
   `prompt_tokens` (e.g., `<|im_start|>assistant` = 3 tokens). The full conversation
   prefix before each marker is discarded. Path reconstruction produces
   `[marker][resp1][marker][resp2]` — missing system prompts and user questions.

1. **Broken `loss_mask`**: `load_trajectories` computes `loss_mask` from
   `prompt_len_total` (sum of marker lengths). For a 3-turn trajectory with 3-token
   markers, `loss_mask=1` starts at position 9, but actual response tokens start much
   earlier. Most response tokens are incorrectly masked as prompt.

Since trie deduplication is not required (user confirmed flat storage is acceptable),
the trie adds complexity without benefit.

## Design

### TrajectoryRecord

```python
@dataclass
class TrajectoryRecord:
    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    reward: float
    turn_response_starts: list[int]  # from loss_mask 0→1 transitions
    turn_response_ends: list[int]    # from loss_mask 1→0 transitions
```

Each record stores the **complete, unpadded** sequence as produced by the rollout. Turn
boundaries are derived from `loss_mask` transitions at insert time — no `turn_splitter`
or assistant marker detection needed.

This works for both AReaL export styles:

- **`individual`**: Each turn dict has one `0→1` transition → one turn boundary
- **`concat`**: Multiple `0→1`/`1→0` transitions → multiple turn boundaries

### MCTSTreeStore

```python
class MCTSTreeStore:
    trajectories: dict[str, list[TrajectoryRecord]]  # query_id → records
    _seq_id_to_key: dict[int, tuple[str, int]]       # seq_id → (query_id, index)
    _query_seq_ids: dict[str, list[int]]             # query_id → seq_ids in insertion order
    _next_seq_id: int
    _visit_counts: dict[int, int]    # seq_id → count
    _total_values: dict[int, float]  # seq_id → total
    _q_values: dict[int, float]      # seq_id → Q
    _trained: dict[int, bool]        # seq_id → trained
    _rewards: dict[int, float]       # seq_id → reward
```

MCTS stats are keyed by `seq_id` (int) instead of `(query_id, id(node))`. This survives
serialization directly — no `rebuild_mcts_stats` needed.

**Removed APIs**: `start_sequence`, `add_turn`, `finish_sequence` (cursor-based trie
building), `insert_trajectory` (single-trajectory convenience),
`_split_metadata_to_turns`, `build_training_history`, `load_trajectory_by_seq_id`.

**Retained helper methods** (used by `_CacheAwareBatchBuilder`):

```python
def get_untrained_count(self, query_id: str) -> int:
    if query_id not in self._query_seq_ids:
        return 0
    return sum(1 for seq_id in self._query_seq_ids[query_id] if not self._trained.get(seq_id, False))

def get_untrained_seq_ids(self, query_id: str, n_samples: int) -> list[int]:
    if query_id not in self._query_seq_ids:
        return []
    result = []
    for seq_id in self._query_seq_ids[query_id]:
        if not self._trained.get(seq_id, False):
            result.append(seq_id)
            if len(result) >= n_samples:
                break
    return result
```

**Simplified `_backup`**: No node walk. Just increment visit count and update Q-value
for the seq_id.

```python
def _backup(self, seq_id: int, reward: float) -> None:
    self._visit_counts[seq_id] = self._visit_counts.get(seq_id, 0) + 1
    self._total_values[seq_id] = self._total_values.get(seq_id, 0.0) + reward
    self._q_values[seq_id] = self._total_values[seq_id] / self._visit_counts[seq_id]
```

### insert_batch

Padding is stripped per sample using `attention_mask` before storing:

```python
def _make_record(self, traj: dict, idx: int, seq_len: int) -> TrajectoryRecord:
    input_ids = traj["input_ids"][idx, :seq_len].tolist()
    loss_mask = traj["loss_mask"][idx, :seq_len].tolist()
    # ... logprobs, versions similarly sliced
    starts, ends = self._find_turn_boundaries(loss_mask)
    return TrajectoryRecord(input_ids=..., loss_mask=..., ..., turn_response_starts=starts, turn_response_ends=ends)
```

### Turn Boundary Detection

```python
@staticmethod
def _find_turn_boundaries(loss_mask: list[int]) -> tuple[list[int], list[int]]:
    starts, ends = [], []
    in_response = False
    for i, v in enumerate(loss_mask):
        if v == 1 and not in_response:
            starts.append(i)
            in_response = True
        elif v == 0 and in_response:
            ends.append(i)
            in_response = False
    if in_response:
        ends.append(len(loss_mask))
    return starts, ends
```

No assistant marker detection. Works purely from the `loss_mask` that AReaL's rollout
pipeline already computes correctly for both export styles.

### Advantage Computation

Q-value is per-trajectory (not per-turn-node), distributed across response tokens:

```python
def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
    qid, idx = self._seq_id_to_key[seq_id]
    record = self.trajectories[qid][idx]
    q_val = self._q_values.get(seq_id, 0.0)
    seq_len = len(record.input_ids)
    advantages = torch.zeros(seq_len, dtype=torch.float32)
    for start, end in zip(record.turn_response_starts, record.turn_response_ends):
        advantages[start:end] = q_val
    return advantages

def get_prompt_mask(self, query_id: str, seq_id: int) -> torch.Tensor:
    qid, idx = self._seq_id_to_key[seq_id]
    record = self.trajectories[qid][idx]
    return torch.tensor(record.loss_mask, dtype=torch.bool)
```

Same Q-value for all turns within a trajectory. Per-turn differentiation requires trie
sharing, which we've removed.

### load_trajectories

Returns stored `input_ids`/`loss_mask` directly — no reconstruction:

```python
def load_trajectories(self, query_id: str, n_samples: int) -> list[dict]:
    untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
    result = []
    for seq_id in untrained_ids:
        qid, idx = self._seq_id_to_key[seq_id]
        record = self.trajectories[qid][idx]
        seq_len = len(record.input_ids)
        result.append({
            "input_ids": torch.tensor(record.input_ids, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(record.loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(record.logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(record.versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(seq_len, dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([record.reward], dtype=torch.float32).unsqueeze(0),
            "_mcts_query_id": query_id,
            "_mcts_seq_id": seq_id,
        })
    return result
```

No padding in stored data (stripped at insert), so `attention_mask` is all ones.

### Checkpoint Format

```json
// query_<query_id>.json
{
  "records": [
    {
      "input_ids": [...],
      "loss_mask": [...],
      "logprobs": [...],
      "versions": [...],
      "reward": 0.75,
      "turn_response_starts": [30],
      "turn_response_ends": [100]
    }
  ]
}

// metadata.json
{
  "next_seq_id": 15,
  "seq_id_to_key": {"7": ["abc", 0], "8": ["abc", 1]},
  "query_seq_ids": {"abc": [7, 8]},
  "visit_counts": {"7": 1, "8": 1},
  "total_values": {"7": 0.75, "8": 0.0},
  "q_values": {"7": 0.75, "8": 0.0},
  "trained": {"7": true, "8": false},
  "rewards": {"7": 0.75, "8": 0.0}
}
```

No `rebuild_mcts_stats` — stats keyed by `seq_id` serialize directly. Old
`TrieNode`-based checkpoints are incompatible; they are discarded.

## Files Changed

| File                 | Change                                                     |
| -------------------- | ---------------------------------------------------------- |
| `mcts_tree_store.py` | Rewrite: flat `TrajectoryRecord` store                     |
| `trie_node.py`       | Delete entirely                                            |
| `turn_splitter.py`   | Delete entirely                                            |
| `advantage.py`       | Minor: adapt to new store internals                        |
| `checkpoint.py`      | Rewrite: new serialization format                          |
| `config.py`          | Remove `assistant_marker` from `TreeBackupConfig`          |
| `trainer.py`         | Remove `make_turn_splitter` call, `assistant_marker` usage |

## Edge Cases

1. **`loss_mask` all zeros**: `_find_turn_boundaries` returns empty lists.
   `get_advantages` returns all-zeros. Log a warning.
1. **`loss_mask` all ones**: Single turn, no prompt. `starts=[0]`, `ends=[seq_len]`.
   Valid.
1. **`concat` mode, strict prefix broken**: AReaL masks out parent turns
   (`loss_mask=0`). Only last turn's response has `loss_mask=1`. One turn boundary
   found. Correct.
1. **Padding**: Stripped at insert via `attention_mask.sum()`. No padding in stored
   records.
1. **Checkpoint migration**: Old checkpoints discarded. Cache is regeneratable.
